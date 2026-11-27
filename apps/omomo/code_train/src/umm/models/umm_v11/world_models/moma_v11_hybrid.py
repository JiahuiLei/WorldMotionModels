"""MotionMachine V11 Hybrid -- per-ctx-timestep P-layer with shared QKV + separate FFN.

Architecture:
  T-layer: full self-attention between ctx+X tokens. ONLY reg-x registers.
    - reg-x bidirectional with X only; cannot see ctx, ctx cannot see reg-x.
    - Dual-width: C_T (narrow) for T-layers, C_P (wide) for P-layers.
    - layer0_mix 3-way: x_state_0 + latest_p_proj + x_state.
    - Register additive carry-over: reg_x = prev_reg_x_state + learnable_embed.

  P-layer: (len_ctx + 1) parallel slots, each (B, P, C_P):
    - Slots 0..len_ctx-1: one per ctx timestep. Pools that ctx token's x_state.
    - Slot len_ctx: X slot. Pools reg-x flat (nrx * C_T).
    - POOL: project into C_P with skip from previous P output.
    - ATTN: shared QKV across all entities (batched). Separate IR embeddings
      for ctx vs X slots. Shared DiT P-block with num_registers=0.
    - FFN: separate MLP for ctx slots vs X slot (norm + ada_modulate + mlp + gate).
    - UNPOOL: ctx P-out -> C_T broadcast to ctx positions; X P-out -> reg + tok.

Compile safety:
  - _build_hybrid_attn_mask: @torch.compiler.disable (Python loops)
  - _sample_traj_ids: @torch.compiler.disable (data-dependent control flow)
  - HybridFFN.forward: @torch.compiler.disable (receives view tensors from
    reshape-after-slice; dynamo cannot track ._base symbolic shapes)
  - All other methods: compile-safe (static shapes, no closures, no math.ceil)
  - len_ctx fixed at init (no dynamic loop trips)
  - P-layer batch dim B*(len_ctx+1) is static

  NOTE on ada_n_layers=2 + compile: PMEM (lrun 0033) warns ada_n_layers=2
  can hit Triton shared memory limit on A6000. The ada MLP runs INSIDE compiled
  t_blocks and shared_p_blocks (NOT protected by HybridFFN's @compiler.disable).
  Empirically verified safe at C_T=128, C_P=512, 4t/3p layers, BS=32 past 1000+
  steps. Monitor if scaling model size or changing Triton tiling.
  When ctx_pool_adaln=True, C_feat increases by ada_extra_dim (default 64),
  widening the ada MLP input. Must re-verify compile safety with this flag.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
)
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly
from umm.models.umm_v11.world_models.rope_utils import DiTBlockSplitRoPE
from umm.models.umm_v11.world_models.so3_utils import compute_entity_mean_9d


class HybridFFN(nn.Module):
    """Standalone FFN module replicating DiTBlockTimeOnly's MLP phase.

    Applies: norm -> ada_modulate(s2, sc2) -> mlp -> gated_residual(g2).
    Owns its own norm and mlp weights but receives ada params externally.

    Args:
        C: Channel dimension.
        mlp_hidden_dims: List of hidden dimensions for the MLP.
        ada_scale_bias_bound: Soft clamping bound for ada scale/bias (<=0 to disable).
        gate_bound: Soft clamping bound for gate (<=0 to disable).
    """

    def __init__(
        self,
        C: int,
        mlp_hidden_dims: list,
        ada_scale_bias_bound: float = -1.0,
        gate_bound: float = -1.0,
    ):
        super().__init__()
        self.ada_scale_bias_bound = ada_scale_bias_bound
        self.gate_bound = gate_bound

        self.norm = nn.LayerNorm(C, elementwise_affine=False)

        # Build MLP (same pattern as DiTBlockTimeOnly)
        layers = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, C))
        self.mlp = nn.Sequential(*layers)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        s2: torch.Tensor,
        sc2: torch.Tensor,
        g2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FFN phase: norm -> ada -> mlp -> gated residual.

        Args:
            x: (B, L, C) input (post-attention state).
            s2, sc2, g2: Ada params from shared P-block, shape (B, L, C).

        Returns:
            (B, L, C) output after FFN.
        """
        h = self.norm(x)
        # Ada modulation: x * (1 + sc) + s (same as DiTBlockTimeOnly._apply_ada)
        # s2=shift (additive), sc2=scale (multiplicative). Only scale is bounded.
        if self.ada_scale_bias_bound > 0.0:
            sc2 = torch.tanh(sc2) * self.ada_scale_bias_bound
        h = h * (1 + sc2) + s2
        h = self.mlp(h)
        # Gated residual: x + g * h (same as DiTBlockTimeOnly._add_gate)
        if self.gate_bound > 0.0:
            g2 = torch.tanh(g2) * self.gate_bound
        return x + g2 * h


class MotionMachineV11Hybrid(nn.Module):
    """Hybrid MotionMachine (V11) with per-ctx-timestep P-layer slots.

    Layer pattern: t-p-t-p-...-t (alternating, configurable via layer_type_string).
    n_t_layers t-layers and n_p_layers p-layers.

    T-layer: full ctx+X self-attention with reg-x only (no reg-c, reg-a).
    P-layer: (len_ctx+1) parallel slots with shared QKV and separate FFN.

    Args:
        C_T: Narrow channel width for t-layers.
        C_P: Wide channel width for p-layers.
        feat_dim: Input feature dimension from identity layer.
        n_head_t: Number of attention heads for t-layers.
        n_head_p: Number of attention heads for p-layers.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        n_t_layers: Number of t-layers. Must be >= 2.
        n_reg_x: Number of x register tokens.
        n_ctx: Maximum context length for position embeddings.
        fc_factor_t: MLP width factor for t-layers.
        fc_factor_p: MLP width factor for p-layers.
        fc_n_layers: Number of hidden layers in each DiT block MLP.
        fc_out_n_layers: Number of layers in output projection.
        num_blk_internal_registers: Internal register count for DiT blocks.
        time_multiplier: Scale factor for timestep values.
        ada_scale_bias_bound: Bound for adaLN scale/bias.
        gate_bound: Bound for gate values.
        ada_n_layers: Number of layers in adaLN projection.
        add_time_to_feat: Use continuous time PE in conditioning.
        time_rope_flag: Use RoPE for time in t-layers.
        qk_norm: Use QK normalization in attention.
        x_pe_dim: If > 0, apply NeRF-style positional encoding to input.
        ctx_trace_indicator_dim: Dimension of ctx/trace indicator embedding.
        spatial_rope_flag: split spatial RoPE + trajectory PE.
        n_traj_tokens: number of trajectory tokens.
        len_ctx: Number of context timesteps for P-layer slots.
        shared_ctx_x: share FFN weights and IR embeddings between ctx and X P-layer slots.
        ctx_pool_adaln: pool ctx tokens (max/min/mean) as extra adaLN conditioning.
        ada_extra_dim: Dimension of the projected ctx-pool feature for adaLN.
        layer_type_string: configurable layer pattern (e.g. "tptptpt").
        p_unpool_to_x: If True (default), unpool P-layer X slot back to X token state.
            When False, X tokens are not updated from P-layer output (saves params).
    """

    def __init__(
        self,
        C_T: int = 128,
        C_P: int = 512,
        feat_dim: int = 512,
        n_head_t: int = 8,
        n_head_p: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        n_t_layers: int = 4,
        n_reg_x: int = 4,
        n_ctx: int = 128,
        #
        fc_factor_t: int = 4,
        fc_factor_p: int = 2,
        fc_n_layers: int = 1,
        fc_out_n_layers: int = 4,
        num_blk_internal_registers: int = 4,
        #
        time_multiplier: float = 1.0,
        ada_scale_bias_bound: float = -1.0,
        gate_bound: float = -1.0,
        ada_n_layers: int = 1,
        #
        add_time_to_feat: bool = False,
        time_rope_flag: bool = False,
        qk_norm: bool = False,
        x_pe_dim: int = 0,
        # Mod 2: ctx/trace indicator embedding
        ctx_trace_indicator_dim: int = 0,
        # split spatial RoPE + trajectory PE
        spatial_rope_flag: bool = False,
        n_traj_tokens: int = 0,
        # Hybrid-specific
        len_ctx: int = 2,
        # shared ctx/x FFN and IR embeddings
        shared_ctx_x: bool = False,
        # ctx pool as extra adaLN conditioning
        ctx_pool_adaln: bool = False,
        ada_extra_dim: int = 64,
        # configurable layer pattern string
        layer_type_string: Optional[str] = None,
        # Hybrid: skip P-layer unpool to X tokens (saves params)
        p_unpool_to_x: bool = True,
    ):
        super().__init__()

        self.C_T = C_T
        self.C_P = C_P
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.len_ctx = len_ctx
        self.shared_ctx_x = shared_ctx_x
        self.ctx_pool_adaln = ctx_pool_adaln
        self.ada_extra_dim = ada_extra_dim
        self.p_unpool_to_x = p_unpool_to_x

        # layer_type_string support
        if layer_type_string is not None:
            assert len(layer_type_string) > 0, "layer_type_string must not be empty"
            assert layer_type_string[0] == "t", (
                f"layer_type_string must start with 't', got '{layer_type_string[0]}'"
            )
            assert layer_type_string[-1] == "t", (
                f"layer_type_string must end with 't', got '{layer_type_string[-1]}'"
            )
            for ch in layer_type_string:
                assert ch in "tp", f"Invalid layer type '{ch}', must be t/p"
            self.n_t_layers = sum(1 for c in layer_type_string if c == "t")
            self.n_p_layers = sum(1 for c in layer_type_string if c == "p")
            self.layer_type_string = layer_type_string
            _string_n_t = self.n_t_layers
            if n_t_layers != 4 and n_t_layers != _string_n_t:
                logging.warning(
                    f"layer_type_string '{layer_type_string}' has {_string_n_t} t-layers "
                    f"but n_t_layers={n_t_layers} was also provided. "
                    f"layer_type_string takes precedence."
                )
        else:
            self.n_t_layers = n_t_layers
            self.n_p_layers = n_t_layers - 1
            self.layer_type_string = "t" + "pt" * (n_t_layers - 1)

        self.n_head_t = n_head_t
        self.n_head_p = n_head_p
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat
        self.spatial_rope_flag = spatial_rope_flag
        self.n_traj_tokens = n_traj_tokens

        # Only reg-x
        self.n_reg_x = n_reg_x
        self.n_total_reg = n_reg_x

        assert self.n_t_layers >= 2, "Need at least 2 t-layers (first and last)"
        assert self.n_reg_x > 0, "Hybrid requires n_reg_x > 0"
        assert not spatial_rope_flag or time_rope_flag, (
            "spatial_rope_flag requires time_rope_flag=True"
        )

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_ct = HardLayerNorm(self.C_T)

        # =============== Mod 2: ctx/trace indicator embedding ===============
        self.ctx_trace_indicator_dim = ctx_trace_indicator_dim
        if ctx_trace_indicator_dim > 0:
            self.ctx_indicator = nn.Parameter(torch.randn(ctx_trace_indicator_dim) * 0.02)
            self.trace_indicator = nn.Parameter(torch.randn(ctx_trace_indicator_dim) * 0.02)
            self.hard_ln_ctx_pos = HardLayerNorm(self.C_T + ctx_trace_indicator_dim)

        # =============== Input Projection ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            from umm.models.umm_v9.world_models.wm_utils import NerfStylePE
            self.x_pe = NerfStylePE(in_channels=1, num_frequencies=x_pe_dim, include_input=True)
            x_proj_in = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.input_dim
        self.x_proj = nn.Linear(x_proj_in, self.C_T)

        # =============== Time/Position Embeddings ===============
        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.C_T)

        self.int_pe = Timesteps(
            self.C_T, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )

        if self.add_time_to_feat:
            self.t_pe = Timesteps(
                self.C_T, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            self.t_proj = nn.Linear(self.C_T + 1, self.C_T)

        # =============== Trajectory PE ===============
        if self.n_traj_tokens > 0:
            self.traj_pe = Timesteps(
                self.C_T, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )

        # =============== Time Embedding Binding (no-RoPE path) ===============
        if not self.time_rope_flag and self.add_time_to_feat:
            self.time_bind_proj = nn.Linear(
                self.C_T + self.C_T + ctx_trace_indicator_dim, self.C_T
            )

        # =============== Conditioning Binders ===============
        self.cond_dim_t = self.feat_dim + self.C_T * 2 + ctx_trace_indicator_dim
        if self.n_traj_tokens > 0:
            self.cond_dim_t += self.C_T

        # t-layer cond binders
        self.t_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_T)
            for _ in range(self.n_t_layers)
        ])

        # p-layer cond binders
        self.p_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_P)
            for _ in range(self.n_p_layers)
        ])

        # =============== Register Embeddings (per-layer, reg-x only) ===============
        self.reg_x_embeds = nn.ParameterList([
            nn.Parameter(torch.empty(1, self.n_reg_x, self.C_T))
            for _ in range(self.n_t_layers)
        ])
        for reg_emb in self.reg_x_embeds:
            nn.init.normal_(reg_emb, std=0.02)

        # =============== Layer-0 Input Mixing ===============
        # 3-way mix: x_state_0 + latest_p_proj + x_state
        self.layer0_mix = nn.ModuleList([
            nn.Linear(3 * self.C_T, self.C_T)
            for _ in range(self.n_t_layers - 1)
        ])

        # =============== t-layer Blocks ===============
        nrx = self.n_reg_x
        _dit_t_kwargs = dict(
            C=self.C_T,
            num_heads=n_head_t,
            num_registers=num_blk_internal_registers,
            mlp_hidden_dims=[self.C_T * fc_factor_t] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_T + (self.ada_extra_dim if self.ctx_pool_adaln else 0),
            qk_norm=qk_norm,
        )
        if self.spatial_rope_flag and self.time_rope_flag:
            _dit_t_kwargs_split = dict(
                n_temporal_heads=n_head_t // 2,
                n_spatial_dims=9,
                **_dit_t_kwargs,
            )
            self.t_blocks = nn.ModuleList([
                DiTBlockSplitRoPE(**_dit_t_kwargs_split)
                for _ in range(self.n_t_layers)
            ])
        else:
            self.t_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_t_kwargs)
                for _ in range(self.n_t_layers)
            ])

        # =============== P-layer Modules (Hybrid) ===============
        _p_mlp_hidden = [self.C_P * fc_factor_p] * fc_n_layers
        _n_slots = self.len_ctx + 1  # ctx slots + X slot
        self.num_blk_internal_registers = num_blk_internal_registers

        # --- Shared P-blocks (one per p-layer, num_registers=0 for shared QKV) ---
        _dit_p_shared_kwargs = dict(
            C=self.C_P,
            num_heads=n_head_p,
            num_registers=0,  # IR handled externally
            mlp_hidden_dims=_p_mlp_hidden,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_P + (self.ada_extra_dim if self.ctx_pool_adaln else 0),
            qk_norm=qk_norm,
        )
        if self.spatial_rope_flag:
            _dit_p_shared_kwargs_split = dict(
                n_temporal_heads=n_head_p // 2,
                n_spatial_dims=9,
                **_dit_p_shared_kwargs,
            )
            _P_Block = DiTBlockSplitRoPE
            _p_shared_block_kwargs = _dit_p_shared_kwargs_split
        else:
            _P_Block = DiTBlockTimeOnly
            _p_shared_block_kwargs = _dit_p_shared_kwargs
        self.shared_p_blocks = nn.ModuleList([
            _P_Block(**_p_shared_block_kwargs)
            for _ in range(self.n_p_layers)
        ])

        # --- IR (Internal Register) embeddings for P-layer ---
        if self.shared_ctx_x:
            # single shared IR embed for ctx and X slots
            self.shared_ir_embed = nn.Parameter(
                torch.randn(1, num_blk_internal_registers, self.C_P) * 0.02
            )
        else:
            # Separate learnable IR for ctx slots vs X slot
            self.ctx_ir_embed = nn.Parameter(
                torch.randn(1, num_blk_internal_registers, self.C_P) * 0.02
            )
            self.x_ir_embed = nn.Parameter(
                torch.randn(1, num_blk_internal_registers, self.C_P) * 0.02
            )

        # --- Pool projections ---
        # ctx slots: extract single ctx timestep's x_state (C_T), project to C_P
        self.ctx_pool_projs = nn.ModuleList()
        for i in range(self.n_p_layers):
            if i == 0:
                self.ctx_pool_projs.append(nn.Linear(self.C_T, self.C_P))
            else:
                self.ctx_pool_projs.append(nn.Linear(self.C_T + self.C_P, self.C_P))

        # X slot: flatten reg-x (nrx * C_T), project to C_P
        self.x_pool_projs = nn.ModuleList()
        for i in range(self.n_p_layers):
            if i == 0:
                self.x_pool_projs.append(nn.Linear(nrx * self.C_T, self.C_P))
            else:
                self.x_pool_projs.append(nn.Linear(nrx * self.C_T + self.C_P, self.C_P))

        # --- Separate FFNs for ctx vs X (or shared when shared_ctx_x=True) ---
        if self.shared_ctx_x:
            # single shared FFN used for both ctx and X slots
            self.shared_p_ffns = nn.ModuleList([
                HybridFFN(self.C_P, _p_mlp_hidden, ada_scale_bias_bound, gate_bound)
                for _ in range(self.n_p_layers)
            ])
        else:
            self.ctx_p_ffns = nn.ModuleList([
                HybridFFN(self.C_P, _p_mlp_hidden, ada_scale_bias_bound, gate_bound)
                for _ in range(self.n_p_layers)
            ])
            self.x_p_ffns = nn.ModuleList([
                HybridFFN(self.C_P, _p_mlp_hidden, ada_scale_bias_bound, gate_bound)
                for _ in range(self.n_p_layers)
            ])

        # --- Unpool projections ---
        # ctx P-out -> C_T (broadcast to ctx positions in x_state)
        self.ctx_unpool_proj = nn.ModuleList([
            nn.Linear(self.C_P, self.C_T)
            for _ in range(self.n_p_layers)
        ])

        # X slot: unpool to reg-x state
        # Per register: cat(reg_x_token[C_T], x_p_out[C_P]) -> FFN -> C_T
        unpool_reg_in = self.C_T + self.C_P
        self.unpool_x_reg_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unpool_reg_in, self.C_T * 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.C_T * 2, self.C_T),
            )
            for _ in range(self.n_p_layers)
        ])

        # X slot: unpool to X token state
        # cat(x_token[C_T], x_p_out[C_P]) -> proj -> C_T
        if self.p_unpool_to_x:
            self.unpool_x_tok_proj = nn.ModuleList([
                nn.Linear(self.C_T + self.C_P, self.C_T)
                for _ in range(self.n_p_layers)
            ])

        # --- P-out -> C_T for layer0_mix (separate for ctx and X) ---
        self.p_to_ct_ctx_proj = nn.ModuleList([
            nn.Linear(self.C_P, self.C_T)
            for _ in range(self.n_p_layers)
        ])
        self.p_to_ct_x_proj = nn.ModuleList([
            nn.Linear(self.C_P, self.C_T)
            for _ in range(self.n_p_layers)
        ])

        # --- ctx pool adaLN projections ---
        if self.ctx_pool_adaln:
            # Projects (max, min, mean) concatenated ctx tokens -> ada_extra_dim
            self.ctx_pool_proj_t = nn.Linear(3 * self.C_T, ada_extra_dim)
            self.ctx_pool_proj_p = nn.Linear(3 * self.C_P, ada_extra_dim)

        # =============== Output Projection ===============
        out_proj_c_in = self.C_T * (self.n_t_layers + 1)
        _seq = []
        for out_proj_c_out in np.linspace(
            out_proj_c_in, self.input_dim, fc_out_n_layers
        )[:-1]:
            out_proj_c_out = int(out_proj_c_out)
            _seq.append(nn.Linear(out_proj_c_in, out_proj_c_out))
            _seq.append(HardLayerNorm(out_proj_c_out))
            _seq.append(nn.LeakyReLU())
            out_proj_c_in = out_proj_c_out
        _seq.append(nn.Linear(out_proj_c_in, self.input_dim))
        self.output_proj = nn.Sequential(*_seq)

        self.__summary__()

    def __summary__(self):
        logging.info("LMM V11 Hybrid Summary:")
        logging.info(f"  C_T (narrow):     {self.C_T}")
        logging.info(f"  C_P (wide):       {self.C_P}")
        logging.info(f"  n_t_layers:       {self.n_t_layers}")
        logging.info(f"  n_p_layers:       {self.n_p_layers}")
        logging.info(f"  n_reg_x:          {self.n_reg_x}")
        logging.info(f"  len_ctx:          {self.len_ctx}")
        logging.info(f"  input_dim:        {self.input_dim}")
        logging.info(f"  feat_dim:         {self.feat_dim}")
        logging.info(f"  time_rope_flag:   {self.time_rope_flag}")
        logging.info(f"  time_bind:        {not self.time_rope_flag and self.add_time_to_feat}")
        logging.info(f"  ctx_trace_ind:    {self.ctx_trace_indicator_dim}")
        logging.info(f"  spatial_rope:     {self.spatial_rope_flag}")
        logging.info(f"  n_traj_tokens:    {self.n_traj_tokens}")
        logging.info(f"  layer_type_string: {self.layer_type_string}")
        logging.info(f"  shared_ctx_x:     {self.shared_ctx_x}")
        logging.info(f"  ctx_pool_adaln:   {self.ctx_pool_adaln}")
        logging.info(f"  p_unpool_to_x:    {self.p_unpool_to_x}")
        if self.ctx_pool_adaln:
            logging.info(f"  ada_extra_dim:    {self.ada_extra_dim}")
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logging.info(f"  params:           {n_params:.2f}M")

    def get_optim_groups(self, weight_decay: float = 0.0, base_lr: Optional[float] = None):
        """Optimizer groups with weight-decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if base_lr is not None:
            for group in optim_groups:
                group["lr"] = base_lr
        return optim_groups

    @torch.compiler.disable
    def _build_hybrid_attn_mask(
        self, N_CTX: int, T: int, R: int, device: torch.device,
    ) -> torch.Tensor:
        """Build the structured attention mask for hybrid t-layer.

        Sequence layout (after DiT internal register prepend):
            [internal_regs(R), reg-x(nrx), ctx(N_CTX), x(T)]

        Attention rules:
          - Internal regs (R): full attention to everything.
          - reg-x <-> reg-x: self-attention.
          - reg-x <-> x: bidirectional.
          - reg-x NOT <-> ctx: no connection.
          - ctx <-> ctx: self-attention.
          - x <-> x: self-attention.
          - ctx <-> x: bidirectional (full dense-style).

        Args:
            N_CTX: Number of context tokens.
            T: Number of x (noisy) tokens.
            R: Number of DiT-internal register tokens.
            device: torch device.

        Returns:
            (L, L) bool mask where True = can attend.
        """
        nrx = self.n_reg_x
        L = R + nrx + N_CTX + T

        mask = torch.zeros(L, L, dtype=torch.bool, device=device)

        # Segment boundaries
        ir_end = R
        rx_end = ir_end + nrx
        ctx_end = rx_end + N_CTX
        # x_end = L

        # Internal registers: full attention to everything
        mask[:ir_end, :] = True
        mask[:, :ir_end] = True

        # reg-x <-> reg-x (self-attention within group)
        mask[ir_end:rx_end, ir_end:rx_end] = True
        # reg-x <-> x (bidirectional)
        mask[ir_end:rx_end, ctx_end:L] = True
        mask[ctx_end:L, ir_end:rx_end] = True

        # ctx <-> ctx (self-attention)
        mask[rx_end:ctx_end, rx_end:ctx_end] = True
        # x <-> x (self-attention)
        mask[ctx_end:L, ctx_end:L] = True

        # ctx <-> x (bidirectional, dense-style)
        mask[rx_end:ctx_end, ctx_end:L] = True
        mask[ctx_end:L, rx_end:ctx_end] = True

        return mask

    @torch.compiler.disable
    def _sample_traj_ids(self, B: int, P: int, device: torch.device) -> torch.Tensor:
        """Sample trajectory IDs for trajectory PE.

        Training: random permutation per batch item.
        Eval: identity ordering (0, 1, ..., P-1).

        Args:
            B: Batch size.
            P: Number of entities.
            device: torch device.

        Returns:
            (B, P) int tensor of trajectory IDs in range [0, n_traj_tokens).
        """
        if self.training:
            ids = torch.stack([
                torch.randperm(self.n_traj_tokens, device=device)[:P]
                for _ in range(B)
            ])  # (B, P)
        else:
            ids = torch.arange(P, device=device)[None].expand(B, -1)  # (B, P)
        return ids

    @torch.compiler.disable
    def _pool_ctx_tokens(
        self,
        x_state: torch.Tensor,
        N_CTX: int,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Pool ctx token features for adaLN conditioning.

        Takes max, min, and mean of the first N_CTX tokens along dim=1,
        concatenates them, and projects to ada_extra_dim.

        Args:
            x_state: (batch, seq, C) token states.
            N_CTX: number of ctx positions at start of seq dimension.
            proj: Linear(3*C, ada_extra_dim).

        Returns:
            (batch, ada_extra_dim) pooled and projected ctx features.
        """
        ctx_tokens = x_state[:, :N_CTX]  # (batch, N_CTX, C)
        pool_max = ctx_tokens.max(dim=1).values   # (batch, C)
        pool_min = ctx_tokens.min(dim=1).values   # (batch, C)
        pool_mean = ctx_tokens.mean(dim=1)         # (batch, C)
        pooled = torch.cat([pool_max, pool_min, pool_mean], dim=-1)  # (batch, 3*C)
        return proj(pooled)  # (batch, ada_extra_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_time: torch.Tensor,
        x_level: torch.Tensor,
        feat: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        ctx_level: Optional[torch.Tensor] = None,
        add_tokens: Optional[torch.Tensor] = None,
        add_tokens_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_tokens_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        x_r_level: Optional[torch.Tensor] = None,
        ctx_r_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with hybrid per-ctx-timestep P-layer.

        Args:
            x: [B, T, P, input_dim] noisy input.
            x_mask: [B, T, P] bool mask.
            x_time: [B, T] timestamps.
            x_level: [B, T, P] diffusion levels.
            feat: [B, P, feat_dim] per-part identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: [B, A_lang, C_P] language tokens.
            add_tokens_mask: [B, A_lang] mask.
            image_tokens: [B, A_img, C_P] image tokens.
            image_tokens_mask: [B, A_img] mask.
            group_ids: unused (kept for interface compat).
            x_r_level: [B, T, P] MeanFlow r-level (optional).
            ctx_r_level: [B, C, P] MeanFlow r-level for context (optional).

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
        device = x.device
        nrx = self.n_reg_x
        len_ctx = self.len_ctx
        n_slots = len_ctx + 1  # total P-layer slots
        x_time = x_time.float().clone() * self.time_multiplier

        # Hybrid requires ctx (P-layer has len_ctx per-ctx-timestep slots)
        assert ctx is not None, (
            f"Hybrid mode requires ctx (len_ctx={len_ctx}). Got ctx=None."
        )

        # ========== Input Processing ==========
        if self.add_time_to_feat:
            ctx_pos_emb = torch.cat(
                [
                    x_time[..., None],
                    self.t_pe(x_time.reshape(-1)).reshape(*x_time.shape[:2], -1),
                ],
                dim=-1,
            )
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # (B, T, C_T)
        else:
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert N_CTX <= self.n_ctx
            assert N_CTX == len_ctx, (
                f"Hybrid requires ctx length ({N_CTX}) == len_ctx ({len_ctx})"
            )
            assert ctx_mask is not None and ctx_level is not None
            x = torch.cat([ctx, x], dim=1)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)
            x_level = torch.cat([ctx_level, x_level], dim=1)
            if x_r_level is not None and ctx_r_level is not None:
                x_r_level = torch.cat([ctx_r_level, x_r_level], dim=1)
            elif x_r_level is not None and ctx_r_level is None:
                # Cannot use x_r_level without ctx_r_level when ctx is provided
                logging.warning(
                    "x_r_level provided but ctx_r_level=None with ctx present; "
                    "discarding x_r_level for this batch"
                )
                x_r_level = None
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # ========== Mod 2: ctx/trace indicator embedding ==========
        if self.ctx_trace_indicator_dim > 0:
            indicator = torch.zeros(
                B, S, self.ctx_trace_indicator_dim,
                device=device, dtype=ctx_pos_emb.dtype,
            )
            indicator[:, :N_CTX] = self.ctx_indicator
            indicator[:, N_CTX:] = self.trace_indicator
            ctx_pos_emb = torch.cat(
                [ctx_pos_emb, indicator], dim=-1
            )  # (B, S, C_T + indicator_dim)

        # Build per-element conditioning: (B, S, P, cond_dim_t)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * P))
            .reshape(B, S, P, self.C_T)
            .to(x)
        )  # (B, S, P, C_T)
        if x_r_level is not None:
            r_emb = (
                self.int_pe(x_r_level.reshape(B * S * P))
                .reshape(B, S, P, self.C_T)
                .to(x)
            )
            level_emb = level_emb + r_emb
        feat_expanded = feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, feat_dim)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(
            -1, -1, P, -1
        )  # (B, S, P, C_T+indicator_dim)

        if self.ctx_trace_indicator_dim > 0:
            ctx_pos_emb_normed = self.hard_ln_ctx_pos(ctx_pos_emb_expanded)
        else:
            ctx_pos_emb_normed = self.hard_ln_ct(ctx_pos_emb_expanded)

        cond_parts = [
            self.hard_ln_feat(feat_expanded),
            self.hard_ln_ct(level_emb),
            ctx_pos_emb_normed,
        ]

        # trajectory PE as 4th conditioning component
        if self.n_traj_tokens > 0:
            traj_ids = self._sample_traj_ids(B, P, device)  # (B, P)
            traj_pe_feat = self.traj_pe(
                traj_ids.reshape(-1).float()
            ).reshape(B, P, self.C_T)  # (B, P, C_T)
            traj_pe_expanded = traj_pe_feat[:, None, :, :].expand(
                -1, S, -1, -1
            )  # (B, S, P, C_T)
            cond_parts.append(self.hard_ln_ct(traj_pe_expanded))

        cond_bsp = torch.cat(cond_parts, dim=-1)  # (B, S, P, cond_dim_t)

        # Embed input
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, P, C_T)

        # ========== Time Embedding Binding (no-RoPE path) ==========
        if not self.time_rope_flag and self.add_time_to_feat:
            time_emb_expanded = ctx_pos_emb[:, :, None, :].expand(
                -1, -1, P, -1
            )  # (B, S, P, C_T + indicator_dim)
            x_emb = self.time_bind_proj(
                torch.cat([x_emb, time_emb_expanded], dim=-1)
            )  # (B, S, P, C_T)

        # Save layer-0 state for mixing
        x_state_0 = x_emb.clone()  # (B, S, P, C_T)

        # ========== Prepare Add Tokens for p-layers ==========
        if add_tokens is not None and image_tokens is not None:
            all_add_tokens = torch.cat([add_tokens, image_tokens], dim=1)
            all_add_mask = torch.cat([add_tokens_mask, image_tokens_mask], dim=1)
        elif add_tokens is not None:
            all_add_tokens = add_tokens
            all_add_mask = add_tokens_mask
        elif image_tokens is not None:
            all_add_tokens = image_tokens
            all_add_mask = image_tokens_mask
        else:
            all_add_tokens = torch.zeros(B, 0, self.C_P, device=device)
            all_add_mask = torch.zeros(B, 0, device=device).bool()
        A = all_add_tokens.shape[1]

        # ========== Build T-layer attention mask ==========
        R = self.t_blocks[0].R  # internal registers in DiT blocks
        struct_mask = self._build_hybrid_attn_mask(N_CTX, T, R, device)  # (L, L)

        # Validity mask for t-layer
        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)
        ir_valid = torch.ones(B * P, R, device=device, dtype=torch.bool)
        reg_valid = torch.ones(B * P, nrx, device=device, dtype=torch.bool)
        full_valid = torch.cat([ir_valid, reg_valid, traj_valid], dim=1)  # (B*P, L)
        val_mask = full_valid[:, None, :] & full_valid[:, :, None]  # (B*P, L, L)
        attn_mask_t = struct_mask[None, :, :] & val_mask  # (B*P, L, L)
        # Force diagonal True
        diag_idx = torch.arange(attn_mask_t.shape[-1], device=device)
        attn_mask_t[:, diag_idx, diag_idx] = True
        attn_mask_t = attn_mask_t[:, None].expand(
            -1, self.n_head_t, -1, -1
        )  # (B*P, H, L, L)

        # ========== Prepare RoPE time ==========
        if self.time_rope_flag:
            t_rope = (
                x_time[:, :, None]
                .expand(-1, -1, P)
                .permute(0, 2, 1)
                .reshape(B * P, T)
            )  # (B*P, T)
        else:
            t_rope = None

        # ========== Spatial positions for split RoPE ==========
        if self.spatial_rope_flag:
            x_9d_spatial = x.detach().permute(0, 2, 1, 3).reshape(B * P, S, 9)
            mean_ctx_9d = (
                compute_entity_mean_9d(x[:, :N_CTX], x_mask[:, :N_CTX])
                if N_CTX > 0 else None
            )
            mean_x_9d = compute_entity_mean_9d(x[:, N_CTX:], x_mask[:, N_CTX:])
            if mean_ctx_9d is None:
                mean_ctx_9d = mean_x_9d
        else:
            x_9d_spatial = None
            mean_ctx_9d = None
            mean_x_9d = None

        # ========== P-layer mask (validity for batched slots) ==========
        p_valid = x_mask.any(dim=1)  # (B, P) - trajectory valid if any time valid

        # ========== Main Loop ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]

        # latest_p_proj for layer0_mix (initialized to zeros)
        latest_p_proj = torch.zeros(
            B, S, P, self.C_T, device=device, dtype=x_emb.dtype
        )  # (B, S, P, C_T)

        # Register state (reg-x only)
        reg_x_state = None  # (B, P, nrx, C_T) or None

        # Previous P-layer outputs for skip connections
        prev_ctx_p_outs = None  # (B, len_ctx, P, C_P) or None
        prev_x_p_out = None  # (B, P, C_P) or None

        t_idx = 0
        p_idx = 0
        for layer_char in self.layer_type_string:
          if layer_char == "t":
            # --------- Layer-0 Mixing ---------
            if t_idx > 0:
                x_state = self.layer0_mix[t_idx - 1](
                    torch.cat([x_state_0, latest_p_proj, x_state], dim=-1)
                )  # (B, S, P, C_T)

            # --------- Build register tokens (additive carry-over) ---------
            rx_learnable = self.reg_x_embeds[t_idx].expand(B * P, -1, -1).reshape(
                B, P, nrx, self.C_T
            )  # (B, P, nrx, C_T)
            rx = rx_learnable if reg_x_state is None else reg_x_state + rx_learnable

            # --------- Reshape for per-trajectory self-attention ---------
            x_traj = x_state.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)

            # Full sequence: [reg-x, ctx+x]
            reg_traj = rx.reshape(B * P, nrx, self.C_T)  # (B*P, nrx, C_T)
            traj_seq = torch.cat(
                [reg_traj, x_traj], dim=1
            )  # (B*P, nrx + S, C_T)

            # --------- Build conditioning for this t-layer ---------
            cond_t = self.t_cond_binders[t_idx](cond_bsp)  # (B, S, P, C_T)
            cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)

            # reg-x conditioning: mean over x tokens
            if N_CTX < S:
                cond_x_mean = cond_t_per_traj[:, N_CTX:].mean(dim=1, keepdim=True)
            else:
                cond_x_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
            cond_reg = cond_x_mean.expand(-1, nrx, -1)  # (B*P, nrx, C_T)

            cond_traj = torch.cat(
                [cond_reg, cond_t_per_traj], dim=1
            )  # (B*P, nrx + S, C_T)

            # ctx_pool_adaln -- append pooled ctx features to cond
            if self.ctx_pool_adaln and N_CTX > 0:
                # x_traj is (B*P, S, C_T); pool from ctx positions [:, :N_CTX]
                ctx_pool_feat_t = self._pool_ctx_tokens(
                    x_traj, N_CTX, self.ctx_pool_proj_t
                )  # (B*P, ada_extra_dim)
                # Expand to match cond_traj shape: (B*P, nrx+S, ada_extra_dim)
                ctx_pool_feat_t = ctx_pool_feat_t[:, None, :].expand(
                    -1, cond_traj.shape[1], -1
                )
                # Cat: (B*P, nrx+S, C_T + ada_extra_dim)
                cond_traj = torch.cat([cond_traj, ctx_pool_feat_t], dim=-1)

            # --------- Run t-block ---------
            if self.spatial_rope_flag:
                self.t_blocks[t_idx].set_spatial_rope_params(
                    spatial_positions=x_9d_spatial,
                    num_external_regs_spatial=nrx,
                )

            traj_out = self.t_blocks[t_idx](
                x=traj_seq,
                cond=cond_traj,
                t=t_rope,
                num_external_regs=nrx + N_CTX,
                attn_mask=attn_mask_t,
            )  # (B*P, nrx + S, C_T)

            # --------- Extract outputs ---------
            reg_x_out = traj_out[:, :nrx]  # (B*P, nrx, C_T)
            x_out = traj_out[:, nrx:]  # (B*P, S, C_T)

            x_state = x_out.reshape(B, P, S, self.C_T).permute(
                0, 2, 1, 3
            )  # (B, S, P, C_T)

            t_skip_list.append(x_state.reshape(B * S, P, self.C_T).clone())
            t_idx += 1

          elif layer_char == "p":
            # --------- Hybrid P-layer: Pool -> Shared Attn -> Separate FFN -> Unpool ---------
            IR = self.num_blk_internal_registers

            # === POOL ===
            # ctx slots: each ctx timestep's x_state (single token per entity)
            # x_state: (B, S, P, C_T), ctx positions are x_state[:, 0:N_CTX]
            ctx_slot_list = []
            for ci in range(len_ctx):
                # Extract ctx timestep ci: (B, 1, P, C_T) -> (B, P, C_T)
                ctx_tok = x_state[:, ci, :, :]  # (B, P, C_T)
                if prev_ctx_p_outs is not None:
                    # Skip: cat previous ctx P-output for this slot
                    ctx_tok = torch.cat(
                        [ctx_tok, prev_ctx_p_outs[:, ci]], dim=-1
                    )  # (B, P, C_T + C_P)
                ctx_pooled = self.ctx_pool_projs[p_idx](ctx_tok)  # (B, P, C_P)
                ctx_slot_list.append(ctx_pooled)

            # X slot: flatten reg-x
            rx_flat = reg_x_out.reshape(B, P, nrx * self.C_T)  # (B, P, nrx*C_T)
            if prev_x_p_out is not None:
                rx_flat = torch.cat(
                    [rx_flat, prev_x_p_out], dim=-1
                )  # (B, P, nrx*C_T + C_P)
            x_pooled = self.x_pool_projs[p_idx](rx_flat)  # (B, P, C_P)

            # Stack all slots: (B, n_slots, P, C_P)
            all_slots = torch.stack(
                ctx_slot_list + [x_pooled], dim=1
            )  # (B, n_slots, P, C_P)

            # === ATTN: Shared QKV across entities ===
            # Prepend IR embeddings per slot type
            if self.shared_ctx_x:
                # single shared IR embed for ctx and X
                ctx_ir = self.shared_ir_embed.expand(B, -1, -1)  # (B, IR, C_P)
                x_ir = self.shared_ir_embed.expand(B, -1, -1)    # (B, IR, C_P)
            else:
                # ctx_ir_embed: (1, IR, C_P), x_ir_embed: (1, IR, C_P)
                ctx_ir = self.ctx_ir_embed.expand(B, -1, -1)  # (B, IR, C_P)
                x_ir = self.x_ir_embed.expand(B, -1, -1)      # (B, IR, C_P)

            # Build per-slot sequences:
            #   ctx slot: [ctx-IR(IR), add_tokens(A), entities(P)]
            #   X slot:   [x-IR(IR), add_tokens(A), entities(P)]
            # Then batch all n_slots together

            # For each slot, build: (B, IR + A + P, C_P)
            slot_seqs = []
            for si in range(n_slots):
                if si < len_ctx:
                    ir_tokens = ctx_ir  # (B, IR, C_P)
                else:
                    ir_tokens = x_ir  # (B, IR, C_P)
                slot_entities = all_slots[:, si]  # (B, P, C_P)
                # seq: [IR, add_tokens, entities]
                seq = torch.cat(
                    [ir_tokens, all_add_tokens, slot_entities], dim=1
                )  # (B, IR + A + P, C_P)
                slot_seqs.append(seq)

            # Stack and reshape for batched attention: (B * n_slots, IR + A + P, C_P)
            batched_seq = torch.stack(slot_seqs, dim=1)  # (B, n_slots, IR+A+P, C_P)
            Bp = B * n_slots
            L_p = IR + A + P
            batched_seq = batched_seq.reshape(Bp, L_p, self.C_P)

            # Build P conditioning
            cond_p_per_point = cond_bsp.mean(dim=1)  # (B, P, cond_dim_t)
            cond_p = self.p_cond_binders[p_idx](cond_p_per_point)  # (B, P, C_P)

            # Expand conditioning for batched slots
            # Each slot gets same entity conditioning (entities are the same P)
            # For add_tokens: use mean across entities
            cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)  # (B, A, C_P)
            cond_ir = cond_p.mean(dim=1, keepdim=True).expand(-1, IR, -1)  # (B, IR, C_P)
            cond_slot = torch.cat(
                [cond_ir, cond_a, cond_p], dim=1
            )  # (B, IR+A+P, C_P)
            # Repeat for n_slots
            cond_batched = cond_slot[:, None].expand(
                -1, n_slots, -1, -1
            ).reshape(Bp, L_p, self.C_P)  # (B*n_slots, IR+A+P, C_P)

            # ctx_pool_adaln -- append pooled ctx features to P cond
            if self.ctx_pool_adaln:
                # Pool from ctx slots: all_slots[:, :len_ctx] is (B, len_ctx, P, C_P)
                ctx_slots_for_pool = all_slots[:, :len_ctx]  # (B, len_ctx, P, C_P)
                # Reshape to (B*P, len_ctx, C_P) for per-entity pooling
                ctx_for_pool = ctx_slots_for_pool.permute(0, 2, 1, 3).reshape(
                    B * P, len_ctx, self.C_P
                )  # (B*P, len_ctx, C_P)
                ctx_pool_feat_p_per_entity = self._pool_ctx_tokens(
                    ctx_for_pool, len_ctx, self.ctx_pool_proj_p
                )  # (B*P, ada_extra_dim)
                # Reshape to (B, P, ada_extra_dim)
                ctx_pool_feat_p = ctx_pool_feat_p_per_entity.reshape(
                    B, P, self.ada_extra_dim
                )

                # Build expanded pool features for cond_batched: (B, IR+A+P, ada_extra_dim)
                ctx_pool_mean_tok = ctx_pool_feat_p.mean(
                    dim=1, keepdim=True
                )  # (B, 1, ada_extra_dim)
                ctx_pool_ir = ctx_pool_mean_tok.expand(-1, IR, -1)  # (B, IR, ada_extra_dim)
                ctx_pool_a = ctx_pool_mean_tok.expand(-1, A, -1)    # (B, A, ada_extra_dim)
                ctx_pool_slot = torch.cat(
                    [ctx_pool_ir, ctx_pool_a, ctx_pool_feat_p], dim=1
                )  # (B, IR+A+P, ada_extra_dim)
                ctx_pool_batched = ctx_pool_slot[:, None].expand(
                    -1, n_slots, -1, -1
                ).reshape(Bp, L_p, self.ada_extra_dim)

                # Cat: (B*n_slots, IR+A+P, C_P + ada_extra_dim)
                cond_batched = torch.cat([cond_batched, ctx_pool_batched], dim=-1)

            # Build P-layer attention mask
            # p_valid: (B, P), add_tokens: always valid for A, IR: always valid
            ir_valid_p = torch.ones(B, IR, device=device, dtype=torch.bool)
            p_seq_valid = torch.cat(
                [ir_valid_p, all_add_mask, p_valid], dim=1
            )  # (B, IR+A+P)
            # Repeat for n_slots
            p_seq_valid_batched = p_seq_valid[:, None].expand(
                -1, n_slots, -1
            ).reshape(Bp, L_p)  # (B*n_slots, IR+A+P)

            # The shared P-block has R=0 (num_registers=0), so left_pad=0
            attn_mask_p = build_attention_mask(
                p_seq_valid_batched, mode="full", left_pad=0,
            )[:, None].expand(
                -1, self.n_head_p, -1, -1
            )  # (B*n_slots, H_p, IR+A+P, IR+A+P)

            # spatial RoPE for P-block
            if self.spatial_rope_flag:
                # For P-block entity tokens: use mean positions
                # IR and add_tokens are external regs from P-block's perspective
                # Entity positions: mean_ctx_9d for ctx slots, mean_x_9d for X slot
                # But since we batch all slots with same entity set, and the
                # P-block does cross-entity attention, use ctx/x means
                _mc = mean_ctx_9d[:, None].expand(-1, n_slots, -1, -1).reshape(
                    Bp, P, 9
                )
                _mx = mean_x_9d[:, None].expand(-1, n_slots, -1, -1).reshape(
                    Bp, P, 9
                )
                self.shared_p_blocks[p_idx].set_dual_spatial_rope_params(
                    pos_a=_mc, pos_b=_mx,
                    num_external_regs=IR + A,
                )

            # Run shared attention (forward_attn_only)
            # num_external_regs = IR + A (both IR and add_tokens are external
            # from DiT's perspective since DiT has R=0)
            x_after_attn, (s2, sc2, g2) = self.shared_p_blocks[p_idx].forward_attn_only(
                x=batched_seq,
                cond=cond_batched,
                t=None,
                num_external_regs=IR + A,
                attn_mask=attn_mask_p,
            )  # x_after_attn: (B*n_slots, IR+A+P, C_P), ada: each (B*n_slots, IR+A+P, C_P)

            # === FFN: Separate for ctx vs X ===
            # Reshape back to (B, n_slots, IR+A+P, C_P)
            x_after_attn_4d = x_after_attn.reshape(B, n_slots, L_p, self.C_P)
            s2_4d = s2.reshape(B, n_slots, L_p, self.C_P)
            sc2_4d = sc2.reshape(B, n_slots, L_p, self.C_P)
            g2_4d = g2.reshape(B, n_slots, L_p, self.C_P)

            # ctx slots: first len_ctx
            ctx_attn = x_after_attn_4d[:, :len_ctx].reshape(
                B * len_ctx, L_p, self.C_P
            )  # (B*len_ctx, IR+A+P, C_P)
            ctx_s2 = s2_4d[:, :len_ctx].reshape(B * len_ctx, L_p, self.C_P)
            ctx_sc2 = sc2_4d[:, :len_ctx].reshape(B * len_ctx, L_p, self.C_P)
            ctx_g2 = g2_4d[:, :len_ctx].reshape(B * len_ctx, L_p, self.C_P)
            if self.shared_ctx_x:
                # use single shared FFN for both ctx and X slots
                ctx_ffn_out = self.shared_p_ffns[p_idx](
                    ctx_attn, ctx_s2, ctx_sc2, ctx_g2
                )  # (B*len_ctx, IR+A+P, C_P)
            else:
                ctx_ffn_out = self.ctx_p_ffns[p_idx](
                    ctx_attn, ctx_s2, ctx_sc2, ctx_g2
                )  # (B*len_ctx, IR+A+P, C_P)

            # X slot: last 1
            x_attn = x_after_attn_4d[:, len_ctx]  # (B, IR+A+P, C_P)
            x_s2 = s2_4d[:, len_ctx]
            x_sc2 = sc2_4d[:, len_ctx]
            x_g2 = g2_4d[:, len_ctx]
            if self.shared_ctx_x:
                # use single shared FFN for both ctx and X slots
                x_ffn_out = self.shared_p_ffns[p_idx](
                    x_attn, x_s2, x_sc2, x_g2
                )  # (B, IR+A+P, C_P)
            else:
                x_ffn_out = self.x_p_ffns[p_idx](
                    x_attn, x_s2, x_sc2, x_g2
                )  # (B, IR+A+P, C_P)

            # Extract entity outputs (strip IR and add_tokens)
            ctx_entity_out = ctx_ffn_out[:, IR + A:].reshape(
                B, len_ctx, P, self.C_P
            )  # (B, len_ctx, P, C_P)
            x_entity_out = x_ffn_out[:, IR + A:]  # (B, P, C_P)

            # Save for skip connections in next P-layer
            prev_ctx_p_outs = ctx_entity_out  # (B, len_ctx, P, C_P)
            prev_x_p_out = x_entity_out  # (B, P, C_P)

            # === UNPOOL ===
            # ctx: project each ctx slot P-out to C_T, add to ctx positions
            # Build ctx update tensor non-inplace: (B, len_ctx, P, C_T)
            ctx_unp_list = []
            for ci in range(len_ctx):
                ctx_p_ci = ctx_entity_out[:, ci]  # (B, P, C_P)
                ctx_unp_list.append(
                    self.ctx_unpool_proj[p_idx](ctx_p_ci)
                )  # (B, P, C_T)
            ctx_unp_stacked = torch.stack(ctx_unp_list, dim=1)  # (B, len_ctx, P, C_T)
            # Non-inplace update: replace ctx portion of x_state
            x_state_ctx_updated = x_state[:, :N_CTX] + ctx_unp_stacked  # (B, N_CTX, P, C_T)

            # X slot: unpool to reg-x state
            # Per register: cat(reg_x_token[C_T], x_p_out[C_P]) -> FFN -> C_T
            x_p_exp = x_entity_out[:, :, None, :].expand(
                -1, -1, nrx, -1
            )  # (B, P, nrx, C_P)
            rx_r = reg_x_out.reshape(B, P, nrx, self.C_T)  # (B, P, nrx, C_T)
            reg_x_state = self.unpool_x_reg_ffn[p_idx](
                torch.cat([rx_r, x_p_exp], dim=-1)
            )  # (B, P, nrx, C_T)

            if self.p_unpool_to_x:
                # X slot: unpool to X token state (x positions in x_state)
                # Mean x_p_out broadcast to all X timesteps
                x_p_tok = x_entity_out  # (B, P, C_P)
                # For each X timestep: cat(x_state_tok[C_T], x_p_out[C_P]) -> proj -> C_T
                x_state_x = x_state[:, N_CTX:]  # (B, T, P, C_T)
                x_p_broadcast = x_p_tok[:, None, :, :].expand(
                    -1, T, -1, -1
                )  # (B, T, P, C_P)
                x_state_x_updated = self.unpool_x_tok_proj[p_idx](
                    torch.cat([x_state_x, x_p_broadcast], dim=-1)
                )  # (B, T, P, C_T)
                # Reassemble x_state non-inplace
                x_state = torch.cat(
                    [x_state_ctx_updated, x_state_x_updated], dim=1
                )  # (B, S, P, C_T)
            else:
                # Skip X-token unpool: keep X positions unchanged
                x_state = torch.cat(
                    [x_state_ctx_updated, x_state[:, N_CTX:]], dim=1
                )  # (B, S, P, C_T)

            # === MIX: Update latest_p_proj for next t-layer's layer0_mix ===
            # ctx positions: project each ctx slot's P-out to C_T
            ctx_p_proj_list = []
            for ci in range(len_ctx):
                cp = self.p_to_ct_ctx_proj[p_idx](
                    ctx_entity_out[:, ci]
                )  # (B, P, C_T)
                ctx_p_proj_list.append(cp)
            # (B, len_ctx, P, C_T)
            ctx_p_proj = torch.stack(ctx_p_proj_list, dim=1)

            # X positions: project X P-out to C_T, broadcast over T timesteps
            x_p_proj = self.p_to_ct_x_proj[p_idx](x_entity_out)  # (B, P, C_T)
            x_p_proj_broadcast = x_p_proj[:, None].expand(
                -1, T, -1, -1
            )  # (B, T, P, C_T)

            # Combine: ctx + X
            latest_p_proj = torch.cat(
                [ctx_p_proj, x_p_proj_broadcast], dim=1
            )  # (B, S, P, C_T)

            p_idx += 1

        # ========== Output: concat t-layer skip outputs ==========
        x_out = torch.cat(t_skip_list, dim=-1)  # (B*S, P, (n_t_layers+1)*C_T)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)

    def inference(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_time: torch.Tensor,
        x_level: torch.Tensor,
        feat: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        ctx_level: Optional[torch.Tensor] = None,
        add_tokens: Optional[torch.Tensor] = None,
        add_tokens_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        image_tokens_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        x_r_level: Optional[torch.Tensor] = None,
        ctx_r_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference method mirroring forward().

        Identical signature and behavior to forward(). Used by visualization
        and eval pipelines to generate predictions from input poses.

        Args:
            (same as forward())

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        return self.forward(
            x=x,
            x_mask=x_mask,
            x_time=x_time,
            x_level=x_level,
            feat=feat,
            ctx=ctx,
            ctx_mask=ctx_mask,
            ctx_level=ctx_level,
            add_tokens=add_tokens,
            add_tokens_mask=add_tokens_mask,
            image_tokens=image_tokens,
            image_tokens_mask=image_tokens_mask,
            group_ids=group_ids,
            x_r_level=x_r_level,
            ctx_r_level=ctx_r_level,
        )


def test_main():
    """Smoke test for MotionMachineV11Hybrid -- various configs."""
    import sys

    B, T, P, C = 2, 8, 64, 9
    CTX = 2  # must match len_ctx
    A = 16
    C_T, C_P = 128, 512

    combos = [
        # 1. Basic hybrid (no RoPE)
        {
            "label": "basic no-RoPE",
            "time_rope_flag": False,
            "ctx_trace_indicator_dim": 0,
            "len_ctx": CTX,
        },
        # 2. With RoPE + indicator
        {
            "label": "RoPE + indicator",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
        },
        # 3. RoPE + spatial RoPE + traj tokens
        {
            "label": "RoPE+spatial+traj",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "spatial_rope_flag": True,
            "n_traj_tokens": 1024,
            "len_ctx": CTX,
        },
        # 4. Custom layer_type_string
        {
            "label": "layer_type_string=tptttpt",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
            "layer_type_string": "tptttpt",
        },
        # 5. Fewer registers (non-default)
        {
            "label": "n_reg_x=2",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "n_reg_x": 2,
            "len_ctx": CTX,
        },
        # 6. len_ctx=3
        {
            "label": "len_ctx=3",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": 3,
        },
        # 7. With add_time_to_feat but no RoPE (time binding path)
        {
            "label": "time_bind no-RoPE",
            "time_rope_flag": False,
            "add_time_to_feat": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
        },
        # 8. shared_ctx_x
        {
            "label": "shared_ctx_x=True",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
            "shared_ctx_x": True,
        },
        # 9. p_unpool_to_x=False
        {
            "label": "p_unpool_to_x=False",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
            "p_unpool_to_x": False,
        },
        # 10. ctx_pool_adaln
        {
            "label": "ctx_pool_adaln=True",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
            "ctx_pool_adaln": True,
            "ada_extra_dim": 64,
        },
        # 11. Both flags together
        {
            "label": "shared_ctx_x + ctx_pool_adaln",
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "len_ctx": CTX,
            "shared_ctx_x": True,
            "ctx_pool_adaln": True,
            "ada_extra_dim": 64,
        },
    ]

    for combo in combos:
        label = combo.pop("label")
        _len_ctx = combo.get("len_ctx", CTX)
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        net = MotionMachineV11Hybrid(
            C_T=C_T,
            C_P=C_P,
            feat_dim=512,
            n_head_t=8,
            n_head_p=16,
            n_t_layers=4,
            n_ctx=16,
            fc_factor_t=4,
            fc_factor_p=2,
            fc_n_layers=2,
            fc_out_n_layers=12,
            num_blk_internal_registers=4,
            ada_scale_bias_bound=1.0,
            gate_bound=2.0,
            add_time_to_feat=combo.pop("add_time_to_feat", True),
            qk_norm=True,
            **combo,
        ).cuda()

        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"Parameters: {n_params:.2f}M")

        x = torch.randn(B, T, P, C).cuda()
        x_mask = torch.ones(B, T, P, dtype=torch.bool).cuda()
        x_time = torch.arange(T).float().cuda()[None].expand(B, -1)
        x_level = torch.randint(0, 100, (B, T, P)).cuda()
        feat = torch.randn(B, P, 512).cuda()

        ctx = torch.randn(B, _len_ctx, P, C).cuda()
        ctx_mask = torch.ones(B, _len_ctx, P, dtype=torch.bool).cuda()
        ctx_level = torch.zeros(B, _len_ctx, P, dtype=torch.long).cuda()

        add_tokens = torch.randn(B, A, C_P).cuda()
        add_tokens_mask = torch.ones(B, A, dtype=torch.bool).cuda()

        out = net(
            x, x_mask, x_time, x_level, feat,
            ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
            add_tokens=add_tokens, add_tokens_mask=add_tokens_mask,
        )

        expected = (B, T + _len_ctx, P, C)
        print(f"Output shape: {out.shape}")
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

        # Test backward
        loss = out.sum()
        loss.backward()
        print("Backward: PASS")

        # Test inference method
        net.eval()
        with torch.no_grad():
            out_inf = net.inference(
                x, x_mask, x_time, x_level, feat,
                ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
                add_tokens=add_tokens, add_tokens_mask=add_tokens_mask,
            )
        assert out_inf.shape == expected, f"Inference shape mismatch: {out_inf.shape}"
        print("Inference: PASS")
        net.train()

        # Reset grads for next combo
        net.zero_grad(set_to_none=True)
        print("PASS")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_main()
