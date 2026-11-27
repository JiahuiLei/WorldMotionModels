"""MotionMachine V11 Sliced -- chunk-based register mechanism.

Based on MotionMachineV11Compact with a fundamentally different register
mechanism: instead of typed per-layer registers (reg-c, reg-x, reg-a),
uses chunk-based ("sliced") registers.

Key differences from compact:
  1. Two global learnable init tokens (reg_ctx_init, reg_x_init) instead of
     per-layer-per-type ParameterLists.  Layer 0 replicates them;
     layer N>0 reuses previous output directly (no additive learnable).
  2. Asymmetric chunk-based attention mask: each register attends to ALL
     registers + its assigned chunk of tokens; each token attends to its
     chunk peers + its chunk's register ONLY.
  3. P-layer operates per-entity across registers (B*P, A+n_reg_total, C_P)
     instead of across entities (B, A+P, C_P).
  4. Mod 4 layer0_mix uses per-chunk broadcast from register P outputs.
"""

import logging
import math
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
from umm.models.umm_v11.world_models.dit_mot import DiTBlockMoT
from umm.models.umm_v11.world_models.so3_utils import compute_entity_mean_9d


class MotionMachineV11Sliced(nn.Module):
    """Chunk-based register MotionMachine (V11 Sliced).

    Layer pattern: t-p-t-p-...-t (alternating, first and last are t-layers).
    n_t_layers t-layers and n_p_layers = n_t_layers - 1 p-layers.

    Register mechanism: instead of typed per-layer registers, uses two global
    init tokens (reg_ctx_init, reg_x_init) that are replicated to cover
    ceil(N_CTX/G) and ceil(T/G) chunks respectively.  Each register attends
    to all other registers + its chunk of tokens.  Each token attends to its
    chunk peers + its chunk's register only.

    Args:
        C_T: Narrow channel width for t-layers.
        C_P: Wide channel width for p-layers.
        feat_dim: Input feature dimension from identify layer.
        n_head_t: Number of attention heads for t-layers.
        n_head_p: Number of attention heads for p-layers.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        n_t_layers: Number of t-layers. Must be >= 2.
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
        ctx_trace_indicator_dim: Mod 2 -- dimension of ctx/trace indicator
            embedding.  0 = disabled.
        spatial_rope_flag: split spatial RoPE + trajectory PE.
        n_traj_tokens: number of trajectory tokens.
        layer_type_string: configurable layer pattern (e.g. "tptptpt").
        slice_group_size: G -- number of tokens per register chunk.
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
        # configurable layer pattern string
        layer_type_string: Optional[str] = None,
        # Sliced register params
        slice_group_size: int = 4,
    ):
        super().__init__()

        self.C_T = C_T
        self.C_P = C_P
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.slice_group_size = slice_group_size

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

        assert self.n_t_layers >= 2, "Need at least 2 t-layers (first and last)"

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

        self.t_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_T)
            for _ in range(self.n_t_layers)
        ])

        self.p_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_P)
            for _ in range(self.n_p_layers)
        ])

        # =============== Sliced Register Embeddings ===============
        # Two global learnable init tokens (shape 1,1,C_T), replicated at runtime.
        self.reg_ctx_init = nn.Parameter(torch.randn(1, 1, C_T) * 0.02)
        self.reg_x_init = nn.Parameter(torch.randn(1, 1, C_T) * 0.02)

        # =============== Layer-0 Input Mixing ===============
        # Mod 4: 3-way mix: x_state_0 + p_proj_broadcast + current_x_state
        self.layer0_mix = nn.ModuleList([
            nn.Linear(3 * self.C_T, self.C_T)
            for _ in range(self.n_t_layers - 1)
        ])

        # =============== Mod 4: p-layer output projection (C_P -> C_T) ===============
        self.p_to_ct_projs = nn.ModuleList([
            nn.Linear(self.C_P, self.C_T)
            for _ in range(self.n_p_layers)
        ])

        # =============== t-layer Blocks ===============
        _dit_t_kwargs = dict(
            C=self.C_T,
            num_heads=n_head_t,
            num_registers=num_blk_internal_registers,
            mlp_hidden_dims=[self.C_T * fc_factor_t] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_T,
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

        # =============== P-layer Blocks ===============
        _dit_p_kwargs = dict(
            C=self.C_P,
            num_heads=n_head_p,
            num_registers=num_blk_internal_registers,
            mlp_hidden_dims=[self.C_P * fc_factor_p] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_P,
            qk_norm=qk_norm,
        )
        if self.spatial_rope_flag:
            _dit_p_kwargs_split = dict(
                n_temporal_heads=n_head_p // 2,
                n_spatial_dims=9,
                **_dit_p_kwargs,
            )
            _P_Block = DiTBlockSplitRoPE
            _p_block_kwargs = _dit_p_kwargs_split
        else:
            _P_Block = DiTBlockTimeOnly
            _p_block_kwargs = _dit_p_kwargs

        self.p_blocks = nn.ModuleList([
            _P_Block(**_p_block_kwargs)
            for _ in range(self.n_p_layers)
        ])

        # =============== Pool Projections (register C_T -> C_P) ===============
        # Layer 0: Linear(C_T, C_P), layer i>0: Linear(C_T + C_P, C_P) with skip
        self.pool_projs = nn.ModuleList()
        for i in range(self.n_p_layers):
            in_dim = self.C_T if i == 0 else self.C_T + self.C_P
            self.pool_projs.append(nn.Linear(in_dim, self.C_P))

        # =============== Unpool FFNs: cat(reg_C_T, p_out_C_P) -> C_T ===============
        # Shared per P-layer, applied element-wise to all registers
        unpool_in = self.C_T + self.C_P
        self.unpool_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unpool_in, self.C_T * 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.C_T * 2, self.C_T),
            )
            for _ in range(self.n_p_layers)
        ])

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
        logging.info("LMM V11 Sliced Summary:")
        logging.info(f"  C_T (narrow):     {self.C_T}")
        logging.info(f"  C_P (wide):       {self.C_P}")
        logging.info(f"  n_t_layers:       {self.n_t_layers}")
        logging.info(f"  n_p_layers:       {self.n_p_layers}")
        logging.info(f"  slice_group_size: {self.slice_group_size}")
        logging.info(f"  input_dim:        {self.input_dim}")
        logging.info(f"  feat_dim:         {self.feat_dim}")
        logging.info(f"  time_rope_flag:   {self.time_rope_flag}")
        logging.info(f"  time_bind:        {not self.time_rope_flag and self.add_time_to_feat}")
        logging.info(f"  ctx_trace_ind:    {self.ctx_trace_indicator_dim}")
        logging.info(f"  spatial_rope:     {self.spatial_rope_flag}")
        logging.info(f"  n_traj_tokens:    {self.n_traj_tokens}")
        logging.info(f"  layer_type_string: {self.layer_type_string}")
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
    def _build_sliced_attn_mask(
        self,
        N_CTX: int,
        T: int,
        n_reg_ctx: int,
        n_reg_x: int,
        R: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build chunk-based (sliced) attention mask for the t-layer.

        Sequence layout (after DiT internal register prepend):
            [internal_regs(R), reg_ctx(n_reg_ctx), reg_x(n_reg_x), ctx(N_CTX), x(T)]

        Attention rules:
          Internal registers (R): full attention to everything.
          reg_ctx[i]: attends ALL registers + ctx_tokens[i*G : min((i+1)*G, N_CTX)].
          reg_x[j]:  attends ALL registers + x_tokens[j*G : min((j+1)*G, T)].
          ctx_token[k]: chunk i = k // G. Attends ctx_tokens[i*G:min((i+1)*G, N_CTX)]
                        + reg_ctx[i] ONLY.
          x_token[k]:  chunk j = k // G. Attends x_tokens[j*G:min((j+1)*G, T)]
                        + reg_x[j] ONLY.

        Args:
            N_CTX: Number of context tokens.
            T: Number of x (noisy) tokens.
            n_reg_ctx: Number of ctx-chunk registers (ceil(N_CTX/G) or 0).
            n_reg_x: Number of x-chunk registers (ceil(T/G)).
            R: Number of DiT-internal register tokens.
            device: torch device.

        Returns:
            (L, L) bool mask where True = can attend.
            L = R + n_reg_ctx + n_reg_x + N_CTX + T.
        """
        G = self.slice_group_size
        n_reg_total = n_reg_ctx + n_reg_x
        L = R + n_reg_total + N_CTX + T

        mask = torch.zeros(L, L, dtype=torch.bool, device=device)

        # Segment boundaries
        ir_end = R  # end of internal registers
        rc_end = ir_end + n_reg_ctx  # end of ctx registers
        rx_end = rc_end + n_reg_x  # end of x registers
        ctx_end = rx_end + N_CTX  # end of ctx tokens
        # x_end = L  # end of x tokens

        # Internal registers: full attention to everything
        mask[:ir_end, :] = True
        mask[:, :ir_end] = True

        # --- Register rows ---

        # All sliced registers attend to ALL registers (including each other)
        # reg_ctx and reg_x self-attention + cross-attention among registers
        mask[ir_end:rx_end, ir_end:rx_end] = True

        # reg_ctx[i] -> ctx_tokens in chunk i
        for i in range(n_reg_ctx):
            chunk_start = rx_end + i * G  # offset into ctx tokens region
            chunk_end = rx_end + min((i + 1) * G, N_CTX)
            reg_row = ir_end + i
            # Register attends to its chunk of ctx tokens
            mask[reg_row, chunk_start:chunk_end] = True
            # Bidirectional: chunk tokens attend to this register
            mask[chunk_start:chunk_end, reg_row] = True

        # reg_x[j] -> x_tokens in chunk j
        for j in range(n_reg_x):
            chunk_start = ctx_end + j * G  # offset into x tokens region
            chunk_end = ctx_end + min((j + 1) * G, T)
            reg_row = ir_end + n_reg_ctx + j
            # Register attends to its chunk of x tokens
            mask[reg_row, chunk_start:chunk_end] = True
            # Bidirectional: chunk tokens attend to this register
            mask[chunk_start:chunk_end, reg_row] = True

        # --- Token rows ---

        # ctx_token[k]: attends to ctx_tokens in same chunk + reg_ctx[chunk_id]
        for i in range(n_reg_ctx):
            cs = rx_end + i * G
            ce = rx_end + min((i + 1) * G, N_CTX)
            # Intra-chunk full attention
            mask[cs:ce, cs:ce] = True

        # x_token[k]: attends to x_tokens in same chunk + reg_x[chunk_id]
        for j in range(n_reg_x):
            cs = ctx_end + j * G
            ce = ctx_end + min((j + 1) * G, T)
            # Intra-chunk full attention
            mask[cs:ce, cs:ce] = True

        return mask

    @torch.compiler.disable
    def _sample_traj_ids(self, B: int, P: int, device: torch.device) -> torch.Tensor:
        """Sample trajectory IDs for trajectory PE.

        Training: random permutation per batch item (shuffle entity ordering).
        Eval: identity ordering (0, 1, 2, ..., P-1).

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
        """Forward pass with chunk-based (sliced) register temporal attention.

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
            x_r_level: [B, T, P] MeanFlow r-level (optional).
            ctx_r_level: [B, C, P] MeanFlow r-level for context (optional).

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
        device = x.device
        G = self.slice_group_size
        x_time = x_time.float().clone() * self.time_multiplier

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
            assert ctx_mask is not None and ctx_level is not None
            x = torch.cat([ctx, x], dim=1)  # (B, N_CTX+T, P, input_dim)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)  # (B, N_CTX+T, P)
            x_level = torch.cat([ctx_level, x_level], dim=1)  # (B, N_CTX+T, P)
            if x_r_level is not None and ctx_r_level is not None:
                x_r_level = torch.cat([ctx_r_level, x_r_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)  # (B, N_CTX, C_T)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)  # (B, S, C_T)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # ========== Mod 2: ctx/trace indicator embedding ==========
        if self.ctx_trace_indicator_dim > 0:
            indicator = torch.zeros(
                B, S, self.ctx_trace_indicator_dim,
                device=device, dtype=ctx_pos_emb.dtype,
            )  # (B, S, indicator_dim)
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
        A = all_add_tokens.shape[1]  # total number of add tokens

        # ========== Compute sliced register counts ==========
        # Use integer arithmetic instead of math.ceil for torch.compile SymInt safety
        n_reg_ctx = (N_CTX + G - 1) // G if N_CTX > 0 else 0
        n_reg_x = (T + G - 1) // G
        n_reg_total = n_reg_ctx + n_reg_x

        # ========== Build attention masks ==========
        R = self.t_blocks[0].R  # internal registers in DiT blocks

        # Sliced attention mask: precomputed once (N_CTX, T fixed within batch)
        struct_mask = self._build_sliced_attn_mask(
            N_CTX, T, n_reg_ctx, n_reg_x, R, device
        )  # (L_t, L_t) where L_t = R + n_reg_total + S

        L_t = R + n_reg_total + S

        # Validity mask for t-layer
        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)
        ir_valid = torch.ones(B * P, R, device=device, dtype=torch.bool)  # (B*P, R)
        reg_valid = torch.ones(
            B * P, n_reg_total, device=device, dtype=torch.bool
        )  # (B*P, n_reg_total)
        full_valid = torch.cat(
            [ir_valid, reg_valid, traj_valid], dim=1
        )  # (B*P, L_t)
        val_mask = full_valid[:, None, :] & full_valid[:, :, None]  # (B*P, L_t, L_t)
        attn_mask_t = struct_mask[None, :, :] & val_mask  # (B*P, L_t, L_t)

        # Force diagonal True
        diag_idx = torch.arange(L_t, device=device)
        attn_mask_t[:, diag_idx, diag_idx] = True
        attn_mask_t = attn_mask_t[:, None].expand(
            -1, self.n_head_t, -1, -1
        )  # (B*P, H_t, L_t, L_t)

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
            x_9d_spatial = x.detach().permute(0, 2, 1, 3).reshape(
                B * P, S, 9
            )  # (B*P, S, 9)
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

        # ========== Main Loop ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]

        # Mod 4: latest p-layer projection for layer0_mix (per-chunk broadcast)
        latest_p_proj = torch.zeros(
            B, S, P, self.C_T, device=device, dtype=x_emb.dtype
        )  # (B, S, P, C_T)

        # Register states: (B*P, n_reg_total, C_T)
        # Initialized from learnable inits at layer 0
        reg_state = None  # set before first t-layer
        prev_p_reg_out = None  # (B*P, n_reg_total, C_P) from previous p-layer

        t_idx = 0
        p_idx = 0
        for layer_char in self.layer_type_string:
          if layer_char == "t":
            # --------- Layer-0 Mixing ---------
            if t_idx > 0:
                x_state = self.layer0_mix[t_idx - 1](
                    torch.cat([x_state_0, latest_p_proj, x_state], dim=-1)
                )  # (B, S, P, C_T)

            # --------- Build register tokens ---------
            if t_idx == 0:
                # Layer 0: replicate init tokens
                reg_ctx_tokens = self.reg_ctx_init.expand(
                    B * P, n_reg_ctx, self.C_T
                )  # (B*P, n_reg_ctx, C_T)
                reg_x_tokens = self.reg_x_init.expand(
                    B * P, n_reg_x, self.C_T
                )  # (B*P, n_reg_x, C_T)
                reg_tokens = torch.cat(
                    [reg_ctx_tokens, reg_x_tokens], dim=1
                )  # (B*P, n_reg_total, C_T)
            else:
                # Layer N>0: reuse previous output directly (no additive)
                reg_tokens = reg_state  # (B*P, n_reg_total, C_T)

            # --------- Reshape for per-trajectory self-attention ---------
            x_traj = x_state.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)

            # Full sequence: [reg_ctx, reg_x, ctx+x]
            traj_seq = torch.cat(
                [reg_tokens, x_traj], dim=1
            )  # (B*P, n_reg_total + S, C_T)

            # --------- Build conditioning for this t-layer ---------
            cond_t = self.t_cond_binders[t_idx](cond_bsp)  # (B, S, P, C_T)
            cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)

            # Register conditioning: mean over all tokens (simple approach)
            cond_all_mean = cond_t_per_traj.mean(
                dim=1, keepdim=True
            )  # (B*P, 1, C_T)
            cond_reg = cond_all_mean.expand(
                -1, n_reg_total, -1
            )  # (B*P, n_reg_total, C_T)

            cond_traj = torch.cat(
                [cond_reg, cond_t_per_traj], dim=1
            )  # (B*P, n_reg_total + S, C_T)

            # --------- Run t-block ---------
            if self.spatial_rope_flag:
                self.t_blocks[t_idx].set_spatial_rope_params(
                    spatial_positions=x_9d_spatial,
                    num_external_regs_spatial=n_reg_total,
                )

            traj_out = self.t_blocks[t_idx](
                x=traj_seq,
                cond=cond_traj,
                t=t_rope,
                num_external_regs=n_reg_total + N_CTX,
                attn_mask=attn_mask_t,
            )  # (B*P, n_reg_total + S, C_T)

            # --------- Extract outputs ---------
            reg_state = traj_out[:, :n_reg_total]  # (B*P, n_reg_total, C_T)
            x_out = traj_out[:, n_reg_total:]  # (B*P, S, C_T)

            x_state = x_out.reshape(B, P, S, self.C_T).permute(
                0, 2, 1, 3
            )  # (B, S, P, C_T)

            t_skip_list.append(x_state.reshape(B * S, P, self.C_T).clone())
            t_idx += 1

          elif layer_char == "p":
            # --------- Pool + P-layer + Unpool ---------

            # Pool: each register token (C_T) -> C_P
            if p_idx == 0:
                pool_in = reg_state  # (B*P, n_reg_total, C_T)
            else:
                # Skip connection from previous P-layer output
                pool_in = torch.cat(
                    [reg_state, prev_p_reg_out], dim=-1
                )  # (B*P, n_reg_total, C_T + C_P)
            pooled_regs = self.pool_projs[p_idx](pool_in)  # (B*P, n_reg_total, C_P)

            # Expand add_tokens for per-entity: (B, A, C_P) -> (B*P, A, C_P)
            add_tokens_expanded = all_add_tokens[:, None].expand(
                -1, P, -1, -1
            ).reshape(B * P, A, self.C_P)  # (B*P, A, C_P)

            # Build P-layer sequence: [add_tokens, registers]
            p_seq = torch.cat(
                [add_tokens_expanded, pooled_regs], dim=1
            )  # (B*P, A + n_reg_total, C_P)

            # Build conditioning for P
            cond_p_per_point = cond_bsp.mean(dim=1)  # (B, P, cond_dim_t)
            cond_p = self.p_cond_binders[p_idx](cond_p_per_point)  # (B, P, C_P)
            # Reshape to per-entity: (B*P, 1, C_P) then expand for registers
            cond_p_flat = cond_p.reshape(B * P, 1, self.C_P)  # (B*P, 1, C_P)
            cond_p_regs = cond_p_flat.expand(
                -1, n_reg_total, -1
            )  # (B*P, n_reg_total, C_P)
            # For add_tokens: use mean across entities
            cond_p_add = cond_p_flat.expand(
                -1, A, -1
            )  # (B*P, A, C_P)
            cond_p_seq = torch.cat(
                [cond_p_add, cond_p_regs], dim=1
            )  # (B*P, A + n_reg_total, C_P)

            # P-layer attention mask: all registers always valid
            # Build validity: add_tokens_mask expanded + all-True for registers
            add_mask_expanded = all_add_mask[:, None].expand(
                -1, P, -1
            ).reshape(B * P, A)  # (B*P, A)
            reg_p_valid = torch.ones(
                B * P, n_reg_total, device=device, dtype=torch.bool
            )  # (B*P, n_reg_total)
            p_seq_valid = torch.cat(
                [add_mask_expanded, reg_p_valid], dim=1
            )  # (B*P, A + n_reg_total)

            R_p = self.p_blocks[p_idx].R
            attn_mask_p = build_attention_mask(
                p_seq_valid, mode="full", left_pad=R_p,
            )[:, None].expand(
                -1, self.n_head_p, -1, -1
            )  # (B*P, H_p, R_p+A+n_reg_total, R_p+A+n_reg_total)

            # set dual spatial RoPE for P-block
            if self.spatial_rope_flag:
                # P-layer entity tokens are registers (n_reg_total per entity).
                # Assign each register its entity's mean 9D position:
                #   ctx registers -> mean_ctx_9d, x registers -> mean_x_9d.
                # mean_ctx_9d: (B, P, 9), mean_x_9d: (B, P, 9)
                # Need: (B*P, n_reg_total, 9) for both pos_a and pos_b
                _mc = mean_ctx_9d.reshape(B * P, 1, 9).expand(
                    -1, n_reg_total, -1
                )  # (B*P, n_reg_total, 9)
                _mx = mean_x_9d.reshape(B * P, 1, 9).expand(
                    -1, n_reg_total, -1
                )  # (B*P, n_reg_total, 9)
                self.p_blocks[p_idx].set_dual_spatial_rope_params(
                    pos_a=_mc,
                    pos_b=_mx,
                    num_external_regs=A,
                )

            p_out_seq = self.p_blocks[p_idx](
                x=p_seq, cond=cond_p_seq, t=None,
                num_external_regs=A, attn_mask=attn_mask_p,
            )  # (B*P, A + n_reg_total, C_P)

            # Extract register P outputs (skip add_tokens portion)
            p_reg_out = p_out_seq[:, A:]  # (B*P, n_reg_total, C_P)
            prev_p_reg_out = p_reg_out

            # Unpool: cat(reg_C_T, p_out_C_P) -> FFN -> C_T
            unpool_in = torch.cat(
                [reg_state, p_reg_out], dim=-1
            )  # (B*P, n_reg_total, C_T + C_P)
            reg_state = self.unpool_ffns[p_idx](unpool_in)  # (B*P, n_reg_total, C_T)

            # --------- Mod 4: per-chunk broadcast of P output ---------
            # Project P output (C_P) to C_T for each register
            p_proj_regs = self.p_to_ct_projs[p_idx](
                p_reg_out
            )  # (B*P, n_reg_total, C_T)

            # Split into ctx and x register projections
            p_proj_ctx = p_proj_regs[:, :n_reg_ctx]  # (B*P, n_reg_ctx, C_T)
            p_proj_x = p_proj_regs[:, n_reg_ctx:]  # (B*P, n_reg_x, C_T)

            # Broadcast: each register covers G tokens, last covers remainder
            if N_CTX > 0 and n_reg_ctx > 0:
                p_ctx_broadcast = p_proj_ctx.repeat_interleave(
                    G, dim=1
                )[:, :N_CTX]  # (B*P, N_CTX, C_T)
            else:
                p_ctx_broadcast = torch.zeros(
                    B * P, 0, self.C_T, device=device, dtype=x_emb.dtype
                )

            p_x_broadcast = p_proj_x.repeat_interleave(
                G, dim=1
            )[:, :T]  # (B*P, T, C_T)

            latest_p_proj_flat = torch.cat(
                [p_ctx_broadcast, p_x_broadcast], dim=1
            )  # (B*P, S, C_T)

            # Reshape to (B, S, P, C_T) for the layer0_mix
            latest_p_proj = latest_p_proj_flat.reshape(
                B, P, S, self.C_T
            ).permute(0, 2, 1, 3)  # (B, S, P, C_T)

            p_idx += 1

        # ========== Output: concat t-layer skip outputs ==========
        # t_skip_list: list of (B*S, P, C_T) tensors, len = n_t_layers + 1
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
    """Smoke test for MotionMachineV11Sliced -- various configs."""
    import sys

    B, T, P, C = 2, 8, 64, 9
    CTX = 3
    A = 16
    C_T, C_P = 128, 512

    combos = [
        # 1. Basic sliced (G=4, no RoPE)
        {
            "label": "basic G=4 no-RoPE",
            "slice_group_size": 4,
            "time_rope_flag": False,
            "ctx_trace_indicator_dim": 0,
        },
        # 2. Sliced with RoPE
        {
            "label": "G=4 with RoPE",
            "slice_group_size": 4,
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
        },
        # 3. Sliced with RoPE + spatial RoPE + traj tokens
        {
            "label": "G=4 RoPE+spatial+traj",
            "slice_group_size": 4,
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "spatial_rope_flag": True,
            "n_traj_tokens": 1024,
        },
        # 4a. G=2
        {
            "label": "G=2",
            "slice_group_size": 2,
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
        },
        # 4b. G=8
        {
            "label": "G=8",
            "slice_group_size": 8,
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
        },
        # 5. Mod 2 indicator
        {
            "label": "Mod2 indicator G=4",
            "slice_group_size": 4,
            "time_rope_flag": False,
            "ctx_trace_indicator_dim": 32,
        },
        # 6. Custom layer_type_string
        {
            "label": "layer_type_string=tptttpt",
            "slice_group_size": 4,
            "time_rope_flag": True,
            "ctx_trace_indicator_dim": 16,
            "layer_type_string": "tptttpt",
        },
    ]

    for combo in combos:
        label = combo.pop("label")
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        net = MotionMachineV11Sliced(
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
            add_time_to_feat=True,
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

        ctx = torch.randn(B, CTX, P, C).cuda()
        ctx_mask = torch.ones(B, CTX, P, dtype=torch.bool).cuda()
        ctx_level = torch.zeros(B, CTX, P, dtype=torch.long).cuda()

        add_tokens = torch.randn(B, A, C_P).cuda()
        add_tokens_mask = torch.ones(B, A, dtype=torch.bool).cuda()

        out = net(
            x, x_mask, x_time, x_level, feat,
            ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
            add_tokens=add_tokens, add_tokens_mask=add_tokens_mask,
        )

        expected = (B, T + CTX, P, C)
        print(f"Output shape: {out.shape}")
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"
        print("PASS")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_main()
