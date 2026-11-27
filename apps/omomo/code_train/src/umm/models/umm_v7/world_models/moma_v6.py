"""MotionMachine V7 -- Register-based temporal attention with structured P-layer.

Architecture: alternating t-p layers (first and last are t-layers).
Pattern: t-p-t-p-t-p-t

Key changes from V6:
- 3 register types (reg-c, reg-x, reg-a) in t-layer with structured attention mask
- Structured P-layer: sub-P-c (half-width) + sub-P-x (half-width) + main P
- Layer-0 input mixing via learned linear for ctx+x tokens (preserves identity)
- C_T=128 (narrow), C_P=512 (wide), fc_factor_t=4

- **t-layer**: Per-trajectory self-attention on [reg-c, reg-x, reg-a, ctx, x].
  reg-c only talks to ctx, reg-x only talks to x, reg-a talks to ctx+x.
  No cross-group register interaction.

- **Pooling (t->p)**: 3 separate pools for reg-c (C_P/2), reg-x (C_P/2), reg-a (C_P).

- **Sub-P-c/x**: Half-width transformers processing reg-c/reg-x pooled features.

- **Merge**: cat(sub-P-c, sub-P-x, reg-a) -> 2*C_P -> Linear -> C_P.

- **Main P**: Full attention on [add_tokens; merged] at C_P.

- **Unpool**: 3 separate FFNs. reg-c and reg-x get skip from sub-P output.

- **Layer-0 mixing**: At each layer > 0, ctx+x tokens are mixed with their layer-0
  representation via a learned linear (not direct addition).

Example::

    net = MotionMachineV7(C_T=128, C_P=512, feat_dim=512, n_t_layers=4)
    out = net(x, x_mask, x_time, x_level, feat)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v7.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
)
from umm.models.umm_v7.world_models.dit_atten import DiTBlockTimeOnly


class MotionMachineV7(nn.Module):
    """Register-based temporal attention MotionMachine.

    Layer pattern: t-p-t-p-...-t (alternating, first and last are t-layers).
    n_t_layers t-layers and n_p_layers = n_t_layers - 1 p-layers.

    Args:
        C_T: Narrow channel width for t-layers.
        C_P: Wide channel width for p-layers.
        feat_dim: Input feature dimension from identify layer.
        n_head_t: Number of attention heads for t-layers.
        n_head_p: Number of attention heads for p-layers.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        n_t_layers: Number of t-layers. Must be >= 2.
        n_reg: Number of register tokens per type (reg-c, reg-x, reg-a).
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
        isolated_ctx_x: If True, ctx and x cannot attend to each other directly
            (Option B -- they communicate only through reg-a). Default False (Option A).
        qk_norm: Use QK normalization in attention.
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
        n_reg: int = 4,
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
        isolated_ctx_x: bool = False,
        unified_p_layer: bool = False,
        full_attention_t: bool = False,
        parallel_p_layer: bool = False,
        n_reg_c: int = -1,
        n_reg_x: int = -1,
        n_reg_a: int = -1,
        qk_norm: bool = False,
        #
        latent_dim: int = -1,
        # Unused (compat with agent config parsing)
        cond_binder_dim: int = -1,
        num_register: int = 0,
        use_reg_binder: bool = False,
        add_first_n_add_token_to_feat: int = -1,
        image_cross_attn_layers: str = "",
        layer_type_string: str = "",
        skip_layer_idx: Optional[list] = None,
        n_latent_queries: int = -1,
        q_pool_mode: str = "",
        fc_factor: int = 2,
        x_pe_dim: int = 0,
        # V6 compat flags (ignored in V7)
        dual_t_layer: bool = False,
        traj_pos_emb: bool = False,
        ctx_p_layer: bool = False,
        dense_p_block_size: int = 0,
        n_dense_p_layers: int = -1,
        num_random_trace_id_pe: int = -1,
        n_Q: int = 4,
    ):
        super().__init__()

        self.C_T = C_T
        self.C_P = C_P
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.n_t_layers = n_t_layers
        self.n_p_layers = n_t_layers - 1
        self.n_reg = n_reg
        self.n_head_t = n_head_t
        self.n_head_p = n_head_p
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat
        self.isolated_ctx_x = isolated_ctx_x
        self.unified_p_layer = unified_p_layer
        self.full_attention_t = full_attention_t
        self.parallel_p_layer = parallel_p_layer

        # Per-type register counts (for parallel_p_layer mode)
        if parallel_p_layer:
            assert n_reg_c > 0 and n_reg_x > 0 and n_reg_a > 0, (
                "parallel_p_layer requires n_reg_c, n_reg_x, n_reg_a > 0"
            )
            self.n_reg_c = n_reg_c
            self.n_reg_x = n_reg_x
            self.n_reg_a = n_reg_a
            self.n_total_reg = n_reg_c + n_reg_x + n_reg_a
        else:
            # Standard mode: n_reg tokens per type, 3 types
            self.n_reg_c = n_reg
            self.n_reg_x = n_reg
            self.n_reg_a = n_reg
            self.n_total_reg = 3 * n_reg

        assert n_t_layers >= 2, "Need at least 2 t-layers (first and last)"

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_ct = HardLayerNorm(self.C_T)

        # =============== Input Projection ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            from umm.models.umm_v7.world_models.wm_utils import NerfStylePE
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

        # =============== Conditioning Binders ===============
        # t-layer conditioning: feat_dim + 2*C_T -> cond_dim_t
        self.cond_dim_t = self.feat_dim + self.C_T * 2

        # t-layer cond binders: one per t-layer
        self.t_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_T)
            for _ in range(self.n_t_layers)
        ])

        # p-layer conditioning: cond_dim_t -> C_P
        self.p_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_P)
            for _ in range(self.n_p_layers)
        ])

        # Sub-P conditioning: cond_dim_t -> C_P//2 (only for structured v7.0 mode)
        if not self.unified_p_layer and not self.parallel_p_layer:
            self.sub_p_c_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])
            self.sub_p_x_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])

        # =============== Register Embeddings (3 types, variable counts) ===============
        self.reg_c_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_reg_c, self.C_T))
            for _ in range(self.n_t_layers)
        ])
        self.reg_x_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_reg_x, self.C_T))
            for _ in range(self.n_t_layers)
        ])
        self.reg_a_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_reg_a, self.C_T))
            for _ in range(self.n_t_layers)
        ])
        for reg_list in [self.reg_c_embeds, self.reg_x_embeds, self.reg_a_embeds]:
            for reg_emb in reg_list:
                nn.init.normal_(reg_emb, std=0.02)

        # =============== Layer-0 Input Mixing ===============
        # For layers 1..n_t_layers-1: mix(layer_0_out, layer_{n-1}_out) for ctx+x tokens
        self.layer0_mix = nn.ModuleList([
            nn.Linear(2 * self.C_T, self.C_T)
            for _ in range(self.n_t_layers - 1)
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
        self.t_blocks = nn.ModuleList([
            DiTBlockTimeOnly(**_dit_t_kwargs)
            for _ in range(self.n_t_layers)
        ])

        # =============== Pooling Projections ===============
        reg_c_pool_in = self.n_reg_c * self.C_T
        reg_x_pool_in = self.n_reg_x * self.C_T
        reg_a_pool_in = self.n_reg_a * self.C_T

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

        if self.parallel_p_layer:
            # v7.5: N parallel P-layers (one per register token)
            N_par = self.n_total_reg  # e.g. 2+1+1 = 4
            # Per-register pool: each register token(s) → C_P
            # reg-c tokens pool, reg-x tokens pool, each reg-a token pools
            self.par_pool_projs = nn.ModuleList()
            for p_idx in range(self.n_p_layers):
                layer_projs = nn.ModuleList()
                for r_idx in range(N_par):
                    # Determine which type this register belongs to
                    if r_idx < self.n_reg_a:
                        pool_in = self.C_T  # single token per register
                    elif r_idx < self.n_reg_a + self.n_reg_x:
                        pool_in = self.C_T
                    else:
                        pool_in = self.C_T
                    if p_idx > 0:
                        pool_in += self.C_P  # skip from prev p-layer output
                    layer_projs.append(nn.Linear(pool_in, self.C_P))
                self.par_pool_projs.append(layer_projs)

            # N parallel P blocks per p-layer
            self.par_p_blocks = nn.ModuleList([
                nn.ModuleList([
                    DiTBlockTimeOnly(**_dit_p_kwargs)
                    for _ in range(N_par)
                ])
                for _ in range(self.n_p_layers)
            ])

            # N parallel cond binders per p-layer
            self.par_p_cond_binders = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.cond_dim_t, self.C_P)
                    for _ in range(N_par)
                ])
                for _ in range(self.n_p_layers)
            ])

            # Bridge projections: Linear((1+N_par)*C_P → C_P) per register per p-layer
            self.par_bridge_projs = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear((1 + N_par) * self.C_P, self.C_P)
                    for _ in range(N_par)
                ])
                for _ in range(self.n_p_layers)
            ])

            # Unpool: cat(reg_token[C_T], p_out[C_P]) → C_T per register per p-layer
            unpool_in = self.C_T + self.C_P
            self.par_unpool_ffns = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(unpool_in, self.C_T * 2),
                        nn.GELU(approximate="tanh"),
                        nn.Linear(self.C_T * 2, self.C_T),
                    )
                    for _ in range(N_par)
                ])
                for _ in range(self.n_p_layers)
            ])

        elif self.unified_p_layer:
            # v7.4: single unified pool — cat all 3 register flats → C_P
            total_pool_in = reg_c_pool_in + reg_x_pool_in + reg_a_pool_in
            self.unified_pool_projs = nn.ModuleList()
            for i in range(self.n_p_layers):
                if i == 0:
                    self.unified_pool_projs.append(
                        nn.Linear(total_pool_in, self.C_P)
                    )
                else:
                    self.unified_pool_projs.append(
                        nn.Linear(total_pool_in + self.C_P, self.C_P)
                    )

            # Main P Blocks
            self.p_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_p_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # Unpool: cat(reg_token[C_T], p_out[C_P]) -> C_T
            unpool_in = self.C_T + self.C_P
            self.unpool_c_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
            self.unpool_x_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
            self.unpool_a_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])

        else:
            # v7.0: 3 separate pool projections
            self.reg_c_pool_projs = nn.ModuleList([
                nn.Linear(reg_c_pool_in, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])
            self.reg_x_pool_projs = nn.ModuleList([
                nn.Linear(reg_x_pool_in, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])
            self.reg_a_pool_projs = nn.ModuleList()
            for i in range(self.n_p_layers):
                if i == 0:
                    self.reg_a_pool_projs.append(nn.Linear(reg_a_pool_in, self.C_P))
                else:
                    self.reg_a_pool_projs.append(
                        nn.Linear(reg_a_pool_in + self.C_P, self.C_P)
                    )

            # Sub-P Blocks (half-width)
            _dit_sub_p_kwargs = dict(
                C=self.C_P // 2,
                num_heads=max(1, n_head_p // 2),
                num_registers=num_blk_internal_registers,
                mlp_hidden_dims=[self.C_P // 2 * fc_factor_p] * fc_n_layers,
                ada_scale_bias_bound=ada_scale_bias_bound,
                gate_bound=gate_bound,
                ada_n_layers=ada_n_layers,
                C_feat=self.C_P // 2,
                qk_norm=qk_norm,
            )
            self.sub_p_c_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_sub_p_kwargs)
                for _ in range(self.n_p_layers)
            ])
            self.sub_p_x_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_sub_p_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # Merge Projections
            self.merge_projs = nn.ModuleList([
                nn.Linear(2 * self.C_P, self.C_P)
                for _ in range(self.n_p_layers)
            ])

            # Main P Blocks
            self.p_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_p_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # Unpool with sub-P skip connections
            self.unpool_c_projs = nn.ModuleList([
                nn.Linear(self.C_P, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])
            self.unpool_x_projs = nn.ModuleList([
                nn.Linear(self.C_P, self.C_P // 2)
                for _ in range(self.n_p_layers)
            ])

            unpool_cx_in = self.C_T + self.C_P
            self.unpool_c_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_cx_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
            self.unpool_x_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_cx_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
            unpool_a_in = self.C_T + self.C_P
            self.unpool_a_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_a_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])

        # =============== Output Projection ===============
        # Concat t-layer skip outputs -> project to input_dim
        # n_t_layers skips + 1 initial = (n_t_layers + 1) * C_T
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
        logging.info("LMM V7 Summary:")
        logging.info(f"  C_T (narrow):     {self.C_T}")
        logging.info(f"  C_P (wide):       {self.C_P}")
        logging.info(f"  n_t_layers:       {self.n_t_layers}")
        logging.info(f"  n_p_layers:       {self.n_p_layers}")
        logging.info(f"  n_reg (per type): {self.n_reg}")
        logging.info(f"  input_dim:        {self.input_dim}")
        logging.info(f"  feat_dim:         {self.feat_dim}")
        logging.info(f"  time_rope_flag:   {self.time_rope_flag}")
        logging.info(f"  isolated_ctx_x:   {self.isolated_ctx_x}")
        logging.info(f"  unified_p_layer:  {self.unified_p_layer}")
        logging.info(f"  full_attention_t: {self.full_attention_t}")
        logging.info(f"  parallel_p_layer: {self.parallel_p_layer}")
        if self.parallel_p_layer:
            logging.info(f"  registers:        {self.n_reg_a}a + {self.n_reg_x}x + {self.n_reg_c}c = {self.n_total_reg}")
        logging.info(f"  pattern:          {'t-p-' * self.n_p_layers}t")

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
    def _build_register_attn_mask(
        self, N_CTX: int, T: int, R: int, device: torch.device,
    ) -> torch.Tensor:
        """Build the structured attention mask for register-based t-layer.

        Sequence layout (after DiT internal register prepend):
            [internal_regs(R), reg-c(n_reg_c), reg-x(n_reg_x), reg-a(n_reg_a), ctx(N_CTX), x(T)]

        When full_attention_t=True, returns all-True mask (all regs are homogeneous).

        Returns:
            (L, L) bool mask where True = can attend.
        """
        nrc = self.n_reg_c
        nrx = self.n_reg_x
        nra = self.n_reg_a
        L = R + nrc + nrx + nra + N_CTX + T

        if self.full_attention_t:
            return torch.ones(L, L, dtype=torch.bool, device=device)

        mask = torch.zeros(L, L, dtype=torch.bool, device=device)

        # Segment boundaries
        ir_end = R
        rc_end = ir_end + nrc
        rx_end = rc_end + nrx
        ra_end = rx_end + nra
        ctx_end = ra_end + N_CTX
        # x_end = L

        # Internal registers: full attention to everything
        mask[:ir_end, :] = True
        mask[:, :ir_end] = True

        # reg-c <-> reg-c (self-attention within group)
        mask[ir_end:rc_end, ir_end:rc_end] = True
        # reg-c <-> ctx (bidirectional)
        mask[ir_end:rc_end, ra_end:ctx_end] = True
        mask[ra_end:ctx_end, ir_end:rc_end] = True

        # reg-x <-> reg-x (self-attention within group)
        mask[rc_end:rx_end, rc_end:rx_end] = True
        # reg-x <-> x (bidirectional)
        mask[rc_end:rx_end, ctx_end:L] = True
        mask[ctx_end:L, rc_end:rx_end] = True

        # reg-a <-> reg-a (self-attention within group)
        mask[rx_end:ra_end, rx_end:ra_end] = True
        # reg-a <-> ctx (bidirectional)
        mask[rx_end:ra_end, ra_end:ctx_end] = True
        mask[ra_end:ctx_end, rx_end:ra_end] = True
        # reg-a <-> x (bidirectional)
        mask[rx_end:ra_end, ctx_end:L] = True
        mask[ctx_end:L, rx_end:ra_end] = True

        # ctx <-> ctx (self-attention)
        mask[ra_end:ctx_end, ra_end:ctx_end] = True
        # x <-> x (self-attention)
        mask[ctx_end:L, ctx_end:L] = True

        if not self.isolated_ctx_x:
            # Option A: ctx <-> x direct communication
            mask[ra_end:ctx_end, ctx_end:L] = True
            mask[ctx_end:L, ra_end:ctx_end] = True

        return mask

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
    ) -> torch.Tensor:
        """Forward pass with register-based temporal attention.

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

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
        n_reg = self.n_reg
        device = x.device
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
            x = torch.cat([ctx, x], dim=1)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)
            x_level = torch.cat([ctx_level, x_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # Build per-element conditioning: (B, S, P, cond_dim_t)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * P))
            .reshape(B, S, P, self.C_T)
            .to(x)
        )  # (B, S, P, C_T)
        feat_expanded = feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, feat_dim)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(
            -1, -1, P, -1
        )  # (B, S, P, C_T)

        cond_bsp = torch.cat(
            [
                self.hard_ln_feat(feat_expanded),
                self.hard_ln_ct(level_emb),
                self.hard_ln_ct(ctx_pos_emb_expanded),
            ],
            dim=-1,
        )  # (B, S, P, cond_dim_t)

        # Embed input
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, P, C_T)

        # Save layer-0 state for mixing (ctx+x tokens only, not registers)
        x_state_0 = x_emb.clone()  # (B, S, P, C_T) — preserved across all layers

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

        # ========== Build attention masks ==========
        # t-layer: structured register attention mask
        R = self.t_blocks[0].R  # internal registers in DiT blocks
        n_total_reg = self.n_total_reg
        struct_mask = self._build_register_attn_mask(N_CTX, T, R, device)  # (L, L)

        # Validity mask for t-layer: registers are always valid
        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)
        ir_valid = torch.ones(B * P, R, device=device, dtype=torch.bool)
        reg_valid = torch.ones(B * P, n_total_reg, device=device, dtype=torch.bool)
        full_valid = torch.cat([ir_valid, reg_valid, traj_valid], dim=1)  # (B*P, L)
        val_mask = full_valid[:, None, :] & full_valid[:, :, None]  # (B*P, L, L)
        attn_mask_t = struct_mask[None, :, :] & val_mask  # (B*P, L, L)
        # Force diagonal True: every token can attend to itself (prevents NaN
        # from softmax over all-False rows for padded/invalid trajectories)
        diag_idx = torch.arange(attn_mask_t.shape[-1], device=device)
        attn_mask_t[:, diag_idx, diag_idx] = True
        attn_mask_t = attn_mask_t[:, None].expand(
            -1, self.n_head_t, -1, -1
        )  # (B*P, H, L, L)

        # p-layer mask: full attention over A+P
        p_valid = x_mask.any(dim=1)  # (B, P) - trajectory valid if valid at any time
        p_seq_valid = torch.cat([all_add_mask, p_valid], dim=1)  # (B, A+P)

        # Sub-P mask (same structure, just different width)
        # Uses p_valid for both sub-P-c and sub-P-x

        # ========== Prepare RoPE time ==========
        if self.time_rope_flag:
            # x_time is (B, T) — only noisy frame timestamps
            # Need to reshape for per-trajectory: (B*P, T)
            t_rope = (
                x_time[:, :, None]
                .expand(-1, -1, P)
                .permute(0, 2, 1)
                .reshape(B * P, T)
            )  # (B*P, T)
        else:
            t_rope = None

        # ========== Main Loop: t-p-t-p-...-t ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]
        prev_p_out = None  # (B, P, C_P) from previous p-layer
        prev_par_p_outs = None  # list of (B, P, C_P) for parallel p-layer mode

        # Register states (initialized from learnable embeddings at layer 0)
        reg_c_state = None  # (B, P, n_reg_c, C_T) or None
        reg_x_state = None
        reg_a_state = None
        nrc = self.n_reg_c
        nrx = self.n_reg_x
        nra = self.n_reg_a

        for layer_idx in range(self.n_t_layers):
            # --------- Layer-0 Mixing ---------
            if layer_idx > 0:
                x_state = self.layer0_mix[layer_idx - 1](
                    torch.cat([x_state_0, x_state], dim=-1)
                )  # (B, S, P, C_T)

            # --------- Build register tokens ---------
            if reg_c_state is not None:
                rc = reg_c_state  # (B, P, n_reg_c, C_T)
                rx = reg_x_state
                ra = reg_a_state
            else:
                rc = self.reg_c_embeds[layer_idx].expand(B * P, -1, -1).reshape(
                    B, P, nrc, self.C_T
                )
                rx = self.reg_x_embeds[layer_idx].expand(B * P, -1, -1).reshape(
                    B, P, nrx, self.C_T
                )
                ra = self.reg_a_embeds[layer_idx].expand(B * P, -1, -1).reshape(
                    B, P, nra, self.C_T
                )

            # --------- Reshape for per-trajectory self-attention ---------
            x_traj = x_state.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)
            rc_traj = rc.reshape(B * P, nrc, self.C_T)
            rx_traj = rx.reshape(B * P, nrx, self.C_T)
            ra_traj = ra.reshape(B * P, nra, self.C_T)

            # Full sequence: [reg-c, reg-x, reg-a, ctx+x]
            traj_seq = torch.cat(
                [rc_traj, rx_traj, ra_traj, x_traj], dim=1
            )  # (B*P, n_total_reg + S, C_T)

            # --------- Build conditioning for this t-layer ---------
            cond_t = self.t_cond_binders[layer_idx](cond_bsp)  # (B, S, P, C_T)
            cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)
            cond_reg_mean = cond_t_per_traj.mean(dim=1, keepdim=True).expand(
                -1, n_total_reg, -1
            )  # (B*P, n_total_reg, C_T)
            cond_traj = torch.cat(
                [cond_reg_mean, cond_t_per_traj], dim=1
            )  # (B*P, n_total_reg + S, C_T)

            # --------- Run t-block ---------
            traj_out = self.t_blocks[layer_idx](
                x=traj_seq,
                cond=cond_traj,
                t=t_rope,
                num_external_regs=n_total_reg + N_CTX,
                attn_mask=attn_mask_t,
            )  # (B*P, n_total_reg + S, C_T)

            # --------- Extract outputs ---------
            reg_c_out = traj_out[:, :nrc]  # (B*P, n_reg_c, C_T)
            reg_x_out = traj_out[:, nrc:nrc + nrx]  # (B*P, n_reg_x, C_T)
            reg_a_out = traj_out[:, nrc + nrx:nrc + nrx + nra]  # (B*P, n_reg_a, C_T)
            x_out = traj_out[:, n_total_reg:]  # (B*P, S, C_T)

            x_state = x_out.reshape(B, P, S, self.C_T).permute(
                0, 2, 1, 3
            )  # (B, S, P, C_T)

            t_skip_list.append(x_state.reshape(B * S, P, self.C_T).clone())

            # --------- Pooling + p-layer + Unpooling (if not last t-layer) ---------
            if layer_idx < self.n_t_layers - 1:
                p_idx = layer_idx

                # === Pool register types ===
                rc_flat = reg_c_out.reshape(B, P, nrc * self.C_T)  # (B, P, nrc*C_T)
                rx_flat = reg_x_out.reshape(B, P, nrx * self.C_T)
                ra_flat = reg_a_out.reshape(B, P, nra * self.C_T)
                cond_p_per_point = cond_bsp.mean(dim=1)  # (B, P, cond_dim_t)

                if self.parallel_p_layer:
                    # === v7.5: Parallel P-layers with bridge sync ===
                    N_par = self.n_total_reg

                    # Build per-register token list: each is (B, P, C_T)
                    # Order: reg-a tokens, then reg-x tokens, then reg-c tokens
                    reg_tokens_list = []
                    # reg-a: nra individual tokens
                    for ri in range(nra):
                        reg_tokens_list.append(
                            reg_a_out[:, ri:ri + 1].reshape(B, P, self.C_T)
                        )
                    # reg-x: nrx individual tokens
                    for ri in range(nrx):
                        reg_tokens_list.append(
                            reg_x_out[:, ri:ri + 1].reshape(B, P, self.C_T)
                        )
                    # reg-c: nrc individual tokens
                    for ri in range(nrc):
                        reg_tokens_list.append(
                            reg_c_out[:, ri:ri + 1].reshape(B, P, self.C_T)
                        )

                    # Pool each register → C_P
                    pooled_list = []
                    for r_idx in range(N_par):
                        pool_in = reg_tokens_list[r_idx]  # (B, P, C_T)
                        if prev_par_p_outs is not None:
                            pool_in = torch.cat(
                                [pool_in, prev_par_p_outs[r_idx]], dim=-1
                            )  # (B, P, C_T + C_P)
                        pooled_list.append(
                            self.par_pool_projs[p_idx][r_idx](pool_in)
                        )  # (B, P, C_P)

                    # Build P-layer input sequences: [add_tokens, pooled_i]
                    # Phase 1: Run attention for all N parallel P-layers
                    R_p = self.par_p_blocks[p_idx][0].R
                    attn_mask_p = build_attention_mask(
                        p_seq_valid, mode="full", left_pad=R_p,
                    )[:, None].expand(-1, self.n_head_p, -1, -1)

                    attn_outs = []  # list of (B, R_p+A+P, C_P)
                    ada_params = []  # list of (s2, sc2, g2)
                    for r_idx in range(N_par):
                        p_seq = torch.cat(
                            [all_add_tokens, pooled_list[r_idx]], dim=1
                        )  # (B, A+P, C_P)
                        cond_p = self.par_p_cond_binders[p_idx][r_idx](
                            cond_p_per_point
                        )  # (B, P, C_P)
                        cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)
                        cond_p_seq = torch.cat([cond_a, cond_p], dim=1)

                        x_after_attn, mlp_params = self.par_p_blocks[p_idx][
                            r_idx
                        ].forward_attn_only(
                            x=p_seq, cond=cond_p_seq, t=None,
                            num_external_regs=A, attn_mask=attn_mask_p,
                        )
                        attn_outs.append(x_after_attn)
                        ada_params.append(mlp_params)

                    # Phase 2: Bridge sync — extract P-portions, cat, project
                    p_portions = [
                        ao[:, R_p + A:, :] for ao in attn_outs
                    ]  # list of (B, P, C_P)
                    global_cat = torch.cat(p_portions, dim=-1)  # (B, P, N_par*C_P)

                    # Phase 3: Inject bridge and run MLP for each
                    par_p_outs = []
                    for r_idx in range(N_par):
                        own_p = p_portions[r_idx]  # (B, P, C_P)
                        bridged = self.par_bridge_projs[p_idx][r_idx](
                            torch.cat([own_p, global_cat], dim=-1)
                        )  # (B, P, C_P)

                        # Replace P-portion in attn_out with bridged value
                        x_in = attn_outs[r_idx].clone()
                        x_in[:, R_p + A:, :] = bridged

                        s2, sc2, g2 = ada_params[r_idx]
                        p_out_r = self.par_p_blocks[p_idx][r_idx].forward_mlp_only(
                            x_in, s2, sc2, g2
                        )  # (B, A+P, C_P)
                        par_p_outs.append(p_out_r[:, A:])  # (B, P, C_P)

                    prev_par_p_outs = par_p_outs

                    # Unpool: each register gets its own FFN
                    all_reg_states = []  # collect in same order as pooling
                    for r_idx in range(N_par):
                        reg_tok = reg_tokens_list[r_idx]  # (B, P, C_T)
                        p_out_r = par_p_outs[r_idx]  # (B, P, C_P)
                        unpool_in = torch.cat(
                            [reg_tok[:, :, None, :], p_out_r[:, :, None, :]], dim=-1
                        )  # (B, P, 1, C_T+C_P)
                        reg_state = self.par_unpool_ffns[p_idx][r_idx](
                            unpool_in
                        )  # (B, P, 1, C_T)
                        all_reg_states.append(reg_state)

                    # Reassemble register states by type
                    # Order was: nra reg-a, nrx reg-x, nrc reg-c
                    ra_states = all_reg_states[:nra]
                    rx_states = all_reg_states[nra:nra + nrx]
                    rc_states = all_reg_states[nra + nrx:]
                    reg_a_state = torch.cat(ra_states, dim=2)  # (B, P, nra, C_T)
                    reg_x_state = torch.cat(rx_states, dim=2)  # (B, P, nrx, C_T)
                    reg_c_state = torch.cat(rc_states, dim=2)  # (B, P, nrc, C_T)

                elif self.unified_p_layer:
                    # === v7.4: Unified P-layer (no sub-P-c/x) ===
                    all_flat = torch.cat([rc_flat, rx_flat, ra_flat], dim=-1)
                    if prev_p_out is not None:
                        all_flat = torch.cat([all_flat, prev_p_out], dim=-1)
                    p_tokens = self.unified_pool_projs[p_idx](all_flat)  # (B, P, C_P)

                    # Main P
                    p_seq = torch.cat([all_add_tokens, p_tokens], dim=1)
                    cond_p = self.p_cond_binders[p_idx](cond_p_per_point)
                    cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)
                    cond_p_seq = torch.cat([cond_a, cond_p], dim=1)

                    attn_mask_p = build_attention_mask(
                        p_seq_valid, mode="full",
                        left_pad=self.p_blocks[p_idx].R,
                    )[:, None].expand(-1, self.n_head_p, -1, -1)

                    p_out_seq = self.p_blocks[p_idx](
                        x=p_seq, cond=cond_p_seq, t=None,
                        num_external_regs=A, attn_mask=attn_mask_p,
                    )
                    p_out = p_out_seq[:, A:]  # (B, P, C_P)
                    prev_p_out = p_out

                    # Unpool: cat(reg_token, p_out) -> FFN for each type
                    p_exp = p_out[:, :, None, :].expand(-1, -1, nrc, -1)
                    rc_r = reg_c_out.reshape(B, P, nrc, self.C_T)
                    reg_c_state = self.unpool_c_ffns[p_idx](torch.cat([rc_r, p_exp], dim=-1))

                    p_exp_x = p_out[:, :, None, :].expand(-1, -1, nrx, -1)
                    rx_r = reg_x_out.reshape(B, P, nrx, self.C_T)
                    reg_x_state = self.unpool_x_ffns[p_idx](torch.cat([rx_r, p_exp_x], dim=-1))

                    p_exp_a = p_out[:, :, None, :].expand(-1, -1, nra, -1)
                    ra_r = reg_a_out.reshape(B, P, nra, self.C_T)
                    reg_a_state = self.unpool_a_ffns[p_idx](torch.cat([ra_r, p_exp_a], dim=-1))

                else:
                    # === v7.0: Structured P-layer (sub-P-c, sub-P-x, main P) ===
                    reg_c_pooled = self.reg_c_pool_projs[p_idx](rc_flat)  # (B, P, C_P//2)
                    reg_x_pooled = self.reg_x_pool_projs[p_idx](rx_flat)  # (B, P, C_P//2)

                    if prev_p_out is not None:
                        ra_pool_in = torch.cat([ra_flat, prev_p_out], dim=-1)
                    else:
                        ra_pool_in = ra_flat
                    reg_a_pooled = self.reg_a_pool_projs[p_idx](ra_pool_in)  # (B, P, C_P)

                    # Sub-P-c (half-width)
                    sub_p_c_cond = self.sub_p_c_cond_binders[p_idx](cond_p_per_point)
                    sub_p_c_mask = build_attention_mask(
                        p_valid, mode="full",
                        left_pad=self.sub_p_c_blocks[p_idx].R,
                    )[:, None].expand(-1, max(1, self.n_head_p // 2), -1, -1)
                    sub_p_c_out = self.sub_p_c_blocks[p_idx](
                        x=reg_c_pooled, cond=sub_p_c_cond, t=None,
                        num_external_regs=0, attn_mask=sub_p_c_mask,
                    )

                    # Sub-P-x (half-width)
                    sub_p_x_cond = self.sub_p_x_cond_binders[p_idx](cond_p_per_point)
                    sub_p_x_mask = build_attention_mask(
                        p_valid, mode="full",
                        left_pad=self.sub_p_x_blocks[p_idx].R,
                    )[:, None].expand(-1, max(1, self.n_head_p // 2), -1, -1)
                    sub_p_x_out = self.sub_p_x_blocks[p_idx](
                        x=reg_x_pooled, cond=sub_p_x_cond, t=None,
                        num_external_regs=0, attn_mask=sub_p_x_mask,
                    )

                    # Merge
                    merged = torch.cat([sub_p_c_out, sub_p_x_out, reg_a_pooled], dim=-1)
                    merged_proj = self.merge_projs[p_idx](merged)  # (B, P, C_P)

                    # Main P
                    p_seq = torch.cat([all_add_tokens, merged_proj], dim=1)
                    cond_p = self.p_cond_binders[p_idx](cond_p_per_point)
                    cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)
                    cond_p_seq = torch.cat([cond_a, cond_p], dim=1)

                    attn_mask_p = build_attention_mask(
                        p_seq_valid, mode="full",
                        left_pad=self.p_blocks[p_idx].R,
                    )[:, None].expand(-1, self.n_head_p, -1, -1)

                    p_out_seq = self.p_blocks[p_idx](
                        x=p_seq, cond=cond_p_seq, t=None,
                        num_external_regs=A, attn_mask=attn_mask_p,
                    )
                    p_out = p_out_seq[:, A:]
                    prev_p_out = p_out

                    # Unpool with sub-P skip connections
                    p_out_c = self.unpool_c_projs[p_idx](p_out)
                    p_out_x = self.unpool_x_projs[p_idx](p_out)

                    p_c_skip = torch.cat([p_out_c, sub_p_c_out], dim=-1)
                    p_c_exp = p_c_skip[:, :, None, :].expand(-1, -1, nrc, -1)
                    rc_r = reg_c_out.reshape(B, P, nrc, self.C_T)
                    reg_c_state = self.unpool_c_ffns[p_idx](
                        torch.cat([rc_r, p_c_exp], dim=-1)
                    )

                    p_x_skip = torch.cat([p_out_x, sub_p_x_out], dim=-1)
                    p_x_exp = p_x_skip[:, :, None, :].expand(-1, -1, nrx, -1)
                    rx_r = reg_x_out.reshape(B, P, nrx, self.C_T)
                    reg_x_state = self.unpool_x_ffns[p_idx](
                        torch.cat([rx_r, p_x_exp], dim=-1)
                    )

                    p_a_exp = p_out[:, :, None, :].expand(-1, -1, nra, -1)
                    ra_r = reg_a_out.reshape(B, P, nra, self.C_T)
                    reg_a_state = self.unpool_a_ffns[p_idx](
                        torch.cat([ra_r, p_a_exp], dim=-1)
                    )

        # ========== Output: concat t-layer skip outputs ==========
        # t_skip_list: list of (B*S, P, C_T) tensors, len = n_t_layers + 1
        x_out = torch.cat(t_skip_list, dim=-1)  # (B*S, P, (n_t_layers+1)*C_T)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)


# Keep backward-compatible name (agent.py may reference MotionMachineV2 for fallback)
MotionMachineV2 = MotionMachineV7


def test_main():
    """Smoke test for MotionMachineV7 — both attention mask options."""
    import sys

    B, T, P, C = 2, 8, 64, 9
    CTX = 3
    A = 16
    C_T, C_P = 128, 512

    combos = [
        {"isolated_ctx_x": False, "time_rope_flag": False},  # V7.0 base (Option A)
        {"isolated_ctx_x": True, "time_rope_flag": False},   # Option B
        {"isolated_ctx_x": False, "time_rope_flag": True},   # Time RoPE variant
    ]

    for combo in combos:
        label = ", ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        net = MotionMachineV7(
            C_T=C_T,
            C_P=C_P,
            feat_dim=512,
            n_head_t=8,
            n_head_p=16,
            n_t_layers=4,
            n_reg=4,
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

        print(f"Output shape: {out.shape}")
        assert out.shape == (B, T + CTX, P, C), f"Expected ({B}, {T+CTX}, {P}, {C}), got {out.shape}"
        print("PASS")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_main()
