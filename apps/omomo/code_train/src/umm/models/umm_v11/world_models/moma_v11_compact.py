"""MotionMachine V11 Compact -- ctx/trace indicator + p-layer skip/mix.

Based on MotionMachineV7 (moma_v6.py) with two modifications:

Mod 2: ctx/trace indicator embedding (configurable dim, default 0 for backward compat).
    Appends a learned indicator to ctx_pos_emb distinguishing context vs trace frames.
    Changes ctx_pos_emb from C_T to C_T + indicator_dim, propagated through cond_dim,
    LayerNorm, cond_binders, and time_bind_proj.

Mod 4: p-layer output projection into t-layer skip/mix.
    After each p-layer, projects p_out (B, P, C_P) to C_T, broadcasts across time,
    and feeds into the next t-layer's layer0_mix (3-way: x_state_0 + p_proj + x_state).
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
from umm.models.umm_v11.world_models.dit_mot import DiTBlockMoT
from umm.models.umm_v11.world_models.so3_utils import compute_entity_mean_9d


class MotionMachineV11Compact(nn.Module):
    """Register-based temporal attention MotionMachine (V11 Compact).

    Layer pattern: t-p-t-p-...-t (alternating, first and last are t-layers).
    n_t_layers t-layers and n_p_layers = n_t_layers - 1 p-layers.

    V11 Compact changes from V7:
    - Mod 2: ctx/trace indicator embedding (ctx_trace_indicator_dim)
    - Mod 4: p-layer output projection fed into t-layer layer0_mix (3-way)

    Args:
        C_T: Narrow channel width for t-layers.
        C_P: Wide channel width for p-layers.
        feat_dim: Input feature dimension from identify layer.
        n_head_t: Number of attention heads for t-layers.
        n_head_p: Number of attention heads for p-layers.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        n_t_layers: Number of t-layers. Must be >= 2.
        n_reg_c: Number of context register tokens (0 to skip).
        n_reg_x: Number of x register tokens (0 to skip).
        n_reg_a: Number of all register tokens (0 to skip).
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
        x_pe_dim: If > 0, apply NeRF-style positional encoding to input.
        ctx_trace_indicator_dim: Mod 2 -- dimension of ctx/trace indicator embedding.
            0 = disabled (backward compat with V7).
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
        n_reg_c: int = 2,
        n_reg_x: int = 2,
        n_reg_a: int = 2,
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
        group_p_layer: bool = False,
        v981_pool_to_adaln: bool = False,
        v982_hyper_group: bool = False,
        v982_hyper_emb_dim: int = 32,
        v982_hyper_hidden_dim: int = 64,
        v983_cross_layer_pool: bool = False,
        qk_norm: bool = False,
        x_pe_dim: int = 0,
        # Mod 2: ctx/trace indicator embedding
        ctx_trace_indicator_dim: int = 0,
        # split spatial RoPE + trajectory PE
        spatial_rope_flag: bool = False,
        n_traj_tokens: int = 0,
        # compact variant selection (None=default, "dual_p"=dual P-layer)
        compact_variant: Optional[str] = None,
        # configurable layer pattern string (e.g., "tptptpt", "tttpt")
        # If provided, overrides n_t_layers. Must start/end with 't', only 't'/'p'.
        layer_type_string: Optional[str] = None,
        # Backward compat: n_reg is ignored if n_reg_c/x/a are explicitly set
        n_reg: int = -1,
    ):
        super().__init__()

        self.C_T = C_T
        self.C_P = C_P
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
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
            # Consistency check: warn if n_t_layers param disagrees with string
            _string_n_t = self.n_t_layers
            if n_t_layers != 4 and n_t_layers != _string_n_t:
                # n_t_layers=4 is the default, so only warn if explicitly set differently
                logging.warning(
                    f"layer_type_string '{layer_type_string}' has {_string_n_t} t-layers "
                    f"but n_t_layers={n_t_layers} was also provided. "
                    f"layer_type_string takes precedence."
                )
            # Assert unified_p_layer for consecutive P's (incompatible with dual_p/parallel_p)
            if "pp" in layer_type_string:
                assert unified_p_layer, (
                    f"Consecutive P's in '{layer_type_string}' require unified_p_layer=True"
                )
                assert compact_variant != "dual_p", (
                    f"Consecutive P's in '{layer_type_string}' incompatible with dual_p"
                )
                assert not parallel_p_layer, (
                    f"Consecutive P's in '{layer_type_string}' incompatible with parallel_p_layer"
                )
                assert not group_p_layer, (
                    f"Consecutive P's in '{layer_type_string}' incompatible with group_p_layer"
                )
                assert not v983_cross_layer_pool, (
                    f"Consecutive P's in '{layer_type_string}' incompatible with v983_cross_layer_pool"
                )

            # Compute P-group info for consecutive P support
            p_groups = []  # list of (start_p_idx, length)
            _p_counter = 0
            i = 0
            while i < len(self.layer_type_string):
                if self.layer_type_string[i] == 'p':
                    start = _p_counter
                    length = 0
                    while i < len(self.layer_type_string) and self.layer_type_string[i] == 'p':
                        length += 1
                        _p_counter += 1
                        i += 1
                    p_groups.append((start, length))
                else:
                    i += 1
            self.p_groups = p_groups
            self.n_p_groups = len(p_groups)

            # Per-P lookup tables
            self._p_is_group_start = {}
            self._p_is_group_end = {}
            self._p_to_group = {}
            for g_idx, (start, length) in enumerate(p_groups):
                for offset in range(length):
                    p_idx = start + offset
                    self._p_is_group_start[p_idx] = (offset == 0)
                    self._p_is_group_end[p_idx] = (offset == length - 1)
                    self._p_to_group[p_idx] = g_idx
        else:
            self.n_t_layers = n_t_layers
            self.n_p_layers = n_t_layers - 1
            # Generate default alternating pattern: t-p-t-p-...-t
            self.layer_type_string = "t" + "pt" * (n_t_layers - 1)
            # Default alternating: each P is its own group
            self.p_groups = [(i, 1) for i in range(self.n_p_layers)]
            self.n_p_groups = self.n_p_layers
            self._p_is_group_start = {i: True for i in range(self.n_p_layers)}
            self._p_is_group_end = {i: True for i in range(self.n_p_layers)}
            self._p_to_group = {i: i for i in range(self.n_p_layers)}
        self.n_head_t = n_head_t
        self.n_head_p = n_head_p
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat
        self.isolated_ctx_x = isolated_ctx_x
        self.unified_p_layer = unified_p_layer
        self.full_attention_t = full_attention_t
        self.parallel_p_layer = parallel_p_layer
        self.group_p_layer = group_p_layer
        self.v981_pool_to_adaln = v981_pool_to_adaln
        self.v982_hyper_group = v982_hyper_group
        self.v982_hyper_emb_dim = v982_hyper_emb_dim
        self.v982_hyper_hidden_dim = v982_hyper_hidden_dim
        self.v983_cross_layer_pool = v983_cross_layer_pool
        self.spatial_rope_flag = spatial_rope_flag
        self.n_traj_tokens = n_traj_tokens
        self.compact_variant = compact_variant
        if self.group_p_layer or self.v983_cross_layer_pool:
            # group_p_layer implies unified_p_layer behavior for Stage 3
            self.unified_p_layer = True

        # Per-type register counts -- always independent
        self.n_reg_c = n_reg_c
        self.n_reg_x = n_reg_x
        self.n_reg_a = n_reg_a
        self.n_total_reg = n_reg_c + n_reg_x + n_reg_a

        assert self.n_t_layers >= 2, "Need at least 2 t-layers (first and last)"

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_ct = HardLayerNorm(self.C_T)

        # =============== Mod 2: ctx/trace indicator embedding ===============
        self.ctx_trace_indicator_dim = ctx_trace_indicator_dim
        if ctx_trace_indicator_dim > 0:
            self.ctx_indicator = nn.Parameter(torch.randn(ctx_trace_indicator_dim) * 0.02)
            self.trace_indicator = nn.Parameter(torch.randn(ctx_trace_indicator_dim) * 0.02)
            # Separate LayerNorm for the expanded ctx_pos_emb (C_T + indicator_dim)
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
        # When RoPE is off and add_time_to_feat is on, bind per-token time embedding
        # into the token representation before the first t-layer:
        #   bound = Linear(cat(projected_token, time_emb), dim=-1) -> C_T
        if not self.time_rope_flag and self.add_time_to_feat:
            # Mod 2: time_bind input is C_T (from x_emb) + C_T + indicator_dim (from ctx_pos_emb)
            self.time_bind_proj = nn.Linear(
                self.C_T + self.C_T + ctx_trace_indicator_dim, self.C_T
            )

        # =============== Conditioning Binders ===============
        # t-layer conditioning: feat_dim + C_T (level_emb) + C_T+indicator_dim (ctx_pos_emb)
        # = feat_dim + 2*C_T + indicator_dim
        self.cond_dim_t = self.feat_dim + self.C_T * 2 + ctx_trace_indicator_dim
        if self.n_traj_tokens > 0:
            self.cond_dim_t += self.C_T

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

        # =============== Register Embeddings (per-layer learnable, random init) ===============
        # Each layer has its own learnable register tokens per type.
        # At layer 0: use just the learnable token.
        # At layer N>0: previous layer's register output + this layer's learnable token (additive).
        if self.n_reg_c > 0:
            self.reg_c_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.n_reg_c, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            for reg_emb in self.reg_c_embeds:
                nn.init.normal_(reg_emb, std=0.02)
        if self.n_reg_x > 0:
            self.reg_x_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.n_reg_x, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            for reg_emb in self.reg_x_embeds:
                nn.init.normal_(reg_emb, std=0.02)
        if self.n_reg_a > 0:
            self.reg_a_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.n_reg_a, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            for reg_emb in self.reg_a_embeds:
                nn.init.normal_(reg_emb, std=0.02)

        # =============== Layer-0 Input Mixing ===============
        # Mod 4: 3-way mix: x_state_0 + p_proj + current_x_state
        # For layers 1..n_t_layers-1: mix(layer_0_out, p_proj, layer_{n-1}_out) for ctx+x tokens
        self.layer0_mix = nn.ModuleList([
            nn.Linear(3 * self.C_T, self.C_T)
            for _ in range(self.n_t_layers - 1)
        ])

        # Mod 4: p-layer output projection for skip mixing (C_P -> C_T)
        # unified_p_layer: one proj per P-group (shared across consecutive P's)
        # other variants: one proj per P-layer
        _n_projs = self.n_p_groups if self.unified_p_layer else self.n_p_layers
        self.p_to_ct_projs = nn.ModuleList([
            nn.Linear(self.C_P, self.C_T)
            for _ in range(_n_projs)
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
        if self.compact_variant == "mot":
            # MoT t-blocks -- 3x ada/norm/FFN, shared attention
            _n_temp = n_head_t // 2 if (self.spatial_rope_flag and self.time_rope_flag) else n_head_t
            _dit_t_mot_kwargs = dict(
                n_external_reg=self.n_total_reg,
                n_temporal_heads=_n_temp,
                n_spatial_dims=9,
                **_dit_t_kwargs,
            )
            self.t_blocks = nn.ModuleList([
                DiTBlockMoT(**_dit_t_mot_kwargs)
                for _ in range(self.n_t_layers)
            ])
        elif self.spatial_rope_flag and self.time_rope_flag:
            # split RoPE -- half temporal, half spatial
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
        # split RoPE kwargs for P-blocks (dual spatial)
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

        if self.compact_variant == "dual_p":
            # =============== Dual P-layer ===============
            # P1 processes reg-a + reg-c, P2 processes reg-a + reg-x.
            # After parallel attention, reg-a outputs merged via cat+proj.
            #
            # Compile audit: all modules are nn.Linear / nn.Sequential with
            # GELU + static shapes. No closures, no SVD, no dynamic control
            # flow. The `compact_variant == "dual_p"` branch is a static
            # Python conditional on a string attribute -- traced as constant.
            # Safe for torch.compile.
            p1_pool_in = reg_c_pool_in + reg_a_pool_in
            p2_pool_in = reg_x_pool_in + reg_a_pool_in

            self.dual_p1_pool_projs = nn.ModuleList()
            self.dual_p2_pool_projs = nn.ModuleList()
            for i in range(self.n_p_layers):
                extra = self.C_P if i > 0 else 0
                self.dual_p1_pool_projs.append(
                    nn.Linear(p1_pool_in + extra, self.C_P)
                )
                self.dual_p2_pool_projs.append(
                    nn.Linear(p2_pool_in + extra, self.C_P)
                )

            # 2 P-blocks per P-layer (separate weights)
            self.dual_p1_blocks = nn.ModuleList([
                _P_Block(**_p_block_kwargs)
                for _ in range(self.n_p_layers)
            ])
            self.dual_p2_blocks = nn.ModuleList([
                _P_Block(**_p_block_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # 2 cond binders per P-layer
            self.dual_p1_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_P)
                for _ in range(self.n_p_layers)
            ])
            self.dual_p2_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_P)
                for _ in range(self.n_p_layers)
            ])

            # reg-a merge: Linear(2*C_P, C_P) per P-layer
            self.dual_p_merge = nn.ModuleList([
                nn.Linear(2 * self.C_P, self.C_P)
                for _ in range(self.n_p_layers)
            ])

            # Unpool FFNs: cat(reg_token[C_T], p_out[C_P]) -> C_T
            # reg-c uses P1 output, reg-x uses P2 output, reg-a uses merged
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

        elif self.parallel_p_layer:
            # v7.5: N parallel P-layers (one per register token)
            N_par = self.n_total_reg  # e.g. 2+1+1 = 4
            # Per-register pool: each register token(s) -> C_P
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
                    _P_Block(**_p_block_kwargs)
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

            # Bridge projections: Linear((1+N_par)*C_P -> C_P) per register per p-layer
            self.par_bridge_projs = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear((1 + N_par) * self.C_P, self.C_P)
                    for _ in range(N_par)
                ])
                for _ in range(self.n_p_layers)
            ])

            # Unpool: cat(reg_token[C_T], p_out[C_P]) -> C_T per register per p-layer
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
            # v7.4: single unified pool -- cat all 3 register flats -> C_P
            # One pool proj per P-group. group_idx==0: no skip. group_idx>0: +C_P skip
            # from last P of previous group.
            total_pool_in = reg_c_pool_in + reg_x_pool_in + reg_a_pool_in
            self.unified_pool_projs = nn.ModuleList()
            for i in range(self.n_p_groups):
                if i == 0:
                    self.unified_pool_projs.append(
                        nn.Linear(total_pool_in, self.C_P)
                    )
                else:
                    self.unified_pool_projs.append(
                        nn.Linear(total_pool_in + self.C_P, self.C_P)
                    )

            # Main P Blocks (Stage 3 for group_p_layer, or the only stage otherwise)
            self.p_blocks = nn.ModuleList([
                _P_Block(**_p_block_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # === V98 Group P-layer: Stage 1 (group attn) + Stage 2 (pool+cat) ===
            if self.group_p_layer:
                # Stage 1: Group attention DiT blocks (same arch as main P)
                self.group_p_blocks = nn.ModuleList([
                    _P_Block(**_p_block_kwargs)
                    for _ in range(self.n_p_layers)
                ])
                # Stage 1: Group attention conditioning
                self.group_p_cond_binders = nn.ModuleList([
                    nn.Linear(self.cond_dim_t, self.C_P)
                    for _ in range(self.n_p_layers)
                ])
                # Stage 2: Group pool + cat projection (2*C_P -> C_P)
                self.group_pool_projs = nn.ModuleList([
                    nn.Linear(2 * self.C_P, self.C_P)
                    for _ in range(self.n_p_layers)
                ])

                # V981: pool-to-adaln -- project pooled and combine with Stage 3 cond
                if self.v981_pool_to_adaln and not self.v982_hyper_group:
                    self.v981_pool_proj = nn.ModuleList([
                        nn.Linear(self.C_P, self.C_P)
                        for _ in range(self.n_p_layers)
                    ])
                    self.v981_adaln_combine = nn.ModuleList([
                        nn.Linear(2 * self.C_P, self.C_P)
                        for _ in range(self.n_p_layers)
                    ])

                # V982: hyper-network group feature (supersedes V981)
                if self.v982_hyper_group:
                    _emb_dim = self.v982_hyper_emb_dim
                    _hid_dim = self.v982_hyper_hidden_dim
                    # Per-P-layer learnable embedding
                    self.v982_hyper_embs = nn.ParameterList([
                        nn.Parameter(torch.randn(1, 1, _emb_dim) * 0.02)
                        for _ in range(self.n_p_layers)
                    ])
                    # Weight generators: pooled -> MLP weights+biases
                    # Layer 1: (_emb_dim -> _hid_dim) weights + bias
                    _l1_params = _emb_dim * _hid_dim + _hid_dim
                    self.v982_wgen_l1 = nn.ModuleList([
                        nn.Linear(self.C_P, _l1_params)
                        for _ in range(self.n_p_layers)
                    ])
                    # Layer 2: (_hid_dim -> C_P) weights + bias
                    _l2_params = _hid_dim * self.C_P + self.C_P
                    self.v982_wgen_l2 = nn.ModuleList([
                        nn.Linear(self.C_P, _l2_params)
                        for _ in range(self.n_p_layers)
                    ])
                    # Combine hyper group feat with adaln cond
                    self.v982_adaln_combine = nn.ModuleList([
                        nn.Linear(2 * self.C_P, self.C_P)
                        for _ in range(self.n_p_layers)
                    ])

            # Unpool: cat(reg_token[C_T], p_out[C_P]) -> C_T
            # One unpool FFN per P-group (only group-end triggers unpool)
            unpool_in = self.C_T + self.C_P
            self.unpool_c_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_groups)
            ])
            self.unpool_x_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_groups)
            ])
            self.unpool_a_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(unpool_in, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_groups)
            ])

            # V983: cross-layer group pooling (no Stage 1, pool after full P attn)
            if self.v983_cross_layer_pool:
                # Learnable init for first P-layer's group feature
                self.v983_group_init = nn.Parameter(
                    torch.randn(1, 1, self.C_P) * 0.02
                )
                # Combine prev group feat with regular adaln cond per P-layer
                self.v983_adaln_combine = nn.ModuleList([
                    nn.Linear(2 * self.C_P, self.C_P)
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
            if self.spatial_rope_flag:
                _sub_n_heads = max(1, n_head_p // 2)
                _dit_sub_p_kwargs_split = dict(
                    n_temporal_heads=_sub_n_heads // 2,
                    n_spatial_dims=9,
                    **_dit_sub_p_kwargs,
                )
                _Sub_P_Block = DiTBlockSplitRoPE
                _sub_p_block_kwargs = _dit_sub_p_kwargs_split
            else:
                _Sub_P_Block = DiTBlockTimeOnly
                _sub_p_block_kwargs = _dit_sub_p_kwargs
            self.sub_p_c_blocks = nn.ModuleList([
                _Sub_P_Block(**_sub_p_block_kwargs)
                for _ in range(self.n_p_layers)
            ])
            self.sub_p_x_blocks = nn.ModuleList([
                _Sub_P_Block(**_sub_p_block_kwargs)
                for _ in range(self.n_p_layers)
            ])

            # Merge Projections
            self.merge_projs = nn.ModuleList([
                nn.Linear(2 * self.C_P, self.C_P)
                for _ in range(self.n_p_layers)
            ])

            # Main P Blocks
            self.p_blocks = nn.ModuleList([
                _P_Block(**_p_block_kwargs)
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
        logging.info("LMM V11 Compact Summary:")
        logging.info(f"  C_T (narrow):     {self.C_T}")
        logging.info(f"  C_P (wide):       {self.C_P}")
        logging.info(f"  n_t_layers:       {self.n_t_layers}")
        logging.info(f"  n_p_layers:       {self.n_p_layers}")
        logging.info(f"  registers:        c={self.n_reg_c} x={self.n_reg_x} a={self.n_reg_a} total={self.n_total_reg}")
        logging.info(f"  input_dim:        {self.input_dim}")
        logging.info(f"  feat_dim:         {self.feat_dim}")
        logging.info(f"  time_rope_flag:   {self.time_rope_flag}")
        logging.info(f"  time_bind:        {not self.time_rope_flag and self.add_time_to_feat}")
        logging.info(f"  isolated_ctx_x:   {self.isolated_ctx_x}")
        logging.info(f"  unified_p_layer:  {self.unified_p_layer}")
        logging.info(f"  group_p_layer:    {self.group_p_layer}")
        logging.info(f"  v981_pool_adaln:  {self.v981_pool_to_adaln}")
        logging.info(f"  v982_hyper_group: {self.v982_hyper_group}")
        logging.info(f"  v983_cross_pool:  {self.v983_cross_layer_pool}")
        logging.info(f"  full_attention_t: {self.full_attention_t}")
        logging.info(f"  parallel_p_layer: {self.parallel_p_layer}")
        logging.info(f"  ctx_trace_ind:    {self.ctx_trace_indicator_dim}")
        logging.info(f"  spatial_rope:     {self.spatial_rope_flag}")
        logging.info(f"  n_traj_tokens:    {self.n_traj_tokens}")
        logging.info(f"  compact_variant:  {self.compact_variant}")
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

    @torch.compiler.disable
    def _build_group_attn_mask(
        self, group_ids: torch.Tensor, p_valid: torch.Tensor, R: int,
    ) -> torch.Tensor:
        """Build attention mask for group P-layer Stage 1.

        Tokens can only attend to other tokens with the same group_id.
        Internal registers (R) attend to everything and are attended by everything.

        Args:
            group_ids: (B, P) int group IDs.
            p_valid: (B, P) bool, True if trajectory is valid.
            R: Number of internal register tokens in DiTBlock.

        Returns:
            (B, n_head_p, R+P, R+P) bool mask.
        """
        B, P = group_ids.shape
        L = R + P

        # Group mask: same group_id -> can attend (B, P, P)
        gid_i = group_ids[:, :, None]  # (B, P, 1)
        gid_j = group_ids[:, None, :]  # (B, 1, P)
        same_group = (gid_i == gid_j)  # (B, P, P)

        # Validity: both tokens must be valid
        val_i = p_valid[:, :, None]  # (B, P, 1)
        val_j = p_valid[:, None, :]  # (B, 1, P)
        both_valid = val_i & val_j  # (B, P, P)

        # Token-token: same group AND both valid
        token_mask = same_group & both_valid  # (B, P, P)

        # Full mask with registers
        mask = torch.ones(B, L, L, dtype=torch.bool, device=group_ids.device)
        # Token-token block: apply group constraint
        mask[:, R:, R:] = token_mask
        # Registers attend to/from valid tokens only
        mask[:, :R, R:] = p_valid[:, None, :]  # reg->token: token must be valid
        mask[:, R:, :R] = p_valid[:, :, None]  # token->reg: token must be valid

        # Diagonal always True (prevent NaN from softmax)
        diag = torch.arange(L, device=group_ids.device)
        mask[:, diag, diag] = True

        # Expand for multi-head: (B, H, L, L)
        return mask[:, None].expand(-1, self.n_head_p, -1, -1)

    @torch.compiler.disable
    def _group_max_pool(
        self, x: torch.Tensor, group_ids: torch.Tensor, p_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool features within each group, scatter back to each token.

        Vectorized via scatter_reduce -- no Python loops.

        Args:
            x: (B, P, C_P) token features.
            group_ids: (B, P) int group IDs.
            p_valid: (B, P) bool.

        Returns:
            (B, P, C_P) per-token group-pooled features.
        """
        B, P, C = x.shape
        src = x.clone()
        src[~p_valid] = float("-inf")

        # scatter_reduce: pool into group buckets via amax
        max_gid = int(group_ids.max().item()) + 1
        pooled = torch.full(
            (B, max_gid, C), float("-inf"), device=x.device, dtype=x.dtype,
        )  # (B, G, C)
        idx = group_ids[:, :, None].expand(-1, -1, C).long()  # (B, P, C)
        pooled.scatter_reduce_(1, idx, src, reduce="amax")

        # Gather back: each token gets its group's pooled feature
        result = pooled.gather(1, idx)  # (B, P, C)

        # Clean up: -inf -> 0 for invalid tokens / empty groups
        result = result.clamp(min=-1e6)
        result[~p_valid] = 0.0
        return result

    def _v982_hyper_forward(
        self, pooled: torch.Tensor, p_idx: int,
    ) -> torch.Tensor:
        """V982: Generate group feature via hyper-network.

        The pooled feature generates weights for a 2-layer MLP.
        A learnable embedding goes through this MLP to produce the group feature.

        Compile audit: All shapes are static (emb_dim, hid_dim, C_P fixed at init).
        Operations are Linear + view + einsum + relu -- all compile-safe.
        No dynamic control flow, no closures over tensors. Safe for torch.compile.

        Args:
            pooled: (B, P, C_P) group-pooled features.
            p_idx: P-layer index.

        Returns:
            (B, P, C_P) group feature vector.
        """
        B, P, C = pooled.shape
        emb_dim = self.v982_hyper_emb_dim
        hid_dim = self.v982_hyper_hidden_dim

        # Get learnable embedding for this P-layer
        emb = self.v982_hyper_embs[p_idx].expand(B, P, -1)  # (B, P, emb_dim)

        # Generate Layer 1 weights+bias from pooled
        l1_params = self.v982_wgen_l1[p_idx](pooled)  # (B, P, emb_dim*hid_dim + hid_dim)
        W1 = l1_params[:, :, :emb_dim * hid_dim].view(B, P, emb_dim, hid_dim)
        b1 = l1_params[:, :, emb_dim * hid_dim:]  # (B, P, hid_dim)

        # Generate Layer 2 weights+bias from pooled
        l2_params = self.v982_wgen_l2[p_idx](pooled)  # (B, P, hid_dim*C_P + C_P)
        W2 = l2_params[:, :, :hid_dim * C].view(B, P, hid_dim, C)
        b2 = l2_params[:, :, hid_dim * C:]  # (B, P, C_P)

        # Forward through hyper-network MLP
        h = torch.einsum("bpe,bpeh->bph", emb, W1) + b1  # (B, P, hid_dim)
        h = F.relu(h)
        group_feat = torch.einsum("bph,bphc->bpc", h, W2) + b2  # (B, P, C_P)

        return group_feat

    @torch.compiler.disable
    def _sample_traj_ids(self, B: int, P: int, device: torch.device) -> torch.Tensor:
        """Sample trajectory IDs for trajectory PE.

        Training: random permutation per batch item (shuffle entity ordering).
        Eval: identity ordering (0, 1, 2, ..., P-1).

        Note: Assumes P <= n_traj_tokens. All production configs have n_traj_tokens >> P
        (e.g., n_traj_tokens=1024 vs max P~64 across all domains).

        Args:
            B: Batch size.
            P: Number of entities.
            device: torch device.

        Returns:
            (B, P) int tensor of trajectory IDs in range [0, n_traj_tokens).
        """
        if self.training:
            # Independent random permutation per batch item (true per-sample augmentation)
            ids = torch.stack([
                torch.randperm(self.n_traj_tokens, device=device)[:P]
                for _ in range(B)
            ])  # (B, P)
        else:
            # Eval: identity ordering (0, 1, ..., P-1)
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
            x_r_level: [B, T, P] MeanFlow r-level (optional).
            ctx_r_level: [B, C, P] MeanFlow r-level for context (optional).

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
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
            if x_r_level is not None and ctx_r_level is not None:
                x_r_level = torch.cat([ctx_r_level, x_r_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # ========== Mod 2: ctx/trace indicator embedding ==========
        if self.ctx_trace_indicator_dim > 0:
            # ctx_pos_emb is (B, S, C_T) where S = N_CTX + T
            indicator = torch.zeros(
                B, S, self.ctx_trace_indicator_dim,
                device=device, dtype=ctx_pos_emb.dtype,
            )
            indicator[:, :N_CTX] = self.ctx_indicator  # context frames
            indicator[:, N_CTX:] = self.trace_indicator  # trace frames
            ctx_pos_emb = torch.cat(
                [ctx_pos_emb, indicator], dim=-1
            )  # (B, S, C_T + indicator_dim)

        # Build per-element conditioning: (B, S, P, cond_dim_t)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * P))
            .reshape(B, S, P, self.C_T)
            .to(x)
        )  # (B, S, P, C_T)
        # MeanFlow: add r-level embedding (additive, same dim as level_emb)
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
        )  # (B, S, P, C_T) or (B, S, P, C_T + indicator_dim)

        # Mod 2: use appropriate LayerNorm for ctx_pos_emb
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
        # When RoPE is off: bind per-token time embedding into the token
        # representation before the first t-layer.
        if not self.time_rope_flag and self.add_time_to_feat:
            # ctx_pos_emb is (B, S, C_T + indicator_dim) -- per-timestep time embedding
            # Expand to (B, S, P, C_T + indicator_dim) and concat with x_emb, then project
            time_emb_expanded = ctx_pos_emb[:, :, None, :].expand(
                -1, -1, P, -1
            )  # (B, S, P, C_T + indicator_dim)
            x_emb = self.time_bind_proj(
                torch.cat([x_emb, time_emb_expanded], dim=-1)
            )  # (B, S, P, C_T)

        # Save layer-0 state for mixing (ctx+x tokens only, not registers)
        x_state_0 = x_emb.clone()  # (B, S, P, C_T) -- preserved across all layers

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
            # x_time is (B, T) -- only noisy frame timestamps
            # Need to reshape for per-trajectory: (B*P, T)
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
            # x is (B, S, P, 9) -- already concatenated (ctx+noisy)
            # t-layer operates per-trajectory: sequence is [regs, ctx+x]
            # spatial positions: (B*P, S, 9) detached
            x_9d_spatial = x.detach().permute(0, 2, 1, 3).reshape(B * P, S, 9)

            # Mean 9D for P-layer dual spatial RoPE
            mean_ctx_9d = (
                compute_entity_mean_9d(x[:, :N_CTX], x_mask[:, :N_CTX])
                if N_CTX > 0 else None
            )
            mean_x_9d = compute_entity_mean_9d(x[:, N_CTX:], x_mask[:, N_CTX:])
            if mean_ctx_9d is None:
                mean_ctx_9d = mean_x_9d  # fallback: use x-mean for both
        else:
            x_9d_spatial = None
            mean_ctx_9d = None
            mean_x_9d = None

        # ========== Main Loop: iterate over layer_type_string ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]
        prev_p_out = None  # (B, P, C_P) from previous p-layer
        prev_par_p_outs = None  # list of (B, P, C_P) for parallel p-layer mode
        prev_group_feat = None  # (B, P, C_P) V983 cross-layer group feature

        # Mod 4: latest p-layer projection for layer0_mix (initialized to zeros)
        latest_p_proj = torch.zeros(
            B, S, P, self.C_T, device=device, dtype=x_emb.dtype
        )  # (B, S, P, C_T)

        # Register states (initialized from learnable embeddings at layer 0)
        reg_c_state = None  # (B, P, n_reg_c, C_T) or None
        reg_x_state = None
        reg_a_state = None
        nrc = self.n_reg_c
        nrx = self.n_reg_x
        nra = self.n_reg_a

        t_idx = 0
        p_idx = 0
        for layer_char in self.layer_type_string:
          if layer_char == "t":
            # --------- Layer-0 Mixing ---------
            if t_idx > 0:
                # Mod 4: 3-way mix: x_state_0 + latest_p_proj + x_state
                x_state = self.layer0_mix[t_idx - 1](
                    torch.cat([x_state_0, latest_p_proj, x_state], dim=-1)
                )  # (B, S, P, C_T)

            # --------- Build register tokens (per-layer additive) ---------
            # Layer 0: just the learnable token
            # Layer N>0: previous layer's register output + this layer's learnable token
            if nrc > 0:
                rc_learnable = self.reg_c_embeds[t_idx].expand(B * P, -1, -1).reshape(
                    B, P, nrc, self.C_T
                )  # (B, P, nrc, C_T)
                rc = rc_learnable if reg_c_state is None else reg_c_state + rc_learnable
            if nrx > 0:
                rx_learnable = self.reg_x_embeds[t_idx].expand(B * P, -1, -1).reshape(
                    B, P, nrx, self.C_T
                )
                rx = rx_learnable if reg_x_state is None else reg_x_state + rx_learnable
            if nra > 0:
                ra_learnable = self.reg_a_embeds[t_idx].expand(B * P, -1, -1).reshape(
                    B, P, nra, self.C_T
                )
                ra = ra_learnable if reg_a_state is None else reg_a_state + ra_learnable

            # --------- Reshape for per-trajectory self-attention ---------
            x_traj = x_state.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)

            # Build register trajectory tensors (skip types with count=0)
            reg_parts = []
            if nrc > 0:
                reg_parts.append(rc.reshape(B * P, nrc, self.C_T))
            if nrx > 0:
                reg_parts.append(rx.reshape(B * P, nrx, self.C_T))
            if nra > 0:
                reg_parts.append(ra.reshape(B * P, nra, self.C_T))

            # Full sequence: [reg-c, reg-x, reg-a, ctx+x]
            if reg_parts:
                traj_seq = torch.cat(
                    reg_parts + [x_traj], dim=1
                )  # (B*P, n_total_reg + S, C_T)
            else:
                traj_seq = x_traj  # (B*P, S, C_T) -- no registers

            # --------- Build conditioning for this t-layer ---------
            # AdaLN conditioning split:
            #   reg-c -> mean over ctx tokens only
            #   reg-x -> mean over x tokens only
            #   reg-a -> mean over all tokens
            cond_t = self.t_cond_binders[t_idx](cond_bsp)  # (B, S, P, C_T)
            cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                B * P, S, self.C_T
            )  # (B*P, S, C_T)

            # Split conditioning by token type
            cond_reg_parts = []
            if nrc > 0:
                if N_CTX > 0:
                    cond_ctx_mean = cond_t_per_traj[:, :N_CTX].mean(dim=1, keepdim=True)
                else:
                    # No ctx tokens: fall back to mean over all
                    cond_ctx_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                cond_reg_parts.append(cond_ctx_mean.expand(-1, nrc, -1))  # (B*P, nrc, C_T)
            if nrx > 0:
                if N_CTX < S:
                    cond_x_mean = cond_t_per_traj[:, N_CTX:].mean(dim=1, keepdim=True)
                else:
                    # No x tokens: fall back to mean over all
                    cond_x_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                cond_reg_parts.append(cond_x_mean.expand(-1, nrx, -1))  # (B*P, nrx, C_T)
            if nra > 0:
                cond_all_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                cond_reg_parts.append(cond_all_mean.expand(-1, nra, -1))  # (B*P, nra, C_T)

            if cond_reg_parts:
                cond_reg = torch.cat(cond_reg_parts, dim=1)  # (B*P, n_total_reg, C_T)
                cond_traj = torch.cat(
                    [cond_reg, cond_t_per_traj], dim=1
                )  # (B*P, n_total_reg + S, C_T)
            else:
                cond_traj = cond_t_per_traj  # (B*P, S, C_T) -- no registers

            # --------- Run t-block ---------
            # set spatial RoPE params before t-block call
            if self.spatial_rope_flag:
                self.t_blocks[t_idx].set_spatial_rope_params(
                    spatial_positions=x_9d_spatial,
                    num_external_regs_spatial=n_total_reg,
                )

            if self.compact_variant == "mot":
                traj_out = self.t_blocks[t_idx](
                    x=traj_seq,
                    cond=cond_traj,
                    t=t_rope,
                    num_external_regs=n_total_reg + N_CTX,
                    attn_mask=attn_mask_t,
                    n_ctx=N_CTX,
                )  # (B*P, n_total_reg + S, C_T)
            else:
                traj_out = self.t_blocks[t_idx](
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
            t_idx += 1

          elif layer_char == "p":
            # --------- Pooling + p-layer + Unpooling ---------
            # === Pool register types ===
            rc_flat = reg_c_out.reshape(B, P, nrc * self.C_T)  # (B, P, nrc*C_T)
            rx_flat = reg_x_out.reshape(B, P, nrx * self.C_T)
            ra_flat = reg_a_out.reshape(B, P, nra * self.C_T)
            cond_p_per_point = cond_bsp.mean(dim=1)  # (B, P, cond_dim_t)

            if self.compact_variant == "dual_p":
                # === Dual P-layer ===
                # P1: reg-a + reg-c, P2: reg-a + reg-x, parallel attention
                p1_pool_in = torch.cat([rc_flat, ra_flat], dim=-1)
                p2_pool_in = torch.cat([rx_flat, ra_flat], dim=-1)
                if prev_p_out is not None:
                    p1_pool_in = torch.cat([p1_pool_in, prev_p_out], dim=-1)
                    p2_pool_in = torch.cat([p2_pool_in, prev_p_out], dim=-1)

                p1_tokens = self.dual_p1_pool_projs[p_idx](p1_pool_in)  # (B, P, C_P)
                p2_tokens = self.dual_p2_pool_projs[p_idx](p2_pool_in)  # (B, P, C_P)

                # Build P-layer sequences: [add_tokens, pooled_entities]
                p1_seq = torch.cat([all_add_tokens, p1_tokens], dim=1)  # (B, A+P, C_P)
                p2_seq = torch.cat([all_add_tokens, p2_tokens], dim=1)  # (B, A+P, C_P)

                # Conditioning
                cond_p1 = self.dual_p1_cond_binders[p_idx](cond_p_per_point)  # (B, P, C_P)
                cond_p2 = self.dual_p2_cond_binders[p_idx](cond_p_per_point)  # (B, P, C_P)
                cond_a1 = cond_p1.mean(dim=1, keepdim=True).expand(-1, A, -1)
                cond_a2 = cond_p2.mean(dim=1, keepdim=True).expand(-1, A, -1)
                cond_p1_seq = torch.cat([cond_a1, cond_p1], dim=1)  # (B, A+P, C_P)
                cond_p2_seq = torch.cat([cond_a2, cond_p2], dim=1)  # (B, A+P, C_P)

                # Attention masks (same for both -- same sequence structure)
                R_p1 = self.dual_p1_blocks[p_idx].R
                R_p2 = self.dual_p2_blocks[p_idx].R
                attn_mask_p1 = build_attention_mask(
                    p_seq_valid, mode="full", left_pad=R_p1,
                )[:, None].expand(-1, self.n_head_p, -1, -1)
                attn_mask_p2 = build_attention_mask(
                    p_seq_valid, mode="full", left_pad=R_p2,
                )[:, None].expand(-1, self.n_head_p, -1, -1)

                # Set dual spatial RoPE for both P-blocks
                if self.spatial_rope_flag:
                    self.dual_p1_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=A,
                    )
                    self.dual_p2_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=A,
                    )

                # Run P1 and P2 attention (no data dependency -- parallel)
                p1_out_seq = self.dual_p1_blocks[p_idx](
                    x=p1_seq, cond=cond_p1_seq, t=None,
                    num_external_regs=A, attn_mask=attn_mask_p1,
                )
                p2_out_seq = self.dual_p2_blocks[p_idx](
                    x=p2_seq, cond=cond_p2_seq, t=None,
                    num_external_regs=A, attn_mask=attn_mask_p2,
                )
                p1_out = p1_out_seq[:, A:]  # (B, P, C_P)
                p2_out = p2_out_seq[:, A:]  # (B, P, C_P)

                # Merge reg-a: Linear(cat(p1_out, p2_out)) -> C_P
                merged_a = self.dual_p_merge[p_idx](
                    torch.cat([p1_out, p2_out], dim=-1)
                )  # (B, P, C_P)
                prev_p_out = merged_a

                # Mod 4: project merged output for next t-layer's layer0_mix
                p_proj = self.p_to_ct_projs[p_idx](merged_a)  # (B, P, C_T)
                latest_p_proj = p_proj[:, None].expand(
                    -1, S, -1, -1
                )  # (B, S, P, C_T)

                # Unpool: reg-c from P1, reg-x from P2, reg-a from merged
                p1_exp = p1_out[:, :, None, :].expand(-1, -1, nrc, -1)
                rc_r = reg_c_out.reshape(B, P, nrc, self.C_T)
                reg_c_state = self.unpool_c_ffns[p_idx](
                    torch.cat([rc_r, p1_exp], dim=-1)
                )

                p2_exp = p2_out[:, :, None, :].expand(-1, -1, nrx, -1)
                rx_r = reg_x_out.reshape(B, P, nrx, self.C_T)
                reg_x_state = self.unpool_x_ffns[p_idx](
                    torch.cat([rx_r, p2_exp], dim=-1)
                )

                ma_exp = merged_a[:, :, None, :].expand(-1, -1, nra, -1)
                ra_r = reg_a_out.reshape(B, P, nra, self.C_T)
                reg_a_state = self.unpool_a_ffns[p_idx](
                    torch.cat([ra_r, ma_exp], dim=-1)
                )

            elif self.parallel_p_layer:
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

                # Pool each register -> C_P
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

                    # set dual spatial RoPE for parallel P-block
                    if self.spatial_rope_flag:
                        self.par_p_blocks[p_idx][r_idx].set_dual_spatial_rope_params(
                            pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                            num_external_regs=A,
                        )

                    x_after_attn, mlp_params = self.par_p_blocks[p_idx][
                        r_idx
                    ].forward_attn_only(
                        x=p_seq, cond=cond_p_seq, t=None,
                        num_external_regs=A, attn_mask=attn_mask_p,
                    )
                    attn_outs.append(x_after_attn)
                    ada_params.append(mlp_params)

                # Phase 2: Bridge sync -- extract P-portions, cat, project
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

                # Mod 4: project p-layer output for next t-layer's layer0_mix
                # Use mean of all parallel p-layer outputs as the p-layer representation
                p_out_mean = torch.stack(par_p_outs, dim=0).mean(dim=0)  # (B, P, C_P)
                p_proj = self.p_to_ct_projs[p_idx](p_out_mean)  # (B, P, C_T)
                latest_p_proj = p_proj[:, None].expand(-1, S, -1, -1)  # (B, S, P, C_T)

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
                # === v7.4: Unified P-layer with consecutive-P group support ===
                group_idx = self._p_to_group[p_idx]

                if self._p_is_group_start[p_idx]:
                    # === Group start: pool registers -> C_P ===
                    all_flat = torch.cat([rc_flat, rx_flat, ra_flat], dim=-1)
                    if prev_p_out is not None:
                        all_flat = torch.cat([all_flat, prev_p_out], dim=-1)
                    p_tokens = self.unified_pool_projs[group_idx](all_flat)  # (B, P, C_P)
                else:
                    # === Non-first P in group: input = previous P output ===
                    p_tokens = prev_p_out  # already (B, P, C_P)

                assert not (self.group_p_layer or self.v983_cross_layer_pool) or group_ids is not None, \
                    "group_p_layer/v983 requires group_ids to be passed"

                # V98/V981/V982: Stage 1+2 (skip for V983)
                pooled = None  # will hold group-pooled feature if applicable
                if self.group_p_layer and group_ids is not None and not self.v983_cross_layer_pool:
                    # === V98: 3-stage group P-layer ===

                    # Stage 1: Inner-group attention
                    group_R = self.group_p_blocks[p_idx].R
                    group_mask = self._build_group_attn_mask(
                        group_ids, p_valid, group_R,
                    )  # (B, n_head_p, R+P, R+P)
                    group_cond = self.group_p_cond_binders[p_idx](
                        cond_p_per_point
                    )  # (B, P, C_P)
                    # set dual spatial RoPE for group P-block
                    if self.spatial_rope_flag:
                        self.group_p_blocks[p_idx].set_dual_spatial_rope_params(
                            pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                            num_external_regs=0,
                        )

                    p_tokens = self.group_p_blocks[p_idx](
                        x=p_tokens, cond=group_cond, t=None,
                        num_external_regs=0, attn_mask=group_mask,
                    )  # (B, P, C_P)

                    # Stage 2: Group pooling + cat + project
                    pooled = self._group_max_pool(
                        p_tokens, group_ids, p_valid,
                    )  # (B, P, C_P)
                    p_tokens = self.group_pool_projs[p_idx](
                        torch.cat([p_tokens, pooled], dim=-1)
                    )  # (B, P, C_P)

                # Stage 3 (or only stage): Standard full attention
                # Build conditioning -- augmented by V981/V982/V983
                cond_p = self.p_cond_binders[p_idx](cond_p_per_point)

                if self.v983_cross_layer_pool and group_ids is not None:
                    # V983: augment with prev layer's group feat
                    if prev_group_feat is None:
                        gf = self.v983_group_init.expand(B, P, -1).to(dtype=cond_p.dtype)
                    else:
                        gf = prev_group_feat
                    cond_p = self.v983_adaln_combine[p_idx](
                        torch.cat([cond_p, gf], dim=-1)
                    )
                elif self.v982_hyper_group and pooled is not None:
                    # V982: hyper-network group feature
                    group_feat = self._v982_hyper_forward(pooled, p_idx)
                    cond_p = self.v982_adaln_combine[p_idx](
                        torch.cat([cond_p, group_feat], dim=-1)
                    )
                elif self.v981_pool_to_adaln and pooled is not None:
                    # V981: projected pool to adaln
                    pool_proj = self.v981_pool_proj[p_idx](pooled)
                    cond_p = self.v981_adaln_combine[p_idx](
                        torch.cat([cond_p, pool_proj], dim=-1)
                    )

                p_seq = torch.cat([all_add_tokens, p_tokens], dim=1)
                cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)
                cond_p_seq = torch.cat([cond_a, cond_p], dim=1)

                attn_mask_p = build_attention_mask(
                    p_seq_valid, mode="full",
                    left_pad=self.p_blocks[p_idx].R,
                )[:, None].expand(-1, self.n_head_p, -1, -1)

                # set dual spatial RoPE for P-block
                if self.spatial_rope_flag:
                    self.p_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=A,
                    )

                p_out_seq = self.p_blocks[p_idx](
                    x=p_seq, cond=cond_p_seq, t=None,
                    num_external_regs=A, attn_mask=attn_mask_p,
                )
                p_out = p_out_seq[:, A:]  # (B, P, C_P)

                # V983: pool AFTER attention, store for next P layer
                if self.v983_cross_layer_pool and group_ids is not None:
                    prev_group_feat = self._group_max_pool(
                        p_out, group_ids, p_valid,
                    )

                prev_p_out = p_out

                if self._p_is_group_end[p_idx]:
                    # === Group end: project + unpool ===
                    # Mod 4: project p-layer output for next t-layer's layer0_mix
                    p_proj = self.p_to_ct_projs[group_idx](p_out)  # (B, P, C_T)
                    latest_p_proj = p_proj[:, None].expand(-1, S, -1, -1)  # (B, S, P, C_T)

                    # Unpool: cat(reg_token, p_out) -> FFN for each type
                    # NOTE: reg_c/x/a_out are from the last t-layer BEFORE this P-group.
                    # For consecutive P's, no t-layer runs between them, so these are
                    # intentionally the same registers for the entire group.
                    p_exp = p_out[:, :, None, :].expand(-1, -1, nrc, -1)
                    rc_r = reg_c_out.reshape(B, P, nrc, self.C_T)
                    reg_c_state = self.unpool_c_ffns[group_idx](torch.cat([rc_r, p_exp], dim=-1))

                    p_exp_x = p_out[:, :, None, :].expand(-1, -1, nrx, -1)
                    rx_r = reg_x_out.reshape(B, P, nrx, self.C_T)
                    reg_x_state = self.unpool_x_ffns[group_idx](torch.cat([rx_r, p_exp_x], dim=-1))

                    p_exp_a = p_out[:, :, None, :].expand(-1, -1, nra, -1)
                    ra_r = reg_a_out.reshape(B, P, nra, self.C_T)
                    reg_a_state = self.unpool_a_ffns[group_idx](torch.cat([ra_r, p_exp_a], dim=-1))

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

                # set dual spatial RoPE for sub-P-c block
                if self.spatial_rope_flag:
                    self.sub_p_c_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=0,
                    )

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

                # set dual spatial RoPE for sub-P-x block
                if self.spatial_rope_flag:
                    self.sub_p_x_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=0,
                    )

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

                # set dual spatial RoPE for v7.0 main P-block
                if self.spatial_rope_flag:
                    self.p_blocks[p_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_9d, pos_b=mean_x_9d,
                        num_external_regs=A,
                    )

                p_out_seq = self.p_blocks[p_idx](
                    x=p_seq, cond=cond_p_seq, t=None,
                    num_external_regs=A, attn_mask=attn_mask_p,
                )
                p_out = p_out_seq[:, A:]
                prev_p_out = p_out

                # Mod 4: project p-layer output for next t-layer's layer0_mix
                p_proj = self.p_to_ct_projs[p_idx](p_out)  # (B, P, C_T)
                latest_p_proj = p_proj[:, None].expand(-1, S, -1, -1)  # (B, S, P, C_T)

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
    """Smoke test for MotionMachineV11Compact -- various configs."""
    import sys

    B, T, P, C = 2, 8, 64, 9
    CTX = 3
    A = 16
    C_T, C_P = 128, 512

    combos = [
        # Baseline: identical to V7 (indicator_dim=0)
        {"isolated_ctx_x": False, "time_rope_flag": False, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 0},
        # Mod 2: with indicator
        {"isolated_ctx_x": False, "time_rope_flag": False, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16},
        # Mod 2 + isolated_ctx_x
        {"isolated_ctx_x": True, "time_rope_flag": False, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16},
        # RoPE (indicator should still work but time_bind_proj is skipped)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16},
        # Zero registers
        {"isolated_ctx_x": False, "time_rope_flag": False, "n_reg_c": 0, "n_reg_x": 0, "n_reg_a": 0,
         "ctx_trace_indicator_dim": 16},
        # spatial split RoPE with trajectory PE
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024},
        # explicit layer_type_string (standard pattern)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "tptptpt"},
        # non-standard pattern (3 t-layers, 1 p-layer)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "tttpt"},
        # t-heavy pattern (5 t-layers, 1 p-layer)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "tpttttt"},
        # Consecutive P: backward compat tptptpt (should match standard)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "tptptpt", "unified_p_layer": True},
        # Consecutive P: ttttpptppt (6t, 4p, 2 groups)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "ttttpptppt", "unified_p_layer": True},
        # Consecutive P: tttpppttt (6t, 3p, 1 group)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "tttpppttt", "unified_p_layer": True},
        # Consecutive P: ttpttptt (6t, 2p, 2 groups)
        {"isolated_ctx_x": False, "time_rope_flag": True, "n_reg_c": 2, "n_reg_x": 2, "n_reg_a": 2,
         "ctx_trace_indicator_dim": 16, "spatial_rope_flag": True, "n_traj_tokens": 1024,
         "layer_type_string": "ttpttptt", "unified_p_layer": True},
    ]

    for combo in combos:
        label = ", ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        net = MotionMachineV11Compact(
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

        print(f"Output shape: {out.shape}")
        assert out.shape == (B, T + CTX, P, C), f"Expected ({B}, {T+CTX}, {P}, {C}), got {out.shape}"
        print("PASS")

    # === Backward compatibility param count check ===
    print(f"\n{'='*60}")
    print("Backward compat: param count check (tptptpt)")
    print(f"{'='*60}")
    base_kwargs = dict(
        C_T=C_T, C_P=C_P, feat_dim=512, n_head_t=8, n_head_p=16,
        n_t_layers=4, n_ctx=16, fc_factor_t=4, fc_factor_p=2,
        fc_n_layers=2, fc_out_n_layers=12, num_blk_internal_registers=4,
        ada_scale_bias_bound=1.0, gate_bound=2.0, add_time_to_feat=True,
        qk_norm=True, isolated_ctx_x=False, time_rope_flag=True,
        n_reg_c=2, n_reg_x=2, n_reg_a=2, ctx_trace_indicator_dim=16,
        spatial_rope_flag=True, n_traj_tokens=1024, unified_p_layer=True,
    )
    net_default = MotionMachineV11Compact(**base_kwargs)
    net_explicit = MotionMachineV11Compact(layer_type_string="tptptpt", **base_kwargs)
    n_default = sum(p.numel() for p in net_default.parameters())
    n_explicit = sum(p.numel() for p in net_explicit.parameters())
    assert n_default == n_explicit, (
        f"Param count mismatch! default={n_default}, explicit tptptpt={n_explicit}"
    )
    print(f"Default (no string): {n_default} params")
    print(f"Explicit tptptpt:    {n_explicit} params")
    print("PASS -- param counts match")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_main()
