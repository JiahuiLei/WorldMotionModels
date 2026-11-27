"""MotionMachine V6 -- Two-scale pooled temporal attention.

Architecture: alternating t-p layers (first and last are t-layers).
Pattern: t-p-t-p-t-p-t

- **t-layer**: Per-trajectory self-attention on [Q learnable tokens ; S timesteps].
  Operates at narrow dimension C_T (~128). P independent sequences in parallel.
  Each t-layer has its OWN learnable Q embeddings.

- **Pooling (t->p)**: Per-trajectory Q tokens flattened to Q*C_T, optionally
  concatenated with previous p-layer output (skip), projected to C_P.

- **p-layer**: Self-attention on [A add-tokens ; P pooled vectors] at wide
  dimension C_P (~512). Applied ONCE (not S times). adaLN conditioning.

- **Unpooling (p->t)**: Each Q token gets p-layer output injected via FFN:
  [Q_i (C_T) ; p_out (C_P)] -> FFN -> C_T.

- **Output**: Read S positions from last t-layer. Skip connections at t-layers only.

Variants:
- **v6.8 (dual_t_layer=True)**: Each t-layer splits into two identical sub-layers.
  Sub-layer A processes [Q_ctx(4); ctx_frames(N_CTX)], sub-layer B processes
  [Q_x(4); x_frames(T)]. Combined 8 Q tokens are pooled together into the p-layer.
  Two separate unpool FFNs (one for ctx Q, one for X Q).

- **v6.9 (traj_pos_emb=True)**: Fixed sinusoidal trajectory position embedding.
  During training, random permutation of [0..P-1]; at inference, identity [0..P-1].
  Two integration paths: (a) concat to projected x + linear proj back,
  (b) additional adaLN conditioning component. Supersedes num_random_trace_id_pe.

Example::

    io = IdentifyLayerV2(latent_dim=128, feat_dim=512)
    net = MotionMachineV6(C_T=128, C_P=512, feat_dim=512, n_t_layers=4, n_p_layers=3)
    feat = io(sem_id, group_id, pcl, pcl_mask)
    out = net(x, x_mask, x_time, x_level, feat)
"""

import logging
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v6_all.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
    NerfStylePE,
)
from umm.models.umm_v6_all.world_models.dit_atten import DiTBlockTimeOnly


class MotionMachineV6(nn.Module):
    """Two-scale pooled temporal attention MotionMachine.

    Layer pattern: t-p-t-p-...-t (alternating, first and last are t-layers).
    n_t_layers t-layers and n_p_layers = n_t_layers - 1 p-layers.

    Key design:
        - C_T: narrow dim for t-layers (operates on all P trajectories)
        - C_P: wide dim for p-layers (operates on pooled representation, once)
        - Q learnable tokens per t-layer gather information (like slot attention)
        - Add tokens (lang + image) only participate in p-layers
        - Output skip connections at t-layers only

    Args:
        C_T: Narrow channel width for t-layers.
        C_P: Wide channel width for p-layers.
        feat_dim: Input feature dimension from identify layer.
        n_head_t: Number of attention heads for t-layers.
        n_head_p: Number of attention heads for p-layers.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        n_t_layers: Number of t-layers. Must be >= 2.
        n_Q: Number of learnable query tokens per t-layer.
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
        num_random_trace_id_pe: Random trace ID PE count (<= 0 disables).
        dual_t_layer: v6.8 -- split each t-layer into ctx/X sub-layers.
        traj_pos_emb: v6.9 -- fixed sinusoidal trajectory position embedding.
        latent_dim: Legacy/compatibility param (equals C_T if not overridden).
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
        n_Q: int = 4,
        n_ctx: int = 128,
        #
        fc_factor_t: int = 2,
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
        num_random_trace_id_pe: int = -1,
        #
        dual_t_layer: bool = False,
        traj_pos_emb: bool = False,
        ctx_p_layer: bool = False,
        dense_p_block_size: int = 0,
        n_dense_p_layers: int = -1,
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
        qk_norm: bool = False,
    ):
        super().__init__()

        self.C_T = C_T
        self.C_P = C_P
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.n_t_layers = n_t_layers
        self.n_p_layers = n_t_layers - 1
        self.n_Q = n_Q
        self.n_head_t = n_head_t
        self.n_head_p = n_head_p
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat
        self.dual_t_layer = dual_t_layer
        self.traj_pos_emb = traj_pos_emb
        self.ctx_p_layer = ctx_p_layer
        self.dense_p_block_size = dense_p_block_size

        assert n_t_layers >= 2, "Need at least 2 t-layers (first and last)"
        if ctx_p_layer:
            assert dual_t_layer, "ctx_p_layer requires dual_t_layer=True"

        # v6.8.5/v6.8.6: block-dense P-layers
        if dense_p_block_size > 0:
            n_actual = n_dense_p_layers if n_dense_p_layers >= 0 else (n_t_layers - 1)
            self._n_dense_p = min(n_actual, n_t_layers - 1)
        else:
            self._n_dense_p = 0

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_ct = HardLayerNorm(self.C_T)

        # =============== Input Projection ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
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

        # Random trace ID PE (superseded by traj_pos_emb when both enabled)
        self.num_random_trace_id_pe = num_random_trace_id_pe
        if self.num_random_trace_id_pe > 0 and not self.traj_pos_emb:
            logging.warning(f"V6: random trace id pe with {self.num_random_trace_id_pe}")
            self.random_trace_id_pe = nn.Embedding(
                self.num_random_trace_id_pe, self.C_T
            )
            with torch.no_grad():
                self.random_trace_id_pe.weight.copy_(
                    self.random_trace_id_pe.weight[0].expand_as(
                        self.random_trace_id_pe.weight
                    )
                )

        # =============== v6.9: Trajectory Position Embedding ===============
        if self.traj_pos_emb:
            logging.warning("V6.9: trajectory position embedding enabled")
            # Fixed sinusoidal PE (same style as int_pe / Timesteps)
            self.traj_pe = Timesteps(
                self.C_T, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            # Path (a): t-layer projection — concat [x(C_T), PE(C_T)] -> C_T
            self.traj_pe_x_proj_t = nn.Linear(2 * self.C_T, self.C_T)
            # Path (a): p-layer projection — concat [pooled(C_P), PE(C_T)] -> C_P
            self.traj_pe_x_proj_p = nn.Linear(self.C_P + self.C_T, self.C_P)
            # Learnable embedding for add tokens (shared across all add tokens)
            self.traj_pe_add_token = nn.Parameter(torch.zeros(1, 1, self.C_T))
            nn.init.normal_(self.traj_pe_add_token, std=0.02)

        # =============== Conditioning Binders ===============
        # t-layer conditioning: feat_dim + 2*C_T [+ optional extras] -> cond_dim_t
        cond_dim_t_base = self.feat_dim + self.C_T * 2
        if self.num_random_trace_id_pe > 0 and not self.traj_pos_emb:
            cond_dim_t_base += self.C_T
        # v6.9 path (b): add trajectory PE as 4th adaLN component
        if self.traj_pos_emb:
            cond_dim_t_base += self.C_T
        self.cond_dim_t = cond_dim_t_base

        # t-layer cond binders: one per t-layer, maps cond_dim_t -> C_T (DiTBlock C_feat)
        if self.dual_t_layer:
            # v6.8: two cond binders per layer (ctx + x sub-layers)
            self.t_cond_binders_ctx = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_T)
                for _ in range(self.n_t_layers)
            ])
            self.t_cond_binders_x = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_T)
                for _ in range(self.n_t_layers)
            ])
        else:
            self.t_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, self.C_T)
                for _ in range(self.n_t_layers)
            ])

        # p-layer conditioning: mean-pool the t-cond over P, project to C_P
        self.p_cond_binders = nn.ModuleList([
            nn.Linear(self.cond_dim_t, self.C_P)
            for _ in range(self.n_p_layers)
        ])

        # =============== Per-t-layer Learnable Q Embeddings ===============
        if self.dual_t_layer:
            # v6.8: separate Q embeds for ctx and X sub-layers
            self.q_embeds_ctx = nn.ParameterList([
                nn.Parameter(torch.zeros(1, n_Q, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            self.q_embeds_x = nn.ParameterList([
                nn.Parameter(torch.zeros(1, n_Q, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            for q_emb in self.q_embeds_ctx:
                nn.init.normal_(q_emb, std=0.02)
            for q_emb in self.q_embeds_x:
                nn.init.normal_(q_emb, std=0.02)
        else:
            self.q_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, n_Q, self.C_T))
                for _ in range(self.n_t_layers)
            ])
            for q_emb in self.q_embeds:
                nn.init.normal_(q_emb, std=0.02)

        # =============== t-layer Blocks ===============
        # Common DiT block kwargs
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

        if self.dual_t_layer:
            # v6.8: two separate block lists (ctx + X sub-layers)
            self.t_blocks_ctx = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_t_kwargs)
                for _ in range(self.n_t_layers)
            ])
            self.t_blocks_x = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_t_kwargs)
                for _ in range(self.n_t_layers)
            ])
        else:
            self.t_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_t_kwargs)
                for _ in range(self.n_t_layers)
            ])

        # =============== Pooling Projections (t->p) ===============
        if self.ctx_p_layer:
            # v6.8.2: separate ctx/x pooling — ctx goes through half-width P-layer first
            ctx_q_pool_dim = n_Q * self.C_T  # ctx Q tokens only
            x_q_pool_dim = n_Q * self.C_T    # x Q tokens only
            self.ctx_pool_projs = nn.ModuleList()
            self.x_pool_projs = nn.ModuleList()
            for i in range(self.n_p_layers):
                self.ctx_pool_projs.append(nn.Linear(ctx_q_pool_dim, self.C_P // 2))
                if i == 0:
                    self.x_pool_projs.append(nn.Linear(x_q_pool_dim, self.C_P))
                else:
                    self.x_pool_projs.append(
                        nn.Linear(x_q_pool_dim + self.C_P, self.C_P)
                    )
        else:
            # v6.8: pool 2*Q*C_T (8 Q tokens total: 4 ctx + 4 X)
            # standard: pool Q*C_T
            if self.dual_t_layer:
                q_pool_dim = 2 * n_Q * self.C_T  # 8 Q tokens flattened
            else:
                q_pool_dim = n_Q * self.C_T

            self.pool_projs = nn.ModuleList()
            for i in range(self.n_p_layers):
                if i == 0:
                    proj_in = q_pool_dim
                else:
                    proj_in = q_pool_dim + self.C_P
                self.pool_projs.append(nn.Linear(proj_in, self.C_P))

        # =============== p-layer Blocks ===============
        self.p_blocks = nn.ModuleList([
            DiTBlockTimeOnly(
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
            for _ in range(self.n_p_layers)
        ])

        # =============== Add-token Projector (latent_dim -> C_P) ===============
        # Add tokens come in at latent_dim (from Agent's lang/image projectors).
        # We need to project them to C_P for p-layer participation.
        # The Agent will set this dim at runtime via a separate projector,
        # but we need to know the expected input dim.
        # We'll accept C_P-dim add_tokens (Agent projects them before passing).

        # =============== v6.8.2: ctx-P Layer (half-width) ===============
        if self.ctx_p_layer:
            ctx_p_width = self.C_P // 2
            self.ctx_p_width = ctx_p_width
            # ctx-P DiT blocks: half width, half heads
            self.ctx_p_blocks = nn.ModuleList([
                DiTBlockTimeOnly(
                    C=ctx_p_width,
                    num_heads=n_head_p // 2,
                    num_registers=num_blk_internal_registers,
                    mlp_hidden_dims=[ctx_p_width * fc_factor_p] * fc_n_layers,
                    ada_scale_bias_bound=ada_scale_bias_bound,
                    gate_bound=gate_bound,
                    ada_n_layers=ada_n_layers,
                    C_feat=ctx_p_width,
                    qk_norm=qk_norm,
                )
                for _ in range(self.n_p_layers)
            ])
            # Project add tokens from C_P to half width for ctx-P layer
            self.ctx_add_token_projs = nn.ModuleList([
                nn.Linear(self.C_P, ctx_p_width)
                for _ in range(self.n_p_layers)
            ])
            # ctx-P conditioning: cond_dim_t -> ctx_p_width
            self.ctx_p_cond_binders = nn.ModuleList([
                nn.Linear(self.cond_dim_t, ctx_p_width)
                for _ in range(self.n_p_layers)
            ])
            # Cat ctx-P output + x pooled → project to C_P
            self.ctx_x_cat_projs = nn.ModuleList([
                nn.Linear(ctx_p_width + self.C_P, self.C_P)
                for _ in range(self.n_p_layers)
            ])

        # =============== v6.8.5/v6.8.6: Block-Dense P-Layer Projections ===============
        if self._n_dense_p > 0:
            # Pool: mean of block frames → project to C_P
            self.block_pool_projs = nn.ModuleList([
                nn.Linear(self.C_T, self.C_P)
                for _ in range(self._n_dense_p)
            ])
            # Unpool: project block P output → C_T for x_state residual update
            self.block_x_update_projs = nn.ModuleList([
                nn.Linear(self.C_P, self.C_T)
                for _ in range(self._n_dense_p)
            ])

        # =============== Unpooling FFNs (p->t) ===============
        # Maps (C_T + C_P) -> C_T for each Q token
        if self.dual_t_layer:
            # v6.8: separate unpool FFNs for ctx Q and X Q
            self.unpool_ffns_ctx = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.C_T + self.C_P, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
            self.unpool_ffns_x = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.C_T + self.C_P, self.C_T * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C_T * 2, self.C_T),
                )
                for _ in range(self.n_p_layers)
            ])
        else:
            self.unpool_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.C_T + self.C_P, self.C_T * 2),
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
        logging.info(f"LMM V6 Summary:")
        logging.info(f"  C_T (narrow): {self.C_T}")
        logging.info(f"  C_P (wide):   {self.C_P}")
        logging.info(f"  n_t_layers:   {self.n_t_layers}")
        logging.info(f"  n_p_layers:   {self.n_p_layers}")
        logging.info(f"  n_Q:          {self.n_Q}")
        logging.info(f"  input_dim:    {self.input_dim}")
        logging.info(f"  feat_dim:     {self.feat_dim}")
        logging.info(f"  pattern:      {'t-p-' * self.n_p_layers}t")
        if self.dual_t_layer:
            logging.info(f"  v6.8:         dual_t_layer ENABLED (ctx/X split)")
        if self.ctx_p_layer:
            logging.info(f"  v6.8.2:       ctx_p_layer ENABLED (half-width={self.C_P // 2})")
        if self._n_dense_p > 0:
            logging.info(f"  v6.8.5/6:     block-dense P-layers: {self._n_dense_p}/{self.n_p_layers} dense, block_size={self.dense_p_block_size}")
        if self.traj_pos_emb:
            logging.info(f"  v6.9:         traj_pos_emb ENABLED (sinusoidal, paths a+b)")

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

    def _run_t_layer_standard(
        self,
        layer_idx: int,
        x_state: torch.Tensor,
        q_state: Optional[torch.Tensor],
        cond_bsp: torch.Tensor,
        t_seq_valid: torch.Tensor,
        t_rope: Optional[torch.Tensor],
        B: int, S: int, P: int, Q: int, N_CTX: int,
        traj_pe_feat_bsp: Optional[torch.Tensor] = None,
    ):
        """Run a standard (non-dual) t-layer.

        Args:
            layer_idx: Index of the current t-layer.
            x_state: (B, S, P, C_T) current state.
            q_state: (B, P, Q, C_T) or None for first layer.
            cond_bsp: (B, S, P, cond_dim_t) conditioning.
            t_seq_valid: (B*P, Q+S) validity mask for [Q; S] sequence.
            t_rope: (B*P, T) or None for RoPE time values.
            B, S, P, Q, N_CTX: shape info.
            traj_pe_feat_bsp: (B, S, P, C_T) or None — v6.9 path (a) PE for t-layers.

        Returns:
            x_state: (B, S, P, C_T) updated state.
            q_out: (B*P, Q, C_T) Q token outputs.
        """
        q_emb = self.q_embeds[layer_idx]  # (1, Q, C_T)

        if q_state is not None:
            q_tokens = q_state  # (B, P, Q, C_T)
        else:
            q_tokens = q_emb.expand(B * P, -1, -1).reshape(B, P, Q, self.C_T)

        # Reshape for per-trajectory self-attention
        x_traj = x_state.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)  # (B*P, S, C_T)

        # v6.9 path (a) for t-layers: concat traj PE to x_traj and project
        if traj_pe_feat_bsp is not None:
            traj_pe_traj = traj_pe_feat_bsp.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)
            x_traj = self.traj_pe_x_proj_t(
                torch.cat([x_traj, traj_pe_traj], dim=-1)
            )  # (B*P, S, C_T)

        q_traj = q_tokens.reshape(B * P, Q, self.C_T)
        traj_seq = torch.cat([q_traj, x_traj], dim=1)  # (B*P, Q+S, C_T)

        # Build conditioning for this t-layer
        cond_t_layer = self.t_cond_binders[layer_idx](cond_bsp)  # (B, S, P, C_T)
        cond_t_per_traj = cond_t_layer.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)
        cond_q_mean = cond_t_per_traj.mean(dim=1, keepdim=True).expand(-1, Q, -1)
        cond_traj = torch.cat([cond_q_mean, cond_t_per_traj], dim=1)  # (B*P, Q+S, C_T)

        # Build attention mask
        attn_mask_t = build_attention_mask(
            t_seq_valid,
            mode="full",
            left_pad=self.t_blocks[layer_idx].R,
        )[:, None].expand(-1, self.n_head_t, -1, -1)

        # Run t-block
        traj_out = self.t_blocks[layer_idx](
            x=traj_seq,
            cond=cond_traj,
            t=t_rope,
            num_external_regs=Q + N_CTX,
            attn_mask=attn_mask_t,
        )  # (B*P, Q+S, C_T)

        q_out = traj_out[:, :Q]  # (B*P, Q, C_T)
        x_out = traj_out[:, Q:]  # (B*P, S, C_T)
        x_state = x_out.reshape(B, P, S, self.C_T).permute(0, 2, 1, 3)  # (B, S, P, C_T)

        return x_state, q_out

    def _run_t_layer_dual(
        self,
        layer_idx: int,
        x_state: torch.Tensor,
        q_state_ctx: Optional[torch.Tensor],
        q_state_x: Optional[torch.Tensor],
        cond_bsp: torch.Tensor,
        ctx_valid_bp: torch.Tensor,
        x_valid_bp: torch.Tensor,
        t_rope: Optional[torch.Tensor],
        B: int, S: int, P: int, Q: int, N_CTX: int, T: int,
        traj_pe_feat_bsp: Optional[torch.Tensor] = None,
    ):
        """Run a dual (v6.8) t-layer: separate ctx and X sub-layers.

        Args:
            layer_idx: Index of the current t-layer.
            x_state: (B, S, P, C_T) current state (S = N_CTX + T).
            q_state_ctx: (B, P, Q, C_T) or None for first layer.
            q_state_x: (B, P, Q, C_T) or None for first layer.
            cond_bsp: (B, S, P, cond_dim_t) conditioning.
            ctx_valid_bp: (B*P, N_CTX) validity mask for ctx frames.
            x_valid_bp: (B*P, T) validity mask for X frames.
            t_rope: (B*P, T) or None for RoPE time values.
            B, S, P, Q, N_CTX, T: shape info.
            traj_pe_feat_bsp: (B, S, P, C_T) or None — v6.9 path (a) PE.

        Returns:
            x_state: (B, S, P, C_T) updated state (concat of ctx_out and x_out).
            q_out_ctx: (B*P, Q, C_T) ctx Q token outputs.
            q_out_x: (B*P, Q, C_T) X Q token outputs.
        """
        # --- Sub-layer A: ctx frames ---
        q_emb_ctx = self.q_embeds_ctx[layer_idx]  # (1, Q, C_T)
        if q_state_ctx is not None:
            q_tokens_ctx = q_state_ctx  # (B, P, Q, C_T)
        else:
            q_tokens_ctx = q_emb_ctx.expand(B * P, -1, -1).reshape(B, P, Q, self.C_T)

        if N_CTX > 0:
            # Extract ctx portion: first N_CTX timesteps
            ctx_traj = x_state[:, :N_CTX].permute(0, 2, 1, 3).reshape(B * P, N_CTX, self.C_T)

            # v6.9 path (a): concat traj PE to ctx portion
            if traj_pe_feat_bsp is not None:
                ctx_pe = traj_pe_feat_bsp[:, :N_CTX].permute(0, 2, 1, 3).reshape(B * P, N_CTX, self.C_T)
                ctx_traj = self.traj_pe_x_proj_t(
                    torch.cat([ctx_traj, ctx_pe], dim=-1)
                )

            q_traj_ctx = q_tokens_ctx.reshape(B * P, Q, self.C_T)
            ctx_seq = torch.cat([q_traj_ctx, ctx_traj], dim=1)  # (B*P, Q+N_CTX, C_T)

            # Conditioning for ctx sub-layer
            cond_ctx_layer = self.t_cond_binders_ctx[layer_idx](cond_bsp[:, :N_CTX])  # (B, N_CTX, P, C_T)
            cond_ctx_traj = cond_ctx_layer.permute(0, 2, 1, 3).reshape(B * P, N_CTX, self.C_T)
            cond_q_ctx = cond_ctx_traj.mean(dim=1, keepdim=True).expand(-1, Q, -1)
            cond_ctx_seq = torch.cat([cond_q_ctx, cond_ctx_traj], dim=1)  # (B*P, Q+N_CTX, C_T)

            # Attention mask for ctx sub-layer
            q_valid_ctx = torch.ones(B * P, Q, dtype=torch.bool, device=x_state.device)
            ctx_seq_valid = torch.cat([q_valid_ctx, ctx_valid_bp], dim=1)  # (B*P, Q+N_CTX)
            attn_mask_ctx = build_attention_mask(
                ctx_seq_valid,
                mode="full",
                left_pad=self.t_blocks_ctx[layer_idx].R,
            )[:, None].expand(-1, self.n_head_t, -1, -1)

            # Run ctx sub-block (no RoPE for ctx — ctx uses positional embedding, not time)
            ctx_out = self.t_blocks_ctx[layer_idx](
                x=ctx_seq,
                cond=cond_ctx_seq,
                t=None,
                num_external_regs=Q + N_CTX,  # all are non-time (Q + ctx positions)
                attn_mask=attn_mask_ctx,
            )  # (B*P, Q+N_CTX, C_T)

            q_out_ctx = ctx_out[:, :Q]  # (B*P, Q, C_T)
        else:
            # No context frames: ctx Q tokens stay at initial embeddings, no ctx output
            q_out_ctx = q_tokens_ctx.reshape(B * P, Q, self.C_T)  # (B*P, Q, C_T)

        # Extract ctx frame outputs (empty tensor when N_CTX=0, which is fine for concat)
        if N_CTX > 0:
            ctx_x_out = ctx_out[:, Q:]  # (B*P, N_CTX, C_T)
        else:
            ctx_x_out = torch.zeros(B * P, 0, self.C_T, device=x_state.device, dtype=x_state.dtype)

        # --- Sub-layer B: X (noisy) frames ---
        q_emb_x = self.q_embeds_x[layer_idx]  # (1, Q, C_T)
        if q_state_x is not None:
            q_tokens_x = q_state_x  # (B, P, Q, C_T)
        else:
            q_tokens_x = q_emb_x.expand(B * P, -1, -1).reshape(B, P, Q, self.C_T)

        # Extract X portion: last T timesteps
        x_traj = x_state[:, N_CTX:].permute(0, 2, 1, 3).reshape(B * P, T, self.C_T)

        # v6.9 path (a): concat traj PE to X portion
        if traj_pe_feat_bsp is not None:
            x_pe = traj_pe_feat_bsp[:, N_CTX:].permute(0, 2, 1, 3).reshape(B * P, T, self.C_T)
            x_traj = self.traj_pe_x_proj_t(
                torch.cat([x_traj, x_pe], dim=-1)
            )

        q_traj_x = q_tokens_x.reshape(B * P, Q, self.C_T)
        x_seq = torch.cat([q_traj_x, x_traj], dim=1)  # (B*P, Q+T, C_T)

        # Conditioning for X sub-layer
        cond_x_layer = self.t_cond_binders_x[layer_idx](cond_bsp[:, N_CTX:])  # (B, T, P, C_T)
        cond_x_traj = cond_x_layer.permute(0, 2, 1, 3).reshape(B * P, T, self.C_T)
        cond_q_x = cond_x_traj.mean(dim=1, keepdim=True).expand(-1, Q, -1)
        cond_x_seq = torch.cat([cond_q_x, cond_x_traj], dim=1)  # (B*P, Q+T, C_T)

        # Attention mask for X sub-layer
        q_valid_x = torch.ones(B * P, Q, dtype=torch.bool, device=x_state.device)
        x_seq_valid = torch.cat([q_valid_x, x_valid_bp], dim=1)  # (B*P, Q+T)
        attn_mask_x = build_attention_mask(
            x_seq_valid,
            mode="full",
            left_pad=self.t_blocks_x[layer_idx].R,
        )[:, None].expand(-1, self.n_head_t, -1, -1)

        # Run X sub-block
        x_out = self.t_blocks_x[layer_idx](
            x=x_seq,
            cond=cond_x_seq,
            t=t_rope,
            num_external_regs=Q,  # only Q tokens are non-time
            attn_mask=attn_mask_x,
        )  # (B*P, Q+T, C_T)

        q_out_x = x_out[:, :Q]  # (B*P, Q, C_T)
        x_x_out = x_out[:, Q:]  # (B*P, T, C_T)

        # --- Merge ctx_out and x_out back to (B, S, P, C_T) ---
        ctx_x_state = ctx_x_out.reshape(B, P, N_CTX, self.C_T).permute(0, 2, 1, 3)  # (B, N_CTX, P, C_T)
        x_x_state = x_x_out.reshape(B, P, T, self.C_T).permute(0, 2, 1, 3)  # (B, T, P, C_T)
        x_state = torch.cat([ctx_x_state, x_x_state], dim=1)  # (B, S, P, C_T)

        return x_state, q_out_ctx, q_out_x

    def _is_dense_p_layer(self, p_idx: int) -> bool:
        """Check if p-layer at index p_idx should use block-dense attention."""
        return self._n_dense_p > 0 and p_idx < self._n_dense_p

    def _run_p_layer_block_dense(
        self,
        p_idx: int,
        x_state: torch.Tensor,
        cond_bsp: torch.Tensor,
        all_add_tokens: torch.Tensor,
        all_add_mask: torch.Tensor,
        p_valid: torch.Tensor,
        B: int, S: int, P: int, A: int,
    ):
        """Run a block-dense P-layer: per-block spatial attention.

        Groups S frames into blocks of K, pools within each block,
        runs P-layer per block, broadcasts output back to frames.

        Args:
            p_idx: P-layer index (used to index into block_pool_projs etc.)
            x_state: (B, S, P, C_T) current state from t-layer.
            cond_bsp: (B, S, P, cond_dim_t) conditioning.
            all_add_tokens: (B, A, C_P) add tokens.
            all_add_mask: (B, A) add token mask.
            p_valid: (B, P) per-part validity mask.
            B, S, P, A: shape info.

        Returns:
            x_state: (B, S, P, C_T) updated state (residual from block P outputs).
            p_out_mean: (B, P, C_P) mean of block P outputs (for prev_p_out skip).
        """
        device = x_state.device
        K = self.dense_p_block_size
        n_blocks = (S + K - 1) // K  # ceil division

        # --- Pad S to multiple of K ---
        pad_len = n_blocks * K - S
        if pad_len > 0:
            x_padded = torch.cat(
                [x_state, x_state[:, -1:].expand(-1, pad_len, -1, -1)], dim=1
            )  # (B, n_blocks*K, P, C_T)
            cond_padded = torch.cat(
                [cond_bsp, cond_bsp[:, -1:].expand(-1, pad_len, -1, -1)], dim=1
            )  # (B, n_blocks*K, P, cond_dim_t)
        else:
            x_padded = x_state
            cond_padded = cond_bsp

        # --- Block-pool: mean over K frames per block ---
        x_blocks = x_padded.reshape(B, n_blocks, K, P, self.C_T)
        x_block_mean = x_blocks.mean(dim=2)  # (B, n_blocks, P, C_T)

        # Project to C_P
        p_tokens_dense = self.block_pool_projs[p_idx](x_block_mean)  # (B, n_blocks, P, C_P)
        p_tokens_flat = p_tokens_dense.reshape(B * n_blocks, P, self.C_P)

        # --- Replicate add tokens per block ---
        if A > 0:
            add_rep = all_add_tokens[:, None].expand(-1, n_blocks, -1, -1).reshape(
                B * n_blocks, A, self.C_P
            )
            add_mask_rep = all_add_mask[:, None].expand(-1, n_blocks, -1).reshape(
                B * n_blocks, A
            )
        else:
            add_rep = torch.zeros(B * n_blocks, 0, self.C_P, device=device, dtype=x_state.dtype)
            add_mask_rep = torch.zeros(B * n_blocks, 0, device=device).bool()

        p_valid_rep = p_valid[:, None].expand(-1, n_blocks, -1).reshape(B * n_blocks, P)
        p_seq_valid_rep = torch.cat([add_mask_rep, p_valid_rep], dim=1)

        # --- P-layer input ---
        p_seq = torch.cat([add_rep, p_tokens_flat], dim=1)  # (B*n_blocks, A+P, C_P)

        # --- Conditioning: per-block average ---
        cond_blocks = cond_padded.reshape(B, n_blocks, K, P, -1).mean(dim=2)  # (B, n_blocks, P, cond_dim_t)
        cond_p = self.p_cond_binders[p_idx](cond_blocks)  # (B, n_blocks, P, C_P)
        cond_p_flat = cond_p.reshape(B * n_blocks, P, self.C_P)
        cond_a = cond_p_flat.mean(dim=1, keepdim=True).expand(-1, A, -1)
        cond_p_seq = torch.cat([cond_a, cond_p_flat], dim=1)  # (B*n_blocks, A+P, C_P)

        # --- Attention mask ---
        attn_mask_p = build_attention_mask(
            p_seq_valid_rep,
            mode="full",
            left_pad=self.p_blocks[p_idx].R,
        )[:, None].expand(-1, self.n_head_p, -1, -1)

        # --- Run P-block ---
        p_out_seq = self.p_blocks[p_idx](
            x=p_seq, cond=cond_p_seq, t=None,
            num_external_regs=A, attn_mask=attn_mask_p,
        )  # (B*n_blocks, A+P, C_P)
        p_out_flat = p_out_seq[:, A:]  # (B*n_blocks, P, C_P)
        p_out_blocks = p_out_flat.reshape(B, n_blocks, P, self.C_P)

        # --- Broadcast back to frames and update x_state ---
        # (B, n_blocks, 1, P, C_P) → expand → (B, n_blocks, K, P, C_P)
        p_out_broadcast = p_out_blocks[:, :, None].expand(-1, -1, K, -1, -1)
        p_out_frames = p_out_broadcast.reshape(B, n_blocks * K, P, self.C_P)[:, :S]  # trim padding

        x_update = self.block_x_update_projs[p_idx](p_out_frames)  # (B, S, P, C_T)
        x_state = x_state + x_update  # residual update

        # --- Mean across blocks for skip connection and Q update ---
        p_out_mean = p_out_blocks.mean(dim=1)  # (B, P, C_P)

        return x_state, p_out_mean

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
        """Forward pass with two-scale pooled temporal attention.

        Args:
            x: [B, T, P, input_dim] noisy input.
            x_mask: [B, T, P] bool mask.
            x_time: [B, T] timestamps.
            x_level: [B, T, P] diffusion levels.
            feat: [B, P, feat_dim] per-part identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: [B, A_lang, C_P] language tokens (already projected to C_P).
            add_tokens_mask: [B, A_lang] mask.
            image_tokens: [B, A_img, C_P] image tokens (already projected to C_P).
            image_tokens_mask: [B, A_img] mask.

        Returns:
            [B, S, P, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
        Q = self.n_Q
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
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(-1, -1, P, -1)  # (B, S, P, C_T)

        cond_parts = [
            self.hard_ln_feat(feat_expanded),
            self.hard_ln_ct(level_emb),
            self.hard_ln_ct(ctx_pos_emb_expanded),
        ]

        # v6.9: trajectory position embedding (supersedes random_trace_id_pe)
        traj_pe_feat = None  # (B, P, C_T) — used for path (a)
        if self.traj_pos_emb:
            # Generate permutation
            if self.training:
                perm = torch.argsort(
                    torch.rand(B, P, device=device), dim=-1
                )  # (B, P) random permutation of [0..P-1]
            else:
                perm = torch.arange(P, device=device)[None].expand(B, -1)  # (B, P) identity

            # Compute fixed sinusoidal PE
            traj_pe_feat = self.traj_pe(
                perm.reshape(-1).float()
            ).reshape(B, P, self.C_T)  # (B, P, C_T)

            # Path (b): add to conditioning as 4th component
            traj_pe_expanded = traj_pe_feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, C_T)
            cond_parts.append(self.hard_ln_ct(traj_pe_expanded))
        elif self.num_random_trace_id_pe > 0:
            # Legacy: random trace ID PE
            if self.training:
                _id = torch.argsort(
                    torch.rand(B, self.num_random_trace_id_pe, device=device), dim=-1
                )[:, :P]
            else:
                _id = torch.arange(P)[None].repeat(B, 1).to(device).long()
            _id_feat = self.random_trace_id_pe(_id)[:, None].repeat(1, S, 1, 1)  # (B, S, P, C_T)
            cond_parts.append(self.hard_ln_ct(_id_feat))

        cond_bsp = torch.cat(cond_parts, dim=-1)  # (B, S, P, cond_dim_t)

        # Embed input
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, P, C_T)

        # v6.9 path (a): prepare traj PE for t-layer concat
        # traj_pe_feat_bsp: (B, S, P, C_T) — broadcast over S dim
        traj_pe_feat_bsp = None
        if self.traj_pos_emb and traj_pe_feat is not None:
            traj_pe_feat_bsp = traj_pe_feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, C_T)

        # ========== Prepare Add Tokens for p-layers ==========
        # Combine language + image tokens (already at C_P dim from Agent)
        if add_tokens is not None and image_tokens is not None:
            all_add_tokens = torch.cat([add_tokens, image_tokens], dim=1)  # (B, A, C_P)
            all_add_mask = torch.cat([add_tokens_mask, image_tokens_mask], dim=1)  # (B, A)
        elif add_tokens is not None:
            all_add_tokens = add_tokens  # (B, A, C_P)
            all_add_mask = add_tokens_mask
        elif image_tokens is not None:
            all_add_tokens = image_tokens
            all_add_mask = image_tokens_mask
        else:
            all_add_tokens = torch.zeros(B, 0, self.C_P, device=device)
            all_add_mask = torch.zeros(B, 0, device=device).bool()
        A = all_add_tokens.shape[1]

        # ========== Build attention masks ==========
        if self.dual_t_layer:
            # v6.8: separate masks for ctx and X sub-layers
            traj_mask_perm = x_mask.permute(0, 2, 1)  # (B, P, S)
            ctx_valid_bp = traj_mask_perm[:, :, :N_CTX].reshape(B * P, N_CTX)  # (B*P, N_CTX)
            x_valid_bp = traj_mask_perm[:, :, N_CTX:].reshape(B * P, T)  # (B*P, T)
        else:
            # Standard: single mask for full [Q; S] sequence
            traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)
            q_valid_t = torch.ones(B * P, Q, dtype=torch.bool, device=device)
            t_seq_valid = torch.cat([q_valid_t, traj_valid], dim=1)  # (B*P, Q+S)

        # p-layer mask: full attention over A+P
        p_valid = x_mask.any(dim=1)  # (B, P) - part is valid if valid at any time
        p_seq_valid = torch.cat([all_add_mask, p_valid], dim=1)  # (B, A+P)

        # ========== Prepare RoPE time ==========
        if self.time_rope_flag:
            t_rope = (
                x_time[:, :, None].expand(-1, -1, P)
                .permute(0, 2, 1).reshape(B * P, T)
            )  # (B*P, T)
        else:
            t_rope = None

        # ========== Main Loop: t-p-t-p-...-t ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]  # initial state
        prev_p_out = None  # (B, P, C_P) from previous p-layer

        if self.dual_t_layer:
            q_state_ctx = None  # (B, P, Q, C_T)
            q_state_x = None   # (B, P, Q, C_T)
        else:
            q_state = None  # (B, P, Q, C_T)

        for layer_idx in range(self.n_t_layers):
            # --------- t-layer ---------
            if self.dual_t_layer:
                x_state, q_out_ctx, q_out_x = self._run_t_layer_dual(
                    layer_idx=layer_idx,
                    x_state=x_state,
                    q_state_ctx=q_state_ctx,
                    q_state_x=q_state_x,
                    cond_bsp=cond_bsp,
                    ctx_valid_bp=ctx_valid_bp,
                    x_valid_bp=x_valid_bp,
                    t_rope=t_rope,
                    B=B, S=S, P=P, Q=Q, N_CTX=N_CTX, T=T,
                    traj_pe_feat_bsp=traj_pe_feat_bsp,
                )
            else:
                x_state, q_out = self._run_t_layer_standard(
                    layer_idx=layer_idx,
                    x_state=x_state,
                    q_state=q_state,
                    cond_bsp=cond_bsp,
                    t_seq_valid=t_seq_valid,
                    t_rope=t_rope,
                    B=B, S=S, P=P, Q=Q, N_CTX=N_CTX,
                    traj_pe_feat_bsp=traj_pe_feat_bsp,
                )

            # Collect t-layer skip
            t_skip_list.append(x_state.reshape(B * S, P, self.C_T).clone())

            # --------- Pooling + p-layer + Unpooling (if not last t-layer) ---------
            if layer_idx < self.n_t_layers - 1:
                p_idx = layer_idx  # p-layer index

                if self._is_dense_p_layer(p_idx):
                    # ========== v6.8.5/6: Block-Dense P-Layer ==========
                    x_state, p_out = self._run_p_layer_block_dense(
                        p_idx=p_idx,
                        x_state=x_state,
                        cond_bsp=cond_bsp,
                        all_add_tokens=all_add_tokens,
                        all_add_mask=all_add_mask,
                        p_valid=p_valid,
                        B=B, S=S, P=P, A=A,
                    )
                    prev_p_out = p_out  # (B, P, C_P) mean of block outputs

                    # Unpool Q tokens using mean block p_out
                    if self.dual_t_layer:
                        q_out_ctx_r = q_out_ctx.reshape(B, P, Q, self.C_T)
                        q_out_x_r = q_out_x.reshape(B, P, Q, self.C_T)
                        p_out_exp = p_out[:, :, None, :].expand(-1, -1, Q, -1)

                        unpool_ctx_in = torch.cat([q_out_ctx_r, p_out_exp], dim=-1)
                        q_state_ctx = self.unpool_ffns_ctx[p_idx](unpool_ctx_in)

                        unpool_x_in = torch.cat([q_out_x_r, p_out_exp], dim=-1)
                        q_state_x = self.unpool_ffns_x[p_idx](unpool_x_in)
                    else:
                        q_out_reshaped = q_out.reshape(B, P, Q, self.C_T)
                        p_out_expanded = p_out[:, :, None, :].expand(-1, -1, Q, -1)
                        unpool_in = torch.cat([q_out_reshaped, p_out_expanded], dim=-1)
                        q_state = self.unpool_ffns[p_idx](unpool_in)

                else:
                    # ========== Standard Pooled P-Layer ==========

                    # ---- Pooling ----
                    if self.ctx_p_layer:
                        # v6.8.2: separate ctx/x pooling with ctx-P pre-layer
                        q_flat_ctx = q_out_ctx.reshape(B, P, Q * self.C_T)
                        q_flat_x = q_out_x.reshape(B, P, Q * self.C_T)

                        ctx_p_tokens = self.ctx_pool_projs[p_idx](q_flat_ctx)

                        if prev_p_out is not None:
                            x_pool_in = torch.cat([q_flat_x, prev_p_out], dim=-1)
                        else:
                            x_pool_in = q_flat_x
                        x_p_tokens = self.x_pool_projs[p_idx](x_pool_in)

                        # ctx-P layer (half-width)
                        if A > 0:
                            add_tokens_half = self.ctx_add_token_projs[p_idx](all_add_tokens)
                        else:
                            add_tokens_half = torch.zeros(
                                B, 0, self.ctx_p_width, device=device,
                                dtype=ctx_p_tokens.dtype,
                            )

                        ctx_p_seq = torch.cat([add_tokens_half, ctx_p_tokens], dim=1)
                        cond_p_per_point = cond_bsp.mean(dim=1)
                        cond_ctx_p = self.ctx_p_cond_binders[p_idx](cond_p_per_point)
                        cond_ctx_a = cond_ctx_p.mean(dim=1, keepdim=True).expand(-1, A, -1)
                        cond_ctx_p_seq = torch.cat([cond_ctx_a, cond_ctx_p], dim=1)

                        attn_mask_ctx_p = build_attention_mask(
                            p_seq_valid, mode="full",
                            left_pad=self.ctx_p_blocks[p_idx].R,
                        )[:, None].expand(-1, self.n_head_p // 2, -1, -1)

                        ctx_p_out_seq = self.ctx_p_blocks[p_idx](
                            x=ctx_p_seq, cond=cond_ctx_p_seq, t=None,
                            num_external_regs=A, attn_mask=attn_mask_ctx_p,
                        )
                        ctx_p_out = ctx_p_out_seq[:, A:]

                        p_tokens = self.ctx_x_cat_projs[p_idx](
                            torch.cat([ctx_p_out, x_p_tokens], dim=-1)
                        )
                    else:
                        # Standard pooling (v6.8 dual or baseline)
                        if self.dual_t_layer:
                            q_flat_ctx = q_out_ctx.reshape(B, P, Q * self.C_T)
                            q_flat_x = q_out_x.reshape(B, P, Q * self.C_T)
                            q_flat = torch.cat([q_flat_ctx, q_flat_x], dim=-1)
                        else:
                            q_flat = q_out.reshape(B, P, Q * self.C_T)

                        if prev_p_out is not None:
                            pool_in = torch.cat([q_flat, prev_p_out], dim=-1)
                        else:
                            pool_in = q_flat
                        p_tokens = self.pool_projs[p_idx](pool_in)

                    # v6.9 path (a) for p-layers
                    if self.traj_pos_emb and traj_pe_feat is not None:
                        p_tokens = self.traj_pe_x_proj_p(
                            torch.cat([p_tokens, traj_pe_feat], dim=-1)
                        )
                        if A > 0:
                            add_pe = self.traj_pe_add_token.expand(B, A, -1)
                            all_add_tokens_proj = self.traj_pe_x_proj_p(
                                torch.cat([all_add_tokens, add_pe], dim=-1)
                            )
                        else:
                            all_add_tokens_proj = all_add_tokens
                    else:
                        all_add_tokens_proj = all_add_tokens

                    # ---- Main p-layer ----
                    p_seq = torch.cat([all_add_tokens_proj, p_tokens], dim=1)

                    if not self.ctx_p_layer:
                        cond_p_per_point = cond_bsp.mean(dim=1)
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

                    # Unpooling: inject p-layer info back into Q tokens
                    if self.dual_t_layer:
                        q_out_ctx_r = q_out_ctx.reshape(B, P, Q, self.C_T)
                        q_out_x_r = q_out_x.reshape(B, P, Q, self.C_T)
                        p_out_exp = p_out[:, :, None, :].expand(-1, -1, Q, -1)

                        unpool_ctx_in = torch.cat([q_out_ctx_r, p_out_exp], dim=-1)
                        q_state_ctx = self.unpool_ffns_ctx[p_idx](unpool_ctx_in)

                        unpool_x_in = torch.cat([q_out_x_r, p_out_exp], dim=-1)
                        q_state_x = self.unpool_ffns_x[p_idx](unpool_x_in)
                    else:
                        q_out_reshaped = q_out.reshape(B, P, Q, self.C_T)
                        p_out_expanded = p_out[:, :, None, :].expand(-1, -1, Q, -1)
                        unpool_in = torch.cat([q_out_reshaped, p_out_expanded], dim=-1)
                        q_state = self.unpool_ffns[p_idx](unpool_in)

        # ========== Output: concat t-layer skip outputs ==========
        # t_skip_list: list of (B*S, P, C_T) tensors, len = n_t_layers + 1
        x_out = torch.cat(t_skip_list, dim=-1)  # (B*S, P, (n_t_layers+1)*C_T)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)


# Keep backward-compatible name
MotionMachineV2 = MotionMachineV6


def test_main():
    """Smoke test for MotionMachineV6 — all 4 flag combinations."""
    import sys
    from umm.models.umm_v6_all.world_models.moma_v6 import MotionMachineV6

    B, T, P, C = 2, 8, 64, 9
    CTX = 3
    A = 16
    C_T, C_P = 128, 512

    combos = [
        {"dual_t_layer": False, "traj_pos_emb": False, "ctx_p_layer": False},
        {"dual_t_layer": True,  "traj_pos_emb": False, "ctx_p_layer": False},
        {"dual_t_layer": False, "traj_pos_emb": True,  "ctx_p_layer": False},
        {"dual_t_layer": True,  "traj_pos_emb": False, "ctx_p_layer": True},
        {"dual_t_layer": True,  "traj_pos_emb": True,  "ctx_p_layer": True},
        # v6.8.5: block-dense ALL p-layers
        {"dual_t_layer": True,  "traj_pos_emb": False, "ctx_p_layer": False,
         "dense_p_block_size": 5, "n_dense_p_layers": -1},
        # v6.8.6: hybrid — first p-layer block-dense, rest pooled
        {"dual_t_layer": True,  "traj_pos_emb": False, "ctx_p_layer": False,
         "dense_p_block_size": 5, "n_dense_p_layers": 1},
    ]

    for combo in combos:
        label = ", ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        net = MotionMachineV6(
            C_T=C_T,
            C_P=C_P,
            feat_dim=512,
            n_head_t=8,
            n_head_p=16,
            input_dim=C,
            n_t_layers=4,
            n_Q=4,
            **combo,
        ).to("cuda")

        n_params = sum(p.numel() for p in net.parameters())
        print(f"  Parameters: {n_params:,}")

        # === Training mode ===
        net.train()
        x = torch.randn(B, T, P, C, device="cuda")
        x_mask = torch.ones(B, T, P, device="cuda").bool()
        t = torch.randint(0, 256, (B, T), device="cuda") / 256.0
        x_level = torch.zeros(B, T, P, device="cuda")
        feat = torch.randn(B, P, 512, device="cuda")

        ctx = torch.randn(B, CTX, P, C, device="cuda")
        ctx_mask = torch.ones(B, CTX, P, device="cuda").bool()
        ctx_level = torch.randint(0, 7, (B, CTX, P), device="cuda")

        add_tk = torch.randn(B, A, C_P, device="cuda")
        add_mk = torch.ones(B, A, device="cuda").bool()

        ret = net(
            x=x, x_mask=x_mask, x_time=t, x_level=x_level, feat=feat,
            ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
            add_tokens=add_tk, add_tokens_mask=add_mk,
        )
        expected = (B, T + CTX, P, C)
        assert ret.shape == expected, f"Expected {expected}, got {ret.shape}"
        print(f"  Train output shape: {ret.shape} [OK]")

        loss = F.mse_loss(ret, ret.detach() - 1)
        loss.backward()
        print(f"  Train loss: {loss.item():.4f}, backward: [OK]")

        # === Eval mode (inference permutation check) ===
        net.eval()
        with torch.no_grad():
            ret_eval = net(
                x=x, x_mask=x_mask, x_time=t, x_level=x_level, feat=feat,
                ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
                add_tokens=add_tk, add_tokens_mask=add_mk,
            )
        assert ret_eval.shape == expected, f"Eval: expected {expected}, got {ret_eval.shape}"
        print(f"  Eval output shape: {ret_eval.shape} [OK]")

        # === No-context test ===
        net.train()
        net.zero_grad()
        ret_no_ctx = net(
            x=x, x_mask=x_mask, x_time=t, x_level=x_level, feat=feat,
            add_tokens=add_tk, add_tokens_mask=add_mk,
        )
        expected_no_ctx = (B, T, P, C)
        assert ret_no_ctx.shape == expected_no_ctx, \
            f"No-ctx: expected {expected_no_ctx}, got {ret_no_ctx.shape}"
        loss_nc = F.mse_loss(ret_no_ctx, ret_no_ctx.detach() - 1)
        loss_nc.backward()
        print(f"  No-ctx output shape: {ret_no_ctx.shape} [OK]")

        print(f"  PASSED: {label}")

        del net
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All {len(combos)} combinations passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_main()
