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

from umm.models.umm_v6_mixed.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
    NerfStylePE,
)
from umm.models.umm_v6_mixed.world_models.dit_atten import DiTBlockTimeOnly


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

        assert n_t_layers >= 2, "Need at least 2 t-layers (first and last)"

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

        # Random trace ID PE
        self.num_random_trace_id_pe = num_random_trace_id_pe
        if self.num_random_trace_id_pe > 0:
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

        # =============== Conditioning Binders ===============
        # t-layer conditioning: feat_dim + 2*C_T [+ optional extras] -> cond_dim_t
        cond_dim_t_base = self.feat_dim + self.C_T * 2
        if self.num_random_trace_id_pe > 0:
            cond_dim_t_base += self.C_T
        self.cond_dim_t = cond_dim_t_base

        # t-layer cond binders: one per t-layer, maps cond_dim_t -> C_T (DiTBlock C_feat)
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
        self.q_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_Q, self.C_T))
            for _ in range(self.n_t_layers)
        ])
        for q_emb in self.q_embeds:
            nn.init.normal_(q_emb, std=0.02)

        # =============== t-layer Blocks ===============
        self.t_blocks = nn.ModuleList([
            DiTBlockTimeOnly(
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
            for _ in range(self.n_t_layers)
        ])

        # =============== Pooling Projections (t->p) ===============
        # First pooling: Q*C_T -> C_P
        # Subsequent poolings: Q*C_T + C_P (skip from prev p-layer) -> C_P
        self.pool_projs = nn.ModuleList()
        for i in range(self.n_p_layers):
            if i == 0:
                proj_in = n_Q * self.C_T
            else:
                proj_in = n_Q * self.C_T + self.C_P
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

        # =============== Unpooling FFNs (p->t) ===============
        # Maps (C_T + C_P) -> C_T for each Q token
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

        if self.num_random_trace_id_pe > 0:
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
        # t-layer mask: per-trajectory temporal attention
        # Shape for t-layer: (B*P, Q+S) -> full attention
        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)
        q_valid_t = torch.ones(B * P, Q, dtype=torch.bool, device=device)
        t_seq_valid = torch.cat([q_valid_t, traj_valid], dim=1)  # (B*P, Q+S)
        # Build mask with internal registers
        # (handled inside DiTBlockTimeOnly via its own register mechanism)

        # p-layer mask: full attention over A+P
        p_valid = x_mask.any(dim=1)  # (B, P) - part is valid if valid at any time
        p_seq_valid = torch.cat([all_add_mask, p_valid], dim=1)  # (B, A+P)

        # ========== Main Loop: t-p-t-p-...-t ==========
        x_state = x_emb.clone()  # (B, S, P, C_T)
        t_skip_list = [x_emb.reshape(B * S, P, self.C_T).clone()]  # initial state
        prev_p_out = None  # (B, P, C_P) from previous p-layer
        q_state = None  # will hold Q tokens between layers: (B, P, Q, C_T)

        for layer_idx in range(self.n_t_layers):
            # --------- t-layer ---------
            # Get this layer's Q embeddings
            q_emb = self.q_embeds[layer_idx]  # (1, Q, C_T)

            if q_state is not None:
                # Use Q state from unpooling (already (B, P, Q, C_T))
                q_tokens = q_state
            else:
                # First t-layer: broadcast learnable Q embeddings
                q_tokens = q_emb.expand(B * P, -1, -1)  # (B*P, Q, C_T)
                q_tokens = q_tokens.reshape(B, P, Q, self.C_T)

            # Reshape for per-trajectory self-attention
            # x_state: (B, S, P, C_T) -> per trajectory: (B, P, S, C_T) -> (B*P, S, C_T)
            x_traj = x_state.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)  # (B*P, S, C_T)
            q_traj = q_tokens.reshape(B * P, Q, self.C_T)  # (B*P, Q, C_T)

            # Concat [Q; S] per trajectory
            traj_seq = torch.cat([q_traj, x_traj], dim=1)  # (B*P, Q+S, C_T)

            # Build conditioning for this t-layer: (B, S, P, cond_dim_t) -> (B*P, S, C_T)
            # Mean-pool cond over S for Q positions, keep per-S for time positions
            cond_t_layer = self.t_cond_binders[layer_idx](cond_bsp)  # (B, S, P, C_T)
            cond_t_per_traj = cond_t_layer.permute(0, 2, 1, 3).reshape(B * P, S, self.C_T)
            # Q tokens get mean-pooled conditioning
            cond_q_mean = cond_t_per_traj.mean(dim=1, keepdim=True).expand(-1, Q, -1)  # (B*P, Q, C_T)
            cond_traj = torch.cat([cond_q_mean, cond_t_per_traj], dim=1)  # (B*P, Q+S, C_T)

            # Prepare time for RoPE (only applies to S positions, not Q)
            if self.time_rope_flag:
                t_rope = (
                    x_time[:, :, None].expand(-1, -1, P)
                    .permute(0, 2, 1).reshape(B * P, T)
                )  # (B*P, T)
            else:
                t_rope = None

            # Build attention mask for t-layer
            attn_mask_t = build_attention_mask(
                t_seq_valid,
                mode="full",
                left_pad=self.t_blocks[layer_idx].R,
            )[:, None].expand(-1, self.n_head_t, -1, -1)  # (B*P, H, Q+S+R, Q+S+R)

            # Run t-block
            traj_out = self.t_blocks[layer_idx](
                x=traj_seq,
                cond=cond_traj,
                t=t_rope,
                num_external_regs=Q + N_CTX,  # Q tokens + ctx treated as non-time
                attn_mask=attn_mask_t,
            )  # (B*P, Q+S, C_T)

            # Split Q and S
            q_out = traj_out[:, :Q]  # (B*P, Q, C_T)
            x_out = traj_out[:, Q:]  # (B*P, S, C_T)

            # Update x_state
            x_state = x_out.reshape(B, P, S, self.C_T).permute(0, 2, 1, 3)  # (B, S, P, C_T)

            # Collect t-layer skip
            t_skip_list.append(x_state.reshape(B * S, P, self.C_T).clone())

            # --------- Pooling + p-layer + Unpooling (if not last t-layer) ---------
            if layer_idx < self.n_t_layers - 1:
                p_idx = layer_idx  # p-layer index

                # Pooling: flatten Q tokens per trajectory -> project to C_P
                q_flat = q_out.reshape(B, P, Q * self.C_T)  # (B, P, Q*C_T)

                if prev_p_out is not None:
                    pool_in = torch.cat([q_flat, prev_p_out], dim=-1)  # (B, P, Q*C_T + C_P)
                else:
                    pool_in = q_flat  # (B, P, Q*C_T)

                p_tokens = self.pool_projs[p_idx](pool_in)  # (B, P, C_P)

                # p-layer: self-attention on [A add-tokens ; P pooled-tokens]
                p_seq = torch.cat([all_add_tokens, p_tokens], dim=1)  # (B, A+P, C_P)

                # p-layer conditioning: keep feat per-P, average time/noise over S only
                cond_p_per_point = cond_bsp.mean(dim=1)  # (B, P, cond_dim_t) — avg over S, keep per-P
                cond_p = self.p_cond_binders[p_idx](cond_p_per_point)  # (B, P, C_P)
                # A (add token) positions get mean-pooled conditioning
                cond_a = cond_p.mean(dim=1, keepdim=True).expand(-1, A, -1)  # (B, A, C_P)
                cond_p_seq = torch.cat([cond_a, cond_p], dim=1)  # (B, A+P, C_P)

                # Build attention mask for p-layer
                attn_mask_p = build_attention_mask(
                    p_seq_valid,
                    mode="full",
                    left_pad=self.p_blocks[p_idx].R,
                )[:, None].expand(-1, self.n_head_p, -1, -1)

                # Run p-block
                p_out_seq = self.p_blocks[p_idx](
                    x=p_seq,
                    cond=cond_p_seq,
                    t=None,
                    num_external_regs=A,
                    attn_mask=attn_mask_p,
                )  # (B, A+P, C_P)

                p_out = p_out_seq[:, A:]  # (B, P, C_P)
                prev_p_out = p_out  # save for next pooling skip

                # Unpooling: inject p-layer info back into Q tokens
                # q_out: (B*P, Q, C_T), p_out: (B, P, C_P)
                q_out_reshaped = q_out.reshape(B, P, Q, self.C_T)  # (B, P, Q, C_T)
                p_out_expanded = p_out[:, :, None, :].expand(-1, -1, Q, -1)  # (B, P, Q, C_P)

                unpool_in = torch.cat([q_out_reshaped, p_out_expanded], dim=-1)  # (B, P, Q, C_T+C_P)
                q_state = self.unpool_ffns[p_idx](unpool_in)  # (B, P, Q, C_T)

        # ========== Output: concat t-layer skip outputs ==========
        # t_skip_list: list of (B*S, P, C_T) tensors, len = n_t_layers + 1
        x_out = torch.cat(t_skip_list, dim=-1)  # (B*S, P, (n_t_layers+1)*C_T)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)


# Keep backward-compatible name
MotionMachineV2 = MotionMachineV6


def test_main():
    """Smoke test for MotionMachineV6."""
    from umm.models.umm_v6_mixed.world_models.moma_v6 import MotionMachineV6

    B, T, P, C = 2, 8, 64, 9
    CTX = 3
    A = 16
    C_T, C_P = 128, 512

    net = MotionMachineV6(
        C_T=C_T,
        C_P=C_P,
        feat_dim=512,
        n_head_t=8,
        n_head_p=16,
        input_dim=C,
        n_t_layers=4,
        n_Q=4,
    ).to("cuda")
    net.train()

    x = torch.randn(B, T, P, C).to("cuda")
    x_mask = torch.ones(B, T, P).to("cuda").bool()
    t = torch.randint(0, 256, (B, T)).to("cuda") / 256.0
    x_level = torch.zeros(B, T, P).to("cuda")
    feat = torch.randn(B, P, 512).to("cuda")

    ctx = torch.randn(B, CTX, P, C).to("cuda")
    ctx_mask = torch.ones(B, CTX, P).to("cuda").bool()
    ctx_level = torch.randint(0, 7, (B, CTX, P)).to("cuda")

    add_tk = torch.randn(B, A, C_P).to("cuda")
    add_mk = torch.ones(B, A).to("cuda").bool()

    ret = net(
        x=x, x_mask=x_mask, x_time=t, x_level=x_level, feat=feat,
        ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
        add_tokens=add_tk, add_tokens_mask=add_mk,
    )
    print(f"Output shape: {ret.shape}")
    assert ret.shape == (B, T + CTX, P, C), f"Expected {(B, T + CTX, P, C)}, got {ret.shape}"

    loss = F.mse_loss(ret, ret.detach() - 1)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print("V6 test passed!")


if __name__ == "__main__":
    test_main()
