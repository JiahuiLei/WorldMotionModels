"""MotionMachine TPA -- Dense TPA attention for Non-DF variant.

Adapted from MotionMachineV2 (HH2.0 architecture) for the non-DF masked
conditioned DDPM framework. Key differences from original V2:

- Input is ``[X; cond_X; mask]`` concatenated (19 channels, not 9).
- Forward signature matches MotionMachineV6 non-DF interface:
  accepts ``cond_x``, ``cond_mask``, ``image_tokens``, ``image_tokens_mask``.
- Per-token AdaLN conditioning via ``cond_token_proj`` that encodes
  ``[cond_X(9); mask(1)]`` and adds it to the level embedding.
- Output is 9 channels (original SE3 param dim).

Architecture: factored temporal ("t"), part ("p"), and global ("a") attention
layers specified by ``layer_type_string`` (e.g., "tpatpapt"), with register
tokens per layer type and skip connections.

Example::

    net = MotionMachineTPA(latent_dim=512, feat_dim=512)
    out = net(x, x_mask, x_time, x_level, feat,
              cond_x=cond_x, cond_mask=cond_mask)
"""

import logging
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v6_nondf.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
    NerfStylePE,
)
from umm.models.umm_v6_nondf.world_models.dit_atten import DiTBlockTimeOnly


class MotionMachineTPA(nn.Module):
    """Dense TPA attention MotionMachine for Non-DF variant.

    Factored temporal/part/all attention layers with register tokens,
    skip connections, and AdaLN-Zero conditioning. Adapted for 19-channel
    non-DF input ``[X; cond_X; mask]``.

    Args:
        latent_dim: Latent dimension for all attention layers.
        feat_dim: Dimension of per-part identity features.
        n_head: Number of attention heads.
        input_dim: Per-element dimension (9 for SE3 param).
        output_dim: Output dimension (defaults to input_dim).
        layer_type_string: Sequence of layer types, e.g. "tpatpapt".
        skip_layer_idx: Layer indices for skip connections.
        cond_binder_dim: Dimension of conditioning binder output.
        fc_out_n_layers: Number of layers in output MLP.
        fc_factor: MLP width factor.
        fc_n_layers: Number of hidden layers per DiT block MLP.
        num_register: Number of register tokens per layer type.
        num_blk_internal_registers: Internal register count for DiT blocks.
        n_ctx: Max context length for position embeddings.
        x_pe_dim: NeRF-style PE frequency count for input (0 = disabled).
        time_multiplier: Scale factor for timestep values.
        use_reg_binder: Whether to use register-to-sequence injection.
        ada_scale_bias_bound: Bound for AdaLN scale/bias.
        gate_bound: Bound for gate values.
        ada_n_layers: Number of AdaLN MLP layers.
        add_first_n_add_token_to_feat: Tokens from add_tokens to fold into feat.
        time_rope_flag: Whether to use RoPE for temporal attention.
        add_time_to_feat: Whether to add continuous time PE to conditioning.
        num_random_trace_id_pe: Random trace ID PE count (<= 0 disables).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 2048,
        n_head: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        layer_type_string: str = "tpatpapt",
        skip_layer_idx: Optional[List[int]] = None,
        cond_binder_dim: int = 512,
        #
        fc_out_n_layers: int = 12,
        fc_factor: int = 2,
        fc_n_layers: int = 2,
        num_register: int = 4,
        num_blk_internal_registers: int = 4,
        n_ctx: int = 128,
        #
        x_pe_dim: int = 0,
        time_multiplier: float = 10.0,
        #
        use_reg_binder: bool = True,
        #
        ada_scale_bias_bound: float = 2.0,
        gate_bound: float = 2.0,
        ada_n_layers: int = 2,
        #
        add_first_n_add_token_to_feat: int = -1,
        time_rope_flag: bool = False,
        add_time_to_feat: bool = True,
        num_random_trace_id_pe: int = -1,
        # Unused (compat with agent config parsing — V6-specific params)
        C_T: int = -1,
        C_P: int = -1,
        n_t_layers: int = -1,
        n_Q: int = -1,
        n_head_t: int = -1,
        n_head_p: int = -1,
        fc_factor_t: int = 2,
        fc_factor_p: int = 2,
        dual_t_layer: bool = False,
        traj_pos_emb: bool = False,
        ctx_p_layer: bool = False,
        dense_p_block_size: int = 0,
        n_dense_p_layers: int = -1,
        qk_norm: bool = False,
        image_cross_attn_layers: str = "",
        n_latent_queries: int = -1,
        q_pool_mode: str = "",
    ):
        super().__init__()

        if skip_layer_idx is None:
            skip_layer_idx = [2, 3, 5]

        self.layer_type_string = layer_type_string
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.cond_binder_dim = cond_binder_dim

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.n_blocks = len(self.layer_type_string)
        # Ensure last layer is always a skip
        skip_layer_idx = list(skip_layer_idx)
        if self.n_blocks - 1 not in skip_layer_idx:
            skip_layer_idx.append(self.n_blocks - 1)
        self.skip_layer_idx = skip_layer_idx
        self.n_head = n_head
        self.num_register = num_register
        self.hard_ln_lat = HardLayerNorm(self.latent_dim)
        self.hard_ln_feat = HardLayerNorm(self.feat_dim)

        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat
        if self.add_time_to_feat:
            self.t_pe = Timesteps(
                self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            self.t_proj = nn.Linear(self.latent_dim + 1, self.latent_dim)

        # =============== PE and EMBs ===============
        self.int_pe = Timesteps(
            self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.time_multiplier = time_multiplier

        # Non-DF: input is [X(9); cond_X(9); mask(1)] = 19 channels
        self.nondf_input_dim = self.input_dim * 2 + 1  # 19

        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )
            x_proj_in = self.x_pe.out_channels * self.nondf_input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.nondf_input_dim
        self.x_proj = nn.Linear(x_proj_in, self.latent_dim)

        # Non-DF: per-token conditioning projection [cond_X(9) + mask(1)] -> latent_dim
        # Added to level_emb for AdaLN, encoding what each position knows/doesn't know.
        self.cond_token_proj = nn.Linear(self.input_dim + 1, self.latent_dim)

        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.latent_dim)

        # =============== Cond Binder ===============
        cond_binder_in = self.feat_dim + self.latent_dim * 2

        self.add_first_n_add_token_to_feat = add_first_n_add_token_to_feat
        if self.add_first_n_add_token_to_feat > 0:
            self.add_to_feat_fc1 = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            self.add_to_feat_fc2 = nn.Linear(
                64 * self.add_first_n_add_token_to_feat, self.latent_dim
            )
            cond_binder_in += self.latent_dim

        self.num_random_trace_id_pe = num_random_trace_id_pe
        if self.num_random_trace_id_pe > 0:
            self.random_trace_id_pe = nn.Embedding(
                self.num_random_trace_id_pe, self.latent_dim
            )
            with torch.no_grad():
                self.random_trace_id_pe.weight.copy_(
                    self.random_trace_id_pe.weight[0].expand_as(
                        self.random_trace_id_pe.weight
                    )
                )
            cond_binder_in += self.latent_dim

        self.cond_binder = nn.ModuleList(
            [
                nn.Linear(cond_binder_in, self.cond_binder_dim)
                for _ in range(self.n_blocks)
            ]
        )

        # =============== Output Projection ===============
        out_proj_c_in = self.latent_dim * (1 + len(self.skip_layer_idx))
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

        # =============== DiT Blocks ===============
        self.num_blk_internal_registers = num_blk_internal_registers
        self.blks = nn.ModuleList(
            [
                DiTBlockTimeOnly(
                    C=self.latent_dim,
                    num_heads=n_head,
                    num_registers=self.num_blk_internal_registers,
                    mlp_hidden_dims=[self.latent_dim * fc_factor] * fc_n_layers,
                    ada_scale_bias_bound=ada_scale_bias_bound,
                    gate_bound=gate_bound,
                    ada_n_layers=ada_n_layers,
                    C_feat=self.cond_binder_dim,
                )
                for _ in range(self.n_blocks)
            ]
        )

        # =============== Register Tokens ===============
        self.a_reg = nn.Embedding(num_register, self.latent_dim)
        self.t_reg = nn.Embedding(num_register, self.latent_dim)
        self.p_reg = nn.Embedding(num_register, self.latent_dim)

        self.use_reg_binder = use_reg_binder
        if self.use_reg_binder:
            self.reg_binder = nn.ModuleList(
                [
                    nn.Linear(
                        self.latent_dim * num_register,
                        self.latent_dim,
                    )
                    for _ in range(self.n_blocks)
                ]
            )

        self.__summary__()

    def __summary__(self):
        """Log model architecture summary."""
        logging.info("LMM TPA Summary:")
        logging.info(f"  latent_dim:         {self.latent_dim}")
        logging.info(f"  input_dim (SE3):    {self.input_dim}")
        logging.info(f"  nondf_input_dim:    {self.nondf_input_dim}")
        logging.info(f"  n_blocks:           {self.n_blocks}")
        logging.info(f"  layer_type_string:  {self.layer_type_string}")
        logging.info(f"  skip_layer_idx:     {self.skip_layer_idx}")
        logging.info(f"  cond_binder_dim:    {self.cond_binder_dim}")
        logging.info(f"  num_register:       {self.num_register}")

    def get_optim_groups(self, weight_decay: float = 0.0, base_lr: Optional[float] = None):
        """Optimizer groups with weight-decay separation.

        2D+ tensors get weight_decay, <2D tensors get 0.

        Args:
            weight_decay: Weight decay for decayable parameters.
            base_lr: If provided, sets ``lr`` on each group.

        Returns:
            list[dict]: Parameter groups for ``torch.optim``.
        """
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
        # Non-DF conditioning (per-token known/unknown info for AdaLN)
        cond_x: Optional[torch.Tensor] = None,  # [B, S, P, 9]
        cond_mask: Optional[torch.Tensor] = None,  # [B, S, P, 1]
    ) -> torch.Tensor:
        """Forward pass with dense TPA attention (Non-DF variant).

        Args:
            x: [B, T, P, 19] noisy input = [X; cond_X; mask].
            x_mask: [B, T, P] bool mask.
            x_time: [B, T] timestamps.
            x_level: [B, T, P] global diffusion levels.
            feat: [B, P, feat_dim] per-part identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: [B, A_lang, latent_dim] language tokens.
            add_tokens_mask: [B, A_lang] mask.
            image_tokens: [B, A_img, latent_dim] image tokens (ignored for TPA).
            image_tokens_mask: [B, A_img] mask.
            cond_x: [B, S, P, 9] condition values for AdaLN.
            cond_mask: [B, S, P, 1] known/unknown mask for AdaLN.

        Returns:
            [B, S, P, input_dim] predicted clean output (9ch). S = T + N_CTX.
        """
        B, T, N, _ = x.shape
        device = x.device
        x_time = x_time.float().clone() * self.time_multiplier

        # ========== Time / Position Embeddings ==========
        if self.add_time_to_feat:
            ctx_pos_emb = torch.cat(
                [
                    x_time[..., None],
                    self.t_pe(x_time.reshape(-1)).reshape(*x_time.shape[:2], -1),
                ],
                dim=-1,
            )
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # (B, T, latent_dim)
        else:
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )

        # ========== Context Concatenation ==========
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

        # ========== Build Conditioning ==========
        # Level embedding (non-DF: same global t for all positions)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * N))
            .reshape(B, S, N, self.latent_dim)
            .to(x)
        )  # (B, S, N, latent_dim)

        # Non-DF: add per-token conditioning from [cond_X; mask]
        if cond_x is not None and cond_mask is not None:
            # cond_x: (B, S, N, 9), cond_mask: (B, S, N, 1)
            cond_token_input = torch.cat([cond_x, cond_mask], dim=-1)  # (B, S, N, 10)
            cond_token_emb = self.cond_token_proj(cond_token_input)  # (B, S, N, latent_dim)
            level_emb = level_emb + cond_token_emb

        feat_expanded = feat[:, None].expand(-1, S, -1, -1)  # (B, S, N, feat_dim)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(-1, -1, N, -1)  # (B, S, N, latent_dim)

        cond = torch.cat(
            [
                self.hard_ln_feat(feat_expanded),
                self.hard_ln_lat(level_emb),
                self.hard_ln_lat(ctx_pos_emb_expanded),
            ],
            dim=-1,
        )  # (B, S, N, feat_dim + 2*latent_dim)

        if self.add_first_n_add_token_to_feat > 0 and add_tokens is not None:
            assert add_tokens.shape[1] >= self.add_first_n_add_token_to_feat
            _add_feat = (
                add_tokens[:, : self.add_first_n_add_token_to_feat].clone().contiguous()
            )
            _add_feat = self.add_to_feat_fc1(_add_feat)
            _add_feat = self.add_to_feat_fc2(_add_feat.reshape(B, -1))
            _add_feat = _add_feat[:, None, None].expand(-1, S, N, -1)
            cond = torch.cat([cond, self.hard_ln_lat(_add_feat)], dim=-1)

        if self.num_random_trace_id_pe > 0:
            assert N <= self.num_random_trace_id_pe
            if self.training:
                _id = torch.argsort(
                    torch.rand(B, self.num_random_trace_id_pe, device=device), dim=-1
                )[:, :N]
            else:
                _id = torch.arange(N)[None].repeat(B, 1).to(device).long()
            _id_feat = self.random_trace_id_pe(_id)[:, None].repeat(1, S, 1, 1)
            cond = torch.cat([cond, self.hard_ln_lat(_id_feat)], dim=-1)

        # ========== Embed Input (19ch -> latent_dim) ==========
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, N, latent_dim)

        # ========== Flatten for TPA attention ==========
        x = x_emb.reshape(B, S * N, self.latent_dim)
        cond = cond.reshape(B, S * N, cond.shape[-1])
        x_time_flat = x_time[:, :, None].expand(-1, -1, N).reshape(B, T * N)

        # ========== Add tokens (combine lang + image) ==========
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
            all_add_tokens = torch.zeros(B, 0, self.latent_dim, device=device)
            all_add_mask = torch.zeros(B, 0, device=device).bool()
        A = all_add_tokens.shape[1]

        # ========== Build Attention Masks and Register Tokens ==========

        # * all-attention layers
        a_reg = self.a_reg(torch.arange(self.num_register, device=device))
        a_reg = a_reg[None].repeat(B, 1, 1)  # (B, R, latent_dim)
        attn_mask_a = build_attention_mask(
            torch.cat([all_add_mask, x_mask.reshape(B, S * N)], dim=1),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_a = all_add_tokens.clone()  # (B, A, latent_dim)

        # * t layers (temporal, per-trajectory)
        t_reg = self.t_reg(torch.arange(self.num_register, device=device))
        t_reg = t_reg[None].repeat(B * N, 1, 1)  # (B*N, R, latent_dim)
        attn_mask_t = build_attention_mask(
            torch.cat(
                [
                    all_add_mask[:, None].expand(-1, N, -1).reshape(B * N, A),
                    x_mask.permute(0, 2, 1).reshape(B * N, S),
                ],
                dim=1,
            ),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_t = (
            all_add_tokens[:, None].expand(-1, N, -1, -1).reshape(B * N, A, self.latent_dim)
        )

        # * p layers (spatial/part, per-timestep)
        p_reg = self.p_reg(torch.arange(self.num_register, device=device))
        p_reg = p_reg[None].repeat(B * S, 1, 1)  # (B*S, R, latent_dim)
        attn_mask_p = build_attention_mask(
            torch.cat(
                [
                    all_add_mask[:, None].expand(-1, S, -1).reshape(B * S, A),
                    x_mask.reshape(B * S, N),
                ],
                dim=1,
            ),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_p = (
            all_add_tokens[:, None].expand(-1, S, -1, -1).reshape(B * S, A, self.latent_dim)
        )

        # ========== Main TPA Loop ==========
        skip_list = [x.clone()]
        for i, blk in enumerate(self.blks):
            _cond = self.cond_binder[i](cond)

            if self.layer_type_string[i] == "a":
                _x = torch.cat([a_reg, add_a, x], dim=1)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B, a_reg.shape[1] + add_a.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                x = blk(x=_x, cond=_cond, t=None, attn_mask=attn_mask_a)
                a_reg = x[:, : self.num_register]
                x = x[:, self.num_register + A :]
                if self.use_reg_binder:
                    x = (
                        x
                        + self.reg_binder[i](
                            a_reg.reshape(B, self.num_register * self.latent_dim)
                        )[:, None]
                    )

            elif self.layer_type_string[i] == "t":
                _track = x.reshape(B, S, N, self.latent_dim).permute(
                    0, 2, 1, 3
                )  # (B, N, S, latent_dim)
                _track = _track.reshape(B * N, S, self.latent_dim)
                _track = torch.cat([t_reg, add_t, _track], dim=1)
                _cond = _cond.reshape(B, S, N, self.cond_binder_dim).permute(0, 2, 1, 3)
                _cond = _cond.reshape(B * N, S, self.cond_binder_dim)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B * N, t_reg.shape[1] + add_t.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                _track = blk(
                    _track,
                    cond=_cond,
                    t=(
                        x_time_flat.reshape(B, T, N).permute(0, 2, 1).reshape(B * N, T)
                        if self.time_rope_flag
                        else None
                    ),
                    num_external_regs=self.num_register + A + N_CTX,
                    attn_mask=attn_mask_t,
                )
                t_reg = _track[:, : self.num_register]
                _track = _track[:, self.num_register + A :]
                if self.use_reg_binder:
                    _track = (
                        _track
                        + self.reg_binder[i](
                            t_reg.reshape(B * N, self.num_register * self.latent_dim)
                        )[:, None]
                    )
                _track = _track.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
                x = _track.reshape(B, S * N, self.latent_dim)

            elif self.layer_type_string[i] == "p":
                _track = x.reshape(B, S, N, self.latent_dim)
                _track = _track.reshape(B * S, N, self.latent_dim)
                _track = torch.cat([p_reg, add_p, _track], dim=1)
                _cond = _cond.reshape(B * S, N, self.cond_binder_dim)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B * S, p_reg.shape[1] + add_p.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                _track = blk(
                    _track,
                    cond=_cond,
                    t=None,
                    num_external_regs=None,
                    attn_mask=attn_mask_p,
                )
                p_reg = _track[:, : self.num_register]
                _track = _track[:, self.num_register + A :]
                if self.use_reg_binder:
                    _track = (
                        _track
                        + self.reg_binder[i](
                            p_reg.reshape(B * S, self.num_register * self.latent_dim)
                        )[:, None]
                    )
                _track = _track.reshape(B, S, N, self.latent_dim)
                x = _track.reshape(B, S * N, self.latent_dim)

            else:
                raise ValueError(f"Invalid layer type: {self.layer_type_string[i]}")

            if i in self.skip_layer_idx:
                skip_list.append(x.clone())

        # ========== Output Projection ==========
        x = torch.cat(skip_list, dim=-1)
        ret = self.output_proj(x.reshape(-1, x.shape[-1]))
        ret = ret.reshape(B, S, N, self.input_dim)
        return ret  # (B, S, N, 9)
