"""MotionMachine V11 Dense — String-based t/p/a layers at unified width.

Architecture: All layers at width C_D=512. String-based layer config.
No pool/unpool needed (same width throughout).

t-layer: Self-attention across time (S) per entity, with structured registers
         (reg-c attends ctx only, reg-x attends x only, reg-a attends all).
p-layer: Cross-attention across entities (P) per timestep, with own registers.
         Operates on BOTH pooled t-features AND per-timestep features:
         pooled entities are included as additional tokens in each timestep's
         attention, giving each timestep access to global + local entity info.
a-layer: Full attention over all S*P tokens.

Example layer strings: "tpatpapt", "tptptptpt"

Example::

    net = MotionMachineV11Dense(
        C_D=512, layer_type_string="tpatpapt",
        n_reg_c=2, n_reg_x=2, n_reg_a=2, n_reg_p=4,
    )
    out = net(x, x_mask, x_time, x_level, feat,
              ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
              add_tokens=lang, add_tokens_mask=lang_mask)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
)
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly

logger = logging.getLogger(__name__)


class MotionMachineV11Dense(nn.Module):
    """Dense LMM with string-based t/p/a layers at unified width C_D.

    Args:
        C_D: Unified channel width for all layers (default 512).
        feat_dim: Input feature dimension from identity layer.
        n_head: Number of attention heads.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        layer_type_string: String of layer types (e.g., "tpatpapt").
            Must contain only 't', 'p', 'a'. Must start with 't'.
        n_reg_c: Context register count for t-layers (0 to skip).
        n_reg_x: X register count for t-layers (0 to skip).
        n_reg_a: All register count for t-layers (0 to skip).
        n_reg_p: Register count for p-layers.
        n_ctx: Maximum context length for position embeddings.
        fc_factor: MLP width factor for all layers.
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
    """

    def __init__(
        self,
        C_D: int = 512,
        feat_dim: int = 512,
        n_head: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        layer_type_string: str = "tpatpapt",
        n_reg_c: int = 2,
        n_reg_x: int = 2,
        n_reg_a: int = 2,
        n_reg_p: int = 4,
        n_ctx: int = 128,
        fc_factor: int = 4,
        fc_n_layers: int = 2,
        fc_out_n_layers: int = 12,
        num_blk_internal_registers: int = 4,
        time_multiplier: float = 1.0,
        ada_scale_bias_bound: float = 1.0,
        gate_bound: float = 2.0,
        ada_n_layers: int = 1,
        add_time_to_feat: bool = True,
        time_rope_flag: bool = False,
        qk_norm: bool = True,
        x_pe_dim: int = 0,
    ):
        super().__init__()

        self.C_D = C_D
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.layer_type_string = layer_type_string
        self.n_layers = len(layer_type_string)
        self.n_head = n_head
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.add_time_to_feat = add_time_to_feat

        # Validate layer string
        assert len(layer_type_string) > 0, "layer_type_string must not be empty"
        assert layer_type_string[0] == "t", (
            f"layer_type_string must start with 't', got '{layer_type_string[0]}'"
        )
        for ch in layer_type_string:
            assert ch in "tpa", f"Invalid layer type '{ch}', must be t/p/a"

        # Register counts
        self.n_reg_c = n_reg_c
        self.n_reg_x = n_reg_x
        self.n_reg_a = n_reg_a
        self.n_reg_p = n_reg_p
        self.n_total_t_reg = n_reg_c + n_reg_x + n_reg_a

        # Count layer types
        self.n_t_layers = sum(1 for c in layer_type_string if c == "t")
        self.n_p_layers = sum(1 for c in layer_type_string if c == "p")
        self.n_a_layers = sum(1 for c in layer_type_string if c == "a")

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_cd = HardLayerNorm(self.C_D)

        # =============== Input Projection ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            from umm.models.umm_v9.world_models.wm_utils import NerfStylePE

            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )
            x_proj_in = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.input_dim
        self.x_proj = nn.Linear(x_proj_in, self.C_D)

        # =============== Time/Position Embeddings ===============
        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.C_D)

        self.int_pe = Timesteps(
            self.C_D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )

        if self.add_time_to_feat:
            self.t_pe = Timesteps(
                self.C_D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            self.t_proj = nn.Linear(self.C_D + 1, self.C_D)

        # =============== Time Binding (no-RoPE path) ===============
        if not self.time_rope_flag and self.add_time_to_feat:
            self.time_bind_proj = nn.Linear(2 * self.C_D, self.C_D)

        # =============== Conditioning ===============
        self.cond_dim = self.feat_dim + 2 * self.C_D

        # Per-layer conditioning binders
        self.cond_binders = nn.ModuleList(
            [nn.Linear(self.cond_dim, self.C_D) for _ in range(self.n_layers)]
        )

        # =============== t-layer Registers (per-layer learnable) ===============
        if self.n_reg_c > 0:
            self.reg_c_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.n_reg_c, self.C_D))
                    for _ in range(self.n_t_layers)
                ]
            )
            for emb in self.reg_c_embeds:
                nn.init.normal_(emb, std=0.02)
        if self.n_reg_x > 0:
            self.reg_x_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.n_reg_x, self.C_D))
                    for _ in range(self.n_t_layers)
                ]
            )
            for emb in self.reg_x_embeds:
                nn.init.normal_(emb, std=0.02)
        if self.n_reg_a > 0:
            self.reg_a_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.n_reg_a, self.C_D))
                    for _ in range(self.n_t_layers)
                ]
            )
            for emb in self.reg_a_embeds:
                nn.init.normal_(emb, std=0.02)

        # =============== p-layer Registers (per-layer learnable) ===============
        if self.n_reg_p > 0:
            self.reg_p_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.n_reg_p, self.C_D))
                    for _ in range(self.n_p_layers)
                ]
            )
            for emb in self.reg_p_embeds:
                nn.init.normal_(emb, std=0.02)

        # =============== Layer-0 Mixing (H5 FIX: gradient highway) ===============
        if self.n_t_layers > 1:
            self.layer0_mix = nn.ModuleList([
                nn.Linear(2 * self.C_D, self.C_D)
                for _ in range(self.n_t_layers - 1)
            ])

        # =============== DiT Blocks (one per position in layer string) ===============
        _dit_kwargs = dict(
            C=self.C_D,
            num_heads=n_head,
            num_registers=num_blk_internal_registers,
            mlp_hidden_dims=[self.C_D * fc_factor] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_D,
            qk_norm=qk_norm,
        )
        self.blocks = nn.ModuleList(
            [DiTBlockTimeOnly(**_dit_kwargs) for _ in range(self.n_layers)]
        )

        # =============== Output Projection ===============
        # Collect skips from t-layers only (like compact) + initial embedding
        n_skips = self.n_t_layers + 1
        out_proj_in = self.C_D * n_skips
        _seq = []
        _c_in = out_proj_in
        for _c_out in np.linspace(out_proj_in, self.input_dim, fc_out_n_layers)[:-1]:
            _c_out = int(_c_out)
            _seq.append(nn.Linear(_c_in, _c_out))
            _seq.append(HardLayerNorm(_c_out))
            _seq.append(nn.LeakyReLU())
            _c_in = _c_out
        _seq.append(nn.Linear(_c_in, self.input_dim))
        self.output_proj = nn.Sequential(*_seq)

        self._log_summary()

    def _log_summary(self):
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info("LMM V11 Dense Summary:")
        logger.info(f"  C_D (unified):     {self.C_D}")
        logger.info(f"  layer_type_string: {self.layer_type_string}")
        logger.info(f"  n_layers:          {self.n_layers}")
        logger.info(f"  t-layers:          {self.n_t_layers}")
        logger.info(f"  p-layers:          {self.n_p_layers}")
        logger.info(f"  a-layers:          {self.n_a_layers}")
        logger.info(f"  t-registers:       c={self.n_reg_c} x={self.n_reg_x} a={self.n_reg_a}")
        logger.info(f"  p-registers:       {self.n_reg_p}")
        logger.info(f"  n_head:            {self.n_head}")
        logger.info(f"  input_dim:         {self.input_dim}")
        logger.info(f"  time_rope_flag:    {self.time_rope_flag}")
        logger.info(f"  params:            {n_params:.2f}M")
        logger.info(
            f"  output skips:      {self.n_t_layers + 1} x {self.C_D}"
            f" = {(self.n_t_layers + 1) * self.C_D}"
        )

    def get_optim_groups(
        self, weight_decay: float = 0.0, base_lr: Optional[float] = None
    ):
        """Optimizer groups with weight-decay separation."""
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
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

    def _build_t_register_attn_mask(
        self, N_CTX: int, T: int, R: int, device: torch.device
    ) -> torch.Tensor:
        """Build structured attention mask for t-layer.

        Sequence layout (after DiT internal register prepend):
            [internal_regs(R), reg-c(nrc), reg-x(nrx), reg-a(nra), ctx(N_CTX), x(T)]

        Registers attend bidirectionally to their assigned group:
            reg-c <-> ctx, reg-x <-> x, reg-a <-> ctx+x.
        ctx <-> x direct communication (not isolated).
        """
        nrc = self.n_reg_c
        nrx = self.n_reg_x
        nra = self.n_reg_a
        L = R + nrc + nrx + nra + N_CTX + T

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

        # reg-c <-> reg-c (self) + ctx
        mask[ir_end:rc_end, ir_end:rc_end] = True
        mask[ir_end:rc_end, ra_end:ctx_end] = True
        mask[ra_end:ctx_end, ir_end:rc_end] = True

        # reg-x <-> reg-x (self) + x
        mask[rc_end:rx_end, rc_end:rx_end] = True
        mask[rc_end:rx_end, ctx_end:L] = True
        mask[ctx_end:L, rc_end:rx_end] = True

        # reg-a <-> reg-a (self) + ctx + x
        mask[rx_end:ra_end, rx_end:ra_end] = True
        mask[rx_end:ra_end, ra_end:ctx_end] = True
        mask[ra_end:ctx_end, rx_end:ra_end] = True
        mask[rx_end:ra_end, ctx_end:L] = True
        mask[ctx_end:L, rx_end:ra_end] = True

        # ctx <-> ctx + x (not isolated)
        mask[ra_end:ctx_end, ra_end:ctx_end] = True
        mask[ctx_end:L, ctx_end:L] = True
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
        group_ids: Optional[torch.Tensor] = None,
        x_r_level: Optional[torch.Tensor] = None,
        ctx_r_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with string-based t/p/a layers.

        Args:
            x: (B, T, P, input_dim) noisy input.
            x_mask: (B, T, P) bool mask.
            x_time: (B, T) timestamps.
            x_level: (B, T, P) diffusion levels.
            feat: (B, P, feat_dim) per-entity identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: (B, A, C_D) language tokens.
            add_tokens_mask: (B, A) mask.
            group_ids: (B, P) group IDs (unused, for interface compat).
            x_r_level: (B, T, P) MeanFlow r-level (optional).
            ctx_r_level: (B, C, P) MeanFlow r-level for context (optional).

        Returns:
            (B, S, P, input_dim) predicted output. S = T + N_CTX.
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
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # (B, T, C_D)
        else:
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert N_CTX <= self.n_ctx
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

        # Build conditioning: (B, S, P, cond_dim)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * P))
            .reshape(B, S, P, self.C_D)
            .to(x)
        )
        # MeanFlow: add r-level embedding (additive, same dim as level_emb)
        if x_r_level is not None:
            r_emb = (
                self.int_pe(x_r_level.reshape(B * S * P))
                .reshape(B, S, P, self.C_D)
                .to(x)
            )
            level_emb = level_emb + r_emb
        feat_expanded = feat[:, None].expand(-1, S, -1, -1)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(-1, -1, P, -1)

        cond_bsp = torch.cat(
            [
                self.hard_ln_feat(feat_expanded),
                self.hard_ln_cd(level_emb),
                self.hard_ln_cd(ctx_pos_emb_expanded),
            ],
            dim=-1,
        )  # (B, S, P, cond_dim)

        # Embed input
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, P, C_D)

        # Time binding (no-RoPE path)
        if not self.time_rope_flag and self.add_time_to_feat:
            time_emb_expanded = ctx_pos_emb[:, :, None, :].expand(-1, -1, P, -1)
            x_emb = self.time_bind_proj(
                torch.cat([x_emb, time_emb_expanded], dim=-1)
            )

        # ========== Prepare Add Tokens ==========
        if add_tokens is not None:
            A = add_tokens.shape[1]
            assert add_tokens_mask is not None
        else:
            A = 0
            add_tokens = torch.zeros(B, 0, self.C_D, device=device)
            add_tokens_mask = torch.zeros(B, 0, device=device).bool()

        # ========== State ==========
        x_state = x_emb.clone()  # (B, S, P, C_D)
        skip_list = [x_emb.reshape(B * S, P, self.C_D).clone()]

        # RoPE time for t-layers
        if self.time_rope_flag:
            t_rope = (
                x_time[:, :, None]
                .expand(-1, -1, P)
                .permute(0, 2, 1)
                .reshape(B * P, T)
            )
        else:
            t_rope = None

        # ========== Build Attention Masks (once before loop) ==========

        # --- t-layer mask ---
        if self.n_t_layers > 0:
            R_t = self.blocks[0].R  # internal registers from first block
            nrc, nrx, nra = self.n_reg_c, self.n_reg_x, self.n_reg_a
            n_total_t_reg = self.n_total_t_reg

            t_struct_mask = self._build_t_register_attn_mask(
                N_CTX, T, R_t, device
            )

            # Validity: (B*P, L) where L = R + n_total_t_reg + S
            traj_valid_t = x_mask.permute(0, 2, 1).reshape(B * P, S)
            ir_valid = torch.ones(B * P, R_t, device=device, dtype=torch.bool)
            t_reg_valid = torch.ones(
                B * P, n_total_t_reg, device=device, dtype=torch.bool
            )
            full_valid_t = torch.cat([ir_valid, t_reg_valid, traj_valid_t], dim=1)
            val_mask_t = full_valid_t[:, None, :] & full_valid_t[:, :, None]
            attn_mask_t = t_struct_mask[None, :, :] & val_mask_t
            # Force diagonal True (prevent NaN from softmax over all-False rows)
            diag_t = torch.arange(attn_mask_t.shape[-1], device=device)
            attn_mask_t[:, diag_t, diag_t] = True
            attn_mask_t = attn_mask_t[:, None].expand(-1, self.n_head, -1, -1)

        # --- p-layer mask ---
        if self.n_p_layers > 0:
            p_block_idx = next(
                i for i, c in enumerate(self.layer_type_string) if c == "p"
            )
            R_p = self.blocks[p_block_idx].R

            # M8 FIX: Exclude fully-padded entities from p-layer attention,
            # but keep per-timestep invalid tokens IN attention.
            # An entity is valid if it has ANY valid timestep across the sequence.
            entity_ever_valid = x_mask.any(dim=1)  # (B, P)
            pert_valid = entity_ever_valid.unsqueeze(1).expand(
                -1, S, -1
            ).reshape(B * S, P)  # (B*S, P)

            # Language tokens tiled over S
            add_mask_tiled = (
                add_tokens_mask.unsqueeze(1)
                .expand(-1, S, -1)
                .reshape(B * S, A)
            )

            p_seq_valid = torch.cat(
                [add_mask_tiled, pert_valid], dim=1
            )  # (B*S, A+P)

            attn_mask_p = build_attention_mask(
                p_seq_valid,
                mode="full",
                left_pad=self.n_reg_p + R_p,
            )[:, None].expand(-1, self.n_head, -1, -1)

        # --- a-layer mask ---
        if self.n_a_layers > 0:
            a_block_idx = next(
                i for i, c in enumerate(self.layer_type_string) if c == "a"
            )
            R_a = self.blocks[a_block_idx].R

            token_valid_a = x_mask.reshape(B, S * P)
            a_seq_valid = torch.cat(
                [add_tokens_mask, token_valid_a], dim=1
            )  # (B, A+S*P)

            attn_mask_a = build_attention_mask(
                a_seq_valid,
                mode="full",
                left_pad=R_a,
            )[:, None].expand(-1, self.n_head, -1, -1)

        # ========== Main Layer Loop ==========
        # C5 FIX: Register state carry-over between t-layers
        prev_reg_c_state = None  # (B*P, nrc, C_D) or None
        prev_reg_x_state = None  # (B*P, nrx, C_D) or None
        prev_reg_a_state = None  # (B*P, nra, C_D) or None
        # H5 FIX: Save initial embedding for layer-0 mixing
        x_state_0 = x_emb.clone()  # (B, S, P, C_D)

        t_layer_idx = 0
        p_layer_idx = 0

        for layer_idx, lt in enumerate(self.layer_type_string):
            block = self.blocks[layer_idx]
            cond_binder = self.cond_binders[layer_idx]

            if lt == "t":
                # ===== t-layer: self-attention across time per entity =====

                # H5 FIX: Layer-0 mixing (gradient highway from initial embedding)
                if self.n_t_layers > 1 and t_layer_idx > 0:
                    x_mix_input = torch.cat([x_state_0, x_state], dim=-1)
                    x_state = self.layer0_mix[t_layer_idx - 1](
                        x_mix_input
                    )  # (B, S, P, C_D)

                # Build register tokens from per-layer learnable embeddings
                # C5 FIX: additive carry-over from previous t-layer's register output
                reg_parts = []
                if nrc > 0:
                    rc = (
                        self.reg_c_embeds[t_layer_idx]
                        .expand(B * P, -1, -1)
                        .reshape(B, P, nrc, self.C_D)
                    )
                    rc = rc.reshape(B * P, nrc, self.C_D)
                    if prev_reg_c_state is not None:
                        rc = rc + prev_reg_c_state
                    reg_parts.append(rc)
                if nrx > 0:
                    rx = (
                        self.reg_x_embeds[t_layer_idx]
                        .expand(B * P, -1, -1)
                        .reshape(B, P, nrx, self.C_D)
                    )
                    rx = rx.reshape(B * P, nrx, self.C_D)
                    if prev_reg_x_state is not None:
                        rx = rx + prev_reg_x_state
                    reg_parts.append(rx)
                if nra > 0:
                    ra = (
                        self.reg_a_embeds[t_layer_idx]
                        .expand(B * P, -1, -1)
                        .reshape(B, P, nra, self.C_D)
                    )
                    ra = ra.reshape(B * P, nra, self.C_D)
                    if prev_reg_a_state is not None:
                        ra = ra + prev_reg_a_state
                    reg_parts.append(ra)

                # Reshape to per-trajectory: (B*P, S, C_D)
                x_traj = x_state.permute(0, 2, 1, 3).reshape(B * P, S, self.C_D)

                # Sequence: [reg-c, reg-x, reg-a, ctx+x]
                if reg_parts:
                    traj_seq = torch.cat(reg_parts + [x_traj], dim=1)
                else:
                    traj_seq = x_traj

                # Conditioning
                cond_t = cond_binder(cond_bsp)  # (B, S, P, C_D)
                cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                    B * P, S, self.C_D
                )

                # Split conditioning by register type (mean of assigned group)
                cond_reg_parts = []
                if nrc > 0:
                    if N_CTX > 0:
                        cond_ctx_mean = cond_t_per_traj[:, :N_CTX].mean(
                            dim=1, keepdim=True
                        )
                    else:
                        cond_ctx_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                    cond_reg_parts.append(cond_ctx_mean.expand(-1, nrc, -1))
                if nrx > 0:
                    if N_CTX < S:
                        cond_x_mean = cond_t_per_traj[:, N_CTX:].mean(
                            dim=1, keepdim=True
                        )
                    else:
                        cond_x_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                    cond_reg_parts.append(cond_x_mean.expand(-1, nrx, -1))
                if nra > 0:
                    cond_all_mean = cond_t_per_traj.mean(dim=1, keepdim=True)
                    cond_reg_parts.append(cond_all_mean.expand(-1, nra, -1))

                if cond_reg_parts:
                    cond_reg = torch.cat(cond_reg_parts, dim=1)
                    cond_traj = torch.cat([cond_reg, cond_t_per_traj], dim=1)
                else:
                    cond_traj = cond_t_per_traj

                # Run t-block
                traj_out = block(
                    x=traj_seq,
                    cond=cond_traj,
                    t=t_rope,
                    num_external_regs=n_total_t_reg + N_CTX,
                    attn_mask=attn_mask_t,
                )

                # C5 FIX: Extract register outputs for carry-over
                # traj_out layout (internal regs stripped by DiT block):
                #   [reg-c(nrc), reg-x(nrx), reg-a(nra), ctx+x(S)]
                reg_offset = 0
                if nrc > 0:
                    prev_reg_c_state = traj_out[:, reg_offset:reg_offset + nrc].clone()
                    reg_offset += nrc
                if nrx > 0:
                    prev_reg_x_state = traj_out[:, reg_offset:reg_offset + nrx].clone()
                    reg_offset += nrx
                if nra > 0:
                    prev_reg_a_state = traj_out[:, reg_offset:reg_offset + nra].clone()

                # Extract entity tokens (skip registers)
                x_out = traj_out[:, n_total_t_reg:]  # (B*P, S, C_D)
                x_state = x_out.reshape(B, P, S, self.C_D).permute(
                    0, 2, 1, 3
                )  # (B, S, P, C_D)

                # Collect skip from t-layer
                skip_list.append(
                    x_state.reshape(B * S, P, self.C_D).clone()
                )
                t_layer_idx += 1

            elif lt == "p":
                # ===== p-layer: cross-attention across entities per timestep =====
                # Dense: processes EVERY timestep individually (not just pooled).
                # Pooled t-features injected as additive residual.

                # Masked pool over time for additive residual (avoid dilution by padding)
                pool_valid = x_mask.float().unsqueeze(-1)  # (B, S, P, 1)
                pool_sum = (x_state * pool_valid).sum(dim=1)  # (B, P, C_D)
                pool_count = pool_valid.sum(dim=1).clamp(min=1)  # (B, P, 1)
                x_pool = pool_sum / pool_count  # (B, P, C_D)

                # Per-timestep entities: (B*S, P, C_D)
                x_flat = x_state.reshape(B * S, P, self.C_D)

                # P-registers: (B*S, n_reg_p, C_D) — skip if n_reg_p=0
                if self.n_reg_p > 0:
                    p_reg = self.reg_p_embeds[p_layer_idx].expand(B * S, -1, -1)

                # Language tokens tiled over S: (B*S, A, C_D)
                add_tiled = (
                    add_tokens.unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(B * S, A, self.C_D)
                )

                # Full sequence: [p_reg, add, entities]
                parts = []
                if self.n_reg_p > 0:
                    parts.append(p_reg)
                parts.extend([add_tiled, x_flat])
                p_seq = torch.cat(parts, dim=1)

                # Conditioning
                cond_p = cond_binder(cond_bsp)  # (B, S, P, C_D)
                cond_flat = cond_p.reshape(B * S, P, self.C_D)

                # Zero conditioning for registers and add tokens
                cond_prefix = torch.zeros(
                    B * S,
                    self.n_reg_p + A,
                    self.C_D,
                    device=device,
                    dtype=cond_flat.dtype,
                )
                cond_full = torch.cat(
                    [cond_prefix, cond_flat], dim=1
                )  # (B*S, n_reg_p+A+P, C_D)

                # Run p-block
                p_out = block(
                    x=p_seq,
                    cond=cond_full,
                    t=None,
                    num_external_regs=self.n_reg_p + A,
                    attn_mask=attn_mask_p,
                )

                # Extract entity tokens (skip registers and add)
                # Sequence inside block: [R_internal, p_reg, add, entities]
                x_flat_out = p_out[:, self.n_reg_p + A :]  # (B*S, P, C_D)
                x_pert = x_flat_out.reshape(B, S, P, self.C_D)

                # Add pooled residual: broadcast over S
                x_state = x_pert + x_pool.unsqueeze(1)

                p_layer_idx += 1

            elif lt == "a":
                # ===== a-layer: full attention over all S*P tokens =====

                # Flatten to (B, S*P, C_D)
                x_flat = x_state.reshape(B, S * P, self.C_D)

                # Sequence: [add_tokens, all_entity_tokens]
                a_seq = torch.cat([add_tokens, x_flat], dim=1)

                # Conditioning
                cond_a = cond_binder(cond_bsp)  # (B, S, P, C_D)
                cond_flat = cond_a.reshape(B, S * P, self.C_D)

                # Zero conditioning for add tokens
                add_cond = torch.zeros(
                    B, A, self.C_D, device=device, dtype=cond_flat.dtype
                )
                cond_full = torch.cat([add_cond, cond_flat], dim=1)

                # Run a-block
                a_out = block(
                    x=a_seq,
                    cond=cond_full,
                    t=None,
                    num_external_regs=A,
                    attn_mask=attn_mask_a,
                )

                # Extract entity tokens
                x_flat_out = a_out[:, A:]  # (B, S*P, C_D)
                x_state = x_flat_out.reshape(B, S, P, self.C_D)

        # ========== Output Projection ==========
        # Use t-layer skips only (like compact)
        x_out = torch.cat(skip_list, dim=-1)  # (B*S, P, (n_t+1)*C_D)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)
