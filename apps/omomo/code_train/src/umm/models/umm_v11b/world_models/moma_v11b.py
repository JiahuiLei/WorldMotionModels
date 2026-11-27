"""MotionMachineV11B -- radically simplified P-only world model.

Replaces all V11 world model variants (compact, dense, sliced, hybrid) with
a single, flat architecture:

    MLP-Input -> P-Layers (DiT self-attention over entities) -> MLP-Output

There are NO t-layers, NO slots, NO pooling/unpooling, NO reg-x, NO
dual-width channels. Each entity's full temporal trajectory is flattened
into a single token processed by self-attention over entities.

MLP Input (per entity, vectorized):
    Concatenates [noise_PE(128) | known_mask(C+T) | time_list(C+T) | 9D_flat((C+T)*9)]
    and projects to C_P via a small MLP.

P-Layers:
    Standard DiTBlockTimeOnly blocks at width C_P with internal registers,
    adaLN conditioning on concat(entity_feat, noise_emb), and padding mask
    for fully-invalid entities.

MLP Output (per entity):
    Projects C_P back to (C+T)*9 and reshapes.

torch.compile audit:
    All forward-path operations are pure tensor ops (cat, expand, reshape,
    matmul) with no closures, dynamic mask construction, or data-dependent
    control flow. DiTBlockTimeOnly is compile-audited in V11. The full
    forward method is compile-safe.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly
from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
)

logger = logging.getLogger(__name__)


class MotionMachineV11B(nn.Module):
    """P-only world model with per-entity temporal flattening.

    Each entity's (C+T) timesteps of 9D data are flattened into a single
    token of dimension C_P, processed by P-layer DiT blocks with
    self-attention over entities, then decoded back to (C+T)*9.

    Args:
        C_P: Channel width for P-layer tokens and DiT blocks.
        feat_dim: Dimension of per-entity identity features from upstream.
        n_p_layers: Number of DiT P-layer blocks.
        n_head_p: Number of attention heads per P-layer.
        len_ctx: Number of context timesteps (C).
        max_T: Maximum number of trace timesteps (T).
        num_blk_internal_registers: Internal register count per DiT block.
        time_multiplier: Scale factor applied to noise_level before PE.
        ada_scale_bias_bound: Soft clamping bound for adaLN scale/bias
            (<=0 disables clamping).
        gate_bound: Soft clamping bound for gates (<=0 disables).
        ada_n_layers: Number of layers in the adaLN projection MLP.
        fc_factor_p: MLP width factor inside DiT blocks (hidden = C_P * fc_factor_p).
        fc_n_layers: Number of hidden layers in each DiT block MLP.
        add_time_to_feat: If True, include noise_level embedding in adaLN
            conditioning (concat with entity_feat).
        qk_norm: Use QK normalization (RMSNorm) in attention.
        noise_pe_dim: Dimension of sinusoidal PE for global noise level.
    """

    def __init__(
        self,
        C_P: int = 512,
        feat_dim: int = 512,
        n_p_layers: int = 3,
        n_head_p: int = 16,
        len_ctx: int = 1,
        max_T: int = 30,
        num_blk_internal_registers: int = 4,
        time_multiplier: float = 30.0,
        ada_scale_bias_bound: float = 1.0,
        gate_bound: float = 2.0,
        ada_n_layers: int = 2,
        fc_factor_p: int = 2,
        fc_n_layers: int = 2,
        add_time_to_feat: bool = True,
        qk_norm: bool = True,
        noise_pe_dim: int = 128,
    ):
        super().__init__()

        self.C_P = C_P
        self.feat_dim = feat_dim
        self.n_p_layers = n_p_layers
        self.len_ctx = len_ctx
        self.max_T = max_T
        self.time_multiplier = time_multiplier
        self.add_time_to_feat = add_time_to_feat
        self.noise_pe_dim = noise_pe_dim

        total_steps = len_ctx + max_T  # C + T
        self.total_steps = total_steps
        self.input_dim = 9

        # =============== Invalid token embedding ===============
        # Learnable 9D embedding substituted for invalid timestep slots
        self.invalid_emb = nn.Parameter(torch.zeros(self.input_dim))
        nn.init.normal_(self.invalid_emb, std=0.02)

        # =============== Noise-level PE ===============
        # Sinusoidal encoding of global noise level (same as V11 Timesteps usage)
        self.noise_pe = Timesteps(
            noise_pe_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )

        # =============== MLP Input Layer ===============
        # Input per entity: noise_PE(noise_pe_dim) + known_mask(total_steps)
        #                  + time_list(total_steps) + 9D_flat(total_steps * 9)
        mlp_input_dim = noise_pe_dim + total_steps + total_steps + total_steps * self.input_dim
        # = noise_pe_dim + (2 + input_dim) * total_steps
        # = noise_pe_dim + 11 * total_steps  (for input_dim=9)

        self.input_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, C_P),
            HardLayerNorm(C_P),
            nn.GELU(approximate="tanh"),
            nn.Linear(C_P, C_P),
        )

        # =============== adaLN conditioning ===============
        # noise_level -> C_P embedding for adaLN conditioning
        self.noise_emb_proj = nn.Sequential(
            nn.Linear(noise_pe_dim, C_P),
            nn.SiLU(),
            nn.Linear(C_P, C_P),
        )

        # C_feat for DiT blocks: concat(feat, noise_emb) -> feat_dim + C_P
        if add_time_to_feat:
            ada_c_feat = feat_dim + C_P
        else:
            ada_c_feat = feat_dim

        # =============== P-Layers (DiT blocks) ===============
        # MLP hidden dims for DiT blocks
        mlp_hidden_dims = []
        for _ in range(fc_n_layers):
            mlp_hidden_dims.append(C_P * fc_factor_p)

        self.p_layers = nn.ModuleList()
        for _ in range(n_p_layers):
            self.p_layers.append(
                DiTBlockTimeOnly(
                    C=C_P,
                    num_heads=n_head_p,
                    mlp_hidden_dims=mlp_hidden_dims,
                    num_registers=num_blk_internal_registers,
                    ada_scale_bias_bound=ada_scale_bias_bound,
                    gate_bound=gate_bound,
                    ada_n_layers=ada_n_layers,
                    C_feat=ada_c_feat,
                    qk_norm=qk_norm,
                )
            )

        # =============== MLP Output Layer ===============
        output_dim = total_steps * self.input_dim  # (C+T) * 9
        self.output_mlp = nn.Sequential(
            nn.Linear(C_P, C_P),
            HardLayerNorm(C_P),
            nn.GELU(approximate="tanh"),
            nn.Linear(C_P, output_dim),
        )

        # Zero-init output projection bias for stable training start
        nn.init.zeros_(self.output_mlp[-1].weight)
        nn.init.zeros_(self.output_mlp[-1].bias)

        # Log architecture summary
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MotionMachineV11B: C_P=%d, n_p_layers=%d, total_steps=%d (ctx=%d, T=%d), "
            "mlp_input_dim=%d, ada_c_feat=%d, params=%.2fM",
            C_P, n_p_layers, total_steps, len_ctx, max_T,
            mlp_input_dim, ada_c_feat, n_params / 1e6,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_time: torch.Tensor,
        x_level: torch.Tensor,
        feat: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        ctx_level: torch.Tensor,
        noise_level: torch.Tensor,
        add_tokens: Optional[torch.Tensor] = None,
        add_tokens_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: flatten per-entity, P-layer attention, decode.

        Args:
            x: (B, T, P, 9) noisy trace data.
            x_mask: (B, T, P) trace validity mask (bool or 0/1).
            x_time: (B, T) real timestep values for trace slots.
            x_level: (B, T, P) diffusion levels per element (UNUSED in V11B,
                kept for interface compatibility with the agent).
            feat: (B, P, feat_dim) per-entity identity features.
            ctx: (B, C, P, 9) context data (clean observations).
            ctx_mask: (B, C, P) context validity mask.
            ctx_level: (B, C, P) context diffusion levels (UNUSED).
            noise_level: (B,) global noise level t in [0, 1].
            add_tokens: (B, L, C_P) language tokens, prepended to P-layer attention.
            add_tokens_mask: (B, L) language token mask (unused, tokens always valid).
            group_ids: (B, P) entity group IDs (ignored in V11B).
            **kwargs: Additional keyword args (ignored, forward compat).

        Returns:
            (B, C+T, P, 9) predicted clean output for all timesteps.
        """
        B, T, P, D = x.shape  # D = 9
        C = ctx.shape[1]  # number of context frames
        S = C + T  # total timesteps
        device = x.device

        # ============================================================
        # 1. Concatenate ctx and x along time dimension
        # ============================================================
        # all_data: (B, S, P, 9)
        all_data = torch.cat([ctx, x], dim=1)
        # all_mask: (B, S, P) -- bool or float, 1=valid
        all_mask = torch.cat([ctx_mask, x_mask], dim=1)
        all_mask_float = all_mask.float()  # (B, S, P)

        # ============================================================
        # 2. Build time_list: ctx slots = 0.0, trace slots = x_time
        # ============================================================
        # ctx times are 0.0; trace times are real timestep values
        ctx_times = torch.zeros(B, C, device=device, dtype=x_time.dtype)  # (B, C)
        time_list = torch.cat([ctx_times, x_time.float()], dim=1)  # (B, S)

        # ============================================================
        # 3. Transpose to per-entity layout and handle invalid tokens
        # ============================================================
        # (B, S, P, 9) -> (B, P, S, 9)
        all_data_pe = all_data.permute(0, 2, 1, 3).contiguous()
        # (B, S, P) -> (B, P, S)
        mask_pe = all_mask_float.permute(0, 2, 1).contiguous()

        # Replace invalid slots (mask=0) with learnable invalid_emb
        # invalid_emb: (9,) -> broadcast to (B, P, S, 9)
        invalid_expanded = self.invalid_emb.view(1, 1, 1, D).expand_as(all_data_pe)
        # mask_pe: (B, P, S) -> (B, P, S, 1) for broadcasting
        mask_3d = mask_pe.unsqueeze(-1)  # (B, P, S, 1)
        all_data_pe = torch.where(mask_3d.bool(), all_data_pe, invalid_expanded)

        # Flatten last two dims: (B, P, S, 9) -> (B, P, S*9)
        flat_9d = all_data_pe.reshape(B, P, S * D)

        # ============================================================
        # 4. Build MLP input vector per entity
        # ============================================================
        # noise_PE: sinusoidal PE of global noise level
        # noise_level: (B,) -> scale -> Timesteps -> (B, noise_pe_dim)
        noise_scaled = noise_level.float() * self.time_multiplier  # (B,)
        noise_pe = self.noise_pe(noise_scaled).to(x.dtype)  # (B, noise_pe_dim)
        # Expand to (B, P, noise_pe_dim)
        noise_pe_expanded = noise_pe.unsqueeze(1).expand(B, P, self.noise_pe_dim)

        # known_mask: (B, P, S) -- already computed as mask_pe
        # time_list: (B, S) -> (B, 1, S) -> (B, P, S)
        time_list_expanded = time_list.unsqueeze(1).expand(B, P, S)

        # Concatenate: [noise_PE | known_mask | time_list | flat_9d]
        # (B, P, noise_pe_dim + S + S + S*9)
        mlp_input = torch.cat(
            [noise_pe_expanded, mask_pe, time_list_expanded, flat_9d],
            dim=-1,
        )

        # MLP input: (B, P, mlp_input_dim) -> (B, P, C_P)
        tokens = self.input_mlp(mlp_input)  # (B, P, C_P)

        # ============================================================
        # 5. Build adaLN conditioning
        # ============================================================
        # noise_emb: (B, noise_pe_dim) -> (B, C_P)
        noise_emb = self.noise_emb_proj(noise_pe)  # (B, C_P)

        if self.add_time_to_feat:
            # cond: concat(feat, noise_emb) -> (B, P, feat_dim + C_P)
            noise_emb_expanded = noise_emb.unsqueeze(1).expand(B, P, self.C_P)
            cond = torch.cat([feat, noise_emb_expanded], dim=-1)  # (B, P, feat_dim + C_P)
        else:
            cond = feat  # (B, P, feat_dim)

        # ============================================================
        # 6. Integrate language tokens into P-layer attention
        # ============================================================
        # Language tokens participate in self-attention alongside entity tokens
        # but are stripped from the output. This allows entities to attend to
        # language conditioning.
        n_lang = 0
        if add_tokens is not None:
            n_lang = add_tokens.shape[1]  # L
            # Prepend language tokens to entity tokens
            tokens = torch.cat([add_tokens, tokens], dim=1)  # (B, L+P, C_P)
            # Language tokens get zero conditioning (adaLN identity)
            lang_cond = torch.zeros(B, n_lang, cond.shape[-1], device=device, dtype=cond.dtype)
            cond = torch.cat([lang_cond, cond], dim=1)  # (B, L+P, C_feat)

        # ============================================================
        # 7. Build padding mask for P-layer attention
        # ============================================================
        # An entity is padded (excluded) if ALL its timesteps are invalid.
        # entity_valid: (B, P) -- True if at least one timestep is valid
        entity_valid = mask_pe.sum(dim=-1) > 0  # (B, P), bool
        # Language tokens are always valid
        if n_lang > 0:
            lang_valid = torch.ones(B, n_lang, device=device, dtype=torch.bool)
            token_valid = torch.cat([lang_valid, entity_valid], dim=1)  # (B, L+P)
        else:
            token_valid = entity_valid  # (B, P)

        R = self.p_layers[0].R  # internal register count
        assert all(layer.R == R for layer in self.p_layers), \
            "All P-layers must have the same number of internal registers"
        p_attn_mask = build_attention_mask(
            token_valid, mode="full", left_pad=R,
        )  # (B, R+L+P, R+L+P)
        p_attn_mask = p_attn_mask.unsqueeze(1)  # (B, 1, R+L+P, R+L+P)

        # ============================================================
        # 8. P-Layers: DiT self-attention over language + entity tokens
        # ============================================================
        for p_layer in self.p_layers:
            tokens = p_layer(
                tokens, cond, t=None, num_external_regs=0, attn_mask=p_attn_mask,
            )  # (B, L+P, C_P)

        # Strip language tokens -- only entity tokens go to output MLP
        if n_lang > 0:
            tokens = tokens[:, n_lang:]  # (B, P, C_P)

        # ============================================================
        # 8. MLP Output: decode per-entity token to (C+T)*9
        # ============================================================
        out_flat = self.output_mlp(tokens)  # (B, P, S*9)
        # Reshape: (B, P, S*9) -> (B, P, S, 9) -> (B, S, P, 9)
        out = out_flat.reshape(B, P, S, D).permute(0, 2, 1, 3).contiguous()

        return out  # (B, C+T, P, 9)

    def get_optim_groups(
        self, weight_decay: float = 0.0, base_lr: Optional[float] = None,
    ):
        """Optimizer parameter groups with weight-decay separation.

        Weight matrices (dim >= 2) get weight_decay; biases and layernorm
        params (dim < 2) get zero weight_decay.

        Args:
            weight_decay: Weight decay for decay group.
            base_lr: If provided, set as explicit lr for all groups.

        Returns:
            List of optimizer param group dicts.
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
