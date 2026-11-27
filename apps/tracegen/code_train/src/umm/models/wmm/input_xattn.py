"""V13.2 V6: cross-attention input layer + lang-register global latent.

Drop-in module invoked from ``Agent.forward`` AFTER the existing ``input_layer``
produces per-entity features and AFTER the language / DINO token projectors
run, BEFORE the LMM call. Back-compat default: the Agent only instantiates
this when ``model.input_xattn.enable: True``.

Architecture:

    Block A (self-attn over concatenated stream, n_layers TransformerEncoderLayers):
        seq = [ lang_proj  |  img_proj?  |  entity_proj ]
        mask = [ lang_mask  |  img_mask?  |  valid_entity_mask ]
        -> splits back into lang_out, img_out?, entity_out_a

    Block B (self-attn over [raw lang || register], n_layers TransformerEncoderLayers):
        seq = [ lang_proj_raw  |  register ]
        -> takes register slice -> reshape channel-wise -> (B, n_register * D1)
        = "global language latent"

    Final per-entity feature:
        entity_combined = cat([entity_out_a, global_latent.expand(P)], dim=-1)
        feat_out = feat_out_proj(entity_combined)  # back to feat_dim

When ``img_proj is None`` (e.g. langtable), Block A drops the image slice with
no None-check crashes.

Widths (langtable base config):
    D1 = lmm.D1 = 512       (lang/img already in this width)
    feat_dim = model.feat_dim = 1024
    feat_in_proj:  Linear(feat_dim, D1)
    feat_out_proj: Linear(D1 + n_register * D1, feat_dim)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class InputLayerCrossAttn(nn.Module):
    """V6 cross-attention input layer.

    Args:
        feat_dim: per-entity feature width from the existing input_layer.
        D1: LMM add_token width (= lmm.D1). Lang/image tokens are already
            projected to this width by the Agent.
        n_layers: number of TransformerEncoderLayers in EACH of Block A and
            Block B.
        n_register: number of learnable register tokens in Block B.
        n_heads: attention heads for the TransformerEncoderLayers.
    """

    def __init__(
        self,
        feat_dim: int,
        D1: int,
        n_layers: int = 2,
        n_register: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        assert feat_dim > 0 and D1 > 0, f"feat_dim={feat_dim} D1={D1}"
        assert n_layers >= 1, f"n_layers={n_layers}"
        assert n_register >= 1, f"n_register={n_register}"
        assert D1 % n_heads == 0, (
            f"D1={D1} must be divisible by n_heads={n_heads}"
        )
        self.feat_dim = feat_dim
        self.D1 = D1
        self.n_layers = n_layers
        self.n_register = n_register
        self.n_heads = n_heads

        self.feat_in_proj = nn.Linear(feat_dim, D1)

        def _make_encoder_stack(n: int) -> nn.ModuleList:
            return nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=D1,
                    nhead=n_heads,
                    dim_feedforward=4 * D1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n)
            ])

        self.block_a = _make_encoder_stack(n_layers)
        self.block_b = _make_encoder_stack(n_layers)

        # Learnable register tokens for Block B. Small-init so Block B starts
        # close to a pure lang-summarizer.
        self.register = nn.Parameter(
            torch.randn(1, n_register, D1) * 0.02
        )

        # Final per-entity projection. Global latent is channel-wise concat of
        # n_register D1-dim register states -> (n_register * D1).
        self.feat_out_proj = nn.Linear(D1 + n_register * D1, feat_dim)

    def forward(
        self,
        lang_proj: torch.Tensor,          # (B, A_lang, D1)
        lang_mask: torch.Tensor,          # (B, A_lang) bool (True = valid)
        img_proj: Optional[torch.Tensor],  # (B, A_img, D1) or None
        img_mask: Optional[torch.Tensor],  # (B, A_img) bool or None
        feat: torch.Tensor,               # (B, P, feat_dim)
        valid_entity_mask: torch.Tensor,  # (B, P) bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Returns (lang_out, img_out, feat_out).

        ``lang_out`` and ``feat_out`` replace the original ``lang_proj`` and
        ``feat`` fed into the LMM. ``img_out`` replaces ``img_proj`` when the
        image stream is present, else None.
        """
        B, P, _ = feat.shape
        A_lang = lang_proj.shape[1]

        # --- Project feat -> D1 for joint attention ---
        entity_proj = self.feat_in_proj(feat)  # (B, P, D1)

        # --- Block A: self-attention over [lang || img? || entity] ---
        if img_proj is not None:
            assert img_mask is not None, (
                "img_mask must be provided when img_proj is not None"
            )
            A_img = img_proj.shape[1]
            seq_a = torch.cat([lang_proj, img_proj, entity_proj], dim=1)
            mask_a = torch.cat([lang_mask, img_mask, valid_entity_mask], dim=1)
        else:
            A_img = 0
            seq_a = torch.cat([lang_proj, entity_proj], dim=1)
            mask_a = torch.cat([lang_mask, valid_entity_mask], dim=1)

        # nn.TransformerEncoderLayer uses src_key_padding_mask where True =
        # IGNORE. Invert our valid-mask (True = valid) into that convention.
        key_padding_a = ~mask_a
        h_a = seq_a
        for layer in self.block_a:
            h_a = layer(h_a, src_key_padding_mask=key_padding_a)

        lang_out = h_a[:, :A_lang]
        if img_proj is not None:
            img_out: Optional[torch.Tensor] = h_a[:, A_lang: A_lang + A_img]
        else:
            img_out = None
        entity_out_a = h_a[:, A_lang + A_img:]  # (B, P, D1)

        # --- Block B: raw lang + learnable registers ---
        # Uses pre-A lang_proj (NOT Block A's lang_out), per spec.
        reg = self.register.expand(B, -1, -1)  # (B, n_register, D1)
        seq_b = torch.cat([lang_proj, reg], dim=1)  # (B, A_lang + n_register, D1)
        reg_mask = torch.ones(
            B, self.n_register, device=lang_mask.device, dtype=torch.bool
        )
        mask_b = torch.cat([lang_mask, reg_mask], dim=1)
        key_padding_b = ~mask_b
        h_b = seq_b
        for layer in self.block_b:
            h_b = layer(h_b, src_key_padding_mask=key_padding_b)

        reg_out = h_b[:, A_lang:]  # (B, n_register, D1)
        global_latent = reg_out.reshape(B, self.n_register * self.D1)

        # --- Final: combine entity with global latent ---
        entity_combined = torch.cat(
            [entity_out_a, global_latent[:, None, :].expand(-1, P, -1)],
            dim=-1,
        )  # (B, P, D1 + n_register * D1)
        feat_out = self.feat_out_proj(entity_combined)  # (B, P, feat_dim)

        return lang_out, img_out, feat_out
