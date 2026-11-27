"""w1.0.2: cross-attention input layer
with the v2 pipeline PLUS a projected pooled-language vector surfaced to the
Agent for insertion into `cond_parts` (replacing the `hard_ln_lat` /
`add_first_n_add_token_to_feat` slot).

Forked from ``input_xattn_v2.py``. Keeps the entire v2 pipeline intact:

    Transformer 1: Language Pooling with RoPE
        seq = [ lang_proj  |  register ]  (B, L+R, D1)
        sequential 1D RoPE positions arange(L+R)
        -> extract register slice -> pooled_lang  (B, R*D1)

    Transformer 2: Entity-Lang-Img Self-Attn with RoPE
        seq = [ lang_proj  |  img_proj?  |  entity_proj ]
        lang/img sequential positions; entity positions random-permuted
        (training) / sequential (eval), offset past lang+img.
        -> extract entity slice -> entity_out  (B, P, D1)

    Fuse:
        entity_combined = cat([entity_out, pooled_lang.expand(P)])
        feat_out = feat_out_proj(entity_combined)  (B, P, feat_dim)

    NEW in v3 (vs v2):
        pooled_lang_proj = pool_proj(pooled_lang)  (B, D1)
        returned as an extra field of the output NamedTuple.

    Optional learned traj_id head (unchanged from v2).

Why v3 (Q6 / highlevel.md R2): v2's `pooled_lang` is ``(B, R*D1) = (B, 1024)``
at default R=2, D1=512. The `cond_parts` slot currently filled by
``hard_ln_lat(lang_cat_first_N_proj)`` expects ``(B, D1) = (B, 512)``. v3
adds an internal ``pool_proj: Linear(R*D1, D1)`` so Agent.forward can feed a
correctly-shaped pooled-lang vector into cond_parts.

Lang/img pass-through (for interface symmetry): the output NamedTuple's
``lang_out`` and ``img_out`` fields are the IDENTICAL input tensors.
Transformer 2 mutates its own internal sequence but — matching v2 — does NOT
export an updated lang/img slice. Callers that need v2-parity can ignore
these fields. They exist so the output tuple is self-describing.

Widths (langtable base, w1.0.2 config):
    D1 = lmm.D1 = 512
    feat_dim = model.feat_dim = 1024
    n_register = 2 -> pool_proj: Linear(1024, 512)
"""

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from umm.models.wmm.input_xattn_v2 import RoPETransformerLayer


class InputXattnV3Out(NamedTuple):
    """V3 forward output.

    Fields:
        lang_out: (B, A_lang, D1) — pass-through of ``lang_proj`` input
            (v3 does NOT update lang tokens; field provided for interface
            self-description). Matches v2 behavior.
        img_out: Optional[(B, A_img, D1)] — pass-through of ``img_proj``
            input. ``None`` if caller passed ``img_proj=None``.
        feat_out: (B, P, feat_dim) — new per-entity features (fused
            entity_out + pooled_lang via feat_out_proj).
        traj_ids: Optional[(B, P) float] — learned traj_ids (sigmoid*K),
            ``None`` if ``learned_traj_id=False``.
        pooled_lang_proj: (B, D1) — pooled language latent projected from
            ``(B, R*D1)`` via ``pool_proj``. Intended for cond_parts
            insertion at the `hard_ln_lat` slot in the LMM.
    """
    lang_out: torch.Tensor
    img_out: Optional[torch.Tensor]
    feat_out: torch.Tensor
    traj_ids: Optional[torch.Tensor]
    pooled_lang_proj: torch.Tensor


class InputLayerCrossAttnV3(nn.Module):
    """V3 input cross-attention: v2 pipeline + projected pooled-lang vector.

    Args:
        feat_dim: per-entity feature width from the existing input_layer.
        D1: LMM add_token width (= lmm.D1).
        n_layers: number of RoPETransformerLayers in EACH transformer.
        n_register: number of learnable register tokens in Transformer 1.
        n_heads: attention heads for the RoPETransformerLayers.
        learned_traj_id: if True, adds a linear head that produces per-entity
            traj_id values via sigmoid * K.
        traj_id_bound_K: upper bound for learned_traj_ids (sigmoid * K).
        traj_id_K_warmup_start: training step at which K warmup begins
            (compile-safe: actual schedule is computed in Agent.forward and
            passed in via ``effective_K``).
        traj_id_K_warmup_duration: steps over which K ramps 0 -> bound_K.
            0 disables warmup.
    """

    def __init__(
        self,
        feat_dim: int,
        D1: int,
        n_layers: int = 2,
        n_register: int = 2,
        n_heads: int = 8,
        learned_traj_id: bool = False,
        traj_id_bound_K: float = 1.0,
        traj_id_K_warmup_start: int = 0,
        traj_id_K_warmup_duration: int = 0,
    ):
        super().__init__()
        assert feat_dim > 0 and D1 > 0, f"feat_dim={feat_dim} D1={D1}"
        assert n_layers >= 1, f"n_layers={n_layers}"
        assert n_register >= 1, f"n_register={n_register}"
        assert D1 % n_heads == 0, (
            f"D1={D1} must be divisible by n_heads={n_heads}"
        )
        assert traj_id_K_warmup_start >= 0, (
            f"traj_id_K_warmup_start={traj_id_K_warmup_start} must be >= 0"
        )
        assert traj_id_K_warmup_duration >= 0, (
            f"traj_id_K_warmup_duration={traj_id_K_warmup_duration} must be >= 0"
        )
        self.feat_dim = feat_dim
        self.D1 = D1
        self.n_layers = n_layers
        self.n_register = n_register
        self.n_heads = n_heads
        self.learned_traj_id = learned_traj_id
        self.traj_id_bound_K = traj_id_bound_K
        self.traj_id_K_warmup_start = traj_id_K_warmup_start
        self.traj_id_K_warmup_duration = traj_id_K_warmup_duration

        # Project entity feats to D1 for joint attention.
        self.feat_in_proj = nn.Linear(feat_dim, D1)

        # Transformer 1: Language Pooling with RoPE.
        self.lang_pool_layers = nn.ModuleList([
            RoPETransformerLayer(D1, n_heads)
            for _ in range(n_layers)
        ])

        # Learnable register tokens (small init).
        self.register = nn.Parameter(
            torch.randn(1, n_register, D1) * 0.02
        )

        # Transformer 2: Entity-Lang-Img Self-Attn with RoPE.
        self.entity_xattn_layers = nn.ModuleList([
            RoPETransformerLayer(D1, n_heads)
            for _ in range(n_layers)
        ])

        # Fuse entity_out + pooled_lang -> feat_dim.
        self.feat_out_proj = nn.Linear(D1 + n_register * D1, feat_dim)

        # NEW in v3: project pooled_lang (B, R*D1) -> (B, D1) so the Agent
        # can feed it into cond_parts at the `hard_ln_lat` slot.
        self.pool_proj = nn.Linear(n_register * D1, D1)

        # Optional learned traj_id head.
        if learned_traj_id:
            self.traj_id_head = nn.Linear(D1, 1)
            nn.init.normal_(self.traj_id_head.weight, std=0.02)
            nn.init.zeros_(self.traj_id_head.bias)

    @torch.compiler.disable
    def _sample_entity_positions(
        self, B: int, P: int, device: torch.device
    ) -> torch.Tensor:
        """Entity position IDs for RoPE.

        Training: random permutation per sample (permutation invariance).
        Eval: sequential arange(P).

        Returns:
            (B, P) long tensor of position indices.
        """
        if self.training:
            return torch.stack(
                [torch.randperm(P, device=device) for _ in range(B)]
            )
        return torch.arange(P, device=device)[None].expand(B, -1)

    def forward(
        self,
        lang_proj: torch.Tensor,            # (B, A_lang, D1)
        lang_mask: torch.Tensor,            # (B, A_lang) bool True=valid
        img_proj: Optional[torch.Tensor],   # (B, A_img, D1) or None
        img_mask: Optional[torch.Tensor],   # (B, A_img) bool or None
        feat: torch.Tensor,                 # (B, P, feat_dim)
        valid_entity_mask: torch.Tensor,    # (B, P) bool
        effective_K: Optional[float] = None,
    ) -> InputXattnV3Out:
        """Returns ``InputXattnV3Out(lang_out, img_out, feat_out, traj_ids,
        pooled_lang_proj)``.

        Args:
            effective_K: if not None, use this Python float as the sigmoid
                upper bound instead of ``self.traj_id_bound_K``. Agent.forward
                passes the warmup-adjusted value to avoid recompilation across
                the compile boundary. Inference callers pass
                ``self.traj_id_bound_K`` directly (no warmup).
        """
        B, P, _ = feat.shape
        device = feat.device
        L = lang_proj.shape[1]  # A_lang
        R = self.n_register

        # ═══════════════════════════════════════════════════════════════
        # Transformer 1: Language Pooling with RoPE
        # ═══════════════════════════════════════════════════════════════
        reg = self.register.expand(B, -1, -1)  # (B, R, D1)
        seq1 = torch.cat([lang_proj, reg], dim=1)  # (B, L+R, D1)

        pos1 = torch.arange(
            L + R, device=device, dtype=torch.float32
        )[None].expand(B, -1)  # (B, L+R)

        reg_mask = torch.ones(B, R, device=device, dtype=torch.bool)
        mask1 = torch.cat([lang_mask, reg_mask], dim=1)  # (B, L+R)
        kpm1 = ~mask1  # True = IGNORE

        h1 = seq1
        for layer in self.lang_pool_layers:
            h1 = layer(h1, positions=pos1, key_padding_mask=kpm1)

        # Pool via register slice -> (B, R*D1) latent.
        pooled_lang = h1[:, L:].reshape(B, R * self.D1)  # (B, R*D1)

        # NEW: project pooled_lang to (B, D1) for cond_parts insertion.
        pooled_lang_proj = self.pool_proj(pooled_lang)  # (B, D1)

        # ═══════════════════════════════════════════════════════════════
        # Transformer 2: Entity-Lang-Img Self-Attn with RoPE
        # ═══════════════════════════════════════════════════════════════
        entity_proj = self.feat_in_proj(feat)  # (B, P, D1)

        if img_proj is not None:
            assert img_mask is not None, (
                "img_mask must be provided when img_proj is not None"
            )
            I = img_proj.shape[1]
            seq2 = torch.cat([lang_proj, img_proj, entity_proj], dim=1)
            mask2 = torch.cat([lang_mask, img_mask, valid_entity_mask], dim=1)

            lang_pos = torch.arange(
                L, device=device, dtype=torch.float32
            )[None].expand(B, -1)  # (B, L)
            img_pos = torch.arange(
                L, L + I, device=device, dtype=torch.float32
            )[None].expand(B, -1)  # (B, I)
            # Entity positions offset past lang+img to avoid collisions.
            entity_pos = self._sample_entity_positions(B, P, device).float() + (L + I)
            pos2 = torch.cat([lang_pos, img_pos, entity_pos], dim=1)  # (B, L+I+P)
        else:
            I = 0
            seq2 = torch.cat([lang_proj, entity_proj], dim=1)
            mask2 = torch.cat([lang_mask, valid_entity_mask], dim=1)

            lang_pos = torch.arange(
                L, device=device, dtype=torch.float32
            )[None].expand(B, -1)
            entity_pos = self._sample_entity_positions(B, P, device).float() + L
            pos2 = torch.cat([lang_pos, entity_pos], dim=1)  # (B, L+P)

        kpm2 = ~mask2  # True = IGNORE

        h2 = seq2
        for layer in self.entity_xattn_layers:
            h2 = layer(h2, positions=pos2, key_padding_mask=kpm2)

        entity_out = h2[:, L + I:]  # (B, P, D1)

        # ═══════════════════════════════════════════════════════════════
        # Fuse: entity_out + pooled_lang -> feat_dim
        # ═══════════════════════════════════════════════════════════════
        entity_combined = torch.cat(
            [entity_out, pooled_lang[:, None, :].expand(-1, P, -1)],
            dim=-1,
        )  # (B, P, D1 + R*D1)
        feat_out = self.feat_out_proj(entity_combined)  # (B, P, feat_dim)

        # Optional learned traj_id head.
        traj_ids = None
        if self.learned_traj_id:
            K = self.traj_id_bound_K if effective_K is None else effective_K
            traj_ids = (
                torch.sigmoid(self.traj_id_head(entity_out).squeeze(-1)) * K
            )  # (B, P)

        return InputXattnV3Out(
            lang_out=lang_proj,
            img_out=img_proj,
            feat_out=feat_out,
            traj_ids=traj_ids,
            pooled_lang_proj=pooled_lang_proj,
        )
