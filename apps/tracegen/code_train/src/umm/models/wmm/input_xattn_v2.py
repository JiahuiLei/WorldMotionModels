"""V13.2 Explore 10: cross-attention input layer with RoPE + learned traj-ids.

Drop-in module invoked from ``Agent.forward`` AFTER the existing ``input_layer``
produces per-entity features and AFTER the language / DINO token projectors
run, BEFORE the LMM call. Back-compat default: the Agent only instantiates
this when ``model.input_xattn_v2.enable: True``.

Architecture:

    Transformer 1 (Language Pooling with RoPE):
        seq = [ lang_proj  |  register ]
        Sequential 1D RoPE positions (arange(L + n_register))
        -> extract register slice -> reshape to (B, n_register * D1)
        = "pooled language latent"

    Transformer 2 (Entity-Lang-Img Self-Attn with RoPE):
        seq = [ lang_proj  |  img_proj?  |  entity_proj ]
        Positions:
            lang:   arange(L), expanded to (B, L)
            img:    arange(L, L+I), expanded to (B, I)
            entity: random permuted per sample (training) / sequential (eval)
        -> extract entity slice -> entity_out (B, P, D1)

    Final per-entity feature:
        entity_combined = cat([entity_out, pooled_lang.expand(P)], dim=-1)
        feat_out = feat_out_proj(entity_combined)  # back to feat_dim

    Optional learned traj_id head:
        traj_ids = sigmoid(traj_id_head(entity_out)) * K  # (B, P) float

Differences from V6 (input_xattn.py):
    - Uses custom RoPETransformerLayer instead of nn.TransformerEncoderLayer.
    - Does NOT return updated lang/img (only entity feats and traj_ids change).
    - Optional learned traj_id head for downstream LMM entity identification.
    - Entity positions are randomly permuted during training (RoPE-based
      permutation invariance learning).

Widths (langtable base config):
    D1 = lmm.D1 = 512       (lang/img already in this width)
    feat_dim = model.feat_dim = 1024
    feat_in_proj:  Linear(feat_dim, D1)
    feat_out_proj: Linear(D1 + n_register * D1, feat_dim)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from umm.models.wmm.world_models.dit_atten import compute_rope_freqs, rope_apply


class RoPETransformerLayer(nn.Module):
    """Transformer encoder layer with Rotary Position Embeddings.

    Pre-norm (LayerNorm) -> QKV projection -> split heads -> RoPE ->
    SDPA with optional key_padding_mask -> output projection -> residual.
    FFN: Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)
    -> residual.

    Args:
        d_model: model dimension.
        n_heads: number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,                              # (B, L, D)
        positions: torch.Tensor,                       # (B, L) float
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) bool, True=IGNORE
    ) -> torch.Tensor:
        """Forward with RoPE on all positions.

        Args:
            x: input tensor (B, L, D).
            positions: per-token position values for RoPE (B, L) float.
            key_padding_mask: bool mask where True = IGNORE (padding).
                Converted to additive float mask for SDPA.

        Returns:
            Output tensor (B, L, D).
        """
        B, L, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        # --- Attention block ---
        h = self.norm1(x)
        qkv = self.qkv(h)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, H, L, Dh)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)

        # Apply RoPE to all positions
        cos, sin = compute_rope_freqs(positions, Dh, x.device)
        # cos, sin: (B, L, Dh/2) -> (B, 1, L, Dh/2) for broadcasting over heads
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = rope_apply(q, cos, sin)
        k = rope_apply(k, cos, sin)

        # Build additive attention mask from key_padding_mask
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) bool, True = IGNORE
            # SDPA attn_mask: (B, 1, 1, L) float, -inf for ignored positions
            attn_mask = torch.zeros(
                B, 1, 1, L, device=x.device, dtype=x.dtype
            )
            attn_mask.masked_fill_(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        # Residual
        x = x + out

        # --- FFN block ---
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x


class InputLayerCrossAttnV2(nn.Module):
    """V13.2 Explore 10: RoPE cross-attention input layer with optional
    learned traj-id head.

    Args:
        feat_dim: per-entity feature width from the existing input_layer.
        D1: LMM add_token width (= lmm.D1). Lang/image tokens are already
            projected to this width by the Agent.
        n_layers: number of RoPETransformerLayers in EACH transformer.
        n_register: number of learnable register tokens in Transformer 1.
        n_heads: attention heads for the RoPETransformerLayers.
        learned_traj_id: if True, adds a linear head that produces per-entity
            traj_id values via sigmoid * K.
        traj_id_bound_K: upper bound for the learned traj_ids (sigmoid * K).
        traj_id_K_warmup_start: training step at which K warmup begins.
            Stored for reference; the actual warmup schedule is computed in
            Agent.forward() and passed in via ``effective_K`` (compile-safe).
        traj_id_K_warmup_duration: number of steps over which K ramps from
            0 to ``traj_id_bound_K``. 0 disables warmup (default).
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

        # Project entity feats to D1 for joint attention
        self.feat_in_proj = nn.Linear(feat_dim, D1)

        # Transformer 1: Language Pooling with RoPE
        self.lang_pool_layers = nn.ModuleList([
            RoPETransformerLayer(D1, n_heads)
            for _ in range(n_layers)
        ])

        # Learnable register tokens (small init)
        self.register = nn.Parameter(
            torch.randn(1, n_register, D1) * 0.02
        )

        # Transformer 2: Entity-Lang-Img Self-Attn with RoPE
        self.entity_xattn_layers = nn.ModuleList([
            RoPETransformerLayer(D1, n_heads)
            for _ in range(n_layers)
        ])

        # Final: fuse entity_out + pooled_lang -> feat_dim
        self.feat_out_proj = nn.Linear(D1 + n_register * D1, feat_dim)

        # Optional learned traj_id head
        if learned_traj_id:
            self.traj_id_head = nn.Linear(D1, 1)
            nn.init.normal_(self.traj_id_head.weight, std=0.02)
            nn.init.zeros_(self.traj_id_head.bias)

    @torch.compiler.disable
    def _sample_entity_positions(
        self, B: int, P: int, device: torch.device
    ) -> torch.Tensor:
        """Sample entity position IDs for RoPE.

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
        lang_mask: torch.Tensor,            # (B, A_lang) bool (True = valid)
        img_proj: Optional[torch.Tensor],   # (B, A_img, D1) or None
        img_mask: Optional[torch.Tensor],   # (B, A_img) bool or None
        feat: torch.Tensor,                 # (B, P, feat_dim)
        valid_entity_mask: torch.Tensor,    # (B, P) bool
        effective_K: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (feat_out, traj_ids).

        ``feat_out`` replaces the original ``feat`` fed into the LMM.
        ``traj_ids`` is (B, P) float if learned_traj_id else None.
        Does NOT return updated lang/img -- only entity feats change.

        Args:
            effective_K: If not None, use this Python float as the sigmoid
                upper bound instead of ``self.traj_id_bound_K``. Used by
                Agent.forward() to apply a K-warmup schedule without crossing
                the compile boundary with a changing int (global_step).
                Inference callers pass ``self.traj_id_bound_K`` directly for
                no-warmup behavior.
        """
        B, P, _ = feat.shape
        device = feat.device
        L = lang_proj.shape[1]  # A_lang
        R = self.n_register

        # ═══════════════════════════════════════════════════════════════
        # Transformer 1: Language Pooling with RoPE
        # ═══════════════════════════════════════════════════════════════
        # seq = [lang_proj | register]  (B, L+R, D1)
        reg = self.register.expand(B, -1, -1)  # (B, R, D1)
        seq1 = torch.cat([lang_proj, reg], dim=1)  # (B, L+R, D1)

        # Sequential positions for RoPE: arange(L+R), same for all batch
        pos1 = torch.arange(
            L + R, device=device, dtype=torch.float32
        )[None].expand(B, -1)  # (B, L+R)

        # Mask: lang_mask + ones for registers
        reg_mask = torch.ones(B, R, device=device, dtype=torch.bool)
        mask1 = torch.cat([lang_mask, reg_mask], dim=1)  # (B, L+R)
        kpm1 = ~mask1  # True = IGNORE

        h1 = seq1
        for layer in self.lang_pool_layers:
            h1 = layer(h1, positions=pos1, key_padding_mask=kpm1)

        # Extract register slice and reshape to pooled language latent
        pooled_lang = h1[:, L:].reshape(B, R * self.D1)  # (B, R*D1)

        # ═══════════════════════════════════════════════════════════════
        # Transformer 2: Entity-Lang-Img Self-Attn with RoPE
        # ═══════════════════════════════════════════════════════════════
        entity_proj = self.feat_in_proj(feat)  # (B, P, D1)

        # Build sequence and positions
        if img_proj is not None:
            assert img_mask is not None, (
                "img_mask must be provided when img_proj is not None"
            )
            I = img_proj.shape[1]
            seq2 = torch.cat([lang_proj, img_proj, entity_proj], dim=1)
            mask2 = torch.cat([lang_mask, img_mask, valid_entity_mask], dim=1)

            # Positions: lang=[0..L), img=[L..L+I), entity=offset random permuted
            # Entity positions are offset by L+I to avoid collision with
            # lang/img sequential positions (exe-critic BLOCK fix).
            lang_pos = torch.arange(
                L, device=device, dtype=torch.float32
            )[None].expand(B, -1)  # (B, L)
            img_pos = torch.arange(
                L, L + I, device=device, dtype=torch.float32
            )[None].expand(B, -1)  # (B, I)
            entity_pos = self._sample_entity_positions(B, P, device).float() + (L + I)  # (B, P)
            pos2 = torch.cat([lang_pos, img_pos, entity_pos], dim=1)  # (B, L+I+P)
        else:
            I = 0
            seq2 = torch.cat([lang_proj, entity_proj], dim=1)
            mask2 = torch.cat([lang_mask, valid_entity_mask], dim=1)

            # Positions: lang=[0..L), entity=offset random permuted
            # Entity positions offset by L to avoid collision (exe-critic BLOCK fix).
            lang_pos = torch.arange(
                L, device=device, dtype=torch.float32
            )[None].expand(B, -1)  # (B, L)
            entity_pos = self._sample_entity_positions(B, P, device).float() + L  # (B, P)
            pos2 = torch.cat([lang_pos, entity_pos], dim=1)  # (B, L+P)

        kpm2 = ~mask2  # True = IGNORE

        h2 = seq2
        for layer in self.entity_xattn_layers:
            h2 = layer(h2, positions=pos2, key_padding_mask=kpm2)

        # Extract entity slice
        entity_out = h2[:, L + I:]  # (B, P, D1)

        # ═══════════════════════════════════════════════════════════════
        # Fuse: entity_out + pooled_lang -> feat_dim
        # ═══════════════════════════════════════════════════════════════
        entity_combined = torch.cat(
            [entity_out, pooled_lang[:, None, :].expand(-1, P, -1)],
            dim=-1,
        )  # (B, P, D1 + R*D1)
        feat_out = self.feat_out_proj(entity_combined)  # (B, P, feat_dim)

        # ═══════════════════════════════════════════════════════════════
        # Optional learned traj_id head
        # ═══════════════════════════════════════════════════════════════
        traj_ids = None
        if self.learned_traj_id:
            K = self.traj_id_bound_K if effective_K is None else effective_K
            # sigmoid * K -> [0, K]
            traj_ids = (
                torch.sigmoid(self.traj_id_head(entity_out).squeeze(-1)) * K
            )  # (B, P)

        return feat_out, traj_ids
