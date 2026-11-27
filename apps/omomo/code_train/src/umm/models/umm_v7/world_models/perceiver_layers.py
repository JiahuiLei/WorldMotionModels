"""Perceiver IO encode/decode layers for V5 latent pooling.

Two layer types for compressing/expanding the part dimension:
- **PerceiverEncodeLayer** (``q`` layer): Q latent queries attend to P dense tokens.
  Supports ``async`` (independent per timestep) and ``synced`` (fused attention
  weights across timesteps) modes.
- **PerceiverDecodeLayer** (``d`` layer): P dense queries attend to Q latent tokens.
  Always async. Outputs are collected as skip connections.

Example::

    q_layer = PerceiverEncodeLayer(C=512, num_heads=16, C_feat=512)
    d_layer = PerceiverDecodeLayer(C=512, num_heads=16, C_feat=512)

    x_q = q_layer(x_q, x_p, cond_q)           # (B*S, Q, D) ← (B*S, P, D)
    x_p_decoded = d_layer(x_p, x_q, cond_p)   # (B*S, P, D) ← (B*S, Q, D)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


def _build_ada(c_in: int, c_out: int, n_layers: int) -> nn.Module:
    """Build AdaLN projection (1-layer linear or multi-layer MLP)."""
    if n_layers == 1:
        layer = nn.Linear(c_in, c_out)
        nn.init.zeros_(layer.bias)
        return layer
    _outdims = np.linspace(c_in, c_out, n_layers).astype(int).tolist()[1:]
    ada = []
    _ci = c_in
    for _co in _outdims:
        ada.append(nn.Linear(_ci, _co))
        ada.append(nn.LayerNorm(_co))
        ada.append(nn.LeakyReLU(0.2))
        _ci = _co
    return nn.Sequential(*ada)


class PerceiverEncodeLayer(nn.Module):
    """Q layer: cross-attention compressing P→Q.

    Latent queries (Q) attend to dense input tokens (P) as key/value.

    Args:
        C: Embedding dimension.
        num_heads: Number of attention heads.
        C_feat: Conditioning feature dimension for AdaLN.
        q_pool_mode: ``"async"`` for independent per-timestep attention,
            ``"synced"`` for fused attention weights across timesteps.
        mlp_hidden_dims: Hidden dims for feedforward MLP. Defaults to ``[C*4]``.
        ada_n_layers: Layers in AdaLN projection.
        ada_scale_bias_bound: Bound for AdaLN scale/bias (<=0 disables).
        gate_bound: Bound for gate values (<=0 disables).
    """

    def __init__(
        self,
        C: int,
        num_heads: int,
        C_feat: Optional[int] = None,
        q_pool_mode: str = "async",
        mlp_hidden_dims: Optional[List[int]] = None,
        ada_n_layers: int = 1,
        ada_scale_bias_bound: float = -1.0,
        gate_bound: float = -1.0,
    ):
        super().__init__()
        assert C % num_heads == 0
        self.C = C
        self.H = num_heads
        self.D = C // num_heads
        assert self.D % 2 == 0
        self.q_pool_mode = q_pool_mode
        self.ada_scale_bias_bound = ada_scale_bias_bound
        self.gate_bound = gate_bound

        if C_feat is None:
            C_feat = C
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C * 4]

        # AdaLN-Zero: 6 outputs (s, sc, g) x 2 sub-layers (cross-attn, MLP)
        self.ada = _build_ada(C_feat, 6 * C, ada_n_layers)

        # Cross-attention projections
        self.norm_q = nn.LayerNorm(C, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(C, elementwise_affine=False)
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)
        self.out_proj = nn.Linear(C, C)

        # MLP
        self.norm_mlp = nn.LayerNorm(C, elementwise_affine=False)
        layers = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, C))
        self.mlp = nn.Sequential(*layers)

        # Zero-init output projections for stable training
        for proj in [self.out_proj, self.mlp[-1]]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _apply_bounds(self, s, sc, g):
        if self.ada_scale_bias_bound > 0:
            s = torch.tanh(s) * self.ada_scale_bias_bound
            sc = torch.tanh(sc) * self.ada_scale_bias_bound
        if self.gate_bound > 0:
            g = torch.tanh(g) * self.gate_bound
        return s, sc, g

    def forward(
        self,
        x_q: torch.Tensor,
        x_p: torch.Tensor,
        cond_q: torch.Tensor,
        S: Optional[int] = None,
        p_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode: Q queries attend to P key-values.

        Args:
            x_q: (B_eff, Q, C) latent queries. B_eff = B*S.
            x_p: (B_eff, P, C) dense input tokens.
            cond_q: (B_eff, Q, C_feat) conditioning for queries.
            S: Time dimension size. Required for synced mode.
            p_mask: (B_eff, P) optional bool mask for KV. True=valid.

        Returns:
            (B_eff, Q, C) updated latent queries.
        """
        B_eff, Q, C = x_q.shape
        P = x_p.shape[1]

        # AdaLN
        ada_out = self.ada(cond_q)  # (B_eff, Q, 6*C)
        s1, sc1, g1, s2, sc2, g2 = ada_out.chunk(6, dim=-1)
        s1, sc1, g1 = self._apply_bounds(s1, sc1, g1)
        s2, sc2, g2 = self._apply_bounds(s2, sc2, g2)

        # Cross-attention: Q attends to P
        q_normed = self.norm_q(x_q) * (1 + sc1) + s1
        kv_normed = self.norm_kv(x_p)

        q = self.q_proj(q_normed).view(B_eff, Q, self.H, self.D).permute(0, 2, 1, 3)
        k = self.k_proj(kv_normed).view(B_eff, P, self.H, self.D).permute(0, 2, 1, 3)
        v = self.v_proj(kv_normed).view(B_eff, P, self.H, self.D).permute(0, 2, 1, 3)

        # Mask invalid KV (zero out K,V for FlashAttention compat)
        if p_mask is not None:
            _m = p_mask[:, None, :, None]  # (B_eff, 1, P, 1)
            k = k * _m
            v = v * _m

        if self.q_pool_mode == "synced" and S is not None and S > 1:
            # Synced: fuse attention weights across S (time) dimension
            B = B_eff // S
            scale = self.D ** -0.5
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*S, H, Q, P)

            # Reshape → mean over S → expand back
            attn_logits = attn_logits.reshape(B, S, self.H, Q, P)
            attn_logits = attn_logits.mean(dim=1, keepdim=True)  # (B, 1, H, Q, P)
            attn_logits = attn_logits.expand(-1, S, -1, -1, -1)
            attn_logits = attn_logits.reshape(B_eff, self.H, Q, P)

            attn_weights = F.softmax(attn_logits, dim=-1)
            out = torch.matmul(attn_weights, v)  # (B_eff, H, Q, D)
        else:
            # Async: standard per-timestep cross-attention
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B_eff, Q, C)
        out = torch.nan_to_num(out, nan=0.0)
        out = self.out_proj(out)

        x_q = x_q + g1 * out  # gated residual

        # MLP with AdaLN
        h = self.norm_mlp(x_q) * (1 + sc2) + s2
        h = self.mlp(h)
        x_q = x_q + g2 * h

        return x_q


class PerceiverDecodeLayer(nn.Module):
    """D layer: cross-attention expanding Q→P.

    Dense tokens (P) attend to latent tokens (Q) as key/value.
    Always operates in async mode (independent per timestep).

    Args:
        C: Embedding dimension.
        num_heads: Number of attention heads.
        C_feat: Conditioning feature dimension for AdaLN.
        mlp_hidden_dims: Hidden dims for feedforward MLP.
        ada_n_layers: Layers in AdaLN projection.
        ada_scale_bias_bound: Bound for AdaLN scale/bias.
        gate_bound: Bound for gate values.
    """

    def __init__(
        self,
        C: int,
        num_heads: int,
        C_feat: Optional[int] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        ada_n_layers: int = 1,
        ada_scale_bias_bound: float = -1.0,
        gate_bound: float = -1.0,
    ):
        super().__init__()
        assert C % num_heads == 0
        self.C = C
        self.H = num_heads
        self.D = C // num_heads
        assert self.D % 2 == 0
        self.ada_scale_bias_bound = ada_scale_bias_bound
        self.gate_bound = gate_bound

        if C_feat is None:
            C_feat = C
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C * 4]

        # AdaLN-Zero: 6 outputs for (cross-attn, MLP)
        self.ada = _build_ada(C_feat, 6 * C, ada_n_layers)

        # Cross-attention projections (P queries, Q key-values)
        self.norm_q = nn.LayerNorm(C, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(C, elementwise_affine=False)
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)
        self.out_proj = nn.Linear(C, C)

        # MLP
        self.norm_mlp = nn.LayerNorm(C, elementwise_affine=False)
        layers = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, C))
        self.mlp = nn.Sequential(*layers)

        # Zero-init
        for proj in [self.out_proj, self.mlp[-1]]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _apply_bounds(self, s, sc, g):
        if self.ada_scale_bias_bound > 0:
            s = torch.tanh(s) * self.ada_scale_bias_bound
            sc = torch.tanh(sc) * self.ada_scale_bias_bound
        if self.gate_bound > 0:
            g = torch.tanh(g) * self.gate_bound
        return s, sc, g

    def forward(
        self,
        x_p: torch.Tensor,
        x_q: torch.Tensor,
        cond_p: torch.Tensor,
    ) -> torch.Tensor:
        """Decode: P queries attend to Q key-values.

        Args:
            x_p: (B_eff, P, C) dense query tokens. B_eff = B*S.
            x_q: (B_eff, Q, C) latent key-value tokens.
            cond_p: (B_eff, P, C_feat) conditioning for dense queries.

        Returns:
            (B_eff, P, C) updated dense tokens.
        """
        B_eff, P, C = x_p.shape
        Q = x_q.shape[1]

        # AdaLN
        ada_out = self.ada(cond_p)  # (B_eff, P, 6*C)
        s1, sc1, g1, s2, sc2, g2 = ada_out.chunk(6, dim=-1)
        s1, sc1, g1 = self._apply_bounds(s1, sc1, g1)
        s2, sc2, g2 = self._apply_bounds(s2, sc2, g2)

        # Cross-attention: P attends to Q
        q_normed = self.norm_q(x_p) * (1 + sc1) + s1
        kv_normed = self.norm_kv(x_q)

        q = self.q_proj(q_normed).view(B_eff, P, self.H, self.D).permute(0, 2, 1, 3)
        k = self.k_proj(kv_normed).view(B_eff, Q, self.H, self.D).permute(0, 2, 1, 3)
        v = self.v_proj(kv_normed).view(B_eff, Q, self.H, self.D).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B_eff, P, C)
        out = torch.nan_to_num(out, nan=0.0)
        out = self.out_proj(out)

        x_p = x_p + g1 * out

        # MLP
        h = self.norm_mlp(x_p) * (1 + sc2) + s2
        h = self.mlp(h)
        x_p = x_p + g2 * h

        return x_p


# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, S, P, Q, C, H = 2, 8, 512, 16, 256, 8
    C_feat = 256
    B_eff = B * S

    x_p = torch.randn(B_eff, P, C, device=device)
    x_q = torch.randn(B_eff, Q, C, device=device)
    cond_q = torch.randn(B_eff, Q, C_feat, device=device)
    cond_p = torch.randn(B_eff, P, C_feat, device=device)

    # Test encode (async)
    enc = PerceiverEncodeLayer(C, H, C_feat, q_pool_mode="async").to(device)
    x_q_out = enc(x_q, x_p, cond_q)
    print(f"Encode async: {x_q.shape} -> {x_q_out.shape}")
    assert x_q_out.shape == (B_eff, Q, C)

    # Test encode (synced)
    enc_sync = PerceiverEncodeLayer(C, H, C_feat, q_pool_mode="synced").to(device)
    x_q_sync = enc_sync(x_q, x_p, cond_q, S=S)
    print(f"Encode synced: {x_q.shape} -> {x_q_sync.shape}")
    assert x_q_sync.shape == (B_eff, Q, C)

    # Test decode
    dec = PerceiverDecodeLayer(C, H, C_feat).to(device)
    x_p_out = dec(x_p, x_q, cond_p)
    print(f"Decode: {x_p.shape} -> {x_p_out.shape}")
    assert x_p_out.shape == (B_eff, P, C)

    # Backward test
    loss = x_q_out.sum() + x_p_out.sum()
    loss.backward()
    print("Backward pass OK")
    print("All tests passed.")
