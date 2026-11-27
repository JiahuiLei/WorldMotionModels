"""Hybrid Equivariant-Invariant Transformer for trajectory diffusion forcing.

Architecture: Equivariant encoder → Invariant transformer → Equivariant decoder.

This is Variant B of the E10 SIM(3)-equivariant model. The key insight (from EFEM):
- Early layers use VN (equivariant) to lift input to a rich equivariant representation
- Extract invariant scalar features via vector norms (SO(3)-invariant)
- Run a standard transformer on the scalar features (more expressive, no equivariance constraint)
- Decode back to equivariant output by modulating the equivariant basis from the encoder

Equivariance proof:
    Under rotation R:
    - VN encoder output: h_enc -> h_enc @ R^T (equivariant)
    - Norms: ||h_enc[c]|| -> ||h_enc[c] @ R^T|| = ||h_enc[c]|| (invariant)
    - Standard transformer: operates on invariant scalars -> output is invariant
    - Decoder: scalar_weights * h_basis @ R^T = (scalar_weights * h_basis) @ R^T (equivariant)
    Combined with canonicalization: SIM(3) equivariant. QED.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from umm.models.umm_e10.vn_layers import (
    VNLinear,
    VNLeakyReLU,
    VNLayerNorm,
    VNLinearLeakyReLU,
)

_EPS = 1e-6


class VNEncoder(nn.Module):
    """Equivariant encoder: lifts 3 VN input channels to d_vn VN channels.

    Also produces an equivariant basis for the decoder to use.
    """

    def __init__(self, d_vn: int, input_channels: int = 3):
        super().__init__()
        self.proj = VNLinearLeakyReLU(input_channels, d_vn)
        self.refine = VNLinearLeakyReLU(d_vn, d_vn)

    def forward(self, x_vn: torch.Tensor) -> torch.Tensor:
        """x_vn: [B, S, C_in, 3] -> [B, S, d_vn, 3]"""
        h = self.proj(x_vn)
        h = self.refine(h)
        return h


class VNToScalar(nn.Module):
    """Extract invariant scalar features from VN features.

    Method: per-channel vector norms + pairwise dot products of basis vectors.
    norms: [B, S, d_vn] — ||v_c|| for each channel c
    dots: [B, S, n_basis*(n_basis-1)/2] — v_i · v_j for basis pairs

    All outputs are SO(3)-invariant by construction.
    """

    def __init__(self, d_vn: int, n_basis: int = 8):
        super().__init__()
        self.d_vn = d_vn
        self.n_basis = min(n_basis, d_vn)
        self.n_dots = self.n_basis * (self.n_basis - 1) // 2
        self.scalar_dim = d_vn + self.n_dots

        # Register upper-triangle indices for dot products
        idx_i, idx_j = torch.triu_indices(self.n_basis, self.n_basis, offset=1)
        self.register_buffer("dot_idx_i", idx_i)
        self.register_buffer("dot_idx_j", idx_j)

    def forward(self, h_vn: torch.Tensor) -> torch.Tensor:
        """h_vn: [B, S, d_vn, 3] -> scalars: [B, S, scalar_dim]"""
        # Per-channel norms: [B, S, d_vn]
        # safe_norm: avoids NaN grad at zero vectors (d/dx ||x|| = 0/0 at x=0)
        norms = (h_vn.pow(2).sum(dim=-1) + _EPS).sqrt()  # [B, S, d_vn]

        # Pairwise dot products of first n_basis channels
        basis = h_vn[..., :self.n_basis, :]  # [B, S, n_basis, 3]
        vi = basis[..., self.dot_idx_i, :]  # [B, S, n_dots, 3]
        vj = basis[..., self.dot_idx_j, :]  # [B, S, n_dots, 3]
        dots = (vi * vj).sum(dim=-1)  # [B, S, n_dots]

        return torch.cat([norms, dots], dim=-1)  # [B, S, scalar_dim]


class ScalarMultiHeadAttention(nn.Module):
    """Standard multi-head attention on scalar features."""

    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_head)
            self.k_norm = nn.LayerNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, S, D], mask: [B, S] bool -> [B, S, D]"""
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)  # [B, H, S, Dh]
        K = self.k_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        if mask is not None:
            key_mask = mask[:, None, None, :]  # [B, 1, 1, S]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # Handle all-masked rows
        out = torch.matmul(attn, V)  # [B, H, S, Dh]
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out_proj(out)


class ScalarAdaLNZero(nn.Module):
    """Standard AdaLN-Zero with scalar conditioning (non-equivariant version)."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model),  # gamma, beta, alpha, gamma_ffn, beta_ffn, gate_ffn
        )
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x: [B, S, D], cond: [B, S, d_cond] -> 6 modulation params + normalized x"""
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        params = self.cond_proj(cond)  # [B, S, 6*D]
        gamma, beta, alpha, gamma_ffn, beta_ffn, gate_ffn = params.chunk(6, dim=-1)

        gamma = 1 + gamma
        gamma_ffn = 1 + gamma_ffn

        x_norm = self.norm(x)
        return gamma, beta, alpha, gamma_ffn, beta_ffn, gate_ffn, x_norm


class ScalarTransformerBlock(nn.Module):
    """Standard transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_cond: int,
        ffn_factor: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attn = ScalarMultiHeadAttention(d_model, n_heads, qk_norm=qk_norm)
        self.adaln = ScalarAdaLNZero(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_factor),
            nn.GELU(),
            nn.Linear(d_model * ffn_factor, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, S, D], cond: [B, S, d_cond], mask: [B, S] -> [B, S, D]"""
        gamma, beta, alpha, gamma_ffn, beta_ffn, gate_ffn, x_norm = self.adaln(x, cond)

        attn_in = gamma * x_norm + beta
        attn_out = self.attn(attn_in, mask=mask)
        x = x + alpha * attn_out

        ffn_in = gamma_ffn * self.ffn_norm(x) + beta_ffn
        x = x + gate_ffn * self.ffn(ffn_in)

        return x


class ScalarAxialBlock(nn.Module):
    """Axial scalar transformer: P-attention then T-attention.

    Same idea as VNAxialBlock but for scalar features.
    Memory: O(P^2 + T^2) instead of O((T*P)^2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_cond: int,
        ffn_factor: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        # P-attention
        self.p_attn = ScalarMultiHeadAttention(d_model, n_heads, qk_norm=qk_norm)
        self.p_adaln = ScalarAdaLNZero(d_model, d_cond)

        # T-attention
        self.t_attn = ScalarMultiHeadAttention(d_model, n_heads, qk_norm=qk_norm)
        self.t_adaln = ScalarAdaLNZero(d_model, d_cond)

        # Shared FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_factor),
            nn.GELU(),
            nn.Linear(d_model * ffn_factor, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        T: int,
        P: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, T*P, D], cond: [B, T*P, d_cond], mask: [B, T*P] -> [B, T*P, D]"""
        B, S, D = x.shape

        # --- P-attention ---
        gamma_p, beta_p, alpha_p, _, _, _, x_norm_p = self.p_adaln(x, cond)
        x_p = (gamma_p * x_norm_p + beta_p).reshape(B, T, P, D).reshape(B * T, P, D)
        mask_p = mask.reshape(B, T, P).reshape(B * T, P) if mask is not None else None
        attn_p = self.p_attn(x_p, mask=mask_p).reshape(B, T, P, D).reshape(B, S, D)
        x = x + alpha_p * attn_p

        # --- T-attention ---
        gamma_t, beta_t, alpha_t, gamma_ffn, beta_ffn, gate_ffn, x_norm_t = self.t_adaln(x, cond)
        x_t = (gamma_t * x_norm_t + beta_t).reshape(B, T, P, D).permute(0, 2, 1, 3).reshape(B * P, T, D)
        mask_t = mask.reshape(B, T, P).permute(0, 2, 1).reshape(B * P, T) if mask is not None else None
        attn_t = self.t_attn(x_t, mask=mask_t).reshape(B, P, T, D).permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + alpha_t * attn_t

        # --- FFN ---
        ffn_in = gamma_ffn * self.ffn_norm(x) + beta_ffn
        x = x + gate_ffn * self.ffn(ffn_in)

        return x


class ScalarToVN(nn.Module):
    """Equivariant decoder: convert scalar features back to VN features.

    Method: scalar weights modulate an equivariant basis.
    The basis comes from the encoder's VN output (equivariant).
    Scalar weights come from the invariant transformer (invariant).
    Product: invariant * equivariant = equivariant.

    Output: linear combination of basis vectors weighted by scalar features.
    """

    def __init__(self, scalar_dim: int, d_vn: int, output_channels: int = 3):
        super().__init__()
        self.d_vn = d_vn
        self.output_channels = output_channels
        # Scalar features -> weights for each (output_channel, basis_channel) pair
        self.weight_proj = nn.Sequential(
            nn.Linear(scalar_dim, d_vn * 2),
            nn.GELU(),
            nn.Linear(d_vn * 2, output_channels * d_vn),
        )
        # Initialize near zero for stable training
        nn.init.zeros_(self.weight_proj[-1].weight)
        nn.init.zeros_(self.weight_proj[-1].bias)

    def forward(
        self,
        scalar_features: torch.Tensor,
        basis_vn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scalar_features: [B, S, scalar_dim] — invariant scalar features
            basis_vn: [B, S, d_vn, 3] — equivariant basis from encoder

        Returns:
            [B, S, output_channels, 3] — equivariant output
        """
        B, S, _ = scalar_features.shape
        # Compute weights: [B, S, output_channels * d_vn]
        weights = self.weight_proj(scalar_features)
        weights = weights.reshape(B, S, self.output_channels, self.d_vn)  # [B, S, C_out, d_vn]

        # Linear combination of basis vectors: weights @ basis_vn
        # weights: [B, S, C_out, d_vn], basis_vn: [B, S, d_vn, 3]
        out = torch.einsum("bscd, bsdx -> bscx", weights, basis_vn)  # [B, S, C_out, 3]

        return out


class VNHybridTransformer(nn.Module):
    """Hybrid Equivariant-Invariant Transformer (Variant B).

    Architecture:
        1. VN Encoder: [B, S, 3, 3] → [B, S, d_vn, 3] (equivariant)
        2. VN→Scalar: [B, S, d_vn, 3] → [B, S, scalar_dim] (extract invariants)
        3. Scalar Transformer: [B, S, d_scalar] → [B, S, d_scalar] (standard, no symmetry constraint)
        4. Scalar→VN Decoder: [B, S, d_scalar] × [B, S, d_vn, 3] → [B, S, 3, 3] (equivariant)

    The scalar transformer can use full expressivity (bias, shift, GELU)
    since it operates on invariant features. The equivariance is maintained
    through the encoder/decoder pair.
    """

    def __init__(
        self,
        d_vn: int = 64,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_factor: int = 4,
        input_vn_channels: int = 3,
        output_vn_channels: int = 3,
        num_types: int = 1024,
        num_groups: int = 1024,
        max_noise_steps: int = 100,
        d_cond: int = 256,
        qk_norm: bool = True,
        time_multiplier: float = 10.0,
        n_basis_dots: int = 8,
    ):
        super().__init__()
        self.d_vn = d_vn
        self.d_cond = d_cond
        self.time_multiplier = time_multiplier

        # 1. Equivariant encoder
        self.vn_encoder = VNEncoder(d_vn, input_vn_channels)

        # 2. VN → Scalar projection
        self.vn_to_scalar = VNToScalar(d_vn, n_basis=n_basis_dots)
        scalar_dim = self.vn_to_scalar.scalar_dim

        # 3. Scalar transformer
        # Project scalar features to d_model for transformer
        d_model = d_vn * 3  # Use d_vn*3 as the scalar model dim (similar param count)
        self.d_model = d_model
        self.scalar_input_proj = nn.Linear(scalar_dim, d_model)

        # Scalar conditioning embeddings
        self.noise_emb = Timesteps(d_cond // 2, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.noise_proj = nn.Sequential(
            nn.Linear(d_cond // 2, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.type_emb = nn.Embedding(num_types, d_cond // 2)
        self.group_emb = nn.Embedding(num_groups, d_cond // 2)
        self.cond_merge = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 2, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.blocks = nn.ModuleList([
            ScalarAxialBlock(d_model, n_heads, d_cond, ffn_factor, qk_norm)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        # 4. Scalar → VN decoder
        self.scalar_to_vn = ScalarToVN(d_model, d_vn, output_vn_channels)

        logging.info(
            f"VNHybridTransformer: d_vn={d_vn}, d_model={d_model}, "
            f"n_heads={n_heads}, n_layers={n_layers}, scalar_dim={scalar_dim}, "
            f"d_cond={d_cond}"
        )

    def forward(
        self,
        x_vn: torch.Tensor,
        noise_level: torch.Tensor,
        sem_type: torch.Tensor,
        group_id: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_vn: [B, S, 3, 3] — 3 equivariant input vectors per token
            noise_level: [B, S] int — diffusion timestep per token
            sem_type: [B, P] int — semantic type per part
            group_id: [B, P] int — group ID per part
            mask: [B, S] bool — True = valid token

        Returns:
            [B, S, 3, 3] — 3 equivariant output vectors per token
        """
        B, S, Cin, D3 = x_vn.shape
        assert Cin == 3 and D3 == 3, f"Expected [B, S, 3, 3], got {x_vn.shape}"

        # 1. Equivariant encoder: [B, S, 3, 3] -> [B, S, d_vn, 3]
        h_vn = self.vn_encoder(x_vn)

        # 2. Extract invariant scalars: [B, S, d_vn, 3] -> [B, S, scalar_dim]
        scalars = self.vn_to_scalar(h_vn)

        # 3. Scalar transformer
        h = self.scalar_input_proj(scalars)  # [B, S, d_model]

        # Build scalar conditioning
        noise_flat = noise_level.reshape(-1).float() * self.time_multiplier
        noise_emb = self.noise_emb(noise_flat).to(x_vn.dtype)  # [B*S, d_cond//2]
        noise_emb = self.noise_proj(noise_emb).reshape(B, S, self.d_cond)  # [B, S, d_cond]

        P = sem_type.shape[1]
        T_plus_C = S // P
        type_emb = self.type_emb(sem_type)  # [B, P, d_cond//2]
        group_emb_val = self.group_emb(group_id)  # [B, P, d_cond//2]
        type_emb = type_emb.unsqueeze(1).expand(-1, T_plus_C, -1, -1).reshape(B, S, -1)
        group_emb_val = group_emb_val.unsqueeze(1).expand(-1, T_plus_C, -1, -1).reshape(B, S, -1)

        cond = self.cond_merge(
            torch.cat([noise_emb, type_emb, group_emb_val], dim=-1)
        )  # [B, S, d_cond]

        for block in self.blocks:
            h = block(h, cond, T=T_plus_C, P=P, mask=mask)

        h = self.output_norm(h)  # [B, S, d_model]

        # 4. Equivariant decoder: scalar features + VN basis -> VN output
        out = self.scalar_to_vn(h, h_vn)  # [B, S, 3, 3]

        return out
