"""SO(3)-equivariant Vector Neuron Transformer for trajectory diffusion forcing.

All features are VN tensors [..., C, 3]. Attention scores are SO(3)-invariant
(computed from dot products of equivariant vectors). Values are equivariant.
Combined with input canonicalization, this gives SIM(3) equivariance.

Architecture:
  - VNMultiHeadAttention: invariant scores + equivariant values
  - VNAdaLNZero: scalar conditioning scales equivariant features (no shift)
  - VNTransformerBlock: attention + FFN + AdaLN
  - VNTransformer: stack of blocks with input/output projections
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


class VNMultiHeadAttention(nn.Module):
    """SO(3)-equivariant multi-head attention.

    Q, K, V are VN features [B, seq, C, 3].
    Attention scores = sum of dot products across VN channels and spatial dims
    -> invariant scalars [B, n_heads, seq, seq].
    Output = weighted sum of V -> equivariant.

    Proof:
        Under R: Q -> Q @ R^T, K -> K @ R^T
        score_ij = sum_c sum_d Q[i,c,d] * K[j,c,d]
                 = sum_c (Q[i,c,:] @ R^T) . (K[j,c,:] @ R^T)
                 = sum_c Q[i,c,:] . K[j,c,:]  (R^T R = I)
        Scores invariant. V -> V @ R^T, attn @ V -> (attn @ V) @ R^T. QED.
    """

    def __init__(self, d_vn: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        assert d_vn % n_heads == 0, f"d_vn={d_vn} must be divisible by n_heads={n_heads}"
        self.d_vn = d_vn
        self.n_heads = n_heads
        self.d_head = d_vn // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head * 3)  # scale by sqrt(d_head * 3)

        self.q_proj = VNLinear(d_vn, d_vn)
        self.k_proj = VNLinear(d_vn, d_vn)
        self.v_proj = VNLinear(d_vn, d_vn)
        self.out_proj = VNLinear(d_vn, d_vn)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = VNLayerNorm(self.d_head)
            self.k_norm = VNLayerNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, C, 3] VN features
            mask: [B, S] bool, True = valid token
        Returns:
            [B, S, C, 3] VN features
        """
        B, S, C, D3 = x.shape  # D3 = 3
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).reshape(B, S, H, Dh, 3)  # [B, S, H, Dh, 3]
        K = self.k_proj(x).reshape(B, S, H, Dh, 3)
        V = self.v_proj(x).reshape(B, S, H, Dh, 3)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Transpose to [B, H, S, Dh, 3] for attention
        Q = Q.permute(0, 2, 1, 3, 4)
        K = K.permute(0, 2, 1, 3, 4)
        V = V.permute(0, 2, 1, 3, 4)

        # Attention scores: sum over Dh and 3 dimensions -> [B, H, S, S]
        scores = torch.einsum("bhscd, bhtcd -> bhst", Q, K) * self.scale

        # Apply mask: set padded positions to -inf
        if mask is not None:
            # mask: [B, S] -> [B, 1, 1, S] for key masking
            key_mask = mask[:, None, None, :]  # [B, 1, 1, S]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Handle all-masked rows: softmax(all -inf) = NaN -> replace with 0
        attn = attn.nan_to_num(0.0)

        # Weighted sum of values: [B, H, S, Dh, 3]
        out = torch.einsum("bhst, bhtcd -> bhscd", attn, V)

        # Back to [B, S, C, 3]
        out = out.permute(0, 2, 1, 3, 4).reshape(B, S, C, 3)
        return self.out_proj(out)


class VNAdaLNZero(nn.Module):
    """Equivariant Adaptive LayerNorm-Zero with scalar conditioning.

    Standard AdaLN has scale (gamma) and shift (beta). For equivariant features,
    shift is FORBIDDEN (no equivariant direction available from scalar conditioning).
    Only scalar multiplication is allowed.

    AdaLN-Zero initializes scale near 1 and gate near 0, so the residual
    connection dominates initially (stable training).

    Conditioning: scalar embedding (noise level, timestep) -> scales for
    each VN channel. Scalar * equivariant = equivariant.
    """

    def __init__(self, d_vn: int, d_cond: int):
        super().__init__()
        self.norm = VNLayerNorm(d_vn)
        # Predict 3 sets of scalars: gamma (pre-attention), alpha (gate), gamma_ffn
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 3 * d_vn),
        )
        # Initialize near identity (gamma=1, alpha=0, gamma_ffn=1)
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: [B, S, C, 3] VN features
            cond: [B, S, d_cond] or [B, d_cond] scalar conditioning

        Returns:
            gamma, alpha, gamma_ffn: each [B, S, C, 1] or broadcastable
            x_norm: [B, S, C, 3] normalized x
        """
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # [B, 1, d_cond]

        params = self.cond_proj(cond)  # [B, S, 3*C]
        gamma, alpha, gamma_ffn = params.chunk(3, dim=-1)  # each [B, S, C]

        # Shift around 1 for scale, 0 for gate
        gamma = (1 + gamma).unsqueeze(-1)  # [B, S, C, 1]
        alpha = alpha.unsqueeze(-1)  # [B, S, C, 1]
        gamma_ffn = (1 + gamma_ffn).unsqueeze(-1)  # [B, S, C, 1]

        x_norm = self.norm(x)

        return gamma, alpha, gamma_ffn, x_norm


class VNFFN(nn.Module):
    """Equivariant Feed-Forward Network.

    VNLinear -> VNLeakyReLU -> VNLinear

    All operations preserve SO(3) equivariance.
    """

    def __init__(self, d_vn: int, factor: int = 4, negative_slope: float = 0.2):
        super().__init__()
        d_inner = d_vn * factor
        self.net = nn.Sequential(
            VNLinear(d_vn, d_inner),
            VNLeakyReLU(d_inner, negative_slope=negative_slope),
            VNLinear(d_inner, d_vn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C, 3] -> [..., C, 3]"""
        return self.net(x)


class VNTransformerBlock(nn.Module):
    """Single equivariant transformer block with AdaLN-Zero conditioning.

    Architecture:
        x_norm = AdaLN_norm(x, cond)
        x = x + alpha * Attention(gamma * x_norm)
        x = x + VNFFN(gamma_ffn * VNNorm(x))
    """

    def __init__(
        self,
        d_vn: int,
        n_heads: int,
        d_cond: int,
        ffn_factor: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attn = VNMultiHeadAttention(d_vn, n_heads, qk_norm=qk_norm)
        self.adaln = VNAdaLNZero(d_vn, d_cond)
        self.ffn = VNFFN(d_vn, factor=ffn_factor)
        self.ffn_norm = VNLayerNorm(d_vn)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, C, 3] VN features
            cond: [B, S, d_cond] scalar conditioning
            mask: [B, S] bool, True = valid token
        Returns:
            [B, S, C, 3]
        """
        gamma, alpha, gamma_ffn, x_norm = self.adaln(x, cond)

        # Attention with pre-norm and gate
        attn_out = self.attn(gamma * x_norm, mask=mask)
        x = x + alpha * attn_out

        # FFN with pre-norm
        ffn_in = gamma_ffn * self.ffn_norm(x)
        x = x + self.ffn(ffn_in)

        return x


class VNAxialBlock(nn.Module):
    """Axial equivariant transformer: P-attention then T-attention.

    Instead of full (T*P) x (T*P) attention, uses:
    1. P-attention: attend across parts at each timestep. [B*T, P, C, 3]
    2. T-attention: attend across timesteps for each part. [B*P, T, C, 3]

    Both are equivariant (VN attention on different axes).
    Memory: O(P^2 + T^2) instead of O((T*P)^2).
    """

    def __init__(
        self,
        d_vn: int,
        n_heads: int,
        d_cond: int,
        ffn_factor: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        # P-attention (across parts)
        self.p_attn = VNMultiHeadAttention(d_vn, n_heads, qk_norm=qk_norm)
        self.p_adaln = VNAdaLNZero(d_vn, d_cond)

        # T-attention (across time)
        self.t_attn = VNMultiHeadAttention(d_vn, n_heads, qk_norm=qk_norm)
        self.t_adaln = VNAdaLNZero(d_vn, d_cond)

        # Shared FFN
        self.ffn = VNFFN(d_vn, factor=ffn_factor)
        self.ffn_norm = VNLayerNorm(d_vn)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        T: int,
        P: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T*P, C, 3] VN features
            cond: [B, T*P, d_cond] scalar conditioning
            T: number of timesteps (C+T_trace)
            P: number of parts
            mask: [B, T*P] bool, True = valid token
        Returns:
            [B, T*P, C, 3]
        """
        B, S, C, D3 = x.shape

        # --- P-attention: attend across parts at each timestep ---
        gamma_p, alpha_p, _, x_norm_p = self.p_adaln(x, cond)
        # Reshape: [B, T*P, C, 3] -> [B, T, P, C, 3] -> [B*T, P, C, 3]
        x_p = (gamma_p * x_norm_p).reshape(B, T, P, C, D3).reshape(B * T, P, C, D3)
        mask_p = mask.reshape(B, T, P).reshape(B * T, P) if mask is not None else None
        attn_p = self.p_attn(x_p, mask=mask_p)  # [B*T, P, C, 3]
        attn_p = attn_p.reshape(B, T, P, C, D3).reshape(B, S, C, D3)
        x = x + alpha_p * attn_p

        # --- T-attention: attend across timesteps for each part ---
        gamma_t, alpha_t, gamma_ffn, x_norm_t = self.t_adaln(x, cond)
        # Reshape: [B, T*P, C, 3] -> [B, T, P, C, 3] -> [B*P, T, C, 3]
        x_t = (gamma_t * x_norm_t).reshape(B, T, P, C, D3).permute(0, 2, 1, 3, 4).reshape(B * P, T, C, D3)
        mask_t = mask.reshape(B, T, P).permute(0, 2, 1).reshape(B * P, T) if mask is not None else None
        attn_t = self.t_attn(x_t, mask=mask_t)  # [B*P, T, C, 3]
        attn_t = attn_t.reshape(B, P, T, C, D3).permute(0, 2, 1, 3, 4).reshape(B, S, C, D3)
        x = x + alpha_t * attn_t

        # --- FFN ---
        ffn_in = gamma_ffn * self.ffn_norm(x)
        x = x + self.ffn(ffn_in)

        return x


class VNTransformer(nn.Module):
    """Full SO(3)-equivariant Vector Neuron Transformer.

    Input: 3 equivariant vectors per token [t_canon, R_col0, R_col1]
    + scalar conditioning (noise level per token, semantic type per token).

    Output: 3 equivariant vectors per token (predicted noise/x0 components).

    Combined with canonicalization (centroid + scale), gives SIM(3) equivariance.
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
    ):
        super().__init__()
        self.d_vn = d_vn
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_cond = d_cond
        self.time_multiplier = time_multiplier

        # Input: lift 3 VN channels to d_vn VN channels
        self.input_proj = VNLinear(input_vn_channels, d_vn)

        # Scalar conditioning embeddings
        # Noise level embedding (per-token diffusion timestep)
        self.noise_emb = Timesteps(d_cond // 2, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.noise_proj = nn.Sequential(
            nn.Linear(d_cond // 2, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Semantic type and group embeddings
        self.type_emb = nn.Embedding(num_types, d_cond // 2)
        self.group_emb = nn.Embedding(num_groups, d_cond // 2)
        self.cond_merge = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 2, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Transformer blocks (axial attention: P-attn + T-attn per block)
        self.blocks = nn.ModuleList([
            VNAxialBlock(d_vn, n_heads, d_cond, ffn_factor, qk_norm)
            for _ in range(n_layers)
        ])

        # Output: project d_vn VN channels back to output_vn_channels
        self.output_norm = VNLayerNorm(d_vn)
        self.output_proj = VNLinear(d_vn, output_vn_channels)

        # Initialize output projection to near-zero for stable training
        nn.init.zeros_(self.output_proj.weight)

        logging.info(
            f"VNTransformer: d_vn={d_vn}, n_heads={n_heads}, n_layers={n_layers}, "
            f"ffn_factor={ffn_factor}, d_cond={d_cond}, "
            f"effective_scalar_dim={d_vn * 3}"
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
                   (canonicalized translation, rot col 0, rot col 1)
            noise_level: [B, S] int — diffusion timestep per token
            sem_type: [B, P] int — semantic type per part (broadcast over time)
            group_id: [B, P] int — group ID per part
            mask: [B, S] bool — True = valid token

        Returns:
            [B, S, 3, 3] — 3 equivariant output vectors per token
        """
        B, S, Cin, D3 = x_vn.shape
        assert Cin == 3 and D3 == 3, f"Expected [B, S, 3, 3], got {x_vn.shape}"

        # 1. Project input VN features: [B, S, 3, 3] -> [B, S, d_vn, 3]
        h = self.input_proj(x_vn)

        # 2. Build scalar conditioning: [B, S, d_cond]
        # Noise level embedding
        noise_flat = noise_level.reshape(-1).float() * self.time_multiplier
        noise_emb = self.noise_emb(noise_flat).to(x_vn.dtype)  # [B*S, d_cond//2], cast to model dtype
        noise_emb = self.noise_proj(noise_emb).reshape(B, S, self.d_cond)  # [B, S, d_cond]

        # Type and group embeddings (per-part, broadcast over time)
        P = sem_type.shape[1]
        T_plus_C = S // P  # number of time frames (context + trace)
        type_emb = self.type_emb(sem_type)  # [B, P, d_cond//2]
        group_emb_val = self.group_emb(group_id)  # [B, P, d_cond//2]
        # Broadcast over time: [B, P, d] -> [B, T+C, P, d] -> [B, S, d]
        type_emb = type_emb.unsqueeze(1).expand(-1, T_plus_C, -1, -1).reshape(B, S, -1)
        group_emb_val = group_emb_val.unsqueeze(1).expand(-1, T_plus_C, -1, -1).reshape(B, S, -1)

        # Merge all scalar conditioning
        cond = self.cond_merge(
            torch.cat([noise_emb, type_emb, group_emb_val], dim=-1)
        )  # [B, S, d_cond]

        # 3. Transformer blocks (axial attention needs T and P dims)
        for block in self.blocks:
            h = block(h, cond, T=T_plus_C, P=P, mask=mask)

        # 4. Output projection: [B, S, d_vn, 3] -> [B, S, 3, 3]
        h = self.output_norm(h)
        out = self.output_proj(h)

        return out
