"""Free (unconstrained) scalar axial transformer for E11 sandwich middle.

These layers operate on invariant scalar features extracted by the
EquivariantEncoder. No equivariance constraint — full AdaLN with shift+scale,
standard GELU, bias in all projections.

This is the key advantage over E10: the middle layers have full expressiveness
while equivariance is maintained through the boundary layers.

Copied from E10 vn_hybrid_transformer.py ScalarAxialBlock with minor cleanup.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps


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
    """Standard AdaLN-Zero with full shift+scale (non-equivariant)."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 6 params: gamma, beta, alpha, gamma_ffn, beta_ffn, gate_ffn
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model),
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


class ScalarAxialBlock(nn.Module):
    """Axial scalar transformer: P-attention then T-attention.

    Memory: O(P^2 + T^2) instead of O((T*P)^2).
    Full AdaLN with shift+scale — no equivariance constraint.
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


class FreeTransformer(nn.Module):
    """Stack of ScalarAxialBlocks with input/output projection and conditioning.

    This is the "free" middle of the E11 sandwich — no equivariance constraint.
    Operates purely on invariant scalar features.
    """

    def __init__(
        self,
        invariant_dim: int,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 6,
        d_cond: int = 256,
        ffn_factor: int = 4,
        qk_norm: bool = True,
        num_types: int = 1024,
        num_groups: int = 1024,
        time_multiplier: float = 10.0,
        multi_layer_skip: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.n_layers = n_layers
        self.time_multiplier = time_multiplier
        self.multi_layer_skip = multi_layer_skip

        # Input projection: invariant features -> d_model
        self.input_proj = nn.Linear(invariant_dim, d_model)

        # Conditioning embeddings
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

        # Axial transformer blocks
        self.blocks = nn.ModuleList([
            ScalarAxialBlock(d_model, n_heads, d_cond, ffn_factor, qk_norm)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        # [R2-A] Multi-layer skip: input-to-output residual connection
        # V9 uses concat(all_layers) → FC but that requires storing all activations.
        # Memory-efficient alternative: blend input projection with final output.
        # This provides 2-scale features (early coarse + late refined) → smoother.
        # h_out = output_norm(h_final) + alpha * input_proj_norm(h_input)
        if multi_layer_skip:
            self.input_skip_norm = nn.LayerNorm(d_model)
            # Learnable blend weight, starts at 0 (pure final output at init)
            self.input_skip_alpha = nn.Parameter(torch.zeros(1))

    @torch.compiler.disable
    def _build_conditioning(
        self,
        noise_level: torch.Tensor,
        sem_type: torch.Tensor,
        group_id: torch.Tensor,
        B: int,
        S: int,
        T: int,
        P: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build per-token conditioning from noise, type, and group embeddings.

        Args:
            noise_level: [B, S] int diffusion timestep per token
            sem_type: [B, P] int semantic type per part
            group_id: [B, P] int group ID per part

        Returns:
            cond: [B, S, d_cond]
        """
        noise_flat = noise_level.reshape(-1).float() * self.time_multiplier
        noise_emb = self.noise_emb(noise_flat).to(dtype)  # [B*S, d_cond//2]
        noise_emb = self.noise_proj(noise_emb).reshape(B, S, self.d_cond)  # [B, S, d_cond]

        type_emb = self.type_emb(sem_type)  # [B, P, d_cond//2]
        group_emb_val = self.group_emb(group_id)  # [B, P, d_cond//2]

        # Broadcast type/group over T
        type_emb = type_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, S, -1)  # [B, S, d_cond//2]
        group_emb_val = group_emb_val.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, S, -1)

        cond = self.cond_merge(
            torch.cat([noise_emb, type_emb, group_emb_val], dim=-1)
        )  # [B, S, d_cond]
        return cond

    def forward(
        self,
        invariant: torch.Tensor,
        noise_level: torch.Tensor,
        sem_type: torch.Tensor,
        group_id: torch.Tensor,
        T: int,
        P: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run free transformer on invariant scalar features.

        Args:
            invariant: [B, S, invariant_dim] from EquivariantEncoder
            noise_level: [B, S] diffusion timesteps
            sem_type: [B, P] semantic type IDs
            group_id: [B, P] group IDs
            T: number of timesteps (C+T)
            P: number of parts
            mask: [B, S] bool, True = valid

        Returns:
            [B, S, d_model] scalar features for EquivariantDecoder
        """
        B, S, _ = invariant.shape

        # Project to d_model
        h = self.input_proj(invariant)  # [B, S, d_model]

        # Build conditioning
        cond = self._build_conditioning(
            noise_level, sem_type, group_id, B, S, T, P, invariant.dtype
        )

        if self.multi_layer_skip:
            # [R2-A] Input-to-output skip: blend early (input_proj) + late (final block)
            h0 = h  # Save input projection output
            for block in self.blocks:
                h = block(h, cond, T=T, P=P, mask=mask)
            # Blend: final output + learned fraction of input
            # alpha starts at 0 → pure final output initially, learns to blend
            h = self.output_norm(h) + self.input_skip_alpha * self.input_skip_norm(h0)
        else:
            # Original: only use final layer output
            for block in self.blocks:
                h = block(h, cond, T=T, P=P, mask=mask)
            h = self.output_norm(h)  # [B, S, d_model]

        return h
