"""Split RoPE utilities for V11 Dense LMM.

Provides multi-dimensional spatial RoPE and a split-head attention block
that applies temporal RoPE to the first half of heads and 9D spatial RoPE
to the second half.

The 9D spatial position is [pos_x, pos_y, pos_z, wing1_x, wing1_y, wing1_z,
wing2_x, wing2_y, wing2_z] -- the SE(3) parameterization used throughout V11.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from umm.models.umm_v9.world_models.dit_atten import (
    DiTBlockTimeOnly,
    apply_rope_explicit_time,
    rope_apply,
)


def compute_nd_rope_freqs(
    positions: torch.Tensor,
    head_dim: int,
    n_dims: int,
    device: torch.device,
) -> tuple:
    """Compute multi-dimensional RoPE frequencies (fully vectorized).

    Each head_dim/2 pair is assigned to one of n_dims spatial dimensions
    via round-robin.  Within each dimension's allocation the standard
    theta=10000 frequency ladder is used.

    Args:
        positions: (..., n_dims) float -- spatial position per token.
        head_dim: dimension of each attention head (must be even).
        n_dims: number of spatial dimensions (e.g. 9).
        device: torch device.

    Returns:
        cos, sin: (..., head_dim//2) ready for ``rope_apply``.
    """
    assert head_dim % 2 == 0
    n_pairs = head_dim // 2

    # Round-robin assignment: pair i -> dimension i % n_dims
    dim_assignment = torch.arange(n_pairs, device=device) % n_dims  # (n_pairs,)

    # Count how many pairs each dimension gets
    # Use one_hot + sum for a compile-friendly vectorized count
    one_hot = F.one_hot(dim_assignment, num_classes=n_dims).float()  # (n_pairs, n_dims)
    pairs_per_dim = one_hot.sum(dim=0)  # (n_dims,)

    # Local index of each pair within its dimension's allocation
    # cumsum within each dimension group, done via masked cumsum
    local_idx = (one_hot.cumsum(dim=0) * one_hot).sum(dim=-1) - 1.0  # (n_pairs,)

    # Frequency per pair: 1 / 10000^(2 * local_idx / pairs_for_that_dim)
    pairs_for_pair = pairs_per_dim[dim_assignment]  # (n_pairs,)
    inv_freq = 1.0 / (
        10000.0 ** (2.0 * local_idx / pairs_for_pair.clamp(min=1))
    )  # (n_pairs,)

    # Gather position value for each pair: positions[..., dim_assignment[i]]
    pos_per_pair = positions[..., dim_assignment]  # (..., n_pairs)

    # Angles: position * frequency
    angles = pos_per_pair * inv_freq  # (..., n_pairs)

    return angles.cos(), angles.sin()


def apply_split_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    t_positions: Optional[torch.Tensor],
    spatial_positions: Optional[torch.Tensor],
    n_temporal_heads: int,
    num_regs: int,
    num_external_regs_temporal: int,
    num_external_regs_spatial: int,
    n_spatial_dims: int = 9,
) -> tuple:
    """Apply split RoPE: temporal on first half of heads, spatial on second half.

    Args:
        q, k: (B, H, L, D) query and key tensors.
        t_positions: (B, T) or (T,) temporal positions for temporal RoPE.
            T = L - num_regs - num_external_regs_temporal (noisy tokens only).
        spatial_positions: (B, S, n_spatial_dims) spatial positions for ALL
            entity tokens (ctx + noisy).  S = L - num_regs - num_external_regs_spatial.
        n_temporal_heads: how many heads get temporal RoPE.
        num_regs: internal register count in the DiT block.
        num_external_regs_temporal: external regs + ctx count for temporal RoPE
            (registers that are NOT noisy time-tokens).
        num_external_regs_spatial: external regs count for spatial RoPE
            (only structured registers, NOT ctx -- ctx tokens get spatial RoPE).
        n_spatial_dims: number of spatial dimensions (default 9).

    Returns:
        q, k: (B, H, L, D) with RoPE applied to respective head groups.
    """
    B, H, L, D = q.shape
    n_spatial_heads = H - n_temporal_heads

    # --- Split heads ---
    q_t, q_s = q[:, :n_temporal_heads], q[:, n_temporal_heads:]
    k_t, k_s = k[:, :n_temporal_heads], k[:, n_temporal_heads:]

    # --- Temporal RoPE on first half ---
    if t_positions is not None and n_temporal_heads > 0:
        q_t, k_t = apply_rope_explicit_time(
            q_t, k_t, t_positions,
            num_regs=num_regs,
            num_external_regs=num_external_regs_temporal,
        )

    # --- Spatial RoPE on second half ---
    if spatial_positions is not None and n_spatial_heads > 0:
        start_idx = num_regs + num_external_regs_spatial
        n_spatial_tokens = L - start_idx
        if n_spatial_tokens > 0:
            q_pre, q_spatial = q_s.split([start_idx, n_spatial_tokens], dim=-2)
            k_pre, k_spatial = k_s.split([start_idx, n_spatial_tokens], dim=-2)

            cos, sin = compute_nd_rope_freqs(
                spatial_positions, D, n_spatial_dims, q.device
            )
            # cos, sin: (B, S, D//2) -- expand for head broadcast
            if cos.dim() == 2:
                # (S, D//2) -> (1, 1, S, D//2)
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.dim() == 3:
                # (B, S, D//2) -> (B, 1, S, D//2)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)

            q_spatial = rope_apply(q_spatial, cos, sin)
            k_spatial = rope_apply(k_spatial, cos, sin)

            q_s = torch.cat([q_pre, q_spatial], dim=-2)
            k_s = torch.cat([k_pre, k_spatial], dim=-2)

    q = torch.cat([q_t, q_s], dim=1)
    k = torch.cat([k_t, k_s], dim=1)
    return q, k


def apply_dual_spatial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    n_heads_a: int,
    num_regs: int,
    num_external_regs: int,
    n_spatial_dims: int = 9,
) -> tuple:
    """Apply dual spatial RoPE: two spatial position signals on split heads.

    Used by P-layer: half heads get ctx-mean-9D, half get x-mean-9D.
    Both are spatial (not temporal). Registers/add-tokens are excluded.

    Args:
        q, k: (B, H, L, D) query and key tensors.
        pos_a: (B, P, n_spatial_dims) spatial positions for first head group.
        pos_b: (B, P, n_spatial_dims) spatial positions for second head group.
        n_heads_a: number of heads that get pos_a.
        num_regs: internal register count in the DiT block.
        num_external_regs: external tokens that should NOT receive RoPE
            (e.g., add_tokens count A).
        n_spatial_dims: number of spatial dimensions (default 9).

    Returns:
        q, k: (B, H, L, D) with dual spatial RoPE applied.
    """
    B, H, L, D = q.shape
    n_heads_b = H - n_heads_a

    # Split heads
    q_a, q_b = q[:, :n_heads_a], q[:, n_heads_a:]
    k_a, k_b = k[:, :n_heads_a], k[:, n_heads_a:]

    start_idx = num_regs + num_external_regs
    n_entity_tokens = L - start_idx

    if n_entity_tokens > 0:
        # Apply pos_a to first head group
        if n_heads_a > 0:
            q_a_pre, q_a_ent = q_a.split([start_idx, n_entity_tokens], dim=-2)
            k_a_pre, k_a_ent = k_a.split([start_idx, n_entity_tokens], dim=-2)
            cos_a, sin_a = compute_nd_rope_freqs(pos_a, D, n_spatial_dims, q.device)
            cos_a = cos_a.unsqueeze(1)  # (B, 1, P, D//2)
            sin_a = sin_a.unsqueeze(1)
            q_a_ent = rope_apply(q_a_ent, cos_a, sin_a)
            k_a_ent = rope_apply(k_a_ent, cos_a, sin_a)
            q_a = torch.cat([q_a_pre, q_a_ent], dim=-2)
            k_a = torch.cat([k_a_pre, k_a_ent], dim=-2)

        # Apply pos_b to second head group
        if n_heads_b > 0:
            q_b_pre, q_b_ent = q_b.split([start_idx, n_entity_tokens], dim=-2)
            k_b_pre, k_b_ent = k_b.split([start_idx, n_entity_tokens], dim=-2)
            cos_b, sin_b = compute_nd_rope_freqs(pos_b, D, n_spatial_dims, q.device)
            cos_b = cos_b.unsqueeze(1)  # (B, 1, P, D//2)
            sin_b = sin_b.unsqueeze(1)
            q_b_ent = rope_apply(q_b_ent, cos_b, sin_b)
            k_b_ent = rope_apply(k_b_ent, cos_b, sin_b)
            q_b = torch.cat([q_b_pre, q_b_ent], dim=-2)
            k_b = torch.cat([k_b_pre, k_b_ent], dim=-2)

    q = torch.cat([q_a, q_b], dim=1)
    k = torch.cat([k_a, k_b], dim=1)
    return q, k


class DiTBlockSplitRoPE(DiTBlockTimeOnly):
    """DiT block with split-head RoPE: temporal + spatial.

    Overrides ``_attn_explicit_time`` to apply temporal RoPE on the
    first ``n_temporal_heads`` heads and 9D spatial RoPE on the rest.

    The spatial positions and configuration are set per-call via
    ``set_spatial_rope_params`` before ``forward`` / ``forward_attn_only``.
    This avoids changing the DiT block's forward signature, keeping it
    compatible with the existing layer loop.
    """

    def __init__(self, n_temporal_heads: int, n_spatial_dims: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.n_temporal_heads = n_temporal_heads
        self.n_spatial_heads = self.H - n_temporal_heads
        self.n_spatial_dims = n_spatial_dims

        # Per-call state (set before each forward)
        self._spatial_positions = None  # type: Optional[torch.Tensor]
        self._num_external_regs_spatial = 0
        # Dual spatial mode (P-layer): two spatial positions, no temporal
        self._dual_spatial_mode = False
        self._dual_spatial_pos_a = None  # type: Optional[torch.Tensor]
        self._dual_spatial_pos_b = None  # type: Optional[torch.Tensor]
        self._dual_spatial_num_ext_regs = 0

    def set_spatial_rope_params(
        self,
        spatial_positions: Optional[torch.Tensor],
        num_external_regs_spatial: int,
    ):
        """Set spatial RoPE state for the next forward call.

        Must be called before every ``forward`` or ``forward_attn_only`` when
        spatial RoPE is active.

        Args:
            spatial_positions: (B, S, 9) spatial positions for entity tokens
                (ctx + noisy).  S = total entity tokens that receive spatial RoPE.
            num_external_regs_spatial: number of external register tokens that
                should NOT receive spatial RoPE (only structured registers).
        """
        self._spatial_positions = spatial_positions
        self._num_external_regs_spatial = num_external_regs_spatial
        self._dual_spatial_mode = False

    def set_dual_spatial_rope_params(
        self,
        pos_a: Optional[torch.Tensor],
        pos_b: Optional[torch.Tensor],
        num_external_regs: int,
    ):
        """Set dual spatial RoPE state for P-layer use.

        Both head groups receive spatial RoPE (no temporal). First half of
        heads gets pos_a (ctx-mean-9D), second half gets pos_b (x-mean-9D).

        Args:
            pos_a: (B, P, 9) spatial positions for first head group.
            pos_b: (B, P, 9) spatial positions for second head group.
            num_external_regs: number of external tokens (add_tokens) that
                should NOT receive RoPE.
        """
        self._dual_spatial_mode = True
        self._dual_spatial_pos_a = pos_a
        self._dual_spatial_pos_b = pos_b
        self._dual_spatial_num_ext_regs = num_external_regs
        # Clear single-spatial state
        self._spatial_positions = None

    def _attn_explicit_time(self, x, t, num_external_regs, attn_mask_bool=None):
        """Multi-head self-attention with split RoPE.

        Supports two modes:
        1. Temporal + Spatial (t-layers): temporal RoPE on first heads,
           9D spatial RoPE on second heads.
        2. Dual Spatial (P-layers): two different spatial positions on
           split heads, no temporal RoPE.

        Args:
            x: (B, L_total, C) input (internal registers already prepended).
            t: (B, T) or (T,) or None -- time values for temporal RoPE.
            num_external_regs: external register count for temporal RoPE.
            attn_mask_bool: optional boolean attention mask.

        Returns:
            (B, L_total, C) attention output.
        """
        B, L, C = x.shape
        qkv = self.qkv(x)  # (B, L, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, L, H, D) -> (B, H, L, D)
        q = q.view(B, L, self.H, self.D).permute(0, 2, 1, 3)
        k = k.view(B, L, self.H, self.D).permute(0, 2, 1, 3)
        v = v.view(B, L, self.H, self.D).permute(0, 2, 1, 3)

        # QK-norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE based on mode
        if self._dual_spatial_mode:
            # Dual spatial mode (P-layer): two spatial positions, no temporal
            if self._dual_spatial_pos_a is not None:
                q, k = apply_dual_spatial_rope(
                    q, k,
                    pos_a=self._dual_spatial_pos_a,
                    pos_b=self._dual_spatial_pos_b,
                    n_heads_a=self.n_temporal_heads,
                    num_regs=self.R,
                    num_external_regs=self._dual_spatial_num_ext_regs,
                    n_spatial_dims=self.n_spatial_dims,
                )
        elif t is not None or self._spatial_positions is not None:
            # Standard temporal + spatial mode (t-layers)
            q, k = apply_split_rope(
                q, k,
                t_positions=t,
                spatial_positions=self._spatial_positions,
                n_temporal_heads=self.n_temporal_heads,
                num_regs=self.R,
                num_external_regs_temporal=num_external_regs,
                num_external_regs_spatial=self._num_external_regs_spatial,
                n_spatial_dims=self.n_spatial_dims,
            )

        # SDPA
        if attn_mask_bool is not None:
            attn_mask = attn_mask_bool.bool()
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        return self.proj(out)
