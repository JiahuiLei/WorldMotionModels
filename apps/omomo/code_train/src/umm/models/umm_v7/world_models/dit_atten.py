"""Diffusion Transformer (DiT) attention blocks with Rotary Position Embeddings (RoPE).

Provides the core transformer building block for the MotionMachine denoising
backbone. Key components:

- **RoPE helpers**: ``compute_rope_freqs``, ``rope_apply``,
  ``apply_rope_explicit_time`` -- apply rotary positional encoding only to
  time-token positions while leaving register tokens untouched.
- **TimestepEmbedder**: Sinusoidal timestep embedding followed by an MLP,
  used for diffusion-step conditioning.
- **DiTBlockTimeOnly**: A single DiT block with AdaLN-Zero conditioning,
  multi-head self-attention (with optional RoPE), internal register tokens,
  and a gated MLP.

Example:
    >>> block = DiTBlockTimeOnly(C=128, num_heads=4, num_registers=4)
    >>> x = torch.randn(2, 34, 128)
    >>> cond = torch.randn(2, 34, 128)
    >>> t = torch.arange(32).float()
    >>> out = block(x, cond, t, num_external_regs=2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (compat with PyTorch < 2.4)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# -------------------------
# RoPE helpers (Explicit Time)
# -------------------------
def compute_rope_freqs(t: torch.Tensor, head_dim: int, device):
    """
    Computes RoPE frequencies for explicit time values.

    Args:
        t: [T] or [B, T] float tensor of time values.
        head_dim: dimension of each head (must be even).
        device: torch device.

    Returns:
        cos, sin: [..., T, head_dim/2]
    """
    assert head_dim % 2 == 0
    # standard theta=10000
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # outer product: t [..., T] x inv_freq [D/2] -> [..., T, D/2]

    if t.dim() == 1:
        freqs = torch.einsum("t,f->tf", t, inv_freq)
    elif t.dim() == 2:
        freqs = torch.einsum("bt,f->btf", t, inv_freq)
    else:
        raise ValueError(f"Unsupported time tensor shape: {t.shape}")

    return freqs.cos(), freqs.sin()


def rope_apply(x, cos, sin):
    """
    Applies Rotary Positional Embeddings (RoPE) to a tensor.

    Args:
        x (torch.Tensor): Input tensor [..., L, D].
        cos (torch.Tensor): Cosine buffer [..., L, D/2].
        sin (torch.Tensor): Sine buffer [..., L, D/2].

    Returns:
        torch.Tensor: Tensor with RoPE applied, shape [..., L, D].
    """

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return y.flatten(-2)


def apply_rope_explicit_time(q, k, t, num_regs: int, num_external_regs: int = 0):
    """
    q,k: [B, H, L, D]
    L = internal_regs + external_regs + Time
    t: [B, Time] or [Time] time values for the Time part.

    We assume the structure of L is:
    [Internal Registers (num_regs) | External Registers (num_external_regs) | Time Tokens (T)]

    RoPE is ONLY applied to the Time Tokens part.
    """
    # Split off the parts that get RoPE
    start_time_idx = num_regs + num_external_regs

    # If there are no time tokens, nothing to do (edge case)
    if start_time_idx >= q.size(-2):
        return q, k

    q_pre, q_time = q.split([start_time_idx, q.size(-2) - start_time_idx], dim=-2)
    k_pre, k_time = k.split([start_time_idx, k.size(-2) - start_time_idx], dim=-2)

    D = q_time.size(-1)

    # Compute cos/sin for these time values
    cos, sin = compute_rope_freqs(t, D, q.device)

    if t.dim() == 1:
        # [T, D/2] -> [1, 1, T, D/2]
        cos = cos.view(1, 1, -1, D // 2)
        sin = sin.view(1, 1, -1, D // 2)
    else:
        # [B, T, D/2] -> [B, 1, T, D/2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_time = rope_apply(q_time, cos, sin)
    k_time = rope_apply(k_time, cos, sin)

    q = torch.cat([q_pre, q_time], dim=-2)
    k = torch.cat([k_pre, k_time], dim=-2)

    return q, k


# -------------------------
# Time embedding (DiT-like) - Preserved
# -------------------------
class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into a learned vector representation.

    Uses sinusoidal positional encoding followed by a two-layer MLP with SiLU
    activation, following the DiT design.

    Args:
        dim (int): Output embedding dimension.
        hidden_dim (int, optional): MLP hidden dimension. Defaults to ``dim * 4``.

    Example:
        >>> embedder = TimestepEmbedder(dim=256)
        >>> t = torch.tensor([0.0, 50.0, 99.0])
        >>> emb = embedder(t)  # [3, 256]
    """

    def __init__(self, dim, hidden_dim=None):
        """Initialize timestep embedder with sinusoidal base + MLP projection.

        Args:
            dim: Output embedding dimension.
            hidden_dim: Hidden MLP dimension. Defaults to ``dim * 4``.
        """
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    @staticmethod
    def sinusoidal_timestep_embedding(t, dim):
        """Compute sinusoidal positional embedding for diffusion timesteps.

        Args:
            t (torch.Tensor): Timestep values of shape ``[B]`` (float or int).
            dim (int): Target embedding dimension.

        Returns:
            torch.Tensor: Sinusoidal embedding of shape ``[B, dim]``.

        Example:
            >>> emb = TimestepEmbedder.sinusoidal_timestep_embedding(
            ...     torch.tensor([0.0, 50.0]), dim=128
            ... )  # [2, 128]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        """Embed timesteps through sinusoidal encoding + MLP.

        Args:
            t (torch.Tensor): Timestep values of shape ``[B]``.

        Returns:
            torch.Tensor: Learned timestep embedding of shape ``[B, dim]``.

        Example:
            >>> embedder = TimestepEmbedder(dim=256)
            >>> out = embedder(torch.tensor([10.0, 20.0]))  # [2, 256]
        """
        emb = self.sinusoidal_timestep_embedding(t, self.mlp[0].in_features)
        return self.mlp(emb)


# -------------------------
# DiT Block (Refactored)
# -------------------------
class DiTBlockTimeOnly(nn.Module):
    """
    A Diffusion Transformer (DiT) block specialized for time-series data with explicit time conditioning via RoPE.
    Features:
    - Adaptive LayerNorm (AdaLN) zero-initialized.
    - Explicit Time RoPE on time tokens.
    - Internal registers for global context aggregation.
    - Gate mechanism for skip connections.

    Args:
        C (int): Input/Output embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_hidden_dims (List[int]): Hidden dimensions for the MLP.
        num_registers (int): Number of internal register tokens.
        ada_scale_bias_bound (float): Soft clamping bound for scale/bias (<=0 to disable).
        gate_bound (float): Soft clamping bound for gate (<=0 to disable).
        ada_n_layers (int): Number of layers in AdaLN projection.
        C_feat (int): Dimension of the conditioning feature input.
    """

    def __init__(
        self,
        C,
        num_heads,
        mlp_hidden_dims: List[int] = None,
        num_registers=0,
        ada_scale_bias_bound=-1.0,
        gate_bound=-1.0,
        ada_n_layers=1,
        C_feat=None,
        qk_norm=False,
    ):
        """Initialize a DiT block.

        Example:
            >>> block = DiTBlockTimeOnly(C=128, num_heads=4, num_registers=2)
        """

        super().__init__()
        assert C % num_heads == 0
        self.C = C
        self.H = num_heads
        self.D = C // num_heads
        assert self.D % 2 == 0, "RoPE needs even head dim D"
        if C_feat is None:
            C_feat = C
        self.C_feat = C_feat

        self.ada_scale_bias_bound = ada_scale_bias_bound
        self.gate_bound = gate_bound
        if self.ada_scale_bias_bound > 0.0:
            logging.warning(f"Ada scale bias bounded by {self.ada_scale_bias_bound}")
        if self.gate_bound > 0.0:
            logging.warning(f"Gate bounded by {self.gate_bound}")

        self.R = int(num_registers)

        # Default MLP if not specified (standard ratio 4.0 approx)
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [int(C * 4.0)]

        # AdaLN-Zero: produce (shift, scale, gate) for attn and mlp
        # outputs: 6*C -> (s1, sc1, g1, s2, sc2, g2)
        if ada_n_layers == 1:
            self.ada = nn.Linear(C_feat, 6 * C)
            nn.init.zeros_(self.ada.bias)
        else:
            assert ada_n_layers > 1
            _outdims = np.linspace(C_feat, 6 * C, ada_n_layers).astype(int).tolist()
            _outdims = _outdims[1:]
            ada = []
            _c_in = C_feat
            for _c_out in _outdims:
                ada.append(nn.Linear(_c_in, _c_out))
                ada.append(nn.LayerNorm(_c_out))
                ada.append(nn.LeakyReLU(0.2))
                _c_in = _c_out
            self.ada = nn.Sequential(*ada)
            logging.warning(f"Using {ada_n_layers} ada layers: {self.ada    }")

        self.norm1 = nn.LayerNorm(C, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(C, elementwise_affine=False)

        # attention qkv + out
        self.qkv = nn.Linear(C, 3 * C)
        self.proj = nn.Linear(C, C)

        # QK-norm: RMSNorm on Q,K per head to bound attention logits
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.D)  # D = head_dim
            self.k_norm = RMSNorm(self.D)

        # Configurable MLP
        layers = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, C))
        self.mlp = nn.Sequential(*layers)

        # learnable registers (shared across N; expanded at runtime)
        # These are "Internal Registers" for local aggregation
        if self.R > 0:
            self.registers = nn.Parameter(torch.zeros(1, self.R, C))
            nn.init.normal_(self.registers, std=0.02)
        else:
            self.registers = None

    def _attn_explicit_time(self, x, t, num_external_regs, attn_mask_bool=None):
        """Multi-head self-attention with optional explicit-time RoPE.

        Computes QKV projections, applies RoPE only to the time-token
        portion of Q and K, then runs scaled dot-product attention.

        Args:
            x (torch.Tensor): Input of shape ``[B, L_total, C]`` where
                ``L_total = R + num_external_regs + T`` (internal registers
                are already prepended).
            t (torch.Tensor | None): Time values for RoPE. ``[B, T]`` or
                ``[T]``. Pass ``None`` to skip RoPE.
            num_external_regs (int): Number of external register tokens
                prepended before the time tokens in the sequence.
            attn_mask_bool (torch.Tensor | None): Boolean attention mask of
                shape ``[B, H, L_total, L_total]`` or broadcastable.

        Returns:
            torch.Tensor: Attention output projected back, shape
                ``[B, L_total, C]``.

        Example:
            >>> block = DiTBlockTimeOnly(C=128, num_heads=4)
            >>> x = torch.randn(2, 10, 128)
            >>> out = block._attn_explicit_time(x, t=None, num_external_regs=2)
        """
        B, L, C = x.shape
        qkv = self.qkv(x)  # [B,L,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,L,H,D] -> [B,H,L,D]
        q = q.view(B, L, self.H, self.D).permute(0, 2, 1, 3)
        k = k.view(B, L, self.H, self.D).permute(0, 2, 1, 3)
        v = v.view(B, L, self.H, self.D).permute(0, 2, 1, 3)

        # QK-norm: bound attention logits to prevent entropy collapse
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE explicitly
        # We need to compute total non-time tokens to offset
        # Structure: [InternalRegs, ExternalRegs, Time]

        if t is not None:
            q, k = apply_rope_explicit_time(
                q, k, t, num_regs=self.R, num_external_regs=num_external_regs
            )

        # attention using SDPA
        if attn_mask_bool is not None:
            attn_mask = attn_mask_bool.bool()  # ! make sure this is bool
            # row is query, col is key
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        return self.proj(out)

    def _apply_ada(self, x, s, sc):
        """Apply AdaLN-Zero shift and scale: ``x * (1 + sc) + s``."""
        if self.ada_scale_bias_bound > 0.0:
            sc = torch.tanh(sc) * self.ada_scale_bias_bound
        return x * (1 + sc) + s

    def _add_gate(self, x, g, h):
        """Apply gated residual: ``x + g * h``."""
        if self.gate_bound > 0.0:
            g = torch.tanh(g) * self.gate_bound
        return x + g * h

    def _prepare_inputs(self, x, cond):
        """Prepend internal registers and pad conditioning.

        Returns:
            (x_in, cond_in): with internal registers prepended.
        """
        B, L_in, C = x.shape
        if self.R > 0:
            regs = self.registers.expand(B, self.R, C)  # [B, R, C]
            x_in = torch.cat([regs, x], dim=1)  # [B, R+L_in, C]
            if cond.dim() == 3 and cond.shape[1] == L_in:
                reg_cond = torch.zeros(
                    B, self.R, self.C_feat, device=x.device, dtype=x.dtype
                )
                cond_in = torch.cat([reg_cond, cond], dim=1)
            else:
                cond_in = cond
        else:
            x_in = x
            cond_in = cond
        return x_in, cond_in

    def forward_attn_only(self, x, cond, t=None, num_external_regs=0, attn_mask=None):
        """Run only the attention phase (no MLP).

        Used by ParallelPLayer to insert a global bridge between attention and MLP.

        Returns:
            x_in: [B, R+L_in, C] — post-attention state WITH internal registers.
            mlp_ada_params: tuple (s2, sc2, g2) for forward_mlp_only.
        """
        B, L_in, C = x.shape
        assert C == self.C
        x_in, cond_in = self._prepare_inputs(x, cond)

        ada_out = self.ada(cond_in)
        s1, sc1, g1, s2, sc2, g2 = ada_out.chunk(6, dim=-1)

        h = self.norm1(x_in)
        h = self._apply_ada(h, s1, sc1)
        h = self._attn_explicit_time(h, t, num_external_regs, attn_mask_bool=attn_mask)
        x_in = self._add_gate(x_in, g1, h)

        return x_in, (s2, sc2, g2)

    def forward_mlp_only(self, x_in, s2, sc2, g2):
        """Run only the MLP phase on post-attention state.

        Args:
            x_in: [B, R+L_in, C] — from forward_attn_only (with internal regs).
            s2, sc2, g2: MLP ada params from forward_attn_only.

        Returns:
            x_out: [B, L_in, C] — internal registers removed.
        """
        h = self.norm2(x_in)
        h = self._apply_ada(h, s2, sc2)
        h = self.mlp(h)
        x_in = self._add_gate(x_in, g2, h)

        if self.R > 0:
            return x_in[:, self.R:, :]
        return x_in

    def forward(self, x, cond, t=None, num_external_regs=0, attn_mask=None):
        """
        Args:
            x: [B, L_input, C]
               L_input = ExternalRegisters + TimeTokens
            cond: [B, L_input, C] OR whatever broadcasts to x.
            t: [B, T] or [T] time values corresponding to the TimeTokens part of x. Optional.
            attn_mask: Optional mask for attention.
        """
        B, L_in, C = x.shape
        assert C == self.C
        x_in, cond_in = self._prepare_inputs(x, cond)

        # 2. AdaLN-Zero
        ada_out = self.ada(cond_in)  # [..., 6C]
        s1, sc1, g1, s2, sc2, g2 = ada_out.chunk(6, dim=-1)

        # 3. Attention
        h = self.norm1(x_in)
        h = self._apply_ada(h, s1, sc1)
        h = self._attn_explicit_time(h, t, num_external_regs, attn_mask_bool=attn_mask)
        x_in = self._add_gate(x_in, g1, h)

        # 4. MLP
        h = self.norm2(x_in)
        h = self._apply_ada(h, s2, sc2)
        h = self.mlp(h)
        x_in = self._add_gate(x_in, g2, h)

        # 5. Remove Internal Registers
        if self.R > 0:
            x_out = x_in[:, self.R :, :]
        else:
            x_out = x_in

        return x_out


# -------------------------
# Example wiring
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example setup
    B, N, T, C = 2, 64, 32, 128
    H = 4
    num_ext_regs = 2
    num_int_regs = 4

    # Input has ext regs + time
    L_in = num_ext_regs + T
    # Flattened batch layout: B_eff = B * N
    B_eff = B * N
    x = torch.randn(B_eff, L_in, C, device=device)

    # Per-token condition (e.g. diffusion forcing)
    cond = torch.randn(B_eff, L_in, C, device=device)

    # Explicit time
    # If using shared time across N, we can just expand or repeat.
    # Block supports [B, T] or [T].
    # Here we typically want [T] to mean "same pattern for every sequence"
    # Or [B_eff, T] if time varies per sequence.
    t = torch.arange(T, device=device).float() + 100

    # Block
    block = DiTBlockTimeOnly(
        C=C, num_heads=H, mlp_hidden_dims=[C * 2, C * 2], num_registers=num_int_regs
    ).to(device)

    y = block(x, cond, t, num_external_regs=num_ext_regs)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert x.shape == y.shape

    # Test with batch time
    # t_batch must match B_eff if 2D
    t_batch = torch.randn(B_eff, T, device=device)
    y2 = block(x, cond, t_batch, num_external_regs=num_ext_regs)
    print("Batch time test passed")

    # Test with mask
    # SDPA supports boolean or float masks
    # Create a dummy boolean mask
    L_total = num_int_regs + num_ext_regs + T
    mask = torch.ones(B_eff, L_total, L_total, dtype=torch.bool, device=device)
    mask[0, :, -1] = False

    # Standard SDPA mask shape: [B, H, L, L] or [B, 1, L, L]
    mask_BC = mask.unsqueeze(1)  # [B_eff, 1, L, L]
    y3 = block(x, cond, t, num_external_regs=num_ext_regs, attn_mask=mask_BC)
    print("Mask test passed (run w/o error)")

    # Test with t=None (No RoPE)
    y4 = block(x, cond, t=None, num_external_regs=num_ext_regs)
    print(f"No-RoPE (t=None) test passed. Output shape: {y4.shape}")
