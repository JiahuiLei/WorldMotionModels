"""Vector Neuron (VNN) equivariant layers for SO(3)/SIM(3)-equivariant networks.

Based on: "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
(Deng et al., ICCV 2021) and "EFEM" (Lei et al., CVPR 2023).

Feature convention: all VN features have shape [..., C, 3] where:
  - C = number of vector neuron channels (learned feature mixing axis)
  - 3 = spatial axis (rotates rigidly with SO(3) action)

Under rotation R in SO(3): f -> f @ R^T  (each row-vector rotated by R).
Equivalently: f[..., c, :] -> R @ f[..., c, :] for each channel c.

All layers are designed so that:
  layer(x @ R^T) = layer(x) @ R^T  for any R in SO(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


class VNLinear(nn.Module):
    """Equivariant linear layer: weight matrix acts on channel axis only.

    Maps [..., C_in, 3] -> [..., C_out, 3].
    NO BIAS (bias would break equivariance — it has no spatial direction).

    Equivariance proof:
        VNLinear(x @ R^T) = W @ (x @ R^T) = (W @ x) @ R^T = VNLinear(x) @ R^T
        because W acts on C-axis, R^T on 3-axis — they commute.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C_in, 3] -> [..., C_out, 3]"""
        # Einstein: ...ij, oi -> ...oj  (i=C_in, o=C_out, j=3)
        return torch.einsum("...ij, oi -> ...oj", x, self.weight)


class VNLeakyReLU(nn.Module):
    """Equivariant leaky ReLU via direction-based reflection.

    For each channel c, learns a reference direction d_c (itself equivariant,
    computed as a linear combination of input vectors). Projects vectors onto
    the positive half-space of d_c; vectors in the negative half-space are
    reflected with a leaky slope.

    Equivariance: d_c rotates with input (it's a VNLinear output), so the
    dot product <v_c, d_c> is invariant, and the reflection plane rotates
    with the input. Full equivariance.
    """

    def __init__(self, c_in: int, negative_slope: float = 0.2, share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        c_dir = 1 if share_nonlinearity else c_in
        self.dir_linear = VNLinear(c_in, c_dir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C, 3] -> [..., C, 3]"""
        d = self.dir_linear(x)  # [..., C_dir, 3]
        # Dot product: project each channel's vector onto its reference direction
        # Need to broadcast if share_nonlinearity (C_dir=1)
        dot = (x * d).sum(dim=-1, keepdim=True)  # [..., C, 1]
        d_norm_sq = (d * d).sum(dim=-1, keepdim=True).clamp(min=_EPS)  # [..., C_dir, 1]

        # Projection of x onto d: (dot / ||d||^2) * d
        proj = (dot / d_norm_sq) * d  # [..., C, 3]

        # Positive half-space: keep x. Negative: reflect with leaky slope.
        mask = (dot >= 0).float()  # [..., C, 1]
        s = self.negative_slope
        # out = mask * x + (1-mask) * (x - (1-s) * proj)
        #     = x - (1-mask) * (1-s) * proj
        out = x - (1 - mask) * (1 - s) * proj
        return out


class VNLayerNorm(nn.Module):
    """Equivariant layer normalization: normalize vector NORMS, preserve directions.

    Standard LayerNorm on xyz components would break equivariance. Instead:
    1. Compute per-channel norms: ||v_c|| -> scalar [C]
    2. Apply standard LayerNorm to the norms -> normalized norms
    3. Rescale vectors: v_c * (norm_normalized / ||v_c||)

    Equivariance: ||R @ v|| = ||v|| (norms are invariant), so the scaling
    factor is unchanged under rotation. The direction v/||v|| -> R @ v/||v||.
    """

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C, 3] -> [..., C, 3]"""
        # safe_norm: (x^2.sum + eps).sqrt() avoids NaN grad at zero vectors
        # d/dx ||x|| = x/||x|| = 0/0 = NaN when x=0; adding eps inside sqrt fixes this
        norms = (x.pow(2).sum(dim=-1) + _EPS).sqrt()  # [..., C]
        norms_normalized = self.norm(norms)  # [..., C]

        # Rescale: preserve direction, replace magnitude
        scale = norms_normalized / norms  # [..., C] — norms already has eps, no clamp needed
        return x * scale.unsqueeze(-1)  # [..., C, 3]


class VNLinearLeakyReLU(nn.Module):
    """Fused VNLinear + VNLeakyReLU."""

    def __init__(self, c_in: int, c_out: int, negative_slope: float = 0.2):
        super().__init__()
        self.linear = VNLinear(c_in, c_out)
        self.act = VNLeakyReLU(c_out, negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class VNMeanPool(nn.Module):
    """Equivariant mean pooling over a specified dimension.

    mean(R @ v_i) = R @ mean(v_i) — trivially equivariant.
    """

    def __init__(self, pool_dim: int):
        self.pool_dim = pool_dim
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Expand mask to [..., C, 3]
            x = x * mask.unsqueeze(-1).unsqueeze(-1).float()
            count = mask.sum(dim=self.pool_dim, keepdim=True).clamp(min=1).float()
            return x.sum(dim=self.pool_dim, keepdim=True) / count.unsqueeze(-1).unsqueeze(-1)
        return x.mean(dim=self.pool_dim, keepdim=True)
