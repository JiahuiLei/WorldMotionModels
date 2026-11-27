"""6D continuous rotation representation (Zhou et al., CVPR 2019).

Provides functions to convert between 3x3 rotation matrices, 6D continuous
representations, and SE(3) 4x4 matrices.  The 9D pose = [3D position, 6D rotation].

All functions are differentiable and torch.compile-safe (no dynamic shapes,
no data-dependent control flow).

Example::

    from umm.models.umm_v11b.rotation_utils import se3_to_9d, ninep_to_se3
    x9 = se3_to_9d(T_4x4)      # (*, 4, 4) -> (*, 9)
    T_hat = ninep_to_se3(x9)    # (*, 9) -> (*, 4, 4)
"""

import torch


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        R: Rotation matrices, shape ``(*, 3, 3)``.

    Returns:
        6D rotation, shape ``(*, 6)``.
    """
    # Concatenate first two columns (column-major, NOT reshape which is row-major)
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # (*, 6)


def sixd_to_rotmat(d: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to rotation matrix via Gram-Schmidt.

    Numerically safe: uses eps in normalization to avoid NaN gradients
    at zero vectors (which shouldn't happen with trained networks but
    guards against initialization edge cases).

    Args:
        d: 6D rotation vectors, shape ``(*, 6)``.

    Returns:
        Rotation matrices, shape ``(*, 3, 3)``.
    """
    _EPS = 1e-6
    a1 = d[..., :3]  # (*, 3)
    a2 = d[..., 3:]  # (*, 3)

    # Normalize first column
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + _EPS)

    # Gram-Schmidt: remove b1 component from a2, then normalize
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + _EPS)

    # Third column via cross product
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (*, 3, 3)


def se3_to_9d(T: torch.Tensor) -> torch.Tensor:
    """Convert SE(3) matrix to 9D representation [position, 6D rotation].

    Args:
        T: SE(3) matrices, shape ``(*, 4, 4)``.

    Returns:
        9D pose vectors, shape ``(*, 9)`` = [pos(3), rot6d(6)].
    """
    pos = T[..., :3, 3]  # (*, 3)
    rot = T[..., :3, :3]  # (*, 3, 3)
    rot6d = rotmat_to_6d(rot)  # (*, 6)
    return torch.cat([pos, rot6d], dim=-1)  # (*, 9)


def ninep_to_se3(x: torch.Tensor) -> torch.Tensor:
    """Convert 9D representation back to SE(3) matrix.

    Args:
        x: 9D pose vectors, shape ``(*, 9)`` = [pos(3), rot6d(6)].

    Returns:
        SE(3) matrices, shape ``(*, 4, 4)``.
    """
    pos = x[..., :3]  # (*, 3)
    rot6d = x[..., 3:]  # (*, 6)
    R = sixd_to_rotmat(rot6d)  # (*, 3, 3)

    # Build 4x4 matrix
    batch_shape = x.shape[:-1]
    T = torch.zeros(*batch_shape, 4, 4, device=x.device, dtype=x.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def se3_to_9d_wing(T: torch.Tensor, wing_len: float = 0.2) -> torch.Tensor:
    """Convert SE(3) to 9D wing representation [pos, pos+R_col0*wl, pos+R_col1*wl].

    All 9 channels live in position space, avoiding the scale imbalance between
    unit-vector rotation (~1.0) and canonicalized position (~0.5).

    Args:
        T: SE(3) matrices, shape ``(*, 4, 4)``.
        wing_len: Wing length scaling factor.

    Returns:
        9D wing vectors, shape ``(*, 9)`` = [pos(3), wing0(3), wing1(3)].
    """
    pos = T[..., :3, 3]       # (*, 3)
    R = T[..., :3, :3]        # (*, 3, 3)
    wing0 = pos + R[..., :, 0] * wing_len  # (*, 3)
    wing1 = pos + R[..., :, 1] * wing_len  # (*, 3)
    return torch.cat([pos, wing0, wing1], dim=-1)  # (*, 9)


def ninep_wing_to_se3(x: torch.Tensor, wing_len: float = 0.2) -> torch.Tensor:
    """Convert 9D wing representation back to SE(3) matrix.

    Recovers rotation columns from wing offsets, applies Gram-Schmidt
    for orthonormality, then builds the SE(3) matrix.

    Args:
        x: 9D wing vectors, shape ``(*, 9)`` = [pos(3), wing0(3), wing1(3)].
        wing_len: Wing length scaling factor (must match encoding).

    Returns:
        SE(3) matrices, shape ``(*, 4, 4)``.
    """
    pos = x[..., :3]     # (*, 3)
    wing0 = x[..., 3:6]  # (*, 3)
    wing1 = x[..., 6:9]  # (*, 3)
    col0 = (wing0 - pos) / wing_len  # (*, 3)
    col1 = (wing1 - pos) / wing_len  # (*, 3)
    rot6d = torch.cat([col0, col1], dim=-1)  # (*, 6)
    R = sixd_to_rotmat(rot6d)  # (*, 3, 3)

    batch_shape = x.shape[:-1]
    T = torch.zeros(*batch_shape, 4, 4, device=x.device, dtype=x.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T
