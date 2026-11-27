"""6D continuous rotation representation (Zhou et al., CVPR 2019).

Provides functions to convert between 3x3 rotation matrices, 6D continuous
representations, and SE(3) 4x4 matrices.  The 9D pose = [3D position, 6D rotation].

All functions are differentiable and torch.compile-safe (no dynamic shapes,
no data-dependent control flow).

Example::

    from umm.models.umm_v10.rotation_utils import se3_to_9d, ninep_to_se3
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
