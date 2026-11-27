"""Torch + autograd port of ``prepare_data/omomo/build_q.py::se3_to_axis_angle``.

Decodes per-joint world-frame SE(3) ``Q`` back into SMPL-X ``(root_trans,
root_orient_aa, body_pose_aa)``. Used by w1.6.2 full-body penetration loss
to feed predicted joint SE(3) into a differentiable SMPL-X FK that yields
mesh verts.

**Differences from the numpy version:**
  * Inputs / outputs are batched torch tensors (B, T, ...).
  * Rotation matrix → axis-angle uses ``pytorch3d.transforms.matrix_to_axis_angle``
    (autograd-supported).
  * Pelvis rest offset is supplied by the caller — typically computed once per
    subject via ``smplx_pelvis_rest_offset_torch`` in ``smplx_joints.py``.

**Math** (mirrors `build_q.py:407-455` docstring):

    R_local[k]  = R_global[parent(k)]^T @ R_global[k]   for k >= 1
    R_local[0]  = R_global[0]                            (root absolute)
    root_trans  = Q[:, 0, :3, 3] - J_rest_shaped[0]      (pelvis offset)
    aa[k]       = matrix_to_axis_angle(R_local[k])

The kinematic-tree parent indices are the same SMPL-X body hierarchy used
elsewhere in the codebase (``SMPLX_BODY_PARENTS`` in
``prepare_data/omomo/build_q.py:52``).
"""
from __future__ import annotations

from typing import Tuple

import torch
from pytorch3d.transforms import matrix_to_axis_angle

# SMPL-X body parents for joints 0..21. Mirrors prepare_data/omomo/build_q.py:52.
# parent of joint k = SMPLX_BODY_PARENTS[k]; root has parent -1.
SMPLX_BODY_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

# Per-device cache for parent index tensor (joints 1..21, parents of body joints).
# Avoids re-allocating this 21-element int64 tensor on every forward call.
_PARENT_IDX_CACHE: dict = {}


def _parent_idx_for(device: torch.device) -> torch.Tensor:
    """Return cached parent-index tensor (21,) int64 on ``device``."""
    key = str(device)
    cached = _PARENT_IDX_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(SMPLX_BODY_PARENTS[1:], dtype=torch.long, device=device)
        _PARENT_IDX_CACHE[key] = cached
    return cached


@torch.compiler.disable
def se3_to_aa_torch(
    Q: torch.Tensor,                # (B, T, 22, 4, 4) float, world-frame per-joint SE(3)
    pelvis_rest: torch.Tensor,      # (B, 3) float, J_rest_shaped[0] per sample
    parents: Tuple[int, ...] = SMPLX_BODY_PARENTS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert per-joint world SE(3) → SMPL-X ``(root_trans, root_orient_aa, body_aa)``.

    Args:
        Q: ``(B, T, 22, 4, 4)`` predicted/world per-joint SE(3). Rotation block
            ``Q[..., :3, :3]`` is global (world-frame) rotation; translation
            block ``Q[..., :3, 3]`` is global joint position.
        pelvis_rest: ``(B, 3)`` shape-blended pelvis rest position
            (``J_rest_shaped[0]``) per sample. Computed via
            ``smplx_pelvis_rest_offset_torch(betas, gender)``.
        parents: SMPL-X body parent indices, default ``SMPLX_BODY_PARENTS``.

    Returns:
        ``(root_trans, root_orient_aa, pose_body_aa)``:
            * ``root_trans``       ``(B, T, 3)`` — pelvis world translation
              minus rest offset
            * ``root_orient_aa``   ``(B, T, 3)`` — root axis-angle
            * ``pose_body_aa``     ``(B, T, 21, 3)`` — body-joint axis-angle
              for joints 1..21

    Raises:
        ValueError: shape mismatch on ``Q`` or ``pelvis_rest``.
    """
    if Q.ndim != 5 or Q.shape[-2:] != (4, 4) or Q.shape[-3] != 22:
        raise ValueError(f"Q must have shape (B, T, 22, 4, 4); got {tuple(Q.shape)}")
    if pelvis_rest.ndim != 2 or pelvis_rest.shape[-1] != 3:
        raise ValueError(f"pelvis_rest must have shape (B, 3); got {tuple(pelvis_rest.shape)}")

    J = Q.shape[-3]
    if Q.shape[0] != pelvis_rest.shape[0]:
        raise ValueError(
            f"batch mismatch: Q.shape[0]={Q.shape[0]} vs pelvis_rest.shape[0]={pelvis_rest.shape[0]}"
        )
    if len(parents) != J:
        raise ValueError(f"parents must have length {J}; got {len(parents)}")
    if parents[0] != -1:
        raise ValueError(f"parents[0] must be -1 (root); got {parents[0]}")

    R_global = Q[..., :3, :3]  # (B, T, 22, 3, 3)
    t_pelvis = Q[..., 0, :3, 3]  # (B, T, 3)

    # Translation: pelvis world position minus shape-blended rest offset.
    # pelvis_rest is (B, 3); broadcast over T.
    root_trans = t_pelvis - pelvis_rest[:, None, :]  # (B, T, 3)

    # Rotation: R_local[k] = R_global[parent(k)]^T @ R_global[k] for k >= 1.
    # Vectorize via gather over the parent index list.
    # Use cached per-device parent-index tensor to avoid per-call allocation.
    if parents is SMPLX_BODY_PARENTS:
        parent_idx = _parent_idx_for(Q.device)  # cached path (default arg)
    else:
        parent_idx = torch.tensor(parents[1:], dtype=torch.long, device=Q.device)  # (21,)
    # R_parent: (B, T, 21, 3, 3) — pull parent rotation for joints 1..21.
    R_parent = R_global[..., parent_idx, :, :]
    R_child = R_global[..., 1:, :, :]  # (B, T, 21, 3, 3)

    # Local rotation = parent^T @ child. Use einsum for clarity.
    R_local_body = torch.einsum("...ji,...jk->...ik", R_parent, R_child)  # (B, T, 21, 3, 3)

    # Concatenate root (absolute) + body-local.
    R_root = R_global[..., 0:1, :, :]  # (B, T, 1, 3, 3)
    R_local_all = torch.cat([R_root, R_local_body], dim=-3)  # (B, T, 22, 3, 3)

    # Rotation matrix → axis-angle. matrix_to_axis_angle accepts (..., 3, 3).
    aa_all = matrix_to_axis_angle(R_local_all)  # (B, T, 22, 3)

    root_orient_aa = aa_all[..., 0, :]  # (B, T, 3)
    pose_body_aa = aa_all[..., 1:, :]   # (B, T, 21, 3)

    return root_trans, root_orient_aa, pose_body_aa
