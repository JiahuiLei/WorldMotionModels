"""Closed-form inverse kinematics: per-joint world SE(3) -> MANO parameters.

Given Q_world (T, 16, 4, 4) containing the world-space SE(3) transform of each
MANO joint, recovers the MANO parameters (trans, global_orient_aa, fpose_aa)
that reproduce those joint positions and orientations when passed through the
MANO forward function.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import smplx
import torch
from pytorch3d.transforms import matrix_to_axis_angle

# MANO kinematic tree: parent index for each of the 16 joints.
# Index -1 denotes no parent (root wrist joint).
# Source: MANO PKL kintree_table[0] with 4294967295 mapped to -1.
MANO_KINTREE: list = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

# Module-level caches
_mano_cache: Dict[tuple, smplx.MANO] = {}
_rest_wrist_cache: Dict[tuple, torch.Tensor] = {}


def project_so3(R: torch.Tensor) -> torch.Tensor:
    """SVD-project a batch of matrices to the nearest SO(3) rotation.

    Uses the standard SVD projection with determinant correction to handle
    reflection matrices (det=-1) that may arise from noisy input.

    Args:
        R: (..., 3, 3) float tensor of rotation-like matrices.

    Returns:
        Projected (..., 3, 3) float tensor with each matrix in SO(3).
    """
    # R: (..., 3, 3)
    U, _S, Vt = torch.linalg.svd(R)  # U: (...,3,3), Vt: (...,3,3); singular values unused
    det_sign = torch.det(U @ Vt)     # (...,)
    Vt = Vt.clone()
    # Flip last row of Vt where det = -1 to enforce det(result) = +1
    Vt[..., -1, :] = Vt[..., -1, :] * det_sign.unsqueeze(-1)
    return U @ Vt  # (..., 3, 3)


def _mano_cache_key(
    mano_pkl_path: str,
    is_rhand: bool,
    vtemplate: Optional[Union[torch.Tensor, np.ndarray]],
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Build the cache key for a MANO model instance."""
    if vtemplate is not None:
        if isinstance(vtemplate, torch.Tensor):
            vt_hash = hash(vtemplate.detach().cpu().numpy().tobytes())
        else:
            vt_hash = hash(vtemplate.tobytes())
    else:
        vt_hash = None
    return (mano_pkl_path, is_rhand, vt_hash, batch_size, str(device))


def get_mano_model(
    mano_pkl_path: str,
    is_rhand: bool,
    vtemplate: Optional[Union[torch.Tensor, np.ndarray]],
    batch_size: int,
    device: torch.device,
) -> smplx.MANO:
    """Cached MANO model factory.

    Returns a MANO model with use_pca=False and flat_hand_mean=True, moved to
    the requested device. Identical calls reuse a cached instance.

    Args:
        mano_pkl_path: Path to the MANO PKL file (LEFT or RIGHT).
        is_rhand: True for right hand, False for left.
        vtemplate: Optional vertex template, shape (778, 3), torch.Tensor or
            np.ndarray. None means MANO uses its default template.
        batch_size: Batch size for the MANO model.
        device: Target torch device.

    Returns:
        smplx.MANO instance on `device`.
    """
    key = _mano_cache_key(mano_pkl_path, is_rhand, vtemplate, batch_size, device)
    if key not in _mano_cache:
        if vtemplate is not None:
            if isinstance(vtemplate, np.ndarray):
                vt = torch.tensor(vtemplate, dtype=torch.float32)
            else:
                vt = vtemplate.float()
        else:
            vt = None
        model = smplx.MANO(
            model_path=mano_pkl_path,
            use_pca=False,
            flat_hand_mean=True,
            is_rhand=is_rhand,
            batch_size=batch_size,
            v_template=vt,
        ).to(device)
        _mano_cache[key] = model
    return _mano_cache[key]


def get_rest_wrist_offset(
    mano_pkl_path: str,
    is_rhand: bool,
    vtemplate: Optional[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the wrist joint position in the MANO rest pose.

    Runs MANO with all-zero pose and translation to find where the wrist joint
    sits in rest pose. This offset must be subtracted from the world wrist
    position to obtain the MANO `transl` parameter.

    Args:
        mano_pkl_path: Path to the MANO PKL file.
        is_rhand: True for right hand, False for left.
        vtemplate: Optional vertex template (778, 3).
        device: Target torch device.
        dtype: Output tensor dtype.

    Returns:
        Shape (3,) tensor: rest-pose wrist joint position.
    """
    # Cache key without batch_size (always batch_size=1)
    if vtemplate is not None:
        if isinstance(vtemplate, torch.Tensor):
            vt_hash = hash(vtemplate.detach().cpu().numpy().tobytes())
        else:
            vt_hash = hash(vtemplate.tobytes())
    else:
        vt_hash = None
    # Cache key omits dtype: stored offset is always float32 and converted at the
    # call site to the caller's dtype (see line below in q_to_mano).
    key = (mano_pkl_path, is_rhand, vt_hash, str(device))

    if key not in _rest_wrist_cache:
        model = get_mano_model(mano_pkl_path, is_rhand, vtemplate, 1, device)
        with torch.no_grad():
            out = model(
                global_orient=torch.zeros(1, 3, device=device, dtype=dtype),
                hand_pose=torch.zeros(1, 45, device=device, dtype=dtype),
                transl=torch.zeros(1, 3, device=device, dtype=dtype),
            )
        # out.joints: (1, 16, 3) — joint 0 is the wrist
        _rest_wrist_cache[key] = out.joints[0, 0].detach()  # (3,)
    return _rest_wrist_cache[key]


def q_to_mano(
    Q_world: torch.Tensor,
    is_rhand: bool,
    vtemplate: Optional[Union[torch.Tensor, np.ndarray]],
    mano_pkl_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form IK: world SE(3) per joint -> MANO parameters.

    Converts per-joint world-space SE(3) transforms to the three MANO
    parameters needed to reproduce those joint poses via MANO forward.

    Algorithm:
        1. Project rotation blocks to SO(3) via SVD.
        2. Recover `trans` as wrist world position minus rest-pose wrist offset.
        3. Recover `global_orient_aa` as axis-angle of the world wrist rotation.
        4. Recover finger `fpose_aa` by composing local relative rotations
           (parent^T @ child) for each of the 15 non-root joints.

    Args:
        Q_world: (T, 16, 4, 4) float tensor of world SE(3) per joint per frame.
            Q_world[t, k, :3, :3] = world rotation of joint k at time t.
            Q_world[t, k, :3,  3] = world position of joint k at time t.
        is_rhand: True for right hand, False for left.
        vtemplate: Optional vertex template (778, 3), torch.Tensor or np.ndarray.
        mano_pkl_path: Path to the MANO PKL file for the appropriate hand.

    Returns:
        Tuple of:
            trans: (T, 3) MANO translation parameter.
            global_orient_aa: (T, 3) wrist global orientation as axis-angle.
            fpose_aa: (T, 45) finger pose as axis-angle (15 joints x 3).

    Note:
        All outputs have the same dtype and device as Q_world. The function is
        fully vectorised over T — no Python loops over frames.
    """
    T = Q_world.shape[0]
    device = Q_world.device
    dtype = Q_world.dtype

    # (T, 16, 3, 3) — project each rotation block to SO(3)
    R_world = project_so3(Q_world[:, :, :3, :3])

    # Rest-pose wrist offset: (3,)
    rest_wrist = get_rest_wrist_offset(
        mano_pkl_path, is_rhand, vtemplate, device, dtype
    ).to(dtype=dtype, device=device)

    # MANO transl: wrist world pos minus rest-pose wrist pos
    trans = Q_world[:, 0, :3, 3] - rest_wrist  # (T, 3)

    # Global orientation: world rotation of joint 0 (wrist)
    global_orient_aa = matrix_to_axis_angle(R_world[:, 0])  # (T, 3)

    # Finger local rotations: R_local[j] = R_world[parent(j)]^T @ R_world[j]
    # parent_idx[i] = parent of joint (i+1), i in 0..14
    parent_idx = MANO_KINTREE[1:]  # [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    R_parent = R_world[:, parent_idx]         # (T, 15, 3, 3)
    R_child = R_world[:, 1:]                  # (T, 15, 3, 3)
    R_local = R_parent.transpose(-1, -2) @ R_child  # (T, 15, 3, 3)
    fpose_aa = matrix_to_axis_angle(R_local).reshape(T, 45)  # (T, 45)

    return trans, global_orient_aa, fpose_aa
