"""SMPL-X joint regressor for hand-object loss.

Returns full SMPL-X joint set (B, T, J, 3) where J=127 in practice.
Caller picks indices.

AUDITED JOINT MAPPING (verified by test/audit_smplx_joints.py, 2026-04-27):
  smplx.create(..., return_joints=True) returns 127 joints (not 55).
  Body joints 0..21 (22 joints, same as SMPL-H 22-joint body):
    j[0]  = pelvis
    j[1]  = l_hip,         j[2]  = r_hip
    j[3]  = spine1
    j[4]  = l_knee,        j[5]  = r_knee
    j[6]  = spine2
    j[7]  = l_ankle,       j[8]  = r_ankle
    j[9]  = spine3
    j[10] = l_foot,        j[11] = r_foot
    j[12] = neck
    j[13] = l_collar,      j[14] = r_collar
    j[15] = head
    j[16] = l_shoulder,    j[17] = r_shoulder
    j[18] = l_elbow,       j[19] = r_elbow
    j[20] = l_wrist,       j[21] = r_wrist   <- WRISTS
  Face joints 22..24:
    j[22] = jaw,  j[23] = left_eye_smplhf,  j[24] = right_eye_smplhf
  Left hand joints 25..39 (15 joints, index1..pinky3):
    j[25] = l_index1 base  <- L PALM PROXY (verified: dist to L wrist = 0.104m)
    j[26] = l_index2
    j[27] = l_index3
    j[28] = l_middle1
    j[29] = l_middle2
    j[30] = l_middle3
    j[31] = l_pinky1
    j[32] = l_pinky2
    j[33] = l_pinky3
    j[34] = l_ring1
    j[35] = l_ring2
    j[36] = l_ring3
    j[37] = l_thumb1
    j[38] = l_thumb2
    j[39] = l_thumb3
  Right hand joints 40..54 (mirrored, index1..pinky3):
    j[40] = r_index1 base  <- R PALM PROXY (verified: dist to R wrist = 0.102m)
    j[41..54]: (right hand fingers, mirrored order)
  Joints 55..126: additional vertex regression results (face, hands, etc.)

COORDINATE CONVENTION (SMPL-X T-pose, zero-params):
  Person's left arm extends in +X direction (j[20].x = +0.670)
  Person's right arm extends in -X direction (j[21].x = -0.672)
  Up = +Y (pelvis at y=-0.35, head at y=+0.27)
  Note: this is OPPOSITE of the "x<0 for left" comment in CHOIS — SMPL-X uses
  body-right = -X, body-left = +X convention in zero pose.

PATH NOTE: smplx.create() model_path must point to the PARENT dir containing
  the 'smplx/' subfolder (i.e. 'data_wmm/omomo_v2', NOT 'data_wmm/omomo_v2/smplx').

For OMOMO hand-object loss:
  wrists:     hand_wrist_joint_idx = [20, 21]   (L, R)
  palm proxy: hand_palm_joint_idx  = [25, 40]   (L index1 base, R index1 base)
"""
from __future__ import annotations

from typing import Dict

import torch

try:
    import smplx
except ImportError as e:
    raise ImportError(
        "smplx package required for hand-object loss. Install: pip install smplx"
    ) from e


_MODEL_CACHE: Dict[str, "smplx.SMPLX"] = {}


def _get_model(smplx_parent_dir: str, gender: str, device: torch.device) -> "smplx.SMPLX":
    """Get or create a cached SMPL-X model.

    Args:
        smplx_parent_dir: Parent directory that contains the 'smplx/' subfolder.
            E.g. 'data_wmm/omomo_v2' (NOT 'data_wmm/omomo_v2/smplx').
        gender: 'neutral', 'male', or 'female'.
        device: Target device.
    """
    key = f"{smplx_parent_dir}::{gender}::{device}"
    if key not in _MODEL_CACHE:
        model = smplx.create(
            model_path=smplx_parent_dir,
            model_type="smplx",
            gender=gender.lower(),
            num_betas=16,
            use_pca=False,
            flat_hand_mean=True,
            ext="npz",
        ).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@torch.compiler.disable
def smplx_joints_forward(
    betas: torch.Tensor,           # (B, 16)
    pose_body_aa: torch.Tensor,    # (B, T, 21, 3)
    root_orient_aa: torch.Tensor,  # (B, T, 3)
    root_trans: torch.Tensor,      # (B, T, 3)
    smplx_dir: str = "data_wmm/omomo_v2",
    gender: str = "neutral",
) -> torch.Tensor:
    """SMPL-X forward pass returning ALL joints in world frame.

    NOTE: smplx_dir must be the PARENT directory containing the 'smplx/' subfolder,
    e.g. 'data_wmm/omomo_v2' (not 'data_wmm/omomo_v2/smplx').

    Args:
        betas: Shape coefficients, (B, 16).
        pose_body_aa: Body joint rotations in axis-angle, (B, T, 21, 3).
            Joints 0..20 of the body (excluding root/global orient).
        root_orient_aa: Global orientation in axis-angle, (B, T, 3).
        root_trans: Root translation in metres, (B, T, 3).
        smplx_dir: Parent dir containing 'smplx/' subfolder with model .npz files.
        gender: 'neutral', 'male', or 'female'.

    Returns:
        joints: (B, T, J, 3) where J=127. All positions in metres, world frame.
            Key indices (see module docstring for full mapping):
              j[20] = L wrist,       j[21] = R wrist
              j[25] = L index1 base, j[40] = R index1 base (palm proxies)
    """
    B, T = pose_body_aa.shape[:2]
    device = betas.device
    model = _get_model(smplx_dir, gender, device)
    BT = B * T
    out = model(
        betas=betas[:, None, :].expand(B, T, 16).reshape(BT, 16),
        body_pose=pose_body_aa.reshape(BT, 21 * 3),
        global_orient=root_orient_aa.reshape(BT, 3),
        transl=root_trans.reshape(BT, 3),
        left_hand_pose=torch.zeros(BT, 45, device=device, dtype=betas.dtype),
        right_hand_pose=torch.zeros(BT, 45, device=device, dtype=betas.dtype),
        # Explicitly pass jaw/eye/expression as zeros to avoid batch-size mismatch
        # when BT > 1 (the model's default stored values have batch_size=1).
        jaw_pose=torch.zeros(BT, 3, device=device, dtype=betas.dtype),
        leye_pose=torch.zeros(BT, 3, device=device, dtype=betas.dtype),
        reye_pose=torch.zeros(BT, 3, device=device, dtype=betas.dtype),
        expression=torch.zeros(BT, model.num_expression_coeffs, device=device, dtype=betas.dtype),
        return_joints=True,
    )
    J = out.joints.shape[1]
    return out.joints.reshape(B, T, J, 3)


@torch.compiler.disable
def smplx_verts_forward(
    betas: torch.Tensor,           # (B, 16)
    pose_body_aa: torch.Tensor,    # (B, T, 21, 3)
    root_orient_aa: torch.Tensor,  # (B, T, 3)
    root_trans: torch.Tensor,      # (B, T, 3)
    smplx_dir: str = "data_wmm/omomo_v2",
    gender: str = "neutral",
) -> torch.Tensor:
    """SMPL-X forward pass returning the FULL MESH (10475 verts).

    Mirrors :func:`smplx_joints_forward` but returns mesh vertices instead
    of joints. Used by the w1.6.2 full-body penetration loss to query the
    object SDF at every body vert (matches the eval P_hand metric, which
    samples all SMPL-X mesh verts).

    NOTE: ``flat_hand_mean=True`` (set in :func:`_get_model`) means the
    rest hand pose is the splayed-flat palm — matches the eval scorer
    (``src/umm/eval/omomo/smplx_fk.py:20``: ``smplx.create(...) is called
    with use_pca=False and flat_hand_mean=True``). Hand pose input is
    hardcoded zeros (no finger articulation modality in the model);
    output hand verts are flat fingers in rest position.

    The numpy reference ``prepare_data/omomo/build_q.py:_get_pelvis_rest_offset``
    uses ``flat_hand_mean=False`` for pelvis rest only; this is mathematically
    equivalent for joint 0 (pelvis) in zero body pose because the
    ``hands_mean`` term affects only finger joint positions, not body joints.
    Parity test ``test/test_se3_to_aa_parity.py`` PASSES across all three
    genders at ~2e-7 (fp32 noise floor), confirming the flag choice is
    correct for our use case.

    Args:
        betas: Shape coefficients, ``(B, 16)``.
        pose_body_aa: Body joint rotations in axis-angle, ``(B, T, 21, 3)``.
        root_orient_aa: Global orientation in axis-angle, ``(B, T, 3)``.
        root_trans: Root translation in metres, ``(B, T, 3)``.
        smplx_dir: Parent dir containing ``smplx/`` subfolder.
        gender: ``'neutral'``, ``'male'``, or ``'female'``.

    **MIXED-GENDER WARNING:** ``gender`` applies to the entire batch. For
    mixed-gender batches, the caller MUST split by gender and call once per
    group. Wrong gender → ~1.5 cm vert error (silent).

    **DTYPE:** SMPL-X is fp32-only. Inputs are cast to float32 internally;
    output verts are float32. Safe to call from a bf16 autocast region.

    Returns:
        verts: ``(B, T, V, 3)`` where V=10475. World-frame metres, float32.
    """
    B, T = pose_body_aa.shape[:2]
    device = betas.device
    model = _get_model(smplx_dir, gender, device)
    BT = B * T
    # SMPL-X is fp32-only — cast all inputs to float32 to avoid bf16/fp16 crashes
    # under autocast. Doing this inside the @torch.compiler.disable boundary so
    # autograd records the cast for backward.
    betas_f = betas.float()
    body_pose_f = pose_body_aa.float()
    global_orient_f = root_orient_aa.float()
    transl_f = root_trans.float()
    out = model(
        betas=betas_f[:, None, :].expand(B, T, 16).reshape(BT, 16),
        body_pose=body_pose_f.reshape(BT, 21 * 3),
        global_orient=global_orient_f.reshape(BT, 3),
        transl=transl_f.reshape(BT, 3),
        left_hand_pose=torch.zeros(BT, 45, device=device, dtype=torch.float32),
        right_hand_pose=torch.zeros(BT, 45, device=device, dtype=torch.float32),
        jaw_pose=torch.zeros(BT, 3, device=device, dtype=torch.float32),
        leye_pose=torch.zeros(BT, 3, device=device, dtype=torch.float32),
        reye_pose=torch.zeros(BT, 3, device=device, dtype=torch.float32),
        expression=torch.zeros(BT, model.num_expression_coeffs, device=device, dtype=torch.float32),
        return_verts=True,
    )
    V = out.vertices.shape[1]
    return out.vertices.reshape(B, T, V, 3)


# Per-(gender, betas) cache for shape-blended pelvis rest offset (J_rest_shaped[0]).
# Used by se3_to_aa_torch to invert SMPL-X FK on predicted joint SE(3).
_PELVIS_REST_CACHE: Dict[tuple, torch.Tensor] = {}


@torch.compiler.disable
def smplx_pelvis_rest_offset_torch(
    betas: torch.Tensor,           # (B, 16)
    gender: str,                   # 'neutral' | 'male' | 'female'
    smplx_dir: str = "data_wmm/omomo_v2",
) -> torch.Tensor:
    """Return shape-blended pelvis rest position J_rest_shaped[0] per sample.

    Equivalent to ``prepare_data/omomo/build_q.py:_get_pelvis_rest_offset`` but:
        * accepts a batched torch tensor of betas
        * caches per ``(gender, betas)`` for repeated subjects
        * runs on the same device as ``betas``

    The returned tensor is detached from the autograd graph (gradient through
    pelvis offset is irrelevant for the penetration loss; we only need root
    translation conditioning).

    Args:
        betas: ``(B, 16)`` float32 SMPL-X shape coefficients.
        gender: ``'neutral' | 'male' | 'female'`` — assumed identical for
            this batch (same gender per call). For mixed-gender batches,
            split the batch and call per-gender.
        smplx_dir: Parent dir containing ``smplx/`` subfolder.

    Returns:
        ``(B, 3)`` pelvis rest position in metres, on ``betas.device``.
    """
    if betas.ndim != 2 or betas.shape[-1] != 16:
        raise ValueError(f"betas must be (B, 16); got {tuple(betas.shape)}")
    B = betas.shape[0]
    device = betas.device
    # SMPL-X is fp32-only — cast betas to float32 to avoid bf16/fp16 crashes
    # under autocast. Pelvis offset doesn't need gradient flow (detached below).
    betas = betas.float()

    # Cache key per-sample.
    out = []
    for b in range(B):
        key = (gender, tuple(betas[b].detach().cpu().tolist()))
        cached = _PELVIS_REST_CACHE.get(key)
        if cached is None or cached.device != device:
            model = _get_model(smplx_dir, gender, device)
            with torch.no_grad():
                fwd = model(
                    betas=betas[b:b + 1],
                    body_pose=torch.zeros(1, 63, device=device, dtype=torch.float32),
                    global_orient=torch.zeros(1, 3, device=device, dtype=torch.float32),
                    transl=torch.zeros(1, 3, device=device, dtype=torch.float32),
                    left_hand_pose=torch.zeros(1, 45, device=device, dtype=torch.float32),
                    right_hand_pose=torch.zeros(1, 45, device=device, dtype=torch.float32),
                    jaw_pose=torch.zeros(1, 3, device=device, dtype=torch.float32),
                    leye_pose=torch.zeros(1, 3, device=device, dtype=torch.float32),
                    reye_pose=torch.zeros(1, 3, device=device, dtype=torch.float32),
                    expression=torch.zeros(1, model.num_expression_coeffs, device=device, dtype=torch.float32),
                    return_joints=True,
                )
            cached = fwd.joints[0, 0].detach().clone()  # (3,) pelvis = joint 0
            _PELVIS_REST_CACHE[key] = cached
        out.append(cached)
    return torch.stack(out, dim=0)  # (B, 3)
