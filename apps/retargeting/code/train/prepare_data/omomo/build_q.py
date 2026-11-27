"""Build per-joint world-frame SE(3) from SMPL-X FK + inverse.

Purpose: enable OMOMO's real-SE(3) trace encoding (WMM convention) while
keeping the axis-angle eval format intact. Forward path needs SMPL-X
(to derive SMPL-X-consistent world joint positions); the inverse
``se3_to_axis_angle`` also requires ``smplx`` to resolve the
(betas, gender)-dependent pelvis rest offset ``J_rest_shaped[0]``.

Public API:
    smplx_world_transforms(root_trans, root_orient_aa, pose_body_aa,
                           betas, gender, smplx_dir)
        -> (global_rot_mat (T, 22, 3, 3), global_jpos (T, 22, 3))
    build_q_from_pickle_fields(global_rot_mat, global_jpos) -> Q
    se3_to_axis_angle(Q, betas, gender, smplx_dir, parents=None)
        -> (root_trans, root_orient_aa, pose_body_aa)
    da_pose_q(gender, betas, smplx_dir) -> (22, 4, 4) float32
        DA-pose world-frame SE(3) per body joint (standing body, arms ~45° down,
        Z-up world). Dataset/callback ctx prior (replaces identity-SE(3) blob).
    SMPLX_BODY_PARENTS # (22,) int32 array
    _verify_parents # one-shot smplx-model-backed sanity helper

IMPORTANT — SMPL-X vs SMPL-H (iter-02 correctness fix)
------------------------------------------------------
OMOMO's raw CHOIS pickle stores ``motion[:, :72]`` as SMPL-H-derived world
joint positions, NOT SMPL-X. SMPL-X's shape-blended rest-pose pelvis
position differs from SMPL-H's by up to ~5.4 mm on female subjects
(verified empirically: sub10=5.4mm, sub9=1.9mm, sub1=2.9mm; male=0.08mm).
Downstream eval (``eval/omomo/eval_chois_table1.py`` → ``_SmplxFK.fk_sequence``)
uses ``smplx.SMPLX``; using SMPL-H joint positions there introduces a
silent ~5 mm bias on every female window.

The **authoritative** path to build Q in the OMOMO rewrite is therefore
``smplx_world_transforms(...)`` — re-FK the stored axis-angle pose
through SMPL-X. DO NOT pass raw pickle's ``motion[:, :72]`` to
``build_q_from_pickle_fields`` directly.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# SMPL-X body kinematic parents (joint indices 0..21; -1 = root)
# ---------------------------------------------------------------------------
# Verified against smplx.SMPLX(...).parents[:22] on 2026-04-24 (male model,
# data_wmm/omomo/smplx/). Values frozen here so that dataset/viz/eval code
# has zero runtime dependency on the smplx package.
SMPLX_BODY_PARENTS: np.ndarray = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# SMPL-X model cache + world-transform helper (authoritative AA -> Q path)
# ---------------------------------------------------------------------------
# Cache keyed by (gender, batch_size) so repeated calls for the same
# (gender, T) amortize the 500+MB SMPL-X model load. Mirrors the pattern in
# ``prepare_data/omomo/omomo_viser.py::_SmplxFK._get_model`` (lines 293-313),
# but keyed locally here to keep build_q.py's public surface self-contained.
_SMPLX_MODEL_CACHE: dict = {}


def _get_smplx_model(gender: str, batch_size: int, smplx_dir: Path):
    """Return a cached smplx.SMPLX model for (gender, batch_size).

    Uses ``use_pca=False, flat_hand_mean=False, num_betas=16`` — matches
    ``_SmplxFK._get_model`` (omomo_viser.py) and the existing round-trip
    test harness (test/wmm/test_omomo_q_roundtrip.py). Keeping the same
    flags everywhere prevents subtle T-pose drift between convert / viz /
    eval.

    Args:
        gender: ``"male" | "female" | "neutral"``.
        batch_size: sequence length ``T`` — cached so repeated calls at the
            same length re-use the model.
        smplx_dir: directory containing ``SMPLX_{GENDER}.npz``.

    Returns:
        ``smplx.SMPLX`` instance (heavyweight, cached module-level).

    Raises:
        ImportError: if ``smplx`` package is not installed.
        FileNotFoundError: if ``SMPLX_{GENDER}.npz`` is missing.
        ValueError: on unsupported gender.
    """
    gender = gender.lower()
    if gender not in ("male", "female", "neutral"):
        raise ValueError(f"unsupported gender: {gender!r}")
    key = (gender, int(batch_size))
    cached = _SMPLX_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import smplx  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "smplx is required for smplx_world_transforms; install it "
            "(pip install smplx). Underlying error: " + str(e)
        ) from e

    smplx_dir = Path(smplx_dir)
    model_path = smplx_dir / f"SMPLX_{gender.upper()}.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

    model = smplx.SMPLX(
        str(model_path),
        gender=gender,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=int(batch_size),
        num_betas=16,
    )
    _SMPLX_MODEL_CACHE[key] = model
    return model


def smplx_world_transforms(
    root_trans: np.ndarray,
    root_orient_aa: np.ndarray,
    pose_body_aa: np.ndarray,
    betas: np.ndarray,
    gender: str,
    smplx_dir,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SMPL-X FK and return SMPL-X-consistent world joint transforms.

    This is the AUTHORITATIVE path for deriving ``(global_rot_mat,
    global_jpos)`` that will feed ``build_q_from_pickle_fields``. It runs
    ``smplx.SMPLX(...).forward(...)`` so the produced positions agree with
    the downstream eval's SMPL-X FK (``eval/omomo/eval_chois_table1.py``'s
    ``_SmplxFK.fk_sequence``).

    Why NOT read ``motion[:, :72]`` from the pickle directly: OMOMO's raw
    CHOIS pickle stores SMPL-H-derived world joint positions, which differ
    from SMPL-X by up to ~5.4 mm on female subjects (empirical: sub10 =
    5.4 mm, sub9 = 1.9 mm, sub1 = 2.9 mm; male subjects = 0.08 mm). Using
    the pickle positions directly introduces a silent ~5 mm pelvis-offset
    bias for every female window. Re-FK'ing via SMPL-X eliminates that
    bias (cost: ~5 min per full split, benefit: correctness on 40%+ of
    windows).

    Rotation strategy:
        Positions come from ``out.joints[:, :22]`` (SMPL-X's internal FK).
        For rotations we **compose the parent chain manually** from the
        per-joint local rotation matrices (Rodrigues on the input AA).
        This matches SMPL-X's internal ``batch_rigid_transform`` exactly
        (up to fp32 rounding; verified by ``test_omomo_q_roundtrip.py``).

    Args:
        root_trans: shape (T, 3) float32, pelvis world translation.
        root_orient_aa: shape (T, 3) float32, root axis-angle.
        pose_body_aa: shape (T, 21, 3) float32, body-joint axis-angle
            (joints 1..21).
        betas: shape (16,) float32, SMPL-X shape coefficients.
        gender: ``"male" | "female" | "neutral"``.
        smplx_dir: ``str | Path`` — directory containing ``SMPLX_{GENDER}.npz``.

    Returns:
        Tuple ``(global_rot_mat, global_jpos)``:
            - ``global_rot_mat``: shape (T, 22, 3, 3) float32, world-frame
              joint rotations (cumulative parent chain via Rodrigues).
            - ``global_jpos``: shape (T, 22, 3) float32, world-frame joint
              positions (``out.joints[:, :22]`` from SMPL-X forward).

    Raises:
        ValueError: if input shapes are inconsistent or ``betas.size != 16``.
        ImportError: if ``smplx`` is not installed.
        FileNotFoundError: if the SMPL-X model file is missing.
    """
    # --- shape / dtype normalization --------------------------------------
    root_trans = np.asarray(root_trans, dtype=np.float32)            # (T, 3)
    root_orient_aa = np.asarray(root_orient_aa, dtype=np.float32)    # (T, 3)
    pose_body_aa = np.asarray(pose_body_aa, dtype=np.float32)        # (T, 21, 3)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)          # (N,)

    if root_trans.ndim != 2 or root_trans.shape[1] != 3:
        raise ValueError(f"root_trans must be (T, 3); got {root_trans.shape}")
    T = int(root_trans.shape[0])
    if root_orient_aa.shape != (T, 3):
        raise ValueError(
            f"root_orient_aa must be (T={T}, 3); got {root_orient_aa.shape}"
        )
    if pose_body_aa.shape != (T, 21, 3):
        raise ValueError(
            f"pose_body_aa must be (T={T}, 21, 3); got {pose_body_aa.shape}"
        )
    if betas.size != 16:
        raise ValueError(
            f"betas must be (16,) to match OMOMO num_betas=16; got size {betas.size}"
        )

    # Local import: keeps module top-level torch-free for fast import in
    # downstream code that only needs SMPLX_BODY_PARENTS.
    import torch  # type: ignore

    model = _get_smplx_model(gender, T, Path(smplx_dir))

    with torch.no_grad():
        betas_t = (
            torch.from_numpy(betas.reshape(1, 16))
            .expand(T, -1)
            .contiguous()
        )                                                             # (T, 16)
        body_pose_t = torch.from_numpy(pose_body_aa.reshape(T, 63))   # (T, 63)
        global_orient_t = torch.from_numpy(root_orient_aa)            # (T, 3)
        transl_t = torch.from_numpy(root_trans)                       # (T, 3)

        out = model(
            betas=betas_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            left_hand_pose=torch.zeros(T, 45),
            right_hand_pose=torch.zeros(T, 45),
            jaw_pose=torch.zeros(T, 3),
            leye_pose=torch.zeros(T, 3),
            reye_pose=torch.zeros(T, 3),
            expression=torch.zeros(T, 10),
            transl=transl_t,
            return_full_pose=True,
        )
    global_jpos = (
        out.joints[:, :22].detach().cpu().numpy().astype(np.float32)
    )  # (T, 22, 3)

    # World joint rotations: compose the SMPL-X body kinematic chain.
    # SMPL-X's internal batch_rigid_transform produces the same up to fp32
    # precision; we recompute here to avoid coupling to a private smplx
    # helper. all_aa: (T, 22, 3); R_local: (T, 22, 3, 3).
    all_aa = np.concatenate(
        [root_orient_aa[:, None, :], pose_body_aa], axis=1
    )  # (T, 22, 3)
    R_local = (
        R.from_rotvec(all_aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    ).astype(np.float32)                                              # (T, 22, 3, 3)

    R_global = np.zeros_like(R_local)                                 # (T, 22, 3, 3)
    R_global[:, 0] = R_local[:, 0]
    for k in range(1, 22):
        parent = int(SMPLX_BODY_PARENTS[k])
        R_global[:, k] = R_global[:, parent] @ R_local[:, k]

    return R_global.astype(np.float32), global_jpos.astype(np.float32)


def build_q_from_pickle_fields(
    global_rot_mat: np.ndarray,
    global_jpos: np.ndarray,
) -> np.ndarray:
    """Assemble per-joint world-frame SE(3) from SMPL-X-derived transforms.

    Bit-exact assembly: copies world joint rotation and world joint position
    straight into an SE(3) matrix. No FK re-derivation inside this function.

    **CRITICAL — input provenance:** ``global_rot_mat`` and ``global_jpos``
    MUST be **SMPL-X-derived** (e.g. output of :func:`smplx_world_transforms`).
    Do NOT pass OMOMO's raw pickle ``motion[:, :72]`` here — those positions
    are SMPL-H-derived and differ from SMPL-X's FK by up to ~5.4 mm on
    female subjects (empirical: sub10 = 5.4 mm, sub9 = 1.9 mm, sub1 = 2.9 mm).
    The downstream eval uses ``smplx.SMPLX`` FK, so any residual bias from
    SMPL-H-derived positions shows up directly as an MPJPE floor on female
    windows.

    Args:
        global_rot_mat: shape (L, 22, 3, 3) float, world-frame joint rotation
            (SMPL-X-composed parent chain; see :func:`smplx_world_transforms`).
        global_jpos: shape (L, 22, 3) float, world-frame joint position
            (SMPL-X FK output ``out.joints[:, :22]``).

    Returns:
        Q: shape (L, 22, 4, 4) float32, per-joint homogeneous SE(3).
            Q[t, k, :3, :3] = global_rot_mat[t, k]
            Q[t, k, :3,  3] = global_jpos[t, k]
            Q[t, k,  3, :] = [0, 0, 0, 1]

    Raises:
        ValueError: if input shapes disagree or do not match (L, 22, ...).
    """
    if global_rot_mat.ndim != 4 or global_rot_mat.shape[1:] != (22, 3, 3):
        raise ValueError(
            f"global_rot_mat must have shape (L, 22, 3, 3); got {global_rot_mat.shape}"
        )
    if global_jpos.ndim != 3 or global_jpos.shape[1:] != (22, 3):
        raise ValueError(
            f"global_jpos must have shape (L, 22, 3); got {global_jpos.shape}"
        )
    if global_rot_mat.shape[0] != global_jpos.shape[0]:
        raise ValueError(
            "global_rot_mat and global_jpos must share L; "
            f"got {global_rot_mat.shape[0]} vs {global_jpos.shape[0]}"
        )

    L = global_rot_mat.shape[0]
    Q = np.zeros((L, 22, 4, 4), dtype=np.float32)  # (L, 22, 4, 4) float32
    Q[:, :, :3, :3] = global_rot_mat.astype(np.float32, copy=False)
    Q[:, :, :3, 3] = global_jpos.astype(np.float32, copy=False)
    Q[:, :, 3, 3] = 1.0
    return Q


# ---------------------------------------------------------------------------
# Shape-dependent pelvis rest offset cache (SMPL-X J_rest_shaped[0])
# ---------------------------------------------------------------------------
# SMPL-X forward kinematics stores the pelvis WORLD position as
#     global_jpos[:, 0] = J_rest_shaped[0] + transl,
# where J_rest_shaped = J_regressor @ (v_template + shapedirs @ betas), i.e.
# the (betas, gender)-blended rest-pose joint positions. Inverting a CHOIS
# pickle Q therefore requires subtracting this offset from Q[:, 0, :3, 3]
# before recording root translation (mirrors CHOIS convert.py:355 which adds
# ``trans2joint = -J_rest_shaped[0]`` when building the npz).
#
# We cache the offset by (gender, tuple(betas.tolist())) so that repeated
# inversions for the same subject amortize the 500+MB SMPL-X model load.
_PELVIS_OFFSET_CACHE: dict = {}


def _get_pelvis_rest_offset(
    betas: np.ndarray,
    gender: str,
    smplx_dir: Path,
) -> np.ndarray:
    """Return ``J_rest_shaped[0]`` (shape (3,) float32) for the given subject.

    Cached by ``(gender, tuple(betas.tolist()))`` — subsequent calls for the
    same subject are O(1) dict lookups.

    Args:
        betas: shape (16,) float32, SMPL-X shape coefficients.
        gender: ``"male" | "female" | "neutral"``.
        smplx_dir: directory containing ``SMPLX_{GENDER}.npz``.

    Returns:
        pelvis: shape (3,) float32, shape-blended pelvis rest position.

    Raises:
        ImportError: if ``smplx`` is not installed.
        FileNotFoundError: if the SMPL-X model file is missing.
        ValueError: if ``betas`` has the wrong shape.
    """
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)  # (N,) float32
    if betas.size != 16:
        raise ValueError(
            f"betas must be (16,) to match OMOMO num_betas=16; got size {betas.size}"
        )

    key = (gender, tuple(betas.tolist()))
    cached = _PELVIS_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import smplx  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "smplx is required for se3_to_axis_angle to compute "
            "J_rest_shaped[0]; install it (pip install smplx). "
            "Underlying error: " + str(e)
        ) from e
    import torch  # local: keep module top-level torch-free for fast import

    smplx_dir = Path(smplx_dir)
    model_path = str(smplx_dir / f"SMPLX_{gender.upper()}.npz")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

    # Forward at pose=0, transl=0, shape=betas -> out.joints[0, 0] == J_rest_shaped[0].
    # Using batch_size=1 keeps this cheap; we cache the resulting 3-vec.
    model = smplx.SMPLX(
        model_path,
        gender=gender,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
        num_betas=16,
    )
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas.reshape(1, 16)),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            jaw_pose=torch.zeros(1, 3),
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3),
            expression=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
        )
    pelvis = out.joints[0, 0].detach().cpu().numpy().astype(np.float32)  # (3,)
    _PELVIS_OFFSET_CACHE[key] = pelvis
    return pelvis


def se3_to_axis_angle(
    Q: np.ndarray,
    betas: np.ndarray,
    gender: str,
    smplx_dir: Path,
    parents: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert per-joint world-frame SE(3) back to SMPL-X axis-angle parameters.

    **Rotation inverse (shape-independent).** SMPL-X FK computes world
    transforms via ``A[k] = A[parent(k)] @ T_k``, where ``T_k``'s rotation
    block is the **local** joint rotation (axis-angle converted to rotmat;
    pelvis uses the global root orientation). Inverting:

        R_local[k] = R_global[parent(k)].T @ R_global[k]   for k >= 1
        R_local[0] = R_global[0]                           (root = global)

    recovers the per-joint local rotation matrix that produced ``A[k]``.
    Each ``R_local`` is then converted to a rotation vector (axis-angle).

    **Translation inverse (shape-dependent).** In SMPL-X, the pelvis WORLD
    position (which is what ``global_jpos[:, 0]`` stores, and therefore what
    ``Q[:, 0, :3, 3]`` carries) is::

        global_jpos[:, 0] = J_rest_shaped[0] + root_trans

    where ``J_rest_shaped = J_regressor @ (v_template + shapedirs @ betas)``
    is the (betas, gender)-blended rest-pose joint table. Inverting requires
    subtracting the pelvis rest offset::

        root_trans = Q[:, 0, :3, 3] - J_rest_shaped[0]

    This matches CHOIS ``convert.py:355`` which computes
    ``root_trans = global_jpos[:, 0] + trans2joint`` with
    ``trans2joint = -J_rest_shaped[0]``. Therefore this function needs the
    subject's ``(gender, betas)`` plus a path to the SMPL-X model dir.

    Args:
        Q: shape (T, 22, 4, 4) float, world-frame per-joint SE(3). Usually
            the output of :func:`build_q_from_pickle_fields`.
        betas: shape (16,) float32, SMPL-X shape coefficients (subject-
            specific). OMOMO stores these in each npz as ``betas``.
        gender: ``"male" | "female" | "neutral"`` — the subject gender used
            to pick ``SMPLX_{GENDER}.npz`` for the rest-offset lookup.
        smplx_dir: directory containing ``SMPLX_{GENDER}.npz`` model files.
        parents: shape (22,) int32. Kinematic parent index per joint; -1 for
            root. Defaults to :data:`SMPLX_BODY_PARENTS`.

    Returns:
        Tuple of three arrays (all float32):
            - root_trans: shape (T, 3), pelvis world translation
              ``= Q[:, 0, :3, 3] - J_rest_shaped[0]``.
            - root_orient_aa: shape (T, 3), root axis-angle
              ``= rotvec(Q[:, 0, :3, :3])``.
            - pose_body_aa: shape (T, 21, 3), body-joint axis-angle for
              joints 1..21 (SMPL-X body convention).

    Raises:
        ValueError: if ``Q`` shape != (T, 22, 4, 4), ``parents`` shape != (22,),
            or ``betas`` size != 16.
        ImportError: if ``smplx`` is not installed (needed for the shape-
            dependent pelvis offset).
        FileNotFoundError: if ``smplx_dir / SMPLX_{GENDER}.npz`` is missing.
    """
    if Q.ndim != 4 or Q.shape[1:] != (22, 4, 4):
        raise ValueError(f"Q must have shape (T, 22, 4, 4); got {Q.shape}")

    if parents is None:
        parents = SMPLX_BODY_PARENTS
    parents = np.asarray(parents, dtype=np.int32)
    if parents.shape != (22,):
        raise ValueError(f"parents must have shape (22,); got {parents.shape}")
    if int(parents[0]) != -1:
        raise ValueError(f"parents[0] must be -1 (root); got {int(parents[0])}")

    T = Q.shape[0]

    # Extract world rotation and translation blocks.
    R_global = Q[:, :, :3, :3].astype(np.float32, copy=False)  # (T, 22, 3, 3)

    # --- translation inverse: subtract shape-blended pelvis rest offset ---
    pelvis_rest = _get_pelvis_rest_offset(betas, gender, Path(smplx_dir))  # (3,)
    root_trans = (
        Q[:, 0, :3, 3].astype(np.float32, copy=False) - pelvis_rest[None, :]
    ).astype(np.float32)  # (T, 3)

    # --- rotation inverse: per-joint local rotmat -> axis-angle -----------
    # Build local rotations per joint. Root (k=0) has no parent: local = global.
    R_local = np.empty_like(R_global)  # (T, 22, 3, 3) float32
    R_local[:, 0] = R_global[:, 0]

    # For k >= 1: R_local[t, k] = R_global[t, parent(k)].T @ R_global[t, k].
    # Vectorize via gather + batched matmul.
    child_idx = np.arange(1, 22, dtype=np.int32)  # (21,)
    parent_idx = parents[1:]  # (21,), parent index for joints 1..21
    R_parent = R_global[:, parent_idx]  # (T, 21, 3, 3)
    R_child = R_global[:, child_idx]  # (T, 21, 3, 3)
    # np.swapaxes transposes the last two axes: gives R_parent^T.
    R_local[:, 1:] = np.matmul(np.swapaxes(R_parent, -1, -2), R_child)

    # Vectorized rotmat -> axis-angle via scipy.
    rot_flat = R_local.reshape(T * 22, 3, 3)  # (T*22, 3, 3)
    aa_flat = R.from_matrix(rot_flat).as_rotvec().astype(np.float32)  # (T*22, 3)
    aa = aa_flat.reshape(T, 22, 3)  # (T, 22, 3)

    root_orient_aa = aa[:, 0].astype(np.float32, copy=False)  # (T, 3)
    pose_body_aa = aa[:, 1:].astype(np.float32, copy=False)  # (T, 21, 3)

    return root_trans, root_orient_aa, pose_body_aa


# ---------------------------------------------------------------------------
# DA-pose ctx helper — standing body reference for dataset/callback ctx
# ---------------------------------------------------------------------------
# The dataset + eval callback previously fed the model an identity-SE(3)
# body ctx (all 22 joints at world origin with identity rotation). Under
# the new real-SE(3) convention that renders as a "collapsed humanoid blob"
# when the Ctx visualizer LBS-skins the ctx SE(3) directly. This helper
# builds a DA-pose (standing body, arms ~45° down, Z-up world) world-frame
# SE(3) per body joint so the model ctx is a meaningful prior, not garbage.
#
# The result depends only on (gender, betas); we cache by
# (gender, tuple(betas.tolist())) — same pattern as _PELVIS_OFFSET_CACHE
# above — so repeated calls amortize the 500+MB SMPL-X model load. The
# dataset calls this with (gender="neutral", betas=zeros) per sample; the
# cache collapses that to a single FK on first access.
_DA_POSE_Q_CACHE: dict = {}


def da_pose_q(
    gender: str = "neutral",
    betas: Optional[np.ndarray] = None,
    smplx_dir=None,
) -> np.ndarray:
    """Return the DA-pose world-frame SE(3) per body joint.

    Computed via SMPL-X FK at a canonical DA-pose (standing body, arms ~45°
    down, Z-up world). Result is cached by ``(gender, betas)`` so the
    heavy SMPL-X model load happens once per subject at most.

    Pose parameters:
        * ``root_trans = [0, 0, 0]``.
        * ``root_orient_aa = [pi/2, 0, 0]`` — rotate SMPL-X Y-up into our
          Z-up world (SMPL-X's canonical T-pose has +Y pointing head-up;
          wmm uses Z-up, so we rotate 90° about +X).
        * ``pose_body_aa[0..20] = 0`` EXCEPT:
            - ``pose_body_aa[15]`` (L_Shoulder) axis-angle ``[0, 0, -pi/4]``:
              rotates the left shoulder ~45° about z, bringing the arm
              down from T-pose.
            - ``pose_body_aa[16]`` (R_Shoulder) axis-angle ``[0, 0,  pi/4]``:
              mirrors for the right arm.

        NOTE: SMPL-X body-pose indices are 0..20, corresponding to SMPL-X
        joints 1..21 (pelvis = joint 0 is root_orient, not body_pose).
        L_Shoulder = joint 16 -> body-pose index 15. R_Shoulder = joint 17
        -> body-pose index 16.

    Args:
        gender: ``"male" | "female" | "neutral"``. Default neutral.
        betas: shape (16,) float32, SMPL-X shape coefficients. Defaults to
            zeros (mean shape) — shape-independent approximation is used by
            the dataset so every sample shares one canonical ctx pose.
        smplx_dir: directory containing ``SMPLX_{GENDER}.npz``. Required
            (no default — the caller must be explicit about which SMPL-X
            model to load).

    Returns:
        Q_da: shape (22, 4, 4) float32 — world-frame SE(3) per SMPL-X body
            joint at the DA-pose. Can be expanded to (22, C, 4, 4) via
            ``np.broadcast_to(Q_da[:, None], (22, C, 4, 4))`` for dataset
            ctx consumption.

    Raises:
        ValueError: if ``smplx_dir`` is None or ``betas`` has wrong size.
        ImportError: if ``smplx`` is not installed.
        FileNotFoundError: if the SMPL-X model file is missing.
    """
    if smplx_dir is None:
        raise ValueError("da_pose_q: smplx_dir must be provided")
    if betas is None:
        betas = np.zeros(16, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.size != 16:
        raise ValueError(
            f"da_pose_q: betas must be (16,); got size {betas.size}"
        )

    key = (gender.lower(), tuple(betas.tolist()))
    cached = _DA_POSE_Q_CACHE.get(key)
    if cached is not None:
        return cached

    # Build the DA-pose axis-angle inputs. Single frame (T=1).
    root_orient_aa = np.array(
        [[np.pi / 2.0, 0.0, 0.0]], dtype=np.float32
    )                                                                # (1, 3)
    root_trans = np.zeros((1, 3), dtype=np.float32)                  # (1, 3)
    pose_body = np.zeros((1, 21, 3), dtype=np.float32)               # (1, 21, 3)
    pose_body[0, 15] = [0.0, 0.0, -np.pi / 4.0]   # L_Shoulder -45° about Z
    pose_body[0, 16] = [0.0, 0.0,  np.pi / 4.0]   # R_Shoulder +45° about Z

    global_rot_mat, global_jpos = smplx_world_transforms(
        root_trans=root_trans,
        root_orient_aa=root_orient_aa,
        pose_body_aa=pose_body,
        betas=betas,
        gender=gender,
        smplx_dir=smplx_dir,
    )                                                                 # (1,22,3,3), (1,22,3)

    Q_da_1 = build_q_from_pickle_fields(global_rot_mat, global_jpos)  # (1, 22, 4, 4)
    Q_da = Q_da_1[0].astype(np.float32, copy=True)                    # (22, 4, 4)
    _DA_POSE_Q_CACHE[key] = Q_da
    return Q_da


def _verify_parents() -> np.ndarray:
    """Load SMPL-X male model once and return ``parents[:22]`` as int32.

    Used for one-time verification that :data:`SMPLX_BODY_PARENTS` has not
    drifted from the upstream ``smplx`` package. Should be called from a
    developer-side script or test, never from hot-path library code.

    Returns:
        parents: shape (22,) int32, SMPL-X kinematic parent table for body
            joints.

    Raises:
        ImportError: if the ``smplx`` package is not installed.
        FileNotFoundError: if the SMPL-X model file cannot be located.
    """
    try:
        import smplx
    except ImportError as e:
        raise ImportError(
            "smplx is required for _verify_parents(); install it or compare "
            "SMPLX_BODY_PARENTS manually. Underlying error: " + str(e)
        ) from e

    # Resolve repo-root model path. build_q.py lives at
    # <repo>/prepare_data/omomo/build_q.py -> 2 parents up.
    import os.path as osp

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    smplx_dir = osp.join(repo_root, "data_wmm", "omomo", "smplx")

    model = smplx.SMPLX(smplx_dir, gender="male", use_pca=False, flat_hand_mean=True)
    parents = model.parents[:22].cpu().numpy().astype(np.int32)
    return parents


if __name__ == "__main__":
    # Developer convenience: verify parents when run as a script.
    got = _verify_parents()
    if np.array_equal(got, SMPLX_BODY_PARENTS):
        print(f"ok: SMPLX_BODY_PARENTS matches smplx model: {got.tolist()}")
    else:
        print(
            f"FAIL: SMPLX_BODY_PARENTS drift.\n"
            f"  file: {SMPLX_BODY_PARENTS.tolist()}\n"
            f"  model: {got.tolist()}"
        )
