"""Visualization helpers for GRAB (two MANO hands + one rigid object).

Renders left hand + right hand + rigid object + 256-pt PCL overlay in world
frame. Mirrors the interface of :mod:`omomo_viz` (``OmomoVisualizer``) and
reuses ``_make_ground_plane``, ``_camera_pose_diagonal``, ``_look_at_z_up``,
``_color_mesh`` and ``_build_pcl_spheres`` from that module.

This file is **additive only** — it does not modify ``omomo_viz``.

Two rendering paths are supported per-frame, switched on which keys the
batch carries:

* **Path A (MANO axis-angle)** — uses ``smplx.MANO.forward()`` with the
  per-frame ``mano_global_orient_aa_*``, ``mano_hand_pose_aa_*`` and
  ``mano_trans_*`` fields. Robust and matches the eval pipeline.
* **Path B (Q-skinning)** — uses linear blend skinning from the per-joint
  world-frame SE(3) tensors ``Q_l_world`` / ``Q_r_world``. Used when only
  Q tensors are present in the batch (e.g. a model output trace).

Both paths use the per-subject ``v_template`` (loaded from the GRAB
``s{N}_{l,r}hand.ply`` files) to avoid silent shape drift.

The world frame is Z-up (consistent with omomo). ``world_is_mirrored`` is
encoded directly in the dataset's Q / mano_*_aa tensors — no per-renderer
mirror compensation is needed.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# CRITICAL: must be set BEFORE pyrender import. pyrender is imported
# transitively via omomo_viz below; this guard keeps EGL active even if
# someone reorders imports later.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender  # type: ignore
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as _R, Slerp

from umm.models.wmm.viz_helpers.omomo_viz import (
    _build_pcl_spheres,
    _camera_pose_diagonal,
    _color_mesh,
    _look_at_z_up,
    _make_ground_plane,
)


# ---------------------------------------------------------------------------
# SLERP interpolation helpers (fix: smooth viz on stride-4 30fps cache).
# ---------------------------------------------------------------------------

# Default upsample factor: 30fps cache -> 120fps display via 3 sub-frames per
# pair. Matches the original raw GRAB framerate (120fps), recovering the
# fast-manipulation smoothness lost to stride-4 sub-sampling.
_VIZ_UPSAMPLE_FACTOR = 4


def _slerp_se3_array(SE3: np.ndarray, n_sub: int) -> np.ndarray:
    """Interpolate an SE(3) array along its T axis.

    Args:
        SE3: ``(T, 4, 4)`` float homogeneous transforms.
        n_sub: Upsample factor; output T_new = (T-1)*n_sub + 1.

    Returns:
        ``(T_new, 4, 4)`` float SE(3) with SLERP-rotation + linear-translation
        interpolation between adjacent input frames.

    Notes:
        - Endpoints are preserved exactly (start at SE3[0], end at SE3[T-1]).
        - For T < 2 (degenerate), returns a copy of the input.
        - Uses scipy ``Slerp`` (quaternion great-circle interpolation) on the
          rotation block; linear interpolation on translation. This matches
          SE(3) twist-coordinate behavior to first order under small-angle
          increments (~5deg/raw-frame in our data — well within first-order
          regime).
    """
    if SE3.ndim != 3 or SE3.shape[1:] != (4, 4):
        raise ValueError(
            f"_slerp_se3_array: expected (T, 4, 4); got {SE3.shape}"
        )
    T = int(SE3.shape[0])
    if T < 2 or n_sub <= 1:
        return SE3.copy()

    R_in = SE3[:, :3, :3].astype(np.float64)
    t_in = SE3[:, :3, 3].astype(np.float64)
    times = np.arange(T, dtype=np.float64)
    rots = _R.from_matrix(R_in)
    slerp = Slerp(times, rots)

    T_new = (T - 1) * n_sub + 1
    times_new = np.linspace(0.0, float(T - 1), T_new)
    R_new = slerp(times_new).as_matrix()                        # (T_new, 3, 3)
    # Per-axis linear interpolation on translation.
    t_new = np.empty((T_new, 3), dtype=np.float64)
    for ax in range(3):
        t_new[:, ax] = np.interp(times_new, times, t_in[:, ax])

    out = np.zeros((T_new, 4, 4), dtype=SE3.dtype)
    out[:, :3, :3] = R_new.astype(SE3.dtype)
    out[:, :3, 3] = t_new.astype(SE3.dtype)
    out[:, 3, 3] = 1.0
    return out


def _slerp_q_array(Q: np.ndarray, n_sub: int) -> np.ndarray:
    """Interpolate a per-joint SE(3) array ``Q (T, K, 4, 4)`` along T.

    Each joint's SE(3) is interpolated INDEPENDENTLY via
    :func:`_slerp_se3_array`. Output shape: ``((T-1)*n_sub + 1, K, 4, 4)``.

    Assumption — per-joint independence is valid because ``Q_l/r_world`` are
    in WORLD frame (each joint k stores its absolute world rotation and
    position), NOT in local/parent-relative frame. This matches the
    OMOMO convention documented in
    ``prepare_data/handobject_new/augment_world_se3.py:13-16``:

        Q[t, k, :3, :3] = world_rot_k(t)   (absolute, not local)
        Q[t, k, :3,  3] = world_jpos_k(t)  (absolute world position)

    Independent SLERP is therefore correct for the LBS-skinning consumer
    (which composes ``Q_world @ inv_bind`` per joint, with ``inv_bind`` a
    constant translation-only matrix). If a future change converts ``Q`` to
    LOCAL/parent-relative SE(3), per-joint independent SLERP would
    accumulate kinematic-chain errors and a different interpolator (e.g.,
    composing parent + child interpolated in their respective local frames)
    would be required.
    """
    if Q.ndim != 4 or Q.shape[2:] != (4, 4):
        raise ValueError(
            f"_slerp_q_array: expected (T, K, 4, 4); got {Q.shape}"
        )
    T, K = int(Q.shape[0]), int(Q.shape[1])
    if T < 2 or n_sub <= 1:
        return Q.copy()
    T_new = (T - 1) * n_sub + 1
    out = np.zeros((T_new, K, 4, 4), dtype=Q.dtype)
    for k in range(K):
        out[:, k] = _slerp_se3_array(Q[:, k], n_sub)
    return out

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colors / constants
# ---------------------------------------------------------------------------

# Per-domain palette as uint8 RGBA (matches ``_color_mesh`` contract — it
# expects an integer-RGBA tuple, not float). Hand colors are warm/cool to
# disambiguate L vs R; object stays neutral so it does not compete visually.
LHAND_COLOR_RGBA: Tuple[int, int, int, int] = (242, 140, 76, 255)   # warm orange
RHAND_COLOR_RGBA: Tuple[int, int, int, int] = (76, 140, 242, 255)   # cool blue
OBJ_COLOR_RGBA:   Tuple[int, int, int, int] = (180, 180, 184, 255)  # neutral gray
PCL_DEFAULT_RGBA: Tuple[int, int, int, int] = (50, 215, 115, 255)

# SEEN/PRED top-bar indicator colors (RGB uint8). Matches omomo_viz so a
# downstream callback can paint a uniform indicator across all domains.
SEEN_INDICATOR = np.array([120, 190, 120], dtype=np.uint8)   # green
PRED_INDICATOR = np.array([220, 90, 90], dtype=np.uint8)     # red

# ---------------------------------------------------------------------------
# GRAB inference-mask helpers (mirrors omomo_viz _set_*_known pattern)
# ---------------------------------------------------------------------------

# Number of MANO hand joints (matches grab_eval._NUM_HAND_JOINTS).
_NUM_HAND_JOINTS = 16
# Eval keyframe index per protocol (matches grab_eval._KEYFRAME_T).
_EVAL_KEYFRAME_T = 49


def _set_eval_keyframe_known(
    trace_known: torch.Tensor, keyframe_t: int = _EVAL_KEYFRAME_T,
) -> None:
    """Reveal exactly one frame (default t=49) for ALL entities.

    Matches training ``mask_scheduler.eval_keyframe`` (the dominant
    GRAB pattern, weight 4.0). In-place edit of ``trace_known`` shape
    ``(1, T, P)``.
    """
    trace_known.zero_()
    T = int(trace_known.shape[1])
    if 0 <= keyframe_t < T:
        trace_known[0, keyframe_t, :] = True


def _set_o2h_known(
    trace_known: torch.Tensor, num_hand_joints: int = _NUM_HAND_JOINTS,
) -> None:
    """Object known, hands unknown.

    GRAB entity layout: slots [0, num_hand_joints) = L hand,
    [num_hand_joints, 2*num_hand_joints) = R hand,
    [2*num_hand_joints] = obj. Matches training ``mask_scheduler.o2h``.
    In-place edit of ``trace_known`` shape ``(1, T, P)``.
    """
    n_hand = 2 * num_hand_joints
    if trace_known.shape[2] >= n_hand:
        trace_known[0, :, :n_hand] = False


def _set_h2o_known(
    trace_known: torch.Tensor, num_hand_joints: int = _NUM_HAND_JOINTS,
) -> None:
    """Hands known, object unknown. Matches training ``mask_scheduler.h2o``.

    In-place edit of ``trace_known`` shape ``(1, T, P)``.
    """
    obj_idx = 2 * num_hand_joints  # slot 32
    if obj_idx < trace_known.shape[2]:
        trace_known[0, :, obj_idx] = False


def _set_random_prefix_future_known(
    trace_known: torch.Tensor, k: int,
) -> None:
    """Prefix [0, k) revealed; suffix [k, T) unknown for ALL entities.

    Matches training ``mask_scheduler.random_prefix_future`` for a
    SPECIFIC k (training samples k ~ Uniform{1..T-1}; viz uses a fixed
    representative k). In-place edit of ``trace_known`` shape
    ``(1, T, P)``.
    """
    T = int(trace_known.shape[1])
    if T <= 1:
        return
    k = max(1, min(int(k), T - 1))
    trace_known[0, k:, :] = False

# Sphere radius for PCL overlay, world frame (meters). Smaller than omomo
# (15 mm) because the GRAB scene scale is hand-sized, not body-sized.
# 8 mm covered too much of a ~12 cm stapler — verifier could not tell
# whether the mesh was overlaid or buried. 3 mm leaves the gray mesh
# silhouette visible between/through the spheres.
_PCL_SPHERE_RADIUS = 0.003  # 3 mm

# Sanity range for an object mesh's largest extent (meters). GRAB contact
# meshes range from small (~5 cm pen) to large (~50 cm laptop). The upper
# bound catches accidental loads of a raw scan in mm units.
_OBJ_BBOX_MIN = 0.01   # 1 cm
_OBJ_BBOX_MAX = 1.5    # 1.5 m

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
# Rest-template cache for LBS path. Keyed by (hand, vtemp_path).
# Value:
#   rest_verts (778, 3) float64
#   rest_homo  (778, 4) float64
#   lbs_w      (778, 16) float32
#   inv_bind   (16, 4, 4) float64  — translation-only inv-bind
#   faces      (1538, 3) int32
#   J_rest     (16, 3) float64    — kept for diagnostics / PathA fallback
_RestTemplate = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]
_grab_rest_template_cache: Dict[Tuple[str, str], _RestTemplate] = {}

# Per-(hand, vtemp_path) smplx.MANO module cache. Loading a MANO PKL takes
# ~50 ms; we cache so per-batch render is not bottlenecked on it.
_mano_model_cache: Dict[Tuple[str, str], object] = {}


# ---------------------------------------------------------------------------
# MANO loading + LBS template
# ---------------------------------------------------------------------------

def _load_vtemp(vtemp_path: Path) -> np.ndarray:
    """Load a (778, 3) float64 v_template from a GRAB subject PLY.

    Args:
        vtemp_path: absolute path to ``s{N}_{l,r}hand.ply``.

    Returns:
        ``(778, 3) float64`` vertex array.
    """
    if not vtemp_path.exists():
        raise FileNotFoundError(
            f"_load_vtemp: subject vtemp PLY missing: {vtemp_path}"
        )
    mesh = trimesh.load(str(vtemp_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"_load_vtemp: expected a single mesh from {vtemp_path}; "
            f"got {type(mesh).__name__}"
        )
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (N, 3)
    if verts.shape != (778, 3):
        raise ValueError(
            f"_load_vtemp: expected (778, 3) verts in {vtemp_path}; "
            f"got {verts.shape}"
        )
    return verts


def _load_mano_model(
    hand: str,
    vtemp_path: Optional[Path],
    mano_dir: Path,
    flat_hand_mean: bool = True,
):
    """Return a cached ``smplx.MANO`` module for one (hand, vtemp) subject.

    ``hand`` must be ``"l"`` or ``"r"``. ``vtemp_path`` may be ``None`` to
    fall back to the default MANO v_template; otherwise the per-subject
    GRAB vtemp is loaded and passed in via ``v_template=``.

    Args:
        hand: ``"l"`` (left) or ``"r"`` (right).
        vtemp_path: optional per-subject vtemp PLY.
        mano_dir: directory containing ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl``.
        flat_hand_mean: per the GRAB protocol in
            ``src/umm/eval/handobject/physical_realism.py``: GRAB uses
            ``flat_hand_mean=True``.

    Returns:
        ``smplx.MANO`` module (cached).
    """
    if hand not in ("l", "r"):
        raise ValueError(f"_load_mano_model: hand must be 'l' or 'r'; got {hand!r}")
    key = (hand, str(vtemp_path) if vtemp_path is not None else "")
    cached = _mano_model_cache.get(key)
    if cached is not None:
        return cached

    try:
        import smplx  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "_load_mano_model requires smplx. pip install smplx. " + str(e)
        ) from e

    is_rhand = (hand == "r")
    mano_dir = Path(mano_dir)
    if not mano_dir.exists():
        raise FileNotFoundError(
            f"_load_mano_model: mano_dir does not exist: {mano_dir}"
        )
    kwargs = dict(
        model_path=str(mano_dir),
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        batch_size=1,
    )
    if vtemp_path is not None:
        v_template = _load_vtemp(vtemp_path)
        kwargs["v_template"] = torch.from_numpy(
            v_template.astype(np.float32)
        )

    model = smplx.MANO(**kwargs)  # type: ignore[arg-type]
    _mano_model_cache[key] = model
    logger.info(
        "grab_viz: loaded MANO hand=%s vtemp=%s (flat_hand_mean=%s)",
        hand, vtemp_path, flat_hand_mean,
    )
    return model


def _get_grab_rest_template(
    hand: str,
    vtemp_path: Path,
    mano_dir: Path,
    flat_hand_mean: bool = True,
) -> _RestTemplate:
    """Build / fetch the LBS rest template for one GRAB hand subject.

    The template is everything LBS skinning needs except the per-frame
    world-joint transforms ``Q``: rest-pose verts, the homogeneous form
    of those verts, the LBS weights, the translation-only inverse-bind
    matrices, the face table, and ``J_rest`` (kept for diagnostics).

    Cached at module level keyed by ``(hand, vtemp_path_str)``.
    """
    key = (hand, str(vtemp_path))
    cached = _grab_rest_template_cache.get(key)
    if cached is not None:
        return cached

    model = _load_mano_model(hand, vtemp_path, mano_dir, flat_hand_mean=flat_hand_mean)

    # Run forward at zero pose / zero transl to recover the (subject)
    # rest verts and rest joint table.
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, 45),
            betas=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
        )
    rest_verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)  # (778, 3)
    J_rest = out.joints[0].detach().cpu().numpy().astype(np.float64)        # (16, 3)
    if rest_verts.shape != (778, 3):
        raise RuntimeError(
            f"_get_grab_rest_template: expected (778,3) rest verts; got {rest_verts.shape}"
        )
    if J_rest.shape != (16, 3):
        raise RuntimeError(
            f"_get_grab_rest_template: expected (16,3) J_rest; got {J_rest.shape}"
        )

    # MANO LBS weights: (778, 16). All body verts have sum_j w = 1 — no
    # truncation / renormalization needed (unlike SMPL-X body-only viz).
    lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)  # (778, 16)
    if lbs_w.shape != (778, 16):
        raise RuntimeError(
            f"_get_grab_rest_template: lbs_weights shape mismatch {lbs_w.shape}"
        )

    # Translation-only inverse bind: inv_bind[j] = [[I, -J_rest[j]], [0, 1]]
    inv_bind = np.tile(np.eye(4, dtype=np.float64)[None, :, :], (16, 1, 1))  # (16, 4, 4)
    inv_bind[:, :3, 3] = -J_rest                                              # (16, 3)

    rest_homo = np.concatenate(
        [rest_verts, np.ones((778, 1), dtype=np.float64)], axis=1,
    )  # (778, 4)

    faces = np.asarray(model.faces, dtype=np.int32)  # (1538, 3)

    template: _RestTemplate = (
        rest_verts, rest_homo, lbs_w, inv_bind, faces, J_rest,
    )
    _grab_rest_template_cache[key] = template
    return template


def _lbs_skin_grab_q(
    Q_world: np.ndarray,         # (T, 16, 4, 4) float
    rest_template: _RestTemplate,
) -> np.ndarray:
    """Skin GRAB hand verts via LBS from per-joint world-frame SE(3).

    Standard formula:
        M[t, j]      = Q_world[t, j] @ inv_bind[j]               # (T, 16, 4, 4)
        v_world[t,v] = sum_j lbs_weights[v, j] * M[t, j] @ rest_v_homo

    Args:
        Q_world: ``(T, 16, 4, 4)`` per-joint world transforms. Joint 0 is
            wrist; joints 1..15 are the 15 finger joints.
        rest_template: tuple from ``_get_grab_rest_template``.

    Returns:
        ``(T, 778, 3) float32`` world-frame vertex positions.
    """
    rest_verts, rest_homo, lbs_w, inv_bind, _, _ = rest_template
    Q = np.asarray(Q_world, dtype=np.float64)
    if Q.ndim != 4 or Q.shape[1:] != (16, 4, 4):
        raise ValueError(
            f"_lbs_skin_grab_q: Q_world must be (T, 16, 4, 4); got {Q.shape}"
        )
    T = int(Q.shape[0])

    # M[t, j] = Q[t, j] @ inv_bind[j]
    M = np.einsum("tjab,jbc->tjac", Q, inv_bind)                  # (T, 16, 4, 4)
    # v_per_joint[t, v, j, r] = M[t, j, r, c] @ rest_homo[v, c]
    v_per_joint = np.einsum("tjrc,vc->tvjr", M, rest_homo)        # (T, 778, 16, 4)
    v_world_h = np.einsum("vj,tvjr->tvr", lbs_w, v_per_joint)     # (T, 778, 4)
    return v_world_h[..., :3].astype(np.float32)                  # (T, 778, 3)


def _mano_forward_world(
    hand: str,
    mano_dir: Path,
    vtemp_path: Optional[Path],
    transl: np.ndarray,             # (T, 3) float32
    global_orient_aa: np.ndarray,   # (T, 3) float32
    hand_pose_aa: np.ndarray,       # (T, 45) float32
    flat_hand_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Path-A: run smplx.MANO forward and return (verts_world, faces).

    Mirrors the GRAB eval reference at
    ``src/umm/eval/handobject/physical_realism.py``: the per-frame world
    verts are obtained by

        verts_world[t] = verts_canonical[t] - joint0_canonical[t] + transl[t]

    so the wrist-anchored MANO output is repositioned to the world wrist
    location supplied by ``transl``.

    Args:
        hand: ``"l"`` or ``"r"``.
        mano_dir: directory with ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl``.
        vtemp_path: optional per-subject vtemp PLY.
        transl: ``(T, 3)`` world wrist position.
        global_orient_aa: ``(T, 3)`` axis-angle for the wrist.
        hand_pose_aa: ``(T, 45)`` axis-angle for the 15 finger joints.
        flat_hand_mean: GRAB convention.

    Returns:
        ``(verts_world (T, 778, 3) float32, faces (1538, 3) int32)``.
    """
    model = _load_mano_model(hand, vtemp_path, mano_dir, flat_hand_mean=flat_hand_mean)

    transl_t = torch.as_tensor(transl, dtype=torch.float32).reshape(-1, 3)              # (T, 3)
    go_t = torch.as_tensor(global_orient_aa, dtype=torch.float32).reshape(-1, 3)        # (T, 3)
    hp_t = torch.as_tensor(hand_pose_aa, dtype=torch.float32).reshape(-1, 45)           # (T, 45)
    T = int(transl_t.shape[0])
    if go_t.shape[0] != T or hp_t.shape[0] != T:
        raise ValueError(
            f"_mano_forward_world: T mismatch — transl={transl_t.shape}, "
            f"global_orient_aa={go_t.shape}, hand_pose_aa={hp_t.shape}"
        )

    # MANO is configured at batch_size=1 in the cache (cheaper / no
    # recompile per T-step). Loop and stack.
    verts_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for t in range(T):
            out = model(
                global_orient=go_t[t:t+1],
                hand_pose=hp_t[t:t+1],
                betas=torch.zeros(1, 10),
                transl=torch.zeros(1, 3),
            )
            verts_canon = out.vertices[0].detach().cpu().numpy()         # (778, 3)
            joint0_canon = out.joints[0, 0].detach().cpu().numpy()        # (3,)
            v_world = (
                verts_canon
                - joint0_canon[None, :]
                + transl_t[t].cpu().numpy()[None, :]
            )                                                              # (778, 3)
            verts_chunks.append(v_world.astype(np.float32))
    verts_world = np.stack(verts_chunks, axis=0)                          # (T, 778, 3)
    faces = np.asarray(model.faces, dtype=np.int32)                       # (1538, 3)
    return verts_world, faces


# ---------------------------------------------------------------------------
# Object mesh loading
# ---------------------------------------------------------------------------

def _load_obj_mesh(obj_mesh_path: Path) -> trimesh.Trimesh:
    """Load a GRAB contact mesh from disk with a basic sanity check."""
    if not obj_mesh_path.exists():
        raise FileNotFoundError(
            f"_load_obj_mesh: object mesh missing: {obj_mesh_path}"
        )
    mesh = trimesh.load(str(obj_mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"_load_obj_mesh: expected single mesh from {obj_mesh_path}; "
            f"got {type(mesh).__name__}"
        )
    extents = np.asarray(mesh.extents, dtype=np.float64)  # (3,) max-min per axis
    max_ext = float(extents.max()) if extents.size else 0.0
    if not (_OBJ_BBOX_MIN < max_ext < _OBJ_BBOX_MAX):
        raise ValueError(
            f"_load_obj_mesh: {obj_mesh_path} max extent {max_ext:.3f} m "
            f"outside [{_OBJ_BBOX_MIN}, {_OBJ_BBOX_MAX}] — wrong file?"
        )
    return mesh


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def _lookup_in_batch(batch: Dict, key: str):
    """Return ``batch[key]`` if present, else ``batch['extra'][key]`` if present, else None.

     GrabDataset emits world-frame SE(3) tensors (``Q_l_world`` etc.)
    inside the per-sample ``extra`` dict, which CanonicalDatasetOutput.collate
    nests under ``batch['extra'][key] -> list-of-B``. Other domains may emit
    the same keys at top-level ``batch[key] -> (B, ...)``. Probe both shapes
    so grab_viz works for either emission style without forcing a callsite
    branch.
    """
    if key in batch:
        return batch[key]
    extra = batch.get("extra")
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return None


def _resolve_actual_len(batch: Dict, bid: int) -> int:
    """Return the actual sequence length T for ``batch[bid]``.

    Probes (in order): ``obj_se3_world``, ``Q_l_world``, ``mano_trans_l``,
    ``mano_global_orient_aa_l``. Returns the time-axis length of whichever
    is present first. Returns 0 if none are available — the caller decides
    how to handle the empty case (typically: log + return).

    Each key is looked up first at top-level ``batch[key]`` and then under
    ``batch['extra'][key]`` (per ``_lookup_in_batch``).
    """
    for key in (
        "obj_se3_world", "Q_l_world", "mano_trans_l", "mano_global_orient_aa_l",
    ):
        v = _lookup_in_batch(batch, key)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.ndim < 2:
                continue
            return int(v.shape[1])  # (B, T, ...)
        if isinstance(v, (list, tuple)):
            if bid < len(v):
                inner = v[bid]
                if isinstance(inner, torch.Tensor) and inner.ndim >= 1:
                    return int(inner.shape[0])
                if isinstance(inner, np.ndarray) and inner.ndim >= 1:
                    return int(inner.shape[0])
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return int(v.shape[1])
    return 0


# ---------------------------------------------------------------------------
# Scene + render helpers
# ---------------------------------------------------------------------------

def _build_grab_scene(
    lhand_mesh: trimesh.Trimesh,
    rhand_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    width: int,
    height: int,
    pcl_world_xyz: Optional[np.ndarray] = None,
    pcl_rgb: Optional[np.ndarray] = None,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    include_ground: bool = True,
) -> pyrender.Scene:
    """Compose the pyrender scene for one GRAB frame.

    Camera framing uses the combined bbox of both hands + object + PCL.
    Light setup mirrors the omomo viz (key + fill + rim) so the visual
    style is consistent across domains.
    """
    bboxes = [lhand_mesh.bounds, rhand_mesh.bounds, obj_mesh.bounds]  # each (2, 3)
    pts: List[np.ndarray] = []
    for b in bboxes:
        pts.extend([b[0], b[1]])
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        pts.append(np.asarray(pcl_world_xyz, dtype=np.float64).min(axis=0))
        pts.append(np.asarray(pcl_world_xyz, dtype=np.float64).max(axis=0))
    pts_np = np.stack(pts, axis=0)                                # (K, 3)
    bmin = pts_np.min(axis=0)
    bmax = pts_np.max(axis=0)
    center = 0.5 * (bmin + bmax)
    diag = float(np.linalg.norm(bmax - bmin))

    # Plain combined bbox center. Omomo's "55% of body height" bias was a
    # body-frame trick that does not apply to hand-scale geometry — at GRAB
    # scale (~0.3-0.6m diag) we just want the camera looking at the middle
    # of what's actually in frame.
    target = np.array(
        [center[0], center[1], center[2]],
        dtype=np.float64,
    )
    # GRAB scene is hand+object scale — drop omomo's body-scale lower clamp
    # (3.2m / 0.9*diag) which over-zoomed-out for a 0.5m scene. Keep a small
    # safety floor (0.6m) and scale linearly with the bbox diagonal.
    cam_pose = _camera_pose_diagonal(target, dist=max(0.6, 1.5 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )

    if include_ground:
        # Hand-scale ground reference. Larger plane (e.g. omomo's 4m) dwarfs
        # a 0.5m hand+object scene and pulls the camera framing wide.
        scene.add(pyrender.Mesh.from_trimesh(
            _make_ground_plane(half_size=1.0, z=0.0), smooth=False,
        ))

    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(lhand_mesh, LHAND_COLOR_RGBA), smooth=True,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(rhand_mesh, RHAND_COLOR_RGBA), smooth=True,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False,
    ))

    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        pcl_mesh = _build_pcl_spheres(
            pcl_world_xyz, pcl_rgb, radius=float(pcl_radius),
        )
        scene.add(pyrender.Mesh.from_trimesh(pcl_mesh, smooth=False))

    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)

    for offset, inten in (
        (np.array([1.5, -1.5, 2.0]), 4.0),
        (np.array([-2.0, 1.5, 1.5]), 2.0),
        (np.array([0.0, 2.0, 1.0]), 2.0),
    ):
        light = pyrender.DirectionalLight(
            color=np.ones(3, dtype=np.float32), intensity=float(inten),
        )
        scene.add(light, pose=_look_at_z_up(eye=target + offset, target=target))

    return scene


def _render_frame_with_renderer(
    renderer: pyrender.OffscreenRenderer,
    lhand_mesh: trimesh.Trimesh,
    rhand_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    width: int,
    height: int,
    pcl_world_xyz: Optional[np.ndarray] = None,
    pcl_rgb: Optional[np.ndarray] = None,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    include_ground: bool = True,
) -> np.ndarray:
    """Render into an EXISTING ``pyrender.OffscreenRenderer``.

    Caller owns the renderer lifecycle. Returns ``(H, W, 3) uint8``.
    """
    scene = _build_grab_scene(
        lhand_mesh, rhand_mesh, obj_mesh, width, height,
        pcl_world_xyz=pcl_world_xyz,
        pcl_rgb=pcl_rgb,
        pcl_radius=pcl_radius,
        include_ground=include_ground,
    )
    color, _ = renderer.render(scene)
    return color[..., :3].astype(np.uint8)


# ---------------------------------------------------------------------------
# Standalone convenience entry points
# ---------------------------------------------------------------------------

def render_grab_window_jpeg(
    npz_path: "str | os.PathLike",
    frame: int,
    out_path: "str | os.PathLike",
    mano_dir: "str | os.PathLike",
    data_root: "str | os.PathLike",
    width: int = 960,
    height: int = 960,
    quality: int = 92,
    num_pcl_points: int = 256,
    pcl_seed: int = 0,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    flat_hand_mean: bool = True,
) -> Dict[str, object]:
    """Render one GRAB frame from a canonical npz to JPEG.

    Uses Path-A (MANO axis-angle) — robust and exactly matches the eval
    pipeline's hand reconstruction.
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    mano_dir = Path(mano_dir)
    data_root = Path(data_root)

    d = np.load(str(npz_path), allow_pickle=True)
    T = int(d["Q_l_world"].shape[0])
    frame = int(min(max(frame, 0), T - 1))

    lvtemp = data_root / str(d["world_lhand_vtemp_path"].item())
    rvtemp = data_root / str(d["world_rhand_vtemp_path"].item())
    obj_path = data_root / str(d["world_obj_mesh_path"].item())

    # Path-A MANO forward at this single frame (T=1).
    transl_l = np.asarray(d["mano_trans_l"], dtype=np.float32)[frame:frame+1]
    go_l = np.asarray(d["mano_global_orient_aa_l"], dtype=np.float32)[frame:frame+1]
    hp_l = np.asarray(d["mano_hand_pose_aa_l"], dtype=np.float32)[frame:frame+1]
    transl_r = np.asarray(d["mano_trans_r"], dtype=np.float32)[frame:frame+1]
    go_r = np.asarray(d["mano_global_orient_aa_r"], dtype=np.float32)[frame:frame+1]
    hp_r = np.asarray(d["mano_hand_pose_aa_r"], dtype=np.float32)[frame:frame+1]

    lverts, lfaces = _mano_forward_world(
        "l", mano_dir, lvtemp, transl_l, go_l, hp_l, flat_hand_mean=flat_hand_mean,
    )
    rverts, rfaces = _mano_forward_world(
        "r", mano_dir, rvtemp, transl_r, go_r, hp_r, flat_hand_mean=flat_hand_mean,
    )

    lhand_mesh = trimesh.Trimesh(
        vertices=lverts[0].astype(np.float64), faces=lfaces, process=False,
    )
    rhand_mesh = trimesh.Trimesh(
        vertices=rverts[0].astype(np.float64), faces=rfaces, process=False,
    )

    obj_se3 = np.asarray(d["obj_se3_world"], dtype=np.float64)[frame]   # (4, 4)
    obj_R = obj_se3[:3, :3]
    obj_t = obj_se3[:3, 3]
    obj_rest = _load_obj_mesh(obj_path)
    V_rest = np.asarray(obj_rest.vertices, dtype=np.float64)            # (V, 3)
    # Standard active SE(3) application: V_w = V @ R.T + t (== R @ V_col + t).
    # Cache obj_se3_world stores R_active (see augment_world_se3.py:_build_obj_se3).
    V_w = V_rest @ obj_R.T + obj_t[None, :]
    obj_mesh = trimesh.Trimesh(
        vertices=V_w, faces=np.asarray(obj_rest.faces), process=False,
    )

    # PCL overlay
    pcl_local = np.asarray(d["obj_pcl"], dtype=np.float64)              # (2048, 3)
    rng = np.random.default_rng(int(pcl_seed))
    idx = rng.choice(int(pcl_local.shape[0]), size=int(num_pcl_points), replace=False)
    pcl_local_sub = pcl_local[idx]                                      # (256, 3)
    pcl_world = pcl_local_sub @ obj_R.T + obj_t[None, :]                # (256, 3)
    pcl_rgb = _pcl_colormap(int(num_pcl_points))                         # (256, 3) uint8

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        frame_rgb = _render_frame_with_renderer(
            renderer, lhand_mesh, rhand_mesh, obj_mesh, width, height,
            pcl_world_xyz=pcl_world, pcl_rgb=pcl_rgb, pcl_radius=pcl_radius,
        )
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).convert("RGB").save(
        str(out_path), format="JPEG", quality=int(quality),
    )

    return {
        "out_path": str(out_path),
        "frame": frame,
        "T": T,
        "obj_mesh_path": str(obj_path),
        "lvtemp_path": str(lvtemp),
        "rvtemp_path": str(rvtemp),
        "text_detailed": str(d["text_detailed"].item()),
    }


def render_grab_window_mp4(
    npz_path: "str | os.PathLike",
    out_path: "str | os.PathLike",
    mano_dir: "str | os.PathLike",
    data_root: "str | os.PathLike",
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    num_pcl_points: int = 256,
    pcl_seed: int = 0,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    flat_hand_mean: bool = True,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """Render the full canonical npz to an MP4 (Path-A)."""
    import imageio  # type: ignore[import-not-found]

    npz_path = Path(npz_path)
    out_path = Path(out_path)
    mano_dir = Path(mano_dir)
    data_root = Path(data_root)

    d = np.load(str(npz_path), allow_pickle=True)
    T = int(d["Q_l_world"].shape[0])
    if max_frames is not None:
        T = int(min(T, max_frames))

    lvtemp = data_root / str(d["world_lhand_vtemp_path"].item())
    rvtemp = data_root / str(d["world_rhand_vtemp_path"].item())
    obj_path = data_root / str(d["world_obj_mesh_path"].item())

    transl_l = np.asarray(d["mano_trans_l"], dtype=np.float32)[:T]
    go_l = np.asarray(d["mano_global_orient_aa_l"], dtype=np.float32)[:T]
    hp_l = np.asarray(d["mano_hand_pose_aa_l"], dtype=np.float32)[:T]
    transl_r = np.asarray(d["mano_trans_r"], dtype=np.float32)[:T]
    go_r = np.asarray(d["mano_global_orient_aa_r"], dtype=np.float32)[:T]
    hp_r = np.asarray(d["mano_hand_pose_aa_r"], dtype=np.float32)[:T]

    lverts, lfaces = _mano_forward_world(
        "l", mano_dir, lvtemp, transl_l, go_l, hp_l, flat_hand_mean=flat_hand_mean,
    )
    rverts, rfaces = _mano_forward_world(
        "r", mano_dir, rvtemp, transl_r, go_r, hp_r, flat_hand_mean=flat_hand_mean,
    )
    obj_rest = _load_obj_mesh(obj_path)
    V_rest = np.asarray(obj_rest.vertices, dtype=np.float64)
    obj_faces = np.asarray(obj_rest.faces)
    obj_se3 = np.asarray(d["obj_se3_world"], dtype=np.float64)[:T]      # (T, 4, 4)
    pcl_local = np.asarray(d["obj_pcl"], dtype=np.float64)              # (2048, 3)
    rng = np.random.default_rng(int(pcl_seed))
    idx = rng.choice(int(pcl_local.shape[0]), size=int(num_pcl_points), replace=False)
    pcl_local_sub = pcl_local[idx]                                       # (256, 3)
    pcl_rgb = _pcl_colormap(int(num_pcl_points))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            str(out_path), fps=int(fps), codec="libx264", quality=8,
        ) as writer:
            for t in range(T):
                lh = trimesh.Trimesh(
                    vertices=lverts[t].astype(np.float64), faces=lfaces,
                    process=False,
                )
                rh = trimesh.Trimesh(
                    vertices=rverts[t].astype(np.float64), faces=rfaces,
                    process=False,
                )
                R = obj_se3[t, :3, :3]
                tr = obj_se3[t, :3, 3]
                V_w = V_rest @ R.T + tr[None, :]  # active SE3 (cache R_active)
                obj = trimesh.Trimesh(
                    vertices=V_w, faces=obj_faces, process=False,
                )
                pcl_world = pcl_local_sub @ R.T + tr[None, :]
                frame_rgb = _render_frame_with_renderer(
                    renderer, lh, rh, obj, width, height,
                    pcl_world_xyz=pcl_world, pcl_rgb=pcl_rgb, pcl_radius=pcl_radius,
                )
                writer.append_data(frame_rgb)
    finally:
        renderer.delete()

    return {
        "out_path": str(out_path),
        "frames": T,
        "obj_mesh_path": str(obj_path),
    }


def _pcl_colormap(n: int) -> np.ndarray:
    """Return an ``(n, 3) uint8`` colormap (turbo) over point indices.

    Uses matplotlib's turbo colormap when available, else falls back to a
    simple HSV-style ramp. Result is in [0, 255].
    """
    try:
        import matplotlib.cm as cm  # type: ignore[import-not-found]
        cmap = cm.get_cmap("turbo")
        t = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
        rgba = cmap(t)                                              # (n, 4) float
        return (np.clip(rgba[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    except (ImportError, ValueError) as e:
        logger.warning(
            "grab_viz._pcl_colormap: matplotlib turbo unavailable (%s); "
            "falling back to HSV ramp", e,
        )
        h = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
        # Simple HSV→RGB at full saturation/value
        i = (h * 6.0).astype(int) % 6
        f = (h * 6.0) - (h * 6.0).astype(int)
        v = np.ones_like(h)
        p = np.zeros_like(h)
        q = 1.0 - f
        t = f
        out = np.zeros((int(n), 3), dtype=np.float64)
        for k, (R, G, B) in enumerate(
            [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
        ):
            mask = (i == k)
            out[mask, 0] = R[mask]
            out[mask, 1] = G[mask]
            out[mask, 2] = B[mask]
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_lr_legend(
    frame: np.ndarray,
    lhand_rgba: Tuple[int, int, int, int] = LHAND_COLOR_RGBA,
    rhand_rgba: Tuple[int, int, int, int] = RHAND_COLOR_RGBA,
) -> np.ndarray:
    """Burn an "L=<color>  R=<color>" legend in the top-right corner.

    Returns a NEW (H, W, 3) uint8 array — input is not mutated. Used by
    the canonical sample frame for unambiguous L/R color verification.

    Falls back to a no-op (returns the input unchanged) if PIL.ImageDraw
    cannot be loaded — never crashes the caller.
    """
    try:
        from PIL import ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError as e:
        logger.warning(
            "grab_viz._draw_lr_legend: PIL.ImageDraw unavailable (%s); "
            "skipping legend", e,
        )
        return frame.copy()

    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a TrueType font; PIL falls back to a tiny default font
    # when the .ttf is unavailable. Both produce readable letters at the
    # sizes we use here.
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 18)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    W, H = img.size
    pad = 6
    box_w = 110
    box_h = 28
    x0 = W - box_w - pad
    y0 = pad
    # Background rectangle for legibility against varied backgrounds.
    draw.rectangle(
        [(x0, y0), (x0 + box_w, y0 + box_h)],
        fill=(255, 255, 255), outline=(40, 40, 40), width=1,
    )
    draw.text(
        (x0 + 6, y0 + 4), "L",
        fill=(int(lhand_rgba[0]), int(lhand_rgba[1]), int(lhand_rgba[2])),
        font=font,
    )
    draw.text(
        (x0 + 22, y0 + 4), "=L-hand",
        fill=(60, 60, 60), font=font,
    )
    draw.text(
        (x0 + 6, y0 + 4), "L",
        fill=(int(lhand_rgba[0]), int(lhand_rgba[1]), int(lhand_rgba[2])),
        font=font,
    )
    # Second line: "R = R-hand" in the right-hand color.
    y1 = y0 + box_h
    box2_h = 24
    draw.rectangle(
        [(x0, y1), (x0 + box_w, y1 + box2_h)],
        fill=(255, 255, 255), outline=(40, 40, 40), width=1,
    )
    draw.text(
        (x0 + 6, y1 + 2), "R",
        fill=(int(rhand_rgba[0]), int(rhand_rgba[1]), int(rhand_rgba[2])),
        font=font,
    )
    draw.text(
        (x0 + 22, y1 + 2), "=R-hand",
        fill=(60, 60, 60), font=font,
    )

    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# GrabVisualizer — training-time callback class
# ---------------------------------------------------------------------------

class GrabVisualizer:
    """Mesh-based visualizer for GRAB hand-object interaction sequences.

    Mirrors the interface of :class:`OmomoVisualizer`. Two render paths
    (Path-A MANO axis-angle, Path-B Q-skinning) are auto-selected per
    batch item from which keys are present.

    Args:
        pl_module: Parent PyTorch Lightning module (``PLModel``).
        grab_cfg: dict from ``model.grab`` or ``visualize.grab``. Optional
            keys (defaults shown):

                mano_dir          (REQUIRED, no default)
                data_root         (default "data_wmm/handobject/imos_deps")
                width             (default 512)
                height            (default 512)
                num_pcl_points    (default 256)
                pcl_seed          (default 0)
                pcl_radius        (default 0.008)
                flat_hand_mean    (default True)

            If the dict is empty / falsey, the visualizer is disabled
            (``self.flag = False``) and no setup is performed.
    """

    def __init__(self, pl_module, grab_cfg: Optional[Dict]):
        self.pl = pl_module
        if not grab_cfg:
            self.flag = False
            return
        if "mano_dir" not in grab_cfg:
            raise KeyError(
                "GrabVisualizer: grab_cfg['mano_dir'] is required "
                "(directory containing MANO_LEFT.pkl / MANO_RIGHT.pkl)"
            )
        self.flag = True
        self.mano_dir = Path(grab_cfg["mano_dir"])
        self.data_root = Path(
            grab_cfg.get("data_root", "data_wmm/handobject/imos_deps"),
        )
        self._W = int(grab_cfg.get("width", 512))
        self._H = int(grab_cfg.get("height", 512))
        self._num_pcl_points = int(grab_cfg.get("num_pcl_points", 256))
        self._pcl_seed = int(grab_cfg.get("pcl_seed", 0))
        self._pcl_radius = float(grab_cfg.get("pcl_radius", _PCL_SPHERE_RADIUS))
        self._flat_hand_mean = bool(grab_cfg.get("flat_hand_mean", True))

        # Cached pyrender renderer + threading lock (matches omomo).
        self._render_lock = threading.Lock()
        self._cached_renderer: Optional[pyrender.OffscreenRenderer] = None

        logger.info(
            "GrabVisualizer wired: mano_dir=%s data_root=%s W=%d H=%d "
            "num_pcl_points=%d flat_hand_mean=%s",
            self.mano_dir, self.data_root, self._W, self._H,
            self._num_pcl_points, self._flat_hand_mean,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_gt(self, batch: Dict, bid: int = 0) -> np.ndarray:
        """Render ``(T, H, W, 3) uint8`` GT frames for ``batch[bid]``.

        Decides per-batch which path to use:

        * If ``mano_global_orient_aa_l`` (or ``..._r``) is present in the
          batch, run **Path A** (MANO axis-angle forward).
        * Else fall back to **Path B** (LBS skinning from
          ``Q_l_world``/``Q_r_world``).

        Reads from batch:
            * ``Q_l_world`` ``(B, T, 16, 4, 4)`` (Path B) OR
              ``mano_global_orient_aa_l/r`` ``(B, T, 3)`` +
              ``mano_hand_pose_aa_l/r`` ``(B, T, 45)`` +
              ``mano_trans_l/r`` ``(B, T, 3)`` (Path A).
            * ``obj_se3_world`` ``(B, T, 4, 4)``.
            * ``obj_pcl_local`` or ``obj_pcl`` ``(B, 2048, 3)``.
            * ``extra["world_obj_mesh_path"]`` (list[str] of length B).
            * ``extra["world_lhand_vtemp_path"]`` (list[str]).
            * ``extra["world_rhand_vtemp_path"]`` (list[str]).
            * ``extra["world_is_mirrored"]`` (optional list[bool]).
              Informational only — the renderer trusts that upstream
              tensors are already in the correct world frame
              post-mirror; no per-renderer mirror compensation is
              applied. The flag is logged at debug level so its value
              is observable in the trace.

        Time-base contract:
            When BOTH Path-A and Path-B fields are present in the batch
            their T-axes MUST match (same fps, same time alignment).
            Mismatch raises ``ValueError`` — silently rendering with a
            4x speed mismatch (e.g. 30Hz mano vs 120Hz Q in raw GRAB
            npz) is the bug class this guard exists to prevent.
        """
        if not self.flag:
            raise RuntimeError("GrabVisualizer is disabled (empty grab_cfg)")

        extra = batch.get("extra", {}) or {}

        # Resolve per-item paths. All three are required.
        obj_rel = self._extract_str(extra, "world_obj_mesh_path", bid)
        lvtemp_rel = self._extract_str(extra, "world_lhand_vtemp_path", bid)
        rvtemp_rel = self._extract_str(extra, "world_rhand_vtemp_path", bid)
        obj_path = self.data_root / obj_rel
        lvtemp = self.data_root / lvtemp_rel
        rvtemp = self.data_root / rvtemp_rel

        # `world_is_mirrored` is informational — the renderer trusts that
        # upstream Q / mano_*_aa tensors are already in the correct world
        # frame post-mirror. Surface the flag in debug logs so its value
        # is observable.
        is_mirrored: Optional[bool] = None
        if "world_is_mirrored" in extra:
            v = extra["world_is_mirrored"]
            try:
                if isinstance(v, (list, tuple)) and bid < len(v):
                    is_mirrored = bool(v[bid])
                elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > bid:
                    is_mirrored = bool(v[bid])
                else:
                    is_mirrored = bool(v)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "grab_viz.draw_gt: unparseable world_is_mirrored=%r (bid=%d): %s",
                    v, bid, e,
                )
            logger.debug(
                "grab_viz: world_is_mirrored=%s for bid=%d "
                "(renderer trusts upstream tensors)",
                is_mirrored, bid,
            )

        # Object mesh (rest pose) + per-frame SE(3) transform.
        obj_se3_b = self._to_np(self._extract_tensor(batch, "obj_se3_world", bid))
        if obj_se3_b.ndim != 3 or obj_se3_b.shape[1:] != (4, 4):
            raise ValueError(
                f"GrabVisualizer.draw_gt: obj_se3_world must be (T, 4, 4); "
                f"got {obj_se3_b.shape}"
            )
        T = int(obj_se3_b.shape[0])

        # Decide path. Prefer Path A (axis-angle) when available because
        # it matches the eval pipeline byte-for-byte.
        # Lookups consult both batch[key] (legacy domains) and
        # batch['extra'][key] (GrabDataset emits SE(3) tensors via
        # extra so they survive CanonicalDatasetOutput collate).
        path_a = all(
            _lookup_in_batch(batch, k) is not None
            for k in (
                "mano_global_orient_aa_l", "mano_hand_pose_aa_l", "mano_trans_l",
                "mano_global_orient_aa_r", "mano_hand_pose_aa_r", "mano_trans_r",
            )
        )
        path_b = (
            _lookup_in_batch(batch, "Q_l_world") is not None
            and _lookup_in_batch(batch, "Q_r_world") is not None
        )
        if not (path_a or path_b):
            raise KeyError(
                "GrabVisualizer.draw_gt: neither Path-A (mano_*_aa_*) nor "
                "Path-B (Q_l_world / Q_r_world) keys are present in batch"
            )

        # frame-mismatch guard (subsumes prior path_a + path_b time-base
        # check; raises before the dead-code path-A branch is reached):
        #
        # Path-A's `mano_*_aa` fields are sourced from the canonical .npz
        # `mano_trans_l/r`, `mano_global_orient_aa_l/r`, `mano_hand_pose_aa_l/r`
        # — per `prepare_data/handobject_new/grab_to_umm_se3.py:19` these are
        # in OBJECT-CENTRIC frame (DiffH2O 117-d derived). `obj_se3_world` is
        # in WORLD frame. Mixing the two renders the hand in object-centric
        # frame while the obj is in world frame: every frame is mis-aligned.
        # This was the user-flagged "rotation buggy frame-by-frame" suspect
        # (team-lead hypothesis 2026-04-27 00:27 PT) but currently dormant
        # because GrabDataset emits no mano_*_aa keys. Fail loud so a future
        # change that propagates mano_*_aa into batch cannot silently produce
        # mis-framed viz.
        #
        # NOTE: the prior (path_a and path_b) time-base validation block was
        # removed because this guard fires unconditionally on path_a==True
        # and the validation block was therefore unreachable.
        if path_a:
            raise ValueError(
                "GrabVisualizer.draw_gt: mano_*_aa keys present in batch. "
                "These are in OBJECT-CENTRIC frame "
                "(prepare_data/handobject_new/grab_to_umm_se3.py:19) and would "
                "mismatch the WORLD-frame obj_se3_world that this renderer "
                "uses. Use Path-B (Q_l_world / Q_r_world / obj_se3_world) for "
                "world-frame rendering. If a synthetic test legitimately needs "
                "Path-A (e.g. zero-pose rest hand at world wrist position), "
                "construct mano_trans from the wrist column of Q_l_world AND "
                "drop obj_se3_world from the batch — but do not mix the two."
            )

        # Path-A branch retained as dead code (post-guard) so the prior
        # path-select structure is visible. The guard above ensures it
        # can never execute. Future maintainers MUST NOT remove this
        # branch without also removing the guard — they form a unit.
        if path_a:                            # pragma: no cover  (unreachable)
            transl_l = self._to_np(self._extract_tensor(batch, "mano_trans_l", bid))
            go_l = self._to_np(self._extract_tensor(batch, "mano_global_orient_aa_l", bid))
            hp_l = self._to_np(self._extract_tensor(batch, "mano_hand_pose_aa_l", bid))
            transl_r = self._to_np(self._extract_tensor(batch, "mano_trans_r", bid))
            go_r = self._to_np(self._extract_tensor(batch, "mano_global_orient_aa_r", bid))
            hp_r = self._to_np(self._extract_tensor(batch, "mano_hand_pose_aa_r", bid))
            T = int(min(T, transl_l.shape[0]))
            lverts, lfaces = _mano_forward_world(
                "l", self.mano_dir, lvtemp,
                transl_l[:T], go_l[:T], hp_l[:T],
                flat_hand_mean=self._flat_hand_mean,
            )
            rverts, rfaces = _mano_forward_world(
                "r", self.mano_dir, rvtemp,
                transl_r[:T], go_r[:T], hp_r[:T],
                flat_hand_mean=self._flat_hand_mean,
            )
        else:
            # Path-B: LBS from Q world tensors.
            Ql = self._to_np(self._extract_tensor(batch, "Q_l_world", bid))    # (T, 16, 4, 4)
            Qr = self._to_np(self._extract_tensor(batch, "Q_r_world", bid))    # (T, 16, 4, 4)
            T = int(min(T, Ql.shape[0], Qr.shape[0]))
            Ql = Ql[:T]
            Qr = Qr[:T]

            # fix: SLERP-interpolate per-joint Q + obj_se3 to recover
            # the smoothness lost to stride-4 sub-sampling in the cache.
            # The cache stores 30fps stride-4 of raw 120fps GRAB. Fast
            # manipulation (e.g. inspect/pass) shows up to ~5deg/raw-frame
            # smooth motion at 120fps but ~19deg/cache-frame jumps at 30fps —
            # interpolating 3 sub-frames between each cache frame restores
            # the original visual smoothness. Translation: linear; rotation:
            # SLERP. Output: (T-1)*4+1 frames per array.
            #
            # Diagnostic + measurement: see
            #
            #   assets/per_frame_window_000704.json
            # (stride-4 amplification ratio measured 3.82x on the 000704
            # stapler suspect window).
            Ql = _slerp_q_array(Ql, _VIZ_UPSAMPLE_FACTOR)                       # ((T-1)*4+1, 16, 4, 4)
            Qr = _slerp_q_array(Qr, _VIZ_UPSAMPLE_FACTOR)
            obj_se3_b = _slerp_se3_array(obj_se3_b[:T], _VIZ_UPSAMPLE_FACTOR)   # ((T-1)*4+1, 4, 4)
            T = int(Ql.shape[0])

            l_template = _get_grab_rest_template(
                "l", lvtemp, self.mano_dir, flat_hand_mean=self._flat_hand_mean,
            )
            r_template = _get_grab_rest_template(
                "r", rvtemp, self.mano_dir, flat_hand_mean=self._flat_hand_mean,
            )
            lverts = _lbs_skin_grab_q(Ql, l_template)                           # (T, 778, 3)
            rverts = _lbs_skin_grab_q(Qr, r_template)                           # (T, 778, 3)
            lfaces = l_template[4]
            rfaces = r_template[4]

        # Object rest mesh + per-frame transform.
        obj_rest = _load_obj_mesh(obj_path)
        V_rest = np.asarray(obj_rest.vertices, dtype=np.float64)               # (V, 3)
        obj_faces = np.asarray(obj_rest.faces)

        # Resolve PCL source — try canonical key first, then GRAB-internal name.
        pcl_local = None
        for key in ("obj_pcl_local", "obj_pcl"):
            if key in batch:
                pcl_local = self._to_np(self._extract_tensor(batch, key, bid))
                break
        pcl_local_sub: Optional[np.ndarray] = None
        pcl_rgb_u8: Optional[np.ndarray] = None
        if pcl_local is not None:
            if pcl_local.ndim != 2 or pcl_local.shape[1] != 3:
                raise ValueError(
                    f"GrabVisualizer.draw_gt: obj_pcl[_local] must be (N, 3); "
                    f"got {pcl_local.shape}"
                )
            n_pts = min(int(self._num_pcl_points), int(pcl_local.shape[0]))
            rng = np.random.default_rng(int(self._pcl_seed))
            idx = rng.choice(int(pcl_local.shape[0]), size=n_pts, replace=False)
            pcl_local_sub = pcl_local[idx].astype(np.float64)                  # (n_pts, 3)
            pcl_rgb_u8 = _pcl_colormap(n_pts)                                  # (n_pts, 3) uint8

        frames: List[np.ndarray] = []
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            for t in range(T):
                lh = trimesh.Trimesh(
                    vertices=lverts[t].astype(np.float64), faces=lfaces,
                    process=False,
                )
                rh = trimesh.Trimesh(
                    vertices=rverts[t].astype(np.float64), faces=rfaces,
                    process=False,
                )
                R = obj_se3_b[t, :3, :3]
                tr = obj_se3_b[t, :3, 3]
                # Standard active SE(3) application. Cache obj_se3_world stores
                # R_active (passive→active transpose in augment_world_se3.py).
                V_w = V_rest @ R.T + tr[None, :]
                obj_mesh = trimesh.Trimesh(
                    vertices=V_w, faces=obj_faces, process=False,
                )
                pcl_world: Optional[np.ndarray] = None
                if pcl_local_sub is not None:
                    pcl_world = pcl_local_sub @ R.T + tr[None, :]              # (n_pts, 3)
                frame_rgb = _render_frame_with_renderer(
                    self._cached_renderer, lh, rh, obj_mesh, self._W, self._H,
                    pcl_world_xyz=pcl_world,
                    pcl_rgb=pcl_rgb_u8,
                    pcl_radius=self._pcl_radius,
                )
                frames.append(frame_rgb)

        return np.stack(frames, axis=0)                                       # (T, H, W, 3)

    def draw_inference(
        self,
        batch: Dict,
        pred_dict: Optional[Dict],
        bid: int = 0,
    ) -> np.ndarray:
        """Render predicted-trace frames the same way as ``draw_gt``.

        ``pred_dict`` is expected to carry overrides for any subset of
        ``Q_l_world``, ``Q_r_world``, ``obj_se3_world``,
        ``mano_global_orient_aa_{l,r}``, ``mano_hand_pose_aa_{l,r}``,
        ``mano_trans_{l,r}``. Whatever it provides is overlaid onto the
        original batch and ``draw_gt`` does the rendering.

        ``pred_dict=None`` → returns GT (useful as a dry-run sentinel).

        Important: when ``pred_dict`` carries any Q-form predictions
        (``Q_l_world`` / ``Q_r_world`` / ``obj_se3_world``), all GT
        ``mano_*_aa_*`` keys are stripped from the merged batch BEFORE
        calling :meth:`draw_gt`. Otherwise Path-A would be selected
        (since GT MANO axis-angle is still present) and the Q-form
        predictions would be silently ignored.
        """
        if not self.flag:
            raise RuntimeError("GrabVisualizer is disabled (empty grab_cfg)")
        if pred_dict is None:
            return self.draw_gt(batch, bid=bid)
        sub = dict(batch)
        for k, v in pred_dict.items():
            sub[k] = v

        # If pred provides Q form, drop GT MANO axis-angle to avoid
        # silently rendering GT instead of predictions.
        Q_PRED_KEYS = ("Q_l_world", "Q_r_world", "obj_se3_world")
        MANO_AA_KEYS = (
            "mano_global_orient_aa_l", "mano_hand_pose_aa_l", "mano_trans_l",
            "mano_global_orient_aa_r", "mano_hand_pose_aa_r", "mano_trans_r",
        )
        pred_has_q = any(k in pred_dict for k in Q_PRED_KEYS)
        pred_has_mano = any(k in pred_dict for k in MANO_AA_KEYS)
        if pred_has_q and not pred_has_mano:
            # Partial-Q safety: Path-B requires all of Q_l_world,
            # Q_r_world, obj_se3_world. If pred carries only some of
            # them, raise loudly rather than letting Path-B fail with
            # a confusing KeyError downstream.
            missing_q = [k for k in Q_PRED_KEYS if k not in pred_dict]
            if missing_q:
                raise ValueError(
                    f"GrabVisualizer.draw_inference: pred_dict provides "
                    f"partial Q-form prediction (missing {missing_q}); "
                    f"cannot mix with GT MANO. Provide all Q_*_world keys "
                    f"or all mano_*_aa keys."
                )
            for k in MANO_AA_KEYS:
                sub.pop(k, None)
            logger.debug(
                "grab_viz.draw_inference: pred is Q-form (%s); stripped "
                "GT mano_*_aa keys to force Path-B rendering",
                [k for k in Q_PRED_KEYS if k in pred_dict],
            )
        return self.draw_gt(sub, bid=bid)

    def draw_ctx(self, batch: Dict, bid: int = 0) -> np.ndarray:
        """Render a single-frame context image (frame 0, no ground plane).

        Used as the "abstract reference" panel — same role as
        :meth:`OmomoVisualizer._render_ctx_panel`. The image carries no
        caption text; callers can paint a caption on top if desired.
        """
        if not self.flag:
            raise RuntimeError("GrabVisualizer is disabled (empty grab_cfg)")

        extra = batch.get("extra", {}) or {}
        obj_rel = self._extract_str(extra, "world_obj_mesh_path", bid)
        lvtemp_rel = self._extract_str(extra, "world_lhand_vtemp_path", bid)
        rvtemp_rel = self._extract_str(extra, "world_rhand_vtemp_path", bid)
        obj_path = self.data_root / obj_rel
        lvtemp = self.data_root / lvtemp_rel
        rvtemp = self.data_root / rvtemp_rel

        # First-frame reconstruction. Prefer Path-A.
        # Use _lookup_in_batch so `extra`-emitted keys (Q_l_world etc.
        # via GrabDataset.grab_extra) are found alongside any top-level
        # batch[key] emission.
        if all(
            _lookup_in_batch(batch, k) is not None
            for k in (
                "mano_global_orient_aa_l", "mano_hand_pose_aa_l", "mano_trans_l",
                "mano_global_orient_aa_r", "mano_hand_pose_aa_r", "mano_trans_r",
            )
        ):
            transl_l = self._to_np(self._extract_tensor(batch, "mano_trans_l", bid))[:1]
            go_l = self._to_np(self._extract_tensor(batch, "mano_global_orient_aa_l", bid))[:1]
            hp_l = self._to_np(self._extract_tensor(batch, "mano_hand_pose_aa_l", bid))[:1]
            transl_r = self._to_np(self._extract_tensor(batch, "mano_trans_r", bid))[:1]
            go_r = self._to_np(self._extract_tensor(batch, "mano_global_orient_aa_r", bid))[:1]
            hp_r = self._to_np(self._extract_tensor(batch, "mano_hand_pose_aa_r", bid))[:1]
            lverts, lfaces = _mano_forward_world(
                "l", self.mano_dir, lvtemp, transl_l, go_l, hp_l,
                flat_hand_mean=self._flat_hand_mean,
            )
            rverts, rfaces = _mano_forward_world(
                "r", self.mano_dir, rvtemp, transl_r, go_r, hp_r,
                flat_hand_mean=self._flat_hand_mean,
            )
        else:
            Ql = self._to_np(self._extract_tensor(batch, "Q_l_world", bid))[:1]
            Qr = self._to_np(self._extract_tensor(batch, "Q_r_world", bid))[:1]
            l_template = _get_grab_rest_template(
                "l", lvtemp, self.mano_dir, flat_hand_mean=self._flat_hand_mean,
            )
            r_template = _get_grab_rest_template(
                "r", rvtemp, self.mano_dir, flat_hand_mean=self._flat_hand_mean,
            )
            lverts = _lbs_skin_grab_q(Ql, l_template)
            rverts = _lbs_skin_grab_q(Qr, r_template)
            lfaces = l_template[4]
            rfaces = r_template[4]

        obj_se3_b = self._to_np(self._extract_tensor(batch, "obj_se3_world", bid))[0]   # (4, 4)
        obj_rest = _load_obj_mesh(obj_path)
        V_rest = np.asarray(obj_rest.vertices, dtype=np.float64)
        # Standard active SE(3): cache obj_se3_world stores R_active.
        V_w = V_rest @ obj_se3_b[:3, :3].T + obj_se3_b[:3, 3][None, :]
        obj_mesh = trimesh.Trimesh(
            vertices=V_w, faces=np.asarray(obj_rest.faces), process=False,
        )
        lh = trimesh.Trimesh(
            vertices=lverts[0].astype(np.float64), faces=lfaces, process=False,
        )
        rh = trimesh.Trimesh(
            vertices=rverts[0].astype(np.float64), faces=rfaces, process=False,
        )
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            return _render_frame_with_renderer(
                self._cached_renderer, lh, rh, obj_mesh, self._W, self._H,
                include_ground=False,
            )

    # ------------------------------------------------------------------
    # Training-time callback
    # ------------------------------------------------------------------

    def grab_viz_routine(
        self,
        batch: Dict,
        phase: str,
        batch_idx: int,
        eval_model,
    ) -> None:
        """Training-time visualization callback (GT-only for now).

        Logs:
            * GT video via ``pl.logger.log_video`` (if present).
            * Single-frame ctx panel via ``pl.logger.log_image`` (if present).

        Inference-mode branches are intentionally deferred: the wmm Agent's
        GRAB inference signature is not yet stable in this branch. When it
        lands, mirror :meth:`OmomoVisualizer.omomo_viz_routine`.
        """
        if not self.flag:
            return

        bid = 0
        # Prefer window_idx==0 batch item when available — same convention
        # as omomo_viz_routine.
        extra = batch.get("extra", {}) or {}
        try:
            window_idx_list = extra.get("window_idx")
            if window_idx_list is not None:
                for i, wi in enumerate(window_idx_list):
                    wi_v = int(wi) if wi is not None else -1
                    if wi_v == 0:
                        bid = i
                        break
        except (TypeError, ValueError, KeyError) as e:
            logger.warning(
                "GrabVisualizer.grab_viz_routine: bad window_idx in extra: %s", e,
            )
            bid = 0

        # Degenerate-T guard: an empty trace would render to a (0, 3, H, W)
        # video tensor that crashes log_video. Resolve the actual length
        # the same way draw_gt does (prefers obj_se3_world's T, falling
        # back to Q_l_world / mano_trans_l) and bail out early when zero.
        actual_len = _resolve_actual_len(batch, bid)
        if actual_len <= 0:
            logger.warning(
                "grab_viz_routine: actual_len=%d <= 0 (phase=%s batch=%d); "
                "skipping viz for this batch",
                actual_len, phase, batch_idx,
            )
            return

        # Per UMM viz_try=False rule: errors from draw_gt / draw_ctx must
        # propagate. The outer VizCallback already wraps the entire viz
        # dispatch in try/except when viz_try=True. An inner try/except
        # here would silently swallow data-pipeline bugs during dev.
        # Mirrors omomo_viz_routine's no-inner-try design.
        log_video = getattr(self.pl.logger, "log_video", None)
        if log_video is None:
            logger.warning(
                "grab_viz_routine: pl.logger has no log_video — frames "
                "rendered but not logged (phase=%s batch=%d)",
                phase, batch_idx,
            )
        else:
            frames_gt = self.draw_gt(batch, bid=bid)                       # (T, H, W, 3)
            video_gt = (
                torch.from_numpy(frames_gt).float().permute(0, 3, 1, 2)
                / 255.0
            )
            # (2026-04-27): draw_gt SLERP-upsamples the 200-frame
            # 30 fps cache to 797 frames (4x). Passing fps=40 to log_video
            # keeps the wandb playback duration ~consistent with the prior
            # 200-frame @ fps=10 (~20 s) display rate. Without this override
            # the wandb player would display an 80 s video.
            log_video(
                phase, f"GT_grab_{phase}", video_gt,
                self.pl.global_step, batch_idx, fps=40,
            )

        log_image = getattr(self.pl.logger, "log_image", None)
        if log_image is None:
            logger.warning(
                "grab_viz_routine: pl.logger has no log_image — ctx panel "
                "skipped (phase=%s batch=%d)", phase, batch_idx,
            )
        else:
            ctx_img = self.draw_ctx(batch, bid=bid)                        # (H, W, 3)
            ctx_chw = (
                torch.from_numpy(ctx_img).float().permute(2, 0, 1) / 255.0
            )
            log_image(
                phase, f"Ctx_grab_{phase}", ctx_chw,
                self.pl.global_step, batch_idx,
            )

        # -----------------------------------------------------------
        # inference-mode panels (mirrors OmomoVisualizer).
        # -----------------------------------------------------------
        if log_video is None:
            return  # already warned above

        # 4 GRAB mask patterns matching mask_scheduler.GRAB_PATTERNS:
        #   eval_keyframe (weight 4.0, dominant): only t=49 known
        #   o2h: object known, hands predicted
        #   h2o: hands known, object predicted
        #   random_prefix_future: k-prefix known (representative k = T // 2)
        T_full = int(batch["trace_mask"].shape[1])
        prefix_k = max(1, T_full // 2)
        modes: List[Dict] = [
            {
                "name": "eval_keyframe",
                "caption": (
                    f"GRAB eval_keyframe (only t={_EVAL_KEYFRAME_T} known) "
                    f"[GT vs Pred]"
                ),
                "set_known": _set_eval_keyframe_known,
            },
            {
                "name": "o2h",
                "caption": "GRAB o2h (object known, predict hands)",
                "set_known": _set_o2h_known,
            },
            {
                "name": "h2o",
                "caption": "GRAB h2o (hands known, predict object)",
                "set_known": _set_h2o_known,
            },
            {
                "name": f"random_prefix_future_k{prefix_k}",
                "caption": (
                    f"GRAB random_prefix_future (seen={prefix_k}/{T_full})"
                ),
                "set_known": lambda tk, k=prefix_k: _set_random_prefix_future_known(tk, k),
            },
        ]

        B = int(batch["trace_mask"].shape[0])
        for mode in modes:
            try:
                gen_trace, tk = self._run_inference_one(
                    batch, bid, eval_model, mode["set_known"],
                )
            except Exception as e:
                # Per project rule (no silent swallow): log + skip this mode.
                logger.warning(
                    "grab_viz_routine: inference mode %s failed: %s",
                    mode["name"], e, exc_info=True,
                )
                continue

            # Slice predicted per-entity SE(3): (1, T, 33, 4, 4) -> per-entity tensors
            # at full-batch shape so draw_gt's `_extract_tensor(batch, key, bid)`
            # finds the right (T, ...) slice. Identity-fill all non-bid samples
            # since only bid is rendered.
            T_pred = int(gen_trace.shape[1])
            pred_l = gen_trace[0, :, 0:_NUM_HAND_JOINTS]                       # (T, 16, 4, 4)
            pred_r = gen_trace[0, :, _NUM_HAND_JOINTS:2 * _NUM_HAND_JOINTS]    # (T, 16, 4, 4)
            pred_obj = gen_trace[0, :, 2 * _NUM_HAND_JOINTS]                   # (T, 4, 4)

            eye4 = torch.eye(4, device=gen_trace.device, dtype=gen_trace.dtype)
            Ql_full = eye4.expand(B, T_pred, _NUM_HAND_JOINTS, 4, 4).clone()
            Qr_full = eye4.expand(B, T_pred, _NUM_HAND_JOINTS, 4, 4).clone()
            obj_full = eye4.expand(B, T_pred, 4, 4).clone()
            Ql_full[bid] = pred_l
            Qr_full[bid] = pred_r
            obj_full[bid] = pred_obj

            pred_dict = {
                "Q_l_world": Ql_full,
                "Q_r_world": Qr_full,
                "obj_se3_world": obj_full,
            }
            try:
                frames = self.draw_inference(batch, pred_dict, bid=bid)        # (T_new, H, W, 3) uint8
            except Exception as e:
                logger.warning(
                    "grab_viz_routine: draw_inference mode %s failed: %s",
                    mode["name"], e, exc_info=True,
                )
                continue

            # SEEN/PRED top-bar indicator: paint top 10 rows green when ANY
            # entity is "seen" (known) at frame t_orig, red otherwise.
            # Frames are SLERP-upsampled by _VIZ_UPSAMPLE_FACTOR; map t_new
            # back to t_orig via integer division.
            try:
                tk_np = tk.detach().cpu().numpy()[0]              # (T, P) bool
                seen_per_frame = tk_np.any(axis=1)                # (T,) bool
                n_frames = int(frames.shape[0])
                n_t = int(tk_np.shape[0])
                for t_new in range(n_frames):
                    t_orig = min(t_new // _VIZ_UPSAMPLE_FACTOR, n_t - 1)
                    color = (
                        SEEN_INDICATOR
                        if (t_orig >= 0 and bool(seen_per_frame[t_orig]))
                        else PRED_INDICATOR
                    )
                    frames[t_new, :10, :, :3] = color[None, None, :]
            except Exception as e:
                logger.warning(
                    "grab_viz_routine: failed to paint SEEN/PRED bar "
                    "(mode=%s): %s", mode["name"], e,
                )

            video = (
                torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            )
            log_video(
                phase, f"{mode['name']}_grab_{phase}", video,
                self.pl.global_step, batch_idx,
                caption=mode["caption"], fps=40,
            )

    # ------------------------------------------------------------------
    # Inference dispatcher used by grab_viz_routine
    # ------------------------------------------------------------------

    def _run_inference_one(
        self,
        batch: Dict,
        bid: int,
        eval_model,
        set_known_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run eval_model.inference for one batch item with one masking mode.

        Builds ``trace_known`` / ``context_known`` tensors (all True),
        applies ``set_known_fn`` in-place to trace_known, then calls
        ``eval_model.inference`` with single-sample-sliced inputs (mirrors
        grab_eval call signature).

        Returns:
            gen_trace: ``(1, T, P, 4, 4)`` predicted per-entity SE(3).
            tk: ``(1, T, P)`` bool — final trace_known mask used (returned
                so the caller can paint the SEEN/PRED top-bar indicator).
        """
        tk = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ck = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        set_known_fn(tk)
        # AND with trace_mask so padded (invalid) tokens never get marked
        # known — matches training pattern fns in mask_scheduler.py which
        # all start from trace_mask.clone() (eval_keyframe / o2h / h2o /
        # random_prefix_future). Without this, sequences with
        # valid_T < T_full=200 hand the model a known-mask distribution it
        # was never trained on.
        tk &= batch["trace_mask"][bid:bid + 1].bool()

        local_pcl = batch.get("local_pcl")
        local_pcl_mask = batch.get("local_pcl_mask")
        if local_pcl is not None:
            local_pcl = local_pcl[bid:bid + 1]
            local_pcl_mask = (
                local_pcl_mask[bid:bid + 1] if local_pcl_mask is not None else None
            )

        text_kwarg = None
        if batch.get("text") is not None:
            text_val = batch["text"]
            if isinstance(text_val, list):
                text_kwarg = [text_val[bid]]
            else:
                text_kwarg = text_val[bid:bid + 1]

        _, gen_trace = eval_model.inference(
            trace=batch["trace"][bid:bid + 1],
            context=batch["context"][bid:bid + 1],
            trace_known_mask=tk,
            context_known_mask=ck,
            trace_mask=batch["trace_mask"][bid:bid + 1],
            context_mask=batch["context_mask"][bid:bid + 1],
            real_timestep=batch["time"][bid:bid + 1],
            group_id=batch["trace_group_id"][bid:bid + 1],
            sem_type=batch["trace_sem_type"][bid:bid + 1],
            rot_mask=batch["trace_rot_mask"][bid:bid + 1],
            local_pcl=local_pcl,
            local_pcl_mask=local_pcl_mask,
            text=text_kwarg,
            verbose=False,
        )
        return gen_trace, tk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_str(extra: Dict, key: str, bid: int) -> str:
        """Pull a per-item string from ``extra`` (list-of-str or scalar)."""
        if key not in extra:
            raise KeyError(
                f"GrabVisualizer: required extra key {key!r} missing"
            )
        v = extra[key]
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError(
                    f"GrabVisualizer: extra[{key!r}] is an empty list"
                )
            return str(v[bid] if bid < len(v) else v[-1])
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return str(v[bid] if v.shape[0] > bid else v[-1])
        return str(v)

    @staticmethod
    def _extract_tensor(batch: Dict, key: str, bid: int):
        """Pull ``batch[key][bid]`` returning the torch.Tensor or np.ndarray.

        Looks up via :func:`_lookup_in_batch` so callers can transparently
        emit data either at top-level ``batch[key]`` (B-batched tensor) or
        in ``batch['extra'][key]`` (list-of-B from CanonicalDatasetOutput).

        Raises KeyError when the key is absent in both locations.
        """
        v = _lookup_in_batch(batch, key)
        if v is None:
            raise KeyError(
                f"GrabVisualizer: required batch key {key!r} missing "
                f"(checked batch and batch['extra'])"
            )
        if isinstance(v, (list, tuple)):
            return v[bid]
        return v[bid]

    @staticmethod
    def _to_np(x) -> np.ndarray:
        """Convert a tensor / array / list to a contiguous float64 ndarray."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


__all__ = [
    "GrabVisualizer",
    "render_grab_window_jpeg",
    "render_grab_window_mp4",
    "SEEN_INDICATOR",
    "PRED_INDICATOR",
    "LHAND_COLOR_RGBA",
    "RHAND_COLOR_RGBA",
    "OBJ_COLOR_RGBA",
    "_draw_lr_legend",
]
