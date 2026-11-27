"""Canonical hand-object sample file schema + I/O helpers.

Version-agnostic eval code. Every sample producer (baselines, UMM sampler, GT loader)
writes to this schema; the single eval pipeline reads it. Bit-identical round-trip
is a hard requirement — validation raises on any shape/dtype mismatch.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)

# Track NPZs that lacked is_lhand/is_rhand so we warn once per file, not once per load.
_LEGACY_HAND_FLAG_WARNED: set = set()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class CanonicalSample:
    """Single hand-object interaction episode in canonical form.

    All time-indexed arrays share the same leading dimension T.

    Fields:
        left_hand_joints:  (T, 21, 3) float32 — world-frame meters, 21 MANO joints w/tip.
        right_hand_joints: (T, 21, 3) float32 — same.
        left_hand_rot:     (T, 16, 3) float32 — axis-angle, 1 global + 15 finger joints.
        right_hand_rot:    (T, 16, 3) float32 — same.
        obj_translation:   (T, 3)     float32 — world-frame meters.
        obj_rotation:      (T, 3, 3)  float32 — rotation matrix, column-vector convention.
        obj_name:          str        — e.g. "airplane", "box".
        valid_mask:        (T,)       bool    — True iff real recorded frame.
        fps:               int        — e.g. 30.
        text:              str        — episode text annotation.
        obj_articulation_angle:
                            (T,) float32 or None — per-frame revolute joint angle
                            (radians) for ARCTIC objects. None for non-articulated
                            datasets (e.g. GRAB). Convention: axis = (0, 0, -1),
                            positive angle rotates the ``top`` part per Text2HOI
                            `get_transformed_obj_pc`.
        is_lhand, is_rhand:
                            bool — which hands are task-active for this episode.
                            Raw Text2HOI does NOT zero idle-hand streams
                            (``j_lhand`` carries full body-rest finger positions
                            even when ``is_lhand=0``). Downstream metrics
                            (phys_realism contact/penetration) must gate on these
                            flags so idle-hand verts never enter contact/penetration
                            scoring. When loading legacy files lacking these keys,
                            both default to ``True``.
                            At least one must be True.
    """

    left_hand_joints: ndarray   # (T, 21, 3) float32
    right_hand_joints: ndarray  # (T, 21, 3) float32
    left_hand_rot: ndarray      # (T, 16, 3) float32
    right_hand_rot: ndarray     # (T, 16, 3) float32
    obj_translation: ndarray    # (T, 3)     float32
    obj_rotation: ndarray       # (T, 3, 3)  float32
    obj_name: str
    valid_mask: ndarray         # (T,)       bool
    fps: int
    text: str
    obj_articulation_angle: Optional[ndarray] = None  # (T,) float32 or None
    is_lhand: bool = True
    is_rhand: bool = True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Expected shape suffixes and dtypes for each array field.
# The leading T dimension is checked separately for consistency.
_ARRAY_FIELD_SPECS: List[tuple] = [
    # (field_name, trailing_shape, dtype)
    ("left_hand_joints",  (21, 3), np.float32),
    ("right_hand_joints", (21, 3), np.float32),
    ("left_hand_rot",     (16, 3), np.float32),
    ("right_hand_rot",    (16, 3), np.float32),
    ("obj_translation",   (3,),    np.float32),
    ("obj_rotation",      (3, 3),  np.float32),
    ("valid_mask",        (),      np.bool_),
]


def validate_canonical_sample(sample: CanonicalSample, strict: bool = False) -> None:
    """Validate all fields of a CanonicalSample. Raises ValueError on mismatch.

    Checks:
    - Shape and dtype of every array field.
    - T consistency across all time-dimensioned fields.
    - fps is a positive int.
    - obj_name and text are Python str.

    When strict=True, additionally checks for each valid-mask frame:
    - obj_rotation[t] is orthonormal: max |R @ R.T - I| < 1e-4
    - det(obj_rotation[t]) > 0 (proper rotation, no reflections)

    strict=False (default) preserves backward compatibility.

    obj_name and text may be empty strings (valid sentinel for unlabeled samples).
    """
    # Collect T values from each array field.
    t_values: Dict[str, int] = {}

    for field_name, trailing_shape, expected_dtype in _ARRAY_FIELD_SPECS:
        arr: ndarray = getattr(sample, field_name)
        if not isinstance(arr, ndarray):
            raise ValueError(
                f"canonical.{field_name}: expected numpy.ndarray, got {type(arr).__name__}"
            )

        # valid_mask is (T,) bool — validated by the general shape/dtype path below.

        # Build the full expected shape using inferred T.
        if arr.ndim == 0:
            raise ValueError(
                f"canonical.{field_name}: expected at least 1 dimension, got scalar"
            )

        T = arr.shape[0]
        t_values[field_name] = T

        expected_ndim = 1 + len(trailing_shape)
        expected_full_shape = (T,) + trailing_shape

        if arr.shape != expected_full_shape:
            # Report with generic T notation.
            generic = f"(T, {', '.join(str(d) for d in trailing_shape)})" if trailing_shape else "(T,)"
            raise ValueError(
                f"canonical.{field_name}: expected shape {generic}, got {arr.shape}"
            )

        if arr.dtype != expected_dtype:
            raise ValueError(
                f"canonical.{field_name}: expected dtype {expected_dtype}, got {arr.dtype}"
            )

    # Check T consistency.
    unique_Ts = set(t_values.values())
    if len(unique_Ts) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in t_values.items())
        raise ValueError(
            f"canonical: inconsistent T across fields — {detail}"
        )

    # fps must be a positive int (numpy ints accepted via cast).
    fps = sample.fps
    if isinstance(fps, (np.integer,)):
        fps = int(fps)
    if not isinstance(fps, int) or isinstance(fps, bool):
        raise ValueError(
            f"canonical.fps: expected Python int, got {type(sample.fps).__name__}"
        )
    if fps <= 0:
        raise ValueError(f"canonical.fps: must be positive, got {fps}")

    # obj_name and text must be Python str.
    for scalar_field in ("obj_name", "text"):
        val = getattr(sample, scalar_field)
        if not isinstance(val, str):
            raise ValueError(
                f"canonical.{scalar_field}: expected str, got {type(val).__name__}"
            )

    # is_lhand / is_rhand must be Python bool (or numpy bool). At least one True.
    for hand_field in ("is_lhand", "is_rhand"):
        v = getattr(sample, hand_field)
        if isinstance(v, np.bool_):
            v = bool(v)
        if not isinstance(v, bool):
            raise ValueError(
                f"canonical.{hand_field}: expected bool, got {type(getattr(sample, hand_field)).__name__}"
            )
    if not (bool(sample.is_lhand) or bool(sample.is_rhand)):
        raise ValueError(
            "canonical: at least one of is_lhand / is_rhand must be True "
            "(both-False episodes are invalid)"
        )

    # Optional articulation angle: (T,) float32 when present, must match T.
    if sample.obj_articulation_angle is not None:
        arr = sample.obj_articulation_angle
        if not isinstance(arr, ndarray):
            raise ValueError(
                f"canonical.obj_articulation_angle: expected numpy.ndarray or None, "
                f"got {type(arr).__name__}"
            )
        T_ref = next(iter(t_values.values()))  # all T already consistent
        if arr.shape != (T_ref,):
            raise ValueError(
                f"canonical.obj_articulation_angle: expected shape (T={T_ref},), "
                f"got {arr.shape}"
            )
        if arr.dtype != np.float32:
            raise ValueError(
                f"canonical.obj_articulation_angle: expected dtype float32, "
                f"got {arr.dtype}"
            )

    # Optional strict orthonormality check on obj_rotation.
    if strict:
        valid_idx = np.where(sample.valid_mask)[0]
        if len(valid_idx) > 0:
            R = sample.obj_rotation[valid_idx]  # (K, 3, 3)
            eye = np.broadcast_to(np.eye(3, dtype=np.float32), R.shape)
            RRT = np.einsum("...ij,...kj->...ik", R, R)  # R @ R.T
            max_dev = float(np.abs(RRT - eye).max())
            if max_dev >= 1e-4:
                raise ValueError(
                    f"canonical.obj_rotation: not orthonormal in strict mode — "
                    f"max |R@R.T - I| = {max_dev:.2e} (threshold 1e-4)"
                )
            dets = np.linalg.det(R)
            if float(dets.min()) <= 0.0:
                raise ValueError(
                    f"canonical.obj_rotation: improper rotation in strict mode — "
                    f"min det = {dets.min():.6f} (must be > 0)"
                )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_canonical_npz(sample: CanonicalSample, path: str, strict: bool = False) -> None:
    """Validate and save a CanonicalSample to a compressed .npz file.

    Creates parent directories if missing. Raises ValueError on validation
    failure before any write occurs.

    Args:
        sample: The sample to save.
        path:   Destination file path (should end in .npz).
        strict: If True, additionally verify obj_rotation orthonormality and det>0.
    """
    validate_canonical_sample(sample, strict=strict)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    fps = sample.fps
    if isinstance(fps, np.integer):
        fps = int(fps)

    extra: Dict[str, Any] = {}
    if sample.obj_articulation_angle is not None:
        extra["obj_articulation_angle"] = sample.obj_articulation_angle

    np.savez_compressed(
        path,
        left_hand_joints=sample.left_hand_joints,
        right_hand_joints=sample.right_hand_joints,
        left_hand_rot=sample.left_hand_rot,
        right_hand_rot=sample.right_hand_rot,
        obj_translation=sample.obj_translation,
        obj_rotation=sample.obj_rotation,
        obj_name=np.array(sample.obj_name),
        valid_mask=sample.valid_mask,
        fps=np.array(fps),
        text=np.array(sample.text),
        is_lhand=np.array(bool(sample.is_lhand)),
        is_rhand=np.array(bool(sample.is_rhand)),
        **extra,
    )


def load_canonical_npz(path: str) -> CanonicalSample:
    """Load a CanonicalSample from a .npz file.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError:          If any expected field is missing from the file.

    Returns:
        The reconstructed CanonicalSample.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"canonical: file not found: {path}")

    data = np.load(path, allow_pickle=False)

    expected_keys = [
        "left_hand_joints", "right_hand_joints",
        "left_hand_rot", "right_hand_rot",
        "obj_translation", "obj_rotation",
        "obj_name", "valid_mask", "fps", "text",
    ]
    for key in expected_keys:
        if key not in data:
            raise KeyError(
                f"canonical: missing field '{key}' in file {path!r}. "
                f"Present keys: {list(data.keys())}"
            )

    obj_arti = None
    if "obj_articulation_angle" in data.files:
        obj_arti = np.asarray(data["obj_articulation_angle"], dtype=np.float32)

    # Backward compat: legacy canonical files may lack these fields.
    # Default to (True, True) which matches the pre-fix unconditional both-hand
    # decode behavior. Warn LOUDLY once per path so silent bimanual assumption
    # is visible — real + NPZs always contain these keys.
    has_l = "is_lhand" in data.files
    has_r = "is_rhand" in data.files
    if not (has_l and has_r) and path not in _LEGACY_HAND_FLAG_WARNED:
        logger.warning(
            "load_canonical_npz: %s is missing is_lhand/is_rhand keys "
            "(has_l=%s has_r=%s); defaulting both to True. This is a legacy "
            "older file — downstream per-hand gating will treat it as "
            "bimanual, which is wrong for unimanual raw Text2HOI samples. "
            "Re-run the adapter to regenerate a canonical NPZ with the keys.",
            path, has_l, has_r,
        )
        _LEGACY_HAND_FLAG_WARNED.add(path)
    is_lhand = bool(data["is_lhand"]) if has_l else True
    is_rhand = bool(data["is_rhand"]) if has_r else True

    return CanonicalSample(
        left_hand_joints=data["left_hand_joints"],
        right_hand_joints=data["right_hand_joints"],
        left_hand_rot=data["left_hand_rot"],
        right_hand_rot=data["right_hand_rot"],
        obj_translation=data["obj_translation"],
        obj_rotation=data["obj_rotation"],
        obj_name=str(data["obj_name"]),
        valid_mask=data["valid_mask"],
        fps=int(data["fps"]),
        text=str(data["text"]),
        obj_articulation_angle=obj_arti,
        is_lhand=is_lhand,
        is_rhand=is_rhand,
    )


# ---------------------------------------------------------------------------
# Directory validation
# ---------------------------------------------------------------------------

def validate_canonical_dir(dirpath: str, strict: bool = False) -> Dict[str, Any]:
    """Scan all .npz files in dirpath, load and validate each.

    Does not recurse into subdirectories. Collects per-file errors without
    raising (except for mixed-fps and missing/empty dir).

    Args:
        dirpath: Directory containing .npz sample files.
        strict:  If True, forward strict=True to per-sample validation
                 (checks obj_rotation orthonormality and det>0).

    Returns:
        Summary dict with keys: dirpath, n_samples, fps_set, T_stats,
        obj_counts, errors.

    Raises:
        FileNotFoundError: If dirpath does not exist or contains no .npz files.
        ValueError:        If more than one distinct fps value is found.
    """
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"canonical: directory not found: {dirpath}")

    npz_files = sorted(
        f for f in os.listdir(dirpath)
        if f.endswith(".npz") and os.path.isfile(os.path.join(dirpath, f))
    )

    if not npz_files:
        raise FileNotFoundError(
            f"canonical: no .npz files found in directory: {dirpath}"
        )

    fps_set: List[int] = []
    t_list: List[int] = []
    obj_counts: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []

    for fname in npz_files:
        fpath = os.path.join(dirpath, fname)
        try:
            sample = load_canonical_npz(fpath)
            validate_canonical_sample(sample, strict=strict)
        except Exception as exc:
            errors.append({"file": fname, "error": str(exc)})
            continue

        fps_val = int(sample.fps)
        if fps_val not in fps_set:
            fps_set.append(fps_val)

        t_list.append(sample.left_hand_joints.shape[0])
        obj_counts[sample.obj_name] = obj_counts.get(sample.obj_name, 0) + 1

    if len(fps_set) > 1:
        raise ValueError(
            f"canonical: mixed fps values in {dirpath!r}: {sorted(fps_set)}. "
            "This is a producer bug — all samples must share the same fps."
        )

    n_valid = len(t_list)
    t_stats: Dict[str, Any] = {}
    if t_list:
        t_stats = {
            "min": int(min(t_list)),
            "max": int(max(t_list)),
            "mean": float(sum(t_list) / len(t_list)),
        }
    else:
        t_stats = {"min": 0, "max": 0, "mean": 0.0}

    return {
        "dirpath": dirpath,
        "n_samples": n_valid,
        "fps_set": sorted(fps_set),
        "T_stats": t_stats,
        "obj_counts": obj_counts,
        "errors": errors,
    }
