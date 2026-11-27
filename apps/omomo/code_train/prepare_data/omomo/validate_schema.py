"""OMOMO window schema validator.

Checks shape and dtype of every field ``convert.py`` writes into each ``.npz`` of a folder.

Usage (from ``apps/omomo/code_train``)::

    python -m prepare_data.omomo.validate_schema --npz_dir <data_wmm/omomo_v3/test_npz>
    python -m prepare_data.omomo.validate_schema --npz_dir <.../train_npz> --limit 5
"""

import argparse
import glob
import os
import os.path as osp
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
# Each entry: field_name -> (shape_fn, dtype)
# shape_fn(T, actual_len) -> tuple  (use T=120)

T_FIXED = 120  # always 120 after padding

SCHEMA: Dict[str, Tuple[Callable[[int, int], tuple], Any]] = {
    # time-varying padded fields
    "root_trans":        (lambda T, L: (T, 3),        np.float32),
    "root_orient_aa":    (lambda T, L: (T, 3),        np.float32),
    "pose_body_aa":      (lambda T, L: (T, 21, 3),    np.float32),
    "lhand_wrist_jpos":  (lambda T, L: (T, 3),        np.float32),
    "rhand_wrist_jpos":  (lambda T, L: (T, 3),        np.float32),
    "obj_trans":         (lambda T, L: (T, 3),        np.float32),
    "obj_rot":           (lambda T, L: (T, 3, 3),     np.float32),
    "obj_scale":         (lambda T, L: (T,),          np.float32),
    "Q":                 (lambda T, L: (T, 22, 4, 4), np.float32),
    # static numeric
    "betas":             (lambda T, L: (16,),          np.float32),
    "obj_bps":           (lambda T, L: (1024, 3),      np.float32),
    # waypoints / first-frame
    "waypoint_indices":  (lambda T, L: (5,),           np.int32),
    "first_root_trans":  (lambda T, L: (3,),           np.float32),
    "first_root_orient_aa": (lambda T, L: (3,),        np.float32),
    "first_pose_body_aa":   (lambda T, L: (21, 3),     np.float32),
    # scalar int metadata
    "window_idx":        (lambda T, L: (),             np.int32),
    "actual_len":        (lambda T, L: (),             np.int32),
    "T":                 (lambda T, L: (),             np.int32),
}

# String fields (stored as numpy object arrays; checked separately)
STRING_FIELDS = ["gender", "obj_mesh_path", "text", "seq_name"]


def validate_file(
    path: str, errors: List[str]
) -> bool:
    """Validate one .npz file against SCHEMA.

    Args:
        path: Absolute path to .npz file.
        errors: List to which error strings are appended.

    Returns:
        True if all checks pass, False otherwise.
    """
    stem = osp.splitext(osp.basename(path))[0]
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        errors.append(f"{stem}: cannot load: {e}")
        return False

    ok = True

    # Read actual_len and T for shape functions
    try:
        actual_len = int(data["actual_len"])
        T_val = int(data["T"])
        assert T_val == T_FIXED, f"T={T_val} expected {T_FIXED}"
    except Exception as e:
        errors.append(f"{stem}: cannot read actual_len/T: {e}")
        return False

    for field, (shape_fn, dtype) in SCHEMA.items():
        if field not in data:
            errors.append(f"{stem}: missing field '{field}'")
            ok = False
            continue
        arr = data[field]
        expected_shape = shape_fn(T_val, actual_len)
        if arr.shape != expected_shape:
            errors.append(
                f"{stem}: '{field}' shape {arr.shape} != expected {expected_shape}"
            )
            ok = False
        if arr.dtype != dtype:
            errors.append(
                f"{stem}: '{field}' dtype {arr.dtype} != expected {dtype}"
            )
            ok = False

    # Check string fields
    for field in STRING_FIELDS:
        if field not in data:
            errors.append(f"{stem}: missing string field '{field}'")
            ok = False
            continue
        arr = data[field]
        val = str(arr.item()) if arr.shape == () else None
        if val is None:
            errors.append(f"{stem}: '{field}' not a scalar, shape={arr.shape}")
            ok = False

    return ok


def main() -> None:
    """Entry point for schema validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate the OMOMO window .npz files written by convert.py."
    )
    parser.add_argument("--npz_dir", required=True, help="folder of window .npz files (test_npz/ or train_npz/)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Check only first N files.",
    )
    args = parser.parse_args()

    data_dir = args.npz_dir
    paths = sorted(glob.glob(osp.join(data_dir, "*.npz")))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"[validate] no .npz files in {data_dir}")

    print(f"[validate] Checking {len(paths)} files in {data_dir}")

    errors: List[str] = []
    passed = 0
    failed = 0

    for path in paths:
        ok = validate_file(path, errors)
        if ok:
            passed += 1
        else:
            failed += 1

    if errors:
        for err in errors:
            print(f"  FAIL: {err}", file=sys.stderr)
        print(
            f"[validate] FAILED: {failed}/{len(paths)} files had errors.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[validate] PASSED: {passed}/{len(paths)} files validated OK."
    )


if __name__ == "__main__":
    main()
