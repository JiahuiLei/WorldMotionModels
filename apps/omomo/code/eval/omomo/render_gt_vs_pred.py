"""Standalone GT-vs-pred MP4 renderer for saved omomo eval dirs.

Reads predictions from ``<eval_dir>/preds_all/`` and matching GT from
``<eval_dir>/_scorer_gt_subset/``, renders each as a side-by-side MP4
(GT left | pred right), and writes the videos under
``<eval_dir>/render_gt_vs_pred/``.

Reuses ``_eval_render_body_obj_frame`` from
``umm.models.wmm.viz_helpers.omomo_viz`` so the body+object FK + mesh
rendering is byte-identical to the training-time eval-viz path.

Usage:

    conda activate umm
    nvidia-smi   # pick a free GPU
    export CUDA_VISIBLE_DEVICES=0
    python eval/omomo/render_gt_vs_pred.py \\
        --eval_dir <path> \\
        --n_samples 5 \\
        --width 384 --height 384 --fps 30
"""
from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch  # noqa: F401  ensure torch loads before pyrender (CUDA pin done by caller)
import imageio
import pyrender
import trimesh

# repo on PYTHONPATH (need both repo root for prepare_data.* imports + src/ for umm.*)
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from umm.models.wmm.viz_helpers.omomo_viz import (  # noqa: E402
    _SmplxFK,
    _assert_obj_height_sane,
    _eval_render_body_obj_frame,
    _resolve_eval_obj_category,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_gt_vs_pred")


# CHOIS waypoint frames (0-indexed): full SE(3) at 0; xy-only at 29/59/89; xyz at end_idx.
# Used only for color-bar overlay.
_CHOIS_FULL_FRAMES = (0,)
_CHOIS_XY_FRAMES = (29, 59, 89)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load all keys from an npz into a dict. allow_pickle=True for str fields."""
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as d:
        for k in d.files:
            out[k] = d[k]
    return out


def _color_bar_overlay(frame: np.ndarray, t: int, end_idx: int) -> np.ndarray:
    """Paint a 6-pixel-tall top bar marking CHOIS waypoint frames.

    Green = full SE(3) anchor (frame 0)
    Yellow = xy-only anchor (frames 29/59/89)
    Cyan = xyz anchor at end_idx
    Red otherwise.
    """
    if t in _CHOIS_FULL_FRAMES:
        color = np.array([46, 204, 113], dtype=np.uint8)  # green
    elif t in _CHOIS_XY_FRAMES and t <= end_idx:
        color = np.array([241, 196, 15], dtype=np.uint8)  # yellow
    elif t == end_idx:
        color = np.array([26, 188, 156], dtype=np.uint8)  # cyan
    else:
        color = np.array([192, 57, 43], dtype=np.uint8)  # red
    out = frame.copy()
    out[:6, :, :] = color[None, None, :]
    return out


def _label_panel(
    frame: np.ndarray, label: str, color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Burn a small text label into the bottom-left corner via pixel manipulation.

    Avoids a font dependency by drawing a 6-pixel-tall block per character
    using simple box-shaped ASCII rendering (no full font — just a marker).
    Replaced by imageio metadata if/when text overlay is required.
    """
    # Minimal: draw a 12px-tall solid colored bar at the bottom whose width
    # is proportional to len(label). This is a marker, not actual text.
    out = frame.copy()
    H, W = out.shape[:2]
    bar_h = 12
    bar_w = min(W, 8 * len(label) + 16)
    color_arr = np.array(color, dtype=np.uint8)
    out[H - bar_h: H, : bar_w, :] = color_arr[None, None, :]
    return out


def _composite_side_by_side(
    gt_frames: List[np.ndarray], pred_frames: List[np.ndarray],
) -> List[np.ndarray]:
    """GT on the left, pred on the right with a 2-pixel divider."""
    assert len(gt_frames) == len(pred_frames), (
        f"frame count mismatch: gt={len(gt_frames)} pred={len(pred_frames)}"
    )
    out: List[np.ndarray] = []
    for g, p in zip(gt_frames, pred_frames):
        H, W, _ = g.shape
        divider = np.full((H, 2, 3), 255, dtype=np.uint8)
        combined = np.concatenate([g, divider, p], axis=1)
        out.append(combined)
    return out


def _render_one_window(
    pred: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    fk: _SmplxFK,
    renderer: pyrender.OffscreenRenderer,
    width: int,
    height: int,
    geo_dir: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Render GT and pred frame sequences for one window.

    Returns (gt_frames, pred_frames, end_idx).
    """
    actual_len = int(gt.get("actual_len", pred["root_trans"].shape[0]))
    actual_len = max(1, min(actual_len, int(pred["root_trans"].shape[0])))
    end_idx = max(actual_len - 1, 0)
    seq_name = str(gt.get("seq_name", "<unknown>"))

    obj_mesh_path = str(gt["obj_mesh_path"])
    category = _resolve_eval_obj_category(obj_mesh_path)
    geo_path = geo_dir / f"{category}.ply"
    if not geo_path.exists():
        raise FileNotFoundError(
            f"render_gt_vs_pred: rest-frame geometry missing: {geo_path}"
        )
    rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
    if not isinstance(rest_mesh, trimesh.Trimesh):
        raise ValueError(
            f"render_gt_vs_pred: expected Trimesh from {geo_path}, "
            f"got {type(rest_mesh).__name__}"
        )
    _assert_obj_height_sane(geo_path, rest_mesh.bounds)

    betas = np.asarray(gt["betas"], dtype=np.float32).reshape(-1)
    gender = str(gt["gender"])

    # GT body and object
    gt_root_trans = np.asarray(gt["root_trans"], dtype=np.float32)
    gt_root_orient_aa = np.asarray(gt["root_orient_aa"], dtype=np.float32)
    gt_pose_body_aa = np.asarray(gt["pose_body_aa"], dtype=np.float32)
    gt_obj_trans = np.asarray(gt["obj_trans"], dtype=np.float64)
    # GT npz uses 'obj_rot' (3x3 matrix); pred npz uses 'obj_rot_mat'.
    gt_obj_rot_mat = np.asarray(gt["obj_rot"], dtype=np.float64)

    # Pred body and object — schema from
    # src/umm/callbacks/wmm_omomo_eval.py::_save_pred_npz
    p_root_trans = np.asarray(pred["root_trans"], dtype=np.float32)
    p_root_orient_aa = np.asarray(pred["root_orient_aa"], dtype=np.float32)
    p_pose_body_aa = np.asarray(pred["pose_body_aa"], dtype=np.float32)
    p_obj_trans = np.asarray(pred["pred_obj_trans"], dtype=np.float64)
    p_obj_rot_mat = np.asarray(pred["pred_obj_rot_mat"], dtype=np.float64)

    gt_frames: List[np.ndarray] = []
    pred_frames: List[np.ndarray] = []
    for t in range(actual_len):
        # GT frame
        try:
            gframe = _eval_render_body_obj_frame(
                fk=fk, renderer=renderer, width=width, height=height,
                root_trans_t=gt_root_trans[t],
                root_orient_aa_t=gt_root_orient_aa[t],
                pose_body_aa_t=gt_pose_body_aa[t],
                betas=betas, gender=gender,
                rest_mesh=rest_mesh,
                obj_R_t=gt_obj_rot_mat[t],
                obj_t_t=gt_obj_trans[t],
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail per-frame
            log.warning("[%s] GT frame %d render failed: %s", seq_name, t, exc)
            gframe = np.zeros((height, width, 3), dtype=np.uint8)
        gframe = _color_bar_overlay(gframe, t, end_idx)
        gframe = _label_panel(gframe, "GT", color=(46, 204, 113))
        gt_frames.append(gframe)

        # Pred frame
        try:
            pframe = _eval_render_body_obj_frame(
                fk=fk, renderer=renderer, width=width, height=height,
                root_trans_t=p_root_trans[t],
                root_orient_aa_t=p_root_orient_aa[t],
                pose_body_aa_t=p_pose_body_aa[t],
                betas=betas, gender=gender,
                rest_mesh=rest_mesh,
                obj_R_t=p_obj_rot_mat[t],
                obj_t_t=p_obj_trans[t],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] PRED frame %d render failed: %s", seq_name, t, exc)
            pframe = np.zeros((height, width, 3), dtype=np.uint8)
        pframe = _color_bar_overlay(pframe, t, end_idx)
        pframe = _label_panel(pframe, "PRED", color=(231, 76, 60))
        pred_frames.append(pframe)

    return gt_frames, pred_frames, end_idx


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render GT-vs-pred MP4s from a saved omomo eval dir.",
    )
    p.add_argument(
        "--eval_dir", required=True,
        help="Path to a saved eval dir (must contain preds_all/ and "
        "_scorer_gt_subset/).",
    )
    p.add_argument(
        "--smplx_dir",
        default=str(_REPO / "data_wmm" / "omomo_v3" / "smplx"),
        help="SMPL-X model dir (must contain SMPLX_NEUTRAL.npz).",
    )
    p.add_argument(
        "--obj_geo_dir",
        default=str(_REPO / "data_wmm" / "omomo_v3" / "rest_object_geo"),
        help="Rest-frame object PLY directory.",
    )
    p.add_argument(
        "--out_subdir", default="render_gt_vs_pred",
        help="Output subdir under --eval_dir (default: render_gt_vs_pred).",
    )
    p.add_argument("--n_samples", type=int, default=5, help="Number of windows to render.")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--seq_filter", default=None,
        help="Optional substring filter on seq_name (e.g. 'clothesstand').",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    preds_dir = eval_dir / "preds_all"
    gt_dir = eval_dir / "_scorer_gt_subset"
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"preds_all/ not found under {eval_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"_scorer_gt_subset/ not found under {eval_dir}")

    out_dir = eval_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("eval_dir = %s", eval_dir)
    log.info("output    = %s", out_dir)

    pred_paths = sorted(preds_dir.glob("*.npz"))
    if args.seq_filter:
        pred_paths = [p for p in pred_paths if args.seq_filter in p.stem]
    if args.n_samples <= 0:
        pred_paths = []
    elif len(pred_paths) > args.n_samples:
        # Evenly-spaced indices across the full sorted list (NOT the first
        # N) so the rendered sample set covers different episodes / objects
        # rather than just the alphabetic prefix.
        idx = [
            int(round(i * (len(pred_paths) - 1) / (args.n_samples - 1)))
            if args.n_samples > 1 else 0
            for i in range(args.n_samples)
        ]
        # Dedupe while preserving order (in case of rounding collisions on
        # very small lists).
        seen = set()
        idx = [i for i in idx if not (i in seen or seen.add(i))]
        pred_paths = [pred_paths[i] for i in idx]
    if not pred_paths:
        log.warning("no preds matched (n_samples=%d filter=%r)",
                    args.n_samples, args.seq_filter)
        return 1
    log.info("rendering %d sample(s): %s",
             len(pred_paths), [p.stem for p in pred_paths])

    fk = _SmplxFK(Path(args.smplx_dir))
    geo_dir = Path(args.obj_geo_dir)
    renderer = pyrender.OffscreenRenderer(
        viewport_width=int(args.width), viewport_height=int(args.height),
    )

    n_ok = 0
    n_fail = 0
    try:
        for pp in pred_paths:
            name = pp.stem
            gp = gt_dir / pp.name
            if not gp.exists():
                log.warning("[%s] missing GT npz at %s — skipping", name, gp)
                n_fail += 1
                continue
            try:
                pred = _load_npz(str(pp))
                gt = _load_npz(str(gp))
                gt_frames, pred_frames, end_idx = _render_one_window(
                    pred=pred, gt=gt, fk=fk, renderer=renderer,
                    width=int(args.width), height=int(args.height),
                    geo_dir=geo_dir,
                )
                combined = _composite_side_by_side(gt_frames, pred_frames)
                out_mp4 = out_dir / f"{name}.mp4"
                imageio.mimwrite(
                    str(out_mp4), combined,
                    fps=int(args.fps), codec="libx264",
                    macro_block_size=1,
                )
                n_ok += 1
                log.info("[%s] wrote %s (T=%d, end_idx=%d)",
                         name, out_mp4.name, len(combined), end_idx)
            except Exception as exc:  # noqa: BLE001 — fail-soft per window
                n_fail += 1
                log.exception("[%s] render failed: %s", name, exc)
    finally:
        try:
            renderer.delete()
        except Exception:  # noqa: BLE001
            pass

    log.info("DONE — %d ok, %d failed → %s", n_ok, n_fail, out_dir)
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
