"""Visual sweep: CHOIS partial-conditioning sampling modes vs the leaky baseline.

Runs ``agent.inference()`` ``N_modes`` times on the SAME ``N_windows`` GT
clips (with the SAME random seed) and renders train-time-style MP4s into
``<project_root>/debug/chois_modes_<ts>/``. Each mode is a different
combination of:

    * ``partial_chois`` — eval-batch flag (``"off"`` = leak full SE(3) at
      object waypoints, current behavior; ``"strict"`` = CHOIS-faithful,
      only obj@0 fully known + xy/xyz channel targets).
    * ``chois_mode``   — sampling-time hook (``"off"`` / ``"replace"`` /
      ``"guidance"``).
    * ``smooth_lambda``— acceleration smoothing weight (0 = off).

Default sweep (5 modes; same seed; same n_steps):

    | tag       | partial_chois | chois_mode | smooth_lambda |
    |-----------|---------------|------------|---------------|
    | baseline  | off           | off        | 0             |
    | M1        | strict        | replace    | 0             |
    | M2        | strict        | guidance   | 0             |
    | M1_M3     | strict        | replace    | smooth_lambda |
    | M2_M3     | strict        | guidance   | smooth_lambda |

Outputs::

    debug/chois_modes_<ts>/
        run_meta.json
        baseline/
            preds/<seq>.npz       # CHOIS-schema pred npz
            videos/<seq>.mp4      # 30 fps train-time-style render
            strips/<seq>.jpg      # 6 evenly-spaced keyframes
        M1/  M2/  M1_M3/  M2_M3/  (same layout)
        index.html                # 5-column grid: one row per clip

Example::

    CUDA_VISIBLE_DEVICES=0 python eval/omomo/eval_chois_modes.py \\
        --ckpt <path> \\
        --config configs/wmm/om/w1.5.0.yaml \\
        --n_windows 3 \\
        --smooth_lambda 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("eval_chois_modes")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Mode table (single source of truth — keep in sync with the driver header).
# ---------------------------------------------------------------------------

MODES: List[Dict[str, Any]] = [
    {"tag": "baseline", "partial_chois": "off",    "chois_mode": "off",
     "smooth_lambda_mult": 0.0},
    {"tag": "M1",       "partial_chois": "strict", "chois_mode": "replace",
     "smooth_lambda_mult": 0.0},
    {"tag": "M2",       "partial_chois": "strict", "chois_mode": "guidance",
     "smooth_lambda_mult": 0.0},
    {"tag": "M1_M3",    "partial_chois": "strict", "chois_mode": "replace",
     "smooth_lambda_mult": 1.0},
    {"tag": "M2_M3",    "partial_chois": "strict", "chois_mode": "guidance",
     "smooth_lambda_mult": 1.0},
]


# ---------------------------------------------------------------------------
# Argparse / setup
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to the trained .ckpt.",
    )
    p.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs/wmm/om/w1.5.0.yaml"),
    )
    p.add_argument(
        "--gt_dir",
        default=str(_REPO_ROOT / "data_wmm/omomo_v2/test_npz"),
    )
    p.add_argument(
        "--smplx_dir",
        default=str(_REPO_ROOT / "data_wmm/omomo_v2/smplx"),
    )
    p.add_argument(
        "--obj_geo_dir",
        default=str(_REPO_ROOT / "data_wmm/omomo_v2/rest_object_geo"),
    )
    p.add_argument("--n_windows", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n_steps", type=int, default=20,
        help="DDIM sampling steps (matches w1.5.0 gen_timesteps=20 default).",
    )
    p.add_argument(
        "--smooth_lambda", type=float, default=0.05,
        help="Acceleration smoothing weight (used by M1+M3 / M2+M3 only).",
    )
    p.add_argument("--chois_guide_scale", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output dir; default debug/chois_modes_<UTC ts>/.")
    p.add_argument("--max_T", type=int, default=120,
                   help="Window length (must match training config).")
    p.add_argument("--num_pts", type=int, default=256,
                   help="PCL points per object (must match training config).")
    p.add_argument("--render_fps", type=int, default=30)
    p.add_argument("--render_size", type=int, default=512)
    p.add_argument("--strip_keyframes", type=int, default=6)
    p.add_argument(
        "--skip_render", action="store_true",
        help="Run inference + save preds, skip the heavy rendering.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _build_batch(
    *,
    gt_paths: List[str],
    device,
    obj_geo_dir: str,
    smplx_dir: str,
    num_pts: int,
    partial_chois: str,
):
    """Wrap the existing batch builder with the new partial_chois flag."""
    from umm.callbacks.wmm_omomo_eval import _build_batch_from_gt_files
    return _build_batch_from_gt_files(
        gt_paths=gt_paths,
        device=device,
        obj_geo_dir=obj_geo_dir,
        smplx_dir=smplx_dir,
        num_pts=num_pts,
        partial_chois=partial_chois,
    )


def _select_inference_agent(pl_module):
    """Prefer the EMA agent when present (matches eval_and_save_wmm)."""
    try:
        from umm.models.wmm.model import PLModel  # noqa: F401  (type only)
    except Exception:
        pass
    ema = getattr(pl_module, "ema_model", None)
    if ema is not None:
        ema_inner = getattr(ema, "module", ema)
        if ema_inner is not None:
            log.info("Using EMA agent for inference.")
            return ema_inner
    log.warning("EMA agent not found — falling back to non-EMA agent.")
    return getattr(pl_module, "model", pl_module)


def _run_inference(
    agent,
    batch: Dict[str, Any],
    *,
    seed: int,
    n_steps: int,
    chois_mode: str,
    chois_guide_scale: float,
    smooth_lambda: float,
):
    """Single inference pass; returns (B, T, P, 4, 4) trace tensor (CPU)."""
    import torch
    pos_target = batch.get("chois_pos_target", None)
    pos_mask = batch.get("chois_pos_mask", None)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = agent.inference(
            trace=batch["trace"],
            context=batch["context"],
            trace_known_mask=batch["known_mask"],
            context_known_mask=batch["context_known_mask"],
            trace_mask=batch["trace_mask"],
            context_mask=batch["context_mask"],
            real_timestep=batch["time"],
            group_id=batch["trace_group_id"],
            sem_type=batch["trace_sem_type"],
            rot_mask=batch["trace_rot_mask"],
            text=batch["text"],
            local_pcl=batch["local_pcl"],
            local_pcl_mask=batch["local_pcl_mask"],
            seed=seed,
            n_steps=n_steps,
            verbose=False,
            scheduler_mode="ddim_gen",
            # CHOIS partial-channel kwargs (no-op when args are None / "off").
            chois_pos_target=pos_target,
            chois_pos_mask=pos_mask,
            chois_mode=chois_mode,
            chois_guide_scale=chois_guide_scale,
            smooth_lambda=smooth_lambda,
        )
    _, ret_trace = result            # (B, T, P, 4, 4)
    return ret_trace.detach().cpu()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


# Top-bar palette (semantic leak indicator). Stored as uint8 RGB so it can be
# blitted directly into the rendered frame's first 10 rows.
_COLOR_FULL = np.array([ 60, 220,  60], dtype=np.uint8)   # green: fully known
_COLOR_XY   = np.array([240, 220,  40], dtype=np.uint8)   # yellow: xy-only
_COLOR_XYZ  = np.array([255, 140,  20], dtype=np.uint8)   # orange: xyz-only
_COLOR_GEN  = np.array([220,  60,  60], dtype=np.uint8)   # red: model-generated


def _waypoint_color(t: int, actual_len: int) -> np.ndarray:
    """Return the per-frame top-bar color for the CHOIS leak palette.

    * frame 0                    → green (fully known)
    * frames 29/59/89 (in range) → yellow (xy-only partial)
    * frame ``actual_len-1``      → orange (xyz-only partial)
    * everything else            → red (model-generated)
    """
    end_idx = max(int(actual_len) - 1, 0)
    if t == 0:
        return _COLOR_FULL
    if t == end_idx and end_idx not in (0, 29, 59, 89):
        return _COLOR_XYZ
    if t in (29, 59, 89) and t < actual_len:
        return _COLOR_XY
    return _COLOR_GEN


def _render_one_window_mp4(
    *,
    fk,
    renderer,
    pred_npz_path: str,
    gt_dict: Dict[str, Any],
    out_mp4: str,
    out_strip_jpg: str,
    geo_dir: str,
    width: int,
    height: int,
    fps: int,
    n_strip: int,
    waypoints: Tuple[int, ...],
):
    """Render a 30-fps MP4 + a strip JPEG for one prediction npz."""
    import numpy as _np
    import trimesh
    import imageio.v2 as iio
    from umm.models.wmm.viz_helpers.omomo_viz import (
        _eval_render_body_obj_frame, _resolve_eval_obj_category,
    )

    with _np.load(pred_npz_path, allow_pickle=True) as d:
        root_trans = _np.asarray(d["root_trans"], dtype=_np.float32)
        root_orient_aa = _np.asarray(d["root_orient_aa"], dtype=_np.float32)
        pose_body_aa = _np.asarray(d["pose_body_aa"], dtype=_np.float32)
        pred_obj_trans = _np.asarray(d["pred_obj_trans"], dtype=_np.float32)
        pred_obj_rot_mat = _np.asarray(d["pred_obj_rot_mat"], dtype=_np.float32)

    betas = _np.asarray(gt_dict["betas"], dtype=_np.float32).reshape(-1)
    gender = str(gt_dict["gender"])
    obj_mesh_path = str(gt_dict["obj_mesh_path"])
    actual_len = int(gt_dict.get("actual_len", root_trans.shape[0]))
    actual_len = max(1, min(actual_len, int(root_trans.shape[0])))

    category = _resolve_eval_obj_category(obj_mesh_path)
    geo_path = Path(geo_dir) / f"{category}.ply"
    if not geo_path.exists():
        raise FileNotFoundError(f"missing rest mesh: {geo_path}")
    rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
    if not isinstance(rest_mesh, trimesh.Trimesh):
        raise ValueError(f"expected Trimesh from {geo_path}")

    frames: List[_np.ndarray] = []
    for t in range(actual_len):
        frame = _eval_render_body_obj_frame(
            fk=fk, renderer=renderer, width=width, height=height,
            root_trans_t=root_trans[t],
            root_orient_aa_t=root_orient_aa[t],
            pose_body_aa_t=pose_body_aa[t],
            betas=betas, gender=gender, rest_mesh=rest_mesh,
            obj_R_t=pred_obj_rot_mat[t].astype(_np.float64),
            obj_t_t=pred_obj_trans[t].astype(_np.float64),
        )
        # Top-bar coloring (leak semantics):
        #   green   = fully-known anchor (frame 0)
        #   yellow  = xy-only partial (CHOIS intermediate: 29/59/89)
        #   orange  = xyz-only partial (CHOIS end frame)
        #   red     = unknown / generated
        color = _waypoint_color(t, actual_len)
        frame[:10, :, :3] = color[None, None, :]
        frames.append(frame)

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", quality=8)

    # Strip: pick n_strip evenly spaced keyframes, hstack into one image.
    n_strip = max(2, min(int(n_strip), actual_len))
    idxs = _np.linspace(0, actual_len - 1, num=n_strip).round().astype(int)
    strip = _np.concatenate([frames[i] for i in idxs], axis=1)
    Path(out_strip_jpg).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_strip_jpg, strip)


def _render_gt_one_window(
    *,
    fk,
    renderer,
    gt_npz_path: str,
    gt_dict: Dict[str, Any],
    out_mp4: str,
    out_strip_jpg: str,
    geo_dir: str,
    width: int,
    height: int,
    fps: int,
    n_strip: int,
    waypoints: Tuple[int, ...],
):
    """Render the GT window so we have a reference column."""
    import numpy as _np
    import trimesh
    import imageio.v2 as iio
    from umm.models.wmm.viz_helpers.omomo_viz import (
        _eval_render_body_obj_frame, _resolve_eval_obj_category,
    )

    with _np.load(gt_npz_path, allow_pickle=True) as d:
        root_trans = _np.asarray(d["root_trans"], dtype=_np.float32)
        root_orient_aa = _np.asarray(d["root_orient_aa"], dtype=_np.float32)
        pose_body_aa = _np.asarray(d["pose_body_aa"], dtype=_np.float32)
        obj_trans = _np.asarray(d["obj_trans"], dtype=_np.float32)
        obj_rot = _np.asarray(d["obj_rot"], dtype=_np.float32)

    betas = _np.asarray(gt_dict["betas"], dtype=_np.float32).reshape(-1)
    gender = str(gt_dict["gender"])
    obj_mesh_path = str(gt_dict["obj_mesh_path"])
    actual_len = int(gt_dict.get("actual_len", root_trans.shape[0]))
    actual_len = max(1, min(actual_len, int(root_trans.shape[0])))

    category = _resolve_eval_obj_category(obj_mesh_path)
    geo_path = Path(geo_dir) / f"{category}.ply"
    rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)

    frames: List[_np.ndarray] = []
    for t in range(actual_len):
        frame = _eval_render_body_obj_frame(
            fk=fk, renderer=renderer, width=width, height=height,
            root_trans_t=root_trans[t],
            root_orient_aa_t=root_orient_aa[t],
            pose_body_aa_t=pose_body_aa[t],
            betas=betas, gender=gender, rest_mesh=rest_mesh,
            obj_R_t=obj_rot[t].astype(_np.float64),
            obj_t_t=obj_trans[t].astype(_np.float64),
        )
        # Same leak-color palette as predictions, so the GT column lines up
        # with the prediction columns frame-by-frame.
        color = _waypoint_color(t, actual_len)
        frame[:10, :, :3] = color[None, None, :]
        frames.append(frame)

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", quality=8)
    n_strip = max(2, min(int(n_strip), actual_len))
    idxs = _np.linspace(0, actual_len - 1, num=n_strip).round().astype(int)
    strip = _np.concatenate([frames[i] for i in idxs], axis=1)
    Path(out_strip_jpg).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_strip_jpg, strip)


# ---------------------------------------------------------------------------
# HTML index
# ---------------------------------------------------------------------------


def _write_html_index(
    out_root: Path,
    seq_names: List[str],
    mode_tags: List[str],
    waypoints: Tuple[int, ...],
):
    """Write index.html with a (clip × mode) grid: video on top, strip below."""
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>CHOIS sampling-mode sweep</title>")
    parts.append(
        "<style>"
        "body{font-family:sans-serif;background:#111;color:#eee;margin:16px}"
        "h2{margin-top:32px}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #333;padding:6px;vertical-align:top}"
        "video{width:100%;max-width:380px;display:block}"
        "img.strip{width:100%;max-width:1100px;display:block;margin-top:6px}"
        ".tag{font-weight:bold;color:#9cf}"
        ".legend{font-size:12px;color:#aaa}"
        "</style></head><body>"
    )
    parts.append("<h1>CHOIS sampling-mode visual sweep</h1>")
    parts.append(
        "<p class='legend'>Modes left-to-right: "
        + " · ".join([f"<span class='tag'>{m}</span>" for m in mode_tags])
        + ". Top-bar palette per frame — "
        "<span style='background:#3cdc3c;color:#000;padding:1px 4px'>"
        "GREEN&nbsp;= fully known (frame 0)</span> · "
        "<span style='background:#f0dc28;color:#000;padding:1px 4px'>"
        "YELLOW = xy-only partial (29/59/89)</span> · "
        "<span style='background:#ff8c14;color:#000;padding:1px 4px'>"
        "ORANGE = xyz-only partial (end frame)</span> · "
        "<span style='background:#dc3c3c;color:#000;padding:1px 4px'>"
        "RED&nbsp;&nbsp;&nbsp;= model-generated</span></p>"
    )
    for seq in seq_names:
        parts.append(f"<h2>{seq}</h2>")
        # videos row
        parts.append("<table><tr>")
        for tag in mode_tags:
            parts.append(f"<th>{tag}</th>")
        parts.append("</tr><tr>")
        for tag in mode_tags:
            mp4 = f"{tag}/videos/{seq}.mp4"
            parts.append(
                f"<td><video src='{mp4}' controls loop muted "
                f"preload='metadata'></video></td>"
            )
        parts.append("</tr></table>")
        # strips row (full-width per mode for readability)
        for tag in mode_tags:
            strip = f"{tag}/strips/{seq}.jpg"
            parts.append(
                f"<p class='tag'>{tag} — keyframes</p>"
                f"<img class='strip' src='{strip}' alt='{seq} {tag}'/>"
            )
    parts.append("</body></html>")
    (out_root / "index.html").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # GPU pin BEFORE torch import (per project rule).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    import torch  # noqa: WPS433  (must come AFTER CUDA pin)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # Output dir
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_root = _REPO_ROOT / "debug" / f"chois_modes_{ts}"
    else:
        out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("out_root=%s", out_root)

    # Resolve clip list (deterministic: alphabetical first N).
    gt_dir = Path(args.gt_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    all_npz = sorted(p for p in gt_dir.glob("*.npz"))
    if len(all_npz) < args.n_windows:
        raise ValueError(
            f"only {len(all_npz)} GT npz in {gt_dir}, need {args.n_windows}"
        )
    gt_paths = [str(p) for p in all_npz[: args.n_windows]]
    log.info("Selected %d GT clips:\n  %s",
             len(gt_paths), "\n  ".join(gt_paths))

    # Load model.
    log.info("Loading PLModel from %s ...", args.ckpt)
    from eval.omomo.eval_and_save_wmm import load_pl_module  # noqa: WPS433
    pl_module, cfg = load_pl_module(args.ckpt, args.config, device)
    agent = _select_inference_agent(pl_module)

    # Build batches once per (partial_chois value) so M1/M2/M3 share.
    batch_off, _ = _build_batch(
        gt_paths=gt_paths, device=device,
        obj_geo_dir=args.obj_geo_dir, smplx_dir=args.smplx_dir,
        num_pts=args.num_pts, partial_chois="off",
    )
    batch_strict, gt_list = _build_batch(
        gt_paths=gt_paths, device=device,
        obj_geo_dir=args.obj_geo_dir, smplx_dir=args.smplx_dir,
        num_pts=args.num_pts, partial_chois="strict",
    )
    log.info("Built two batches (off + strict). N=%d", len(gt_list))

    # Sanity log on the strict batch's chois mask coverage.
    msk = batch_strict["chois_pos_mask"]
    log.info(
        "strict.chois_pos_mask: shape=%s, n_True=%d "
        "(per-sample sum: %s)",
        tuple(msk.shape), int(msk.sum().item()),
        msk.reshape(msk.shape[0], -1).sum(dim=1).tolist(),
    )

    # Run all 5 modes.
    mode_traces: Dict[str, torch.Tensor] = {}
    for mode in MODES:
        tag = mode["tag"]
        b = batch_off if mode["partial_chois"] == "off" else batch_strict
        smooth_lambda = float(args.smooth_lambda) * float(mode["smooth_lambda_mult"])
        log.info(
            "[%s] partial_chois=%s chois_mode=%s smooth_lambda=%.4f",
            tag, mode["partial_chois"], mode["chois_mode"], smooth_lambda,
        )
        t0 = time.time()
        ret_trace = _run_inference(
            agent, b,
            seed=args.seed,
            n_steps=args.n_steps,
            chois_mode=mode["chois_mode"],
            chois_guide_scale=args.chois_guide_scale,
            smooth_lambda=smooth_lambda,
        )
        dt = time.time() - t0
        log.info("[%s] inference done in %.2fs", tag, dt)
        mode_traces[tag] = ret_trace

    # Save preds in each mode dir.
    from umm.callbacks.wmm_omomo_eval import _save_pred_npz
    seq_names: List[str] = []
    for mode in MODES:
        tag = mode["tag"]
        preds_dir = out_root / tag / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        ret_trace = mode_traces[tag]
        for b_idx, gt in enumerate(gt_list):
            stem = str(gt["seq_name"])
            if tag == MODES[0]["tag"]:
                seq_names.append(stem)
            _save_pred_npz(
                str(preds_dir), stem,
                ret_trace[b_idx], gt, args.smplx_dir,
            )

    # Render.
    if args.skip_render:
        log.info("--skip_render set, skipping MP4/strip rendering.")
    else:
        log.info("Rendering MP4s + strips at %dx%d / %d fps",
                 args.render_size, args.render_size, args.render_fps)
        # Lazy-import the renderer (pulls pyrender → needs EGL).
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender  # noqa: WPS433
        from umm.models.wmm.viz_helpers.omomo_viz import _SmplxFK  # noqa: WPS433
        fk = _SmplxFK(Path(args.smplx_dir))
        renderer = pyrender.OffscreenRenderer(
            viewport_width=int(args.render_size),
            viewport_height=int(args.render_size),
        )
        try:
            # GT column (one render per clip, shared across modes).
            gt_dir_out = out_root / "GT"
            for b_idx, gt in enumerate(gt_list):
                stem = str(gt["seq_name"])
                _render_gt_one_window(
                    fk=fk, renderer=renderer,
                    gt_npz_path=gt_paths[b_idx], gt_dict=gt,
                    out_mp4=str(gt_dir_out / "videos" / f"{stem}.mp4"),
                    out_strip_jpg=str(gt_dir_out / "strips" / f"{stem}.jpg"),
                    geo_dir=args.obj_geo_dir,
                    width=args.render_size, height=args.render_size,
                    fps=args.render_fps, n_strip=args.strip_keyframes,
                    waypoints=(0, 29, 30, 59, 60, 89, 90, 119),
                )
                log.info("rendered GT %s", stem)
            # Predictions per mode.
            for mode in MODES:
                tag = mode["tag"]
                preds_dir = out_root / tag / "preds"
                vids_dir = out_root / tag / "videos"
                strips_dir = out_root / tag / "strips"
                for b_idx, gt in enumerate(gt_list):
                    stem = str(gt["seq_name"])
                    _render_one_window_mp4(
                        fk=fk, renderer=renderer,
                        pred_npz_path=str(preds_dir / f"{stem}.npz"),
                        gt_dict=gt,
                        out_mp4=str(vids_dir / f"{stem}.mp4"),
                        out_strip_jpg=str(strips_dir / f"{stem}.jpg"),
                        geo_dir=args.obj_geo_dir,
                        width=args.render_size, height=args.render_size,
                        fps=args.render_fps, n_strip=args.strip_keyframes,
                        waypoints=(0, 29, 30, 59, 60, 89, 90, 119),
                    )
                    log.info("rendered %s/%s", tag, stem)
        finally:
            renderer.delete()

        # HTML index — GT first, then the 5 modes.
        _write_html_index(
            out_root,
            seq_names=seq_names,
            mode_tags=["GT"] + [m["tag"] for m in MODES],
            waypoints=(0, 29, 59, 89, 119),
        )

    # Provenance.
    meta = {
        "ckpt": args.ckpt,
        "config": args.config,
        "gt_dir": args.gt_dir,
        "smplx_dir": args.smplx_dir,
        "obj_geo_dir": args.obj_geo_dir,
        "n_windows": args.n_windows,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "smooth_lambda": args.smooth_lambda,
        "chois_guide_scale": args.chois_guide_scale,
        "modes": MODES,
        "host": socket.gethostname(),
        "gpu": int(args.gpu),
        "out_root": str(out_root),
        "gt_paths": gt_paths,
        "seq_names": seq_names,
        "rendered": not args.skip_render,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("run_meta.json written.")
    log.info("=== DONE — %s/index.html ===", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
