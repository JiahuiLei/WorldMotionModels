"""MP4 video visualizers for TG (TraceGen) domain.

Two animations, paired with the static JPG viz in ``tg_viz.py``:

    1. ``render_tg_2d_flow``      — 2D flow animation on the anchor RGB frame.
    2. ``render_tg_3d_depth``     — 3D scene with depth-backprojected point
                                     cloud + animated trajectory polylines
                                     rendered via cv2-rasterised projection
                                     (no pyrender, no OpenGL context).

Coordinate convention: OpenCV (+X right, +Y down, +Z forward). TG has only the
anchor camera frame, so the "world" frame equals the anchor-camera frame and
``T_wc`` is implicitly the identity. Intrinsics are assumed pinhole with
``fx = fy = image_size * fx_factor`` and ``cx = cy = image_size / 2``.

Helpers (``_generate_track_colors``, ``_make_timeline_bar``, ``_look_at_opencv``)
are imported from the V13 real-viz pipeline — they are pure utilities, not
model-version-specific code, so the cross-version import is acceptable per
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from umm.models.umm_v13.viz_helpers.real_viz import (
    _generate_track_colors,
    _make_timeline_bar,
)
from umm.models.umm_v13.viz_helpers.behavior_viz_v2 import _look_at_opencv

logger = logging.getLogger(__name__)

_DEFAULT_YFOV = np.pi / 3.0


# ---------------------------------------------------------------------------
# Backprojection helpers (TG assumes T_wc = I)
# ---------------------------------------------------------------------------
# Note: previous file revision exported ``select_drawn_tracks`` (top-N moving
# tracks). It had ZERO callers in src/umm (the tg_viz routine sets
# ``drawn_idx = np.where(gt_mask_p)[0]`` directly) so the helper was pruned
# in ( 20260424-2353-a89fef1b). Confirmed via
# ``grep -rn select_drawn_tracks src/umm ext/`` -> no remaining references.


def backproject_trace_tg(
    trace_pt3_xyz: np.ndarray,
    image_size: int = 384,
    fx_factor: float = 1.0,
) -> np.ndarray:
    """Backproject a normalized-uv + metric-z trace into anchor-camera-world space.

    Input ``trace_pt3_xyz[p, t] = (x_n, y_n, z_m)`` with xy normalized to
    ``[0, 1]`` and z in metres. Pinhole with
    ``fx = fy = image_size * fx_factor`` and ``cx = cy = image_size / 2``.

    Args:
        trace_pt3_xyz: ``(P, T, 3)`` float — (x_n, y_n, z_m).
        image_size: target square image size in pixels.
        fx_factor: intrinsic focal scale relative to ``image_size``.

    Returns:
        ``(T, P, 3)`` float32 — transposed to match the ``(T, P, 3)``
        ordering that ``render_tg_3d_depth`` consumes.
    """
    if trace_pt3_xyz.ndim != 3 or trace_pt3_xyz.shape[-1] != 3:
        raise ValueError(
            f"backproject_trace_tg: expected (P, T, 3); got {trace_pt3_xyz.shape}"
        )
    W = H = float(image_size)
    fx = fy = float(image_size) * float(fx_factor)
    cx = cy = float(image_size) / 2.0

    xn = trace_pt3_xyz[..., 0].astype(np.float32)  # (P, T)
    yn = trace_pt3_xyz[..., 1].astype(np.float32)  # (P, T)
    z = trace_pt3_xyz[..., 2].astype(np.float32)   # (P, T)

    x_w = (xn * W - cx) * z / fx                   # (P, T)
    y_w = (yn * H - cy) * z / fy                   # (P, T)
    z_w = z                                        # (P, T)

    world_pt3 = np.stack([x_w, y_w, z_w], axis=-1).astype(np.float32)  # (P, T, 3)
    return world_pt3.transpose(1, 0, 2).copy()  # (T, P, 3)


def backproject_depth_tg(
    depth_raw_hw: Optional[np.ndarray],
    anchor_rgb: Optional[np.ndarray] = None,
    image_size: int = 384,
    fx_factor: float = 1.0,
    max_points: int = 20000,
    stride: int = 2,
) -> Optional[np.ndarray]:
    """Backproject a depth image into a colored (N, 6) point cloud.

    The depth image may have a different resolution than the RGB anchor
    (e.g. TG ships depth at ~294x518 while RGB is 384x384). Pixel-pinhole
    math is therefore done at the ``image_size`` scale — we first rescale
    depth-pixel coords to the normalized ``image_size`` frame where the
    trace lives, then apply the same intrinsics as ``backproject_trace_tg``.

    Args:
        depth_raw_hw: ``(H_d, W_d)`` float32 depth in metres, or ``None``.
        anchor_rgb: Optional ``(H_r, W_r, 3) uint8`` RGB used to color
            each point. Sampled at integer-clamped ``(u_n * W_r, v_n * H_r)``.
        image_size: target square image size in pixels (matches the trace
            normalization frame).
        fx_factor: intrinsic focal scale relative to ``image_size``.
        max_points: subsample to at most this many points (seeded 42).
        stride: pixel stride when sampling the depth map.

    Returns:
        ``(N, 6) float32 [x, y, z, r, g, b]`` or ``None`` if depth is
        empty / ``None`` / all invalid.
    """
    if depth_raw_hw is None:
        return None
    depth = np.asarray(depth_raw_hw)
    if depth.ndim != 2 or depth.size == 0:
        return None
    depth = depth.astype(np.float32)
    H_d, W_d = depth.shape

    # Stride grid of depth-pixel coords.
    u_grid, v_grid = np.meshgrid(
        np.arange(0, W_d, stride), np.arange(0, H_d, stride)
    )
    u_d = u_grid.ravel().astype(np.float32)  # (N_grid,) depth-pixel u
    v_d = v_grid.ravel().astype(np.float32)  # (N_grid,) depth-pixel v
    d = depth[v_grid, u_grid].ravel()        # (N_grid,)

    valid = (d > 0.01) & np.isfinite(d)
    if not valid.any():
        return None
    u_d, v_d, d = u_d[valid], v_d[valid], d[valid]

    # Depth-pixel -> normalized [0, 1] -> image_size-pixel frame.
    u_n = u_d / float(W_d)                   # (N,) in [0, 1]
    v_n = v_d / float(H_d)                   # (N,) in [0, 1]
    u_img = u_n * float(image_size)          # (N,) pixels at image_size
    v_img = v_n * float(image_size)

    fx = fy = float(image_size) * float(fx_factor)
    cx = cy = float(image_size) / 2.0

    x_w = (u_img - cx) * d / fx              # (N,)
    y_w = (v_img - cy) * d / fy              # (N,)
    z_w = d                                  # (N,)
    pts_world = np.stack([x_w, y_w, z_w], axis=-1).astype(np.float32)  # (N, 3)

    # Sample colors from anchor_rgb at normalized uv if provided.
    if anchor_rgb is not None:
        if anchor_rgb.ndim != 3 or anchor_rgb.shape[-1] != 3:
            raise ValueError(
                f"backproject_depth_tg: anchor_rgb must be (H,W,3); "
                f"got {anchor_rgb.shape}"
            )
        H_r, W_r = anchor_rgb.shape[:2]
        u_int = np.clip((u_n * W_r).astype(np.int32), 0, W_r - 1)
        v_int = np.clip((v_n * H_r).astype(np.int32), 0, H_r - 1)
        rgb = anchor_rgb[v_int, u_int]       # (N, 3) uint8
    else:
        rgb = np.full((pts_world.shape[0], 3), 180, dtype=np.uint8)

    # Subsample if too many points.
    if pts_world.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(
            pts_world.shape[0], max_points, replace=False
        )
        pts_world = pts_world[idx]
        rgb = rgb[idx]

    out = np.zeros((pts_world.shape[0], 6), dtype=np.float32)
    out[:, :3] = pts_world
    out[:, 3:6] = rgb.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 2D flow animation
# ---------------------------------------------------------------------------


def render_tg_2d_flow(
    anchor_rgb: np.ndarray,
    gt_trace_tp3: np.ndarray,
    pred_trace_tp3: Optional[np.ndarray],
    trace_valid_tp: np.ndarray,
    first_kp_p2: np.ndarray,
    drawn_idx: np.ndarray,
    image_size: int = 384,
    known_n: int = 1,
    known_start: int = 0,
    mode: str = "both",
) -> List[np.ndarray]:
    """Render a 2D flow animation over the anchor RGB frame.

    For each timestep ``t`` in ``[0, T-1]`` we paint a copy of the anchor
    RGB, draw GT polylines 0..t (solid, lw=2, per-track color) + optional
    pred polylines 0..t (dashed = every-second-segment drawn, lw=1, color
    darkened by 0.65), mark the current dot, plant the anchor cross at
    ``first_kp``, and stack a ``_make_timeline_bar`` at top.

    Args:
        anchor_rgb: ``(image_size, image_size, 3) uint8`` RGB frame.
        gt_trace_tp3: ``(T, P, 3)`` GT trace (xy normalized + z metres).
        pred_trace_tp3: same-shape pred trace or ``None``.
        trace_valid_tp: ``(T, P)`` bool validity.
        first_kp_p2: ``(P, 2)`` float32 anchor kp in normalized ``[0, 1]``.
        drawn_idx: ``(N_drawn,) int32`` index into ``P`` — already capped.
        image_size: square frame size in pixels.
        known_n: number of known frames for the timeline bar (TG uses 1).
        known_start: start of the known block (TG uses 0).

    Returns:
        List of ``(image_size + bar_h, image_size, 3) uint8`` frames of
        length ``T``.
    """
    T, P, _ = gt_trace_tp3.shape
    if mode not in ("gt", "pred", "both"):
        raise ValueError(f"render_tg_2d_flow: mode must be gt/pred/both, got {mode!r}")
    if mode == "pred" and pred_trace_tp3 is None:
        raise ValueError("render_tg_2d_flow: mode='pred' requires pred_trace_tp3")
    if anchor_rgb.shape[:2] != (image_size, image_size):
        raise ValueError(
            f"render_tg_2d_flow: anchor_rgb {anchor_rgb.shape} does not match "
            f"image_size={image_size}"
        )
    if trace_valid_tp.shape != (T, P):
        raise ValueError(
            f"render_tg_2d_flow: trace_valid_tp shape {trace_valid_tp.shape} "
            f"does not match GT trace (T,P)={T,P}"
        )

    H = W = int(image_size)
    tl_h = max(int(H * 0.06), 12)

    draw_gt = mode in ("gt", "both")
    draw_pred = mode in ("pred", "both") and pred_trace_tp3 is not None
    n_drawn = int(drawn_idx.size)
    # Tighten line / dot sizes for high-density (all-400) renders.
    line_w = 1 if n_drawn > 50 else 2
    dot_r_track = 2 if n_drawn > 50 else 3
    if n_drawn == 0:
        frames: List[np.ndarray] = []
        for t_i in range(T):
            canvas = anchor_rgb.copy()
            tl_bar = _make_timeline_bar(
                W, T, t_i, known_start, known_n, bar_h=tl_h
            )
            frames.append(np.concatenate([tl_bar, canvas], axis=0))
        return frames

    # Per-track colors for the drawn subset.
    colors = _generate_track_colors(n_drawn)                 # (n_drawn, 3) uint8
    py_gt = [tuple(int(x) for x in colors[i].tolist()) for i in range(n_drawn)]
    py_pr = [
        (
            int(max(0, min(255, colors[i, 0] * 0.65))),
            int(max(0, min(255, colors[i, 1] * 0.65))),
            int(max(0, min(255, colors[i, 2] * 0.65))),
        )
        for i in range(n_drawn)
    ]

    # Pixel-space conversions. (T, N_drawn, 2) int32.
    gt_xy_px = np.stack(
        [
            gt_trace_tp3[:, drawn_idx, 0] * float(W),
            gt_trace_tp3[:, drawn_idx, 1] * float(H),
        ],
        axis=-1,
    )
    gt_xy_int = np.nan_to_num(gt_xy_px, nan=0.0).astype(np.int32)

    pred_xy_int: np.ndarray = np.zeros((T, n_drawn, 2), dtype=np.int32)
    if pred_trace_tp3 is not None:
        if pred_trace_tp3.shape != gt_trace_tp3.shape:
            raise ValueError(
                f"render_tg_2d_flow: pred shape {pred_trace_tp3.shape} != GT "
                f"{gt_trace_tp3.shape}"
            )
        pred_xy_px = np.stack(
            [
                pred_trace_tp3[:, drawn_idx, 0] * float(W),
                pred_trace_tp3[:, drawn_idx, 1] * float(H),
            ],
            axis=-1,
        )
        pred_xy_int = np.nan_to_num(pred_xy_px, nan=0.0).astype(np.int32)

    sel_valid = trace_valid_tp[:, drawn_idx]                 # (T, n_drawn) bool
    anchor_px = (first_kp_p2[drawn_idx] * float(image_size)).astype(np.int32)

    frames = []
    # Pred-only mode draws pred for every timestep regardless of GT validity.
    pred_use_all_t = (mode == "pred")
    # Pred line is solid in pred-only mode, dashed in "both" mode.
    pred_dashed = (mode == "both")
    for t_i in range(T):
        canvas = anchor_rgb.copy()
        for pi in range(n_drawn):
            gt_color = py_gt[pi]
            # In pred-only mode use full-bright color (not the 0.65-darkened
            # `py_pr`) so the single-stream render reads as the primary signal.
            pr_color = py_gt[pi] if mode == "pred" else py_pr[pi]

            # GT polyline 0..t.
            if draw_gt:
                for tt in range(1, t_i + 1):
                    if not (sel_valid[tt, pi] and sel_valid[tt - 1, pi]):
                        continue
                    x1 = int(gt_xy_int[tt - 1, pi, 0]); y1 = int(gt_xy_int[tt - 1, pi, 1])
                    x2 = int(gt_xy_int[tt, pi, 0]);     y2 = int(gt_xy_int[tt, pi, 1])
                    if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
                        continue
                    cv2.line(canvas, (x1, y1), (x2, y2), gt_color, line_w, cv2.LINE_AA)

            # Pred polyline 0..t.
            if draw_pred:
                for tt in range(1, t_i + 1):
                    if pred_dashed and (tt % 2) == 0:
                        continue  # dashed effect for "both" mode
                    if (not pred_use_all_t) and not (sel_valid[tt, pi] and sel_valid[tt - 1, pi]):
                        continue
                    x1 = int(pred_xy_int[tt - 1, pi, 0]); y1 = int(pred_xy_int[tt - 1, pi, 1])
                    x2 = int(pred_xy_int[tt, pi, 0]);     y2 = int(pred_xy_int[tt, pi, 1])
                    if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
                        continue
                    cv2.line(canvas, (x1, y1), (x2, y2), pr_color, line_w, cv2.LINE_AA)

            # Current-t dots.
            if draw_gt and sel_valid[t_i, pi]:
                cx_gt = int(gt_xy_int[t_i, pi, 0]); cy_gt = int(gt_xy_int[t_i, pi, 1])
                if 0 <= cx_gt < W and 0 <= cy_gt < H:
                    cv2.circle(canvas, (cx_gt, cy_gt), dot_r_track, gt_color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx_gt, cy_gt), dot_r_track, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_pred:
                pred_dot_valid = pred_use_all_t or sel_valid[t_i, pi]
                if pred_dot_valid:
                    cx_p = int(pred_xy_int[t_i, pi, 0]); cy_p = int(pred_xy_int[t_i, pi, 1])
                    if 0 <= cx_p < W and 0 <= cy_p < H:
                        cv2.circle(canvas, (cx_p, cy_p), dot_r_track, pr_color, -1, cv2.LINE_AA)

            # Anchor cross at first_kp for this drawn track.
            ax_ = int(anchor_px[pi, 0]); ay_ = int(anchor_px[pi, 1])
            if 0 <= ax_ < W and 0 <= ay_ < H:
                anchor_color = gt_color if draw_gt else pr_color
                cv2.drawMarker(
                    canvas, (ax_, ay_), anchor_color,
                    cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA,
                )

        tl_bar = _make_timeline_bar(W, T, t_i, known_start, known_n, bar_h=tl_h)
        frame = np.concatenate([tl_bar, canvas], axis=0)
        frames.append(frame)

    logger.info(
        f"render_tg_2d_flow: produced {len(frames)} frames "
        f"(T={T}, n_drawn={n_drawn}, mode={mode}, has_pred={draw_pred})"
    )
    return frames


# ---------------------------------------------------------------------------
# 3D depth animation (cv2-rasterised, no pyrender)
# ---------------------------------------------------------------------------


def _auto_yfov(
    scene_pts_world: np.ndarray,
    T_cw_viz: np.ndarray,
    H: int,
    W: int,
) -> float:
    """Tight yfov so the scene fills the frame, matching render_3d_depth_traces.

    Args:
        scene_pts_world: ``(N, 3)`` world points to frame.
        T_cw_viz: ``(4, 4)`` world-to-camera for the viz camera.
        H, W: output resolution.

    Returns:
        yfov in radians, clipped to ``[0.1, 0.9*pi]``; default on empty
        front-of-camera set.
    """
    if scene_pts_world.size == 0:
        return _DEFAULT_YFOV
    pts_cam = (T_cw_viz[:3, :3] @ scene_pts_world.T + T_cw_viz[:3, 3:4]).T
    in_front = pts_cam[:, 2] > 0.01
    if not in_front.any():
        return _DEFAULT_YFOV
    pf = pts_cam[in_front]
    half_y = np.abs(np.arctan2(pf[:, 1], pf[:, 2]))
    half_x = np.abs(np.arctan2(pf[:, 0], pf[:, 2]))
    aspect = W / H
    needed = max(float(np.max(half_y)), float(np.max(half_x)) / aspect)
    return float(np.clip(needed * 2.0 * 1.15, 0.1, np.pi * 0.9))


def render_tg_3d_depth(
    depth_pcl_n6: Optional[np.ndarray],
    gt_trace_world_tp3: np.ndarray,
    pred_trace_world_tp3: Optional[np.ndarray],
    trace_valid_tp: np.ndarray,
    drawn_idx: np.ndarray,
    H: int = 720,
    W: int = 1280,
    known_n: int = 1,
    known_start: int = 0,
    mode: str = "both",
) -> List[np.ndarray]:
    """Render a 3D animation with depth point cloud + animated trajectories.

    Auto-frames a side-perspective viz camera from the trace bbox and uses
    the combined (trace + depth-pcl) points to compute a tight yfov. The
    depth dot layer is rasterised once (far-to-near) and reused for every
    frame; on top we draw per-timestep polylines, current-t dots, and the
    timeline bar. OpenCV convention, no pyrender, no OpenGL.

    Args:
        depth_pcl_n6: ``(N, 6) float32 [x,y,z,r,g,b]`` scene scaffold or
            ``None``.
        gt_trace_world_tp3: ``(T, P, 3)`` GT traces in world (anchor-camera)
            space, i.e. output of ``backproject_trace_tg``.
        pred_trace_world_tp3: same-shape pred or ``None``.
        trace_valid_tp: ``(T, P)`` bool validity.
        drawn_idx: ``(N_drawn,) int32`` index into ``P``.
        H, W: output resolution (video body; timeline bar adds extra rows).
        known_n: number of known frames for the timeline bar.
        known_start: start of the known block.

    Returns:
        List of ``(H + bar_h, W, 3) uint8`` frames of length ``T``.
    """
    T, P, _ = gt_trace_world_tp3.shape
    if trace_valid_tp.shape != (T, P):
        raise ValueError(
            f"render_tg_3d_depth: trace_valid_tp {trace_valid_tp.shape} "
            f"!= (T,P)={T,P}"
        )
    tl_h = max(int(H * 0.025), 12)
    bg = np.full((H, W, 3), 242, dtype=np.uint8)

    n_drawn = int(drawn_idx.size)
    if n_drawn == 0:
        frames: List[np.ndarray] = []
        for t_i in range(T):
            tl_bar = _make_timeline_bar(
                W, T, t_i, known_start, known_n, bar_h=tl_h
            )
            frames.append(np.concatenate([tl_bar, bg.copy()], axis=0))
        return frames

    sel_valid = trace_valid_tp[:, drawn_idx]                 # (T, n_drawn)
    sel_gt = gt_trace_world_tp3[:, drawn_idx, :]             # (T, n_drawn, 3)
    has_pred = pred_trace_world_tp3 is not None
    # Narrow pred_trace_world_tp3 to a named, non-Optional handle so downstream
    # code blocks can safely reference it without Pyright "possibly unbound"
    # warnings under conditional control flow.
    if pred_trace_world_tp3 is not None:
        if pred_trace_world_tp3.shape != gt_trace_world_tp3.shape:
            raise ValueError(
                f"render_tg_3d_depth: pred shape {pred_trace_world_tp3.shape} "
                f"!= GT {gt_trace_world_tp3.shape}"
            )
        sel_pred: np.ndarray = pred_trace_world_tp3[:, drawn_idx, :]  # (T, n_drawn, 3)
    else:
        sel_pred = np.zeros((T, n_drawn, 3), dtype=np.float32)

    # --- Auto-frame a side-perspective viz camera -----------------------
    all_gt = sel_gt[sel_valid]                               # (N_valid, 3)
    if all_gt.size == 0:
        centroid = np.zeros(3, dtype=np.float32)
        extent = 1.0
    else:
        bbox_min = all_gt.min(axis=0)
        bbox_max = all_gt.max(axis=0)
        centroid = 0.5 * (bbox_min + bbox_max)
        extent = float(max((bbox_max - bbox_min).max(), 1e-3))

    elev = np.deg2rad(45.0)
    azim = np.deg2rad(45.0)
    distance = max(extent * 1.8 / np.tan(_DEFAULT_YFOV / 2.0), 0.8)
    eye_offset = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=np.float32,
    )
    eye = (centroid + distance * eye_offset).astype(np.float32)
    viz_T_cam = _look_at_opencv(eye, centroid.astype(np.float32))  # (4, 4)
    T_cw_viz = np.linalg.inv(viz_T_cam).astype(np.float32)

    # --- Compute auto-yfov from all scene points ------------------------
    scene_pts_list = [all_gt] if all_gt.size > 0 else []
    if has_pred:
        pred_valid = sel_pred[sel_valid]
        if pred_valid.size > 0:
            scene_pts_list.append(pred_valid)
    if depth_pcl_n6 is not None and depth_pcl_n6.size > 0:
        scene_pts_list.append(depth_pcl_n6[:, :3])
    if scene_pts_list:
        scene_pts = np.concatenate(scene_pts_list, axis=0)
    else:
        scene_pts = np.zeros((0, 3), dtype=np.float32)
    yfov = _auto_yfov(scene_pts, T_cw_viz, H, W)
    fy_px = (H / 2.0) / np.tan(yfov / 2.0)
    fx_px = fy_px

    def _project(pts_3d: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 2) pixel coords."""
        pts_cam = (T_cw_viz[:3, :3] @ pts_3d.T + T_cw_viz[:3, 3:4]).T
        z = pts_cam[:, 2].clip(1e-6, None)
        u = fx_px * pts_cam[:, 0] / z + W / 2.0
        v = fy_px * pts_cam[:, 1] / z + H / 2.0
        return np.stack([u, v], axis=-1).astype(np.float32)

    # --- Pre-project traces ---------------------------------------------
    gt_uv = np.full((T, n_drawn, 2), np.nan, dtype=np.float32)
    for t_i in range(T):
        vt = sel_valid[t_i]
        if vt.any():
            gt_uv[t_i, vt] = _project(sel_gt[t_i, vt])
    gt_uv_int = np.nan_to_num(gt_uv, nan=0.0).astype(np.int32)

    pr_uv: np.ndarray = np.full((T, n_drawn, 2), np.nan, dtype=np.float32)
    pr_uv_int: np.ndarray = np.zeros((T, n_drawn, 2), dtype=np.int32)
    if has_pred:
        for t_i in range(T):
            vt = sel_valid[t_i]
            if vt.any():
                pr_uv[t_i, vt] = _project(sel_pred[t_i, vt])
        pr_uv_int = np.nan_to_num(pr_uv, nan=0.0).astype(np.int32)

    # --- Pre-render depth dot layer -------------------------------------
    depth_dot_r = max(3, int(min(H, W) * 0.005))
    track_dot_r = max(2, int(min(H, W) * 0.004))

    depth_canvas = bg.copy()
    if depth_pcl_n6 is not None and depth_pcl_n6.size > 0:
        pts_cam_d = (
            T_cw_viz[:3, :3] @ depth_pcl_n6[:, :3].T + T_cw_viz[:3, 3:4]
        ).T
        in_front = pts_cam_d[:, 2] > 0.01
        if in_front.any():
            pf = pts_cam_d[in_front]
            cols_f = depth_pcl_n6[in_front, 3:6].astype(np.uint8)
            u_d = fx_px * pf[:, 0] / pf[:, 2] + W / 2.0
            v_d = fy_px * pf[:, 1] / pf[:, 2] + H / 2.0
            z_d = pf[:, 2]
            order = np.argsort(-z_d)  # far-to-near

            depth_layer = np.zeros((H, W, 3), dtype=np.uint8)
            depth_mask = np.zeros((H, W), dtype=np.float32)
            d_uv_int = np.round(np.stack([u_d, v_d], axis=-1)).astype(np.int32)
            d_rgb_list = [tuple(int(x) for x in cols_f[i].tolist()) for i in range(cols_f.shape[0])]
            for idx in order:
                cx, cy = int(d_uv_int[idx, 0]), int(d_uv_int[idx, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(
                        depth_layer, (cx, cy), depth_dot_r,
                        d_rgb_list[idx], -1, cv2.LINE_AA,
                    )
                    cv2.circle(depth_mask, (cx, cy), depth_dot_r, 1, -1, cv2.LINE_AA)
            a3 = (depth_mask * 0.4)[:, :, None]
            depth_canvas = (
                depth_layer.astype(np.float32) * a3
                + bg.astype(np.float32) * (1.0 - a3)
            ).clip(0, 255).astype(np.uint8)

    # --- Per-track color tuples -----------------------------------------
    colors = _generate_track_colors(n_drawn)
    py_gt = [tuple(int(x) for x in colors[i].tolist()) for i in range(n_drawn)]
    py_pr = [
        (
            int(max(0, min(255, colors[i, 0] * 0.65))),
            int(max(0, min(255, colors[i, 1] * 0.65))),
            int(max(0, min(255, colors[i, 2] * 0.65))),
        )
        for i in range(n_drawn)
    ]

    if mode not in ("gt", "pred", "both"):
        raise ValueError(f"render_tg_3d_depth: mode must be gt/pred/both, got {mode!r}")
    if mode == "pred" and pred_trace_world_tp3 is None:
        raise ValueError("render_tg_3d_depth: mode='pred' requires pred_trace_world_tp3")
    draw_gt = mode in ("gt", "both")
    draw_pred = mode in ("pred", "both") and has_pred
    pred_use_all_t = (mode == "pred")
    pred_dashed = (mode == "both")
    line_w_3d = 1 if n_drawn > 50 else 2
    track_dot_r = max(2 if n_drawn > 50 else 3, track_dot_r)

    frames = []
    for t_i in range(T):
        canvas = depth_canvas.copy()
        for pi in range(n_drawn):
            gt_color = py_gt[pi]
            # Pred-only mode: full-bright color so single-stream is the primary signal.
            pr_color = py_gt[pi] if mode == "pred" else py_pr[pi]

            if draw_gt:
                for tt in range(1, t_i + 1):
                    if not (sel_valid[tt, pi] and sel_valid[tt - 1, pi]):
                        continue
                    if np.isnan(gt_uv[tt - 1, pi]).any() or np.isnan(gt_uv[tt, pi]).any():
                        continue
                    x1 = int(gt_uv_int[tt - 1, pi, 0]); y1 = int(gt_uv_int[tt - 1, pi, 1])
                    x2 = int(gt_uv_int[tt, pi, 0]);     y2 = int(gt_uv_int[tt, pi, 1])
                    cv2.line(canvas, (x1, y1), (x2, y2), gt_color, line_w_3d, cv2.LINE_AA)

            if draw_pred:
                for tt in range(1, t_i + 1):
                    if pred_dashed and (tt % 2) == 0:
                        continue
                    if (not pred_use_all_t) and not (sel_valid[tt, pi] and sel_valid[tt - 1, pi]):
                        continue
                    if np.isnan(pr_uv[tt - 1, pi]).any() or np.isnan(pr_uv[tt, pi]).any():
                        continue
                    x1 = int(pr_uv_int[tt - 1, pi, 0]); y1 = int(pr_uv_int[tt - 1, pi, 1])
                    x2 = int(pr_uv_int[tt, pi, 0]);     y2 = int(pr_uv_int[tt, pi, 1])
                    cv2.line(canvas, (x1, y1), (x2, y2), pr_color, line_w_3d, cv2.LINE_AA)

            # Current-t dots.
            if draw_gt and sel_valid[t_i, pi] and not np.isnan(gt_uv[t_i, pi]).any():
                cx = int(gt_uv_int[t_i, pi, 0]); cy = int(gt_uv_int[t_i, pi, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(canvas, (cx, cy), track_dot_r, gt_color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), track_dot_r, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_pred:
                pred_dot_valid = pred_use_all_t or sel_valid[t_i, pi]
                if pred_dot_valid and not np.isnan(pr_uv[t_i, pi]).any():
                    cxp = int(pr_uv_int[t_i, pi, 0]); cyp = int(pr_uv_int[t_i, pi, 1])
                    if 0 <= cxp < W and 0 <= cyp < H:
                        cv2.circle(canvas, (cxp, cyp), track_dot_r, pr_color, -1, cv2.LINE_AA)

        tl_bar = _make_timeline_bar(W, T, t_i, known_start, known_n, bar_h=tl_h)
        frames.append(np.concatenate([tl_bar, canvas], axis=0))

    logger.info(
        f"render_tg_3d_depth: produced {len(frames)} frames "
        f"(T={T}, n_drawn={n_drawn}, mode={mode}, has_pred={draw_pred}, "
        f"depth_pts={0 if depth_pcl_n6 is None else int(depth_pcl_n6.shape[0])}, "
        f"yfov={yfov:.3f})"
    )
    return frames
