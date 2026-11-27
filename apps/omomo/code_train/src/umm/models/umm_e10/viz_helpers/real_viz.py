"""Visualization for real-world point-tracking domains (DAVIS, Bridge, etc.).

Two visualization modes:
  1. **2D flow**: Projects 3D traces onto the anchor RGB frame using per-frame
     cameras. Draws colored trajectory curves and dot markers.
  2. **3D depth**: Renders a depth-based point cloud background with trace line
     segments (cylinders) via pyrender.

Example::

    viz = RealVisualizer(pl_module, real_cfg)
    viz.real_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from umm.models.umm_e10.viz_helpers.utils import (
    put_caption_at_bottom,
)
from umm.models.umm_e10.viz_helpers.behavior_viz_v2 import (
    render_pixel_query_overlay,
    _look_at_opencv,
)

import io
import os.path as osp
import zipfile

BACKGROUND_COLOR = (0.95, 0.95, 0.95, 1.0)
GROUND_COLOR = (0.85, 0.85, 0.85, 1.0)
_DEFAULT_YFOV = np.pi / 3.0



def _generate_track_colors(n: int) -> np.ndarray:
    """Generate N distinct colors using HSV colormap.  Returns (N, 3) uint8."""
    hues = np.linspace(0, 170, n, dtype=np.uint8)  # avoid wrap to red
    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = hues
    hsv[0, :, 1] = 220
    hsv[0, :, 2] = 230
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb[0]  # (N, 3) uint8


def _make_timeline_bar(W: int, T: int, t_i: int,
                       known_start: int, known_n: int,
                       bar_h: int = 18) -> np.ndarray:
    """Render a full-width timeline bar outside the video frame.

    Red = predicted, green = known (from known_start to known_start+known_n).
    White cursor at current playing position.

    Args:
        W: image width in pixels.
        T: total number of timesteps.
        t_i: current playing position.
        known_start: first known frame index.
        known_n: number of consecutive known frames from known_start.
        bar_h: bar height in pixels.

    Returns:
        (bar_h, W, 3) uint8 RGB.
    """
    bar = np.full((bar_h, W, 3), 200, dtype=np.uint8)  # base: red (predicted)
    if T <= 0:
        return bar

    # Fill entire bar with red (predicted)
    bar[:, :] = (200, 60, 60)

    # Green for known region: known_start .. known_start+known_n (full height)
    if known_n > 0:
        g_x0 = int(W * known_start / T)
        g_x1 = int(W * (known_start + known_n) / T)
        g_x1 = max(g_x1, g_x0 + 2)  # at least 2px wide
        bar[:, g_x0:min(g_x1, W)] = (60, 180, 75)

    # White cursor at current playing position (full height)
    cx = int(W * t_i / max(T - 1, 1))
    cx = int(np.clip(cx, 0, W - 1))
    bar[:, max(cx - 1, 0):min(cx + 2, W)] = 255

    # Label — right-aligned, compact text
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(bar_h / 40.0, 0.3)
    thick = 1
    label = f"t={t_i}/{T}"
    (tw, th), _ = cv2.getTextSize(label, font, sc, thick)
    ty = (bar_h + th) // 2  # vertically centered
    tx = max(W - tw - 4, 2)
    cv2.putText(bar, label, (tx, ty), font, sc,
                (255, 255, 255), thick, cv2.LINE_AA)

    return bar


def render_2d_flow(
    rgb_frame: np.ndarray,
    positions: np.ndarray,
    trace_mask: np.ndarray,
    K_per_frame: np.ndarray,
    T_wc_per_frame: np.ndarray,
    T_align: np.ndarray,
    anchor_uv: Optional[np.ndarray] = None,
    known_n: int = -1,
    known_start: int = 0,
    max_tracks: int = 256,
    anchor_ith: int = 0,
    scene_scale: float = 1.0,
) -> list:
    """Render 2D flow curves on the anchor RGB frame.

    All 3D trace positions are projected through the **fixed anchor camera**
    so curves show true 3D world-space motion from a static viewpoint.
    A timeline bar at the top shows history (green) vs predicted (red) with
    a cursor at the current timestep.

    Args:
        rgb_frame: (H, W, 3) uint8 — anchor frame in resized resolution.
        positions: (T, P, 3) xyz in aligned frame (T_align applied).
        trace_mask: (T, P) bool validity mask.
        K_per_frame: (T, 3, 3) resized intrinsics per frame.
        T_wc_per_frame: (T, 4, 4) camera-to-world (original).
        T_align: (4, 4) scene alignment transform applied to traces.
        anchor_uv: (P, 2) pixel coords in resized image for static markers.
        known_n: frames 0..known_n-1 are "seen", rest are predicted.
        max_tracks: subsample for visual clarity.
        anchor_ith: index of anchor frame within the window (for K and T_wc).
        scene_scale: per-scene depth rescaling factor applied to traces.

    Returns:
        List of (H + bar_h, W, 3) uint8 frames (one per timestep, curve grows).
    """
    T, P = trace_mask.shape
    H, W = rgb_frame.shape[:2]

    # Subsample tracks for clarity
    valid_parts = np.where(trace_mask.any(axis=0))[0]
    if len(valid_parts) == 0:
        return [rgb_frame.copy() for _ in range(max(T, 1))]
    if len(valid_parts) > max_tracks:
        rng = np.random.RandomState(42)
        sel = rng.choice(valid_parts, max_tracks, replace=False)
    else:
        sel = valid_parts

    colors = _generate_track_colors(len(sel))

    # Undo T_align (+ scene_scale) to recover original world positions for projection.
    T_align_inv = np.linalg.inv(T_align).astype(np.float32)  # (4, 4)
    pos = positions[:, sel, :].copy()  # (T, N_sel, 3)
    if scene_scale != 1.0 and scene_scale > 1e-8:
        pos /= scene_scale
    ones = np.ones((*pos.shape[:2], 1), dtype=np.float32)
    pos_h = np.concatenate([pos, ones], axis=-1)  # (T, N_sel, 4)
    pos = np.einsum('ij,tnj->tni', T_align_inv, pos_h)[:, :, :3]

    sel_mask = trace_mask[:, sel]  # (T, N_sel)

    # Project ALL 3D points through the fixed anchor camera
    T_cw_anchor = np.linalg.inv(T_wc_per_frame[anchor_ith])  # (4, 4) world-to-camera
    K_anchor = K_per_frame[anchor_ith]  # (3, 3)
    uv_2d = np.full((T, len(sel), 2), np.nan, dtype=np.float32)
    for t_i in range(T):
        pts_cam = (T_cw_anchor[:3, :3] @ pos[t_i].T + T_cw_anchor[:3, 3:4]).T  # (N_sel, 3)
        z = pts_cam[:, 2].clip(1e-6, None)
        uv_2d[t_i, :, 0] = K_anchor[0, 0] * pts_cam[:, 0] / z + K_anchor[0, 2]
        uv_2d[t_i, :, 1] = K_anchor[1, 1] * pts_cam[:, 1] / z + K_anchor[1, 2]

    # Timeline bar height
    tl_h = max(int(H * 0.06), 12)

    # Build video frames — each frame shows curves up to that timestep
    # Pre-convert colors to Python int tuples (cv2 rejects numpy scalars)
    py_colors = [tuple(int(x) for x in colors[i].tolist()) for i in range(len(colors))]
    # Pre-convert uv_2d to Python int array for cv2 compatibility
    uv_int = np.nan_to_num(uv_2d, nan=0.0).astype(np.int32)  # (T, N_sel, 2)

    frames = []
    for t_i in tqdm(range(T), desc="2D flow", leave=False):
        canvas = rgb_frame.copy()

        for pi in range(len(sel)):
            color = py_colors[pi]

            # Draw curve from frame 0 to t_i
            for tt in range(1, t_i + 1):
                if not sel_mask[tt, pi] or not sel_mask[tt - 1, pi]:
                    continue
                x1, y1 = int(uv_int[tt - 1, pi, 0]), int(uv_int[tt - 1, pi, 1])
                x2, y2 = int(uv_int[tt, pi, 0]), int(uv_int[tt, pi, 1])
                if not (0 <= x1 < W and 0 <= y1 < H
                        and 0 <= x2 < W and 0 <= y2 < H):
                    continue
                thickness = 1 if (known_n >= 0 and tt >= known_n) else 2
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Draw dot at current position
            if sel_mask[t_i, pi]:
                cx, cy = int(uv_int[t_i, pi, 0]), int(uv_int[t_i, pi, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(canvas, (cx, cy), 3, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), 1, cv2.LINE_AA)

        # Static anchor cross markers for selected tracks only
        if anchor_uv is not None:
            for pi in range(len(sel)):
                part_idx = sel[pi]
                if part_idx < len(anchor_uv) and not np.isnan(anchor_uv[part_idx]).any():
                    ax = int(anchor_uv[part_idx, 0].item())
                    ay = int(anchor_uv[part_idx, 1].item())
                    if 0 <= ax < W and 0 <= ay < H:
                        cv2.drawMarker(
                            canvas, (ax, ay), py_colors[pi],
                            cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA,
                        )

        # Add timeline bar at top
        tl_bar = _make_timeline_bar(W, T, t_i, known_start, known_n, bar_h=tl_h)
        canvas = np.concatenate([tl_bar, canvas], axis=0)

        frames.append(canvas)

    return frames


def load_depth_pcl(
    data_root: str,
    seq_id: str,
    frame_idx: int,
    K_orig: np.ndarray,
    T_wc: np.ndarray,
    T_align: np.ndarray,
    max_points: int = 20000,
    stride: int = 2,
    scene_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """Load depth map and unproject to 3D in the aligned coordinate frame.

    Args:
        data_root: Path to data root.
        seq_id: Sequence identifier.
        frame_idx: Absolute frame index.
        K_orig: (3, 3) original camera intrinsics (not resized).
        T_wc: (4, 4) camera-to-world for this frame.
        T_align: (4, 4) scene alignment transform (applied to world positions).
        max_points: Maximum points to return.
        stride: Pixel stride for subsampling depth.
        scene_scale: Per-scene depth rescaling factor (applied after T_align).

    Returns:
        (N, 6) float32: [x, y, z, r, g, b] in transformed frame, or None.
    """
    depth_path = osp.join(data_root, seq_id, "depths.zip")
    if not osp.exists(depth_path):
        return None

    fname = f"{frame_idx:06d}.npz"
    try:
        with zipfile.ZipFile(depth_path) as zf:
            with zf.open(fname) as f:
                depth = np.load(io.BytesIO(f.read()))["depth"]  # (H, W) float16
    except (KeyError, zipfile.BadZipFile):
        return None

    depth = depth.astype(np.float32)
    H, W = depth.shape

    # Subsample with stride
    u_grid, v_grid = np.meshgrid(
        np.arange(0, W, stride), np.arange(0, H, stride)
    )
    u_flat = u_grid.ravel().astype(np.float32)
    v_flat = v_grid.ravel().astype(np.float32)
    d_flat = depth[v_grid, u_grid].ravel()

    # Filter invalid depth
    valid = d_flat > 0.01
    u_flat, v_flat, d_flat = u_flat[valid], v_flat[valid], d_flat[valid]

    # Unproject to camera space
    fx, fy = K_orig[0, 0], K_orig[1, 1]
    cx, cy = K_orig[0, 2], K_orig[1, 2]
    x_cam = (u_flat - cx) * d_flat / fx
    y_cam = (v_flat - cy) * d_flat / fy
    z_cam = d_flat
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Camera to world
    pts_world = (T_wc[:3, :3] @ pts_cam.T + T_wc[:3, 3:4]).T  # (N, 3)

    # Apply same T_align as traces (unified for yaw and camera modes)
    pts_world = (T_align[:3, :3] @ pts_world.T + T_align[:3, 3:4]).T  # (N, 3)

    # Apply per-scene depth rescaling (same as dataset)
    if scene_scale != 1.0:
        pts_world *= scene_scale

    # Get colors from the image at the same pixel locations
    images_path = osp.join(data_root, seq_id, "images.zip")
    rgb_colors = np.full((len(pts_world), 3), 180, dtype=np.uint8)  # default gray
    try:
        with zipfile.ZipFile(images_path) as zf:
            img_name = f"{frame_idx:06d}.jpg"
            with zf.open(img_name) as f:
                img_data = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # (H, W, 3) BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Sample colors at the original pixel coordinates (already filtered by valid mask)
            u_int = np.clip(u_flat, 0, img.shape[1] - 1).astype(int)
            v_int = np.clip(v_flat, 0, img.shape[0] - 1).astype(int)
            rgb_colors = img[v_int, u_int]  # (N, 3) uint8
    except Exception:
        pass  # keep default gray

    # Subsample if too many points
    if len(pts_world) > max_points:
        idx = np.random.RandomState(42).choice(len(pts_world), max_points, replace=False)
        pts_world = pts_world[idx]
        rgb_colors = rgb_colors[idx]

    # Return (N, 6): xyz + rgb
    result = np.zeros((len(pts_world), 6), dtype=np.float32)
    result[:, :3] = pts_world
    result[:, 3:6] = rgb_colors.astype(np.float32)
    return result


def render_3d_depth_traces(
    positions: np.ndarray,
    trace_mask: np.ndarray,
    depth_pcl: Optional[np.ndarray],
    viz_T_cam: np.ndarray,
    track_colors: Optional[np.ndarray] = None,
    known_n: int = -1,
    known_start: int = 0,
    max_tracks: int = 256,
    H: int = 720,
    W: int = 1280,
    fixed_yfov: Optional[float] = None,
) -> list:
    """Render 3D scene with depth point cloud + trace line segments.

    Args:
        positions: (T, P, 3) xyz in transformed frame.
        trace_mask: (T, P) bool validity.
        depth_pcl: (N, 6) [x,y,z,r,g,b] background cloud, or None.
        viz_T_cam: (4, 4) camera-to-world for viz.
        track_colors: (N_sel, 3) uint8 per-track colors.
        known_n: boundary between seen and predicted frames.
        max_tracks: subsample tracks for clarity.
        H, W: output resolution.

    Returns:
        List of (H, W, 4) uint8 RGBA frames.
    """
    T, P = trace_mask.shape

    # Subsample tracks
    valid_parts = np.where(trace_mask.any(axis=0))[0]
    if len(valid_parts) == 0:
        return [np.ones((H, W, 4), dtype=np.uint8) * 255 for _ in range(T)]
    if len(valid_parts) > max_tracks:
        sel = np.random.RandomState(42).choice(valid_parts, max_tracks, replace=False)
    else:
        sel = valid_parts

    if track_colors is None:
        track_colors = _generate_track_colors(len(sel))

    sel_mask = trace_mask[:, sel]  # (T, N_sel)
    sel_pos = positions[:, sel, :]  # (T, N_sel, 3)

    # Compute scene extent
    all_pos = sel_pos[sel_mask]
    if len(all_pos) == 0:
        return [np.ones((H, W, 4), dtype=np.uint8) * 242 for _ in range(T)], _DEFAULT_YFOV

    # Auto-zoom: compute tight yfov so scene fills the frame
    T_cw_viz = np.linalg.inv(viz_T_cam)  # (4, 4)
    if fixed_yfov is not None:
        _auto_yfov = fixed_yfov
    else:
        if depth_pcl is not None and len(depth_pcl) > 0:
            _scene_pts = np.concatenate([all_pos, depth_pcl[:, :3]], axis=0)
        else:
            _scene_pts = all_pos
        pts_cam_scene = (T_cw_viz[:3, :3] @ _scene_pts.T + T_cw_viz[:3, 3:4]).T  # (N, 3)
        _in_front = pts_cam_scene[:, 2] > 0.01
        if _in_front.any():
            pts_front = pts_cam_scene[_in_front]
            half_y = np.abs(np.arctan2(pts_front[:, 1], pts_front[:, 2]))
            half_x = np.abs(np.arctan2(pts_front[:, 0], pts_front[:, 2]))
            aspect = W / H
            needed = max(float(np.max(half_y)), float(np.max(half_x)) / aspect)
            _auto_yfov = float(np.clip(needed * 2.0 * 1.15, 0.1, np.pi * 0.9))
        else:
            _auto_yfov = _DEFAULT_YFOV

    # Focal length from yfov for all 2D projections
    fy_px = (H / 2.0) / np.tan(_auto_yfov / 2.0)
    fx_px = fy_px  # square pixels

    def _project_to_2d(pts_3d: np.ndarray) -> np.ndarray:
        """Project (N, 3) world points to (N, 2) pixel coords."""
        pts_cam = (T_cw_viz[:3, :3] @ pts_3d.T + T_cw_viz[:3, 3:4]).T  # (N, 3)
        z = pts_cam[:, 2].clip(1e-6, None)
        u = fx_px * pts_cam[:, 0] / z + W / 2.0
        v = fy_px * pts_cam[:, 1] / z + H / 2.0
        return np.stack([u, v], axis=-1).astype(np.float32)

    # Pre-project ALL track positions to 2D once — (T, N_sel, 2)
    track_uv = np.full((T, len(sel), 2), np.nan, dtype=np.float32)
    for t_i in range(T):
        valid_t = sel_mask[t_i]  # (N_sel,)
        if valid_t.any():
            track_uv[t_i, valid_t] = _project_to_2d(sel_pos[t_i, valid_t])

    # Pre-project depth points to 2D
    _depth_uv = None   # (N, 2) pixel coords
    _depth_rgb = None   # (N, 3) uint8 colors
    _depth_z = None     # (N,) depth for sorting
    if depth_pcl is not None and len(depth_pcl) > 0:
        pts_cam_d = (T_cw_viz[:3, :3] @ depth_pcl[:, :3].T + T_cw_viz[:3, 3:4]).T
        in_front = pts_cam_d[:, 2] > 0.01
        if in_front.any():
            pts_f = pts_cam_d[in_front]
            cols_f = depth_pcl[in_front, 3:6].astype(np.uint8)
            u = fx_px * pts_f[:, 0] / pts_f[:, 2] + W / 2.0
            v = fy_px * pts_f[:, 1] / pts_f[:, 2] + H / 2.0
            _depth_uv = np.stack([u, v], axis=-1).astype(np.float32)
            _depth_rgb = cols_f
            _depth_z = pts_f[:, 2]

    # Pixel sizes for drawing
    depth_dot_r = max(3, int(min(H, W) * 0.005))
    track_dot_r = max(2, int(min(H, W) * 0.004))

    # Background color
    bg = np.full((H, W, 3), 242, dtype=np.uint8)  # light gray
    tl_h = max(int(H * 0.025), 12)  # timeline bar height

    # Pre-convert track colors to Python int tuples (cv2 rejects numpy scalars)
    py_colors = []
    py_pred_colors = []
    for pi in range(len(sel)):
        cr, cg, cb = int(track_colors[pi, 0]), int(track_colors[pi, 1]), int(track_colors[pi, 2])
        py_colors.append((cr, cg, cb))
        py_pred_colors.append((max(0, min(255, int(cr * 0.6 + 100))),
                               max(0, min(255, int(cg * 0.6 + 100))),
                               max(0, min(255, int(cb * 0.6 + 100)))))

    # Pre-convert track_uv to int32 for cv2 (NaN → 0)
    track_uv_int = np.nan_to_num(track_uv, nan=0.0).astype(np.int32)  # (T, N_sel, 2)

    # Pre-render depth dots layer (static across frames — render once)
    _depth_canvas = None
    if _depth_uv is not None:
        depth_layer = np.zeros((H, W, 3), dtype=np.uint8)
        depth_mask_layer = np.zeros((H, W), dtype=np.float32)
        order = np.argsort(-_depth_z)  # far-to-near
        # Pre-convert depth colors to Python tuples
        _d_uv_int = np.round(_depth_uv).astype(np.int32)
        _d_rgb_list = [tuple(int(x) for x in _depth_rgb[i].tolist()) for i in range(len(_depth_rgb))]
        for idx in order:
            cx, cy = int(_d_uv_int[idx, 0]), int(_d_uv_int[idx, 1])
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(depth_layer, (cx, cy), depth_dot_r, _d_rgb_list[idx], -1, cv2.LINE_AA)
                cv2.circle(depth_mask_layer, (cx, cy), depth_dot_r, 1, -1, cv2.LINE_AA)
        # Pre-blend depth onto background (same for every frame)
        a3 = (depth_mask_layer * 0.4)[:, :, None]
        _depth_canvas = (depth_layer.astype(np.float32) * a3
                         + bg.astype(np.float32) * (1.0 - a3)).clip(0, 255).astype(np.uint8)

    frames = []
    for t_i in tqdm(range(T), desc="3D depth", leave=False):
        canvas = _depth_canvas.copy() if _depth_canvas is not None else bg.copy()

        # 2. Draw track lines and dots
        for pi in range(len(sel)):
            color = py_colors[pi]
            pred_color = py_pred_colors[pi]

            # Draw line segments from frame 0 to t_i
            for tt in range(1, t_i + 1):
                if not sel_mask[tt, pi] or not sel_mask[tt - 1, pi]:
                    continue
                if np.isnan(track_uv[tt - 1, pi]).any() or np.isnan(track_uv[tt, pi]).any():
                    continue
                x1, y1 = int(track_uv_int[tt - 1, pi, 0]), int(track_uv_int[tt - 1, pi, 1])
                x2, y2 = int(track_uv_int[tt, pi, 0]), int(track_uv_int[tt, pi, 1])
                is_pred = (known_n >= 0 and tt >= known_n)
                col = pred_color if is_pred else color
                thick = 1 if is_pred else 2
                cv2.line(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)

            # Draw dot at current position
            if sel_mask[t_i, pi] and not np.isnan(track_uv[t_i, pi]).any():
                cx = int(track_uv_int[t_i, pi, 0])
                cy = int(track_uv_int[t_i, pi, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    is_pred = (known_n >= 0 and t_i >= known_n)
                    col = pred_color if is_pred else color
                    cv2.circle(canvas, (cx, cy), track_dot_r, col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), track_dot_r, (255, 255, 255), 1, cv2.LINE_AA)

        # Timeline bar on top (consistent with 2D flow viz)
        tl_bar = _make_timeline_bar(W, T, t_i, known_start, known_n, bar_h=tl_h)
        canvas = np.concatenate([tl_bar, canvas], axis=0)  # (H+tl_h, W, 3)

        # Output as RGBA for compatibility
        alpha = np.full((canvas.shape[0], W, 1), 255, dtype=np.uint8)
        frames.append(np.concatenate([canvas, alpha], axis=-1))

    return frames, _auto_yfov


def load_gt_video_frames(
    data_root: str,
    seq_id: str,
    frame_indices: np.ndarray,
    target_wh: Optional[tuple] = None,
) -> Optional[np.ndarray]:
    """Load GT video frames from images.zip for the selected frame indices.

    Args:
        data_root: Path to data root.
        seq_id: Sequence identifier.
        frame_indices: (T,) array of absolute frame indices.
        target_wh: Optional (W, H) to resize frames. None = original resolution.

    Returns:
        (T, H, W, 3) uint8 RGB array, or None on failure.
    """
    images_path = osp.join(data_root, seq_id, "images.zip")
    if not osp.exists(images_path):
        return None

    frames = []
    try:
        with zipfile.ZipFile(images_path) as zf:
            for fi in frame_indices:
                fname = f"{int(fi):06d}.jpg"
                with zf.open(fname) as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # (H, W, 3) BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if target_wh is not None:
                    img = cv2.resize(img, target_wh, interpolation=cv2.INTER_LINEAR)
                frames.append(img)
    except Exception as e:
        logging.debug(f"Failed to load GT video frames: {e}")
        return None

    if not frames:
        return None
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


class RealVisualizer:
    """Visualizer for real-world domains with 2D flow and 3D depth trace modes.

    Produces two types of visualization:
      - **2D flow videos**: trajectory curves overlaid on the anchor RGB frame.
      - **3D depth videos**: depth point cloud + trace cylinders in 3D.

    Example::

        viz = RealVisualizer(pl_module, real_cfg)
        viz.real_viz_routine(batch, "val", 0, eval_model, step_output)
    """

    def __init__(self, pl_module, real_cfg: dict):
        self.pl = pl_module
        if real_cfg == {} or len(real_cfg) == 0 or real_cfg is None:
            self.flag = False
            return
        self.flag = True

        self.real_viz_cfg = self.pl.cfg.get("visualize", {}).get("real", {})
        self.add_cfg = self.real_viz_cfg.get("add_cfg", [1.0])
        self.max_tracks = self.real_viz_cfg.get("max_tracks", 256)

    def make_viz_T_cam(self, batch, bid):
        """Auto-frame camera: compute bounding box of all points, look at centroid.

        Places the camera at an elevated position looking down at the centroid
        of all visible trajectory positions.

        Args:
            batch: Batch data dict.
            bid: Batch item index.

        Returns:
            np.ndarray: 4x4 camera-to-world matrix (OpenCV convention).
        """
        # Gather ALL positions from trace + context for camera framing
        n_steps = int(batch["trace_mask"][bid].any(dim=1).sum())
        T_wl = batch["trace"][bid, :n_steps].cpu().numpy()  # (T, N, 4, 4)
        ctx = batch["context"][bid].cpu().numpy()  # (C, N, 4, 4)

        # Combine trace and context positions
        trace_pos = T_wl[:, :, :3, 3].reshape(-1, 3)  # (T*N, 3)
        ctx_pos = ctx[:, :, :3, 3].reshape(-1, 3)  # (C*N, 3)
        all_pos = np.concatenate([trace_pos, ctx_pos], axis=0)

        # Filter out NaN and near-zero positions (padding)
        valid = ~np.isnan(all_pos).any(axis=1)
        positions = all_pos[valid]
        norms = np.linalg.norm(positions, axis=1)
        positions = positions[norms > 0.01]
        if len(positions) == 0:
            positions = np.zeros((1, 3))

        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)

        centroid = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        max_extent = max(extent[0], extent[1], extent[2], 0.5)
        half_fov = _DEFAULT_YFOV / 2.0
        distance = max((max_extent / 2.0) / np.tan(half_fov) * 1.8, 0.8)

        logging.debug(
            f"Real viz cam: centroid={centroid}, extent={extent}, "
            f"n_valid={len(positions)}, distance={distance:.2f}"
        )

        # Slightly elevated viewpoint
        elev = np.radians(45.0)
        azimuth = np.radians(45.0)

        eye = centroid + np.array(
            [
                distance * np.cos(elev) * np.cos(azimuth),
                distance * np.cos(elev) * np.sin(azimuth),
                distance * np.sin(elev),
            ]
        )
        return _look_at_opencv(eye, centroid)

    @torch.inference_mode()
    def real_viz_routine(
        self, batch, phase, batch_idx, eval_model, step_output=None
    ):
        """Top-level visualization routine for real-world domains.

        Same structure as behavior_viz_routine: GT, X0_PRED, and inference
        outputs. No robot UID filtering -- uses bid=0 directly.

        Args:
            batch: Batch data dict.
            phase: ``"train"`` or ``"val"``.
            batch_idx: Index of the current batch.
            eval_model: EMA or main model for inference.
            step_output: Output dict from training/validation step.
        """
        if not self.flag:
            return

        # Use bid=0 -- no robot UID check needed for real-world domains
        bid = 0
        if len(batch["trace"]) == 0:
            return

        viz_T_cam = self.make_viz_T_cam(batch, bid)
        # Count valid timesteps: any part valid at that timestep counts
        n_steps = int(batch["trace_mask"][bid].any(dim=1).sum())

        logging.info(
            f"VIZ Real phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}"
        )

        # --- GT multiview RGB logging ---
        extra = batch.get("extra", {})
        if "images" in extra and extra["images"][bid] is not None:
            _imgs = extra["images"][bid]
            if isinstance(_imgs, torch.Tensor):
                _imgs_np = _imgs.cpu().numpy()
            else:
                _imgs_np = _imgs
            # _imgs_np is (V, 3, H, W) or (V, H, W, 3)
            if _imgs_np.ndim == 4 and _imgs_np.shape[1] == 3:
                _imgs_np = _imgs_np.transpose(0, 2, 3, 1)
            if _imgs_np.dtype != np.uint8:
                if _imgs_np.max() <= 1.0:
                    _imgs_np = (_imgs_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    _imgs_np = _imgs_np.clip(0, 255).astype(np.uint8)
            # Log each view as a static image
            for vi in range(_imgs_np.shape[0]):
                img_t = (
                    torch.from_numpy(_imgs_np[vi]).float().permute(2, 0, 1) / 255.0
                )
                self.pl.logger.log_image(
                    phase,
                    f"real_input_view_{vi}",
                    img_t,
                    self.pl.global_step,
                    batch_idx,
                )

        # --- GT video clip ---
        if "_data_root" in extra and extra["_data_root"][bid] is not None:
            _gt_data_root = extra["_data_root"][bid]
            _gt_seq_id = extra.get("_seq_id", [None])[bid]
            _gt_fi = extra.get("_frame_indices", [None])[bid]
            if _gt_seq_id and _gt_fi is not None:
                if isinstance(_gt_fi, torch.Tensor):
                    _gt_fi = _gt_fi.cpu().numpy()
                # Determine target size from anchor image if available
                _gt_wh = None
                if "images" in extra and extra["images"][bid] is not None:
                    _tmp = extra["images"][bid]
                    if isinstance(_tmp, torch.Tensor):
                        _tmp = _tmp.cpu().numpy()
                    # (V, 3, H, W) or (V, H, W, 3)
                    if _tmp.ndim == 4 and _tmp.shape[1] == 3:
                        _gt_wh = (_tmp.shape[3], _tmp.shape[2])  # (W, H)
                    elif _tmp.ndim == 4:
                        _gt_wh = (_tmp.shape[2], _tmp.shape[1])  # (W, H)
                gt_video = load_gt_video_frames(
                    data_root=_gt_data_root,
                    seq_id=_gt_seq_id,
                    frame_indices=_gt_fi,
                    target_wh=_gt_wh,
                )
                if gt_video is not None:
                    gt_vid_t = (
                        torch.from_numpy(gt_video).float().permute(0, 3, 1, 2) / 255.0
                    )  # (T, 3, H, W)
                    self.pl.logger.log_video(
                        phase,
                        "real_GT_video",
                        gt_vid_t,
                        self.pl.global_step,
                        batch_idx,
                        caption=f"{_gt_seq_id} ({len(_gt_fi)} frames)",
                    )

        # --- Pixel query click overlay ---
        if (
            "images" in extra
            and extra["images"][bid] is not None
            and "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
        ):
            overlay = render_pixel_query_overlay(
                images=extra["images"][bid],
                pixel_uv=extra["pixel_uv_normalized"][bid],
                pixel_view_idx=extra["pixel_view_idx"][bid],
                pixel_query_mask=extra["pixel_query_mask"][bid],
            )
            if overlay is not None:
                overlay_t = (
                    torch.from_numpy(overlay).float().permute(2, 0, 1) / 255.0
                )
                self.pl.logger.log_image(
                    phase,
                    "real_pixel_query_clicks",
                    overlay_t,
                    self.pl.global_step,
                    batch_idx,
                )

        # --- Build camera data for 2D flow projection ---
        has_cam_data = False
        K_window = None
        T_wc_window = None
        anchor_rgb = None
        anchor_uv_resized = None
        anchor_ith_val = 0
        scene_scale_val = 1.0
        T_align_np = np.eye(4, dtype=np.float32)  # default: identity (no transform)

        # Extract T_align — unified scene alignment transform
        if "_T_align" in extra and extra["_T_align"][bid] is not None:
            _ta = extra["_T_align"][bid]
            if isinstance(_ta, torch.Tensor):
                T_align_np = _ta.cpu().numpy().astype(np.float32)
            else:
                T_align_np = np.asarray(_ta, dtype=np.float32)

        # Extract scene_scale early — needed for both 2D flow and 3D depth viz
        if "_scene_scale" in extra and extra["_scene_scale"][bid] is not None:
            _ss = extra["_scene_scale"][bid]
            if isinstance(_ss, torch.Tensor):
                scene_scale_val = float(_ss.cpu().item())
            else:
                scene_scale_val = float(_ss)

        if "_cam_K_all" in extra and extra["_cam_K_all"][bid] is not None:
            _K_all_o = extra["_cam_K_all"][bid]  # (T_total, 3, 3)
            _T_wc_all = extra["_cam_T_wc_all"][bid]  # (T_total, 4, 4)
            _fi = extra["_frame_indices"][bid]  # (t_len,) abs frame indices
            _sxy = extra["_scale_xy"][bid]  # (2,)

            # Convert to numpy if needed
            if isinstance(_K_all_o, torch.Tensor):
                _K_all_o = _K_all_o.cpu().numpy()
            if isinstance(_T_wc_all, torch.Tensor):
                _T_wc_all = _T_wc_all.cpu().numpy()
            if isinstance(_fi, torch.Tensor):
                _fi = _fi.cpu().numpy()
            if isinstance(_sxy, torch.Tensor):
                _sxy = _sxy.cpu().numpy()

            fi_int = _fi.astype(int)
            K_window = _K_all_o[fi_int].copy()  # (t_len, 3, 3)
            T_wc_window = _T_wc_all[fi_int].copy()  # (t_len, 4, 4)

            # Scale K for the resize
            sx, sy = float(_sxy[0]), float(_sxy[1])
            K_window[:, 0, 0] *= sx
            K_window[:, 0, 2] *= sx
            K_window[:, 1, 1] *= sy
            K_window[:, 1, 2] *= sy
            has_cam_data = True

            # Compute anchor frame index within window
            _anchor_abs = extra.get("_anchor_frame", [None])[bid]
            _anchor_ith_raw = extra.get("_anchor_ith", [None])[bid]
            if _anchor_ith_raw is not None:
                anchor_ith_val = int(_anchor_ith_raw)
            elif _anchor_abs is not None:
                # Fallback: find anchor in frame_indices
                matches = np.where(fi_int == int(_anchor_abs))[0]
                anchor_ith_val = int(matches[0]) if len(matches) > 0 else 0

        # Get anchor RGB for 2D flow background
        if "images" in extra and extra["images"][bid] is not None:
            _img_flow = extra["images"][bid]
            if isinstance(_img_flow, torch.Tensor):
                _img_flow = _img_flow.cpu().numpy()
            if _img_flow.ndim == 4 and _img_flow.shape[1] == 3:
                _img_flow = _img_flow.transpose(0, 2, 3, 1)  # (V, H, W, 3)
            if _img_flow.dtype != np.uint8:
                if _img_flow.max() <= 1.0:
                    _img_flow = (_img_flow * 255).clip(0, 255).astype(np.uint8)
                else:
                    _img_flow = _img_flow.clip(0, 255).astype(np.uint8)
            anchor_rgb = _img_flow[0]  # (H, W, 3) view 0

        # Anchor UV in resized pixel space for static markers
        if (
            "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
            and anchor_rgb is not None
        ):
            _uv_n = extra["pixel_uv_normalized"][bid]
            if isinstance(_uv_n, torch.Tensor):
                _uv_n = _uv_n.cpu().numpy()
            rH, rW = anchor_rgb.shape[:2]
            anchor_uv_resized = np.stack([
                _uv_n[:, 0] * (rW - 1),
                _uv_n[:, 1] * (rH - 1),
            ], axis=-1)  # (P, 2) pixel coords in resized image

        def _log_2d_flow(trace_tensor, name, known_n_val, known_start_val=0):
            """Render and log a 2D flow video."""
            if not has_cam_data or anchor_rgb is None:
                return
            tr = trace_tensor.cpu().numpy() if torch.is_tensor(trace_tensor) else trace_tensor
            n_fr = tr.shape[0]
            n_cam = K_window.shape[0]
            n_use = min(n_fr, n_cam)
            positions = tr[:n_use, :, :3, 3]  # (n_use, P, 3)
            mask_np = batch["trace_mask"][bid, :n_use].cpu().numpy().astype(bool)

            # Pass full K/T_wc window (not sliced) — anchor_ith indexes into full window
            flow_frames = render_2d_flow(
                rgb_frame=anchor_rgb,
                positions=positions,
                trace_mask=mask_np,
                K_per_frame=K_window,
                T_wc_per_frame=T_wc_window,
                T_align=T_align_np,
                anchor_uv=anchor_uv_resized,
                known_n=known_n_val,
                known_start=known_start_val,
                max_tracks=self.max_tracks,
                anchor_ith=anchor_ith_val,
                scene_scale=scene_scale_val,
            )
            flow_frames = put_caption_at_bottom(flow_frames, batch["text"][bid])
            vid_t = (
                torch.from_numpy(np.stack(flow_frames, 0)).float().permute(0, 3, 1, 2)
                / 255.0
            )
            self.pl.logger.log_video(
                phase,
                f"real_2dflow_{name}",
                vid_t,
                self.pl.global_step,
                batch_idx,
                caption=batch["text"][bid],
            )

        # --- Load depth point cloud for 3D viz (on-demand) ---
        depth_pcl_3d = None
        depth_viz_cam = viz_T_cam  # fallback to auto-framed camera
        if has_cam_data:
            _anchor_abs = extra.get("_anchor_frame", [None])[bid]
            _K_all_orig = extra["_cam_K_all"][bid]
            _T_wc_all_raw = extra["_cam_T_wc_all"][bid]
            if isinstance(_K_all_orig, torch.Tensor):
                _K_all_orig = _K_all_orig.cpu().numpy()
            if isinstance(_T_wc_all_raw, torch.Tensor):
                _T_wc_all_raw = _T_wc_all_raw.cpu().numpy()
            if _anchor_abs is not None:
                _anchor_abs = int(_anchor_abs)
                _data_root = extra.get("_data_root", [None])[bid]
                _seq_id = extra.get("_seq_id", [None])[bid]
                if _data_root and _seq_id:
                    try:
                        depth_pcl_3d = load_depth_pcl(
                            data_root=_data_root,
                            seq_id=_seq_id,
                            frame_idx=_anchor_abs,
                            K_orig=_K_all_orig[_anchor_abs],
                            T_wc=_T_wc_all_raw[_anchor_abs],
                            T_align=T_align_np,
                            max_points=20000,
                            stride=2,
                            scene_scale=scene_scale_val,
                        )
                    except Exception as e:
                        logging.debug(f"Failed to load depth pcl: {e}")

            # Compute 3D viz camera: side perspective looking at depth centroid.
            # Start from anchor camera in aligned frame, offset for better 3D view.
            if _anchor_abs is not None:
                T_wc_orig = _T_wc_all_raw[_anchor_abs].copy()  # (4, 4)
                # Unified: anchor camera in aligned frame = T_align @ T_wc_orig
                anchor_cam = (T_align_np @ T_wc_orig).astype(np.float32)

                # Offset camera to a side perspective:
                # Move to side (+X in camera frame) and slightly up (+Y lift),
                # then look at the centroid of the depth cloud or trace center.
                cam_pos = anchor_cam[:3, 3].copy()
                cam_right = anchor_cam[:3, 0]  # X axis of camera
                cam_up = -anchor_cam[:3, 1]    # -Y is up in OpenCV convention

                # Compute look-at target: centroid of depth cloud or trace data
                if depth_pcl_3d is not None and len(depth_pcl_3d) > 0:
                    look_center = depth_pcl_3d[:, :3].mean(axis=0)
                else:
                    # Use trace data centroid as fallback
                    n_s = int(batch["trace_mask"][bid].any(dim=1).sum())
                    _tr = batch["trace"][bid, :n_s].cpu().numpy()
                    _tr_pos = _tr[:, :, :3, 3].reshape(-1, 3)
                    _valid = ~np.isnan(_tr_pos).any(axis=1)
                    if _valid.any():
                        look_center = _tr_pos[_valid].mean(axis=0)
                    else:
                        look_center = np.zeros(3)

                # Distance from camera to centroid
                dist = float(np.linalg.norm(cam_pos - look_center))
                side_offset = dist * 0.4   # 40% of distance to the side
                up_offset = dist * 0.15    # 15% upward

                eye = cam_pos + cam_right * side_offset + cam_up * up_offset
                depth_viz_cam = _look_at_opencv(eye, look_center)

        # GT 3D viz determines camera for all generated outputs
        _gt_3d_yfov = [None]  # mutable container for closure

        def _log_3d_depth(trace_tensor, name, known_n_val, known_start_val=0):
            """Render and log 3D depth + trace line video."""
            tr = trace_tensor.cpu().numpy() if torch.is_tensor(trace_tensor) else trace_tensor
            n_fr = tr.shape[0]
            positions = tr[:n_fr, :, :3, 3]  # (n_fr, P, 3)
            mask_np = batch["trace_mask"][bid, :n_fr].cpu().numpy().astype(bool)

            d3_frames, computed_yfov = render_3d_depth_traces(
                positions=positions,
                trace_mask=mask_np,
                depth_pcl=depth_pcl_3d,
                viz_T_cam=depth_viz_cam,
                known_n=known_n_val,
                known_start=known_start_val,
                max_tracks=self.max_tracks,
                H=720,
                W=1280,
                fixed_yfov=_gt_3d_yfov[0],
            )
            # First call (GT) sets the yfov for all subsequent calls
            if _gt_3d_yfov[0] is None:
                _gt_3d_yfov[0] = computed_yfov
            d3_frames = put_caption_at_bottom(d3_frames, batch["text"][bid])
            vid_t = (
                torch.from_numpy(np.stack(d3_frames, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0
            )
            self.pl.logger.log_video(
                phase,
                f"real_3d_{name}",
                vid_t,
                self.pl.global_step,
                batch_idx,
                caption=batch["text"][bid],
            )

        # --- GT trace ---
        _log_2d_flow(batch["trace"][bid, :n_steps], "GT", n_steps)
        _log_3d_depth(batch["trace"][bid, :n_steps], "GT", n_steps)

        # --- Context (goal) 3D visualization ---
        # Large bright markers on depth background so context positions are unmistakable
        # Placed after GT trace so _gt_3d_yfov[0] is set for consistent camera
        ctx_np = batch["context"][bid].cpu().numpy()  # (P, C, 4, 4)
        ctx_mask_np = batch["context_mask"][bid].cpu().numpy()  # (P, C)
        ctx_pos = ctx_np[:, 0, :3, 3]  # (P, 3) in transformed frame
        ctx_valid = ctx_mask_np[:, 0] > 0.5  # (P,)

        if ctx_valid.any() and depth_viz_cam is not None:
            cH, cW = 720, 1280
            yfov = _gt_3d_yfov[0] if _gt_3d_yfov[0] is not None else _DEFAULT_YFOV
            T_cw = np.linalg.inv(depth_viz_cam)
            fy = (cH / 2.0) / np.tan(yfov / 2.0)
            fx = fy

            # Start with depth background (same as 3D trace viz)
            canvas = np.full((cH, cW, 3), 242, dtype=np.uint8)
            if depth_pcl_3d is not None and len(depth_pcl_3d) > 0:
                d_cam = (T_cw[:3, :3] @ depth_pcl_3d[:, :3].T + T_cw[:3, 3:4]).T
                d_front = d_cam[:, 2] > 0.01
                if d_front.any():
                    d_f = d_cam[d_front]
                    d_cols = depth_pcl_3d[d_front, 3:6].astype(np.uint8)
                    d_u = (fx * d_f[:, 0] / d_f[:, 2] + cW / 2.0).astype(np.int32)
                    d_v = (fy * d_f[:, 1] / d_f[:, 2] + cH / 2.0).astype(np.int32)
                    d_rgb_list = [tuple(int(x) for x in d_cols[i].tolist()) for i in range(len(d_cols))]
                    d_layer = np.zeros((cH, cW, 3), dtype=np.uint8)
                    d_mask = np.zeros((cH, cW), dtype=np.float32)
                    d_r = max(3, int(min(cH, cW) * 0.005))
                    for i in np.argsort(-d_f[:, 2]):
                        cx, cy = int(d_u[i]), int(d_v[i])
                        if 0 <= cx < cW and 0 <= cy < cH:
                            cv2.circle(d_layer, (cx, cy), d_r, d_rgb_list[i], -1, cv2.LINE_AA)
                            cv2.circle(d_mask, (cx, cy), d_r, 1, -1, cv2.LINE_AA)
                    a3 = (d_mask * 0.4)[:, :, None]
                    canvas = (d_layer.astype(np.float32) * a3
                              + canvas.astype(np.float32) * (1.0 - a3)).clip(0, 255).astype(np.uint8)

            # Project context points and draw as large bright markers
            ctx_valid_pos = ctx_pos[ctx_valid]  # (N_valid, 3)
            c_cam = (T_cw[:3, :3] @ ctx_valid_pos.T + T_cw[:3, 3:4]).T
            c_front = c_cam[:, 2] > 0.01
            if c_front.any():
                c_f = c_cam[c_front]
                c_u = (fx * c_f[:, 0] / c_f[:, 2] + cW / 2.0).astype(np.int32)
                c_v = (fy * c_f[:, 1] / c_f[:, 2] + cH / 2.0).astype(np.int32)
                ctx_colors = _generate_track_colors(int(c_front.sum()))
                ctx_py_colors = [tuple(int(x) for x in ctx_colors[i].tolist())
                                 for i in range(len(ctx_colors))]
                marker_r = max(5, int(min(cH, cW) * 0.008))
                # Draw far-to-near so closer ones are on top
                for i in np.argsort(-c_f[:, 2]):
                    cx, cy = int(c_u[i]), int(c_v[i])
                    if 0 <= cx < cW and 0 <= cy < cH:
                        # White outline then colored fill — stands out on any background
                        cv2.circle(canvas, (cx, cy), marker_r + 2, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (cx, cy), marker_r, ctx_py_colors[i], -1, cv2.LINE_AA)

            ctx_t = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
            self.pl.logger.log_image(
                phase,
                "real_ctx_goal",
                ctx_t,
                self.pl.global_step,
                batch_idx,
            )

        # --- Inference outputs ---
        if not hasattr(eval_model, "inference"):
            logging.warning("eval_model has no 'inference' method -- skipping AR viz")
            return

        # Extract V2 image data from batch["extra"] for inference
        device = batch["trace"].device
        P = batch["trace"].shape[2]
        inf_images, inf_pixel_uv, inf_pixel_view, inf_pixel_qmask = (
            None,
            None,
            None,
            None,
        )
        extra = batch.get("extra", {})
        if "images" in extra and extra["images"][bid] is not None:
            inf_images = (
                torch.as_tensor(extra["images"][bid]).float().unsqueeze(0).to(device)
            )
        if (
            "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
        ):
            raw_uv = torch.as_tensor(extra["pixel_uv_normalized"][bid]).float()
            raw_view = torch.as_tensor(extra["pixel_view_idx"][bid]).long()
            raw_qmask = torch.as_tensor(extra["pixel_query_mask"][bid]).bool()
            P_raw = raw_uv.shape[0]
            if P_raw < P:
                raw_uv = torch.cat([raw_uv, torch.zeros(P - P_raw, 2)], 0)
                raw_view = torch.cat(
                    [raw_view, torch.zeros(P - P_raw, dtype=torch.long)], 0
                )
                raw_qmask = torch.cat(
                    [raw_qmask, torch.zeros(P - P_raw, dtype=torch.bool)], 0
                )
            inf_pixel_uv = raw_uv[:P].unsqueeze(0).to(device)
            inf_pixel_view = raw_view[:P].unsqueeze(0).to(device)
            inf_pixel_qmask = raw_qmask[:P].unsqueeze(0).to(device)

        ar_iter1 = [False]
        ar_iter2 = [""]
        seeds = [42]
        for ar_flag, prefix in zip(ar_iter1, ar_iter2):
            for seed in seeds:
                for cfg in self.add_cfg:
                    dataset_trace_mask = batch["trace_mask"][bid]
                    dataset_ctx_mask = batch["context_mask"][bid]
                    pad_mask = batch["trace_mask"][bid].any(0) | batch["context_mask"][
                        bid
                    ].any(0)
                    pad_mask_trace = (
                        pad_mask[None]
                        .bool()
                        .clone()
                        .repeat(batch["trace_mask"].shape[1], 1)
                    )[None]
                    pad_mask_ctx = pad_mask[None, None].bool().clone()

                    trace_known_mask = dataset_trace_mask[None].bool().clone()
                    context_known_mask = dataset_ctx_mask[None].bool().clone()
                    n_steps_int = int((batch["time"][bid] <= 0).sum())
                    trace_known_mask[0, n_steps_int:] = False

                    _, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=context_known_mask,
                        trace_mask=pad_mask_trace,
                        context_mask=pad_mask_ctx,
                        real_timestep=batch["time"][bid : bid + 1],
                        group_id=batch["trace_group_id"][bid : bid + 1],
                        sem_type=batch["trace_sem_type"][bid : bid + 1],
                        rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                        local_pcl=batch["local_pcl"][bid : bid + 1],
                        local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                        text=[batch["text"][bid]],
                        seed=seed,
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
                    )
                    _log_2d_flow(gen_trace[0], f"{prefix}Future_wgoal_{cfg}", n_steps_int)
                    _log_3d_depth(gen_trace[0], f"{prefix}Future_wgoal_{cfg}", n_steps_int)

                    _, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=torch.zeros_like(context_known_mask).bool(),
                        trace_mask=pad_mask_trace,
                        context_mask=pad_mask_ctx,
                        real_timestep=batch["time"][bid : bid + 1],
                        group_id=batch["trace_group_id"][bid : bid + 1],
                        sem_type=batch["trace_sem_type"][bid : bid + 1],
                        rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                        local_pcl=batch["local_pcl"][bid : bid + 1],
                        local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                        text=[batch["text"][bid]],
                        seed=seed,
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
                    )
                    _log_2d_flow(gen_trace[0], f"{prefix}Future_nogoal_{cfg}", n_steps_int)

                    cur_i = n_steps_int - 1
                    trace_known_mask_cur = torch.zeros_like(trace_known_mask)
                    trace_known_mask_cur[0, cur_i] = True
                    _, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask_cur,
                        context_known_mask=torch.zeros_like(context_known_mask).bool(),
                        trace_mask=pad_mask_trace,
                        context_mask=pad_mask_ctx,
                        real_timestep=batch["time"][bid : bid + 1],
                        group_id=batch["trace_group_id"][bid : bid + 1],
                        sem_type=batch["trace_sem_type"][bid : bid + 1],
                        rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                        local_pcl=batch["local_pcl"][bid : bid + 1],
                        local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                        text=[batch["text"][bid]],
                        seed=seed,
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
                    )
                    _log_2d_flow(gen_trace[0], f"{prefix}OnlyCur_nogoal_{cfg}", 1, known_start_val=cur_i)
                    _log_3d_depth(gen_trace[0], f"{prefix}OnlyCur_nogoal_{cfg}", 1, known_start_val=cur_i)
