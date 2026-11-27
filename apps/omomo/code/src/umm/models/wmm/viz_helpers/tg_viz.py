"""TraceGen (TG) viz: depth-faded scaffold + GT/pred trajectory overlay.

Single-panel viz for a TG sample. Layers (bottom->top):
  1. RGB base (384x384).
  2. Depth-faded scene scaffold: backproject depth pixels via pinhole approx,
     downsampled, alpha-faded by z (closer = more opaque).
  3. GT trajectory polylines (cool: viridis), per-keypoint hue.
  4. Predicted trajectory polylines (warm: plasma), per-keypoint hue.
  5. Tiny anchor dots at valid first keypoints.
  6. Title + legend.

Dark figure background. Square layout. JPEG output. No try/except -- caller
handles failures via the existing viz_try shield. Python 3.10.
"""
import logging
import os
from typing import Optional, Union

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Cap on number of moving keypoint trajectories drawn per viz panel.
# 12 well-chosen tracks beats 100+ overlapping spaghetti for legibility.
MAX_DISPLAY_KP = 12


def render_dino_query_overlay(
    image: Union[torch.Tensor, np.ndarray],
    pixel_uv_normalized: Union[torch.Tensor, np.ndarray],
    pixel_query_mask: Union[torch.Tensor, np.ndarray],
    radius: int = 3,
) -> np.ndarray:
    """Render DINO query points on the first-frame image.

    Convention (audited 2026-04-25, see audit-dino-query.md):
        ``pixel_uv_normalized[k] = (u, v) = (x_pixel / W, y_pixel / H)``
        in [0, 1]. Same convention as ``tg_viz`` 2D-track viz (line 161,
        367-368) -- markers placed at ``(u * W, v * H)``.

    Args:
        image: ``(3, H, W)`` torch tensor or ``(H, W, 3)`` ndarray. Float in
            [0, 1] OR uint8 in [0, 255]. Auto-detected and denormalized.
        pixel_uv_normalized: ``(P, 2)`` tensor/ndarray in [0, 1].
            Column 0 = u (x/W); column 1 = v (y/H).
        pixel_query_mask: ``(P,)`` bool tensor/ndarray. True -> green dot
            (active DINO query); False -> faint gray dot (masked).
        radius: marker radius in pixels. Default 3.

    Returns:
        ``(H, W, 3)`` uint8 RGB ndarray with markers drawn.

    Notes:
        Tensors are detached and moved to CPU exactly once at entry per the
        PMEM lesson. Active markers use solid green ``(0, 200, 0)``,
        masked markers use a halved-brightness gray ``(64, 64, 64)`` (a
        cheap stand-in for alpha=0.5 over the image background).
    """
    # --- 1. Image: handle (V,3,H,W) / (3,H,W) torch / (H,W,3) np, float / uint8 ---
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)
    # Strip leading view dim when (V, 3, H, W) — the training-loop callsite
    # passes the raw single-view DINO image (V=1). Use view 0.
    if img_np.ndim == 4 and img_np.shape[1] == 3:
        img_np = img_np[0]  # (3, H, W)
    if img_np.ndim == 3 and img_np.shape[0] == 3 and img_np.shape[-1] != 3:
        # (3, H, W) -> (H, W, 3)
        img_np = img_np.transpose(1, 2, 0)
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError(
            f"render_dino_query_overlay: expected (V,3,H,W) / (3,H,W) / "
            f"(H,W,3) image, got shape {img_np.shape}"
        )
    if img_np.dtype != np.uint8:
        # float [0,1] -> uint8 [0,255]
        img_u8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = img_np
    H, W = img_u8.shape[:2]

    # --- 2. UV + mask to numpy (detach/cpu once) -----------------------------
    if isinstance(pixel_uv_normalized, torch.Tensor):
        uv_np = pixel_uv_normalized.detach().cpu().numpy()
    else:
        uv_np = np.asarray(pixel_uv_normalized)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        raise ValueError(
            f"render_dino_query_overlay: pixel_uv_normalized must be (P,2), "
            f"got {uv_np.shape}"
        )

    if isinstance(pixel_query_mask, torch.Tensor):
        mask_np = pixel_query_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(pixel_query_mask)
    mask_np = mask_np.astype(bool).reshape(-1)
    if mask_np.shape[0] != uv_np.shape[0]:
        raise ValueError(
            f"render_dino_query_overlay: mask shape {mask_np.shape} != uv "
            f"shape {uv_np.shape[:1]}"
        )

    # --- 3. Draw with PIL ----------------------------------------------------
    pil_img = Image.fromarray(img_u8.copy(), mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    color_active = (0, 200, 0)        # green
    color_masked = (64, 64, 64)       # halved gray -> faint over typical bg

    # Pixel coordinates: u -> x_pixel, v -> y_pixel.
    xy_px = uv_np.astype(np.float32) * np.array([W, H], dtype=np.float32)

    for k in range(xy_px.shape[0]):
        x_px = float(xy_px[k, 0])
        y_px = float(xy_px[k, 1])
        # Skip OOB markers (they would wrap or land on image border anyway).
        if not (np.isfinite(x_px) and np.isfinite(y_px)):
            continue
        if mask_np[k]:
            color = color_active
            r = radius
        else:
            color = color_masked
            r = max(radius - 1, 1)  # masked markers slightly smaller
        # Filled circle: ImageDraw.ellipse takes a bounding box.
        bbox = (x_px - r, y_px - r, x_px + r, y_px + r)
        draw.ellipse(bbox, fill=color, outline=None)

    return np.asarray(pil_img, dtype=np.uint8)


def _to_numpy(x):
    """Tensor -> ndarray; pass-through ndarray; None -> None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_numpy_req(x) -> np.ndarray:
    """Required tensor -> ndarray. Raises ValueError if None."""
    if x is None:
        raise ValueError("_to_numpy_req: got None")
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _backproject_depth(
    depth: np.ndarray,
    image_size: int = 384,
    max_points: int = 2500,
    rng_seed: int = 0,
):
    """Project depth-map pixels to (u_img, v_img, z) in image pixel space.

    TG ships no intrinsics; uv is rescaled directly from depth-native pixel
    space to image-pixel space -- a true pinhole projection is unnecessary
    because we render in 2D and only need depth to drive scatter alpha.

    Args:
        depth: (H, W) float32 meters; H~294, W~518 for TG.
        image_size: target square image size in pixels (rgb is image_size^2).
        max_points: subsample cap so the scatter does not dominate.
        rng_seed: deterministic subsampling.

    Returns:
        u_px, v_px, z   each (N,) float32 with N <= max_points.
        u_px and v_px are in [0, image_size] pixel coords (rescaled to RGB).
    """
    assert depth.ndim == 2, f"depth must be (H,W), got {depth.shape}"
    H, W = depth.shape
    valid = (depth > 0) & np.isfinite(depth)
    if not valid.any():
        return (np.zeros(0, np.float32),) * 3
    vs, us = np.where(valid)
    z = depth[vs, us].astype(np.float32)

    u_img = (us.astype(np.float32) / float(W)) * float(image_size)
    v_img = (vs.astype(np.float32) / float(H)) * float(image_size)

    n = u_img.shape[0]
    if n > max_points:
        rng = np.random.default_rng(rng_seed)
        sel = rng.choice(n, size=max_points, replace=False)
        u_img, v_img, z = u_img[sel], v_img[sel], z[sel]
    return u_img, v_img, z


def tg_viz(
    sample: dict,
    pred_trace_xyz=None,
    save_path: Optional[str] = None,
    env_name: Optional[str] = None,
    episode_id: Optional[str] = None,
    frame_id: Optional[str] = None,
    global_step: Optional[int] = None,
    title_extra: str = "",
    image_size: int = 384,
    max_scaffold_pts: int = 2500,
    mode: str = "both",
    draw_all: bool = False,
) -> str:
    """Render a single TG sample with optional prediction overlay.

    Args:
        sample: per-sample dict from ``TracegenDataset.__getitem__``.
        pred_trace_xyz: optional (P, T, 3) tensor/ndarray of predicted positions
            in the same convention as ``sample["trace"][:, :, :3, 3]`` (xy
            normalized [0,1], z raw meters). Same (P, T, 3) shape as GT --
            pred values at GT-invalid timesteps are SILENTLY DROPPED so the
            predicted polyline spans the same observed timesteps as GT
            (visual comparability). If you need to plot pred at all
            timesteps regardless of GT validity, post-fill pred at invalid
            timesteps to a sentinel and override the validity mask upstream.
            None => GT-only viz.
        save_path: output JPEG path. If None, raises ValueError. Caller must
            ensure the parent directory exists; tg_viz does not mkdir.
        env_name, episode_id, frame_id, global_step: shown in title; if any
            are None, fall back to fields from ``sample``.
        title_extra: optional appended title text.
        image_size: RGB canvas size (TG dataset uses 384).
        max_scaffold_pts: cap on depth scaffold points (<= this many drawn).
        mode: ``"gt"`` (only GT polylines), ``"pred"`` (only pred polylines —
            requires ``pred_trace_xyz``), ``"both"`` (legacy GT+pred overlay).
        draw_all: if True, draw every valid keypoint (no movement filter, no
            top-N cap). Line width and alpha are tightened to keep the
            high-density viz legible.

    Returns:
        save_path (the JPEG written).

    Backend:
        Sets ``matplotlib.use("Agg", force=True)`` at runtime before the first
        pyplot import if the current backend is not Agg or an inline (notebook)
        backend. This is safe inside training-callback render loops (called at
        most once per viz batch) and idempotent if the backend is already Agg.
    """
    if save_path is None:
        raise ValueError("tg_viz: save_path is required")
    if mode not in ("gt", "pred", "both"):
        raise ValueError(f"tg_viz: mode must be gt/pred/both, got {mode!r}")
    if mode == "pred" and pred_trace_xyz is None:
        raise ValueError("tg_viz: mode='pred' requires pred_trace_xyz to be provided")

    # Headless rendering. Set backend BEFORE the first pyplot import so it
    # actually takes effect (pyplot import locks the backend). On headless
    # servers Agg is already the default; on display machines or notebooks
    # we explicitly opt in.
    _backend_lower = matplotlib.get_backend().lower()
    if _backend_lower not in ("agg",) and "inline" not in _backend_lower:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # -- Unpack -----------------------------------------------------------
    img = _to_numpy_req(sample["image"])  # (1,3,H,W) float [0,1] OR (3,H,W)
    if img.ndim == 4:
        img = img[0]
    img = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)  # (H,W,3)

    fkp = _to_numpy_req(sample["first_keypoint"]) * float(image_size)  # (P,2) px
    gt_mask = _to_numpy_req(sample["gt_mask"]).astype(bool)             # (P,)
    trace_xyz = _to_numpy_req(sample["trace"])[:, :, :3, 3]              # (P,T,3)
    trace_xy_px = trace_xyz[:, :, :2] * float(image_size)                # (P,T,2)
    trace_mask = _to_numpy_req(sample["trace_mask"]).astype(bool)        # (P,T,3) per-coord
    depth = _to_numpy(sample.get("depth_raw"))                           # (H,W) or None

    # Validate pred shape against GT shape -- silent-wrong-data guard.
    P_gt = trace_xyz.shape[0]
    T_gt = trace_xyz.shape[1]
    if pred_trace_xyz is not None:
        pred_arr = _to_numpy_req(pred_trace_xyz)
        if pred_arr.shape != (P_gt, T_gt, 3):
            raise ValueError(
                f"tg_viz: pred_trace_xyz shape {pred_arr.shape} != GT shape "
                f"{(P_gt, T_gt, 3)}; caller must pass full (P,T,3) absolute "
                f"positions in normalized image coords (xy in [0,1], z meters)."
            )
        pred_xy_px = pred_arr[:, :, :2] * float(image_size)              # (P,T,2)
    else:
        pred_xy_px = None

    # movement_bool is a required key in the TG sample schema. Refuse to
    # silently fall back -- a missing mask would render all 400 valid
    # keypoints as moving and produce visual spaghetti that looks "fine".
    if "movement_bool" not in sample:
        raise KeyError(
            "tg_viz: sample missing required key 'movement_bool'. "
            "All current TG samples ship this; if you are calling from a new "
            "data pipeline, add movement_bool[(P,) bool] to the sample dict."
        )
    mvmt = _to_numpy_req(sample["movement_bool"])

    env_name = env_name or sample.get("env_name", "?")
    episode_id = episode_id or sample.get("episode_id", "?")
    frame_id = frame_id or sample.get("frame_id", "?")

    # -- Figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.imshow(img, extent=(0, image_size, image_size, 0), interpolation="bilinear")
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- (b) Depth-faded scaffold -----------------------------------------
    scaffold_drawn = False
    if depth is not None and depth.size > 0:
        u_px, v_px, z = _backproject_depth(
            depth, image_size=image_size, max_points=max_scaffold_pts
        )
        if u_px.size > 0:
            z_min = float(np.percentile(z, 5))
            z_max = float(np.percentile(z, 95))
            denom = max(z_max - z_min, 1e-6)
            alpha = np.clip(1.0 - (z - z_min) / denom, 0.12, 0.55)
            # Brighter sky-blue so the scaffold reads as scene context, not noise.
            colors = np.zeros((z.shape[0], 4), dtype=np.float32)
            colors[:, 0] = 0.45  # R
            colors[:, 1] = 0.75  # G
            colors[:, 2] = 0.95  # B
            colors[:, 3] = alpha
            ax.scatter(u_px, v_px, s=2.5, c=colors, marker="o",
                       linewidths=0, zorder=1.5)
            scaffold_drawn = True

    # -- Trajectory drawing helpers ---------------------------------------
    mvmt_bool = mvmt.astype(bool) & gt_mask
    moving_idx = np.where(mvmt_bool)[0]
    static_idx = np.where(gt_mask & ~mvmt_bool)[0]
    n_moving_total = int(moving_idx.size)
    n_static = int(static_idx.size)
    n_valid_total = int(gt_mask.sum())

    if draw_all:
        # Honest mode: every valid keypoint is drawn (no movement filter,
        # no top-N cap). Static kps are absorbed into drawn_idx so the
        # static-anchor-dot scatter is suppressed below.
        drawn_idx = np.where(gt_mask)[0]
        static_idx = np.zeros(0, dtype=np.int64)
        n_static = 0
    elif n_moving_total > MAX_DISPLAY_KP:
        # Cap the number of drawn moving tracks. Rank by TOTAL MOVEMENT MAGNITUDE
        # (sum of per-step L2 xy deltas on valid frames).
        dxy = np.diff(trace_xy_px[:, :, :2], axis=1)
        dxy = np.nan_to_num(dxy, nan=0.0, posinf=0.0, neginf=0.0)
        step_valid = (trace_mask[:, :-1, 0] & trace_mask[:, 1:, 0]).astype(np.float32)
        step_dist = np.sqrt((dxy * dxy).sum(axis=-1)) * step_valid
        total_move = step_dist.sum(axis=1)                         # (P,)
        move_scores = total_move[moving_idx]
        valid_counts = trace_mask[moving_idx, :, 0].sum(axis=1)
        order = np.lexsort((moving_idx, -valid_counts, -move_scores))
        keep = order[:MAX_DISPLAY_KP]
        drawn_idx = moving_idx[keep]
    else:
        drawn_idx = moving_idx
    n_drawn = int(drawn_idx.size)

    # Color palette: tab10 cycles for ≤10 tracks, HSV around the wheel for
    # the all-400 mode so adjacent indices are still distinguishable.
    base_cmap = plt.get_cmap("tab10")
    if n_drawn <= 10:
        track_colors = base_cmap(np.arange(n_drawn) % 10)
    else:
        track_colors = plt.get_cmap("hsv")(
            np.linspace(0.0, 1.0, n_drawn, endpoint=False)
        )
    gt_colors = track_colors.copy()
    pred_colors = track_colors.copy()
    pred_colors[:, :3] *= 0.65  # darken pred RGB; keep alpha

    # Drawing gates per mode. In single-mode renders we tighten line width
    # and alpha so 400 polylines remain legible.
    draw_gt = mode in ("gt", "both") and n_drawn > 0
    draw_pred = mode in ("pred", "both") and (pred_xy_px is not None) and n_drawn > 0
    pred_drawn = draw_pred  # legend gate
    if n_drawn > 50:
        line_lw = 0.6
        line_alpha = 0.65
        end_marker_s = 4
    else:
        line_lw = 1.4
        line_alpha = 0.90
        end_marker_s = 12

    def _polyline(k: int, base_xy_px: np.ndarray,
                  color, lw: float, alpha: float, zorder: float,
                  linestyle: str = "-",
                  use_all_t: bool = False):
        """Draw a polyline through valid timesteps.

        ``use_all_t=False`` (default) gates by GT ``trace_mask`` (per-coord t
        validity) — the comparable-timesteps regime used by GT polylines.

        ``use_all_t=True`` draws all 33 timesteps regardless of GT validity.
        Used for pred-only mode so the model output is shown honestly across
        the full predicted horizon, including timesteps where GT is missing.
        """
        if use_all_t:
            T_full = base_xy_px.shape[1]
            valid_t = np.ones(T_full, dtype=bool)
        else:
            valid_t = trace_mask[k, :, 0]
        if int(valid_t.sum()) < 2:
            return
        future_xy = base_xy_px[k, valid_t, :]
        anchor = fkp[k:k + 1, :]
        first_valid_t = int(np.argmax(valid_t))
        if first_valid_t == 0:
            uv = future_xy
        else:
            uv = np.concatenate([anchor, future_xy], axis=0)
        ax.plot(uv[:, 0], uv[:, 1], color=color, lw=lw, alpha=alpha,
                linestyle=linestyle, solid_capstyle="round",
                antialiased=True, zorder=zorder)

    # -- (c) GT polylines -------------------------------------------------
    if draw_gt:
        for pos, k in enumerate(drawn_idx):
            _polyline(k, trace_xy_px, gt_colors[pos], lw=line_lw,
                      alpha=line_alpha, zorder=2.5, linestyle="-")
            valid_t_k = trace_mask[k, :, 0]
            if int(valid_t_k.sum()) >= 1:
                last_t = int(np.where(valid_t_k)[0][-1])
                end_xy = trace_xy_px[k, last_t, :]
                ax.scatter(end_xy[0], end_xy[1],
                           s=end_marker_s, marker="o", color=gt_colors[pos],
                           edgecolors="white", linewidths=0.3,
                           zorder=2.7)

    # -- (d) Predicted polylines -----------------------------------------
    if draw_pred:
        assert pred_xy_px is not None  # narrowed by draw_pred guard
        # Pred-only renders (mode="pred") show the model's full 33-step
        # output regardless of GT validity. In "both" mode we keep the
        # legacy GT-validity gating for visual comparability.
        pred_use_all_t = (mode == "pred")
        # Pred-only gets a solid line; in "both" mode it stays dashed to
        # distinguish from GT.
        pred_linestyle = "-" if mode == "pred" else "--"
        T_full = pred_xy_px.shape[1]
        for pos, k in enumerate(drawn_idx):
            _polyline(k, pred_xy_px, pred_colors[pos], lw=line_lw,
                      alpha=max(line_alpha - 0.05, 0.5), zorder=3.0,
                      linestyle=pred_linestyle, use_all_t=pred_use_all_t)
            if pred_use_all_t:
                last_t = T_full - 1
            else:
                valid_t_k = trace_mask[k, :, 0]
                if int(valid_t_k.sum()) < 1:
                    continue
                last_t = int(np.where(valid_t_k)[0][-1])
            end_xy = pred_xy_px[k, last_t, :]
            ax.scatter(end_xy[0], end_xy[1],
                       s=max(end_marker_s + 2, 6), marker="^",
                       color=pred_colors[pos],
                       edgecolors="white", linewidths=0.3,
                       zorder=3.5)

    # -- (e) Anchor dots: muted static under, colored drawn-moving over --
    if n_static > 0:
        static_color = np.tile(
            np.array([[0.55, 0.55, 0.55, 0.45]], dtype=np.float32),
            (n_static, 1),
        )
        ax.scatter(
            fkp[static_idx, 0], fkp[static_idx, 1],
            s=4, c=static_color,
            linewidths=0,
            zorder=2.4,
        )
    if n_drawn > 0:
        ax.scatter(
            fkp[drawn_idx, 0], fkp[drawn_idx, 1],
            s=12, c=gt_colors,
            edgecolors="white", linewidths=0.3,
            zorder=4.0,
        )

    # -- (f) Title + legend -----------------------------------------------
    step_str = f"step={global_step}" if global_step is not None else ""
    if draw_all:
        track_str = f"drawn={n_drawn}/{n_valid_total} (all valid)"
    else:
        moving_str = f"moving={n_moving_total}/{n_valid_total}"
        if n_moving_total > MAX_DISPLAY_KP:
            moving_str += f" (showing top {MAX_DISPLAY_KP})"
        track_str = f"valid={n_valid_total}  {moving_str}"
    mode_label = {"gt": "GT only", "pred": "PRED only", "both": "GT+PRED"}[mode]
    title_bits = [
        f"{env_name} [{mode_label}]",
        f"ep={episode_id}", f"frame={frame_id}",
        track_str,
    ]
    if step_str:
        title_bits.append(step_str)
    if title_extra:
        title_bits.append(title_extra)
    title = " | ".join(title_bits)
    ax.set_title(title, color="white", fontsize=11,
                 fontfamily="monospace", pad=10)

    # Legend uses tab10(0) (blue) as the swatch reference; pred uses
    # the same hue darkened by 0.65 in "both" mode.
    legend_blue = base_cmap(0)
    legend_pred_blue = (
        legend_blue[0] * 0.65, legend_blue[1] * 0.65, legend_blue[2] * 0.65,
        legend_blue[3],
    )
    handles = []
    if draw_gt:
        handles.append(Line2D([], [], color=legend_blue, lw=2.0, linestyle="-",
                              label="GT"))
    if draw_pred:
        pred_ls = "-" if mode == "pred" else "--"
        pred_swatch = legend_blue if mode == "pred" else legend_pred_blue
        handles.append(Line2D([], [], color=pred_swatch, lw=2.0,
                              linestyle=pred_ls, label="Pred"))
    if draw_gt:
        handles.append(Line2D([], [], color=legend_blue, marker="o",
                              linestyle="none", markersize=6,
                              markeredgecolor="white", markeredgewidth=0.4,
                              label="GT endpoint"))
    if draw_pred:
        pred_swatch = legend_blue if mode == "pred" else legend_pred_blue
        handles.append(Line2D([], [], color=pred_swatch, marker="^",
                              linestyle="none", markersize=7,
                              markeredgecolor="white", markeredgewidth=0.4,
                              label="Pred endpoint"))
    if scaffold_drawn:
        handles.append(Line2D([], [], color=(0.45, 0.75, 0.95, 0.7),
                              marker="o", linestyle="none",
                              markersize=5, label="Depth scaffold"))
    leg = ax.legend(handles=handles, loc="upper right",
                    facecolor="#2a2a2a", edgecolor="#444444",
                    fontsize=8, framealpha=0.92)
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout(pad=0.6)
    # Caller is responsible for ensuring the parent directory exists. We only
    # verify here; we do not create. This avoids a stat() syscall on every
    # render in training-callback hot loops.
    parent = os.path.dirname(os.path.abspath(save_path))
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"tg_viz: save_path parent does not exist: {parent}. "
            f"Caller must create the directory ahead of time."
        )
    fig.savefig(save_path, dpi=140, format="jpg",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(
        f"tg_viz: saved {save_path} (env={env_name} ep={episode_id} "
        f"valid_kp={n_valid_total} moving={n_moving_total} drawn={n_drawn} "
        f"static={n_static} scaffold={scaffold_drawn} pred={pred_drawn})"
    )
    return save_path


# ---------------------------------------------------------------------------
# Wiring for PLModel._init_visualizers.
# Minimal wrapper around tg_viz() so `visualize.tg:` YAML block is no
# longer inert. GT-only viz; inference/pred overlay is deferred to a
# follow-up SG — the contract here is to prove the dispatch path fires.
# ---------------------------------------------------------------------------


def _sample_from_tg_batch(batch, idx: int = 0):
    """Reconstruct a tg_viz-shaped per-sample dict from a collated TG batch.

    Returns None if the batch is not TG-shaped (e.g. mixed dataset without
    the TG extras). Safe to call before the callback knows which dataset
    the current batch came from.
    """
    extra = None
    if isinstance(batch, dict):
        extra = batch.get("extra")
    if not isinstance(extra, dict) or "trace_valid_mask_per_coord" not in extra:
        return None

    def _pick(container, key):
        if not isinstance(container, dict) or key not in container:
            return None
        val = container[key]
        if isinstance(val, (list, tuple)):
            return val[idx] if idx < len(val) else None
        if hasattr(val, "shape") and len(val.shape) > 0:
            return val[idx]
        return val

    trace = _pick(batch, "trace")                      # (B, T_full, P, 4, 4)
    per_coord = _pick(extra, "trace_valid_mask_per_coord")  # (T, P, 3) after idx
    images = _pick(extra, "images")                    # (1, 3, H, W)
    first_kp = _pick(extra, "first_keypoint")          # (P, 2)
    gt_mask = _pick(extra, "gt_mask")                  # (P,) bool
    movement_bool = _pick(extra, "movement_bool")      # (P,) bool
    depth_raw = _pick(extra, "depth_raw")              # (H, W) or None
    env_name = _pick(extra, "env_name")
    episode_id = _pick(extra, "episode_id")
    frame_id = _pick(extra, "frame_id")

    # Required fields; if any absent the batch is not TG-shaped.
    if trace is None or images is None or first_kp is None or gt_mask is None:
        return None
    if per_coord is None:
        return None

    # Per-sample `trace` from the batch is (T_full, P, 4, 4) after idx.
    # tg_viz expects (P, T, 4, 4). For current TG configs T_full == T_viz
    # (no context prepend in `trace`; context lives on a separate key) — the
    # trim below is a no-op. If a future pipeline change prepends context
    # onto `trace`, the trim keeps the last T_viz frames to align with
    # per_coord. A log-warn surfaces any unexpected mismatch so the silent
    # drop does not hide a real bug.
    if hasattr(trace, "permute"):
        trace_pt = trace.permute(1, 0, 2, 3).contiguous()   # (P, T_full, 4, 4)
    else:
        trace_pt = np.transpose(trace, (1, 0, 2, 3))
    if hasattr(per_coord, "permute"):
        per_coord_pt = per_coord.permute(1, 0, 2).contiguous()  # (P, T_viz, 3)
    else:
        per_coord_pt = np.transpose(per_coord, (1, 0, 2))
    T_viz = per_coord_pt.shape[1]
    T_full = trace_pt.shape[1]
    if T_full != T_viz:
        logger.warning(
            f"_sample_from_tg_batch: trace T_full={T_full} != per_coord "
            f"T_viz={T_viz}; trimming trace to last {T_viz} frames. "
            "Verify the collate schema has not changed."
        )
    trace_pt = trace_pt[:, -T_viz:, :, :]

    return {
        "image": images,
        "trace": trace_pt,
        "trace_mask": per_coord_pt,
        "first_keypoint": first_kp,
        "gt_mask": gt_mask,
        "movement_bool": movement_bool if movement_bool is not None else gt_mask,
        "depth_raw": depth_raw,
        "env_name": env_name,
        "episode_id": episode_id,
        "frame_id": frame_id,
    }


def _build_tg_inference_args(batch, bid: int, device):
    """Assemble kwargs for ``AgentV11.inference`` from a training TG batch.

    The training DataModule (wmm.datamodule._TGAdapterDataset + _tg_stack_collate)
    produces a batch with:
      - batched tensors on the top level (``trace``, ``trace_mask``, ...)
      - ``extra[*]`` as PER-SAMPLE LISTS for image + pixel metadata
        (``CanonicalDatasetOutput.collate`` does not stack them)
      - ``extra['trace_valid_mask_per_coord']`` is stacked by _tg_stack_collate
        but is not needed by inference (only by the eval-callback metric).

    This helper unpacks a single sample ``bid`` from the batch and re-packs it
    into the (B=1, ...) tensor format that ``AgentV11.inference`` expects.
    Mirrors the arg table in
    ``$/sg/01-tg-dataset/inference_signature_mapping.md`` §4 and the
    production callback in ``tg_eval_callback._forward_batch``.

    Args:
        batch: Training-time batch dict from TG DataLoader.
        bid: Index of the sample inside the batch to run inference on.
        device: Target device for all inference tensors.

    Returns:
        Dict of kwargs ready to splat into ``eval_model.inference(**kw)``.
    """
    extra = batch.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError(
            "tg_viz: batch['extra'] is missing or not a dict; "
            "TG viz requires the TG datamodule collate path."
        )

    def _pick_list(key):
        """Return the bid-th entry from a list-valued extra field (or None)."""
        val = extra.get(key)
        if isinstance(val, (list, tuple)):
            return val[bid] if bid < len(val) else None
        return None

    trace = batch["trace"][bid:bid + 1].to(device)              # (1, T, P, 4, 4)
    trace_mask_any = batch["trace_mask"][bid:bid + 1].to(device)  # (1, T, P) bool
    context = trace[:, :1].clone()                               # (1, 1, P, 4, 4)

    # Prefer the mask_scheduler-produced known masks when the batch carries
    # them (from _TGAdapterDataset); else fall back to "t=0 known" convention.
    T, P = trace.shape[1], trace.shape[2]
    if "known_mask" in batch:
        trace_known = batch["known_mask"][bid:bid + 1].to(device).bool()
    else:
        trace_known = torch.zeros(1, T, P, dtype=torch.bool, device=device)
        trace_known[:, 0] = True
    if "context_known_mask" in batch:
        ctx_known = batch["context_known_mask"][bid:bid + 1].to(device).bool()
    else:
        ctx_known = torch.ones(1, 1, P, dtype=torch.bool, device=device)

    if "context_mask" in batch:
        ctx_mask = batch["context_mask"][bid:bid + 1].to(device).bool()
    else:
        # context validity = per-kp gt-mask broadcast to time-dim 1.
        gt_m = _pick_list("gt_mask")
        if gt_m is None:
            raise ValueError(
                "tg_viz: neither batch['context_mask'] nor extra['gt_mask'] "
                "available — cannot build inference context_mask."
            )
        ctx_mask = torch.as_tensor(gt_m).to(device).bool().unsqueeze(0).unsqueeze(0)

    if "time" in batch:
        real_timestep = batch["time"][bid:bid + 1].to(device).float()
    else:
        # Fallback: only fires when the caller passed a raw batch without
        # the wmm "time" field. Mirrors TGAdapter (wmm/datamodule.py:155)
        # with the dataset class defaults; YAML-overridden values flow
        # through batch["time"], not through this branch.
        time_unit = 1.0 / 30.0
        real_timestep = (
            torch.arange(T, dtype=torch.float32, device=device) * time_unit
        ).unsqueeze(0)
    group_id = batch["trace_group_id"][bid:bid + 1].to(device).long()
    sem_type = batch["trace_sem_type"][bid:bid + 1].to(device).long()
    rot_mask = batch["trace_rot_mask"][bid:bid + 1].to(device).bool()

    text = [batch["text"][bid]]

    # extra list-of-tensors → (1, ...) tensors on device.
    images_t = _pick_list("images")
    if images_t is None:
        raise ValueError(
            "tg_viz: batch['extra']['images'] missing — TG inference needs a DINO image."
        )
    images = torch.as_tensor(images_t).float().to(device).unsqueeze(0)  # (1, 1, 3, H, W)

    pixel_uv_t = _pick_list("pixel_uv_normalized")
    if pixel_uv_t is None:
        pixel_uv_t = _pick_list("first_keypoint")
    if pixel_uv_t is None:
        raise ValueError(
            "tg_viz: batch['extra']['pixel_uv_normalized'] (or first_keypoint) missing."
        )
    # NB: explicit ``is None`` checks above; Python ``or`` on tensors raises
    # ``RuntimeError: Boolean value of Tensor with more than one value is
    # ambiguous`` because ``bool(tensor)`` is undefined for multi-element
    # tensors. Same pattern applies to ``pixel_qmask_t`` below.
    pixel_uv = torch.as_tensor(pixel_uv_t).float().to(device).unsqueeze(0)  # (1, P, 2)

    pixel_view_idx_t = _pick_list("pixel_view_idx")
    if pixel_view_idx_t is None:
        pixel_view_idx = torch.zeros(1, P, dtype=torch.long, device=device)
    else:
        pixel_view_idx = torch.as_tensor(pixel_view_idx_t).long().to(device).unsqueeze(0)

    pixel_qmask_t = _pick_list("pixel_query_mask")
    if pixel_qmask_t is None:
        pixel_qmask_t = _pick_list("gt_mask")
    if pixel_qmask_t is None:
        raise ValueError(
            "tg_viz: batch['extra']['pixel_query_mask'] (or gt_mask) missing."
        )
    pixel_query_mask = torch.as_tensor(pixel_qmask_t).bool().to(device).unsqueeze(0)

    view_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

    # w1.4.0: cur_frame_idx for bias_to_cur_frame LMM flag. TG anchors at
    # trace t=0; a per-(b, p) zeros tensor is correct here. The LMM ignores
    # this when bias_to_cur_frame=False.
    cur_frame_idx = torch.zeros(1, P, dtype=torch.long, device=device)

    return dict(
        trace=trace,
        context=context,
        trace_known_mask=trace_known,
        context_known_mask=ctx_known,
        trace_mask=trace_mask_any,
        context_mask=ctx_mask,
        real_timestep=real_timestep,
        group_id=group_id,
        sem_type=sem_type,
        rot_mask=rot_mask,
        text=text,
        local_pcl=None,
        local_pcl_mask=None,
        images=images,
        view_ids=view_ids,
        pixel_uv_normalized=pixel_uv,
        pixel_view_idx=pixel_view_idx,
        pixel_query_mask=pixel_query_mask,
        cur_frame_idx=cur_frame_idx,
    )


class TgVisualizer:
    """Minimal dispatch wrapper around ``tg_viz()``.

    Instantiated by ``PLModel._init_visualizers`` when ``cfg.visualize.tg``
    is a non-empty dict. Exposes a ``tg_viz_routine`` method matching the
    signature used by other domain visualizers so ``PLModel.viz()`` can
    call it uniformly. Renders both GT (solid) and predicted (dashed)
    trajectory polylines.
    """

    def __init__(self, pl_model, cfg):
        self.pl = pl_model
        self.cfg = dict(cfg) if cfg else {}
        self.env_name = self.cfg.get("env_name", "tg")
        self.image_size = int(self.cfg.get("image_size", 384))
        self.max_scaffold_pts = int(self.cfg.get("max_scaffold_pts", 2500))
        self.flag = bool(self.cfg)
        logger.info(
            f"TgVisualizer wired: env={self.env_name}, "
            f"image_size={self.image_size}, flag={self.flag}"
        )

    def tg_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Render GT + Pred viz for ``batch[0]`` to ``<log_dir>/tg_viz/...jpg``.

        Signature matches other visualizers in the WMM dispatch. ``step_output``
        is accepted for API parity and currently unused. ``eval_model`` is the
        uncompiled inference model selected by ``PLModel.viz`` (see
        ``src/umm/models/wmm/model.py`` ``viz()`` — it picks ``ema_model`` when
        ``use_ema`` is on, else ``model``). Calling inference through the
        compiled training graph would force a recompile on inference-shape
        tensors → OOM (same reason ``TgEvalCallback`` uses ``_select_eval_model``).
        """
        del step_output  # unused — reserved for future use
        sample = _sample_from_tg_batch(batch, idx=0)
        if sample is None:
            logger.info(
                "TgVisualizer.tg_viz_routine: skipped — batch is not TG-shaped"
            )
            return

        # --- DINO query overlay -----------
        # Render a per-sample overlay showing where each DINO pixel query
        # lands on the first-frame image. Convention matches tg_viz lines
        # 161 and 367-368: uv -> (u*W, v*H). Active queries (green) line up
        # with the tracked-keypoint anchor dots in tg_viz; masked queries
        # (faint gray) are zeroed inside dino_encoder. See
        extra = batch.get("extra", {}) if isinstance(batch, dict) else {}
        pixel_uv_list = extra.get("pixel_uv_normalized") if isinstance(extra, dict) else None
        pixel_qmask_list = extra.get("pixel_query_mask") if isinstance(extra, dict) else None
        if pixel_uv_list is None:
            # Fall back to first_keypoint when datamodule didn't promote it.
            pixel_uv_list = extra.get("first_keypoint") if isinstance(extra, dict) else None
        if pixel_qmask_list is None:
            pixel_qmask_list = extra.get("gt_mask") if isinstance(extra, dict) else None

        # extra fields are per-sample lists in the TG collate. Render sample 0
        # to match the rest of tg_viz_routine's single-sample contract.
        sample_idx_for_overlay = 0
        if (
            isinstance(pixel_uv_list, (list, tuple))
            and isinstance(pixel_qmask_list, (list, tuple))
            and len(pixel_uv_list) > sample_idx_for_overlay
            and len(pixel_qmask_list) > sample_idx_for_overlay
        ):
            uv_per_sample = pixel_uv_list[sample_idx_for_overlay]
            qmask_per_sample = pixel_qmask_list[sample_idx_for_overlay]
            overlay_np = render_dino_query_overlay(
                image=sample["image"],
                pixel_uv_normalized=uv_per_sample,
                pixel_query_mask=qmask_per_sample,
            )
            # log_image expects (C, H, W) float in [0, 1].
            overlay_chw = (
                torch.from_numpy(overlay_np).permute(2, 0, 1).float() / 255.0
            )
            self.pl.logger.log_image(
                phase,
                f"tg_viz/dino_query_overlay/sample_{sample_idx_for_overlay}",
                overlay_chw,
                int(self.pl.global_step),
                batch_idx,
            )
        else:
            logger.warning(
                "TgVisualizer: skipping DINO query overlay -- "
                "pixel_uv_normalized or pixel_query_mask absent from batch.extra."
            )
        # Resolve save dir: prefer trainer.log_dir (canonical PL property that
        # reads through the attached logger), then the logger's own log_dir,
        # then a fixed fallback so the viz path is never silently dropped.
        # Note: fast_dev_run sets trainer.loggers=[] -> both resolutions
        # return None; the fallback is load-bearing for smoke runs. A
        # log.warning makes the fallback visible rather than silent.
        trainer = getattr(self.pl, "trainer", None)
        log_dir = (
            getattr(trainer, "log_dir", None)
            or getattr(self.pl.logger, "log_dir", None)
        )
        if not log_dir:
            log_dir = "./logs/tg_viz_out"
            logger.warning(
                f"TgVisualizer: trainer/logger log_dir unavailable "
                f"(fast_dev_run?); writing viz to fallback '{log_dir}'."
            )
        save_root = os.path.join(log_dir, "tg_viz")
        os.makedirs(save_root, exist_ok=True)
        gstep = int(self.pl.global_step)

        # --- Run inference for pred overlay -----------------------------
        # Fail-fast (no try/except): a broken inference path in viz is a bug
        # that must surface, not silently degrade to GT-only viz. Production
        # runs can still wrap the viz call-site in ``viz_try`` if needed.
        pred_trace_xyz = None
        if eval_model is not None:
            device = getattr(self.pl, "device", None)
            if device is None:
                device = next(eval_model.parameters()).device
            inf_kwargs = _build_tg_inference_args(batch, bid=0, device=device)
            was_training = eval_model.training
            eval_model.eval()
            try:
                with torch.inference_mode():
                    _, ret_trace = eval_model.inference(
                        seed=42,
                        verbose=False,
                        **inf_kwargs,
                    )
            finally:
                if was_training:
                    eval_model.train()
            # ret_trace: (1, T, P, 4, 4) SE(3). Extract absolute xyz (Q9 + 
            # convention: xy normalized [0,1], z raw meters). Transpose to
            # (P, T, 3).
            pred_xyz_tp = ret_trace[0, :, :, :3, 3].detach().float().cpu().numpy()
            pred_trace_xyz = pred_xyz_tp.transpose(1, 0, 2)  # (P, T, 3)

        # --- Static JPGs: TWO honest single-stream views, all valid kps ---
        gt_jpg = os.path.join(
            save_root,
            f"{phase}_b{batch_idx}_gs{gstep}_{self.env_name}_gt.jpg",
        )
        tg_viz(
            sample,
            pred_trace_xyz=None,
            save_path=gt_jpg,
            env_name=self.env_name,
            image_size=self.image_size,
            max_scaffold_pts=self.max_scaffold_pts,
            global_step=gstep,
            title_extra=f"{phase} b{batch_idx}",
            mode="gt",
            draw_all=True,
        )
        pred_jpg = None
        if pred_trace_xyz is not None:
            pred_jpg = os.path.join(
                save_root,
                f"{phase}_b{batch_idx}_gs{gstep}_{self.env_name}_pred.jpg",
            )
            tg_viz(
                sample,
                pred_trace_xyz=pred_trace_xyz,
                save_path=pred_jpg,
                env_name=self.env_name,
                image_size=self.image_size,
                max_scaffold_pts=self.max_scaffold_pts,
                global_step=gstep,
                title_extra=f"{phase} b{batch_idx}",
                mode="pred",
                draw_all=True,
            )
        logger.info(
            f"TgVisualizer.tg_viz_routine: wrote JPGs gt={gt_jpg} "
            f"pred={'no' if pred_jpg is None else pred_jpg}"
        )

        # --- MP4 videos: 2D flow + 3D depth, also split into GT/pred ----
        from umm.models.wmm.viz_helpers.tg_viz_3d import (
            backproject_trace_tg,
            backproject_depth_tg,
            render_tg_2d_flow,
            render_tg_3d_depth,
        )

        trace_np = (
            sample["trace"].detach().cpu().numpy()
            if hasattr(sample["trace"], "detach")
            else np.asarray(sample["trace"])
        )
        gt_xyz_pt3 = trace_np[:, :, :3, 3]  # (P, T, 3)
        trace_mask_pt3 = (
            sample["trace_mask"].detach().cpu().numpy()
            if hasattr(sample["trace_mask"], "detach")
            else np.asarray(sample["trace_mask"])
        )
        trace_valid_pt = trace_mask_pt3.any(axis=-1).astype(bool)  # (P, T)
        first_kp_p2 = _to_numpy_req(sample["first_keypoint"])       # (P, 2)
        gt_mask_p = _to_numpy_req(sample["gt_mask"]).astype(bool)   # (P,)
        depth_raw_np = _to_numpy(sample.get("depth_raw"))           # or None

        img = _to_numpy_req(sample["image"])
        if img.ndim == 4:
            img = img[0]
        anchor_rgb = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

        # Honest "all valid keypoints" mode -- no movement filter, no top-N cap.
        drawn_idx = np.where(gt_mask_p)[0].astype(np.int32)

        gt_tp3 = gt_xyz_pt3.transpose(1, 0, 2)
        pred_tp3 = (
            pred_trace_xyz.transpose(1, 0, 2)
            if pred_trace_xyz is not None
            else None
        )
        valid_tp = trace_valid_pt.T  # (T, P)

        try:
            import imageio.v2 as _imageio
        except Exception:
            import imageio as _imageio

        def _save_2d_flow(mode: str, save_suffix: str, log_key: str,
                          gt_in: Optional[np.ndarray],
                          pred_in: Optional[np.ndarray]) -> str:
            frames = render_tg_2d_flow(
                anchor_rgb=anchor_rgb,
                gt_trace_tp3=gt_tp3 if gt_in is None else gt_in,  # always pass GT for shape
                pred_trace_tp3=pred_in,
                trace_valid_tp=valid_tp,
                first_kp_p2=first_kp_p2,
                drawn_idx=drawn_idx,
                image_size=self.image_size,
                known_n=1,
                mode=mode,
            )
            stack_thwc = np.stack(frames, axis=0)
            vid_t = (
                torch.from_numpy(stack_thwc).float().permute(0, 3, 1, 2) / 255.0
            )
            self.pl.logger.log_video(
                phase,
                f"tg_viz/flow_2d_{self.env_name}_{log_key}",
                vid_t,
                gstep,
                batch_idx,
                caption=(
                    f"env={self.env_name} ep={sample.get('episode_id', '?')} "
                    f"gs={gstep} {log_key}"
                ),
            )
            local_path = os.path.join(
                save_root,
                f"flow_2d_{phase}_b{batch_idx}_gs{gstep}_{self.env_name}_{save_suffix}.mp4",
            )
            _imageio.mimwrite(local_path, list(stack_thwc), fps=10)
            return local_path

        # GT-only 2D MP4.
        flow_gt_path = _save_2d_flow("gt", "gt", "gt", gt_tp3, None)
        flow_pred_path = None
        if pred_tp3 is not None:
            flow_pred_path = _save_2d_flow("pred", "pred", "pred", gt_tp3, pred_tp3)

        # 3D depth MP4 (world = anchor-camera frame, fabricated 53° FOV).
        gt_world_tp3 = backproject_trace_tg(gt_xyz_pt3, image_size=self.image_size)
        pred_world_tp3 = (
            backproject_trace_tg(pred_trace_xyz, image_size=self.image_size)
            if pred_trace_xyz is not None
            else None
        )
        depth_pcl = (
            backproject_depth_tg(depth_raw_np, anchor_rgb=anchor_rgb,
                                 image_size=self.image_size)
            if depth_raw_np is not None
            else None
        )

        def _save_3d_depth(mode: str, save_suffix: str, log_key: str,
                           pred_world_in: Optional[np.ndarray]) -> str:
            frames = render_tg_3d_depth(
                depth_pcl_n6=depth_pcl,
                gt_trace_world_tp3=gt_world_tp3,
                pred_trace_world_tp3=pred_world_in,
                trace_valid_tp=valid_tp,
                drawn_idx=drawn_idx,
                H=720,
                W=1280,
                known_n=1,
                mode=mode,
            )
            stack_thwc = np.stack(frames, axis=0)
            vid_t = (
                torch.from_numpy(stack_thwc).float().permute(0, 3, 1, 2) / 255.0
            )
            self.pl.logger.log_video(
                phase,
                f"tg_viz/depth_3d_{self.env_name}_{log_key}",
                vid_t,
                gstep,
                batch_idx,
                caption=(
                    f"env={self.env_name} ep={sample.get('episode_id', '?')} "
                    f"gs={gstep} {log_key}"
                ),
            )
            local_path = os.path.join(
                save_root,
                f"depth_3d_{phase}_b{batch_idx}_gs{gstep}_{self.env_name}_{save_suffix}.mp4",
            )
            _imageio.mimwrite(local_path, list(stack_thwc), fps=10)
            return local_path

        depth_gt_path = _save_3d_depth("gt", "gt", "gt", None)
        depth_pred_path = None
        if pred_world_tp3 is not None:
            depth_pred_path = _save_3d_depth(
                "pred", "pred", "pred", pred_world_tp3
            )

        logger.info(
            f"TgVisualizer.tg_viz_routine: wrote MP4s 2d_gt={flow_gt_path}, "
            f"2d_pred={flow_pred_path}, 3d_gt={depth_gt_path}, "
            f"3d_pred={depth_pred_path}"
        )
