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
from typing import Optional

import matplotlib
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cap on number of moving keypoint trajectories drawn per viz panel.
# 12 well-chosen tracks beats 100+ overlapping spaghetti for legibility.
MAX_DISPLAY_KP = 12


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

    # Cap the number of drawn moving tracks. Rank by TOTAL MOVEMENT MAGNITUDE
    # (sum of per-step L2 xy deltas on valid frames). This avoids the
    # "grid-top keypoint" bias of the earlier valid_count-only rank: TG keypoints
    # are stored in row-major grid order, so valid_counts commonly tie across
    # many kps — tie-break by moving_idx then picked the lowest indices, which
    # sit on the TOP row of the image (y≈0). The symptom in Epic viz: polylines
    # clustered at the dark wall/ceiling area above the counter. Ranking by
    # movement magnitude surfaces the actually-dynamic kps (hands, tools).
    # Secondary key keeps tied (same-movement) kps stable via valid_count then idx.
    if n_moving_total > MAX_DISPLAY_KP:
        # Step-to-step xy deltas in pixel space (P, T-1, 2); L2 norm; mask by
        # "both endpoints valid" to avoid spurious deltas across invalid gaps.
        # Invalid-kp rows (all -inf from the dataset fill step) would inject
        # NaN/inf into the sum and make lexsort keys undefined; ``nan_to_num``
        # normalises any ±inf/NaN to 0 BEFORE masking so the rank is robust
        # regardless of the trace's fill policy for fully-invalid kps.
        dxy = np.diff(trace_xy_px[:, :, :2], axis=1)
        dxy = np.nan_to_num(dxy, nan=0.0, posinf=0.0, neginf=0.0)
        step_valid = (trace_mask[:, :-1, 0] & trace_mask[:, 1:, 0]).astype(np.float32)
        step_dist = np.sqrt((dxy * dxy).sum(axis=-1)) * step_valid
        total_move = step_dist.sum(axis=1)                         # (P,)
        move_scores = total_move[moving_idx]
        valid_counts = trace_mask[moving_idx, :, 0].sum(axis=1)    # (n_moving,)
        # lexsort keys are sorted last-key-first; primary key = -move_scores.
        order = np.lexsort((moving_idx, -valid_counts, -move_scores))
        keep = order[:MAX_DISPLAY_KP]
        drawn_idx = moving_idx[keep]
    else:
        drawn_idx = moving_idx
    n_drawn = int(drawn_idx.size)

    # Discrete tab10 colors -- GT and pred share the SAME hue per track,
    # distinguished by linestyle (solid vs dashed) and endpoint marker
    # (circle vs triangle). Pred is a darker shade for stack contrast.
    base_cmap = plt.get_cmap("tab10")
    track_colors = base_cmap(np.arange(n_drawn) % 10)  # (n_drawn, 4) RGBA
    gt_colors = track_colors.copy()
    pred_colors = track_colors.copy()
    pred_colors[:, :3] *= 0.65  # darken pred RGB; keep alpha

    # pred_drawn is True only when pred input is provided AND there is at
    # least one drawn track to render it on. Gates the legend entries.
    pred_drawn = (pred_xy_px is not None) and (n_drawn > 0)

    def _polyline(k: int, base_xy_px: np.ndarray,
                  color, lw: float, alpha: float, zorder: float,
                  linestyle: str = "-"):
        """Draw a polyline through (anchor_uv, future_uv[valid_t]).

        Pred polylines are gated by the SAME GT validity mask as GT (see
        tg_viz docstring); pred values at GT-invalid timesteps are silently
        dropped so the predicted line spans the same observed timesteps as
        GT (visual comparability).
        """
        # Note: pred polylines are gated by the SAME GT validity mask as GT,
        # documented in tg_viz docstring. This is intentional for visual
        # comparability across the same observed timesteps.
        valid_t = trace_mask[k, :, 0]
        if int(valid_t.sum()) < 2:
            return
        # Build uv: prepend anchor at t=0 if not already first valid.
        future_xy = base_xy_px[k, valid_t, :]      # (T_valid, 2)
        anchor = fkp[k:k + 1, :]                    # (1, 2)
        # If first valid is t==0 and matches anchor, skip prepend.
        first_valid_t = int(np.argmax(valid_t))
        if first_valid_t == 0:
            uv = future_xy
        else:
            uv = np.concatenate([anchor, future_xy], axis=0)
        ax.plot(uv[:, 0], uv[:, 1], color=color, lw=lw, alpha=alpha,
                linestyle=linestyle, solid_capstyle="round",
                antialiased=True, zorder=zorder)

    # -- (c) GT polylines (drawn moving keypoints, solid) ----------------
    for pos, k in enumerate(drawn_idx):
        _polyline(k, trace_xy_px, gt_colors[pos], lw=1.4, alpha=0.90,
                  zorder=2.5, linestyle="-")
        # GT endpoint marker (circle).
        valid_t_k = trace_mask[k, :, 0]
        if int(valid_t_k.sum()) >= 1:
            last_t = int(np.where(valid_t_k)[0][-1])
            end_xy = trace_xy_px[k, last_t, :]
            ax.scatter(end_xy[0], end_xy[1],
                       s=12, marker="o", color=gt_colors[pos],
                       edgecolors="white", linewidths=0.3,
                       zorder=2.7)

    # -- (d) Predicted polylines (drawn moving keypoints, dashed) --------
    # Gated by pred_drawn (pred input present AND >=1 drawn track) so the
    # legend swatches reflect what was actually rendered.
    if pred_drawn:
        assert pred_xy_px is not None  # narrowed by pred_drawn guard
        for pos, k in enumerate(drawn_idx):
            _polyline(k, pred_xy_px, pred_colors[pos], lw=1.4, alpha=0.85,
                      zorder=3.0, linestyle="--")
            # Pred endpoint marker (triangle).
            valid_t_k = trace_mask[k, :, 0]
            if int(valid_t_k.sum()) >= 1:
                last_t = int(np.where(valid_t_k)[0][-1])
                end_xy = pred_xy_px[k, last_t, :]
                ax.scatter(end_xy[0], end_xy[1],
                           s=14, marker="^", color=pred_colors[pos],
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
    moving_str = f"moving={n_moving_total}/{n_valid_total}"
    if n_moving_total > MAX_DISPLAY_KP:
        moving_str += f" (showing top {MAX_DISPLAY_KP})"
    title_bits = [
        f"{env_name}", f"ep={episode_id}", f"frame={frame_id}",
        f"valid={n_valid_total}  {moving_str}",
    ]
    if step_str:
        title_bits.append(step_str)
    if title_extra:
        title_bits.append(title_extra)
    title = " | ".join(title_bits)
    ax.set_title(title, color="white", fontsize=11,
                 fontfamily="monospace", pad=10)

    # Legend uses tab10(0) (blue) as the swatch reference; pred uses
    # the same hue darkened by 0.65, matching the per-track convention.
    legend_blue = base_cmap(0)
    legend_pred_blue = (
        legend_blue[0] * 0.65, legend_blue[1] * 0.65, legend_blue[2] * 0.65,
        legend_blue[3],
    )
    handles = [
        Line2D([], [], color=legend_blue, lw=2.0, linestyle="-",
               label="GT"),
    ]
    if pred_drawn:
        handles.append(Line2D([], [], color=legend_pred_blue, lw=2.0,
                              linestyle="--", label="Pred"))
    handles.append(Line2D([], [], color=legend_blue, marker="o",
                          linestyle="none", markersize=6,
                          markeredgecolor="white", markeredgewidth=0.4,
                          label="GT endpoint"))
    if pred_drawn:
        handles.append(Line2D([], [], color=legend_pred_blue, marker="^",
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
# Wiring for PLModel._init_visualizers (scope extension).
# Minimal wrapper around tg_viz() so `visualize.tg:` YAML block is no
# longer inert. GT-only viz; inference/pred overlay is deferred to a
# follow-up change — the contract here is to prove the dispatch path fires.
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
    the TG inference-signature mapping §4 and the
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
    _B, T, P, _, _ = trace.shape
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
        real_timestep = torch.zeros(1, T, dtype=torch.float32, device=device)
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
        save_path = os.path.join(
            save_root,
            f"{phase}_b{batch_idx}_gs{gstep}_{self.env_name}.jpg",
        )

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
            # Track-and-restore training mode so this routine is safe to call
            # in isolation. The caller (``PLModel.viz``) also restores via
            # ``self.train()`` on exit — this is belt-and-suspenders for any
            # future caller that does not.
            was_training = eval_model.training
            eval_model.eval()
            try:
                with torch.inference_mode():
                    _ret_ctx, ret_trace = eval_model.inference(
                        seed=42,
                        verbose=False,
                        **inf_kwargs,
                    )
            finally:
                if was_training:
                    eval_model.train()
            # ret_trace: (1, T, P, 4, 4) SE(3). Extract absolute xyz (Q9 +
            # convention: xy normalized [0,1], z raw meters). Transpose to
            # (P, T, 3) to match the shape ``tg_viz`` expects for overlay.
            pred_xyz_tp = ret_trace[0, :, :, :3, 3].detach().float().cpu().numpy()
            pred_trace_xyz = pred_xyz_tp.transpose(1, 0, 2)  # (P, T, 3)

        tg_viz(
            sample,
            pred_trace_xyz=pred_trace_xyz,
            save_path=save_path,
            env_name=self.env_name,
            image_size=self.image_size,
            max_scaffold_pts=self.max_scaffold_pts,
            global_step=gstep,
            title_extra=f"{phase} b{batch_idx}",
        )
        logger.info(
            f"TgVisualizer.tg_viz_routine: wrote {save_path} "
            f"(pred={'yes' if pred_trace_xyz is not None else 'no'})"
        )
