"""Standalone per-sample save eval for WMM TraceGen (Droid / Epic).

Runs the TG test set through the UMM WMM model, saves every inference output
to disk, and verifies the reference metrics match bit-identically between the
in-memory accumulation and the disk-recomputed version.

Mirrors the LT backup-then-load pattern from
``eval/langtable_official/eval_and_save_wmm.py``:
  - checkpoint is copied into dst first, model loaded from copy
  - src_bck/ rsync of src, configs, eval, train.py
  - git_info.txt (branch/HEAD/status/diff/log)
  - _script_backup.py verbatim + _metrics_script_backup.py

Strict no-skip: NO try/except around inference, save, or MP4.
Any error aborts the run immediately.

CLI::

    python eval/tracegen/eval_and_save_wmm.py \\
        --checkpoint <ckpt.ckpt> \\
        --config <yaml> \\
        --output <output_dir>/<run_name> \\
        --env droid|epic \\
        [--eval_effort 1.0] [--seed 42] [--gen_timesteps 20] \\
        [--batch_size 4] [--num_workers 4] \\
        [--shard_id 0 --num_shards N]

Use ``CUDA_VISIBLE_DEVICES`` to pick GPU; the script always uses ``cuda:0``
inside.

For a single shard, ``_YYYYMMDD-HHMMSS`` is appended to ``--output``.
Eval saves can be 25-40 GB per full run.

Output layout::

    <dst>/
      rollout_config.json
      inference_config.json
      git_info.txt
      src_bck/
      _script_backup.py
      _metrics_script_backup.py
      last.ckpt
      filter_indices.json
      samples/sample_<NNNNNN>.npz
      samples/sample_<NNNNNN>.json
      debug_renders/sample_<NNNNNN>_{flow2d,depth3d}.mp4
      metrics.json
      metrics.txt
      eval.log
"""

import argparse
import json
import logging
import os
import os.path as osp
import queue
import shutil
import subprocess
import sys
import threading
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(PROJECT, "src"))
# `prepare_data` lives at the project root (e.g., prepare_data/ms3/schema.py is
# imported transitively by umm.datasets.wmm.__init__). Add project root so it
# resolves when this script is invoked from anywhere.
sys.path.insert(0, PROJECT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Verbatim copies from tg_eval_callback.py
# ────────────────────────────────────────────────────────────────────────────


def _select_eval_model(pl_module):
    """Return the uncompiled eval model.

    UMM convention: training forward goes through ``pl_module.model`` (often
    ``torch.compile``-d). Inference must run through ``pl_module.ema_model``
    when ``use_ema`` is enabled -- the uncompiled copy kept in step with EMA
    weights. Calling ``inference`` through the compiled model causes
    ``torch.compile`` to recompile on inference-shape tensors and OOM. See
    ``src/umm/models/wmm/model.py`` around line 321 where the viz path does
    the same selection.

    Args:
        pl_module: LightningModule with ``model`` + optional ``ema_model``.

    Returns:
        The nn.Module to call ``.inference(...)`` on.
    """
    if (
        getattr(pl_module, "use_ema", False)
        and getattr(pl_module, "ema_model", None) is not None
    ):
        return pl_module.ema_model
    return pl_module.model


def _tg_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of ``TracegenDataset`` dicts into one batch dict.

    Produces exactly the keys the inference-mapping pseudocode from
    expects to read. The per-coord trajectory mask is stored at
    ``batch["extra"]["trace_valid_mask_per_coord"]`` (shape
    ``(B, T=33, P=400, 3)``) AND also reduced to attention-gating shape
    ``(B, T, P)`` at ``batch["trace_mask"]`` via ``.any(-1)``.

    Args:
        samples: List of per-sample dicts from ``TracegenDataset``.

    Returns:
        Dict with stacked tensors + ``extra`` sub-dict.
    """
    B = len(samples)

    # Stack tensors (all samples share identical shapes by construction)
    trace = torch.stack([s["trace"] for s in samples], dim=0)            # (B, 400, 33, 4, 4)
    trace_mask_pc = torch.stack([s["trace_mask"] for s in samples], dim=0)  # (B, 400, 33, 3) bool
    first_kp = torch.stack([s["first_keypoint"] for s in samples], dim=0)  # (B, 400, 2)
    gt_mask = torch.stack([s["gt_mask"] for s in samples], dim=0)         # (B, 400) bool
    movement_bool = torch.stack([s["movement_bool"] for s in samples], dim=0)  # (B, 400)
    valid_steps = torch.stack([s["valid_steps"] for s in samples], dim=0)   # (B,) int64
    images = torch.stack([s["image"] for s in samples], dim=0)              # (B, 1, 3, 384, 384)

    # Permute the per-sample (400, 33, *) convention → batched (B, 33, 400, *)
    # at the metric / inference site. We keep the raw per-sample layout too
    # for downstream consumers that may need it, but the inference call and
    # metric code below assume (B, T, P, ...).
    trace = trace.permute(0, 2, 1, 3, 4).contiguous()           # (B, 33, 400, 4, 4)
    trace_mask_pc = trace_mask_pc.permute(0, 2, 1, 3).contiguous()  # (B, 33, 400, 3)
    trace_mask_any = trace_mask_pc.any(dim=-1)                   # (B, 33, 400) bool

    texts = [s["text"] for s in samples]
    depth_raws = [s.get("depth_raw") for s in samples]  # mixed None/Tensor

    extra = {
        "trace_valid_mask_per_coord": trace_mask_pc,  # (B, 33, 400, 3) — METRIC SITE
        "first_keypoint": first_kp,                    # (B, 400, 2)
        "gt_mask": gt_mask,                            # (B, 400) bool
        "images": images,                              # (B, 1, 3, 384, 384)
        "depth_raw": depth_raws,                       # list[Optional[Tensor]]
        "env_name": samples[0]["env_name"],
        "episode_id": [s["episode_id"] for s in samples],
        "frame_id": [s["frame_id"] for s in samples],
    }

    return {
        "trace": trace,
        "trace_mask": trace_mask_any,   # (B, 33, 400) — attention gate
        "movement_bool": movement_bool, # (B, 400)
        "valid_steps": valid_steps,     # (B,) int64
        "text": texts,                  # list[str] len B
        "extra": extra,
    }


def _filter_batch(batch: Dict[str, Any], valid_mask: torch.Tensor) -> Dict[str, Any]:
    """Apply authors' valid_mask across tensors and list fields.

    Args:
        batch: Output of ``_tg_collate``.
        valid_mask: ``(B,)`` bool mask from authors' filter.

    Returns:
        New batch dict with ``valid_mask`` applied to every batched field.
    """
    idx = valid_mask.nonzero(as_tuple=True)[0].tolist()

    def _take_tensor(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, torch.as_tensor(idx, device=t.device))

    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = _take_tensor(v)
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx]
        elif isinstance(v, dict):
            sub: Dict[str, Any] = {}
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    sub[kk] = _take_tensor(vv)
                elif isinstance(vv, list):
                    sub[kk] = [vv[i] for i in idx]
                else:
                    sub[kk] = vv
            out[k] = sub
        else:
            out[k] = v
    return out


def _compute_trajectory_metrics_abs(
    pred_abs: torch.Tensor,        # (N, T, 3)  absolute predicted traj
    gt_abs: torch.Tensor,          # (N, T, 3)  absolute ground-truth traj
    all_masks: torch.Tensor,       # (N, T, 3)  per-coord bool/0-1 mask
) -> Dict[str, float]:
    """Authors' ``_compute_trajectory_metrics`` inlined, ``absolute_action=True``.

    Attribution: adapted from
    ``data_wmm/tracegen/repo/trainer/trainer.py:790-846`` — the
    ``_compute_trajectory_metrics`` method body.

    Deviations from the source (all intentional and documented):

    1. ``cumsum`` step SKIPPED. Authors' pred / gt are relative deltas;
       they accumulate via ``torch.cumsum(_, dim=1)`` for the
       ``absolute_action=False`` branch. Our pred + gt are already
       absolute (Q9), so the ``absolute_action=True`` branch is
       used, which passes them through unchanged.
    2. ``torch.cat((targets[:, :1, :], preds), dim=1)`` step SKIPPED.
       Authors' ``predicted_trajectory`` has 32 steps (future only);
       they prepend GT step-0 to match the target's 33 steps. Our
       ``AgentV11.inference`` receives ``trace`` with
       ``trace_known_mask[:, 0] = True``, so ``ret_trace[:, 0]``
       is the input's known GT at t=0. The concat is therefore a
       no-op on our pred and is removed for clarity.
    3. ``TRACEGEN_STRIP_Z`` env gating OMITTED. Droid / Epic reference
       numbers in ``data_wmm/tracegen/eval.md`` were reproduced without
       stripping z. For 3D envs the metric is defined over (x, y, z);
       no gating is needed.
    4. ``all_masks is None`` fallback OMITTED. The authors' fallback
       calls ``torch.ones(gt_abs, ...)`` which is itself buggy (passes
       a tensor as the ``size`` argument); we always pass a concrete
       mask, so the branch is unreachable and removed.

    Inputs are expected already flattened to ``(N, T, 3)`` where ``N``
    covers ``B * 400`` keypoints across the batch; the caller flattens.

    Returns:
        Dict with keys ``trajectory_mse``, ``trajectory_mae``, ``endpoint_mse``.
        Returns an empty dict if ``mask.sum() <= 0``.
    """
    # ---- Authors' body (absolute_action=True branch) ----
    # See the docstring above for the explicit list of every deviation
    # from the source. The arithmetic below is byte-identical to
    # trainer.py:816-844.
    all_masks = all_masks.to(pred_abs.dtype)
    total_valid = all_masks.sum()
    if total_valid.item() <= 0:
        log.warning("_compute_trajectory_metrics_abs: no valid masked entries for trajectory metrics")
        return {}

    diff = pred_abs - gt_abs                                # (N, T, 3)
    trajectory_mse = (diff.pow(2) * all_masks).sum() / total_valid
    trajectory_mae = (diff.abs() * all_masks).sum() / total_valid

    valid_mask_time = (all_masks.sum(dim=-1) > 0).to(pred_abs.dtype)  # (N, T)
    valid_counts = valid_mask_time.sum(dim=1)
    has_valid = valid_counts > 0
    if has_valid.any():
        last_idx = (valid_counts.clamp(min=1) - 1).long()
        gather_index = last_idx.view(-1, 1, 1).expand(-1, 1, pred_abs.size(-1))
        pred_endpoints = (pred_abs * all_masks).gather(1, gather_index).squeeze(1)
        target_endpoints = (gt_abs * all_masks).gather(1, gather_index).squeeze(1)
        endpoint_mse = F.mse_loss(
            pred_endpoints[has_valid], target_endpoints[has_valid]
        ).item()
    else:
        endpoint_mse = 0.0

    return {
        "trajectory_mse": trajectory_mse.item(),
        "trajectory_mae": trajectory_mae.item(),
        "endpoint_mse": endpoint_mse,
    }


# ────────────────────────────────────────────────────────────────────────────
# Source snapshot + git
# ────────────────────────────────────────────────────────────────────────────


def get_git_sha(project_root: str) -> str:
    """Return short HEAD sha or 'unknown' on failure (no exceptions)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception as e:
        log.warning("get_git_sha: git rev-parse failed: %r", e)
        return "unknown"


def assert_dst_empty(dst: str) -> None:
    """Refuse to run if ``dst`` exists and is non-empty."""
    if osp.isdir(dst) and len(os.listdir(dst)) > 0:
        log.error(
            "Output directory %s already exists and is non-empty. "
            "Refusing to overwrite. Pick a fresh path or remove it manually.",
            dst,
        )
        sys.exit(2)


def snapshot_src(dst: str, project_root: str) -> float:
    """Mirror training's src_bck convention into ``<dst>/src_bck/``.

    Copies ``src/``, ``configs/`` (FULL), ``eval/``, and ``train.py``.
    Excludes large data artifacts (npz/ckpt/pdf/mp4/...) and Python
    bytecode. Re-uses rsync.

    Args:
        dst: Destination root (the eval output dir).
        project_root: Project root directory.

    Returns:
        Wall seconds spent in the snapshot.
    """
    t0 = time.time()
    src_bck = osp.join(dst, "src_bck")
    os.makedirs(src_bck, exist_ok=True)
    excludes = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        # serialized data
        "--exclude=*.npz", "--exclude=*.npy", "--exclude=*.pkl",
        "--exclude=*.p",
        "--exclude=*.pt", "--exclude=*.ckpt", "--exclude=*.bin",
        "--exclude=*.zip", "--exclude=*.tar", "--exclude=*.tar.gz",
        "--exclude=*.gz", "--exclude=*.h5", "--exclude=*.hdf5",
        # docs / media
        "--exclude=*.pdf", "--exclude=*.mp4",
        "--exclude=*.png", "--exclude=*.jpg", "--exclude=*.jpeg",
        # 3D mesh / sim assets
        "--exclude=*.obj", "--exclude=*.ply", "--exclude=*.stl",
        "--exclude=*.dae", "--exclude=*.glb", "--exclude=*.gltf",
        # any nested data/ dir under code trees
        "--exclude=data/",
    ]
    # If dst is itself under <project_root>/<sub>/ (e.g., output dir is
    # eval/tracegen/results/X), we MUST exclude the dst path from the
    # rsync of that sub — otherwise rsync recursively copies its own
    # growing output and never terminates. Compute per-sub relative path.
    dst_abs = osp.abspath(dst)
    for sub in ("src", "configs", "eval"):
        src_path = osp.join(project_root, sub)
        if not osp.isdir(src_path):
            log.warning("snapshot_src: %s not found, skipping", src_path)
            continue
        sub_excludes: List[str] = []
        # If dst is inside src_path, add a relative-from-src_path exclude.
        try:
            rel = osp.relpath(dst_abs, osp.abspath(src_path))
        except ValueError:
            rel = ".."  # different drive on Windows; safe fallback
        if not rel.startswith("..") and rel != ".":
            sub_excludes.append("--exclude=/{}".format(rel))
            log.info(
                "snapshot_src: dst is under %s; adding self-exclude=/%s",
                sub, rel,
            )
        dst_path = osp.join(src_bck, sub) + "/"
        subprocess.run(
            ["rsync", "-a"] + excludes + sub_excludes + [src_path + "/", dst_path],
            check=True,
        )
    train_py = osp.join(project_root, "train.py")
    if osp.isfile(train_py):
        shutil.copyfile(train_py, osp.join(src_bck, "train.py"))
    return time.time() - t0


def write_git_info(dst: str, project_root: str) -> None:
    """Write ``<dst>/git_info.txt`` with branch/HEAD/status/diff/log -5.

    Args:
        dst: Destination root.
        project_root: Project root directory.
    """
    info_path = osp.join(dst, "git_info.txt")
    queries = [
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("HEAD", ["git", "rev-parse", "HEAD"]),
        ("HEAD short", ["git", "rev-parse", "--short", "HEAD"]),
        ("upstream", ["git", "rev-parse", "--abbrev-ref", "@{u}"]),
        ("upstream HEAD", ["git", "rev-parse", "@{u}"]),
        ("log -1",
         ["git", "log", "-1",
          "--pretty=format:%h %an %ad %s", "--date=iso"]),
        ("log -5 (compact)",
         ["git", "log", "-5", "--oneline"]),
        ("status (porcelain)", ["git", "status", "--porcelain"]),
        ("diff --stat (unstaged)", ["git", "diff", "--stat"]),
        ("diff --stat (staged)", ["git", "diff", "--cached", "--stat"]),
        ("remote -v", ["git", "remote", "-v"]),
    ]
    parts = [
        "timestamp: {}".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        "project_root: {}".format(project_root),
        "hostname: {}".format(os.uname().nodename),
    ]
    for label, cmd in queries:
        try:
            out = subprocess.check_output(
                cmd, cwd=project_root, stderr=subprocess.DEVNULL
            ).decode().rstrip()
        except Exception as e:
            out = "<error: {}: {}>".format(type(e).__name__, e)
        parts.append("\n--- {} ---\n{}".format(label, out))
    with open(info_path, "w") as f:
        f.write("\n".join(parts) + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Threaded save queue (overlap disk I/O with next batch's GPU work)
# ────────────────────────────────────────────────────────────────────────────
#
# `np.savez` (uncompressed) instead of `np.savez_compressed`: gzip on the main
# thread was the bottleneck. Per-sample npz is ~3-5 MB raw vs ~1.7 MB gzipped;
# disk writes at ~500 MB/s vs gzip CPU ~50 MB/s → ~10x save speedup.
#
# A single background thread drains a bounded queue and runs np.savez + json
# write per item. Strict no-skip: any exception on the worker is captured and
# re-raised on the main thread before the next enqueue (and at drain).


_SAVE_QUEUE: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=8)
_SAVE_ERROR: List[BaseException] = []


def _save_worker_loop() -> None:
    """Drain the save queue. Capture any exception for main-thread re-raise."""
    while True:
        item = _SAVE_QUEUE.get()
        if item is None:
            _SAVE_QUEUE.task_done()
            return
        try:
            np.savez(item["npz_path"], **item["npz_kwargs"])
            with open(item["json_path"], "w") as f:
                json.dump(item["json_payload"], f, indent=2)
        except BaseException as e:  # noqa: BLE001 — re-raised in main thread
            _SAVE_ERROR.append(e)
        finally:
            _SAVE_QUEUE.task_done()


def _check_save_error() -> None:
    """Re-raise any captured save-thread exception. Strict no-skip."""
    if _SAVE_ERROR:
        raise RuntimeError(
            "save thread crashed: {}: {}".format(
                type(_SAVE_ERROR[0]).__name__, _SAVE_ERROR[0]
            )
        ) from _SAVE_ERROR[0]


# ────────────────────────────────────────────────────────────────────────────
# Debug MP4 renders (inline, first 2 kept samples)
# ────────────────────────────────────────────────────────────────────────────


def _render_debug_mp4s(
    dst: str,
    subset_global_idx: int,
    image_uint8_chw: np.ndarray,      # (3, 384, 384) uint8
    gt_trace_se3_tpkk: np.ndarray,    # (33, 400, 4, 4)
    pred_trace_se3_tpkk: np.ndarray,  # (33, 400, 4, 4)
    mask_pc_tpc: np.ndarray,           # (33, 400, 3) bool
    first_kp_p2: np.ndarray,          # (400, 2)
    gt_mask_p: np.ndarray,            # (400,) bool
    depth_raw_np: Optional[np.ndarray],  # (H, W) float32 or None
) -> None:
    """Render flow2d and depth3d MP4s for a debug sample.

    Args:
        dst: Eval output root (debug_renders/ is created inside it).
        subset_global_idx: Zero-padded index for filename.
        image_uint8_chw: Anchor RGB ``(3, 384, 384) uint8``.
        gt_trace_se3_tpkk: GT SE(3) trace ``(33, 400, 4, 4)``.
        pred_trace_se3_tpkk: Pred SE(3) trace ``(33, 400, 4, 4)``.
        mask_pc_tpc: Per-coord validity ``(33, 400, 3) bool``.
        first_kp_p2: Anchor keypoints ``(400, 2) float32``.
        gt_mask_p: Valid keypoint mask ``(400,) bool``.
        depth_raw_np: Depth ``(H, W) float32`` or ``None``.
    """
    from umm.models.wmm.viz_helpers.tg_viz_3d import (
        backproject_depth_tg,
        backproject_trace_tg,
        render_tg_2d_flow,
        render_tg_3d_depth,
    )

    debug_dir = osp.join(dst, "debug_renders")
    os.makedirs(debug_dir, exist_ok=True)

    # anchor_rgb: (384, 384, 3) uint8
    anchor_rgb = image_uint8_chw.transpose(1, 2, 0)

    # Extract translations
    gt_trace_tpc3 = gt_trace_se3_tpkk[..., :3, 3]    # (33, 400, 3)
    pred_trace_tpc3 = pred_trace_se3_tpkk[..., :3, 3]  # (33, 400, 3)

    # trace_valid_tp: (33, 400) bool — any() over coord axis
    trace_valid_tp = mask_pc_tpc.any(axis=-1)           # (33, 400)

    # gt_xyz_pt3: (400, 33, 3) for backproject_trace_tg (which expects (P,T,3))
    gt_xyz_pt3 = gt_trace_tpc3.transpose(1, 0, 2)       # (400, 33, 3)
    pred_xyz_pt3 = pred_trace_tpc3.transpose(1, 0, 2)   # (400, 33, 3)

    drawn_idx = np.where(gt_mask_p)[0].astype(np.int32)

    # 2D flow
    frames_2d = render_tg_2d_flow(
        anchor_rgb=anchor_rgb,
        gt_trace_tp3=gt_trace_tpc3,       # (T=33, P=400, 3)
        pred_trace_tp3=pred_trace_tpc3,   # (T=33, P=400, 3)
        trace_valid_tp=trace_valid_tp,    # (T=33, P=400)
        first_kp_p2=first_kp_p2,          # (400, 2)
        drawn_idx=drawn_idx,
        image_size=384,
        known_n=1,
        mode="both",
    )
    path_2d = osp.join(
        debug_dir, "sample_{:06d}_flow2d.mp4".format(subset_global_idx)
    )
    imageio.mimwrite(path_2d, frames_2d, fps=10)
    log.info("debug render 2D flow -> %s", path_2d)

    # 3D depth
    gt_world_tp3 = backproject_trace_tg(gt_xyz_pt3, image_size=384)    # (T, P, 3)
    pred_world_tp3 = backproject_trace_tg(pred_xyz_pt3, image_size=384)  # (T, P, 3)
    depth_pcl = (
        backproject_depth_tg(depth_raw_np, anchor_rgb=anchor_rgb, image_size=384)
        if depth_raw_np is not None
        else None
    )
    frames_3d = render_tg_3d_depth(
        depth_pcl_n6=depth_pcl,
        gt_trace_world_tp3=gt_world_tp3,    # (T, P, 3)
        pred_trace_world_tp3=pred_world_tp3,  # (T, P, 3)
        trace_valid_tp=trace_valid_tp,      # (T, P)
        drawn_idx=drawn_idx,
        H=720,
        W=1280,
        known_n=1,
        mode="both",
    )
    path_3d = osp.join(
        debug_dir, "sample_{:06d}_depth3d.mp4".format(subset_global_idx)
    )
    imageio.mimwrite(path_3d, frames_3d, fps=10)
    log.info("debug render 3D depth -> %s", path_3d)


# ────────────────────────────────────────────────────────────────────────────
# End-of-run integrity gate
# ────────────────────────────────────────────────────────────────────────────

_REQUIRED_NPZ_KEYS = {
    "pred_trace_abs",      # (33, 400, 3)
    "gt_trace_abs",        # (33, 400, 3)
    "pred_trace_se3",      # (33, 400, 4, 4)
    "gt_trace_se3",        # (33, 400, 4, 4)
    "trace_valid_mask_per_coord",  # (33, 400, 3)
    "valid_steps",         # scalar int64
    "movement_bool",       # (400,)
    "gt_mask",             # (400,)
    "first_keypoint",      # (400, 2)
    "image_uint8",         # (3, 384, 384)
    "depth_raw",           # (H, W) or (0, 0) placeholder
    "kept_by_filter",      # scalar bool
}

_REQUIRED_JSON_KEYS = {
    "episode_id",
    "frame_id",
    "text",
    "env_name",
    "seed_used",
    "kept_by_filter",
    "valid_steps_int",
    "movement_count",
    "gt_mask_count",
    "subset_global_idx",
    "has_depth",
}

_NPZ_EXPECTED_SHAPES = {
    "pred_trace_abs": (33, 400, 3),
    "gt_trace_abs": (33, 400, 3),
    "pred_trace_se3": (33, 400, 4, 4),
    "gt_trace_se3": (33, 400, 4, 4),
    "trace_valid_mask_per_coord": (33, 400, 3),
    "movement_bool": (400,),
    "gt_mask": (400,),
    "first_keypoint": (400, 2),
    "image_uint8": (3, 384, 384),
}


def _run_integrity_gate(dst: str, n_subset: int) -> None:
    """Verify all output files exist with correct shapes.

    Args:
        dst: Eval output root.
        n_subset: Expected number of subset samples.

    Raises:
        RuntimeError: If any file is missing or has wrong shape.
    """
    npz_files = sorted(glob(osp.join(dst, "samples", "*.npz")))
    json_files = sorted(glob(osp.join(dst, "samples", "*.json")))

    if len(npz_files) != n_subset:
        raise RuntimeError(
            "Integrity gate: expected {} npz files, got {}.".format(
                n_subset, len(npz_files)
            )
        )
    if len(json_files) != n_subset:
        raise RuntimeError(
            "Integrity gate: expected {} json files, got {}.".format(
                n_subset, len(json_files)
            )
        )

    for jpath in json_files:
        with open(jpath) as f:
            jdata = json.load(f)
        missing = _REQUIRED_JSON_KEYS - set(jdata.keys())
        if missing:
            raise RuntimeError(
                "Integrity gate: json {} missing keys: {}".format(jpath, missing)
            )

    for npath in npz_files:
        with np.load(npath) as npz:
            present = set(npz.files)
            missing = _REQUIRED_NPZ_KEYS - present
            if missing:
                raise RuntimeError(
                    "Integrity gate: npz {} missing keys: {}".format(npath, missing)
                )
            for key, expected_shape in _NPZ_EXPECTED_SHAPES.items():
                actual = npz[key].shape
                if actual != expected_shape:
                    raise RuntimeError(
                        "Integrity gate: npz {} key '{}' shape {} != expected {}.".format(
                            npath, key, actual, expected_shape
                        )
                    )

    log.info("Integrity gate PASSED: %d npz + %d json verified.", n_subset, n_subset)


# ────────────────────────────────────────────────────────────────────────────
# Config helpers
# ────────────────────────────────────────────────────────────────────────────


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested OmegaConf / dict with dot-path keys.

    Args:
        cfg: OmegaConf DictConfig or plain dict.
        *keys: Key path (each element is one level).
        default: Value to return if path is missing.

    Returns:
        Value at path or ``default``.
    """
    node = cfg
    for k in keys:
        try:
            node = node[k]
        except (KeyError, TypeError, AttributeError):
            try:
                node = getattr(node, k)
            except AttributeError:
                return default
    return node


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901 — linear flow, intentionally long
    """Run TraceGen eval, save per-sample outputs, verify metrics."""
    parser = argparse.ArgumentParser(
        description="WMM TraceGen standalone eval with per-sample disk save."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to UMM WMM .ckpt")
    parser.add_argument("--config", required=True,
                        help="Path to training YAML config")
    parser.add_argument(
        "--output", required=True,
        help=("Output directory prefix (must be empty/missing). A full run "
              "writes 25-40 GB.")
    )
    parser.add_argument("--env", required=True, choices=["droid", "epic"],
                        help="TraceGen env name")
    parser.add_argument("--eval_effort", type=float, default=1.0,
                        help="Fraction of test set to evaluate (1.0=full)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed passed to every model.inference() call")
    parser.add_argument("--gen_timesteps", type=int, default=None,
                        help="Override diffusion gen_timesteps (default: from cfg)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index (0-based). Shard 0 owns dst setup + finalize.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help=("Total number of shards. When >1, each shard processes "
                              "a contiguous chunk of the deterministic subset and "
                              "writes to the SAME dst. Shard 0 finalizes."))
    parser.add_argument(
        "--sw", action="store_true",
        help=("Sliding-window inference: dispatch to "
              "AgentV11.inference_sliding_window(K_train, anchor_count). "
              "K_train is read from cfg.dataset.configs[0].num_kps and "
              "anchor_count from cfg.model.diffusion.sliding_window_anchors. "
              "Required when K_train != P_test (e.g. K=128 model on P=400 set)."),
    )
    args = parser.parse_args()

    if not (0.0 < args.eval_effort <= 1.0):
        raise ValueError(
            "--eval_effort must be in (0.0, 1.0], got {}".format(args.eval_effort)
        )
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            "--shard_id must be in [0, num_shards); got shard_id={} num_shards={}"
            .format(args.shard_id, args.num_shards)
        )

    # Auto-append _YYYYMMDD-HHMMSS to --output IN PLACE for single-shard
    # runs so all later code that reads args.output gets the suffixed value
    # automatically. For multi-shard, the launcher MUST compute the
    # timestamp once and pass the SAME --output to every shard (otherwise
    # each shard would compute a different ts -> different dst -> broken).
    # Capture the user's raw input on a separate attr for provenance.
    args.output_user_input = args.output
    if args.num_shards == 1:
        _ts_compact = time.strftime("%Y%m%d-%H%M%S")
        args.output = osp.abspath(osp.expanduser(f"{args.output}_{_ts_compact}"))
        print("[main] resolved dst (auto-suffixed): {}".format(args.output))
    else:
        args.output = osp.abspath(osp.expanduser(args.output))
        print("[main] multi-shard: dst preserved verbatim (no auto-suffix): {}"
              .format(args.output))

    is_leader = (args.shard_id == 0)
    is_multi = (args.num_shards > 1)

    t_total_start = time.time()

    # ── 1. Guard + makedirs ──────────────────────────────────────────────────
    # Multi-shard: ONLY the leader (shard 0) does the empty-dst guard and
    # creates the dirs. Workers wait for the `.shard_ready` sentinel.
    if is_leader:
        assert_dst_empty(args.output)
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(osp.join(args.output, "samples"), exist_ok=True)
        if is_multi:
            os.makedirs(osp.join(args.output, "_shards"), exist_ok=True)
    else:
        # Worker shard. Wait for leader to finish setup (up to 15 min).
        ready_path = osp.join(args.output, ".shard_ready")
        deadline = time.time() + 900
        while not osp.exists(ready_path):
            if time.time() > deadline:
                raise RuntimeError(
                    "shard {} timed out waiting for leader's .shard_ready sentinel "
                    "(15 min) under {}".format(args.shard_id, args.output)
                )
            time.sleep(2)

    # File handler — leader writes eval.log; workers write eval.log.shard{N}
    # to avoid handle conflicts on the same file.
    log_basename = "eval.log" if is_leader else "eval.log.shard{}".format(args.shard_id)
    _fh = logging.FileHandler(osp.join(args.output, log_basename))
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(_fh)

    device = torch.device("cuda:0")
    log.info(
        "Device = %s (CUDA_VISIBLE_DEVICES=%s) shard=%d/%d leader=%s",
        device,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        args.shard_id, args.num_shards, is_leader,
    )

    # ── 2. Backup-then-load: copy ckpt into dst first ────────────────────────
    # Multi-shard: only the leader copies; workers reuse the existing copy.
    src_ckpt = osp.abspath(args.checkpoint)
    dst_ckpt = osp.join(args.output, osp.basename(src_ckpt))
    if is_leader:
        ckpt_size_mb = osp.getsize(src_ckpt) / (1 << 20)
        log.info(
            "Copying ckpt: %s -> %s (size=%.1f MB)", src_ckpt, dst_ckpt, ckpt_size_mb
        )
        t_cp = time.time()
        shutil.copyfile(src_ckpt, dst_ckpt)
        log.info("Ckpt copy done in %.1fs", time.time() - t_cp)
    else:
        if not osp.isfile(dst_ckpt):
            raise RuntimeError(
                "shard {} expected ckpt copy at {} (leader's job); not found"
                .format(args.shard_id, dst_ckpt)
            )

    # ── 3. Source snapshot — leader only ─────────────────────────────────────
    if is_leader:
        log.info("Snapshotting src/configs/eval into %s/src_bck/ ...", args.output)
        snap_sec = snapshot_src(args.output, PROJECT)
        log.info("src snapshot done in %.1fs", snap_sec)

    # ── 4. Git info — leader only ────────────────────────────────────────────
    if is_leader:
        write_git_info(args.output, PROJECT)
        log.info("Wrote git_info.txt")

    # ── 5. Script backups — leader only ──────────────────────────────────────
    if is_leader:
        shutil.copyfile(__file__, osp.join(args.output, "_script_backup.py"))
        metrics_script = osp.join(osp.dirname(osp.abspath(__file__)),
                                  "eval_compute_metrics_wmm.py")
        shutil.copyfile(metrics_script, osp.join(args.output, "_metrics_script_backup.py"))
        log.info("Wrote _script_backup.py and _metrics_script_backup.py")

    # ── 6. Load model ────────────────────────────────────────────────────────
    from umm.training_utils import get_config
    from umm.models.wmm.model import PLModel

    argv_bak = sys.argv[:]
    sys.argv = ["eval", "--config", args.config]
    try:
        cfg = get_config()
    finally:
        sys.argv = argv_bak

    log.info("Loading model from %s ...", dst_ckpt)
    # `strict=False` mirrors the LT pattern. To keep the audit trail honest,
    # we explicitly count and log any missing/unexpected keys after load,
    # so silent partial-init never goes unnoticed.
    _load_result = PLModel.load_from_checkpoint(
        dst_ckpt, cfg=cfg, map_location=device, strict=False
    )
    pl_module = _load_result
    pl_module.to(device).eval()
    # Read the raw state_dict keys vs ckpt keys to surface any mismatch.
    _ckpt_dict = torch.load(dst_ckpt, map_location="cpu")
    _ckpt_keys = set(_ckpt_dict.get("state_dict", {}).keys())
    _model_keys = set(pl_module.state_dict().keys())
    _missing = _model_keys - _ckpt_keys
    _unexpected = _ckpt_keys - _model_keys
    if _missing or _unexpected:
        log.warning(
            "load_from_checkpoint(strict=False): %d missing, %d unexpected keys",
            len(_missing), len(_unexpected),
        )
        if _missing:
            log.warning("  first 5 missing: %s", sorted(_missing)[:5])
        if _unexpected:
            log.warning("  first 5 unexpected: %s", sorted(_unexpected)[:5])
    else:
        log.info("load_from_checkpoint: state_dict keys match exactly")
    del _ckpt_dict, _ckpt_keys, _model_keys, _missing, _unexpected

    eval_model = _select_eval_model(pl_module)
    eval_model.eval()
    log.info(
        "Model loaded. use_ema=%s  eval_model type=%s",
        getattr(pl_module, "use_ema", False),
        type(eval_model).__name__,
    )

    # ── 7. Resolve dataset parameters from cfg ───────────────────────────────
    # Prefer tg_eval block if present; fall back to dataset.configs[0]
    ds_cfg = _cfg_get(cfg, "dataset", "configs")
    ds0 = ds_cfg[0] if ds_cfg is not None else {}
    tg_eval_blk = _cfg_get(cfg, "tg_eval")

    # Explicit None-check (NOT `or`-chain) so that explicit False/0/0.0 in
    # cfg are not silently overridden. `or` would skip a legitimate False.
    def _resolve(*candidates, default=None):
        for v in candidates:
            if v is not None:
                return v
        return default

    base_fps = float(_resolve(
        _cfg_get(tg_eval_blk, "base_fps"),
        _cfg_get(ds0, "base_fps"),
        default=30.0,
    ))
    t_rate = int(_resolve(
        _cfg_get(tg_eval_blk, "t_rate"),
        _cfg_get(ds0, "t_rate"),
        default=1,
    ))
    dummy_id_rot = bool(_resolve(
        _cfg_get(tg_eval_blk, "dummy_id_rot"),
        _cfg_get(ds0, "dummy_id_rot"),
        default=False,
    ))
    data_root = _resolve(
        _cfg_get(tg_eval_blk, "test_data_root"),
        _cfg_get(ds0, "data_root"),
        default=None,
    )
    if data_root is None:
        raise RuntimeError(
            "Cannot resolve data_root from cfg. "
            "Expected tg_eval.test_data_root or dataset.configs[0].data_root"
        )

    # gen_timesteps: CLI override > cfg
    if args.gen_timesteps is not None:
        gen_timesteps = int(args.gen_timesteps)
    else:
        gen_timesteps = int(
            _cfg_get(cfg, "model", "diffusion", "gen_timesteps") or 20
        )

    # use_ema from cfg
    use_ema = bool(getattr(pl_module, "use_ema", False))

    # ── Sliding-window dispatch params (only used when --sw is set) ──────────
    # Mirror Tg128EvalCallback: K_train comes from dataset.configs[0].num_kps;
    # anchor_count from model.diffusion.sliding_window_anchors. Read regardless
    # of --sw so the cfg shape is validated, but only enforced under --sw.
    sw_K_train: Optional[int] = None
    sw_anchor_count: Optional[int] = None
    if args.sw:
        _k_raw = _cfg_get(ds0, "num_kps")
        if _k_raw is None:
            raise RuntimeError(
                "--sw requires cfg.dataset.configs[0].num_kps; got None"
            )
        sw_K_train = int(_k_raw)
        if sw_K_train <= 0:
            raise ValueError(
                f"--sw: K_train must be a positive int, got {sw_K_train}"
            )
        _ac_raw = _cfg_get(cfg, "model", "diffusion", "sliding_window_anchors")
        if _ac_raw is None:
            sw_anchor_count = sw_K_train // 2
            log.warning(
                "--sw: model.diffusion.sliding_window_anchors not in cfg; "
                "defaulting anchor_count to K_train // 2 = %d", sw_anchor_count,
            )
        else:
            sw_anchor_count = int(_ac_raw)
        if not (0 <= sw_anchor_count < sw_K_train):
            raise ValueError(
                f"--sw: anchor_count must be in [0, K_train); "
                f"got anchor_count={sw_anchor_count}, K_train={sw_K_train}"
            )
        log.info(
            "--sw ENABLED: K_train=%d anchor_count=%d",
            sw_K_train, sw_anchor_count,
        )

    log.info(
        "Eval params: env=%s data_root=%s base_fps=%.1f t_rate=%d "
        "dummy_id_rot=%s gen_timesteps=%d use_ema=%s eval_effort=%.4f seed=%d "
        "sw=%s",
        args.env, data_root, base_fps, t_rate, dummy_id_rot,
        gen_timesteps, use_ema, args.eval_effort, args.seed, args.sw,
    )

    # ── 8. Build dataset ─────────────────────────────────────────────────────
    from umm.datasets.wmm.tg_dataset import TracegenDataset

    ds = TracegenDataset(
        data_root=str(data_root),
        env_name=args.env,
        split="test",
        base_fps=base_fps,
        t_rate=t_rate,
        dummy_id_rot=dummy_id_rot,
    )
    N = len(ds)
    log.info("TracegenDataset: %d test samples total", N)

    # ── 9. Apply eval_effort subset (leader computes; workers read) ─────────
    if is_leader:
        n_eval = max(1, int(round(N * args.eval_effort)))
        indices = np.linspace(0, N - 1, n_eval).round().astype(int).tolist()
        with open(osp.join(args.output, "filter_indices.json"), "w") as f:
            json.dump({"N_total": N, "subset_indices": indices}, f, indent=2)
    else:
        # Worker reads what the leader wrote (single source of truth).
        with open(osp.join(args.output, "filter_indices.json")) as f:
            fi = json.load(f)
        if int(fi["N_total"]) != N:
            raise RuntimeError(
                "shard {} dataset N mismatch: leader_wrote={} my_N={}"
                .format(args.shard_id, fi["N_total"], N)
            )
        indices = list(fi["subset_indices"])

    n_subset = len(indices)
    # Per-shard slice (contiguous chunk). All shards share the SAME global
    # numbering = position in `indices`, so npz filenames `sample_<NNNNNN>.npz`
    # never collide across shards.
    n_per_shard = (n_subset + args.num_shards - 1) // args.num_shards
    my_start = args.shard_id * n_per_shard
    my_end = min((args.shard_id + 1) * n_per_shard, n_subset)
    my_indices = indices[my_start:my_end]
    my_global_idx_offset = my_start
    log.info(
        "eval_effort=%.4f -> subset %d / %d  shard=%d/%d  my_range=[%d, %d) (n=%d)",
        args.eval_effort, n_subset, N,
        args.shard_id, args.num_shards, my_start, my_end, len(my_indices),
    )
    if len(my_indices) == 0:
        # Acceptable when num_shards > n_subset; shard does nothing but still
        # writes its done sentinel so the leader's wait does not stall.
        log.warning("shard %d has no samples (num_shards=%d > n_subset=%d)",
                    args.shard_id, args.num_shards, n_subset)

    # ── 10. DataLoader (per-shard) ──────────────────────────────────────────
    my_subset = Subset(ds, my_indices)
    loader = DataLoader(
        my_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        num_workers=args.num_workers,
        collate_fn=_tg_collate,
    )

    # ── 11. Write rollout_config.json + inference_config.json (leader only) ─
    if is_leader:
        git_sha = get_git_sha(PROJECT)
        rollout_cfg = {
            "checkpoint": args.checkpoint,
            "checkpoint_src": src_ckpt,
            "checkpoint_loaded_from": dst_ckpt,
            "config": args.config,
            "output": args.output,
            "env": args.env,
            "eval_effort": args.eval_effort,
            "num_samples_total": N,
            "num_samples_subset": n_subset,
            "num_shards": args.num_shards,
            "git_sha": git_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reference_locked": True,
        }
        with open(osp.join(args.output, "rollout_config.json"), "w") as f:
            json.dump(rollout_cfg, f, indent=2)

        inference_cfg = {
            "seed": args.seed,
            "gen_timesteps": gen_timesteps,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_shards": args.num_shards,
            "dtype": "bfloat16-autocast",
            "device": str(device),
            "use_ema": use_ema,
            "base_fps": base_fps,
            "t_rate": t_rate,
            "dummy_id_rot": dummy_id_rot,
            "sw": bool(args.sw),
            "sw_K_train": sw_K_train,
            "sw_anchor_count": sw_anchor_count,
        }
        with open(osp.join(args.output, "inference_config.json"), "w") as f:
            json.dump(inference_cfg, f, indent=2)

        # Sentinel: workers may now proceed.
        from pathlib import Path as _Path
        _Path(osp.join(args.output, ".shard_ready")).touch()
        log.info(
            "Leader wrote rollout_config.json, inference_config.json, .shard_ready"
        )

    # ── 12. Inference loop ───────────────────────────────────────────────────
    from umm.datasets.wmm.dataset_config import SEM_REAL_SCENE

    # subset_global_idx counts the GLOBAL position in the deterministic
    # subset (NOT the per-shard local position). This guarantees per-sample
    # filenames `sample_<NNNNNN>.npz` are unique across shards.
    subset_global_idx = my_global_idx_offset
    n_debug_renders_done = 0  # render MP4 for first 2 kept samples (leader only)

    # In-memory accumulators (CPU) for kept-by-filter samples
    pred_chunks: List[torch.Tensor] = []
    gt_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    n_samples_valid = 0

    # Start save-thread BEFORE the loop. Worker drains _SAVE_QUEUE; bounded
    # at maxsize=8 so a slow disk back-pressures GPU work instead of OOMing.
    save_thread = threading.Thread(target=_save_worker_loop, daemon=True)
    save_thread.start()
    log.info("Started save-thread (daemon, queue maxsize=8)")

    for batch_i, batch in enumerate(loader):
        # Authors' filter — same line as callback:600
        valid_mask = (
            (batch["valid_steps"] > 3)
            & (batch["movement_bool"].sum(dim=1) > 0)
            & (batch["extra"]["gt_mask"].sum(dim=1) > 0)
        )  # (B,) bool

        # Build inference inputs (do NOT pre-filter — we inference EVERYTHING)
        trace = batch["trace"].to(device)                      # (B, 33, 400, 4, 4)
        trace_mask_any = batch["trace_mask"].to(device)        # (B, 33, 400) bool
        mask_pc = batch["extra"]["trace_valid_mask_per_coord"].to(device)  # (B,33,400,3)

        B, T, P, _, _ = trace.shape
        assert T == 33 and P == 400, "Unexpected (T,P): ({},{})".format(T, P)

        context = trace[:, :1].clone()                          # (B, 1, 400, 4, 4)
        trace_known = torch.zeros(B, T, P, dtype=torch.bool, device=device)
        trace_known[:, 0] = True
        ctx_known = torch.ones(B, 1, P, dtype=torch.bool, device=device)

        gt_mask_dev = batch["extra"]["gt_mask"].to(device)     # (B, 400)
        ctx_mask = gt_mask_dev.unsqueeze(1)                    # (B, 1, 400)

        time_unit = float(t_rate) / float(base_fps)
        real_timestep = (
            torch.arange(T, dtype=torch.float32, device=device) * time_unit
        ).unsqueeze(0).expand(B, -1).contiguous()
        group_id = torch.zeros(B, P, dtype=torch.long, device=device)
        sem_type = torch.full(
            (B, P), fill_value=int(SEM_REAL_SCENE), dtype=torch.long, device=device
        )
        if dummy_id_rot:
            rot_mask = torch.ones(B, P, dtype=torch.bool, device=device)
        else:
            rot_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        images = batch["extra"]["images"].to(device)           # (B, 1, 3, 384, 384)
        view_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        pixel_uv = batch["extra"]["first_keypoint"].to(device) # (B, 400, 2)
        pixel_view_idx = torch.zeros(B, P, dtype=torch.long, device=device)
        pixel_query_mask = gt_mask_dev                         # (B, 400)

        # Build inference_kwargs.
        # `seed` and `n_steps` are ALWAYS passed using the resolved values
        # (CLI override or cfg). Hard-fixing both eliminates any drift between
        # the model's internal default and what is recorded in
        # inference_config.json — the audit trail must match the actual call.
        infer_kwargs: Dict[str, Any] = {
            "seed": args.seed,
            "verbose": False,
            "n_steps": gen_timesteps,
        }

        eval_model.eval()
        # bf16 autocast is the DEFAULT for tracegen eval — bf16 matmul on Ampere
        # tensor cores is ~3× faster than fp32 with negligible quality loss
        # (verified vs prior w1.3.3 bf16 runs). The provenance string in
        # inference_config.json must match (search "dtype" above).
        # Dispatch: --sw selects inference_sliding_window when K_train != P;
        # falls through to plain inference() when K_train == P (matches the
        # Tg128EvalCallback dispatch contract).
        _common_kwargs = dict(
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
            text=batch["text"],
            local_pcl=None,
            local_pcl_mask=None,
            images=images,
            view_ids=view_ids,
            pixel_uv_normalized=pixel_uv,
            pixel_view_idx=pixel_view_idx,
            pixel_query_mask=pixel_query_mask,
        )
        # When --sw is set, sw_K_train / sw_anchor_count are guaranteed
        # non-None by the resolution block above. Narrow once for the type
        # checker so the dispatch branch below is clean.
        _use_sw = bool(args.sw and sw_K_train is not None and int(sw_K_train) != int(P))
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if _use_sw:
                assert sw_K_train is not None and sw_anchor_count is not None  # for pyright
                ret_context, ret_trace = eval_model.inference_sliding_window(
                    K_train=sw_K_train,
                    anchor_count=sw_anchor_count,
                    **_common_kwargs,
                    **infer_kwargs,
                )
            else:
                # Mirror Tg128EvalCallback: when --sw is set but K_train == P,
                # fall through to plain inference() — but log it loudly once
                # so the audit trail isn't a silent fallback.
                if args.sw and batch_i == 0 and sw_K_train is not None:
                    log.warning(
                        "--sw set but K_train (%d) == P (%d); falling through "
                        "to plain inference() (matches Tg128EvalCallback contract).",
                        int(sw_K_train), int(P),
                    )
                ret_context, ret_trace = eval_model.inference(
                    **_common_kwargs,
                    **infer_kwargs,
                )

        # (B, 33, 400, 4, 4) SE(3) — to CPU float32 numpy
        pred_trace_se3 = ret_trace.detach().float().cpu().numpy()  # (B,33,400,4,4)
        gt_trace_se3 = trace.detach().float().cpu().numpy()        # (B,33,400,4,4)
        pred_trace_abs = pred_trace_se3[..., :3, 3]               # (B,33,400,3)
        gt_trace_abs = gt_trace_se3[..., :3, 3]                   # (B,33,400,3)
        mask_pc_np = mask_pc.detach().cpu().numpy()               # (B,33,400,3) bool

        # Per-sample save
        for i in range(B):
            # NaN/Inf guard — strict, no skip
            assert np.isfinite(pred_trace_abs[i]).all(), (
                "NaN/Inf in pred at subset_global_idx={}".format(subset_global_idx)
            )

            kept = bool(valid_mask[i].item())

            # image_uint8: squeeze view dim, (3,384,384) -> uint8
            img_float = batch["extra"]["images"][i]          # (1, 3, 384, 384)
            img_uint8 = (
                img_float[0].cpu().numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)                  # (3, 384, 384)

            # depth_raw: may be None
            depth_raw_i = batch["extra"]["depth_raw"][i]
            if depth_raw_i is not None:
                depth_raw_np = depth_raw_i.cpu().numpy().astype(np.float32)
                has_depth = True
            else:
                depth_raw_np = np.zeros((0, 0), dtype=np.float32)
                has_depth = False

            # Build npz kwargs + json payload, ENQUEUE for the save-thread.
            # All numpy arrays are .copy()'d so the worker thread does not see
            # tensors mutated by the main thread on the next batch.
            gt_mask_arr = batch["extra"]["gt_mask"][i].cpu().numpy().astype(bool)
            movement_arr = batch["movement_bool"][i].cpu().numpy().astype(bool)
            episode_id = batch["extra"]["episode_id"][i]
            frame_id = batch["extra"]["frame_id"][i]
            text_i = batch["text"][i]

            npz_path = osp.join(
                args.output, "samples",
                "sample_{:06d}.npz".format(subset_global_idx)
            )
            json_path = osp.join(
                args.output, "samples",
                "sample_{:06d}.json".format(subset_global_idx)
            )

            npz_kwargs = {
                "pred_trace_abs": np.ascontiguousarray(pred_trace_abs[i], dtype=np.float32),
                "gt_trace_abs": np.ascontiguousarray(gt_trace_abs[i], dtype=np.float32),
                "pred_trace_se3": np.ascontiguousarray(pred_trace_se3[i], dtype=np.float32),
                "gt_trace_se3": np.ascontiguousarray(gt_trace_se3[i], dtype=np.float32),
                "trace_valid_mask_per_coord": np.ascontiguousarray(mask_pc_np[i]),
                "valid_steps": np.int64(batch["valid_steps"][i].item()),
                "movement_bool": movement_arr,
                "gt_mask": gt_mask_arr,
                "first_keypoint": np.ascontiguousarray(
                    batch["extra"]["first_keypoint"][i].cpu().numpy(), dtype=np.float32
                ),
                "image_uint8": img_uint8,
                "depth_raw": depth_raw_np,
                "kept_by_filter": np.bool_(kept),
            }
            sample_meta = {
                "episode_id": episode_id,
                "frame_id": frame_id,
                "text": text_i,
                "env_name": args.env,
                "seed_used": args.seed,
                "kept_by_filter": kept,
                "valid_steps_int": int(batch["valid_steps"][i].item()),
                "movement_count": int(movement_arr.sum()),
                "gt_mask_count": int(gt_mask_arr.sum()),
                "subset_global_idx": subset_global_idx,
                "has_depth": has_depth,
            }
            # Strict no-skip: re-raise any prior save-thread crash before
            # enqueueing the next item.
            _check_save_error()
            _SAVE_QUEUE.put({
                "npz_path": npz_path,
                "npz_kwargs": npz_kwargs,
                "json_path": json_path,
                "json_payload": sample_meta,
            })

            # Accumulate in-memory metric tensors (kept only)
            if kept:
                pred_chunks.append(
                    torch.from_numpy(pred_trace_abs[i]).float()
                )  # (33, 400, 3)
                gt_chunks.append(
                    torch.from_numpy(gt_trace_abs[i]).float()
                )  # (33, 400, 3)
                mask_chunks.append(
                    torch.from_numpy(mask_pc_np[i]).bool()
                )  # (33, 400, 3)
                n_samples_valid += 1

                # Debug MP4s for first 2 kept samples — leader only.
                if is_leader and n_debug_renders_done < 2:
                    _render_debug_mp4s(
                        dst=args.output,
                        subset_global_idx=subset_global_idx,
                        image_uint8_chw=img_uint8,
                        gt_trace_se3_tpkk=gt_trace_se3[i],
                        pred_trace_se3_tpkk=pred_trace_se3[i],
                        mask_pc_tpc=mask_pc_np[i],
                        first_kp_p2=batch["extra"]["first_keypoint"][i].cpu().numpy().astype(np.float32),
                        gt_mask_p=gt_mask_arr,
                        depth_raw_np=(
                            depth_raw_np if has_depth else None
                        ),
                    )
                    n_debug_renders_done += 1

            subset_global_idx += 1

        if (batch_i + 1) % 10 == 0:
            log.info(
                "Progress: batch %d/%d  samples_done=%d  kept=%d",
                batch_i + 1,
                len(loader),
                subset_global_idx,
                n_samples_valid,
            )

    log.info(
        "Inference loop done: subset_global_idx=%d  n_samples_valid=%d",
        subset_global_idx,
        n_samples_valid,
    )

    # ── 12.5 Drain save-thread BEFORE the integrity gate ────────────────────
    log.info("Draining save-thread queue (size now=%d) ...", _SAVE_QUEUE.qsize())
    t_drain = time.time()
    _SAVE_QUEUE.put(None)              # sentinel → worker exits cleanly
    _SAVE_QUEUE.join()                 # wait for all enqueued items to finish
    save_thread.join(timeout=60.0)
    if save_thread.is_alive():
        raise RuntimeError(
            "save-thread did not exit within 60s after sentinel — strict no-skip"
        )
    _check_save_error()                # re-raise any captured exception
    log.info("Save-thread drained in %.1fs", time.time() - t_drain)

    # ── 12.6 Per-shard DONE sentinel — every shard writes one ───────────────
    # Records this shard's contribution so the leader can verify all shards
    # finished and aggregate counts. Written AFTER save-thread drain so a
    # missing .done file means the shard truly didn't finish saving.
    if is_multi:
        shard_done = osp.join(
            args.output, "_shards", "shard_{:03d}.done".format(args.shard_id)
        )
        with open(shard_done, "w") as f:
            json.dump({
                "shard_id": args.shard_id,
                "n_processed": int(subset_global_idx - my_global_idx_offset),
                "n_kept": int(n_samples_valid),
                "global_range": [my_start, my_end],
            }, f, indent=2)
        log.info("Wrote shard done sentinel: %s", shard_done)

    # ── 12.7 Workers exit here. Leader continues to integrity + metrics ────
    if not is_leader:
        wall_clock_s = time.time() - t_total_start
        log.info(
            "Worker shard %d finished in %.1fs. n_processed=%d n_kept=%d. Exiting.",
            args.shard_id, wall_clock_s,
            subset_global_idx - my_global_idx_offset, n_samples_valid,
        )
        return

    # Leader: wait for all worker `.done` sentinels (timeout 12 h).
    if is_multi:
        log.info("Leader waiting for all %d shard .done sentinels ...",
                 args.num_shards)
        deadline = time.time() + 12 * 3600
        expected = {"shard_{:03d}.done".format(i) for i in range(args.num_shards)}
        while True:
            present = set(os.listdir(osp.join(args.output, "_shards")))
            missing = expected - present
            if not missing:
                break
            if time.time() > deadline:
                raise RuntimeError(
                    "Leader timed out (12h) waiting for shards: missing={}"
                    .format(sorted(missing))
                )
            time.sleep(5)
        log.info("All %d shards reported done.", args.num_shards)

    # ── 13. End-of-run integrity gate ────────────────────────────────────────
    _run_integrity_gate(args.output, n_subset)

    # ── 14. In-memory metric (only for single-shard runs) ───────────────────
    # In multi-shard runs the leader's in-mem accumulator only covers its own
    # slice, not the full subset. The disk path is the source of truth.
    if is_multi:
        log.info(
            "Multi-shard run (num_shards=%d): in-mem metric skipped; disk metric only.",
            args.num_shards,
        )
        metrics_inmem = None  # type: ignore[assignment]
    elif len(pred_chunks) == 0:
        log.warning("No kept-by-filter samples; metrics will be NaN.")
        metrics_inmem: Dict[str, float] = {
            "trajectory_mse_x100": float("nan"),
            "trajectory_mae_x100": float("nan"),
            "endpoint_mse_x100": float("nan"),
        }
    else:
        pred_cat = torch.stack(pred_chunks, dim=0)   # (n_valid, 33, 400, 3)
        gt_cat = torch.stack(gt_chunks, dim=0)       # (n_valid, 33, 400, 3)
        mask_cat = torch.stack(mask_chunks, dim=0)   # (n_valid, 33, 400, 3)

        B_eff, T_eff, P_eff, C_eff = pred_cat.shape
        assert T_eff == 33 and P_eff == 400 and C_eff == 3, (
            "Unexpected shape: {}".format(pred_cat.shape)
        )
        pred_flat = pred_cat.permute(0, 2, 1, 3).reshape(B_eff * P_eff, T_eff, C_eff)
        gt_flat = gt_cat.permute(0, 2, 1, 3).reshape(B_eff * P_eff, T_eff, C_eff)
        mask_flat = mask_cat.permute(0, 2, 1, 3).reshape(B_eff * P_eff, T_eff, C_eff)

        raw = _compute_trajectory_metrics_abs(pred_flat, gt_flat, mask_flat)
        if not raw:
            metrics_inmem = {
                "trajectory_mse_x100": float("nan"),
                "trajectory_mae_x100": float("nan"),
                "endpoint_mse_x100": float("nan"),
            }
        else:
            metrics_inmem = {
                "trajectory_mse_x100": float(raw["trajectory_mse"] * 100.0),
                "trajectory_mae_x100": float(raw["trajectory_mae"] * 100.0),
                "endpoint_mse_x100": float(raw["endpoint_mse"] * 100.0),
            }

    # ── 15. Auto-call disk metrics script ────────────────────────────────────
    metrics_script_path = osp.join(
        osp.dirname(osp.abspath(__file__)), "eval_compute_metrics_wmm.py"
    )
    log.info("Calling disk metrics script: %s %s", metrics_script_path, args.output)
    subprocess.run(
        [sys.executable, metrics_script_path, args.output],
        check=True,
    )

    # ── 16. Read disk metrics ────────────────────────────────────────────────
    with open(osp.join(args.output, "metrics.json")) as f:
        metrics_disk = json.load(f)

    # ── 17. Bit-identity assertion (single-shard only) ──────────────────────
    # metrics.json is written RFC-8259-compliant (NaN -> null). Translate
    # `None` from the JSON back to float('nan') for the comparison.
    def _disk_to_float(v: Any) -> float:
        if v is None:
            return float("nan")
        return float(v)

    if not is_multi:
        for key in ("trajectory_mse_x100", "trajectory_mae_x100", "endpoint_mse_x100"):
            v_inmem = metrics_inmem[key]  # type: ignore[index]
            v_disk = _disk_to_float(metrics_disk.get(key))
            # Both NaN case is acceptable
            if (
                isinstance(v_inmem, float) and isinstance(v_disk, float)
                and (v_inmem != v_inmem) and (v_disk != v_disk)  # NaN check
            ):
                continue
            diff = abs(v_inmem - v_disk)
            assert diff <= 1e-6, (
                "in-mem vs disk metric mismatch on {}: inmem={} disk={} diff={}".format(
                    key, v_inmem, v_disk, diff
                )
            )
        log.info("Bit-identity assertion PASSED (atol=1e-6) for all 3 metrics.")

    # ── 18. Print summary ────────────────────────────────────────────────────
    wall_clock_s = time.time() - t_total_start
    # Use disk metrics for summary (always available, valid in single + multi).
    summary_mse = _disk_to_float(metrics_disk.get("trajectory_mse_x100"))
    summary_mae = _disk_to_float(metrics_disk.get("trajectory_mae_x100"))
    summary_ep = _disk_to_float(metrics_disk.get("endpoint_mse_x100"))
    summary_n_eval = int(metrics_disk.get("n_samples_evaluated", n_subset))
    summary_n_valid = int(metrics_disk.get("n_samples_valid", 0))
    integrity_status = (
        "inmem==disk: PASS (atol=1e-6)" if not is_multi else "multi-shard: disk-only"
    )
    print(
        "\n=== TRACEGEN EVAL — env={env} ckpt={ckpt} ===\n"
        "  num_shards={ns}\n"
        "  n_samples_evaluated={n_eval}  n_samples_valid={n_valid}\n"
        "  trajectory_mse_x100={mse:.6f}  trajectory_mae_x100={mae:.6f}"
        "  endpoint_mse_x100={ep:.6f}\n"
        "  wall_clock_s={wc:.1f}  {integ}".format(
            env=args.env,
            ckpt=osp.basename(args.checkpoint),
            ns=args.num_shards,
            n_eval=summary_n_eval,
            n_valid=summary_n_valid,
            mse=summary_mse,
            mae=summary_mae,
            ep=summary_ep,
            wc=wall_clock_s,
            integ=integrity_status,
        )
    )


if __name__ == "__main__":
    main()
