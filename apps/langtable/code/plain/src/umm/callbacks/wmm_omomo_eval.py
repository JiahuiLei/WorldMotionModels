"""On-the-fly CHOIS-Table-1 eval callback for wmm/OMOMO training.

Fires every ``val_every_n_iter`` gradient steps.  Two branches:

* **Debug** (``val_every_n_iter < debug_threshold``):
  Load 8 GT npz files, run K=1 / n_steps=10 inference, Stage A only.
  Designed to complete in <60 s and verify the sampling pipeline.

* **Production** (``val_every_n_iter >= debug_threshold``):
  Load all 482 GT test files, run K=20 / ``scheduler_mode="ddpm_full"``
  inference (all ``num_train_timesteps`` DDPM steps), Stage A + Stage B +
  build_ours_row.  Auto-searches max batch-size starting at ``max_bs_start``.

  **Note on step count**: ``"ddpm_full"`` runs exactly ``num_train_timesteps``
  DDPM steps (set in training config), which may differ from the CHOIS paper
  protocol (1000 steps, model trained with 1000 timesteps).  Provenance records
  the actual step count under ``"n_steps"`` for auditability.

Metrics are written atomically as
``<log_dir>/eval_metrics_step_<step>.json`` (19-key schema).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from umm.datasets.wmm.mask_scheduler import chois_obj_waypoints_to_all

_LOG = logging.getLogger(__name__)

# Number of body joints (pelvis + 21 body) — matches OmomoDataset
_NUM_BODY_JOINTS = 22
# Fixed window length (OMOMO CHOIS protocol)
_T_FIXED = 120
# Number of part-entities per sample (22 body + 1 object)
_P = 23
# Number of diverse samples in production eval (CHOIS-Table-1 protocol)
_K_PRODUCTION = 20


def _ensure_eval_in_path() -> None:
    """Add repo root to sys.path so ``eval.omomo`` is importable regardless of CWD.

    Resolves from __file__: src/umm/callbacks/wmm_omomo_eval.py -> parents[3] = repo root.
    Safe to call multiple times (idempotent).
    """
    import sys as _sys
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Tensor helpers (replicate dataset SE3-building logic without importing it)
# ---------------------------------------------------------------------------

def _aa_to_mat(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues axis-angle -> (*, 3, 3) rotation matrix."""
    eps = 1.0e-6
    angle = aa.norm(dim=-1, keepdim=True)
    is_small = angle < eps
    angle_safe = torch.where(is_small, torch.ones_like(angle), angle)
    axis = aa / angle_safe
    ax, ay, az = axis.unbind(dim=-1)
    zero = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zero, -az, ay], dim=-1),
            torch.stack([az, zero, -ax], dim=-1),
            torch.stack([-ay, ax, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    sin_a = angle.sin().unsqueeze(-1)
    cos_a = angle.cos().unsqueeze(-1)
    R = eye + sin_a * K + (1 - cos_a) * (K @ K)
    return torch.where(is_small.unsqueeze(-1), eye, R)


def _build_body_se3(
    root_trans: np.ndarray,      # (T, 3)
    root_orient_aa: np.ndarray,  # (T, 3)
    pose_body_aa: np.ndarray,    # (T, 21, 3)
) -> torch.Tensor:
    """Return (22, T, 4, 4) body SE3 tensor (pelvis world-frame, rest local)."""
    root_t = torch.as_tensor(root_trans, dtype=torch.float32)
    root_aa = torch.as_tensor(root_orient_aa, dtype=torch.float32)
    body_aa = torch.as_tensor(pose_body_aa, dtype=torch.float32)
    T = root_t.shape[0]
    all_aa = torch.cat([root_aa.unsqueeze(1), body_aa], dim=1)  # (T,22,3)
    rot = _aa_to_mat(all_aa)                                     # (T,22,3,3)
    Q = torch.eye(4, dtype=torch.float32).expand(T, 22, 4, 4).contiguous()
    Q[..., :3, :3] = rot
    Q[:, 0, :3, 3] = root_t
    return Q.permute(1, 0, 2, 3).contiguous()  # (22, T, 4, 4)


def _build_object_se3(
    obj_rot: np.ndarray,    # (T, 3, 3)
    obj_trans: np.ndarray,  # (T, 3)
) -> torch.Tensor:
    """Return (1, T, 4, 4) object SE3 tensor."""
    rot_t = torch.as_tensor(obj_rot, dtype=torch.float32)
    trans_t = torch.as_tensor(obj_trans, dtype=torch.float32)
    T = rot_t.shape[0]
    Q = torch.eye(4, dtype=torch.float32).expand(T, 4, 4).contiguous()
    Q[:, :3, :3] = rot_t
    Q[:, :3, 3] = trans_t
    return Q.unsqueeze(0)  # (1, T, 4, 4)


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------

def _load_gt_npz(path: str) -> Dict[str, Any]:
    """Load and validate a single GT npz file, returning a plain dict."""
    with np.load(path, allow_pickle=True) as d:
        root_trans = d["root_trans"].astype(np.float32)         # (T, 3)
        root_orient_aa = d["root_orient_aa"].astype(np.float32) # (T, 3)
        pose_body_aa = d["pose_body_aa"].astype(np.float32)     # (T, 21, 3)
        if "obj_rot" in d.files:
            obj_rot = d["obj_rot"].astype(np.float32)
        elif "obj_rot_mat" in d.files:
            obj_rot = d["obj_rot_mat"].astype(np.float32)
        else:
            raise KeyError(f"npz {path} has neither 'obj_rot' nor 'obj_rot_mat'")
        obj_trans = d["obj_trans"].astype(np.float32)
        obj_mesh_path = str(d["obj_mesh_path"].item())
        seq_name = str(d["seq_name"].item()) if "seq_name" in d.files else Path(path).stem
        betas = d["betas"].astype(np.float32) if "betas" in d.files else np.zeros(16, dtype=np.float32)
        gender = str(d["gender"].item()) if "gender" in d.files else "neutral"
        # obj_scale may be per-frame (T,) or scalar () — take first value
        if "obj_scale" in d.files:
            _os = np.asarray(d["obj_scale"]).flat[0]
            obj_scale = float(_os)
        else:
            obj_scale = 1.0
        actual_len = int(d["actual_len"].item()) if "actual_len" in d.files else int(root_trans.shape[0])
        window_idx = int(d["window_idx"].item()) if "window_idx" in d.files else 0
        text = str(d["text"].item()) if "text" in d.files else ""

    return dict(
        root_trans=root_trans,
        root_orient_aa=root_orient_aa,
        pose_body_aa=pose_body_aa,
        obj_rot=obj_rot,
        obj_trans=obj_trans,
        obj_mesh_path=obj_mesh_path,
        seq_name=seq_name,
        betas=betas,
        gender=gender,
        obj_scale=obj_scale,
        actual_len=actual_len,
        window_idx=window_idx,
        text=text,
    )


def _save_pred_npz(
    preds_dir: str,
    stem: str,
    ret_trace_b: torch.Tensor,  # (T, P, 4, 4) on CPU
    gt: Dict[str, Any],
) -> None:
    """Write one prediction npz matching the Q12 schema expected by run_eval.

    Args:
        preds_dir: Output directory.
        stem: Filename stem (no extension).
        ret_trace_b: (T, P, 4, 4) SE3 output for one batch item (CPU).
        gt: GT dict from ``_load_gt_npz`` (provides metadata fields).
    """
    from pytorch3d.transforms import matrix_to_axis_angle  # local import

    root_trans = ret_trace_b[:, 0, :3, 3].numpy()                 # (T, 3)
    root_orient_aa = matrix_to_axis_angle(
        ret_trace_b[:, 0, :3, :3]
    ).numpy()                                                       # (T, 3)
    pose_body_aa = matrix_to_axis_angle(
        ret_trace_b[:, 1:22, :3, :3]
    ).numpy()                                                       # (T, 21, 3)
    pred_obj_trans = ret_trace_b[:, 22, :3, 3].numpy()            # (T, 3)
    pred_obj_rot_mat = ret_trace_b[:, 22, :3, :3].numpy()         # (T, 3, 3)

    save_kwargs: Dict[str, Any] = dict(
        root_trans=root_trans,
        root_orient_aa=root_orient_aa,
        pose_body_aa=pose_body_aa,
        pred_obj_trans=pred_obj_trans,
        pred_obj_rot_mat=pred_obj_rot_mat,
        # Copy metadata from GT
        betas=gt["betas"],
        gender=gt["gender"],
        seq_name=gt["seq_name"],
        obj_mesh_path=gt["obj_mesh_path"],
        obj_rot_mat=gt["obj_rot"],   # GT obj rot (full traj)
        obj_trans=gt["obj_trans"],
        obj_scale=gt["obj_scale"],
        # Required by preprocess_chois_preds.py (Stage B); silently skips without these.
        actual_len=np.array(gt["actual_len"], dtype=np.int64),
        window_idx=np.array(gt["window_idx"], dtype=np.int64),
    )

    out_path = os.path.join(preds_dir, f"{stem}.npz")
    np.savez(out_path, **save_kwargs)


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def _build_batch_from_gt_files(
    gt_paths: List[str],
    device: torch.device,
    obj_geo_dir: str,
    num_pts: int = 256,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build a collated batch dict (matching CanonicalDatasetOutput.collate output)
    from a list of GT npz files.

    Returns:
        Tuple of (batch_dict, gt_list) where gt_list has one entry per path.

    Args:
        gt_paths: Absolute paths to GT npz files.
        device: Target torch device.
        obj_geo_dir: Directory containing ``<category>.ply`` files.
        num_pts: PCL points per object.
    """
    # Lazy import to avoid import-at-load-time issues
    from umm.datasets.wmm.dataset_config import (
        SEM_HUMAN,
        SEM_HUMAN_GLOBAL,
        SEM_OBJECT,
        UID_HUMOTO_OBJECT_GROUP,
        UID_SMPLX55_GROUP,
    )
    from umm.datasets.wmm.omomo_dataset import (
        _obj_category_from_mesh_path,
        _ply_to_pcl,
    )

    T = _T_FIXED
    P = _P
    C = 1  # context size = 1 (identity)

    gt_list: List[Dict[str, Any]] = []
    traces: List[torch.Tensor] = []         # each (P, T, 4, 4)
    trace_masks: List[torch.Tensor] = []    # each (P, T) bool
    contexts: List[torch.Tensor] = []       # each (P, C, 4, 4)
    context_masks: List[torch.Tensor] = []  # each (P, C) bool
    known_masks: List[torch.Tensor] = []    # each (P, T) bool — all False (generate all)
    ctx_known_masks: List[torch.Tensor] = []
    group_ids: List[torch.Tensor] = []
    rot_masks: List[torch.Tensor] = []
    sem_types: List[torch.Tensor] = []
    uid_list: List[torch.Tensor] = []
    times: List[torch.Tensor] = []
    texts: List[str] = []
    pcl_list: List[torch.Tensor] = []      # each (P, num_pts, 6)
    pcl_mask_list: List[torch.Tensor] = []  # each (P,) bool

    pcl_cache: Dict[str, np.ndarray] = {}

    for path in gt_paths:
        gt = _load_gt_npz(path)
        gt_list.append(gt)

        actual_len = min(gt["actual_len"], T)

        # Pad to T if needed (replace padded frames with identity/zeros)
        root_trans = gt["root_trans"].copy()
        root_orient_aa = gt["root_orient_aa"].copy()
        pose_body_aa = gt["pose_body_aa"].copy()
        obj_rot = gt["obj_rot"].copy()
        obj_trans = gt["obj_trans"].copy()
        if actual_len < T:
            root_trans[actual_len:] = 0.0
            root_orient_aa[actual_len:] = 0.0
            pose_body_aa[actual_len:] = 0.0
            obj_rot[actual_len:] = np.eye(3, dtype=obj_rot.dtype)
            obj_trans[actual_len:] = 0.0

        # Build SE3
        body_se3 = _build_body_se3(root_trans, root_orient_aa, pose_body_aa)  # (22,T,4,4)
        obj_se3 = _build_object_se3(obj_rot, obj_trans)                        # (1,T,4,4)
        trace_pt = torch.cat([body_se3, obj_se3], dim=0)                       # (23,T,4,4)

        # trace_mask
        body_mask = torch.zeros(_NUM_BODY_JOINTS, T, dtype=torch.bool)
        body_mask[:, :actual_len] = True
        obj_mask = torch.zeros(1, T, dtype=torch.bool)
        obj_mask[:, :actual_len] = True
        trace_mask_pt = torch.cat([body_mask, obj_mask], dim=0)                # (23,T)

        # context: single identity frame
        ctx_pt = torch.eye(4, dtype=torch.float32).expand(P, C, 4, 4).contiguous()
        ctx_mask_pt = torch.ones(P, C, dtype=torch.bool)

        # semantic types and group ids
        sem = np.full((P,), SEM_HUMAN, dtype=np.int64)
        sem[0] = SEM_HUMAN_GLOBAL
        sem[22] = SEM_OBJECT
        grp = np.zeros(P, dtype=np.int64)
        grp[22] = 1  # object is group 1

        # Strict CHOIS Table-1 protocol: object known at [0, 30, 60, 90, 119], body all unknown, ctx all known.
        # Mirrors exactly the dominant training pattern (chois_obj_waypoints_to_all, weight 8.0 in OMOMO_PATTERNS).
        sem_tensor = torch.as_tensor(sem, dtype=torch.int32)
        known_pt, ctx_known_pt = chois_obj_waypoints_to_all(
            trace_mask=trace_mask_pt,
            context_mask=ctx_mask_pt,
            sem_type=sem_tensor,
            waypoint_indices=[0, 30, 60, 90, 119],  # CHOIS protocol (confirmed from Q12 npz inspection)
        )

        body_uids = np.asarray(UID_SMPLX55_GROUP[:_NUM_BODY_JOINTS], dtype=np.int64)
        obj_uid = np.asarray([UID_HUMOTO_OBJECT_GROUP[0]], dtype=np.int64)
        uid = np.concatenate([body_uids, obj_uid], axis=0)

        # rot_mask: all entities have rotation
        rot_mask_pt = torch.ones(P, dtype=torch.bool)

        # time stamps
        base_fps = 30
        t_rate = 1
        time_unit = float(t_rate) / float(base_fps)
        time_stamp = torch.arange(T, dtype=torch.float32) * time_unit

        # Local PCL (only object = slot 22 has real PCL; body slots = zeros)
        obj_category = _obj_category_from_mesh_path(gt["obj_mesh_path"])
        if obj_category not in pcl_cache:
            ply_path = os.path.join(obj_geo_dir, f"{obj_category}.ply")
            pcl_cache[obj_category] = _ply_to_pcl(ply_path)
        pcl_full = pcl_cache[obj_category]
        # Subsample deterministically for eval
        rng = np.random.default_rng(seed=0)
        if pcl_full.shape[0] >= num_pts:
            idx = rng.choice(pcl_full.shape[0], num_pts, replace=False)
        else:
            idx = rng.choice(pcl_full.shape[0], num_pts, replace=True)
        pcl_obj = torch.as_tensor(pcl_full[idx], dtype=torch.float32)  # (num_pts, 6)

        # Build (P, num_pts, 6): body joints get zero PCL
        pcl_all = torch.zeros(P, num_pts, 6, dtype=torch.float32)
        pcl_all[22] = pcl_obj
        pcl_mask_pt = torch.zeros(P, dtype=torch.bool)
        pcl_mask_pt[22] = True

        # text
        texts.append(gt.get("text", "") or "")

        traces.append(trace_pt)
        trace_masks.append(trace_mask_pt)
        contexts.append(ctx_pt)
        context_masks.append(ctx_mask_pt)
        known_masks.append(known_pt)
        ctx_known_masks.append(ctx_known_pt)
        group_ids.append(torch.as_tensor(grp, dtype=torch.int32))
        rot_masks.append(rot_mask_pt)
        sem_types.append(torch.as_tensor(sem, dtype=torch.int32))
        uid_list.append(torch.as_tensor(uid, dtype=torch.int32))
        times.append(time_stamp)
        pcl_list.append(pcl_all)
        pcl_mask_list.append(pcl_mask_pt)

    # Collate: permute (B, P, T, ...) -> (B, T, P, ...) matching CanonicalDatasetOutput.collate
    batch: Dict[str, Any] = {
        "trace":           torch.stack(traces, 0).float().permute(0, 2, 1, 3, 4).to(device),
        "trace_mask":      torch.stack(trace_masks, 0).bool().permute(0, 2, 1).to(device),
        "context":         torch.stack(contexts, 0).float().permute(0, 2, 1, 3, 4).to(device),
        "context_mask":    torch.stack(context_masks, 0).bool().permute(0, 2, 1).to(device),
        "known_mask":      torch.stack(known_masks, 0).bool().permute(0, 2, 1).to(device),
        "context_known_mask": torch.stack(ctx_known_masks, 0).bool().permute(0, 2, 1).to(device),
        "trace_group_id":  torch.stack(group_ids, 0).int().to(device),
        "trace_rot_mask":  torch.stack(rot_masks, 0).bool().to(device),
        "trace_sem_type":  torch.stack(sem_types, 0).int().to(device),
        "trace_uid":       torch.stack(uid_list, 0).int().to(device),
        "time":            torch.stack(times, 0).float().to(device),
        "text":            texts,
        "local_pcl":       torch.stack(pcl_list, 0).float().to(device),
        "local_pcl_mask":  torch.stack(pcl_mask_list, 0).bool().to(device),
    }
    return batch, gt_list


# ---------------------------------------------------------------------------
# Main callback class
# ---------------------------------------------------------------------------


class WmmOmomoEvalCallback(pl.Callback):
    """On-the-fly CHOIS-Table-1 eval callback for wmm/OMOMO training.

    Fires at ``global_step % val_every_n_iter == 0 and global_step > 0``.
    Two branches based on ``val_every_n_iter`` vs ``debug_threshold``:

    * Debug (< threshold): 8 seqs, K=1, n_steps=10 (DDIM). Stage A only. <60 s.
    * Production (>= threshold): 482 seqs, K=20, scheduler_mode="ddpm_full"
      (all num_train_timesteps DDPM steps — not necessarily 1000; provenance
      logs the actual count).  Stage A + B + build_ours_row.  Auto-searches
      max batch size.

    Args:
        gt_dir: Directory with 482 GT test npz files.
        smplx_dir: SMPLX model directory.
        sdf_folder: SDF directory for run_eval.
        obj_geo_dir: Directory containing ``<category>.ply`` rest meshes.
        classifier_ckpt: Path to ``finest.tar`` for Stage B.
        val_every_n_iter: Steps between eval runs.
        log_dir: Output directory for ``eval_metrics_step_*.json``.
        gpu: CUDA GPU index (for Stage B subprocess ``CUDA_VISIBLE_DEVICES``).
        debug_threshold: ``val_every_n_iter < debug_threshold`` -> debug branch.
        max_bs_start: Starting candidate for batch-size autosearch.
        bs_autosearch_log: Optional path to log BS autosearch progression.
        max_budget_s: Maximum wall time (seconds) for eval; log warning if exceeded.
        budget_md_path: Optional path to write a Markdown budget summary.
    """

    def __init__(
        self,
        gt_dir: str,
        smplx_dir: str,
        sdf_folder: str,
        obj_geo_dir: str,
        classifier_ckpt: str,
        val_every_n_iter: int,
        log_dir: str,
        gpu: int = 0,
        debug_threshold: int = 100,
        max_bs_start: int = 8,
        bs_autosearch_log: Optional[str] = None,
        max_budget_s: float = 2100.0,
        budget_md_path: Optional[str] = None,
        eval_sample_ratio: float = 1.0,
    ):
        super().__init__()
        self._gt_dir = str(Path(gt_dir).resolve())
        self._smplx_dir = str(Path(smplx_dir).resolve())
        self._sdf_folder = str(Path(sdf_folder).resolve())
        self._obj_geo_dir = str(Path(obj_geo_dir).resolve())
        self._classifier_ckpt = str(Path(classifier_ckpt).resolve())
        self._val_every_n_iter = int(val_every_n_iter)
        self._log_dir = str(Path(log_dir).resolve())
        self._gpu = gpu
        self._debug_threshold = int(debug_threshold)
        self._max_bs_start = int(max_bs_start)
        self._bs_autosearch_log = bs_autosearch_log
        self._max_budget_s = float(max_budget_s)
        self._budget_md_path = budget_md_path
        self._eval_sample_ratio = float(eval_sample_ratio)

        # Cached list of GT npz paths (sorted, 482 for full eval)
        self._gt_paths: Optional[List[str]] = None
        # Last successful autosearched BS (persists across eval calls)
        self._cached_max_bs: Optional[int] = None

        Path(self._log_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Fire eval if ``global_step % val_every_n_iter == 0``."""
        step = trainer.global_step
        if step == 0 or (step % self._val_every_n_iter) != 0:
            return

        _LOG.info("WmmOmomoEvalCallback: starting eval at step=%d", step)
        t_wall_start = time.time()

        try:
            self._run_eval(pl_module, step)
        except Exception as exc:  # pylint: disable=broad-except
            # Write failure marker; do NOT crash training.
            fail_path = os.path.join(self._log_dir, f"metrics_FAILED_{step}.txt")
            import traceback as _tb
            with open(fail_path, "w") as fh:
                fh.write(f"step={step}\nhost={socket.gethostname()}\n\n")
                _tb.print_exc(file=fh)
            _LOG.error(
                "WmmOmomoEvalCallback: eval at step=%d FAILED — see %s. Error: %s",
                step, fail_path, exc, exc_info=True,
            )
            return

        wall = time.time() - t_wall_start
        if wall > self._max_budget_s:
            _LOG.warning(
                "WmmOmomoEvalCallback: eval at step=%d took %.1f s > budget %.1f s",
                step, wall, self._max_budget_s,
            )

    # ------------------------------------------------------------------
    # Core eval dispatcher
    # ------------------------------------------------------------------

    def _run_eval(self, pl_module: pl.LightningModule, step: int) -> None:
        """Dispatch to debug or production eval branch and write metrics.json."""
        is_debug = self._val_every_n_iter < self._debug_threshold
        if is_debug:
            metrics = self._run_debug_eval(pl_module, step)
        else:
            metrics = self._run_production_eval(pl_module, step)

        # Log scalar metrics to wandb under omomo_eval/<metric> namespace.
        # Skip NaN (Stage B failures), non-numeric (provenance dict), and nested dicts.
        scalar_metrics = {}
        for k, v in metrics.items():
            if k in ("paper_chois_ours", "verdicts", "provenance"):
                continue
            if isinstance(v, (int, float)) and not (math.isnan(v) if isinstance(v, float) else False):
                scalar_metrics[f"omomo_eval/{k}"] = float(v)
        # Also log wall_time from provenance
        if "provenance" in metrics and isinstance(metrics["provenance"].get("wall_time_s"), (int, float)):
            scalar_metrics["omomo_eval/wall_time_s"] = float(metrics["provenance"]["wall_time_s"])

        if scalar_metrics:
            try:
                pl_module.log_dict(
                    scalar_metrics,
                    on_step=True, on_epoch=False,
                    sync_dist=False, rank_zero_only=True,
                    batch_size=1,  # arbitrary; required by PL logger API
                )
                _LOG.info(
                    "WmmOmomoEvalCallback: logged %d scalars to wandb at step=%d",
                    len(scalar_metrics), step,
                )
            except Exception as exc:
                _LOG.warning(
                    "WmmOmomoEvalCallback: wandb log failed (non-fatal): %s", exc,
                )

        # Atomic write: tmp dir on SAME filesystem as log_dir (avoids cross-fs move on NFS).
        tmp_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_eval_{step}_", dir=self._log_dir)
        tmp_json = os.path.join(tmp_dir, "metrics.json")
        try:
            with open(tmp_json, "w") as fh:
                json.dump(metrics, fh, indent=2, allow_nan=False)
            final_path = os.path.join(
                self._log_dir, f"eval_metrics_step_{step:08d}.json"
            )
            shutil.move(tmp_json, final_path)
            _LOG.info("WmmOmomoEvalCallback: metrics written -> %s", final_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Debug branch
    # ------------------------------------------------------------------

    def _run_debug_eval(
        self, pl_module: pl.LightningModule, step: int
    ) -> Dict[str, Any]:
        """Debug eval: 8 seqs, K=1, n_steps=10, Stage A only."""
        _LOG.info("WmmOmomoEvalCallback: debug branch (8 seqs, K=1, n_steps=10)")
        t0 = time.time()

        gt_paths = self._get_gt_paths()[:8]
        device = next(pl_module.parameters()).device

        preds_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_preds_debug_{step}_")
        stage_a_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_stagea_debug_{step}_")
        # Subset GT dir — run_eval requires exact stem parity between gt and pred.
        # When debug uses fewer than all 482 seqs, symlink only the evaluated ones.
        tmp_gt_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_gt_debug_{step}_")
        try:
            for gt_path in gt_paths:
                os.symlink(gt_path, os.path.join(tmp_gt_dir, Path(gt_path).name))

            # Build batch from GT files
            batch, gt_list = _build_batch_from_gt_files(
                gt_paths, device, self._obj_geo_dir
            )

            # Single inference pass (ddim_gen, 10 steps)
            t_sample_start = time.time()
            ret_trace = self._run_inference_batch(
                pl_module=pl_module,
                batch=batch,
                seed=0,
                n_steps=10,
                scheduler_mode="ddim_gen",
            )
            t_sample = time.time() - t_sample_start

            # Save predictions (k=0, the only sample)
            for b, gt_path in enumerate(gt_paths):
                stem = Path(gt_path).stem
                ret_trace_b = ret_trace[b].cpu()  # (T, P, 4, 4)
                _save_pred_npz(preds_dir, stem, ret_trace_b, gt_list[b])

            # Render a few eval-prediction MP4s to wandb for visual inspection.
            # Wrapped in try/except — eval must never be killed by viz failures.
            # Debug branch: _save_pred_npz produces <stem>.npz (K=1, no suffix).
            try:
                from umm.models.wmm.viz_helpers.omomo_viz import omomo_viz_eval_preds
                pred_paths = [os.path.join(preds_dir, f"{Path(p).stem}.npz") for p in gt_paths[:3]]
                pred_paths = [p for p in pred_paths if os.path.exists(p)]
                if pred_paths:
                    omomo_viz_eval_preds(
                        pl_module=pl_module,
                        pred_npz_paths=pred_paths,
                        gt_list=gt_list[:len(pred_paths)],
                        step=step,
                        phase="eval",
                        n_samples=3,
                        smplx_dir=self._smplx_dir,
                        geo_dir=self._obj_geo_dir,
                    )
                    _LOG.info(
                        "WmmOmomoEvalCallback: logged %d eval-pred videos to wandb", len(pred_paths),
                    )
            except Exception as exc:
                _LOG.warning(
                    "WmmOmomoEvalCallback: eval-pred viz failed (non-fatal): %s", exc,
                )

            # Stage A -- pass the subset gt dir to avoid stem-parity errors
            t_stagea_start = time.time()
            stage_a_out_json = os.path.join(stage_a_dir, "metrics_raw.json")
            _ensure_eval_in_path()
            raw = self._run_stage_a(preds_dir, stage_a_out_json, device="cuda",
                                    gt_dir=tmp_gt_dir)
            t_stagea = time.time() - t_stagea_start

            agg = raw.get("aggregate", {})
            # Build partial ours row (no Stage B)
            _ensure_eval_in_path()
            from eval.omomo.eval_chois_table1 import (  # local import
                build_ours_row,
                COLUMN_ORDER,
                PAPER_CHOIS_OURS,
                _verdict,
            )
            ours = build_ours_row(agg, classifier=None)
            verdicts = {
                col: _verdict(col, ours[col], PAPER_CHOIS_OURS[col])
                for col in COLUMN_ORDER
            }

            def _nan_safe(v: Any) -> Any:
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            metrics: Dict[str, Any] = {
                col: _nan_safe(ours[col]) for col in COLUMN_ORDER
            }
            metrics["paper_chois_ours"] = dict(PAPER_CHOIS_OURS)
            metrics["verdicts"] = verdicts
            metrics["provenance"] = {
                "branch": "debug",
                "step": step,
                "n_seqs": len(gt_paths),
                "K": 1,
                "n_steps": 10,
                "scheduler_mode": "ddim_gen",
                "eval_sample_ratio": self._eval_sample_ratio,
                "eval_n_windows": len(gt_paths),
                "skip_fid": True,
                "wall_time_s": round(time.time() - t0, 2),
                "t_sample_s": round(t_sample, 2),
                "t_stagea_s": round(t_stagea, 2),
                "host": socket.gethostname(),
            }

        finally:
            shutil.rmtree(preds_dir, ignore_errors=True)
            shutil.rmtree(stage_a_dir, ignore_errors=True)
            shutil.rmtree(tmp_gt_dir, ignore_errors=True)

        return metrics

    # ------------------------------------------------------------------
    # Production branch
    # ------------------------------------------------------------------

    def _run_production_eval(
        self, pl_module: pl.LightningModule, step: int
    ) -> Dict[str, Any]:
        """Production eval: N seqs, K=20, scheduler_mode=ddpm_full (all train timesteps), Stage A + B."""
        _LOG.info(
            "WmmOmomoEvalCallback: production branch (K=%d, n_steps=full DDPM)",
            _K_PRODUCTION,
        )
        t0 = time.time()

        gt_paths = self._get_gt_paths()
        device = next(pl_module.parameters()).device
        N_test = len(gt_paths)

        # Stage A uses k=0 canonical predictions (one file per stem).
        preds_dir_a = tempfile.mkdtemp(prefix=f"wmm_omomo_preds_a_{step}_")
        # Stage B uses all K samples (files named <stem>_k<k>.npz).
        preds_dir_b = tempfile.mkdtemp(prefix=f"wmm_omomo_preds_b_{step}_")
        stage_a_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_stagea_prod_{step}_")
        # Subset GT dir -- run_eval requires exact stem parity. During eval subsets
        # (e.g., smoke test cb._gt_paths override), only the evaluated stems exist.
        tmp_gt_dir = tempfile.mkdtemp(prefix=f"wmm_omomo_gt_prod_{step}_")
        try:
            for gt_path in gt_paths:
                os.symlink(gt_path, os.path.join(tmp_gt_dir, Path(gt_path).name))

            # Find or reuse cached max_bs
            max_bs = self._autosearch_max_bs(pl_module, gt_paths[:min(1, N_test)], device, step)

            t_sample_start = time.time()
            # Run K=20 inference passes per chunk.
            # preds_dir_a: k=0 only as <stem>.npz (canonical for Stage A).
            # preds_dir_b: all k as <stem>_k<k>.npz (diversity for Stage B).
            # n_steps=None: let the DDPM scheduler use all num_train_timesteps steps
            # ("ddpm_full" mode). Passing n_steps=N would call set_timesteps(N) which
            # fails for models with num_train_timesteps < N.
            for chunk_start in range(0, N_test, max_bs):
                chunk_paths = gt_paths[chunk_start: chunk_start + max_bs]
                chunk_batch, chunk_gt = _build_batch_from_gt_files(
                    chunk_paths, device, self._obj_geo_dir
                )
                for k in range(_K_PRODUCTION):
                    ret_trace = self._run_inference_batch(
                        pl_module=pl_module,
                        batch=chunk_batch,
                        seed=k,
                        n_steps=None,
                        scheduler_mode="ddpm_full",
                    )
                    for b, gt_path in enumerate(chunk_paths):
                        stem = Path(gt_path).stem
                        ret_b = ret_trace[b].cpu()
                        # Stage B: all K samples as <stem>_k<k>.npz
                        _save_pred_npz(preds_dir_b, f"{stem}_k{k}", ret_b, chunk_gt[b])
                        # Stage A: only k=0 canonical as <stem>.npz
                        if k == 0:
                            _save_pred_npz(preds_dir_a, stem, ret_b, chunk_gt[b])
            t_sample = time.time() - t_sample_start

            # Render a few eval-prediction MP4s to wandb for visual inspection.
            # Wrapped in try/except — eval must never be killed by viz failures.
            # Production branch: preds_dir_a holds k=0 canonical as <stem>.npz.
            # gt_list built on-demand for the first few paths (production does not
            # keep a single gt_list across chunks).
            try:
                from umm.models.wmm.viz_helpers.omomo_viz import omomo_viz_eval_preds
                pred_paths = [os.path.join(preds_dir_a, f"{Path(p).stem}.npz") for p in gt_paths[:3]]
                pred_paths = [p for p in pred_paths if os.path.exists(p)]
                if pred_paths:
                    viz_gt_list = [_load_gt_npz(gt_paths[i]) for i in range(len(pred_paths))]
                    omomo_viz_eval_preds(
                        pl_module=pl_module,
                        pred_npz_paths=pred_paths,
                        gt_list=viz_gt_list,
                        step=step,
                        phase="eval",
                        n_samples=3,
                        smplx_dir=self._smplx_dir,
                        geo_dir=self._obj_geo_dir,
                    )
                    _LOG.info(
                        "WmmOmomoEvalCallback: logged %d eval-pred videos to wandb", len(pred_paths),
                    )
            except Exception as exc:
                _LOG.warning(
                    "WmmOmomoEvalCallback: eval-pred viz failed (non-fatal): %s", exc,
                )

            # Stage A -- pass subset gt dir for exact stem parity
            t_stagea_start = time.time()
            stage_a_out_json = os.path.join(stage_a_dir, "metrics_raw.json")
            _ensure_eval_in_path()
            raw = self._run_stage_a(preds_dir_a, stage_a_out_json, device="cuda",
                                    gt_dir=tmp_gt_dir)
            t_stagea = time.time() - t_stagea_start

            # Stage B (FID + R_prec) -- uses all K diverse samples
            t_stageb_start = time.time()
            try:
                _ensure_eval_in_path()
                from eval.omomo.eval_chois_table1 import run_stage_b  # local import
                tag = f"step{step:08d}"
                classifier = run_stage_b(
                    preds_dir=preds_dir_b,
                    classifier_ckpt=self._classifier_ckpt,
                    out_dir=stage_a_dir,
                    gpu=self._gpu,
                    tag=tag,
                )
            except Exception as exc:  # pylint: disable=broad-except
                _LOG.warning(
                    "WmmOmomoEvalCallback: Stage B failed: %s -- R_prec/FID = NaN", exc
                )
                classifier = None
            t_stageb = time.time() - t_stageb_start

            # Build canonical row
            _ensure_eval_in_path()
            from eval.omomo.eval_chois_table1 import (  # local import
                build_ours_row,
                COLUMN_ORDER,
                PAPER_CHOIS_OURS,
                _verdict,
            )
            agg = raw.get("aggregate", {})
            ours = build_ours_row(agg, classifier)
            verdicts = {
                col: _verdict(col, ours[col], PAPER_CHOIS_OURS[col])
                for col in COLUMN_ORDER
            }

            def _nan_safe(v: Any) -> Any:
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            metrics: Dict[str, Any] = {
                col: _nan_safe(ours[col]) for col in COLUMN_ORDER
            }
            metrics["paper_chois_ours"] = dict(PAPER_CHOIS_OURS)
            metrics["verdicts"] = verdicts
            # Resolve actual DDPM step count from scheduler (matches num_train_timesteps).
            # "ddpm_full" runs exactly num_train_timesteps steps — may differ from 1000
            # for models trained with fewer train timesteps (e.g. 100 for v13.8).
            _agent = getattr(pl_module, "agent", None) or getattr(pl_module, "model", None)
            _diffusion = getattr(_agent, "diffusion", None)
            _actual_ddpm_steps = int(
                getattr(_diffusion, "num_train_timesteps", "?")
                if _diffusion is not None else "?"
            ) if _diffusion is not None else "?"
            metrics["provenance"] = {
                "branch": "production",
                "step": step,
                "n_seqs": N_test,
                "K": _K_PRODUCTION,
                # Actual number of DDPM inference steps — equals num_train_timesteps
                # from the training config. "ddpm_full" mode (alias: "ddpm_1000") uses
                # all training timesteps; this is NOT necessarily 1000.
                "n_steps": f"ddpm_full:{_actual_ddpm_steps}",
                "scheduler_mode": "ddpm_full",
                "eval_sample_ratio": self._eval_sample_ratio,
                "eval_n_windows": N_test,
                "skip_fid": classifier is None,
                "max_bs": max_bs,
                "wall_time_s": round(time.time() - t0, 2),
                "t_sample_s": round(t_sample, 2),
                "t_stagea_s": round(t_stagea, 2),
                "t_stageb_s": round(t_stageb, 2),
                "host": socket.gethostname(),
                # preds_dir_a / preds_dir_b are tmp dirs deleted in finally block;
                # not included in provenance to avoid stale paths (R6 fix).
            }

            _LOG.info(
                "WmmOmomoEvalCallback step=%d: sample=%.1fs stageA=%.1fs stageB=%.1fs total=%.1fs",
                step, t_sample, t_stagea, t_stageb, time.time() - t0,
            )

        finally:
            shutil.rmtree(preds_dir_a, ignore_errors=True)
            shutil.rmtree(preds_dir_b, ignore_errors=True)
            shutil.rmtree(stage_a_dir, ignore_errors=True)
            shutil.rmtree(tmp_gt_dir, ignore_errors=True)

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_gt_paths(self) -> List[str]:
        """Return GT npz paths (cached after first call).

        Applies ``eval_sample_ratio`` on first call: if < 1.0, selects
        ``round(ratio * total)`` paths via deterministic shuffle
        (``random.Random(seed=0)``).  Using a random sample avoids the
        lexicographic-first-N bias that would skew metrics when ratio < 1.0.
        The seed is fixed for reproducibility across eval calls.
        """
        import random as _random
        if self._gt_paths is None:
            paths = sorted(
                str(p) for p in Path(self._gt_dir).glob("*.npz")
            )
            if not paths:
                raise FileNotFoundError(
                    f"WmmOmomoEvalCallback: no .npz files found in gt_dir={self._gt_dir}"
                )
            n_full = len(paths)
            if self._eval_sample_ratio < 1.0:
                n_eval = max(1, round(self._eval_sample_ratio * n_full))
                # Deterministic shuffle + take first n_eval — equivalent to
                # a fixed-seed random sample without replacement.
                shuffled = list(paths)
                _random.Random(0).shuffle(shuffled)
                paths = sorted(shuffled[:n_eval])  # re-sort for stable iteration order
                _LOG.info(
                    "WmmOmomoEvalCallback: eval_sample_ratio=%.3f -> %d/%d windows "
                    "(random.Random(0) sample, not lexicographic)",
                    self._eval_sample_ratio, n_eval, n_full,
                )
            self._gt_paths = paths
            _LOG.info(
                "WmmOmomoEvalCallback: found %d GT npz files in %s",
                len(paths), self._gt_dir,
            )
        return self._gt_paths

    def _run_inference_batch(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
        seed: int,
        n_steps: Optional[int],
        scheduler_mode: str,
    ) -> torch.Tensor:
        """Run agent.inference() and return ret_trace (B, T, P, 4, 4).

        Args:
            pl_module: The LightningModule (has ``pl_module.agent``).
            batch: Collated batch dict built by ``_build_batch_from_gt_files``.
            seed: Random seed for this inference pass.
            n_steps: Denoising steps, or None to use the scheduler's default
                timesteps (required for ``ddpm_full`` mode to avoid
                ``set_timesteps`` errors on models with <1000 train timesteps).
            scheduler_mode: ``"ddim_gen"`` (debug) or ``"ddpm_full"`` (prod).

        Returns:
            ret_trace tensor of shape ``(B, T, P, 4, 4)`` on device.
        """
        # Support both .agent (new convention) and .model (legacy wmm PLModel).
        agent = getattr(pl_module, "agent", None) or getattr(pl_module, "model", None)
        if agent is None:
            raise AttributeError(
                "WmmOmomoEvalCallback: pl_module has neither .agent nor .model attribute"
            )
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
            scheduler_mode=scheduler_mode,
        )
        # result is (ret_context, ret_trace); ret_trace is (B, T, P, 4, 4)
        _, ret_trace = result
        return ret_trace

    def _run_stage_a(
        self,
        preds_dir: str,
        out_json: str,
        device: str = "cuda",
        gt_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Stage A (run_eval in-process). Wraps the eval harness call.

        Args:
            preds_dir: Directory with prediction npz files.
            out_json: Path to write the raw Stage A JSON output.
            device: CUDA device string.
            gt_dir: GT directory override. If None, uses ``self._gt_dir``.
                    Pass a subset dir when evaluating fewer than the full 482 seqs.
        """
        _ensure_eval_in_path()
        from eval.omomo.eval_chois_table1 import run_stage_a  # local import
        return run_stage_a(
            preds_dir=preds_dir,
            gt_dir=gt_dir or self._gt_dir,
            smplx_dir=self._smplx_dir,
            sdf_folder=self._sdf_folder,
            out_json=out_json,
            device=device,
            cwd_for_obj_geo=str(Path(self._obj_geo_dir).parents[3]),
        )

    def _autosearch_max_bs(
        self,
        pl_module: pl.LightningModule,
        probe_paths: List[str],
        device: torch.device,
        step: int,
    ) -> int:
        """Binary-doubling search for max feasible batch size.

        Starts at ``max_bs_start``, doubles until OOM, returns last success.
        Result is cached for subsequent eval calls.

        Args:
            pl_module: The LightningModule.
            probe_paths: At least one GT path to use for the probe batch.
            device: Target device.
            step: Current training step (for logging).

        Returns:
            Maximum feasible batch size.
        """
        if self._cached_max_bs is not None:
            _LOG.info(
                "WmmOmomoEvalCallback: reusing cached max_bs=%d", self._cached_max_bs
            )
            return self._cached_max_bs

        log_lines: List[str] = [f"BS autosearch at step={step}"]
        candidate = self._max_bs_start
        # None until first successful probe — ensures we don't cache a fake "success"
        # if even the starting batch size OOMs.
        last_success: Optional[int] = None

        while True:
            # Build probe batch of size `candidate`
            probe_size = candidate
            expanded = (probe_paths * math.ceil(probe_size / len(probe_paths)))[:probe_size]
            try:
                probe_batch, _ = _build_batch_from_gt_files(
                    expanded, device, self._obj_geo_dir
                )
                _ = self._run_inference_batch(
                    pl_module=pl_module,
                    batch=probe_batch,
                    seed=0,
                    n_steps=None,          # match production: full DDPM steps
                    scheduler_mode="ddpm_full",  # match production memory profile
                )
                last_success = candidate
                msg = f"  bs={candidate}: OK"
                _LOG.info("WmmOmomoEvalCallback autosearch: %s", msg)
                log_lines.append(msg)
                torch.cuda.empty_cache()
                candidate = candidate * 2
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                    raise  # not an OOM -- re-raise
                torch.cuda.empty_cache()
                if last_success is None:
                    raise RuntimeError(
                        f"WmmOmomoEvalCallback: even bs={candidate} OOMs; cannot run eval. "
                        f"Reduce max_bs_start (current={self._max_bs_start})."
                    )
                msg = f"  bs={candidate}: OOM — stopping"
                _LOG.info("WmmOmomoEvalCallback autosearch: %s (using bs=%d)", msg, last_success)
                log_lines.append(msg)
                log_lines.append(f"  -> max_bs={last_success}")
                break

        if self._bs_autosearch_log is not None:
            try:
                with open(self._bs_autosearch_log, "a") as fh:
                    fh.write("\n".join(log_lines) + "\n")
            except OSError as exc:
                _LOG.warning(
                    "WmmOmomoEvalCallback: could not write bs_autosearch_log: %s", exc
                )

        self._cached_max_bs = last_success
        _LOG.info(
            "WmmOmomoEvalCallback: autosearch complete, max_bs=%d", last_success
        )
        return last_success
