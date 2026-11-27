"""On-the-fly canonical-eval callback for wmm/GRAB (handobject_new) training.

Fires every ``val_every_n_iter`` gradient steps. Two branches:

* **Debug** (``val_every_n_iter < debug_threshold``):
  N=4 sample seqs, ``n_steps=10``, ``scheduler_mode="ddim_gen"``. Designed
  to complete in <60 s. Used as a wiring smoke test.

* **Production** (``val_every_n_iter >= debug_threshold``):
  ``round(eval_sample_ratio * len(test_seqs))`` sample seqs, model-default
  ``n_steps`` (None), ``scheduler_mode`` from cfg (default ``ddim_gen``).
  Target ≤8 min wall time per cycle (hard ≤10 min).

Pipeline (mirrors WmmOmomoEvalCallback structurally):

  1. Subset ``data_wmm/handobject_new/canonical/<split>/test/*.npz``
     deterministically (random.Random(0) shuffle, take first N).
     Skip ``M``-prefix mirrored duplicates.
  2. Build a collated batch matching ``GrabDataset.__getitem_inner__``
     (33 entities = 16 L hand + 16 R hand + 1 obj). Force the
     ``eval_keyframe`` known-mask pattern (only t=49 of L+R+obj revealed).
  3. ``agent.inference(...)`` -> ``ret_trace`` of shape ``(B, T, 33, 4, 4)``.
  4. Per-sample IK via ``q_to_mano`` for both hands; pull
     ``obj_trans``, ``obj_rot_mat`` from slot 32. Cache vtemplates per
     ``(subject_id, hand_side)``.
  5. Write canonical-schema pred .npz per sample. **No ``norm_seq_117d``.**
  6. Spawn ``eval/handobject_new/canonical_eval.py`` as a subprocess.
  7. Parse ``metrics.json`` and log ``grab_eval/{split}/{IV,ID,CR,SD,OD,AR}``
     plus latency probes to wandb via ``pl_module.log_dict``.
  8. Soft budget warning at ``cycle_total > soft_budget_s``.
  9. Atomic write of metrics summary to
     ``<log_dir>/grab_eval_step_<step>.json`` for offline review.

The callback never crashes training: ``_run_eval`` is wrapped in
``try/except`` and writes a ``metrics_FAILED_<step>.txt`` marker on error.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback as _tb
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh

from umm.datasets.wmm.base_dataset import PromptV0
from umm.datasets.wmm.dataset_config import (
    SEM_LEFT_MANO,
    SEM_OBJECT,
    SEM_RIGHT_MANO,
    UID_HUMOTO_OBJECT_GROUP,
    UID_MANO_LEFT_HAND,
    UID_MANO_RIGHT_HAND,
)
from umm.datasets.wmm.dataset_utils import sample_list
from umm.datasets.wmm.mask_scheduler import eval_keyframe

_LOG = logging.getLogger(__name__)

# Constants matching GrabDataset
_NUM_HAND_JOINTS = 16
_KEYFRAME_T = 49
_P = _NUM_HAND_JOINTS * 2 + 1  # 33 = 16 L + 16 R + 1 obj
_BASE_FPS = 30
_T_RATE = 1


# ---------------------------------------------------------------------------
# Repo-root resolution: callback file is
# src/umm/training_utils/callbacks/grab_eval.py -> parents[4] = repo root.
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[4])


# ---------------------------------------------------------------------------
# Vtemplate loader (cached per absolute PLY path).
# ---------------------------------------------------------------------------

_VTEMP_CACHE: Dict[str, np.ndarray] = {}


def _load_vtemplate(abs_ply_path: str) -> np.ndarray:
    """Load a 778x3 MANO vtemplate from a .ply, cached by absolute path.

    Fails loud if the file is not a single ``trimesh.Trimesh`` (e.g. a Scene
    with multiple geometries) — MANO vtemps are required to be a single mesh.
    """
    if abs_ply_path not in _VTEMP_CACHE:
        if not osp.exists(abs_ply_path):
            raise FileNotFoundError(
                f"WmmGrabEvalCallback: vtemplate PLY not found: {abs_ply_path}"
            )
        loaded = trimesh.load(abs_ply_path, process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(
                f"WmmGrabEvalCallback: vtemplate {abs_ply_path} loaded as "
                f"{type(loaded).__name__}, expected trimesh.Trimesh"
            )
        verts = np.asarray(loaded.vertices, dtype=np.float32)
        if verts.shape != (778, 3):
            raise ValueError(
                f"WmmGrabEvalCallback: vtemplate {abs_ply_path} has shape "
                f"{verts.shape}, expected (778, 3)"
            )
        _VTEMP_CACHE[abs_ply_path] = verts
    return _VTEMP_CACHE[abs_ply_path]


# ---------------------------------------------------------------------------
# Batch builder — mirrors GrabDataset.__getitem_inner__ exactly, except the
# known-mask pattern is forced to ``eval_keyframe`` (no MaskScheduler call).
# ---------------------------------------------------------------------------

def _build_batch_from_canonical_files(
    paths: List[str],
    device: torch.device,
    max_T: int = 200,
    num_pts: int = 256,
    max_C: int = 1,
    skip_mirrored: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build a collated batch dict from canonical handobject_new test .npz files.

    Mirrors ``GrabDataset.__getitem_inner__`` for trace/context construction;
    forces the ``eval_keyframe`` pattern at ``t=_KEYFRAME_T=49``.

    Args:
        paths: Absolute paths to canonical test .npz files.
        device: Target torch device.
        max_T: Time-window length at 30 fps.
        num_pts: PCL points per object.
        max_C: Number of context slots (default 1).
        skip_mirrored: If True, paths starting with 'M' are filtered.

    Returns:
        Tuple of (batch_dict, extras_list). ``extras_list[i]`` carries
        per-sample metadata used for IK / pred-npz writes.
    """
    if skip_mirrored:
        paths = [p for p in paths if not osp.basename(p).startswith("M")]
    if not paths:
        raise ValueError("WmmGrabEvalCallback: no canonical paths provided to batch builder")

    traces: List[torch.Tensor] = []
    trace_masks: List[torch.Tensor] = []
    contexts: List[torch.Tensor] = []
    context_masks: List[torch.Tensor] = []
    known_masks: List[torch.Tensor] = []
    ctx_known_masks: List[torch.Tensor] = []
    rot_masks: List[torch.Tensor] = []
    group_ids: List[torch.Tensor] = []
    sem_types: List[torch.Tensor] = []
    uids: List[torch.Tensor] = []
    times: List[torch.Tensor] = []
    texts: List[str] = []
    pcls: List[torch.Tensor] = []
    pcl_masks: List[torch.Tensor] = []
    extras: List[Dict[str, Any]] = []

    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            for required in (
                "Q_l_world_30fps", "Q_r_world_30fps", "obj_se3_world_30fps",
                "world_grasp_raw_idx", "obj_pcl", "world_is_mirrored",
                "seq_index", "data_id", "obj_name", "subject_id",
                "mano_betas_l", "mano_betas_r",
            ):
                if required not in d.files:
                    raise KeyError(
                        f"WmmGrabEvalCallback: {p} missing canonical field {required!r}"
                    )
            Q_l_30 = d["Q_l_world_30fps"].astype(np.float32)
            Q_r_30 = d["Q_r_world_30fps"].astype(np.float32)
            obj_se3_30 = d["obj_se3_world_30fps"].astype(np.float32)
            obj_pcl_raw = d["obj_pcl"].astype(np.float32)
            world_is_mirrored = bool(d["world_is_mirrored"])
            seq_index = str(d["seq_index"])
            data_id = str(d["data_id"])
            obj_name = str(d["obj_name"])
            subject_id = str(d["subject_id"])
            text_detailed = (
                str(d["text_detailed"]) if "text_detailed" in d.files else ""
            )
            world_lhand_vtemp_path = (
                str(d["world_lhand_vtemp_path"])
                if "world_lhand_vtemp_path" in d.files else ""
            )
            world_rhand_vtemp_path = (
                str(d["world_rhand_vtemp_path"])
                if "world_rhand_vtemp_path" in d.files else ""
            )
            world_obj_mesh_path = (
                str(d["world_obj_mesh_path"])
                if "world_obj_mesh_path" in d.files else ""
            )
            mano_betas_l = d["mano_betas_l"].astype(np.float32)
            mano_betas_r = d["mano_betas_r"].astype(np.float32)
            world_grasp_raw_idx = int(d["world_grasp_raw_idx"])

        # Sanity
        T_30_cache = Q_l_30.shape[0]
        if T_30_cache <= _KEYFRAME_T:
            raise ValueError(
                f"WmmGrabEvalCallback: {p} T_30_cache={T_30_cache} <= {_KEYFRAME_T}"
            )
        if Q_l_30.shape != (T_30_cache, _NUM_HAND_JOINTS, 4, 4):
            raise ValueError(f"bad Q_l shape {Q_l_30.shape}")
        if Q_r_30.shape != (T_30_cache, _NUM_HAND_JOINTS, 4, 4):
            raise ValueError(f"bad Q_r shape {Q_r_30.shape}")
        if obj_se3_30.shape != (T_30_cache, 4, 4):
            raise ValueError(f"bad obj_se3 shape {obj_se3_30.shape}")

        T = int(max_T)
        valid_T = min(T_30_cache, T)

        def _pad_se3(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] >= T:
                return arr[:T]
            pad_T = T - arr.shape[0]
            pad_shape = (pad_T,) + arr.shape[1:]
            pad = np.broadcast_to(np.eye(4, dtype=np.float32), pad_shape).copy()
            return np.concatenate([arr, pad], axis=0)

        Q_l_30 = _pad_se3(Q_l_30)
        Q_r_30 = _pad_se3(Q_r_30)
        obj_se3_30 = _pad_se3(obj_se3_30)

        Q_l = torch.from_numpy(Q_l_30).permute(1, 0, 2, 3).contiguous()  # (16, T, 4, 4)
        Q_r = torch.from_numpy(Q_r_30).permute(1, 0, 2, 3).contiguous()
        obj_se3 = torch.from_numpy(obj_se3_30).unsqueeze(0).contiguous()  # (1, T, 4, 4)

        hand_trace_mask = torch.zeros(_NUM_HAND_JOINTS, T, dtype=torch.bool)
        hand_trace_mask[:, :valid_T] = True
        obj_trace_mask = torch.zeros(1, T, dtype=torch.bool)
        obj_trace_mask[:, :valid_T] = True

        prompt = PromptV0(
            P=_P, T=T, C=int(max_C), PCL=int(num_pts), PCL_dim=6,
        )
        hand_ctx = torch.eye(4, dtype=torch.float32).expand(
            _NUM_HAND_JOINTS, int(max_C), 4, 4,
        ).contiguous()
        ctx_mask_full = torch.ones(_NUM_HAND_JOINTS, int(max_C))
        prompt.append_trace(
            trace=Q_l, trace_mask=hand_trace_mask,
            context=hand_ctx, context_mask=ctx_mask_full,
            group_id=0, sem_type=SEM_LEFT_MANO, uid=UID_MANO_LEFT_HAND,
        )
        prompt.append_trace(
            trace=Q_r, trace_mask=hand_trace_mask.clone(),
            context=hand_ctx.clone(), context_mask=ctx_mask_full.clone(),
            group_id=1, sem_type=SEM_RIGHT_MANO, uid=UID_MANO_RIGHT_HAND,
        )
        # Object
        obj_ctx = torch.eye(4, dtype=torch.float32).expand(1, int(max_C), 4, 4).contiguous()
        # Mirror handling on PCL: x-flip when world_is_mirrored (matches GrabDataset)
        pcl_np = obj_pcl_raw
        if world_is_mirrored:
            pcl_np = pcl_np.copy()
            pcl_np[:, 0] *= -1.0
        sampled_xyz = sample_list(pcl_np, int(num_pts), False)  # (num_pts, 3) deterministic
        rgb = np.zeros_like(sampled_xyz, dtype=np.float32)
        local_pcl = torch.as_tensor(
            np.concatenate([sampled_xyz, rgb], axis=1), dtype=torch.float32,
        ).unsqueeze(0)  # (1, num_pts, 6)
        prompt.append_trace(
            trace=obj_se3, trace_mask=obj_trace_mask,
            context=obj_ctx, context_mask=torch.ones(1, int(max_C)),
            group_id=2, sem_type=SEM_OBJECT,
            uid=int(UID_HUMOTO_OBJECT_GROUP[0]),
            local_pcl=local_pcl, local_pcl_mask=torch.ones(1),
        )
        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )
        # Force eval_keyframe pattern (only t=49 revealed for ALL valid trajectories)
        known_mask, ctx_known_mask = eval_keyframe(
            trace_mask=trace_mask,
            context_mask=context_mask,
            keyframe_t=_KEYFRAME_T,
        )
        time_unit = float(_T_RATE) / float(_BASE_FPS)
        time_stamp = (torch.arange(T) - _KEYFRAME_T) * time_unit
        local_pcl_full, local_pcl_mask_full = prompt.cat_pcl()

        traces.append(trace)
        trace_masks.append(trace_mask.bool())
        contexts.append(context)
        context_masks.append(context_mask.bool())
        known_masks.append(known_mask.bool())
        ctx_known_masks.append(ctx_known_mask.bool())
        rot_masks.append(rot_mask.bool())
        group_ids.append(group_id.int())
        sem_types.append(sem_type.int())
        uids.append(uid.int())
        times.append(time_stamp.float())
        texts.append(text_detailed or "")
        pcls.append(local_pcl_full.float())
        pcl_masks.append(local_pcl_mask_full.bool())
        extras.append(dict(
            data_id=data_id,
            seq_index=seq_index,
            subject_id=subject_id,
            obj_name=obj_name,
            world_is_mirrored=world_is_mirrored,
            world_grasp_raw_idx=world_grasp_raw_idx,
            world_lhand_vtemp_path=world_lhand_vtemp_path,
            world_rhand_vtemp_path=world_rhand_vtemp_path,
            world_obj_mesh_path=world_obj_mesh_path,
            mano_betas_l=mano_betas_l,
            mano_betas_r=mano_betas_r,
            valid_T=int(valid_T),
            T=int(T),
        ))

    # Collate to (B, T, P, ...) per CanonicalDatasetOutput.collate convention
    batch = {
        "trace":              torch.stack(traces, 0).float().permute(0, 2, 1, 3, 4).to(device),
        "trace_mask":         torch.stack(trace_masks, 0).bool().permute(0, 2, 1).to(device),
        "context":            torch.stack(contexts, 0).float().permute(0, 2, 1, 3, 4).to(device),
        "context_mask":       torch.stack(context_masks, 0).bool().permute(0, 2, 1).to(device),
        "known_mask":         torch.stack(known_masks, 0).bool().permute(0, 2, 1).to(device),
        "context_known_mask": torch.stack(ctx_known_masks, 0).bool().permute(0, 2, 1).to(device),
        "trace_group_id":     torch.stack(group_ids, 0).int().to(device),
        "trace_rot_mask":     torch.stack(rot_masks, 0).bool().to(device),
        "trace_sem_type":     torch.stack(sem_types, 0).int().to(device),
        "trace_uid":          torch.stack(uids, 0).int().to(device),
        "time":               torch.stack(times, 0).float().to(device),
        "text":               texts,
        "local_pcl":          torch.stack(pcls, 0).float().to(device),
        "local_pcl_mask":     torch.stack(pcl_masks, 0).bool().to(device),
        "extra": {
            "data_id":              [e["data_id"] for e in extras],
            "seq_index":            [e["seq_index"] for e in extras],
            "subject_id":           [e["subject_id"] for e in extras],
            "obj_name":             [e["obj_name"] for e in extras],
            "world_is_mirrored":    [e["world_is_mirrored"] for e in extras],
            "world_grasp_raw_idx":  [e["world_grasp_raw_idx"] for e in extras],
        },
    }
    return batch, extras


# ---------------------------------------------------------------------------
# Main callback class
# ---------------------------------------------------------------------------


class WmmGrabEvalCallback(pl.Callback):
    """On-the-fly canonical-eval callback for wmm/GRAB training.

    Mirrors :class:`umm.callbacks.wmm_omomo_eval.WmmOmomoEvalCallback`
    structurally but targets the GRAB / handobject_new pipeline.

    Args:
        test_root: Root of canonical handobject_new test files
            (e.g. ``data_wmm/handobject_new/canonical``).
        split: ``"s1"`` or ``"s2"``.
        run_tag: Method tag used for the predictions/<run_tag>/ directory and
            for keying ``metrics["methods"][<run_tag>][<SPLIT>]``.
        mano_dir: Directory holding ``MANO_LEFT.pkl`` and ``MANO_RIGHT.pkl``.
        diffh2o_dir: DiffH2O dir for 117-d normalisation (optional).
        clf_path: Path to ``clf_all.pt`` for the AR classifier metric (optional).
        sbj_model_path: Per-subject vtemplate root for canonical_eval (optional).
        data_root: Root used to resolve ``world_*hand_vtemp_path`` for IK
            (default ``data_wmm/handobject/imos_deps``).
        mesh_dir: Object contact-mesh directory passed to ``canonical_eval.py``
            via ``--mesh_dir``. ``None`` lets canonical_eval auto-detect from
            its built-in candidate list. Set explicitly when meshes live
            outside the default locations (e.g. ``data_wmm/handobject/
            raw_registered/grab_unzip/contact_meshes``).
        val_every_n_iter: Steps between eval runs.
        log_dir: Output directory for ``grab_eval_step_*.json`` summaries.
        predictions_root: Base directory for per-cycle pred subdirs
            (``<predictions_root>/<run_tag>/<step>/``).
        eval_sample_ratio: Ratio of test seqs to evaluate (production branch).
        debug_threshold: ``val_every_n_iter < debug_threshold`` -> debug branch
            (N=4, n_steps=10).
        n_viz_seqs: Number of GT-vs-pred panels to render. ``0`` disables viz.
        soft_budget_s: Cycle wall-time soft budget; emit warning above this.
        hard_budget_s: Subprocess timeout for canonical_eval.
        cleanup_predictions: If True, delete per-cycle pred dir after eval.
        scheduler_mode: Diffusion scheduler mode passed to ``agent.inference``.
        n_steps_debug: ``n_steps`` for the debug branch (default 10).
        n_steps_prod: ``n_steps`` for production (None = model default).
        max_T: Max time-window length at 30 fps (default 200).
        num_pts: PCL points per object (default 256).
        skip_mirrored: Filter ``M``-prefix duplicates from the test set.
    """

    def __init__(
        self,
        test_root: str,
        split: str,
        run_tag: str,
        mano_dir: str,
        diffh2o_dir: Optional[str],
        clf_path: Optional[str],
        sbj_model_path: Optional[str],
        data_root: str,
        mesh_dir: Optional[str],
        val_every_n_iter: int,
        log_dir: str,
        predictions_root: str,
        eval_sample_ratio: float = 0.04,
        debug_threshold: int = 100,
        n_viz_seqs: int = 0,
        soft_budget_s: float = 480.0,
        hard_budget_s: float = 600.0,
        cleanup_predictions: bool = True,
        scheduler_mode: str = "ddim_gen",
        n_steps_debug: int = 10,
        n_steps_prod: Optional[int] = None,
        max_T: int = 200,
        num_pts: int = 256,
        skip_mirrored: bool = True,
    ) -> None:
        super().__init__()
        if split not in ("s1", "s2"):
            raise ValueError(f"WmmGrabEvalCallback: split must be 's1'|'s2', got {split!r}")
        self._test_root = osp.abspath(test_root)
        self._split = split
        self._run_tag = run_tag
        self._mano_dir = osp.abspath(mano_dir)
        self._diffh2o_dir = osp.abspath(diffh2o_dir) if diffh2o_dir else None
        self._clf_path = osp.abspath(clf_path) if clf_path else None
        self._sbj_model_path = osp.abspath(sbj_model_path) if sbj_model_path else None
        self._data_root = osp.abspath(data_root)
        self._mesh_dir = osp.abspath(mesh_dir) if mesh_dir else None
        self._val_every_n_iter = int(val_every_n_iter)
        self._log_dir = osp.abspath(log_dir)
        self._predictions_root = osp.abspath(predictions_root)
        self._eval_sample_ratio = float(eval_sample_ratio)
        self._debug_threshold = int(debug_threshold)
        self._n_viz_seqs = int(n_viz_seqs)
        self._soft_budget_s = float(soft_budget_s)
        self._hard_budget_s = float(hard_budget_s)
        self._cleanup_predictions = bool(cleanup_predictions)
        self._scheduler_mode = str(scheduler_mode)
        self._n_steps_debug = int(n_steps_debug)
        self._n_steps_prod = (int(n_steps_prod) if n_steps_prod is not None else None)
        self._max_T = int(max_T)
        self._num_pts = int(num_pts)
        self._skip_mirrored = bool(skip_mirrored)

        self._mano_l_pkl = osp.join(self._mano_dir, "MANO_LEFT.pkl")
        self._mano_r_pkl = osp.join(self._mano_dir, "MANO_RIGHT.pkl")

        os.makedirs(self._log_dir, exist_ok=True)

        # Cached subset of test files (deterministic across cycles)
        self._cached_paths: Optional[List[str]] = None

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
        """Fire eval at ``global_step % val_every_n_iter == 0`` (step > 0)."""
        step = int(getattr(trainer, "global_step", 0))
        if step == 0 or (step % self._val_every_n_iter) != 0:
            return
        _LOG.info("WmmGrabEvalCallback: starting eval at step=%d", step)
        try:
            self._run_eval(pl_module, step)
        except Exception as exc:  # pylint: disable=broad-except
            fail_path = osp.join(self._log_dir, f"metrics_FAILED_{step}.txt")
            try:
                with open(fail_path, "w") as fh:
                    fh.write(f"step={step}\nhost={socket.gethostname()}\n\n")
                    _tb.print_exc(file=fh)
            except OSError as io_exc:
                _LOG.error("WmmGrabEvalCallback: also failed to write FAILED marker: %s", io_exc)
            _LOG.error(
                "WmmGrabEvalCallback: eval at step=%d FAILED — see %s. Error: %s",
                step, fail_path, exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_test_paths(self) -> List[str]:
        """Return cached deterministic subset of test .npz paths."""
        if self._cached_paths is not None:
            return self._cached_paths
        split_dir = osp.join(self._test_root, self._split, "test")
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"WmmGrabEvalCallback: missing {split_dir}")
        all_paths = sorted(
            osp.join(split_dir, f)
            for f in os.listdir(split_dir) if f.endswith(".npz")
        )
        if self._skip_mirrored:
            all_paths = [p for p in all_paths if not osp.basename(p).startswith("M")]
        if not all_paths:
            raise FileNotFoundError(
                f"WmmGrabEvalCallback: no .npz files in {split_dir} (skip_mirrored={self._skip_mirrored})"
            )
        n_full = len(all_paths)
        is_debug = self._val_every_n_iter < self._debug_threshold
        if is_debug:
            n_eval = min(4, n_full)
        else:
            n_eval = max(1, round(self._eval_sample_ratio * n_full))
        shuffled = list(all_paths)
        random.Random(0).shuffle(shuffled)
        chosen = sorted(shuffled[:n_eval])
        _LOG.info(
            "WmmGrabEvalCallback: split=%s, n_full=%d, n_eval=%d (debug=%s)",
            self._split, n_full, n_eval, is_debug,
        )
        self._cached_paths = chosen
        return chosen

    def _resolve_vtemp(self, rel_path: str) -> np.ndarray:
        """Resolve ``world_*hand_vtemp_path`` against ``data_root`` and load verts."""
        if not rel_path:
            raise ValueError("WmmGrabEvalCallback: empty vtemp path")
        if osp.isabs(rel_path):
            abs_path = rel_path
        else:
            abs_path = osp.join(self._data_root, rel_path)
        return _load_vtemplate(abs_path)

    def _run_inference_batch(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
        seed: int,
        n_steps: Optional[int],
        scheduler_mode: str,
    ) -> torch.Tensor:
        """Run agent.inference -> (B, T, P=33, 4, 4) ret_trace."""
        agent = getattr(pl_module, "agent", None) or getattr(pl_module, "model", None)
        if agent is None:
            raise AttributeError(
                "WmmGrabEvalCallback: pl_module has neither .agent nor .model"
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
        # (ret_context, ret_trace)
        _, ret_trace = result
        return ret_trace

    def _ik_one_sample(
        self,
        ret_trace_b: torch.Tensor,
        extra: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """IK one sample. ``ret_trace_b`` shape: (T, P=33, 4, 4) on CPU.

        Returns dict with the canonical-schema arrays for one sample.
        """
        from umm.eval.handobject_new.q_to_mano_ik import q_to_mano  # lazy

        T = int(ret_trace_b.shape[0])
        if ret_trace_b.shape != (T, _P, 4, 4):
            raise ValueError(
                f"WmmGrabEvalCallback: ret_trace_b shape {tuple(ret_trace_b.shape)} != ({T}, {_P}, 4, 4)"
            )
        Q_l = ret_trace_b[:, 0:_NUM_HAND_JOINTS]                      # (T, 16, 4, 4)
        Q_r = ret_trace_b[:, _NUM_HAND_JOINTS:2 * _NUM_HAND_JOINTS]   # (T, 16, 4, 4)
        obj_se3 = ret_trace_b[:, 2 * _NUM_HAND_JOINTS]                # (T, 4, 4)

        vt_l = self._resolve_vtemp(extra["world_lhand_vtemp_path"])
        vt_r = self._resolve_vtemp(extra["world_rhand_vtemp_path"])

        trans_l, gor_l, fp_l = q_to_mano(
            Q_l, is_rhand=False, vtemplate=vt_l, mano_pkl_path=self._mano_l_pkl,
        )
        trans_r, gor_r, fp_r = q_to_mano(
            Q_r, is_rhand=True, vtemplate=vt_r, mano_pkl_path=self._mano_r_pkl,
        )

        obj_trans = obj_se3[:, :3, 3]      # (T, 3)
        obj_rot_mat = obj_se3[:, :3, :3]   # (T, 3, 3)

        return dict(
            trans_rhand_l=trans_l.detach().cpu().numpy().astype(np.float32),
            trans_rhand_r=trans_r.detach().cpu().numpy().astype(np.float32),
            global_orient_rhand_aa_l=gor_l.detach().cpu().numpy().astype(np.float32),
            global_orient_rhand_aa_r=gor_r.detach().cpu().numpy().astype(np.float32),
            fpose_rhand_aa_l=fp_l.detach().cpu().numpy().astype(np.float32),
            fpose_rhand_aa_r=fp_r.detach().cpu().numpy().astype(np.float32),
            obj_trans=obj_trans.detach().cpu().numpy().astype(np.float32),
            obj_rot_mat=obj_rot_mat.detach().cpu().numpy().astype(np.float32),
            betas_l=np.asarray(extra["mano_betas_l"], dtype=np.float32),
            betas_r=np.asarray(extra["mano_betas_r"], dtype=np.float32),
        )

    @staticmethod
    def _save_pred_npz(
        pred_dir: str,
        seq_index: str,
        ik_dict: Dict[str, np.ndarray],
        data_id: str,
    ) -> str:
        """Write a pred .npz in canonical schema (no ``norm_seq_117d``).

        Filename is ``<seq_index>.npz`` so canonical_eval finds the matching
        test file via ``_load_test_file``.
        """
        out_path = osp.join(pred_dir, f"{seq_index}.npz")
        np.savez(
            out_path,
            trans_rhand_l=ik_dict["trans_rhand_l"],
            trans_rhand_r=ik_dict["trans_rhand_r"],
            global_orient_rhand_aa_l=ik_dict["global_orient_rhand_aa_l"],
            global_orient_rhand_aa_r=ik_dict["global_orient_rhand_aa_r"],
            fpose_rhand_aa_l=ik_dict["fpose_rhand_aa_l"],
            fpose_rhand_aa_r=ik_dict["fpose_rhand_aa_r"],
            betas_l=ik_dict["betas_l"],
            betas_r=ik_dict["betas_r"],
            obj_trans=ik_dict["obj_trans"],
            obj_rot_mat=ik_dict["obj_rot_mat"],
            data_id=data_id,
        )
        return out_path

    # ------------------------------------------------------------------
    # Subprocess + metrics parse
    # ------------------------------------------------------------------

    def _spawn_canonical_eval(
        self,
        pred_root: str,
        metrics_json_path: str,
    ) -> Tuple[bool, float, str]:
        """Run canonical_eval.py as subprocess. Returns (ok, wall_s, err_tail)."""
        repo_root = _repo_root()
        cmd = [
            sys.executable,
            osp.join(repo_root, "eval", "handobject_new", "canonical_eval.py"),
            "--test_root", self._test_root,
            "--predictions_root", pred_root,
            "--out", metrics_json_path,
            "--mano_dir", self._mano_dir,
            "--methods", self._run_tag,
            "--split", self._split,
        ]
        if self._diffh2o_dir:
            cmd += ["--diffh2o_dir", self._diffh2o_dir]
        if self._clf_path:
            cmd += ["--clf_path", self._clf_path]
        if self._sbj_model_path:
            cmd += ["--sbj_model_path", self._sbj_model_path]
        if self._mesh_dir:
            cmd += ["--mesh_dir", self._mesh_dir]

        t0 = time.time()
        err_tail = ""
        try:
            result = subprocess.run(
                cmd, cwd=repo_root,
                capture_output=True, text=True,
                timeout=self._hard_budget_s, check=False,
            )
            wall = time.time() - t0
            if result.returncode != 0:
                err_tail = result.stderr[-2000:] if result.stderr else ""
                _LOG.warning(
                    "WmmGrabEvalCallback: canonical_eval rc=%d (took %.1fs); stderr_tail=%s",
                    result.returncode, wall, err_tail,
                )
                return False, wall, err_tail
            return True, wall, ""
        except subprocess.TimeoutExpired as exc:
            wall = time.time() - t0
            _LOG.warning(
                "WmmGrabEvalCallback: canonical_eval timed out after %.1fs (hard_budget_s=%.1f)",
                wall, self._hard_budget_s,
            )
            return False, wall, f"TimeoutExpired after {wall:.1f}s"

    def _parse_metrics(self, metrics_json_path: str) -> Dict[str, Any]:
        """Parse metrics.json[methods][run_tag][SPLIT]. Returns {} on missing."""
        if not osp.exists(metrics_json_path):
            return {}
        try:
            with open(metrics_json_path) as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("WmmGrabEvalCallback: cannot parse metrics.json: %s", exc)
            return {}
        methods = doc.get("methods", {})
        method = methods.get(self._run_tag, {})
        if not isinstance(method, dict):
            return {}
        # canonical_eval keys splits as "S1" / "S2" upper case
        split_key = self._split.upper()
        split_metrics = method.get(split_key, {})
        if not isinstance(split_metrics, dict):
            return {}
        return split_metrics

    # ------------------------------------------------------------------
    # Core eval driver
    # ------------------------------------------------------------------

    def _run_eval(self, pl_module: pl.LightningModule, step: int) -> None:
        """Run one full eval cycle at ``step``. Logs to wandb + writes summary."""
        cycle_t0 = time.time()
        is_debug = self._val_every_n_iter < self._debug_threshold

        n_steps = self._n_steps_debug if is_debug else self._n_steps_prod
        scheduler_mode = "ddim_gen" if is_debug else self._scheduler_mode

        # 1. Pick test seqs (cached deterministically). _get_test_paths
        # returns the debug-capped subset when val_every_n_iter < debug_threshold,
        # so no further slice is needed here.
        paths = self._get_test_paths()
        n_seqs = len(paths)

        # canonical_eval expects <pred_root>/<method_tag>/*.npz, so create a
        # fresh per-cycle base directory and a method subdir inside it.
        cycle_base = osp.join(self._predictions_root, f"_grab_eval_step_{step:08d}")
        pred_method_dir = osp.join(cycle_base, self._run_tag)
        os.makedirs(pred_method_dir, exist_ok=True)
        metrics_json_path = osp.join(cycle_base, "metrics.json")

        try:
            device = next(pl_module.parameters()).device
        except StopIteration as exc:
            raise RuntimeError(
                "WmmGrabEvalCallback: pl_module has no parameters; "
                "cannot infer device for inference"
            ) from exc

        # Save original training mode and switch to eval for inference
        was_training = bool(pl_module.training)
        if was_training:
            pl_module.eval()

        t_inf = 0.0
        t_ik = 0.0
        try:
            # 2. Build batch
            batch, extras = _build_batch_from_canonical_files(
                paths,
                device=device,
                max_T=self._max_T,
                num_pts=self._num_pts,
                max_C=1,
                skip_mirrored=self._skip_mirrored,
            )

            # 3. Inference
            t_inf_s = time.time()
            with torch.inference_mode():
                ret_trace = self._run_inference_batch(
                    pl_module=pl_module,
                    batch=batch,
                    seed=0,
                    n_steps=n_steps,
                    scheduler_mode=scheduler_mode,
                )
            t_inf = time.time() - t_inf_s

            # 4. Per-sample IK + 5. write pred .npz
            t_ik_s = time.time()
            for b, extra in enumerate(extras):
                ret_b = ret_trace[b].detach().to("cpu")
                ik_dict = self._ik_one_sample(ret_b, extra)
                self._save_pred_npz(
                    pred_method_dir,
                    seq_index=extra["seq_index"],
                    ik_dict=ik_dict,
                    data_id=extra["data_id"],
                )
            t_ik = time.time() - t_ik_s

            # 6. canonical_eval subprocess
            ok, t_canonical_eval, err_tail = self._spawn_canonical_eval(
                pred_root=cycle_base,
                metrics_json_path=metrics_json_path,
            )

            # 7. Parse metrics and log
            metrics = self._parse_metrics(metrics_json_path) if ok else {}

            cycle_total = time.time() - cycle_t0
            over_budget = cycle_total > self._soft_budget_s
            if over_budget:
                warnings.warn(
                    f"WmmGrabEvalCallback: cycle_total {cycle_total:.1f}s exceeded "
                    f"soft_budget_s={self._soft_budget_s:.1f}s",
                    RuntimeWarning,
                )

            log_dict: Dict[str, float] = {
                "grab_eval/t_inference_per_seq": float(t_inf / max(1, n_seqs)),
                "grab_eval/t_ik_per_seq": float(t_ik / max(1, n_seqs)),
                "grab_eval/t_canonical_eval_total": float(t_canonical_eval),
                "grab_eval/n_seqs_evaluated": float(n_seqs),
                "grab_eval/cycle_total_seconds": float(cycle_total),
                "grab_eval/over_budget": 1.0 if over_budget else 0.0,
                "grab_eval/canonical_eval_ok": 1.0 if ok else 0.0,
            }

            # Drill into split metrics; tolerate missing keys (e.g. AR with no clf)
            for src, dst in [
                ("IV_cm3", "IV"), ("ID_mm", "ID"),
                ("CR", "CR"), ("SD", "SD"), ("OD", "OD"),
            ]:
                v = metrics.get(src)
                if isinstance(v, (int, float)):
                    log_dict[f"grab_eval/{self._split}/{dst}"] = float(v)
            ar_v = metrics.get("HandObj_AR_clf")
            if isinstance(ar_v, (int, float)):
                log_dict[f"grab_eval/{self._split}/AR"] = float(ar_v)

            try:
                pl_module.log_dict(
                    log_dict,
                    on_step=True, on_epoch=False,
                    sync_dist=False, rank_zero_only=True,
                    batch_size=1,
                )
                _LOG.info(
                    "WmmGrabEvalCallback: logged %d scalars to wandb (step=%d, ok=%s)",
                    len(log_dict), step, ok,
                )
            except Exception as log_exc:  # pylint: disable=broad-except
                _LOG.warning(
                    "WmmGrabEvalCallback: log_dict failed (non-fatal): %s", log_exc,
                )

            # 8. Atomic write metrics summary for offline review
            summary = {
                "step": int(step),
                "n_seqs": int(n_seqs),
                "split": self._split,
                "run_tag": self._run_tag,
                "eval_sample_ratio": self._eval_sample_ratio,
                "scheduler_mode": scheduler_mode,
                "n_steps": n_steps,
                "is_debug": bool(is_debug),
                "t_inference_s": round(float(t_inf), 3),
                "t_ik_s": round(float(t_ik), 3),
                "t_canonical_eval_s": round(float(t_canonical_eval), 3),
                "cycle_total_s": round(float(cycle_total), 3),
                "canonical_eval_ok": bool(ok),
                "canonical_eval_err_tail": err_tail,
                "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in metrics.items()},
                "host": socket.gethostname(),
            }
            tmp_dir = tempfile.mkdtemp(
                prefix=f"wmm_grab_eval_{step}_", dir=self._log_dir,
            )
            try:
                tmp_json = osp.join(tmp_dir, "metrics.json")
                with open(tmp_json, "w") as fh:
                    json.dump(summary, fh, indent=2)
                final_path = osp.join(
                    self._log_dir, f"grab_eval_step_{step:08d}.json",
                )
                shutil.move(tmp_json, final_path)
                _LOG.info("WmmGrabEvalCallback: summary -> %s", final_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # 10. Optional viz panel: deferred to follow-up iteration
            # (gated by self._n_viz_seqs > 0).
            # TODO(followup): Implement GT-vs-pred panels via
            # GrabVisualizer.draw_inference (see notes/sg03-viz-api.md).
            if self._n_viz_seqs > 0:
                _LOG.info(
                    "WmmGrabEvalCallback: viz panel deferred to follow-up iteration "
                    "(n_viz_seqs=%d). Set n_viz_seqs=0 to silence this notice.",
                    self._n_viz_seqs,
                )

        finally:
            # 11. Cleanup pred dir
            if self._cleanup_predictions:
                shutil.rmtree(cycle_base, ignore_errors=True)
            if was_training:
                pl_module.train()
