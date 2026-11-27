"""TraceGen (TG) evaluation callback — of.

Periodically runs the FULL TG test set through the WMM ``AgentV11`` and
reports the authors' EXACT trajectory metrics (``trajectory_mse``,
``trajectory_mae``, ``endpoint_mse``) in reference protocol ``×1e-2`` format on wandb.

Inference runs through ``pl_module.ema_model`` (UNCOMPILED copy) when
``use_ema`` is enabled — see ``_select_eval_model`` and ``model.py:321``.
Calling inference through the possibly torch-compiled ``pl_module.model``
triggers recompile / fallback on inference-shape tensors → OOM.

This callback is the binding between our model and the TraceGen reference protocol
benchmark (see ``data_wmm/tracegen/eval.md``). Numbers are directly
copyable into Table 1 of that document.

Key contract pins (consumed by the implementation below):

* Metric code is a near-byte-copy of the authors' ``absolute_action=True``
   branch at ``data_wmm/tracegen/repo/trainer/trainer.py:790-846`` -- the
   ``_compute_trajectory_metrics`` body. Pred and GT are already absolute
   in our pipeline (Q9 +), so we *skip* the cumsum step that the
   authors apply to their delta-format GT.
* Per-sample trajectory mask is kept per-coord
   ``(B, 33, 400, 3)`` in ``batch["extra"]["trace_valid_mask_per_coord"]``
   and flattened to ``(B*400, 33, 3)`` at the metric site -- the shape
   authors' metric operates on. Any-reducing the coord axis would
   destroy bit-identity with reference protocol numbers.
* ``inference`` is called on the selected eval model (``ema_model`` when
   ``use_ema``, else ``model``) with the EXACT kwarg names documented in
   ``$TRUN_DIR/sg/01-tg-dataset/inference_signature_mapping.md``. No
   renaming.

Source-of-truth files:

* mapping doc (this callback's dispatch):
   ``$TRUN_DIR/sg/01-tg-dataset/inference_signature_mapping.md``
* dataset:
   ``src/umm/datasets/wmm/tg_dataset.py``
* Authors' test filter + metric body (attribution in-code):
   ``data_wmm/tracegen/repo/trainer/trainer.py`` around L780-L920.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset, Subset

# Lazy-import TracegenDataset + SEM_REAL_SCENE constant at call time to
# avoid pulling dataset-heavy deps (PIL, etc.) during module import -- the
# same pattern ``datamodule.py`` uses.

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Eval-model selection
# ────────────────────────────────────────────────────────────────────────


def _select_eval_model(pl_module):
    """Return the uncompiled eval model.

    WMM convention: training forward goes through ``pl_module.model`` (often
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


# ────────────────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────
# Metric (byte-copy of authors' absolute_action=True branch)
# ────────────────────────────────────────────────────────────────────────


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
       absolute (Q9 +), so the ``absolute_action=True`` branch is
       used, which passes them through unchanged.
    2. ``torch.cat((targets[:, :1, :], preds), dim=1)`` step SKIPPED.
       Authors' ``predicted_trajectory`` has 32 steps (future only);
       they prepend GT step-0 to match the target's 33 steps. Our
       ``AgentV11.inference`` receives ``trace`` with
       ``trace_known_mask[:, 0] = True``, so ``ret_trace[:, 0]``
       is the input's known GT at t=0. The concat is therefore a
       no-op on our pred and is removed for clarity.
    3. ``TRACEGEN_STRIP_Z`` env gating OMITTED. Droid / Epic reference protocol
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
        Returns an empty dict if ``mask.sum <= 0``.
    """
    # ---- Authors' body (absolute_action=True branch) ----
    # See the docstring above for the explicit list of every deviation
    # from the source. The arithmetic below is byte-identical to
    # trainer.py:816-844.
    all_masks = all_masks.to(pred_abs.dtype)
    total_valid = all_masks.sum()
    if total_valid.item() <= 0:
        logger.warning("TgEvalCallback: no valid masked entries for trajectory metrics")
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


# ────────────────────────────────────────────────────────────────────────
# Callback
# ────────────────────────────────────────────────────────────────────────


class TgEvalCallback(Callback):
    """Run the TG test set through WMM at a cadence and log reference protocol metrics.

    Lifecycle hook: ``on_train_batch_end``. Fires ``_run_eval("production")``
    when ``trainer.global_step % eval_every_n_steps == 0`` (and > 0), and
    ``_run_eval("debug")`` when ``debug_every_n_steps > 0`` and
    ``global_step % debug_every_n_steps == 0`` (includes step 0 for
    smoke runs). Debug mode uses a deterministic ``debug_subset_n_samples``
    subset so the first numbers appear quickly during ``fast_dev_run``.

    Args:
        env_name: ``"droid"`` or ``"epic"`` (TG env).
        test_data_root: Path containing ``Droid/`` / ``Epickitchen/``
            subdirectories (e.g., ``<path>``).
        splits: Ignored keyword retained for parity with the spec; the
            callback always evaluates the ``"test"`` split because the
            TG reference protocol metric is defined on test. Accepts ``str`` or ``dict``.
        eval_every_n_steps: Cadence for the full-test production eval.
        batch_size: DataLoader batch size for eval.
        debug_every_n_steps: ``-1`` disables debug cadence. Otherwise fires
            at each multiple of this step count (including step 0).
        debug_subset_n_samples: Sample count for the deterministic debug
            subset (seed 42).
        num_workers: Eval DataLoader workers. Set small for unit tests.
        eval_effort: Ratio of test samples to evaluate in PRODUCTION mode
            (1.0 = full test set, 0.05 = 5% evenly spaced). Indices are
            deterministic (``np.linspace(0, N-1, n_eval).round.astype(int)``).
            Does NOT apply to debug mode. Default ``1.0``.
        log_prefix: Wandb key prefix (default ``"tg_eval"``).
        inference_kwargs: Additional kwargs forwarded to
            ``model.inference(...)`` (``seed``, ``n_steps``, ``verbose``,
            ``cfg_scale`` ...). Default ``{"seed": 42, "verbose": False}``.
    """

    def __init__(
        self,
        env_name: str,
        test_data_root: str,
        splits: Union[dict, str, None] = None,
        eval_every_n_steps: int = 10000,
        batch_size: int = 4,
        debug_every_n_steps: int = -1,
        debug_subset_n_samples: int = 16,
        num_workers: int = 4,
        eval_effort: float = 1.0,
        log_prefix: str = "tg_eval",
        inference_kwargs: Optional[dict] = None,
        base_fps: float = 30.0,
        t_rate: int = 1,
        dummy_id_rot: bool = False,
    ):
        super().__init__()
        self.env_name = env_name
        self.test_data_root = test_data_root
        self.splits = splits  # kept for API parity; not used (see docstring)
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.batch_size = int(batch_size)
        self.debug_every_n_steps = int(debug_every_n_steps)
        self.debug_subset_n_samples = int(debug_subset_n_samples)
        self.num_workers = int(num_workers)
        # Time axis for LMM RoPE — must mirror TGAdapter (wmm/datamodule.py).
        # Defaults match the dataset class defaults; YAML can override.
        self.base_fps = float(base_fps)
        self.t_rate = int(t_rate)
        # Match the training-side dataset flag (datamodule.py:155). When True,
        # the eval batch's ``rot_mask`` is set all-True so rotation channels
        # enter inference's diffusion pipeline the same way they did during
        # training. False (default) keeps legacy behavior (rot zeroed).
        self.dummy_id_rot = bool(dummy_id_rot)
        if not (0.0 < float(eval_effort) <= 1.0):
            raise ValueError(
                f"TgEvalCallback: eval_effort must be in (0.0, 1.0], got {eval_effort!r}"
            )
        self.eval_effort = float(eval_effort)
        self.log_prefix = log_prefix
        self.inference_kwargs = dict(inference_kwargs) if inference_kwargs else {}
        self.inference_kwargs.setdefault("seed", 42)
        self.inference_kwargs.setdefault("verbose", False)

        # Internal cached debug subset so every debug call looks at the same
        # samples (determinism test depends on this).
        self._debug_subset: Optional[Subset] = None
        self._production_dataset: Optional[Dataset] = None
        self._production_subset: Optional[Subset] = None

        # Track steps we already evaluated to avoid duplicate fires when
        # a global step hits both cadences.
        self._last_eval_step_production: int = -1
        self._last_eval_step_debug: int = -1

        # Warn-once guard for the "empty filtered batch" case.
        self._warned_empty_filter: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Trigger eval at production or debug cadences.

        Fires ``_run_eval("production")`` and/or ``_run_eval("debug")`` at
        the configured step multiples. Both may fire on the same step
        (production dominates if both match).
        """
        step = int(trainer.global_step)

        fire_production = (
            self.eval_every_n_steps > 0
            and step > 0
            and step % self.eval_every_n_steps == 0
            and step != self._last_eval_step_production
        )
        fire_debug = (
            self.debug_every_n_steps > 0
            and step % self.debug_every_n_steps == 0
            and step != self._last_eval_step_debug
        )

        if fire_production:
            self._run_eval_safe(trainer, pl_module, mode="production")
            self._last_eval_step_production = step
        # Debug can also fire at the production step; keep both for
        # transparency (the production numbers are on full test, the
        # debug numbers are on the fixed subset — different meaning).
        if fire_debug:
            self._run_eval_safe(trainer, pl_module, mode="debug")
            self._last_eval_step_debug = step

    def _run_eval_safe(self, trainer, pl_module, mode: str) -> None:
        try:
            self._run_eval(trainer, pl_module, mode=mode)
        except Exception as e:  # noqa: BLE001 -- logged + re-raised in strict
            logger.error(
                f"[TgEvalCallback] {self.env_name} {mode}: eval failed: {e!r}",
                exc_info=True,
            )
            # Surface the error — training should still continue because
            # eval is monitoring, but we want the stack in the log.

    # ── Core eval ──────────────────────────────────────────────────────

    @rank_zero_only
    @torch.no_grad()
    def _run_eval(
        self, trainer, pl_module, mode: str
    ) -> Optional[Dict[str, float]]:
        """Run the full (production) or subset (debug) TG eval.

        Args:
            trainer: Lightning ``Trainer``.
            pl_module: The ``LightningModule`` (has ``.model`` = ``AgentV11``).
            mode: ``"production"`` or ``"debug"``.

        Returns:
            Dict with keys ``trajectory_mse_x100``, ``trajectory_mae_x100``,
            ``endpoint_mse_x100``, ``n_samples_valid``, ``wall_clock_s``.
            **Under DDP** the ``@rank_zero_only`` decorator causes this
            to return ``None`` on non-rank-0 processes; callers that
            read the dict must handle the ``None`` case.
        """
        assert mode in ("production", "debug")
        t0 = time.time()

        loader = self._make_loader(mode=mode)
        n_samples_evaluated = len(loader.dataset)

        was_training = pl_module.training
        pl_module.eval()

        pred_chunks: List[torch.Tensor] = []
        gt_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        n_samples_valid = 0

        for batch in loader:
            # ── Authors' filter — EXACT line quoted from
            # data_wmm/tracegen/repo/trainer/trainer.py:920
            # valid_mask = (batch['valid_steps'] > 3)
            #              & (batch['movement_bool'].sum(dim=1) > 0)
            #              & (batch['gt_mask'].sum(dim=1) > 0)
            valid_mask = (
                (batch["valid_steps"] > 3)
                & (batch["movement_bool"].sum(dim=1) > 0)
                & (batch["extra"]["gt_mask"].sum(dim=1) > 0)
            )
            n_valid_this_batch = int(valid_mask.sum().item())
            if n_valid_this_batch == 0:
                if not self._warned_empty_filter:
                    logger.warning(
                        "[TgEvalCallback] empty filtered batch encountered; "
                        "skipping (warn-once)"
                    )
                    self._warned_empty_filter = True
                continue

            batch = _filter_batch(batch, valid_mask)
            n_samples_valid += n_valid_this_batch

            pred_abs, gt_abs, mask_pc = self._forward_batch(pl_module, batch)
            # Accumulate on CPU to bound peak memory during full-test eval.
            pred_chunks.append(pred_abs.detach().to(torch.float32).cpu())
            gt_chunks.append(gt_abs.detach().to(torch.float32).cpu())
            mask_chunks.append(mask_pc.detach().to(torch.bool).cpu())

        wall_clock = time.time() - t0

        # Restore training mode before logging (logger may sync state).
        if was_training:
            pl_module.train()

        if len(pred_chunks) == 0:
            logger.warning(
                f"[TgEvalCallback] {self.env_name} {mode}: no valid samples "
                f"(filter rejected everything)."
            )
            return {
                "trajectory_mse_x100": float("nan"),
                "trajectory_mae_x100": float("nan"),
                "endpoint_mse_x100": float("nan"),
                "n_samples_evaluated": n_samples_evaluated,
                "n_samples_valid": 0,
                "wall_clock_s": wall_clock,
            }

        pred_abs = torch.cat(pred_chunks, dim=0)   # (B, 33, 400, 3)
        gt_abs = torch.cat(gt_chunks, dim=0)       # (B, 33, 400, 3)
        mask_pc = torch.cat(mask_chunks, dim=0)    # (B, 33, 400, 3)

        # Flatten (B, T=33, P=400, 3) → (B*400, T=33, 3), matching authors'
        # (B*400, T, 3) layout at the metric site.
        B_eff, T, P, C = pred_abs.shape
        assert T == 33 and P == 400 and C == 3, (
            f"TgEvalCallback: unexpected pred/gt shape after flatten: "
            f"{pred_abs.shape}"
        )
        pred_flat = pred_abs.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)
        gt_flat = gt_abs.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)
        mask_flat = mask_pc.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)

        raw_metrics = _compute_trajectory_metrics_abs(pred_flat, gt_flat, mask_flat)
        if not raw_metrics:
            metrics = {
                "trajectory_mse_x100": float("nan"),
                "trajectory_mae_x100": float("nan"),
                "endpoint_mse_x100": float("nan"),
                "n_samples_evaluated": n_samples_evaluated,
                "n_samples_valid": n_samples_valid,
                "wall_clock_s": wall_clock,
            }
        else:
            metrics = {
                "trajectory_mse_x100": float(raw_metrics["trajectory_mse"] * 100.0),
                "trajectory_mae_x100": float(raw_metrics["trajectory_mae"] * 100.0),
                "endpoint_mse_x100": float(raw_metrics["endpoint_mse"] * 100.0),
                "n_samples_evaluated": n_samples_evaluated,
                "n_samples_valid": n_samples_valid,
                "wall_clock_s": wall_clock,
            }

        self._log_to_wandb(pl_module, metrics, mode)

        logger.info(
            f"[TgEvalCallback] {self.env_name} {mode}: "
            f"n_evaluated={metrics['n_samples_evaluated']}, n_valid={metrics['n_samples_valid']}, "
            f"wall_clock={metrics['wall_clock_s']:.2f}s, "
            f"mse_x100={metrics['trajectory_mse_x100']:.4f}, "
            f"mae_x100={metrics['trajectory_mae_x100']:.4f}, "
            f"endpoint_x100={metrics['endpoint_mse_x100']:.4f}"
        )

        return metrics

    # ── Helpers ────────────────────────────────────────────────────────

    def _make_loader(self, mode: str) -> DataLoader:
        """Build a DataLoader for production-full or debug-subset eval.

        Datasets are cached on the callback so repeated evals don't
        re-glob episodes every cadence fire.
        """
        # Lazy import — keeps module import cheap.
        from umm.datasets.wmm.tg_dataset import TracegenDataset

        if self._production_dataset is None:
            self._production_dataset = TracegenDataset(
                data_root=self.test_data_root,
                env_name=self.env_name,
                split="test",
                base_fps=self.base_fps,
                t_rate=self.t_rate,
                dummy_id_rot=self.dummy_id_rot,
            )
        ds: Dataset = self._production_dataset

        # Production-mode eval_effort: evenly spaced subset of the full test
        # set. Deterministic across runs (np.linspace + round). Debug mode
        # unaffected — keeps its fixed seed-42 subset below.
        if mode == "production" and self.eval_effort < 1.0:
            if self._production_subset is None:
                N = len(self._production_dataset)
                n_eval = max(1, int(round(N * self.eval_effort)))
                indices = (
                    np.linspace(0, N - 1, n_eval).round().astype(int).tolist()
                )
                self._production_subset = Subset(self._production_dataset, indices)
                logger.info(
                    f"[TgEvalCallback] {self.env_name} production subset: "
                    f"eval_effort={self.eval_effort}, N={N}, n_eval={n_eval} "
                    f"(evenly spaced indices)"
                )
            ds = self._production_subset

        if mode == "debug":
            if self._debug_subset is None:
                gen = torch.Generator().manual_seed(42)
                n = min(self.debug_subset_n_samples, len(ds))
                perm = torch.randperm(len(ds), generator=gen)[:n].tolist()
                self._debug_subset = Subset(ds, perm)
            ds = self._debug_subset

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            num_workers=self.num_workers,
            collate_fn=_tg_collate,
        )

    def _forward_batch(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
    ):
        """Call ``inference`` on one filtered batch, return (pred, gt, mask).

        Runs through ``_select_eval_model(pl_module)`` -- i.e.
        ``pl_module.ema_model`` when ``use_ema`` else ``pl_module.model``.
        Avoids invoking inference through the torch-compiled training graph,
        which would force recompile / fallback on inference-shape tensors
        and OOM (root cause).

        Arg mapping is VERBATIM from
        ``$TRUN_DIR/sg/01-tg-dataset/inference_signature_mapping.md`` §4.
        No kwarg renaming.
        """
        # Lazy import to reuse the -approved constant value.
        from umm.datasets.wmm.dataset_config import SEM_REAL_SCENE

        device = pl_module.device
        trace = batch["trace"].to(device)                      # (B, 33, 400, 4, 4)
        trace_mask_any = batch["trace_mask"].to(device)        # (B, 33, 400) bool
        mask_pc = batch["extra"]["trace_valid_mask_per_coord"].to(device)  # (B,33,400,3)

        B, T, P, _, _ = trace.shape
        assert T == 33 and P == 400, f"TgEvalCallback: bad (T,P): ({T},{P})"

        # Context = duplicated trace[:, 0:1] (first frame); fully known.
        context = trace[:, :1].clone()                          # (B, 1, 400, 4, 4)
        trace_known = torch.zeros(B, T, P, dtype=torch.bool, device=device)
        trace_known[:, 0] = True                                 # t=0 only known
        ctx_known = torch.ones(B, 1, P, dtype=torch.bool, device=device)

        # context validity = gt_mask broadcast to time dim 1.
        gt_mask = batch["extra"]["gt_mask"].to(device)           # (B, 400) bool
        ctx_mask = gt_mask.unsqueeze(1)                          # (B, 1, 400)

        # Mirror TGAdapter time formula (wmm/datamodule.py:155): anchor at
        # t=0, future positive. base_fps + t_rate are configurable on the
        # callback (defaults 30.0 / 1) and MUST match the YAML dataset
        # entry. Required so the LMM t-axis RoPE has distinct phases
        # across the 33 trace rows.
        time_unit = float(self.t_rate) / float(self.base_fps)
        real_timestep = (
            torch.arange(T, dtype=torch.float32, device=device) * time_unit
        ).unsqueeze(0).expand(B, -1).contiguous()
        group_id = torch.zeros(B, P, dtype=torch.long, device=device)
        sem_type = torch.full(
            (B, P), fill_value=int(SEM_REAL_SCENE), dtype=torch.long, device=device
        )
        # Match the training-side rot_mask convention (datamodule.py:155).
        # When ``dummy_id_rot`` is enabled, rotation channels are part of the
        # diffusion pipeline; eval inference must see the same mask shape.
        if self.dummy_id_rot:
            rot_mask = torch.ones(B, P, dtype=torch.bool, device=device)
        else:
            rot_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        images = batch["extra"]["images"].to(device)            # (B, 1, 3, 384, 384)
        view_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        pixel_uv = batch["extra"]["first_keypoint"].to(device)  # (B, 400, 2)
        pixel_view_idx = torch.zeros(B, P, dtype=torch.long, device=device)
        pixel_query_mask = gt_mask                               # (B, 400) bool

        eval_model = _select_eval_model(pl_module)
        # Mirror the viz path in ``src/umm/models/wmm/model.py:324`` — put
        # the eval model explicitly into eval mode. ``pl_module.eval()`` on
        # the caller already cascades to registered submodules, but an
        # explicit call here is cheap insurance against BN / dropout
        # layers firing in training mode on inference batches.
        eval_model.eval()
        # ``torch.inference_mode`` is stronger than ``torch.no_grad`` and is
        # safer around compiled paths: it disables view tracking too, which
        # avoids subtle autograd state leaks between training and eval.
        # ``@torch.no_grad()`` is already set at the method level above; this
        # is the belt-and-suspenders call-site wrap.
        # w1.4.0: TG anchors at trace t=0 — pass zeros(B, P) long. Required
        # by LMM when bias_to_cur_frame=True; ignored when False.
        cur_frame_idx = torch.zeros(B, P, dtype=torch.long, device=device)
        with torch.inference_mode():
            ret_context, ret_trace = eval_model.inference(
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
                cur_frame_idx=cur_frame_idx,
                **self.inference_kwargs,
            )
        # ret_trace: (B, 33, 400, 4, 4) SE(3). Extract translation only.
        pred_abs = ret_trace[..., :3, 3]                          # (B, 33, 400, 3)
        gt_abs = trace[..., :3, 3]                                # (B, 33, 400, 3)

        return pred_abs, gt_abs, mask_pc

    def _log_to_wandb(
        self,
        pl_module: pl.LightningModule,
        metrics: Dict[str, float],
        mode: str,
    ) -> None:
        """Log the scalar metrics through whatever logger is attached.

        Uses ``pl_module.log`` which fans out to the configured logger
        (wandb or tensorboard or stdout). Keys are namespaced per
        ``{log_prefix}/{env_name}/...``; mode is appended for debug runs
        so both coexist in the same run.
        """
        suffix = "" if mode == "production" else "_debug"
        base = f"{self.log_prefix}/{self.env_name}"
        for key in (
            "trajectory_mse_x100",
            "trajectory_mae_x100",
            "endpoint_mse_x100",
            "n_samples_evaluated",
            "n_samples_valid",
            "wall_clock_s",
        ):
            try:
                # on_epoch=True belt-and-suspenders: this callback fires from
                # on_train_batch_end (training=True), where on_step=True alone
                # publishes the bare key into trainer.callback_metrics. If the
                # eval call is ever moved to a non-training hook, on_step=True
                # + on_epoch=False would NOT publish the bare key and the
                # ckpt monitor (`tg_eval/{env}/trajectory_mae_x100`) would
                # silently miss every step → permanent-mutation fallback in
                # ModelCheckpoint would wipe the user-configured monitor.
                pl_module.log(
                    f"{base}/{key}{suffix}",
                    torch.as_tensor(float(metrics[key])).float(),
                    on_step=True,
                    on_epoch=True,
                    rank_zero_only=True,
                )
            except Exception as e:  # noqa: BLE001 -- logging must not crash eval
                logger.warning(
                    f"[TgEvalCallback] wandb log for {base}/{key}{suffix} "
                    f"failed: {e!r}"
                )
