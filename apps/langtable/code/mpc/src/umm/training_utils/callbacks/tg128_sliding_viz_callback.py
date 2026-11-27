"""Tg128SlidingVizCallback — periodic full-K=400 sliding-window viz for K=128 training.

 Production K=128 training only logs the
sliding-window inference *metric* (via ``Tg128EvalCallback`` at 10k cadence).
This callback adds **visual** monitoring: every ``viz_every_n_train_steps``
training steps, it pulls a fixed reference 1-sample from the val split (with
``num_kps=400``), runs ``agent.inference_sliding_window`` to produce a
P=400 trajectory prediction, and renders the GT + prediction as MP4 videos
under ``logs/<expname>/version_<V>/tg_viz/``. The cached sample is
deterministic (always the same trajectory) so the user sees the model
improving over time on a fixed reference.

Filenames are suffixed ``_K400sw_<step>_{gt,pred}.mp4`` to distinguish them
from the regular K=128 training-time viz (which uses standard ``inference``
on K=128 windows). The "sw" tag stands for "sliding-window".

This is **viz-as-monitoring**: the entire fire body is wrapped in
try/except. A viz failure cannot kill training.

Wires into the trainer alongside ``Tg128EvalCallback`` via the
``tg128_sliding_viz`` cfg block (see ``robot_sim_callbacks.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from umm.training_utils.callbacks.tg_eval_callback import (
    _select_eval_model,
    _tg_collate,
)

logger = logging.getLogger(__name__)


class Tg128SlidingVizCallback(Callback):
    """Periodic K=128 -> P=400 sliding-window inference viz, MP4 output.

    Fires every ``viz_every_n_train_steps`` training steps (excluding
    step 0). Uses the same fixed sample every fire so the viz tracks
    model improvement on a deterministic reference. Renders 2D-flow
    and (when depth available) 3D-depth MP4 videos via the existing
    ``tg_viz`` helpers.

    Args:
        env_name: ``"droid"`` or ``"epic"`` (TG env).
        test_data_root: Root containing ``Droid/`` / ``Epickitchen/``.
        viz_every_n_train_steps: Cadence in training steps (e.g. 500).
        K_train: Trained window width (e.g. 128).
        anchor_count: Anchor token count for sub-steps after sub-step 0.
            Defaults to ``K_train // 2`` when ``None``.
        image_size: RGB canvas size (TG dataset uses 384).
        inference_kwargs: Additional kwargs forwarded to
            ``inference_sliding_window`` (e.g. ``seed``, ``n_steps``).
            Default ``{"seed": 42, "verbose": False}``. Must NOT include
            ``K_train`` or ``anchor_count`` -- those are constructor
            kwargs and a collision raises a clear ValueError (mirrors
            the peer ``Tg128EvalCallback`` guard).
    """

    def __init__(
        self,
        env_name: str,
        test_data_root: str,
        viz_every_n_train_steps: int,
        K_train: int,
        # sw-viz-debug fix: per-frame timestamps for the LMM temporal RoPE.
        # REQUIRED kwargs — no defaults. Robot_sim_callbacks reads these
        # from cfg.dataset.configs[0] (single source of truth, must match the
        # values the model was trained on). A missing dataset entry fails
        # loud at the wiring site, not silently here with wrong defaults.
        base_fps: float,
        t_rate: int,
        # iter 04b: dummy_id_rot must match dataset (training-time rot_mask).
        # When True, rotation channels carry an identity placeholder that the
        # model was supervised on; viz inference must pass rot_mask=ones.
        dummy_id_rot: bool,
        anchor_count: Optional[int] = None,
        image_size: int = 384,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.env_name = env_name
        self.test_data_root = test_data_root
        self.viz_every_n_train_steps = int(viz_every_n_train_steps)
        if self.viz_every_n_train_steps <= 0:
            raise ValueError(
                "Tg128SlidingVizCallback: viz_every_n_train_steps must be > 0, "
                f"got {viz_every_n_train_steps!r}"
            )
        self.K_train = int(K_train)
        if self.K_train <= 0:
            raise ValueError(
                f"Tg128SlidingVizCallback: K_train must be a positive int, "
                f"got {K_train!r}"
            )
        if anchor_count is None:
            self.anchor_count = self.K_train // 2
        else:
            self.anchor_count = int(anchor_count)
        if not (0 <= self.anchor_count < self.K_train):
            raise ValueError(
                f"Tg128SlidingVizCallback: anchor_count must be in [0, K_train), "
                f"got anchor_count={self.anchor_count}, K_train={self.K_train}"
            )
        self.image_size = int(image_size)
        self.base_fps = float(base_fps)
        self.t_rate = int(t_rate)
        self.dummy_id_rot = bool(dummy_id_rot)
        if self.base_fps <= 0:
            raise ValueError(
                f"Tg128SlidingVizCallback: base_fps must be > 0, got {base_fps!r}"
            )
        if self.t_rate <= 0:
            raise ValueError(
                f"Tg128SlidingVizCallback: t_rate must be > 0, got {t_rate!r}"
            )
        self.inference_kwargs = dict(inference_kwargs) if inference_kwargs else {}
        # Mirror Tg128EvalCallback's collision guard: K_train/anchor_count are
        # constructor kwargs; if they leak into inference_kwargs the splat at
        # the inference_sliding_window call site below would raise a confusing
        # ``TypeError: multiple values for keyword 'K_train'`` 500 steps into
        # training. Fail loud at init instead.
        for _bad_kw in ("K_train", "anchor_count"):
            if _bad_kw in self.inference_kwargs:
                raise ValueError(
                    f"Tg128SlidingVizCallback: inference_kwargs must not "
                    f"include {_bad_kw!r}; pass it as a constructor kwarg "
                    f"instead."
                )
        self.inference_kwargs.setdefault("seed", 42)
        self.inference_kwargs.setdefault("verbose", False)

        # Cached single-sample batch (built once on first fire).
        self._cached_batch: Optional[Dict[str, Any]] = None
        # Sentinel: ``True`` once the cache build has *failed* permanently
        # (e.g. empty val split). Subsequent fires short-circuit instead of
        # rebuilding the DataLoader (which would re-glob every TG episode).
        self._cache_dead: bool = False
        # Track last-fired step to avoid duplicate fires inside the same step.
        self._last_fired_step: int = -1

        logger.info(
            f"[Tg128SlidingViz] init: env={self.env_name}, "
            f"viz_every_n_train_steps={self.viz_every_n_train_steps}, "
            f"K_train={self.K_train}, anchor_count={self.anchor_count}"
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Fire viz when ``global_step`` matches the configured cadence.

        Latch update (``self._last_fired_step``) lives INSIDE ``_render``
        (after the rank/cache early-returns) so the cadence latch never
        advances on a no-op fire (rank>0 in DDP, dead cache, etc.). This
        is dormant today (single-GPU per training run) but matters if
        multi-rank ever lands.
        """
        step = int(trainer.global_step)
        if step <= 0:
            return
        if step % self.viz_every_n_train_steps != 0:
            return
        if step == self._last_fired_step:
            return

        # Wrap entire fire body — viz must NEVER kill training.
        try:
            self._render(trainer, pl_module, step)
        except Exception as e:  # noqa: BLE001 — logged + swallowed by design
            logger.error(
                f"[Tg128SlidingViz] step={step}: viz failed: {e!r}",
                exc_info=True,
            )

    # ── Cached sample ──────────────────────────────────────────────────

    def _get_cached_batch(self) -> Optional[Dict[str, Any]]:
        """Return the cached single-sample batch; build it on first call.

        Uses ``num_kps=400`` so the sliding-window dispatcher (K_train=128
        != P=400) actually engages. The DataLoader is shuffle=False and
        we take the first sample, so the same trajectory is returned
        every fire.

        Returns ``None`` (and sets ``self._cache_dead``) if the val split
        is empty or unreadable. Subsequent calls short-circuit on the
        sentinel instead of re-globbing every TG episode per fire.
        """
        if self._cached_batch is not None:
            return self._cached_batch
        if self._cache_dead:
            return None

        # Lazy import to keep module import cheap and mirror TgEvalCallback.
        from umm.datasets.wmm.tg_dataset import TracegenDataset

        ds = TracegenDataset(
            data_root=self.test_data_root,
            env_name=self.env_name,
            split="val",
            num_kps=400,
            image_size=self.image_size,
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=_tg_collate,
        )
        try:
            self._cached_batch = next(iter(loader))
        except StopIteration:
            self._cache_dead = True
            logger.error(
                f"[Tg128SlidingViz] val split for env={self.env_name} is "
                f"empty — disabling viz callback for the rest of training."
            )
            return None
        logger.info(
            f"[Tg128SlidingViz] cached 1-sample reference "
            f"(env={self.env_name}, split=val, P=400)"
        )
        return self._cached_batch

    # ── Forward (mirrors Tg128EvalCallback._forward_batch) ─────────────

    def _forward_sliding(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Run sliding-window inference on one batch, return predicted xyz.

        Mirrors :class:`Tg128EvalCallback._forward_batch` byte-for-byte for the
        common-kwargs construction. GT comes from the batch directly via
        ``_sample_from_tg_batch`` in the caller, so this only returns the
        predicted ``(B, T, P, 3)`` absolute xyz.
        """
        from umm.datasets.wmm.dataset_config import SEM_REAL_SCENE

        device = pl_module.device
        trace = batch["trace"].to(device)                           # (1, 33, 400, 4, 4)
        trace_mask_any = batch["trace_mask"].to(device)             # (1, 33, 400) bool

        B, T, P, _, _ = trace.shape
        assert T == 33, f"Tg128SlidingViz: bad T: {T} (expected 33)"

        context = trace[:, :1].clone()                              # (1, 1, 400, 4, 4)
        trace_known = torch.zeros(B, T, P, dtype=torch.bool, device=device)
        trace_known[:, 0] = True
        ctx_known = torch.ones(B, 1, P, dtype=torch.bool, device=device)
        # ``mask_pc`` (per-coord trace mask) is consumed by the eval-callback
        # metric path but not by viz/inference; we deliberately do NOT move
        # it to GPU here (avoids a wasted device alloc per fire).

        gt_mask = batch["extra"]["gt_mask"].to(device)             # (1, 400) bool
        ctx_mask = gt_mask.unsqueeze(1)                            # (1, 1, 400)

        # CRITICAL FIX (sw-viz-debug): per-frame timestamps must NOT be all
        # zeros — the LMM multiplies by time_multiplier=10 to build temporal
        # RoPE phase. Zero timestamps → all 33 frames receive the same time
        # phase → model cannot differentiate frames → predicts the same value
        # at every frame → after de-canon all tracks collapse to scene mean
        # (the original "tracks fly to center" symptom of K400sw viz).
        # Mirror TgEvalCallback (tg_eval_callback.py:650-653):
        #   arange(T) * (t_rate / base_fps), broadcast to (B, T).
        time_unit = float(self.t_rate) / float(self.base_fps)
        real_timestep = (
            torch.arange(T, dtype=torch.float32, device=device) * time_unit
        ).unsqueeze(0).expand(B, -1).contiguous()  # (B, T)
        group_id = torch.zeros(B, P, dtype=torch.long, device=device)
        sem_type = torch.full(
            (B, P), fill_value=int(SEM_REAL_SCENE), dtype=torch.long, device=device,
        )
        # iter 04b FIX: rot_mask must match training. With dummy_id_rot=True
        # (set in dataset YAML), training datamodule sets rot_mask=ones for
        # every trajectory (rotation channels carry identity, supervised).
        # Hardcoding zeros here zeroed the rotation channels at viz time —
        # an OOD distribution the model never saw — so predictions in canon
        # space drift toward zero (= scene mean after de-canonicalization).
        # Mirror TgEvalCallback (tg_eval_callback.py:661-664).
        if self.dummy_id_rot:
            rot_mask = torch.ones(B, P, dtype=torch.bool, device=device)
        else:
            rot_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        images = batch["extra"]["images"].to(device)
        view_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        pixel_uv = batch["extra"]["first_keypoint"].to(device)
        pixel_view_idx = torch.zeros(B, P, dtype=torch.long, device=device)
        pixel_query_mask = gt_mask

        eval_model = _select_eval_model(pl_module)
        eval_model.eval()

        common_kwargs = dict(
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

        with torch.inference_mode():
            _ret_context, ret_trace = eval_model.inference_sliding_window(
                K_train=self.K_train,
                anchor_count=self.anchor_count,
                **common_kwargs,
                **self.inference_kwargs,
            )

        pred_abs = ret_trace[..., :3, 3]                            # (1, 33, 400, 3)
        return pred_abs

    # ── Render ─────────────────────────────────────────────────────────

    @rank_zero_only
    def _render(self, trainer, pl_module, step: int) -> None:
        """Run inference + render MP4s for the cached reference sample."""
        from umm.models.wmm.viz_helpers.tg_viz import _sample_from_tg_batch

        batch = self._get_cached_batch()
        if batch is None:
            return  # cache permanently dead — see _get_cached_batch sentinel.

        # Latch advances only after the cache early-return passes. On rank>0
        # (DDP), @rank_zero_only short-circuits this entire method before the
        # latch update, so the cadence stays consistent across ranks.
        self._last_fired_step = step

        was_training = pl_module.training
        pl_module.eval()
        try:
            pred_abs = self._forward_sliding(pl_module, batch)
        finally:
            if was_training:
                pl_module.train()

        # Convert pred (1, T, P, 3) -> (P, T, 3) numpy for tg_viz consumers.
        pred_pt3 = pred_abs[0].detach().float().cpu().numpy()        # (T, P, 3)
        pred_trace_xyz = pred_pt3.transpose(1, 0, 2)                  # (P, T, 3)

        # Reconstruct sample dict (image + GT + masks) from collated batch.
        sample = _sample_from_tg_batch(batch, idx=0)
        if sample is None:
            logger.warning(
                "[Tg128SlidingViz] step=%d: _sample_from_tg_batch returned None; "
                "skipping render.", step,
            )
            return

        # Resolve viz output dir same way TgVisualizer does.
        log_dir = (
            getattr(trainer, "log_dir", None)
            or getattr(getattr(pl_module, "logger", None), "log_dir", None)
        )
        if not log_dir:
            log_dir = "./logs/tg_viz_out"
            logger.warning(
                f"[Tg128SlidingViz] log_dir unavailable; "
                f"writing viz to fallback {log_dir!r}"
            )
        save_root = os.path.join(log_dir, "tg_viz")
        os.makedirs(save_root, exist_ok=True)

        self._render_mp4s(pl_module, sample, pred_trace_xyz, save_root, step)

    def _render_mp4s(
        self,
        pl_module: pl.LightningModule,
        sample: Dict[str, Any],
        pred_trace_xyz: np.ndarray,
        save_root: str,
        step: int,
    ) -> None:
        """Render 2D-flow + (optional) 3D-depth MP4s for GT and pred.

        Mirrors ``TgVisualizer.tg_viz_routine`` MP4 logic but writes files
        with the ``K400sw`` suffix. Each MP4 is also logged to wandb via
        ``pl_module.logger.log_video`` (same call signature as the standard
        ``tg_viz`` MP4 path -- see
        ``src/umm/models/wmm/viz_helpers/tg_viz.py`` ~lines 1042/1097), so
        K400sw videos appear in the wandb media panel alongside the regular
        K=128 viz at every fire. The wandb call is wrapped in a defensive
        try/except: PL disables loggers under ``fast_dev_run`` (logger may
        be ``None`` / lack ``log_video``), and a logger failure must not
        prevent the local MP4 (or kill training under ``viz_try``).
        """
        from umm.models.wmm.viz_helpers.tg_viz import (
            _to_numpy_req,
            _to_numpy,
        )
        from umm.models.wmm.viz_helpers.tg_viz_3d import (
            backproject_trace_tg,
            backproject_depth_tg,
            render_tg_2d_flow,
            render_tg_3d_depth,
        )

        try:
            import imageio.v2 as _imageio
        except Exception:
            import imageio as _imageio

        # Unpack the GT-side fields from sample (mirrors TgVisualizer).
        trace_np = (
            sample["trace"].detach().cpu().numpy()
            if hasattr(sample["trace"], "detach")
            else np.asarray(sample["trace"])
        )
        gt_xyz_pt3 = trace_np[:, :, :3, 3]                            # (P, T, 3)
        trace_mask_pt3 = (
            sample["trace_mask"].detach().cpu().numpy()
            if hasattr(sample["trace_mask"], "detach")
            else np.asarray(sample["trace_mask"])
        )
        trace_valid_pt = trace_mask_pt3.any(axis=-1).astype(bool)     # (P, T)
        first_kp_p2 = _to_numpy_req(sample["first_keypoint"])         # (P, 2)
        gt_mask_p = _to_numpy_req(sample["gt_mask"]).astype(bool)     # (P,)
        depth_raw_np = _to_numpy(sample.get("depth_raw"))             # or None

        img = _to_numpy_req(sample["image"])
        if img.ndim == 4:
            img = img[0]
        anchor_rgb = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

        drawn_idx = np.where(gt_mask_p)[0].astype(np.int32)
        gt_tp3 = gt_xyz_pt3.transpose(1, 0, 2)                        # (T, P, 3)
        pred_tp3 = pred_trace_xyz.transpose(1, 0, 2)                  # (T, P, 3)
        valid_tp = trace_valid_pt.T                                   # (T, P)

        env_name = self.env_name

        def _log_wandb_video(stack_thwc: np.ndarray, log_key: str,
                             caption: str) -> None:
            """Log a (T, H, W, C) uint8 stack to wandb via CustomLogger.log_video.

            Mirrors the call pattern in ``tg_viz.py`` ~lines 1042/1097:
            cast to (T, 3, H, W) float in [0, 1], then call
            ``pl_module.logger.log_video(phase, key, video_t3hw, gstep, batch_idx, caption)``.
            See ``src/umm/training_utils/log_helpers.py:388`` -- the
            ``log_video`` method handles wandb upload + tb + local copy.

            Defensive: catch all errors. PL's ``fast_dev_run`` may set the
            logger to a no-op that lacks ``log_video``; viz must never kill
            training (this whole callback is wrapped in try/except in
            ``on_train_batch_end`` too).
            """
            try:
                lg = getattr(pl_module, "logger", None)
                if lg is None or not hasattr(lg, "log_video"):
                    logger.info(
                        "[Tg128SlidingViz] step=%d: pl_module.logger has no "
                        "log_video (likely fast_dev_run); skipping wandb log "
                        "for key=%r.", step, log_key,
                    )
                    return
                vid_t = (
                    torch.from_numpy(stack_thwc).float().permute(0, 3, 1, 2)
                    / 255.0
                )
                lg.log_video(
                    "train",
                    log_key,
                    vid_t,
                    step,
                    0,  # batch_idx -- the cached single-sample reference
                    caption=caption,
                )
            except Exception as e:  # noqa: BLE001 -- viz must never break train
                logger.warning(
                    "[Tg128SlidingViz] step=%d: wandb log failed for "
                    "key=%r: %r (mp4 still on disk).",
                    step, log_key, e,
                )

        # ---- 2D flow MP4s ----
        def _save_2d_flow(mode: str, save_suffix: str,
                          gt_in: Optional[np.ndarray],
                          pred_in: Optional[np.ndarray]) -> str:
            frames = render_tg_2d_flow(
                anchor_rgb=anchor_rgb,
                gt_trace_tp3=gt_tp3 if gt_in is None else gt_in,
                pred_trace_tp3=pred_in,
                trace_valid_tp=valid_tp,
                first_kp_p2=first_kp_p2,
                drawn_idx=drawn_idx,
                image_size=self.image_size,
                known_n=1,
                mode=mode,
            )
            stack_thwc = np.stack(frames, axis=0)
            local_path = os.path.join(
                save_root,
                f"flow_2d_{env_name}_K400sw_{step}_{save_suffix}.mp4",
            )
            _imageio.mimwrite(local_path, list(stack_thwc), fps=10)
            _log_wandb_video(
                stack_thwc,
                f"tg_viz/flow_2d_{env_name}_K400sw_{save_suffix}",
                caption=f"K400sw env={env_name} gs={step} {save_suffix}",
            )
            return local_path

        flow_gt_path = _save_2d_flow("gt", "gt", gt_tp3, None)
        flow_pred_path = _save_2d_flow("pred", "pred", gt_tp3, pred_tp3)

        # ---- 3D depth MP4s (only when depth available) ----
        depth_gt_path = None
        depth_pred_path = None
        if depth_raw_np is not None:
            gt_world_tp3 = backproject_trace_tg(
                gt_xyz_pt3, image_size=self.image_size
            )
            pred_world_tp3 = backproject_trace_tg(
                pred_trace_xyz, image_size=self.image_size
            )
            depth_pcl = backproject_depth_tg(
                depth_raw_np, anchor_rgb=anchor_rgb, image_size=self.image_size
            )

            def _save_3d_depth(mode: str, save_suffix: str,
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
                local_path = os.path.join(
                    save_root,
                    f"depth_3d_{env_name}_K400sw_{step}_{save_suffix}.mp4",
                )
                _imageio.mimwrite(local_path, list(stack_thwc), fps=10)
                _log_wandb_video(
                    stack_thwc,
                    f"tg_viz/depth_3d_{env_name}_K400sw_{save_suffix}",
                    caption=f"K400sw env={env_name} gs={step} {save_suffix}",
                )
                return local_path

            depth_gt_path = _save_3d_depth("gt", "gt", None)
            depth_pred_path = _save_3d_depth("pred", "pred", pred_world_tp3)

        logger.info(
            f"[Tg128SlidingViz] step={step}: wrote MP4s "
            f"2d_gt={flow_gt_path}, 2d_pred={flow_pred_path}, "
            f"3d_gt={depth_gt_path}, 3d_pred={depth_pred_path}"
        )
