"""Tg128EvalCallback — sliding-window TG eval for K_train != P_test.

The K=128 DROID configs train on
windows of K=128 trajectories per sample but the TraceGen test set
serves P=400 trajectories per sample. This subclass overrides ONLY
``_forward_batch`` to dispatch to ``inference_sliding_window`` when
``K_train != P_test`` and to the parent's ``inference`` path otherwise
(K_train == P).

The metric path, dataset construction, filter, and wandb logging are all
inherited from :class:`TgEvalCallback` byte-identically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch

from umm.training_utils.callbacks.tg_eval_callback import (
    TgEvalCallback,
    _select_eval_model,
)

logger = logging.getLogger(__name__)


class Tg128EvalCallback(TgEvalCallback):
    """TgEvalCallback variant that runs sliding-window inference.

    Overrides only :meth:`_forward_batch`. All other behaviour
    (cadence, dataset, filter, metric path, wandb keys) is inherited
    from :class:`TgEvalCallback`.

    Args:
        K_train: Trained window width. Required.
        anchor_count: Anchor token count for sub-steps after sub-step 0.
            Defaults to ``K_train // 2`` when ``None``.
        *args, **kwargs: Forwarded to :class:`TgEvalCallback`.
    """

    def __init__(
        self,
        *args: Any,
        K_train: int,
        anchor_count: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # Guard: K_train / anchor_count must NOT also appear inside
        # inference_kwargs, otherwise the dispatch site below would raise
        # a duplicate-keyword TypeError. Caller intent is ambiguous —
        # fail loud (review notes #1).
        for _bad_kw in ("K_train", "anchor_count"):
            if _bad_kw in self.inference_kwargs:
                raise ValueError(
                    f"Tg128EvalCallback: inference_kwargs must not include "
                    f"{_bad_kw!r}; pass it as a constructor kwarg instead."
                )
        self.K_train = int(K_train)
        if self.K_train <= 0:
            raise ValueError(
                f"Tg128EvalCallback: K_train must be a positive int, "
                f"got {K_train!r}"
            )
        if anchor_count is None:
            self.anchor_count = self.K_train // 2
        else:
            self.anchor_count = int(anchor_count)
        if not (0 <= self.anchor_count < self.K_train):
            raise ValueError(
                f"Tg128EvalCallback: anchor_count must be in [0, K_train), "
                f"got anchor_count={self.anchor_count}, K_train={self.K_train}"
            )
        logger.info(
            f"[Tg128EvalCallback] init: K_train={self.K_train}, "
            f"anchor_count={self.anchor_count}"
        )

    def _forward_batch(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
    ):
        """Sliding-window forward — same outputs as parent ``_forward_batch``.

        Mirrors the parent implementation EXCEPT:

        * The ``P == 400`` assertion is dropped (P can be anything).
        * Dispatches to ``eval_model.inference_sliding_window(...)`` when
          ``K_train != P``; otherwise falls through to the parent's
          ``inference()`` path (so ``K=400`` configs that route through
          this subclass remain byte-identical to the parent).
        """
        from umm.datasets.wmm.dataset_config import SEM_REAL_SCENE

        device = pl_module.device
        trace = batch["trace"].to(device)                      # (B, 33, P, 4, 4)
        trace_mask_any = batch["trace_mask"].to(device)        # (B, 33, P) bool
        mask_pc = batch["extra"]["trace_valid_mask_per_coord"].to(device)  # (B,33,P,3)

        B, T, P, _, _ = trace.shape
        assert T == 33, f"Tg128EvalCallback: bad T: {T} (expected 33)"

        context = trace[:, :1].clone()                          # (B, 1, P, 4, 4)
        trace_known = torch.zeros(B, T, P, dtype=torch.bool, device=device)
        trace_known[:, 0] = True
        ctx_known = torch.ones(B, 1, P, dtype=torch.bool, device=device)

        gt_mask = batch["extra"]["gt_mask"].to(device)           # (B, P) bool
        ctx_mask = gt_mask.unsqueeze(1)                          # (B, 1, P)

        # CRITICAL FIX (sw-viz-debug): per-frame timestamps must NOT be all
        # zeros — the LMM multiplies by time_multiplier=10 to build temporal
        # RoPE phase. Zero timestamps → all 33 frames receive the same time
        # phase → model cannot differentiate frames → predicts the same value
        # at every frame → after de-canon all tracks collapse to scene mean.
        # Mirror the parent TgEvalCallback formula (tg_eval_callback.py:650-
        # 653): arange(T) * (t_rate / base_fps), broadcast to (B, T).
        # self.t_rate / self.base_fps are inherited via TgEvalCallback init.
        time_unit = float(self.t_rate) / float(self.base_fps)
        real_timestep = (
            torch.arange(T, dtype=torch.float32, device=device) * time_unit
        ).unsqueeze(0).expand(B, -1).contiguous()  # (B, T)
        group_id = torch.zeros(B, P, dtype=torch.long, device=device)
        sem_type = torch.full(
            (B, P), fill_value=int(SEM_REAL_SCENE), dtype=torch.long, device=device,
        )
        # iter 04b FIX: use self.dummy_id_rot (inherited from TgEvalCallback)
        # to mirror training. Hardcoded zeros zeroed the rotation channels at
        # eval time → OOD distribution → predictions drift to canon zero (=
        # scene mean). Mirror parent (tg_eval_callback.py:661-664).
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
            if int(self.K_train) != int(P):
                ret_context, ret_trace = eval_model.inference_sliding_window(
                    K_train=self.K_train,
                    anchor_count=self.anchor_count,
                    **common_kwargs,
                    **self.inference_kwargs,
                )
            else:
                ret_context, ret_trace = eval_model.inference(
                    **common_kwargs,
                    **self.inference_kwargs,
                )

        pred_abs = ret_trace[..., :3, 3]
        gt_abs = trace[..., :3, 3]
        return pred_abs, gt_abs, mask_pc
