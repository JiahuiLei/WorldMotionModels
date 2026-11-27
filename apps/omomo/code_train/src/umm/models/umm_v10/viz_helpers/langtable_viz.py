"""V10 visualization for Language Table tabletop manipulation domain.

Delegates mesh/PCL rendering to V9's LangTableVisualizer. Handles V10-specific
mask construction and inference calls.

Viz modes: GT, x0-pred, future_with_goal, future_without_goal.

Example::

    viz = V10LangTableViz(pl_module, langtable_cfg)
    viz.viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging

import numpy as np
import torch

from umm.datasets.umm_v10.dataset_config import (
    UID_LANGTABLE_OBJECT_GROUP,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
)
from umm.models.umm_v9.viz_helpers.langtable_viz import LangTableVisualizer
from umm.models.umm_v9.viz_helpers.utils import (
    make_square_cat_image,
    put_caption_at_bottom,
)
from umm.models.umm_v10.agent import AgentV10
from umm.models.umm_v10.viz_helpers.base_viz import run_v10_inference

logger = logging.getLogger(__name__)

ALL_LT_UIDS = UID_LANGTABLE_OBJECT_GROUP + [UID_LANGTABLE_EE, UID_LANGTABLE_ACT]


class V10LangTableViz:
    """V10 Language Table visualizer.

    Wraps V9 LangTableVisualizer for rendering while using V10-specific
    mask patterns and inference calls.

    Args:
        pl_module: V10 PLModel instance.
        langtable_cfg: Config dict for LangTable viz.
    """

    def __init__(self, pl_module, langtable_cfg):
        self.pl = pl_module
        self.renderer = LangTableVisualizer(pl_module, langtable_cfg)
        self.flag = self.renderer.flag
        self.langtable_viz_cfg = pl_module.cfg.get("visualize", {}).get(
            "langtable", {}
        )

    def _find_lt_bid(self, batch):
        """Find first batch item with LangTable entities."""
        for i in range(len(batch["trace"])):
            uid = batch["trace_uid"][i]
            mask = AgentV10.get_type_mask(uid, ALL_LT_UIDS)
            if mask.any():
                return i
        return None

    def _render_and_log(self, T_wl, batch, bid, phase, batch_idx, name,
                        known_n_real, seed, ref_T_wl=None):
        """Render LangTable frames and log to wandb."""
        viz_T_cam = self.renderer.make_viz_T_cam(batch)
        T, N = T_wl.shape[:2]

        # Build marking mask: True = prediction (marked differently)
        trace_marker_mask = torch.ones(T, N).bool()
        if isinstance(known_n_real, list):
            for idx in known_n_real:
                trace_marker_mask[idx] = False
        else:
            trace_marker_mask[:known_n_real] = False

        viz = self.renderer.render_viz_frames(
            prompt_type=batch["trace_uid"][bid],
            T_wl=T_wl,
            viz_T_cam=viz_T_cam,
            local_pcl=batch["local_pcl"][bid],
            local_pcl_mask=batch["local_pcl_mask"][bid],
            marking_mask=trace_marker_mask,
            H=360, W=640,
        )
        viz = put_caption_at_bottom(viz, batch["text"][bid])
        self.pl.logger.log_video(
            phase, f"LT_{name}_seed{seed}",
            torch.from_numpy(np.stack(viz, 0))[..., :3]
            .float().permute(0, 3, 1, 2) / 255.0,
            self.pl.global_step, batch_idx,
            caption=batch["text"][bid],
        )

        if ref_T_wl is not None:
            C_len = ref_T_wl.shape[0]
            ctx_marker = torch.ones(C_len, ref_T_wl.shape[1]).bool()
            viz_ctx = self.renderer.render_viz_frames(
                prompt_type=batch["trace_uid"][bid],
                T_wl=ref_T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                marking_mask=ctx_marker,
                H=360, W=640,
            )
            viz_img = make_square_cat_image(viz_ctx)[..., :3]
            self.pl.logger.log_image(
                phase, f"LT_ctx_{name}_seed{seed}",
                viz_img.transpose(2, 0, 1).astype(np.float32) / 255.0,
                self.pl.global_step, batch_idx,
            )

    def viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """V10 LangTable visualization routine.

        Renders GT, x0-pred, future_with_goal, future_without_goal.

        Args:
            batch: Full training batch dict.
            phase: "train" or "val".
            batch_idx: Batch index.
            eval_model: AgentV10 (or EMA copy) for inference.
            step_output: Training step output dict.
        """
        if not self.flag:
            return

        bid = self._find_lt_bid(batch)
        if bid is None:
            logger.warning(f"phase={phase} batch_idx={batch_idx}: no LT data, skip viz")
            return

        logger.warning(f"V10 LT viz: phase={phase} batch_idx={batch_idx} bid={bid}")

        n_steps = batch["trace_mask"][bid][:, 0].sum().item()

        # --- GT ---
        self._render_and_log(
            batch["trace"][bid].float().cpu(),
            batch, bid, phase, batch_idx, "GT",
            known_n_real=n_steps, seed="gt",
            ref_T_wl=batch["context"][bid].float().cpu(),
        )

        # --- x0-pred ---
        if "x0_pred_trace" in step_output:
            self._render_and_log(
                step_output["x0_pred_trace"][bid].float().cpu(),
                batch, bid, phase, batch_idx, "X0PRED",
                known_n_real=-1, seed=-1,
                ref_T_wl=step_output.get("x0_pred_context", batch["context"])[bid].float().cpu(),
            )

        # --- Future with goal: first steps known, context known ---
        for seed in [42, 1234]:
            trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
            ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
            # Determine number of steps with real timestamps (not padding)
            n_real = int((batch["time"][bid] <= 0).sum().item())
            if n_real <= 0:
                n_real = 1
            trace_known[0, n_real:] = False

            gen_ctx, gen_trace = run_v10_inference(
                eval_model, batch, bid, trace_known, ctx_known, seed=seed,
            )
            self._render_and_log(
                gen_trace[0].float().cpu(),
                batch, bid, phase, batch_idx, "Future(w.goal)",
                known_n_real=n_real, seed=seed,
            )

            # --- Future without goal: first steps known, context UNKNOWN ---
            ctx_known_none = torch.zeros_like(ctx_known).bool()
            gen_ctx, gen_trace = run_v10_inference(
                eval_model, batch, bid, trace_known, ctx_known_none, seed=seed,
            )
            self._render_and_log(
                gen_trace[0].float().cpu(),
                batch, bid, phase, batch_idx, "Future(no.goal)",
                known_n_real=n_real, seed=seed,
                ref_T_wl=gen_ctx[0].float().cpu(),
            )
