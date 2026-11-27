"""V10 visualization for TACO (hand-object interaction) domain.

Delegates mesh rendering to V9's TacoVisualizer. Handles V10-specific
mask construction and inference calls.

Viz modes: GT, O2H, H2O, half_future.

Example::

    viz = V10TacoViz(pl_module, taco_cfg)
    viz.viz_routine(batch, "val", 0, eval_model)
"""

import logging

import numpy as np
import torch

from umm.datasets.umm_v10.dataset_config import (
    SEM_LEFT_MANO,
    SEM_LEFT_MANO_FINGERS,
    SEM_LEFT_MANO_WRIST,
    SEM_OBJECT,
    SEM_RIGHT_MANO,
    SEM_RIGHT_MANO_FINGERS,
    SEM_RIGHT_MANO_WRIST,
    UID_MANO_LEFT_HAND,
    UID_MANO_RIGHT_HAND,
)
from umm.models.umm_v10.agent import AgentV10
from umm.models.umm_v9.viz_helpers.taco_viz import TacoVisualizer
from umm.models.umm_v9.viz_helpers.utils import make_viz_T_cam
from umm.models.umm_v10.viz_helpers.base_viz import run_v10_inference

logger = logging.getLogger(__name__)

HAND_SEMS = [
    SEM_LEFT_MANO, SEM_RIGHT_MANO,
    SEM_LEFT_MANO_WRIST, SEM_LEFT_MANO_FINGERS,
    SEM_RIGHT_MANO_WRIST, SEM_RIGHT_MANO_FINGERS,
]
OBJECT_SEMS = [SEM_OBJECT]


class V10TacoViz:
    """V10 TACO (hand-object) visualizer.

    Wraps V9 TacoVisualizer for mesh rendering while using V10-specific
    mask patterns and inference calls.

    Args:
        pl_module: V10 PLModel instance.
        taco_cfg: Config dict for TACO viz (model_path, mesh_fn).
    """

    def __init__(self, pl_module, taco_cfg):
        self.pl = pl_module
        self.renderer = TacoVisualizer(pl_module, taco_cfg)
        self.flag = self.renderer.flag
        self.taco_viz_cfg = pl_module.cfg.get("visualize", {}).get("taco", {})

    def _find_taco_bid(self, batch):
        """Find first batch item with both hands and objects."""
        for i in range(len(batch["trace"])):
            uid = batch["trace_uid"][i]
            l_mask = AgentV10.get_type_mask(uid, UID_MANO_LEFT_HAND)
            r_mask = AgentV10.get_type_mask(uid, UID_MANO_RIGHT_HAND)
            has_tool = "tool_id" in batch.get("extra", {})
            has_target = "target_id" in batch.get("extra", {})
            if l_mask.any() and r_mask.any() and (has_tool or has_target):
                return i
        return None

    def _sem_entity_mask(self, sem_type, sem_ids):
        """Return (P,) bool mask for entities matching any sem_id."""
        mask = torch.zeros_like(sem_type, dtype=torch.bool)
        for sid in sem_ids:
            mask |= (sem_type == sid)
        return mask

    def _viz_subroutine(self, gen_trace, gen_context, batch, bid, phase,
                        batch_idx, viz_seen_step, name):
        """Render TACO frames via V9 renderer and log."""
        n_steps = batch["trace_mask"][bid][:, 0].sum().item()
        viz_T_cam = make_viz_T_cam(self.taco_viz_cfg, batch)
        extra = batch.get("extra", {})

        # Render with meshes (if tool/target IDs available)
        taco_frames = self.renderer.viz_taco(
            batch["trace_uid"][bid],
            gen_trace[:n_steps],
            viz_T_cam,
            viz_seen_step,
            l_beta=extra.get("beta_l", [None])[bid],
            r_beta=extra.get("beta_r", [None])[bid],
            tool_id=extra.get("tool_id", [None])[bid] if "tool_id" in extra else None,
            target_id=extra.get("target_id", [None])[bid] if "target_id" in extra else None,
            ref_T_wl=gen_context if gen_context is not None else None,
        )
        self.pl.logger.log_video(
            phase, f"TACO_{name}",
            torch.from_numpy(np.stack(taco_frames, 0))[..., :3]
            .float().permute(0, 3, 1, 2) / 255.0,
            self.pl.global_step, batch_idx,
        )

    def viz_routine(self, batch, phase, batch_idx, eval_model):
        """V10 TACO visualization routine.

        Renders GT, half_future, O2H, H2O modes.

        Args:
            batch: Full training batch dict.
            phase: "train" or "val".
            batch_idx: Batch index for logging.
            eval_model: AgentV10 (or EMA copy) for inference.
        """
        if not self.flag:
            return

        bid = self._find_taco_bid(batch)
        if bid is None:
            logger.warning(f"phase={phase} batch_idx={batch_idx}: no TACO data, skip viz")
            return

        logger.warning(f"V10 TACO viz: phase={phase} batch_idx={batch_idx} bid={bid}")

        n_steps = batch["trace_mask"][bid][:, 0].sum().item()
        sem_type = batch["trace_sem_type"][bid]  # (P,)

        # --- GT ---
        self._viz_subroutine(
            gen_trace=batch["trace"][bid],
            gen_context=batch["context"][bid],
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=n_steps, name=f"GT_{phase}",
        )

        # --- Half future ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        trace_known[0, n_steps // 2:] = False
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_subroutine(
            gen_trace=gen_trace[0], gen_context=gen_ctx[0],
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=n_steps // 2, name=f"HalfFuture_{phase}",
        )

        # --- O2H: objects known, hands unknown ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        hand_mask = self._sem_entity_mask(sem_type, HAND_SEMS)
        trace_known[0, :, hand_mask] = False
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_subroutine(
            gen_trace=gen_trace[0], gen_context=gen_ctx[0],
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=0, name=f"O2H_{phase}",
        )

        # --- H2O: hands known, objects unknown ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        obj_mask = self._sem_entity_mask(sem_type, OBJECT_SEMS)
        trace_known[0, :, obj_mask] = False
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_subroutine(
            gen_trace=gen_trace[0], gen_context=gen_ctx[0],
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=0, name=f"H2O_{phase}",
        )
