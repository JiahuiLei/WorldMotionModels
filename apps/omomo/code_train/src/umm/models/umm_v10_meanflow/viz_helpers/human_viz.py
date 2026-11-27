"""V10 visualization for human-humanoid (HH) domain.

Delegates mesh rendering to V9's HumanVisualizer._viz_hh_subroutine.
Handles V10-specific mask construction and inference calls.

Viz modes: GT, H2R, R2H, half_future, infilling.

Example::

    viz = V10HumanViz(pl_module, human_cfg, humanoid_cfg)
    viz.viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging

import torch

from umm.datasets.umm_v10.dataset_config import (
    SEM_HUMAN,
    SEM_HUMAN_GLOBAL,
    SEM_HUMANOID,
    SEM_HUMANOID_GLOBAL,
)
from umm.models.umm_v9.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_v9.viz_helpers.utils import make_viz_T_cam
from umm.models.umm_v10.viz_helpers.base_viz import run_v10_inference

logger = logging.getLogger(__name__)

HUMAN_SEMS = [SEM_HUMAN, SEM_HUMAN_GLOBAL]
ROBOT_SEMS = [SEM_HUMANOID, SEM_HUMANOID_GLOBAL]


class V10HumanViz:
    """V10 human-humanoid visualizer.

    Wraps V9 HumanVisualizer for mesh rendering while using V10-specific
    mask patterns and inference calls.

    Args:
        pl_module: V10 PLModel instance.
        human_cfg: Config dict for human body model.
        humanoid_cfg: Config dict for humanoid robot meshes.
    """

    def __init__(self, pl_module, human_cfg, humanoid_cfg):
        self.pl = pl_module
        # V9 renderer handles mesh loading + frame rendering + logging
        self.renderer = HumanVisualizer(pl_module, human_cfg, humanoid_cfg)
        self.human_flag = self.renderer.human_flag
        self.humanoid_flag = self.renderer.humanoid_flag
        self.humanhumanoid_viz_cfg = pl_module.cfg.get("visualize", {}).get(
            "humanhumanoid", {}
        )

    def _find_hh_bid(self, batch):
        """Find first batch item with both human and humanoid entities.

        Uses batch["extra"] humanoid_type/human_model_type (same logic as V9).
        Falls back to sem_type check if extra keys are missing.
        """
        B = len(batch["trace"])
        bid_both = None
        bid_any = None
        for i in range(B):
            has_human = (
                "human_model_type" in batch.get("extra", {})
                and batch["extra"]["human_model_type"][i] is not None
                and "smplx" in batch["extra"]["human_model_type"][i]
            )
            has_humanoid = False
            if (
                "humanoid_type" in batch.get("extra", {})
                and batch["extra"]["humanoid_type"][i] is not None
            ):
                for ht in batch["extra"]["humanoid_type"][i]:
                    if ht is not None:
                        has_humanoid = True
                        break
            if has_human and has_humanoid:
                bid_both = i
                break
            if bid_any is None and (has_human or has_humanoid):
                bid_any = i
        return bid_both if bid_both is not None else bid_any

    def _sem_entity_mask(self, sem_type, sem_ids):
        """Return (P,) bool mask for entities matching any sem_id."""
        mask = torch.zeros_like(sem_type, dtype=torch.bool)
        for sid in sem_ids:
            mask |= (sem_type == sid)
        return mask

    def _render_and_log(
        self, gen_trace, gen_context, batch, bid, phase, batch_idx,
        viz_seen_step, name,
    ):
        """Render HH frames using V9's _viz_hh_subroutine and log to wandb.

        Delegates entirely to V9 renderer which handles:
        - batch["extra"] extraction (human_uids, humanoid_uids, gid, beta, etc.)
        - SMPL/SMPLX mesh animation
        - Humanoid link mesh rendering
        - Blended overlay
        - Logging to wandb
        """
        viz_T_cam = make_viz_T_cam(self.humanhumanoid_viz_cfg, batch)
        n_steps = batch["trace_mask"][bid][:, 0].sum().item()
        dataset_trace_mask = batch["trace_mask"][bid]
        dataset_ctx_mask = batch["context_mask"][bid]

        self.renderer._viz_hh_subroutine(
            gen_trace=gen_trace[:n_steps] if gen_trace is not None else None,
            gen_context=gen_context,
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=viz_seen_step,
            viz_trace_mask=dataset_trace_mask,
            viz_ctx_mask=dataset_ctx_mask,
            human_viz_name=f"HH_human_{name}",
            humanoid_viz_name=f"HH_robot_{name}",
            compose_viz_name=f"HH_blend_{name}",
            context_viz_name=f"HH_ctx_{name}",
        )

    def viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """V10 HH visualization routine.

        Renders GT, H2R, R2H, half_future, and infilling modes.

        Args:
            batch: Full training batch dict.
            phase: "train" or "val".
            batch_idx: Batch index for logging.
            eval_model: AgentV10 (or EMA copy) for inference.
            step_output: Training step output dict.
        """
        if not (self.human_flag or self.humanoid_flag):
            return

        bid = self._find_hh_bid(batch)
        if bid is None:
            logger.warning(f"phase={phase} batch_idx={batch_idx}: no HH data, skip viz")
            return

        logger.warning(f"V10 HH viz: phase={phase} batch_idx={batch_idx} bid={bid}")

        n_steps = int(batch["trace_mask"][bid][:, 0].sum().item())
        sem_type = batch["trace_sem_type"][bid]  # (P,)

        # --- GT ---
        self._render_and_log(
            gen_trace=batch["trace"][bid],
            gen_context=batch["context"][bid],
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=n_steps, name=f"GT_{phase}",
        )

        # --- H2R: human known, robot unknown ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        robot_mask = self._sem_entity_mask(sem_type, ROBOT_SEMS)  # (P,)
        trace_known[0, :, robot_mask] = False
        h2r_result = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
            return_intermediates=True,
        )
        gen_trace = h2r_result["trace"]
        self._render_and_log(
            gen_trace=gen_trace[0], gen_context=None,
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=0, name=f"H2R_{phase}",
        )
        # Render intermediate X0 predictions
        for inter in h2r_result.get("intermediates", []):
            if inter["pct"] < 100:  # skip 100% — already rendered above
                self._render_and_log(
                    gen_trace=inter["trace"][0], gen_context=None,
                    batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
                    viz_seen_step=0,
                    name=f"H2R_x0_{inter['pct']}pct_step{inter['step']}of{inter['total']}_{phase}",
                )

        # --- R2H: robot known, human unknown ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        human_mask = self._sem_entity_mask(sem_type, HUMAN_SEMS)
        trace_known[0, :, human_mask] = False
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._render_and_log(
            gen_trace=gen_trace[0], gen_context=None,
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=0, name=f"R2H_{phase}",
        )

        # --- Half future: first T/2 known ---
        trace_known = batch["trace_mask"][bid:bid + 1].clone().bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        trace_known[0, n_steps // 2:] = False
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._render_and_log(
            gen_trace=gen_trace[0], gen_context=None,
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=n_steps // 2, name=f"HalfFuture_{phase}",
        )

        # --- Infilling: first + last frame known ---
        trace_known = torch.zeros_like(batch["trace_mask"][bid:bid + 1]).bool()
        ctx_known = batch["context_mask"][bid:bid + 1].clone().bool()
        trace_known[0, 0] = batch["trace_mask"][bid, 0].bool()
        if n_steps > 1:
            trace_known[0, n_steps - 1] = batch["trace_mask"][bid, n_steps - 1].bool()
        gen_ctx, gen_trace = run_v10_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._render_and_log(
            gen_trace=gen_trace[0], gen_context=None,
            batch=batch, bid=bid, phase=phase, batch_idx=batch_idx,
            viz_seen_step=[0, n_steps - 1], name=f"Infilling_{phase}",
        )
