"""V11 human/humanoid visualization — thin adapter over V9 viz.

Inherits all rendering from V9 HumanVisualizer. Overrides only the
viz routine to use V11's inference() signature.
"""

import logging

from umm.models.umm_v9.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_v9.viz_helpers.utils import make_viz_T_cam

logger = logging.getLogger(__name__)


class HumanVisualizerV11(HumanVisualizer):
    """V11 adapter: same rendering, V11 inference signature."""

    def _v11_inference(self, eval_model, batch, bid, trace_known_mask,
                       context_known_mask, **kwargs):
        """Call V11 inference with local_pcl/local_pcl_mask."""
        return eval_model.inference(
            trace=batch["trace"][bid:bid + 1],
            context=batch["context"][bid:bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid:bid + 1],
            context_mask=batch["context_mask"][bid:bid + 1],
            real_timestep=batch["time"][bid:bid + 1],
            group_id=batch["trace_group_id"][bid:bid + 1],
            sem_type=batch["trace_sem_type"][bid:bid + 1],
            rot_mask=batch["trace_rot_mask"][bid:bid + 1],
            text=[batch["text"][bid]],
            local_pcl=batch["local_pcl"][bid:bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
            **kwargs,
        )

    def human_humanoid_viz_routine(
        self, step_output, batch, phase, batch_idx, eval_model, lite=True
    ):
        """V11 viz routine: GT + pred-x0 + H2R + R2H + future prediction.

        Same structure as V9 but uses V11 inference() signature.
        """
        if not (self.human_flag or self.humanoid_flag):
            return

        # Find best batch item (prefer human+humanoid)
        B = len(batch["trace"])
        bid_both = None
        bid_any = None
        for i in range(B):
            has_human = (
                "human_model_type" in batch["extra"]
                and batch["extra"]["human_model_type"][i] is not None
                and "smplx" in batch["extra"]["human_model_type"][i]
            )
            has_humanoid = False
            if (
                "humanoid_type" in batch["extra"]
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
        bid = bid_both if bid_both is not None else bid_any
        if bid is None:
            logger.warning(
                f"{phase} batch_idx={batch_idx}: no human/humanoid, skip viz"
            )
            return

        viz_T_cam = make_viz_T_cam(self.humanhumanoid_viz_cfg, batch)
        dataset_trace_mask = batch["trace_mask"][bid]   # (T, P)
        dataset_ctx_mask = batch["context_mask"][bid]    # (C, P)
        n_steps = dataset_trace_mask[:, 0].sum()

        # --- GT ---
        self._viz_hh_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            gen_context=batch["context"][bid],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_trace_mask=dataset_trace_mask,
            viz_ctx_mask=dataset_ctx_mask,
            human_viz_name=f"GT_human_{phase}",
            humanoid_viz_name=f"GT_humanoid_{phase}",
            compose_viz_name=f"GT_blended_{phase}",
            context_viz_name=f"GT_context_{phase}",
        )

        # --- Pred-X0 (from training step output) ---
        if not lite and step_output is not None:
            if "x0_pred_trace" in step_output and "x0_pred_context" in step_output:
                self._viz_hh_subroutine(
                    gen_trace=step_output["x0_pred_trace"][bid, :n_steps].float(),
                    gen_context=step_output["x0_pred_context"][bid].float(),
                    viz_T_cam=viz_T_cam,
                    batch=batch,
                    phase=phase,
                    bid=bid,
                    batch_idx=batch_idx,
                    viz_seen_step=n_steps,
                    viz_trace_mask=dataset_trace_mask,
                    viz_ctx_mask=dataset_ctx_mask,
                    human_viz_name=f"PREDX0_human_{phase}",
                    humanoid_viz_name=f"PREDX0_humanoid_{phase}",
                    compose_viz_name=f"PREDX0_blended_{phase}",
                    context_viz_name=f"PREDX0_context_{phase}",
                )

        extra_dict = {k: v[bid] for k, v in batch["extra"].items()}

        # --- H2R (human known → generate robot) ---
        _h2r_ok = (
            "human_model_type" in extra_dict
            and extra_dict["human_model_type"] is not None
            and "smplx" in extra_dict["human_model_type"]
        )
        if _h2r_ok:
            _idx = extra_dict["human_model_type"].index("smplx")
            human_mask = eval_model.get_type_mask(
                batch["trace_uid"][bid], extra_dict["human_uids"][_idx]
            )
            human_mask = human_mask & eval_model.get_type_mask(
                batch["trace_group_id"][bid], extra_dict["gid"][_idx]
            )
            trace_known = dataset_trace_mask[None].bool().clone()
            ctx_known = dataset_ctx_mask[None].bool().clone()
            trace_known[0, :, ~human_mask] = False

            gen_context, gen_trace = self._v11_inference(
                eval_model, batch, bid, trace_known, ctx_known,
            )
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=None,
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                human_viz_name=f"H2R_human_gt_{phase}",
                humanoid_viz_name=f"H2R_humanoid_retargeted_{phase}",
                compose_viz_name=f"H2R_blended_{phase}",
            )

        # --- R2H (robot known → generate human) ---
        humanoid_mask = None
        if "humanoid_type" in extra_dict and extra_dict["humanoid_type"] is not None:
            for i in range(len(extra_dict["humanoid_type"])):
                if extra_dict["humanoid_type"][i] is not None:
                    humanoid_mask = eval_model.get_type_mask(
                        batch["trace_uid"][bid], extra_dict["humanoid_uids"][i]
                    )
                    humanoid_mask = humanoid_mask & eval_model.get_type_mask(
                        batch["trace_group_id"][bid], extra_dict["gid"][i]
                    )
                    break

        if humanoid_mask is not None:
            trace_known = dataset_trace_mask[None].bool().clone()
            ctx_known = dataset_ctx_mask[None].bool().clone()
            trace_known[0, :, ~humanoid_mask] = False

            gen_context, gen_trace = self._v11_inference(
                eval_model, batch, bid, trace_known, ctx_known,
            )
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=None,
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                human_viz_name=f"R2H_human_retargeted_{phase}",
                humanoid_viz_name=f"R2H_humanoid_gt_{phase}",
                compose_viz_name=f"R2H_blended_{phase}",
            )

        # --- Half future prediction ---
        trace_known = dataset_trace_mask[None].bool().clone()
        ctx_known = dataset_ctx_mask[None].bool().clone()
        trace_known[0, n_steps // 2:] = False

        gen_context, gen_trace = self._v11_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_hh_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=None,
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            human_viz_name=f"FuturePred_human_seen_half_{phase}",
            humanoid_viz_name=f"FuturePred_humanoid_seen_half_{phase}",
            compose_viz_name=f"FuturePred_blended_seen_half_{phase}",
        )

        # --- 1-step future prediction ---
        trace_known = dataset_trace_mask[None].bool().clone()
        ctx_known = dataset_ctx_mask[None].bool().clone()
        trace_known[:, 1:] = False

        gen_context, gen_trace = self._v11_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_hh_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=None,
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=1,
            human_viz_name=f"FuturePred_human_seen_1_{phase}",
            humanoid_viz_name=f"FuturePred_humanoid_seen_1_{phase}",
            compose_viz_name=f"FuturePred_blended_seen_1_{phase}",
        )
