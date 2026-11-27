"""V11 human/humanoid visualization — thin adapter over V9 viz.

Inherits all rendering from V9 HumanVisualizer. Overrides only the
viz routine to use V11's inference() signature, and adds a "deploy
video" path that round-trips predicted link poses through the G1
kinematics chain (RK -> dof_pos -> FK) to visualise what a real G1
policy would execute.
"""

import logging

import torch

from umm.models.umm_v9.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_v9.viz_helpers.utils import make_viz_T_cam

try:
    from lib_twist.g1_interface import G1Interface
    _G1_INTERFACE_AVAILABLE = True
except ImportError:  # lib_twist optional; guard so training never crashes
    G1Interface = None
    _G1_INTERFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

_DEPLOY_VIZ_WARNED = False

# URDF path pattern used across UMM (see src/umm/sims/save_twist_rollout.py)
_G1_URDF_PATH = (
    "./ext/lib_twist/src/lib_twist/assets/g1/g1_custom_collision_with_fixed_hand.urdf"
)


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

    def _get_g1_interface(self, device):
        """Lazily build and cache a single G1Interface instance.

        The interface is allocated on CPU by default (matching
        ``save_twist_rollout``), which avoids moving the entire kinematic
        chain onto a training GPU just for viz. Call sites explicitly move
        tensors to ``self._g1_interface.device`` before invoking RK/FK.

        Args:
            device: Preferred device string. Currently ignored — interface
                always lives on CPU to keep viz cheap and crash-safe.
        """
        if not _G1_INTERFACE_AVAILABLE:
            return None
        if getattr(self, "_g1_interface", None) is None:
            # Force CPU: RK/FK is fast enough for viz sequences (T < ~200)
            # and avoids fighting the training GPU for memory.
            self._g1_interface = G1Interface(_G1_URDF_PATH, device="cpu")
        return self._g1_interface

    def _viz_hh_deploy_subroutine(
        self,
        gen_trace,           # (T, P, 4, 4) already-decoded SE(3) link poses
        viz_T_cam,
        batch,
        phase,
        bid,
        batch_idx,
        viz_seen_step,
        viz_trace_mask=None,
        human_viz_name=None,      # forwarded; deploy only renders humanoid
        humanoid_viz_name=None,
        compose_viz_name=None,    # forwarded but unused (no compose)
        context_viz_name=None,    # forwarded but unused
        deploy_tag="",
        eval_model=None,
    ):
        """Render a "deploy video": predicted link poses -> RK -> dof_pos -> FK.

        For every g1 humanoid in the batch item, this:
          1. Slices the robot's predicted link poses out of ``gen_trace``.
          2. Undoes ``local_global_trans_factor_flag`` if set so poses are
             in world frame.
          3. Calls ``G1Interface.RK`` to recover (root_pos, root_quat, dof_pos).
          4. Calls ``G1Interface.FK`` to rebuild world-frame link poses
             from that dof_pos + root.
          5. Re-applies ``local_global_trans_factor_flag`` encoding so the
             reconstructed slice matches the format the existing humanoid
             rendering path expects.
          6. Writes the reconstructed slice back into a COPY of ``gen_trace``
             and calls ``_viz_hh_subroutine`` with human/compose logging
             disabled so only the deploy humanoid video is logged.

        A deploy-viz failure MUST NEVER crash training: the entire body is
        wrapped in try/except and logs the error loudly, but returns. Also
        skipped silently (after a single warning) if ``lib_twist`` is not
        importable.
        """
        global _DEPLOY_VIZ_WARNED
        if not _G1_INTERFACE_AVAILABLE:
            if not _DEPLOY_VIZ_WARNED:
                logger.warning(
                    "deploy viz disabled: lib_twist.g1_interface not importable"
                )
                _DEPLOY_VIZ_WARNED = True
            return

        if gen_trace is None:
            return

        try:
            if "humanoid_type" not in batch["extra"]:
                return
            extra_dict = {k: v[bid] for k, v in batch["extra"].items()}
            humanoid_types = extra_dict.get("humanoid_type", None)
            if humanoid_types is None or len(humanoid_types) == 0:
                return

            g1i = self._get_g1_interface(gen_trace.device)
            if g1i is None:
                return

            local_trans_flag = self.humanhumanoid_viz_cfg.get(
                "local_global_trans_factor_flag", False
            )
            local_full_flag = self.humanhumanoid_viz_cfg.get(
                "local_global_factor_flag", False
            )

            gen_trace_deploy = gen_trace.clone()

            any_g1 = False
            for idx, ht in enumerate(humanoid_types):
                if ht is None or str(ht).lower() != "g1":
                    continue

                robot_uids = extra_dict["humanoid_uids"][idx]
                robot_gid = extra_dict["gid"][idx]
                link_names = extra_dict["humanoid_link_names"][idx]
                if robot_uids is None or link_names is None:
                    continue

                robot_mask = eval_model.get_type_mask(
                    batch["trace_uid"][bid], robot_uids
                )
                robot_mask = robot_mask & eval_model.get_type_mask(
                    batch["trace_group_id"][bid], [robot_gid]
                )
                if int(robot_mask.sum()) == 0:
                    continue
                if int(robot_mask.sum()) != len(link_names):
                    logger.warning(
                        f"{deploy_tag} deploy viz: robot_mask sum "
                        f"{int(robot_mask.sum())} != len(link_names) "
                        f"{len(link_names)}, skipping this robot"
                    )
                    continue

                # (T, K, 4, 4) in whatever encoding the viz layer expects.
                sliced = gen_trace_deploy[:, robot_mask].clone().float()

                # --- Undo local encoding so poses are all in world frame ---
                if local_full_flag:
                    _ref_Q = sliced[:, :1]
                    _local_Q = sliced[:, 1:]
                    _local_Q = torch.einsum(
                        "...ij,...jk->...ik", _ref_Q, _local_Q
                    )
                    sliced_world = torch.cat([_ref_Q, _local_Q], dim=1)
                elif local_trans_flag:
                    _ref_Q = sliced[:, :1]
                    _ref_Q_t = _ref_Q[..., :3, 3]
                    _local_Q = sliced[:, 1:].clone()
                    _local_Q[..., :3, 3] = _local_Q[..., :3, 3] + _ref_Q_t
                    sliced_world = torch.cat([_ref_Q, _local_Q], dim=1)
                else:
                    sliced_world = sliced

                # --- RK -> FK round trip on the G1I device (cpu) ---
                sliced_world_cpu = sliced_world.to(g1i.device)
                # RK expects (T, K+1, 4, 4) with root at 0. By dataset
                # convention (hh_dataset.__load_humanoid__), link_names[0]
                # IS the root link, so sliced_world[:, 0] is already the
                # pelvis pose. Prepend a duplicate to match RK's contract
                # (this matches rtg_l2._convert_to_commands).
                rk_input = torch.cat(
                    [sliced_world_cpu[:, :1], sliced_world_cpu], dim=1
                )  # (T, K+1, 4, 4)
                root_pos, root_quat_xyzw, dof_pos = g1i.RK(
                    rk_input, names=link_names
                )
                fk_out = g1i.FK(
                    root_pos, root_quat_xyzw, dof_pos, names=link_names
                )  # (T, K+1, 4, 4), entry 0 is root, 1..K are link_names

                # Reconstructed link poses in world frame; drop the root.
                reconstructed_world = fk_out[:, 1:]  # (T, K, 4, 4)
                reconstructed_world = reconstructed_world.to(sliced)

                # --- Re-apply the local encoding used by viz_humanoid ---
                if local_full_flag:
                    _ref_Q = reconstructed_world[:, :1]
                    _ref_Q_inv = torch.linalg.inv(_ref_Q.float()).to(_ref_Q)
                    _local_Q = reconstructed_world[:, 1:]
                    _local_Q = torch.einsum(
                        "...ij,...jk->...ik", _ref_Q_inv, _local_Q
                    )
                    reconstructed_enc = torch.cat(
                        [_ref_Q, _local_Q], dim=1
                    )
                elif local_trans_flag:
                    _ref_Q = reconstructed_world[:, :1]
                    _ref_Q_t = _ref_Q[..., :3, 3]
                    _local_Q = reconstructed_world[:, 1:].clone()
                    _local_Q[..., :3, 3] = _local_Q[..., :3, 3] - _ref_Q_t
                    reconstructed_enc = torch.cat(
                        [_ref_Q, _local_Q], dim=1
                    )
                else:
                    reconstructed_enc = reconstructed_world

                # Write back into the deploy copy; downstream
                # _viz_hh_subroutine will slice with the same robot_mask
                # and reapply local handling, exactly like the normal path.
                gen_trace_deploy[:, robot_mask] = reconstructed_enc.to(
                    gen_trace_deploy
                )
                any_g1 = True

            if not any_g1:
                return

            # Render the modified trace through the existing path. Only
            # humanoid video is logged — human/compose would duplicate the
            # non-deploy viz for those channels.
            self._viz_hh_subroutine(
                gen_trace=gen_trace_deploy,
                gen_context=None,
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=viz_seen_step,
                viz_trace_mask=viz_trace_mask,
                viz_ctx_mask=None,
                human_viz_name=human_viz_name or "",
                humanoid_viz_name=humanoid_viz_name or "",
                compose_viz_name=compose_viz_name or "",
                context_viz_name=context_viz_name or "",
                log_human_flag=False,
                log_humanoid_flag=True,
                log_compose_flag=False,
            )
        except Exception as e:
            logger.warning(
                f"{deploy_tag} deploy viz failed: {e}", exc_info=True
            )
            return

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
        self._viz_hh_deploy_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_trace_mask=dataset_trace_mask,
            humanoid_viz_name=f"GT_humanoid_deploy_{phase}",
            deploy_tag="GT",
            eval_model=eval_model,
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
                self._viz_hh_deploy_subroutine(
                    gen_trace=step_output["x0_pred_trace"][bid, :n_steps].float(),
                    viz_T_cam=viz_T_cam,
                    batch=batch,
                    phase=phase,
                    bid=bid,
                    batch_idx=batch_idx,
                    viz_seen_step=n_steps,
                    viz_trace_mask=dataset_trace_mask,
                    humanoid_viz_name=f"PREDX0_humanoid_deploy_{phase}",
                    deploy_tag="PREDX0",
                    eval_model=eval_model,
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
            self._viz_hh_deploy_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                humanoid_viz_name=f"H2R_humanoid_deploy_{phase}",
                deploy_tag="H2R",
                eval_model=eval_model,
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
            self._viz_hh_deploy_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                humanoid_viz_name=f"R2H_humanoid_deploy_{phase}",
                deploy_tag="R2H",
                eval_model=eval_model,
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
        self._viz_hh_deploy_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            humanoid_viz_name=f"FuturePred_humanoid_deploy_seen_half_{phase}",
            deploy_tag="FuturePred_half",
            eval_model=eval_model,
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
        self._viz_hh_deploy_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=1,
            humanoid_viz_name=f"FuturePred_humanoid_deploy_seen_1_{phase}",
            deploy_tag="FuturePred_1",
            eval_model=eval_model,
        )
