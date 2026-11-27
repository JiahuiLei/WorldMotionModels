"""Visualization helpers for Language Table tabletop manipulation domain (V11).

Renders robot end-effector and block object meshes/point-clouds for GT and
inference visualization during training. Based on the V1 langtable_viz but
adapted for the V11 architecture (no add_cfg / left2right_AR_sweep_flag).

Supports GT, x0-pred, Future(w./no goal), OnlyCur, Mask inference patterns,
and closed-loop environment rollout deploy video (lazy-imported
``language_table`` / ``tqdm``; wrapped in try/except so viz failures can
never crash training).

Example::

    viz = LangTableVisualizer(pl_module, langtable_cfg)
    viz.langtable_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging

import numpy as np
import torch
import trimesh
from trimesh.creation import icosphere as TriSphere
from copy import deepcopy
import matplotlib.colors as mcolors

from umm.datasets.umm_v11.dataset_config import (
    UID_LANGTABLE_OBJECT_GROUP,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
    SEM_LANGTABLE_ACT,
    SEM_LANGTABLE_EE,
    SEM_OBJECT,
)
from umm.models.umm_v14.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_v14.viz_helpers.mesh_visualizer import visualize_a_mesh

# Constants
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)

# One-shot warning flag for deploy video failures (e.g., missing
# language_table dependency). Set to True after the first failure to avoid
# log spam across thousands of training steps.
_DEPLOY_VIZ_WARNED = False


class LangTableVisualizer:
    """Visualizer for Language Table environment trajectories (V6_All).

    Handles rendering of robot end-effector, action target, and object meshes
    as trimesh objects. Supports GT, x0-pred, and inference visualization modes
    during training.

    Args:
        pl_module: Parent PyTorch Lightning module (PLModel).
        langtable_cfg: Config dict from ``model.langtable`` with mesh paths,
            ``num_pts``, and rollout parameters.
    """

    def __init__(self, pl_module, langtable_cfg):
        self.pl = pl_module
        if not langtable_cfg or len(langtable_cfg) == 0:
            self.flag = False
            self.robot_data = None
            return
        self.flag = True

        self.robot_data = np.load(langtable_cfg.get("model_path", ""))
        self.object_data = np.load(langtable_cfg.get("object_path", ""))

        # Build robot and object meshes keyed by UID
        self.mesh = {}
        self.ee_uid = langtable_cfg.get("ee_uid", 16777219)
        _v = self.robot_data[f"{self.ee_uid}_v"]
        _f = self.robot_data[f"{self.ee_uid}_f"]
        self.mesh[UID_LANGTABLE_EE] = trimesh.Trimesh(vertices=_v, faces=_f)
        self.mesh[UID_LANGTABLE_ACT] = trimesh.Trimesh(vertices=_v, faces=_f)
        object_ids = [k[:-2] for k in self.object_data.keys() if k.endswith("_v")]
        for k in object_ids:
            _v = self.object_data[k + "_v"]
            _f = self.object_data[k + "_f"]
            self.mesh[UID_LANGTABLE_OBJECT_GROUP[int(k)]] = trimesh.Trimesh(
                vertices=_v, faces=_f
            )

        self.num_pts = langtable_cfg.get("num_pts", -1)
        assert self.num_pts > 0, "must set num_pts in langtable config"

        self.window = langtable_cfg.get("window", 20)
        self.hist = langtable_cfg.get("hist", 1)
        self.act_horizon = langtable_cfg.get("act_horizon", 10)

        if not isinstance(self.act_horizon, list):
            self.act_horizon = [self.act_horizon]
        if not isinstance(self.hist, list):
            self.hist = [self.hist]
        if not isinstance(self.window, list):
            self.window = [self.window]

        # Rollout disabled by default (requires language_table env dependency)
        self.rollout_flag = langtable_cfg.get("rollout_flag", False)
        self.max_steps = langtable_cfg.get("max_steps", 200)
        logging.info(
            f"LangTableVisualizer (v11): rollout_flag={self.rollout_flag}, "
            f"max_steps={self.max_steps}"
        )

        self.langtable_viz_cfg = self.pl.cfg.get("visualize", {}).get("langtable", {})
        self.add_cfg = self.langtable_viz_cfg.get("add_cfg", [1.0])

    def __make_dummy_frames__(self, T, H, W):
        """Create blank placeholder frames."""
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(T)]

    def render_viz_frames(
        self,
        prompt_type,
        T_wl,        # (T, N, 4, 4)
        viz_T_cam,
        local_pcl,   # (N, PCL, 3+)
        local_pcl_mask,  # (N,)
        H=512,
        W=512,
        local_pcl_max_N=100,
        local_pcl_r=0.005,
        pcl_alpha=1.0,
        marking_mask=None,
        marking_alpha=0.75,
        viz_mask=None,
        pcl_mode=False,
    ):
        """Render Language Table scene frames.

        Args:
            prompt_type: UID for each part (N,).
            T_wl: World-to-link transforms (T, N, 4, 4).
            viz_T_cam: Camera-to-world transform (4, 4).
            local_pcl: Local point clouds (N, PCL, 6).
            local_pcl_mask: Mask for valid point clouds (N,).
            H: Image height.
            W: Image width.
            marking_mask: Per-frame per-part mask for marking predictions (T, N).

        Returns:
            list[np.ndarray]: Rendered RGBA frames.
        """
        if self.robot_data is None:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)

        T, N = T_wl.shape[:2]

        if marking_mask is not None:
            assert marking_mask.shape == (T, N)
        else:
            marking_mask = torch.zeros((T, N), dtype=torch.bool)

        if viz_mask is not None:
            assert viz_mask.shape == (T, N)
        else:
            viz_mask = torch.ones((T, N), dtype=torch.bool)

        frames = []
        for t in range(T):
            _mm = []
            for i in range(N):
                if not local_pcl_mask[i]:
                    continue
                if not viz_mask[t, i]:
                    continue

                _pcl = local_pcl[i]  # (PCL, 6)
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]
                _rgb = _pcl[:, 3:].clone().cpu()
                if prompt_type[i] == UID_LANGTABLE_EE:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 1] = 1.0  # green for EE
                elif prompt_type[i] == UID_LANGTABLE_ACT:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0  # red for action target

                # Color marking for predicted vs known
                if marking_mask[t, i]:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = 1.0
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)
                else:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = _hsv[..., 1] * 0.7
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)

                _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha
                _alpha_u8 = int(np.clip(_pcl_alpha * 255.0, 0, 255))
                if pcl_mode:
                    _pcl = (T_wl[t, i][:3, :3] @ _pcl[:, :3].T).T + T_wl[t, i][
                        None, :3, 3
                    ]
                    _m_pcl = []
                    for _idx in range(len(_pcl)):
                        sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                        sph.apply_translation(_pcl[_idx].cpu().numpy())
                        _rgb_u8 = np.clip(_rgb[_idx] * 255.0, 0, 255).astype(np.uint8)
                        sph.visual.vertex_colors = _rgb_u8.tolist() + [_alpha_u8]
                        _m_pcl.append(sph)
                    if len(_m_pcl) == 0:
                        continue
                    _m_pcl = trimesh.util.concatenate(_m_pcl)
                else:
                    _mesh = self.mesh[int(prompt_type[i])]
                    _mesh = _mesh.copy()
                    _mesh.apply_transform(T_wl[t, i].cpu().numpy())
                    _rgb_u8 = np.clip(_rgb.mean(0) * 255.0, 0, 255).astype(np.uint8)
                    _mesh.visual.vertex_colors = _rgb_u8.tolist() + [_alpha_u8]
                    _m_pcl = _mesh
                _mm.append(_m_pcl)

            if len(_mm) == 0:
                frames.append(self.__make_dummy_frames__(1, H=H, W=W)[0])
                continue
            viz_m = trimesh.util.concatenate(_mm)
            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam,
                mesh=viz_m,
                background_color=BACKGROUND_COLOR,
                ground_color=GROUND_COLOR,
                lighting=True,
                ground_z=0.0,
                ground_grid_size=0.1,
                width=W,
                height=H,
            )
            frames.append(_fr)
        return frames

    def make_viz_T_cam(self, batch):
        """Build visualization camera from config.

        Args:
            batch: Training batch (unused for fixed-camera viz).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.
        """
        return make_viz_T_cam(self.langtable_viz_cfg, batch)

    @staticmethod
    def get_type_mask(uid_tensor, uid_list):
        """Create a boolean mask for parts matching any UID in uid_list.

        Args:
            uid_tensor: (P,) tensor of UIDs.
            uid_list: list of UIDs to match.

        Returns:
            torch.Tensor: (P,) boolean mask.
        """
        uid_cpu = uid_tensor.cpu()
        mask = torch.zeros(len(uid_cpu), dtype=torch.bool)
        for uid in uid_list:
            mask = mask | (uid_cpu == uid)
        return mask

    # ------------------------------------------------------------------
    # Deploy-video helpers (closed-loop env rollout)
    # ------------------------------------------------------------------

    def prepare_trace_attr_oracle(self, obj_uids, ee_uids):
        """Prepare trace attributes for oracle rollout.

        Assembles group IDs, semantic types, and local point clouds for
        objects and the robot end-effector, matching the format the V11
        agent expects.

        Args:
            obj_uids (list[int]): Local object UIDs from the environment.
            ee_uids (list[int]): Local end-effector UIDs.

        Returns:
            tuple: ``(group_id, sem_type, local_pcl)`` -- (N,) long tensor,
                (N,) long tensor, (N, PCL, 6) float tensor.
        """
        N = len(obj_uids) + 2
        group_id = torch.zeros(N).long()
        group_id[-2:] = 1
        sem_type = torch.zeros(N).long()
        sem_type[-1] = SEM_LANGTABLE_ACT
        sem_type[-2] = SEM_LANGTABLE_EE
        sem_type[:-2] = SEM_OBJECT
        local_pcl = []
        for obj_uid in obj_uids:
            _pcl = self.object_data[f"{obj_uid}_pcl"]
            _pcl_c = self.object_data[f"{obj_uid}_pcl_c"]
            _pcl = np.concatenate([_pcl, _pcl_c], axis=-1)
            _pcl = sample_list(_pcl, self.num_pts)
            local_pcl.append(_pcl.copy())
        ee_pcl = self.robot_data[f"{ee_uids[0]}_pcl"]
        ee_pcl = sample_list(ee_pcl, self.num_pts)
        ee_pcl_c = np.ones_like(ee_pcl).astype(np.float32)
        ee_pcl = np.concatenate([ee_pcl, ee_pcl_c], axis=-1)
        local_pcl.append(ee_pcl.copy())
        local_pcl.append(ee_pcl.copy())
        local_pcl = np.stack(local_pcl, axis=0)
        local_pcl = torch.as_tensor(local_pcl, dtype=torch.float32)
        return group_id, sem_type, local_pcl

    @torch.no_grad()
    def infer_agent(
        self,
        agent,
        hist_trace,
        group_id,
        sem_type,
        local_pcl,
        window,
        fps,
        device,
        seed,
        instruction,
    ):
        """Run a single agent inference step for Language Table rollouts.

        Pads the history trace to the full window length, constructs
        masks and relative timestamps, then calls ``agent.inference`` with
        the V11 signature (no ``add_cfg`` / ``left2right_AR_sweep_flag``).

        Args:
            agent: The V11 inference agent (``AgentV14`` nn.Module).
            hist_trace (list or np.ndarray): History frames (H, N, 4, 4).
            group_id (torch.Tensor): Group IDs (N,).
            sem_type (torch.Tensor): Semantic types (N,).
            local_pcl (torch.Tensor): Local point clouds (N, PCL, 6).
            window (int): Total prediction window.
            fps (float): Environment control frequency.
            device (torch.device): Compute device.
            seed (int): Random seed.
            instruction (str): Language instruction.

        Returns:
            tuple: ``(gen_context, gen_trace)`` -- ``(N, 4, 4)`` context and
                ``(window, N, 4, 4)`` trace, both on the given device.
        """
        # np.stack first for the fast path (avoid slow list-of-arrays conversion).
        if isinstance(hist_trace, list):
            hist_trace = np.stack(hist_trace, axis=0)
        hist_trace = torch.as_tensor(hist_trace, dtype=torch.float32)
        H, N, _, _ = hist_trace.shape
        assert H <= window, (
            f"infer_agent: hist length {H} must be <= window {window}"
        )
        pad = window - H
        pad_trace = torch.eye(4).view(1, 1, 4, 4).expand(pad, N, 4, 4)
        trace = torch.cat([hist_trace, pad_trace])
        trace_mask = torch.ones((window, N))
        trace_mask[H:] = 0
        context = torch.eye(4).view(1, 1, 4, 4).expand(1, N, 4, 4).to(device)
        context_mask = torch.zeros((1, N))
        realtime = torch.arange(window, dtype=torch.float32) - H + 1
        realtime = realtime / fps

        gen_context, gen_trace = agent.inference(
            trace=trace[None].to(device),
            context=context[None].to(device),
            trace_known_mask=trace_mask[None].bool().to(device),
            context_known_mask=context_mask[None].bool().to(device),
            # all rows useful (no pad)
            trace_mask=torch.ones_like(trace_mask[None]).bool().to(device),
            context_mask=torch.ones_like(context_mask[None]).bool().to(device),
            real_timestep=realtime[None].to(device),
            group_id=group_id[None].to(device),
            sem_type=sem_type[None].to(device),
            rot_mask=torch.ones((1, N)).bool().to(device),
            local_pcl=local_pcl[None].to(device),
            local_pcl_mask=torch.ones((1, N)).bool().to(device),
            text=[instruction],
            seed=seed,
            verbose=False,
        )
        gen_context = gen_context[0, 0]  # (N, 4, 4)
        gen_trace = gen_trace[0]          # (window, N, 4, 4)
        return gen_context, gen_trace

    def reset_env(self, env):
        """Reset the Language Table environment and extract metadata.

        Args:
            env: Language Table environment instance.

        Returns:
            tuple: ``(obs, instruction, obj_uids, ee_uids)``
        """
        obs = env.reset()
        instruction = env.decode_instruction(obs["instruction"])
        obj_uids = env.get_all_active_object_ids()
        ee_uids = [k for k, v in obs["instance_map"].items() if v == 4]
        assert len(ee_uids) == 1, "EE not found"
        logging.info(f"all objs: {obj_uids}, ee: {ee_uids}")
        return obs, instruction, obj_uids, ee_uids

    def rollout_full_states_trivial_policy(
        self,
        agent,
        seed,
        max_steps,
        hist,
        window,
        act_horizon,
        device=None,
    ):
        """Perform a closed-loop rollout using a trivial oracle policy.

        Runs the agent in a receding-horizon loop: every ``act_horizon``
        steps, queries the agent for a future-trace prediction, extracts
        the EE action target, and steps the environment.

        ``language_table`` and ``tqdm`` are imported lazily inside this
        method so the module stays importable even when those optional
        dependencies are missing.

        Args:
            agent: The V11 inference agent.
            seed (int): Random seed.
            max_steps (int): Maximum environment steps.
            hist (int): Number of history frames.
            window (int): Total prediction window.
            act_horizon (int): Steps between re-planning.
            device (torch.device, optional): Compute device. Defaults to
                ``cuda:0`` if ``None``.

        Returns:
            tuple: ``(succ, succ_steps, trace, ctx, uid, pcl,
                viz_frames, instruction, detailed_buffer)``
        """
        # Lazy imports — these are OPTIONAL deps and must not break module load
        from tqdm import tqdm
        from language_table.environments import language_table, blocks  # type: ignore[import-not-found]
        from language_table.environments.rewards import block2block  # type: ignore[import-not-found]

        if device is None:
            device = torch.device("cuda:0")

        logging.info(f"langtable_rollout seed={seed}")
        rng_state = np.random.get_state()
        np.random.seed(seed)

        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.N_CHOOSE_K,
            reward_factory=block2block.BlockToBlockReward,
            render_text_in_image=True,
            render_depth=True,
            render_segmentation=True,
            render_semantic_segmentation=True,
            render_instance_poses=True,
        )
        try:
            env.seed(seed)

            obs, instruction, obj_uids, ee_uids = self.reset_env(env)
            group_id, sem_type, local_pcl = self.prepare_trace_attr_oracle(
                obj_uids, ee_uids
            )

            trace_buffer, ctx_buffer = [], []
            detailed_buffer, viz_frames = [], []
            succ = False
            succ_steps = -1

            viz_frames.append(obs["rgb"])

            # [obj-states | ee-state | ee-act]
            cur_state_T_wl = np.asarray(
                [obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
            )
            ee_T_wl = cur_state_T_wl[-1].copy()
            trace_buffer.append(
                np.concatenate([cur_state_T_wl, cur_state_T_wl[-1:].copy()], axis=0)
            )

            action_target = []
            gen_context = None
            for timestep in tqdm(range(max_steps)):
                if timestep % act_horizon == 0:
                    # Update the target buffer
                    hist_trace = trace_buffer[-hist:]
                    gen_context, gen_trace = self.infer_agent(
                        agent=agent,
                        hist_trace=hist_trace,
                        group_id=group_id,
                        sem_type=sem_type,
                        local_pcl=local_pcl,
                        window=window,
                        fps=env._control_frequency,
                        device=device,
                        seed=seed,
                        instruction=instruction,
                    )
                    detailed_buffer.append(
                        (gen_trace.cpu().numpy(), len(hist_trace), timestep)
                    )

                    future_action_T = gen_trace[len(hist_trace):, -1]
                    for i in range(len(future_action_T)):
                        if timestep + i >= max_steps:
                            break
                        if len(action_target) <= timestep + i:
                            action_target.append([])
                        action_target[timestep + i].append(
                            future_action_T[i].cpu().numpy()
                        )

                action_T = action_target[timestep]
                assert len(action_T) > 0
                action_T = action_T[-1]  # latest pred
                applied_act = action_T[:2, 3] - ee_T_wl[:2, 3]

                # Step environment
                next_obs, _reward, done, _info = env.step(applied_act)
                viz_frames.append(next_obs["rgb"])

                assert gen_context is not None  # set on first timestep (0 % H == 0)
                ctx_buffer.append(gen_context.cpu().numpy())

                cur_state_T_wl = np.asarray(
                    [next_obs["instance_poses"][_uid] for _uid in obj_uids + ee_uids]
                )
                ee_T_wl = cur_state_T_wl[-1].copy()
                trace_buffer.append(
                    np.concatenate([cur_state_T_wl, action_T[None]], axis=0)
                )

                if done:
                    succ = True
                    succ_steps = timestep
                    break
            logging.warning(f"seed: {seed}, succ: {succ}, succ_steps: {succ_steps}")
        finally:
            # Always release env resources (physics threads, shared memory) even
            # if rollout raised mid-loop. Missing close() caused resource leaks
            # across viz cycles with multi-element window/hist/act_horizon lists.
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                try:
                    close_fn()
                except Exception as _close_e:
                    logging.warning(f"env.close() failed: {_close_e}")

        np.random.set_state(rng_state)

        ret_trace = np.stack(trace_buffer, 0)  # (T, N, 4, 4)
        ret_ctx = np.stack(ctx_buffer, 0)       # (T, N, 4, 4)
        ret_uid = [UID_LANGTABLE_OBJECT_GROUP[uid] for uid in obj_uids] + [
            UID_LANGTABLE_EE,
            UID_LANGTABLE_ACT,
        ]
        ret_pcl = local_pcl
        return (
            succ,
            succ_steps,
            ret_trace,
            ret_ctx,
            ret_uid,
            ret_pcl,
            viz_frames,
            instruction,
            detailed_buffer,
        )

    @torch.inference_mode()
    def langtable_trivial_policy_rollout_viz_routine(
        self,
        eval_model,
        rollout_seed,
        viz_T_cam,
        batch_idx,
        max_steps,
        window,
        hist,
        act_horizon,
        viz_skip=1,
        postfix="_policy",
    ):
        """Run a policy rollout and log RGB + 3D visualization videos.

        Executes ``rollout_full_states_trivial_policy``, then logs the
        environment RGB video, detailed generation snapshots, and a
        side-by-side context+trace 3D video.

        Args:
            eval_model: Agent model used for inference.
            rollout_seed (int): Random seed.
            viz_T_cam: Camera-to-world transform.
            batch_idx (int): Batch index for logging.
            max_steps (int): Maximum environment steps.
            window (int): Prediction window.
            hist (int): History length.
            act_horizon (int): Action horizon.
            viz_skip (int): Frame skip for 3D visualization.
            postfix (str): Suffix for logger tags.
        """
        (
            succ,
            succ_steps,
            rollout_trace,
            rollout_ctx,
            rollout_uid,
            rollout_pcl,
            rollout_viz_frames,
            rollout_instruction,
            rollout_detailed_gen_trace,
        ) = self.rollout_full_states_trivial_policy(
            agent=eval_model,
            seed=rollout_seed,
            max_steps=max_steps,
            window=window,
            hist=hist,
            act_horizon=act_horizon,
        )

        postfix = f"W{window}_H{hist}_A{act_horizon}{postfix}"
        caption_prefix = (
            f"[Succ={succ}, Steps={succ_steps}, W={window}, H={hist}, "
            f"A={act_horizon}]"
        )

        rollout_viz_frames = put_caption_at_bottom(
            rollout_viz_frames, f"{caption_prefix} {rollout_instruction}"
        )
        self.pl.logger.log_video(
            "deploy",
            f"rollout_{postfix}",
            np.stack(rollout_viz_frames, 0).transpose(0, 3, 1, 2).astype(np.float32)
            / 255.0,
            self.pl.global_step,
            batch_idx,
            caption=f"{caption_prefix} {rollout_instruction}",
        )

        # Also visualize the detailed gen trace at three timesteps
        for i, name in [
            (0, "start"),
            (len(rollout_detailed_gen_trace) // 2, "middle"),
            (len(rollout_detailed_gen_trace) - 1, "end"),
        ]:
            gen_trace, H, timestep = rollout_detailed_gen_trace[i]
            marker_mask = torch.zeros(*gen_trace.shape[:2]).bool()
            marker_mask[H:] = True

            # Double the true future alongside the generated one
            coupled_real_rollout = []
            coupled_real_marker_mask = []
            for _t in range(timestep - H + 1, timestep - H + 1 + len(gen_trace)):
                if _t < 0:
                    coupled_real_rollout.append(rollout_trace[0])
                    coupled_real_marker_mask.append(False)
                elif _t < len(rollout_trace):
                    coupled_real_rollout.append(rollout_trace[_t])
                    coupled_real_marker_mask.append(True)
                else:
                    coupled_real_rollout.append(rollout_trace[-1])
                    coupled_real_marker_mask.append(False)
            coupled_real_rollout = torch.as_tensor(np.stack(coupled_real_rollout, 0))
            coupled_real_marker_mask = torch.as_tensor(
                np.stack(coupled_real_marker_mask, 0)
            )[:, None].expand(-1, gen_trace.shape[1])
            viz_rollout_uid = rollout_uid + rollout_uid
            viz_T_wl = torch.cat(
                [torch.from_numpy(gen_trace), coupled_real_rollout], dim=1
            )
            viz_marker_mask = torch.cat(
                [marker_mask, torch.zeros_like(marker_mask)], dim=1
            )
            viz_rollout_pcl = torch.cat(
                [torch.as_tensor(rollout_pcl), torch.as_tensor(rollout_pcl)], dim=0
            )

            viz = self.render_viz_frames(
                prompt_type=viz_rollout_uid,
                T_wl=viz_T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=viz_rollout_pcl,
                local_pcl_mask=torch.ones(len(viz_rollout_uid)).bool(),
                marking_mask=viz_marker_mask,
                H=360,
                W=640,
            )
            viz = put_caption_at_bottom(
                viz, f"{caption_prefix} {rollout_instruction}"
            )
            self.pl.logger.log_video(
                "deploy",
                f"rollout_detailed_gen_trace_{name}_{postfix}",
                torch.from_numpy(np.stack(viz, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=f"{caption_prefix} {rollout_instruction}",
            )

        # Visualize the RGB and 3D
        rollout_trace = rollout_trace[::viz_skip]
        rollout_ctx = rollout_ctx[::viz_skip]
        viz = self.render_viz_frames(
            prompt_type=rollout_uid,
            T_wl=torch.as_tensor(rollout_trace),
            viz_T_cam=viz_T_cam,
            local_pcl=torch.as_tensor(rollout_pcl),
            local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
            H=360,
            W=640,
        )
        marker_mask = torch.ones(*rollout_ctx.shape[:2]).bool()
        viz_ctx = self.render_viz_frames(
            prompt_type=rollout_uid,
            T_wl=torch.as_tensor(rollout_ctx),
            viz_T_cam=viz_T_cam,
            local_pcl=torch.as_tensor(rollout_pcl),
            local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
            marking_mask=marker_mask,
            H=360,
            W=640,
        )
        if len(viz_ctx) < len(viz):
            viz_ctx.append(np.full_like(viz_ctx[0], 255))
        viz_rollout = [
            np.concatenate([viz_ctx[i], viz[i]], axis=1)[..., :3]
            for i in range(len(viz_ctx))
        ]
        viz_rollout = put_caption_at_bottom(
            viz_rollout, f"{caption_prefix} {rollout_instruction}"
        )
        self.pl.logger.log_video(
            "deploy",
            f"rollout_3dviz_{postfix}",
            torch.from_numpy(np.stack(viz_rollout, 0))[..., :3]
            .float()
            .permute(0, 3, 1, 2)
            / 255.0,
            self.pl.global_step,
            batch_idx,
            caption=f"{caption_prefix} {rollout_instruction}",
        )
        return

    @torch.inference_mode()
    def langtable_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Top-level visualization routine for Language Table.

        Finds the first batch item with LangTable UIDs, logs GT, x0-pred, and
        inference visualizations. Supports image-conditioned inference when DINO
        is enabled.

        Args:
            batch: Training batch dict.
            phase: "train" or "val".
            batch_idx: Batch index for logging.
            eval_model: Agent model for inference.
            step_output: Training step outputs with x0_pred_trace/context.
        """
        if not self.flag:
            return

        viz_T_cam = self.make_viz_T_cam(batch)

        # --- Deploy video: closed-loop env rollout -----------------------
        # Wrapped in try/except with a one-shot warning flag so any failure
        # (missing language_table dep, env crash, CUDA OOM during rollout)
        # can never kill training. Disabled by default via rollout_flag.
        global _DEPLOY_VIZ_WARNED
        if self.rollout_flag:
            try:
                rollout_seed = int(np.random.randint(0, 10000))
                for _window in self.window:
                    for _horizon in self.act_horizon:
                        for _hist in self.hist:
                            self.langtable_trivial_policy_rollout_viz_routine(
                                eval_model=eval_model,
                                rollout_seed=rollout_seed,
                                viz_T_cam=viz_T_cam,
                                batch_idx=batch_idx,
                                max_steps=self.max_steps,
                                window=_window,
                                hist=_hist,
                                act_horizon=_horizon,
                            )
            except Exception as e:
                if not _DEPLOY_VIZ_WARNED:
                    logging.warning(
                        "LT deploy video rollout failed (one-shot warning): "
                        f"{type(e).__name__}: {e}"
                    )
                    _DEPLOY_VIZ_WARNED = True

        # Find a batch item corresponding to LangTable
        bid = None
        for i in range(len(batch["trace"])):
            mask = self.get_type_mask(
                batch["trace_uid"][i],
                UID_LANGTABLE_OBJECT_GROUP + [UID_LANGTABLE_EE, UID_LANGTABLE_ACT],
            )
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no langtable data, skip viz"
            )
            return

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(f"LT VIZ phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}")

        # --- Subroutine: render and log a single trace visualization ---
        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
            trace_marker_mask = torch.ones(*T_wl.shape[:2]).bool()
            if isinstance(known_n_real, list):
                for idx in known_n_real:
                    trace_marker_mask[idx] = False
            else:
                trace_marker_mask[:known_n_real] = False

            viz = self.render_viz_frames(
                prompt_type=batch["trace_uid"][bid],
                T_wl=T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                marking_mask=trace_marker_mask,
                H=360,
                W=640,
            )
            viz = put_caption_at_bottom(viz, batch["text"][bid])
            self.pl.logger.log_video(
                phase,
                f"{name}_seed{seed}",
                torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=batch["text"][bid],
            )
            if ref_T_wl is not None:
                ctx_marker_mask = torch.ones(*ref_T_wl.shape[:2]).bool()
                if isinstance(known_n_ctx, list):
                    for idx in known_n_ctx:
                        ctx_marker_mask[idx] = False
                else:
                    ctx_marker_mask[:known_n_ctx] = False
                viz_ctx = self.render_viz_frames(
                    prompt_type=batch["trace_uid"][bid],
                    T_wl=ref_T_wl,
                    viz_T_cam=viz_T_cam,
                    local_pcl=batch["local_pcl"][bid],
                    local_pcl_mask=batch["local_pcl_mask"][bid],
                    marking_mask=ctx_marker_mask,
                    H=360,
                    W=640,
                )
                viz_img = make_square_cat_image(viz_ctx)
                viz_img = viz_img[..., :3]
                # V13: show the language prompt at the bottom of the ctx image.
                viz_img = put_caption_at_bottom([viz_img], batch["text"][bid])[0]
                self.pl.logger.log_image(
                    phase,
                    f"ctx_{name}_seed{seed}",
                    viz_img[::1, ::1].transpose(2, 0, 1).astype(np.float32) / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

        # --- GT trace ---
        _subroutine(
            batch["trace"][bid].float().cpu(),
            batch["context"][bid].float().cpu(),
            "GT",
            n_steps,
            100000,
            "gt",
        )

        # --- x0-pred ---
        _subroutine(
            step_output["x0_pred_trace"][bid].float().cpu(),
            step_output["x0_pred_context"][bid].float().cpu(),
            "X0_PRED",
            -1,
            -1,
            -1,
        )

        # --- Prepare image data for inference (if DINO is enabled) ---
        # V11 AgentV14 has no DINO support yet — guard with getattr.
        images_for_inf = None
        view_ids_for_inf = None
        if getattr(eval_model, "use_dino", False) and "extra" in batch:
            extra = batch["extra"]
            if "images" in extra and extra["images"][bid] is not None:
                images_for_inf = torch.as_tensor(
                    np.stack([extra["images"][bid]]),
                    dtype=torch.float32,
                ).to(batch["trace"].device)  # (1, V, 3, H, W)
            if "view_ids" in extra and extra["view_ids"][bid] is not None:
                view_ids_for_inf = torch.as_tensor(
                    np.stack([extra["view_ids"][bid]])
                ).to(batch["trace"].device)

        # --- Inference: future prediction with goal ---
        dataset_trace_mask = batch["trace_mask"][bid]  # (T, P)
        dataset_ctx_mask = batch["context_mask"][bid]  # (C, P)

        pad_mask = batch["trace_mask"][bid].any(0) | batch["context_mask"][bid].any(0)
        pad_mask_trace = (
            pad_mask[None].bool().clone().repeat(batch["trace_mask"].shape[1], 1)
        )[None]
        pad_mask_ctx = pad_mask[None, None].bool().clone()

        for seed in [42, 1234]:
            trace_known_mask = dataset_trace_mask[None].bool().clone()
            context_known_mask = dataset_ctx_mask[None].bool().clone()
            n_steps_int = int((batch["time"][bid] <= 0).sum())
            trace_known_mask[0, n_steps_int:] = False

            gen_context, gen_trace = eval_model.inference(
                trace=batch["trace"][bid:bid + 1],
                context=batch["context"][bid:bid + 1],
                trace_known_mask=trace_known_mask,
                context_known_mask=context_known_mask,
                trace_mask=pad_mask_trace,
                context_mask=pad_mask_ctx,
                real_timestep=batch["time"][bid:bid + 1],
                group_id=batch["trace_group_id"][bid:bid + 1],
                sem_type=batch["trace_sem_type"][bid:bid + 1],
                rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                local_pcl=batch["local_pcl"][bid:bid + 1],
                local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                text=[batch["text"][bid]],
                seed=seed,
            )
            _subroutine(
                gen_trace[0].float().cpu(),
                None,
                "Future(w. goal)",
                n_steps_int,
                -1,
                seed,
            )

            # --- Inference: future prediction without goal (text-conditioned) ---
            # V13.3.5: CFG inference -- each cfg value produces different output.
            context_known_mask[...] = False
            for cfg in self.add_cfg:
                gen_context, gen_trace = eval_model.inference(
                    trace=batch["trace"][bid:bid + 1],
                    context=batch["context"][bid:bid + 1],
                    trace_known_mask=trace_known_mask,
                    context_known_mask=context_known_mask,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid:bid + 1],
                    group_id=batch["trace_group_id"][bid:bid + 1],
                    sem_type=batch["trace_sem_type"][bid:bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                    local_pcl=batch["local_pcl"][bid:bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                    cfg_scale=cfg,
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    f"cfg{cfg}_Future(no goal)",
                    n_steps_int,
                    0,
                    seed,
                )

            # --- Inference: OnlyCur pattern ---------------------------
            # Single cur_i step is known (all P parts True at cur_i),
            # rest of trace is unknown, context is all False.
            trace_known_mask_cur = torch.zeros_like(
                dataset_trace_mask[None]
            ).bool()
            context_known_mask_cur = torch.zeros_like(
                dataset_ctx_mask[None]
            ).bool()
            cur_i = int((batch["time"][bid] <= 0).sum()) - 1
            if cur_i >= 0:
                trace_known_mask_cur[0, cur_i] = True
                gen_context, gen_trace = eval_model.inference(
                    trace=batch["trace"][bid:bid + 1],
                    context=batch["context"][bid:bid + 1],
                    trace_known_mask=trace_known_mask_cur,
                    context_known_mask=context_known_mask_cur,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid:bid + 1],
                    group_id=batch["trace_group_id"][bid:bid + 1],
                    sem_type=batch["trace_sem_type"][bid:bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                    local_pcl=batch["local_pcl"][bid:bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    "OnlyCur(no goal)",
                    [cur_i],
                    0,
                    seed,
                )

            # --- Inference: Mask pattern (base_random-style) ----------
            # Randomly mark ~half of the valid (trace, part) positions as
            # known, rest unknown; context is all False.
            g_mask = torch.Generator()
            g_mask.manual_seed(seed)
            valid_trace_pp = dataset_trace_mask.bool()  # (T, P)
            n_valid = int(valid_trace_pp.sum().item())
            if n_valid >= 2:
                n_known = max(1, n_valid // 2)
                flat_valid_idx = (
                    valid_trace_pp.flatten().nonzero(as_tuple=False).flatten()
                )
                perm = torch.randperm(len(flat_valid_idx), generator=g_mask)
                chosen = flat_valid_idx[perm[:n_known]]
                trace_known_mask_rand = (
                    torch.zeros_like(dataset_trace_mask).bool().flatten()
                )
                trace_known_mask_rand[chosen] = True
                trace_known_mask_rand = trace_known_mask_rand.view_as(
                    dataset_trace_mask
                )[None]
                context_known_mask_rand = torch.zeros_like(
                    dataset_ctx_mask[None]
                ).bool()
                gen_context, gen_trace = eval_model.inference(
                    trace=batch["trace"][bid:bid + 1],
                    context=batch["context"][bid:bid + 1],
                    trace_known_mask=trace_known_mask_rand,
                    context_known_mask=context_known_mask_rand,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid:bid + 1],
                    group_id=batch["trace_group_id"][bid:bid + 1],
                    sem_type=batch["trace_sem_type"][bid:bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                    local_pcl=batch["local_pcl"][bid:bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                )
                # Mark trace rows that have ANY known part as "known" so the
                # marker overlay highlights the generated (unknown) rows.
                known_t_rows = []
                for t_i in range(dataset_trace_mask.shape[0]):
                    if trace_known_mask_rand[0, t_i].any().item():
                        known_t_rows.append(t_i)
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    "Mask(random)",
                    known_t_rows,
                    0,
                    seed,
                )

        # --- V13.3.5: Two-stage inference ---
        if getattr(eval_model, 'twostage_inference', False):
            for seed in [42, 1234]:
                trace_known_mask = dataset_trace_mask[None].bool().clone()
                context_known_mask = dataset_ctx_mask[None].bool().clone()
                n_steps_int = int((batch["time"][bid] <= 0).sum())
                trace_known_mask[0, n_steps_int:] = False

                gen_context, gen_trace = eval_model.inference_twostage(
                    trace=batch["trace"][bid:bid + 1],
                    context=batch["context"][bid:bid + 1],
                    trace_known_mask=trace_known_mask,
                    context_known_mask=context_known_mask,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid:bid + 1],
                    group_id=batch["trace_group_id"][bid:bid + 1],
                    sem_type=batch["trace_sem_type"][bid:bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                    local_pcl=batch["local_pcl"][bid:bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                    verbose=False,
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    "twostage/Future(w. goal)",
                    n_steps_int,
                    0,
                    seed,
                )
