"""Visualization helpers for Language Table tabletop manipulation domain.

Renders robot end-effector and block object meshes/point-clouds,
supports closed-loop policy and MPC rollouts with visualization, and
provides GT, x0-pred, and autoregressive future-prediction logging modes.

Example::

    viz = LangTableVisualizer(pl_module, langtable_cfg)
    viz.langtable_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import trimesh
from trimesh.creation import icosphere as TriSphere
from copy import deepcopy
import sys
import os.path as osp
import matplotlib.colors as mcolors
import imageio
import gc

from umm.datasets.umm_v1.dataset_config import (
    UID_LANGTABLE_OBJECT_GROUP,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
    SEM_LANGTABLE_ACT,
    SEM_LANGTABLE_EE,
    SEM_OBJECT,
)
from lib_io import UnifiedIO

# Setup paths for language_table and oracle
from language_table.environments import language_table
from language_table.environments import blocks
from language_table.environments.rewards import block2block

from umm.models.umm_v1.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_v1.agent import Agent
from umm.models.umm_v1.viz_helpers.mesh_visualizer import visualize_a_mesh

# Constants from model.py
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class LangTableVisualizer:
    """
    Visualizer for Language Table environment trajectories.
    Handles rendering of robot, objects, and point clouds.
    """

    def __init__(self, pl_module, langtable_cfg):
        """Initialize the Language Table visualizer.

        Loads robot EE and block object meshes from ``.npz`` archives,
        and configures rollout parameters (window, hist, act_horizon).

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            langtable_cfg (dict): Config dict with ``model_path``,
                ``object_path``, ``num_pts``, and rollout parameters.

        Example::

            viz = LangTableVisualizer(pl_module, cfg["visualize"]["langtable"])
        """
        self.pl = pl_module
        if langtable_cfg == {} or len(langtable_cfg) == 0 or langtable_cfg is None:
            self.flag = False
            self.robot_data = (
                None  # Ensure langtable_data is initialized even if flag is False
            )
            return
        self.flag = True

        self.robot_data = np.load(langtable_cfg.get("model_path", ""))
        self.object_data = np.load(langtable_cfg.get("object_path", ""))

        # make robot and object mesh
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
        assert self.num_pts > 0, " must set self.num_pts in config"

        self.window = langtable_cfg.get("window", 20)
        self.hist = langtable_cfg.get("hist", 1)
        self.act_horizon = langtable_cfg.get("act_horizon", 10)

        if not isinstance(self.act_horizon, list):
            self.act_horizon = [self.act_horizon]
        if not isinstance(self.hist, list):
            self.hist = [self.hist]
        if not isinstance(self.window, list):
            self.window = [self.window]

        self.max_steps = langtable_cfg.get("max_steps", 200)

        self.rollout_flag = langtable_cfg.get("rollout_flag", True)
        logging.info(f"viz langtable rollout_flag: {self.rollout_flag}")

        self.langtable_viz_cfg = self.pl.cfg.get("visualize", {}).get("langtable", {})
        self.add_cfg = self.langtable_viz_cfg.get("add_cfg", [1.0])
        self.io = UnifiedIO()

    def __make_dummy_frames__(self, T, H, W):
        """Create a list of blank placeholder frames.

        Args:
            T (int): Number of frames.
            H (int): Frame height.
            W (int): Frame width.

        Returns:
            list[np.ndarray]: List of blank RGBA frames (H, W, 4).

        Example::

            blanks = self.__make_dummy_frames__(10, 360, 640)
        """
        # Placeholder for when visualization is disabled
        frames = []
        for _ in range(T):
            dummy_frame = np.zeros((H, W, 4), dtype=np.uint8)
            frames.append(dummy_frame)
        return frames

    def render_viz_frames(
        self,
        prompt_type,
        T_wl,  # T,N,4,4
        viz_T_cam,
        local_pcl,  # N,PCL,3
        local_pcl_mask,  # N
        H=512,
        W=512,
        local_pcl_max_N=100,
        local_pcl_r=0.005,
        pcl_alpha=1.0,
        # a predict maks
        marking_mask=None,
        marking_alpha=0.75,
        #
        viz_mask=None,
        pcl_mode=False,
    ):
        """
        Visualize the Language Table scene.

        Args:
            prompt_type (torch.Tensor): Prompt types for objects/robot.
            rot_mask (torch.Tensor): Rotation mask.
            T_wl (torch.Tensor): World-to-link transforms [T, N, 4, 4].
            viz_T_cam (torch.Tensor): Camera pose.
            local_pcl (torch.Tensor): Local point clouds [N, PCL, 3].
            local_pcl_mask (torch.Tensor): Mask for local point clouds [N].
            H (int): Image height.
            W (int): Image width.
            local_pcl_max_N (int): Max points per local cloud.
            local_pcl_r (float): Radius for point cloud spheres.
            pcl_alpha (float): Alpha for point clouds.
            marking_mask (torch.Tensor, optional): Mask for marking specific objects (e.g., prediction).
            marking_alpha (float): Alpha for marked objects.
            viz_mask (torch.Tensor, optional): Mask for visibility.
            pcl_mode (bool): If True, render as point clouds instead of meshes.

        Returns:
            list: List of rendered frames.
        """
        # ! this is for the patch wise version!
        # todo: draw small local frame
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
            # make the robot mesh
            _mm = []
            for i in range(N):
                if not local_pcl_mask[i]:
                    continue
                if not viz_mask[t, i]:
                    continue

                _pcl = local_pcl[i]  # N,6
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]
                _rgb = _pcl[:, 3:].clone().cpu()
                if prompt_type[i] == UID_LANGTABLE_EE:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 1] = 1.0
                elif prompt_type[i] == UID_LANGTABLE_ACT:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0
                # marking the pcl
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
                if pcl_mode:
                    _pcl = (T_wl[t, i][:3, :3] @ _pcl[:, :3].T).T + T_wl[t, i][
                        None, :3, 3
                    ]
                    _m_pcl = []
                    for _idx in range(len(_pcl)):
                        sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                        sph.apply_translation(_pcl[_idx].cpu().numpy())
                        sph.visual.vertex_colors = _rgb[_idx].tolist() + [_pcl_alpha]
                        _m_pcl.append(sph)
                    _m_pcl = trimesh.util.concatenate(_m_pcl)
                else:
                    _mesh = self.mesh[int(prompt_type[i])]
                    _mesh = _mesh.copy()
                    _mesh.apply_transform(T_wl[t, i].cpu().numpy())
                    _mesh.visual.vertex_colors = _rgb.mean(0).tolist() + [_pcl_alpha]
                    _m_pcl = _mesh
                _mm.append(_m_pcl)

            viz_m = trimesh.util.concatenate(_mm)
            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam,
                mesh=viz_m,
                background_color=BACKGROUND_COLOR,
                ground_color=GROUND_COLOR,
                lighting=True,
                ground_z=0.0,  # for better viz the object ctx ref
                # ground_grid_color=background_color,
                # ground_grid_line_width=0.01,
                ground_grid_size=0.1,
                width=W,
                height=H,
            )
            # imageio.imsave("./debug/dbg.jpg", _fr[...,:3])
            # _fr[-15:, ..., :3] = np.asarray(background_color)[:3] * 255.0
            frames.append(_fr)
        return frames

    def prepare_trace_attr_oracle(self, obj_uids, ee_uids):
        """Prepare trace attributes for oracle rollout.

        Assembles group IDs, semantic types, and local point clouds for
        objects and the robot end-effector.

        Args:
            obj_uids (list[int]): Local object UIDs from the environment.
            ee_uids (list[int]): Local end-effector UIDs.

        Returns:
            tuple: ``(group_id, sem_type, local_pcl)`` -- (N,) long tensor,
                (N,) long tensor, (N, PCL, 6) float tensor.

        Example::

            gid, sem, pcl = viz.prepare_trace_attr_oracle(obj_uids, ee_uids)
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
        """Run a single agent inference step for Language Table.

        Pads the history trace to the full window length, constructs
        masks and relative timestamps, then calls ``agent.inference``.

        Args:
            agent: The inference agent (``Agent`` nn.Module).
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
            tuple: ``(gen_context, gen_trace)`` -- context (N, 4, 4)
                and trace (window, N, 4, 4), both on the given device.

        Example::

            ctx, trace = viz.infer_agent(agent, hist, gid, sem, pcl,
                window=20, fps=10.0, device=dev, seed=42, instruction="push red")
        """
        hist_trace = torch.as_tensor(hist_trace, dtype=torch.float32)
        H, N, _, _ = hist_trace.shape
        # H,N,4,4
        pad = window - H
        pad_trace = torch.eye(4).view(1, 1, 4, 4).expand(pad, N, 4, 4)
        trace = torch.cat([hist_trace, pad_trace])
        trace_mask = torch.ones((window, N))
        trace_mask[H:] = 0
        context = torch.eye(4).view(1, 1, 4, 4).expand(1, N, 4, 4).to(device)
        context_mask = torch.zeros((1, N))
        realtime = torch.arange(window, dtype=torch.float32) - H + 1  # !
        realtime = realtime / fps

        gen_context, gen_trace = agent.inference(
            trace=trace[None].to(device),
            context=context[None].to(device),
            trace_known_mask=trace_mask[None].bool().to(device),
            context_known_mask=context_mask[None].bool().to(device),
            # ! all rows useful
            trace_mask=torch.ones_like(trace_mask[None]).bool().to(device),
            context_mask=torch.ones_like(context_mask[None]).bool().to(device),
            #
            real_timestep=realtime[None].to(device),
            group_id=group_id[None].to(device),
            sem_type=sem_type[None].to(device),
            rot_mask=torch.ones((1, N)).bool().to(device),
            #
            local_pcl=local_pcl[None].to(device),
            local_pcl_mask=torch.ones((1, N)).bool().to(device),
            text=[instruction],
            seed=seed,
            left2right_AR_sweep_flag=False,
            verbose=False,
        )
        gen_context = gen_context[0, 0]  # N,4,4
        gen_trace = gen_trace[0]  # window,N,4,4
        return gen_context, gen_trace

    def reset_env(self, env):
        """Reset the Language Table environment and extract metadata.

        Args:
            env: Language Table environment instance.

        Returns:
            tuple: ``(obs, instruction, obj_uids, ee_uids)``

        Example::

            obs, instr, obj_uids, ee_uids = viz.reset_env(env)
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
        device=torch.device("cuda:0"),
    ):
        """Perform a closed-loop rollout using a trivial oracle policy.

        Runs the agent in a receding-horizon loop: every ``act_horizon``
        steps, queries the agent for a future-trace prediction, extracts
        the EE action target, and steps the environment.

        Args:
            agent: The inference agent (``Agent`` nn.Module).
            seed (int): Random seed.
            max_steps (int): Maximum environment steps.
            hist (int): Number of history frames.
            window (int): Total prediction window.
            act_horizon (int): Steps between re-planning.
            device (torch.device): Compute device.

        Returns:
            tuple: ``(succ, succ_steps, trace, ctx, uid, pcl,
                viz_frames, instruction, detailed_buffer)``

        Example::

            result = viz.rollout_full_states_trivial_policy(
                agent, seed=42, max_steps=200, hist=1, window=20,
                act_horizon=10)
        """
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

        # [obj-states|ee-state|ee-act]
        cur_state_T_wl = np.asarray(
            [obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
        )
        ee_T_wl = cur_state_T_wl[-1].copy()
        trace_buffer.append(
            np.concatenate([cur_state_T_wl, cur_state_T_wl[-1:].copy()], axis=0)
        )

        action_target = []
        for timestep in tqdm(range(max_steps)):
            # action = env.action_space.sample()
            if timestep % act_horizon == 0:
                # * update the tgt buffer
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

                future_action_T = gen_trace[len(hist_trace) :, -1]
                for i in range(len(future_action_T)):
                    if timestep + i >= max_steps:
                        break
                    if len(action_target) <= timestep + i:
                        action_target.append([])
                    action_target[timestep + i].append(future_action_T[i].cpu().numpy())

            ######################

            action_T = action_target[timestep]
            assert len(action_T) > 0
            action_T = action_T[-1]  # latest pred
            applied_act = action_T[:2, 3] - ee_T_wl[:2, 3]

            # Step environment
            next_obs, reward, done, info = env.step(applied_act)
            viz_frames.append(next_obs["rgb"])

            # update
            ctx_buffer.append(gen_context.cpu().numpy())

            # ! [obj-states|ee-state|ee-act]
            cur_state_T_wl = np.asarray(
                [next_obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
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

        np.random.set_state(rng_state)

        ret_trace = np.stack(trace_buffer, 0)  # T,N,4,4
        ret_ctx = np.stack(ctx_buffer, 0)  # T,N,4,4
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

    def __plot_to_ndarray__(
        self, value_record: list[np.ndarray], title_prefix: str = ""
    ):
        """Render a list of per-step value arrays as a matplotlib line plot.

        Args:
            value_record (list[np.ndarray]): List of arrays, one per
                optimization step.  Each element has shape ``(K,)``
                representing K parallel samples.
            title_prefix (str): Prefix for the plot title.

        Returns:
            np.ndarray: RGB image of the plot as uint8 array ``(H, W, 3)``.

        Example::

            img = viz.__plot_to_ndarray__(values, title_prefix="Step[0]")
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        import io
        from PIL import Image

        if len(value_record) == 0:
            # Return blank image if no data
            return np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Convert to 2D array: (num_steps, num_samples)
        value_array = np.stack(value_record, axis=0)  # (steps, K)
        num_steps, num_samples = value_array.shape

        # Create figure with nice styling
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot each sample as a line
        steps = np.arange(num_steps)

        # Plot the 0th curve specially (baseline/reference)
        if num_samples > 0:
            ax.plot(
                steps,
                value_array[:, 0],
                "r-",
                alpha=0.9,
                linewidth=2.5,
                label="Sample 0 (Baseline)",
                zorder=9,
            )

        # Plot other samples with transparency
        for i in range(1, num_samples):
            alpha = 0.3 if num_samples > 1 else 1.0
            ax.plot(steps, value_array[:, i], "b-", alpha=alpha, linewidth=1.5)

        # Plot mean and std if multiple samples
        if num_samples > 1:
            mean_values = value_array.mean(axis=1)
            std_values = value_array.std(axis=1)
            ax.plot(steps, mean_values, "k-", linewidth=2.5, label="Mean", zorder=10)
            ax.fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
                color="gray",
                label="±1 std",
            )

        # Styling
        ax.set_xlabel("Optimization Step", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value (Loss)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{title_prefix} MPC Optimization Progress", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        if num_samples > 1:
            ax.legend(loc="best", framealpha=0.9)

        # Tight layout
        fig.tight_layout()

        # Convert figure to numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)[..., :3]  # Remove alpha channel if present
        plt.close(fig)

        return img_array.astype(np.uint8)

    def rollout_full_states_trivial_mpc(
        self,
        agent,
        seed,
        max_steps,
        window,
        act_horizon,
        K=16,
        noise_std=0.03,
        lr_list=[1.0] * 5,
        device=torch.device("cuda:0"),
    ):
        """Perform a closed-loop rollout with gradient-based MPC.

        Generates an initial goal via the agent, then runs K parallel
        noisy action candidates through the agent with gradients
        enabled, optimizes them to minimize a position-matching value
        function, and selects the best candidate.

        Args:
            agent: The inference agent (``Agent`` nn.Module).
            seed (int): Random seed.
            max_steps (int): Maximum environment steps.
            window (int): Total prediction window.
            act_horizon (int): Steps between re-planning.
            K (int): Number of parallel MPC candidates.
            noise_std (float): Std-dev for action noise initialization.
            lr_list (list[float]): Per-iteration learning rates for SGD.
            device (torch.device): Compute device.

        Returns:
            tuple: ``(succ, succ_steps, trace, ctx, uid, pcl,
                viz_frames, instruction, detailed_buffer,
                optimization_value_records, optimization_trace_records,
                optimization_trace_mask_records)``

        Example::

            result = viz.rollout_full_states_trivial_mpc(
                agent, seed=42, max_steps=200, window=20, act_horizon=10)
        """
        torch.cuda.empty_cache()

        logging.info(f"langtable_mpc seed={seed}")
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
        env.seed(seed)

        inner_loop_steps = len(lr_list)
        logging.info(f"inner_loop_steps={inner_loop_steps}, lr:{lr_list}")

        obs, instruction, obj_uids, ee_uids = self.reset_env(env)
        group_id, sem_type, local_pcl = self.prepare_trace_attr_oracle(
            obj_uids, ee_uids
        )

        trace_buffer, ctx_buffer = [], []
        detailed_buffer, viz_frames = [], []
        succ = False
        succ_steps = -1

        viz_frames.append(obs["rgb"])

        # [obj-states|ee-state|ee-act]
        cur_state_T_wl = np.asarray(
            [obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
        )
        ee_T_wl = cur_state_T_wl[-1].copy()
        trace_buffer.append(
            np.concatenate([cur_state_T_wl, cur_state_T_wl[-1:].copy()], axis=0)
        )

        action_tgt_buf = []

        # record viz
        optimization_value_records = []
        optimization_trace_records = []
        optimization_trace_mask_records = []
        for timestep in tqdm(range(max_steps)):
            # action = env.action_space.sample()
            if timestep % act_horizon == 0:
                # * update the act buffer

                # * 1. gen temp goal
                current_goal, gen_trace = self.infer_agent(
                    agent=agent,
                    hist_trace=trace_buffer[-1:],
                    group_id=group_id,
                    sem_type=sem_type,
                    local_pcl=local_pcl,
                    window=window,
                    fps=env._control_frequency,
                    device=device,
                    seed=seed,
                    instruction=instruction,  # ! with language
                )

                # * 2. define the value function
                def _value_func(gen_trace: torch.Tensor):
                    """Compute per-candidate position-matching cost.

                    Args:
                        gen_trace (torch.Tensor): Candidate traces (K, T, N, 4, 4).

                    Returns:
                        torch.Tensor: Per-candidate value (K,).

                    Example::

                        value = _value_func(gen_trace)
                    """
                    # gen_trace: K,T,N,4,4
                    # use the last act time to compare
                    _s = gen_trace[:, act_horizon, :-2].clone()  # B,O,4,4
                    _g = current_goal[:-2].detach().clone()  # O,4,4
                    _pos_diff = (_s[:, :, :2, 3] - _g[None, :, :2, 3]).norm(dim=-1)
                    _value = _pos_diff.mean(dim=1)
                    return _value

                # * 3. MPC: start from baseline (generated action) and randomly augmented actions, optimize the actions, with no condition on any text or goal, such that the ending state looks more like the imagined goal.
                a0 = gen_trace[None, 1 : min(window, 1 + act_horizon), -1:].expand(
                    K, -1, -1, -1, -1
                )  # K,T-1,1, 4,4
                noise = torch.randn(K, a0.shape[1], 1, 1, 2, device=device) * noise_std
                noise[0] = 0.0
                a0_start = a0.clone()
                a0_start[:, :, :, :, :2] += noise  # B,T-1,1,2

                # Make sure agent is frozen:
                for p in agent.parameters():
                    p.requires_grad = False
                with torch.set_grad_enabled(True):

                    param = a0_start.detach().clone().to(device)
                    param = param.requires_grad_(True)
                    # optimizer = torch.optim.Adam([param], lr=lr)
                    optimizer = torch.optim.SGD([param], lr=lr_list[0])

                    value_record = []
                    gen_trace_record = []
                    gen_trace_known_mask_record = []
                    for i_inner_loop in range(inner_loop_steps):
                        # Adjust learning rate from the lr list
                        if i_inner_loop < len(lr_list):
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr_list[i_inner_loop]
                        optimizer.zero_grad()

                        N = len(group_id)
                        pad = window - 1
                        pad_trace = torch.eye(4).view(1, 1, 4, 4).expand(pad, N, 4, 4)
                        trace = torch.cat(
                            [
                                torch.as_tensor(
                                    trace_buffer[-1:], dtype=torch.float32
                                ).clone(),
                                pad_trace,
                            ]
                        ).to(device)
                        trace = trace[None].repeat(K, 1, 1, 1, 1).to(device)
                        trace[:, 1 : 1 + a0.shape[1], -1:] = (
                            trace[:, 1 : 1 + a0.shape[1], -1:].clone() * 0.0
                            + param.clone()
                        )
                        trace_mask = torch.ones((K, window, N))
                        trace_mask[:, 1:] = False
                        trace_mask[:, 1 : 1 + a0.shape[1], -1] = True
                        context = (
                            torch.eye(4).view(1, 1, 4, 4).expand(K, N, 4, 4).to(device)
                        )[:, None]
                        context_mask = torch.zeros((K, 1, N))
                        realtime = (
                            torch.arange(window, dtype=torch.float32) - 1 + 1
                        )  # !
                        realtime = realtime / env._control_frequency
                        realtime = realtime[None].repeat(K, 1).to(device)

                        # gen infer
                        _gen_context, _gen_trace = agent.inference_with_grad(
                            trace=trace,
                            context=context,
                            trace_known_mask=trace_mask.bool().to(device),
                            context_known_mask=context_mask.bool().to(device),
                            # ! all rows useful
                            trace_mask=torch.ones_like(trace_mask).bool().to(device),
                            context_mask=torch.ones_like(context_mask)
                            .bool()
                            .to(device),
                            #
                            real_timestep=realtime.to(device),
                            group_id=group_id[None].expand(K, -1).to(device),
                            sem_type=sem_type[None].expand(K, -1).to(device),
                            rot_mask=torch.ones((K, N)).bool().to(device),
                            #
                            local_pcl=local_pcl[None].expand(K, -1, -1, -1).to(device),
                            local_pcl_mask=torch.ones((K, N)).bool().to(device),
                            text=[""] * K,
                            seed=seed,
                            left2right_AR_sweep_flag=False,
                            verbose=False,
                        )

                        value = _value_func(_gen_trace)
                        value_record.append(value.detach().cpu().numpy())
                        loss = value.mean()
                        loss.backward()
                        optimizer.step()
                        # logging.info(
                        #     f"loss: {loss.item()}, value: {value}"
                        # )
                        _gen_trace = _gen_trace[:, : 1 + act_horizon]
                        _gen_trace_known_mask = trace_mask[:, : 1 + act_horizon]
                        gen_trace_record.append(_gen_trace.detach().cpu().numpy())
                        gen_trace_known_mask_record.append(
                            _gen_trace_known_mask.detach().cpu().numpy()
                        )

                        # print(
                        #     abs(
                        #         _gen_trace[:, :1].detach().cpu().numpy()
                        #         - trace_buffer[-1]
                        #     ).max()
                        # )

                # todo: record the inner loop and viz
                # value_record = self.__plot_to_ndarray__(value_record)
                # imageio.imsave(f"value_record_{timestep}.jpg", value_record)
                optimization_value_records.append(value_record)
                optimization_trace_records.append(gen_trace_record)
                optimization_trace_mask_records.append(gen_trace_known_mask_record)

                detailed_buffer.append((gen_trace.cpu().numpy(), 1, timestep))

                select_idx = value.argmin()
                logging.info(f"select_idx: {select_idx}")
                # future_action_T = gen_trace[1 :, -1]
                future_action_T = param[select_idx.item(), :, -1].detach().cpu().numpy()
                for i in range(len(future_action_T)):
                    if timestep + i >= max_steps:
                        break
                    if len(action_tgt_buf) <= timestep + i:
                        action_tgt_buf.append([])
                    action_tgt_buf[timestep + i].append(future_action_T[i])

            ######################

            action_T = action_tgt_buf[timestep]
            assert len(action_T) > 0
            action_T = action_T[-1]  # latest pred
            applied_act = action_T[:2, 3] - ee_T_wl[:2, 3]

            # Step environment
            next_obs, reward, done, info = env.step(applied_act)
            viz_frames.append(next_obs["rgb"])

            # update
            ctx_buffer.append(current_goal.cpu().numpy())

            # ! [obj-states|ee-state|ee-act]
            cur_state_T_wl = np.asarray(
                [next_obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
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

        np.random.set_state(rng_state)

        ret_trace = np.stack(trace_buffer, 0)  # T,N,4,4
        ret_ctx = np.stack(ctx_buffer, 0)  # T,N,4,4
        ret_uid = [UID_LANGTABLE_OBJECT_GROUP[uid] for uid in obj_uids] + [
            UID_LANGTABLE_EE,
            UID_LANGTABLE_ACT,
        ]
        ret_pcl = local_pcl

        torch.cuda.empty_cache()
        gc.collect()
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
            optimization_value_records,
            optimization_trace_records,
            optimization_trace_mask_records,
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

        Example::

            viz.langtable_trivial_policy_rollout_viz_routine(
                eval_model, 42, cam, 0, 200, 20, 1, 10)
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
            f"[Succ={succ}, Steps={succ_steps}, W={window}, H={hist}, A={act_horizon}]"
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

        # also visualize the detailed gen trace in the middle
        for i, name in [
            (0, "start"),
            (len(rollout_detailed_gen_trace) // 2, "middle"),
            (len(rollout_detailed_gen_trace) - 1, "end"),
        ]:
            gen_trace, H, timestep = rollout_detailed_gen_trace[i]
            marker_mask = torch.zeros(*gen_trace.shape[:2]).bool()
            marker_mask[H:] = True

            # ! double the true future
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
            viz = put_caption_at_bottom(viz, f"{caption_prefix} {rollout_instruction}")
            self.pl.logger.log_video(
                "deploy",
                f"rollout_detailed_gen_trace_{name}_{postfix}",
                torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=f"{caption_prefix} {rollout_instruction}",
            )

        # viz the rbg and the 3D
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
            viz_ctx.append(np.ones_like(viz_ctx[0]) * 255)
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

    # @torch.inference_mode()
    def langtable_trivial_mpc_rollout_viz_routine(
        self,
        eval_model,
        rollout_seed,
        viz_T_cam,
        batch_idx,
        max_steps,
        window,
        act_horizon,
        K=16,
        noise_std=0.03,
        lr_list=[1.0] * 5,
        viz_skip=1,
        postfix="_mpc",
    ):
        """Run an MPC rollout and log RGB + 3D + optimization visualizations.

        Executes ``rollout_full_states_trivial_mpc``, then logs the
        environment RGB video, per-step optimization dream traces,
        value-iteration plots, detailed generation snapshots, and a
        side-by-side context+trace 3D video.

        Args:
            eval_model: Agent model used for inference.
            rollout_seed (int): Random seed.
            viz_T_cam: Camera-to-world transform.
            batch_idx (int): Batch index for logging.
            max_steps (int): Maximum environment steps.
            window (int): Prediction window.
            act_horizon (int): Action horizon.
            K (int): Number of MPC candidates.
            noise_std (float): Std-dev for noise initialization.
            lr_list (list[float]): Per-iteration learning rates.
            viz_skip (int): Frame skip for 3D visualization.
            postfix (str): Suffix for logger tags.

        Example::

            viz.langtable_trivial_mpc_rollout_viz_routine(
                eval_model, 42, cam, 0, 200, 20, 10)
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
            optimization_value_records,
            optimization_trace_records,
            optimization_trace_mask_records,
        ) = self.rollout_full_states_trivial_mpc(
            agent=eval_model,
            seed=rollout_seed,
            max_steps=max_steps,
            window=window,
            act_horizon=act_horizon,
            K=K,
            noise_std=noise_std,
            lr_list=lr_list,
        )

        postfix = f"W{window}_H{1}_A{act_horizon}{postfix}"
        caption_prefix = (
            f"[Succ={succ},Steps={succ_steps},W={window},H={1},A={act_horizon}]"
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

        # * plot dream traces
        max_n_sampeles = 8
        for i, name in [
            (0, "start"),
            (len(optimization_trace_records) // 2, "middle"),
            (len(optimization_trace_records) - 1, "end"),
        ]:
            viz_dream_traces = []
            _trace_records = optimization_trace_records[i]  # list of steps
            _trace_mask_records = optimization_trace_mask_records[i]
            for _step in tqdm(range(len(_trace_records))):
                # each step forms one big image
                _frame = []
                for _sample_idx in range(
                    min(max_n_sampeles, len(_trace_records[_step]))
                ):
                    _trace = _trace_records[_step][_sample_idx]
                    _knonw_mask = _trace_mask_records[_step][_sample_idx] > 0
                    _unknown_mask = ~_knonw_mask
                    _trace_viz = self.render_viz_frames(
                        prompt_type=rollout_uid,
                        T_wl=torch.as_tensor(_trace),
                        viz_T_cam=viz_T_cam,
                        local_pcl=torch.as_tensor(rollout_pcl),
                        local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
                        marking_mask=_unknown_mask,
                        H=360,
                        W=640,
                    )
                    _trace_viz = np.concatenate(_trace_viz, axis=1)  # H,W*(A+1),3
                    _frame.append(_trace_viz)
                _frame = np.concatenate(_frame, axis=0)  # H,W*(A+1)*max_n_sampeles,3
                viz_dream_traces.append(_frame)
            viz_dream_traces = put_caption_at_bottom(
                viz_dream_traces, f"{caption_prefix} {rollout_instruction}"
            )
            self.pl.logger.log_video(
                "deploy",
                f"rollout_dream_traces_{name}_{postfix}",
                np.stack(viz_dream_traces, 0)[..., :3]
                .transpose(0, 3, 1, 2)
                .astype(np.float32)
                / 255.0,
                self.pl.global_step,
                batch_idx,
            )

        # * make the fig of values
        value_fig = []
        for i, v_list in enumerate(optimization_value_records):
            v_list = self.__plot_to_ndarray__(
                v_list, title_prefix=f"Step[{i}] MPC-value-iters"
            )
            value_fig.append(v_list)
        value_fig = make_square_cat_image(value_fig)
        self.pl.logger.log_image(
            "deploy",
            f"deploy/MPC-value-iters_{postfix}",
            value_fig.astype(np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0,
            self.pl.global_step,
            batch_idx,
        )

        # also visualize the detailed gen trace in the middle
        for i, name in [
            (0, "start"),
            (len(rollout_detailed_gen_trace) // 2, "middle"),
            (len(rollout_detailed_gen_trace) - 1, "end"),
        ]:
            gen_trace, H, timestep = rollout_detailed_gen_trace[i]
            marker_mask = torch.zeros(*gen_trace.shape[:2]).bool()
            marker_mask[H:] = True

            # ! double the true future
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
            viz = put_caption_at_bottom(viz, f"{caption_prefix} {rollout_instruction}")
            self.pl.logger.log_video(
                "deploy",
                f"rollout_detailed_gen_trace_{name}_{postfix}",
                torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=f"{caption_prefix} {rollout_instruction}",
            )

        # viz the rbg and the 3D
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
            viz_ctx.append(np.ones_like(viz_ctx[0]) * 255)
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

    def make_viz_T_cam(self, batch=None):
        """Construct the visualization camera pose for Language Table.

        Args:
            batch (dict, optional): Training batch (forwarded to
                ``make_viz_T_cam``).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.

        Example::

            cam = viz.make_viz_T_cam(batch)
        """
        if self.langtable_viz_cfg.get("viz_stand_at", None) is None:
            # * if not specified, use a default one
            viz_T_cam = make_viz_T_cam(self.langtable_viz_cfg, batch)
        else:
            viz_T_cam = make_viz_T_cam(self.langtable_viz_cfg, batch)
        return viz_T_cam

    @torch.inference_mode()
    def langtable_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Top-level visualization routine for Language Table.

        Finds the first batch item with LangTable UIDs, logs GT RGB,
        x0-pred, and autoregressive future-prediction (with/without goal)
        modes.  Optionally runs closed-loop policy rollouts.

        Args:
            batch (dict): Training batch with ``trace``, ``trace_uid``,
                ``context``, ``local_pcl``, ``text``, etc.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference.
            step_output (dict): Training step outputs containing
                ``x0_pred_trace`` and ``x0_pred_context``.

        Example::

            viz.langtable_viz_routine(batch, "val", 0, eval_model, step_output)
        """
        ar_iter1, ar_iter2 = [False], [""]

        if not self.flag:
            return

        viz_T_cam = self.make_viz_T_cam(batch)
        if self.rollout_flag:
            rollout_seed = np.random.randint(0, 10000)
            for _window in self.window:
                for _horizon in self.act_horizon:
                    for _hist in self.hist:
                        self.langtable_trivial_policy_rollout_viz_routine(
                            eval_model,
                            rollout_seed,
                            viz_T_cam,
                            batch_idx,
                            max_steps=self.max_steps,
                            window=_window,
                            hist=_hist,
                            act_horizon=_horizon,
                        )

        # find a bid corresponds to langtable
        bid = None
        for i in range(len(batch["trace"])):
            mask = Agent.get_type_mask(
                batch["trace_uid"][i],
                UID_LANGTABLE_OBJECT_GROUP + [UID_LANGTABLE_EE, UID_LANGTABLE_ACT],
            )
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no human or humanoid, skip viz"
            )
            return

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(f"VIZ phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}")

        # put the video and text caption
        gt_rgb_fn = batch["extra"]["rgbs_path"][bid]
        gt_rgb_indices = batch["extra"]["rgbs_indices"][bid]
        gt_rgb = self.io.load_rgbs(path=gt_rgb_fn, indices=gt_rgb_indices)
        gt_rgb = torch.as_tensor(gt_rgb)
        gt_rgb_frames = put_caption_at_bottom(
            [gt_rgb[i].numpy() for i in range(len(gt_rgb))],
            batch["text"][bid],
        )
        gt_rgb = torch.from_numpy(np.stack(gt_rgb_frames, 0))
        self.pl.logger.log_video(
            phase,
            "gt_rgb",
            gt_rgb.float().permute(0, 3, 1, 2).float() / 255.0,
            self.pl.global_step,
            batch_idx,
            caption=batch["text"][bid],
        )

        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
            """Render and log a single trace+context visualization pair.

            Args:
                T_wl (torch.Tensor): Trace transforms (T, N, 4, 4).
                ref_T_wl (torch.Tensor or None): Context transforms.
                name (str): Logger tag prefix.
                known_n_real (int or list): Known-step boundary for trace.
                known_n_ctx (int or list): Known-step boundary for context.
                seed: Seed label appended to the logger tag.

            Example::

                _subroutine(batch["trace"][0], batch["context"][0],
                    "GT", n_steps, 100000, seed="gt")
            """
            trace_marker_mask = torch.ones(*T_wl.shape[:2]).bool()
            if isinstance(known_n_real, list):
                for i in known_n_real:  # known is not marked
                    trace_marker_mask[i] = False
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
                    for i in known_n_ctx:
                        ctx_marker_mask[i] = False
                else:
                    ctx_marker_mask[:known_n_ctx] = False
                viz = self.render_viz_frames(
                    prompt_type=batch["trace_uid"][bid],
                    T_wl=ref_T_wl,
                    viz_T_cam=viz_T_cam,
                    local_pcl=batch["local_pcl"][bid],
                    local_pcl_mask=batch["local_pcl_mask"][bid],
                    marking_mask=ctx_marker_mask,
                    H=360,
                    W=640,
                )
                viz = make_square_cat_image(viz)
                viz = viz[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"ctx_{name}_seed{seed}",
                    viz[::1, ::1].transpose(2, 0, 1).astype(np.float32) / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

        # * generate
        _subroutine(
            step_output["x0_pred_trace"][bid].float(),
            step_output["x0_pred_context"][bid].float(),
            "X0_PRED",
            -1,
            -1,
            -1,
        )

        # seed = 42
        cfgs = self.pl.cfg
        dataset_trace_mask = batch["trace_mask"][bid]  # T,P
        dataset_ctx_mask = batch["context_mask"][bid]  # C,P

        pad_mask = batch["trace_mask"][bid].any(0) | batch["context_mask"][bid].any(0)
        pad_mask_trace = (
            pad_mask[None].bool().clone().repeat(batch["trace_mask"].shape[1], 1)
        )[None]
        pad_mask_ctx = pad_mask[None, None].bool().clone()

        for seed in [42, 1234]:
            trace_known_mask = dataset_trace_mask[None].bool().clone()
            context_known_mask = dataset_ctx_mask[None].bool().clone()
            n_steps = int((batch["time"][bid] <= 0).sum())
            trace_known_mask[0, n_steps:] = False
            for ar, prefix in zip(ar_iter1, ar_iter2):
                gen_context, gen_trace = eval_model.inference(
                    trace=batch["trace"][bid : bid + 1],
                    context=batch["context"][bid : bid + 1],
                    trace_known_mask=trace_known_mask,
                    context_known_mask=context_known_mask,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid : bid + 1],
                    group_id=batch["trace_group_id"][bid : bid + 1],
                    sem_type=batch["trace_sem_type"][bid : bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                    local_pcl=batch["local_pcl"][bid : bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                    left2right_AR_sweep_flag=ar,
                )
                _subroutine(
                    gen_trace[0], None, f"{prefix}Future(w. goal)", n_steps, -1, seed
                )

            context_known_mask[...] = False
            for cfg in self.add_cfg:
                for ar, prefix in zip(ar_iter1, ar_iter2):
                    gen_context, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=context_known_mask,
                        trace_mask=pad_mask_trace,
                        context_mask=pad_mask_ctx,
                        real_timestep=batch["time"][bid : bid + 1],
                        group_id=batch["trace_group_id"][bid : bid + 1],
                        sem_type=batch["trace_sem_type"][bid : bid + 1],
                        rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                        local_pcl=batch["local_pcl"][bid : bid + 1],
                        local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                        text=[batch["text"][bid]],
                        seed=seed,
                        cfg_scale=cfg,
                        left2right_AR_sweep_flag=ar,
                    )
                    _subroutine(
                        gen_trace[0],
                        gen_context[0],
                        f"{prefix}Future(no goal)_{cfg}",
                        n_steps,
                        0,
                        seed,
                    )

            context_known_mask[...] = False
            cur_i = int((batch["time"][bid] <= 0).sum()) - 1
            trace_known_mask[...] = False
            trace_known_mask[0, cur_i] = True
            for cfg in self.add_cfg:
                for ar, prefix in zip(ar_iter1, ar_iter2):
                    gen_context, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=context_known_mask,
                        trace_mask=pad_mask_trace,
                        context_mask=pad_mask_ctx,
                        real_timestep=batch["time"][bid : bid + 1],
                        group_id=batch["trace_group_id"][bid : bid + 1],
                        sem_type=batch["trace_sem_type"][bid : bid + 1],
                        rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                        local_pcl=batch["local_pcl"][bid : bid + 1],
                        local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                        text=[batch["text"][bid]],
                        seed=seed,
                        cfg_scale=cfg,
                        left2right_AR_sweep_flag=ar,
                    )
                    _subroutine(
                        gen_trace[0],
                        gen_context[0],
                        f"{prefix}OnlyCur(no goal)_{cfg}",
                        [cur_i],
                        0,
                        seed,
                    )

        # * GT
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        _subroutine(
            batch["trace"][bid, :n_steps],
            batch["context"][bid],
            "GT",
            n_steps,
            100000,
            seed="gt",
        )

        return
