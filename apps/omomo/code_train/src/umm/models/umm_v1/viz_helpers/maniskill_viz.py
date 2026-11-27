"""Visualization helpers for ManiSkill tabletop manipulation domain.

Renders robot end-effector, finger, and object meshes/point-clouds,
supports closed-loop policy rollouts with visualization, and provides
GT, x0-pred, and autoregressive future-prediction logging modes.

Example::

    viz = ManiskillVisualizer(pl_module, maniskill_cfg)
    viz.maniskill_viz_routine(batch, "val", 0, eval_model, step_output)
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
import os
import gc

from umm.datasets.umm_v1.dataset_config import (
    UID_MANISKILL_TABLETOP_EE,
    UID_MANISKILL_TABLETOP_ACT,
    UID_MANISKILL_TABLETOP_LEFT_FINGER,
    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    ######
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
)
from lib_io import UnifiedIO

from umm.models.umm_v1.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_v1.agent import Agent
from umm.models.umm_v1.viz_helpers.mesh_visualizer import visualize_a_mesh

from lib_mani.utils.env_utils import make_maniskill_viz_env
from lib_mani.utils.action_utils import tgt_to_state_action
from umm.datasets.umm_v1.dataset_config import (
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
)
from lib_mani.utils.misc_utils import get_total_pose_matrix

# Constants
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class ManiskillVisualizer:
    """
    Visualizer for Maniskill environment trajectories.
    Handles rendering of robot, objects, and point clouds.
    """

    def __init__(self, pl_module, maniskill_cfg: dict):
        """Initialize the ManiSkill visualizer.

        Loads per-task robot and object meshes from ``.npz`` archives
        specified in ``maniskill_cfg``. Also reads rollout parameters
        (window, hist, act_horizon) for closed-loop evaluation.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            maniskill_cfg (dict): Per-task config with ``model_path``,
                ``object_path``, and ``num_pts`` keys.

        Example::

            viz = ManiskillVisualizer(pl_module, cfg["visualize"]["maniskill"])
        """
        self.pl = pl_module
        if maniskill_cfg == {} or len(maniskill_cfg) == 0 or maniskill_cfg is None:
            self.flag = False
            self.robot_data = None
            return
        self.flag = True

        # build dict of assets for different tasks
        self.tasks = maniskill_cfg.keys()

        self.assets = {}

        def load_mesh(data, local_id):
            """Load a trimesh from an npz archive by local actor ID.

            Args:
                data (np.lib.npyio.NpzFile): Loaded ``.npz`` archive.
                local_id (int): Local actor identifier used as key prefix.

            Returns:
                trimesh.Trimesh: The loaded mesh.

            Raises:
                ValueError: If vertex key is not found in ``data``.

            Example::

                palm = load_mesh(robot_data, 3)
            """
            key_v = f"{local_id}_v"
            key_f = f"{local_id}_f"
            key_c = f"{local_id}_c"
            if key_v in data:
                _v = data[key_v]
                _f = data[key_f]
                _c = data[key_c]
                tm = trimesh.Trimesh(vertices=_v, faces=_f, vertex_colors=_c)
                return tm
            else:
                logging.warning(f"Mesh not found for uid: {local_id}")
                raise ValueError(f"Mesh not found for uid: {local_id}")

        for task in self.tasks:
            logging.info(f"Loading assets for task: {task}")
            model_path = maniskill_cfg[task].get("model_path", "")
            object_path = maniskill_cfg[task].get("object_path", "")
            states_dir = osp.join(osp.dirname(model_path), "states")
            states_files = [f for f in os.listdir(states_dir) if f.endswith(".npz")]
            states_files.sort()
            state_data = np.load(
                osp.join(states_dir, states_files[0]), allow_pickle=True
            )
            sem_type_uid = state_data["sem_type_uid"].item()
            ee_uid_local = sem_type_uid["endeffector"][0]
            left_finger_uid_local = sem_type_uid["gripper_left_finger"][0]
            right_finger_uid_local = sem_type_uid["gripper_right_finger"][0]

            num_pts = maniskill_cfg[task].get("num_pts", 256)

            robot_data = np.load(model_path)
            object_data = np.load(object_path)

            palm_mesh = load_mesh(robot_data, ee_uid_local)
            left_finger_mesh = load_mesh(robot_data, left_finger_uid_local)
            right_finger_mesh = load_mesh(robot_data, right_finger_uid_local)

            object_uids = [k[:-2] for k in object_data.keys() if k.endswith("_v")]

            task_obj_uids = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[task]
            meshes = {
                task_obj_uids[int(k)]: load_mesh(object_data, int(k))
                for k in object_uids
            }
            meshes[UID_MANISKILL_TABLETOP_EE] = palm_mesh
            meshes[UID_MANISKILL_TABLETOP_ACT] = palm_mesh
            meshes[UID_MANISKILL_TABLETOP_LEFT_FINGER] = left_finger_mesh
            meshes[UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT] = left_finger_mesh
            meshes[UID_MANISKILL_TABLETOP_RIGHT_FINGER] = right_finger_mesh
            meshes[UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT] = right_finger_mesh
            self.assets[task] = {
                "meshes": meshes,
                "num_pts": num_pts,
                "sem_type_uid": sem_type_uid,
                "robot_data": robot_data,
                "object_data": object_data,
            }

        self.maniskill_viz_cfg = self.pl.cfg.get("visualize", {}).get("maniskill", {})
        self.add_cfg = self.maniskill_viz_cfg.get("add_cfg", [1.0])

        # Rollout configurations
        self.rollout_flag = self.maniskill_viz_cfg.get("rollout", True)
        self.window = self.maniskill_viz_cfg.get("rollout_window", [20])
        self.hist = self.maniskill_viz_cfg.get("rollout_hist", [1])
        self.act_horizon = self.maniskill_viz_cfg.get("rollout_act_horizon", [10])
        self.max_steps = self.maniskill_viz_cfg.get("rollout_max_steps", 200)

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

            blanks = self.__make_dummy_frames__(10, 512, 512)
        """
        frames = []
        for _ in range(T):
            dummy_frame = np.zeros((H, W, 4), dtype=np.uint8)
            frames.append(dummy_frame)
        return frames

    def prepare_trace_attr_oracle(self, task):
        """Prepare trace attributes for oracle rollout.

        Assembles group IDs, semantic types, local point clouds, and
        actor UIDs for the given task by combining object and robot
        actor data from pre-loaded assets.

        Args:
            task (str): ManiSkill task name (e.g. ``"PickCube-v0"``).

        Returns:
            tuple: ``(group_ids, sem_types, local_pcl, actor_uids)`` where
                group_ids is (N,) long tensor, sem_types is (N,) long tensor,
                local_pcl is (N, PCL, 6) float tensor, and actor_uids is a
                list of UID ints.

        Example::

            gid, sem, pcl, uids = viz.prepare_trace_attr_oracle("PickCube-v0")
        """
        sem_type_uid = self.assets[task]["sem_type_uid"]
        task_obj_uids = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[task]

        actor_uids = []
        sem_types = []
        local_pcl = []
        group_ids = []

        # ---------------------
        # 1. OBJECTS (Group ID = 1)
        # ---------------------
        obj_local_uids = sem_type_uid["objects"]
        object_gid = 0

        for local_uid in obj_local_uids:
            actor_uids.append(task_obj_uids[local_uid])
            sem_types.append(SEM_OBJECT)
            group_ids.append(object_gid)

            # Colors and Point Cloud Extraction for Objects
            _pcl_all = self.assets[task]["object_data"][f"{local_uid}_pcl"]
            if f"{local_uid}_c" in self.assets[task]["object_data"]:
                _c = self.assets[task]["object_data"][f"{local_uid}_c"]
                _c = _c.astype(np.float32)
                if _c.max() > 1.0:
                    _c = _c / 255.0
                _pcl_c = np.tile(_c[None, :], (_pcl_all.shape[0], 1)).astype(np.float32)
            else:
                raise ValueError(f"Object {local_uid} missing color data")

            _pcl = np.concatenate([_pcl_all, _pcl_c], axis=-1)
            _pcl = sample_list(_pcl, self.assets[task]["num_pts"], random_flag=False)
            local_pcl.append(_pcl)

        # ---------------------
        # 2. ACTORS (Group ID = 2)
        # ---------------------
        ee_uid = sem_type_uid["endeffector"][0]
        left_finger_uid = sem_type_uid["gripper_left_finger"][0]
        right_finger_uid = sem_type_uid["gripper_right_finger"][0]

        actor_uids += [
            UID_MANISKILL_TABLETOP_EE,
            UID_MANISKILL_TABLETOP_LEFT_FINGER,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER,
            UID_MANISKILL_TABLETOP_ACT,
            UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]

        sem_types += [
            SEM_MANISKILL_TABLETOP_EE,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
            SEM_MANISKILL_TABLETOP_ACT,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]

        actor_group_id = 1
        group_ids += [actor_group_id] * 6

        actor_pcl = []
        for _uid in [ee_uid, left_finger_uid, right_finger_uid]:
            _pcl_all = self.assets[task]["robot_data"][f"{_uid}_pcl"]
            if f"{_uid}_c" in self.assets[task]["robot_data"]:
                _c = self.assets[task]["robot_data"][f"{_uid}_c"]
                _c = _c.astype(np.float32)
                if _c.max() > 1.0:
                    _c = _c / 255.0
                _pcl_c = np.tile(_c[None, :], (_pcl_all.shape[0], 1)).astype(np.float32)
            else:
                raise ValueError(f"Actor {_uid} missing color data")
            _pcl = np.concatenate([_pcl_all, _pcl_c], axis=-1)
            _pcl = sample_list(_pcl, self.assets[task]["num_pts"], random_flag=False)
            actor_pcl.append(_pcl)
        # Combinations logic
        actor_pcl = actor_pcl + actor_pcl
        local_pcl += actor_pcl

        # Stack everything
        local_pcl = np.stack(local_pcl, axis=0)  # N, PCL, 6
        group_ids = np.array(group_ids, dtype=np.int64)  # N
        sem_types = np.array(sem_types, dtype=np.int64)  # N

        local_pcl = torch.as_tensor(local_pcl, dtype=torch.float32)
        group_ids = torch.as_tensor(group_ids, dtype=torch.long)
        sem_types = torch.as_tensor(sem_types, dtype=torch.long)

        # ! order [Objects|Palm|Left Finger|Right Finger|PalmACT|Left FingerACT|Right FingerACT]
        return group_ids, sem_types, local_pcl, actor_uids

    def extract_poses_from_env(self, task, env):
        """Extract current SE(3) poses from a live ManiSkill environment.

        Reads actor and robot link poses via the environment's scene API
        and stacks them in the canonical trace order (objects, then
        EE/fingers).

        Args:
            task (str): ManiSkill task name.
            env: Live ManiSkill environment instance.

        Returns:
            np.ndarray: Stacked poses (N, 4, 4).

        Example::

            T_wl = viz.extract_poses_from_env("PickCube-v0", env)
        """
        sem_type_uid = self.assets[task]["sem_type_uid"]

        uid_to_T_wl = {}
        for actor in env.unwrapped.scene.get_all_actors():
            if hasattr(actor, "per_scene_id"):
                local_uid = int(actor.per_scene_id)
                uid_to_T_wl[local_uid] = get_total_pose_matrix(actor)
        for link in env.unwrapped.agent.robot.get_links():
            if hasattr(link, "per_scene_id"):
                local_uid = int(link.per_scene_id)
                uid_to_T_wl[local_uid] = get_total_pose_matrix(link)
        ret_states = []

        # 1. OBJECTS
        obj_local_uids = sem_type_uid["objects"]
        for local_uid in obj_local_uids:
            if local_uid not in uid_to_T_wl:
                ret_states.append(np.eye(4))
                continue
            ret_states.append(uid_to_T_wl[local_uid])

        # 2. ACTORS
        ee_uid = sem_type_uid["endeffector"][0]
        left_finger_uid = sem_type_uid["gripper_left_finger"][0]
        right_finger_uid = sem_type_uid["gripper_right_finger"][0]

        palm_T_wl = uid_to_T_wl.get(ee_uid, np.eye(4))
        left_finger_T_wl = uid_to_T_wl.get(left_finger_uid, np.eye(4))
        right_finger_T_wl = uid_to_T_wl.get(right_finger_uid, np.eye(4))

        ret_states += [palm_T_wl, left_finger_T_wl, right_finger_T_wl]

        ret_states = np.stack(ret_states, axis=0)  # N, 4, 4
        return ret_states

    def render_viz_frames(
        self,
        task,
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
        marking_mask=None,
        marking_alpha=0.75,
        viz_mask=None,
        pcl_mode=False,
    ):
        """Render 3D visualization frames for a ManiSkill trajectory.

        For each timestep, builds a combined mesh (or point-cloud) scene
        from all visible entities, applies per-UID coloring and
        marking highlights, and renders via ``visualize_a_mesh``.

        Args:
            task (str): ManiSkill task name for mesh lookup.
            prompt_type (torch.Tensor or list): Per-entity UID (N,).
            T_wl (torch.Tensor): World-to-link transforms (T, N, 4, 4).
            viz_T_cam: Camera-to-world transform (4, 4).
            local_pcl (torch.Tensor): Local point clouds (N, PCL, 6).
            local_pcl_mask (torch.Tensor): Boolean mask (N,).
            H (int): Output height.
            W (int): Output width.
            local_pcl_max_N (int): Max points per local cloud.
            local_pcl_r (float): Sphere radius for PCL mode.
            pcl_alpha (float): Default alpha.
            marking_mask (torch.Tensor, optional): Highlight mask (T, N).
            marking_alpha (float): Alpha for highlighted entities.
            viz_mask (torch.Tensor, optional): Visibility mask (T, N).
            pcl_mode (bool): If True, render as point-cloud spheres
                instead of meshes.

        Returns:
            list[np.ndarray]: Rendered RGBA frames (H, W, 4).

        Example::

            frames = viz.render_viz_frames("PickCube-v0", uid, T_wl,
                cam, pcl, pcl_mask)
        """
        if self.flag is False:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)
        ALL_MESHES = self.assets[task]["meshes"]

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

                _pcl = local_pcl[i]
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]

                # Colors
                _rgb = _pcl[:, 3:].clone().cpu().numpy()  # Use proper numpy mapping
                uid = int(prompt_type[i])

                if uid in [
                    UID_MANISKILL_TABLETOP_EE,
                    UID_MANISKILL_TABLETOP_LEFT_FINGER,
                    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
                ]:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 2] = 1.0  # Blue-ish
                elif uid in [
                    UID_MANISKILL_TABLETOP_ACT,
                    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
                    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
                ]:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0  # Yellow-ish
                    _rgb[:, 1] = 1.0

                # Marking
                if marking_mask[t, i]:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = 1.0
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)
                else:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    # _hsv[..., 1] = _hsv[..., 1] * 0.7
                    # Retain original lighting
                    # _hsv[..., 2] = _hsv[..., 2] * 0.9 + 0.1
                    _rgb = mcolors.hsv_to_rgb(_hsv)

                _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha

                # Render
                # Use mesh if available and not pcl_mode
                use_mesh = (not pcl_mode) and (uid in ALL_MESHES)
                if not pcl_mode and uid not in ALL_MESHES:
                    print(f"Warning: Mesh not found for uid: {uid}")

                if not use_mesh:
                    # PCL mode
                    _pcl_t = (T_wl[t, i][:3, :3] @ _pcl[:, :3].T).T + T_wl[t, i][
                        None, :3, 3
                    ]
                    _m_pcl = []
                    for _idx in range(len(_pcl_t)):
                        sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                        sph.apply_translation(_pcl_t[_idx].cpu().numpy())
                        sph.visual.vertex_colors = _rgb[_idx].tolist() + [_pcl_alpha]
                        _m_pcl.append(sph)
                    if len(_m_pcl) > 0:
                        _m_pcl = trimesh.util.concatenate(_m_pcl)
                        _mm.append(_m_pcl)
                else:
                    # Mesh mode
                    _mesh = ALL_MESHES[uid].copy()
                    _mesh.apply_transform(T_wl[t, i].cpu().numpy())
                    # Ensure color is parsed properly 0-255
                    mean_color = np.clip(_rgb.mean(0) * 255.0, 0, 255).astype(np.uint8)
                    _mesh.visual.vertex_colors = mean_color.tolist() + [
                        int(_pcl_alpha * 255)
                    ]
                    _mm.append(_mesh)

            if len(_mm) > 0:
                viz_m = trimesh.util.concatenate(_mm)
                _fr = visualize_a_mesh(
                    cam_T_wc=viz_T_cam,
                    mesh=viz_m,
                    background_color=BACKGROUND_COLOR,
                    ground_color=GROUND_COLOR,
                    lighting=True,
                    ground_z=0.0,
                    ground_grid_size=0.5,
                    width=W,
                    height=H,
                )
                frames.append(_fr)
            else:
                frames.append(np.ones((H, W, 4), dtype=np.uint8) * 255)  # Empty

        return frames

    def make_viz_T_cam(self, batch=None):
        """Construct the visualization camera pose for ManiSkill.

        Args:
            batch (dict, optional): Training batch (currently unused
                beyond forwarding to ``make_viz_T_cam``).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.

        Raises:
            NotImplementedError: If ``viz_stand_at`` is not set in config.

        Example::

            cam = viz.make_viz_T_cam(batch)
        """
        if self.maniskill_viz_cfg.get("viz_stand_at", None) is not None:
            viz_T_cam = make_viz_T_cam(self.maniskill_viz_cfg, batch)
        else:
            # viz_T_cam = move_camera_T_wc(self.maniskill_viz_cfg, batch)
            raise NotImplementedError("viz_stand_at is not implemented")
        return viz_T_cam

    @torch.inference_mode()
    def rollout_full_states_trivial_policy(
        self,
        agent,
        task,
        seed,
        max_steps,
        hist,
        window,
        act_horizon,
        device=torch.device("cuda:0"),
    ):
        """Perform a closed-loop rollout in ManiSkill using an oracle policy.

        Runs the agent in a receding-horizon loop: every ``act_horizon``
        steps, queries the agent for a future-trace prediction, converts
        it to ``pd_ee_delta_pose`` actions, and steps the environment.

        Args:
            agent: The inference agent (``Agent`` nn.Module).
            task (str): ManiSkill task name.
            seed (int): Random seed for reproducibility.
            max_steps (int): Maximum environment steps.
            hist (int): Number of history frames to condition on.
            window (int): Total prediction window length.
            act_horizon (int): Steps between re-planning.
            device (torch.device): Compute device.

        Returns:
            tuple: ``(succ, succ_steps, trace, ctx, uid, pcl,
                viz_frames, instruction, detailed_buffer)``

        Example::

            result = viz.rollout_full_states_trivial_policy(
                agent, "PickCube-v0", seed=42, max_steps=200,
                hist=1, window=20, act_horizon=10)
        """
        logging.info(f"maniskill_rollout seed={seed} task={task}")
        rng_state = np.random.get_state()
        np.random.seed(seed)

        env = make_maniskill_viz_env(
            task,
            seed=seed,
            control_mode="pd_ee_delta_pose",
            sim_backend="physx_cpu",
        )
        obs, info = env.reset(seed=seed)

        # ! use task as instruction as the dataset does
        instruction = task

        group_id, sem_type, local_pcl, trace_uids = self.prepare_trace_attr_oracle(task)

        cur_state_T_wl = self.extract_poses_from_env(task, env)
        cur_state_T_wl = np.concatenate([cur_state_T_wl, cur_state_T_wl[-3:]], axis=0)

        trace_buffer, ctx_buffer = [], []
        detailed_buffer, viz_frames = [], []
        succ = False
        succ_steps = -1

        # To get the initial RGB image, we can just use the render function
        # Or if OBSWrapper isn't used, we can call env.render()
        img = env.render()
        viz_frames.append(img.detach().cpu())

        # Initial trace buffer: [1, N, 4, 4]
        trace_buffer.append(cur_state_T_wl)

        action_target = []
        for timestep in tqdm(range(max_steps)):
            if timestep % act_horizon == 0:
                hist_trace = np.stack(trace_buffer[-hist:], axis=0)  # H,N,4,4
                H, N, _, _ = hist_trace.shape

                # pad to window
                pad = window - H
                pad_trace = (
                    np.eye(4)[None, None, ...].repeat(pad, axis=0).repeat(N, axis=1)
                )
                trace = np.concatenate([hist_trace, pad_trace], axis=0)
                trace_mask = np.ones((window, N), dtype=bool)
                trace_mask[H:] = False

                context = np.eye(4)[None, None, ...].repeat(1, axis=0).repeat(N, axis=1)
                context_mask = np.zeros((1, N), dtype=bool)

                # realtime relative to the last known frame
                realtime = np.arange(window, dtype=np.float32) - H + 1
                fps = 10.0  # Standard Maniskill FPS
                realtime = realtime / fps

                gen_context, gen_trace = agent.inference(
                    trace=torch.as_tensor(trace)[None].float().to(device),
                    context=torch.as_tensor(context)[None].float().to(device),
                    trace_known_mask=torch.as_tensor(trace_mask)[None].to(device),
                    context_known_mask=torch.as_tensor(context_mask)[None].to(device),
                    trace_mask=torch.ones(
                        (1, window, N), dtype=torch.bool, device=device
                    ),
                    context_mask=torch.ones((1, 1, N), dtype=torch.bool, device=device),
                    real_timestep=torch.as_tensor(realtime)[None].to(device),
                    group_id=group_id[None].to(device),
                    sem_type=sem_type[None].to(device),
                    rot_mask=torch.ones((1, N), dtype=torch.bool, device=device),
                    local_pcl=local_pcl[None].to(device),
                    local_pcl_mask=torch.ones((1, N), dtype=torch.bool, device=device),
                    text=[instruction],
                    seed=seed,
                    left2right_AR_sweep_flag=False,
                    verbose=False,
                )

                gen_context = gen_context[0, 0]  # N,4,4
                gen_trace = gen_trace[0]  # window,N,4,4

                detailed_buffer.append(
                    (gen_trace.cpu().numpy(), len(hist_trace), timestep)
                )

                future_trace = gen_trace[len(hist_trace) :]
                for i in range(len(future_trace)):
                    if timestep + i >= max_steps:
                        break
                    if len(action_target) <= timestep + i:
                        action_target.append([])
                    action_target[timestep + i].append(future_trace[i].cpu().numpy())

            # Compute action for environment
            action_T = action_target[timestep].copy()
            assert len(action_T) > 0
            action_T = action_T[-1].copy()  # latest pred, shape N,4,4

            palm_idx = trace_uids.index(UID_MANISKILL_TABLETOP_EE)
            act_palm_idx = trace_uids.index(UID_MANISKILL_TABLETOP_ACT)
            act_left_idx = trace_uids.index(UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT)
            act_right_idx = trace_uids.index(UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT)

            # Use inverse kinematics / action conversion to get PD EE delta action
            current_palm_pose = torch.from_numpy(cur_state_T_wl[palm_idx]).float()
            target_palm_pose = torch.from_numpy(action_T[act_palm_idx]).float()
            target_left_finger_pose = torch.from_numpy(action_T[act_left_idx]).float()
            target_right_finger_pose = torch.from_numpy(action_T[act_right_idx]).float()

            applied_act = tgt_to_state_action(
                current_palm_pose=current_palm_pose,
                target_palm_pose=target_palm_pose,
                target_left_finger_pose=target_left_finger_pose,
                target_right_finger_pose=target_right_finger_pose,
                pos_limit=0.1,
                rot_limit=0.1,
            )

            # The control mode is pd_ee_delta_pose which takes [dx, dy, dz, drx, dry, drz, gripper]
            if applied_act.ndim == 1:
                applied_act = applied_act.unsqueeze(0)
            applied_act = applied_act.numpy()
            # logging.warning(f"[{timestep}]act={applied_act}" )

            # Step environment
            next_obs, reward, done, truncated, info = env.step(applied_act)
            viz_frames.append(env.render().detach().cpu().numpy())
            ctx_buffer.append(gen_context.detach().cpu().numpy())

            # Update trace from environment
            cur_state_T_wl = self.extract_poses_from_env(task, env)
            cur_state_T_wl = np.concatenate(
                [
                    cur_state_T_wl,
                    action_T[act_palm_idx][None],
                    action_T[act_left_idx][None],
                    action_T[act_right_idx][None],
                ],
                axis=0,
            )
            trace_buffer.append(cur_state_T_wl)

            if done or info.get("success", False):
                succ = True
                succ_steps = timestep
                break

        logging.warning(f"seed: {seed}, succ: {succ}, succ_steps: {succ_steps}")
        np.random.set_state(rng_state)

        ret_trace = np.stack(trace_buffer, 0)
        ret_ctx = np.stack(ctx_buffer, 0)
        ret_uid = trace_uids
        ret_pcl = local_pcl.cpu().numpy()

        viz_frames = [it.squeeze(0) for it in viz_frames]

        torch.cuda.empty_cache()
        gc.collect()

        if hasattr(env.unwrapped, "close"):
            env.unwrapped.close()
        elif hasattr(env, "close"):
            env.close()

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
    def maniskill_trivial_policy_rollout_viz_routine(
        self,
        eval_model,
        rollout_seed,
        viz_T_cam,
        batch_idx,
        task,
        max_steps,
        window,
        hist,
        act_horizon,
        viz_skip=1,
        postfix="_policy",
    ):
        """Run a rollout and log the RGB + 3D visualization videos.

        Executes ``rollout_full_states_trivial_policy``, then logs the
        environment RGB video, a side-by-side context+trace 3D video,
        and detailed generation snapshots at start/middle/end.

        Args:
            eval_model: Agent model used for inference.
            rollout_seed (int): Random seed.
            viz_T_cam: Camera-to-world transform.
            batch_idx (int): Batch index for logging.
            task (str): ManiSkill task name.
            max_steps (int): Maximum environment steps.
            window (int): Prediction window.
            hist (int): History length.
            act_horizon (int): Action horizon.
            viz_skip (int): Frame skip for 3D visualization.
            postfix (str): Suffix for logger tags.

        Example::

            viz.maniskill_trivial_policy_rollout_viz_routine(
                eval_model, 42, cam, 0, "PickCube-v0", 200, 20, 1, 10)
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
            task=task,
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

        # We also need to get the side-by-side video rendering with context (gen_context -> rendered)
        # However, for simplicity here, we only have viz_frames (the RGB camera renders from ManiSkill)
        # If we want the goal frame side by side, we can use the gen_context 3D renders.

        rollout_viz_frames = put_caption_at_bottom(
            rollout_viz_frames, f"{caption_prefix} {rollout_instruction}"
        )
        self.pl.logger.log_video(
            "deploy",
            f"rollout_rgb_{task}_{postfix}",
            np.stack(rollout_viz_frames, 0).transpose(0, 3, 1, 2).astype(np.float32)
            / 255.0,
            self.pl.global_step,
            batch_idx,
            caption=f"{caption_prefix} {rollout_instruction}",
        )

        # Render the 3D Trace and Context just like langtable
        viz_trace = self.render_viz_frames(
            task=task,
            prompt_type=rollout_uid,
            T_wl=torch.as_tensor(rollout_trace),
            viz_T_cam=viz_T_cam,
            local_pcl=torch.as_tensor(rollout_pcl),
            local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
            H=512,
            W=512,
            pcl_mode=not self.maniskill_viz_cfg.get("mesh_mode", True),
        )

        marker_mask_ctx = torch.ones(*rollout_ctx.shape[:2]).bool()
        viz_ctx = self.render_viz_frames(
            task=task,
            prompt_type=rollout_uid,
            T_wl=torch.as_tensor(rollout_ctx),
            viz_T_cam=viz_T_cam,
            local_pcl=torch.as_tensor(rollout_pcl),
            local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
            marking_mask=marker_mask_ctx,
            H=512,
            W=512,
            pcl_mode=not self.maniskill_viz_cfg.get("mesh_mode", True),
        )

        # Combine [viz_ctx, viz_trace] horizontally
        if len(viz_ctx) < len(viz_trace):
            viz_ctx.append(np.ones_like(viz_ctx[0]) * 255)

        combined_viz_frames = []
        for i in range(len(viz_trace)):
            combined = np.concatenate(
                [viz_ctx[i][..., :3], viz_trace[i][..., :3]], axis=1
            )
            combined_viz_frames.append(combined)

        combined_viz_frames = put_caption_at_bottom(
            combined_viz_frames, f"{caption_prefix} {rollout_instruction}"
        )
        self.pl.logger.log_video(
            "deploy",
            f"rollout_3dviz_{task}_{postfix}",
            np.stack(combined_viz_frames, 0).transpose(0, 3, 1, 2).astype(np.float32)
            / 255.0,
            self.pl.global_step,
            batch_idx,
            caption=f"{caption_prefix} {rollout_instruction}",
        )

        # also visualize the detailed gen trace in the middle
        # Similar to LangTable implementation
        for i, name in [
            (0, "start"),
            (len(rollout_detailed_gen_trace) // 2, "middle"),
            (len(rollout_detailed_gen_trace) - 1, "end"),
        ]:
            if len(rollout_detailed_gen_trace) <= i:
                continue
            gen_trace, H, timestep = rollout_detailed_gen_trace[i]
            marker_mask = torch.zeros(*gen_trace.shape[:2]).bool()
            marker_mask[H:] = True

            coupled_real_rollout = []
            for j in range(len(gen_trace)):
                if timestep + j < len(rollout_trace):
                    coupled_real_rollout.append(rollout_trace[timestep + j])
                else:
                    coupled_real_rollout.append(rollout_trace[-1])
            coupled_real_rollout = np.stack(coupled_real_rollout, 0)

            viz = self.render_viz_frames(
                task=task,
                prompt_type=rollout_uid,
                T_wl=torch.from_numpy(gen_trace),
                viz_T_cam=viz_T_cam,
                local_pcl=torch.from_numpy(rollout_pcl),
                local_pcl_mask=torch.ones(len(rollout_uid)).bool(),
                marking_mask=marker_mask,
                H=720,
                W=1280,
                pcl_mode=False,
            )
            viz = put_caption_at_bottom(viz, f"gen_{name} {rollout_instruction}")
            self.pl.logger.log_video(
                "deploy",
                f"gen_detailed_{task}_{name}_{postfix}",
                torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=f"gen_{name} {rollout_instruction}",
            )

    @torch.inference_mode()
    def maniskill_viz_routine(
        self, batch, phase, batch_idx, eval_model, step_output=None
    ):
        """Top-level visualization routine for ManiSkill.

        Finds the first ManiSkill batch item, optionally runs closed-loop
        rollouts, then renders GT, x0-pred, and autoregressive
        future-prediction (with/without goal) modes.

        Args:
            batch (dict): Training batch.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference.
            step_output (dict, optional): Training step outputs containing
                ``x0_pred_trace`` and ``x0_pred_context``.

        Example::

            viz.maniskill_viz_routine(batch, "val", 0, eval_model, step_output)
        """
        if not self.flag:
            return

        viz_T_cam = self.make_viz_T_cam(batch)

        # find a bid corresponds to maniskill
        bid = None
        for i in range(len(batch["trace"])):
            # Check if any UID is Maniskill UID
            uids = batch["trace_uid"][i]
            if (
                eval_model.get_type_mask(
                    uids,
                    [
                        UID_MANISKILL_TABLETOP_EE,
                        UID_MANISKILL_TABLETOP_ACT,
                        UID_MANISKILL_TABLETOP_LEFT_FINGER,
                        UID_MANISKILL_TABLETOP_RIGHT_FINGER,
                    ],
                ).any()
                and batch["extra"]["task"][i] in self.tasks
            ):
                bid = i
                break

        if bid is None:
            return

        task = batch["extra"]["task"][bid]

        if self.rollout_flag:
            rollout_seed = np.random.randint(0, 10000)
            for _window in self.window:
                for _horizon in self.act_horizon:
                    for _hist in self.hist:
                        self.maniskill_trivial_policy_rollout_viz_routine(
                            eval_model,
                            rollout_seed,
                            viz_T_cam,
                            batch_idx,
                            task=task,
                            max_steps=self.max_steps,
                            window=_window,
                            hist=_hist,
                            act_horizon=_horizon,
                        )

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(
            f"VIZ Maniskill phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}"
        )

        # GT RGB Video
        if "rgbs_path" in batch["extra"]:
            gt_rgb_fn = batch["extra"]["rgbs_path"][bid]
            gt_rgb_indices = batch["extra"]["rgbs_indices"][bid]
            gt_rgb = self.io.load_rgbs(path=gt_rgb_fn, indices=gt_rgb_indices)
            gt_rgb = torch.as_tensor(gt_rgb)
            if len(gt_rgb) > 0:
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
                for i in known_n_real:
                    trace_marker_mask[i] = False
            else:
                trace_marker_mask[:known_n_real] = False

            viz = self.render_viz_frames(
                task=task,
                prompt_type=batch["trace_uid"][bid],
                T_wl=T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                marking_mask=trace_marker_mask,
                H=720,
                W=1280,
                pcl_mode=not self.maniskill_viz_cfg.get(
                    "mesh_mode", True
                ),  # Try mesh first
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
                    task=task,
                    prompt_type=batch["trace_uid"][bid],
                    T_wl=ref_T_wl,
                    viz_T_cam=viz_T_cam,
                    local_pcl=batch["local_pcl"][bid],
                    local_pcl_mask=batch["local_pcl_mask"][bid],
                    marking_mask=ctx_marker_mask,
                    H=720,
                    W=1280,
                    pcl_mode=not self.maniskill_viz_cfg.get(
                        "mesh_mode", True
                    ),  # Try mesh first
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

        # * GT Visualization
        _subroutine(
            batch["trace"][bid, :n_steps],
            batch["context"][bid],
            "GT",
            n_steps,
            100000,
            seed="gt",
        )

        # * Generated Visualization (X0 Pred)
        if "x0_pred_trace" in step_output:
            _subroutine(
                step_output["x0_pred_trace"][bid].float(),
                step_output["x0_pred_context"][bid].float(),
                "X0_PRED",
                -1,
                -1,
                -1,
            )

        # * Autoregressive Generation (Future Prediction)

        ar_iter1 = [False]
        ar_iter2 = [""]
        seeds = [42]
        for ar_flag, prefix in zip(ar_iter1, ar_iter2):
            for seed in seeds:
                for cfg in self.add_cfg:

                    dataset_trace_mask = batch["trace_mask"][bid]
                    dataset_ctx_mask = batch["context_mask"][bid]
                    pad_mask = batch["trace_mask"][bid].any(0) | batch["context_mask"][
                        bid
                    ].any(0)
                    pad_mask_trace = (
                        pad_mask[None]
                        .bool()
                        .clone()
                        .repeat(batch["trace_mask"].shape[1], 1)
                    )[None]
                    pad_mask_ctx = pad_mask[None, None].bool().clone()

                    trace_known_mask = dataset_trace_mask[None].bool().clone()
                    context_known_mask = dataset_ctx_mask[None].bool().clone()
                    n_steps_int = int((batch["time"][bid] <= 0).sum())
                    trace_known_mask[0, n_steps_int:] = False

                    # Simple future prediction
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
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                    )
                    _subroutine(
                        gen_trace[0],
                        None,
                        f"{prefix}Future(w. goal)_{cfg}",
                        n_steps_int,
                        -1,
                        seed,
                    )

                    # future prediction no goal
                    gen_context, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=torch.zeros_like(context_known_mask).bool(),
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
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                    )
                    _subroutine(
                        gen_trace[0],
                        gen_context[0],
                        f"{prefix}Future(no goal)_{cfg}",
                        n_steps_int,
                        -1,
                        seed,
                    )

                    cur_i = int((batch["time"][bid] <= 0).sum()) - 1
                    trace_known_mask[...] = False
                    trace_known_mask[0, cur_i] = True
                    gen_context, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask,
                        context_known_mask=torch.zeros_like(context_known_mask).bool(),
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
                        left2right_AR_sweep_flag=ar_flag,
                        add_cfg=cfg,
                    )
                    _subroutine(
                        gen_trace[0],
                        gen_context[0],
                        f"{prefix}OnlyCur(no goal)_{cfg}",
                        [cur_i],
                        0,
                        seed,
                    )

        return
