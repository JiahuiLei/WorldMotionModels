"""Visualization helpers for ManiSkill tabletop manipulation domain (V6_All).

Renders robot end-effector, finger, and object meshes/point-clouds for GT,
x0-pred, and inference visualization during training. Based on the V1
maniskill_viz but adapted for the V6_All architecture with optional DINO
image conditioning.

NOTE: Environment rollout visualization is disabled in this version to avoid
the lib_mani/ManiSkill environment dependency. Only GT + x0-pred + inference
modes are active.

Example::

    viz = ManiskillVisualizer(pl_module, maniskill_cfg)
    viz.maniskill_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from trimesh.creation import icosphere as TriSphere
import matplotlib.colors as mcolors

from umm.datasets.umm_e10.dataset_config import (
    UID_MANISKILL_TABLETOP_EE,
    UID_MANISKILL_TABLETOP_ACT,
    UID_MANISKILL_TABLETOP_LEFT_FINGER,
    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
    UID_MANISKILL_TABLETOP_OBJECT_ALL,
)
from umm.models.umm_e10.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_e10.viz_helpers.mesh_visualizer import visualize_a_mesh

# Constants
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)

# All ManiSkill UIDs for batch item detection
ALL_MANISKILL_UIDS = (
    [UID_MANISKILL_TABLETOP_EE, UID_MANISKILL_TABLETOP_ACT,
     UID_MANISKILL_TABLETOP_LEFT_FINGER, UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
     UID_MANISKILL_TABLETOP_RIGHT_FINGER, UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT]
    + UID_MANISKILL_TABLETOP_OBJECT_ALL
)


class ManiskillVisualizer:
    """Visualizer for ManiSkill environment trajectories (V6_All).

    Handles rendering of robot end-effector, fingers, and object meshes
    as trimesh objects. Supports GT, x0-pred, and inference visualization
    modes during training.

    Args:
        pl_module: Parent PyTorch Lightning module (PLModel).
        maniskill_cfg: Config dict from ``model.maniskill`` with per-task
            ``model_path``, ``object_path``, and ``num_pts`` keys.
    """

    def __init__(self, pl_module, maniskill_cfg):
        self.pl = pl_module
        if not maniskill_cfg or len(maniskill_cfg) == 0:
            self.flag = False
            return
        self.flag = True

        # Build dict of assets for different tasks
        self.assets = {}

        def load_mesh(data, local_id):
            """Load a trimesh from an npz archive by local actor ID."""
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
                raise ValueError(f"Mesh not found for uid: {local_id}")

        for task in maniskill_cfg.keys():
            logging.info(f"Loading ManiSkill viz assets for task: {task}")
            model_path = maniskill_cfg[task].get("model_path", "")
            object_path = maniskill_cfg[task].get("object_path", "")
            num_pts = maniskill_cfg[task].get("num_pts", 256)

            # Load state data from first sequence to get sem_type_uid
            states_dir = osp.join(osp.dirname(model_path), "states")
            states_files = sorted(
                [f for f in os.listdir(states_dir) if f.endswith(".npz")]
            )
            state_data = np.load(
                osp.join(states_dir, states_files[0]), allow_pickle=True
            )
            sem_type_uid = state_data["sem_type_uid"].item()
            ee_uid_local = sem_type_uid["endeffector"][0]
            left_finger_uid_local = sem_type_uid["gripper_left_finger"][0]
            right_finger_uid_local = sem_type_uid["gripper_right_finger"][0]

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

        # Rollout disabled by default (requires ManiSkill env dependency)
        self.rollout_flag = self.maniskill_viz_cfg.get("rollout_flag", False)
        logging.info(
            f"ManiskillVisualizer (v6_all): rollout_flag={self.rollout_flag}"
        )

    def __make_dummy_frames__(self, T, H, W):
        """Create blank placeholder frames."""
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(T)]

    def _detect_task(self, batch, bid):
        """Detect which ManiSkill task this batch item belongs to.

        Checks the extra dict for task name, otherwise tries to match UIDs
        to known task allocations.

        Args:
            batch: Training batch dict.
            bid: Batch item index.

        Returns:
            Task name string, or None if not detected.
        """
        # Check extra dict for task annotation
        if "extra" in batch and "task" in batch["extra"]:
            task_list = batch["extra"]["task"]
            if task_list[bid] is not None:
                return str(task_list[bid])

        # Fallback: match UIDs to task allocation pools
        uids = batch["trace_uid"][bid].cpu().tolist()
        for task, task_uids in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP.items():
            if task in self.assets:
                for uid in uids:
                    if uid in task_uids:
                        return task
        return None

    def render_viz_frames(
        self,
        task,
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
        """Render ManiSkill scene frames.

        Args:
            task: ManiSkill task name for mesh lookup.
            prompt_type: UID for each part (N,).
            T_wl: World-to-link transforms (T, N, 4, 4).
            viz_T_cam: Camera-to-world transform (4, 4).
            local_pcl: Local point clouds (N, PCL, 6).
            local_pcl_mask: Mask for valid point clouds (N,).
            marking_mask: Per-frame per-part mask for marking predictions (T, N).

        Returns:
            list[np.ndarray]: Rendered RGBA frames.
        """
        if not self.flag or task not in self.assets:
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

                _pcl = local_pcl[i]  # (PCL, 6)
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]

                _rgb = _pcl[:, 3:].clone().cpu().numpy()
                uid = int(prompt_type[i])

                # Color coding by UID
                if uid in [
                    UID_MANISKILL_TABLETOP_EE,
                    UID_MANISKILL_TABLETOP_LEFT_FINGER,
                    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
                ]:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 2] = 1.0  # Blue for robot state
                elif uid in [
                    UID_MANISKILL_TABLETOP_ACT,
                    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
                    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
                ]:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0  # Yellow for action target
                    _rgb[:, 1] = 1.0

                # Marking for predicted vs known
                if marking_mask[t, i]:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = 1.0
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)
                else:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _rgb = mcolors.hsv_to_rgb(_hsv)

                _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha

                use_mesh = (not pcl_mode) and (uid in ALL_MESHES)
                if not pcl_mode and uid not in ALL_MESHES:
                    logging.debug(f"Mesh not found for uid: {uid}")

                if not use_mesh:
                    # PCL mode
                    _pcl_t = (
                        T_wl[t, i][:3, :3] @ _pcl[:, :3].T
                    ).T + T_wl[t, i][None, :3, 3]
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
                    mean_color = np.clip(_rgb.mean(0) * 255.0, 0, 255).astype(
                        np.uint8
                    )
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
                frames.append(np.ones((H, W, 4), dtype=np.uint8) * 255)

        return frames

    def make_viz_T_cam(self, batch=None):
        """Construct the visualization camera pose for ManiSkill.

        Args:
            batch: Training batch (currently unused beyond forwarding).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.
        """
        return make_viz_T_cam(self.maniskill_viz_cfg, batch)

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

    @torch.inference_mode()
    def maniskill_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Top-level visualization routine for ManiSkill.

        Finds the first batch item with ManiSkill UIDs, logs GT, x0-pred, and
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

        # Find a batch item corresponding to ManiSkill
        bid = None
        for i in range(len(batch["trace"])):
            mask = self.get_type_mask(
                batch["trace_uid"][i],
                ALL_MANISKILL_UIDS,
            )
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no ManiSkill data, skip viz"
            )
            return

        # Detect the task for this batch item
        task = self._detect_task(batch, bid)
        if task is None or task not in self.assets:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, "
                f"could not detect ManiSkill task (task={task}), skip viz"
            )
            return

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(
            f"MS VIZ phase: {phase}, batch_idx: {batch_idx}, "
            f"task: {task}, n_steps: {n_steps}"
        )

        mesh_mode = self.maniskill_viz_cfg.get("mesh_mode", True)

        # --- Subroutine: render and log a single trace visualization ---
        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
            trace_marker_mask = torch.ones(*T_wl.shape[:2]).bool()
            if isinstance(known_n_real, list):
                for idx in known_n_real:
                    trace_marker_mask[idx] = False
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
                H=360,
                W=640,
                pcl_mode=not mesh_mode,
            )
            viz = put_caption_at_bottom(viz, batch["text"][bid])
            self.pl.logger.log_video(
                phase,
                f"ms_{name}_seed{seed}",
                torch.from_numpy(np.stack(viz, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
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
                    task=task,
                    prompt_type=batch["trace_uid"][bid],
                    T_wl=ref_T_wl,
                    viz_T_cam=viz_T_cam,
                    local_pcl=batch["local_pcl"][bid],
                    local_pcl_mask=batch["local_pcl_mask"][bid],
                    marking_mask=ctx_marker_mask,
                    H=360,
                    W=640,
                    pcl_mode=not mesh_mode,
                )
                viz_img = make_square_cat_image(viz_ctx)
                viz_img = viz_img[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"ms_ctx_{name}_seed{seed}",
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
        images_for_inf = None
        view_ids_for_inf = None
        if eval_model.use_dino and "extra" in batch:
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
                left2right_AR_sweep_flag=False,
                images=images_for_inf,
                view_ids=view_ids_for_inf,
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
                    add_cfg=cfg,
                    left2right_AR_sweep_flag=False,
                    images=images_for_inf,
                    view_ids=view_ids_for_inf,
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    f"Future(no goal)_{cfg}",
                    n_steps_int,
                    0,
                    seed,
                )
