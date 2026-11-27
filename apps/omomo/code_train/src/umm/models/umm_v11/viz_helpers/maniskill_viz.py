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

import imageio
import numpy as np
import torch
import trimesh
from trimesh.creation import icosphere as TriSphere
import matplotlib.colors as mcolors

from umm.datasets.umm_v11.dataset_config import (
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
from umm.datasets.umm_v11.dataset_template import CanonicalDatasetOutput
from umm.datasets.umm_v11.maniskill_oracle_dataset import ManiskillOracleDataset
from umm.models.umm_v11.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_v11.viz_helpers.mesh_visualizer import visualize_a_mesh

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
            # Some tasks use a non-gripper end-effector (e.g., PushT-v1 uses a
            # T-stick) and have empty `gripper_*_finger` entries in sem_type_uid.
            # Fall back to the endeffector mesh so viz still renders an EE.
            lf_list = sem_type_uid.get("gripper_left_finger", [])
            rf_list = sem_type_uid.get("gripper_right_finger", [])
            left_finger_uid_local = lf_list[0] if lf_list else ee_uid_local
            right_finger_uid_local = rf_list[0] if rf_list else ee_uid_local

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

        # --- V12.1 deployment-test knobs ---
        # deploy_chunk_size: length of the predicted chunk inside one inference.
        #   Matches lt5.6 act_horizon=10. Setting 0 disables the deployment test.
        # deploy_hist_len: number of leading frames kept as history / known
        #   conditioning (matches lt5.6 hist=5).
        # deploy_max_tasks: hard cap on tasks rendered per viz cycle (None = all).
        # deploy_seeds: inference seeds, one rendered clip per seed per task.
        self.deploy_chunk_size = int(
            self.maniskill_viz_cfg.get("deploy_chunk_size", 10)
        )
        self.deploy_hist_len = int(self.maniskill_viz_cfg.get("deploy_hist_len", 5))
        self.deploy_max_tasks = self.maniskill_viz_cfg.get("deploy_max_tasks", None)
        self.deploy_seeds = list(
            self.maniskill_viz_cfg.get("deploy_seeds", [42, 1234])
        )
        logging.info(
            f"ManiskillVisualizer (v6_all): rollout_flag={self.rollout_flag}, "
            f"deploy_chunk_size={self.deploy_chunk_size}, "
            f"deploy_hist_len={self.deploy_hist_len}, "
            f"deploy_max_tasks={self.deploy_max_tasks}, "
            f"deploy_seeds={self.deploy_seeds}"
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
        # V11 AgentV11 has no DINO support yet — guard with getattr.
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
            # Note: V11 inference does NOT support add_cfg/left2right_AR_sweep_flag/images/view_ids.
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
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    f"Future(no goal)_{cfg}",
                    n_steps_int,
                    0,
                    seed,
                )

        # --- V12.1 per-task deployment test ---
        # Iterate every task in self.assets, build a single-sample batch from
        # that task's val_dataset, call inference with kwargs mirroring forward,
        # and render GT + full inference + deploy-chunk slice.
        # Outer VizCallback already honours the viz_try flag, so failures here
        # surface loudly in dev and are swallowed only when viz_try=True.
        if self.deploy_chunk_size > 0 and phase in ("train", "val", "test"):
            self.deployment_test_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

    # ------------------------------------------------------------------
    # V12.1 per-task deployment test
    # ------------------------------------------------------------------
    def _find_task_val_dataset(self, task_name):
        """Locate the ManiskillOracleDataset instance for ``task_name``.

        Walks ``self.pl.trainer.datamodule.val_dataset`` (a ConcatDataset of
        per-domain datasets) and returns the first child dataset whose class
        is :class:`ManiskillOracleDataset` and whose ``task_name`` matches.

        Args:
            task_name: ManiSkill task name (e.g. ``"PickCube-v1"``).

        Returns:
            ManiskillOracleDataset instance, or None if not found.
        """
        trainer = getattr(self.pl, "trainer", None)
        if trainer is None:
            return None
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return None
        val_dataset = getattr(dm, "val_dataset", None)
        if val_dataset is None:
            return None
        children = getattr(val_dataset, "datasets", None)
        if children is None:
            # Not a ConcatDataset — treat as single dataset
            children = [val_dataset]
        for ds in children:
            if isinstance(ds, ManiskillOracleDataset) and ds.task_name == task_name:
                return ds
        return None

    def _print_sample_batch_diff(self, task, raw_sample, raw_extra, batch_dict):
        """Print a sanity-check diff of a raw dataset sample vs the collated batch.

        Logs per-field shapes, dtypes and a handful of scalar invariants
        (number of unique UIDs, number of valid trace frames, text string).
        This is MANDATORY — not decorative — to catch silent pipeline drift
        between training forward and deployment inference.

        Args:
            task: Task name string.
            raw_sample: CanonicalDatasetOutput returned by the dataset.
            raw_extra: Extra dict returned alongside the sample.
            batch_dict: Dict produced by ``CanonicalDatasetOutput.collate``.
        """
        lines = [f"MS DEPLOY DIFF [{task}]"]

        def _fmt(x):
            if isinstance(x, torch.Tensor):
                return f"Tensor shape={tuple(x.shape)} dtype={x.dtype}"
            if isinstance(x, np.ndarray):
                return f"ndarray shape={x.shape} dtype={x.dtype}"
            if isinstance(x, str):
                return f"str={x!r}"
            if isinstance(x, (list, tuple)):
                return f"{type(x).__name__} len={len(x)}"
            return f"{type(x).__name__}={x!r}"

        raw_trace_valid = int(raw_sample.trace_mask.any(dim=0).sum())
        raw_uniq_uids = int(torch.unique(raw_sample.trace_uid).numel())
        lines.append(
            "  raw sample: "
            f"trace={_fmt(raw_sample.trace)} "
            f"trace_mask={_fmt(raw_sample.trace_mask)} "
            f"context={_fmt(raw_sample.context)} "
            f"local_pcl={_fmt(raw_sample.local_pcl)} "
            f"uid={_fmt(raw_sample.trace_uid)} "
            f"text={_fmt(raw_sample.text)} "
            f"time={_fmt(raw_sample.time)} "
            f"valid_T={raw_trace_valid} uniq_uids={raw_uniq_uids}"
        )
        if isinstance(raw_extra, dict):
            lines.append(
                f"  raw extra keys: {sorted(raw_extra.keys())}"
            )

        batch_trace_valid = int(batch_dict["trace_mask"][0].any(dim=1).sum())
        batch_uniq_uids = int(torch.unique(batch_dict["trace_uid"][0]).numel())
        lines.append(
            "  batch (B=1): "
            f"trace={_fmt(batch_dict['trace'])} "
            f"trace_mask={_fmt(batch_dict['trace_mask'])} "
            f"context={_fmt(batch_dict['context'])} "
            f"local_pcl={_fmt(batch_dict['local_pcl'])} "
            f"uid={_fmt(batch_dict['trace_uid'])} "
            f"text={_fmt(batch_dict['text'])} "
            f"time={_fmt(batch_dict['time'])} "
            f"valid_T={batch_trace_valid} uniq_uids={batch_uniq_uids}"
        )
        # Raw permutation is (P, T, 4, 4); batch is (B, T, P, 4, 4).
        # Verify the collate permute actually reordered the axes.
        lines.append(
            f"  axis check: raw trace (P,T,...) -> batch (B,T,P,...): "
            f"raw shape (P,T)=({raw_sample.trace.shape[0]},"
            f"{raw_sample.trace.shape[1]}) vs "
            f"batch (T,P)=({batch_dict['trace'].shape[1]},"
            f"{batch_dict['trace'].shape[2]})"
        )
        logging.info("\n".join(lines))

    def _build_single_sample_batch(self, task_dataset, device, sample_idx=0):
        """Fetch one sample from a task dataset and run it through the canonical
        collate to produce a single-sample batch ready for inference.

        Args:
            task_dataset: A ManiskillOracleDataset instance.
            device: Target device to move the batch tensors onto.
            sample_idx: Raw index into the dataset. Defaults to 0; callers may
                pass a batch-index-derived value for variety across viz cycles.

        Returns:
            Tuple ``(raw_sample, raw_extra, batch_dict)`` where ``batch_dict``
            has every tensor field moved to ``device``.
        """
        if len(task_dataset) == 0:
            raise RuntimeError(
                f"ManiSkill task {task_dataset.task_name} val dataset is empty"
            )
        idx = sample_idx % len(task_dataset)
        raw_sample, raw_extra = task_dataset[idx]
        batch_dict = CanonicalDatasetOutput.collate([(raw_sample, raw_extra)])
        # Move tensor fields to device. Skip list/dict fields.
        tensor_keys = [
            "trace",
            "trace_mask",
            "context",
            "context_mask",
            "known_mask",
            "context_known_mask",
            "trace_group_id",
            "trace_rot_mask",
            "trace_sem_type",
            "trace_uid",
            "time",
            "local_pcl",
            "local_pcl_mask",
        ]
        for k in tensor_keys:
            if k in batch_dict and isinstance(batch_dict[k], torch.Tensor):
                batch_dict[k] = batch_dict[k].to(device)
        return raw_sample, raw_extra, batch_dict

    def _ms_deploy_save_dir(self):
        """Return the directory where deployment MP4 clips are dumped.

        Priority: ``UMM_MS_DEPLOY_VIZ_DIR`` env var (useful for smoke tests /
        fast_dev_run evidence collection, which suppresses the Lightning
        logger) → ``<logger.video_save_dir>/ms_deploy`` → ``./ms_deploy`` as
        a last-resort fallback.
        """
        env_dir = os.environ.get("UMM_MS_DEPLOY_VIZ_DIR")
        if env_dir:
            target = env_dir
        else:
            logger = getattr(self.pl, "logger", None)
            video_save_dir = getattr(logger, "video_save_dir", None)
            target = (
                osp.join(video_save_dir, "ms_deploy")
                if video_save_dir
                else "./ms_deploy"
            )
        os.makedirs(target, exist_ok=True)
        return target

    def _save_ms_deploy_mp4(
        self, task, kind, frames, seed, global_step, batch_idx
    ):
        """Dump a deployment clip (list of RGB frames) as an MP4 on disk.

        This is independent of the Lightning logger so that viz evidence is
        still produced under ``fast_dev_run`` (which suppresses logging) and
        for offline smoke tests.

        Args:
            task: ManiSkill task name (e.g. ``"PickCube-v1"``).
            kind: Clip kind: ``"GT"``, ``"INF"``, or ``"CHUNK"``.
            frames: List of ``uint8`` RGB frames (H, W, 3) or
                ``(H, W, 4)`` — alpha channel is stripped.
            seed: Inference seed (unused for GT; pass 0).
            global_step: Global step at time of render (0 for fast_dev_run).
            batch_idx: Current val batch index.
        """
        out_dir = self._ms_deploy_save_dir()
        seed_tag = f"_seed{seed}" if kind != "GT" else ""
        fn = osp.join(
            out_dir,
            f"g{int(global_step)}_b{int(batch_idx)}_{task}_{kind}{seed_tag}.mp4",
        )
        # imageio expects a list of uint8 HxWx3 frames.
        frame_list = [f[..., :3].astype(np.uint8) for f in frames]
        imageio.mimwrite(fn, frame_list, fps=10)
        logging.info(f"MS DEPLOY SAVED: {fn}")

    @torch.inference_mode()
    def deployment_test_routine(
        self, batch, phase, batch_idx, eval_model, step_output
    ):
        """Per-task deployment visualization test.

        Iterates every task in ``self.assets`` (capped by ``deploy_max_tasks``),
        pulls a single sample from that task's val ``ManiskillOracleDataset``,
        builds a single-sample batch via :meth:`CanonicalDatasetOutput.collate`,
        prints a shape/value diff sanity check, then calls
        ``eval_model.inference`` with kwargs that EXACTLY mirror the training
        forward call (no silent modality dropping).

        For each task + seed, renders three clips: GT, full inference
        (max_T frames), and the deployment chunk slice
        ``[deploy_hist_len : deploy_hist_len + deploy_chunk_size]``.
        Text prompts are always taken verbatim from the dataset —
        never templated.

        Args:
            batch: Current training batch (unused except for device lookup).
            phase: "train", "val", or "test".
            batch_idx: Batch index for logger call.
            eval_model: Agent model in eval mode (EMA-unwrapped by caller).
            step_output: Training step output dict (unused here).
        """
        if not self.flag or self.deploy_chunk_size <= 0:
            return

        device = batch["trace"].device
        viz_T_cam = self.make_viz_T_cam(batch)
        mesh_mode = self.maniskill_viz_cfg.get("mesh_mode", True)

        # Sort tasks for deterministic ordering across runs.
        all_tasks = sorted(self.assets.keys())
        if self.deploy_max_tasks is not None:
            all_tasks = all_tasks[: int(self.deploy_max_tasks)]
        logging.info(
            f"MS DEPLOY phase={phase} batch_idx={batch_idx}: "
            f"iterating {len(all_tasks)}/{len(self.assets)} tasks, "
            f"hist_len={self.deploy_hist_len} chunk={self.deploy_chunk_size}"
        )

        for task in all_tasks:
            task_ds = self._find_task_val_dataset(task)
            if task_ds is None:
                logging.warning(
                    f"MS DEPLOY [{task}]: no matching val dataset, skipping"
                )
                continue

            # Vary sample_idx per viz cycle so consecutive runs exercise
            # different episodes, not just index 0 forever.
            raw_sample, raw_extra, single_batch = self._build_single_sample_batch(
                task_ds, device, sample_idx=int(self.pl.global_step) + batch_idx
            )

            # Mandatory sanity check: print raw vs batch diff.
            self._print_sample_batch_diff(task, raw_sample, raw_extra, single_batch)

            bid = 0  # single sample
            # ---- Build inference kwargs that MIRROR training forward ----
            # Start from the fully-unmasked batch trace_mask and wipe the
            # trace_known_mask past the history length so the model must
            # predict the deployment chunk.
            trace_mask_all = single_batch["trace_mask"]  # (1, T, P) bool
            context_mask_all = single_batch["context_mask"]  # (1, C, P) bool

            # Prompt-visibility pad mask (P entities valid anywhere in window)
            pad_p = trace_mask_all[bid].any(0) | context_mask_all[bid].any(0)
            pad_mask_trace = (
                pad_p[None, None, :].expand(1, trace_mask_all.shape[1], -1).clone()
            )
            pad_mask_ctx = (
                pad_p[None, None, :].expand(1, context_mask_all.shape[1], -1).clone()
            )

            # known_mask: history (< hist_len) known, rest (>= hist_len) unknown.
            # Use the dataset trace_mask as the baseline so padded entities stay False.
            hist_len = min(self.deploy_hist_len, trace_mask_all.shape[1])
            trace_known_mask = trace_mask_all.clone()
            trace_known_mask[:, hist_len:, :] = False

            # Context known: keep goal state available (deployment wants to
            # condition on the target context), mirrors Future(w.goal) path.
            context_known_mask = context_mask_all.clone()

            # Text verbatim from dataset — single-element list.
            text_list = [single_batch["text"][bid]]

            # Task UID for the GT scene render.
            prompt_uid = single_batch["trace_uid"][bid]
            local_pcl_single = single_batch["local_pcl"][bid]
            local_pcl_mask_single = single_batch["local_pcl_mask"][bid]

            # --- Render GT clip (full window) ---
            gt_T_wl = single_batch["trace"][bid].float().cpu()  # (T, P, 4, 4)
            gt_frames = self.render_viz_frames(
                task=task,
                prompt_type=prompt_uid,
                T_wl=gt_T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=local_pcl_single,
                local_pcl_mask=local_pcl_mask_single,
                H=360,
                W=640,
                pcl_mode=not mesh_mode,
            )
            gt_frames_cap = put_caption_at_bottom(
                gt_frames, f"{task} | GT | {text_list[0]}"
            )
            self.pl.logger.log_video(
                phase,
                f"ms_deploy_{task}_GT",
                torch.from_numpy(np.stack(gt_frames_cap, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=f"{task} | GT | {text_list[0]}",
            )
            # Disk save (independent of Lightning logger — survives fast_dev_run).
            self._save_ms_deploy_mp4(
                task,
                "GT",
                gt_frames_cap,
                seed=0,
                global_step=self.pl.global_step,
                batch_idx=batch_idx,
            )

            # --- Inference for each deployment seed ---
            for seed in self.deploy_seeds:
                gen_context, gen_trace = eval_model.inference(
                    trace=single_batch["trace"],
                    context=single_batch["context"],
                    trace_known_mask=trace_known_mask,
                    context_known_mask=context_known_mask,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=single_batch["time"],
                    group_id=single_batch["trace_group_id"],
                    sem_type=single_batch["trace_sem_type"],
                    rot_mask=single_batch["trace_rot_mask"],
                    local_pcl=single_batch["local_pcl"],
                    local_pcl_mask=single_batch["local_pcl_mask"],
                    text=text_list,
                    seed=seed,
                )

                full_trace = gen_trace[0].float().cpu()  # (T, P, 4, 4)
                T_total = full_trace.shape[0]
                chunk_end = min(hist_len + self.deploy_chunk_size, T_total)
                deploy_slice = full_trace[hist_len:chunk_end]
                logging.info(
                    f"MS DEPLOY [{task}] seed={seed}: "
                    f"inference trace={tuple(full_trace.shape)} "
                    f"hist_len={hist_len} chunk_size={self.deploy_chunk_size} "
                    f"deploy_slice={tuple(deploy_slice.shape)}"
                )

                # Marker: known frames (< hist_len) get False, predicted get True.
                full_marker = torch.ones(*full_trace.shape[:2], dtype=torch.bool)
                full_marker[:hist_len] = False

                inf_frames = self.render_viz_frames(
                    task=task,
                    prompt_type=prompt_uid,
                    T_wl=full_trace,
                    viz_T_cam=viz_T_cam,
                    local_pcl=local_pcl_single,
                    local_pcl_mask=local_pcl_mask_single,
                    marking_mask=full_marker,
                    H=360,
                    W=640,
                    pcl_mode=not mesh_mode,
                )
                inf_frames_cap = put_caption_at_bottom(
                    inf_frames,
                    f"{task} | INF_full | seed={seed} | {text_list[0]}",
                )
                self.pl.logger.log_video(
                    phase,
                    f"ms_deploy_{task}_INF_seed{seed}",
                    torch.from_numpy(np.stack(inf_frames_cap, 0))[..., :3]
                    .float()
                    .permute(0, 3, 1, 2)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                    caption=f"{task} | INF_full | seed={seed} | {text_list[0]}",
                )
                self._save_ms_deploy_mp4(
                    task,
                    "INF",
                    inf_frames_cap,
                    seed=seed,
                    global_step=self.pl.global_step,
                    batch_idx=batch_idx,
                )

                # Deploy chunk slice — all predicted.
                chunk_marker = torch.ones(
                    *deploy_slice.shape[:2], dtype=torch.bool
                )
                chunk_frames = self.render_viz_frames(
                    task=task,
                    prompt_type=prompt_uid,
                    T_wl=deploy_slice,
                    viz_T_cam=viz_T_cam,
                    local_pcl=local_pcl_single,
                    local_pcl_mask=local_pcl_mask_single,
                    marking_mask=chunk_marker,
                    H=360,
                    W=640,
                    pcl_mode=not mesh_mode,
                )
                chunk_frames_cap = put_caption_at_bottom(
                    chunk_frames,
                    f"{task} | DEPLOY_chunk[{hist_len}:{chunk_end}] "
                    f"| seed={seed} | {text_list[0]}",
                )
                self.pl.logger.log_video(
                    phase,
                    f"ms_deploy_{task}_CHUNK_seed{seed}",
                    torch.from_numpy(np.stack(chunk_frames_cap, 0))[..., :3]
                    .float()
                    .permute(0, 3, 1, 2)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                    caption=(
                        f"{task} | DEPLOY_chunk[{hist_len}:{chunk_end}] "
                        f"| seed={seed} | {text_list[0]}"
                    ),
                )
                self._save_ms_deploy_mp4(
                    task,
                    "CHUNK",
                    chunk_frames_cap,
                    seed=seed,
                    global_step=self.pl.global_step,
                    batch_idx=batch_idx,
                )
