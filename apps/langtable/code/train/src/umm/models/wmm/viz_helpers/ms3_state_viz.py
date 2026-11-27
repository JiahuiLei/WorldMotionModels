"""Visualization helpers for ManiSkill3 schema-v3 state dataset (WMM).

Renders robot end-effector, finger, and object meshes/point-clouds for GT,
x0-pred, and inference visualization during training. Single-task per-config
design matching the ms3_state dataset's one-task-per-config architecture.

Loads robot + object meshes from ``robot_embodiment.npz`` / ``object_embodiment.npz``
keyed by SAPIEN per_scene_id (psid). Renders GT + x0-pred + 2-seed inference
clips per viz call.

NOTE: No deployment_test_routine — this module is the simple training-time
visualizer only (~400 lines). Deployment tests are out of scope.

Example::

    viz = Ms3StateVisualizer(pl_module, ms3_state_cfg, task="PushCube-v1")
    viz.ms3_state_viz_routine(batch, "val", 0, eval_model, step_output)
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

from umm.models.wmm.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    put_caption_at_bottom,
)
from umm.models.wmm.viz_helpers.mesh_visualizer import visualize_a_mesh

# Constants — same as maniskill_viz.py
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class Ms3StateVisualizer:
    """ManiSkill3 schema-v3 training-time visualizer (WMM).

    Gated on model.ms3_state config block. Single-task per-config design
    (MS3 training configs are one-task-per-config). Loads robot + object
    meshes from robot_embodiment.npz / object_embodiment.npz keyed by
    SAPIEN per_scene_id (psid). Renders GT + x0-pred + 2-seed inference
    clips per viz call.

    Args:
        pl_module: parent PLModel.
        ms3_state_cfg: dict with ``{model_path, object_path, num_pts}``.
        task: task name string (e.g. "PushCube-v1"). If None, raises.
    """

    def __init__(self, pl_module, ms3_state_cfg, task):
        self.pl = pl_module
        self.task = task

        # Import maps (UID constants only — SEM constants not needed at viz
        # time, meshes are keyed by UID and colors come from per-part RGB in
        # `local_pcl`).
        from umm.datasets.wmm.dataset_config import (
            MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
            UID_MANISKILL_TABLETOP_EE,
            UID_MANISKILL_TABLETOP_ACT,
            UID_MANISKILL_TABLETOP_LEFT_FINGER,
            UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_OBJECT_ALL,
        )
        from umm.datasets.wmm.ms3_state_dataset import (
            _TASK_FINGER_PSIDS,
            _TASK_OBJ_PSIDS,
            _PALM_PSID,
        )

        # Gate: empty config or task not in map → disable
        if not ms3_state_cfg or task not in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP:
            self.flag = False
            logging.warning(
                f"Ms3StateVisualizer: disabled (ms3_state_cfg={bool(ms3_state_cfg)}, "
                f"task={task!r} in map={task in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP})"
            )
            return

        model_path = ms3_state_cfg.get("model_path", "")
        object_path = ms3_state_cfg.get("object_path", "")
        num_pts = ms3_state_cfg.get("num_pts", 256)

        def load_mesh(data, psid):
            """Load a trimesh from an npz archive by SAPIEN per_scene_id."""
            key_v = f"{psid}_v"
            key_f = f"{psid}_f"
            key_c = f"{psid}_c"
            if key_v in data:
                _v = data[key_v]
                _f = data[key_f]
                _c = data[key_c]
                tm = trimesh.Trimesh(vertices=_v, faces=_f, vertex_colors=_c)
                return tm
            else:
                raise ValueError(f"Mesh not found for psid: {psid}")

        robot_data = np.load(model_path)
        object_data = np.load(object_path)

        # Build meshes dict
        meshes = {}
        palm_mesh = load_mesh(robot_data, _PALM_PSID)  # psid 10 = panda_hand

        # --- Frame correction: palm mesh is stored in `panda_hand` local
        # frame (per `lib_mani.mesh_utils._get_meshes_internal`, which bakes
        # `shape.local_pose` into vertices) BUT the dataset's `palm_T_wl`
        # equals `tcp_pose` (= `panda_hand_tcp.pose`, the MS3 controller
        # ee_link). Per `panda_v3.urdf::panda_hand_tcp_joint`, the TCP frame
        # is offset `xyz="0 0 0.1034"` (identity rot) from `panda_hand`.
        # Without correction, the palm mesh renders at the TCP origin, which
        # sits ~10.3 cm along hand-z from the true hand origin. At grasp
        # poses where `hand.z` points DOWN in world, this mis-places the
        # palm mesh ~10.3 cm below the true palm, causing fingers (rendered
        # correctly via `panda_leftfinger`/`panda_rightfinger` poses) to
        # appear ABOVE the palm — the "upside-down gripper" visualization
        # bug.
        #
        # Fix: pre-translate mesh vertices by `-0.1034` along local z so
        # the mesh now lives in the TCP local frame. Applying `tcp_pose`
        # to this shifted mesh renders the palm at its true world location.
        #
        # This is a VIZ-ONLY correction; the underlying `robot_embodiment.npz`
        # and the training targets (which legitimately use TCP pose for the
        # pd_ee_delta_pos/pose controller) are unchanged.
        #
        # Known follow-up (not fixed here): the palm point-cloud in
        # ``batch["local_pcl"]`` is sampled from the same panda_hand-frame
        # mesh (see ``lib_mani.mesh_utils._save_meshes_to_npz``) and is NOT
        # shifted by this visualizer. Under ``mesh_mode=True`` (all 4
        # production MS3 configs today), ``render_viz_frames`` uses the
        # shifted mesh above and ignores ``local_pcl``, so the rendered
        # output is correct. But under ``pcl_mode=True`` (or
        # ``mesh_mode=False``), spheres are rendered at ``T_wl @ local_pcl``
        # which reintroduces the 10.3 cm offset for the palm. If anyone
        # flips the config to pcl_mode, the PCL path needs a parallel
        # fix in ``Ms3StateDataset._load_robot_pcl(psid=10)``.
        #
        # The constant is cross-verified at init against PandaFK (see
        # ``_verify_tcp_offset`` below) so a URDF change that moves the TCP
        # joint is detected at import time rather than silently degrading.
        PANDA_HAND_TO_TCP_Z_OFFSET_M = 0.1034  # URDF: panda_hand_tcp_joint
        self._verify_tcp_offset(PANDA_HAND_TO_TCP_Z_OFFSET_M)
        palm_mesh = palm_mesh.copy()
        # Defensive explicit vertex copy: trimesh.copy() is documented as a
        # deep copy in >=3.9, but an in-place `-=` on a shared buffer would
        # corrupt the original npz-backed array if that ever changed.
        palm_mesh.vertices = palm_mesh.vertices.copy()
        palm_mesh.vertices[:, 2] -= PANDA_HAND_TO_TCP_Z_OFFSET_M

        left_psid = _TASK_FINGER_PSIDS[task]["left"]
        right_psid = _TASK_FINGER_PSIDS[task]["right"]

        # Note: palm_mesh and palm_act both reference the SAME shifted mesh
        # object — intentional to save memory (render_viz_frames does
        # ALL_MESHES[uid].copy() per frame, so the aliasing is safe at
        # render time).  Regression test
        # `test_palm_mesh_shifted_to_tcp_frame` locks the aliasing.
        meshes[UID_MANISKILL_TABLETOP_EE] = palm_mesh
        meshes[UID_MANISKILL_TABLETOP_ACT] = palm_mesh
        meshes[UID_MANISKILL_TABLETOP_LEFT_FINGER] = load_mesh(robot_data, left_psid)
        meshes[UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT] = meshes[UID_MANISKILL_TABLETOP_LEFT_FINGER]
        meshes[UID_MANISKILL_TABLETOP_RIGHT_FINGER] = load_mesh(robot_data, right_psid)
        meshes[UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT] = meshes[UID_MANISKILL_TABLETOP_RIGHT_FINGER]

        task_obj_uids = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[task]
        for obj_uid, obj_psid in zip(task_obj_uids, _TASK_OBJ_PSIDS[task]):
            meshes[obj_uid] = load_mesh(object_data, obj_psid)

        self.flag = True
        self.assets = {task: {"meshes": meshes, "num_pts": num_pts}}

        self.ms3_state_viz_cfg = self.pl.cfg.get("visualize", {}).get("ms3_state", {})
        self.add_cfg = self.ms3_state_viz_cfg.get("add_cfg", [1.0])

        # Store UID constants for use in the routine
        self._ALL_MS3_STATE_UIDS = (
            [
                UID_MANISKILL_TABLETOP_EE,
                UID_MANISKILL_TABLETOP_ACT,
                UID_MANISKILL_TABLETOP_LEFT_FINGER,
                UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
                UID_MANISKILL_TABLETOP_RIGHT_FINGER,
                UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
            ]
            + list(UID_MANISKILL_TABLETOP_OBJECT_ALL)
        )
        self._UID_EE = UID_MANISKILL_TABLETOP_EE
        self._UID_ACT = UID_MANISKILL_TABLETOP_ACT
        self._UID_LF = UID_MANISKILL_TABLETOP_LEFT_FINGER
        self._UID_LF_ACT = UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT
        self._UID_RF = UID_MANISKILL_TABLETOP_RIGHT_FINGER
        self._UID_RF_ACT = UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT

        logging.info(f"Ms3StateVisualizer: task={task}, n_meshes={len(meshes)}")

    @staticmethod
    def _verify_tcp_offset(expected_z_offset: float, tol: float = 1e-3) -> None:
        """Runtime sanity check that hardcoded TCP z-offset matches PandaFK.

        PandaFK loads ``panda_v3.urdf`` via ManiSkill at import time, so
        evaluating ``fk.forward(zero_qpos)["panda_hand_tcp"]`` vs
        ``["panda_hand"]`` yields the ground-truth ``T_hand_to_tcp``. If a
        mani_skill upgrade ever ships a URDF revision that moves the TCP
        joint, this check raises at Visualizer construction time — loud and
        early — rather than silently mis-rendering the palm.

        Raises ``RuntimeError`` if the measured offset disagrees with the
        hardcoded constant by more than ``tol`` (default 1 mm) OR the
        hand→tcp rotation is non-identity.
        """
        import torch
        from lib_mani.utils.panda_fk import PandaFK

        fk = PandaFK()
        qpos = torch.zeros(1, 9, dtype=torch.float32)
        out = fk.forward(qpos)
        T_hand = out["panda_hand"][0].numpy()
        T_tcp = out["panda_hand_tcp"][0].numpy()
        T_h2tcp = np.linalg.inv(T_hand) @ T_tcp
        measured = float(T_h2tcp[2, 3])
        R_dev = float(np.abs(T_h2tcp[:3, :3] - np.eye(3, dtype=np.float32)).max())
        if abs(measured - expected_z_offset) > tol:
            raise RuntimeError(
                f"Ms3StateVisualizer: hardcoded TCP z-offset "
                f"{expected_z_offset:.4f} m disagrees with PandaFK runtime "
                f"{measured:.4f} m by {abs(measured - expected_z_offset):.4f} m. "
                f"URDF `panda_hand_tcp_joint` may have changed. Update the "
                f"constant and re-verify the viz."
            )
        if R_dev > tol:
            raise RuntimeError(
                f"Ms3StateVisualizer: measured T_hand_to_tcp rotation "
                f"deviates from identity by {R_dev:.4f}. URDF "
                f"`panda_hand_tcp_joint` rpy may have changed. The "
                f"viz fix assumes pure-translation offset."
            )

    def __make_dummy_frames__(self, T, H, W):
        """Create blank placeholder frames."""
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(T)]

    def render_viz_frames(
        self,
        task,
        prompt_type,
        T_wl,             # (T, N, 4, 4)
        viz_T_cam,
        local_pcl,        # (N, PCL, 3+)
        local_pcl_mask,   # (N,)
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
        """Render MS3-state scene frames.

        Args:
            task: MS3 task name for mesh lookup.
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
                    self._UID_EE,
                    self._UID_LF,
                    self._UID_RF,
                ]:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 2] = 1.0  # Blue for robot state
                elif uid in [
                    self._UID_ACT,
                    self._UID_LF_ACT,
                    self._UID_RF_ACT,
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
                        sph.visual.vertex_colors = _rgb[_idx].tolist() + [_pcl_alpha]  # type: ignore[attr-defined]
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
        """Construct the visualization camera pose for MS3-state.

        Args:
            batch: Training batch (currently unused beyond forwarding).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.
        """
        return make_viz_T_cam(self.ms3_state_viz_cfg, batch)

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

    def _viz_save_dir(self):
        """Return the directory where viz MP4 clips are dumped.

        Priority: ``UMM_MS_DEPLOY_VIZ_DIR`` env var → ``<logger.video_save_dir>/ms3_state``
        → ``./ms3_state`` as a last-resort fallback.
        """
        env_dir = os.environ.get("UMM_MS_DEPLOY_VIZ_DIR")
        if env_dir:
            target = env_dir
        else:
            logger = getattr(self.pl, "logger", None)
            video_save_dir = getattr(logger, "video_save_dir", None)
            target = (
                osp.join(video_save_dir, "ms3_state")
                if video_save_dir
                else "./ms3_state"
            )
        os.makedirs(target, exist_ok=True)
        return target

    def _save_viz_mp4(self, kind, frames, seed, global_step, batch_idx):
        """Dump a viz clip as MP4 on disk for evidence collection.

        Independent of Lightning logger — survives fast_dev_run. Tag is
        ``ms3_state`` (not ``ms_deploy``).

        Args:
            kind: Clip kind: ``"GT"``, ``"X0_PRED"``, ``"Future_w_goal"``, etc.
            frames: List of ``uint8`` RGBA frames (H, W, 4).
            seed: Inference seed; pass ``"gt"`` for GT clips.
            global_step: Global step at time of render.
            batch_idx: Current val batch index.
        """
        out_dir = self._viz_save_dir()
        seed_tag = f"_seed{seed}" if kind != "GT" else ""
        fn = osp.join(
            out_dir,
            f"g{int(global_step)}_b{int(batch_idx)}_ms3_state_{kind}{seed_tag}.mp4",
        )
        frame_list = [f[..., :3].astype(np.uint8) for f in frames]
        imageio.mimwrite(fn, frame_list, fps=10)
        logging.info(f"MS3_STATE SAVED: {fn}")

    @torch.inference_mode()
    def ms3_state_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Top-level visualization routine for MS3-state.

        Finds the first batch item with MS3 UIDs, logs GT, x0-pred, and
        inference visualizations (future with/without goal, 2 seeds each).

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

        # Find first batch item with any MS3 UID
        bid = None
        for i in range(len(batch["trace"])):
            mask = self.get_type_mask(batch["trace_uid"][i], self._ALL_MS3_STATE_UIDS)
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"MS_STATE VIZ phase={phase} batch_idx={batch_idx}: "
                f"no MS3 data in batch, skipping"
            )
            return

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(
            f"MS_STATE VIZ phase={phase} batch_idx={batch_idx} "
            f"task={self.task} n_steps={n_steps}"
        )

        mesh_mode = self.ms3_state_viz_cfg.get("mesh_mode", True)

        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
            trace_marker_mask = torch.ones(*T_wl.shape[:2]).bool()
            if isinstance(known_n_real, list):
                for idx in known_n_real:
                    trace_marker_mask[idx] = False
            else:
                trace_marker_mask[:known_n_real] = False

            viz = self.render_viz_frames(
                task=self.task,
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
                f"ms3_state_{name}_seed{seed}",
                torch.from_numpy(np.stack(viz, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=batch["text"][bid],
            )
            # Disk save for fast_dev_run evidence collection
            self._save_viz_mp4(
                name, viz, seed=seed,
                global_step=self.pl.global_step, batch_idx=batch_idx,
            )

            if ref_T_wl is not None:
                ctx_marker_mask = torch.ones(*ref_T_wl.shape[:2]).bool()
                if isinstance(known_n_ctx, list):
                    for idx in known_n_ctx:
                        ctx_marker_mask[idx] = False
                else:
                    ctx_marker_mask[:known_n_ctx] = False
                viz_ctx = self.render_viz_frames(
                    task=self.task,
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
                viz_img = make_square_cat_image(viz_ctx)[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"ms3_state_ctx_{name}_seed{seed}",
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
        if step_output is not None and "x0_pred_trace" in step_output:
            _subroutine(
                step_output["x0_pred_trace"][bid].float().cpu(),
                step_output["x0_pred_context"][bid].float().cpu(),
                "X0_PRED",
                -1,
                -1,
                -1,
            )

        # --- Prepare masks for inference ---
        dataset_trace_mask = batch["trace_mask"][bid]   # (T, P)
        dataset_ctx_mask = batch["context_mask"][bid]   # (C, P)

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

            # --- Inference: future prediction with goal ---
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
                "Future_w_goal",
                n_steps_int,
                -1,
                seed,
            )

            # --- Inference: future prediction without goal ---
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
                    f"Future_no_goal_{cfg}",
                    n_steps_int,
                    0,
                    seed,
                )
