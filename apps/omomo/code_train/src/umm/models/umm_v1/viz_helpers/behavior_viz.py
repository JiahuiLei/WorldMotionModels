"""Visualization helpers for BEHAVIOR-1K bimanual mobile manipulation domain.

Renders R1-Pro robot link meshes (ZED camera, grippers, fingers) alongside
local point-cloud observations, with coordinate-frame overlays and
auto-computed overhead camera placement. Supports GT replay, x0-pred,
and autoregressive future-prediction logging modes.

Example::

    viz = BehaviorVisualizer(pl_module, behavior_cfg)
    viz.behavior_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import torch
import numpy as np
import logging
import trimesh
from trimesh.creation import icosphere as TriSphere
import matplotlib.colors as mcolors
from PIL import Image as PILImage
from tqdm import tqdm

from umm.datasets.umm_v1.dataset_config import (
    UID_R1PRO_ZED,
    UID_R1PRO_LEFT_GRIPPER,
    UID_R1PRO_LEFT_FINGER1,
    UID_R1PRO_LEFT_FINGER2,
    UID_R1PRO_RIGHT_GRIPPER,
    UID_R1PRO_RIGHT_FINGER1,
    UID_R1PRO_RIGHT_FINGER2,
    UID_R1PRO_ZED_ACT,
    UID_R1PRO_LEFT_GRIPPER_ACT,
    UID_R1PRO_LEFT_FINGER1_ACT,
    UID_R1PRO_LEFT_FINGER2_ACT,
    UID_R1PRO_RIGHT_GRIPPER_ACT,
    UID_R1PRO_RIGHT_FINGER1_ACT,
    UID_R1PRO_RIGHT_FINGER2_ACT,
)

from umm.models.umm_v1.viz_helpers.utils import (
    make_square_cat_image,
    put_caption_at_bottom,
)
from umm.models.umm_v1.viz_helpers.mesh_visualizer import visualize_a_mesh
from lib_io import UnifiedIO

BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
_DEFAULT_YFOV = np.pi / 3.0  # 60° — must match visualize_a_mesh default


def _look_at_opencv(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in OpenCV convention.

    OpenCV camera frame: +X right, +Y down, +Z forward (optical axis).

    Args:
        eye: Camera position in world coordinates (3,).
        target: Point the camera looks at in world coordinates (3,).
        up: World up vector (3,).  Default is +Z up.

    Returns:
        (4, 4) camera-to-world homogeneous transform.

    Example::
        T = _look_at_opencv(np.array([0, -2, 1]), np.array([0, 0, 0]))
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is parallel to up — pick an arbitrary perpendicular
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            right_norm = np.linalg.norm(right)
    right = right / right_norm
    # In OpenCV, camera +Y = "down in image".  We want world-up to be
    # image-up, so camera +Y should oppose world-up.
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = eye
    return T


def _make_frame_mesh(
    T: np.ndarray,
    length: float = 0.1,
    radius: float = 0.005,
) -> trimesh.Trimesh:
    """Create a coordinate-frame visualization (XYZ axes) as a Trimesh at pose T.

    Each axis is a cylinder: X=red, Y=green, Z=blue.  The cylinders start at the
    frame origin (T[:3,3]) and extend along the corresponding column of T[:3,:3].

    Args:
        T: (4,4) homogeneous transform — world pose of the frame.
        length: Axis cylinder length in world units.
        radius: Axis cylinder radius in world units.

    Returns:
        Concatenated trimesh with the three coloured axis cylinders.

    Example::
        frame = _make_frame_mesh(np.eye(4), length=0.15)
        scene = trimesh.Scene([frame])
        scene.show()
    """
    axis_colors = [
        np.array([255, 0, 0, 255], dtype=np.uint8),  # X: red
        np.array([0, 255, 0, 255], dtype=np.uint8),  # Y: green
        np.array([0, 0, 255, 255], dtype=np.uint8),  # Z: blue
    ]
    origin = T[:3, 3]
    parts = []
    for i, color in enumerate(axis_colors):
        axis_dir = T[:3, i]  # i-th column of rotation matrix
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        # Default cylinder is centred at origin along +Z; align it to axis_dir.
        z = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z, axis_dir)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-6:
            R = np.eye(3) if axis_dir[2] > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            angle = np.arccos(float(np.clip(np.dot(z, axis_dir), -1.0, 1.0)))
            R = trimesh.transformations.rotation_matrix(angle, cross / cross_norm)[
                :3, :3
            ]
        T_cyl = np.eye(4)
        T_cyl[:3, :3] = R
        T_cyl[:3, 3] = origin + axis_dir * (length / 2.0)
        cyl.apply_transform(T_cyl)
        cyl.visual.vertex_colors = color
        parts.append(cyl)
    return trimesh.util.concatenate(parts)


class BehaviorVisualizer:
    """Visualizer for BEHAVIOR-1K R1-Pro bimanual mobile manipulation scenes.

    Loads robot link meshes from a pre-computed ``.npz`` archive, maps
    UIDs to link names (including ACT copies), and provides rendering
    routines with coordinate-frame overlays and auto-computed camera
    placement.

    Example::

        viz = BehaviorVisualizer(pl_module, behavior_cfg)
        viz.behavior_viz_routine(batch, "val", 0, eval_model, step_output)
    """

    def __init__(self, pl_module, behavior_cfg: dict):
        """Initialize the BEHAVIOR-1K visualizer.

        Loads robot link meshes from an ``.npz`` archive, maps UIDs
        to link names, and configures visualization parameters.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            behavior_cfg (dict): Config dict with ``robot_path`` and
                optional ``add_cfg`` for classifier-free guidance weights.

        Example::

            viz = BehaviorVisualizer(pl_module, cfg["visualize"]["behavior"])
        """
        self.pl = pl_module
        if behavior_cfg == {} or len(behavior_cfg) == 0 or behavior_cfg is None:
            self.flag = False
            return
        self.flag = True

        self.robot_path = behavior_cfg.get(
            "robot_path",
            "/datasets/pointw-internal/current/behavior_proc_debug/robot_embodiment.npz",
        )
        self.io = UnifiedIO()
        self.robot_data = self.io.load_npzs(self.robot_path)

        # Preload meshes
        self.meshes = {}

        uid_to_link = {
            UID_R1PRO_ZED: "zed_link",
            UID_R1PRO_LEFT_GRIPPER: "left_gripper_link",
            UID_R1PRO_LEFT_FINGER1: "left_gripper_finger_link1",
            UID_R1PRO_LEFT_FINGER2: "left_gripper_finger_link2",
            UID_R1PRO_RIGHT_GRIPPER: "right_gripper_link",
            UID_R1PRO_RIGHT_FINGER1: "right_gripper_finger_link1",
            UID_R1PRO_RIGHT_FINGER2: "right_gripper_finger_link2",
        }

        # ACT uses the same meshes
        act_map = {
            UID_R1PRO_ZED_ACT: UID_R1PRO_ZED,
            UID_R1PRO_LEFT_GRIPPER_ACT: UID_R1PRO_LEFT_GRIPPER,
            UID_R1PRO_LEFT_FINGER1_ACT: UID_R1PRO_LEFT_FINGER1,
            UID_R1PRO_LEFT_FINGER2_ACT: UID_R1PRO_LEFT_FINGER2,
            UID_R1PRO_RIGHT_GRIPPER_ACT: UID_R1PRO_RIGHT_GRIPPER,
            UID_R1PRO_RIGHT_FINGER1_ACT: UID_R1PRO_RIGHT_FINGER1,
            UID_R1PRO_RIGHT_FINGER2_ACT: UID_R1PRO_RIGHT_FINGER2,
        }
        uid_to_link.update({k: uid_to_link[v] for k, v in act_map.items()})

        for uid, link_name in uid_to_link.items():
            if (
                f"{link_name}_v" in self.robot_data
                and f"{link_name}_f" in self.robot_data
            ):
                _v = self.robot_data[f"{link_name}_v"]
                _f = self.robot_data[f"{link_name}_f"]
                _c = self.robot_data.get(f"{link_name}_c", None)
                if _c is None and f"{link_name}_pcl_c" in self.robot_data:
                    # just fallback if needed
                    _c = np.ones((_v.shape[0], 3)) * 255.0
                elif _c is None:
                    _c = np.ones((_v.shape[0], 3)) * 255.0

                self.meshes[uid] = trimesh.Trimesh(
                    vertices=_v, faces=_f, vertex_colors=_c
                )
            else:
                logging.warning(f"Mesh not found for {link_name}")

        self.behavior_viz_cfg = self.pl.cfg.get("visualize", {}).get("behavior", {})
        self.add_cfg = self.behavior_viz_cfg.get("add_cfg", [1.0])

        self.ACT_UIDS = [
            UID_R1PRO_ZED_ACT,
            UID_R1PRO_LEFT_GRIPPER_ACT,
            UID_R1PRO_LEFT_FINGER1_ACT,
            UID_R1PRO_LEFT_FINGER2_ACT,
            UID_R1PRO_RIGHT_GRIPPER_ACT,
            UID_R1PRO_RIGHT_FINGER1_ACT,
            UID_R1PRO_RIGHT_FINGER2_ACT,
        ]

        self.ROBOT_UIDS = [
            UID_R1PRO_ZED,
            UID_R1PRO_LEFT_GRIPPER,
            UID_R1PRO_LEFT_FINGER1,
            UID_R1PRO_LEFT_FINGER2,
            UID_R1PRO_RIGHT_GRIPPER,
            UID_R1PRO_RIGHT_FINGER1,
            UID_R1PRO_RIGHT_FINGER2,
        ]

    def render_viz_frames(
        self,
        prompt_type,
        T_wl,
        viz_T_cam,
        local_pcl,
        local_pcl_mask,
        H=720,
        W=1280,
        local_pcl_max_N=80,
        local_pcl_r=0.02,
        pcl_alpha=1.0,
        marking_mask=None,
        marking_alpha=0.5,
        viz_mask=None,
        show_frames: bool = True,
        frame_length: float = 0.1,
        frame_radius: float = 0.005,
    ):
        """Render 3D visualization frames for a BEHAVIOR trajectory.

        For each timestep, builds a combined mesh (or point-cloud) scene
        from all visible entities with optional coordinate-frame overlays,
        applies per-UID coloring, and renders via ``visualize_a_mesh``.

        Args:
            prompt_type (torch.Tensor): Per-entity UID (N,).
            T_wl (torch.Tensor): World-to-link transforms (T, N, 4, 4).
            viz_T_cam (np.ndarray): Camera-to-world transform (4, 4).
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
            show_frames (bool): Whether to overlay coordinate frames.
            frame_length (float): Length of frame axis cylinders.
            frame_radius (float): Radius of frame axis cylinders.

        Returns:
            list[np.ndarray]: Rendered RGBA frames (H, W, 4).

        Example::

            frames = viz.render_viz_frames(uid, T_wl, cam, pcl, mask)
        """
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
        for t in tqdm(range(T)):
            _mm = []
            for i in range(N):
                if not local_pcl_mask[i]:
                    continue
                if not viz_mask[t, i]:
                    continue

                uid = int(prompt_type[i])
                _pcl = local_pcl[i]
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]

                _rgb = _pcl[:, 3:].clone().cpu().numpy()

                if uid in self.ROBOT_UIDS:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 2] = 1.0  # Blue-ish
                elif uid in self.ACT_UIDS:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0  # Yellow-ish
                    _rgb[:, 1] = 1.0

                # if marking_mask[t, i]:
                #     _hsv = mcolors.rgb_to_hsv(_rgb)
                #     _hsv[..., 1] = 1.0
                #     _hsv[..., 2] = 1.0
                #     _rgb = mcolors.hsv_to_rgb(_hsv)
                # else:
                #     _hsv = mcolors.rgb_to_hsv(_rgb)
                #     _rgb = mcolors.hsv_to_rgb(_hsv)

                _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha

                use_mesh = uid in self.meshes

                if not use_mesh:
                    _pcl_t = (T_wl[t, i][:3, :3] @ _pcl[:, :3].T).T + T_wl[t, i][
                        None, :3, 3
                    ]
                    _m_pcl = []
                    _rgb_int = np.clip(_rgb * 255.0, 0, 255).astype(np.uint8)
                    alpha_int = int(_pcl_alpha * 255)
                    for _idx in range(len(_pcl_t)):
                        sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                        sph.apply_translation(_pcl_t[_idx].cpu().numpy())
                        sph.visual.vertex_colors = _rgb_int[_idx].tolist() + [alpha_int]
                        _m_pcl.append(sph)
                    if len(_m_pcl) > 0:
                        _m_pcl = trimesh.util.concatenate(_m_pcl)
                        _mm.append(_m_pcl)
                else:
                    _mesh = self.meshes[uid].copy()
                    _mesh.apply_transform(T_wl[t, i].cpu().numpy())
                    mean_color = np.clip(_rgb.mean(0) * 255.0, 0, 255).astype(np.uint8)
                    _mesh.visual.vertex_colors = mean_color.tolist() + [
                        int(_pcl_alpha * 255)
                    ]
                    _mm.append(_mesh)

                if show_frames:
                    _mm.append(
                        _make_frame_mesh(
                            T_wl[t, i].cpu().numpy(),
                            length=frame_length,
                            radius=frame_radius,
                        )
                    )

            # # ! debug
            # trimesh.util.concatenate(_mm).export(f"./debug/dbg_{t}.obj")
            # # ! debug
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

    def _collect_scene_bounds(
        self,
        T_wl: torch.Tensor,
        prompt_type: torch.Tensor,
        local_pcl: torch.Tensor,
        local_pcl_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the world-space axis-aligned bounding box of the full scene.

        Iterates over **all timesteps** in ``T_wl`` so the camera encloses
        the entire trajectory, not just the first frame.

        Args:
            T_wl: (T, N, 4, 4) world poses across all timesteps.
            prompt_type: (N,) UID per entity.
            local_pcl: (N, P, D) local point clouds.
            local_pcl_mask: (N,) boolean mask.

        Returns:
            (bbox_min, bbox_max) each (3,) in world coordinates.

        Example::
            lo, hi = viz._collect_scene_bounds(
                batch["trace"][0], batch["trace_uid"][0],
                batch["local_pcl"][0], batch["local_pcl_mask"][0],
            )
        """
        pts: list[np.ndarray] = []
        T_total = T_wl.shape[0]
        N = T_wl.shape[1]
        for t in range(T_total):
            for i in range(N):
                if not local_pcl_mask[i]:
                    continue
                uid = int(prompt_type[i])
                T_np = T_wl[t, i].cpu().numpy()
                if uid in self.meshes:
                    m = self.meshes[uid].copy()
                    m.apply_transform(T_np)
                    pts.append(m.vertices)
                else:
                    pcl_xyz = local_pcl[i][:, :3].cpu().numpy()
                    world_xyz = (T_np[:3, :3] @ pcl_xyz.T).T + T_np[:3, 3]
                    pts.append(world_xyz)
        if len(pts) == 0:
            return np.zeros(3), np.ones(3)
        all_pts = np.concatenate(pts, axis=0)
        return all_pts.min(axis=0), all_pts.max(axis=0)

    def make_viz_T_cam(self, batch, bid):
        """Compute a static overhead camera that sees the full scene.

        Uses the bounding box of all meshes/point-clouds across **every
        timestep** so that even the last frame stays in view.  The camera
        is placed above the centroid at a 70-degree elevation angle, looking
        down at the scene centre, with distance derived from the FOV so that
        the entire bounding box fits the frame.

        Args:
            batch (dict): Training batch containing ``trace``, ``trace_mask``,
                ``trace_uid``, ``local_pcl``, and ``local_pcl_mask``.
            bid (int): Batch item index to visualize.

        Returns:
            np.ndarray: Camera-to-world 4x4 homogeneous transform.

        Example::

            T_cam = viz.make_viz_T_cam(batch, 0)
        """
        n_steps = int(batch["trace_mask"][bid][:, 0].sum())
        bbox_min, bbox_max = self._collect_scene_bounds(
            T_wl=batch["trace"][bid, :n_steps],
            prompt_type=batch["trace_uid"][bid],
            local_pcl=batch["local_pcl"][bid],
            local_pcl_mask=batch["local_pcl_mask"][bid],
        )

        centroid = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        # Horizontal footprint determines how far back the camera must be
        max_horizontal = max(extent[0], extent[1], 0.5)
        max_vertical = max(extent[2], 0.3)
        half_fov = _DEFAULT_YFOV / 2.0
        # Distance so the horizontal extent fits, with 40 % margin
        dist_h = (max_horizontal / 2.0) / np.tan(half_fov) * 1.4
        dist_v = (max_vertical / 2.0) / np.tan(half_fov) * 1.4
        distance = max(dist_h, dist_v, 0.8)

        # Elevation angle from horizontal (70° = mostly looking down)
        elev = np.radians(70.0)
        # Azimuth offset: use ZED forward projected onto XY to pick a
        # viewing direction that shows the robot's front workspace.
        trace_uids = batch["trace_uid"][bid].cpu().numpy().tolist()
        if UID_R1PRO_ZED in trace_uids:
            idx = trace_uids.index(UID_R1PRO_ZED)
            T_zed = batch["trace"][bid, 0, idx].cpu().numpy()
            fwd_xy = T_zed[:2, 2]  # ZED +Z projected onto XY
            azimuth = np.arctan2(fwd_xy[1], fwd_xy[0])
        else:
            azimuth = 0.0

        eye = centroid + np.array(
            [
                distance * np.cos(elev) * np.cos(azimuth),
                distance * np.cos(elev) * np.sin(azimuth),
                distance * np.sin(elev),
            ]
        )
        return _look_at_opencv(eye, centroid)

    @torch.inference_mode()
    def behavior_viz_routine(
        self, batch, phase, batch_idx, eval_model, step_output=None
    ):
        """Top-level visualization routine for BEHAVIOR-1K.

        Finds the first batch item containing R1-Pro robot UIDs, renders
        GT multi-view RGB, x0-pred, and autoregressive future-prediction
        (with/without goal, with/without context) modes.

        Args:
            batch (dict): Training batch.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference.
            step_output (dict, optional): Training step outputs containing
                ``x0_pred_trace`` and ``x0_pred_context``.

        Example::

            viz.behavior_viz_routine(batch, "val", 0, eval_model, step_output)
        """
        if not self.flag:
            return

        bid = None
        for i in range(len(batch["trace"])):
            uids = batch["trace_uid"][i]
            if eval_model.get_type_mask(uids, self.ROBOT_UIDS).any():
                bid = i
                break

        if bid is None:
            return

        viz_T_cam = self.make_viz_T_cam(batch, bid)
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(
            f"VIZ Behavior phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}"
        )

        if "rgbs_paths" in batch["extra"]:
            view_paths = batch["extra"]["rgbs_paths"][bid]
            view_names = batch["extra"]["rgbs_views"][bid]
            frame_indices = batch["extra"]["rgbs_indices"][bid]
            per_view_frames: list[list[np.ndarray]] = []
            for vp in view_paths:
                loaded = self.io.load_rgbs(path=vp, indices=frame_indices)
                if len(loaded) > 0:
                    per_view_frames.append(loaded)
            if len(per_view_frames) > 0:
                n_frames = min(len(vf) for vf in per_view_frames)
                target_h = max(vf[0].shape[0] for vf in per_view_frames)
                cat_frames: list[np.ndarray] = []
                for fi in range(n_frames):
                    row: list[np.ndarray] = []
                    for vf in per_view_frames:
                        img = vf[fi]
                        if img.shape[0] != target_h:
                            scale = target_h / img.shape[0]
                            new_w = int(img.shape[1] * scale)
                            img = np.array(
                                PILImage.fromarray(img).resize(
                                    (new_w, target_h), PILImage.BILINEAR
                                )
                            )
                        row.append(img)
                    cat_frames.append(np.concatenate(row, axis=1))
                cat_frames = put_caption_at_bottom(cat_frames, batch["text"][bid])
                cat_tensor = torch.from_numpy(np.stack(cat_frames, 0))
                self.pl.logger.log_video(
                    phase,
                    "gt_multiview_rgb",
                    cat_tensor.float().permute(0, 3, 1, 2) / 255.0,
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
                prompt_type=batch["trace_uid"][bid],
                T_wl=T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                marking_mask=trace_marker_mask,
                H=720,
                W=1280,
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
                    H=720,
                    W=1280,
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

        _subroutine(
            batch["trace"][bid, :n_steps],
            batch["context"][bid],
            "GT",
            n_steps,
            100000,
            seed="gt",
        )

        # return # ! debug

        if "x0_pred_trace" in step_output:
            _subroutine(
                step_output["x0_pred_trace"][bid].float(),
                step_output["x0_pred_context"][bid].float(),
                "X0_PRED",
                -1,
                -1,
                -1,
            )

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

                    # Future prediction without goal
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

                    # Only current frame known, no goal
                    cur_i = n_steps_int - 1
                    trace_known_mask_cur = torch.zeros_like(trace_known_mask)
                    trace_known_mask_cur[0, cur_i] = True
                    gen_context, gen_trace = eval_model.inference(
                        trace=batch["trace"][bid : bid + 1],
                        context=batch["context"][bid : bid + 1],
                        trace_known_mask=trace_known_mask_cur,
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
