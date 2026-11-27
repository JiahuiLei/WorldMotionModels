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

from umm.datasets.umm_v2.dataset_config import (
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

from umm.models.umm_v2.viz_helpers.utils import (
    make_square_cat_image,
    put_caption_at_bottom,
)
from umm.models.umm_v2.viz_helpers.mesh_visualizer import visualize_a_mesh
from lib_io import UnifiedIO

BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
_DEFAULT_YFOV = np.pi / 3.0  # 60° — must match visualize_a_mesh default

# Distinct colors for pixel query markers (RGB uint8)
_QUERY_COLORS = [
    (255, 0, 0),      # red
    (0, 200, 0),      # green
    (0, 80, 255),     # blue
    (255, 165, 0),    # orange
    (160, 32, 240),   # purple
    (0, 200, 200),    # cyan
    (255, 105, 180),  # hot pink
    (255, 255, 0),    # yellow
    (0, 128, 0),      # dark green
    (139, 69, 19),    # saddle brown
]


def render_pixel_query_overlay(images, pixel_uv, pixel_view_idx, pixel_query_mask):
    """Draw colored markers on camera images at pixel query UV positions.

    For each valid pixel query, draws a filled circle with a contrasting
    outline on the corresponding camera view image. Queries on the same
    view get distinct colors. Returns one annotated image per camera view,
    concatenated horizontally.

    Args:
        images: Camera images (V, 3, H, W) or (V, H, W, 3), float [0,1]
            or uint8 [0,255]. Can be numpy or torch tensor.
        pixel_uv: Normalized UV coords (P, 2) in [0,1], where
            uv[:,0] = horizontal, uv[:,1] = vertical.
        pixel_view_idx: View index per query (P,), int.
        pixel_query_mask: Valid query mask (P,), bool.

    Returns:
        np.ndarray: Annotated image (H, W_total, 3) uint8 with all views
            concatenated side by side, or None if inputs are invalid.
    """
    import cv2

    if images is None or pixel_uv is None:
        return None

    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(pixel_uv, torch.Tensor):
        pixel_uv = pixel_uv.cpu().numpy()
    if isinstance(pixel_view_idx, torch.Tensor):
        pixel_view_idx = pixel_view_idx.cpu().numpy()
    if isinstance(pixel_query_mask, torch.Tensor):
        pixel_query_mask = pixel_query_mask.cpu().numpy()

    # images: ensure (V, H, W, 3) uint8
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)  # (V, 3, H, W) -> (V, H, W, 3)
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (images * 255).clip(0, 255).astype(np.uint8)
        else:
            images = images.clip(0, 255).astype(np.uint8)

    V, H, W, _ = images.shape
    # Make writable copies for drawing
    annotated = [img.copy() for img in images]

    n_colors = len(_QUERY_COLORS)
    # Track per-view query count for color assignment
    view_query_count = [0] * V

    P = len(pixel_uv)
    for qi in range(P):
        if not pixel_query_mask[qi]:
            continue
        vi = int(pixel_view_idx[qi])
        if vi < 0 or vi >= V:
            continue
        u_norm, v_norm = float(pixel_uv[qi, 0]), float(pixel_uv[qi, 1])
        px_x = int(np.clip(u_norm * (W - 1), 0, W - 1))
        px_y = int(np.clip(v_norm * (H - 1), 0, H - 1))

        color_idx = view_query_count[vi] % n_colors
        color = _QUERY_COLORS[color_idx]
        view_query_count[vi] += 1

        # Circle radius scales with image size
        radius = max(4, min(H, W) // 80)
        # Draw filled circle with black outline for visibility
        cv2.circle(annotated[vi], (px_x, px_y), radius + 2, (0, 0, 0), -1)
        cv2.circle(annotated[vi], (px_x, px_y), radius, color, -1)

    # Concatenate views horizontally
    return np.concatenate(annotated, axis=1)  # (H, V*W, 3) uint8


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
        viz_obj_pcl=None,
        group_ids=None,
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

        # Pre-compute averaged SE(3) per group for multi-query object UIDs
        group_avg_T_wl = None
        if viz_obj_pcl is not None and group_ids is not None:
            obj_part_indices = [
                i
                for i in range(N)
                if not local_pcl_mask[i]
                and i < len(viz_obj_pcl)
                and viz_obj_pcl[i].abs().sum() > 0
            ]
            if obj_part_indices:
                from collections import defaultdict

                gid_to_parts = defaultdict(list)
                for i in obj_part_indices:
                    gid = int(group_ids[i])
                    gid_to_parts[gid].append(i)
                group_avg_T_wl = {}
                for gid, parts in gid_to_parts.items():
                    if len(parts) == 1:
                        group_avg_T_wl[gid] = T_wl[:, parts[0]]  # (T, 4, 4)
                    else:
                        avg_T = T_wl[:, parts[0]].clone()  # (T, 4, 4)
                        for p in parts[1:]:
                            avg_T[:, :3, 3] += T_wl[:, p, :3, 3]
                        avg_T[:, :3, 3] /= len(parts)
                        group_avg_T_wl[gid] = avg_T

        frames = []
        for t in tqdm(range(T)):
            _mm = []
            for i in range(N):
                if not viz_mask[t, i]:
                    continue

                uid = int(prompt_type[i])

                if not local_pcl_mask[i]:
                    # Object part: use viz_obj_pcl if available
                    if (
                        viz_obj_pcl is None
                        or i >= len(viz_obj_pcl)
                        or viz_obj_pcl[i].abs().sum() == 0
                    ):
                        continue

                    _pcl = viz_obj_pcl[i]
                    if len(_pcl) > local_pcl_max_N:
                        _pcl = _pcl[:local_pcl_max_N]
                    _rgb = _pcl[:, 3:].clone().cpu().numpy()

                    gid = int(group_ids[i]) if group_ids is not None else -1
                    if group_avg_T_wl is not None and gid in group_avg_T_wl:
                        T_for_anim = group_avg_T_wl[gid][t]
                    else:
                        T_for_anim = T_wl[t, i]

                    _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha

                    _pcl_t = (T_for_anim[:3, :3] @ _pcl[:, :3].T).T + T_for_anim[
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
                        _mm.append(trimesh.util.concatenate(_m_pcl))

                    if show_frames:
                        _mm.append(
                            _make_frame_mesh(
                                T_for_anim.cpu().numpy(),
                                length=frame_length,
                                radius=frame_radius,
                            )
                        )
                    continue

                # --- Original robot link rendering below (unchanged) ---
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
        viz_obj_pcl=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the world-space axis-aligned bounding box of the full scene.

        Iterates over **all timesteps** in ``T_wl`` so the camera encloses
        the entire trajectory, not just the first frame.

        Args:
            T_wl: (T, N, 4, 4) world poses across all timesteps.
            prompt_type: (N,) UID per entity.
            local_pcl: (N, P, D) local point clouds.
            local_pcl_mask: (N,) boolean mask.
            viz_obj_pcl: Optional (N, num_pts, 6) object viz PCLs.

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
                    if (
                        viz_obj_pcl is not None
                        and i < len(viz_obj_pcl)
                        and viz_obj_pcl[i].abs().sum() > 0
                    ):
                        pcl_xyz = viz_obj_pcl[i][:, :3].cpu().numpy()
                        T_np = T_wl[t, i].cpu().numpy()
                        world_xyz = (T_np[:3, :3] @ pcl_xyz.T).T + T_np[:3, 3]
                        pts.append(world_xyz)
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
        viz_obj_pcl_data = None
        if "viz_obj_pcl" in batch.get("extra", {}):
            _raw = batch["extra"]["viz_obj_pcl"][bid]
            if _raw is not None:
                _device = batch["trace"].device
                viz_obj_pcl_data = torch.as_tensor(_raw).float().to(_device)
        bbox_min, bbox_max = self._collect_scene_bounds(
            T_wl=batch["trace"][bid, :n_steps],
            prompt_type=batch["trace_uid"][bid],
            local_pcl=batch["local_pcl"][bid],
            local_pcl_mask=batch["local_pcl_mask"][bid],
            viz_obj_pcl=viz_obj_pcl_data,
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
        viz_obj_pcl_tensor = None
        if "viz_obj_pcl" in batch.get("extra", {}):
            _raw = batch["extra"]["viz_obj_pcl"][bid]
            if _raw is not None:
                _device = batch["trace"].device
                viz_obj_pcl_tensor = torch.as_tensor(_raw).float().to(_device)
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

        # --- Pixel query click overlay ---
        extra = batch.get("extra", {})
        if (
            "images" in extra
            and extra["images"][bid] is not None
            and "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
        ):
            overlay = render_pixel_query_overlay(
                images=extra["images"][bid],
                pixel_uv=extra["pixel_uv_normalized"][bid],
                pixel_view_idx=extra["pixel_view_idx"][bid],
                pixel_query_mask=extra["pixel_query_mask"][bid],
            )
            if overlay is not None:
                # overlay is (H, W_total, 3) uint8 -> (3, H, W_total) float [0,1]
                overlay_t = (
                    torch.from_numpy(overlay).float().permute(2, 0, 1) / 255.0
                )
                self.pl.logger.log_image(
                    phase,
                    "pixel_query_clicks",
                    overlay_t,
                    self.pl.global_step,
                    batch_idx,
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
                viz_obj_pcl=viz_obj_pcl_tensor,
                group_ids=batch["trace_group_id"][bid],
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
                    viz_obj_pcl=viz_obj_pcl_tensor,
                    group_ids=batch["trace_group_id"][bid],
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

        if step_output is not None and "x0_pred_trace" in step_output:
            _subroutine(
                step_output["x0_pred_trace"][bid].float(),
                step_output["x0_pred_context"][bid].float(),
                "X0_PRED",
                -1,
                -1,
                -1,
            )

        if not hasattr(eval_model, "inference"):
            logging.warning("eval_model has no 'inference' method — skipping AR viz")
            return

        # ── Extract V2 image data from batch["extra"] for inference ──
        device = batch["trace"].device
        P = batch["trace"].shape[2]  # padded P
        inf_images, inf_pixel_uv, inf_pixel_view, inf_pixel_qmask = (
            None,
            None,
            None,
            None,
        )
        extra = batch.get("extra", {})
        if "images" in extra and extra["images"][bid] is not None:
            inf_images = (
                torch.as_tensor(extra["images"][bid]).float().unsqueeze(0).to(device)
            )  # (1, V, 3, H, W)
        if (
            "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
        ):
            raw_uv = torch.as_tensor(
                extra["pixel_uv_normalized"][bid]
            ).float()  # (P_raw, 2)
            raw_view = torch.as_tensor(extra["pixel_view_idx"][bid]).long()  # (P_raw,)
            raw_qmask = torch.as_tensor(
                extra["pixel_query_mask"][bid]
            ).bool()  # (P_raw,)
            # Pad to match collated P
            P_raw = raw_uv.shape[0]
            if P_raw < P:
                raw_uv = torch.cat([raw_uv, torch.zeros(P - P_raw, 2)], 0)
                raw_view = torch.cat(
                    [raw_view, torch.zeros(P - P_raw, dtype=torch.long)], 0
                )
                raw_qmask = torch.cat(
                    [raw_qmask, torch.zeros(P - P_raw, dtype=torch.bool)], 0
                )
            inf_pixel_uv = raw_uv[:P].unsqueeze(0).to(device)  # (1, P, 2)
            inf_pixel_view = raw_view[:P].unsqueeze(0).to(device)  # (1, P)
            inf_pixel_qmask = raw_qmask[:P].unsqueeze(0).to(device)  # (1, P)

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
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
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
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
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
                        images=inf_images,
                        pixel_uv_normalized=inf_pixel_uv,
                        pixel_view_idx=inf_pixel_view,
                        pixel_query_mask=inf_pixel_qmask,
                    )
                    _subroutine(
                        gen_trace[0],
                        gen_context[0],
                        f"{prefix}OnlyCur(no goal)_{cfg}",
                        [cur_i],
                        0,
                        seed,
                    )
