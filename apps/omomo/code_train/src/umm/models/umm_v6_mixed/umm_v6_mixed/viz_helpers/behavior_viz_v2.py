"""Hybrid visualization for BEHAVIOR-1K V5 domain.

Robot links are rendered as meshes (from robot_embodiment.npz).
Scene/background points (pixel queries) are rendered as colored icospheres
at their trace positions — no local PCLs, no coordinate frames.

Drop-in replacement for ``behavior_viz.py``: same class name
``BehaviorVisualizer``, same method signatures.

Example::

    viz = BehaviorVisualizer(pl_module, behavior_cfg)
    viz.behavior_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import torch
import numpy as np
import logging
import trimesh
from PIL import Image as PILImage

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

from umm.models.umm_v6_mixed.viz_helpers.utils import (
    make_square_cat_image,
    put_caption_at_bottom,
)
from umm.models.umm_v6_mixed.viz_helpers.mesh_visualizer import visualize_a_mesh

BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
_DEFAULT_YFOV = np.pi / 3.0

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

    Args:
        images: Camera images (V, 3, H, W) or (V, H, W, 3), float [0,1]
            or uint8 [0,255]. Can be numpy or torch tensor.
        pixel_uv: Normalized UV coords (P, 2) in [0,1].
        pixel_view_idx: View index per query (P,), int.
        pixel_query_mask: Valid query mask (P,), bool.

    Returns:
        np.ndarray: Annotated image (H, W_total, 3) uint8, or None.
    """
    import cv2

    if images is None or pixel_uv is None:
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(pixel_uv, torch.Tensor):
        pixel_uv = pixel_uv.cpu().numpy()
    if isinstance(pixel_view_idx, torch.Tensor):
        pixel_view_idx = pixel_view_idx.cpu().numpy()
    if isinstance(pixel_query_mask, torch.Tensor):
        pixel_query_mask = pixel_query_mask.cpu().numpy()

    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (images * 255).clip(0, 255).astype(np.uint8)
        else:
            images = images.clip(0, 255).astype(np.uint8)

    V, H, W, _ = images.shape
    annotated = [img.copy() for img in images]

    n_colors = len(_QUERY_COLORS)
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

        radius = max(4, min(H, W) // 80)
        cv2.circle(annotated[vi], (px_x, px_y), radius + 2, (0, 0, 0), -1)
        cv2.circle(annotated[vi], (px_x, px_y), radius, color, -1)

    return np.concatenate(annotated, axis=1)


def _look_at_opencv(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in OpenCV convention."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    true_up = np.cross(right, forward)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = -true_up  # OpenCV Y-down
    T[:3, 2] = forward  # OpenCV Z-forward
    T[:3, 3] = eye
    return T


# Default gray for scene points when no RGB is available
_DEFAULT_SCENE_COLOR = np.array([180, 180, 180, 255], dtype=np.uint8)


class BehaviorVisualizer:
    """Hybrid visualizer: robot meshes + scene point icospheres.

    Example::

        viz = BehaviorVisualizer(pl_module, behavior_cfg)
        viz.behavior_viz_routine(batch, "val", 0, eval_model, step_output)
    """

    def __init__(self, pl_module, behavior_cfg: dict):
        self.pl = pl_module
        if behavior_cfg == {} or len(behavior_cfg) == 0 or behavior_cfg is None:
            self.flag = False
            return
        self.flag = True

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

        # Colors
        self._robot_color = np.array([50, 50, 255, 255], dtype=np.uint8)
        self._act_color = np.array([255, 200, 50, 255], dtype=np.uint8)
        self._robot_uid_set = set(self.ROBOT_UIDS)
        self._act_uid_set = set(self.ACT_UIDS)
        self._mesh_uid_set = set(self.ROBOT_UIDS + self.ACT_UIDS)

        # Load robot meshes (same as full behavior_viz.py)
        self.meshes = {}
        robot_path = behavior_cfg.get(
            "robot_path",
            "/datasets/pointw-internal/current/behavior_proc_debug/robot_embodiment.npz",
        )
        try:
            from lib_io import UnifiedIO
            robot_data = UnifiedIO().load_npzs(robot_path)

            uid_to_link = {
                UID_R1PRO_ZED: "zed_link",
                UID_R1PRO_LEFT_GRIPPER: "left_gripper_link",
                UID_R1PRO_LEFT_FINGER1: "left_gripper_finger_link1",
                UID_R1PRO_LEFT_FINGER2: "left_gripper_finger_link2",
                UID_R1PRO_RIGHT_GRIPPER: "right_gripper_link",
                UID_R1PRO_RIGHT_FINGER1: "right_gripper_finger_link1",
                UID_R1PRO_RIGHT_FINGER2: "right_gripper_finger_link2",
            }
            # ACT uses same meshes
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
                    f"{link_name}_v" in robot_data
                    and f"{link_name}_f" in robot_data
                ):
                    _v = robot_data[f"{link_name}_v"]
                    _f = robot_data[f"{link_name}_f"]
                    _c = robot_data.get(f"{link_name}_c", None)
                    if _c is None:
                        _c = np.ones((_v.shape[0], 3)) * 255.0
                    self.meshes[uid] = trimesh.Trimesh(
                        vertices=_v, faces=_f, vertex_colors=_c
                    )
                else:
                    logging.warning(f"Mesh not found for {link_name}")
        except Exception as e:
            logging.warning(f"Could not load robot meshes: {e}. Using dots for robot.")

        # Icosphere template for scene points (vectorized)
        self._sphere_tpl = trimesh.creation.icosphere(subdivisions=1, radius=0.008)
        self._tpl_v = np.array(self._sphere_tpl.vertices, dtype=np.float32)
        self._tpl_f = np.array(self._sphere_tpl.faces, dtype=np.int64)

    def render_viz_frames(
        self,
        prompt_type,
        T_wl,
        viz_T_cam,
        local_pcl,
        local_pcl_mask,
        H=720,
        W=1280,
        marking_mask=None,
        marking_alpha=0.5,
        viz_obj_pcl=None,
        group_ids=None,
        scene_rgb=None,
        **kwargs,
    ):
        """Render frames: robot as meshes, scene points as icosphere dots.

        All parts are rendered regardless of local_pcl_mask. Robot/ACT UIDs
        get mesh rendering; everything else gets colored icosphere at position.

        Args:
            scene_rgb: (N, 3) uint8 per-part RGB colors from UV-sampled images.
                       Only used for scene parts (non-robot/act). If provided,
                       overrides group_id tab20 coloring.
        """
        T_wl_np = T_wl.cpu().numpy() if torch.is_tensor(T_wl) else T_wl  # (T, N, 4, 4)
        T_steps, N = T_wl_np.shape[:2]
        prompt_np = (
            prompt_type.cpu().numpy() if torch.is_tensor(prompt_type) else prompt_type
        )
        # Classify parts
        scene_idx = []  # indices of scene/object parts (rendered as dots)
        mesh_idx = []   # indices of robot/act parts (rendered as meshes)
        for i in range(N):
            uid = int(prompt_np[i])
            if uid in self._mesh_uid_set:
                mesh_idx.append(i)
            else:
                scene_idx.append(i)

        # Precompute scene colors  # (len(scene_idx), 4) uint8
        # Uses real RGB from UV-sampled image pixels; falls back to gray.
        scene_colors = np.tile(_DEFAULT_SCENE_COLOR, (len(scene_idx), 1))
        if scene_rgb is not None:
            for j, i in enumerate(scene_idx):
                scene_colors[j, :3] = scene_rgb[i, :3]
                scene_colors[j, 3] = 255

        Vs = self._tpl_v.shape[0]
        Fs = self._tpl_f.shape[0]

        frames = []
        for t in range(T_steps):
            _mm = []

            # --- Robot/ACT meshes ---
            for i in mesh_idx:
                uid = int(prompt_np[i])
                _alpha = 255
                if marking_mask is not None:
                    mk = marking_mask[t, i]
                    if torch.is_tensor(mk):
                        mk = mk.item()
                    if mk:
                        _alpha = int(marking_alpha * 255)

                if uid in self.meshes:
                    _mesh = self.meshes[uid].copy()
                    _mesh.apply_transform(T_wl_np[t, i])
                    # Color: blue for robot, yellow for act
                    if uid in self._robot_uid_set:
                        c = [50, 50, 255, _alpha]
                    else:
                        c = [255, 200, 50, _alpha]
                    _mesh.visual.vertex_colors = c
                    _mm.append(_mesh)
                else:
                    # Fallback: dot
                    sph = self._sphere_tpl.copy()
                    sph.apply_translation(T_wl_np[t, i, :3, 3])
                    if uid in self._robot_uid_set:
                        sph.visual.vertex_colors = [50, 50, 255, _alpha]
                    else:
                        sph.visual.vertex_colors = [255, 200, 50, _alpha]
                    _mm.append(sph)

            # --- Scene points as vectorized icospheres ---
            M = len(scene_idx)
            if M > 0:
                positions = T_wl_np[t, scene_idx, :3, 3]  # (M, 3)
                all_verts = self._tpl_v[None, :, :] + positions[:, None, :]  # (M, Vs, 3)
                all_verts = all_verts.reshape(M * Vs, 3)

                offsets = (np.arange(M) * Vs)[:, None, None]
                all_faces = self._tpl_f[None, :, :] + offsets
                all_faces = all_faces.reshape(M * Fs, 3)

                part_colors = scene_colors.copy()
                if marking_mask is not None:
                    mk_t = marking_mask[t, scene_idx]
                    if torch.is_tensor(mk_t):
                        mk_t = mk_t.cpu().numpy()
                    predicted = mk_t.astype(bool)
                    part_colors[predicted, 3] = int(marking_alpha * 255)
                vert_colors = np.repeat(part_colors, Vs, axis=0)  # (M*Vs, 4)

                scene_mesh = trimesh.Trimesh(
                    vertices=all_verts, faces=all_faces, process=False
                )
                scene_mesh.visual.vertex_colors = vert_colors
                _mm.append(scene_mesh)

            # Render
            if len(_mm) > 0:
                combined = trimesh.util.concatenate(_mm)
                rgb = visualize_a_mesh(
                    cam_T_wc=viz_T_cam,
                    mesh=combined,
                    width=W,
                    height=H,
                    background_color=BACKGROUND_COLOR,
                    ground_color=GROUND_COLOR,
                    draw_ground=True,
                    ground_z=0.0,
                )
                frames.append(rgb)
            else:
                frames.append(np.ones((H, W, 4), dtype=np.uint8) * 255)

        return frames

    def make_viz_T_cam(self, batch, bid):
        """Compute camera that sees all parts (robot + scene points)."""
        n_steps = int(batch["trace_mask"][bid][:, 0].sum())
        T_wl = batch["trace"][bid, :n_steps].cpu().numpy()  # (T, N, 4, 4)
        N = T_wl.shape[1]

        # Include ALL parts' positions for camera framing (not just valid mask)
        positions = T_wl[:, :, :3, 3].reshape(-1, 3)  # (T*N, 3)
        # Filter out zero positions (padding)
        norms = np.linalg.norm(positions, axis=1)
        positions = positions[norms > 0.01]
        if len(positions) == 0:
            positions = np.zeros((1, 3))

        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)

        centroid = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        max_horizontal = max(extent[0], extent[1], 0.5)
        half_fov = _DEFAULT_YFOV / 2.0
        distance = max((max_horizontal / 2.0) / np.tan(half_fov) * 1.4, 0.8)

        elev = np.radians(70.0)

        # Use ZED forward for azimuth
        trace_uids = batch["trace_uid"][bid].cpu().numpy().tolist()
        if UID_R1PRO_ZED in trace_uids:
            idx = trace_uids.index(UID_R1PRO_ZED)
            T_zed = batch["trace"][bid, 0, idx].cpu().numpy()
            fwd_xy = T_zed[:2, 2]
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

        Same structure as behavior_viz.py: GT, X0_PRED, and inference modes.
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

        # GT multiview RGB logging
        if "rgbs_paths" in batch.get("extra", {}):
            view_paths = batch["extra"]["rgbs_paths"][bid]
            frame_indices = batch["extra"]["rgbs_indices"][bid]
            from lib_io import UnifiedIO

            io = UnifiedIO()
            per_view_frames: list[list[np.ndarray]] = []
            for vp in view_paths:
                loaded = io.load_rgbs(path=vp, indices=frame_indices)
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

        # --- Extract per-point RGB from images via UV coordinates ---
        scene_rgb_np = None  # (N, 3) uint8 or None
        if "images" in extra and extra["images"][bid] is not None:
            _imgs = extra["images"][bid]  # (V, 3, H, W) float or uint8
            if isinstance(_imgs, torch.Tensor):
                _imgs = _imgs.cpu().numpy()
            # _imgs is (V, 3, H, W) — transpose to (V, H, W, 3)
            if _imgs.ndim == 4 and _imgs.shape[1] == 3:
                _imgs = _imgs.transpose(0, 2, 3, 1)  # (V, H, W, 3)
            if _imgs.dtype != np.uint8:
                if _imgs.max() <= 1.0:
                    _imgs = (_imgs * 255).clip(0, 255).astype(np.uint8)
                else:
                    _imgs = _imgs.clip(0, 255).astype(np.uint8)

            N_parts = batch["trace_uid"][bid].shape[0]
            scene_rgb_np = np.full((N_parts, 3), 180, dtype=np.uint8)  # default gray

            if (
                "pixel_uv_normalized" in extra
                and extra["pixel_uv_normalized"][bid] is not None
            ):
                _uv = extra["pixel_uv_normalized"][bid]  # (P_raw, 2) float [0,1]
                _vi = extra["pixel_view_idx"][bid]  # (P_raw,) int
                _qm = extra["pixel_query_mask"][bid]  # (P_raw,) bool
                if isinstance(_uv, torch.Tensor):
                    _uv = _uv.cpu().numpy()
                if isinstance(_vi, torch.Tensor):
                    _vi = _vi.cpu().numpy()
                if isinstance(_qm, torch.Tensor):
                    _qm = _qm.cpu().numpy()

                # Map pixel queries back to part indices
                # Scene parts start after robot/act parts in the prompt
                uids_np = batch["trace_uid"][bid]
                if torch.is_tensor(uids_np):
                    uids_np = uids_np.cpu().numpy()
                scene_part_indices = [
                    i for i in range(N_parts) if int(uids_np[i]) not in self._mesh_uid_set
                ]

                P_raw = len(_uv)
                n_scene = len(scene_part_indices)
                # Each pixel query corresponds to one scene part in order
                for qi in range(min(P_raw, n_scene)):
                    if not _qm[qi]:
                        continue
                    part_i = scene_part_indices[qi]
                    view_i = int(_vi[qi])
                    if view_i >= len(_imgs):
                        continue
                    img_hw = _imgs[view_i]  # (H, W, 3)
                    _H, _W = img_hw.shape[:2]
                    px_u = int(np.clip(_uv[qi, 0] * (_W - 1), 0, _W - 1))
                    px_v = int(np.clip(_uv[qi, 1] * (_H - 1), 0, _H - 1))
                    scene_rgb_np[part_i] = img_hw[px_v, px_u]

        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
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
                group_ids=batch["trace_group_id"][bid],
                scene_rgb=scene_rgb_np,
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
                    group_ids=batch["trace_group_id"][bid],
                    scene_rgb=scene_rgb_np,
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

        # Extract V2 image data from batch["extra"] for inference
        device = batch["trace"].device
        P = batch["trace"].shape[2]
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
            )
        if (
            "pixel_uv_normalized" in extra
            and extra["pixel_uv_normalized"][bid] is not None
        ):
            raw_uv = torch.as_tensor(extra["pixel_uv_normalized"][bid]).float()
            raw_view = torch.as_tensor(extra["pixel_view_idx"][bid]).long()
            raw_qmask = torch.as_tensor(extra["pixel_query_mask"][bid]).bool()
            P_raw = raw_uv.shape[0]
            if P_raw < P:
                raw_uv = torch.cat([raw_uv, torch.zeros(P - P_raw, 2)], 0)
                raw_view = torch.cat(
                    [raw_view, torch.zeros(P - P_raw, dtype=torch.long)], 0
                )
                raw_qmask = torch.cat(
                    [raw_qmask, torch.zeros(P - P_raw, dtype=torch.bool)], 0
                )
            inf_pixel_uv = raw_uv[:P].unsqueeze(0).to(device)
            inf_pixel_view = raw_view[:P].unsqueeze(0).to(device)
            inf_pixel_qmask = raw_qmask[:P].unsqueeze(0).to(device)

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
