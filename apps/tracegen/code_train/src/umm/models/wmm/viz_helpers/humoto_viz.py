"""Visualization helpers for HUMOTO (human-object interaction) domain.

Renders LBS-skinned body meshes and rigidly-transformed object meshes via
``HumotoSeq`` from the data preparation pipeline.  Camera follows the human
pelvis.  Supports GT replay, half-future, 1-step, o2h (objects known / body
unknown), and h2o (body known / objects unknown) inference modes.

Ported to work with the v13 model architecture (AgentV11).

Example::

    viz = HumotoVisualizer(pl_module, humoto_cfg)
    viz.humoto_viz_routine(batch, "val", 0, eval_model)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from umm.models.wmm.viz_helpers.utils import (
    get_viz_T_wc,
    make_square_cat_image,
    put_caption_at_bottom,
)
from umm.models.wmm.viz_helpers.mesh_visualizer import visualize_a_mesh
from umm.models.wmm.agent import AgentV11
from umm.datasets.wmm.dataset_config import (
    UID_HUMOTO_BODY_GROUP,
    UID_HUMOTO_OBJECT_GROUP,
)

from prepare_data.humoto.humoto_seq import HumotoSeq

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Colors (RGBA uint8 for trimesh vertex_colors)
# --------------------------------------------------------------------------
BODY_COLOR = (245, 222, 200, 255)       # bright peach/skin
EYES_COLOR = (60, 60, 60, 255)          # dark gray
# Per-object palette -- bright, saturated, high contrast (cycles if > 5)
OBJ_COLORS: Tuple[Tuple[int, int, int, int], ...] = (
    (230, 100, 60, 255),    # bright orange-red
    (60, 160, 230, 255),    # bright blue
    (100, 210, 80, 255),    # bright green
    (220, 80, 180, 255),    # bright magenta
    (240, 200, 40, 255),    # bright yellow
)
# White background for all frames
WHITE_BG = (1.0, 1.0, 1.0, 1.0)
# Top-bar indicator colors (RGB uint8)
SEEN_INDICATOR = np.array([120, 190, 120], dtype=np.uint8)   # green
PRED_INDICATOR = np.array([220, 90, 90], dtype=np.uint8)     # red

# HUMOTO body_T_w rotation columns are 0.01-scale in the raw data.
# The dataset normalizes to unit norm for the model. LBS skinning with
# the original inv_bind requires the original 0.01 scale.
_HUMOTO_ROT_SCALE = 0.01


def _color_mesh(
    mesh: trimesh.Trimesh,
    rgba: Tuple[int, int, int, int],
    alpha: Optional[int] = None,
) -> trimesh.Trimesh:
    """Return a copy of *mesh* with uniform vertex colors. If *alpha* is
    given, it overrides rgba[3]."""
    n = mesh.vertices.shape[0]
    r, g, b, a = rgba
    if alpha is not None:
        a = int(alpha)
    colors = np.tile(np.asarray((r, g, b, a), dtype=np.uint8), (n, 1))
    return trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        vertex_colors=colors,
        process=False,
    )


def _build_pcl_dots(
    local_pcl_xyz: np.ndarray,       # (256, 3) float64
    pose: np.ndarray,                # (4, 4) float64
    rgba: Tuple[int, int, int, int], # alpha=255 opaque
    radius: float = 0.02,
) -> trimesh.Trimesh:
    """Transform 256 local-frame points by *pose* and build a single trimesh
    containing uv_sphere glyphs (subdivisions=1) at each world position.

    Returns ONE merged trimesh (concatenation is cheaper than N meshes).
    """
    # Transform: world = (pose[:3, :3] @ local.T + pose[:3, 3:4]).T
    local = local_pcl_xyz.astype(np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    world = (local @ R.T) + t[None, :]   # (N, 3)

    # Build a single tiny sphere, translate copies, concat
    base = trimesh.creation.uv_sphere(radius=float(radius), count=[4, 4])  # small
    spheres = []
    for p in world:
        s = base.copy()
        s.apply_translation(p)
        spheres.append(s)
    if not spheres:
        return base
    merged = trimesh.util.concatenate(spheres)
    n = merged.vertices.shape[0]
    colors = np.tile(np.asarray(rgba, dtype=np.uint8), (n, 1))
    return trimesh.Trimesh(
        vertices=merged.vertices,
        faces=merged.faces,
        vertex_colors=colors,
        process=False,
    )


class HumotoVisualizer:
    """Mesh-based visualizer for HUMOTO human-object interaction sequences.

    Uses ``HumotoSeq`` LBS skinning for the body and rigid transforms for
    objects.  Camera tracks the human pelvis.

    Args:
        pl_module: Parent PyTorch Lightning module (``PLModel``).
        humoto_cfg (dict): Configuration dict from ``model.humoto`` YAML block.
            Expects ``processed_dir`` key.
    """

    def __init__(self, pl_module, humoto_cfg):
        self.pl = pl_module
        if not humoto_cfg or len(humoto_cfg) == 0:
            self.flag = False
            return
        self.flag = True

        self.processed_dir = humoto_cfg.get("processed_dir", "")

        self.body_uids = UID_HUMOTO_BODY_GROUP
        self.object_uids = UID_HUMOTO_OBJECT_GROUP

        self.humoto_viz_cfg = self.pl.cfg.get("visualize", {}).get("humoto", {})

        # Seq cache: {seq_name: HumotoSeq}
        self._seq_cache: Dict[str, HumotoSeq] = {}

        # Render resolution
        self._H = self.humoto_viz_cfg.get("height", 512)
        self._W = self.humoto_viz_cfg.get("width", 512)

    # ------------------------------------------------------------------
    # Sequence loading (cached)
    # ------------------------------------------------------------------
    def _load_seq(self, seq_name: str) -> HumotoSeq:
        """Load a HumotoSeq from processed_dir, with caching."""
        if seq_name not in self._seq_cache:
            self._seq_cache[seq_name] = HumotoSeq.load(
                self.processed_dir, seq_name,
            )
        return self._seq_cache[seq_name]

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _compute_camera(self, pelvis_pos: np.ndarray) -> np.ndarray:
        """Compute camera-to-world looking at pelvis from behind and above.

        Args:
            pelvis_pos: (3,) world position of pelvis joint.

        Returns:
            (4, 4) cam_T_wc numpy array.
        """
        # Look at body center (~0.3m above pelvis, roughly chest/navel)
        look_at = pelvis_pos.copy().astype(np.float64)
        look_at[2] += 0.3

        # Camera: 3m along +X (side view), 0.5m above center -- full body in frame
        stand_at = pelvis_pos.copy().astype(np.float64)
        stand_at[0] += 3.0
        stand_at[2] += 0.5

        # Use existing utility (returns torch tensor)
        cam_T_wc = get_viz_T_wc(stand_at.tolist(), look_at.tolist())
        return cam_T_wc.numpy()  # (4, 4) float32

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------
    def _render_meshes(
        self,
        mesh_list: List[trimesh.Trimesh],
        cam_T_wc: np.ndarray,
    ) -> np.ndarray:
        """Render a list of colored meshes and composite onto white background.

        Returns (H, W, 4) uint8 image.
        """
        if len(mesh_list) == 0:
            return np.ones((self._H, self._W, 4), dtype=np.uint8) * 255

        merged = trimesh.util.concatenate(mesh_list)
        frame = visualize_a_mesh(
            cam_T_wc=cam_T_wc,
            mesh=merged,
            background_color=WHITE_BG,
            lighting=True,
            draw_ground=False,
            width=self._W,
            height=self._H,
        )
        # Composite against white: pyrender RGBA has alpha<255 for background
        if frame.shape[-1] == 4:
            alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
            rgb = frame[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            composited = (rgb * alpha + white * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            frame = np.concatenate([composited, frame[:, :, 3:4]], axis=-1)
        return frame

    def _render_gt_frame(
        self,
        seq: HumotoSeq,
        t: int,
        cam_T_wc: np.ndarray,
        obj_local_pcls: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Render one GT frame using raw HumotoSeq data directly (no injection).

        Uses seq.skin_body(t) and seq.transform_object(name, t) on the
        ORIGINAL unmodified data. This avoids the rotation scale mismatch
        that occurs when injecting rotation-normalized batch data.

        If *obj_local_pcls* is provided, renders 256 opaque PCL dots at the
        object's current-frame world position on top of the translucent mesh.
        """
        mesh_list: List[trimesh.Trimesh] = []
        body_dict = seq.skin_body(t)
        mesh_list.append(_color_mesh(body_dict["body"], BODY_COLOR))
        mesh_list.append(_color_mesh(body_dict["eyes"], EYES_COLOR))
        for i, name in enumerate(seq.objects):
            try:
                obj_mesh = seq.transform_object(name, t)
                mesh_list.append(_color_mesh(obj_mesh, OBJ_COLORS[i % len(OBJ_COLORS)], alpha=128))
            except Exception as e:
                logger.warning("humoto_viz: GT transform_object(%s, %d) failed: %s", name, t, e)
                continue
            # Add 256 dots at current-frame object world positions
            if obj_local_pcls is not None and i < len(obj_local_pcls) and obj_local_pcls[i] is not None:
                try:
                    pose = seq.objects[name][t]   # (4, 4)
                    pose_np = np.asarray(pose, dtype=np.float64)
                    mesh_list.append(_build_pcl_dots(
                        obj_local_pcls[i], pose_np, OBJ_COLORS[i % len(OBJ_COLORS)],
                    ))
                except Exception as e:
                    logger.warning("humoto_viz: GT pcl dots(%s, %d) failed: %s", name, t, e)
        return self._render_meshes(mesh_list, cam_T_wc)

    def _render_pred_frame(
        self,
        seq: HumotoSeq,
        body_T_w_all: np.ndarray,       # (T, 65, 4, 4) unit-norm rotations
        object_poses: Dict[str, np.ndarray],  # {name: (T, 4, 4)}
        t: int,
        cam_T_wc: np.ndarray,
        obj_local_pcls: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Render one predicted frame by injecting scaled-back transforms.

        The model outputs unit-norm rotations. LBS skinning with the original
        inv_bind requires 0.01-scale rotations. This method rescales before
        injection.

        If *obj_local_pcls* is provided, renders 256 opaque PCL dots at the
        predicted object's current-frame world position.
        """
        mesh_list: List[trimesh.Trimesh] = []

        # Rescale body rotations from unit-norm back to original 0.01 scale
        body_scaled = body_T_w_all.copy()
        body_scaled[:, :, :3, :3] *= _HUMOTO_ROT_SCALE

        orig_body = seq.body_T_w
        seq.body_T_w = body_scaled
        try:
            body_dict = seq.skin_body(t)
            mesh_list.append(_color_mesh(body_dict["body"], BODY_COLOR))
            mesh_list.append(_color_mesh(body_dict["eyes"], EYES_COLOR))
        finally:
            seq.body_T_w = orig_body

        # Objects: inject and transform (objects don't need rescaling)
        orig_objects = {}
        added_names = []
        for name in object_poses:
            if name in seq.objects:
                orig_objects[name] = seq.objects[name]
            else:
                added_names.append(name)
            seq.objects[name] = object_poses[name]
        try:
            for i, name in enumerate(object_poses):
                try:
                    obj_mesh = seq.transform_object(name, t)
                    mesh_list.append(_color_mesh(obj_mesh, OBJ_COLORS[i % len(OBJ_COLORS)], alpha=128))
                except Exception as e:
                    logger.warning("humoto_viz: pred transform_object(%s, %d) failed: %s", name, t, e)
                    continue
                # Add 256 dots at current-frame predicted object world position
                if obj_local_pcls is not None and i < len(obj_local_pcls) and obj_local_pcls[i] is not None:
                    try:
                        pose_np = np.asarray(object_poses[name][t], dtype=np.float64)
                        mesh_list.append(_build_pcl_dots(
                            obj_local_pcls[i], pose_np, OBJ_COLORS[i % len(OBJ_COLORS)],
                        ))
                    except Exception as e:
                        logger.warning("humoto_viz: pred pcl dots(%s, %d) failed: %s", name, t, e)
        finally:
            for name, orig in orig_objects.items():
                seq.objects[name] = orig
            for name in added_names:
                seq.objects.pop(name, None)

        return self._render_meshes(mesh_list, cam_T_wc)

    def _render_ctx_frame(
        self,
        seq: HumotoSeq,
        ctx_body_T_w: np.ndarray,                 # (65, 4, 4) T-pose (pelvis-centered)
        object_names: List[str],
        ctx_obj_poses: np.ndarray,                # (n_obj, 4, 4) should be ~identity
        obj_local_pcls: Optional[List[np.ndarray]],  # per-object (256, 3) list
        cam_T_wc: np.ndarray,
    ) -> np.ndarray:
        """Render a CTX frame: T-pose body pelvis-centered + objects at identity/origin.

        Reuses the same injection mechanism as _render_pred_frame so rotation
        scales are handled correctly.
        """
        mesh_list: List[trimesh.Trimesh] = []

        # Rescale body rotations from unit-norm back to original 0.01 scale
        body_scaled = ctx_body_T_w.copy().astype(np.float64)
        body_scaled[:, :3, :3] *= _HUMOTO_ROT_SCALE

        # Inject body as a (T=1, 65, 4, 4) array so seq.skin_body(0) works
        orig_body = seq.body_T_w
        seq.body_T_w = body_scaled[None, ...]  # (1, 65, 4, 4)
        try:
            body_dict = seq.skin_body(0)
            mesh_list.append(_color_mesh(body_dict["body"], BODY_COLOR))
            mesh_list.append(_color_mesh(body_dict["eyes"], EYES_COLOR))
        finally:
            seq.body_T_w = orig_body

        # Objects: inject identity poses and render at origin
        orig_objects = {}
        added_names = []
        for i, name in enumerate(object_names):
            if i >= ctx_obj_poses.shape[0]:
                break
            if name in seq.objects:
                orig_objects[name] = seq.objects[name]
            else:
                added_names.append(name)
            # (T=1, 4, 4) pose
            pose_single = ctx_obj_poses[i].astype(np.float64)
            seq.objects[name] = pose_single[None, ...]
        try:
            for i, name in enumerate(object_names):
                if i >= ctx_obj_poses.shape[0]:
                    break
                try:
                    obj_mesh = seq.transform_object(name, 0)
                    mesh_list.append(_color_mesh(obj_mesh, OBJ_COLORS[i % len(OBJ_COLORS)], alpha=128))
                except Exception as e:
                    logger.warning("humoto_viz: ctx transform_object(%s) failed: %s", name, e)
                    continue
                # Add dots at origin (pose should be identity)
                if obj_local_pcls is not None and i < len(obj_local_pcls) and obj_local_pcls[i] is not None:
                    mesh_list.append(_build_pcl_dots(
                        obj_local_pcls[i], ctx_obj_poses[i].astype(np.float64),
                        OBJ_COLORS[i % len(OBJ_COLORS)],
                    ))
        finally:
            for name, orig in orig_objects.items():
                seq.objects[name] = orig
            for name in added_names:
                seq.objects.pop(name, None)

        return self._render_meshes(mesh_list, cam_T_wc)

    # ------------------------------------------------------------------
    # Data extraction from batch
    # ------------------------------------------------------------------
    def _extract_from_batch(
        self,
        batch: dict,
        bid: int,
    ) -> Tuple[
        torch.Tensor,     # body_trace: (T, n_body, 4, 4)
        torch.Tensor,     # obj_trace: (T, n_obj, 4, 4)
        torch.BoolTensor, # body_mask: (P,)
        torch.BoolTensor, # obj_mask: (P,)
        List[str],        # object_names
        str,              # seq_name
        int,              # n_steps (valid time steps)
    ]:
        """Extract body/object traces, masks, and metadata for one batch item."""
        trace = batch["trace"]          # (B, T, P, 4, 4)
        uid = batch["trace_uid"]        # (B, P)

        body_mask = AgentV11.get_type_mask(uid[bid], self.body_uids)   # (P,) bool
        obj_mask = AgentV11.get_type_mask(uid[bid], self.object_uids)  # (P,) bool

        body_trace = trace[bid, :, body_mask]   # (T, n_body, 4, 4)
        obj_trace = trace[bid, :, obj_mask]     # (T, n_obj, 4, 4)

        # Object names from extra
        extra = batch.get("extra", {})
        obj_names_list = extra.get("object_names", [None] * trace.shape[0])
        object_names = obj_names_list[bid] if obj_names_list[bid] is not None else []

        # Seq name
        seq_name = batch["seq_name"][bid]

        # Valid time steps: trace_mask (B, T, P), count along T for entity 0
        n_steps = int(batch["trace_mask"][bid][:, 0].sum().item())

        return body_trace, obj_trace, body_mask, obj_mask, object_names, seq_name, n_steps

    def _build_body_T_w(
        self,
        body_trace: torch.Tensor,   # (T, 65, 4, 4) or subset
        n_steps: int,
    ) -> np.ndarray:
        """Convert body trace tensor to numpy (T, 65, 4, 4) float64."""
        return body_trace[:n_steps].cpu().numpy().astype(np.float64)

    def _build_object_poses(
        self,
        obj_trace: torch.Tensor,    # (T, n_obj, 4, 4)
        object_names: List[str],
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Convert object trace tensor to {name: (T, 4, 4)} float64 dict."""
        poses = {}
        obj_np = obj_trace[:n_steps].cpu().numpy().astype(np.float64)  # (T, n_obj, 4, 4)
        for i, name in enumerate(object_names):
            if i < obj_np.shape[1]:
                poses[name] = obj_np[:, i]  # (T, 4, 4)
        return poses

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    def _run_inference(
        self,
        eval_model,
        batch: dict,
        bid: int,
        trace_known_mask: torch.Tensor,
        context_known_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run eval_model.inference for one batch item.

        Returns:
            (gen_context, gen_trace) -- full batch output from inference.
        """
        _local_pcl = batch.get("local_pcl")
        _local_pcl_mask = batch.get("local_pcl_mask")
        if _local_pcl is not None:
            _local_pcl = _local_pcl[bid:bid + 1]
            _local_pcl_mask = _local_pcl_mask[bid:bid + 1]

        gen_context, gen_trace = eval_model.inference(
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
            local_pcl=_local_pcl,
            local_pcl_mask=_local_pcl_mask,
            text=[batch["text"][bid]],
        )
        return gen_context, gen_trace

    # ------------------------------------------------------------------
    # Render + log a single visualization mode
    # ------------------------------------------------------------------
    def _viz_humoto_subroutine(
        self,
        seq: HumotoSeq,
        gen_trace: Optional[torch.Tensor],     # (T, P, 4, 4)
        gen_context: Optional[torch.Tensor],   # (C, P, 4, 4)
        body_mask: torch.BoolTensor,           # (P,)
        obj_mask: torch.BoolTensor,            # (P,)
        object_names: List[str],
        batch: dict,
        phase: str,
        bid: int,
        batch_idx: int,
        n_steps: int,
        viz_seen_step: int,
        video_viz_name: str = "",
        context_viz_name: str = "",
        caption: str = "",
        viz_context_flag: bool = True,
        use_raw_gt: bool = False,
        raw_frame_offset: int = 0,
        obj_local_pcls: Optional[List[np.ndarray]] = None,
    ):
        """Render and log a HUMOTO mesh-based visualization for one mode.

        Camera follows pelvis per-frame.  All frames use white background.

        Args:
            seq: Loaded HumotoSeq for mesh assets.
            gen_trace: Generated/GT trace (T, P, 4, 4) or None.
            gen_context: Generated/GT context (C, P, 4, 4) or None.
            body_mask: Boolean mask for body entities in P dim.
            obj_mask: Boolean mask for object entities in P dim.
            object_names: List of object name strings.
            batch: Full training batch.
            phase: "train" or "val".
            bid: Batch item index.
            batch_idx: Global batch index for logging.
            n_steps: Number of valid time steps.
            viz_seen_step: Boundary between seen/predicted frames (for caption).
            video_viz_name: Logger tag for the video.
            context_viz_name: Logger tag for the context image.
            caption: Text caption to overlay.
            viz_context_flag: Whether to visualize/log context.
            use_raw_gt: If True, render directly from seq (raw data, correct scale).
            raw_frame_offset: First raw frame index for raw GT mode.
            obj_local_pcls: Optional per-object (256, 3) numpy xyz lists for
                overlaying opaque PCL dots. Length matches obj_mask count.
        """
        if gen_trace is not None:
            # Build body + object arrays from trace
            body_trace = gen_trace[:n_steps, body_mask]    # (T, 65, 4, 4)
            obj_trace = gen_trace[:n_steps, obj_mask]      # (T, n_obj, 4, 4)

            body_T_w = self._build_body_T_w(body_trace, n_steps)
            obj_poses = self._build_object_poses(obj_trace, object_names, n_steps)

            frames = []
            for t in tqdm(range(n_steps), desc=f"humoto_viz:{video_viz_name}"):
                if use_raw_gt:
                    # Use raw seq data directly (correct rotation scale)
                    raw_t = min(raw_frame_offset + t, seq.T - 1)
                    pelvis_pos = seq.body_T_w[raw_t, 0, :3, 3]
                    cam_T_wc = self._compute_camera(pelvis_pos)
                    frame = self._render_gt_frame(
                        seq, raw_t, cam_T_wc, obj_local_pcls=obj_local_pcls,
                    )
                else:
                    # Predicted data: rescale rotations for skinning
                    pelvis_pos = body_T_w[t, 0, :3, 3]
                    cam_T_wc = self._compute_camera(pelvis_pos)
                    frame = self._render_pred_frame(
                        seq, body_T_w, obj_poses, t, cam_T_wc,
                        obj_local_pcls=obj_local_pcls,
                    )
                # Top bar indicator: green=seen, red=predicted (like HH viz)
                indicator_c = SEEN_INDICATOR if t < viz_seen_step else PRED_INDICATOR
                frame[:10, :, :3] = indicator_c[None, None, :]
                frames.append(frame)

            # Add caption if provided
            if caption:
                frames = put_caption_at_bottom(frames, caption)

            self.pl.logger.log_video(
                phase,
                video_viz_name,
                torch.from_numpy(np.stack(frames, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
            )

        if gen_context is not None and viz_context_flag:
            ctx_body = gen_context[:, body_mask]   # (C, 65, 4, 4)
            ctx_obj = gen_context[:, obj_mask]      # (C, n_obj, 4, 4)
            n_ctx = ctx_body.shape[0]

            ctx_body_np = ctx_body.cpu().numpy().astype(np.float64)
            ctx_obj_np = ctx_obj.cpu().numpy().astype(np.float64)

            ctx_frames = []
            for t in range(n_ctx):
                pelvis_pos = ctx_body_np[t, 0, :3, 3]
                cam_T_wc = self._compute_camera(pelvis_pos)
                frame = self._render_ctx_frame(
                    seq=seq,
                    ctx_body_T_w=ctx_body_np[t],           # (65, 4, 4)
                    object_names=object_names,
                    ctx_obj_poses=ctx_obj_np[t],           # (n_obj, 4, 4)
                    obj_local_pcls=obj_local_pcls,
                    cam_T_wc=cam_T_wc,
                )
                ctx_frames.append(frame)

            context_img = make_square_cat_image(ctx_frames)
            self.pl.logger.log_image(
                phase,
                context_viz_name,
                context_img[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0,
                self.pl.global_step,
                batch_idx,
            )

    # ------------------------------------------------------------------
    # Top-level routine
    # ------------------------------------------------------------------
    def humoto_viz_routine(self, batch, phase, batch_idx, eval_model):
        """Top-level visualization routine for HUMOTO.

        Finds a suitable batch item containing HUMOTO body joints, then
        renders: GT, half-future, 1-step, o2h, and h2o visualization modes.

        Args:
            batch (dict): Training batch.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference (``AgentV11``).
        """
        if not self.flag:
            return

        # Find first batch item with HUMOTO body joints
        bid = None
        for i in range(len(batch["trace"])):
            body_mask = AgentV11.get_type_mask(
                batch["trace_uid"][i], self.body_uids,
            )
            if body_mask.any():
                bid = i
                break

        if bid is None:
            logger.warning(
                "phase: %s, batch_idx: %d, no HUMOTO data in batch, skip viz",
                phase, batch_idx,
            )
            return

        # Extract data
        (body_trace, obj_trace, body_mask, obj_mask,
         object_names, seq_name, n_steps) = self._extract_from_batch(batch, bid)

        # Load HumotoSeq for mesh assets
        try:
            seq = self._load_seq(seq_name)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error("humoto_viz: failed to load seq %s: %s", seq_name, e)
            return

        text_caption = batch["text"][bid] if batch["text"][bid] else ""

        # Extract per-object local_pcl xyz from batch (256 points each)
        obj_local_pcls: Optional[List[np.ndarray]] = None
        local_pcl_tensor = batch.get("local_pcl")
        if local_pcl_tensor is not None:
            # (B, P, 256, 6) -> per-object list of (256, 3) numpy xyz
            pcl_bid = local_pcl_tensor[bid]                 # (P, 256, 6)
            obj_pcl = pcl_bid[obj_mask][..., :3]             # (n_obj, 256, 3)
            obj_local_pcls = [obj_pcl[i].detach().cpu().numpy().astype(np.float64)
                              for i in range(obj_pcl.shape[0])]

        # === GT (use raw seq data for correct mesh rendering) ===
        # Render min(n_steps, seq.T) frames from raw seq starting at frame 0
        gt_n = min(n_steps, seq.T)
        self._viz_humoto_subroutine(
            seq=seq,
            gen_trace=batch["trace"][bid, :gt_n],
            gen_context=batch["context"][bid],
            body_mask=body_mask,
            obj_mask=obj_mask,
            object_names=object_names,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            n_steps=gt_n,
            viz_seen_step=gt_n,
            video_viz_name=f"GT_humoto_{phase}",
            context_viz_name=f"GT_humoto_context_{phase}",
            caption=f"GT | {text_caption}",
            use_raw_gt=True,
            raw_frame_offset=0,
            obj_local_pcls=obj_local_pcls,
        )

        # === Half-future prediction ===
        trace_known = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ctx_known = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        trace_known[0, n_steps // 2:] = False

        gen_context, gen_trace = self._run_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_humoto_subroutine(
            seq=seq,
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            body_mask=body_mask,
            obj_mask=obj_mask,
            object_names=object_names,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            n_steps=n_steps,
            viz_seen_step=n_steps // 2,
            video_viz_name=f"FuturePred_humoto_seen_half_{phase}",
            context_viz_name=f"FuturePred_humoto_context_seen_half_{phase}",
            caption=f"half-future (seen={n_steps // 2}/{n_steps}) | {text_caption}",
            viz_context_flag=False,
            obj_local_pcls=obj_local_pcls,
        )

        # === 1-step prediction ===
        trace_known = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ctx_known = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        trace_known[:, 1:] = False

        gen_context, gen_trace = self._run_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_humoto_subroutine(
            seq=seq,
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            body_mask=body_mask,
            obj_mask=obj_mask,
            object_names=object_names,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            n_steps=n_steps,
            viz_seen_step=1,
            video_viz_name=f"FuturePred_humoto_seen_1_{phase}",
            context_viz_name=f"FuturePred_humoto_context_seen_1_{phase}",
            caption=f"1-step (seen=1/{n_steps}) | {text_caption}",
            viz_context_flag=False,
            obj_local_pcls=obj_local_pcls,
        )

        # === o2h: objects known, body unknown ===
        trace_known = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ctx_known = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        # Mask out body entities (unknown)
        trace_known[:, :, body_mask] = False

        gen_context, gen_trace = self._run_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_humoto_subroutine(
            seq=seq,
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            body_mask=body_mask,
            obj_mask=obj_mask,
            object_names=object_names,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            n_steps=n_steps,
            viz_seen_step=0,
            video_viz_name=f"o2h_humoto_{phase}",
            context_viz_name=f"o2h_humoto_context_{phase}",
            caption=f"o2h (objects known, body predicted) | {text_caption}",
            viz_context_flag=False,
            obj_local_pcls=obj_local_pcls,
        )

        # === h2o: body known, objects unknown ===
        trace_known = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ctx_known = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        # Mask out object entities (unknown)
        trace_known[:, :, obj_mask] = False

        gen_context, gen_trace = self._run_inference(
            eval_model, batch, bid, trace_known, ctx_known,
        )
        self._viz_humoto_subroutine(
            seq=seq,
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            body_mask=body_mask,
            obj_mask=obj_mask,
            object_names=object_names,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            n_steps=n_steps,
            viz_seen_step=0,
            video_viz_name=f"h2o_humoto_{phase}",
            context_viz_name=f"h2o_humoto_context_{phase}",
            caption=f"h2o (body known, objects predicted) | {text_caption}",
            viz_context_flag=False,
            obj_local_pcls=obj_local_pcls,
        )
