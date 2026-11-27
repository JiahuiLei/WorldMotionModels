"""Visualization helpers for OMOMO (human-object interaction, CHOIS protocol).

Renders SMPL-X body mesh + rest-frame object mesh + 256-point local PCL
overlay for OMOMO CHOIS-protocol windows. Mirrors the interface of
``humoto_viz.py`` so the training viz callback can drop it in.

Uses ``prepare_data.omomo.omomo_viser`` (``_SmplxFK``, ``_resolve_obj_mesh_path``,
etc.) as the FK + geometry backbone — same pattern as ``humoto_viz.py``
importing ``prepare_data.humoto.humoto_seq.HumotoSeq``.

This file is **additive-only**: it does not modify any existing viz_helpers
module.

Scope
-----------------------------------------------------------------
``OmomoVisualizer.draw_gt(batch)`` and
``OmomoVisualizer.draw_inference(batch, pred_x0)`` are fully implemented
and verified end-to-end on collated OMOMO batches. The training-time
``omomo_viz_routine`` renders GT and logs via ``pl.logger.log_video``.

The four inference-mode branches (half-future / 1-step / o2h / h2o) are
deferred. They depend on the ``scheduler_mode`` kwarg that is still to be added
to ``wmm/agent.py::inference(...)`` — that API does not exist yet on
this branch, and writing against a non-existent API would fail
silently at training time. **Inference-mode hooks land in a follow-up
change once `agent.inference()` signature stabilizes**. Until then, those branches log a
one-shot WARN and return.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# CRITICAL: before pyrender import.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender  # type: ignore
import torch
import trimesh
from PIL import Image

from prepare_data.omomo.omomo_viser import (
    _SmplxFK,
    _color_mesh,
    _make_ground_plane,
    _look_at_z_up,
    _camera_pose_diagonal,
    OmomoWindow,
    BODY_COLOR_RGBA,
    OBJ_COLOR_RGBA,
)

logger = logging.getLogger(__name__)

# Per-point sphere radius (object-local frame, meters).
_PCL_SPHERE_RADIUS = 0.015

# Height sanity range for rest-frame object meshes (meters).
# The UPPER bound (2.5 m) is the critical guard: it fails loud if someone
# accidentally hands us the raw-scan mesh from captured_objects/*.obj
# (~4.5 m extent, off-centre, wrong orientation) instead of the canonical
# rest_object_geo/<name>.ply.
#
# The LOWER bound catches empty / degenerate meshes. It is 0.1 m — wide
# enough to admit legitimate small-object categories that ARE physically
# small (smallbox 0.164 m, plasticbox 0.302 m, trashcan 0.337 m), narrow
# enough to flag a 0-extent or near-0-extent artefact. Note that this
# deviates from the dispatch (which proposed 0.3 m lower).
_OBJ_BBOX_Z_MIN = 0.1
_OBJ_BBOX_Z_MAX = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pcl_spheres(
    pcl_world_xyz: np.ndarray,
    pcl_rgb: Optional[np.ndarray] = None,
    radius: float = _PCL_SPHERE_RADIUS,
) -> trimesh.Trimesh:
    """Construct a single concatenated trimesh of uv_spheres at every point.

    Args:
        pcl_world_xyz: ``(N, 3)`` float64 world coordinates.
        pcl_rgb: optional ``(N, 3)`` uint8 per-point RGB. If None, uses
            ``OBJ_COLOR_RGBA``.
        radius: sphere radius in meters.

    Returns:
        Single merged ``trimesh.Trimesh`` with per-vertex colors.
    """
    pts = np.asarray(pcl_world_xyz, dtype=np.float64)
    N = pts.shape[0]

    # Build a minimal base sphere once, copy + translate for each point.
    base = trimesh.creation.uv_sphere(radius=float(radius), count=[4, 4])
    n_verts_per_sphere = base.vertices.shape[0]

    spheres: List[trimesh.Trimesh] = []
    for i, p in enumerate(pts):
        s = base.copy()
        s.apply_translation(p)
        spheres.append(s)

    if not spheres:
        # Return empty mesh
        return trimesh.Trimesh(process=False)

    merged = trimesh.util.concatenate(spheres)
    total_verts = merged.vertices.shape[0]

    # Assign per-vertex colors: repeat each point's color across its sphere verts.
    colors = np.zeros((total_verts, 4), dtype=np.uint8)
    for i in range(N):
        start = i * n_verts_per_sphere
        end = start + n_verts_per_sphere
        if pcl_rgb is not None:
            r, g, b = int(pcl_rgb[i, 0]), int(pcl_rgb[i, 1]), int(pcl_rgb[i, 2])
            colors[start:end] = np.array([r, g, b, 255], dtype=np.uint8)
        else:
            colors[start:end] = np.array(OBJ_COLOR_RGBA, dtype=np.uint8)

    return trimesh.Trimesh(
        vertices=merged.vertices,
        faces=merged.faces,
        vertex_colors=colors,
        process=False,
    )


def _assert_obj_height_sane(mesh_path: "Path | str", bbox: np.ndarray) -> None:
    """Assert the rest-frame object mesh height is within expected bounds.

    Args:
        mesh_path: path to the mesh file (for error messages).
        bbox: ``(2, 3)`` array of (min, max) corners.

    Raises:
        ValueError: if Z-extent is outside ``[_OBJ_BBOX_Z_MIN, _OBJ_BBOX_Z_MAX]``.
    """
    z_extent = float(bbox[1, 2] - bbox[0, 2])
    if not (_OBJ_BBOX_Z_MIN < z_extent < _OBJ_BBOX_Z_MAX):
        raise ValueError(
            f"OmomoViz: {mesh_path} bbox_z = {z_extent:.3f}m, "
            f"expected {_OBJ_BBOX_Z_MIN}..{_OBJ_BBOX_Z_MAX} — obj_mesh_path trap?"
        )


def _sample_pcl_from_mesh(
    mesh: trimesh.Trimesh,
    num_pts: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``num_pts`` vertices from ``mesh.vertices`` with optional RGB.

    Args:
        mesh: source trimesh.
        num_pts: number of points to sample.
        rng: optional ``np.random.Generator`` for reproducibility.

    Returns:
        ``(num_pts, 6)`` float32 array: xyz + rgb in [-0.5, 0.5].
        If the mesh has no vertex colors, RGB is filled with zeros.
    """
    if rng is None:
        rng = np.random.default_rng()

    verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V, 3)
    V = verts.shape[0]

    # Sample with replacement if mesh has fewer vertices than requested.
    idx = rng.integers(0, V, size=num_pts)
    xyz = verts[idx]  # (num_pts, 3)

    # RGB: use vertex colors if available, else zeros.
    # trimesh.visual can be ColorVisuals (has vertex_colors) or TextureVisuals
    # (no vertex_colors — PLYs without per-vertex color land here). Pyright
    # cannot discriminate these at static-analysis time; the getattr guard
    # plus the except fallback keeps us safe at runtime AND silences the
    # TextureVisuals branch warning.
    visual = getattr(mesh, "visual", None)
    vc_raw = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vc_raw is not None:
        try:
            vc = np.asarray(vc_raw, dtype=np.float32)  # (V, 4) uint8-based
            # vertex_colors are [0,255]; normalize to [0,1] then shift to [-0.5, 0.5]
            rgb_raw = vc[idx, :3] / 255.0 - 0.5  # (num_pts, 3)
        except (ValueError, IndexError, TypeError) as e:
            # Log the deviation — do not swallow silently. Malformed
            # per-vertex colors fall back to zero RGB (HUMOTO PCL-NaN
            # lesson: never leave RGB as NaN; zero is the safe default).
            logger.warning(
                "omomo_viz: vertex_colors malformed, zero-filling RGB "
                "(V=%d, err=%s)", V, e,
            )
            rgb_raw = np.zeros((num_pts, 3), dtype=np.float32)
    else:
        rgb_raw = np.zeros((num_pts, 3), dtype=np.float32)

    return np.concatenate([xyz, rgb_raw.astype(np.float32)], axis=1)  # (num_pts, 6)


def _render_scene_with_pcl(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    pcl_world_xyz: Optional[np.ndarray],
    pcl_rgb: Optional[np.ndarray],
    out_path: "Path | str",
    width: int = 960,
    height: int = 960,
    quality: int = 92,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
) -> None:
    """Render body + object + PCL spheres + ground to JPEG via pyrender EGL.

    Args:
        body_mesh: SMPL-X body trimesh (world frame).
        obj_mesh: rigid-transformed object trimesh (world frame).
        pcl_world_xyz: ``(N, 3)`` float64 world-frame PCL points.
        pcl_rgb: optional ``(N, 3)`` uint8 per-point RGB.
        out_path: output JPEG path.
        width, height: output resolution.
        quality: JPEG quality (0-95).
    """
    out_path = Path(out_path)

    # Compute combined bbox of body, object, and PCL for camera framing.
    b_bbox = body_mesh.bounds      # (2, 3)
    o_bbox = obj_mesh.bounds       # (2, 3)
    all_pts = [b_bbox[0], b_bbox[1], o_bbox[0], o_bbox[1]]
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        all_pts.append(pcl_world_xyz.min(axis=0))
        all_pts.append(pcl_world_xyz.max(axis=0))
    all_pts_np = np.stack(all_pts, axis=0)  # (K, 3)
    both_min = all_pts_np.min(axis=0)
    both_max = all_pts_np.max(axis=0)
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))

    # Aim at upper-body height, biased toward combined scene center.
    body_h = float(b_bbox[1, 2] - b_bbox[0, 2])
    target = np.array(
        [center[0], center[1], b_bbox[0, 2] + 0.55 * max(body_h, 0.2)],
        dtype=np.float64,
    )
    cam_pose = _camera_pose_diagonal(target, dist=max(3.2, 0.9 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )

    # Ground plane
    scene.add(
        pyrender.Mesh.from_trimesh(_make_ground_plane(half_size=4.0, z=0.0), smooth=False)
    )

    # Body mesh
    scene.add(
        pyrender.Mesh.from_trimesh(_color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False)
    )

    # Object mesh (opaque)
    scene.add(
        pyrender.Mesh.from_trimesh(_color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False)
    )

    # PCL spheres
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        pcl_sphere_mesh = _build_pcl_spheres(
            pcl_world_xyz, pcl_rgb, radius=pcl_radius,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(pcl_sphere_mesh, smooth=False)
        )

    # Camera
    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)

    # Lights: key + fill + rim directional
    key = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=4.0)
    scene.add(
        key,
        pose=_look_at_z_up(eye=target + np.array([2.0, -2.0, 4.0]), target=target),
    )
    fill = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(
        fill,
        pose=_look_at_z_up(eye=target + np.array([-3.0, 2.0, 2.0]), target=target),
    )
    rim = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(
        rim,
        pose=_look_at_z_up(eye=target + np.array([0.0, 3.0, 1.5]), target=target),
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).convert("RGB").save(str(out_path), format="JPEG", quality=quality)


def _extract_scalar_field(extra: Dict, key: str, bid: int, default=None):
    """Pull ``extra[key]`` either as a per-item list or a batch-wide scalar.

    OMOMO collates ``obj_category``, ``betas``, ``gender`` into
    ``batch["extra"]`` as dict-of-lists per CanonicalDatasetOutput. When
    the dataloader uses a naive default collate they may become a flat
    scalar shared across the batch — handle both.

    Returns ``default`` if ``key`` is absent.
    """
    if key not in extra:
        return default
    v = extra[key]
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        return v[bid] if bid < len(v) else v[-1]
    # Tensors / arrays: per-item first-dim indexing when possible.
    if isinstance(v, (np.ndarray, torch.Tensor)):
        if v.ndim >= 1 and v.shape[0] > bid:
            return v[bid]
        return v
    return v


def _build_training_scene(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    width: int,
    height: int,
) -> pyrender.Scene:
    """Build the pyrender scene + lights + camera for training-time viz.

    Shared by the cached-renderer path (``_render_frame_given_renderer``).
    """
    b_bbox = body_mesh.bounds
    o_bbox = obj_mesh.bounds
    both_min = np.minimum(b_bbox[0], o_bbox[0])
    both_max = np.maximum(b_bbox[1], o_bbox[1])
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))
    body_h = float(b_bbox[1, 2] - b_bbox[0, 2])
    target = np.array(
        [center[0], center[1], b_bbox[0, 2] + 0.55 * max(body_h, 0.2)],
        dtype=np.float64,
    )
    cam_pose = _camera_pose_diagonal(target, dist=max(3.2, 0.9 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(
        _make_ground_plane(half_size=4.0, z=0.0), smooth=False,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False,
    ))
    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)
    for offset, inten in (
        (np.array([2.0, -2.0, 4.0]), 4.0),
        (np.array([-3.0, 2.0, 2.0]), 2.0),
        (np.array([0.0, 3.0, 1.5]), 2.0),
    ):
        light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=inten)
        scene.add(light, pose=_look_at_z_up(eye=target + offset, target=target))
    return scene


def _render_frame_given_renderer(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    renderer: pyrender.OffscreenRenderer,
    width: int,
    height: int,
) -> np.ndarray:
    """Render into an EXISTING renderer; caller owns its lifecycle."""
    scene = _build_training_scene(body_mesh, obj_mesh, width, height)
    color, _ = renderer.render(scene)
    return color[..., :3].astype(np.uint8)


# ---------------------------------------------------------------------------
# OmomoVisualizer — training-time callback class
# ---------------------------------------------------------------------------

class OmomoVisualizer:
    """Mesh-based visualizer for OMOMO human-object interaction sequences.

    Uses SMPL-X forward kinematics for the body and rigid transforms for
    the single object. Mirrors the interface of ``HumotoVisualizer``.

    Scope:
        * ``draw_gt(batch)`` is fully implemented — renders a per-frame
          GT video from the batch's trace (no model inference required)
          and logs it via ``pl_module.logger.log_video``.
        * ``draw_inference(batch, pred_x0)`` is implemented for the
          minimal "GT-substituted-for-pred" path (useful when a caller
          passes a prediction tensor but the training harness hasn't
          yet wired the wmm Agent's OMOMO inference path).
        * ``omomo_viz_routine(batch, phase, batch_idx, eval_model)``
          renders GT only for now. The half-future / 1-step / o2h / h2o
          multi-mode branches log a WARNING and skip, because the wmm
          Agent's OMOMO ``inference()`` signature is not stable in
          this version. They will be wired when the
          Agent API stabilizes.

    Args:
        pl_module: Parent PyTorch Lightning module (``PLModel``).
        omomo_cfg (dict): YAML block from ``model.omomo``. Optional keys:
            ``smplx_dir`` (default ``data_wmm/omomo/smplx``),
            ``geo_dir`` (default ``data_wmm/omomo/rest_object_geo``),
            ``width`` (default 512), ``height`` (default 512),
            ``num_pcl_points`` (default 256),
            ``pcl_seed`` (default 0).
    """

    def __init__(self, pl_module, omomo_cfg):
        self.pl = pl_module
        # Defensive defaults initialized BEFORE the cfg-empty early-return,
        # so partial instances still have a consistent attribute surface
        # (review-04 NOTE #2).
        self._inference_defer_logged = False
        if not omomo_cfg or len(omomo_cfg) == 0:
            self.flag = False
            return
        self.flag = True
        self.smplx_dir = Path(omomo_cfg.get("smplx_dir", "data_wmm/omomo/smplx"))
        self.geo_dir = Path(omomo_cfg.get("geo_dir", "data_wmm/omomo/rest_object_geo"))
        # data_root is needed to read betas/gender from the source npz
        # when the batch does not carry them in extras. Defaults to
        # geo_dir.parent = data_wmm/omomo/ which contains test_npz/ +
        # train_npz/.
        self._data_root = Path(omomo_cfg.get("data_root", self.geo_dir.parent))
        self.omomo_viz_cfg = self.pl.cfg.get("visualize", {}).get("omomo", {})
        # Resolution: visualize.omomo.{height,width} preferred; fall back
        # to omomo_cfg.{height,width}; final default 512.
        self._H = int(
            self.omomo_viz_cfg.get("height", omomo_cfg.get("height", 512))
        )
        self._W = int(
            self.omomo_viz_cfg.get("width", omomo_cfg.get("width", 512))
        )
        self._num_pcl_points = int(omomo_cfg.get("num_pcl_points", 256))
        self._pcl_seed = int(omomo_cfg.get("pcl_seed", 0))
        self._fk = _SmplxFK(self.smplx_dir)
        # Cache a single pyrender offscreen renderer for the training-time
        # draw_gt path — avoids 120× EGL init/teardown per batch.
        # Protected by a lock because the training callback may be
        # invoked from non-main threads in some PL configurations.
        self._render_lock = __import__("threading").Lock()
        self._cached_renderer: Optional[pyrender.OffscreenRenderer] = None

    # ------------------------------------------------------------------
    # Public helpers for inspection / ad-hoc use
    # ------------------------------------------------------------------

    def draw_gt(self, batch: Dict, bid: int = 0) -> np.ndarray:
        """Render a (T_actual, H, W, 3) uint8 GT video from batch[bid].

        Expects the wmm OMOMO batch layout produced by ``OmomoDataset``:
        ``trace[bid]`` contains 22 body joints (group 0) + 1 object
        (group 1), each carrying SMPL-X axis-angle rotation embedded in
        its SE(3) rotation block (pelvis SE(3) also carries root trans).
        Body joint axis-angles are reconstructed by matrix-logging each
        rotation column; SMPL-X FK then produces the per-frame mesh.

        Args:
            batch: collated training batch dict. Expected keys:
                ``trace`` ``(B, T, P, 4, 4)``, ``trace_mask`` ``(B, T, P)``,
                ``extra`` ``{obj_category, entry, ...}``. Exact layout
                enforced at runtime; deviations raise loudly.
            bid: batch item index.

        Returns:
            ``(T_actual, H, W, 3)`` uint8 frames.
        """
        if not self.flag:
            raise RuntimeError("OmomoVisualizer is disabled (empty omomo_cfg)")

        body_verts, body_faces, obj_R, obj_t, actual_len, obj_category = \
            self._fk_from_batch(batch, bid)

        rest_mesh, _, _ = self._load_rest(obj_category)
        V_rest = np.asarray(rest_mesh.vertices, dtype=np.float64)
        obj_faces = np.asarray(rest_mesh.faces)

        frames: List[np.ndarray] = []
        for t in range(actual_len):
            body_mesh = trimesh.Trimesh(
                vertices=body_verts[t].astype(np.float64),
                faces=body_faces, process=False,
            )
            V_w = V_rest @ obj_R[t].T + obj_t[t][None, :]
            obj_mesh = trimesh.Trimesh(
                vertices=V_w, faces=obj_faces, process=False,
            )
            # No PCL overlay in training-time viz (keeps frame budget low);
            # the standalone render_omomo_window_jpeg path is the
            # PCL-overlay gate used for VLM verification. We reuse
            # the instance-cached renderer across frames (review-02 #5 —
            # avoids 120x EGL init/teardown).
            frame = self._render_frame_with_cached_renderer(body_mesh, obj_mesh)
            frames.append(frame)
        return np.stack(frames, axis=0)  # (T_actual, H, W, 3)

    def _render_frame_with_cached_renderer(
        self, body_mesh: trimesh.Trimesh, obj_mesh: trimesh.Trimesh,
    ) -> np.ndarray:
        """Thread-safe render using a single OffscreenRenderer per instance."""
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            return _render_frame_given_renderer(
                body_mesh, obj_mesh, self._cached_renderer,
                self._W, self._H,
            )

    def draw_inference(
        self,
        batch: Dict,
        pred_x0: Optional[torch.Tensor],
        bid: int = 0,
    ) -> np.ndarray:
        """Render predicted trace the same way ``draw_gt`` renders GT.

        If ``pred_x0`` is None, falls back to GT (useful for dry-run).
        Expects ``pred_x0`` shape matching ``batch['trace']``. The helper
        swaps the selected batch item's trace for the predicted one and
        calls ``draw_gt`` internally.
        """
        if not self.flag:
            raise RuntimeError("OmomoVisualizer is disabled (empty omomo_cfg)")
        if pred_x0 is None:
            return self.draw_gt(batch, bid=bid)

        # Shallow copy batch and replace the trace
        sub_batch = dict(batch)
        # pred_x0 must match the collated layout (B, T, P, 4, 4) — the
        # same contract draw_gt enforces via _fk_from_batch. Use
        # .clone() so we don't alias the caller's tensor.
        sub_batch["trace"] = pred_x0.detach().clone()
        return self.draw_gt(sub_batch, bid=bid)

    # ------------------------------------------------------------------
    # Internal helpers — batch extraction + FK
    # ------------------------------------------------------------------

    def _fk_from_batch(
        self,
        batch: Dict,
        bid: int,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]":
        """Extract SMPL-X FK + object pose from a wmm OMOMO batch item.

        The wmm OMOMO trace layout (see ``OmomoDataset``) is P = 23:
        joints 0..21 = SMPL-X body axis-angles encoded as SE(3)
        rotations (pelvis joint carries root translation in its
        translation block), joint 22 = rigid object SE(3).

        This helper recovers axis-angle from each body SE(3)'s rotation
        matrix (by ``scipy.spatial.transform.Rotation.as_rotvec``, or
        numpy matrix-log fallback), then runs SMPL-X FK using the
        batch's ``betas`` + ``gender``. The object pose is taken
        directly from joint 22.

        Expected batch keys:
            * ``trace`` — ``(B, T, P, 4, 4)`` float tensor (collated
              layout; see ``CanonicalDatasetOutput.collate``).
            * ``trace_mask`` — ``(B, T, P)`` bool tensor (True if frame
              is valid). Non-negotiable; deviations raise loudly.
            * ``extra["betas"]`` — ``(16,)`` per item OR batch-wide
              fallback. If absent: raise.
            * ``extra["gender"]`` — str per item OR batch-wide.

        Returns:
            ``(body_verts (T_actual,10475,3) float32,
               body_faces (F,3) int32,
               obj_R (T_actual,3,3) float64,
               obj_t (T_actual,3) float64,
               actual_len int,
               obj_category str)``.
        """
        # CONTRACT (matches CanonicalDatasetOutput.collate +
        # humoto_viz.py:415,432):
        #   batch["trace"]      = (B, T, P, 4, 4) — time-major, entity-minor
        #   batch["trace_mask"] = (B, T, P)       — bool
        # We trust this exact shape. Any deviation raises loudly, not
        # silently remapped.
        trace = batch["trace"]
        tmask = batch.get("trace_mask")
        if trace.ndim != 5:
            raise ValueError(
                f"OmomoVisualizer: expected trace ndim=5 (B,T,P,4,4), got {trace.ndim}"
            )
        if tmask is None or tmask.ndim != 3:
            raise ValueError(
                "OmomoVisualizer: batch['trace_mask'] must be (B,T,P) bool "
                "— required to determine actual_len from the collated batch"
            )
        if trace.shape[:3] != tmask.shape[:3]:
            raise ValueError(
                f"OmomoVisualizer: trace.shape[:3]={tuple(trace.shape[:3])} "
                f"!= trace_mask.shape[:3]={tuple(tmask.shape[:3])}"
            )

        # bTrace: (T, P, 4, 4); bMask: (T, P)
        bTrace = trace[bid]
        bMask = tmask[bid]
        T, P = bTrace.shape[0], bTrace.shape[1]
        if P < 23:
            raise ValueError(
                f"OmomoVisualizer: OMOMO contract expects P>=23 "
                f"(22 body + 1 object), got P={P}"
            )

        # actual_len = # time steps where pelvis (entity 0) is valid.
        # bMask[:, 0] is the time-axis validity for entity 0.
        actual_len = int(bMask[:, 0].sum().item())
        actual_len = max(1, min(actual_len, T))

        bTrace_np = bTrace.detach().cpu().numpy().astype(np.float64)

        # Body joints 0..21 rotations along time.
        # body_rot: (T_a, 22, 3, 3) in time-major layout.
        body_rot = bTrace_np[:actual_len, :22, :3, :3]
        # Rodrigues inverse (matrix -> axis-angle). scipy's as_rotvec is
        # the numerically stable path; the closed-form fallback is for
        # envs lacking scipy.
        try:
            from scipy.spatial.transform import Rotation as _R
            flat = body_rot.reshape(-1, 3, 3)
            aa_flat = _R.from_matrix(flat).as_rotvec()      # (T_a*22, 3)
            aa = aa_flat.reshape(actual_len, 22, 3)          # (T_a, 22, 3)
        except ImportError:
            # Closed-form: angle = acos((trace-1)/2); axis from skew.
            tr = np.einsum("...ii", body_rot)                # (T_a, 22)
            angle = np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
            sin_a = np.sin(angle)
            ax = body_rot[..., 2, 1] - body_rot[..., 1, 2]
            ay = body_rot[..., 0, 2] - body_rot[..., 2, 0]
            az = body_rot[..., 1, 0] - body_rot[..., 0, 1]
            axis = np.stack([ax, ay, az], axis=-1)           # (T_a, 22, 3)
            denom = 2.0 * np.where(sin_a > 1.0e-9, sin_a, 1.0)
            axis = axis / denom[..., None]
            aa = axis * angle[..., None]

        root_aa = aa[:, 0, :]                                 # (T_a, 3)
        body_aa = aa[:, 1:22, :]                              # (T_a, 21, 3)
        # Pelvis world translation is carried in joint 0's SE(3)[:3, 3].
        root_trans = bTrace_np[:actual_len, 0, :3, 3]         # (T_a, 3)

        # Object pose at entity index 22.
        obj_R = bTrace_np[:actual_len, 22, :3, :3]            # (T_a, 3, 3)
        obj_t = bTrace_np[:actual_len, 22, :3, 3]             # (T_a, 3)

        # Resolve betas + gender + obj_category. OmomoDataset emits
        # extra = {obj_category, obj_mesh_path, waypoint_indices,
        # actual_len, entry}. betas + gender are NOT in extras — we
        # read them from the source npz using `entry`. This keeps
        # dataset contract untouched.
        extra = batch.get("extra", {})
        obj_category = _extract_scalar_field(
            extra, "obj_category", bid, default=None,
        )
        if obj_category is None:
            raise KeyError(
                "OmomoVisualizer: batch['extra']['obj_category'] is required "
                "to locate rest_object_geo/<cat>.ply"
            )
        betas_np, gender = self._resolve_betas_gender(extra, bid)

        # SMPL-X FK (CPU).
        verts, faces = self._fk.fk_sequence(
            root_trans.astype(np.float32),
            root_aa.astype(np.float32),
            body_aa.astype(np.float32),
            betas_np, gender,
        )
        return verts, faces, obj_R, obj_t, actual_len, str(obj_category)

    def _resolve_betas_gender(
        self, extra: Dict, bid: int,
    ) -> "tuple[np.ndarray, str]":
        """Return ``(betas (16,) float32, gender str)`` for the batch item.

        Looks first at ``extra['betas']`` / ``extra['gender']`` (in case
        a future dataset extension populates them). If absent, loads
        from ``self._npz_dir_for(entry)/<entry>.npz`` using the entry
        stem stored in ``extra['entry']`` (populated by
        ``OmomoDataset.__getitem_inner__``).

        Fails loud if neither path works.
        """
        betas = _extract_scalar_field(extra, "betas", bid, default=None)
        gender = _extract_scalar_field(extra, "gender", bid, default=None)
        if betas is not None and gender is not None:
            return (
                np.asarray(betas, dtype=np.float32).reshape(-1),
                str(gender),
            )

        # Fallback: load from the source npz via `entry`.
        entry = _extract_scalar_field(extra, "entry", bid, default=None)
        if entry is None:
            raise KeyError(
                "OmomoVisualizer._resolve_betas_gender: extras lack both "
                "('betas','gender') AND 'entry' — cannot locate source npz. "
                "OmomoDataset.__getitem_inner__ must populate at least 'entry'."
            )
        npz_path = self._find_npz_for_entry(str(entry))
        with np.load(str(npz_path), allow_pickle=True) as d:
            betas_np = np.asarray(d["betas"], dtype=np.float32).reshape(-1)
            gender_str = str(d["gender"].item())
        return betas_np, gender_str

    def _find_npz_for_entry(self, entry: str) -> Path:
        """Return absolute path to the npz for a given entry stem.

        Searches ``<data_root>/{test,train}_npz/<entry>.npz`` in that
        order. ``data_root`` is resolved from ``omomo_cfg['data_root']``
        or inferred from ``geo_dir.parent``. Fails loud if not found.
        """
        data_root = getattr(self, "_data_root", None)
        if data_root is None:
            data_root = self.geo_dir.parent
        for split in ("test", "train"):
            cand = Path(data_root) / f"{split}_npz" / f"{entry}.npz"
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"OmomoVisualizer: cannot find npz for entry {entry!r} under "
            f"{data_root}/(test|train)_npz/"
        )

    def _load_rest(
        self, obj_category: str,
    ) -> "tuple[trimesh.Trimesh, Path, float]":
        """Load the rest-frame PLY for an object category, with height check."""
        geo_path = self.geo_dir / f"{obj_category}.ply"
        if not geo_path.exists():
            raise FileNotFoundError(
                f"OmomoVisualizer: rest-frame geometry missing: {geo_path}"
            )
        rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
        if not isinstance(rest_mesh, trimesh.Trimesh):
            raise ValueError(
                f"OmomoVisualizer: expected single mesh from {geo_path}, "
                f"got {type(rest_mesh).__name__}"
            )
        _assert_obj_height_sane(geo_path, rest_mesh.bounds)
        bbox_z_m = float(rest_mesh.bounds[1, 2] - rest_mesh.bounds[0, 2])
        return rest_mesh, geo_path, bbox_z_m

    # ------------------------------------------------------------------
    # Training-time callback (partial — GT only)
    # ------------------------------------------------------------------

    def omomo_viz_routine(
        self,
        batch: Dict,
        phase: str,
        batch_idx: int,
        eval_model,
    ) -> None:
        """Training-time visualization callback.

        Scope:
          * GT video rendered and logged via
            ``pl_module.logger.log_video``.
          * Half-future / 1-step / o2h / h2o inference modes log a
            WARNING and skip (wmm Agent OMOMO inference not yet wired).

        Args:
            batch: training batch dict.
            phase: ``"train"`` or ``"val"``.
            batch_idx: batch index.
            eval_model: callable for running model inference.
        """
        if not self.flag:
            return
        # eval_model is accepted for interface parity with HumotoVisualizer;
        # it is not invoked here because the wmm Agent's OMOMO inference
        # path is not stable in master. When it is, add half-future /
        # 1-step / o2h / h2o branches below.
        _ = eval_model
        # --- Prefer window_idx=0 batch item for cleaner viz ---------------
        # Train-split
        # sliding windows sample mid-action frames at arbitrary
        # start_t_idx > 0; rendering those as "GT" produces visually
        # surprising (though physically correct) mp4s with objects
        # already tilted at frame 0. For training-time viz only, pick
        # a batch item whose `extra["window_idx"] == 0` when present —
        # this makes wandb train-viz look like test-viz. Training
        # itself uses all windows (no change to loss / data pipeline).
        bid = 0
        try:
            extra = batch.get("extra", {}) or {}
            window_idx_list = extra.get("window_idx")
            if window_idx_list is not None:
                for i, wi in enumerate(window_idx_list):
                    wi_v = int(wi) if wi is not None else -1
                    if wi_v == 0:
                        bid = i
                        break
        except (TypeError, ValueError, KeyError):
            bid = 0
        # NO blanket except around draw_gt. If the batch contract is wrong,
        # that MUST surface (review-02 #14). The only exception we contain
        # is the logger call itself — a missing `log_video` attribute
        # on the Lightning logger is a wiring concern, not a viz bug.
        frames = self.draw_gt(batch, bid=bid)
        # frames: (T_a, H, W, 3) uint8 -> (T_a, 3, H, W) float32 in [0, 1]
        video = (
            torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        )
        log_video = getattr(self.pl.logger, "log_video", None)
        if log_video is None:
            logger.warning(
                "omomo_viz_routine: pl.logger has no log_video attribute "
                "— frames rendered but not logged (phase=%s batch=%d)",
                phase, batch_idx,
            )
            return
        log_video(
            phase,
            f"GT_omomo_{phase}",
            video,
            self.pl.global_step,
            batch_idx,
        )
        # The inference-mode branches (half-future / 1-step / o2h / h2o)
        # are deferred. Log the notice ONCE per visualizer instance to
        # avoid flooding long training runs.
        if not self._inference_defer_logged:
            logger.warning(
                "omomo_viz_routine: GT video logged; inference modes "
                "(half-future / 1-step / o2h / h2o) are NOT YET WIRED — "
                "wmm Agent OMOMO inference path must stabilize first. "
                "This notice is logged ONCE; "
                "subsequent calls are silent. phase=%s batch=%d",
                phase, batch_idx,
            )
            self._inference_defer_logged = True


# ---------------------------------------------------------------------------
# Standalone entry point: render_omomo_window_jpeg
# ---------------------------------------------------------------------------

def render_omomo_window_jpeg(
    npz_path: "str | os.PathLike",
    frame: int,
    out_path: "str | os.PathLike",
    smplx_dir: "str | os.PathLike" = "data_wmm/omomo/smplx",
    geo_dir: "str | os.PathLike" = "data_wmm/omomo/rest_object_geo",
    width: int = 960,
    height: int = 960,
    include_pcl: bool = True,
    num_pcl_points: int = 256,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    pcl_seed: int = 0,
    fk_cache: Optional[_SmplxFK] = None,
    window_cache: Optional[OmomoWindow] = None,
) -> Dict[str, object]:
    """Render one OMOMO window frame to JPEG with optional PCL overlay.

    Reads an OMOMO Q12 .npz, runs SMPL-X FK, loads the CANONICAL rest-frame
    object mesh from ``geo_dir/<category>.ply`` (NOT the obj_mesh_path raw
    scan), builds a scene with ground plane, body, object, and optionally
    256 PCL spheres, and writes a JPEG.

    The rest-frame geo path is resolved by stripping the ``_cleaned_simplified.obj``
    suffix from the npz's ``obj_mesh_path`` to recover ``<category>``, then
    loading ``<geo_dir>/<category>.ply``. This path is printed + returned in
    the metadata so the caller can verify it.

    Args:
        npz_path: OMOMO Q12 .npz window path.
        frame: frame index in [0, T-1]. **Silently clamped** to
            ``actual_len - 1`` if the caller passes a frame beyond the
            valid range. Callers that want to reject out-of-range frames
            MUST check ``actual_len`` themselves BEFORE calling
            (``render_gate.py`` does this). The clamp is defensive only
            and never masks a real "teleport-to-origin" zero-pad.
        out_path: output JPEG path.
        smplx_dir: SMPL-X models (SMPLX_MALE/FEMALE/NEUTRAL.npz).
        geo_dir: directory with ``<category>.ply`` rest-frame geometry.
        width, height: output resolution.
        include_pcl: overlay 256 PCL spheres transformed by frame-t pose.
        num_pcl_points: points to sample (default 256, matches OmomoDataset).
        pcl_radius: sphere radius (meters).
        pcl_seed: RNG seed for reproducible PCL sampling.
        fk_cache, window_cache: reuse for multi-frame rendering.

    Returns:
        dict with keys: ``out_path``, ``geo_path`` (resolved rest PLY),
        ``obj_category``, ``frame``, ``actual_len``, ``seq_name``, ``text``,
        ``bbox_z_m`` (object height sanity).
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    geo_dir = Path(geo_dir)
    smplx_dir = Path(smplx_dir)

    # ------------------------------------------------------------------
    # 1. Load npz fields
    # ------------------------------------------------------------------
    d = np.load(str(npz_path), allow_pickle=True)
    actual_len = int(d["actual_len"].item())
    seq_name = str(d["seq_name"].item())
    text = str(d["text"].item())
    obj_mesh_path_raw = str(d["obj_mesh_path"].item())
    betas = np.asarray(d["betas"], dtype=np.float32).reshape(-1)  # (16,)
    gender = str(d["gender"].item())
    root_trans = np.asarray(d["root_trans"], dtype=np.float32)      # (T, 3)
    root_orient_aa = np.asarray(d["root_orient_aa"], dtype=np.float32)  # (T, 3)
    pose_body_aa = np.asarray(d["pose_body_aa"], dtype=np.float32)   # (T, 21, 3)
    obj_trans = np.asarray(d["obj_trans"], dtype=np.float32)         # (T, 3)
    obj_rot = np.asarray(d["obj_rot"], dtype=np.float32)             # (T, 3, 3)

    # ------------------------------------------------------------------
    # 2. Frame clamping
    # ------------------------------------------------------------------
    frame = min(int(frame), actual_len - 1)

    # ------------------------------------------------------------------
    # 3. Category resolution: strip suffix from obj_mesh_path basename
    # ------------------------------------------------------------------
    base = Path(obj_mesh_path_raw).name
    SUFFIXES = (
        "_cleaned_simplified.obj",
        "_cleaned_simplified.ply",
        "_simplified.obj",
        ".obj",
        ".ply",
    )
    category = None
    for suf in SUFFIXES:
        if base.endswith(suf):
            category = base[: -len(suf)]
            break
    if category is None:
        raise ValueError(
            f"Cannot strip object suffix from {obj_mesh_path_raw!r}; "
            f"expected one of {SUFFIXES}"
        )

    geo_path = geo_dir / f"{category}.ply"
    if not geo_path.exists():
        raise FileNotFoundError(
            f"Rest-frame geometry not found: {geo_path}. "
            f"Cannot fall back to raw obj_mesh_path — raw scan is not in "
            f"canonical frame. geo_dir={geo_dir}"
        )

    # ------------------------------------------------------------------
    # 4. Load rest-frame PLY and sanity-check height
    # ------------------------------------------------------------------
    rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
    if not isinstance(rest_mesh, trimesh.Trimesh):
        raise ValueError(
            f"Expected single mesh from {geo_path}, got {type(rest_mesh).__name__}"
        )
    _assert_obj_height_sane(geo_path, rest_mesh.bounds)
    bbox_z_m = float(rest_mesh.bounds[1, 2] - rest_mesh.bounds[0, 2])

    # ------------------------------------------------------------------
    # 5. SMPL-X FK (reuse window_cache when available to skip re-FK)
    # ------------------------------------------------------------------
    if (
        window_cache is not None
        and getattr(window_cache, "name", None) == npz_path.stem
    ):
        verts = np.asarray(window_cache.body_verts, dtype=np.float32)
        faces = np.asarray(window_cache.body_faces, dtype=np.int32)
    else:
        fk = fk_cache if fk_cache is not None else _SmplxFK(smplx_dir)
        verts, faces = fk.fk_sequence(
            root_trans, root_orient_aa, pose_body_aa, betas, gender
        )

    # ------------------------------------------------------------------
    # 6. Build frame-t meshes
    # ------------------------------------------------------------------
    body_mesh = trimesh.Trimesh(
        vertices=verts[frame].astype(np.float64),
        faces=faces,
        process=False,
    )

    # Object: V_world = V_rest @ R.T + t (no scale — protocol fix #2)
    R = obj_rot[frame].astype(np.float64)     # (3, 3)
    t_obj = obj_trans[frame].astype(np.float64)  # (3,)
    V_rest = np.asarray(rest_mesh.vertices, dtype=np.float64)  # (V, 3)
    V_w = V_rest @ R.T + t_obj[None, :]       # (V, 3)
    obj_mesh = trimesh.Trimesh(
        vertices=V_w,
        faces=np.asarray(rest_mesh.faces),
        process=False,
    )

    # ------------------------------------------------------------------
    # 7. PCL overlay
    # ------------------------------------------------------------------
    pcl_world_xyz: Optional[np.ndarray] = None
    pcl_rgb_u8: Optional[np.ndarray] = None

    if include_pcl:
        rng = np.random.default_rng(pcl_seed)
        pcl_data = _sample_pcl_from_mesh(rest_mesh, num_pcl_points, rng=rng)
        # pcl_data: (N, 6) float32 — xyz + rgb in [-0.5, 0.5]
        pcl_xyz_rest = pcl_data[:, :3].astype(np.float64)   # (N, 3)
        pcl_rgb_centered = pcl_data[:, 3:]                  # (N, 3) float32

        # Transform PCL to world frame
        pcl_world_xyz = pcl_xyz_rest @ R.T + t_obj[None, :]  # (N, 3)

        # Convert RGB from [-0.5, 0.5] back to uint8
        pcl_rgb_u8 = np.clip(
            (pcl_rgb_centered + 0.5) * 255.0, 0, 255
        ).astype(np.uint8)  # (N, 3)

    # ------------------------------------------------------------------
    # 8. Render
    # ------------------------------------------------------------------
    _render_scene_with_pcl(
        body_mesh=body_mesh,
        obj_mesh=obj_mesh,
        pcl_world_xyz=pcl_world_xyz,
        pcl_rgb=pcl_rgb_u8,
        out_path=out_path,
        width=width,
        height=height,
        pcl_radius=pcl_radius,
    )

    logger.info(
        "omomo_viz: rendered %s f%d -> %s (geo=%s, bbox_z=%.3fm)",
        seq_name, frame, out_path, geo_path.name, bbox_z_m,
    )

    return {
        "out_path": str(out_path),
        "geo_path": str(geo_path),
        "obj_category": category,
        "frame": frame,
        "actual_len": actual_len,
        "seq_name": seq_name,
        "text": text,
        "bbox_z_m": bbox_z_m,
    }


__all__ = [
    "OmomoVisualizer",
    "render_omomo_window_jpeg",
]
