"""Viser hand-object viewer — scene build, V972 guard, render_sample API.

implementation for the handobject eval viz gate ( 20260419-1657-a308104e).

Public surface
--------------
- `render_sample(sample, out_dir, views, ...)` — 's programmatic entry point.
  Launches a viser server, opens a headless chromium client via playwright+xvfb,
  renders each requested view using `client.camera.get_render(H, W)`, writes one
  PNG + a JSON sidecar per view. Cleanup is in finally-blocks.
- `_safe_add_point_cloud(...)` — render-boundary V972 guard. Only enforces the
  world-frame check when `obj_pose` AND `mesh_extent` are both passed. Opt-in.
- `transform_object_points_world(pcl_local, R, t)` — the function callers MUST
  use before handing PCL into `_safe_add_point_cloud(obj_pose=..., mesh_extent=...)`.
- `_load_object_mesh(dataset, obj_name)` — GRAB .ply / ARCTIC .obj loader with
  mm→m rescale for ARCTIC. Fails loud on missing files / empty meshes.
- `build_scene(server, sample, obj_mesh, articulation_angle=None)` — idempotent
  per-frame scene builder; separated from the screenshot harness so can
  drive the scene programmatically.

CLI
---
`python -m umm.eval.handobject.viser_viewer --source {umm,native,canonical}
 --dataset {grab,arctic} --episode-id N [--npz-path P] [--headless]
 [--port P] [--image-size HxW] [--views top,side,iso] [--out-dir DIR]`

Module independence
-------------------
Intra-UMM imports restricted to `umm.eval.handobject.*`. Third-party: numpy,
viser, trimesh, scipy, Pillow; playwright imported lazily inside the
screenshot harness to keep top-level light.

Deviations from spec
--------------------
- `render_sample` accepts `mesh_path_override: Path | None` as a single mesh
  path (keeps the public signature simple; old design's `mesh_source_paths`
  dict became unused after ARCTIC frozen-mesh decision).
- Wrist-chirality gizmo: uses derived orientation from `index_mcp - wrist` and
  `thumb_mcp - wrist`; clearly documented as approximate. Fixed fallback if
  either input vector collapses to zero.
- `--image-size HxW` on CLI is parsed as `HxW` (e.g. `768x768`).
- ARCTIC banner uses both a GUI markdown label and an in-scene text label
  anchored above the object (per spec §3).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import trimesh  # type: ignore
import viser  # type: ignore
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy  # type: ignore

from umm.eval.handobject.canonical import CanonicalSample, load_canonical_npz
from umm.eval.handobject.gt_loader import (
    iter_arctic_articulation,
    iter_gt_episodes,
)


LOGGER = logging.getLogger("umm.eval.handobject.viser_viewer")
if not LOGGER.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constants — MANO 21-joint bones and known object lists
# ---------------------------------------------------------------------------

# MANO joint order (Text2HOI pre-decoded `j_lhand`/`j_rhand`):
#   0 = wrist
#   1..4 = index MCP, PIP, DIP, TIP
#   5..8 = middle ...
#   9..12 = ring ...
#   13..16 = pinky ...
#   17..20 = thumb ...
# Source: MANO official joint order (right-handed, Romero et al.); verified
# against Text2HOI's lib/utils/proc.py which uses MANO's standard topology.
#
# Spec asked for the joint topology as "thumb: 0->1..4" etc; the physical bone
# connectivity doesn't depend on that relabelling since we only render lines
# between adjacent joints within each finger chain.
MANO_BONES: List[Tuple[int, int]] = [
    # index
    (0, 1), (1, 2), (2, 3), (3, 4),
    # middle
    (0, 5), (5, 6), (6, 7), (7, 8),
    # pinky
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # thumb
    (0, 17), (17, 18), (18, 19), (19, 20),
]
if len(MANO_BONES) != 20:
    raise RuntimeError("MANO 21-joint skeleton has exactly 20 bones")

# For wrist-chirality gizmo: which joint-index gives an "index" and "thumb" ray
# from the wrist. Based on the same MANO ordering above.
_WRIST_INDEX_MCP_IDX = 1
_WRIST_THUMB_MCP_IDX = 17

# ARCTIC's 11 articulated objects.
ARCTIC_OBJ_NAMES = {
    "box", "capsulemachine", "espressomachine", "ketchup", "laptop",
    "microwave", "mixer", "notebook", "phone", "scissors", "waffleiron",
}

# Colors
COLOR_LEFT = (230, 40, 40)    # red
COLOR_RIGHT = (40, 190, 60)   # green

# Default paths (CLI convenience; `render_sample` does not depend on these).
_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../umm
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "handobject"
_DEFAULT_UMM_ROOT = _DEFAULT_DATA_ROOT / "umm"
_DEFAULT_TEXT2HOI_ROOT = _DEFAULT_DATA_ROOT / "text2hoi_raw"
_DEFAULT_BIN_MAP = _DEFAULT_DATA_ROOT
_DEFAULT_GRAB_MESH_DIR = _DEFAULT_DATA_ROOT / "raw_registered" / "grab_unzip" / "contact_meshes"
_DEFAULT_ARCTIC_MESH_ROOT = _DEFAULT_DATA_ROOT / "raw_registered" / "arctic_unzip" / "meta" / "object_vtemplates"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _rotmat_to_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to viser's (w, x, y, z) quaternion."""
    q_xyzw = R_scipy.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return (w, x, y, z)


def transform_object_points_world(
    pcl_local: np.ndarray,
    obj_R: np.ndarray,
    obj_t: np.ndarray,
) -> np.ndarray:
    """Apply `R @ pcl_local + t` to lift object-local points into world frame.

    Callers MUST invoke this BEFORE passing `obj_pose=...` to
    `_safe_add_point_cloud`, otherwise the V972 guard raises. This is the
    *single* blessed transform path for object-attached PCL in the viewer.

    Args:
        pcl_local: (N, 3) float array of points in the object's local frame.
        obj_R: (3, 3) rotation matrix (column-vector convention, same as
            `CanonicalSample.obj_rotation[t]`).
        obj_t: (3,) translation.

    Returns:
        (N, 3) float array of world-frame points.
    """
    pcl_local = np.asarray(pcl_local, dtype=np.float64)
    obj_R = np.asarray(obj_R, dtype=np.float64)
    obj_t = np.asarray(obj_t, dtype=np.float64).reshape(3)
    if pcl_local.ndim != 2 or pcl_local.shape[1] != 3:
        raise ValueError(f"pcl_local: expected (N,3), got {pcl_local.shape}")
    if obj_R.shape != (3, 3):
        raise ValueError(f"obj_R: expected (3,3), got {obj_R.shape}")
    # Column-vector convention: p_world = R @ p_local + t.
    # pcl_local is (N,3); want (N,3). R @ p = (R @ p.T).T = p @ R.T.
    return (pcl_local @ obj_R.T + obj_t).astype(np.float32)


def _safe_add_point_cloud(
    server: "viser.ViserServer",
    name: str,
    points_world: np.ndarray,
    *,
    obj_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    mesh_extent: Optional[float] = None,
    **kw: Any,
):
    """V972 render-boundary guard for point clouds.

    If BOTH `obj_pose` (translation, rotation) and `mesh_extent` (max axis of
    mesh bounding box in meters) are given, assert that the world-frame
    centroid of `points_world` lies within `mesh_extent * 1.2` meters of
    `obj_pose[0]`. Otherwise behave as a plain passthrough to
    `server.scene.add_point_cloud`.

    The check is intentionally relative (mesh-scale driven) — no 0.5 m magic
    number — so GRAB objects near the world origin still trip the guard when a
    local-frame PCL is placed far from `obj_t`.

    ### Known coverage gap

    This guard is a smell-test, not a guarantee. It CANNOT detect:

    1. **Rotation-only bugs**: if the caller applied `t` correctly but forgot `R`,
       the centroid is offset by `t` and may land within threshold even though
       the points are in the wrong orientation.
    2. **Pure-zero-obj_t bugs**: if `obj_t` is exactly or near zero (object at the
       world origin), both a correct PCL and an untransformed local-frame PCL would
       cluster near the origin, so the far-from-obj_t check cannot distinguish them.

    The near-origin case (PCL centroid near world origin while obj_t is non-trivial)
    is now caught by the second check below.

    Args:
        server: ViserServer instance.
        name: Scene path.
        points_world: (N, 3) float array — MUST already be in world frame.
        obj_pose: Optional (obj_t (3,), obj_R (3,3)) of the object the PCL is
            attached to. Pass this to opt into the V972 check.
        mesh_extent: Optional float — max of the object mesh's bounding box
            extents in meters. Required alongside `obj_pose`.
        **kw: Forwarded to `add_point_cloud` (e.g. colors, point_size).

    Raises:
        ValueError: If the V972 invariant fails.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"{name}: expected (N,3) points, got {points_world.shape}")
    if obj_pose is not None and mesh_extent is not None:
        obj_t = np.asarray(obj_pose[0], dtype=np.float64).reshape(3)
        centroid = points_world.astype(np.float64).mean(axis=0)
        dist = float(np.linalg.norm(centroid - obj_t))
        threshold = float(mesh_extent) * 1.2
        if dist > threshold:
            raise ValueError(
                f"V972 guard [{name}]: centroid={centroid.tolist()} is {dist:.3f} m "
                f"from obj_t={obj_t.tolist()} (threshold={threshold:.3f} m, "
                f"mesh_extent={mesh_extent:.3f} m). "
                f"Did you forget transform_object_points_world(pcl_local, R, t)?"
            )
        # Near-origin detection: if the PCL's centroid is near the WORLD origin
        # (not the object origin) AND obj_t is non-trivial AND the centroid is
        # meaningfully far from obj_t, the PCL is very likely un-transformed
        # local-frame. This catches the V972 case where |obj_t| < mesh_extent*1.2
        # (common on GRAB objects at table height) so the first check is silent.
        # The extra `dist > mesh_extent * 0.5` guard prevents false positives when
        # a correctly-transformed PCL has its centroid near origin because obj_t
        # itself is small (centroid ≈ obj_t → dist is small → skip this check).
        centroid_origin_dist = float(np.linalg.norm(centroid))
        obj_origin_dist = float(np.linalg.norm(obj_t))
        if (
            centroid_origin_dist < mesh_extent
            and obj_origin_dist > mesh_extent * 0.5
            and dist > mesh_extent * 0.5
        ):
            raise ValueError(
                f"V972 guard [{name}]: centroid near world origin "
                f"({centroid_origin_dist:.3f} m < {mesh_extent:.3f}) while obj_t is "
                f"non-trivial ({obj_origin_dist:.3f} m > {mesh_extent * 0.5:.3f} m) "
                f"and centroid is far from obj_t ({dist:.3f} m > {mesh_extent * 0.5:.3f} m). "
                f"Looks like untransformed local-frame PCL. Did you forget "
                f"transform_object_points_world(pcl_local, R, t)?"
            )
    return server.scene.add_point_cloud(name=name, points=points_world, **kw)


def _wrist_gizmo_wxyz(
    joints_t: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Derive an approximate wrist-frame quaternion for the chirality gizmo.

    +X axis = normalize(index_mcp - wrist)
    +Y axis = normalize(thumb_mcp - wrist - projection onto +X)
    +Z axis = X cross Y

    If any input vector degenerates to zero, returns identity quaternion. The
    gizmo is intentionally heuristic — its purpose is to surface mirror-chirality
    bugs visually, not to be kinematically exact.

    Args:
        joints_t: (21, 3) float array of a single hand's joints at one frame.

    Returns:
        (w, x, y, z) quaternion.
    """
    j = np.asarray(joints_t, dtype=np.float64)
    if j.shape != (21, 3):
        raise ValueError(f"_wrist_gizmo_wxyz: expected (21,3), got {j.shape}")
    wrist = j[0]
    v_x = j[_WRIST_INDEX_MCP_IDX] - wrist
    v_y_raw = j[_WRIST_THUMB_MCP_IDX] - wrist
    nx = np.linalg.norm(v_x)
    if nx < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    x_axis = v_x / nx
    v_y_perp = v_y_raw - np.dot(v_y_raw, x_axis) * x_axis
    ny = np.linalg.norm(v_y_perp)
    if ny < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    y_axis = v_y_perp / ny
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    # Re-orthonormalise via SVD to suppress numerical skew.
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        R_clean[:, 2] *= -1
    return _rotmat_to_wxyz(R_clean)


# ---------------------------------------------------------------------------
# Object mesh loading
# ---------------------------------------------------------------------------

def _grab_mesh_name_map(obj_name: str) -> str:
    """Map `CanonicalSample.obj_name` (which may be space-separated for GRAB,
    e.g. "alarm clock") to the mesh filename stem ("alarmclock")."""
    return obj_name.lower().replace(" ", "")


def _load_object_mesh(
    dataset: str,
    obj_name: str,
    data_root: Optional[Path] = None,
) -> trimesh.Trimesh:
    """Load an object mesh for GRAB or ARCTIC; rescale ARCTIC from mm to m.

    Args:
        dataset: 'grab' or 'arctic'.
        obj_name: Canonical object name as stored on `CanonicalSample`.
        data_root: Optional override root (default: `<repo>/data/handobject`).

    Returns:
        A `trimesh.Trimesh` in meters, with non-zero vertex and face counts.

    Raises:
        FileNotFoundError: If the expected mesh file does not exist.
        ValueError: If the mesh has 0 faces / 0 vertices / invalid dataset.
    """
    if dataset not in ("grab", "arctic"):
        raise ValueError(f"_load_object_mesh: dataset must be 'grab' or 'arctic', got {dataset!r}")

    data_root = data_root or _DEFAULT_DATA_ROOT

    if dataset == "grab":
        stem = _grab_mesh_name_map(obj_name)
        mesh_path = Path(data_root) / "raw_registered" / "grab_unzip" / "contact_meshes" / f"{stem}.ply"
    else:
        mesh_path = Path(data_root) / "raw_registered" / "arctic_unzip" / "meta" / "object_vtemplates" / obj_name / "mesh.obj"

    if not mesh_path.exists():
        raise FileNotFoundError(
            f"_load_object_mesh[{dataset}]: mesh not found for obj_name={obj_name!r} "
            f"at {mesh_path}"
        )

    mesh_any = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh_any, trimesh.Trimesh):
        raise ValueError(
            f"_load_object_mesh[{dataset}]: expected trimesh.Trimesh for {mesh_path}, "
            f"got {type(mesh_any).__name__}"
        )
    mesh = mesh_any
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError(
            f"_load_object_mesh[{dataset}]: empty mesh at {mesh_path} "
            f"(nv={len(mesh.vertices)}, nf={len(mesh.faces)})"
        )

    # Unit detection: ARCTIC meshes are stored in mm. Rescale to meters if the
    # bounding-box extent dwarfs any sane hand-object scale (> 2 m).
    extent = float(np.max(mesh.bounding_box.extents))
    if extent > 2.0:
        factor = 0.001
        LOGGER.info(
            "_load_object_mesh[%s %s]: extent %.3f > 2.0 m — rescaling by %.4f (mm->m)",
            dataset, obj_name, extent, factor,
        )
        mesh.apply_scale(factor)

    return mesh


# ---------------------------------------------------------------------------
# Scene build (stateless per-frame updater + static elements)
# ---------------------------------------------------------------------------

@dataclass
class SceneHandles:
    """Container for scene handles updated per frame (so callers can re-tick)."""
    server: "viser.ViserServer"
    sample: CanonicalSample
    mesh_extent: float
    obj_handle: Any          # GlbHandle
    lhand_points: Any        # PointCloudHandle
    rhand_points: Any        # PointCloudHandle
    lhand_bones: Any         # LineSegmentsHandle
    rhand_bones: Any         # LineSegmentsHandle
    lhand_gizmo: Any         # FrameHandle
    rhand_gizmo: Any         # FrameHandle
    arctic_label: Any        # LabelHandle or None
    articulation_angle: Optional[np.ndarray]


def _bones_endpoints(joints_t: np.ndarray) -> np.ndarray:
    """Build an (20, 2, 3) array of line endpoints from (21, 3) joints."""
    return np.stack(
        [np.stack([joints_t[a], joints_t[b]], axis=0) for (a, b) in MANO_BONES],
        axis=0,
    ).astype(np.float32)


def _bone_colors(rgb: Tuple[int, int, int]) -> np.ndarray:
    """Per-segment color array, shape (20, 2, 3) uint8."""
    arr = np.tile(np.asarray(rgb, dtype=np.uint8), (len(MANO_BONES), 2, 1))
    return arr


def build_scene(
    server: "viser.ViserServer",
    sample: CanonicalSample,
    obj_mesh: trimesh.Trimesh,
    *,
    frame_index: int,
    articulation_angle: Optional[np.ndarray] = None,
) -> SceneHandles:
    """Populate a viser scene with the static + per-frame elements.

    Args:
        server: ViserServer.
        sample: CanonicalSample.
        obj_mesh: Object mesh in meters.
        frame_index: Which frame the initial scene pose reflects.
        articulation_angle: Optional (T,) float — if given, renders an ARCTIC
            articulation banner with the angle at `frame_index`.

    Returns:
        SceneHandles for follow-up per-frame updates via `tick_scene`.
    """
    server.scene.set_up_direction("+z")

    # Static ground grid — 2 m x 2 m.
    server.scene.add_grid(
        "/ground",
        width=2.0, height=2.0, plane="xy",
        position=(0.0, 0.0, 0.0),
    )

    # Title markdown in GUI.
    title = f"Text: {sample.text[:80]} | Obj: {sample.obj_name} | T={sample.left_hand_joints.shape[0]} | fps={sample.fps}"
    server.gui.add_markdown(title)
    if articulation_angle is not None:
        # Loud banner; does not articulate ARCTIC meshes.
        server.gui.add_markdown(
            "⚠ **ARTICULATION NOT RENDERED** — ARCTIC mesh is frozen; "
            "per-frame angle shown as in-scene label."
        )

    T = int(sample.left_hand_joints.shape[0])
    if frame_index < 0 or frame_index >= T:
        raise ValueError(f"build_scene: frame_index={frame_index} out of [0,{T})")

    lhj = np.asarray(sample.left_hand_joints, dtype=np.float32)
    rhj = np.asarray(sample.right_hand_joints, dtype=np.float32)

    # Joint points.
    # guard wrapper: passthrough here (no obj_pose); ensures future PCL callers
    # can't bypass the V972 check.
    # V972 SCOPE: the guard covers `add_point_cloud` only. `add_line_segments`,
    # `add_frame`, and `add_mesh_trimesh` below pass world-frame data driven
    # by explicit `wxyz`/`position` handles — they do NOT go through the guard.
    # This is correct for today's usage (pre-decoded world joints + explicit
    # SE3 mesh driver). A future caller adding object-attached line_segments
    # must world-transform vertices before this point.
    lhand_points = _safe_add_point_cloud(
        server,  # type: ignore[arg-type]
        "/lhand_joints",
        lhj[frame_index],
        colors=np.tile(np.asarray(COLOR_LEFT, dtype=np.uint8), (21, 1)),
        point_size=0.006,
        point_shape="circle",
    )
    rhand_points = _safe_add_point_cloud(
        server,  # type: ignore[arg-type]
        "/rhand_joints",
        rhj[frame_index],
        colors=np.tile(np.asarray(COLOR_RIGHT, dtype=np.uint8), (21, 1)),
        point_size=0.006,
        point_shape="circle",
    )

    # Bones.
    lhand_bones = server.scene.add_line_segments(
        "/lhand_bones",
        points=_bones_endpoints(lhj[frame_index]),
        colors=_bone_colors(COLOR_LEFT),
        line_width=2.5,
    )
    rhand_bones = server.scene.add_line_segments(
        "/rhand_bones",
        points=_bones_endpoints(rhj[frame_index]),
        colors=_bone_colors(COLOR_RIGHT),
        line_width=2.5,
    )

    # Wrist chirality gizmos.
    lhand_gizmo = server.scene.add_frame(
        "/lhand_wrist_gizmo",
        wxyz=_wrist_gizmo_wxyz(lhj[frame_index]),
        position=tuple(float(x) for x in lhj[frame_index, 0]),
        axes_length=0.04, axes_radius=0.002,
    )
    rhand_gizmo = server.scene.add_frame(
        "/rhand_wrist_gizmo",
        wxyz=_wrist_gizmo_wxyz(rhj[frame_index]),
        position=tuple(float(x) for x in rhj[frame_index, 0]),
        axes_length=0.04, axes_radius=0.002,
    )

    # Object mesh — upload ONCE, update wxyz/position per frame.
    obj_R = np.asarray(sample.obj_rotation[frame_index], dtype=np.float32)
    obj_t = np.asarray(sample.obj_translation[frame_index], dtype=np.float32)
    obj_handle = server.scene.add_mesh_trimesh(
        "/object",
        mesh=obj_mesh,
        wxyz=_rotmat_to_wxyz(obj_R),
        position=tuple(float(x) for x in obj_t),
    )
    mesh_extent = float(np.max(obj_mesh.bounding_box.extents))

    arctic_label = None
    if articulation_angle is not None:
        ang_deg = float(np.rad2deg(articulation_angle[frame_index]))
        arctic_label = server.scene.add_label(
            "/arctic_banner",
            text=f"ARTIC: {ang_deg:.1f}° (FROZEN)",
            position=(
                float(obj_t[0]),
                float(obj_t[1]),
                float(obj_t[2]) + 0.15,
            ),
        )

    return SceneHandles(
        server=server,
        sample=sample,
        mesh_extent=mesh_extent,
        obj_handle=obj_handle,
        lhand_points=lhand_points,
        rhand_points=rhand_points,
        lhand_bones=lhand_bones,
        rhand_bones=rhand_bones,
        lhand_gizmo=lhand_gizmo,
        rhand_gizmo=rhand_gizmo,
        arctic_label=arctic_label,
        articulation_angle=articulation_angle,
    )


def tick_scene(handles: SceneHandles, frame_index: int) -> None:
    """Update per-frame handles in place for a new frame index."""
    s = handles.sample
    T = int(s.left_hand_joints.shape[0])
    if frame_index < 0 or frame_index >= T:
        raise ValueError(f"tick_scene: frame_index={frame_index} out of [0,{T})")

    lhj_t = np.asarray(s.left_hand_joints[frame_index], dtype=np.float32)
    rhj_t = np.asarray(s.right_hand_joints[frame_index], dtype=np.float32)
    obj_R = np.asarray(s.obj_rotation[frame_index], dtype=np.float32)
    obj_t = np.asarray(s.obj_translation[frame_index], dtype=np.float32)

    handles.lhand_points.points = lhj_t
    handles.rhand_points.points = rhj_t
    handles.lhand_bones.points = _bones_endpoints(lhj_t)
    handles.rhand_bones.points = _bones_endpoints(rhj_t)
    handles.lhand_gizmo.wxyz = _wrist_gizmo_wxyz(lhj_t)
    handles.lhand_gizmo.position = tuple(float(x) for x in lhj_t[0])
    handles.rhand_gizmo.wxyz = _wrist_gizmo_wxyz(rhj_t)
    handles.rhand_gizmo.position = tuple(float(x) for x in rhj_t[0])
    handles.obj_handle.wxyz = _rotmat_to_wxyz(obj_R)
    handles.obj_handle.position = tuple(float(x) for x in obj_t)
    if handles.arctic_label is not None and handles.articulation_angle is not None:
        ang_deg = float(np.rad2deg(handles.articulation_angle[frame_index]))
        handles.arctic_label.text = f"ARTIC: {ang_deg:.1f}° (FROZEN)"
        handles.arctic_label.position = (
            float(obj_t[0]),
            float(obj_t[1]),
            float(obj_t[2]) + 0.15,
        )


# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

# Each preset is (view, center, factor) -> dict with position/look_at/up/wxyz/fov.
# Center is the mean of (obj_translation[t], lhj[t,0], rhj[t,0]) over valid hand
# wrists, so the camera frames the interaction regardless of global scene offset.
# The radius is driven by `max(mesh_extent, 0.15 m) * factor` so small GRAB
# objects and large ARCTIC objects both fit.

# View-specific radius factors, calibrated so a 0.15 m object still fits in a
# 55° FOV render and a 0.5 m object does not clip the frame.
_VIEW_RADIUS_FACTORS: Dict[str, float] = {
    "top": 4.0,
    "side": 4.0,
    "iso": 3.5,
}


_HAND_WRIST_PROXIMITY_M = 0.5


def _derive_camera_center(
    sample: "CanonicalSample",
    frame_index: int,
) -> np.ndarray:
    """Mean of obj_t + any wrist within 0.5 m of obj_t at frame_index.

    Single-hand GRAB episodes (is_lhand=False or is_rhand=False) leave
    residual unrelated joints in the inactive hand slot — including them
    biases the camera center ~1 m away from the active scene, clipping
    hand+object out of the frame (V972-sibling bug surfaced on eps 73, 249).
    We exclude wrists that are farther than 0.5 m from the object — a
    natural arm-reach distance — which keeps both hands of bimanual
    episodes AND drops inactive hands of single-hand episodes.

    Guarantees at least one point (obj_t) is always included.
    """
    t = int(frame_index)
    obj_t = np.asarray(sample.obj_translation[t], dtype=np.float64).reshape(3)
    pts = [obj_t]
    for wrist_src in (sample.left_hand_joints, sample.right_hand_joints):
        wrist = np.asarray(wrist_src[t, 0], dtype=np.float64)
        if np.linalg.norm(wrist - obj_t) <= _HAND_WRIST_PROXIMITY_M:
            pts.append(wrist)
    return np.mean(np.stack(pts, axis=0), axis=0)


def _camera_preset(
    view: str,
    center: np.ndarray,
    *,
    mesh_extent: float = 0.15,
) -> Dict[str, Any]:
    """Compute position / look_at / up / wxyz / fov for one of 3 views.

    Args:
        view: one of "top", "side", "iso".
        center: (3,) scene center — camera aims here.
        mesh_extent: max bounding-box extent of the object mesh (m). Used as a
            floor via `max(mesh_extent, 0.15)` to keep the framing sane on small
            objects and proportional on large ones.
    """
    if view not in _VIEW_RADIUS_FACTORS:
        raise ValueError(f"_camera_preset: unknown view {view!r}")
    center = np.asarray(center, dtype=np.float64).reshape(3)
    factor = _VIEW_RADIUS_FACTORS[view]
    radius = float(max(float(mesh_extent), 0.15) * factor)
    # Base unit offsets per view (unitless; scaled by `radius` below).
    # World-up = +Z (see build_scene -> set_up_direction("+z")).
    if view == "top":
        # Looking straight down along -Z from directly above.
        unit_offset = np.array([0.0, 0.0, 1.0])
        up = np.array([1.0, 0.0, 0.0])  # +X up in top view, per spec
    elif view == "side":
        # Looking in -X direction, with +Z up.
        unit_offset = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
    else:  # iso
        # 3/4 view: camera in +X, -Y, +Z octant; world-up +Z.
        unit_offset = np.array([1.0, -1.0, 1.0])
        unit_offset = unit_offset / np.linalg.norm(unit_offset)
        up = np.array([0.0, 0.0, 1.0])
    position = center + unit_offset * radius
    look_at = center

    # Derive a wxyz from (position, look_at, up) as a fallback for clients that
    # don't consume look_at/up_direction separately. Camera convention:
    # local -Z points toward look_at; local +Y = up (after re-ortho).
    forward = look_at - position
    fn = np.linalg.norm(forward)
    if fn < 1e-6:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / fn
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # up parallel to forward — pick an orthogonal fallback.
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
    right = right / rn
    up_rect = np.cross(right, forward)
    # Camera frame: x=right, y=up, z=-forward (right-handed).
    R_cam = np.column_stack([right, up_rect, -forward])
    wxyz = _rotmat_to_wxyz(R_cam)
    return {
        "wxyz": wxyz,
        "position": tuple(float(x) for x in position),
        "look_at": tuple(float(x) for x in look_at),
        "up": tuple(float(x) for x in up),
        "fov": float(np.deg2rad(55.0)),
        "view_name": view,
        "radius": radius,
    }


# ---------------------------------------------------------------------------
# Headless screenshot harness (playwright + xvfb)
# ---------------------------------------------------------------------------

def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_xvfb_playwright_chromium(url: str, log_dir: Path) -> Tuple[Any, Any, Any]:
    """Launch chromium via playwright (headless); open `url`.

    Playwright chromium in headless mode uses --ozone-platform=headless and
    does NOT require DISPLAY. We keep Xvfb as a fallback (in case a caller
    switches to headed chromium), but do NOT leak DISPLAY across calls:
    any Xvfb we start is terminated in the finally block AND the DISPLAY
    env var we set is unset, so a second render_sample() call re-entering
    this helper gets a fresh environment. The previous code leaked DISPLAY
    → second call saw stale DISPLAY pointing at a dead Xvfb → chromium
    hung → no viser client connected → TimeoutError.

    Returns (playwright_mgr, browser, page). Caller must close browser and
    stop playwright in `finally`.
    """
    from playwright.sync_api import sync_playwright  # lazy import
    log_dir.mkdir(parents=True, exist_ok=True)
    xvfb_proc: Optional[subprocess.Popen] = None
    set_display = False
    # Headless chromium does not require X; skip Xvfb unless explicitly forced.
    if os.environ.get("UMM_VISER_FORCE_XVFB") == "1" and not os.environ.get("DISPLAY"):
        display_num = 99 + (os.getpid() % 900)
        display_str = f":{display_num}"
        log_path = log_dir / "xvfb.log"
        log_fd = open(log_path, "wb")
        try:
            xvfb_proc = subprocess.Popen(
                ["Xvfb", display_str, "-screen", "0", "1600x1200x24"],
                stdout=log_fd,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_fd.close()
            raise
        log_fd.close()  # safe: Popen dup2'd the fd; closing the parent copy is correct
        time.sleep(0.5)
        os.environ["DISPLAY"] = display_str
        set_display = True
        LOGGER.info("Started Xvfb on DISPLAY=%s (pid=%d)", display_str, xvfb_proc.pid)

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
    except Exception:
        pw.stop()
        if xvfb_proc is not None:
            xvfb_proc.terminate()
        if set_display:
            os.environ.pop("DISPLAY", None)
        raise
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    return (pw, browser, xvfb_proc, set_display), browser, page


def _wait_for_client(server: "viser.ViserServer", timeout_s: float = 60.0) -> "viser.ClientHandle":
    """Poll `server.get_clients()` until at least one client is connected."""
    start = time.time()
    while time.time() - start < timeout_s:
        clients = server.get_clients()
        if clients:
            client = next(iter(clients.values()))
            LOGGER.info("Client connected after %.1f s (id=%s)", time.time() - start, next(iter(clients.keys())))
            return client
        time.sleep(0.2)
    raise TimeoutError(
        f"No viser client connected within {timeout_s:.0f} s. "
        f"Check xvfb/chromium subprocess, server port, and network."
    )


# ---------------------------------------------------------------------------
# Public API — render_sample
# ---------------------------------------------------------------------------

def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        LOGGER.warning("_git_sha: could not read sha: %s", e, exc_info=True)
        return "unknown"


def render_sample(
    sample: CanonicalSample,
    out_dir: Path,
    views: Sequence[Literal["top", "side", "iso"]] = ("top", "side", "iso"),
    *,
    port: int = 0,
    image_size: Tuple[int, int] = (768, 768),
    frame_index: Optional[int] = None,
    mesh_path_override: Optional[Path] = None,
    name_prefix: str = "sample",
    articulation_angle: Optional[float] = None,
) -> List[Path]:
    """Render `sample` from each `view` via headless viser; write PNGs + JSON.

    Args:
        sample: A fully validated CanonicalSample.
        out_dir: Destination directory (created if missing).
        views: Subset of ("top","side","iso"). Unknown names raise.
        port: Server port. 0 = auto-pick a free port.
        image_size: (H, W) pixels for each PNG.
        frame_index: Which frame to render. Default: middle valid frame.
        mesh_path_override: If given, load this file as the object mesh instead
            of inferring from `obj_name`. Useful for tests / synthetic samples.
        name_prefix: PNG filename stem prefix; files are
            `{prefix}_{view}.png` + `{prefix}_{view}.json`.
        articulation_angle: If given, an ARCTIC articulation angle (radians)
            at `frame_index`. The banner is rendered; no mesh articulation.

    Returns:
        List of PNG paths (one per view, in order).
    """
    if len(views) == 0:
        raise ValueError("render_sample: at least one view required")
    for v in views:
        if v not in ("top", "side", "iso"):
            raise ValueError(f"render_sample: unknown view {v!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = int(image_size[0]), int(image_size[1])
    T = int(sample.left_hand_joints.shape[0])
    if frame_index is None:
        valid_idx = np.where(np.asarray(sample.valid_mask))[0]
        if len(valid_idx) == 0:
            frame_index = T // 2
        else:
            frame_index = int(valid_idx[len(valid_idx) // 2])

    # Load mesh.
    if mesh_path_override is not None:
        mesh_any = trimesh.load(str(mesh_path_override), force="mesh")
        if not isinstance(mesh_any, trimesh.Trimesh):
            raise ValueError(f"mesh_path_override: {mesh_path_override} did not load as Trimesh")
        obj_mesh = mesh_any
        if len(obj_mesh.faces) == 0:
            raise ValueError(f"mesh_path_override: empty mesh at {mesh_path_override}")
    else:
        # ARCTIC-vs-GRAB detection by obj_name membership. Safe today: all 11
        # ARCTIC obj names (ARCTIC_OBJ_NAMES) are disjoint from GRAB's 57 names.
        # If a future dataset adds a conflicting name, pass mesh_path_override
        # to skip this heuristic.
        dataset = "arctic" if sample.obj_name in ARCTIC_OBJ_NAMES else "grab"
        obj_mesh = _load_object_mesh(dataset, sample.obj_name)

    # Build articulation-per-frame array if a scalar angle was passed.
    # Input articulation_angle is a scalar today (frozen-mesh decision). The (T,)
    # array shape is plumbing for future per-frame support; ARCTIC banner currently
    # reads articulation_arr[frame_index], which equals the input scalar.
    articulation_arr: Optional[np.ndarray] = None
    if articulation_angle is not None:
        articulation_arr = np.full((T,), float(articulation_angle), dtype=np.float32)

    server = None
    pw_bundle = None
    browser = None
    try:
        # Pick port with TOCTOU retry: _pick_free_port() binds then closes,
        # leaving a window for another process to claim it. Retry up to 5x.
        # If a caller-supplied `port` is busy, falls back to auto-pick after 1 retry.
        requested_port = port
        for _attempt in range(5):
            try:
                if port == 0:
                    port = _pick_free_port()
                    LOGGER.info("render_sample: auto-picked free port %d (attempt %d)", port, _attempt + 1)
                server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
                break
            except OSError as _e:
                if "Address" in str(_e) or getattr(_e, "errno", None) == 98:
                    if _attempt == 4:
                        raise
                    if requested_port != 0 and _attempt == 0:
                        LOGGER.warning(
                            "render_sample: requested port %d already in use, "
                            "falling back to auto-pick", requested_port
                        )
                        port = 0
                    else:
                        LOGGER.warning(
                            "render_sample: port %d already in use, retrying (%d/5)", port, _attempt + 1
                        )
                        port = 0
                    continue
                raise
        # Note: no `else: raise` — the inner `if _attempt == 4: raise` always
        # triggers on the 5th attempt, so the loop either breaks on success
        # or re-raises the OSError. The only way to reach code after the
        # for-block is via `break`, which indicates successful bind.
        handles = build_scene(
            server, sample, obj_mesh,
            frame_index=frame_index,
            articulation_angle=articulation_arr,
        )

        url = f"http://127.0.0.1:{port}"
        LOGGER.info("render_sample: viser listening at %s", url)

        pw_bundle, browser, _page = _start_xvfb_playwright_chromium(
            url, log_dir=out_dir / "_harness_logs"
        )

        client = _wait_for_client(server, timeout_s=60.0)
        # Larger settle so meshes upload completely AND the first camera
        # update (which viser applies asynchronously over websocket) takes
        # effect before the first get_render(). 1.5 s was too tight and
        # occasionally produced a top-view PNG that looked like the default
        # iso orbit pose.
        time.sleep(3.0)

        repo_root = Path(__file__).resolve().parents[4]
        git_sha = _git_sha(repo_root)
        viser_ver = getattr(viser, "__version__", "unknown")

        # Collect playwright/chromium version for sidecar harness block.
        # playwright has no __version__ attribute; use importlib.metadata.
        _playwright_version = "unknown"
        try:
            import importlib.metadata as _md
            _playwright_version = _md.version("playwright")
        except (ImportError, _md.PackageNotFoundError) as e:
            LOGGER.warning("render_sample: could not read playwright version: %s", e, exc_info=True)
        _chromium_path = "unknown"
        if pw_bundle is not None:
            try:
                _chromium_path = str(pw_bundle[0].chromium.executable_path)
            except (AttributeError, TypeError) as e:
                LOGGER.warning("render_sample: could not read chromium path: %s", e, exc_info=True)

        out_paths: List[Path] = []
        center = _derive_camera_center(sample, frame_index)
        mesh_extent = float(np.max(obj_mesh.bounding_box.extents))
        for view in views:
            preset = _camera_preset(view, center, mesh_extent=mesh_extent)
            # Viser only sends position / look_at / up_direction / fov / near
            # / far to the client. We drive those four directly and let the
            # client recompute wxyz from them. Ordering:
            # 1. position (moves the camera, drags look_at along to preserve
            #    orientation — so we still need step 2).
            # 2. look_at (re-aims; this calls _update_wxyz, which can divide
            #    by zero if the previous up_direction is parallel to the new
            #    forward — suppressed via np.errstate; the bogus wxyz is
            #    harmless because the client consumes the explicit messages,
            #    not wxyz).
            # 3. up_direction (anchors world-up; this is the core fix for the
            #    pre-fix bug where chromium's orbit-control default picked
            #    the wrong up and drew the grid above the hands).
            # 4. fov.
            with np.errstate(invalid="ignore", divide="ignore"):
                client.camera.position = preset["position"]
                client.camera.look_at = preset["look_at"]
                client.camera.up_direction = preset["up"]
            client.camera.fov = preset["fov"]
            # Let the camera update propagate. 1.2 s is empirically enough;
            # below 1.0 s the FIRST view sometimes renders with the default
            # orbit pose because the websocket camera update hadn't landed.
            time.sleep(1.2)
            # Warm-up: render a tiny throwaway frame. This forces the client
            # to flush any pending camera/scene updates and serialize a fresh
            # frame with the new camera. The second get_render is then
            # guaranteed to reflect the preset.
            try:
                _ = client.camera.get_render(height=64, width=64, transport_format="jpeg")
            except (TimeoutError, OSError, RuntimeError) as e:
                LOGGER.warning("render_sample: warm-up render failed: %s", e, exc_info=True)
            time.sleep(0.3)

            img = client.camera.get_render(height=H, width=W, transport_format="jpeg")
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            # One retry if the render came back nearly uniform (blank /
            # default pose). std < 3 means the image is effectively empty.
            if float(arr.std()) < 3.0:
                LOGGER.warning(
                    "render_sample[%s]: first render std=%.2f — retrying after longer settle",
                    view, float(arr.std()),
                )
                time.sleep(1.5)
                img = client.camera.get_render(height=H, width=W, transport_format="jpeg")
                arr = np.asarray(img)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]

            png_path = out_dir / f"{name_prefix}_{view}.png"
            Image.fromarray(arr).save(png_path)
            sidecar = {
                "camera": {
                    "wxyz": list(preset["wxyz"]),
                    "position": list(preset["position"]),
                    "look_at": list(preset["look_at"]),
                    "up": list(preset["up"]),
                    "fov": float(preset["fov"]),
                    "radius": float(preset["radius"]),
                    "center": [float(x) for x in center],
                    "mesh_extent": float(mesh_extent),
                    "host": "127.0.0.1",
                    "port": port,
                },
                "frame_index": int(frame_index),
                "sample_text": sample.text,
                "obj_name": sample.obj_name,
                "obj_translation": [float(x) for x in sample.obj_translation[frame_index]],
                "obj_rotation": [float(x) for x in np.asarray(sample.obj_rotation[frame_index]).reshape(-1)],
                "articulation_angle_rad": (
                    float(articulation_angle) if articulation_angle is not None else None
                ),
                "view": view,
                "image_size": [H, W],
                "git_sha": git_sha,
                "viser_version": viser_ver,
                "harness": {
                    "playwright_version": _playwright_version,
                    "chromium_path": _chromium_path,
                },
            }
            (out_dir / f"{name_prefix}_{view}.json").write_text(
                json.dumps(sidecar, indent=2)
            )
            out_paths.append(png_path)
            LOGGER.info(
                "render_sample: wrote %s (bytes=%d)",
                png_path, png_path.stat().st_size,
            )

        return out_paths

    finally:
        # Cleanup in reverse order.
        if browser is not None:
            try:
                browser.close()
            except Exception as e:  # noqa: BLE001 — log and continue cleanup
                LOGGER.warning("render_sample: browser.close() failed: %s", e)
        if pw_bundle is not None:
            pw, _br, xvfb_proc, set_display = pw_bundle
            try:
                pw.stop()
            except Exception as e:
                LOGGER.warning("render_sample: playwright.stop() failed: %s", e)
            if xvfb_proc is not None:
                try:
                    xvfb_proc.terminate()
                    xvfb_proc.wait(timeout=5)
                except Exception as e:
                    LOGGER.warning("render_sample: Xvfb cleanup failed: %s", e)
            if set_display:
                os.environ.pop("DISPLAY", None)
        if server is not None:
            try:
                server.stop()
            except Exception as e:
                LOGGER.warning("render_sample: server.stop() failed: %s", e)


# ---------------------------------------------------------------------------
# Data source loaders (CLI glue)
# ---------------------------------------------------------------------------

def _find_umm_episode(dataset: str, episode_id: int, umm_root: Path) -> Path:
    """Locate data/handobject/umm/<dataset>/{train,val,test}/NNNNNN.npz."""
    stem = f"{episode_id:06d}.npz"
    for split in ("train", "val", "test"):
        candidate = Path(umm_root) / dataset / split / stem
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"_find_umm_episode: no {stem} under {umm_root}/{dataset}/{{train,val,test}}"
    )


def _load_native_episode(
    dataset: str,
    episode_id: int,
    text2hoi_root: Path,
    bin_map_root: Path,
) -> Tuple[CanonicalSample, Optional[float]]:
    """Load a native (GT) episode via gt_loader; return (sample, artic_angle_at_mid)."""
    npz_path = Path(text2hoi_root) / dataset / "data.npz"
    text_json = Path(text2hoi_root) / dataset / "text.json"
    bin_map = Path(bin_map_root) / f"bin_map_{dataset}.json"
    for p in (npz_path, text_json, bin_map):
        if not p.exists():
            raise FileNotFoundError(f"_load_native_episode: missing {p}")

    it = iter_gt_episodes(
        dataset=dataset,
        bin_map_path=str(bin_map),
        npz_path=str(npz_path),
        text_json_path=str(text_json),
        max_episodes=episode_id + 1,
    )
    try:
        sample = next(itertools.islice(it, episode_id, episode_id + 1))
    except StopIteration:
        raise ValueError(
            f"_load_native_episode: episode_id={episode_id} out of range in {npz_path}"
        ) from None

    artic_angle: Optional[float] = None
    if dataset == "arctic":
        # Pick articulation at the middle valid frame.
        valid_idx = np.where(np.asarray(sample.valid_mask))[0]
        mid = int(valid_idx[len(valid_idx) // 2]) if len(valid_idx) else 0
        it_artic = iter_arctic_articulation(str(npz_path))
        angle_arr = next(itertools.islice(it_artic, episode_id, episode_id + 1))
        artic_angle = float(angle_arr[mid])

    return sample, artic_angle


def _load_sample(
    source: str,
    dataset: str,
    episode_id: int,
    *,
    npz_path: Optional[Path] = None,
    umm_root: Path = _DEFAULT_UMM_ROOT,
    text2hoi_root: Path = _DEFAULT_TEXT2HOI_ROOT,
    bin_map_root: Path = _DEFAULT_BIN_MAP,
) -> Tuple[CanonicalSample, Optional[float]]:
    """Dispatch by `source`. Returns (sample, articulation_angle_or_None)."""
    if source == "umm":
        path = _find_umm_episode(dataset, episode_id, umm_root)
        sample = load_canonical_npz(str(path))
        artic: Optional[float] = None
        if dataset == "arctic":
            # UMM adapter does NOT store articulation in canonical npz; fetch from native npz.
            try:
                _, artic = _load_native_episode(dataset, episode_id, text2hoi_root, bin_map_root)
            except (FileNotFoundError, KeyError) as e:
                LOGGER.warning(
                    "_load_sample[umm/arctic]: could not fetch articulation angle: %s", e
                )
        return sample, artic
    if source == "native":
        return _load_native_episode(dataset, episode_id, text2hoi_root, bin_map_root)
    if source == "canonical":
        if npz_path is None:
            raise ValueError("--source canonical requires --npz-path")
        sample = load_canonical_npz(str(npz_path))
        return sample, None
    raise ValueError(f"_load_sample: unknown source {source!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_image_size(s: str) -> Tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--image-size: expected HxW, got {s!r}")
    return (int(parts[0]), int(parts[1]))


def _parse_views(s: str) -> List[str]:
    vs = [v.strip() for v in s.split(",") if v.strip()]
    for v in vs:
        if v not in ("top", "side", "iso"):
            raise argparse.ArgumentTypeError(f"--views: unknown view {v!r}")
    return vs


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umm.eval.handobject.viser_viewer",
        description="Viser hand-object viewer.",
    )
    p.add_argument("--source", choices=("umm", "native", "canonical"), required=True)
    p.add_argument("--dataset", choices=("grab", "arctic"), required=True)
    p.add_argument("--episode-id", type=int, required=True)
    p.add_argument("--npz-path", type=Path, default=None,
                   help="Required for --source canonical.")
    p.add_argument("--headless", action="store_true",
                   help="Render PNGs and exit; do not hold the server.")
    p.add_argument("--port", type=int, default=0, help="0 = auto-pick.")
    p.add_argument("--image-size", type=_parse_image_size, default=(768, 768),
                   help="HxW (default 768x768).")
    p.add_argument("--views", type=_parse_views, default=["top", "side", "iso"])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/viser_out"))
    p.add_argument("--frame-index", type=int, default=None)
    p.add_argument("--umm-root", type=Path, default=_DEFAULT_UMM_ROOT)
    p.add_argument("--text2hoi-root", type=Path, default=_DEFAULT_TEXT2HOI_ROOT)
    p.add_argument("--bin-map-root", type=Path, default=_DEFAULT_BIN_MAP)
    p.add_argument("--name-prefix", type=str, default=None,
                   help="PNG stem prefix. Default: {dataset}_{source}_ep{id}.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    sample, artic_angle = _load_sample(
        source=args.source,
        dataset=args.dataset,
        episode_id=args.episode_id,
        npz_path=args.npz_path,
        umm_root=args.umm_root,
        text2hoi_root=args.text2hoi_root,
        bin_map_root=args.bin_map_root,
    )

    prefix = args.name_prefix or f"{args.dataset}_{args.source}_ep{args.episode_id:06d}"

    if args.headless:
        paths = render_sample(
            sample,
            out_dir=args.out_dir,
            views=tuple(args.views),
            port=args.port,
            image_size=tuple(args.image_size),
            frame_index=args.frame_index,
            name_prefix=prefix,
            articulation_angle=artic_angle,
        )
        for p in paths:
            print(str(p))
        return 0

    # Interactive mode — hold server until Ctrl+C.
    # Retry port binding to handle TOCTOU between _pick_free_port and ViserServer.
    _cli_port = args.port
    for _attempt in range(5):
        try:
            port = _cli_port or _pick_free_port()
            server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
            break
        except OSError as _e:
            if _cli_port or ("Address" not in str(_e) and getattr(_e, "errno", None) != 98):
                raise
            if _attempt == 4:
                raise
            LOGGER.warning("main: port %d in use, retrying (%d/5)", port, _attempt + 1)
    try:
        if args.dataset == "arctic" and sample.obj_name in ARCTIC_OBJ_NAMES:
            dataset = "arctic"
        else:
            dataset = "grab"
        obj_mesh = _load_object_mesh(dataset, sample.obj_name)
        T = int(sample.left_hand_joints.shape[0])
        fi = args.frame_index if args.frame_index is not None else T // 2
        artic_arr: Optional[np.ndarray] = None
        if artic_angle is not None:
            artic_arr = np.full((T,), float(artic_angle), dtype=np.float32)
        build_scene(server, sample, obj_mesh, frame_index=fi, articulation_angle=artic_arr)
        print(f"Viser listening at http://127.0.0.1:{port}")
        print("Ctrl+C to exit.")
        server.sleep_forever()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
