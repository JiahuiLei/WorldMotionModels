"""omomo_penetration_viz.py — World-space penetration viz overlay for OMOMO.

Renders a sequence of frames: object mesh + L/R palm spheres color-coded by
SDF value (red=inside, green=on surface, blue=outside) + numeric depth panel.

Gated by ``viz_penetration_flag: True`` in the training config.

# TODO B2: hook into omomo_viz.py at viz call site (Phase C integration deferred).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

from umm.models.wmm.sdf_utils import load_sdf_asset, query_sdf_at_world_points


def _color_for_sdf(sdf_m: float) -> tuple:
    """Map SDF value in metres to an (R, G, B) tuple 0..255.

    Args:
        sdf_m: signed distance in metres. Negative = inside object.

    Returns:
        (R, G, B) as ints 0-255.
    """
    if sdf_m < -0.005:
        return (220, 30, 30)    # red — penetrating
    elif sdf_m < 0.01:
        return (30, 200, 30)    # green — on surface
    else:
        return (30, 80, 220)    # blue — far outside


def _look_at(
    eye: tuple,
    target: tuple,
    up: tuple = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Build a 4x4 camera-pose matrix in the pyrender (OpenGL) convention.

    Args:
        eye: camera position (x, y, z).
        target: point to look at (x, y, z).
        up: world up vector.

    Returns:
        (4, 4) float64 pose matrix.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    R = np.stack([s, u, -f], axis=1)   # (3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = eye
    return pose


def render_penetration_sequence(
    obj_mesh_path: str,
    sdf_stem: str,
    obj_R_seq_world: np.ndarray,        # (T, 3, 3)
    obj_t_seq_world: np.ndarray,        # (T, 3)
    palm_world_seq: np.ndarray,         # (T, 2, 3)  index 0=Left, 1=Right
    out_dir: str,
    width: int = 512,
    height: int = 512,
    cam_look_at: tuple = (0.0, 0.0, 1.0),
    cam_stand_at: tuple = (3.0, 3.0, 2.0),
    save_n_frames: int = 0,
    renderer: Optional["pyrender.OffscreenRenderer"] = None,
) -> List[str]:
    """Render T frames; save JPEGs; return list of saved paths.

    Each frame shows:
    - Object mesh in world pose (grey, translucent).
    - Left and right palm spheres (radius 2.5 cm) color-coded by SDF value.
    - Bottom bar with numeric L/R penetration depth in mm.

    Args:
        obj_mesh_path: path to the object mesh file (PLY, OBJ, etc.).
        sdf_stem: stem path for :func:`~umm.models.wmm.sdf_utils.load_sdf_asset`,
            e.g. ``"data_wmm/omomo_v2/rest_object_sdf_v2/clothesstand"``.
        obj_R_seq_world: ``(T, 3, 3)`` rotation matrices (world frame).
        obj_t_seq_world: ``(T, 3)`` translation vectors (world frame, metres).
        palm_world_seq: ``(T, 2, 3)`` palm positions (world frame, metres).
            Axis-1 index 0 = Left hand, 1 = Right hand.
        out_dir: output directory for JPEG frames.
        width: render width in pixels.
        height: render height in pixels.
        cam_look_at: world point the camera looks at.
        cam_stand_at: camera eye position in world space.
        save_n_frames: if 0, render every frame; otherwise sample this many
            frames evenly from ``[0, T-1]``.

    Returns:
        List of absolute paths to the saved JPEG files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load object mesh
    mesh: trimesh.Trimesh = trimesh.load(obj_mesh_path, force="mesh", process=False)

    T: int = obj_R_seq_world.shape[0]

    # Load SDF asset
    sdf_np, c_np, e_np = load_sdf_asset(sdf_stem)
    sdf_t = torch.from_numpy(sdf_np)[None]   # (1, D, H, W)
    c_t = torch.from_numpy(c_np)[None]       # (1, 3)
    e_t = torch.from_numpy(e_np)[None]       # (1, 3)

    # Decide which frames to render
    if save_n_frames == 0:
        idxs: List[int] = list(range(T))
    else:
        idxs = list(np.linspace(0, T - 1, save_n_frames).astype(int))

    # Camera pose (constant across frames)
    cam_pose = _look_at(cam_stand_at, cam_look_at)

    # Reuse caller-provided renderer when available — creating extra
    # pyrender.OffscreenRenderer instances during a viz cycle corrupts
    # the EGL state of any cached renderer in the same process and
    # silently breaks subsequent renders. Only fall back to creating
    # our own renderer when running standalone (e.g. unit tests).
    own_renderer: bool = renderer is None
    if own_renderer:
        # Pre-bind to None so the `finally` block never sees an unbound
        # name if `OffscreenRenderer(...)` raises.
        new_renderer: Optional["pyrender.OffscreenRenderer"] = None
        new_renderer = pyrender.OffscreenRenderer(width, height)
        renderer = new_renderer
    else:
        # Caller-supplied renderer: enforce that requested width/height
        # match the renderer's viewport. Mismatched dimensions would
        # leave the depth-panel text drawn at the wrong y-offset.
        rw = int(getattr(renderer, "viewport_width", width))
        rh = int(getattr(renderer, "viewport_height", height))
        if (rw, rh) != (int(width), int(height)):
            raise ValueError(
                f"render_penetration_sequence: renderer viewport ({rw}x{rh}) "
                f"does not match requested width/height ({width}x{height})."
            )

    paths: List[str] = []
    try:
        for fi in idxs:
            R_w: np.ndarray = obj_R_seq_world[fi]     # (3, 3)
            t_w: np.ndarray = obj_t_seq_world[fi]     # (3,)
            palms: np.ndarray = palm_world_seq[fi]    # (2, 3)

            # SDF query at both palm positions
            # query_sdf_at_world_points expects (B, T, P, 3) / (B, T, 3, 3) / (B, T, 3)
            palm_t_torch = torch.from_numpy(palms[None, None]).float()   # (1, 1, 2, 3)
            R_t_torch = torch.from_numpy(R_w[None, None]).float()        # (1, 1, 3, 3)
            t_t_torch = torch.from_numpy(t_w[None, None]).float()        # (1, 1, 3)

            sdf_at_palms: np.ndarray = query_sdf_at_world_points(
                palm_t_torch, R_t_torch, t_t_torch, sdf_t, c_t, e_t,
            )[0, 0].numpy()   # (2,)

            # Build pyrender scene
            scene = pyrender.Scene(
                bg_color=(255, 255, 255, 0),
                ambient_light=(0.4, 0.4, 0.4),
            )

            # Object mesh in world pose
            obj_mesh_world: trimesh.Trimesh = mesh.copy()
            T4 = np.block([[R_w, t_w[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
            obj_mesh_world.apply_transform(T4)
            try:
                obj_mesh_world.visual.vertex_colors = [180, 180, 180, 100]
            except Exception:
                pass
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh_world, smooth=False))

            # Palm spheres
            for hi in range(2):
                sph: trimesh.Trimesh = trimesh.creation.uv_sphere(radius=0.025)
                color = _color_for_sdf(float(sdf_at_palms[hi]))
                sph.visual.vertex_colors = list(color) + [255]
                sph.apply_translation(palms[hi])
                scene.add(pyrender.Mesh.from_trimesh(sph, smooth=True))

            # Camera + light
            cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
            scene.add(cam, pose=cam_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            scene.add(light, pose=cam_pose)

            # Off-screen render — reuse the renderer chosen above. Do NOT
            # construct a new pyrender.OffscreenRenderer per frame; doing
            # so corrupts the EGL state of any other cached renderer in
            # the same process.
            rgb, _ = renderer.render(scene)

            # Depth panel overlay
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            l_mm = float(sdf_at_palms[0]) * 1000.0
            r_mm = float(sdf_at_palms[1]) * 1000.0
            try:
                font: Optional[ImageFont.ImageFont] = ImageFont.load_default()
            except Exception:
                font = None
            draw.rectangle([0, height - 36, width, height], fill=(0, 0, 0, 180))
            draw.text(
                (8, height - 32),
                f"L: {l_mm:+.1f} mm",
                fill=_color_for_sdf(float(sdf_at_palms[0])),
                font=font,
            )
            draw.text(
                (width // 2, height - 32),
                f"R: {r_mm:+.1f} mm",
                fill=_color_for_sdf(float(sdf_at_palms[1])),
                font=font,
            )

            out_path = os.path.join(out_dir, f"frame_{fi:04d}.jpg")
            img.convert("RGB").save(out_path, quality=85)
            paths.append(out_path)
    finally:
        if own_renderer and renderer is not None:
            renderer.delete()

    return paths
