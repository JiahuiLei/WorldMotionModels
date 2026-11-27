"""Headless mesh rendering via pyrender with EGL/OSMesa backend.

Provides a thread-safe, lazily-initialized offscreen renderer
(``MeshRenderer``) and a convenience function ``visualize_a_mesh`` for
rendering trimesh scenes from a given camera pose. Used by every
domain-specific visualizer to produce video/image frames for logging.

Example::

    from umm.models.umm_v14.viz_helpers.mesh_visualizer import visualize_a_mesh
    rgb = visualize_a_mesh(cam_T_wc=np.eye(4), mesh=my_trimesh)
"""

import os
import threading
from typing import Optional, Tuple, Union

# Set environment variables for headless rendering BEFORE importing pyrender
# Try EGL first, fall back to osmesa if needed
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import trimesh


class MeshRenderer:
    """
    Lightweight mesh renderer optimized for training loops.
    Reuses renderer instances and minimizes allocations for efficiency.
    Designed to handle EGL/OSMesa contexts for headless rendering.
    """

    def __init__(self, width: int = 512, height: int = 512):
        """Initialize a MeshRenderer with the given viewport size.

        Args:
            width (int): Viewport width in pixels.
            height (int): Viewport height in pixels.

        Example::

            renderer = MeshRenderer(width=1024, height=768)
        """
        self.width = width
        self.height = height
        self._renderer = None
        self._lock = threading.Lock()  # Thread safety for training loops

    def _get_renderer(self):
        """Get or create the pyrender OffscreenRenderer (lazy init).

        Returns:
            pyrender.OffscreenRenderer: The renderer instance.

        Raises:
            RuntimeError: If EGL/OSMesa context creation fails.

        Example::

            r = mesh_renderer._get_renderer()
            rgb, depth = r.render(scene, flags=flags)
        """
        if self._renderer is None:
            try:
                self._renderer = pyrender.OffscreenRenderer(
                    viewport_width=self.width,
                    viewport_height=self.height,
                    point_size=1.0,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize OpenGL renderer: {e}. "
                    "Please ensure EGL/OSMesa is available."
                )
        return self._renderer

    def __del__(self):
        """Clean up renderer resources.

        Example::

            del mesh_renderer  # releases GPU/EGL resources
        """
        try:
            if self._renderer is not None:
                self._renderer.delete()
        except (ImportError, AttributeError):
            # Ignore cleanup errors during Python shutdown
            pass


# Global renderer instance for efficiency
_global_renderer = None
_renderer_lock = threading.Lock()


def get_global_renderer(width: int = 512, height: int = 512) -> MeshRenderer:
    """Get or create a process-global MeshRenderer singleton.

    If the existing renderer has a different viewport size, it is
    replaced with a new one.

    Args:
        width (int): Desired viewport width.
        height (int): Desired viewport height.

    Returns:
        MeshRenderer: The shared renderer instance.

    Example::

        renderer = get_global_renderer(1280, 720)
    """
    global _global_renderer
    with _renderer_lock:
        if (
            _global_renderer is None
            or _global_renderer.width != width
            or _global_renderer.height != height
        ):
            if _global_renderer is not None:
                del _global_renderer
            _global_renderer = MeshRenderer(width, height)
        return _global_renderer


def visualize_a_mesh(
    mesh_v: Optional[np.ndarray] = None,
    mesh_f: Optional[np.ndarray] = None,
    cam_T_wc: Optional[np.ndarray] = None,
    width: int = 512,
    height: int = 512,
    focal_length: Optional[Union[float, Tuple[float, float]]] = None,
    principal_point: Optional[Tuple[float, float]] = None,
    mesh_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    lighting: bool = True,
    mesh: Optional[trimesh.Trimesh] = None,
    draw_ground: bool = True,
    ground_z: float = 0.0,
    ground_color: Tuple[float, float, float, float] = (0.08, 0.08, 0.08, 1.0),
    ground_size: float = 50.0,  # Reduced to 50.0 for shadow precision
    ground_grid_size: float = 1.0,  # Grid spacing
    ground_grid_color: Optional[Tuple[float, float, float, float]] = None,
    ground_grid_line_width: Optional[float] = None,  # World-space width for grid lines
    yfov: float = np.pi / 3.0,
) -> np.ndarray:
    """
    Render a mesh from a specified camera pose.

    Args:
        mesh_v (np.ndarray, optional): Vertices [V, 3].
        mesh_f (np.ndarray, optional): Faces [F, 3].
        cam_T_wc (np.ndarray): Camera-to-world transform [4, 4].
        width (int): Image width.
        height (int): Image height.
        focal_length (float or tuple, optional): Focal length (fx, fy) or scalar.
        principal_point (tuple, optional): Principal point (cx, cy).
        mesh_color (tuple): RGBA color for the mesh.
        background_color (tuple): RGBA color for the background.
        lighting (bool): Enable lighting.
        mesh (trimesh.Trimesh, optional): Pre-existing Trimesh object.
        draw_ground (bool): Whether to add a ground plane to the scene.
        ground_z (float): Z height for the XY ground plane.
        ground_color (tuple): RGBA color for the ground plane material.
        ground_size (float): Size of the ground plate.
        ground_grid_size (float): Spacing of the grid lines.
        ground_grid_color (tuple, optional): Color of grid lines.
        ground_grid_line_width (float, optional): Width of grid lines.
        yfov (float): Field of view in Y direction.

    Returns:
        np.ndarray: Rendered RGB image [H, W, 3] (or RGBA depending on flags).
    """

    # Input validation - support dual mode
    mesh_from_vertices = False  # Track if mesh was created from vertices
    if mesh is not None:
        # Mode 1: mesh object provided
        tri_mesh = mesh
    else:
        # Mode 2: vertices and faces provided separately
        if mesh_v is None or mesh_f is None:
            raise ValueError(
                "Either 'mesh' must be provided, or both 'mesh_v' and 'mesh_f' must be provided"
            )
        assert mesh_v.shape[1] == 3, f"Vertices must be (V, 3), got {mesh_v.shape}"
        assert mesh_f.shape[1] == 3, f"Faces must be (F, 3), got {mesh_f.shape}"
        tri_mesh = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f)
        mesh_from_vertices = True

    if cam_T_wc is None:
        raise ValueError("cam_T_wc must be provided")
    assert cam_T_wc.shape == (4, 4), f"Transform must be (4, 4), got {cam_T_wc.shape}"

    # Get global renderer for efficiency
    renderer_obj = get_global_renderer(width, height)

    with renderer_obj._lock:  # Thread safety
        try:
            # Create material
            use_custom_material = mesh_from_vertices

            if not mesh_from_vertices:
                # For passed-in mesh objects, check if they have their own colors
                use_custom_material = True
                try:
                    # Check if mesh has vertex colors
                    visual = tri_mesh.visual
                    if hasattr(visual, "vertex_colors"):
                        vertex_colors = getattr(visual, "vertex_colors", None)
                        if vertex_colors is not None and len(vertex_colors) > 0:
                            use_custom_material = False
                    # Check if mesh has material
                    if use_custom_material and hasattr(visual, "material"):
                        mat = getattr(visual, "material", None)
                        if mat is not None:
                            use_custom_material = False
                except (AttributeError, TypeError):
                    use_custom_material = True

            # Create pyrender mesh
            if use_custom_material:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=mesh_color
                )
                pyrender_mesh = pyrender.Mesh.from_trimesh(
                    tri_mesh, material=material, smooth=True
                )
            else:
                pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

            # Create scene
            # Reduce ambient light to avoid washing out colors
            scene = pyrender.Scene(
                ambient_light=(0.02, 0.02, 0.02) if lighting else (0.1, 0.1, 0.1),
                bg_color=background_color,
            )

            # Add mesh to scene
            scene.add(pyrender_mesh, "mesh")

            # === GROUND DRAWING LOGIC ===
            if draw_ground:
                # 1. Very large ground plate
                plate_thickness = 1.0
                # Create a box centered at the origin
                ground_box = trimesh.creation.box(
                    extents=[ground_size, ground_size, plate_thickness]
                )

                # Align top face to ground_z
                ground_box.apply_translation(
                    [0.0, 0.0, ground_z - plate_thickness / 2.0]
                )

                # Rough, non-metallic material
                ground_material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    baseColorFactor=ground_color,
                )
                ground_mesh = pyrender.Mesh.from_trimesh(
                    ground_box, material=ground_material, smooth=False
                )
                scene.add(ground_mesh, name="ground")

                # 2. Grid Lines
                if ground_grid_size > 0:
                    # Decide grid color
                    if ground_grid_color is None:
                        grid_vis_color = (
                            min(ground_color[0] + 0.3, 1.0),
                            min(ground_color[1] + 0.3, 1.0),
                            min(ground_color[2] + 0.3, 1.0),
                            1.0,
                        )
                    else:
                        grid_vis_color = ground_grid_color

                    # We constrain the grid to a reasonable area to avoid generating millions of lines
                    # A 500 meter grid is usually sufficient for visual "infinity" in typical scenes
                    grid_limit = 500.0
                    extent = min(ground_size / 2.0, grid_limit)
                    spacing = ground_grid_size

                    # Number of lines
                    num_lines = int((extent * 2) / spacing)
                    coords = np.linspace(-extent, extent, num_lines + 1)

                    z_grid = ground_z + 0.005

                    if (
                        ground_grid_line_width is not None
                        and ground_grid_line_width > 0
                    ):
                        # Implement thick lines as flat ribbons (geometry)
                        w_half = ground_grid_line_width / 2.0

                        # We will construct vertices and faces directly
                        # Total quads = 2 * (num_lines + 1)
                        # Each quad has 4 vertices, 2 faces

                        # List collection is faster then stacking later
                        all_vertices = []
                        all_faces = []

                        v_idx = 0

                        # X-parallel lines (varying Y)
                        # Quad: (-extent, y-w), (extent, y-w), (extent, y+w), (-extent, y+w)
                        for y in coords:
                            # 4 corners
                            v0 = [-extent, y - w_half, z_grid]
                            v1 = [extent, y - w_half, z_grid]
                            v2 = [extent, y + w_half, z_grid]
                            v3 = [-extent, y + w_half, z_grid]

                            all_vertices.extend([v0, v1, v2, v3])
                            # 2 triangles: 0-1-2, 0-2-3
                            all_faces.extend(
                                [
                                    [v_idx, v_idx + 1, v_idx + 2],
                                    [v_idx, v_idx + 2, v_idx + 3],
                                ]
                            )
                            v_idx += 4

                        # Y-parallel lines (varying X)
                        for x in coords:
                            v0 = [x - w_half, -extent, z_grid]
                            v1 = [x + w_half, -extent, z_grid]
                            v2 = [x + w_half, extent, z_grid]
                            v3 = [x - w_half, extent, z_grid]

                            all_vertices.extend([v0, v1, v2, v3])
                            all_faces.extend(
                                [
                                    [v_idx, v_idx + 1, v_idx + 2],
                                    [v_idx, v_idx + 2, v_idx + 3],
                                ]
                            )
                            v_idx += 4

                        ribbon_mesh = trimesh.Trimesh(
                            vertices=np.array(all_vertices, dtype=np.float32),
                            faces=np.array(all_faces, dtype=np.int32),
                        )

                        # Ribbons need a material
                        ribbon_material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            roughnessFactor=1.0,
                            baseColorFactor=grid_vis_color,
                            doubleSided=True,
                        )
                        grid_mesh = pyrender.Mesh.from_trimesh(
                            ribbon_mesh, material=ribbon_material, smooth=False
                        )
                        scene.add(grid_mesh, name="grid_ribbons")

                    else:
                        # Use simple GL_LINES
                        bg_lines = []
                        # Lines parallel to X
                        for y in coords:
                            bg_lines.append([-extent, y, z_grid])
                            bg_lines.append([extent, y, z_grid])

                        # Lines parallel to Y
                        for x in coords:
                            bg_lines.append([x, -extent, z_grid])
                            bg_lines.append([x, extent, z_grid])

                        bg_lines = np.array(bg_lines, dtype=np.float32)
                        colors = np.tile(np.array(grid_vis_color), (len(bg_lines), 1))

                        grid_primitive = pyrender.Primitive(
                            positions=bg_lines,
                            mode=pyrender.constants.GLTF.LINES,
                            color_0=colors,
                        )
                        grid_mesh = pyrender.Mesh(
                            primitives=[grid_primitive], is_visible=True
                        )
                        scene.add(grid_mesh, name="grid")

            # === CAMERA LOGIC ===
            if focal_length is None:
                camera = pyrender.PerspectiveCamera(
                    yfov=yfov, aspectRatio=width / height
                )
            else:
                if isinstance(focal_length, (int, float)):
                    fx = fy = float(focal_length)
                else:
                    fx, fy = focal_length

                if principal_point is None:
                    cx, cy = width / 2.0, height / 2.0
                else:
                    cx, cy = principal_point

                camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

            # Camera Pose Conversion (OpenCV -> OpenGL)
            camera_pos = cam_T_wc[:3, 3]
            camera_rot = cam_T_wc[:3, :3]

            # OpenCV: +Z forward, +Y down, +X right
            # OpenGL: -Z forward, +Y up, +X right
            # Rotation change: 180 deg around X
            opencv_to_opengl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            opengl_rot = camera_rot @ opencv_to_opengl

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = opengl_rot
            camera_pose[:3, 3] = camera_pos

            if np.allclose(camera_pos, 0):
                camera_pose[:3, 3] = [0, 0, 3]

            scene.add(camera, pose=camera_pose)

            # === LIGHTING ===
            if lighting:
                # Moderate intensity to preserve color accuracy
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

                # Attached to camera
                scene.add(light, pose=camera_pose)

                # Fill lights
                lp2 = camera_pose.copy()
                lp2[:3, 3] += np.array([1.0, 1.0, 0.5])
                scene.add(light, pose=lp2)

                lp3 = camera_pose.copy()
                lp3[:3, 3] += np.array([-1.0, -1.0, 0.5])
                scene.add(light, pose=lp3)

                # Overhead light for shadows - reduced intensity
                overhead_light = pyrender.DirectionalLight(
                    color=[1.0, 1.0, 1.0], intensity=1.0
                )
                overhead_pose = np.eye(4)
                overhead_pose[:3, 3] = np.array([0.0, 0.0, ground_z + 10.0])
                # Pointing down (-Z)
                scene.add(overhead_light, pose=overhead_pose)

            # === RENDER ===
            renderer = renderer_obj._get_renderer()
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
            for shadow_flag_name in ("SHADOWS_DIRECTIONAL", "SHADOWS"):
                if hasattr(pyrender.RenderFlags, shadow_flag_name):
                    flags |= getattr(pyrender.RenderFlags, shadow_flag_name)
                    break

            result = renderer.render(scene, flags=flags)
            if result is None:
                raise RuntimeError("Renderer returned None")
            rgb, depth = result
            return rgb.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Mesh rendering failed: {str(e)}")
