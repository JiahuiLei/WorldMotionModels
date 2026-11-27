import sapien.core as sapien
import trimesh
import numpy as np
import os
import os.path as osp
import torch

# Semantic IDs as per make_langtable_data.py
SEMANTIC_ID_BACKGROUND = 0
SEMANTIC_ID_TABLE = 1
SEMANTIC_ID_ROBOT = 2
SEMANTIC_ID_OBJECTS = 3
SEMANTIC_ID_EE = 4

# EE Link names for Panda (common in ManiSkill)
EE_BASE_LINK = "panda_hand"
GRIPPER_LINKS = ["panda_leftfinger", "panda_rightfinger"]

def get_robot_meshes(env):
    """
    Returns meshes for the robot and end-effector.
    Dictionary keys are instance IDs.
    """
    return _get_meshes_internal(env, include_robot=True, include_objects=False)


def get_object_meshes(env):
    """
    Returns meshes for the objects.
    Dictionary keys are instance IDs.
    """
    return _get_meshes_internal(env, include_robot=False, include_objects=True)


def get_instance_meshes(env):
    """
    Extracts meshes for all relevant objects (robot + objects).
    """
    return _get_meshes_internal(env, include_robot=True, include_objects=True)


def _get_meshes_internal(env, include_robot=True, include_objects=True):
    instance_meshes = {}
    
    def process_obj(obj):
        # obj can be a ManiSkill Link/Actor or a SAPIEN component/Entity
        if hasattr(obj, "per_scene_id"):
            entity = obj
        elif hasattr(obj, "entity"):
            entity = obj.entity
        else:
            return

        instance_id = entity.per_scene_id
        if torch.is_tensor(instance_id):
            instance_id = instance_id.item()
        instance_id = int(instance_id)

        # Visual meshes
        render_shapes = []
        if hasattr(obj, "render_shapes"):
            render_shapes = obj.render_shapes
        else:
            # SAPIEN 3 Entity or Component
            components = entity.components if hasattr(entity, "components") else []
            for comp in components:
                if hasattr(comp, "render_shapes"):
                    render_shapes.extend(comp.render_shapes)

        for shape in render_shapes:
            # SAPIEN 3 RenderShape might have parts
            parts = shape.parts if hasattr(shape, "parts") else [shape]
            
            for part in parts:
                # SAPIEN 3 uses 'vertices' and 'triangles' (not 'indices')
                # Also check primitive shapes (box, sphere, etc.) which might not have 'vertices'
                if hasattr(part, "vertices") and hasattr(part, "triangles"):
                    vertices = part.vertices # (N, 3)
                    indices = part.triangles # (M, 3)
                else:
                    # Handle primitives if necessary, but for ManiSkill meshes are common
                    continue
                
                # Combine poses
                total_local_pose = shape.local_pose
                mat = total_local_pose.to_transformation_matrix()
                
                if torch.is_tensor(mat):
                    mat = mat.cpu().numpy()
                if mat.ndim == 3: # GPU sim
                    mat = mat[0]

                # Apply scale / primitive dimensions
                if hasattr(shape, "scale"):
                    scale = shape.scale
                    if torch.is_tensor(scale): scale = scale.cpu().numpy()
                    if scale.ndim == 2: scale = scale[0]
                    vertices = vertices * scale
                elif hasattr(shape, "half_size"):
                    hs = shape.half_size
                    if torch.is_tensor(hs): hs = hs.cpu().numpy()
                    if hs.ndim == 2: hs = hs[0]
                    vertices = vertices * hs
                elif hasattr(shape, "radius"):
                    r = shape.radius
                    if torch.is_tensor(r): r = r.cpu().numpy()
                    if isinstance(r, np.ndarray) and r.ndim >= 1: r = r.item() if r.size == 1 else r[0]
                    
                    if hasattr(shape, "half_length"):
                        hl = shape.half_length
                        if torch.is_tensor(hl): hl = hl.cpu().numpy()
                        if isinstance(hl, np.ndarray) and hl.ndim >= 1: hl = hl.item() if hl.size == 1 else hl[0]
                        # Cylinder or Capsule?
                        # In SAPIEN, primitive vertices for sphere/cylinder are usually unit sized
                        # Let's assume they need scaling.
                        vertices = vertices * np.array([r, r, hl])
                    else:
                        # Sphere
                        vertices = vertices * r

                mesh = trimesh.Trimesh(vertices=vertices, faces=indices.reshape(-1, 3))
                mesh.apply_transform(mat)

                # Extract color from material if possible
                material = None
                if hasattr(part, "material") and part.material is not None:
                    material = part.material
                elif hasattr(shape, "material"):
                    try:
                        material = shape.material
                    except:
                        pass
                
                if material is not None:
                    color = material.base_color
                    if torch.is_tensor(color): color = color.cpu().numpy()
                    color = np.array(color)
                    if color.ndim == 2: color = color[0]
                    mesh.visual.face_colors = (color[:3] * 255).astype(np.uint8)

                if instance_id in instance_meshes:
                    instance_meshes[instance_id] = trimesh.util.concatenate(
                        [instance_meshes[instance_id], mesh]
                    )
                else:
                    instance_meshes[instance_id] = mesh

    scene = env.unwrapped.scene
    
    if include_objects:
        for actor in scene.get_all_actors():
            # Check if actor has any render shapes
            has_visual = False
            for comp in actor.components:
                if hasattr(comp, "render_shapes") and len(comp.render_shapes) > 0:
                    has_visual = True
                    break
            
            if has_visual and actor.name not in ["ground", "plane", "surface"]:
                process_obj(actor)

    if include_robot:
        for articulation in scene.get_all_articulations():
            for link in articulation.get_links():
                process_obj(link)

    return instance_meshes


def save_robot_meshes(output_dir, env, num_points=16384):
    print(f"Extracting robot embodiment meshes (N={num_points})...")
    robot_meshes = get_robot_meshes(env)
    _save_meshes_to_npz(robot_meshes, osp.join(output_dir, "robot_embodiment.npz"), num_points)


def save_object_meshes(output_dir, env, num_points=2048):
    print(f"Extracting object meshes (N={num_points})...")
    object_meshes = get_object_meshes(env)
    _save_meshes_to_npz(object_meshes, osp.join(output_dir, "object_embodiment.npz"), num_points)


def _save_meshes_to_npz(meshes, save_path, num_points):
    data_dict = {}
    for uid, mesh in meshes.items():
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        data_dict[f"{uid}_v"] = np.array(mesh.vertices, dtype=np.float32)
        data_dict[f"{uid}_f"] = np.array(mesh.faces, dtype=np.int32)
        
        # Save color if available (take mean of face colors)
        if hasattr(mesh.visual, "face_colors"):
            colors = mesh.visual.face_colors
            if colors.ndim == 2:
                # Average face colors to get a single color for the part
                avg_color = colors.mean(axis=0)[:3].astype(np.uint8)
                data_dict[f"{uid}_c"] = avg_color
            else:
                data_dict[f"{uid}_c"] = colors[:3].astype(np.uint8)
        else:
            data_dict[f"{uid}_c"] = np.array([150, 150, 150], dtype=np.uint8)

        try:
            # Check if mesh is empty
            if len(mesh.vertices) == 0:
                 pcl = np.zeros((num_points, 3), dtype=np.float32)
            else:
                # Sample Point Cloud
                pcl, face_indices = trimesh.sample.sample_surface(mesh, num_points)
                if len(pcl) < num_points:
                    needed = num_points - len(pcl)
                    pcl_fill, face_indices_fill = trimesh.sample.sample_surface(mesh, needed)
                    pcl = np.vstack([pcl, pcl_fill])
                elif len(pcl) > num_points:
                    pcl = pcl[:num_points]

            data_dict[f"{uid}_pcl"] = np.array(pcl, dtype=np.float32)
            data_dict[f"{uid}_pcl_c"] = np.ones((num_points, 3), dtype=np.float32) * 0.5

        except Exception as e:
            print(f"Warning: Failed to sample surface for {uid}: {e}")
            data_dict[f"{uid}_pcl"] = np.zeros((num_points, 3), dtype=np.float32)
            data_dict[f"{uid}_pcl_c"] = np.zeros((num_points, 3), dtype=np.float32)

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **data_dict)
    print(f"Saved meshes to {save_path}")
