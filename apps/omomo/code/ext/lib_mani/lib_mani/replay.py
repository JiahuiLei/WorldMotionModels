import os
import os.path as osp
import numpy as np
import h5py
import json
import time
from tqdm import tqdm
import gymnasium as gym
import torch
from mani_skill.utils.common import to_tensor
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply, quaternion_invert
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
from lib_io import UnifiedIO
from .mesh_utils import (
    get_robot_meshes, 
    get_object_meshes, 
    SEMANTIC_ID_BACKGROUND,
    SEMANTIC_ID_TABLE,
    SEMANTIC_ID_ROBOT,
    SEMANTIC_ID_OBJECTS,
    SEMANTIC_ID_EE,
    EE_BASE_LINK,
    GRIPPER_LINKS
)

from .utils import load_h5_recursive


def quaternion_invert(q):
    """Invert a quaternion (assuming normalized). q: [w, x, y, z]"""
    # conjugate = [w, -x, -y, -z]
    if isinstance(q, torch.Tensor):
        q_inv = q.clone()
        q_inv[..., 1:] *= -1
        return q_inv
    elif isinstance(q, np.ndarray):
        q_inv = q.copy()
        q_inv[..., 1:] *= -1
        return q_inv
    else:
        raise ValueError("Unsupported type for quaternion_invert")
        
from pytorch3d.transforms import matrix_to_euler_angles as p3d_matrix_to_euler_angles

def matrix_to_euler_angles(matrix, convention="XYZ"):
    """
    Convert rotation matrix to Euler angles using pytorch3d.
    """
    # Ensure tensor
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
    
    # Handle batch dimension
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
        
    euler = p3d_matrix_to_euler_angles(matrix, convention)
    
    if single:
        return euler[0]
    return euler

class ManiSkillSaver:
    def __init__(self, root):
        self.root = osp.abspath(root)
        self.img_dir = osp.join(self.root, "images")
        self.dep_dir = osp.join(self.root, "depths")
        self.inst_dir = osp.join(self.root, "instances")
        self.state_dir = osp.join(self.root, "states")
        self.cam_dir = osp.join(self.root, "cameras")
        self.text_dir = osp.join(self.root, "text")
        self.meta_dir = osp.join(self.root, "meta")
        self.packed_dir = osp.join(self.root, "packed")

        for d in [self.img_dir, self.dep_dir, self.inst_dir, self.state_dir, 
                  self.cam_dir, self.text_dir, self.meta_dir, self.packed_dir]:
            os.makedirs(d, exist_ok=True)

        self.io = UnifiedIO()

    def save_episode(self, name, episode_data, fps=20.0):
        # episode_data is a list of frames, each frame is a dict
        rgbs = [f["rgb"] for f in episode_data]
        self.io.save_rgbs(rgbs, osp.join(self.img_dir, f"{name}.zip"))

        deps = [f["depth"] for f in episode_data]
        self.io.save_depths(deps, osp.join(self.dep_dir, f"{name}.zip"))

        inst_segs = [f["segmentation"] for f in episode_data]
        inst_anno = self.io.save_inst_segs(inst_segs, osp.join(self.inst_dir, f"{name}.zip"))

        # sem_type_uid mapping
        sem_type_uid = episode_data[0]["sem_type_uid"]
        all_uids = episode_data[0]["all_uids"]

        states = {
            str(uid): np.stack([f["instance_poses"][uid] for f in episode_data])
            for uid in all_uids if uid in episode_data[0]["instance_poses"]
        }

        actions = np.stack([f["action"] for f in episode_data])
        
        self.io.save_npzs(
            osp.join(self.state_dir, f"{name}.npz"),
            states=states,
            actions=actions,
            sem_type_uid=sem_type_uid,
            all_uids=all_uids,
            fps=float(fps),
        )

        T_wc = np.stack([f["camera_pose"] for f in episode_data])
        K = np.stack([f["camera_intrinsics"] for f in episode_data])
        self.io.save_npzs(osp.join(self.cam_dir, f"{name}.npz"), T_wc=T_wc, K=K)

        self.io.save_text(episode_data[0]["instruction"], osp.join(self.text_dir, f"{name}.txt"))

        self.io.save_npzs(
            osp.join(self.meta_dir, f"{name}.npz"),
            seed=episode_data[0].get("seed", 0),
            env_id=episode_data[0].get("env_id", ""),
        )

        # Packed version
        # Include all parts of EE and objects
        interesting_uids = (
            sem_type_uid.get("objects", []) + 
            sem_type_uid.get("endeffector", []) +
            sem_type_uid.get("gripper_left_finger", []) +
            sem_type_uid.get("gripper_left_pad", []) +
            sem_type_uid.get("gripper_right_finger", []) +
            sem_type_uid.get("gripper_right_pad", [])
        )
        u, v = np.meshgrid(np.arange(rgbs[0].shape[1]), np.arange(rgbs[0].shape[0]))
        uv = np.stack([u, v], axis=-1).astype(np.int16)
        
        packed = []
        packed_meta = []
        for t in range(len(episode_data)):
            frame = {}
            frame_meta = {}
            _rgb = rgbs[t]
            _depth = deps[t]
            _inst = inst_segs[t]
            
            for uid in interesting_uids:
                mask = _inst == uid
                cnt = mask.sum()
                if cnt > 0:
                    frame_meta[uid] = int(cnt)
                    frame[uid] = {
                        "rgb": _rgb[mask].astype(np.uint8),
                        "depth": _depth[mask].astype(np.float16),
                        "uv": uv[mask].copy().astype(np.int16),
                    }
            packed.append(frame)
            packed_meta.append(frame_meta)

        self.io.save_packed_zip(packed, packed_meta, osp.join(self.packed_dir, f"{name}.zip"))


class ReplayHandler:
    def __init__(self, env_id):
        self.env_id = env_id

    def _convert_action_to_ee_delta(self, env, original_action, device):
        """
        Convert any action to PD EE Delta Pose format [dx, dy, dz, drx, dry, drz, gripper]
        Assuming Panda robot for limits.
        """
        agent = env.unwrapped.agent
        if 'arm' not in agent.controller.controllers:
            return original_action
            
        arm_ctrl = agent.controller.controllers['arm']
        gripper_ctrl = agent.controller.controllers['gripper']
        
        # Ensure action is tensor
        if isinstance(original_action, np.ndarray):
            original_action_t = torch.from_numpy(original_action).float().to(device)
        else:
            original_action_t = original_action.float().to(device)
            
        if original_action_t.ndim == 1:
            original_action_t = original_action_t.unsqueeze(0)
            
        # Identify gripper part based on dimensions
        total_dim = original_action_t.shape[-1]
        
        # Default: Keep original if we can't figure it out
        new_action = original_action_t
        
        if isinstance(arm_ctrl, PDJointPosController) and total_dim == 8:
            # Case: Joint Delta -> EE Delta
            
            # Save state
            original_qpos = agent.robot.get_qpos()
            
            # Compute Target Joint Positions
            # Feed full action to controller to get target
            agent.controller.set_action(original_action_t)
            target_qpos_arm = arm_ctrl._target_qpos
            target_qpos_gripper = gripper_ctrl._target_qpos # usually mimic
            
            # Gripper Action (Keep absolute/normalized as is)
            # For Panda, gripper is last dim
            gripper_action_raw = original_action_t[:, 7:]
            
            # FK for Target EE Pose
            full_target_qpos = original_qpos.clone()
            full_target_qpos[..., arm_ctrl.active_joint_indices] = target_qpos_arm
            full_target_qpos[..., gripper_ctrl.active_joint_indices] = target_qpos_gripper
            
            agent.robot.set_qpos(full_target_qpos)
            target_ee_pose = agent.tcp_pose
            
            # Restore
            agent.robot.set_qpos(original_qpos)
            
            # Current EE Pose
            current_ee_pose = agent.tcp_pose # or arm_ctrl.ee_pose_at_base which accounts for root offset if any
            
            # Compute Delta: Target - Current
            # Using Panda limits for normalization: pos=0.1, rot=0.1
            POS_LIMIT = 0.1
            ROT_LIMIT = 0.1
            
            # Position Delta
            delta_pos = target_ee_pose.p - current_ee_pose.p
            
            # Rotation Delta (Euler XYZ)
            # q_target = q_delta * q_prev -> q_delta = q_target * q_prev^-1 (for "root_translation:root_aligned_body_rotation")
            q_target = target_ee_pose.q
            q_current = current_ee_pose.q
            q_current_inv = quaternion_invert(q_current)
            q_delta = quaternion_multiply(q_target, q_current_inv)
            
            # Convert q_delta to Matrix then Euler
            mat_delta = quaternion_to_matrix(q_delta)
            euler_delta = matrix_to_euler_angles(mat_delta, "XYZ").to(device)
            
            # Normalize
            norm_delta_pos = torch.clamp(delta_pos / POS_LIMIT, -1.0, 1.0)
            norm_delta_rot = torch.clamp(euler_delta / ROT_LIMIT, -1.0, 1.0)
            
            # Combine
            new_action = torch.cat([norm_delta_pos, norm_delta_rot, gripper_action_raw], dim=-1)

        return new_action.cpu().numpy()[0]

    def replay_trajectory(self, traj_path, save_dir, count=None):
        # Determine control mode from metadata
        control_mode = "pd_joint_delta_pos" # Default
        json_path = traj_path.replace(".h5", ".json")
        if osp.exists(json_path):
            with open(json_path, "r") as f:
                import json
                meta = json.load(f)
                if "env_info" in meta and "env_kwargs" in meta["env_info"]:
                    control_mode = meta["env_info"]["env_kwargs"].get("control_mode", control_mode)
        
        print(f"Replaying with control_mode: {control_mode}")

        with h5py.File(traj_path, "r") as f:
            traj_keys = list(f.keys())
            if count is not None:
                # Sort keys to be deterministic
                traj_keys = sorted([k for k in traj_keys if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
                traj_keys = traj_keys[:count]

            # Use composite obs_mode to get everything in one go
            env = gym.make(self.env_id, obs_mode="rgb+depth+segmentation", control_mode=control_mode)
            saver = ManiSkillSaver(save_dir)

            for key in tqdm(traj_keys, desc=f"Replaying {self.env_id}"):
                group = f[key]
                actions = group["actions"][:]
                
                # Load all env states if available
                env_states = None
                if "env_states" in group:
                    env_states = load_h5_recursive(group["env_states"])
                
                # Setup environment to initial state
                if env_states is not None:
                    # Extract state at t=0
                    def get_state_at_t(d, t):
                        if isinstance(d, dict):
                            return {k: get_state_at_t(v, t) for k, v in d.items()}
                        # d is numpy array (T, ...)
                        v = to_tensor(d[t])
                        if v.ndim == 1: v = v.unsqueeze(0)
                        return v
                    
                    initial_state = get_state_at_t(env_states, 0)
                    env.reset()
                    env.unwrapped.scene.set_sim_state(initial_state)
                elif "seed" in group:
                    env.reset(seed=int(group["seed"][()]))
                else:
                    env.reset()

                episode_data = []
                
                # Get camera name
                cam_name = "base_camera" 
                if cam_name not in env.unwrapped.scene.sensors:
                    cam_name = list(env.unwrapped.scene.sensors.keys())[0]

                # Identify robot links for categorization
                agent = env.unwrapped.agent
                robot = agent.robot
                links = robot.get_links()
                
                link_map = {l.name: int(l.per_scene_id.item() if torch.is_tensor(l.per_scene_id) else l.per_scene_id) for l in links}
                
                # Categorize UIDs - Detailed EE parts
                sem_type_uid = {
                    "background": [SEMANTIC_ID_BACKGROUND],
                    "table": [SEMANTIC_ID_TABLE],
                    "robotbody": [],
                    "objects": [],
                    "endeffector": [],
                    "gripper_left_finger": [],
                    "gripper_left_pad": [],
                    "gripper_right_finger": [],
                    "gripper_right_pad": []
                }
                
                all_uids = []
                
                # Populate robot links
                for name, uid in link_map.items():
                    all_uids.append(uid)
                    if name == EE_BASE_LINK:
                        sem_type_uid["endeffector"].append(uid)
                    elif "leftfinger" in name:
                        if "pad" in name:
                            sem_type_uid["gripper_left_pad"].append(uid)
                        else:
                            sem_type_uid["gripper_left_finger"].append(uid)
                    elif "rightfinger" in name:
                        if "pad" in name:
                            sem_type_uid["gripper_right_pad"].append(uid)
                        else:
                            sem_type_uid["gripper_right_finger"].append(uid)
                    else:
                        sem_type_uid["robotbody"].append(uid)
                
                # Identify objects and other entities
                for actor in env.unwrapped.scene.get_all_actors():
                    uid = int(actor.per_scene_id.item() if torch.is_tensor(actor.per_scene_id) else actor.per_scene_id)
                    if uid not in all_uids:
                        all_uids.append(uid)
                        
                        has_visual = False
                        for comp in actor.components:
                            if hasattr(comp, "render_shapes") and len(comp.render_shapes) > 0:
                                has_visual = True
                                break
                        
                        if has_visual:
                            # Check background keywords
                            is_background = any(k in actor.name for k in ["ground", "plane", "surface"])
                            
                            if "table" in actor.name or "workspace" in actor.name:
                                sem_type_uid["table"].append(uid)
                            elif is_background:
                                sem_type_uid["background"].append(uid)
                            else:
                                sem_type_uid["objects"].append(uid)

                for t in range(len(actions)):
                    # Force state if available
                    if env_states is not None:
                        state_t = get_state_at_t(env_states, t)
                        env.unwrapped.scene.set_sim_state(state_t)
                    
                    # Capture observation
                    obs = env.unwrapped.get_obs()
                    
                    # Process observation
                    sensor_data = obs["sensor_data"][cam_name]
                    sensor_param = obs["sensor_param"][cam_name]
                    
                    rgb = sensor_data["rgb"][0].cpu().numpy()
                    depth = sensor_data["depth"][0, ..., 0].cpu().numpy()
                    seg = sensor_data["segmentation"][0, ..., 0].cpu().numpy().astype(np.uint32)
                    
                    # Convert GL camera pose to CV camera pose
                    T_gl_cv = np.array([
                        [1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]
                    ])
                    cam2world_gl = sensor_param["cam2world_gl"][0].cpu().numpy()
                    cam2world_cv = cam2world_gl @ T_gl_cv
                    
                    # Convert action to EE Delta if needed (ALWAYS, as requested)
                    try:
                        final_action = self._convert_action_to_ee_delta(env, actions[t], env.device)
                    except Exception as e:
                        # Fallback if conversion fails
                        print(f"Warning: Action conversion failed at step {t}: {e}")
                        final_action = actions[t]

                    frame = {
                        "rgb": rgb,
                        "depth": depth,
                        "segmentation": seg,
                        "action": final_action,
                        "camera_pose": cam2world_cv,
                        "camera_intrinsics": sensor_param["intrinsic_cv"][0].cpu().numpy(),
                        "instruction": "Task: " + self.env_id,
                        "sem_type_uid": sem_type_uid,
                        "all_uids": all_uids,
                        "instance_poses": {},
                        "env_id": self.env_id,
                        "seed": int(group.get("seed", [0])[0]) if "seed" in group else 0
                    }
                    
                    # Capture all instance poses
                    for actor in env.unwrapped.scene.get_all_actors():
                        uid = int(actor.per_scene_id.item() if torch.is_tensor(actor.per_scene_id) else actor.per_scene_id)
                        mat = actor.pose.to_transformation_matrix()
                        if torch.is_tensor(mat): mat = mat.cpu().numpy()[0]
                        frame["instance_poses"][uid] = mat
                    for articulation in env.unwrapped.scene.get_all_articulations():
                        for link in articulation.get_links():
                            entity = link.entity
                            uid = int(entity.per_scene_id.item() if torch.is_tensor(entity.per_scene_id) else entity.per_scene_id)
                            mat = link.pose.to_transformation_matrix()
                            if torch.is_tensor(mat): mat = mat.cpu().numpy()[0]
                            frame["instance_poses"][uid] = mat

                    episode_data.append(frame)
                    
                    # If we don't have env_states, we step
                    if env_states is None:
                        env.step(actions[t])

                saver.save_episode(f"{self.env_id}_{key}", episode_data)

            env.close()
