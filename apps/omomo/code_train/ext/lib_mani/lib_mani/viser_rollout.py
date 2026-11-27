import os
import os.path as osp
import numpy as np
import viser
import viser.transforms as tf
import time
from lib_io import UnifiedIO
import argparse
import torch
from scipy.spatial.transform import Rotation
from lib_mani.utils.action_utils import compute_target_pose_for_frame

def colorize_segmentation(seg):
    unique_ids = np.unique(seg)
    # Use consistent hashing for colors based on UID
    # This ensures the same UID always gets the same color across frames
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    
    for uid in unique_ids:
        if uid == 0: continue # Skip background if needed, or color it consistently
        
        # Simple deterministic hash for color
        np.random.seed(int(uid))
        color = np.random.rand(3)
        
        mask = (seg == uid)
        seg_color[mask] = color
        
    return (seg_color * 255).astype(np.uint8)

class ManiSkillViser:
    def __init__(self, root_dir):
        self.root = osp.abspath(root_dir)
        self.io = UnifiedIO()
        self.server = viser.ViserServer()
        
        # Load embodiment meshes
        self.robot_meshes = self._load_meshes(osp.join(self.root, "robot_embodiment.npz"))
        self.object_meshes = self._load_meshes(osp.join(self.root, "object_embodiment.npz"))
        
        # Merge meshes
        self.all_meshes = {**self.robot_meshes, **self.object_meshes}

    def _load_meshes(self, path):
        if not osp.exists(path):
            print(f"Warning: {path} not found.")
            return {}
        data = np.load(path, allow_pickle=True)
        meshes = {}
        # Keys are {uid}_v, {uid}_f, {uid}_pcl, {uid}_pcl_c
        uids = set([k.split('_')[0] for k in data.keys()])
        for uid in uids:
            if f"{uid}_v" in data and f"{uid}_f" in data:
                meshes[int(uid)] = {
                    "v": data[f"{uid}_v"],
                    "f": data[f"{uid}_f"],
                    "c": data[f"{uid}_c"] if f"{uid}_c" in data else np.array([150, 150, 150], dtype=np.uint8)
                }
        return meshes

    def load_episode(self, episode_name):
        # Load all data for this episode
        print(f"Loading episode: {episode_name}")
        
        rgbs = self.io.load_rgbs(osp.join(self.root, "images", f"{episode_name}.zip"))
        depths = self.io.load_depths(osp.join(self.root, "depths", f"{episode_name}.zip"))
        segs, _ = self.io.load_inst_segs(osp.join(self.root, "instances", f"{episode_name}.zip"))
        
        states_data = self.io.load_npzs(osp.join(self.root, "states", f"{episode_name}.npz"))
        states = states_data["states"].item() # It's a dict
        actions = states_data["actions"]
        sem_type_uid = states_data["sem_type_uid"].item()
        
        cams_data = self.io.load_npzs(osp.join(self.root, "cameras", f"{episode_name}.npz"))
        T_wc = cams_data["T_wc"]
        K = cams_data["K"]
        
        return {
            "rgbs": rgbs,
            "depths": depths,
            "segs": segs,
            "states": states,
            "actions": actions,
            "sem_type_uid": sem_type_uid,
            "T_wc": T_wc,
            "K": K,
            "name": episode_name
        }

    def visualize(self, episode_name):
        data = self.load_episode(episode_name)
        num_frames = len(data["rgbs"])
        
        # Pre-load meshes in Viser
        # Identify all UIDs present in the episode states
        episode_uids = set([int(uid) for uid in data["states"].keys()])
        mesh_handles = {}
        ghost_handles = {} # For Target Visualization

        # Identify EE UIDs for Ghost Visualization
        ee_uids = []
        if "sem_type_uid" in data:
            stu = data["sem_type_uid"]
            # Collect all EE-related UIDs
            for key in ["endeffector", "gripper_left_finger", "gripper_left_pad", "gripper_right_finger", "gripper_right_pad"]:
                if key in stu:
                    ee_uids.extend(stu[key])
        ee_uids = set(ee_uids)
        
        # Create handles for all relevant meshes (initially hidden)
        for uid in episode_uids:
            if uid in self.all_meshes:
                mesh_info = self.all_meshes[uid]
                # Ensure color is int tuple
                color_tuple = tuple(map(int, mesh_info["c"].tolist()))
                
                # Normal Mesh
                mesh_handles[uid] = self.server.scene.add_mesh_simple(
                    f"/meshes/{uid}",
                    vertices=mesh_info["v"],
                    faces=mesh_info["f"],
                    color=color_tuple,
                    visible=False 
                )

                # Ghost Mesh (for EE parts)
                if uid in ee_uids:
                     ghost_handles[uid] = self.server.scene.add_mesh_simple(
                        f"/ghost/{uid}",
                        vertices=mesh_info["v"],
                        faces=mesh_info["f"],
                        color=(0, 255, 0), # Green
                        opacity=0.5,       # Transparent
                        visible=False
                    )
        
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            with client.gui.add_folder("Playback"):
                fr_slider = client.gui.add_slider("Frame", 0, num_frames - 1, 1, 0)
                play_btn = client.gui.add_button("Pause") # Default state is playing
            
            with client.gui.add_folder("Vis"):
                c_mode = client.gui.add_dropdown("Color", ["RGB", "Instance"], initial_value="RGB")
                
                # Granular Controls
                with client.gui.add_folder("Scene Elements"):
                    s_pcl = client.gui.add_checkbox("Point Cloud", True)
                    s_mesh = client.gui.add_checkbox("Robot/Object Meshes", True)
                    s_frames = client.gui.add_checkbox("Coordinate Frames", False)
                
                with client.gui.add_folder("PD Target Visualization"):
                    s_ghost = client.gui.add_checkbox("Show PD Target (Green)", True)
                    s_align = client.gui.add_checkbox("Align Target to Next Frame", True, hint="Show target computed from t-1 at time t (for comparison)")

            h_pcl = None
            h_frames = {}
            play = True # Auto-play by default

            import threading
            def playback_loop():
                while True:
                    if play:
                        fr_slider.value = (fr_slider.value + 1) % num_frames
                    time.sleep(0.05)
            
            # Start background thread for playback
            threading.Thread(target=playback_loop, daemon=True).start()

            @play_btn.on_click
            def _(_):
                nonlocal play
                play = not play
                play_btn.label = "Pause" if play else "Play"

            def update():
                nonlocal h_pcl
                t = fr_slider.value
                obs_rgb = data["rgbs"][t]
                obs_depth = data["depths"][t]
                obs_seg = data["segs"][t]
                obs_K = data["K"][t]
                obs_T_wc = data["T_wc"][t]
                
                # 1. Point Cloud
                if s_pcl.value:
                    # Scale depth to meters if it looks like mm
                    # Heuristic: if mean > 10, assume mm
                    depth_m = obs_depth.astype(np.float32)
                    if depth_m.mean() > 10.0:
                        depth_m /= 1000.0
                        
                    H, W = obs_depth.shape
                    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    xc = (u - obs_K[0, 2]) / obs_K[0, 0] * depth_m
                    yc = (v - obs_K[1, 2]) / obs_K[1, 1] * depth_m
                    zc = depth_m
                    
                    pc_cam = np.stack([xc, yc, zc, np.ones_like(zc)], axis=-1).reshape(-1, 4)
                    pc_world = (obs_T_wc @ pc_cam.T).T[:, :3]
                    
                    if c_mode.value == "RGB":
                        colors = obs_rgb.reshape(-1, 3)
                    else:
                        colors = colorize_segmentation(obs_seg).reshape(-1, 3)
                    
                    # Filter out zero depth
                    mask = depth_m.reshape(-1) > 0
                    h_pcl = client.scene.add_point_cloud("/pcl", pc_world[mask], colors[mask], point_size=0.002)
                elif h_pcl:
                    h_pcl.remove()
                    h_pcl = None

                # 2. Meshes (Visibility Toggle)
                if s_mesh.value:
                    for uid, handle in mesh_handles.items():
                        uid_str = str(uid)
                        if uid_str in data["states"]:
                            pose = data["states"][uid_str][t]
                            handle.visible = True
                            handle.position = pose[:3, 3]
                            handle.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
                        else:
                            handle.visible = False
                else:
                    for handle in mesh_handles.values():
                        handle.visible = False

                # 3. Ghost Meshes (Target)
                if s_ghost.value:
                    # Determine which action/state to use based on alignment setting
                    if s_align.value:
                        # "Align Target to Next Frame":
                        # We want to see the target that RESULTED in the current state (t).
                        # That target came from the action at (t-1).
                        # So we compute target using State(t-1) and Action(t-1).
                        
                        if t > 0:
                            prev_t = t - 1
                            prev_states = {k: v[prev_t] for k, v in data["states"].items()}
                            prev_action = data["actions"][prev_t]
                            
                            try:
                                target_poses = compute_target_pose_for_frame(
                                    prev_states, 
                                    prev_action, 
                                    data["sem_type_uid"]
                                )
                            except Exception as e:
                                print(f"Ghost compute error: {e}")
                                target_poses = {}
                        else:
                            # At t=0, there is no previous action that led here. Hide ghost.
                            target_poses = {}
                    else:
                        # "Current Target":
                        # Show where the current action (at t) is trying to go (for t+1).
                        # Use State(t) and Action(t).
                        curr_states = {k: v[t] for k, v in data["states"].items()}
                        curr_action = data["actions"][t]
                        
                        try:
                            target_poses = compute_target_pose_for_frame(
                                curr_states, 
                                curr_action, 
                                data["sem_type_uid"]
                            )
                        except Exception as e:
                            print(f"Ghost compute error: {e}")
                            target_poses = {}

                    # Update Ghost Visibility and Pose
                    for uid, handle in ghost_handles.items():
                        if uid in target_poses:
                            handle.visible = True
                            pose = target_poses[uid]
                            handle.position = pose[:3, 3]
                            handle.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
                        else:
                            handle.visible = False
                            
                else:
                    for handle in ghost_handles.values():
                        handle.visible = False

                # 4. Frames
                if s_frames.value:
                    for uid_str, poses in data["states"].items():
                        uid = int(uid_str)
                        pose = poses[t]
                        if uid not in h_frames:
                            h_frames[uid] = client.scene.add_frame(f"/frames/{uid}", axes_length=0.1, axes_radius=0.005)
                        h_frames[uid].visible = True
                        h_frames[uid].position = pose[:3, 3]
                        h_frames[uid].wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
                else:
                    for h in h_frames.values(): h.remove()
                    h_frames.clear()

            @fr_slider.on_update
            def _(_): update()
            @c_mode.on_update
            def _(_): update()
            @s_pcl.on_update
            def _(_): update()
            @s_mesh.on_update
            def _(_): update()
            @s_ghost.on_update
            def _(_): update()
            @s_align.on_update
            def _(_): update()
            @s_frames.on_update
            def _(_): update()
            
            update()

        print(f"Viser server started at {self.server.get_host()}:{self.server.get_port()}")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.server.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="test/maniskill_test_output", help="Data root")
    parser.add_argument("--episode", type=str, help="Episode name (e.g., PickCube-v1_traj_0)")
    args = parser.parse_args()
    
    vis = ManiSkillViser(args.root)
    
    episode = args.episode
    if episode is None:
        # Pick first episode from image dir
        img_dir = osp.join(args.root, "images")
        if not osp.exists(img_dir):
             print(f"Error: {img_dir} does not exist.")
             return
        episodes = [f.replace(".zip", "") for f in os.listdir(img_dir) if f.endswith(".zip")]
        if len(episodes) == 0:
            print("No episodes found.")
            return
        episode = episodes[0]
        
    vis.visualize(episode)

if __name__ == "__main__":
    main()
