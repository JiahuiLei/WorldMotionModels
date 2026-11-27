"""Utilities to parse ManiSkill trajectories and recorded observations.

This module provides high-level functions to:
- Read original demonstration HDF5/JSON to extract joint-space proprioception and actions
- Read re-recorded HDF5 created by ManiSkill's RecordEpisode to extract observations (RGB/Depth/etc)
- Instantiate a single SAPIEN robot articulation for the environment (via Gym/ManiSkill)

Returned data is intended for use in offline visualization/animation workflows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import trimesh
import h5py
import numpy as np
from copy import deepcopy
import gymnasium as gym
import mani_skill.envs  # ! super important to import this after gym!!!!
from mani_skill.utils import common

# Check for torch availability
import torch

# -----------------------------
# Internal helpers
# -----------------------------


def _safe_numpy(x: Any) -> Any:
    """Convert h5py datasets to numpy arrays; leave scalars/others as-is."""
    if isinstance(x, h5py.Dataset):
        return x[()]
    return x


def _read_h5_group_recursive(group: h5py.Group) -> Dict[str, Any]:
    """Recursively read an HDF5 group into a nested Python dict of numpy arrays.

    Each dataset is converted to a numpy array. Groups become dicts.
    """
    out: Dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_h5_group_recursive(item)
        else:
            out[key] = _safe_numpy(item)
    return out


def _iter_nested_arrays(nested: Dict[str, Any], prefix: str = ""):
    """Yield tuples of (path, array) for all numpy arrays inside a nested dict."""
    for k, v in nested.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from _iter_nested_arrays(v, path)
        else:
            if isinstance(v, np.ndarray):
                yield path, v


def _find_first_matching_array(
    nested: Dict[str, Any], key_substrings: List[str]
) -> Optional[np.ndarray]:
    """Find the first array whose path contains any of the provided substrings."""
    lowered = [s.lower() for s in key_substrings]
    for path, arr in _iter_nested_arrays(nested):
        lp = path.lower()
        if any(s in lp for s in lowered):
            return arr
    return None


def _collect_all_matching_arrays(
    nested: Dict[str, Any], key_substrings: List[str]
) -> List[Tuple[str, np.ndarray]]:
    """Collect all arrays whose path contains any of the provided substrings."""
    results: List[Tuple[str, np.ndarray]] = []
    lowered = [s.lower() for s in key_substrings]
    for path, arr in _iter_nested_arrays(nested):
        lp = path.lower()
        if any(s in lp for s in lowered):
            results.append((path, arr))
    return results


def _default_json_path_from_h5(h5_path: str) -> str:
    return h5_path.replace(".h5", ".json")


def _load_json(json_path: str) -> Dict[str, Any]:
    # Lightweight JSON loader; ManiSkill also provides io_utils, but avoid importing internal utils here
    import json

    with open(json_path, "r") as f:
        return json.load(f)


def _pick_episode_id(json_data: Dict[str, Any], episode_index: int) -> int:
    episodes = json_data.get("episodes", [])
    if not episodes:
        raise ValueError("No episodes listed in JSON metadata.")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"episode_index {episode_index} out of range [0, {len(episodes)-1}]"
        )
    return int(episodes[episode_index]["episode_id"])  # ensure int


def _get_traj_group(h5_file: h5py.File, episode_id: int) -> h5py.Group:
    key = f"traj_{episode_id}"
    if key not in h5_file:
        # Fallback: if trajectory ids are contiguous, try first/only group
        keys = [k for k in h5_file.keys() if k.startswith("traj_")]
        raise KeyError(f"Trajectory group '{key}' not found. Available: {keys}")
    return h5_file[key]


# -----------------------------
# Public data structures
# -----------------------------


@dataclass
class RobotArticulationHandle:
    """A thin handle to the environment and its robot articulation.

    Keep this object alive while animating; it holds references to the Gym env.
    """

    env: Any
    articulation: Any

    def set_qpos(self, joint_positions: np.ndarray) -> None:
        """Set joint positions on the underlying articulation, best-effort across versions."""
        art = self.articulation
        if hasattr(art, "set_qpos"):
            art.set_qpos(joint_positions)
            return
        if hasattr(art, "set_joint_positions"):
            art.set_joint_positions(joint_positions)
            return
        # Fallback: try to set via joints
        if hasattr(art, "get_active_joints") and callable(art.get_active_joints):
            joints = art.get_active_joints()
            n = min(len(joints), int(joint_positions.shape[0]))
            for i in range(n):
                if hasattr(joints[i], "set_qpos"):
                    joints[i].set_qpos(float(joint_positions[i]))

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


# -----------------------------
# Core parsing functions
# -----------------------------


def parse_proprio_and_actions(
    h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Parse proprioception, states, and actions from ManiSkill trajectory HDF5.

    Extracts qpos, qvel from obs/agent and tcp_pose from obs/extra.
    Falls back to env_states for qpos/qvel if not found in obs.
    Optionally loads full env_states as 'states' (flattened vector).

    Args:
        load_states: If True, includes full env_states as 'states' in proprioception

    Returns (proprioception, actions) where proprioception contains:
    - 'qpos', 'qvel' (if available), 'tcp_pose' (if available)
    - 'states' (if load_states=True): full env_states flattened into single vector
    """
    # Auto-detect episode_id if no JSON provided
    if json_path is None:
        json_path = _default_json_path_from_h5(h5_path)

    episode_id = episode_index  # Default to episode_index as episode_id
    if json_path and os.path.exists(json_path):
        try:
            json_data = _load_json(json_path)
            episode_id = _pick_episode_id(json_data, episode_index)
        except Exception:
            # If JSON fails, fall back to using episode_index directly
            episode_id = episode_index

    with h5py.File(h5_path, "r") as f:
        # Try to get trajectory group, fallback to episode_index if needed
        try:
            traj = _get_traj_group(f, episode_id)
        except KeyError:
            # If episode_id doesn't work, try episode_index directly
            episode_id = episode_index
            traj = _get_traj_group(f, episode_id)

        # Actions
        if "actions" not in traj:
            raise KeyError(f"'actions' dataset not found in group traj_{episode_id}")
        converted_actions = np.array(traj["actions"])  # shape (T, A')
        actions = np.array(traj["original_actions"])  # shape (T, A)

        # Initialize
        qpos: Optional[np.ndarray] = None
        qvel: Optional[np.ndarray] = None
        tcp_pose: Optional[np.ndarray] = None
        states: Optional[np.ndarray] = None

        # Extract from obs structure (preferred)
        if "obs" in traj:
            obs_group = traj["obs"]
            # Get qpos/qvel from obs/agent
            if "agent" in obs_group:
                agent_group = obs_group["agent"]
                if "qpos" in agent_group:
                    qpos = np.array(agent_group["qpos"])
                if "qvel" in agent_group:
                    qvel = np.array(agent_group["qvel"])

            # Get tcp_pose from obs/extra
            if "extra" in obs_group:
                extra_group = obs_group["extra"]
                if "tcp_pose" in extra_group:
                    tcp_pose = np.array(extra_group["tcp_pose"])

        # make states
        if len(traj["obs"]) > 0:
            state_dict = _read_h5_group_recursive(traj["obs"])
            state_dict.pop("sensor_data")
            state_dict.pop("sensor_param")
            state_dict["extra"] = {
                k: v[:, None] if len(v.shape) == 1 else v
                for k, v in state_dict["extra"].items()
            }  # dirty fix for data that has one dimension (e.g. is_grasped)
            states = common.flatten_state_dict(state_dict, use_torch=True)

    # Assemble results
    proprio: Dict[str, np.ndarray] = {}
    if qpos is not None:
        proprio["qpos"] = np.asarray(qpos)
    if qvel is not None:
        proprio["qvel"] = np.asarray(qvel)
    if tcp_pose is not None:
        proprio["tcp_pose"] = np.asarray(tcp_pose)
    if states is not None:
        proprio["states"] = np.asarray(states)

    return proprio, actions, converted_actions


def list_env_state_arrays(
    original_h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
) -> List[Tuple[str, Tuple[int, ...]]]:
    """List available arrays under traj_X with their shapes.

    Returns a list of (path, shape) tuples to help users identify qpos/qvel.
    Includes both obs/agent and env_states arrays.
    """
    # Auto-detect episode_id if no JSON provided
    if json_path is None:
        json_path = _default_json_path_from_h5(original_h5_path)

    episode_id = episode_index  # Default to episode_index as episode_id
    if json_path and os.path.exists(json_path):
        try:
            json_data = _load_json(json_path)
            episode_id = _pick_episode_id(json_data, episode_index)
        except Exception:
            episode_id = episode_index

    with h5py.File(original_h5_path, "r") as f:
        # Try to get trajectory group, fallback to episode_index if needed
        try:
            traj = _get_traj_group(f, episode_id)
        except KeyError:
            episode_id = episode_index
            traj = _get_traj_group(f, episode_id)

        listed: List[Tuple[str, Tuple[int, ...]]] = []

        # Check obs/agent structure (newer format)
        if "obs" in traj and "agent" in traj["obs"]:
            agent_nested = _read_h5_group_recursive(traj["obs"]["agent"])  # type: ignore[arg-type]
            for path, arr in _iter_nested_arrays(agent_nested):
                try:
                    shape = tuple(int(x) for x in arr.shape)
                except Exception:
                    shape = tuple()
                listed.append((f"obs/agent/{path}", shape))

        # Check env_states structure (older format)
        if "env_states" in traj:
            env_states_nested = _read_h5_group_recursive(traj["env_states"])  # type: ignore[arg-type]
            for path, arr in _iter_nested_arrays(env_states_nested):
                try:
                    shape = tuple(int(x) for x in arr.shape)
                except Exception:
                    shape = tuple()
                listed.append((f"env_states/{path}", shape))

    listed.sort(key=lambda x: x[0])
    return listed


def parse_recorded_observations(
    recorded_h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Parse recorded observations (RGB/Depth/etc) from a re-recorded HDF5.

    Args:
        recorded_h5_path: Path to the re-recorded HDF5 file
        json_path: Optional path to JSON metadata file
        episode_index: Episode index to parse (default: 0)

    Returns:
        List of dicts (per timestep). Each dict maps an observation key
        like 'camera/rgb' or 'camera/depth' to a numpy array for that timestep.
        If camera parameters exist, they are included as 'camera_intrinsics' and 'camera_extrinsics'.
    """
    # Auto-detect episode_id if no JSON provided
    if json_path is None:
        json_path = _default_json_path_from_h5(recorded_h5_path)

    episode_id = episode_index  # Default to episode_index as episode_id
    if json_path and os.path.exists(json_path):
        try:
            json_data = _load_json(json_path)
            episode_id = _pick_episode_id(json_data, episode_index)
        except Exception:
            episode_id = episode_index

    with h5py.File(recorded_h5_path, "r") as f:
        # Try to get trajectory group, fallback to episode_index if needed
        try:
            traj = _get_traj_group(f, episode_id)
        except KeyError:
            episode_id = episode_index
            traj = _get_traj_group(f, episode_id)

        # Try standard locations first
        observation_nested: Optional[Dict[str, Any]] = None
        if "observation" in traj:
            observation_nested = _read_h5_group_recursive(traj["observation"])  # type: ignore[arg-type]
        elif "observations" in traj:
            observation_nested = _read_h5_group_recursive(traj["observations"])  # type: ignore[arg-type]
        else:
            # Some datasets may store observations at the root of the trajectory group
            observation_nested = _read_h5_group_recursive(traj)

    # Collect all arrays that look like images/depths
    image_like_keys = [
        "rgb",
        "rgba",
        "depth",
        "image",
        "seg",
        "semantic",
        "normal",
    ]
    image_arrays = _collect_all_matching_arrays(observation_nested, image_like_keys)  # type: ignore[arg-type]

    if not image_arrays:
        # As a fallback, include everything; downstream can filter
        image_arrays = list(_iter_nested_arrays(observation_nested))  # type: ignore[arg-type]

    # Determine time length from the first dimensions of arrays
    time_lengths = [arr.shape[0] for _, arr in image_arrays if arr.ndim >= 1]
    if not time_lengths:
        return []
    T = int(max(time_lengths))

    # Always try to extract camera parameters
    camera_params = parse_camera_parameters(
        recorded_h5_path=recorded_h5_path,
        json_path=json_path,
        episode_index=episode_index,
    )

    per_timestep: List[Dict[str, np.ndarray]] = []
    for t in range(T):
        obs_t: Dict[str, np.ndarray] = {}

        # Add image observations
        for path, arr in image_arrays:
            if arr.ndim == 0:
                continue
            idx = min(t, arr.shape[0] - 1)
            obs_t[path] = np.asarray(arr[idx])

        # Add camera parameters if available
        if camera_params:
            camera_intrinsics = {}
            camera_extrinsics = {}

            for cam_name, cam_data in camera_params.items():
                if "intrinsics" in cam_data:
                    t_idx = min(t, cam_data["intrinsics"].shape[0] - 1)
                    camera_intrinsics[cam_name] = cam_data["intrinsics"][t_idx]
                if "extrinsics" in cam_data:
                    t_idx = min(t, cam_data["extrinsics"].shape[0] - 1)
                    camera_extrinsics[cam_name] = cam_data["extrinsics"][t_idx]

            if camera_intrinsics:
                obs_t["camera_intrinsics"] = camera_intrinsics
            if camera_extrinsics:
                obs_t["camera_extrinsics"] = camera_extrinsics

        per_timestep.append(obs_t)

    return per_timestep


def parse_converted_actions(
    recorded_h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
    *,
    key_hints: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Parse converted actions (e.g., end-effector space) from a recorded HDF5.

    This searches common locations and key names inside the recorded trajectory
    for arrays named like 'converted_action'. Returns None if not found.
    """
    # Auto-detect episode_id if no JSON provided
    if json_path is None:
        json_path = _default_json_path_from_h5(recorded_h5_path)

    episode_id = episode_index  # Default to episode_index as episode_id
    if json_path and os.path.exists(json_path):
        try:
            json_data = _load_json(json_path)
            episode_id = _pick_episode_id(json_data, episode_index)
        except Exception:
            episode_id = episode_index

    with h5py.File(recorded_h5_path, "r") as f:
        # Try to get trajectory group, fallback to episode_index if needed
        try:
            traj = _get_traj_group(f, episode_id)
        except KeyError:
            episode_id = episode_index
            traj = _get_traj_group(f, episode_id)

        # Most recorded demos store converted actions directly under 'actions'
        if "actions" in traj:
            return np.array(traj["actions"])  # (T, converted_dim)

        # Fallback: Read entire trajectory group to search flexibly
        traj_nested = _read_h5_group_recursive(traj)

    # Compose hints
    default_hints = [
        "converted_action",
        "converted_actions",
        "ee_action",
        "end_effector_action",
        "end_effector/actions",
        "action/end_effector",
        "actions/end_effector",
        "controller_action",
    ]
    search_hints = (key_hints or []) + default_hints

    arr = _find_first_matching_array(traj_nested, search_hints)
    if arr is None:
        return None
    return np.asarray(arr)


def parse_camera_parameters(
    recorded_h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Parse camera intrinsics and extrinsics from a re-recorded HDF5 file.

    Returns a dict of camera parameters if available, None if not found.
    Structure: {camera_name: {'intrinsics': np.ndarray, 'extrinsics': np.ndarray}}

    - intrinsics: Camera intrinsic matrix K, shape (T, 3, 3) or (T, 4, 4)
    - extrinsics: Camera-to-world transformation matrix, shape (T, 4, 4)
    """
    # Auto-detect episode_id if no JSON provided
    if json_path is None:
        json_path = _default_json_path_from_h5(recorded_h5_path)

    episode_id = episode_index  # Default to episode_index as episode_id
    if json_path and os.path.exists(json_path):
        try:
            json_data = _load_json(json_path)
            episode_id = _pick_episode_id(json_data, episode_index)
        except Exception:
            episode_id = episode_index

    with h5py.File(recorded_h5_path, "r") as f:
        # Try to get trajectory group, fallback to episode_index if needed
        try:
            traj = _get_traj_group(f, episode_id)
        except KeyError:
            episode_id = episode_index
            traj = _get_traj_group(f, episode_id)

        # Check if camera_params group exists
        if "camera_params" not in traj:
            return None

        camera_params_group = traj["camera_params"]
        camera_data: Dict[str, Dict[str, np.ndarray]] = {}

        for camera_name in camera_params_group.keys():
            camera_group = camera_params_group[camera_name]
            camera_info: Dict[str, np.ndarray] = {}

            # Extract intrinsics if available
            if "intrinsics" in camera_group:
                camera_info["intrinsics"] = np.array(camera_group["intrinsics"])

            # Extract extrinsics if available
            if "extrinsics" in camera_group:
                camera_info["extrinsics"] = np.array(camera_group["extrinsics"])

            # Only include cameras that have at least one parameter
            if camera_info:
                camera_data[camera_name] = camera_info

        return camera_data if camera_data else None


def build_sapien_robot_articulation(
    env_json_path: str,
) -> RobotArticulationHandle:
    """Instantiate the environment and return a handle to its robot articulation.

    The articulation is created using the same `env_id` and `env_kwargs` stored in
    the trajectory JSON's `env_info`. Keep the returned handle alive while using
    the articulation; call `handle.close()` when done.
    """
    if not os.path.exists(env_json_path):
        raise FileNotFoundError(
            f"JSON file not found: {env_json_path}. Cannot create robot articulation without environment info."
        )

    json_data = _load_json(env_json_path)
    env_info = json_data.get("env_info", {})
    env_id = env_info.get("env_id")
    env_kwargs = env_info.get("env_kwargs", {})
    if not env_id:
        raise ValueError("env_id missing in JSON env_info")

    # Use CPU sim by default unless specified
    if "sim_backend" not in env_kwargs:
        env_kwargs["sim_backend"] = "physx_cpu"

    env = gym.make(env_id, **env_kwargs)
    env.reset()

    base = env.unwrapped
    articulation = None

    # Try common accessors across ManiSkill versions
    for attr in ("agent", "robot", "articulation"):
        if hasattr(base, attr):
            candidate = getattr(base, attr)
            # Some have agent.articulation
            if hasattr(candidate, "articulation"):
                articulation = getattr(candidate, "articulation")
                break
            # Or the attr itself is an articulation
            if hasattr(candidate, "set_qpos") or hasattr(
                candidate, "get_active_joints"
            ):
                articulation = candidate
                break

    if articulation is None:
        raise RuntimeError(
            "Could not locate robot articulation on the environment. "
            "Please adjust accessors in build_sapien_robot_articulation()."
        )

    return RobotArticulationHandle(env=env, articulation=articulation)


# -----------------------------
# Action-to-QPos mapping discovery
# -----------------------------


def discover_action_qpos_mapping(env: Any, verbose: bool = True) -> Tuple[
    Dict[str, Any],
    Callable[[Union[np.ndarray, Any]], Union[np.ndarray, Any]],
    Callable[[Union[np.ndarray, Any]], Union[np.ndarray, Any]],
]:
    """
    Dynamically discover how actions map to qpos by analyzing the environment.

    Args:
        env: The ManiSkill environment instance
        verbose: Whether to print discovery information

    Returns:
        Tuple of:
        - mapping_info: Dictionary with discovered mapping information
        - map_action_to_qpos_target: Function that converts actions to qpos targets
        - map_qpos_to_action: Function that converts qpos back to actions
    """
    # Extract dimensions directly from environment
    # Get action dimension from action space
    try:
        action_dim = env.action_space.shape[0]
    except (AttributeError, IndexError):
        try:
            action_dim = env.action_space.n  # discrete action space
        except AttributeError:
            raise ValueError(
                "Could not determine action dimension from environment action space!"
            )

    # Get qpos dimension from robot's active joints
    qpos_dim = None
    robot = None

    if verbose:
        print("🔧 DISCOVERING ACTION-TO-QPOS MAPPING FROM SAPIEN")
        print("=" * 55)
        print(f"🤖 Environment type: {type(env)}")
        print(f"   Action dimensionality: {action_dim}")

    # Get robot and active joints info to determine qpos_dim
    qpos_dim = None
    active_joints = []
    try:
        base = env.unwrapped

        # First try to get robot from agent (most common case)
        if hasattr(base, "agent") and hasattr(base.agent, "robot"):
            robot = base.agent.robot
            if hasattr(robot, "get_active_joints"):
                active_joints = robot.get_active_joints()
                qpos_dim = len(active_joints)
                if verbose:
                    print(f"   Robot type: {type(robot)}")
                    print(f"   Active joints: {len(active_joints)}")
                    print(f"   QPos dimensionality: {qpos_dim}")

        # Fallback: try direct attributes on environment
        if qpos_dim is None:
            for attr in ("robot", "articulation"):
                if hasattr(base, attr):
                    candidate = getattr(base, attr)
                    if hasattr(candidate, "get_active_joints"):
                        active_joints = candidate.get_active_joints()
                        qpos_dim = len(active_joints)
                        if verbose:
                            print(f"   Found robot via {attr}: {type(candidate)}")
                            print(f"   Active joints: {len(active_joints)}")
                            print(f"   QPos dimensionality: {qpos_dim}")
                        break

    except Exception as e:
        if verbose:
            print(f"   Warning: Could not get robot info: {e}")

    # Fallback: try to get qpos_dim from observation space if robot method failed
    if qpos_dim is None:
        try:
            obs_space = env.observation_space
            if verbose:
                print(f"   Trying observation space fallback: {type(obs_space)}")

            # Look for qpos in observation space (if it's a Dict space)
            if hasattr(obs_space, "spaces") and isinstance(obs_space.spaces, dict):
                if "agent" in obs_space.spaces and hasattr(
                    obs_space.spaces["agent"], "spaces"
                ):
                    agent_obs = obs_space.spaces["agent"].spaces
                    if "qpos" in agent_obs:
                        qpos_dim = agent_obs["qpos"].shape[0]
                        if verbose:
                            print(f"   QPos dimensionality from obs space: {qpos_dim}")
                elif "qpos" in obs_space.spaces:
                    qpos_dim = obs_space.spaces["qpos"].shape[0]
                    if verbose:
                        print(f"   QPos dimensionality from obs space: {qpos_dim}")

            # Alternative: try to get a sample observation to infer qpos
            if qpos_dim is None:
                try:
                    sample_obs, _ = env.reset()
                    if hasattr(sample_obs, "shape"):
                        if verbose:
                            print(f"   Sample obs shape: {sample_obs.shape}")
                        # For simple Box observation spaces, we can't easily infer qpos dim
                        # This is why we need the robot method to work
                except Exception as obs_e:
                    if verbose:
                        print(f"   Could not get sample observation: {obs_e}")

        except Exception as e:
            if verbose:
                print(f"   Warning: Could not get qpos_dim from observation space: {e}")

    if qpos_dim is None:
        raise ValueError(
            "Could not determine qpos dimension from environment! "
            "Unable to find robot active joints or qpos in observation space."
        )

    mapping_info = {
        "action_dim": action_dim,
        "qpos_dim": qpos_dim,
        "action_to_qpos_indices": None,
        "mapping_method": "unknown",
    }

    # Method 1: Try to get mapping from controller
    try:
        if hasattr(env, "agent") and hasattr(env.agent, "controller"):
            controller = env.agent.controller
            if verbose:
                print(f"🎮 Controller type: {type(controller)}")

            # Look for direct mapping attributes
            for attr_name in [
                "joint_indices",
                "qpos_indices",
                "action_to_qpos_mapping",
                "controlled_joint_indices",
            ]:
                if hasattr(controller, attr_name):
                    indices = getattr(controller, attr_name)
                    if indices is not None:
                        mapping_info["action_to_qpos_indices"] = (
                            list(indices) if hasattr(indices, "__iter__") else indices
                        )
                        mapping_info["mapping_method"] = f"controller.{attr_name}"
                        if verbose:
                            print(f"   ✅ Found mapping via {attr_name}: {indices}")
                        break

            # Look for controlled joints info
            if mapping_info["action_to_qpos_indices"] is None and hasattr(
                controller, "controlled_joints"
            ):
                controlled_joints = controller.controlled_joints
                joint_to_qpos_map = []
                for controlled_joint in controlled_joints:
                    # Find this joint's index in all active joints
                    for i, active_joint in enumerate(active_joints):
                        if controlled_joint == active_joint:
                            joint_to_qpos_map.append(i)
                            break

                if joint_to_qpos_map:
                    mapping_info["action_to_qpos_indices"] = joint_to_qpos_map
                    mapping_info["mapping_method"] = "controlled_joints_analysis"
                    if verbose:
                        print(
                            f"   ✅ Derived mapping from controlled joints: {joint_to_qpos_map}"
                        )

    except Exception as e:
        if verbose:
            print(f"   ⚠️  Controller method failed: {e}")

    # Method 2: Pattern-based inference
    if mapping_info["action_to_qpos_indices"] is None:
        if verbose:
            print(f"   🧮 Attempting pattern analysis...")

        if action_dim == qpos_dim:
            # 1:1 mapping
            mapping_info["action_to_qpos_indices"] = list(range(action_dim))
            mapping_info["mapping_method"] = "1to1_direct"
            if verbose:
                print(
                    f"   ✅ Assuming 1:1 mapping: {mapping_info['action_to_qpos_indices']}"
                )

        elif action_dim < qpos_dim:
            # Most common case: action controls subset of joints, with some coupling
            basic_mapping = list(range(action_dim))

            # For the common gripper case, duplicate last action for remaining joints
            if qpos_dim - action_dim <= 2:  # Likely gripper (1-2 extra joints)
                # Last action controls remaining joints (gripper coupling)
                for extra_joint in range(action_dim, qpos_dim):
                    basic_mapping.append(action_dim - 1)  # Point to last action

                mapping_info["action_to_qpos_indices"] = basic_mapping
                mapping_info["mapping_method"] = "gripper_coupling_inference"
                if verbose:
                    print(
                        f"   ✅ Inferred coupling mapping: {mapping_info['action_to_qpos_indices']}"
                    )
            else:
                if verbose:
                    print(f"   ⚠️  Complex coupling detected, using partial mapping")
                mapping_info["action_to_qpos_indices"] = basic_mapping
                mapping_info["mapping_method"] = "partial_mapping"

        else:  # action_dim > qpos_dim
            if verbose:
                print(f"   ⚠️  Action dim > QPos dim: using truncated mapping")
            mapping_info["action_to_qpos_indices"] = list(range(qpos_dim))
            mapping_info["mapping_method"] = "truncated_mapping"

    if verbose:
        print(f"\n📊 Final mapping info:")
        for key, value in mapping_info.items():
            print(f"   {key}: {value}")

    # Create the mapper function
    def map_action_to_qpos_target(
        action_input: Union[np.ndarray, Any],
    ) -> Union[np.ndarray, Any]:
        """
        Convert actions to qpos targets using discovered mapping.
        Handles both numpy arrays and torch tensors, preserving input type.
        Works with arbitrary prefix shapes: (..., action_dim) -> (..., qpos_dim)
        """
        if mapping_info["action_to_qpos_indices"] is None:
            raise ValueError(
                "Could not determine action-to-qpos mapping! Please specify manually."
            )

        # Determine input type and convert to numpy for processing
        is_torch_tensor = hasattr(action_input, "dtype") and str(
            action_input.dtype
        ).startswith("torch")
        original_device = None

        if is_torch_tensor:
            original_device = action_input.device
            action_np = action_input.detach().cpu().numpy()
        else:
            action_np = np.asarray(action_input)

        # Get the shape info
        prefix_shape = action_np.shape[:-1]  # All dimensions except the last one
        action_dim_actual = action_np.shape[-1]

        # Flatten prefix dimensions for easier processing
        action_flat = action_np.reshape(-1, action_dim_actual)
        batch_size = action_flat.shape[0]

        # Create target qpos
        qpos_target_flat = np.zeros(
            (batch_size, mapping_info["qpos_dim"]), dtype=action_np.dtype
        )

        # Apply the mapping
        action_indices = mapping_info["action_to_qpos_indices"]

        for qpos_idx, action_idx in enumerate(action_indices):
            if action_idx < action_dim_actual:
                qpos_target_flat[:, qpos_idx] = action_flat[:, action_idx]
            else:
                # This shouldn't happen with proper mapping, but handle gracefully
                qpos_target_flat[:, qpos_idx] = 0.0

        # Reshape back to original prefix shape + qpos_dim
        target_shape = prefix_shape + (mapping_info["qpos_dim"],)
        qpos_target = qpos_target_flat.reshape(target_shape)

        # Convert back to original type if needed
        if is_torch_tensor:
            qpos_target = torch.from_numpy(qpos_target).to(original_device)

        return torch.as_tensor(qpos_target)

    def map_qpos_to_action(
        qpos_input: Union[np.ndarray, Any],
    ) -> Union[np.ndarray, Any]:
        """
        Convert qpos back to actions using discovered mapping (inverse operation).
        Handles both numpy arrays and torch tensors, preserving input type.
        Works with arbitrary prefix shapes: (..., qpos_dim) -> (..., action_dim)
        """
        if mapping_info["action_to_qpos_indices"] is None:
            raise ValueError(
                "Could not determine action-to-qpos mapping! Please specify manually."
            )

        # Determine input type and convert to numpy for processing
        is_torch_tensor = hasattr(qpos_input, "dtype") and str(
            qpos_input.dtype
        ).startswith("torch")
        original_device = None

        if is_torch_tensor:
            original_device = qpos_input.device
            qpos_np = qpos_input.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos_input)

        # Get the shape info
        prefix_shape = qpos_np.shape[:-1]  # All dimensions except the last one
        qpos_dim_actual = qpos_np.shape[-1]

        # Flatten prefix dimensions for easier processing
        qpos_flat = qpos_np.reshape(-1, qpos_dim_actual)
        batch_size = qpos_flat.shape[0]

        # Create target actions
        action_target_flat = np.zeros(
            (batch_size, mapping_info["action_dim"]), dtype=qpos_np.dtype
        )

        # Apply the inverse mapping
        action_indices = mapping_info["action_to_qpos_indices"]

        # Create reverse mapping: from action index to qpos indices
        action_to_qpos_map = {}
        for qpos_idx, action_idx in enumerate(action_indices):
            if action_idx not in action_to_qpos_map:
                action_to_qpos_map[action_idx] = []
            action_to_qpos_map[action_idx].append(qpos_idx)

        # Extract actions from qpos
        for action_idx in range(mapping_info["action_dim"]):
            if action_idx in action_to_qpos_map:
                qpos_indices = action_to_qpos_map[action_idx]
                # If multiple qpos indices map to same action (like gripper coupling),
                # use the first one as representative
                qpos_idx = qpos_indices[0]
                if qpos_idx < qpos_dim_actual:
                    action_target_flat[:, action_idx] = qpos_flat[:, qpos_idx]
                else:
                    # This shouldn't happen with proper mapping, but handle gracefully
                    action_target_flat[:, action_idx] = 0.0

        # Reshape back to original prefix shape + action_dim
        target_shape = prefix_shape + (mapping_info["action_dim"],)
        action_target = action_target_flat.reshape(target_shape)

        # Convert back to original type if needed
        if is_torch_tensor:
            action_target = torch.from_numpy(action_target).to(original_device)

        return torch.as_tensor(action_target)

    return mapping_info, map_action_to_qpos_target, map_qpos_to_action


# -----------------------------
# High-level convenience
# -----------------------------
def episode_index_list(json_path: str) -> List[int]:
    """List all episode indices in the JSON file."""
    if json_path.endswith(".h5"):
        json_path = _default_json_path_from_h5(json_path)
    json_data = _load_json(json_path)
    episodes = json_data.get("episodes", [])
    return [int(episode["episode_id"]) for episode in episodes]

# def episode_index_list(recorded_h5_path: str) -> List[int]:
#     with h5py.File(recorded_h5_path, "r") as f:
#         episode_ids = [it for it in f.keys() if it.startswith("traj_")]
#     episode_ids = [int(it.split("_")[-1]) for it in episode_ids]
#     episode_ids.sort()
#     return episode_ids


def extract_demo(
    recorded_h5_path: str,
    json_path: Optional[str] = None,
    episode_index: int = 0,
) -> Tuple[
    List[Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
]:
    """Extract observations, proprioception, and actions from trajectory files.

    Args:
        original_h5_path: Path to original trajectory HDF5 file
        recorded_h5_path: Path to re-recorded HDF5 file with observations
        json_path: Optional path to JSON metadata file
        episode_index: Episode index to parse (default: 0)
        qpos_path: Optional explicit path to qpos data
        qvel_path: Optional explicit path to qvel data
        qpos_key_hints: Optional list of key hints for finding qpos
        qvel_key_hints: Optional list of key hints for finding qvel
        load_states: If True, includes full env_states as flattened 'states' in proprioception

    Returns:
        Tuple of (observations_list, proprioception, actions, converted_actions):
        1) observations_list: List[Dict[str, np.ndarray]] per timestep (RGB/Depth/etc + camera params if available)
        2) proprioception: Dict with 'qpos', 'qvel', 'tcp_pose', 'states' (if available)
        3) actions: np.ndarray of shape (T, action_dim) from original HDF5
        4) converted_actions: Optional[np.ndarray] from recorded HDF5 if available

        Camera parameters are automatically included in observations_list as 'camera_intrinsics' and 'camera_extrinsics' if available.
    """
    if json_path is None:
        json_path = _default_json_path_from_h5(recorded_h5_path)

    # _, actions = parse_proprio_and_actions(
    #     h5_path=original_h5_path,
    #     json_path=json_path,
    #     episode_index=episode_index,
    # )

    proprio, actions, converted_actions = parse_proprio_and_actions(
        h5_path=recorded_h5_path,
        json_path=json_path,
        episode_index=episode_index,
    )
    observations = parse_recorded_observations(
        recorded_h5_path=recorded_h5_path,
        json_path=_default_json_path_from_h5(recorded_h5_path),
        episode_index=episode_index,
    )
    # converted_actions = parse_converted_actions(
    #     recorded_h5_path=recorded_h5_path,
    #     json_path=_default_json_path_from_h5(recorded_h5_path),
    #     episode_index=episode_index,
    # )

    return observations, proprio, actions, converted_actions


def get_sapien_robot(env_id: str = "StackCube-v1"):
    """Get a SAPIEN robot instance that you can use in your own code.

    Args:
        env_id: ManiSkill environment ID

    Returns:
        Tuple of (robot_articulation, environment) - keep env alive to use robot!
    """
    env = gym.make(env_id, sim_backend="physx_cpu")
    env.reset()

    # Get robot from agent
    base = env.unwrapped
    agent = base.agent
    robot = agent.robot  # This is the SAPIEN Articulation

    print(f"✅ SAPIEN robot obtained: {type(robot)}")
    print(f"🔗 Active joints: {len(robot.get_active_joints())}")
    print(f"📦 Links: {len(robot.get_links())}")
    print(f"⚠️  Keep the environment alive to use the robot!")

    return robot, env  # Return both so user can close env when done


def extract_link_meshes(robot: Any, exclude_names: List[str] = None):
    """Extract visual geometry (trimesh) for each link that has visual geometry.

    Args:
        robot: SAPIEN robot articulation object
        exclude_names: List of strings. If a link name contains any of these strings,
                      the mesh will be ignored and None will be returned for that link.

    Returns:
        List of tuples with same length as robot links: [(link_name, trimesh_geometry_or_None), ...]
        Each tuple contains (link_name, mesh) where mesh is a trimesh object or None.
    """
    if exclude_names is None:
        exclude_names = []

    ret = []

    try:
        # Get all links from the robot
        links = robot.get_links()

        for link in links:
            link_name = getattr(link, "name", f"link_{id(link)}")

            # Check if this link should be excluded
            should_exclude = any(
                exclude_str in link_name for exclude_str in exclude_names
            )

            if should_exclude:
                ret.append((link_name, None))
                continue

            try:
                # Try generate_mesh method which works more reliably
                def accept_all_filter(link, render_shape):
                    return True

                generated_meshes = link.generate_mesh(
                    filter=accept_all_filter, mesh_name=f"{link_name}_mesh"
                )

                if generated_meshes and len(generated_meshes) > 0:
                    # generated_meshes is a list of trimesh objects
                    if len(generated_meshes) == 1:
                        combined_mesh = generated_meshes[0]
                    else:
                        # Import trimesh here to avoid dependency issues if not needed
                        try:
                            import trimesh

                            combined_mesh = trimesh.util.concatenate(generated_meshes)
                        except ImportError:
                            # If trimesh not available, just use the first mesh
                            combined_mesh = generated_meshes[0]

                    ret.append((link_name, combined_mesh))
                else:
                    # No meshes generated, append None
                    ret.append((link_name, None))

            except Exception as e:
                # If generate_mesh fails, the link likely has no visual geometry
                # This is normal for some links (like joints without visual representation)
                ret.append((link_name, None))

    except Exception as e:
        print(f"Warning: Failed to extract link meshes from robot: {e}")

    return ret


def sample_link_pcls(link_meshes, N_pcl: int = 16384):
    ret = []
    for link_name, mesh in link_meshes:
        if mesh is None:
            ret.append(np.zeros((N_pcl, 3)))
        else:
            sampled = np.zeros((0, 3))
            while len(sampled) < N_pcl:
                _sampled, _ = trimesh.sample.sample_surface_even(mesh, N_pcl)
                sampled = np.concatenate([sampled, _sampled], axis=0)
            sampled = sampled[:N_pcl]
            # randomly shuffle
            perm = np.random.permutation(len(sampled))
            sampled = sampled[perm]
            ret.append(sampled)
    return np.stack(ret, axis=0).astype(np.float16)


__all__ = [
    "parse_proprio_and_actions",
    "parse_recorded_observations",
    "parse_converted_actions",
    "parse_camera_parameters",
    "build_sapien_robot_articulation",
    "extract_demo",
    "list_env_state_arrays",
    "get_sapien_robot",
    "discover_action_qpos_mapping",
    "extract_link_meshes",
    "RobotArticulationHandle",
]
