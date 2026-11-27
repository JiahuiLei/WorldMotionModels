# https://raw.githubusercontent.com/haosulab/ManiSkill/refs/heads/main/mani_skill/trajectory/replay_trajectory.py

# ! this is the debug version, try to add 1. depth 2. all camera posese 3. segmentation
# todo: 4. scene mesh per frame

"""Replay ManiSkill trajectories stored in HDF5 (.h5) format

The replayed trajectory can use different observation modes and control modes.

We support translating actions from certain controllers to a limited number of controllers.

The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import copy
import gc
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


def safe_convert(matrix):
    if hasattr(matrix, "cpu"):
        result = matrix.cpu().numpy()
    elif hasattr(matrix, "numpy"):
        result = matrix.numpy()
    else:
        result = np.array(matrix)

    # Remove unnecessary batch dimension (1, H, W) -> (H, W)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    return result


def force_memory_cleanup():
    """Force garbage collection and CUDA memory cleanup to prevent memory leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_camera_params(env):
    """Extract REAL camera intrinsics and extrinsics from ManiSkill environment.

    Returns:
        dict: Dictionary containing camera parameters for each camera
            Format: {camera_name: {'intrinsic': K_matrix, 'extrinsic': pose_matrix}}
    """
    camera_params = {}

    try:
        # ONLY extract from verified ManiSkill environment - NO FALLBACKS
        base_env = getattr(env, "unwrapped", env)

        # ManiSkill stores cameras in _sensors
        if not hasattr(base_env, "_sensors"):
            logger.warning(
                "Environment has no _sensors - cannot extract camera parameters"
            )
            return {}

        cameras = {k: v for k, v in base_env._sensors.items() if "camera" in k.lower()}
        if not cameras:
            logger.warning("No cameras found in environment sensors")
            return {}

        for camera_name, camera in cameras.items():
            # ONLY use ManiSkill's verified get_params() method - NO OTHER METHODS
            params = camera.get_params()
            intrinsic = params["intrinsic_cv"]
            extrinsic = camera.config.pose.to_transformation_matrix()
            # ! this is in sapien format
            # find the mounted point # ! this need to be call every step because camera moves!!!!
            if camera.entity is not None:
                T_wl = camera.entity.pose.to_transformation_matrix()
                extrinsic = T_wl @ extrinsic

            clean_intrinsic = safe_convert(intrinsic)
            clean_extrinsic = safe_convert(extrinsic)

            camera_params[camera_name] = {
                "intrinsic": clean_intrinsic,
                "extrinsic": clean_extrinsic,
            }
    except Exception as e:
        logger.error(f"Failed to extract camera parameters: {e}")
        return {}
    return camera_params


def setup_camera_resolution(env_kwargs, camera_width, camera_height):
    """Setup camera resolution PROPERLY in ManiSkill environment kwargs.

    Args:
        env_kwargs: Environment configuration dictionary
        camera_width: Desired camera width
        camera_height: Desired camera height
    """
    if camera_width is not None and camera_height is not None:
        # Set camera resolution in sensor_configs - the PROPER ManiSkill way
        if "sensor_configs" not in env_kwargs:
            env_kwargs["sensor_configs"] = {}

        # Set resolution parameters that ManiSkill understands
        env_kwargs["sensor_configs"]["width"] = camera_width
        env_kwargs["sensor_configs"]["height"] = camera_height

        logger.info(
            f"Set camera resolution in env_kwargs: {camera_width}x{camera_height}"
        )


def save_camera_params_to_h5(h5_file, traj_group, camera_params_sequence):
    """Save camera parameters sequence to h5 file with memory optimization.

    Args:
        h5_file: Open h5py file handle
        traj_group: Trajectory group in h5 file
        camera_params_sequence: List of camera params for each timestep
    """
    if not camera_params_sequence:
        return

    # Create camera_params group
    if "camera_params" in traj_group:
        del traj_group["camera_params"]
    camera_group = traj_group.create_group("camera_params")

    # Get all camera names from first non-empty timestep
    first_params = None
    for params in camera_params_sequence:
        if params:
            first_params = params
            break

    if not first_params:
        logger.warning("No camera parameters found in any timestep")
        return

    camera_names = list(first_params.keys())

    # Process one camera at a time to reduce memory usage
    for camera_name in camera_names:
        cam_group = camera_group.create_group(camera_name)

        # Collect intrinsics and extrinsics across all timesteps
        intrinsics_list = []
        extrinsics_list = []

        for t, timestep_params in enumerate(camera_params_sequence):
            if camera_name in timestep_params:
                params = timestep_params[camera_name]
                if params.get("intrinsic") is not None:
                    intrinsics_list.append(params["intrinsic"])
                if params.get("extrinsic") is not None:
                    extrinsics_list.append(params["extrinsic"])

        # Save as datasets with immediate cleanup
        if intrinsics_list:
            intrinsics_array = np.stack(intrinsics_list, axis=0)
            cam_group.create_dataset("intrinsics", data=intrinsics_array)
            del intrinsics_array  # Explicitly free memory
            del intrinsics_list   # Free the list as well

        if extrinsics_list:
            extrinsics_array = np.stack(extrinsics_list, axis=0)
            cam_group.create_dataset("extrinsics", data=extrinsics_array)
            del extrinsics_array  # Explicitly free memory
            del extrinsics_list   # Free the list as well
        
        # Force garbage collection after each camera to free memory
        gc.collect()

    logger.info(
        f"Saved camera parameters for {len(camera_names)} cameras across {len(camera_params_sequence)} timesteps"
    )


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""

    # ! jiahui lei
    # New features for resolution control and camera data
    camera_width: Optional[int] = None
    """Width resolution for camera images. If specified, will override default camera resolution."""
    camera_height: Optional[int] = None
    """Height resolution for camera images. If specified, will override default camera resolution."""
    save_camera_params: bool = False
    """Whether to save per-frame camera intrinsics K and extrinsic pose matrices to the h5 file."""


@dataclass
class ReplayResult:
    num_replays: int
    successful_replays: int


def sanity_check_and_format_seed(episode):
    """sanity checks the trajectory seed aligns with the episode seed. reformats the reset kwargs seed if missing or formatted wrong"""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):

            assert (
                len(episode["reset_kwargs"]["seed"]) == 1
            ), f"found multiple seeds for one trajectory (id={episode['episode_id']}) in the reset kwargs which means it is ambiguous which seed to use"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert (
            episode["reset_kwargs"]["seed"] == episode["episode_seed"]
        ), f"found mismatch between trajectory seed and episode seed (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def extract_camera_params_batch(env, num_envs):
    """Extract camera parameters from all environments in a parallelized batch.
    
    Args:
        env: The parallelized environment 
        num_envs: Number of environments in the batch
        
    Returns:
        List of camera parameter dicts, one for each environment
    """
    batch_camera_params = []
    
    try:
        # Get the base environment which handles multiple parallel environments
        base_env = getattr(env, "unwrapped", env)
        
        # For each environment in the batch
        for env_idx in range(num_envs):
            camera_params = {}
            
            # Check if _sensors exists
            if not hasattr(base_env, "_sensors"):
                logger.warning(f"Environment {env_idx} has no _sensors - cannot extract camera parameters")
                batch_camera_params.append({})
                continue
                
            cameras = {k: v for k, v in base_env._sensors.items() if "camera" in k.lower()}
            if not cameras:
                logger.warning(f"No cameras found in environment {env_idx} sensors")
                batch_camera_params.append({})
                continue
                
            for camera_name, camera in cameras.items():
                try:
                    # Get camera parameters - in batched mode, we need to extract for specific env index
                    params = camera.get_params()
                    
                    # For batched environments, intrinsics might be batched too
                    intrinsic = params["intrinsic_cv"]
                    if hasattr(intrinsic, "shape") and len(intrinsic.shape) > 2:
                        # If batched, extract for this environment
                        intrinsic = intrinsic[env_idx] if env_idx < intrinsic.shape[0] else intrinsic[0]
                    
                    # Get extrinsic matrix
                    extrinsic = camera.config.pose.to_transformation_matrix()
                    if hasattr(extrinsic, "shape") and len(extrinsic.shape) > 2:
                        # If batched, extract for this environment  
                        extrinsic = extrinsic[env_idx] if env_idx < extrinsic.shape[0] else extrinsic[0]
                    
                    # Handle mounted cameras
                    if camera.entity is not None:
                        T_wl = camera.entity.pose.to_transformation_matrix()
                        if hasattr(T_wl, "shape") and len(T_wl.shape) > 2:
                            # If batched, extract for this environment
                            T_wl = T_wl[env_idx] if env_idx < T_wl.shape[0] else T_wl[0]
                        extrinsic = T_wl @ extrinsic
                        
                    clean_intrinsic = safe_convert(intrinsic)
                    clean_extrinsic = safe_convert(extrinsic)
                    
                    camera_params[camera_name] = {
                        "intrinsic": clean_intrinsic,
                        "extrinsic": clean_extrinsic,
                    }
                except Exception as e:
                    logger.warning(f"Failed to extract camera {camera_name} params for env {env_idx}: {e}")
                    continue
                    
            batch_camera_params.append(camera_params)
            
    except Exception as e:
        logger.error(f"Failed to extract batch camera parameters: {e}")
        # Return empty dicts for all environments
        batch_camera_params = [{} for _ in range(num_envs)]
        
    return batch_camera_params


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    pbar.reset(total=len(episodes))
    warned_reset_kwargs_options = False
    # split all episodes into batches of args.num_envs environments and process each batch in parallel, truncating where necessary
    # add fake episode padding to the end of the episodes to make sure all batches are the same size
    episode_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        np.array(episodes),
        (0, episode_pad),
        mode="constant",
        constant_values=episodes[-1],
    ).reshape(-1, args.num_envs)

    successful_replays = 0
    if pbar is not None:
        pbar.reset(total=len(episodes))
    
    # Initialize camera parameters sequences for all batches
    all_camera_params_sequences = []  # Will store camera params for each episode
    
    for episode_batch_index, episode_batch in enumerate(batches):
        trajectory_ids = [episode["episode_id"] for episode in episode_batch]
        episode_lens = np.array([episode["elapsed_steps"] for episode in episode_batch])
        ori_control_mode = episode_batch[0]["control_mode"]
        assert all(
            [episode["control_mode"] == ori_control_mode for episode in episode_batch]
        ), "Replay trajectory with parallelized environments is only supported for trajectories with the same control mode"
        episode_batch_max_len = max(episode_lens)
        seeds = torch.tensor(
            [episode["episode_seed"] for episode in episode_batch],
            device=env.base_env.device,
        )
        env.reset(seed=seeds)

        # generate batched env states and actions
        env_states_list = []
        original_actions_batch = []
        env_states_batch = []  # list of batched env states shape (max_steps, D)
        for i, trajectory_id in enumerate(trajectory_ids):

            # sanity check seeds and warn user if reset kwargs includes options (which are not supported in GPU sim replay)
            traj = trajectories[f"traj_{trajectory_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            if not warned_reset_kwargs_options and "options" in episode["reset_kwargs"]:
                logger.warning(
                    f"Reset kwargs includes options, which are not supported in GPU sim replay and will be ignored."
                )
                warned_reset_kwargs_options = True

            # note (stao): this code to reformat the trajectories into a list of batched dicts can be optimized
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.array(traj["actions"])

            # padding
            for _ in range(episode_batch_max_len + 1 - len(env_states)):
                env_states.append(env_states[-1])
            if len(actions) < episode_batch_max_len:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (episode_batch_max_len - len(actions), actions.shape[1])
                        ),
                    ],
                    axis=0,
                )
            env_states_list.append(env_states)
            original_actions_batch.append(actions)
        for t in range(episode_batch_max_len + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )

        original_actions_batch = np.stack(original_actions_batch, axis=1)
        if args.use_first_env_state or args.use_env_states:
            # set the first environment state to the first states in the trajectories given
            env.base_env.set_state_dict(env_states_batch[0])
            if args.save_traj:
                # replace the first saved env state
                # since we set state earlier and RecordEpisode will save the reset to state.
                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, :] = y[-1, :]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(
                    env._trajectory_buffer.state, common.batch(env_states_batch[0])
                )
                recursive_replace(
                    env._trajectory_buffer.observation,
                    common.to_numpy(common.batch(env.base_env.get_obs())),
                )

        # replay with env states / actions
        if (
            args.target_control_mode is None
            or ori_control_mode == args.target_control_mode
        ):
            flushed_trajectories = np.zeros(len(episode_batch), dtype=bool)
            # mark the fake padding trajectories as flushed
            if episode_batch_index == len(batches) - 1 and episode_pad > 0:
                flushed_trajectories[-episode_pad:] = True
                
            # Initialize camera parameter sequences for this batch
            batch_camera_params_sequences = []
            if args.save_camera_params:
                batch_camera_params_sequences = [[] for _ in range(len(episode_batch))]
                
            for t, a in enumerate(original_actions_batch):
                _, _, _, truncated, info = env.step(a)
                if args.use_env_states:
                    # NOTE (stao): due to the high precision nature of some tasks even taking a single step in GPU simulation (in e.g. PushT-v1) can lead
                    # to some non-deterministic behaviors leading to some steps labeled with slightly wrong observations/rewards/success/fail data (1e-4 error).
                    # I unfortunately do not have a good solution for this apart from using the same number of parallel environments to replay demos as the original trajectory collection.
                    env.base_env.set_state_dict(env_states_batch[t])
                if args.vis:
                    env.base_env.render_human()
                    
                # Extract camera parameters if requested
                if args.save_camera_params:
                    try:
                        batch_camera_params = extract_camera_params_batch(env.base_env, len(episode_batch))
                        if len(batch_camera_params) == len(episode_batch):
                            for env_idx, camera_params in enumerate(batch_camera_params):
                                if camera_params:  # Only append if non-empty
                                    batch_camera_params_sequences[env_idx].append(camera_params)
                                else:
                                    logger.warning(f"Empty camera params for batch {episode_batch_index}, env {env_idx}, timestep {t}")
                        else:
                            logger.error(f"Camera params batch size mismatch: got {len(batch_camera_params)}, expected {len(episode_batch)}")
                    except Exception as e:
                        logger.error(f"Failed to extract camera parameters at batch {episode_batch_index}, timestep {t}: {e}")
                        
                # if the elapsed_steps mark saved in the trajectory is reached for any env, flush that trajectory buffer

                if args.save_traj:
                    envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                    flushed_trajectories |= envs_to_flush
                    if envs_to_flush.sum() > 0:
                        pbar.update(n=envs_to_flush.sum())
                        if not args.allow_failure:
                            if "success" in info:
                                success_tensor = info["success"]
                                if hasattr(success_tensor, "cpu"):
                                    envs_to_flush &= (success_tensor == True).cpu().numpy()
                                else:
                                    envs_to_flush &= (success_tensor == True)
                        if args.discard_timeout:
                            if hasattr(truncated, "cpu"):
                                envs_to_flush &= (truncated == False).cpu().numpy()
                            else:
                                envs_to_flush &= (truncated == False)
                        successful_replays += envs_to_flush.sum()
                        
                        # Flush trajectories first
                        env.flush_trajectory(
                            env_idxs_to_flush=np.where(envs_to_flush)[0]
                        )
                        
                        # Save original actions and camera parameters for successfully flushed environments
                        if args.save_camera_params or True:  # Always save original actions
                            try:
                                # Access the h5 file from RecordEpisode wrapper
                                h5_file = env._h5_file
                                
                                # For each successfully flushed environment
                                for local_env_idx in np.where(envs_to_flush)[0]:
                                    episode_id = episode_batch[local_env_idx]["episode_id"]
                                    
                                    # Get original actions for this episode
                                    ori_actions = original_actions_batch[:episode_lens[local_env_idx], local_env_idx]
                                    
                                    # Find the most recently added trajectory in h5 file
                                    available_keys = list(h5_file.keys())
                                    traj_keys = [k for k in available_keys if k.startswith("traj_")]
                                    
                                    if traj_keys:
                                        # Get the latest trajectory number and work backwards to find our trajectory
                                        traj_nums = [int(k.split("_")[1]) for k in traj_keys]
                                        latest_traj_num = max(traj_nums)
                                        
                                        # Find the trajectory for this episode (this is approximate in GPU mode)
                                        traj_key = f"traj_{latest_traj_num - (envs_to_flush.sum() - 1) + np.where(envs_to_flush)[0].tolist().index(local_env_idx)}"
                                        
                                        if traj_key in h5_file:
                                            traj_group = h5_file[traj_key]
                                            
                                            # Save original actions
                                            try:
                                                if "original_actions" in traj_group:
                                                    del traj_group["original_actions"]
                                                traj_group.create_dataset("original_actions", data=ori_actions)
                                            except (AttributeError, KeyError, TypeError) as e:
                                                logger.warning(f"Failed to save original actions: {e}")
                                            
                                            # Save original control mode
                                            try:
                                                if "original_control_mode" in traj_group.attrs:
                                                    del traj_group.attrs["original_control_mode"]
                                                traj_group.attrs["original_control_mode"] = ori_control_mode
                                            except (AttributeError, KeyError, TypeError) as e:
                                                logger.warning(f"Failed to save original control mode: {e}")
                                            
                                            logger.info(f"✅ GPU: SAVED original actions: episode {episode_id} -> {traj_key} ({len(ori_actions)} actions)")
                                            
                                            # Save camera parameters if available
                                            if args.save_camera_params and local_env_idx < len(batch_camera_params_sequences):
                                                camera_params_sequence = batch_camera_params_sequences[local_env_idx]
                                                if camera_params_sequence:
                                                    save_camera_params_to_h5(h5_file, traj_group, camera_params_sequence)
                                                    logger.info(f"✅ GPU: SAVED camera params: episode {episode_id} -> {traj_key} ({len(camera_params_sequence)} timesteps)")
                                                else:
                                                    logger.warning(f"❌ GPU: No camera params to save for episode {episode_id}")
                                        else:
                                            logger.warning(f"GPU: Trajectory {traj_key} not found for episode {episode_id}")
                                    else:
                                        logger.warning("GPU: No trajectory keys found in h5 file")
                                        
                            except Exception as e:
                                logger.error(f"GPU: Failed to save additional data: {e}")
                                # Don't let this break the main replay process
                                
            # Clean up camera parameter sequences after processing this batch
            if args.save_camera_params and 'batch_camera_params_sequences' in locals():
                try:
                    for seq in batch_camera_params_sequences:
                        if seq is not None:
                            del seq[:]  # Clear the sequence contents
                    del batch_camera_params_sequences  # Delete the list itself
                except (NameError, UnboundLocalError):
                    pass  # Variable doesn't exist, nothing to clean up
                
        else:
            raise NotImplementedError(
                "Replay with different control modes are not supported when replaying on GPU parallelized environments"
            )
        
        # Cleanup memory after each batch to prevent accumulation
        if episode_batch_index > 0 and episode_batch_index % 5 == 0:  # Every 5 batches
            force_memory_cleanup()
    
    # Cleanup memory after all batches are processed
    force_memory_cleanup()
    
    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def replay_cpu_sim(
    args: Args, env: RecordEpisode, ori_env, pbar, episodes, trajectories
):
    successful_replays = 0
    for episode in episodes:
        sanity_check_and_format_seed(episode)
        episode_id = episode["episode_id"]
        traj_id = f"traj_{episode_id}"
        reset_kwargs = episode["reset_kwargs"]
        ori_control_mode = episode["control_mode"]
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")
        if traj_id not in trajectories:
            tqdm.write(f"{traj_id} does not exist in {args.traj_path}")
            continue

        for _ in range(args.max_retry + 1):
            # Each trial for each trajectory to replay, we reset the environment
            # and optionally set the first environment state
            env.reset(**reset_kwargs)
            if ori_env is not None:
                ori_env.reset(**reset_kwargs)

            # set first environment state and update recorded env state
            ori_env_states = None  # Initialize to prevent unbound variable issues
            if args.use_first_env_state or args.use_env_states:
                ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    trajectories[traj_id]["env_states"]
                )
                if ori_env is not None:
                    ori_env.set_state_dict(ori_env_states[0])
                env.base_env.set_state_dict(ori_env_states[0])
                ori_env_states = ori_env_states[1:]
                if args.save_traj:
                    # replace the first saved env state
                    # since we set state earlier and RecordEpisode will save the reset to state.
                    def recursive_replace(x, y):
                        if isinstance(x, np.ndarray):
                            x[-1, :] = y[-1, :]
                        else:
                            for k in x.keys():
                                recursive_replace(x[k], y[k])

                    recursive_replace(
                        env._trajectory_buffer.state, common.batch(ori_env_states[0])
                    )
                    fixed_obs = env.base_env.get_obs()
                    recursive_replace(
                        env._trajectory_buffer.observation,
                        common.to_numpy(common.batch(fixed_obs)),
                    )
            # Original actions to replay - load as copy to allow cleanup
            ori_actions = np.array(trajectories[traj_id]["actions"][:])
            info = {}
            truncated = False  # Initialize truncated to prevent unbound variable issues

            # Initialize camera parameters sequence for saving
            camera_params_sequence = []

            # Without conversion between control modes
            assert (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
                or not args.use_env_states
            ), "Cannot use env states when trying to \
                convert from one control mode to another. This is because control mode conversion causes there to be changes \
                in how many actions are taken to achieve the same states"
            if (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
            ):
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, truncated, info = env.step(a)
                    if args.use_env_states and ori_env_states is not None:
                        env.base_env.set_state_dict(ori_env_states[t])
                    if args.vis:
                        env.base_env.render_human()

                    # Extract camera parameters if requested
                    if args.save_camera_params:
                        try:
                            camera_params = extract_camera_params(env.base_env)
                            if not camera_params:
                                logger.error(
                                    f"Episode {episode_id}, timestep {t}: Camera parameter extraction returned empty dict - CRITICAL FAILURE!"
                                )
                                raise RuntimeError(
                                    f"Camera parameter extraction failed at timestep {t}"
                                )
                            camera_params_sequence.append(camera_params)
                        except Exception as e:
                            logger.error(
                                f"Episode {episode_id}, timestep {t}: Camera parameter extraction FAILED: {e}"
                            )
                            # This will cause the trajectory to be rejected in validation
                            camera_params_sequence = []
                            break

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                # Create a wrapper to extract camera parameters during conversion
                if args.save_camera_params:
                    logger.info(
                        f"Extracting camera parameters at every timestep during pd_joint_pos -> {args.target_control_mode} conversion"
                    )
                    
                    # Monkey patch the env.step method to capture camera parameters
                    original_step = env.step
                    def step_with_camera_extraction(action):
                        # Extract camera parameters BEFORE stepping
                        try:
                            camera_params = extract_camera_params(env.base_env)
                            if not camera_params:
                                logger.error(
                                    f"Episode {episode_id}, timestep {len(camera_params_sequence)}: Camera parameter extraction returned empty dict during conversion - CRITICAL FAILURE!"
                                )
                                raise RuntimeError(
                                    f"Camera parameter extraction failed at timestep {len(camera_params_sequence)} during conversion"
                                )
                            camera_params_sequence.append(camera_params)
                        except Exception as e:
                            logger.error(
                                f"Episode {episode_id}, timestep {len(camera_params_sequence)}: Camera parameter extraction FAILED during conversion: {e}"
                            )
                            # Don't break the conversion, but mark as failed
                            pass
                        
                        # Call original step method
                        return original_step(action)
                    
                    # Apply the monkey patch
                    env.step = step_with_camera_extraction
                    
                    try:
                        info = action_conversion.from_pd_joint_pos(
                            args.target_control_mode,
                            ori_actions,
                            ori_env,
                            env,
                            render=args.vis,
                            pbar=pbar,
                            verbose=args.verbose,
                        )
                        
                        logger.info(
                            f"Extracted {len(camera_params_sequence)} camera parameter sets during pd_joint_pos conversion"
                        )
                    finally:
                        # Restore original step method
                        env.step = original_step
                else:
                    # Use standard conversion without camera extraction
                    info = action_conversion.from_pd_joint_pos(
                        args.target_control_mode,
                        ori_actions,
                        ori_env,
                        env,
                        render=args.vis,
                        pbar=pbar,
                        verbose=args.verbose,
                    )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                # Create a wrapper to extract camera parameters during conversion
                if args.save_camera_params:
                    logger.info(
                        f"Extracting camera parameters at every timestep during pd_joint_delta_pos -> {args.target_control_mode} conversion"
                    )
                    
                    # Monkey patch the env.step method to capture camera parameters
                    original_step = env.step
                    def step_with_camera_extraction(action):
                        # Extract camera parameters BEFORE stepping
                        try:
                            camera_params = extract_camera_params(env.base_env)
                            if not camera_params:
                                logger.error(
                                    f"Episode {episode_id}, timestep {len(camera_params_sequence)}: Camera parameter extraction returned empty dict during conversion - CRITICAL FAILURE!"
                                )
                                raise RuntimeError(
                                    f"Camera parameter extraction failed at timestep {len(camera_params_sequence)} during conversion"
                                )
                            camera_params_sequence.append(camera_params)
                        except Exception as e:
                            logger.error(
                                f"Episode {episode_id}, timestep {len(camera_params_sequence)}: Camera parameter extraction FAILED during conversion: {e}"
                            )
                            # Don't break the conversion, but mark as failed
                            pass
                        
                        # Call original step method
                        return original_step(action)
                    
                    # Apply the monkey patch
                    env.step = step_with_camera_extraction
                    
                    try:
                        info = action_conversion.from_pd_joint_delta_pos(
                            args.target_control_mode,
                            ori_actions,
                            ori_env,
                            env,
                            render=args.vis,
                            pbar=pbar,
                            verbose=args.verbose,
                        )
                        
                        logger.info(
                            f"Extracted {len(camera_params_sequence)} camera parameter sets during pd_joint_delta_pos conversion"
                        )
                    finally:
                        # Restore original step method
                        env.step = original_step
                else:
                    # Use standard conversion without camera extraction
                    info = action_conversion.from_pd_joint_delta_pos(
                        args.target_control_mode,
                        ori_actions,
                        ori_env,
                        env,
                        render=args.vis,
                        pbar=pbar,
                        verbose=args.verbose,
                    )
            else:
                raise NotImplementedError(
                    f"Script currently does not support converting {ori_control_mode} to {args.target_control_mode}"
                )

            success = info.get("success", False)
            if args.discard_timeout:
                success = success and (not truncated)

            # VALIDATE CAMERA PARAMETERS if required
            camera_params_valid = True
            if args.save_camera_params:
                if len(camera_params_sequence) == 0:
                    logger.error(
                        f"Episode {episode_id}: NO camera parameters extracted - EXCLUDING from training!"
                    )
                    camera_params_valid = False
                else:
                    # Validate that all timesteps have complete camera data
                    for t, params in enumerate(camera_params_sequence):
                        if not params:  # Empty dict
                            logger.error(
                                f"Episode {episode_id}: Camera parameters MISSING at timestep {t} - EXCLUDING from training!"
                            )
                            camera_params_valid = False
                            break

                        for camera_name, camera_data in params.items():
                            if camera_data.get("intrinsic") is None:
                                logger.error(
                                    f"Episode {episode_id}: Camera {camera_name} MISSING intrinsic at timestep {t} - EXCLUDING from training!"
                                )
                                camera_params_valid = False
                                break
                            if camera_data.get("extrinsic") is None:
                                logger.error(
                                    f"Episode {episode_id}: Camera {camera_name} MISSING extrinsic at timestep {t} - EXCLUDING from training!"
                                )
                                camera_params_valid = False
                                break

                        if not camera_params_valid:
                            break

                    if camera_params_valid:
                        logger.info(
                            f"Episode {episode_id}: Camera parameters VALIDATED - {len(camera_params_sequence)} timesteps, {len(camera_params_sequence[0])} cameras"
                        )

            # Only save trajectory if both success AND camera parameters are valid (if required)
            final_success = (success or args.allow_failure) and camera_params_valid

            if final_success:
                successful_replays += 1
                if args.save_traj:
                    # First flush the trajectory to ensure it's written
                    env.flush_trajectory()

                    # Save original actions to the h5 file - ALWAYS save original actions
                    try:
                        # Access the h5 file from RecordEpisode wrapper
                        h5_file = env._h5_file

                        # Find the trajectory we just saved
                        # In multiprocessing, each process has its own h5 file
                        # Find the most recently added trajectory (the one we just flushed)
                        available_keys = list(h5_file.keys())
                        traj_keys = [
                            k for k in available_keys if k.startswith("traj_")
                        ]

                        if traj_keys:
                            # Use the highest numbered trajectory key (most recent)
                            traj_nums = [int(k.split("_")[1]) for k in traj_keys]
                            latest_traj_num = max(traj_nums)
                            traj_key = f"traj_{latest_traj_num}"
                        else:
                            logger.error("No trajectory keys found in h5 file")
                            raise RuntimeError("No trajectory keys found after flush")
                        
                        if traj_key in h5_file:
                            traj_group = h5_file[traj_key]
                            
                            # Save original actions as a separate dataset
                            try:
                                if "original_actions" in traj_group:
                                    del traj_group["original_actions"]
                                traj_group.create_dataset("original_actions", data=ori_actions)
                            except (AttributeError, KeyError, TypeError) as e:
                                logger.warning(f"Failed to save original actions: {e}")
                            
                            # Also save original control mode metadata
                            try:
                                if "original_control_mode" in traj_group.attrs:
                                    del traj_group.attrs["original_control_mode"]
                                traj_group.attrs["original_control_mode"] = ori_control_mode
                            except (AttributeError, KeyError, TypeError) as e:
                                logger.warning(f"Failed to save original control mode: {e}")
                            
                            logger.info(
                                f"✅ SAVED original actions: episode {episode_id} -> {traj_key} ({len(ori_actions)} actions, control_mode: {ori_control_mode})"
                            )
                        else:
                            logger.error(f"Trajectory {traj_key} not found after flush")
                            raise RuntimeError(f"Trajectory {traj_key} missing after flush")
                            
                    except Exception as e:
                        logger.error(f"CRITICAL: Failed to save original actions for episode {episode_id}: {e}")
                        raise RuntimeError(f"Original actions saving failed: {e}")

                    # Then save camera parameters to the h5 file
                    if args.save_camera_params and len(camera_params_sequence) > 0:
                        try:
                            # Access the h5 file from RecordEpisode wrapper
                            h5_file = env._h5_file

                            # In multiprocessing, each process has its own h5 file
                            # Find the most recently added trajectory (the one we just flushed)
                            available_keys = list(h5_file.keys())
                            traj_keys = [
                                k for k in available_keys if k.startswith("traj_")
                            ]

                            if traj_keys:
                                # Use the highest numbered trajectory key (most recent)
                                traj_nums = [int(k.split("_")[1]) for k in traj_keys]
                                latest_traj_num = max(traj_nums)
                                traj_key = f"traj_{latest_traj_num}"

                                if traj_key in h5_file:
                                    traj_group = h5_file[traj_key]
                                    save_camera_params_to_h5(
                                        h5_file, traj_group, camera_params_sequence
                                    )
                                    logger.info(
                                        f"✅ SAVED camera params: episode {episode_id} -> {traj_key} ({len(camera_params_sequence)} timesteps)"
                                    )
                                    # Immediately clear camera_params_sequence after saving
                                    del camera_params_sequence[:]  # Clear the list contents
                                    force_memory_cleanup()  # Force cleanup after H5 save
                                else:
                                    logger.error(
                                        f"Trajectory {traj_key} not found after flush"
                                    )
                                    raise RuntimeError(
                                        f"Trajectory {traj_key} missing after flush"
                                    )
                            else:
                                logger.error(
                                    f"No trajectories found in h5 file - available: {available_keys}"
                                )
                                raise RuntimeError(f"No trajectories in h5 file")
                        except Exception as e:
                            logger.error(
                                f"CRITICAL: Failed to save camera parameters for episode {episode_id}: {e}"
                            )
                            raise RuntimeError(f"Camera parameter saving failed: {e}")
                if args.save_video:
                    env.flush_video(ignore_empty_transition=False)
                break
            else:
                # REJECT this trajectory due to camera parameter failure
                if not camera_params_valid:
                    logger.error(
                        f"❌ REJECTING episode {episode_id}: Camera parameter extraction FAILED"
                    )
                else:
                    logger.warning(
                        f"❌ REJECTING episode {episode_id}: Task execution failed"
                    )
                env.flush_video(save=False)  # Don't save failed trajectories
                if args.verbose:
                    print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")
        
        # CRITICAL: Clean up memory after each episode to prevent leaks
        try:
            if 'ori_actions' in locals() and ori_actions is not None:
                del ori_actions
        except (NameError, UnboundLocalError):
            pass  # Variable doesn't exist
        try:
            if 'camera_params_sequence' in locals() and camera_params_sequence is not None:
                del camera_params_sequence
        except (NameError, UnboundLocalError):
            pass  # Variable doesn't exist
        try:
            if 'ori_env_states' in locals() and ori_env_states is not None:
                del ori_env_states
        except (NameError, UnboundLocalError):
            pass  # Variable doesn't exist
        force_memory_cleanup()

    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def _main_helper(x):
    return _main(*x)


def _main(
    args: Args,
    use_cpu_backend,
    env_id,
    env_kwargs,
    ori_env_kwargs,
    record_episode_kwargs,
    proc_id: int = 0,
    num_procs=1,
):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    env = gym.make(env_id, **env_kwargs)

    if isinstance(env.action_space, gym.spaces.Dict):
        logger.warning(
            "We currently do not track which wrappers are used when recording trajectories but majority of the time in multi-agent envs with dictionary action spaces the actions are stored as flat vectors. We will flatten the action space with the ManiSkill provided FlattenActionSpaceWrapper. If you do not want this behavior you can copy the replay trajectory code yourself and modify it as needed."
        )
        env = FlattenActionSpaceWrapper(env)
    # TODO (support adding wrappers to the recorded data?)

    # if pbar is not None:
    #     pbar.set_postfix(
    #         {
    #             "control_mode": env_kwargs.get("control_mode"),
    #             "obs_mode": env_kwargs.get("obs_mode"),
    #         }
    #     )

    ### Prepare for recording ###

    # note for maniskill trajectory datasets the general naming format is <trajectory_name>.<obs_mode>.<control_mode>.<sim_backend>.h5
    # If it is called <file_name>.h5 then we assume obs_mode=None, control_mode=pd_joint_pos, and sim_backend=physx_cpu
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode,
        env.unwrapped.control_mode,
        env.unwrapped.backend.sim_backend,
    )
    new_traj_name = ori_traj_name + "." + suffix
    if use_cpu_backend:
        if num_procs > 1:
            new_traj_name = new_traj_name + "." + str(proc_id)
        if args.target_control_mode is not None:
            ori_env = gym.make(env_id, **ori_env_kwargs)
        else:
            ori_env = None
    else:
        pass

    env = wrappers.RecordEpisode(
        env,
        output_dir,
        trajectory_name=new_traj_name,
        video_fps=(
            args.video_fps if args.video_fps is not None else env.unwrapped.control_freq
        ),
        **record_episode_kwargs,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    if use_cpu_backend:
        inds = np.arange(len(episodes))
        inds = np.array_split(inds, num_procs)[proc_id]
        # Get episodes for this process
        proc_episodes = [episodes[index] for index in inds]

        # Keep original episode IDs for proper h5 file trajectory naming

        replay_result = replay_cpu_sim(
            args, env, ori_env, pbar, proc_episodes, ori_h5_file
        )
    else:
        replay_result = replay_parallelized_sim(args, env, pbar, episodes, ori_h5_file)

    env.close()
    ori_h5_file.close()
    
    # Final cleanup to ensure all memory is freed
    force_memory_cleanup()
    
    return output_h5_path, replay_result


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def main(args: Args):
    traj_path = args.traj_path
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    # Clean up existing output files at the beginning if save_traj is enabled
    if args.save_traj:
        output_dir = os.path.dirname(traj_path)
        ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
        parts = ori_traj_name.split(".")
        if len(parts) > 1:
            ori_traj_name = parts[0]

        # Determine the target control mode and obs mode for output file naming
        target_control_mode = args.target_control_mode or ori_env_kwargs.get(
            "control_mode", "pd_joint_pos"
        )
        target_obs_mode = args.obs_mode or ori_env_kwargs.get("obs_mode", "none")
        sim_backend = args.sim_backend or ori_env_kwargs.get("sim_backend", "physx_cpu")

        # Construct expected output file pattern
        suffix = f"{target_obs_mode}.{target_control_mode}.{sim_backend}"
        output_pattern = f"{ori_traj_name}.{suffix}"

        # Clean up potential output files
        import glob

        cleanup_patterns = [
            f"{output_dir}/{output_pattern}.h5",
            f"{output_dir}/{output_pattern}.json",
            f"{output_dir}/{output_pattern}.*.h5",  # multiprocessing files
            f"{output_dir}/{output_pattern}.*.json",
        ]

        files_removed = 0
        for pattern in cleanup_patterns:
            matching_files = glob.glob(pattern)
            for file_path in matching_files:
                try:
                    os.remove(file_path)
                    files_removed += 1
                    logger.info(
                        f"Cleaned up existing file: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

        if files_removed > 0:
            logger.info(
                f"🧹 Cleaned up {files_removed} existing output files before starting"
            )
        else:
            logger.info("No existing output files found to clean up")

    ### Checks and setting up env kwargs ###
    # First we determine how to setup the environment to replay demonstrations and raise relevant warnings to the user
    if (
        "sim_backend" in ori_env_kwargs
        and ori_env_kwargs["sim_backend"] != args.sim_backend
        and args.use_env_states
    ):
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']} "
            f"or replay by environment states by passing --use-env-states instead."
        )
    if args.sim_backend is None:
        # try to guess which sim backend to use
        if "sim_backend" not in ori_env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = ori_env_kwargs["sim_backend"]

    ori_env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["sim_backend"] = args.sim_backend

    # modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader  # change all shaders
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs["render_mode"] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    # Setup camera resolution if specified
    if args.camera_width is not None and args.camera_height is not None:
        setup_camera_resolution(env_kwargs, args.camera_width, args.camera_height)

    record_episode_kwargs = dict(
        save_on_reset=False,
        save_trajectory=args.save_traj,
        save_video=args.save_video,
        record_reward=args.record_rewards,
    )

    if args.count is not None and args.count > len(json_data["episodes"]):
        logger.warning(
            f"Warning: Requested to replay {args.count} demos but there are only {len(json_data['episodes'])} demos collected, replaying all demos now"
        )
        args.count = len(json_data["episodes"])
    elif args.count is None:
        args.count = len(json_data["episodes"])

    pbar = tqdm(total=args.count, unit="step", dynamic_ncols=True)

    # if missing info or auto sim backend is provided, we try to infer which backend is being used
    if "sim_backend" not in env_kwargs or (
        env_kwargs["sim_backend"] == "auto"
        and ("num_envs" not in env_kwargs or env_kwargs["num_envs"] == 1)
    ):
        env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = args.num_envs
    if env_kwargs["sim_backend"] not in CPU_SIM_BACKENDS:
        record_episode_kwargs["max_steps_per_video"] = env_info["max_episode_steps"]
        _, replay_result = _main(
            args,
            use_cpu_backend=False,
            env_id=env_id,
            env_kwargs=env_kwargs,
            ori_env_kwargs=ori_env_kwargs,
            record_episode_kwargs=record_episode_kwargs,
            proc_id=0,
            num_procs=1,
        )

    else:
        env_kwargs["num_envs"] = 1
        ori_env_kwargs["num_envs"] = 1

        # Camera parameter extraction works with multiprocessing - each process handles its own episodes

        if args.num_envs > 1:
            pool = mp.Pool(args.num_envs)
            proc_args = [
                (
                    copy.deepcopy(args),
                    True,
                    env_id,
                    env_kwargs,
                    ori_env_kwargs,
                    record_episode_kwargs,
                    i,
                    args.num_envs,
                )
                for i in range(args.num_envs)
            ]
            # res = pool.starmap(_main, proc_args)
            res = list(tqdm(pool.imap(_main_helper, proc_args), total=args.count))
            replay_results_list = [x[1] for x in res]
            trajectory_paths = [x[0] for x in res]
            pool.close()
            if args.save_traj:
                # A hack to find the path
                output_path = trajectory_paths[0][: -len("0.h5")] + "h5"
                merge_trajectories(output_path, trajectory_paths)
                for h5_path in trajectory_paths:
                    tqdm.write(f"Remove {h5_path}")
                    os.remove(h5_path)
                    json_path = h5_path.replace(".h5", ".json")
                    tqdm.write(f"Remove {json_path}")
                    os.remove(json_path)
            replay_result = ReplayResult(
                num_replays=sum([x.num_replays for x in replay_results_list]),
                successful_replays=sum(
                    [x.successful_replays for x in replay_results_list]
                ),
            )
        else:
            _, replay_result = _main(
                args,
                use_cpu_backend=True,
                env_id=env_id,
                env_kwargs=env_kwargs,
                ori_env_kwargs=ori_env_kwargs,
                record_episode_kwargs=record_episode_kwargs,
                proc_id=0,
                num_procs=1,
            )

    pbar.close()
    print(
        f"Replayed {replay_result.num_replays} episodes, "
        f"{replay_result.successful_replays}/{replay_result.num_replays}={replay_result.successful_replays/replay_result.num_replays*100:.2f}% demos saved"
    )
    
    # Final comprehensive cleanup to prevent any memory leaks
    force_memory_cleanup()
    logger.info("🧹 Final memory cleanup completed")


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())
