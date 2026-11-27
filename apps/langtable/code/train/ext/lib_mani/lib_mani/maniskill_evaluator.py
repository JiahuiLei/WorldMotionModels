from collections import defaultdict
import gymnasium
import numpy as np
import torch
from mani_skill.utils import common
from typing import Optional
import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper
from mani_skill.envs.sapien_env import BaseEnv
from functools import partial
from typing import Dict
import os
import os.path as osp
from .sapien_robot_wrapper import RobotKinematicsConverter
import logging
import gc

# rank 0 only
from pytorch_lightning.utilities import rank_zero_only


class ManiSkillEvaluator:
    def __init__(
        self,
        robot_link_pcl_fn=None,
        env_id="StackCube-v1",
        sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pos",
        num_eval_envs=10,
        num_eval_episodes=10,
        max_episode_steps=None,
        video_dir=None,
        # sensor
        H: int = 512,
        W: int = 512,
        cuda_visible_devices: Optional[str] = None,
        max_viz_envs: int = -1,
    ):
        self.H = H
        self.W = W

        self.env_id = env_id
        self.sim_backend = sim_backend
        self.control_mode = control_mode
        self.num_eval_envs = num_eval_envs
        self.num_eval_episodes = num_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.video_dir = video_dir
        self.max_viz_envs = max_viz_envs
        # Note: CUDA_VISIBLE_DEVICES must be set before importing ManiSkill modules
        # Setting it here is too late - SAPIEN has already initialized GPU contexts
        if cuda_visible_devices is not None:
            print(
                f"Warning: cuda_visible_devices parameter ({cuda_visible_devices}) is ineffective. "
                f"Set CUDA_VISIBLE_DEVICES environment variable before importing ManiSkill."
            )

        self.robot_link_pcl_fn = robot_link_pcl_fn
        if robot_link_pcl_fn is not None:
            self.robot_link_pcl = np.load(self.robot_link_pcl_fn)["link_pcls"].astype(
                np.float16
            )
        else:
            self.robot_link_pcl = None

        self.envs = None
        return

    def reset(self):
        if self.envs is not None:
            self.envs.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the vectorized environments to prevent cleanup errors."""
        if hasattr(self, "envs") and self.envs is not None:
            try:
                self.envs.close()
            except Exception as e:
                print(f"Warning: Error closing environments: {e}")
            finally:
                self.envs = None

    def __del__(self):
        """Ensure environments are closed when the evaluator is garbage collected."""
        self.close()

    def make_env(self):
        env_kwargs = dict(
            control_mode=self.control_mode,
            reward_mode="sparse",
            obs_mode="rgb+depth+segmentation",
            render_mode="rgb_array",
        )
        if self.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = self.max_episode_steps
        env_kwargs["sensor_configs"] = dict(width=self.W, height=self.H)
        envs = make_eval_envs(
            self.env_id,
            self.num_eval_envs,
            self.sim_backend,
            env_kwargs,
            other_kwargs={},
            video_dir=self.video_dir,
            wrappers=[partial(OBSWrapper, robot_link_pcl=self.robot_link_pcl)],
            max_viz_envs=self.max_viz_envs,
        )
        return envs

    # NOTE: keep a single robust close implementation (defined above)

    @torch.no_grad()
    @rank_zero_only
    def evaluate(
        self,
        policy,  # callable: action = policy_func(obs); this function it self has to deal with all the preproces and post process!
        predict_horizon,  # ! this is coupled with policy func!
        # other config
        temporal_agg=True,
        temporal_agg_max_timesteps=None,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        sim_backend="physx_cpu",
    ):
        if self.envs is None:
            self.envs = self.make_env()
        eval_envs = self.envs
        if temporal_agg_max_timesteps is None:
            temporal_agg_max_timesteps = self.max_episode_steps
        if temporal_agg_max_timesteps is None:
            temporal_agg_max_timesteps = 200  # fallback
        if eval_envs is None:
            raise RuntimeError("Environments are not initialized or have been closed")

        use_visual_obs = isinstance(eval_envs.single_observation_space.sample(), dict)

        # create action table for temporal ensembling
        action_dim = eval_envs.action_space.shape[-1]
        num_envs = eval_envs.num_envs
        if temporal_agg:
            query_frequency = 1
            all_time_actions = torch.zeros(
                [
                    num_envs,
                    temporal_agg_max_timesteps,
                    temporal_agg_max_timesteps + predict_horizon,
                    action_dim,
                ],
                device=device,
            )
            actions_to_take = None  # Initialize for clarity
        else:
            query_frequency = predict_horizon
            actions_to_take = torch.zeros(
                [num_envs, predict_horizon, action_dim], device=device
            )
            all_time_actions = None  # Initialize for clarity

        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        ts, eps_count = 0, 0
        while eps_count < self.num_eval_episodes:
            logging.info(
                f"Evaluating episode [{eps_count}/{self.num_eval_episodes}] step [{ts}/{temporal_agg_max_timesteps}]"
            )
            # pre-process obs
            if use_visual_obs:
                obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            else:
                obs = common.to_tensor(obs, device)

            # query policy
            if ts % query_frequency == 0:
                action_seq = policy(obs)  # (num_envs, num_queries, action_dim)
                # assert action_seq.shape == (num_envs, predict_horizon, action_dim)

            # we assume ignore_terminations=True. Otherwise, some envs could be done
            # earlier, so we would need to temporally ensemble at corresponding timestep
            # for each env.
            if temporal_agg:
                assert (
                    query_frequency == 1
                ), "query_frequency != 1 has not been implemented for temporal_agg==1."
                assert (
                    all_time_actions is not None
                ), "all_time_actions should be initialized for temporal_agg"
                all_time_actions[:, ts, ts : ts + predict_horizon] = (
                    action_seq  # (num_envs, num_queries, act_dim)
                )
                actions_for_curr_step = all_time_actions[
                    :, :, ts
                ]  # (num_envs, max_timesteps, act_dim)
                # since we pad the action with 0 in 'delta_pos' control mode, this causes error.
                # actions_populated = torch.all(actions_for_curr_step[0] != 0, axis=1) # (max_timesteps,)
                actions_populated = torch.zeros(
                    temporal_agg_max_timesteps, dtype=torch.bool, device=device
                )  # (max_timesteps,)
                actions_populated[max(0, ts + 1 - predict_horizon) : ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[
                    :, actions_populated
                ]  # (num_envs, num_populated, act_dim)
                k = 0.01
                if ts < predict_horizon:
                    exp_weights = torch.exp(
                        -k * torch.arange(len(actions_for_curr_step[0]), device=device)
                    )  # (num_populated,)
                    exp_weights = exp_weights / exp_weights.sum()  # (num_populated,)
                    exp_weights = torch.tile(
                        exp_weights, (num_envs, 1)
                    )  # (num_envs, num_populated)
                    exp_weights = torch.unsqueeze(
                        exp_weights, -1
                    )  # (num_envs, num_populated, 1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(
                        dim=1
                    )  # (num_envs, act_dim)
                else:
                    # Use simple averaging when not in initial prediction horizon
                    raw_action = actions_for_curr_step.mean(dim=1)
            else:
                assert (
                    actions_to_take is not None
                ), "actions_to_take should be initialized for non-temporal_agg"
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            action = raw_action  # (num_envs, act_dim)
            if sim_backend == "physx_cpu":
                action = action.cpu().numpy()

            # step the environment
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1
            # collect episode info
            if truncated.any():
                assert (
                    truncated.all() == truncated.any()
                ), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                # new episodes begin
                eps_count += num_envs
                ts = 0
                if temporal_agg:
                    all_time_actions = torch.zeros(
                        [
                            num_envs,
                            temporal_agg_max_timesteps,
                            temporal_agg_max_timesteps + predict_horizon,
                            action_dim,
                        ],
                        device=device,
                    )

        for k in eval_metrics.keys():
            eval_metrics[k] = np.stack(eval_metrics[k])

        # Reset environments and collect garbage
        eval_envs.reset()
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return eval_metrics


class OBSWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, robot_link_pcl) -> None:
        self.base_env: BaseEnv = env.unwrapped
        # se3 convernter
        self.kin_converter = RobotKinematicsConverter(self.base_env.agent.robot)
        self.robot_link_pcl = (
            torch.from_numpy(robot_link_pcl) if robot_link_pcl is not None else None
        )
        super().__init__(env)
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

        return

    def __get_camera_params__(self) -> Dict[str, Dict[str, np.ndarray]]:
        camera_params = {}
        if hasattr(self.base_env, "_sensors"):
            cameras = {
                k: v for k, v in self.base_env._sensors.items() if "camera" in k.lower()
            }
            for camera_name, camera in cameras.items():
                try:
                    # Get camera parameters using ManiSkill's get_params() method
                    params = camera.get_params()
                    intrinsic = params["intrinsic_cv"]

                    # Get extrinsic matrix (camera pose in world frame)
                    extrinsic = camera.config.pose.to_transformation_matrix()

                    # If camera is mounted on an entity, transform by entity pose
                    if camera.entity is not None:
                        T_wl = camera.entity.pose.to_transformation_matrix()
                        extrinsic = T_wl @ extrinsic

                    # Convert to numpy and handle batch dimension
                    if hasattr(intrinsic, "cpu"):
                        clean_intrinsic = intrinsic.cpu().numpy()
                    elif hasattr(intrinsic, "numpy"):
                        clean_intrinsic = intrinsic.numpy()
                    else:
                        clean_intrinsic = np.array(intrinsic)

                    if hasattr(extrinsic, "cpu"):
                        clean_extrinsic = extrinsic.cpu().numpy()
                    elif hasattr(extrinsic, "numpy"):
                        clean_extrinsic = extrinsic.numpy()
                    else:
                        clean_extrinsic = np.array(extrinsic)

                    clean_intrinsic = clean_intrinsic.squeeze(0)
                    clean_extrinsic = clean_extrinsic.squeeze(0)  # 4x4
                    R_sapien_opencv = np.array(
                        [
                            [0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                        ]
                    )
                    T_wc = clean_extrinsic.copy()
                    T_wc[:3, :3] = T_wc[:3, :3] @ R_sapien_opencv

                    camera_params[camera_name] = {
                        "K": clean_intrinsic,
                        "T_wc": T_wc,
                    }
                    assert clean_intrinsic.shape == (3, 3)
                    assert T_wc.shape == (4, 4)

                except Exception as e:
                    logging.warning(
                        f"Failed to extract camera parameters for {camera_name}: {e}"
                    )
                    continue

        return camera_params

    def observation(self, observation: Dict):
        qpos = observation["agent"]["qpos"].detach().clone()  # B,Q
        qvel = observation["agent"]["qvel"].detach().clone()  # B,Q
        qpos_se3 = torch.stack(
            [self.kin_converter.qpos_to_link_poses(q) for q in qpos], 0
        )  # B,Q,4,4

        # animate the local pcl
        if self.robot_link_pcl is not None:
            link_pcl = self.robot_link_pcl.clone().to(qpos)  # Q,N,3
            assert len(link_pcl) == qpos_se3.shape[1]
            animated_pcl = (
                torch.einsum("bqij,qnj->bqni", qpos_se3[..., :3, :3], link_pcl)
                + qpos_se3[..., None, :3, 3]
            )
            # # debug
            # np.savetxt("./debug/robot_pcl_qpos.xyz", animated_pcl[0,:,:512].reshape(-1, 3).cpu().numpy(), fmt="%.6f")

        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb, images_depth, images_segmentation = [], [], []
        for cam_data in sensor_data.values():
            images_rgb.append(cam_data["rgb"].permute(0, 3, 1, 2))
            depth = cam_data["depth"].to(torch.float32) / 1024  # .to(torch.float16)
            images_depth.append(depth.permute(0, 3, 1, 2))
            images_segmentation.append(cam_data["segmentation"].permute(0, 3, 1, 2))

        rgb = torch.stack(images_rgb, dim=1)  # (1, num_cams, C, H, W), uint8
        depth = torch.stack(images_depth, dim=1)  # (1, num_cams, C, H, W), float16
        segmentation = torch.stack(
            images_segmentation, dim=1
        )  # (1, num_cams, C, H, W), int32

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)

        # read camera T_wc and K
        cam = self.__get_camera_params__()
        K, T_wc = [], []
        for sensor_name in sensor_data.keys():
            K_cam = cam[sensor_name]["K"]
            T_wc_cam = cam[sensor_name]["T_wc"]
            # Ensure arrays have batch dimension
            K.append(K_cam)
            T_wc.append(T_wc_cam)
        K = np.stack(K, axis=0)[None]
        T_wc = np.stack(T_wc, axis=0)[None]
        assert len(K) == len(T_wc) == len(rgb)
        # print(K.shape, T_wc.shape)
        K, T_wc = torch.as_tensor(K), torch.as_tensor(T_wc)

        ret = dict(
            state=observation.clone(),
            rgb=rgb.clone(),
            depth=depth.clone(),
            segmentation=segmentation.clone(),
            qpos=qpos.clone(),
            qvel=qvel.clone(),
            qpos_se3=qpos_se3.clone(),
            K=K.clone(),
            T_wc=T_wc.clone(),
        )
        if self.robot_link_pcl is not None:
            animated_pcl = torch.cat(
                [animated_pcl, torch.zeros_like(animated_pcl)], dim=-1
            )
            ret["link_pcl"] = animated_pcl.clone()  # B,Q,N,3

        for k in ret.keys():
            if isinstance(ret[k], torch.Tensor):
                ret[k] = ret[k].detach().clone().contiguous()

        del observation
        gc.collect()
        return ret


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list = [],
    max_viz_envs: int = -1,
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                logging.info(
                    f"cpu_make_env:thunk: {env_id}, {seed}, {video_dir}, {env_kwargs}, {other_kwargs}"
                )
                env = gym.make(
                    env_id,
                    reconfiguration_freq=1,
                    **env_kwargs,
                    sim_backend="physx_cpu",
                    enable_shadow=False,
                    render_backend="gpu",
                )
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="act",
                        # source_desc="act evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        # tmp = cpu_make_env(env_id, 0, None, env_kwargs, other_kwargs)()
        # try:
        #     obs_space, act_space = tmp.observation_space, tmp.action_space
        # finally:
        #     tmp.close()

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(
                x,
                context="spawn",
                # context="forkserver",
                # observation_space=obs_space,
                # action_space=act_space,
            )
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    (
                        osp.join(video_dir, f"env_{seed}")
                        if video_dir and (max_viz_envs < 0 or seed < max_viz_envs)
                        else None
                    ),
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs,
            render_backend="gpu",  # Use SAPIEN GPU backend
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="act",
                source_desc="act evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


if __name__ == "__main__":
    Q = 30

    # Use context manager to ensure proper cleanup
    evaluator = ManiSkillEvaluator(
        num_eval_episodes=4,
        num_eval_envs=2,
        video_dir="../../debug/mani_evaluator2",
    )

    def policy_func(obs):
        a = np.stack([evaluator.envs.action_space.sample() for _ in range(Q)], 1)
        return torch.as_tensor(a)

    results = evaluator.evaluate(
        policy=policy_func,
        predict_horizon=Q,
    )
    evaluator.close()
    print("Evaluation completed successfully!")
    print(f"Results: {results}")
