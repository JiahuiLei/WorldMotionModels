"""Environment creation utilities for ACT evaluation.

Provides factory functions for constructing vectorized ManiSkill
environments with optional video recording, supporting both CPU
and GPU simulation backends.
"""

from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.

    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: The environment id (e.g., 'PickCube-v1').
        num_envs: The number of parallel environments.
        sim_backend: The simulation backend to use. Can be 'physx_cpu' or
            'physx_gpu'.
        env_kwargs: The environment keyword arguments. You can also pass in
            max_episode_steps in env_kwargs to override the default max
            episode steps for the environment.
        other_kwargs: Additional keyword arguments passed through to the
            CPU environment factory.
        video_dir: The directory to save the videos. If None no videos are
            recorded.
        wrappers: The list of wrappers to apply to the environment.

    Returns:
        A vectorized gymnasium environment ready for evaluation.

    Example:
        >>> env = make_eval_envs('PickCube-v1', num_envs=4,
        ...     sim_backend='physx_cpu', env_kwargs={}, other_kwargs={})
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            """Create a single CPU environment wrapped for evaluation.

            Args:
                env_id: The environment id string.
                seed: Random seed for the environment.
                video_dir: Directory for video recording, or None.
                env_kwargs: Environment keyword arguments.
                other_kwargs: Additional keyword arguments (unused).

            Returns:
                A callable thunk that creates the environment when invoked.
            """

            def thunk():
                """Create and wrap a single environment instance.

                Returns:
                    The wrapped gymnasium environment.
                """
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
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
                        source_desc="act evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="spawn")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
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
