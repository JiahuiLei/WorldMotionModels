import gymnasium as gym
import logging
import os

# To prevent SAPIEN's Vulkan/EGL initialization from destroying PyRender's EGL context,
# we force SAPIEN to initialize its renderer here, BEFORE PyRender ever gets imported or used.
# Since this module is imported early, the first SapienRenderer is created safely.
try:
    import sapien.render
    _dummy_renderer = sapien.render.SapienRenderer()
except Exception as e:
    logging.warning(f"Failed to pre-initialize SapienRenderer: {e}")

def make_maniskill_viz_env(
    env_id,
    seed=0,
    sim_backend="physx_cpu",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=None,
    H=512,
    W=512,
    video_dir=None,
):
    """
    Creates a single ManiSkill environment for visualization/rollout.
    """
    env_kwargs = dict(
        control_mode=control_mode,
        reward_mode="sparse",
        obs_mode="rgb+depth+segmentation",
        render_mode="rgb_array",
        sensor_configs=dict(width=W, height=H),
    )
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    logging.info(f"make_maniskill_viz_env: {env_id}, seed={seed}, backend={sim_backend}")
    
    env = gym.make(
        env_id,
        reconfiguration_freq=1,
        **env_kwargs,
        sim_backend=sim_backend,
        enable_shadow=False,
        # Default back to GPU rendering for performance
        render_backend="gpu",
    )
    
    # Optionally add video recording wrapper if needed
    if video_dir is not None:
        from mani_skill.utils.wrappers import RecordEpisode
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            info_on_video=True,
            source_type="act",
        )
        
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
