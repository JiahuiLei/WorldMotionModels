"""protocol-strict standalone Protocol A eval for WMM LangTable WITH per-rollout save.

This script extends ``eval_protocol_a_wmm.py`` by saving — for every
rollout — full per-step env observations, the rollout helper's full
return tuple (trace, ctx, uid_list, local_pcl, detailed_buffer, viz_frames),
the initial env state, and global rollout/viz configs. The saved metadata
is sufficient to *re-render* the rollout post-hoc through the WMM deploy
viz functions (``LangTableVisualizer.render_viz_frames``) without
re-running the simulator.

The eval semantics (env build, plan-feasibility loop, per-trial seeding,
``skip_env_reset=True``, ``strict_action_check=True``, 5 reward factories,
50 rollouts, env_seed=0, max_steps=200, BLOCK_8) are bit-identical to
``eval_protocol_a_wmm.py``. The protocol-locked params are NOT CLI-overridable.

CLI::

    python eval/langtable_official/eval_and_save_wmm.py \\
        --checkpoint <ckpt> --config <yaml> --output <dst> \\
        [--hist N] [--window N] [--act_horizon N]

``--hist`` / ``--window`` / ``--act_horizon`` default to the first entry
of ``cfg.model.langtable.{hist,window,act_horizon}`` (those fields are
lists). Use ``CUDA_VISIBLE_DEVICES`` to pick GPU; the script always uses
``cuda:0`` inside.

Output layout::

    <dst>/
      rollout_config.json    # global per-run config + git sha + timestamp
      viz_config.json        # viz_stand_at/look_at + computed viz_T_cam (4x4)
      results.json           # per-reward success table + profile (rewritten per-reward)
      _script_backup.py      # verbatim copy of this script
      <task_name>/
        <NNNN>/
          rollout.npz        # all per-step arrays + helper outputs (np.savez_compressed)
          meta.json          # success, steps, instruction, profile_sec, ...
          env.mp4            # viz_frames at fps=10 via imageio.mimwrite

Refuses to run if ``<dst>`` exists and is non-empty.
"""
import argparse
import contextlib
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch

PROJECT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(PROJECT, "src"))
sys.path.insert(0, osp.join(PROJECT, "ext", "lib_langtable"))
sys.path.insert(0, osp.join(PROJECT, "ext", "lib_io"))

from language_table.environments import blocks, language_table  # noqa: E402
from language_table.environments.rewards import (  # noqa: E402
    block2absolutelocation,
    block2block,
    block2block_relative_location,
    block2relativelocation,
    separate_blocks,
)
from language_table.replay.push_oracle_rrt_slowdown import (  # noqa: E402
    ObstacleOrientedPushOracleBoard2dRRT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Same key order as eval_protocol_a_wmm.py and eval/main.py:59-70.
REWARDS = {
    "blocktoblock": block2block.BlockToBlockReward,
    "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
    "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
    "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
    "separate": separate_blocks.SeparateBlocksReward,
}

# Protocol-locked params per the evaluation protocol.
# These are intentionally NOT exposed as CLI args.
PROTOCOL_NUM_ROLLOUTS = 50
PROTOCOL_ENV_SEED = 0
PROTOCOL_MAX_STEPS = 200
PROTOCOL_BLOCK_MODE = "BLOCK_8"
MP4_FPS = 10

# Smoke-test override. Set to a small int (e.g. 1) to run the smoke flow,
# then REVERT to None before commit. The final committed code MUST keep
# this at None so the script defaults to protocol_NUM_ROLLOUTS.
_NUM_ROLLOUTS_SMOKE: Optional[int] = None


# ---------------------------------------------------------------------------
# Loading / setup
# ---------------------------------------------------------------------------


def load_wmm_agent(
    checkpoint_path: str, config_path: str, device: torch.device
) -> Tuple[torch.nn.Module, Any, Any, Any]:
    """Load PLModel and return ``(eval_model, visualizer, cfg, pl_module)``.

    Mirrors ``eval_protocol_a_wmm.py::load_wmm_agent`` but also returns
    ``cfg`` and ``pl_module`` so the caller can read
    ``cfg.model.langtable`` (hist/window/act_horizon) and
    ``cfg.visualize.langtable`` (viz_stand_at/look_at).
    """
    from umm.training_utils import get_config
    from umm.models.wmm.model import PLModel

    argv_bak = sys.argv[:]
    sys.argv = ["eval", "--config", config_path]
    try:
        cfg = get_config()
    finally:
        sys.argv = argv_bak

    pl_module = PLModel.load_from_checkpoint(
        checkpoint_path, cfg=cfg, map_location=device, strict=False
    )
    pl_module.to(device).eval()
    eval_model = (
        pl_module.ema_model
        if getattr(pl_module, "use_ema", False)
        else pl_module.model
    )
    eval_model.eval()

    viz = pl_module.langtable_visualizer
    if viz is None or not getattr(viz, "flag", False):
        raise RuntimeError(
            "pl_module.langtable_visualizer is None or disabled — "
            "cfg.model.langtable block is missing or empty."
        )
    return eval_model, viz, cfg, pl_module


def get_git_sha(project_root: str) -> str:
    """Return short HEAD sha or 'unknown' on failure (no exceptions)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception as e:  # subprocess error / not a git repo
        log.warning("get_git_sha: git rev-parse failed: %r", e)
        return "unknown"


def assert_dst_empty(dst: str) -> None:
    """Refuse to run if ``dst`` exists and is non-empty."""
    if osp.isdir(dst) and len(os.listdir(dst)) > 0:
        log.error(
            "Output directory %s already exists and is non-empty. "
            "Refusing to overwrite. Pick a fresh path or remove it manually.",
            dst,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Source + git snapshot
# ---------------------------------------------------------------------------


def snapshot_src(dst: str, project_root: str) -> float:
    """Mirror training's `src_bck` convention into `<dst>/src_bck/`.

    Copies `src/`, `configs/` (FULL — every domain), `eval/`, and
    `train.py`. Excludes large data artifacts (npz/ckpt/pdf/mp4/...) and
    Python bytecode so the snapshot stays in the tens of MB rather than
    the GB range. Re-uses rsync (already a hard dep on Linux dev boxes).

    Returns wall seconds spent in the snapshot.
    """
    t0 = time.time()
    src_bck = osp.join(dst, "src_bck")
    os.makedirs(src_bck, exist_ok=True)
    excludes = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        # serialized data
        "--exclude=*.npz", "--exclude=*.npy", "--exclude=*.pkl",
        "--exclude=*.p",
        "--exclude=*.pt", "--exclude=*.ckpt", "--exclude=*.bin",
        "--exclude=*.zip", "--exclude=*.tar", "--exclude=*.tar.gz",
        "--exclude=*.gz", "--exclude=*.h5", "--exclude=*.hdf5",
        # docs / media
        "--exclude=*.pdf", "--exclude=*.mp4",
        "--exclude=*.png", "--exclude=*.jpg", "--exclude=*.jpeg",
        # 3D mesh / sim assets — large in eval/<domain>/data/
        "--exclude=*.obj", "--exclude=*.ply", "--exclude=*.stl",
        "--exclude=*.dae", "--exclude=*.glb", "--exclude=*.gltf",
        # any nested data/ dir under code trees (e.g. eval/omomo/data/)
        "--exclude=data/",
    ]
    for sub in ("src", "configs", "eval"):
        src_path = osp.join(project_root, sub)
        if not osp.isdir(src_path):
            log.warning("snapshot_src: %s not found, skipping", src_path)
            continue
        dst_path = osp.join(src_bck, sub) + "/"
        subprocess.run(
            ["rsync", "-a"] + excludes + [src_path + "/", dst_path],
            check=True,
        )
    train_py = osp.join(project_root, "train.py")
    if osp.isfile(train_py):
        shutil.copyfile(train_py, osp.join(src_bck, "train.py"))
    return time.time() - t0


def write_git_info(dst: str, project_root: str) -> None:
    """Write `<dst>/git_info.txt` with branch/HEAD/status/diff/log -1.

    Captures every signal needed to identify the exact code state behind
    the eval — even if the branch is later force-pushed or the local
    workspace is wiped. Read-only git operations only.
    """
    info_path = osp.join(dst, "git_info.txt")
    queries = [
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("HEAD", ["git", "rev-parse", "HEAD"]),
        ("HEAD short", ["git", "rev-parse", "--short", "HEAD"]),
        ("upstream", ["git", "rev-parse", "--abbrev-ref", "@{u}"]),
        ("upstream HEAD", ["git", "rev-parse", "@{u}"]),
        ("log -1",
         ["git", "log", "-1",
          "--pretty=format:%h %an %ad %s", "--date=iso"]),
        ("log -5 (compact)",
         ["git", "log", "-5", "--oneline"]),
        ("status (porcelain)", ["git", "status", "--porcelain"]),
        ("diff --stat (unstaged)", ["git", "diff", "--stat"]),
        ("diff --stat (staged)", ["git", "diff", "--cached", "--stat"]),
        ("remote -v", ["git", "remote", "-v"]),
    ]
    parts = [
        f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"project_root: {project_root}",
        f"hostname: {os.uname().nodename}",
    ]
    for label, cmd in queries:
        try:
            out = subprocess.check_output(
                cmd, cwd=project_root, stderr=subprocess.DEVNULL
            ).decode().rstrip()
        except Exception as e:
            out = f"<error: {type(e).__name__}: {e}>"
        parts.append(f"\n--- {label} ---\n{out}")
    with open(info_path, "w") as f:
        f.write("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Global config writers
# ---------------------------------------------------------------------------


def write_global_configs(
    dst: str,
    cfg: Any,
    args: argparse.Namespace,
    hist: int,
    window: int,
    act_horizon: int,
    device: torch.device,
    project_root: str,
    num_rollouts: int,
) -> None:
    """Write rollout_config.json, viz_config.json, and copy this script."""
    os.makedirs(dst, exist_ok=True)
    from umm.models.wmm.viz_helpers.utils import make_viz_T_cam  # local

    # rollout_config.json --------------------------------------------------
    # The eval loads from the LOCAL COPY of the ckpt (see main()), so
    # `checkpoint` records the original src path for traceability while
    # `checkpoint_loaded_from` records the actual on-disk file the model
    # was loaded from. They differ by a basename swap into `dst/`.
    src_ckpt = osp.abspath(args.checkpoint)
    dst_ckpt_basename = osp.basename(src_ckpt)
    rollout_cfg = {
        "checkpoint": args.checkpoint,
        "checkpoint_src": src_ckpt,
        "checkpoint_loaded_from": osp.join(args.output, dst_ckpt_basename),
        "config": args.config,
        "output": args.output,
        "hist": int(hist),
        "window": int(window),
        "act_horizon": int(act_horizon),
        "num_rollouts": int(num_rollouts),
        "env_seed": int(PROTOCOL_ENV_SEED),
        "max_steps": int(PROTOCOL_MAX_STEPS),
        "block_mode": PROTOCOL_BLOCK_MODE,
        "device": str(device),
        "git_sha": get_git_sha(project_root),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_locked": True,
        "smoke_override_active": _NUM_ROLLOUTS_SMOKE is not None,
    }
    with open(osp.join(dst, "rollout_config.json"), "w") as f:
        json.dump(rollout_cfg, f, indent=2)

    # viz_config.json ------------------------------------------------------
    viz_cfg_dict = cfg.get("visualize", {}).get("langtable", {})
    viz_stand_at = viz_cfg_dict.get("viz_stand_at", None)
    viz_look_at = viz_cfg_dict.get("viz_look_at", [0.0, 0.0, 0.0])
    if viz_stand_at is None:
        raise RuntimeError(
            "cfg.visualize.langtable.viz_stand_at is not set — required to "
            "compute viz_T_cam for re-rendering."
        )
    # ``make_viz_T_cam`` ignores the ``batch`` arg when ``viz_stand_at`` is
    # provided. Pass an empty dict to keep the API call honest.
    viz_T_cam = make_viz_T_cam(
        {"viz_stand_at": viz_stand_at, "viz_look_at": viz_look_at}, {}
    )
    viz_T_cam_np = viz_T_cam.cpu().numpy().astype(np.float64)
    viz_config = {
        "viz_stand_at": list(viz_stand_at),
        "viz_look_at": list(viz_look_at),
        "viz_T_cam": viz_T_cam_np.tolist(),  # 4x4 row-major
    }
    with open(osp.join(dst, "viz_config.json"), "w") as f:
        json.dump(viz_config, f, indent=2)

    # _script_backup.py ----------------------------------------------------
    shutil.copyfile(__file__, osp.join(dst, "_script_backup.py"))
    log.info("Wrote global configs + script backup to %s", dst)


# ---------------------------------------------------------------------------
# Step capture context manager — monkey-patches env.step to log per-step data.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def step_capture(env: Any):
    """Context manager: monkey-patches ``env.step`` to record every call.

    Yields a list that is appended to on each ``env.step(action)`` call as
    ``{"action": np.ndarray(2,), "next_obs": dict, "reward": float,
       "done": bool, "info": dict}``.

    The original ``env.step`` is ALWAYS restored in ``finally``, even if
    the wrapped block raises. This protects against the
    ``_owns_env`` pattern.
    """
    captured: List[Dict[str, Any]] = []
    original_step = env.step

    def wrapped(action):
        result = original_step(action)
        # LT step returns (obs, reward, done, info); record a *copy* of
        # action to decouple from any caller-side mutation.
        captured.append({
            "action": np.asarray(action, dtype=np.float32).copy(),
            "next_obs": result[0],
            "reward": float(result[1]),
            "done": bool(result[2]),
            "info": result[3],
        })
        return result

    env.step = wrapped
    try:
        yield captured
    finally:
        env.step = original_step


# ---------------------------------------------------------------------------
# Per-step obs assembly
# ---------------------------------------------------------------------------


def _stack_obs_field(
    initial_obs: Dict[str, Any],
    captured: List[Dict[str, Any]],
    field: str,
    dtype: Any,
) -> np.ndarray:
    """Stack obs[field] across (initial + captured next_obs) into ndarray.

    `dtype` is ``Any`` because numpy accepts a wide union (np.float32 type,
    np.dtype instance, dtype-like string) — annotating it strictly as
    ``np.dtype`` makes Pyright reject the np.float32 / np.int32 type
    objects that callers pass in.
    """
    frames = [initial_obs[field]]
    for entry in captured:
        frames.append(entry["next_obs"][field])
    return np.asarray(np.stack(frames, axis=0), dtype=dtype)


def _build_instance_poses_per_step(
    initial_obs: Dict[str, Any],
    captured: List[Dict[str, Any]],
    obj_uids: List[Any],
    ee_uids: List[Any],
) -> np.ndarray:
    """(T+1, N, 4, 4) per-step pose stack matching helper's uid order."""
    uid_order = list(obj_uids) + list(ee_uids)
    out = []
    for obs in [initial_obs] + [c["next_obs"] for c in captured]:
        per_step = np.asarray(
            [obs["instance_poses"][uid] for uid in uid_order], dtype=np.float32
        )
        out.append(per_step)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Per-episode runner
# ---------------------------------------------------------------------------


def run_one_episode(
    env: Any,
    agent: torch.nn.Module,
    visualizer: Any,
    ep_num: int,
    hist: int,
    window: int,
    act_horizon: int,
    device: torch.device,
    env_seed: int,
    max_steps: int,
    episode_dst: str,
) -> Dict[str, Any]:
    """Run ONE episode + save per-rollout artefacts to ``episode_dst``.

    Returns a per-episode summary dict. Always restores ``env.step`` even
    on failure.
    """
    os.makedirs(episode_dst, exist_ok=True)

    # 1. Plan-feasibility retry — same loop shape as eval_protocol_a_wmm.py.
    oracle = ObstacleOrientedPushOracleBoard2dRRT(env, use_ee_planner=True)
    plan_ok = False
    raw_state: Any = None  # bound below; explicit init silences "possibly unbound"
    for _ in range(20):
        env.reset()
        raw_state = env.compute_state()
        if oracle.get_plan(raw_state):
            plan_ok = True
            break
    if not plan_ok:
        log.warning("ep %d: no feasible init after 20 retries", ep_num)
        # Write a minimal meta so the directory is not silently empty.
        meta = {
            "ep_num": int(ep_num),
            "succ": False,
            "succ_steps": -1,
            "seed": int(env_seed + ep_num),
            "instruction": "",
            "task_name": osp.basename(osp.dirname(episode_dst)),
            "num_steps": 0,
            "reason": "no_feasible_init",
            "profile_sec": {"rollout": 0.0, "save": 0.0},
        }
        with open(osp.join(episode_dst, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return {
            "ep": ep_num,
            "success": False,
            "steps": -1,
            "reason": "no_feasible_init",
            "rollout_sec": 0.0,
            "save_sec": 0.0,
        }

    # 2. Reuse raw_state from the LAST successful plan-feasibility
    # iteration as initial_state, and build initial_obs from it via
    # env._compute_observation(state=raw_state). This routes through the
    # state-arg branch of _compute_observation that does NOT call
    # _compute_state again — avoiding any extra _compute_state /
    # _set_task_info invocation beyond what eval_protocol_a_wmm.py does.
    # Net result: same _compute_state call count as the reference script
    # between plan-feasibility-OK and helper entry.
    initial_state = raw_state
    initial_obs = env._compute_observation(state=raw_state)
    obj_uids = env.get_all_active_object_ids()
    ee_uids = [k for k, v in initial_obs["instance_map"].items() if v == 4]
    # Helper line 462 asserts len(ee_uids)==1 internally; we keep the
    # same contract here so the failure surfaces at the per-episode try
    # boundary in run_one_reward (recorded as a failure entry) rather
    # than crashing inside the helper after the rollout has started.
    if len(ee_uids) != 1:
        raise RuntimeError(
            f"ep {ep_num}: expected exactly 1 EE in instance_map, got "
            f"{len(ee_uids)} — matches helper's `assert len(ee_uids)==1`"
        )

    # 3. Wrap env.step + run rollout via helper.
    t_roll_start = time.time()
    with step_capture(env) as captured:
        (
            succ,
            succ_steps,
            trace,
            ctx,
            uid_list,
            local_pcl,
            viz_frames,
            instruction,
            detailed_buffer,
        ) = visualizer.rollout_full_states_trivial_policy(
            agent=agent,
            seed=env_seed + ep_num,
            max_steps=max_steps,
            hist=hist,
            window=window,
            act_horizon=act_horizon,
            device=device,
            env=env,
            strict_action_check=True,
            skip_env_reset=True,
        )
    rollout_sec = time.time() - t_roll_start

    # 4. Assemble per-step arrays. T = number of env.step calls.
    T = len(captured)
    if len(viz_frames) != T + 1:
        log.warning(
            "ep %d: viz_frames length %d != T+1 (%d). Helper invariant violated.",
            ep_num, len(viz_frames), T + 1,
        )

    # rgb_per_step: prefer viz_frames as truth (LT renders rgb identically
    # to obs["rgb"]; this also gives us mp4 frames in lockstep).
    rgb_per_step = np.asarray(np.stack(viz_frames, axis=0), dtype=np.uint8)

    # depth: env _may_ return uint16 internally; cast to float32 to be safe.
    depth_per_step = _stack_obs_field(initial_obs, captured, "depth", np.float32)

    # Instance segmentation (raw pybullet ids). Cast to int64 — pybullet
    # encodes seg_id as obj_id + ((link_idx+1) << 24), which can exceed
    # int32 range with high obj counts × deep link indices. int64 is
    # safe for any conceivable env without a memory penalty at our
    # rollout sizes.
    instance_seg_per_step = _stack_obs_field(
        initial_obs, captured, "segmentation", np.int64
    )

    instance_poses_per_step = _build_instance_poses_per_step(
        initial_obs, captured, obj_uids, ee_uids
    )

    ee_target_per_step = _stack_obs_field(
        initial_obs, captured, "effector_target_translation", np.float32
    )
    effector_translation_per_step = _stack_obs_field(
        initial_obs, captured, "effector_translation", np.float32
    )

    if T > 0:
        applied_action_per_step = np.stack(
            [c["action"] for c in captured], axis=0
        ).astype(np.float32)
    else:
        applied_action_per_step = np.zeros((0, 2), dtype=np.float32)

    # detailed_buffer -> structured object array
    gen_trace_snapshots = np.array(
        [
            {"trace": np.asarray(t, dtype=np.float32),
             "hist_len": int(h),
             "timestep": int(ts)}
            for (t, h, ts) in detailed_buffer
        ],
        dtype=object,
    )

    # 5. Save artefacts.
    t_save_start = time.time()
    npz_path = osp.join(episode_dst, "rollout.npz")
    # Helper returns trace/ctx as numpy (np.stack), local_pcl as a torch
    # Tensor (CPU, float32). Normalize to numpy for npz storage.
    if isinstance(local_pcl, torch.Tensor):
        local_pcl_np = local_pcl.detach().cpu().numpy().astype(np.float32)
    else:
        local_pcl_np = np.asarray(local_pcl, dtype=np.float32)
    trace_np = trace.detach().cpu().numpy() if isinstance(trace, torch.Tensor) else np.asarray(trace)
    ctx_np = ctx.detach().cpu().numpy() if isinstance(ctx, torch.Tensor) else np.asarray(ctx)

    np.savez_compressed(
        npz_path,
        # Helper outputs
        trace=trace_np.astype(np.float32),
        ctx=ctx_np.astype(np.float32),
        uid_list=np.array(uid_list, dtype=object),
        local_pcl=local_pcl_np,
        # Per-step env captures
        instance_poses_per_step=instance_poses_per_step,
        rgb_per_step=rgb_per_step,
        depth_per_step=depth_per_step,
        instance_seg_per_step=instance_seg_per_step,
        ee_target_per_step=ee_target_per_step,
        effector_translation_per_step=effector_translation_per_step,
        applied_action_per_step=applied_action_per_step,
        # Snapshots from detailed_buffer
        gen_trace_snapshots=gen_trace_snapshots,
        # Initial state (env.compute_state at reset, post-plan-feasibility)
        initial_state=np.array([initial_state], dtype=object),
        # Helper-side uid grouping context (handy at re-render time)
        obj_uids=np.array(obj_uids, dtype=object),
        ee_uids=np.array(ee_uids, dtype=object),
    )

    # MP4
    mp4_path = osp.join(episode_dst, "env.mp4")
    try:
        imageio.mimwrite(
            mp4_path,
            [np.asarray(f, dtype=np.uint8) for f in viz_frames],
            fps=MP4_FPS,
        )
    except Exception:
        log.exception("ep %d: imageio.mimwrite failed for %s", ep_num, mp4_path)

    save_sec = time.time() - t_save_start

    meta = {
        "ep_num": int(ep_num),
        "succ": bool(succ),
        "succ_steps": int(succ_steps),
        "seed": int(env_seed + ep_num),
        "instruction": str(instruction) if instruction is not None else "",
        "task_name": osp.basename(osp.dirname(episode_dst)),
        "num_steps": int(T),
        "profile_sec": {"rollout": float(rollout_sec), "save": float(save_sec)},
        "rollout_params": {
            "hist": int(hist),
            "window": int(window),
            "act_horizon": int(act_horizon),
        },
        "ckpt_path": None,  # filled in by run_one_reward via post-edit (kept None here for purity)
        "config_path": None,
    }
    with open(osp.join(episode_dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "[ep %d] succ=%s steps=%d roll=%.1fs save=%.1fs",
        ep_num, succ, succ_steps, rollout_sec, save_sec,
    )
    return {
        "ep": int(ep_num),
        "success": bool(succ),
        "steps": int(succ_steps),
        "rollout_sec": float(rollout_sec),
        "save_sec": float(save_sec),
        "instruction": str(instruction) if instruction is not None else "",
    }


# ---------------------------------------------------------------------------
# Per-reward runner
# ---------------------------------------------------------------------------


def run_one_reward(
    reward_name: str,
    reward_cls: Any,
    agent: torch.nn.Module,
    visualizer: Any,
    num_rollouts: int,
    max_steps: int,
    env_seed: int,
    hist: int,
    window: int,
    act_horizon: int,
    device: torch.device,
    dst_task_dir: str,
    ckpt_path: str,
    config_path: str,
) -> Dict[str, Any]:
    """Run num_rollouts episodes for one reward factory."""
    log.info(
        "[%s] starting %d rollouts, seed=%d", reward_name, num_rollouts, env_seed,
    )
    os.makedirs(dst_task_dir, exist_ok=True)
    env = language_table.LanguageTable(
        block_mode=getattr(blocks.LanguageTableBlockVariants, PROTOCOL_BLOCK_MODE),
        reward_factory=reward_cls,
        render_text_in_image=True,
        render_depth=True,
        render_segmentation=True,
        render_semantic_segmentation=True,
        render_instance_poses=True,
        seed=env_seed,
    )

    successes = 0
    per_ep: List[Dict[str, Any]] = []
    rollout_sec_total = 0.0
    save_sec_total = 0.0

    try:
        for ep_num in range(num_rollouts):
            episode_dst = osp.join(dst_task_dir, f"{ep_num:04d}")
            # Per-episode try/except: a single rollout crash (sim
            # segfault, save failure, OOM during inference) MUST NOT kill
            # the entire reward's results. Append a failure entry and
            # continue. Note: silent C-extension crash
            # mid-reward — per-reward checkpoint saved 2 of 5 rewards;
            # we extend the same idea to per-episode resilience.
            try:
                res = run_one_episode(
                    env=env,
                    agent=agent,
                    visualizer=visualizer,
                    ep_num=ep_num,
                    hist=hist,
                    window=window,
                    act_horizon=act_horizon,
                    device=device,
                    env_seed=env_seed,
                    max_steps=max_steps,
                    episode_dst=episode_dst,
                )
            except Exception as e:
                log.exception(
                    "[%s] ep %d: rollout/save FAILED — recording as failure "
                    "entry, continuing to next episode", reward_name, ep_num,
                )
                per_ep.append({
                    "ep": int(ep_num),
                    "success": False,
                    "succ_steps": -1,
                    "rollout_sec": 0.0,
                    "save_sec": 0.0,
                    "instruction": "",
                    "reason": f"exception: {type(e).__name__}: {e}",
                })
                continue

            # Patch ckpt_path / config_path into per-episode meta.json.
            meta_path = osp.join(episode_dst, "meta.json")
            if osp.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    meta["ckpt_path"] = ckpt_path
                    meta["config_path"] = config_path
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception as e:
                    log.warning("ep %d: meta.json patch failed: %r", ep_num, e)

            per_ep.append(res)
            if res.get("success"):
                successes += 1
            rollout_sec_total += float(res.get("rollout_sec", 0.0))
            save_sec_total += float(res.get("save_sec", 0.0))
    finally:
        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            try:
                close_fn()
            except Exception as e:
                log.warning("env.close failed: %r", e)

    rate = successes / max(num_rollouts, 1)
    log.info(
        "[%s] done: %d/%d = %.3f (roll_total=%.1fs save_total=%.1fs)",
        reward_name, successes, num_rollouts, rate,
        rollout_sec_total, save_sec_total,
    )
    return {
        "reward": reward_name,
        "successes": int(successes),
        "total": int(num_rollouts),
        "rate": float(rate),
        "episodes": per_ep,
        "rollout_sec_total": float(rollout_sec_total),
        "save_sec_total": float(save_sec_total),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _resolve_int_param(
    cli_val: Optional[int], cfg_field: Any, name: str
) -> int:
    """If CLI value None, take cfg_field[0] (lists in YAML). Validate."""
    if cli_val is not None:
        return int(cli_val)
    if cfg_field is None:
        raise RuntimeError(
            f"--{name} not given and cfg.model.langtable.{name} not set"
        )
    if isinstance(cfg_field, list):
        if len(cfg_field) == 0:
            raise RuntimeError(
                f"cfg.model.langtable.{name} is an empty list"
            )
        return int(cfg_field[0])
    return int(cfg_field)


def _write_results_json(dst: str, all_results: Dict[str, Any], elapsed: float) -> None:
    """Write/overwrite results.json. Called after every reward."""
    rollout_sec_total = sum(
        r.get("rollout_sec_total", 0.0) for r in all_results.values()
    )
    save_sec_total = sum(
        r.get("save_sec_total", 0.0) for r in all_results.values()
    )
    if all_results:
        avg_rate = float(np.mean([r["rate"] for r in all_results.values()]))
    else:
        avg_rate = 0.0
    payload = {
        "rewards": all_results,
        "avg_rate": avg_rate,
        "elapsed_sec": float(elapsed),
        "profile": {
            "rollout_sec_total": float(rollout_sec_total),
            "save_sec_total": float(save_sec_total),
            "per_task": {
                k: {
                    "rollout_sec": v.get("rollout_sec_total", 0.0),
                    "save_sec": v.get("save_sec_total", 0.0),
                }
                for k, v in all_results.items()
            },
        },
    }
    with open(osp.join(dst, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "protocol-strict standalone Protocol A WMM eval that saves full "
            "per-rollout metadata for post-hoc re-rendering."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to UMM WMM .ckpt")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--output", required=True, help="Output directory (must be empty/missing)")
    parser.add_argument("--hist", type=int, default=None,
                        help="Override cfg.model.langtable.hist[0]")
    parser.add_argument("--window", type=int, default=None,
                        help="Override cfg.model.langtable.window[0]")
    parser.add_argument("--act_horizon", type=int, default=None,
                        help="Override cfg.model.langtable.act_horizon[0]")
    args = parser.parse_args()

    # Resolve number of rollouts (smoke override applies if set).
    num_rollouts = (
        _NUM_ROLLOUTS_SMOKE if _NUM_ROLLOUTS_SMOKE is not None else PROTOCOL_NUM_ROLLOUTS
    )
    if _NUM_ROLLOUTS_SMOKE is not None:
        log.warning(
            "SMOKE OVERRIDE ACTIVE: _NUM_ROLLOUTS_SMOKE=%d (protocol default is %d). "
            "REVERT TO None BEFORE COMMIT.",
            _NUM_ROLLOUTS_SMOKE, PROTOCOL_NUM_ROLLOUTS,
        )

    assert_dst_empty(args.output)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda:0")
    log.info("Device = %s (CUDA_VISIBLE_DEVICES=%s)",
             device, os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    # Backup-then-load: copy the source ckpt into the dst FIRST, then
    # load the agent from the COPY. This guarantees every artifact dir
    # is a self-contained snapshot — `last.ckpt` lives next to
    # `results.json`, `rollout_config.json`, and `_script_backup.py`.
    # If the original training log is deleted/rotated/moved later, the
    # copy in dst is still loadable for re-render or re-eval.
    src_ckpt = osp.abspath(args.checkpoint)
    dst_ckpt = osp.join(args.output, osp.basename(src_ckpt))
    log.info("Copying ckpt: %s -> %s (size=%.1f MB)",
             src_ckpt, dst_ckpt, osp.getsize(src_ckpt) / (1 << 20))
    t_cp = time.time()
    shutil.copyfile(src_ckpt, dst_ckpt)
    log.info("Ckpt copy done in %.1fs", time.time() - t_cp)

    # Source snapshot + git state — full reproducibility. Mirrors the
    # `src_bck` convention used by training (logs/<expname>/version_N/
    # src_bck/{src,configs,eval,train.py}). git_info.txt captures
    # branch/HEAD/status/diff so the exact code state is recoverable
    # even if the local tree changes or the branch is force-pushed.
    log.info("Snapshotting src/configs/eval into %s/src_bck/ ...", args.output)
    snap_sec = snapshot_src(args.output, PROJECT)
    log.info("src snapshot done in %.1fs", snap_sec)
    write_git_info(args.output, PROJECT)
    log.info("Wrote git_info.txt")

    # Load model + visualizer + cfg from the dst copy (NOT from src).
    eval_model, viz, cfg, _pl_module = load_wmm_agent(
        dst_ckpt, args.config, device
    )

    lt_cfg = cfg.get("model", {}).get("langtable", {})
    hist = _resolve_int_param(args.hist, lt_cfg.get("hist"), "hist")
    window = _resolve_int_param(args.window, lt_cfg.get("window"), "window")
    act_horizon = _resolve_int_param(
        args.act_horizon, lt_cfg.get("act_horizon"), "act_horizon"
    )
    log.info(
        "Resolved rollout params: hist=%d window=%d act_horizon=%d num_rollouts=%d",
        hist, window, act_horizon, num_rollouts,
    )

    # Write global configs + script backup BEFORE any rollout so partial
    # crashes still leave the rollout context on disk.
    write_global_configs(
        dst=args.output,
        cfg=cfg,
        args=args,
        hist=hist,
        window=window,
        act_horizon=act_horizon,
        device=device,
        project_root=PROJECT,
        num_rollouts=num_rollouts,
    )

    all_results: Dict[str, Any] = {}
    t0 = time.time()
    with torch.no_grad():
        for reward_name, reward_cls in REWARDS.items():
            dst_task_dir = osp.join(args.output, reward_name)
            res = run_one_reward(
                reward_name=reward_name,
                reward_cls=reward_cls,
                agent=eval_model,
                visualizer=viz,
                num_rollouts=num_rollouts,
                max_steps=PROTOCOL_MAX_STEPS,
                env_seed=PROTOCOL_ENV_SEED,
                hist=hist,
                window=window,
                act_horizon=act_horizon,
                device=device,
                dst_task_dir=dst_task_dir,
                ckpt_path=args.checkpoint,
                config_path=args.config,
            )
            all_results[reward_name] = res
            # Per-reward checkpoint write.
            _write_results_json(args.output, all_results, time.time() - t0)

    # Final summary --------------------------------------------------------
    elapsed = time.time() - t0
    avg_rate = float(np.mean([r["rate"] for r in all_results.values()])) if all_results else 0.0
    print("\n=== PROTOCOL A SUMMARY (WMM, eval_and_save) ===")
    for name, r in all_results.items():
        print(f"  {name}: {r['successes']}/{r['total']} = {r['rate']:.3f} "
              f"(roll={r['rollout_sec_total']:.0f}s save={r['save_sec_total']:.0f}s)")
    print(f"  AVG: {avg_rate:.3f}  total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
