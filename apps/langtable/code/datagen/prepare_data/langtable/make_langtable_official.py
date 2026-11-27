"""LT oracle data generator for Protocol A (eval/main.py aligned).

Generates 5-reward, BLOCK_8 trajectory data matching the canonical LT
``eval/main.py`` evaluation distribution exactly. Per-episode output is written
in the format already consumed by ``umm.datasets.umm_v13.langtable_oracle_dataset.LangTableOracleDataset``.

Key differences from ``make_langtable_data.py`` (which we preserve untouched):

- Five reward factories selectable via ``--task`` (plus ``all``).
- ``block_mode=BLOCK_8`` (matches ``eval/main.py:81``; was ``N_CHOOSE_K``).
- Plan-feasibility retry loop (mirrors ``eval/main.py:109-117``): we call
  ``oracle.get_plan(env.compute_state())`` after every ``env.reset()`` and
  retry up to ``plan_retries`` times before giving up on a seed.
- Per-task ``generation_stats.json`` records attempts / plan-success /
  episode-success / skipped / mean-steps so post-hoc yield is auditable.

Output layout: ``<output_dir>/<task_name>/{images,depths,states,cameras,text,meta,packed,instances}/<seq_id>.*``
plus ``robot_embodiment.npz`` / ``object_embodiment.npz`` once per task dir
(byte-identical across tasks since all 16 block URDFs always load — accepted
for simplicity).

Determinism: practically deterministic within a process (Test 1b: L1=0.113/255
on 80-step replay). NOT bit-identical across workers — ``separate_blocks._sample_instruction``
uses the global numpy RNG. Saved ``instruction`` strings are what counts for
training; we do not rely on bit-for-bit cross-run equality.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
import traceback
from multiprocessing import Process

import numpy as np

# Reuse existing building blocks (import locally inside function for path setup).
_THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from make_langtable_data import (  # noqa: E402
    setup_paths,
    save_robot_meshes,
    save_object_meshes,
    Saver,
)

setup_paths()

from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
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


TASK_REGISTRY = {
    "b2b": ("block2block", block2block.BlockToBlockReward),
    "b2al": ("block2absolutelocation", block2absolutelocation.BlockToAbsoluteLocationReward),
    "b2brl": ("block2block_relative_location", block2block_relative_location.BlockToBlockRelativeLocationReward),
    "b2rl": ("block2relativelocation", block2relativelocation.BlockToRelativeLocationReward),
    "separate": ("separate_blocks", separate_blocks.SeparateBlocksReward),
}

# Max pre-rollout reset retries to find an oracle-feasible init.
DEFAULT_PLAN_RETRIES = 20


def _make_env_and_oracle(reward_factory_cls, oracle_std):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory_cls,
        render_text_in_image=True,
        render_depth=True,
        render_segmentation=True,
        render_semantic_segmentation=True,
        render_instance_poses=True,
    )
    oracle = ObstacleOrientedPushOracleBoard2dRRT(
        env, use_ee_planner=True, action_noise_std=oracle_std
    )
    return env, oracle


def _rollout_from_current_state(env, oracle, max_steps=200):
    """Roll out an episode from the env's CURRENT state.

    Unlike ``make_langtable_data.generate_episode`` this does NOT call
    ``env.reset()`` — it assumes the caller has already placed the env in
    the state we want to roll out from (e.g. verified feasible via
    ``oracle.get_plan``). We do NOT call ``pybullet_client.saveState`` — we
    capture the state via ``env.get_pybullet_state()`` (dict snapshot, no
    in-client state ID) to avoid pool leaks across thousands of episodes.

    We also do NOT call ``oracle.reset()`` explicitly — ``oracle.action(is_first=True)``
    on the first step invokes its internal reset + re-plan (see
    ``push_oracle_rrt_slowdown.action`` → ``oriented_push_oracle.py:235-236``),
    so an explicit reset here is redundant.
    """
    initial_state = env.get_pybullet_state()

    all_obs = []
    actions = []
    done = False

    obs = env._compute_observation()

    for i in range(max_steps):
        a = oracle.action(is_first=(i == 0))
        actions.append(a)
        obs["action"] = a
        next_obs, _, done, _ = env.step(a)
        all_obs.append(obs)
        obs = next_obs
        if done:
            all_obs.append(obs)
            break

    instruction = env.decode_instruction(obs["instruction"])
    return all_obs, initial_state, actions, instruction, bool(done)


def worker_process_official(
    worker_id,
    task_key,
    num_episodes,
    output_dir,
    oracle_std,
    base_seed,
    plan_retries,
    save_failure,
    stats_out_path,
):
    """Worker that generates `num_episodes` feasible successful episodes for one task.

    Writes per-worker stats to `stats_out_path` as JSON on exit.
    """
    setup_paths()  # Re-import in child in case fork loses it.

    task_subdir_name, reward_cls = TASK_REGISTRY[task_key]
    task_out = osp.join(output_dir, task_subdir_name)
    os.makedirs(task_out, exist_ok=True)

    stats = {
        "worker_id": worker_id,
        "task": task_key,
        "attempted": 0,
        "plan_success": 0,
        "plan_fail_skips": 0,
        "episode_success": 0,
        "episode_fail_skips": 0,
        "saved": 0,
        "steps_on_success": [],
        "wall_time_sec": 0.0,
    }
    t_worker_start = time.time()

    try:
        env, oracle = _make_env_and_oracle(reward_cls, oracle_std)
        fps = env._control_frequency
        saver = Saver(root=task_out)
        seed_counter = 0
        # Hard ceiling on total attempts: prevents infinite loop if a task
        # suddenly becomes universally infeasible. 10x headroom covers the
        # worst observed yield (~20%) plus variance. Raise via --max_attempt_multiplier
        # if you hit the ceiling without being infeasibility-limited.
        max_total_attempts = max(num_episodes * 10, 200)

        while stats["saved"] < num_episodes and stats["attempted"] < max_total_attempts:
            seed = base_seed + worker_id * 1_000_000 + seed_counter
            seed_counter += 1
            stats["attempted"] += 1

            # Plan-feasibility retry — mirror eval/main.py:109-117.
            # We seed ONLY the env RNG (not global numpy) to avoid polluting
            # separate_blocks._sample_instruction's global-RNG draw path.
            # Modular seed formula: keep within numpy's uint32 range while
            # decorrelating retries. 2**31 - 1 is a prime, any per-try offset
            # stays well inside the space.
            _SEED_MOD = 2**31 - 1
            plan_ok = False
            for plan_try in range(plan_retries):
                env.seed((seed + plan_try * 9973) % _SEED_MOD)
                env.reset()
                raw_state = env.compute_state(request_task_update=True)
                if oracle.get_plan(raw_state):
                    plan_ok = True
                    break

            if not plan_ok:
                stats["plan_fail_skips"] += 1
                continue
            stats["plan_success"] += 1

            # Roll out from the verified state WITHOUT another env.reset().
            try:
                episode_obs, initial_state, actions, text, success = (
                    _rollout_from_current_state(env, oracle)
                )
            except Exception as e:
                print(f"[{task_key}|w{worker_id}|seed{seed}] rollout raised: {e}", flush=True)
                traceback.print_exc()
                stats["episode_fail_skips"] += 1
                continue

            if success:
                stats["episode_success"] += 1
                stats["steps_on_success"].append(len(actions))
            else:
                stats["episode_fail_skips"] += 1

            if success or save_failure:
                name = f"s{seed:012d}_w{worker_id:03d}_n{stats['saved']:06d}"
                try:
                    saver.save_episode(
                        name=name,
                        all_obs=episode_obs,
                        actions=actions,
                        instruction=text,
                        initial_state=initial_state,
                        initial_state_id=0,  # no pybullet saveState — state dict is authoritative
                        seed=seed,
                        fps=fps,
                    )
                    stats["saved"] += 1
                except Exception as e:
                    print(f"[{task_key}|w{worker_id}|seed{seed}] save_episode raised: {e}", flush=True)
                    traceback.print_exc()
                    stats["episode_fail_skips"] += 1

            if seed_counter % 10 == 0:
                elapsed = time.time() - t_worker_start
                rate = stats["saved"] / max(elapsed, 1e-3)
                remain = max(num_episodes - stats["saved"], 0) / max(rate, 1e-6)
                print(
                    f"[{task_key}|w{worker_id}] attempt={stats['attempted']} saved={stats['saved']}/{num_episodes} "
                    f"plan_skip={stats['plan_fail_skips']} ep_skip={stats['episode_fail_skips']} "
                    f"rate={rate:.3f} ep/s eta={remain/60:.1f}min",
                    flush=True,
                )
        if stats["saved"] < num_episodes:
            print(
                f"[{task_key}|w{worker_id}] STOPPED at saved={stats['saved']}/{num_episodes} "
                f"after {stats['attempted']} attempts (max_total_attempts={max_total_attempts})",
                flush=True,
            )
    except Exception as e:
        print(f"[{task_key}|w{worker_id}] CRITICAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        stats["wall_time_sec"] = time.time() - t_worker_start
        if stats["steps_on_success"]:
            stats["mean_steps_on_success"] = float(np.mean(stats["steps_on_success"]))
            stats["p50_steps_on_success"] = float(np.percentile(stats["steps_on_success"], 50))
            stats["p95_steps_on_success"] = float(np.percentile(stats["steps_on_success"], 95))
        os.makedirs(osp.dirname(stats_out_path), exist_ok=True)
        with open(stats_out_path, "w") as f:
            json.dump(stats, f, indent=2)
        try:
            env.close()
        except Exception:
            pass
        print(f"[{task_key}|w{worker_id}] DONE saved={stats['saved']} wall={stats['wall_time_sec']:.1f}s", flush=True)


def run_task(
    task_key,
    output_dir,
    num_workers,
    num_episodes_per_worker,
    oracle_std,
    base_seed,
    plan_retries,
    save_failure,
    save_meshes_once,
):
    task_subdir_name, reward_cls = TASK_REGISTRY[task_key]
    task_out = osp.join(output_dir, task_subdir_name)
    os.makedirs(task_out, exist_ok=True)

    if save_meshes_once and not osp.exists(osp.join(task_out, "object_embodiment.npz")):
        save_robot_meshes(task_out)
        save_object_meshes(task_out)

    stats_dir = osp.join(task_out, "_stats")
    os.makedirs(stats_dir, exist_ok=True)
    t0 = time.time()
    procs = []
    for wid in range(num_workers):
        stats_path = osp.join(stats_dir, f"worker_{wid:03d}.json")
        p = Process(
            target=worker_process_official,
            args=(
                wid,
                task_key,
                num_episodes_per_worker,
                output_dir,
                oracle_std,
                base_seed,
                plan_retries,
                save_failure,
                stats_path,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    wall = time.time() - t0
    total = {"task": task_key, "num_workers": num_workers, "num_episodes_per_worker": num_episodes_per_worker, "wall_time_sec": wall, "workers": []}
    for wid in range(num_workers):
        sp = osp.join(stats_dir, f"worker_{wid:03d}.json")
        if osp.exists(sp):
            total["workers"].append(json.load(open(sp)))

    sums = {"attempted": 0, "plan_success": 0, "plan_fail_skips": 0, "episode_success": 0, "episode_fail_skips": 0, "saved": 0}
    steps = []
    for w in total["workers"]:
        for k in sums:
            sums[k] += w.get(k, 0)
        steps.extend(w.get("steps_on_success", []))
    sums["mean_steps_on_success"] = float(np.mean(steps)) if steps else 0.0
    sums["p50_steps_on_success"] = float(np.percentile(steps, 50)) if steps else 0.0
    sums["p95_steps_on_success"] = float(np.percentile(steps, 95)) if steps else 0.0
    total["aggregate"] = sums

    with open(osp.join(task_out, "generation_stats.json"), "w") as f:
        json.dump(total, f, indent=2)

    a = sums
    rate = a["saved"] / max(wall, 1e-3)
    print(
        f"[TASK {task_key}] wall={wall:.1f}s saved={a['saved']} attempted={a['attempted']} "
        f"plan_ok={a['plan_success']} plan_skip={a['plan_fail_skips']} ep_skip={a['episode_fail_skips']} "
        f"rate={rate:.3f} ep/s mean_steps={a['mean_steps_on_success']:.1f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Language-Table 5-reward BLOCK_8 oracle data gen (Protocol A aligned).")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_REGISTRY.keys()) + ["all"],
        help="Task to generate (or 'all' for all 5 sequentially).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/languagetable_official/oracle_gen_block8",
        help="Root output dir. Per-task subdirs created under this.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_episodes_per_worker", type=int, default=10)
    parser.add_argument("--oracle_std", type=float, default=0.0)
    parser.add_argument("--base_seed", type=int, default=20260419)
    parser.add_argument("--plan_retries", type=int, default=DEFAULT_PLAN_RETRIES)
    parser.add_argument("--save_failure", action="store_true")
    parser.add_argument("--no_save_meshes", action="store_true", help="Skip saving mesh npzs (use if already saved).")
    args = parser.parse_args()

    # Use fully-resolved absolute path (dereference symlinks once) so a
    # subsequent symlink replacement mid-run cannot divert writes onto a
    # different filesystem.
    output_dir = osp.realpath(osp.abspath(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resolved output_dir: {output_dir}", flush=True)

    tasks = list(TASK_REGISTRY.keys()) if args.task == "all" else [args.task]
    print(f"Running tasks: {tasks}")
    print(f"Output root: {output_dir}")
    print(f"Workers={args.num_workers} eps/worker={args.num_episodes_per_worker} -> "
          f"{args.num_workers*args.num_episodes_per_worker} eps/task, "
          f"{args.num_workers*args.num_episodes_per_worker*len(tasks)} total target")

    for tk in tasks:
        run_task(
            tk,
            output_dir,
            args.num_workers,
            args.num_episodes_per_worker,
            args.oracle_std,
            args.base_seed,
            args.plan_retries,
            args.save_failure,
            save_meshes_once=not args.no_save_meshes,
        )


if __name__ == "__main__":
    main()
