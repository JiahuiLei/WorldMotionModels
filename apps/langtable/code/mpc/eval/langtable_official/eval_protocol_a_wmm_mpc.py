"""Protocol A LangTable eval with optional gradient-based MPC + sharding.

Single user-facing entry point. Internally forks N child shards, runs
each on a separate GPU, and merges results into one unified output dir.

Reproducibility (CRITICAL — different from ``eval_protocol_a_wmm.py``):

  - A FRESH env is built per episode, seeded by ``env_seed + ep_num``.
    The plan-feasibility loop reseeds per retry (deterministic in
    ``ep_num``). All MPC RNG (candidate noise, agent inference) is
    seeded from the per-replan seed = ``ep_seed * 1000 + replan_idx``.
  - Episode E's (succ, succ_steps, frames) is a deterministic function
    of (env_seed, ep_num) ALONE — independent of:
      * shard count (1 vs 10 shards → byte-identical results)
      * shard ordering (shard 3 vs shard 7 → identical for ep E)
      * episode order WITHIN a task

This is INTENTIONALLY different from ``eval_protocol_a_wmm.py``'s
single-env, sequential-reset convention. That convention's results
depend on cumulative env state and are NOT shard-invariant.

Two phases — eval ≠ render:

  Phase 1 (this script): rollout, save raw artifacts
    - frames.npz (T, H, W, 3) uint8
    - intermediates.npz (MPC only) — full per-replan optimization data
    - real_trace.npy, imagined_goals.npy
    - meta.json
    NO mp4, NO matplotlib, NO 3D rendering.

  Phase 2 (``render_mpc_eval_videos.py``): user-triggered post-eval
    rendering of mp4 + plots from any episode subset. Cheap re-runs.

Sharding model:

  - Episode-shard ONLY (no task-shard). Each shard iterates::

        for task in tasks:
            for ep_num in range(num_rollouts):
                if ep_num % num_shards == shard_id:
                    run(task, ep_num)

    All shards work on task 0 in parallel, then task 1, etc.
    No barrier — shards drift; that's fine.
  - GPU selection precedence:
      1. ``--gpus 0,1,...`` (explicit)
      2. ``CUDA_VISIBLE_DEVICES`` env var (filter free GPUs from that mask)
      3. Default: scan all GPUs, pick free ones (mem<500MiB AND util<10%)

Output layout::

    <output>/
        results.json                    # final aggregate (after merge)
        rollout_config.json             # all CLI args
        <reward>/
            ep_<env_seed+ep_num>/
                meta.json
                frames.npz
                intermediates.npz       # MPC only
                real_trace.npy
                imagined_goals.npy

Example (auto-shard, MPC mode)::

    python eval/langtable_official/eval_protocol_a_wmm_mpc.py \\
        --checkpoint debug/w1.0.3_8ball_debug.ckpt \\
        --config configs/wmm/lt/w1.0.3_b8all.yaml \\
        --output debug/eval_mpc_run1 \\
        --policy_mode mpc \\
        --num_rollouts 50

Example (2-shard smoke, plain)::

    python eval/langtable_official/eval_protocol_a_wmm_mpc.py \\
        --checkpoint ... --config ... --output ... \\
        --policy_mode plain --num_rollouts 4 --num_shards 2 \\
        --gpus 6,7 --reward_filter blocktoblock
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np
import torch

PROJECT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(PROJECT, "src"))
sys.path.insert(0, osp.join(PROJECT, "ext", "lib_langtable"))
sys.path.insert(0, osp.join(PROJECT, "ext", "lib_io"))
sys.path.insert(0, PROJECT)  # for `demo.langtable.*`


REWARD_NAMES = [
    "blocktoblock",
    "blocktoabsolutelocation",
    "blocktoblockrelativelocation",
    "blocktorelativelocation",
    "separate",
]


def parse_lr_schedule(s: str) -> List[float]:
    """Parse compact lr schedule, e.g. ``"1.0x4,0.5x4,0.1x4"`` -> 12 floats.

    Each chunk is ``<lr>x<count>``. Supports plain ``<lr>`` (count=1).
    """
    out: List[float] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk:
            lr_str, n_str = chunk.split("x", 1)
            out.extend([float(lr_str)] * int(n_str))
        else:
            out.append(float(chunk))
    if not out:
        raise ValueError(f"empty lr_schedule: {s!r}")
    return out


def autodetect_free_gpus(min_free_mem_mib: int = 500, max_util_pct: int = 10) -> List[int]:
    """Return absolute physical GPU IDs that are idle.

    Uses ``nvidia-smi --query-gpu=index,memory.used,utilization.gpu``.
    Honors the parent's ``CUDA_VISIBLE_DEVICES`` if set: filters to
    only those physical IDs. Returns IDs sorted ascending.

    A GPU is "free" iff ``memory.used < min_free_mem_mib`` AND
    ``utilization.gpu < max_util_pct``. Conservative: any background
    process disqualifies the GPU. Override with ``--gpus``.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    free: List[int] = []
    for line in out.strip().splitlines():
        idx_s, mem_s, util_s = [x.strip() for x in line.split(",")]
        idx, mem, util = int(idx_s), int(mem_s), int(util_s)
        if mem < min_free_mem_mib and util < max_util_pct:
            free.append(idx)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd != "":
        try:
            allowed = {int(x) for x in cvd.split(",") if x.strip()}
            free = [g for g in free if g in allowed]
        except ValueError:
            pass  # malformed CVD — ignore the filter
    return sorted(free)


def resolve_gpus(args_gpus: Optional[str]) -> List[int]:
    """Decide which physical GPUs the parent will spawn child shards on."""
    if args_gpus:
        return [int(x) for x in args_gpus.split(",") if x.strip()]
    free = autodetect_free_gpus()
    if not free:
        raise RuntimeError(
            "No free GPUs detected (memory.used < 500 MiB and util < 10%). "
            "Pass --gpus explicitly to override."
        )
    return free


# ────────────────────────────────────────────────────────────────────
# Provenance helpers — copied verbatim from eval_and_save_wmm.py so
# the MPC eval dir is self-contained (ckpt + src_bck + git_info +
# script backup), matching the protocol eval's snapshot convention.
# ────────────────────────────────────────────────────────────────────

def snapshot_src(dst: str, project_root: str) -> float:
    """Mirror training's `src_bck` convention into `<dst>/src_bck/`.

    Copies `src/`, `configs/`, `eval/`, and `train.py`. Excludes large
    data artifacts so the snapshot stays in the tens of MB.
    """
    t0 = time.time()
    src_bck = osp.join(dst, "src_bck")
    os.makedirs(src_bck, exist_ok=True)
    excludes = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=*.npz", "--exclude=*.npy", "--exclude=*.pkl",
        "--exclude=*.p",
        "--exclude=*.pt", "--exclude=*.ckpt", "--exclude=*.bin",
        "--exclude=*.zip", "--exclude=*.tar", "--exclude=*.tar.gz",
        "--exclude=*.gz", "--exclude=*.h5", "--exclude=*.hdf5",
        "--exclude=*.pdf", "--exclude=*.mp4",
        "--exclude=*.png", "--exclude=*.jpg", "--exclude=*.jpeg",
        "--exclude=*.obj", "--exclude=*.ply", "--exclude=*.stl",
        "--exclude=*.dae", "--exclude=*.glb", "--exclude=*.gltf",
        "--exclude=data/",
    ]
    log_local = logging.getLogger("eval_mpc")
    for sub in ("src", "configs", "eval"):
        src_path = osp.join(project_root, sub)
        if not osp.isdir(src_path):
            log_local.warning("snapshot_src: %s not found, skipping", src_path)
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

    Read-only git ops only. Captures every signal needed to identify
    the exact code state behind the eval.
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


# ────────────────────────────────────────────────────────────────────
# Inner shard logic — runs in a child process with one GPU visible
# ────────────────────────────────────────────────────────────────────

def _run_one_episode(
    task: str, ep_num: int, env_seed: int, args, viz, agent, device, log,
):
    """Run plain or MPC for one episode; save raw artifacts."""
    from demo.langtable.mpc_harness import (
        MPCConfig, build_env, _ensure_plan_feasible,
        run_plain_rollout, run_mpc_rollout,
    )

    ep_seed = int(env_seed) + int(ep_num)
    out_root = osp.join(args.output, task, f"ep_{ep_seed}")
    os.makedirs(out_root, exist_ok=True)

    if args.policy_mode == "mpc":
        cfg = MPCConfig(
            K=args.K,
            noise_std=args.noise_std,
            lr_list=parse_lr_schedule(args.lr_schedule),
            n_grad_steps=args.n_grad_steps,
            act_horizon=args.act_horizon,
            window=args.window,
            hist=args.hist,
            max_steps=args.max_steps,
            n_steps=args.n_steps,
            save_intermediates=bool(args.save_intermediates),
        )
    else:
        cfg = MPCConfig(
            act_horizon=args.act_horizon,
            window=args.window,
            hist=args.hist,
            max_steps=args.max_steps,
        )

    env = build_env(task, ep_seed)
    try:
        if not _ensure_plan_feasible(env, base_seed=ep_seed):
            log.warning(f"[{task}/ep_{ep_seed}] no feasible init in 20 retries")
            with open(osp.join(out_root, "meta.json"), "w") as f:
                json.dump({
                    "task": task, "ep_num": ep_num, "ep_seed": ep_seed,
                    "succ": False, "succ_steps": -1,
                    "reason": "no_feasible_init",
                    "policy_mode": args.policy_mode,
                }, f, indent=2)
            return {"ep_num": ep_num, "succ": False, "succ_steps": -1,
                    "reason": "no_feasible_init"}
        t0 = time.time()
        if args.policy_mode == "mpc":
            res = run_mpc_rollout(viz, agent, env, cfg, ep_seed, device)
        else:
            res = run_plain_rollout(viz, agent, env, cfg, ep_seed, device)
    finally:
        env.close()
    elapsed = time.time() - t0

    # Save raw artifacts (no rendering).
    np.savez_compressed(
        osp.join(out_root, "frames.npz"),
        frames=np.stack(res["frames"], axis=0).astype(np.uint8),
    )
    np.save(osp.join(out_root, "real_trace.npy"), res["real_trace"])
    np.save(osp.join(out_root, "imagined_goals.npy"), res["imagined_goals"])
    # local_pcl is needed by render_goal_vs_real_video / render_dream_traces.
    # Tiny (~24 KB at default sizes) — save per episode rather than rely on
    # a shared file so each episode dir is self-contained.
    np.save(osp.join(out_root, "local_pcl.npy"), res["pcl"].cpu().numpy())
    if args.policy_mode == "mpc" and bool(args.save_intermediates):
        from demo.langtable.mpc_harness import save_mpc_intermediates
        save_mpc_intermediates(
            res["intermediates"], osp.join(out_root, "intermediates.npz"),
        )

    meta = {
        "task": task, "ep_num": ep_num, "ep_seed": ep_seed,
        "succ": res["succ"], "succ_steps": res["succ_steps"],
        "instruction": res["instruction"],
        "policy_mode": args.policy_mode,
        "elapsed_sec": elapsed,
        "uid": [int(u) for u in res["uid"]],
    }
    if args.policy_mode == "mpc":
        meta["mpc_config"] = {
            "K": cfg.K, "noise_std": cfg.noise_std,
            "lr_list": cfg.lr_list, "n_grad_steps": cfg.n_grad_steps,
            "act_horizon": cfg.act_horizon, "window": cfg.window,
            "hist": cfg.hist, "max_steps": cfg.max_steps,
            "n_steps": cfg.n_steps,
            "save_intermediates": cfg.save_intermediates,
        }
    with open(osp.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        f"[{task}/ep_{ep_seed}] succ={res['succ']} steps={res['succ_steps']} ({elapsed:.1f}s)"
    )
    return {
        "ep_num": ep_num, "ep_seed": ep_seed,
        "succ": bool(res["succ"]), "succ_steps": int(res["succ_steps"]),
    }


def run_shard(args):
    """Inner shard: this child process runs its slice of episodes."""
    from demo.langtable.mpc_harness import load_agent_and_visualizer

    log = logging.getLogger(f"shard{args._internal_shard_id}")
    log.info(f"shard {args._internal_shard_id}/{args._internal_num_shards} starting on cuda:0 "
             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')})")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in shard process")
    device = torch.device("cuda:0")  # 0 inside the visible-devices mask

    # Resolve hist/window/act_horizon from CLI (with config-derived defaults
    # already applied by the parent before the spawn).
    agent, viz, _pl = load_agent_and_visualizer(args.checkpoint, args.config, device)

    rewards_to_run = (
        [r.strip() for r in args.reward_filter.split(",") if r.strip()]
        if args.reward_filter else REWARD_NAMES
    )
    for r in rewards_to_run:
        if r not in REWARD_NAMES:
            raise ValueError(f"unknown reward {r!r}; expected one of {REWARD_NAMES}")

    sid = args._internal_shard_id
    nshards = args._internal_num_shards
    manifest_dir = osp.join(args.output, "_shards")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = osp.join(manifest_dir, f"shard_{sid:03d}_of_{nshards:03d}.json")

    per_task: dict = {}
    for task in rewards_to_run:
        per_task[task] = []
        for ep_num in range(args.num_rollouts):
            if ep_num % nshards != sid:
                continue
            ep_res = _run_one_episode(
                task, ep_num, args.env_seed, args, viz, agent, device, log,
            )
            per_task[task].append(ep_res)
            # Incremental save in case the shard crashes mid-task.
            with open(manifest_path, "w") as f:
                json.dump({
                    "shard_id": sid, "num_shards": nshards,
                    "per_task": per_task,
                    "args": _safe_vars(args),
                }, f, indent=2)

    log.info(f"shard {sid} done, manifest={manifest_path}")


def _safe_vars(args) -> dict:
    """Return vars(args) minus internal fields, JSON-serializable."""
    d = {k: v for k, v in vars(args).items() if not k.startswith("_internal")}
    return d


# ────────────────────────────────────────────────────────────────────
# Parent: spawn children, wait, merge
# ────────────────────────────────────────────────────────────────────

def merge_manifests(args):
    """Read all per-shard manifests, build final results.json."""
    manifest_dir = osp.join(args.output, "_shards")
    paths = sorted([
        osp.join(manifest_dir, f) for f in os.listdir(manifest_dir)
        if f.startswith("shard_") and f.endswith(".json")
    ])
    if not paths:
        raise RuntimeError(f"no manifests found in {manifest_dir}")

    rewards_to_run = (
        [r.strip() for r in args.reward_filter.split(",") if r.strip()]
        if args.reward_filter else REWARD_NAMES
    )
    merged: dict = {r: {} for r in rewards_to_run}  # task -> {ep_num: ep_res}
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        for task, ep_list in m["per_task"].items():
            for ep in ep_list:
                merged[task][int(ep["ep_num"])] = ep

    all_results = {}
    for task in rewards_to_run:
        eps = sorted(merged.get(task, {}).items())
        succ_count = sum(1 for _, e in eps if e.get("succ"))
        total = len(eps)
        rate = succ_count / total if total else 0.0
        all_results[task] = {
            "reward": task,
            "successes": int(succ_count),
            "total": int(total),
            "rate": float(rate),
            "episodes": [e for _, e in eps],
        }

    avg_rate = float(np.mean([r["rate"] for r in all_results.values()])) if all_results else 0.0
    out = {
        "policy_mode": args.policy_mode,
        "rewards": all_results,
        "avg_rate": avg_rate,
        "args": _safe_vars(args),
    }
    out_path = osp.join(args.output, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # Clean up per-shard manifests once merged successfully.
    shutil.rmtree(manifest_dir, ignore_errors=True)
    return out_path, avg_rate


def main():
    parser = argparse.ArgumentParser(
        description="Protocol A LangTable eval with optional MPC + sharding."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True,
                        help="Must NOT exist (errors out otherwise).")
    parser.add_argument("--policy_mode", choices=["plain", "mpc"], default="plain")
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--reward_filter", type=str, default="",
                        help="Comma-separated reward names; empty = all 5.")

    # Rollout knobs (config-derived in eval_protocol_a_wmm.py; explicit here).
    parser.add_argument("--hist", type=int, default=5)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--act_horizon", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)

    # MPC knobs (defaults match the notebook cfg).
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--lr_schedule", type=str, default="1.0x4,0.5x4,0.1x4")
    parser.add_argument("--n_grad_steps", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=10,
                        help="DDIM denoising steps per inference.")
    parser.add_argument("--save_intermediates", type=int, default=1,
                        help="1=keep per-replan dream traces; 0=skip (saves disk).")

    # Sharding.
    parser.add_argument("--num_shards", type=int, default=0,
                        help="0 = auto (= number of free / specified GPUs).")
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated physical GPU IDs. Empty = auto-detect.")

    # Internal — set by the parent on child invocations.
    parser.add_argument("--_internal_shard_id", type=int, default=-1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_internal_num_shards", type=int, default=-1,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("eval_mpc")

    # Child invocation: just run the shard and return.
    if args._internal_shard_id >= 0:
        run_shard(args)
        return

    # ── Parent ──
    if osp.exists(args.output):
        raise SystemExit(
            f"--output already exists: {args.output}. Refusing to overwrite. "
            f"Pick a fresh path."
        )
    os.makedirs(args.output)

    # Backup-then-load: copy the source ckpt into the dst FIRST. Children
    # are launched with --checkpoint=<dst_ckpt> (local copy) so the output
    # dir is a self-contained snapshot; `args.checkpoint` itself is left
    # holding the original user-supplied path so it serializes intact in
    # rollout_config.json["checkpoint"]. Mirrors eval_and_save_wmm.py.
    src_ckpt = osp.abspath(args.checkpoint)
    dst_ckpt = osp.join(args.output, osp.basename(src_ckpt))
    log.info("Copying ckpt: %s -> %s (size=%.1f MB)",
             src_ckpt, dst_ckpt, osp.getsize(src_ckpt) / (1 << 20))
    t_cp = time.time()
    shutil.copyfile(src_ckpt, dst_ckpt)
    log.info("Ckpt copy done in %.1fs", time.time() - t_cp)

    # Source snapshot + git state — full reproducibility, mirrors protocol.
    log.info("Snapshotting src/configs/eval into %s/src_bck/ ...", args.output)
    snap_sec = snapshot_src(args.output, PROJECT)
    log.info("src snapshot done in %.1fs", snap_sec)
    write_git_info(args.output, PROJECT)
    log.info("Wrote git_info.txt")

    # Verbatim copy of THIS script for re-render / re-eval traceability.
    shutil.copyfile(__file__, osp.join(args.output, "_script_backup.py"))
    log.info("Wrote _script_backup.py")

    gpus = resolve_gpus(args.gpus or None)
    log.info(f"gpus = {gpus}")
    if args.num_shards <= 0:
        args.num_shards = len(gpus)
    if args.num_shards > len(gpus):
        raise SystemExit(
            f"--num_shards={args.num_shards} but only {len(gpus)} GPUs available "
            f"({gpus}). Either lower --num_shards or pass --gpus."
        )
    log.info(f"num_shards = {args.num_shards}")

    # Persist resolved CLI for reproducibility before spawning anything.
    # Record both the original src path and the loaded-from local copy
    # (matches eval_and_save_wmm.py's checkpoint_src / checkpoint_loaded_from).
    with open(osp.join(args.output, "rollout_config.json"), "w") as f:
        json.dump({
            **_safe_vars(args),
            "checkpoint_src": src_ckpt,
            "checkpoint_loaded_from": dst_ckpt,
            "_resolved_gpus": gpus,
            "_resolved_num_shards": args.num_shards,
        }, f, indent=2)

    if args.num_shards == 1:
        # In-process — keep CUDA_VISIBLE_DEVICES from caller's env or pin.
        log.info("num_shards=1 → running in-process (no subprocess fork).")
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        args._internal_shard_id = 0
        args._internal_num_shards = 1
        # Point in-process load at the local copy (parallel to the cmd
        # list below, which passes --checkpoint=dst_ckpt to children).
        args.checkpoint = dst_ckpt
        run_shard(args)
    else:
        # Spawn N children, one per GPU.
        children = []
        log_dir = osp.join(args.output, "_logs")
        os.makedirs(log_dir, exist_ok=True)
        for i in range(args.num_shards):
            gpu = gpus[i]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable, osp.abspath(__file__),
                "--checkpoint", dst_ckpt,
                "--config", args.config,
                "--output", args.output,
                "--policy_mode", args.policy_mode,
                "--num_rollouts", str(args.num_rollouts),
                "--env_seed", str(args.env_seed),
                "--reward_filter", args.reward_filter,
                "--hist", str(args.hist),
                "--window", str(args.window),
                "--act_horizon", str(args.act_horizon),
                "--max_steps", str(args.max_steps),
                "--K", str(args.K),
                "--noise_std", str(args.noise_std),
                "--lr_schedule", args.lr_schedule,
                "--n_grad_steps", str(args.n_grad_steps),
                "--n_steps", str(args.n_steps),
                "--save_intermediates", str(args.save_intermediates),
                "--_internal_shard_id", str(i),
                "--_internal_num_shards", str(args.num_shards),
            ]
            log_path = osp.join(log_dir, f"shard_{i:03d}.log")
            log.info(f"spawning shard {i} on GPU {gpu} (log={log_path})")
            f_log = open(log_path, "w")
            children.append((i, gpu, subprocess.Popen(
                cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT,
            ), f_log))

        # Wait for all to finish.
        rcs = []
        for i, gpu, p, f_log in children:
            rc = p.wait()
            f_log.close()
            log.info(f"shard {i} (GPU {gpu}) exited rc={rc}")
            rcs.append((i, rc))

        bad = [(i, rc) for i, rc in rcs if rc != 0]
        if bad:
            raise SystemExit(
                f"shards failed: {bad}. See {osp.join(args.output, '_logs')}/ "
                f"for stderr."
            )

    # Merge.
    out_path, avg_rate = merge_manifests(args)
    log.info(f"merged → {out_path}, avg_rate={avg_rate:.3f}")
    print(f"\n=== {args.policy_mode.upper()} EVAL DONE ===")
    print(f"  output : {args.output}")
    print(f"  results: {out_path}")
    print(f"  avg    : {avg_rate:.3f}")


if __name__ == "__main__":
    main()
