"""MS3 side-by-side comparison — .

Re-evaluates two policy targets (A and B) under an identical shared-evaluator
protocol and reports side-by-side 3-seed statistics.

Protocol notes
--------------
* **3-seed protocol**: single-seed runs show up
  to 12 pp swing on BC ckpts. Three seeds reduce the noise floor to ±~6 pp and
  give a meaningful median. Seeds 0, 1, 2 are the default.
* **PRIMARY metric = ``success_once``** per ``highlevel.md:31``. Both metrics
  are reported; pass/attribution decisions use ``success_once``.
* **SECONDARY metric = ``success_at_end``**: last-step success indicator as
  used by bc.py:354 and the vendor paper. Also reported in the summary table.
* Both targets run through the same subprocess invocation of
  ``python -m eval.ms3.evaluator`` (Mode A CLI) so the env stack, seed
  handling, obs unflatten, and metric aggregation are byte-for-byte identical.

Output files
------------
``<output-dir>/comparison.json``
    Nested: per-target × per-seed × per-episode + cross-seed stats.
``<output-dir>/comparison.csv``
    Flat: one row per (target_label, seed, episode_index).

Known limitations
-----------------
* ``--control-mode`` is NOT forwarded to the evaluator subprocess. Both
  targets run under the evaluator's default ``pd_ee_delta_pos`` control mode
  (the canary). If a future comparison crosses control modes, extend
  ``BC_CMD``-style args in :pyfunc:`run_one` to forward the flag.
* ``--smoke`` sets ``n_episodes=5`` / ``num_envs=2``; the evaluator
  counts episodes in full batches of ``num_envs`` (``eps_count += num_envs``
  at evaluator.py:697), so the actual per-seed episode count is **6**
  (first batch reaches ``eps_count=2 < 5``, second ``4 < 5``, third ``6 ≥ 5``
  → loop exits). Documented; not a bug.

Usage example (smoke test — no real ckpts needed)::

    python eval/ms3/run_comparison.py \\
        --target-a-ckpt NONE \\
        --target-a-adapter eval.ms3.policies.random_policy:RandomPolicy \\
        --target-a-label random_a \\
        --target-b-ckpt NONE \\
        --target-b-adapter eval.ms3.policies.bc_official_adapter:BCOfficialAdapter \\
        --target-b-label bc_random_init \\
        --smoke
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_one(
    ckpt: Optional[str],
    adapter_spec: str,
    task: str,
    n_eps: int,
    num_envs: int,
    max_ep_steps: int,
    seed: int,
    output_dir: str,
) -> dict:
    """Run eval.ms3.evaluator as a subprocess for one (target, seed) pair.

    Spawns ``python -m eval.ms3.evaluator`` with Mode-A CLI arguments. The
    evaluator writes a timestamped JSON into ``output_dir``; this function
    locates the most-recently-written JSON matching
    ``<task>-<AdapterClass>.json`` and parses it.

    Args:
        ckpt: Path to checkpoint file, or ``None`` to omit ``--ckpt`` (used
            by adapters that handle random-init when ckpt_path=None, e.g.
            RandomPolicy and BCOfficialAdapter in smoke mode).
        adapter_spec: ``module:ClassName`` string passed verbatim to
            ``--policy-adapter``.
        task: MS3 task id, e.g. ``"PushCube-v1"``.
        n_eps: Number of episodes.
        num_envs: Number of parallel envs.
        max_ep_steps: Max steps per episode.
        seed: Outer reset seed.
        output_dir: Directory where the evaluator writes its JSON output.
            Also the directory this function scans to find the result file.

    Returns:
        ``{"per_episode": [...], "aggregate": {...}, "seed": seed}``

    Raises:
        RuntimeError: If the subprocess exits non-zero or no output JSON is
            found after the run.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Adapter class tag matches evaluator.py:988 normalization: split on ':'
    adapter_cls_tag = adapter_spec.split(":")[-1]

    # Snapshot files present before the run so we can find the new one.
    pattern = os.path.join(output_dir, f"*-{task}-{adapter_cls_tag}.json")
    before = set(glob.glob(pattern))

    cmd = [
        sys.executable, "-m", "eval.ms3.evaluator",
        "--task", task,
        "--policy-adapter", adapter_spec,
        "--n-episodes", str(n_eps),
        "--num-envs", str(num_envs),
        "--max-episode-steps", str(max_ep_steps),
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if ckpt is not None:
        cmd += ["--ckpt", ckpt]

    logger.info(
        "run_one: seed=%d adapter=%s ckpt=%s",
        seed, adapter_spec, ckpt,
    )
    logger.info("cmd: %s", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print("[run_one] SUBPROCESS STDERR:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"eval.ms3.evaluator exited with code {proc.returncode} "
            f"(seed={seed}, adapter={adapter_spec}, ckpt={ckpt})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    logger.info("run_one: elapsed=%.1fs", elapsed)

    # Find the newly written JSON — the file that appears after 'before'.
    # Fail loud if no new file: a silent fallback to the most-recent file
    # in the dir would return stale data from a previous seed/run and
    # corrupt the comparison (review: "2-seed comparison with one seed
    # counted twice"). Subprocess returned 0 but produced no file ⇒
    # evaluator contract violated ⇒ raise.
    after = set(glob.glob(pattern))
    new_files = sorted(after - before)
    if not new_files:
        raise RuntimeError(
            f"run_one: subprocess exited 0 but wrote no new JSON in "
            f"{output_dir} matching pattern *-{task}-{adapter_cls_tag}.json. "
            f"Evaluator CLI contract broken — refusing to use stale JSON "
            f"from a previous run."
        )

    # Use the most recent new file (there should be exactly one).
    result_path = sorted(new_files, key=os.path.getmtime)[-1]
    logger.info("run_one: parsing result from %s", result_path)

    with open(result_path) as fh:
        payload = json.load(fh)

    result = payload["result"]
    return {
        "per_episode": result["per_episode"],
        "aggregate": result["aggregate"],
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------


def _cross_seed_stats(
    per_seed_results: List[dict],
    metric: str,
) -> Dict[str, float]:
    """Compute mean, std, median of a per-aggregate metric across seeds.

    Args:
        per_seed_results: List of seed-result dicts from ``run_one``.
        metric: Key inside ``result["aggregate"]``, e.g.
            ``"success_once_mean"`` or ``"success_at_end_mean"``.

    Returns:
        Dict with keys ``mean``, ``std``, ``median``.
    """
    vals = [float(r["aggregate"][metric]) for r in per_seed_results]
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals)  # population std (N seeds, not N-1)
    median = statistics.median(vals)
    return {"mean": mean, "std": std, "median": median}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_json(output_dir: str, comparison: dict) -> str:
    """Write comparison.json; returns the path."""
    path = os.path.join(output_dir, "comparison.json")
    with open(path, "w") as fh:
        json.dump(comparison, fh, indent=2)
    return path


def _write_csv(output_dir: str, comparison: dict) -> str:
    """Write comparison.csv — flat, one row per (target, seed, episode).

    Columns: target_label, ckpt, seed, episode_index, env_index,
             success_at_end, success_once, episode_len.

    Args:
        output_dir: Directory to write into.
        comparison: The comparison dict as produced by the main loop.

    Returns:
        Absolute path of the written CSV file.
    """
    path = os.path.join(output_dir, "comparison.csv")
    fieldnames = [
        "target_label",
        "ckpt",
        "seed",
        "episode_index",
        "env_index",
        "success_at_end",
        "success_once",
        "episode_len",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label, tdata in comparison["targets"].items():
            ckpt_val = tdata.get("ckpt") or ""
            for seed_rec in tdata["per_seed"]:
                seed = seed_rec["seed"]
                for ep in seed_rec["per_episode"]:
                    writer.writerow({
                        "target_label": label,
                        "ckpt": ckpt_val,
                        "seed": seed,
                        "episode_index": ep["episode_index"],
                        "env_index": ep["env_index"],
                        "success_at_end": ep["success_at_end"],
                        "success_once": ep["success_once"],
                        "episode_len": ep["episode_len"],
                    })
    return path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(comparison: dict, label_a: str, label_b: str) -> None:
    """Print a human-readable comparison table to stdout."""
    seeds = comparison["seeds"]
    task = comparison["task"]
    n_eps = comparison["n_episodes"]
    num_envs = comparison["num_envs"]

    print()
    print("=== side-by-side comparison ===")
    print(f"task={task}  n_eps={n_eps}  seeds={seeds}  num_envs={num_envs}")
    print()

    col_w = 22
    hdr = (
        f"{'target':<{col_w}}"
        f"{'once (mean+-std)':>18}"
        f"{'once (median)':>15}"
        f"{'at_end (mean+-std)':>20}"
        f"{'at_end (median)':>17}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for label in (label_a, label_b):
        tdata = comparison["targets"][label]
        st = tdata["stats"]
        once_ms = f"{st['success_once_mean']:.3f} +- {st['success_once_std']:.3f}"
        once_med = f"{st['success_once_median']:.3f}"
        sae_ms = f"{st['success_at_end_mean']:.3f} +- {st['success_at_end_std']:.3f}"
        sae_med = f"{st['success_at_end_median']:.3f}"
        print(
            f"{label:<{col_w}}"
            f"{once_ms:>18}"
            f"{once_med:>15}"
            f"{sae_ms:>20}"
            f"{sae_med:>17}"
        )
        rows[label] = st

    print("-" * len(hdr))

    # Deltas (A - B)
    delta_once = rows[label_a]["success_once_mean"] - rows[label_b]["success_once_mean"]
    delta_sae = rows[label_a]["success_at_end_mean"] - rows[label_b]["success_at_end_mean"]
    print(
        f"Delta once (mean) = {delta_once * 100:+.2f} pp  "
        f"[{label_a} - {label_b}]"
    )
    print(
        f"Delta at_end (mean) = {delta_sae * 100:+.2f} pp  "
        f"[{label_a} - {label_b}]"
    )
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Side-by-side comparison: re-eval two policy targets under "
            "identical MS3 protocol (3-seed, primary=success_once per "
            "highlevel.md:31, secondary=success_at_end)."
        )
    )

    # Target A
    p.add_argument(
        "--target-a-ckpt",
        required=True,
        help=(
            "Path to target-A checkpoint file, or 'NONE' to pass ckpt=None "
            "to the adapter (random-init / smoke mode)."
        ),
    )
    p.add_argument(
        "--target-a-adapter",
        required=True,
        metavar="MODULE:CLASS",
        help="Target-A adapter spec, format 'module.path:ClassName'.",
    )
    p.add_argument(
        "--target-a-label",
        default="target_a",
        help="Human-readable label for target A in outputs (default: target_a).",
    )

    # Target B
    p.add_argument(
        "--target-b-ckpt",
        required=True,
        help="Path to target-B checkpoint, or 'NONE'.",
    )
    p.add_argument(
        "--target-b-adapter",
        required=True,
        metavar="MODULE:CLASS",
        help="Target-B adapter spec, format 'module.path:ClassName'.",
    )
    p.add_argument(
        "--target-b-label",
        default="target_b",
        help="Human-readable label for target B (default: target_b).",
    )

    # Protocol
    p.add_argument("--task", default="PushCube-v1")
    p.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Episodes per (target, seed) evaluation (default: 100).",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="Parallel eval envs (default: 10).",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Max steps per episode. Default resolves via "
            "eval.ms3.task_protocol.TASK_MAX_EP_STEPS[task] at run-time "
            "(100 for PushCube/PickCube, 200 for StackCube, 300 for "
            "PegInsertionSide)."
        ),
    )
    p.add_argument(
        "--seeds",
        default="0,1,2",
        help=(
            "Comma-separated seed integers (default: '0,1,2'). "
            "3-seed protocol eliminates single-seed 12pp swing."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="eval/ms3/results",
        help="Directory for comparison.json and comparison.csv (default: eval/ms3/results).",
    )

    # Smoke mode
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Force n-episodes=5, num-envs=2, seeds='0,1,2'. Accepts 'NONE' "
            "ckpts. Useful for plumbing validation without real checkpoints."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Sentinel resolution
# ---------------------------------------------------------------------------


def _resolve_ckpt(raw: str) -> Optional[str]:
    """Convert CLI ckpt string to Python value.

    ``'NONE'`` (case-insensitive) → ``None``.
    Any other string → returned as-is.
    """
    if raw.strip().upper() == "NONE":
        return None
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for side-by-side comparison.

    Returns:
        0 on success, non-zero on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = _build_argparser().parse_args()

    # Smoke overrides
    if args.smoke:
        args.n_episodes = 5
        args.num_envs = 2
        args.seeds = "0,1,2"
        logger.info("--smoke: n_episodes=5  num_envs=2  seeds=0,1,2")

    # Resolve per-task default max_episode_steps from paper-protocol reference.
    if args.max_episode_steps is None:
        from eval.ms3.task_protocol import TASK_MAX_EP_STEPS
        if args.task not in TASK_MAX_EP_STEPS:
            print(
                f"ERROR: --max-episode-steps not given and task "
                f"'{args.task}' is not in TASK_MAX_EP_STEPS. Supported: "
                f"{sorted(TASK_MAX_EP_STEPS.keys())}",
                file=sys.stderr,
            )
            return 1
        args.max_episode_steps = TASK_MAX_EP_STEPS[args.task]
        logger.info(
            "max_episode_steps resolved from TASK_MAX_EP_STEPS[%r] = %d",
            args.task, args.max_episode_steps,
        )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("ERROR: --seeds produced an empty list.", file=sys.stderr)
        return 1

    ckpt_a = _resolve_ckpt(args.target_a_ckpt)
    ckpt_b = _resolve_ckpt(args.target_b_ckpt)

    os.makedirs(args.output_dir, exist_ok=True)

    # Intermediate evaluator output goes into a sub-dir to avoid polluting
    # the comparison output dir with per-run JSON files.
    eval_runs_dir = os.path.join(args.output_dir, "runs")
    os.makedirs(eval_runs_dir, exist_ok=True)

    targets_cfg = [
        (args.target_a_label, ckpt_a, args.target_a_adapter),
        (args.target_b_label, ckpt_b, args.target_b_adapter),
    ]

    # Duplicate-label guard: identical labels would silently overwrite the
    # first target's entry in targets_out. Fail
    # loud at arg-parse time.
    if args.target_a_label == args.target_b_label:
        raise SystemExit(
            f"ERROR: --target-a-label and --target-b-label must differ. "
            f"Both are '{args.target_a_label}' — comparison would collapse "
            f"into a single entry."
        )

    targets_out: Dict[str, dict] = {}

    for label, ckpt, adapter_spec in targets_cfg:
        print(f"\nRunning target '{label}' ({len(seeds)} seeds) ...")
        per_seed_results: List[dict] = []
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_one(
                ckpt=ckpt,
                adapter_spec=adapter_spec,
                task=args.task,
                n_eps=args.n_episodes,
                num_envs=args.num_envs,
                max_ep_steps=args.max_episode_steps,
                seed=seed,
                output_dir=eval_runs_dir,
            )
            elapsed = time.time() - t0
            once = result["aggregate"]["success_once_mean"]
            sae = result["aggregate"]["success_at_end_mean"]
            print(
                f"done ({elapsed:.1f}s)  "
                f"once={once:.3f}  at_end={sae:.3f}"
            )
            per_seed_results.append(result)

        # Cross-seed stats — compute each metric once, then splat into the
        # output dict (review: previous code called _cross_seed_stats
        # 6× per target — copy-paste divergence surface).
        once_stats = _cross_seed_stats(per_seed_results, "success_once_mean")
        at_end_stats = _cross_seed_stats(per_seed_results, "success_at_end_mean")
        stats = {
            "success_once_mean": once_stats["mean"],
            "success_once_std": once_stats["std"],
            "success_once_median": once_stats["median"],
            "success_at_end_mean": at_end_stats["mean"],
            "success_at_end_std": at_end_stats["std"],
            "success_at_end_median": at_end_stats["median"],
        }

        # Use the loop variable `ckpt` rather than an inline
        # label-identity comparison — avoids silent cross-assignment if
        # label strings are ever permuted.
        targets_out[label] = {
            "ckpt": ckpt,
            "adapter": adapter_spec,
            "per_seed": per_seed_results,
            "stats": stats,
        }

    # Build comparison document
    comparison = {
        "task": args.task,
        "n_episodes": args.n_episodes,
        "max_ep_steps": args.max_episode_steps,
        "num_envs": args.num_envs,
        "seeds": seeds,
        "targets": targets_out,
    }

    json_path = _write_json(args.output_dir, comparison)
    csv_path = _write_csv(args.output_dir, comparison)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

    _print_summary(comparison, args.target_a_label, args.target_b_label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
