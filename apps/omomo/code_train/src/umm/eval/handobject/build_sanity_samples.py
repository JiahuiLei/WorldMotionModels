"""Build canonical sanity samples using GTIdentitySampler.

CLI:
    python -m umm.eval.handobject.build_sanity_samples \\
        --dataset {grab,arctic} \\
        --data_root data/handobject/text2hoi_raw/<dataset>/ \\
        --bin_map data/handobject/bin_map_<dataset>.json \\
        --out data/handobject/umm_sanity_samples/<dataset>/ \\
        --split_indices <path_to_split_indices.json from train_classifier run, optional> \\
        --seed 42

Writes:
    <out>/samples/<ep_idx:05d>.npz   -- one per unique text prompt (strict validated)
    <out>/config.json                -- sampler metadata + checksum
    <out>/prompts.json               -- {ep_idx: text} flat mapping
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Set

import numpy as np

from umm.eval.handobject.canonical import (
    CanonicalSample,
    save_canonical_npz,
    validate_canonical_dir,
)
from umm.eval.handobject.gt_identity_sampler import GTIdentitySampler
from umm.eval.handobject.gt_loader import iter_gt_episodes
from umm.eval.handobject.ummsampler import SamplerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_commit_sha() -> str:
    """Return the current git commit SHA (short). Returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("git rev-parse failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return "unknown"
    except Exception as exc:
        logger.warning("Could not determine git SHA: %s", exc)
        return "unknown"


def _checksum_dir(dirpath: str) -> str:
    """SHA-256 of sorted file sizes + names in dirpath. Fast integrity check."""
    h = hashlib.sha256()
    for fname in sorted(os.listdir(dirpath)):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            h.update(f"{fname}:{fsize}\n".encode())
    return h.hexdigest()


def _load_split_indices(path: str) -> Set[int]:
    """Load test-split indices from a split_indices.json file.

    Expects a dict with a "test" key containing a list of ints.

    Args:
        path: Path to split_indices.json.

    Returns:
        Set of int episode indices in the test split.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError:          If "test" key is missing.
        ValueError:        If indices are not ints.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"split_indices: file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if "test" not in data:
        raise KeyError(
            f"split_indices {path!r}: missing 'test' key. Present keys: {list(data.keys())}"
        )
    raw = data["test"]
    if not isinstance(raw, list):
        raise ValueError(
            f"split_indices {path!r}: 'test' must be a list, got {type(raw).__name__}"
        )
    indices: Set[int] = set()
    for v in raw:
        if not isinstance(v, (int, float)):
            raise ValueError(f"split_indices: non-numeric index {v!r}")
        indices.add(int(v))
    logger.info("Loaded %d test-split indices from %s", len(indices), path)
    return indices


def _resolve_data_paths(data_root: str, dataset: str) -> tuple:
    """Resolve and validate the three required data file paths from data_root.

    Expected layout inside data_root:
        data.npz      -- Text2HOI episode data
        text.json     -- paraphrase dict

    Args:
        data_root: Directory containing data.npz and text.json.
        dataset:   "grab" or "arctic" (informational only here).

    Returns:
        (npz_path, text_json_path) as absolute strings.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    npz_path = os.path.join(data_root, "data.npz")
    text_json_path = os.path.join(data_root, "text.json")

    for p in (npz_path, text_json_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"build_sanity_samples: required file not found: {p}"
            )
    return npz_path, text_json_path


def _print_summary_table(
    n_gt_total: int,
    n_gt_kept: int,
    n_unique_prompts: int,
    n_written: int,
    out_dir: str,
    elapsed: float,
) -> None:
    """Print a summary table to stdout."""
    print()
    print("=" * 60)
    print("  build_sanity_samples — summary")
    print("=" * 60)
    print(f"  GT episodes total:        {n_gt_total:>8d}")
    print(f"  GT episodes kept (split): {n_gt_kept:>8d}")
    print(f"  Unique text prompts:      {n_unique_prompts:>8d}")
    print(f"  Samples written:          {n_written:>8d}")
    print(f"  Output directory:         {out_dir}")
    print(f"  Elapsed:                  {elapsed:.1f}s")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the build_sanity_samples CLI."""
    parser = argparse.ArgumentParser(
        description="Build canonical sanity samples using GTIdentitySampler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("grab", "arctic"),
        help='Dataset name: "grab" or "arctic".',
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Directory containing data.npz and text.json.",
    )
    parser.add_argument(
        "--bin_map",
        required=True,
        help="Path to bin_map_<dataset>.json.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root directory. Samples go to <out>/samples/.",
    )
    parser.add_argument(
        "--split_indices",
        default=None,
        help=(
            "Optional path to split_indices.json. If given, only test-split "
            "episodes are used. If omitted, ALL episodes are used (a warning is logged)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for GTIdentitySampler.",
    )
    args = parser.parse_args(argv)

    t0 = time.time()

    # --- Resolve paths ---
    npz_path, text_json_path = _resolve_data_paths(args.data_root, args.dataset)
    bin_map_path = args.bin_map
    if not os.path.exists(bin_map_path):
        raise FileNotFoundError(f"build_sanity_samples: bin_map not found: {bin_map_path}")

    out_dir = args.out
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # --- Load test split indices if given ---
    test_indices: Optional[Set[int]] = None
    if args.split_indices is not None:
        test_indices = _load_split_indices(args.split_indices)
    else:
        warnings.warn(
            "build_sanity_samples: --split_indices not provided. "
            "Using ALL episodes (no train/test filtering). "
            "This may include training data in the sanity samples.",
            UserWarning,
            stacklevel=1,
        )
        logger.warning(
            "No --split_indices given: using ALL episodes without train/test filtering."
        )

    # --- Load and filter GT episodes ---
    logger.info("Loading GT episodes from %s ...", npz_path)

    all_episodes: List[CanonicalSample] = []
    kept_episodes: List[CanonicalSample] = []
    kept_original_indices: List[int] = []

    for ep_idx, ep in enumerate(
        iter_gt_episodes(
            dataset=args.dataset,
            bin_map_path=bin_map_path,
            npz_path=npz_path,
            text_json_path=text_json_path,
        )
    ):
        all_episodes.append(ep)
        if test_indices is None or ep_idx in test_indices:
            kept_episodes.append(ep)
            kept_original_indices.append(ep_idx)

    n_gt_total = len(all_episodes)
    n_gt_kept = len(kept_episodes)
    logger.info(
        "GT episodes: total=%d, kept after split filter=%d", n_gt_total, n_gt_kept
    )

    if n_gt_kept == 0:
        raise ValueError(
            "build_sanity_samples: no episodes remain after split filtering. "
            "Check --split_indices and that the indices match this dataset."
        )

    # --- Build GTIdentitySampler ---
    config = SamplerConfig(dataset=args.dataset, seed=args.seed, n_repeats_per_prompt=1)
    sampler = GTIdentitySampler(config, kept_episodes)

    logger.info(
        "GTIdentitySampler: %d unique prompts across %d episodes",
        len(sampler.unique_prompts),
        sampler.n_episodes,
    )

    # --- Sample and write ---
    # One output file per kept episode (using original episode index as filename).
    # We call sampler.sample(ep.text) for each episode and take the first result
    # (n_repeats_per_prompt=1). The output sample is the GT episode itself.

    prompts_map: Dict[str, str] = {}   # ep_idx_str -> text
    n_written = 0

    for i, (ep, orig_idx) in enumerate(zip(kept_episodes, kept_original_indices)):
        out_fname = f"{orig_idx:05d}.npz"
        out_path = os.path.join(samples_dir, out_fname)

        samples = sampler.sample(ep.text)
        if len(samples) != 1:
            raise ValueError(
                f"GTIdentitySampler returned {len(samples)} samples, expected 1 "
                f"(ep_idx={orig_idx}, text={ep.text!r})"
            )
        sample = samples[0]

        # strict=True: validates obj_rotation orthonormality and det>0 for valid frames.
        save_canonical_npz(sample, out_path, strict=True)
        prompts_map[str(orig_idx)] = ep.text
        n_written += 1

        if (i + 1) % 100 == 0:
            logger.info("Written %d / %d samples ...", i + 1, n_gt_kept)

    logger.info("Wrote %d sample files to %s", n_written, samples_dir)

    # --- Validate the output directory ---
    logger.info("Validating output directory ...")
    dir_report = validate_canonical_dir(samples_dir, strict=True)
    if dir_report["errors"]:
        n_err = len(dir_report["errors"])
        first_err = dir_report["errors"][0]
        raise ValueError(
            f"build_sanity_samples: {n_err} validation error(s) in {samples_dir}. "
            f"First error: {first_err}"
        )
    logger.info(
        "Directory validation passed: n_valid=%d, fps_set=%s, T_stats=%s",
        dir_report["n_samples"],
        dir_report["fps_set"],
        dir_report["T_stats"],
    )

    # --- Write config.json ---
    commit_sha = _get_commit_sha()
    checksum = _checksum_dir(samples_dir)
    cfg_out: Dict[str, Any] = {
        "method": "gt_identity",
        "dataset": args.dataset,
        "seed": args.seed,
        "n_episodes_total": n_gt_total,
        "n_episodes_kept": n_gt_kept,
        "n_samples_written": n_written,
        "n_unique_prompts": len(sampler.unique_prompts),
        "split_indices_path": args.split_indices,
        "data_root": args.data_root,
        "bin_map": args.bin_map,
        "commit_sha": commit_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checksum_samples_dir": checksum,
        "fps_set": dir_report["fps_set"],
        "T_stats": dir_report["T_stats"],
        "obj_counts": dir_report["obj_counts"],
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    logger.info("Wrote config.json to %s", config_path)

    # --- Write prompts.json ---
    prompts_path = os.path.join(out_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts_map, f, indent=2, ensure_ascii=False)
    logger.info("Wrote prompts.json to %s", prompts_path)

    elapsed = time.time() - t0
    _print_summary_table(
        n_gt_total=n_gt_total,
        n_gt_kept=n_gt_kept,
        n_unique_prompts=len(sampler.unique_prompts),
        n_written=n_written,
        out_dir=out_dir,
        elapsed=elapsed,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
