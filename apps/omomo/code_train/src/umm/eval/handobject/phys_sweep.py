"""Physical realism threshold sweep CLI.

Two sub-commands:

  dump  -- Loads samples (generated or GT) for one (method, dataset) combo,
           runs compute_physical_realism_raw on each sample, and saves a
           per-sample cache .npz so any threshold grid can be applied later
           without re-running MANO or KNN.

  analyze -- Given a cache .npz and a 4x4 threshold grid, produces a 4x4
             heatmap of per-sample mean physical realism and saves both a .npy
             matrix and a matplotlib .png.

Usage:
  python -m umm.eval.handobject.phys_sweep dump \\
      --dataset {grab,arctic} \\
      --source {samples,gt} \\
      --samples-dir <dir>       # for --source samples \\
      --gt-dir <dir>            # for --source gt (text2hoi_raw/grab style) \\
      --out <cache.npz> \\
      --max-frames 30

  python -m umm.eval.handobject.phys_sweep analyze \\
      --cache <cache.npz> \\
      --out-npy <m.npy> --out-png <m.png> \\
      --label "Text2HOI x GRAB"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold grid for the 4x4 sweep
# ---------------------------------------------------------------------------

_CONTACT_THRESHOLDS = [0.01, 0.02, 0.03, 0.04]    # meters
_PENET_THRESHOLDS   = [0.005, 0.01, 0.015, 0.02]  # meters


# ---------------------------------------------------------------------------
# Sample loading helpers (mirrors eval_imos internal helpers)
# ---------------------------------------------------------------------------

def _load_samples(
    source: str,
    dataset: str,
    samples_dir: Optional[str],
    gt_dir: Optional[str],
) -> "List":
    """Load CanonicalSample list from samples dir or GT dir.

    Args:
        source:      'samples' or 'gt'.
        dataset:     'grab' or 'arctic'.
        samples_dir: Path to generated samples canonical npz dir.
        gt_dir:      Path to text2hoi raw root (for source='gt').

    Returns:
        List of CanonicalSample.

    Raises:
        ValueError: If required path is not provided.
        FileNotFoundError: If the path does not exist.
    """
    from umm.eval.handobject.canonical import load_canonical_npz, validate_canonical_dir
    from umm.eval.handobject.eval_imos import _load_canonical_dir, _load_gt_samples

    if source == "samples":
        if not samples_dir:
            raise ValueError("--samples-dir is required when --source samples")
        if not os.path.isdir(samples_dir):
            raise FileNotFoundError(
                f"phys_sweep dump: --samples-dir not found: {samples_dir!r}"
            )
        return _load_canonical_dir(samples_dir)

    elif source == "gt":
        if not gt_dir:
            raise ValueError("--gt-dir is required when --source gt")
        if not os.path.exists(gt_dir):
            raise FileNotFoundError(
                f"phys_sweep dump: --gt-dir not found: {gt_dir!r}"
            )
        return _load_gt_samples(gt_dir, dataset=dataset)

    else:
        raise ValueError(f"phys_sweep dump: unknown --source {source!r}; expected 'samples' or 'gt'")


# ---------------------------------------------------------------------------
# dump sub-command
# ---------------------------------------------------------------------------

def cmd_dump(args: argparse.Namespace) -> None:
    """Run MANO+KNN on each sample and save raw per-frame cache .npz."""
    from umm.eval.handobject.eval_imos import _collect_obj_pcs, _collect_obj_top_masks
    from umm.eval.handobject.physical_realism import compute_physical_realism_raw

    dataset: str = args.dataset
    source: str = args.source
    max_frames: Optional[int] = args.max_frames if args.max_frames > 0 else None
    out_path: str = args.out

    logger.info(
        "phys_sweep dump: dataset=%s source=%s max_frames=%s -> %s",
        dataset, source, max_frames, out_path,
    )

    # Load samples.
    samples = _load_samples(
        source=source,
        dataset=dataset,
        samples_dir=getattr(args, "samples_dir", None),
        gt_dir=getattr(args, "gt_dir", None),
    )
    if len(samples) == 0:
        raise ValueError("phys_sweep dump: no samples loaded; aborting.")
    logger.info("phys_sweep dump: loaded %d samples", len(samples))

    # Optional deterministic subsample for speed. For threshold selection the
    # distance distribution is stable at ~1k-1500 samples; the generation sets
    # of 8928 (text2hoi_grab) and 3300 (text2hoi_arctic) are overkill here.
    max_samples: int = getattr(args, "max_samples", 0) or 0
    if max_samples > 0 and len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(samples), size=max_samples, replace=False))
        samples = [samples[int(i)] for i in idx]
        logger.info(
            "phys_sweep dump: subsampled to %d samples (seed=42, sorted indices)",
            len(samples),
        )

    # Load obj pcs + top masks.
    obj_pcs = _collect_obj_pcs(dataset=dataset)
    obj_top_masks = _collect_obj_top_masks(dataset=dataset)

    if not obj_pcs:
        raise RuntimeError(
            f"phys_sweep dump: obj_pcs is empty for dataset={dataset!r}. "
            "Check DATA_ROOT and data layout."
        )

    # Per-sample raw computation.
    obj_names_out: List[str] = []
    n_valid_out: List[int] = []
    l_tip_min_dist_chunks: List[np.ndarray] = []
    r_tip_min_dist_chunks: List[np.ndarray] = []
    l_max_penet_chunks: List[np.ndarray] = []
    r_max_penet_chunks: List[np.ndarray] = []
    frame_offsets: List[int] = [0]  # cumulative

    n_skipped = 0
    for idx, s in enumerate(samples):
        if s.obj_name not in obj_pcs:
            logger.warning(
                "phys_sweep dump: sample %d/%d obj_name=%r not in obj_pcs -- skipping",
                idx + 1, len(samples), s.obj_name,
            )
            n_skipped += 1
            continue

        try:
            raw = compute_physical_realism_raw(
                sample=s,
                obj_pcs=obj_pcs,
                dataset=dataset,
                obj_top_masks=obj_top_masks,
                max_frames=max_frames,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning(
                "phys_sweep dump: sample %d/%d obj_name=%r -- compute_physical_realism_raw "
                "raised %s: %s -- skipping",
                idx + 1, len(samples), s.obj_name, type(exc).__name__, exc,
            )
            n_skipped += 1
            continue

        n_valid = raw["n_valid_frames"]
        if n_valid == 0:
            logger.warning(
                "phys_sweep dump: sample %d/%d obj_name=%r has n_valid=0 -- skipping",
                idx + 1, len(samples), s.obj_name,
            )
            n_skipped += 1
            continue

        obj_names_out.append(s.obj_name)
        n_valid_out.append(n_valid)
        l_tip_min_dist_chunks.append(raw["l_tip_min_dist"])
        r_tip_min_dist_chunks.append(raw["r_tip_min_dist"])
        l_max_penet_chunks.append(raw["l_max_penet"])
        r_max_penet_chunks.append(raw["r_max_penet"])
        frame_offsets.append(frame_offsets[-1] + n_valid)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            logger.info(
                "phys_sweep dump: processed %d/%d samples (kept=%d skipped=%d)",
                idx + 1, len(samples), len(obj_names_out), n_skipped,
            )

    if n_skipped > 0:
        logger.warning("phys_sweep dump: total skipped samples: %d", n_skipped)

    if len(obj_names_out) == 0:
        raise RuntimeError(
            "phys_sweep dump: all samples skipped; nothing to save. "
            "Check obj_pcs, valid_mask, and sample loading."
        )

    # Build arrays.
    obj_names_arr = np.array(obj_names_out, dtype="U32")        # (S,)
    n_valid_arr = np.array(n_valid_out, dtype=np.int32)          # (S,)
    l_tip_flat = np.concatenate(l_tip_min_dist_chunks).astype(np.float32)
    r_tip_flat = np.concatenate(r_tip_min_dist_chunks).astype(np.float32)
    l_penet_flat = np.concatenate(l_max_penet_chunks).astype(np.float32)
    r_penet_flat = np.concatenate(r_max_penet_chunks).astype(np.float32)
    offsets_arr = np.array(frame_offsets, dtype=np.int64)        # (S+1,)

    # Save.
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        out_path,
        obj_names=obj_names_arr,
        n_valid_per_sample=n_valid_arr,
        l_tip_min_dist=l_tip_flat,
        r_tip_min_dist=r_tip_flat,
        l_max_penet=l_penet_flat,
        r_max_penet=r_penet_flat,
        sample_frame_offsets=offsets_arr,
    )
    logger.info(
        "phys_sweep dump: saved cache -> %s  (S=%d, total_frames=%d)",
        out_path, len(obj_names_out), int(offsets_arr[-1]),
    )


# ---------------------------------------------------------------------------
# analyze sub-command
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    """Given a cache .npz, sweep a 4x4 threshold grid and produce heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_path: str = args.cache
    out_npy: str = args.out_npy
    out_png: str = args.out_png
    label: str = getattr(args, "label", "")

    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"phys_sweep analyze: cache file not found: {cache_path!r}"
        )

    data = np.load(cache_path, allow_pickle=False)
    obj_names = data["obj_names"]                          # (S,)
    n_valid_per_sample = data["n_valid_per_sample"]        # (S,)
    l_tip_flat = data["l_tip_min_dist"]                    # (total_frames,) float32
    r_tip_flat = data["r_tip_min_dist"]                    # (total_frames,) float32
    l_penet_flat = data["l_max_penet"]                     # (total_frames,) float32
    r_penet_flat = data["r_max_penet"]                     # (total_frames,) float32
    offsets = data["sample_frame_offsets"]                 # (S+1,) int64

    S = int(obj_names.shape[0])

    # Validate cache coherence: offsets[-1] MUST equal the flat-array length.
    # A mismatch indicates cache corruption / mismatched concatenation, which
    # would silently produce wrong per-sample slices downstream.
    if offsets[-1] != l_tip_flat.shape[0]:
        raise ValueError(
            f"phys_sweep analyze: cache corruption -- sample_frame_offsets[-1]="
            f"{int(offsets[-1])} != l_tip_min_dist.shape[0]={l_tip_flat.shape[0]}"
        )
    if not (
        l_tip_flat.shape[0]
        == r_tip_flat.shape[0]
        == l_penet_flat.shape[0]
        == r_penet_flat.shape[0]
    ):
        raise ValueError(
            "phys_sweep analyze: cache corruption -- flat arrays have unequal lengths: "
            f"l_tip={l_tip_flat.shape[0]} r_tip={r_tip_flat.shape[0]} "
            f"l_penet={l_penet_flat.shape[0]} r_penet={r_penet_flat.shape[0]}"
        )
    if offsets.shape[0] != S + 1:
        raise ValueError(
            f"phys_sweep analyze: cache corruption -- offsets length {offsets.shape[0]} "
            f"!= S+1 = {S + 1}"
        )

    logger.info(
        "phys_sweep analyze: loaded cache %s  S=%d total_frames=%d",
        cache_path, S, int(offsets[-1]),
    )

    contact_thresholds = _CONTACT_THRESHOLDS
    penet_thresholds = _PENET_THRESHOLDS
    n_contact = len(contact_thresholds)
    n_penet = len(penet_thresholds)

    # Build 4x4 matrix: rows = contact (ascending), cols = penet (ascending).
    # Entry [i, j] = mean over samples of (fraction of frames that are realistic).
    heatmap = np.zeros((n_contact, n_penet), dtype=np.float64)

    for ci, c_thresh in enumerate(contact_thresholds):
        for pi, p_thresh in enumerate(penet_thresholds):
            sample_realism_values: List[float] = []
            for s_idx in range(S):
                lo = int(offsets[s_idx])
                hi = int(offsets[s_idx + 1])
                if hi <= lo:
                    continue

                l_tip = l_tip_flat[lo:hi]
                r_tip = r_tip_flat[lo:hi]
                l_penet = l_penet_flat[lo:hi]
                r_penet = r_penet_flat[lo:hi]

                # Contact: left OR right fingertip within c_thresh.
                is_contact = (l_tip <= c_thresh) | (r_tip <= c_thresh)

                # Penetration: left OR right max_penet >= p_thresh.
                has_penetration = (l_penet >= p_thresh) | (r_penet >= p_thresh)

                is_realistic = is_contact & ~has_penetration
                sample_realism_values.append(float(is_realistic.mean()))

            if sample_realism_values:
                heatmap[ci, pi] = float(np.mean(sample_realism_values))
            else:
                heatmap[ci, pi] = float("nan")

    logger.info("phys_sweep analyze: heatmap computed:\n%s", str(np.round(heatmap, 4)))

    # Save npy.
    out_npy_dir = os.path.dirname(os.path.abspath(out_npy))
    os.makedirs(out_npy_dir, exist_ok=True)
    np.save(out_npy, heatmap)
    logger.info("phys_sweep analyze: saved npy -> %s", out_npy)

    # Save png heatmap.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean physical realism")

    ax.set_xticks(range(n_penet))
    ax.set_xticklabels([f"{p*1000:.1f}" for p in penet_thresholds])
    ax.set_xlabel("Penetration threshold (mm)")

    ax.set_yticks(range(n_contact))
    ax.set_yticklabels([f"{c*100:.0f}" for c in contact_thresholds])
    ax.set_ylabel("Contact threshold (cm)")

    title = f"Physical realism sweep"
    if label:
        title = f"{label}\n{title}"
    ax.set_title(title)

    # Annotate each cell with the value.
    for ci in range(n_contact):
        for pi in range(n_penet):
            val = heatmap[ci, pi]
            txt = f"{val:.3f}" if not np.isnan(val) else "nan"
            ax.text(
                pi, ci, txt,
                ha="center", va="center",
                color="white" if val < 0.5 else "black",
                fontsize=9,
            )

    fig.tight_layout()
    out_png_dir = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(out_png_dir, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("phys_sweep analyze: saved png -> %s", out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physical realism threshold sweep — dump raw cache or analyze.",
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    # ---- dump ----
    p_dump = sub.add_parser(
        "dump",
        help="Run MANO+KNN and save per-frame raw cache .npz.",
    )
    p_dump.add_argument(
        "--dataset", required=True, choices=("grab", "arctic"),
        help="Dataset name.",
    )
    p_dump.add_argument(
        "--source", required=True, choices=("samples", "gt"),
        help="'samples' = generated samples dir; 'gt' = text2hoi raw root.",
    )
    p_dump.add_argument(
        "--samples-dir", dest="samples_dir", default=None,
        help="Path to canonical npz directory (for --source samples).",
    )
    p_dump.add_argument(
        "--gt-dir", dest="gt_dir", default=None,
        help="Path to text2hoi raw root directory (for --source gt).",
    )
    p_dump.add_argument(
        "--out", required=True,
        help="Output .npz cache path.",
    )
    p_dump.add_argument(
        "--max-frames", dest="max_frames", type=int, default=30,
        help="Max valid frames per sample (uniform subsample). 0=no cap. Default: 30.",
    )
    p_dump.add_argument(
        "--max-samples", dest="max_samples", type=int, default=0,
        help="Max samples to process (deterministic rng seed=42 subsample). 0=no cap.",
    )

    # ---- analyze ----
    p_analyze = sub.add_parser(
        "analyze",
        help="Given a cache .npz, sweep a 4x4 threshold grid and save heatmap.",
    )
    p_analyze.add_argument(
        "--cache", required=True,
        help="Input .npz cache produced by 'dump'.",
    )
    p_analyze.add_argument(
        "--out-npy", dest="out_npy", required=True,
        help="Output .npy path for the 4x4 heatmap matrix.",
    )
    p_analyze.add_argument(
        "--out-png", dest="out_png", required=True,
        help="Output .png path for the heatmap image.",
    )
    p_analyze.add_argument(
        "--label", default="",
        help="Plot title prefix (e.g. 'Text2HOI x GRAB').",
    )

    return parser


def main() -> None:
    """Entry point for python -m umm.eval.handobject.phys_sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcmd == "dump":
        cmd_dump(args)
    elif args.subcmd == "analyze":
        cmd_analyze(args)
    else:
        parser.error(f"Unknown subcommand: {args.subcmd!r}")


if __name__ == "__main__":
    main()
