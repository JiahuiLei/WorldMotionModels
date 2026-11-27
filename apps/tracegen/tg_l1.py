"""Recompute the TraceGen paper table (tables/exp_tg.tex) from released per-sample results.

The metric is the TraceGen authors' ``_compute_trajectory_metrics`` (absolute-action
branch), applied to all test samples that pass the authors' filter. It is written here
as per-sample sums so every sample's contribution can be inspected:

    MSE      = sum_samples sse    / sum_samples n_coord    (x100)
    MAE      = sum_samples sae    / sum_samples n_coord    (x100)
    Endpoint = sum_samples ep_sse / sum_samples ep_coord   (x100)
"""

import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# (table row, env) -> eval run directory under L1_results/runs/
RUNS = {
    ("per-dataset (P=128)", "droid"): "w1.6.0_droid_k128_v9_50k_droid_10way_20260501-150526",
    ("per-dataset (P=128)", "epic"): "w1.6.0_epic_k128_v10_30k_epic_10way_20260501-175931",
    ("per-dataset (P=400)", "droid"): "w1.6.0_droid_k400_v4_80k_droid_10way_20260501-163216",
    ("per-dataset (P=400)", "epic"): "w1.6.0_epic_k400_v9_60k_epic_10way_20260502-155314",
    ("Joint (P=128)", "droid"): "w1.6.0_droidepic_k128_v9_45k_droid_10way_20260501-100224",
    ("Joint (P=128)", "epic"): "w1.6.0_droidepic_k128_v9_45k_epic_10way_20260501-112956",
    ("Joint (P=400)", "droid"): "w1.6.0_droidepic_k400_v10_68k_droid_10way_20260501-123412",
    ("Joint (P=400)", "epic"): "w1.6.0_droidepic_k400_v10_68k_epic_10way_20260501-140123",
}

# Transcribed from the paper's TraceGen table (camera-ready version).
# Columns: Droid MSE, MAE, Endpoint, EpicKitchen MSE, MAE, Endpoint (all x100).
PAPER = {
    "TraceGen (quoted)": [0.206, 1.289, 0.285, 0.445, 2.721, 0.791],
    "per-dataset (P=128)": [0.151, 1.129, 0.200, 0.322, 2.130, 0.556],
    "per-dataset (P=400)": [0.178, 1.169, 0.236, 0.329, 2.167, 0.580],
    "Joint (P=128)": [0.167, 1.172, 0.225, 0.346, 2.256, 0.615],
    "Joint (P=400)": [0.187, 1.179, 0.259, 0.351, 2.287, 0.621],
}
COLUMNS = ["Dr MSE", "Dr MAE", "Dr Endp.", "Ep MSE", "Ep MAE", "Ep Endp."]
QUOTED = {"TraceGen (quoted)": "Lee et al., TraceGen (arXiv 2511.21690), Table 1"}
ENVS = ("droid", "epic")
N_TOTAL = {"droid": 10816, "epic": 7928}
N_KEPT = {"droid": 8421, "epic": 7119}


def l1_root() -> Path:
    """Return release_data/tracegen/L1_results; $WMM_DATA overrides the release_data root."""
    base = Path(os.environ.get("WMM_DATA", REPO_ROOT / "release_data"))
    root = base / "tracegen" / "L1_results"
    if not root.is_dir():
        raise FileNotFoundError("TraceGen L1 results not found at %s" % root)
    return root


def authors_filter(valid_steps: np.ndarray, movement_bool: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """TraceGen authors' test filter: >3 valid frames, some keypoint moves, some keypoint has GT.

    Args:
        valid_steps: (n,) int64.
        movement_bool, gt_mask: (n, 400) bool.

    Returns:
        (n,) bool, True for samples that enter the metric.
    """
    return (valid_steps > 3) & movement_bool.any(axis=1) & gt_mask.any(axis=1)


def per_sample_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """Per-sample error sums.

    Args:
        pred, gt: (n, 33, 400, 3) float32 absolute 3D trajectories (time, keypoint, xyz).
        mask: (n, 33, 400, 3) bool, True where the ground truth is valid.

    Returns:
        dict of (n,) float64 arrays: sse, sae, n_coord, ep_sse, ep_coord.
    """
    m = mask.astype(np.float64)
    p = pred.astype(np.float64) * m
    g = gt.astype(np.float64) * m
    diff = p - g
    sse = (diff ** 2).sum(axis=(1, 2, 3))
    sae = np.abs(diff).sum(axis=(1, 2, 3))
    n_coord = m.sum(axis=(1, 2, 3))
    steps = mask.any(axis=3).sum(axis=1)                       # (n, 400) valid time steps per keypoint
    take = (np.clip(steps, 1, None) - 1)[:, None, :, None]     # (n, 1, 400, 1) last valid step
    ep_diff = np.take_along_axis(diff, take, axis=1)[:, 0]     # (n, 400, 3)
    has = steps > 0                                            # (n, 400)
    ep_sse = ((ep_diff ** 2).sum(axis=2) * has).sum(axis=1)
    ep_coord = 3.0 * has.sum(axis=1)
    return dict(sse=sse, sae=sae, n_coord=n_coord, ep_sse=ep_sse, ep_coord=ep_coord)


def run_per_sample(run: str, env: str, root: Path = None) -> pd.DataFrame:
    """Per-sample error table for one eval run over all test samples; column 'kept' is the filter."""
    root = root or l1_root()
    gt_dir, run_dir = root / "gt" / env, root / "runs" / run
    chunks = sorted(p.name for p in gt_dir.glob("chunk_*.npz"))
    if not chunks or sorted(p.name for p in run_dir.glob("chunk_*.npz")) != chunks:
        raise RuntimeError("%s: prediction chunks do not match ground-truth chunks" % run)
    meta = pd.read_csv(gt_dir / "samples.csv")
    parts = []
    for name in chunks:
        with np.load(gt_dir / name) as g, np.load(run_dir / name) as p:
            if not np.array_equal(g["sample_index"], p["sample_index"]):
                raise RuntimeError("%s/%s: sample order mismatch" % (run, name))
            kept = authors_filter(g["valid_steps"], g["movement_bool"], g["gt_mask"])
            if not np.array_equal(kept, g["kept_by_filter"]):
                raise RuntimeError("%s: recomputed filter differs from stored flag" % name)
            err = per_sample_errors(p["pred_trace_abs"], g["gt_trace_abs"], g["trace_valid_mask_per_coord"])
            parts.append(pd.DataFrame(dict(sample=g["sample_index"], kept=kept, **err)))
    df = pd.concat(parts, ignore_index=True)
    if len(df) != N_TOTAL[env] or not np.array_equal(df["sample"], meta["subset_global_idx"]):
        raise RuntimeError("%s: expected %d samples in order" % (run, N_TOTAL[env]))
    return pd.concat([meta[["episode_id", "text"]], df], axis=1)


def aggregate(df: pd.DataFrame) -> dict:
    """Table metrics (x100) over the samples kept by the authors' filter."""
    k = df[df["kept"]]
    return {
        "MSE": 100 * k["sse"].sum() / k["n_coord"].sum(),
        "MAE": 100 * k["sae"].sum() / k["n_coord"].sum(),
        "Endp.": 100 * k["ep_sse"].sum() / k["ep_coord"].sum(),
        "n_valid": int(len(k)),
    }


def cell_status(value: float, paper: float) -> str:
    """Compare a recomputed value with the paper's 3-decimal number (rounding half up)."""
    three = Decimal("0.001")
    if Decimal(repr(float(value))).quantize(three, ROUND_HALF_UP) == Decimal(repr(paper)):
        return "match"
    if Decimal("%.4f" % float(value)).quantize(three, ROUND_HALF_UP) == Decimal(repr(paper)):
        return "match (paper rounded the 4-decimal value)"
    return "MISMATCH"
