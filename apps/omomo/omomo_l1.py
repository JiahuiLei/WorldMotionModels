"""Recompute the paper's OMOMO table from per-window results.

Every number in our row is the mean over the 482 test windows of a per-window value stored in
``metrics_raw.json["per_window"]``, using exactly the aggregation rules of the scorer
(``code/eval/omomo/eval_chois_table1.py``).
"""

import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Optional

COLUMNS = ["T_s", "T_e", "T_xy", "H_feet", "FS", "C_prec", "C_rec", "C_F1", "C_%",
           "MPJPE", "T_root", "T_obj", "O_obj"]

HIGHER_IS_BETTER = {"C_prec", "C_rec", "C_F1", "C_%"}

# Transcribed from the paper's OMOMO table (camera-ready version).
# None = the paper prints "--".
PAPER_TABLE: Dict[str, List[Optional[float]]] = {
    "InterDiff":       [None, 158.84, 72.72, 0.90, 0.42, 0.63, 0.28, 0.33, 0.27, 25.91, 63.44, 88.35, 1.65],
    "MDM":             [5.18, 33.07, 19.42, 6.72, 0.48, 0.72, 0.47, 0.53, 0.43, 17.86, 34.16, 24.46, 1.85],
    "Pred-OMOMO":      [2.39, 8.03, 4.15, 7.08, 0.40, 0.73, 0.66, 0.66, 0.62, 18.66, 28.39, 16.36, 1.05],
    "CHOIS w/o L_obj": [5.76, 14.16, 8.44, 6.55, 0.40, 0.75, 0.50, 0.55, 0.43, 14.34, 21.97, 15.53, 0.98],
    "CHOIS w/o F_all": [1.75, 6.61, 2.69, 6.64, 0.38, 0.78, 0.49, 0.55, 0.41, 15.23, 24.13, 11.51, 0.99],
    "CHOIS":           [1.71, 6.31, 2.87, 4.20, 0.35, 0.80, 0.64, 0.67, 0.54, 15.30, 24.43, 12.53, 0.99],
    "WMM (ours)":      [0.15, 6.48, 4.40, 0.84, 0.56, 0.79, 0.67, 0.76, 0.59, 12.07, 19.87, 16.53, 1.03],
}

# Transcribed from the CHOIS paper (Li et al., ECCV 2024, arXiv 2312.03913),
# sec_eccv/4_experiment.tex, Table 1, restricted to the 13 columns above.
# All baseline rows of the WMM table are quoted from here.
CHOIS_TABLE1: Dict[str, List[float]] = {
    "InterDiff":       [0.00, 158.84, 72.72, 0.90, 0.42, 0.63, 0.28, 0.33, 0.27, 25.91, 63.44, 88.35, 1.65],
    "MDM":             [5.18, 33.07, 19.42, 6.72, 0.48, 0.72, 0.47, 0.53, 0.43, 17.86, 34.16, 24.46, 1.85],
    "Pred-OMOMO":      [2.39, 8.03, 4.15, 7.08, 0.40, 0.73, 0.66, 0.66, 0.62, 18.66, 28.39, 16.36, 1.05],
    "CHOIS w/o L_obj": [5.76, 14.16, 8.44, 6.55, 0.40, 0.75, 0.50, 0.55, 0.43, 14.34, 21.97, 15.53, 0.98],
    "CHOIS w/o F_all": [1.75, 6.61, 2.69, 6.64, 0.38, 0.78, 0.49, 0.55, 0.41, 15.23, 24.13, 11.51, 0.99],
    "CHOIS":           [1.71, 6.31, 2.87, 4.20, 0.35, 0.80, 0.64, 0.67, 0.54, 15.30, 24.43, 12.53, 0.99],
}

OURS_DIR = "ours_wmm_w1.8.1.1_165k_M7nofuzzy"


def l1_dir() -> Path:
    """``$WMM_DATA/omomo/L1_results`` if WMM_DATA is set, else ``<repo>/release_data/omomo/L1_results``."""
    root = os.environ.get("WMM_DATA")
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "release_data"
    path = base / "omomo" / "L1_results"
    if not path.is_dir():
        raise FileNotFoundError(f"OMOMO L1 results not found at {path}; set WMM_DATA to the release_data folder")
    return path


def load_per_window(run_dir: Path) -> List[dict]:
    with open(run_dir / "metrics_raw.json") as f:
        per_window = json.load(f)["per_window"]
    if len(per_window) != 482:
        raise ValueError(f"{run_dir}: expected 482 test windows, found {len(per_window)}")
    return per_window


def _mean(values: List[float]) -> float:
    if not values:
        raise ValueError("no valid values to average")
    return sum(values) / len(values)


def _valid(w: dict, key: str) -> bool:
    v = w.get(key)
    return isinstance(v, (int, float)) and not math.isnan(float(v))


def table_row(per_window: List[dict]) -> Dict[str, float]:
    """Average per-window values into the 13 table columns (lengths in cm)."""
    def avg(key: str, scale: float = 1.0) -> float:
        return _mean([float(w[key]) * scale for w in per_window if _valid(w, key)])

    return {
        "T_s": avg("t_s_cm"),
        "T_e": avg("t_e_cm"),
        "T_xy": avg("t_xy_cm"),
        "H_feet": avg("h_feet_cm"),
        "FS": avg("foot_sliding", 0.1),          # mm -> cm
        "C_prec": avg("cprec"),
        "C_rec": avg("crec"),
        # F1 averaged only over windows whose ground truth has at least one contact frame.
        "C_F1": _mean([float(w["cf1"]) for w in per_window
                       if _valid(w, "cf1") and (w.get("gt_contact_frames") or 0) > 0]),
        # Fraction of frames in which the prediction has hand-object contact.
        "C_%": _mean([w["pred_contact_frames"] / w["n_frames_valid"] for w in per_window
                      if w.get("pred_contact_frames") is not None and w["pred_contact_frames"] >= 0
                      and w.get("n_frames_valid")]),
        "MPJPE": avg("mpjpe_mm", 0.1),           # mm -> cm
        "T_root": avg("troot_mm", 0.1),          # mm -> cm
        "T_obj": avg("t_obj_cm"),
        "O_obj": avg("o_obj"),
    }


def round_half_up(x: float, digits: int) -> Decimal:
    return Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def compare(recomputed: Dict[str, float], paper: List[Optional[float]], digits: int = 2) -> List[dict]:
    """Cell-by-cell comparison of a recomputed row against printed paper values.

    status: "match"   recomputed value rounded to the printed precision equals the paper;
            "match (3-dp rounding)" paper was rounded from a 3-decimal intermediate report
                      (e.g. 6.4746 -> 6.475 -> 6.48);
            "MISMATCH" anything else.
    """
    out = []
    for col, p in zip(COLUMNS, paper):
        x = recomputed[col]
        if p is None:
            out.append(dict(column=col, recomputed=x, paper=None, diff=None, status="not printed"))
            continue
        target = Decimal(repr(p)).quantize(Decimal(1).scaleb(-digits))
        if round_half_up(x, digits) == target:
            status = "match"
        elif round_half_up(float(round_half_up(x, 3)), digits) == target:
            status = "match (3-dp rounding)"
        else:
            status = "MISMATCH"
        out.append(dict(column=col, recomputed=x, paper=p, diff=x - p, status=status))
    return out
