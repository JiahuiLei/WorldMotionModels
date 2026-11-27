"""CLI entry point for hand-object evaluation.

Usage:
    python -m umm.eval.handobject_new.run_eval \\
        --preds_dir <preds>/ \\
        --gt_dir    <gt>/    \\
        --out       metrics.json

Directory layout expected (one .npz per object):
    preds_dir/<object>.npz  with key 'verts' shape (P, V, 3)
                            optional 'contact' shape (P, V) bool
    gt_dir   /<object>.npz  with key 'verts' shape (G, V, 3)
                            optional 'contact' shape (G, V) bool

For each predicted grasp, the closest GT grasp is selected by v2v_mae
(see eval/handobject_new/baselines/papers/tex_sources/2008.11200/
GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:5-7). The
reported per-object metric is the mean over predicted grasps of the
closest-GT v2v (in cm per paper Table 1) and contact % error.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .closest_gt import closest_gt_by_v2v
from .metrics import contact_error, v2v_mae

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a vertex (+ optional contact) npz.

    Args:
        path: Path to a .npz file with key ``verts`` (required) and optional
            key ``contact``.

    Returns:
        (verts, contact) where verts is (N, V, 3) float and contact is
        (N, V) bool-castable or None.
    """
    with np.load(path) as data:
        if "verts" not in data.files:
            raise KeyError(f"{path}: missing required key 'verts'")
        verts = np.asarray(data["verts"])
        contact = np.asarray(data["contact"]) if "contact" in data.files else None
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"{path}: 'verts' must be (N, V, 3); got {verts.shape}")
    return verts, contact


def _match_objects(preds_dir: Path, gt_dir: Path) -> List[str]:
    """Return sorted list of object stems present in both dirs.

    Args:
        preds_dir: Directory of prediction npz files.
        gt_dir:    Directory of ground-truth npz files.

    Returns:
        Sorted list of object stems (filename without .npz) present in both.
    """
    pred_stems = {p.stem for p in preds_dir.glob("*.npz")}
    gt_stems = {p.stem for p in gt_dir.glob("*.npz")}
    common = sorted(pred_stems & gt_stems)
    missing_in_gt = sorted(pred_stems - gt_stems)
    missing_in_pred = sorted(gt_stems - pred_stems)
    if missing_in_gt:
        logger.warning("Prediction objects with no GT: %s", missing_in_gt)
    if missing_in_pred:
        logger.warning("GT objects with no prediction: %s", missing_in_pred)
    if not common:
        raise RuntimeError(
            f"No overlapping objects between {preds_dir} and {gt_dir}"
        )
    return common


def evaluate_object(
    verts_pred: np.ndarray,  # (P, V, 3) float
    verts_gt: np.ndarray,  # (G, V, 3) float
    contact_pred: Optional[np.ndarray] = None,  # (P, V) bool
    contact_gt: Optional[np.ndarray] = None,  # (G, V) bool
) -> Dict[str, float]:
    """Compute mean closest-GT v2v and contact error over P predictions.

    Args:
        verts_pred: Predicted hand verts (P, V, 3) in metres.
        verts_gt:   GT hand verts (G, V, 3) in metres.
        contact_pred: Optional predicted contact maps (P, V) bool.
        contact_gt:   Optional GT contact maps (G, V) bool.

    Returns:
        dict with ``v2v_mae_cm`` (mean per-pred closest-GT v2v, in cm), and
        optionally ``contact_pct`` (mean per-pred closest-GT contact %).
    """
    per_pred_v2v_m: List[float] = []  # metres, one per pred
    per_pred_contact: List[float] = []  # %, one per pred
    P = verts_pred.shape[0]
    for i in range(P):
        best_idx, best_m = closest_gt_by_v2v(verts_pred[i], verts_gt)
        per_pred_v2v_m.append(best_m)
        if contact_pred is not None and contact_gt is not None:
            per_pred_contact.append(
                contact_error(contact_pred[i], contact_gt[best_idx])
            )
    out: Dict[str, float] = {
        # Paper Table 1 "Vertices mean(cm)" — metres * 100.
        "v2v_mae_cm": float(np.mean(per_pred_v2v_m)) * 100.0,
        "v2v_mae_mm": float(np.mean(per_pred_v2v_m)) * 1000.0,
        "n_preds": int(P),
    }
    if per_pred_contact:
        out["contact_pct"] = float(np.mean(per_pred_contact))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. See module docstring for usage."""
    parser = argparse.ArgumentParser(
        description="Hand-object eval: v2v_mae + contact %% (GrabNet port)"
    )
    parser.add_argument("--preds_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    objects = _match_objects(args.preds_dir, args.gt_dir)
    per_object: Dict[str, Dict[str, float]] = {}
    for obj in objects:
        p_verts, p_contact = _load_npz(args.preds_dir / f"{obj}.npz")
        g_verts, g_contact = _load_npz(args.gt_dir / f"{obj}.npz")
        if p_verts.shape[1:] != g_verts.shape[1:]:
            raise ValueError(
                f"{obj}: vertex shape mismatch pred {p_verts.shape} vs "
                f"gt {g_verts.shape}"
            )
        per_object[obj] = evaluate_object(
            p_verts, g_verts, p_contact, g_contact
        )
        logger.info("%s: %s", obj, per_object[obj])

    # Average across objects (paper Table 1 final "Average" row).
    avg: Dict[str, float] = {}
    for key in ("v2v_mae_cm", "v2v_mae_mm", "contact_pct"):
        vals = [v[key] for v in per_object.values() if key in v]
        if vals:
            avg[key] = float(np.mean(vals))

    result = {"per_object": per_object, "average": avg}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
