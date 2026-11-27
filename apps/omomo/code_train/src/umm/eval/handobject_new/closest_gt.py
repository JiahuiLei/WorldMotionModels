"""Closest-GT grasp selection.

The GRAB paper's Table 1 metrics are computed "vs the closest-GT grasp"
(source: eval/handobject_new/baselines/papers/tex_sources/2008.11200/
GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:5-7). For each
predicted grasp, we pick the GT grasp with the minimum v2v_mae and
report that minimum.
"""
from typing import Tuple

import numpy as np

from .metrics import v2v_mae


def closest_gt_by_v2v(
    verts_pred: np.ndarray, verts_gt_set: np.ndarray
) -> Tuple[int, float]:
    """Pick the GT grasp with minimum v2v error against ``verts_pred``.

    Implements the "vs closest-GT grasp" selection rule from GRAB paper
    Table 1 (2008.11200): for a single predicted hand mesh and a set of
    candidate GT meshes, return the index of the closest GT and its
    v2v_mae value (in metres, see :func:`v2v_mae`).

    Args:
        verts_pred: Predicted hand vertices, shape (V, 3), float.
        verts_gt_set: Candidate GT vertices, shape (N, V, 3) with ``N``
            GT grasps.

    Returns:
        Tuple of (``best_idx``, ``best_v2v_metres``).

    Shape comments:
        verts_pred:   (V, 3) float
        verts_gt_set: (N, V, 3) float, N >= 1
    """
    if verts_gt_set.ndim != 3:
        raise ValueError(
            f"verts_gt_set must be (N, V, 3); got shape {verts_gt_set.shape}"
        )
    if verts_pred.shape != verts_gt_set.shape[1:]:
        raise ValueError(
            f"vertex shape mismatch: pred {verts_pred.shape} vs "
            f"gt[0] {verts_gt_set.shape[1:]}"
        )
    if verts_gt_set.shape[0] == 0:
        raise ValueError("verts_gt_set must contain at least one GT grasp")

    # Per-GT v2v error, reuse v2v_mae() so units (metres) match.
    # (N,) array of mean L1 errors against each GT.
    errs = np.asarray(
        [v2v_mae(verts_pred, verts_gt_set[i]) for i in range(verts_gt_set.shape[0])],
        dtype=np.float64,
    )
    best_idx = int(np.argmin(errs))
    return best_idx, float(errs[best_idx])
