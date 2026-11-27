"""Pure-numpy port of GrabNet evaluation metrics.

Sources
-------
- v2v_mae: ext/baselines/grabnet/grabnet/train/trainer.py:523,528,536.
  The GrabNet formula is
      torch.mean(torch.abs(dorig['verts_rhand'] - verts_hand_cnet) * MESH_SCALER)
  with MESH_SCALER = 1000 (converts metres -> millimetres).
  We return the raw metres value (no scaler) so callers may convert to
  cm (divide by 10) or mm (multiply by 1000) as needed.
- contact_error: element-wise percentage disagreement between two binary
  contact maps. GrabNet's trainer.py does NOT implement this metric; the
  formula is taken directly from the paper's Table 1
  (eval/handobject_new/baselines/papers/tex_sources/2008.11200/
  GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:14-23):
      ``Contact %'' = % of mesh vertices whose contact state (bool)
      disagrees with the closest-GT grasp contact map.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Metrics:
    """Container for a single (pred, gt) evaluation result.

    Attributes:
        v2v_mae_m: Mean per-vertex L1 error in metres. Multiply by 100 for cm,
            or by 1000 for mm (GrabNet's reported unit).
        contact_pct: Percent of vertices with disagreeing binary contact state.
            Range [0, 100]. ``None`` when contact maps are not provided.
        n_vertices: Number of hand-mesh vertices (for sanity checks).
    """

    v2v_mae_m: float
    contact_pct: Optional[float]
    n_vertices: int


def v2v_mae(verts_pred: np.ndarray, verts_gt: np.ndarray) -> float:
    """Mean L1 vertex-to-vertex error in metres.

    Mirrors GrabNet's formula exactly (without the ``* MESH_SCALER`` that
    converts to millimetres). From
    ext/baselines/grabnet/grabnet/train/trainer.py:528
        torch.mean(torch.abs(dorig['verts_rhand'] - verts_hand_cnet) * MESH_SCALER)

    Args:
        verts_pred: Predicted hand vertices, shape (..., V, 3), float.
            Any leading batch dims are allowed; reduction is over the full
            tensor.
        verts_gt: Ground-truth hand vertices, same shape as ``verts_pred``.

    Returns:
        Scalar float: mean of element-wise absolute differences, in the
        native unit of the inputs (metres for MANO).

    Shape comments:
        verts_pred: (..., V, 3) float
        verts_gt:   (..., V, 3) float
    """
    if verts_pred.shape != verts_gt.shape:
        raise ValueError(
            f"shape mismatch: verts_pred {verts_pred.shape} vs "
            f"verts_gt {verts_gt.shape}"
        )
    # From ext/baselines/grabnet/grabnet/train/trainer.py:528
    # torch.mean(torch.abs(dorig['verts_rhand'] - verts_hand_cnet) * MESH_SCALER)
    # We omit the * MESH_SCALER (== 1000 mm/m) so the return value is in metres.
    return float(np.mean(np.abs(verts_pred - verts_gt)))


def contact_error(contact_pred: np.ndarray, contact_gt: np.ndarray) -> float:
    """Percent of vertices with disagreeing binary contact state.

    Formula sourced from the paper Table 1 caption and column header,
    ``Contact %'', at
    eval/handobject_new/baselines/papers/tex_sources/2008.11200/
    GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:14-23.
    GrabNet's released code (ext/baselines/grabnet/) does not implement
    a contact-map metric, so we use the paper's straightforward
    formulation:
        (contact_pred != contact_gt).mean() * 100

    Args:
        contact_pred: Binary contact map, shape (..., V). Any dtype cast-able
            to bool (e.g. float thresholded at 0.5 upstream, or int {0,1}).
        contact_gt: Binary contact map, same shape as ``contact_pred``.

    Returns:
        Scalar float in ``[0, 100]``: percentage of vertices that disagree.

    Shape comments:
        contact_pred: (..., V) bool-castable
        contact_gt:   (..., V) bool-castable
    """
    if contact_pred.shape != contact_gt.shape:
        raise ValueError(
            f"shape mismatch: contact_pred {contact_pred.shape} vs "
            f"contact_gt {contact_gt.shape}"
        )
    # From GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:14-16
    # "Contact %" column = element-wise disagreement % vs closest-GT contact map.
    a = contact_pred.astype(bool)
    b = contact_gt.astype(bool)
    return float(np.mean(a != b)) * 100.0


# Public aliases that match the task's import names.
def compute_v2v_mae(verts_pred: np.ndarray, verts_gt: np.ndarray) -> float:
    """Alias for :func:`v2v_mae`. See that function's docstring."""
    return v2v_mae(verts_pred, verts_gt)


def compute_contact_error(
    contact_pred: np.ndarray, contact_gt: np.ndarray
) -> float:
    """Alias for :func:`contact_error`. See that function's docstring."""
    return contact_error(contact_pred, contact_gt)
