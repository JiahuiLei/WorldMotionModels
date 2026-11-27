"""Hand-object evaluation metrics (GrabNet port).

Pure-numpy port of GrabNet's v2v_mae and the paper's contact %-error,
used to match Table 1 of the GRAB paper (2008.11200).
"""
from .metrics import (
    Metrics,
    compute_v2v_mae,
    compute_contact_error,
    v2v_mae,
    contact_error,
)
from .closest_gt import closest_gt_by_v2v

__all__ = [
    "Metrics",
    "compute_v2v_mae",
    "compute_contact_error",
    "v2v_mae",
    "contact_error",
    "closest_gt_by_v2v",
]
