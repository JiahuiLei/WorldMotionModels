"""Thin wrapper module re-exporting foot-sliding + floor-height helpers.

Kept to satisfy the module layout. The real implementations live in
:mod:`umm.eval.omomo.metrics`.
"""

from umm.eval.omomo.metrics import (
    compute_foot_sliding_for_smpl as compute_foot_sliding,
)
from umm.eval.omomo.metrics import (
    determine_floor_height_and_contacts as determine_floor_height,
)

__all__ = ["compute_foot_sliding", "determine_floor_height"]
