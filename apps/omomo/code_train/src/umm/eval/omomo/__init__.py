"""OMOMO evaluation metrics port.

Ports OMOMO's ``evaluation_metrics.py`` and its SMPL-X FK dependency into
``umm``. The original baseline uses ``human_body_prior.BodyModel`` for FK;
this port uses the ``smplx`` pip package so it runs in the umm env without
the ``human_body_prior`` dependency.

Unit convention is preserved: distance metrics are returned in **mm**
(the baseline multiplies meters by 1000 internally). Paper Table 2 reports
**cm**, so paper values are ``/10`` of values produced here.

Public API
----------
* :class:`SmplxFK` — SMPL-X forward kinematics for (BS, T, 22, 3) axis-angle.
* :func:`compute_metrics` — full joint/vertex/rotation/trans/foot metrics.
* :func:`compute_collision` — Pbody (object-vs-human SDF penetration).
* :func:`compute_foot_sliding` — alias of ``compute_foot_sliding_for_smpl``.
* :func:`determine_floor_height` — alias of ``determine_floor_height_and_contacts``.
"""

from umm.eval.omomo.smplx_fk import SmplxFK
from umm.eval.omomo.metrics import (
    compute_metrics,
    compute_foot_sliding_for_smpl,
    determine_floor_height_and_contacts,
    get_frobenious_norm_rot_only,
)
from umm.eval.omomo.collision import compute_collision
from umm.eval.omomo.foot_sliding import compute_foot_sliding, determine_floor_height

__all__ = [
    "SmplxFK",
    "compute_metrics",
    "compute_collision",
    "compute_foot_sliding",
    "compute_foot_sliding_for_smpl",
    "determine_floor_height",
    "determine_floor_height_and_contacts",
    "get_frobenious_norm_rot_only",
]
