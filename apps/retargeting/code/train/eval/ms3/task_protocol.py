"""Single source of truth for ManiSkill3 benchmark paper protocol.

ALL downstream ms3 code (evaluator, viser, data converter, policy adapters,
round-trip diff, BC / ACT / DP launchers) MUST import constants from this
module. Do NOT duplicate these values elsewhere.

Citations (all from ``ext/maniskill_baselines/``):

* ``examples/baselines/bc/examples.sh``                         — BC canary.
* ``examples/baselines/act/baselines.sh:1-36``                  — ACT state-obs training.
* ``examples/baselines/diffusion_policy/baselines.sh:1-31``     — DP state-obs training.
* ``examples/baselines/<algo>/*.py``                            — per-algo default total_iters.
* ``mani_skill/envs/tasks/tabletop/*.py``                       — env success thresholds.

The paper referenced is Tao et al. 2024 (arxiv:2410.00425) Appendix B Table 2.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Per-task control mode (paper IL 4-task benchmark)
# ---------------------------------------------------------------------------
#
# PushCube / PickCube / StackCube use ``pd_ee_delta_pos`` (3 pos + 1 gripper
# = 4-DOF). PegInsertionSide uses ``pd_ee_delta_pose`` (3 pos + 3 axis-angle
# rotation + 1 gripper = 7-DOF) because the insertion task requires full
# 6-DOF EE control.
TASK_CONTROL_MODE: Dict[str, str] = {
    "PickCube-v1":         "pd_ee_delta_pos",
    "PushCube-v1":         "pd_ee_delta_pos",
    "StackCube-v1":        "pd_ee_delta_pos",
    "PegInsertionSide-v1": "pd_ee_delta_pose",
}

# ---------------------------------------------------------------------------
# Per-task max episode steps (cites act/baselines.sh + dp/baselines.sh)
# ---------------------------------------------------------------------------
#
# PickCube / PushCube = 100 steps (vendor default 50 — examples.sh overrides
# to 100).
# StackCube           = 200 steps (``act/baselines.sh:15``).
# PegInsertionSide    = 300 steps (``act/baselines.sh:22`` + ``dp/baselines.sh:27``).
TASK_MAX_EP_STEPS: Dict[str, int] = {
    "PickCube-v1":         100,
    "PushCube-v1":         100,
    "StackCube-v1":        200,
    "PegInsertionSide-v1": 300,
}

# ---------------------------------------------------------------------------
# Demo source convention (filename fragment under ~/.maniskill/demos/<task>/)
# ---------------------------------------------------------------------------
TASK_DEMO_SOURCE: Dict[str, str] = {
    "PickCube-v1":         "motionplanning",
    "PushCube-v1":         "motionplanning",
    "StackCube-v1":        "motionplanning",
    "PegInsertionSide-v1": "motionplanning",
}

# ---------------------------------------------------------------------------
# Per-algo training iterations (vendor baselines.sh)
# ---------------------------------------------------------------------------
#
# BC  uses 10k universally (bc/examples.sh).
# ACT uses 30k for cubes, 100k for PegInsertionSide (act/baselines.sh).
# DP  uses 30k universally (diffusion_policy/baselines.sh).
ALGO_TOTAL_ITERS: Dict[str, Dict[str, int]] = {
    "BC": {
        "PickCube-v1":         10000,
        "PushCube-v1":         10000,
        "StackCube-v1":        10000,
        "PegInsertionSide-v1": 10000,
    },
    "ACT": {
        "PickCube-v1":         30000,
        "PushCube-v1":         30000,
        "StackCube-v1":        30000,
        "PegInsertionSide-v1": 100000,
    },
    "DP": {
        "PickCube-v1":         30000,
        "PushCube-v1":         30000,
        "StackCube-v1":        30000,
        "PegInsertionSide-v1": 30000,
    },
}

# ---------------------------------------------------------------------------
# Control-mode → action_dim
# ---------------------------------------------------------------------------
CONTROL_MODE_ACTION_DIM: Dict[str, int] = {
    "pd_ee_delta_pos":  4,   # [dx, dy, dz, gripper]
    "pd_ee_delta_pose": 7,   # [dx, dy, dz, drx, dry, drz, gripper] axis-angle
}

#: Tuple of the 4 tasks covered by the paper's IL benchmark (for bulk iteration).
PAPER_SUPPORTED_TASKS: Tuple[str, ...] = tuple(TASK_CONTROL_MODE)


def task_action_dim(task: str) -> int:
    """Return the env action_dim for ``task`` under its canonical control mode.

    Raises:
        KeyError: if ``task`` is not in :data:`TASK_CONTROL_MODE` or the
            resulting control mode is not in :data:`CONTROL_MODE_ACTION_DIM`.
    """
    if task not in TASK_CONTROL_MODE:
        raise KeyError(
            f"task '{task}' is not in TASK_CONTROL_MODE. Supported: "
            f"{sorted(TASK_CONTROL_MODE.keys())}"
        )
    cm = TASK_CONTROL_MODE[task]
    if cm not in CONTROL_MODE_ACTION_DIM:
        raise KeyError(
            f"control mode '{cm}' (task='{task}') has no entry in "
            f"CONTROL_MODE_ACTION_DIM. Supported: "
            f"{sorted(CONTROL_MODE_ACTION_DIM.keys())}"
        )
    return CONTROL_MODE_ACTION_DIM[cm]


def demo_h5_path(task: str, data_root: str = "~/.maniskill/demos") -> str:
    """Return canonical h5 path for the paper-protocol demo of ``task``.

    Args:
        task: MS3 task id (e.g. ``"PushCube-v1"``).
        data_root: Root of the demo tree. Expanded via
            :func:`os.path.expanduser`.

    Returns:
        Absolute path to the ``trajectory.state.<control_mode>.physx_cpu.h5``
        file for this task's canonical paper-protocol demos.
    """
    source = TASK_DEMO_SOURCE[task]
    cm = TASK_CONTROL_MODE[task]
    sim = "physx_cpu"
    return os.path.expanduser(
        f"{data_root}/{task}/{source}/trajectory.state.{cm}.{sim}.h5"
    )
