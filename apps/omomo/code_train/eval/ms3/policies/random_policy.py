"""Random policy for evaluator smoke testing (per-task action_dim aware).

Not a real baseline — produces uniform random actions in ``[-1, 1]`` sized
to match each obs_dict's ``task_action_dim`` field (4 for pd_ee_delta_pos
PushCube/PickCube/StackCube; 7 for pd_ee_delta_pose PegInsertionSide).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from eval.ms3.policies.base import MS3PolicyAdapter


class RandomPolicy(MS3PolicyAdapter):
    """Uniform-random action policy; sizes action to each task's action_dim."""

    def __init__(self, ckpt_path: Optional[str] = None, task: str = ""):
        # ckpt_path intentionally ignored — random policy has no weights.
        super().__init__(ckpt_path=ckpt_path, task=task)
        self._rng = np.random.default_rng(0)

    def __call__(self, obs_dict_batch: List[dict]) -> np.ndarray:
        # Every obs_dict in the batch carries the same task_action_dim (it
        # is task-wide, not env-wide). Read from the first and fail loud if
        # the evaluator failed to inject it.
        if not obs_dict_batch:
            return np.zeros((0, 4), dtype=np.float32)
        first = obs_dict_batch[0]
        if "task_action_dim" not in first:
            raise KeyError(
                "RandomPolicy: obs_dict missing 'task_action_dim' — "
                "evaluator must populate it via build_obs_dict (per "
                "protocol-audit item #6)."
            )
        action_dim = int(first["task_action_dim"])
        num_envs = len(obs_dict_batch)
        action = self._rng.uniform(
            low=-1.0, high=1.0, size=(num_envs, action_dim)
        ).astype(np.float32)
        return action
