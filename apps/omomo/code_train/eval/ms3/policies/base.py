"""MS3PolicyAdapter abstract base class.

Single source of truth for the contract that every MS3 evaluator adapter must
interface and plug into :class:`eval.ms3.evaluator.MS3Evaluator` — they MUST
NOT re-derive the protocol from bc.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class MS3PolicyAdapter(ABC):
    """Abstract adapter between an inference model and the MS3 evaluator.

    Contract — non-negotiable:

    * ``__init__(ckpt_path, task)`` loads a checkpoint (or ignores it for
      random baselines) and prepares the policy for a specific task id
      (e.g. ``"PushCube-v1"``). Device placement is the adapter's
      responsibility.
    * ``__call__(obs_dict_batch)`` receives a length-``num_envs`` list of
      observation dicts (schema in :pyattr:`obs_dict_spec`) and returns a
      ``numpy.ndarray`` of shape ``(num_envs, action_dim)`` with dtype
      ``float32``. **Numpy is required** for the cpu backend — the evaluator
      hands the array directly to :meth:`gym.vector.VectorEnv.step`, which
      fails on torch tensors at the gym action-space boundary
      (bc.py:341-342).
    * If the adapter was trained with state normalization (bc.py:175-180,
      ``normalize_states=True``), it MUST apply that normalization inside
      ``__call__`` — the evaluator hands raw unnormalized obs (bc_anatomy.md
      §(c); bc.py default ``normalize_states=False``).

    obs_dict schema (every entry is required; missing keys raise ``KeyError``
    when the adapter accesses them):

    ========================  =====================  =======================
    Key                       Shape / dtype          Source
    ========================  =====================  =======================
    ``raw_state``             ``(state_dim,)`` f32   flat obs vector from
                                                    ``CPUGymWrapper``
                                                    (sapien_env.py:535-544)
    ``task_id``               str                    evaluator CLI ``--task``
    ``step``                  int                    step counter since
                                                    last reset
    ``task_action_dim``       int                    env's action_dim for
                                                    this task — 4 for
                                                    pd_ee_delta_pos,
                                                    7 for pd_ee_delta_pose
                                                    (protocol-audit #6)
    ``control_mode``          str                    e.g. ``"pd_ee_delta_pos"``
                                                    or ``"pd_ee_delta_pose"``
    ``robot.qpos``            ``(9,)`` f32           slice of raw_state
                                                    (Panda 7+2 DoF)
    ``robot.qvel``            ``(9,)`` f32           slice of raw_state
    ``robot.ee_pose``         ``(4,4)`` f32 SE(3)    built from tcp_pose
                                                    ``[pos(3), quat_wxyz(4)]``
    ``objects.<name>.pose``   ``(4,4)`` f32 SE(3)    per-task object poses
    ``goal.<sub>``            task-specific          per-task goal fields
    ``scene_params``          dict[str, np.ndarray]  per-episode scene
                                                    geometry (schema v3;
                                                    data-side addition).
                                                    PegIns: ``{peg_half_sizes
                                                    (3,) f32,
                                                    peg_head_offsets
                                                    (3,) f32}``; empty
                                                    ``{}`` for Push/Pick/
                                                    StackCube (no per-env
                                                    rand). Adapters MAY
                                                    ignore this — existing
                                                    adapters remain
                                                    compatible.
    ========================  =====================  =======================

    **Per-task action_dim.** The returned action's last dim MUST equal
    ``obs_dict["task_action_dim"]`` — 4 for pd_ee_delta_pos (PushCube,
    PickCube, StackCube) or 7 for pd_ee_delta_pose (PegInsertionSide,
    per act/baselines.sh:22-26 and diffusion_policy/baselines.sh:27-31).
    Task-agnostic policies read ``task_action_dim`` from the obs dict;
    task-specific ckpts carry the right shape by construction.

    ``raw_state`` is the guaranteed contract — every adapter can rely on it.
    Rich fields (``robot.*``, ``objects.*``, ``goal.*``) are derived by the
    evaluator via a hard-coded per-task unflatten table (see
    :pyfunc:`eval.ms3.evaluator.unflatten_raw_state`). Coord frame: ManiSkill
    world frame; units: meters / radians; quat convention: ``wxyz``.
    """

    #: Human-readable spec of required/optional keys in the obs dict passed
    #: to ``__call__``. Adapters can reference this as documentation.
    obs_dict_spec: str = (
        "raw_state:(state_dim,)f32 | task_id:str | step:int | "
        "task_action_dim:int | control_mode:str | "
        "robot.qpos:(9,)f32 | robot.qvel:(9,)f32 | robot.ee_pose:(4,4)f32 | "
        "objects.<name>.pose:(4,4)f32 (per task) | goal.<sub>:task-specific | "
        "scene_params:dict[str,ndarray] (schema v3; PegIns: peg_half_sizes, "
        "peg_head_offsets; else empty)"
    )

    def __init__(self, ckpt_path: Optional[str], task: str):
        self.ckpt_path = ckpt_path
        self.task = task

    @abstractmethod
    def __call__(self, obs_dict_batch: List[dict]) -> np.ndarray:
        """Map a batch of per-env obs dicts to a numpy action batch.

        Args:
            obs_dict_batch: length-``num_envs`` list of observation dicts.

        Returns:
            ``np.ndarray`` of shape ``(num_envs, action_dim)``, dtype
            ``float32``. The evaluator passes this directly to
            ``env.step(action)``.
        """
        raise NotImplementedError
