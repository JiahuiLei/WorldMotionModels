"""MS3 shared evaluator — single-owner protocol transcription.

This module is the **single source of truth** for the MS3 imitation-learning
evaluation protocol inside UMM. All downstream SGs (05A BC adapter, 05B ACT,
07A UMM-as-BC, 07B DP, 08 report) consume this module and MUST NOT re-read
``ext/maniskill_baselines/examples/baselines/bc/bc.py`` to derive their own
eval loops.

The transcription is lifted from:

* ``ext/maniskill_baselines/examples/baselines/bc/bc.py``
* ``ext/maniskill_baselines/examples/baselines/bc/behavior_cloning/evaluate.py``
* ``ext/maniskill_baselines/examples/baselines/bc/behavior_cloning/make_env.py``

and from the independent anatomy document


Protocol audit (all 5 items fixed with file:line citations)
-----------------------------------------------------------

1. **Eval seed list.** Two stages:

   a. Per-sub-env ``action_space`` / ``observation_space`` seeding with
      ``range(num_envs)`` inside the thunk factory — ``make_env.py:31-32,37``.
   b. A single outer ``envs.reset(seed=args.seed)`` with ``args.seed``
      (default 1) — ``bc.py:36,301`` and ``evaluate.py:21``.

   Episode-to-episode auto-resets are deterministic from the outer seed but
   not re-seeded per episode (vendor behaviour). We replicate both stages
   exactly.

2. **Episode count.** ``num_eval_episodes=100`` — ``bc.py:73``. The CLI
   default here is 100 and the in-training callback default is 20.

3. **Success metric.**

   * **PRIMARY = ``success_at_end``** (transcription rule) — last-step
     success indicator under ``ignore_terminations=True``
     (``mani_skill/utils/wrappers/gymnasium.py:74-80``). This is the metric
     ``bc.py:354`` saves on best.
   * **SECONDARY = ``success_once``** — cumulative OR of per-step success
     across the episode (``gymnasium.py:63-65``).

   **Run-level canary threshold decision:** which metric is thresholded
   against the ≥55% pass criterion is a run-level choice, NOT an evaluator
   choice. For run ``20260420-0128-86467e35``, the design notes
   pins ``success_once`` as the PRIMARY canary metric. The evaluator
   always reports both and leaves the thresholding decision to the
   consuming step. See the design notes
   §Q1 for the reconciliation.

   The two are NOT the same (bc_anatomy.md §(d)): ``success_once`` captures
   "was the goal satisfied at any step?"; ``success_at_end`` captures "was
   the goal satisfied at the last step of the episode?" — which is the
   published definition.

4. **Env vector parallelism.** ``num_eval_envs=10`` (``bc.py:75``),
   ``CPUGymWrapper(ignore_terminations=True, record_metrics=True)``
   (``make_env.py:28``), ``reconfiguration_freq=1`` on ``gym.make``
   (``make_env.py:25``), ``SyncVectorEnv`` when ``num_envs == 1`` else
   ``AsyncVectorEnv(context="forkserver")`` (``make_env.py:35``). Replicated
   exactly in :pyfunc:`build_eval_envs`.

5. **Obs normalization.** Default ``normalize_states=False``
   (``bc.py:60-61``); the vendor does NOT apply normalization inside
   ``evaluate`` — only the training dataset is normalized in-place
   (``bc.py:175-180``). The evaluator therefore passes **raw** observations
   to the adapter; adapters trained with normalization MUST apply it
   themselves inside ``__call__``. DEVIATION: none (we match vendor).

6. **Per-task control mode** (protocol-audit item #6, added 2026-04-20).
   Not every task uses the same control mode: PushCube / PickCube /
   StackCube use ``pd_ee_delta_pos`` (3 pos + 1 gripper = 4-DOF);
   PegInsertionSide requires ``pd_ee_delta_pose`` (3 pos + 3 axis-angle
   rotation + 1 gripper = 7-DOF). Cites:
   ``ext/maniskill_baselines/examples/baselines/act/baselines.sh:22-26``
   and ``.../diffusion_policy/baselines.sh:27-31``. The evaluator resolves
   the control mode via :pydata:`TASK_CONTROL_MODE` inside
   :pyfunc:`build_eval_envs` and injects the expected action_dim into
   every obs_dict (``obs_dict["task_action_dim"]``) so policies can be
   task-agnostic. ``MS3Evaluator.__init__`` fails loud on an
   ``env.single_action_space`` mismatch at construction time.

sample_fn numpy contract
------------------------

For the cpu backend (``sim_backend == "cpu"``), the adapter action MUST be a
``numpy.ndarray`` of shape ``(num_envs, action_dim)`` with dtype
``float32`` — see ``bc.py:341-342``. Passing a torch tensor to
``env.step(action)`` under ``gym.vector`` (cpu) fails at the action-space
boundary. The :class:`eval.ms3.policies.base.MS3PolicyAdapter` contract
enforces this.

obs_dict contract + per-task unflatten
--------------------------------------

Each sub-env's ``CPUGymWrapper`` returns a flat state vector
(``mani_skill/envs/sapien_env.py:535-544``). To build the rich obs dict
documented on :class:`MS3PolicyAdapter` we slice ``raw_state`` using a
hard-coded per-task offset table derived from each task's
``_get_obs_state_dict`` / ``_get_obs_extra`` (see file citations inside
:pydata:`TASK_SPEC`). This avoids fragile cross-process ``env.call(...)``
round-trips under ``AsyncVectorEnv(forkserver)``.

DEVIATIONS from bc.py
---------------------

* We pass a **list of dicts** (one per env) to the policy instead of a raw
  flat tensor. The flat tensor remains available as
  ``obs_dict["raw_state"]`` — adapters that only need the flat vector
  (e.g. bc's MLP) work unchanged. DEVIATION reason: UMM adapters need
  semantic fields (qpos, tcp_pose, object poses) and the evaluator is the
  correct place to project the flat vector into a structured dict.
* Video recording (2026-04-22). ``MS3Evaluator`` accepts optional
  ``record_video``, ``video_dir``, ``video_fps``. When enabled, the first
  rollout episode's frames are captured via ``envs.call("render")[0]`` and
  written as an mp4 on truncation. ``MS3EvalCallback`` threads the same
  params and logs the mp4 to wandb via ``pl_module.logger.experiment.log``
  keyed by ``eval/ms3/<task>/rollout_mp4``. Default remains
  ``record_video=False`` for the plain protocol CLI (Mode B).
* Per-episode seed recording is **best-effort**: the vector env derives
  per-env seeds internally and they are not exposed. We record the global
  ``args.seed`` plus the env index that produced each episode so the runs
  are traceable, not re-seedable. DEVIATION reason: upstream doesn't expose
  it either.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import wandb
from wandb.errors import Error as WandbError

# NOTE: gymnasium import — make_env.py:2 confirms the upstream uses
# `import gymnasium as gym`. MS3 no longer uses the legacy `gym` package.
import gymnasium as gym

# Side-effect import: registers all MS3 envs (including PushCube-v1). This
# mirrors bc.py:? via `import mani_skill.envs` inside make_env.py:3.
import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.utils.wrappers import CPUGymWrapper

import pytorch_lightning as pl

from prepare_data.ms3.schema import PEG_INS_OBS_PEG_HALF_SIZE_SLICE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task unflatten specification
# ---------------------------------------------------------------------------
#
# flatten_state_dict (mani_skill/utils/common.py:195-244) concatenates the
# dict values in insertion order. `_get_obs_state_dict` returns
# ``dict(agent=self._get_obs_agent(), extra=self._get_obs_extra(info))``
# (sapien_env.py:546-551). For each task below we cite the task-specific
# `_get_obs_extra` (and the agent component which is always qpos, qvel).
#
# Everything is empirically verified against `single_observation_space.shape`
# in bc_anatomy.md §(a).


@dataclass(frozen=True)
class TaskSpec:
    """Static offset table for a canary MS3 task.

    ``fields`` is a list of ``(key, length, kind)`` tuples applied in order
    against ``raw_state``. ``kind`` determines how the slice is rendered in
    the final obs_dict:

    * ``"qpos"``/``"qvel"`` → stashed under ``robot.qpos`` / ``robot.qvel``
    * ``"tcp_pose"`` → 7-vec ``[pos(3), quat_wxyz(4)]`` → converted to
      ``(4,4)`` SE(3) matrix under ``robot.ee_pose``.
    * ``"goal_pos"`` → placed at ``goal.pos`` (shape ``(3,)``).
    * ``"obj_pose"``/``"cubeA_pose"``/``"cubeB_pose"``/``"peg_pose"``/
      ``"box_hole_pose"`` → 7-vec converted to ``(4,4)`` under
      ``objects.<name>.pose``.
    * ``"box_hole_pose_goal"`` → same as above but lives at
      ``goal.box_hole_pose`` (PegInsertionSide only).
    * ``"raw"`` → kept as a flat numpy slice under ``extras.<name>``.
    """

    state_dim: int
    fields: List[Tuple[str, int, str]]


#: PushCube-v1 — push_cube.py:194-207 + agent(qpos/qvel). Total 35.
PUSH_CUBE_SPEC = TaskSpec(
    state_dim=35,
    fields=[
        ("qpos", 9, "qpos"),
        ("qvel", 9, "qvel"),
        ("tcp_pose", 7, "tcp_pose"),
        ("goal_pos", 3, "goal_pos"),
        ("obj_pose", 7, "obj_pose"),
    ],
)

#: PickCube-v1 — pick_cube.py:132-145 + agent. Total 42.
PICK_CUBE_SPEC = TaskSpec(
    state_dim=42,
    fields=[
        ("qpos", 9, "qpos"),
        ("qvel", 9, "qvel"),
        ("is_grasped", 1, "raw"),
        ("tcp_pose", 7, "tcp_pose"),
        ("goal_pos", 3, "goal_pos"),
        ("obj_pose", 7, "obj_pose"),
        ("tcp_to_obj_pos", 3, "raw"),
        ("obj_to_goal_pos", 3, "raw"),
    ],
)

#: StackCube-v1 — stack_cube.py:133-143 + agent. Total 48.
STACK_CUBE_SPEC = TaskSpec(
    state_dim=48,
    fields=[
        ("qpos", 9, "qpos"),
        ("qvel", 9, "qvel"),
        ("tcp_pose", 7, "tcp_pose"),
        ("cubeA_pose", 7, "obj_pose"),
        ("cubeB_pose", 7, "obj_pose"),
        ("tcp_to_cubeA_pos", 3, "raw"),
        ("tcp_to_cubeB_pos", 3, "raw"),
        ("cubeA_to_cubeB_pos", 3, "raw"),
    ],
)

#: PegInsertionSide-v1 — peg_insertion_side.py:289-298 + agent. Total 43.
PEG_INSERTION_SIDE_SPEC = TaskSpec(
    state_dim=43,
    fields=[
        ("qpos", 9, "qpos"),
        ("qvel", 9, "qvel"),
        ("tcp_pose", 7, "tcp_pose"),
        ("peg_pose", 7, "obj_pose"),
        ("peg_half_size", 3, "raw"),
        ("box_hole_pose", 7, "box_hole_pose_goal"),
        ("box_hole_radius", 1, "raw"),
    ],
)


TASK_SPEC: Dict[str, TaskSpec] = {
    "PushCube-v1": PUSH_CUBE_SPEC,
    "PickCube-v1": PICK_CUBE_SPEC,
    "StackCube-v1": STACK_CUBE_SPEC,
    "PegInsertionSide-v1": PEG_INSERTION_SIDE_SPEC,
}


# ---------------------------------------------------------------------------
# Per-task control mode (protocol-audit item #6) — imported from reference
# ---------------------------------------------------------------------------
# The MS3 canonical baselines use different control modes per task. All
# constants live in :mod:`eval.ms3.task_protocol` — see that module for
# citations. This evaluator re-exports the names under the original spelling
# so existing importers keep working.
from eval.ms3.task_protocol import (  # noqa: E402
    CONTROL_MODE_ACTION_DIM,
    TASK_CONTROL_MODE,
    TASK_MAX_EP_STEPS,
    task_action_dim,
)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion ``(4,)`` to a 3x3 rotation matrix.

    ManiSkill uses wxyz (bc_anatomy.md §(c); ``Pose.raw_pose``). Fail loud on
    wrong input shape.
    """
    if q.shape != (4,):
        raise ValueError(f"quat must be shape (4,) wxyz, got {q.shape}")
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def _pose7_to_mat4(pose7: np.ndarray) -> np.ndarray:
    """Convert ``[pos(3), quat_wxyz(4)]`` to a ``(4,4)`` SE(3) matrix."""
    if pose7.shape != (7,):
        raise ValueError(f"pose7 must be shape (7,), got {pose7.shape}")
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_wxyz_to_rotmat(pose7[3:7].astype(np.float32))
    T[:3, 3] = pose7[:3].astype(np.float32)
    return T


def unflatten_raw_state(raw_state: np.ndarray, task: str) -> Dict[str, Any]:
    """Project a flat state vector into a rich per-env obs dict.

    See :pydata:`TASK_SPEC` for the per-task slice table. Raises ``KeyError``
    for unknown tasks and ``ValueError`` on shape mismatch.
    """
    if task not in TASK_SPEC:
        raise KeyError(
            f"unknown MS3 task '{task}' — supported: {sorted(TASK_SPEC.keys())}"
        )
    spec = TASK_SPEC[task]
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if raw.shape[0] != spec.state_dim:
        raise ValueError(
            f"task={task}: expected state_dim={spec.state_dim}, "
            f"got raw_state.shape={raw.shape}"
        )

    out: Dict[str, Any] = {"raw_state": raw}
    objects: Dict[str, Dict[str, np.ndarray]] = {}
    goal: Dict[str, np.ndarray] = {}
    extras: Dict[str, np.ndarray] = {}

    offset = 0
    for key, length, kind in spec.fields:
        slc = raw[offset : offset + length]
        offset += length
        if kind == "qpos":
            out["robot.qpos"] = slc.copy()
        elif kind == "qvel":
            out["robot.qvel"] = slc.copy()
        elif kind == "tcp_pose":
            out["robot.ee_pose"] = _pose7_to_mat4(slc)
        elif kind == "goal_pos":
            goal["pos"] = slc.copy()
        elif kind == "obj_pose":
            # object pose 7-vec → (4,4). Object name is derived from the
            # field name: "obj_pose" → "obj"; "cubeA_pose" → "cubeA";
            # "peg_pose" → "peg"; etc. Strip trailing "_pose" suffix.
            # Fail loud if the key does not end in "_pose" — silent
            # fallback would produce a bogus `objects.<full_key>.pose`
            # entry (review-02 #5 LATENT-RISK).
            if not key.endswith("_pose"):
                raise ValueError(
                    f"task={task}: obj_pose-kind field key '{key}' must end "
                    f"in '_pose' (object-name derivation relies on the "
                    f"suffix). Offending TaskSpec entry: ({key!r}, "
                    f"{length}, 'obj_pose')."
                )
            name = key[: -len("_pose")]
            objects[name] = {"pose": _pose7_to_mat4(slc)}
        elif kind == "box_hole_pose_goal":
            # PegInsertionSide: goal is the hole pose (SE(3))
            goal["box_hole_pose"] = _pose7_to_mat4(slc)
        elif kind == "raw":
            extras[key] = slc.copy()
        else:
            raise ValueError(f"unknown field kind '{kind}' in task spec")

    if offset != spec.state_dim:
        raise ValueError(
            f"task={task}: consumed {offset} of {spec.state_dim} slots"
        )

    # Rich namespaced keys. We also flatten 'objects.<name>.pose' into the
    # top level as a dotted key for easy direct access.
    for name, fields in objects.items():
        for sub, val in fields.items():
            out[f"objects.{name}.{sub}"] = val
    for sub, val in goal.items():
        out[f"goal.{sub}"] = val
    for name, val in extras.items():
        out[f"extras.{name}"] = val
    return out


def _extract_scene_params_from_raw(
    raw_state: np.ndarray, task_id: str,
) -> Dict[str, np.ndarray]:
    """Return per-episode scene_params for this task from a single raw obs.

    Mirrors ``prepare_data.ms3.ms3_replay_handler._extract_scene_params``:
    same source of truth (``peg_half_size`` slice of the vendor-recorded
    obs at [32:35] for PegIns), same derivation for ``peg_head_offsets``.

    Tasks without per-env geometry randomization return ``{}``.
    """
    if task_id != "PegInsertionSide-v1":
        return {}
    # PegIns peg_half_size slice — single source of truth is
    # schema.PEG_INS_OBS_PEG_HALF_SIZE_SLICE (imported at module top).
    # Locks the evaluator's runtime extraction to the same slice the
    # handler uses at conversion time.
    s, e = PEG_INS_OBS_PEG_HALF_SIZE_SLICE
    peg_half_sizes = np.asarray(raw_state[s:e], dtype=np.float32).copy()
    peg_head_offsets = np.array(
        [peg_half_sizes[0], 0.0, 0.0], dtype=np.float32,
    )
    return {
        "peg_half_sizes":   peg_half_sizes,
        "peg_head_offsets": peg_head_offsets,
    }


def build_obs_dict(
    raw_flat_obs_batch: np.ndarray,
    task_id: str,
    step_counters: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build one obs dict per env from a ``(num_envs, state_dim)`` batch.

    Adds ``task_id``, ``step``, ``task_action_dim``, ``control_mode`` and
    ``scene_params`` (schema v3) to every dict (per design contract).
    ``step_counters`` is a ``(num_envs,)`` int array of per-env steps since
    last reset — episode-scoped, NOT a global monotonic counter.

    ``scene_params`` (dict[str, np.ndarray]) — per-episode scene geometry
    params. For PegInsertionSide-v1: ``{peg_half_sizes, peg_head_offsets}``
    extracted from the per-env raw_state. Empty ``{}`` for tasks without
    per-env geometry randomization. Mirrors the per-episode npz contract
    (``prepare_data/ms3/schema.py`` v3 docstring).
    """
    if raw_flat_obs_batch.ndim != 2:
        raise ValueError(
            f"expected (num_envs, state_dim) batch, got shape "
            f"{raw_flat_obs_batch.shape}"
        )
    num_envs = raw_flat_obs_batch.shape[0]
    if step_counters.shape != (num_envs,):
        raise ValueError(
            f"step_counters must have shape ({num_envs},), got "
            f"{step_counters.shape}"
        )
    action_dim = task_action_dim(task_id)
    control_mode = TASK_CONTROL_MODE[task_id]
    out: List[Dict[str, Any]] = []
    for i in range(num_envs):
        raw_i = raw_flat_obs_batch[i]
        d = unflatten_raw_state(raw_i, task_id)
        d["task_id"] = task_id
        d["step"] = int(step_counters[i])
        # Per-task action contract (protocol-audit item #6). Adapters that
        # want to be task-agnostic read these fields instead of hardcoding 4.
        d["task_action_dim"] = action_dim
        d["control_mode"] = control_mode
        # Per-episode scene_params (schema v3). See
        # prepare_data/ms3/schema.py v3 docstring — non-breaking: existing
        # adapters that ignore this key are unaffected.
        d["scene_params"] = _extract_scene_params_from_raw(raw_i, task_id)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Env builder — mirrors behavior_cloning/make_env.py:22-37 exactly.
# ---------------------------------------------------------------------------


#: Canary env kwargs — matches the canary protocol in ``examples.sh`` for
#: PushCube-v1 / PickCube-v1 (the two scripts that ship with the vendor).
#:
#: Env kwargs common to all tasks. ``control_mode`` is INTENTIONALLY absent
#: here — it is resolved per-task via :pydata:`TASK_CONTROL_MODE` inside
#: :pyfunc:`build_eval_envs` (protocol-audit item #6). DEVIATION from
#: ``bc.py:81`` default ``pd_joint_delta_pos``: we follow the canonical
#: per-task protocol from act/diffusion_policy ``baselines.sh``.
DEFAULT_ENV_KWARGS: Dict[str, Any] = {
    "reward_mode": "sparse",
    "obs_mode": "state",
    "render_mode": "rgb_array",
}


def build_eval_envs(
    task: str,
    num_envs: int,
    max_ep_steps: int,
    seed: int,
    env_kwargs_override: Optional[Dict[str, Any]] = None,
) -> gym.vector.VectorEnv:
    """Construct the MS3 cpu eval vector env.

    Replicates the vendor recipe line-for-line:

    * ``gym.make(task, reconfiguration_freq=1, control_mode="pd_ee_delta_pos",
      reward_mode="sparse", obs_mode="state", render_mode="rgb_array",
      max_episode_steps=max_ep_steps)`` — make_env.py:25 + bc.py:251-258
    * Wrapped in ``CPUGymWrapper(ignore_terminations=True,
      record_metrics=True)`` — make_env.py:28
    * Per-sub-env ``action_space.seed(i)`` / ``observation_space.seed(i)``
      for ``i in range(num_envs)`` — make_env.py:31-32,37
    * ``SyncVectorEnv`` if ``num_envs == 1`` else
      ``AsyncVectorEnv(context="forkserver")`` — make_env.py:35

    The outer ``envs.reset(seed=seed)`` is the caller's responsibility
    (MS3Evaluator.run performs it once — bc.py:301).
    """

    env_kwargs: Dict[str, Any] = dict(DEFAULT_ENV_KWARGS)
    # Resolve per-task control mode (protocol-audit item #6) unless the
    # caller explicitly overrides. PushCube/PickCube/StackCube →
    # pd_ee_delta_pos (4-DOF); PegInsertionSide → pd_ee_delta_pose (7-DOF).
    env_kwargs["control_mode"] = TASK_CONTROL_MODE.get(task, "pd_ee_delta_pos")
    if env_kwargs_override:
        env_kwargs.update(env_kwargs_override)
    env_kwargs["max_episode_steps"] = int(max_ep_steps)

    def cpu_make_env(env_seed: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(task, reconfiguration_freq=1, **env_kwargs)
            # NOTE: no user wrappers in — keep the wrapper stack
            # identical to vendor defaults for this run.
            env = CPUGymWrapper(
                env, ignore_terminations=True, record_metrics=True
            )
            env.action_space.seed(env_seed)
            env.observation_space.seed(env_seed)
            return env

        return thunk

    thunks = [cpu_make_env(i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(thunks)
    else:
        envs = gym.vector.AsyncVectorEnv(thunks, context="forkserver")
    return envs


# ---------------------------------------------------------------------------
# MS3Evaluator — transcribes behavior_cloning/evaluate.py:6-39 with
# per-episode record extraction on top.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """Per-episode metrics emitted by :pymeth:`MS3Evaluator.run`.

    ``episode_index`` is a globally-unique 0-based id across the whole
    rollout. ``env_index`` is the slot within a truncation batch
    (``0..num_envs-1``) — useful for tracing back to the per-sub-env space
    seed but NOT unique across batches.
    """

    task_id: str
    episode_index: int
    env_index: int
    success_at_end: float
    success_once: float
    episode_len: int
    seed_used: int  # global seed (per-episode seed not exposed by gym vector)


class MS3Evaluator:
    """Task-agnostic MS3 IL evaluator.

    Usage::

        ev = MS3Evaluator(task="PushCube-v1", num_envs=10,
                          max_ep_steps=100, seed=1)
        result = ev.run(policy, n_episodes=100)
        print(result["aggregate"]["success_at_end_mean"])
        ev.close()
    """

    def __init__(
        self,
        task: str,
        num_envs: int = 10,
        max_ep_steps: int = 100,
        seed: int = 1,
        env_kwargs_override: Optional[Dict[str, Any]] = None,
        record_video: bool = False,
        video_dir: Optional[str] = None,
        video_fps: int = 20,
        sbs_renderer: Optional[Any] = None,
        sbs_video_dir: Optional[str] = None,
    ):
        if task not in TASK_SPEC:
            raise KeyError(
                f"task '{task}' is not in TASK_SPEC; supported: "
                f"{sorted(TASK_SPEC.keys())}"
            )
        if record_video and video_dir is None:
            raise ValueError(
                "record_video=True requires video_dir to be set"
            )
        self.task = task
        self.num_envs = int(num_envs)
        self.max_ep_steps = int(max_ep_steps)
        self.seed = int(seed)
        self.env_kwargs_override = env_kwargs_override
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        self.sbs_renderer = sbs_renderer
        self.sbs_video_dir = sbs_video_dir
        if self.sbs_renderer is not None:
            if self.sbs_video_dir is None:
                raise ValueError(
                    "MS3Evaluator: sbs_renderer given but sbs_video_dir is None"
                )
            os.makedirs(self.sbs_video_dir, exist_ok=True)
        if self.record_video:
            if self.video_dir is None:
                raise ValueError(
                    "MS3Evaluator: record_video=True requires a non-None "
                    "video_dir; got None."
                )
            os.makedirs(self.video_dir, exist_ok=True)
        self.envs = build_eval_envs(
            task=task,
            num_envs=self.num_envs,
            max_ep_steps=self.max_ep_steps,
            seed=self.seed,
            env_kwargs_override=env_kwargs_override,
        )

        # Guard against silent state_dim growth (e.g., a task variant with
        # `controller.use_target=True` appends target-pose fields after qvel,
        # invalidating ALL per-task unflatten offsets in TASK_SPEC). Fail
        # loud at construction time rather than corrupt downstream rich
        # fields. See review-01.md finding #8 (LATENT-RISK).
        #
        # If the guard fires, close the already-spawned vector env before
        # re-raising — otherwise forkserver children leak (review-02 #1).
        try:
            actual_dim = int(self.envs.single_observation_space.shape[0])
            expected_dim = TASK_SPEC[task].state_dim
            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"state_dim mismatch for '{task}': env reports "
                    f"single_observation_space.shape=({actual_dim},), "
                    f"TASK_SPEC expects ({expected_dim},). Did the "
                    f"controller acquire `use_target=True` or did obs_mode "
                    f"change? bc_anatomy.md §(a) fixes the expected "
                    f"obs_dim per task."
                )
            # Per-task action_dim guard (protocol-audit item #6): verify the
            # env's action space matches our TASK_CONTROL_MODE promise.
            actual_action_dim = int(self.envs.single_action_space.shape[0])
            expected_action_dim = task_action_dim(task)
            if actual_action_dim != expected_action_dim:
                raise RuntimeError(
                    f"action_dim mismatch for '{task}': env reports "
                    f"single_action_space.shape=({actual_action_dim},), "
                    f"TASK_CONTROL_MODE[{task!r}]="
                    f"'{TASK_CONTROL_MODE[task]}' expects "
                    f"({expected_action_dim},). Did env_kwargs_override "
                    f"change control_mode?"
                )
        except BaseException:
            try:
                self.envs.close()
            except Exception as close_err:  # pragma: no cover
                logger.warning(
                    "failed to close vector env during failed __init__: %s",
                    close_err,
                )
            raise

    # ..................................................................
    def close(self) -> None:
        try:
            self.envs.close()
        except Exception as e:  # fail loud but don't crash on shutdown
            logger.warning("env close raised: %s", e)

    # ..................................................................
    def _extract_final_info(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize per-env final_info extraction.

        Matches ``evaluate.py:28-34`` exactly: handle both the dict branch
        (batched) and the list branch (per-env).
        """
        fi = info["final_info"]
        if isinstance(fi, dict):
            # GPU/batched path — not the cpu vector path. We still handle it
            # to mirror vendor.
            per_env: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
            ep = fi["episode"]
            for k, v in ep.items():
                arr = v.float().cpu().numpy()  # torch tensor on gpu path
                for i in range(self.num_envs):
                    per_env[i][k] = float(arr[i])
            return per_env
        # cpu vector path — list-of-dicts with per-env final_info.
        # Under ``ignore_terminations=True`` the vendor guarantees all envs
        # truncate together (evaluate.py:26 comment: "no partial resets"),
        # so every entry MUST carry an 'episode' key. Fail loud if not —
        # any None/missing entry means the canary protocol assumption is
        # broken and silently dropping records would miscount ``eps_count``
        # vs the actual per-episode sample size.
        out: List[Dict[str, Any]] = []
        for idx, final_info in enumerate(fi):
            if final_info is None:
                raise RuntimeError(
                    f"cpu vector path: env {idx} returned None final_info at "
                    f"truncation — vendor protocol guarantees all envs "
                    f"truncate together. Did ignore_terminations=True get "
                    f"disabled? (evaluate.py:26)"
                )
            if "episode" not in final_info:
                raise RuntimeError(
                    f"cpu vector path: env {idx} final_info missing "
                    f"'episode' key. Available keys: "
                    f"{sorted(final_info.keys())}"
                )
            ep = final_info["episode"]
            out.append({k: v for k, v in ep.items()})
        return out

    # ..................................................................
    def run(
        self,
        policy: Callable[[List[Dict[str, Any]]], np.ndarray],
        n_episodes: int = 100,
        video_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the vectorized rollout loop and collect per-episode records.

        Exactly replicates ``evaluate.py:21-35`` but:

        * uses ``envs.reset(seed=self.seed)`` per ``bc.py:301`` (the vendor's
          training loop resets once with ``args.seed`` BEFORE calling
          ``evaluate``; ``evaluate.py:21`` then calls another reset
          WITHOUT a seed — we fold both into a single seeded reset here
          because there is no training loop around us. DEVIATION: single
          seeded reset instead of seeded reset + un-seeded reset. Effect:
          identical per-episode sequence because ``ignore_terminations=True``
          means there is no pre-step-1 termination between the two resets).
        * projects the flat obs to an obs_dict via
          :pyfunc:`build_obs_dict`.
        """

        per_episode: List[Dict[str, Any]] = []
        eps_count = 0
        # Video capture state — reset at the top of every run() call.
        first_ep_frames: List[np.ndarray] = []
        first_ep_done: bool = False
        saved_video_path: Optional[str] = None
        first_ep_sbs_records: List[Tuple[np.ndarray, torch.Tensor, int]] = []
        saved_sbs_video_path: Optional[str] = None

        obs_flat, _ = self.envs.reset(seed=self.seed)
        # Per-env step counter: steps since last reset. Episode-scoped —
        # used BOTH as the ``step`` field in obs_dict AND as the recorded
        # ``episode_len``. NOT a global monotonic counter.
        step_counters = np.zeros(self.num_envs, dtype=np.int64)

        # MS3-cpu returns obs as float64 np.ndarray of shape
        # (num_envs, state_dim). Keep a defensive assertion.
        if not isinstance(obs_flat, np.ndarray):
            raise TypeError(
                f"expected np.ndarray obs from vector env, got "
                f"{type(obs_flat).__name__}"
            )

        while eps_count < n_episodes:
            obs_dict_batch = build_obs_dict(
                obs_flat.astype(np.float32, copy=False),
                self.task,
                step_counters,
            )
            action = policy(obs_dict_batch)
            if not isinstance(action, np.ndarray):
                raise TypeError(
                    "MS3 cpu backend requires numpy action; got "
                    f"{type(action).__name__}. Adapters MUST convert "
                    "torch→numpy before returning (bc.py:341-342)."
                )
            if action.dtype != np.float32:
                action = action.astype(np.float32, copy=False)
            obs_flat, _, _, truncated, info = self.envs.step(action)
            step_counters += 1

            # Video capture: record first env's frames for the first episode.
            rendered: Optional[list] = None
            if self.record_video and not first_ep_done:
                rendered = self.envs.call("render")
                if rendered and rendered[0] is not None:
                    first_ep_frames.append(np.asarray(rendered[0]))
            # SBS snapshot reuses `rendered` if available.
            if (
                self.sbs_renderer is not None
                and self.record_video
                and not first_ep_done
                and rendered is not None
                and rendered[0] is not None
            ):
                ctx = getattr(policy, "last_hallucinated_context", None)
                if ctx is not None:
                    first_ep_sbs_records.append(
                        (
                            np.asarray(rendered[0]),
                            ctx.detach().cpu().clone(),
                            int(step_counters[0]),
                        )
                    )

            if truncated.any():
                finals = self._extract_final_info(info)
                # Under ignore_terminations=True all envs truncate together
                # (evaluate.py:26). Mirror the vendor: record every env.
                for i, fi in enumerate(finals):
                    if "success_at_end" not in fi:
                        raise RuntimeError(
                            f"env {i} final_info['episode'] missing "
                            f"'success_at_end'. Keys: {sorted(fi.keys())}. "
                            f"gymnasium.py:78 should have injected it under "
                            f"record_metrics=True."
                        )
                    if "success_once" not in fi:
                        raise RuntimeError(
                            f"env {i} final_info['episode'] missing "
                            f"'success_once'. Keys: {sorted(fi.keys())}. "
                            f"gymnasium.py:64-65 should have injected it "
                            f"under record_metrics=True alongside "
                            f"success_at_end."
                        )
                    rec = EpisodeRecord(
                        task_id=self.task,
                        episode_index=len(per_episode),
                        env_index=i,
                        success_at_end=float(fi["success_at_end"]),
                        success_once=float(fi["success_once"]),
                        episode_len=int(step_counters[i]),
                        seed_used=self.seed,
                    )
                    per_episode.append(rec.__dict__)
                # Vendor counts one full batch of episodes per truncation —
                # evaluate.py:35.
                eps_count += self.num_envs
                # Save first-episode video if recording enabled.
                if self.record_video and not first_ep_done and len(first_ep_frames) > 0:
                    assert self.video_dir is not None, (
                        "record_video=True but video_dir is None — "
                        "__init__ guard should have prevented this."
                    )
                    fname = video_name or "episode_000.mp4"
                    out_path = osp.join(self.video_dir, fname)
                    imageio.mimsave(
                        out_path,
                        [np.asarray(f) for f in first_ep_frames],
                        fps=self.video_fps,
                    )
                    logger.info(f"MS3Evaluator: saved rollout video to {out_path}")
                    saved_video_path = out_path
                    # DELIBERATE: do NOT set first_ep_done here. SBS build
                    # below also gates on `not first_ep_done`. The single
                    # first_ep_done=True assignment lives after the SBS
                    # block so both videos are emitted in the same
                    # truncation batch. Clearing first_ep_frames here is
                    # fine because we own the list and won't re-append.
                    first_ep_frames = []  # free memory
                # SBS mp4 — build from collected records if any exist for
                # env 0. One mp4 per `run()` call (first episode only).
                if (
                    self.sbs_renderer is not None
                    and not first_ep_done
                    and len(first_ep_sbs_records) > 0
                ):
                    assert self.sbs_video_dir is not None, (
                        "sbs_renderer set but sbs_video_dir is None — "
                        "__init__ guard should have prevented this."
                    )
                    # Fail LOUD if the policy lacks the SBS viz helpers —
                    # silent fallback masks a misconfiguration (caller
                    # wired sbs_renderer for a non-WmmAdapter policy).
                    get_pcl = getattr(policy, "get_viz_local_pcl", None)
                    get_uids = getattr(policy, "get_viz_prompt_types", None)
                    if get_pcl is None or get_uids is None:
                        raise AttributeError(
                            f"MS3Evaluator: sbs_renderer is set but policy "
                            f"{type(policy).__name__} lacks get_viz_local_pcl "
                            f"/ get_viz_prompt_types. SBS viz is WmmAdapter-"
                            f"specific; either pass a WmmAdapter policy or "
                            f"leave sbs_renderer=None."
                        )
                    # env 0 success_at_end: index 0 in `finals`.
                    success_at_end = bool(finals[0].get("success_at_end", False))
                    total_steps_sbs = len(first_ep_sbs_records)
                    sbs_fname = (
                        (video_name or "episode_000.mp4")
                        .rsplit(".", 1)[0]
                        + "_sbs.mp4"
                    )
                    sbs_out_path = osp.join(self.sbs_video_dir, sbs_fname)
                    try:
                        viz_pcl = get_pcl()
                        viz_uids = get_uids()
                        sbs_frames: List[np.ndarray] = []
                        for rgb, ctx_pose, _step in first_ep_sbs_records:
                            left = self.sbs_renderer.render_left_frame(
                                ctx_poses=ctx_pose,
                                local_pcl=viz_pcl,
                                prompt_types=viz_uids,
                            )
                            sbs = self.sbs_renderer.stitch_sbs(left, rgb)
                            sbs_frames.append(sbs)
                        self.sbs_renderer.build_mp4(
                            sbs_frames=sbs_frames,
                            out_path=sbs_out_path,
                            success=success_at_end,
                            task=self.task,
                            total_steps=total_steps_sbs,
                        )
                        saved_sbs_video_path = sbs_out_path
                        logger.info(
                            "MS3Evaluator: saved sbs rollout video to %s",
                            sbs_out_path,
                        )
                    except (OSError, RuntimeError, ValueError) as _sbs_e:
                        # Narrow: I/O + runtime render failures are
                        # recoverable (the training run must continue).
                        # Programmer errors (AttributeError, TypeError,
                        # KeyError) already fail loud at the guard above
                        # or propagate from renderer methods.
                        logger.warning(
                            "MS3Evaluator: SBS mp4 build failed: %s: %s",
                            type(_sbs_e).__name__, _sbs_e,
                        )
                    # Free captured records regardless of success/fail so
                    # the next run() call starts clean.
                    first_ep_sbs_records = []
                # Consolidated first-episode-done gate: both normal and SBS
                # video paths have now run. This must be the only place we
                # flip first_ep_done within run().
                if self.record_video and not first_ep_done:
                    first_ep_done = True
                # Reset per-env step counters after a batch truncation. The
                # vector env auto-resets; a new episode starts at step 0.
                step_counters = np.zeros(self.num_envs, dtype=np.int64)

        # Aggregate
        if len(per_episode) == 0:
            raise RuntimeError(
                "no episodes completed; check max_ep_steps and n_episodes"
            )
        sae = np.array([r["success_at_end"] for r in per_episode], dtype=np.float32)
        so = np.array([r["success_once"] for r in per_episode], dtype=np.float32)
        lens = np.array([r["episode_len"] for r in per_episode], dtype=np.float32)
        aggregate = {
            "success_at_end_mean": float(sae.mean()),
            "success_once_mean": float(so.mean()),
            "episode_len_mean": float(lens.mean()),
            "n_episodes": int(len(per_episode)),
            "task": self.task,
            "seed": self.seed,
            "max_ep_steps": self.max_ep_steps,
            "num_envs": self.num_envs,
        }
        result: Dict[str, Any] = {"per_episode": per_episode, "aggregate": aggregate}
        # Include video_path ONLY when a video was saved in this run.
        if saved_video_path is not None:
            result["video_path"] = saved_video_path
        if saved_sbs_video_path is not None:
            result["sbs_video_path"] = saved_sbs_video_path
        return result


# ---------------------------------------------------------------------------
# MS3EvalCallback — Lightning callback variant for in-training eval.
# ---------------------------------------------------------------------------


class MS3EvalCallback(pl.Callback):
    """PyTorch Lightning Callback: run MS3 rollouts every K steps.

    Instantiates a fresh :class:`MS3Evaluator` on first use and runs
    ``n_episodes`` rollouts every ``eval_every_n_steps`` global steps, logs
    mean ``success_at_end`` / ``success_once`` to the Lightning logger, and
    maintains a top-K checkpoint tracker (sorted descending by
    ``success_at_end``).

    .. note::
        This callback is consumed by
        ``src/umm/training_utils/trainer_helper.py:153`` via a new branch in
        ``get_eval_callback`` — that wiring is the job, not this module.
         only guarantees the class imports cleanly from
        ``eval.ms3.evaluator`` and is a proper ``pl.Callback`` subclass.

    **Fork safety (caution).** MS3Evaluator builds an
    ``AsyncVectorEnv(context="forkserver")`` when ``num_envs > 1``. Forking
    from a live Lightning training process that already holds GPU / CUDA
    context can hang or corrupt CUDA state. The recommended callback
    configuration therefore passes ``num_envs=1`` (SyncVectorEnv, single
    process), trading parallelism for safety. If a caller *really* wants
    multi-env eval inside training, set ``num_envs>1`` and verify
    ``multiprocessing.get_start_method()`` is ``'spawn'`` or ``'forkserver'``
    at trainer startup (never ``'fork'``).

    **Checkpoint format (DEVIATION from bc.py).**  bc.py:210-215 saves
    ``{"actor": actor.state_dict()}``. We save a richer wrapper dict::

        {
          "pl_state_dict": pl_module.state_dict(),
          "global_step": int,
          "success_at_end": float,
          "success_once": float,
          "task": str,
        }

    because ``pl_module`` here is a full LightningModule (not the raw BC
    Actor), and callers need the step + SR metadata to rank ckpts.
    the adapter is responsible for translating from ``pl_state_dict``
    into a bc-compatible ``{"actor": ...}`` ckpt if required.
    """

    def __init__(
        self,
        task: str,
        ckpt_dir: str,
        num_envs: int = 10,
        n_episodes: int = 20,
        eval_every_n_steps: int = 5000,
        max_ep_steps: int = 100,
        seed: int = 1,
        policy_factory: Optional[
            Callable[[Any], "MS3PolicyAdapter"]  # noqa: F821
        ] = None,
        top_k: int = 5,
        record_video: bool = False,
        video_dir: Optional[str] = None,
        video_fps: int = 20,
        sbs_video_dir: Optional[str] = None,
    ):
        super().__init__()
        if policy_factory is None:
            raise ValueError(
                "policy_factory is required — returns an MS3PolicyAdapter "
                "given a LightningModule"
            )

        self.task = task
        self.ckpt_dir = ckpt_dir
        self.num_envs = int(num_envs)
        self.n_episodes = int(n_episodes)
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.video_fps = int(video_fps)
        self.max_ep_steps = int(max_ep_steps)
        self.seed = int(seed)
        self.policy_factory = policy_factory
        self.top_k = int(top_k)
        self.record_video = record_video
        self.video_dir = video_dir
        self.sbs_video_dir = sbs_video_dir
        #: list of ``(success_at_end, global_step, ckpt_path)`` — descending.
        self._topk: List[Tuple[float, int, str]] = []
        #: lazily built evaluator to avoid forking an env for unused runs.
        self._evaluator: Optional[MS3Evaluator] = None
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # Lightning calls this on every train batch end.
    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or (step % self.eval_every_n_steps) != 0:
            return
        self._run_once(trainer, pl_module, step)

    def _run_once(self, trainer: Any, pl_module: Any, step: int) -> None:
        if self._evaluator is None:
            sbs_renderer = None
            resolved_sbs_dir = self.sbs_video_dir
            if self.record_video:
                # Lazy import — keeps Ms3StateVisualizer imports off the
                # cold path for non-video callbacks.
                try:
                    from umm.models.wmm.viz_helpers.ms3_sbs_rollout import (
                        Ms3SbsRolloutRenderer,
                    )
                    ms3_state_cfg = (
                        pl_module.cfg.get("model", {}).get("ms3_state", {})
                    )
                    sbs_renderer = Ms3SbsRolloutRenderer(
                        pl_module=pl_module,
                        ms3_state_cfg=ms3_state_cfg,
                        task=self.task,
                        video_fps=self.video_fps,
                    )
                    if resolved_sbs_dir is None and self.video_dir is not None:
                        # Sibling of video_dir by default.
                        resolved_sbs_dir = osp.join(
                            osp.dirname(self.video_dir.rstrip("/")),
                            "ms3_eval_sbs_videos",
                        )
                except (ImportError, RuntimeError, KeyError) as e:
                    logger.warning(
                        "MS3EvalCallback: SBS renderer construction failed: "
                        "%s: %s — falling back to non-SBS rollout.",
                        type(e).__name__, e,
                    )
                    sbs_renderer = None

            self._evaluator = MS3Evaluator(
                task=self.task,
                num_envs=self.num_envs,
                max_ep_steps=self.max_ep_steps,
                seed=self.seed,
                record_video=self.record_video,
                video_dir=self.video_dir,
                video_fps=self.video_fps,
                sbs_renderer=sbs_renderer,
                sbs_video_dir=resolved_sbs_dir,
            )
        policy = self.policy_factory(pl_module)
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                result = self._evaluator.run(
                    policy,
                    n_episodes=self.n_episodes,
                    video_name=f"step{step}_ep000.mp4",
                )
        finally:
            if was_training:
                pl_module.train()

        agg = result["aggregate"]
        sae = float(agg["success_at_end_mean"])
        so = float(agg["success_once_mean"])

        # Log rollout video to wandb if recorded.
        # Use the global `wandb.run` directly (not Lightning's logger-list
        # resolution). Lightning's `pl_module.loggers` iteration was
        # unreliable in production — post-resume the list did not expose a
        # callable `.experiment.log`, so the hotfix (c7c060a7) silently
        # skipped the upload. Going through `wandb.run` is the authoritative
        # path: it's the same singleton the WandbLogger wraps, and it's
        # guaranteed present whenever `wandb.init(...)` has been called
        # upstream (trainer_helper wires it).
        if self.record_video and "video_path" in result:
            mp4_path = result["video_path"]
            wandb_run = getattr(wandb, "run", None)
            if os.path.isfile(mp4_path) and wandb_run is not None:
                # Narrow the exception surface: swallow ONLY I/O /
                # wandb-runtime failures (transient outage, disk I/O,
                # socket error, wandb backend errors). Let programmer
                # errors (TypeError, AttributeError, KeyError, ImportError)
                # propagate — those signal the wandb schema changed or we
                # mis-constructed the log dict.
                # Project rule: "No bare except. No broad except Exception."
                # wandb.errors.Error inherits directly from Exception —
                # must be listed explicitly. review-02 #1.
                try:
                    wandb_run.log({
                        f"eval/ms3/{self.task}/rollout_mp4": wandb.Video(
                            mp4_path, format="mp4"
                        ),
                        "trainer/global_step": step,
                    })
                    logger.info(
                        f"MS3EvalCallback: logged rollout mp4 to wandb: "
                        f"{mp4_path}"
                    )
                except (
                    OSError,
                    RuntimeError,
                    ConnectionError,
                    WandbError,
                ) as e:
                    logger.warning(
                        f"MS3EvalCallback: wandb video log transient "
                        f"failure for {mp4_path}: "
                        f"{type(e).__name__}: {e}"
                    )
        # Log SBS rollout mp4 to wandb.
        if self.record_video and "sbs_video_path" in result:
            sbs_mp4_path = result["sbs_video_path"]
            wandb_run = getattr(wandb, "run", None)
            if os.path.isfile(sbs_mp4_path) and wandb_run is not None:
                try:
                    wandb_run.log({
                        f"eval/ms3/{self.task}/rollout_sbs_mp4": wandb.Video(
                            sbs_mp4_path, format="mp4"
                        ),
                        "trainer/global_step": step,
                    })
                    logger.info(
                        f"MS3EvalCallback: logged SBS rollout mp4 to wandb: "
                        f"{sbs_mp4_path}"
                    )
                except (
                    OSError,
                    RuntimeError,
                    ConnectionError,
                    WandbError,
                ) as e:
                    logger.warning(
                        f"MS3EvalCallback: wandb SBS video log transient "
                        f"failure for {sbs_mp4_path}: "
                        f"{type(e).__name__}: {e}"
                    )
        # Lightning logger
        pl_module.log(
            f"eval/ms3/{self.task}/success_at_end", sae, on_step=True,
            on_epoch=False, prog_bar=False,
        )
        pl_module.log(
            f"eval/ms3/{self.task}/success_once", so, on_step=True,
            on_epoch=False, prog_bar=False,
        )

        # Top-K checkpoint tracker — DEVIATION from bc.py:210-215; see
        # class docstring "Checkpoint format". Skip write when the current
        # eval falls below the top-K threshold (review-01 finding #4 — save
        # disk I/O on long runs after top-K saturates).
        if len(self._topk) >= self.top_k and sae <= self._topk[-1][0]:
            return
        ckpt_path = os.path.join(
            self.ckpt_dir, f"step{step}-sr{sae:.3f}.pt"
        )
        torch.save(
            {
                "pl_state_dict": pl_module.state_dict(),
                "global_step": int(step),
                "success_at_end": sae,
                "success_once": so,
                "task": self.task,
            },
            ckpt_path,
        )
        self._topk.append((sae, step, ckpt_path))
        self._topk.sort(key=lambda t: t[0], reverse=True)
        evicted = self._topk[self.top_k :]
        self._topk = self._topk[: self.top_k]
        for _, _, path in evicted:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("failed to evict ckpt %s: %s", path, e)

    # ..................................................................
    # Teardown hooks — close the forkserver vector env on train end AND on
    # exception (review-01 finding #3 LATENT-RISK: leaked forkserver pool
    # on crash).
    def _close_evaluator(self) -> None:
        if self._evaluator is not None:
            try:
                self._evaluator.close()
            except Exception as e:  # pragma: no cover — best-effort cleanup
                logger.warning("MS3EvalCallback close raised: %s", e)
            finally:
                self._evaluator = None

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._close_evaluator()

    def on_exception(
        self, trainer: Any, pl_module: Any, exception: BaseException
    ) -> None:
        self._close_evaluator()

    def teardown(
        self, trainer: Any, pl_module: Any, stage: Optional[str] = None
    ) -> None:
        self._close_evaluator()


# ---------------------------------------------------------------------------
# CLI — Mode A
# ---------------------------------------------------------------------------


def _load_adapter(adapter_spec: str) -> type:
    """Resolve ``module:ClassName`` to a class object."""
    if ":" not in adapter_spec:
        raise ValueError(
            f"--policy-adapter must be 'module:ClassName', got "
            f"'{adapter_spec}'"
        )
    mod_name, cls_name = adapter_spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, cls_name):
        raise AttributeError(
            f"module '{mod_name}' has no attribute '{cls_name}'"
        )
    cls = getattr(mod, cls_name)
    return cls


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS3 shared IL evaluator.")
    p.add_argument("--task", required=True, choices=sorted(TASK_SPEC.keys()))
    p.add_argument(
        "--policy-adapter",
        required=True,
        help="adapter spec, format 'module.path:ClassName'",
    )
    p.add_argument("--ckpt", default=None, help="checkpoint path (optional)")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "max steps per episode. Default resolves to "
            "TASK_MAX_EP_STEPS[task] from eval.ms3.task_protocol "
            "(100 for PushCube/PickCube, 200 for StackCube, 300 for "
            "PegInsertionSide). Override only if you have a reason."
        ),
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output-dir",
        default="eval/ms3/runs",
        help="directory to write JSON results into",
    )
    p.add_argument(
        "--control-mode",
        default=None,
        help=(
            "override env control_mode (default 'pd_ee_delta_pos' = "
            "action_dim 4, canary). Use 'pd_joint_delta_pos' (action_dim 8) "
            "for bc.py default; adapter action_dim must match."
        ),
    )
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_argparser().parse_args()

    adapter_cls = _load_adapter(args.policy_adapter)
    policy = adapter_cls(ckpt_path=args.ckpt, task=args.task)

    # Resolve max_episode_steps default from paper-protocol reference if the
    # caller didn't override it on the CLI.
    max_ep_steps = args.max_episode_steps
    if max_ep_steps is None:
        if args.task not in TASK_MAX_EP_STEPS:
            raise KeyError(
                f"--max-episode-steps not given and task '{args.task}' is "
                f"not in TASK_MAX_EP_STEPS. Supported: "
                f"{sorted(TASK_MAX_EP_STEPS.keys())}"
            )
        max_ep_steps = TASK_MAX_EP_STEPS[args.task]

    env_kwargs_override: Optional[Dict[str, Any]] = None
    if args.control_mode is not None:
        env_kwargs_override = {"control_mode": args.control_mode}
    evaluator = MS3Evaluator(
        task=args.task,
        num_envs=args.num_envs,
        max_ep_steps=max_ep_steps,
        seed=args.seed,
        env_kwargs_override=env_kwargs_override,
    )
    try:
        result = evaluator.run(policy, n_episodes=args.n_episodes)
    finally:
        evaluator.close()

    agg = result["aggregate"]
    print(
        f"[MS3Evaluator] task={agg['task']} n={agg['n_episodes']} "
        f"success_at_end={agg['success_at_end_mean']:.4f} "
        f"success_once={agg['success_once_mean']:.4f} "
        f"ep_len={agg['episode_len_mean']:.2f}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    # Normalize adapter tag: 'mod.path:Cls' -> 'Cls'
    adapter_tag = args.policy_adapter.split(":")[-1]
    out_path = os.path.join(
        args.output_dir, f"{ts}-{args.task}-{adapter_tag}.json"
    )
    payload = {
        "cli_args": vars(args),
        "result": result,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[MS3Evaluator] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
