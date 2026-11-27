"""ms3 per-episode schema (SPEC ONLY).

This module is the authoritative specification for every field written to
`data/ms3/umm_format/<TASK>/<N>/traj_<NNNN>.npz`. It is intentionally NOT a
torch Dataset — it only defines shape/dtype/units, plus save/load helpers.

Task directory layout::

    data/ms3/umm_format/<TASK>/
        robot_embodiment.npz         # per-task robot meshes (shared across demo counts)
        object_embodiment.npz        # per-task REFERENCE object meshes (env 0 only)
        100/
            meta.json
            traj_0000.npz
            traj_0001.npz
            ...
        1000/                        # optional; shares the same *_embodiment.npz
            meta.json
            traj_*.npz

The two ``*_embodiment.npz`` files are written ONCE per task at the task
root by ``ms3_replay_handler.save_task_meshes`` (forked from
``prepare_data/maniskill/replay.py:25-30``). Demo-count subdirectories
reuse the same mesh files — they are independent of which episodes were
converted.

Per-episode fields (T = number of environment transitions the demo recorded;
T+1 qpos/qvel/tcp/obj_pose/goal frames are stored — one per state, including
the terminal state after the last action):

    qpos         (T+1, 9)   float32   Panda 7-DoF arm + 2-DoF gripper (rad + m)
    qvel         (T+1, 9)   float32   Panda joint velocities
    tcp_pose     (T+1, 7)   float32   EE SE(3) world frame [pos(3), quat_wxyz(4)]
    obj_poses    dict[str, (T+1, 7) float32]   per-task object SE(3) poses
    goal         dict[str, array]            per-task goal fields (e.g. goal_pos)
    scene_params dict[str, np.ndarray]       per-episode geometry params (v3+)
    action       (T, 4|7)   float32   [dx, dy, dz, gripper] or
                                       [dx, dy, dz, drx, dry, drz, gripper]
    success      (T,)       bool      raw per-step success from demo h5
    terminated   (T,)       bool      raw per-step terminated from demo h5
    truncated    (T,)       bool      raw per-step truncated from demo h5
    meta         dict                 {seed, env_id, control_mode, action_dim,
                                       episode_id, schema_version}

Note: the demo h5 stores `env_states` with length T+1 and
`actions/success/terminated/truncated` with length T. This schema preserves
that asymmetry exactly (no padding, no truncation).

Object-pose key sets per task are frozen in `TASK_OBJECT_FIELDS` below.

Per-episode scene_params (v3 — see DATALOADER / VIZ CONTRACT)
=============================================================

Some MS3 tasks randomize scene geometry PER ENV at ``_load_scene`` time (so
each episode has different object dimensions even though all episodes share
one "task").

    PegInsertionSide-v1:
        peg_half_sizes    (3,) f32  [length, radius, radius] of the peg, where
                                    length ~ U(0.085, 0.125) and
                                    radius ~ U(0.015, 0.025) per-env
                                    (peg_insertion_side.py:114-123).
        peg_head_offsets  (3,) f32  peg-local offset of the peg *head* tip,
                                    = [length, 0, 0] — derivable but stored
                                    for consumer convenience and schema
                                    stability (peg_insertion_side.py:124-126).
        box_hole_offsets  (3,) f32  box-local offset of the hole center,
                                    = [0, center_y, center_z] with
                                    center_{y,z} ~ 0.5*(length-radius)*U(-1,1)
                                    per-env (peg_insertion_side.py:116-130).
                                    REQUIRED for correct box-actor pose at
                                    viz time: the stored obj_poses['box_hole_pose']
                                    is the HOLE world pose = box.pose *
                                    box_hole_offsets (line 261), NOT the box
                                    actor's pose. To set the box actor
                                    correctly, consumers compute
                                      box.pose = box_hole_pose * box_hole_offsets.inv()
                                    — see viser_rollout._compute_box_actor_pose.

    PushCube-v1 / PickCube-v1 / StackCube-v1:
        {}  — no per-env geometry randomization (cube_half_size is a frozen
             class constant 0.02 m; scene is reconstructed identically
             across envs modulo actor initial poses).

DATALOADER / VIZ CONTRACT — NON-NEGOTIABLE
-------------------------------------------
Consumers (dataset classes, viz, evaluators) MUST read
``meta["scene_params"]`` at load time and reconstruct the correct per-episode
geometry from it. Assuming a single fixed mesh per task will produce wrong
geometry for tasks with randomized scene params (e.g. PegInsertionSide will
show a visible gap at insertion because ``object_embodiment.npz`` captures
env 0's peg only).

For PegIns specifically, the visual peg mesh must be reconstructed at load
time as two axis-aligned boxes (head orange at +x, tail white at -x) with
the per-episode ``peg_half_sizes`` — matching the exact mesh construction
in ``_load_scene`` (peg_insertion_side.py:134-164).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Frozen task metadata
# -----------------------------------------------------------------------------

# Per-task expected obj_poses keys (read from env.unwrapped._get_obs_state_dict
# under the `extra.*_pose` namespace).
TASK_OBJECT_FIELDS: Dict[str, List[str]] = {
    "PushCube-v1": ["obj_pose"],
    "PickCube-v1": ["obj_pose"],
    "StackCube-v1": ["cubeA_pose", "cubeB_pose"],
    "PegInsertionSide-v1": ["peg_pose", "box_hole_pose"],
}

# Per-task expected goal keys (each is (T+1, D) float32). Only keys present on
# the task's state_dict are stored; missing tasks get an empty `goal` dict.
TASK_GOAL_FIELDS: Dict[str, List[str]] = {
    "PushCube-v1": ["goal_pos"],
    "PickCube-v1": ["goal_pos"],
    "StackCube-v1": [],
    "PegInsertionSide-v1": [],
}

QPOS_DIM: int = 9    # Panda 7-DoF arm + 2-DoF gripper
TCP_POSE_DIM: int = 7  # [x, y, z, qw, qx, qy, qz]

# Per-task control mode + action_dim come from the -wide reference protocol
# :mod:`eval.ms3.task_protocol`. Re-exported here under the legacy plural
# spelling (``CONTROL_MODE_ACTION_DIMS``) so existing schema consumers
# (ms3_replay_handler, verify_replay, etc.) keep working without churn.
from eval.ms3.task_protocol import (  # noqa: E402
    CONTROL_MODE_ACTION_DIM as _CONTROL_MODE_ACTION_DIM_REF,
    TASK_CONTROL_MODE,
)

# Plural alias kept for back-compat; identical dict to the reference protocol.
CONTROL_MODE_ACTION_DIMS: Dict[str, int] = dict(_CONTROL_MODE_ACTION_DIM_REF)


# Schema version marker stored in every per-episode meta["schema_version"].
# v1 = env-reconstruction pipeline (scene.set_sim_state -> _get_obs_state_dict)
#      → gripper qpos drifted by ~2cm (mimic-joint doubling mismatch) and
#      qvel had ~1e-6 noise at step 0.
# v2 = direct h5.obs slicing pipeline — bit-exact to the vendor-recorded obs.
# v3 = v2 + per-episode ``scene_params`` dict (peg_half_sizes, peg_head_offsets
#      for PegIns; empty dict for Push/Pick/StackCube). Required to fix the
#      PegInsertionSide visual-mesh-mismatch bug (dataloader was using env 0's
#      peg mesh for every episode → visible gap at insertion because each
#      episode has a different randomized peg length).
# Downstream consumers MUST check meta["schema_version"] when the drift
# matters (e.g. policy input normalization statistics) AND must read
# meta["scene_params"] at load time for tasks that randomize scene geometry.
SCHEMA_VERSION: int = 3


# Per-task scene_params field spec. Each entry maps a task id to the
# expected keys + shapes. See module docstring for semantic description.
# Tasks not listed are treated as empty — ``{}``.
TASK_SCENE_PARAM_FIELDS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "PushCube-v1": {},
    "PickCube-v1": {},
    "StackCube-v1": {},
    "PegInsertionSide-v1": {
        # [length, radius, radius] of the peg (peg_insertion_side.py:123).
        "peg_half_sizes":   (3,),
        # [length, 0, 0] — peg-local offset of peg head tip
        # (peg_insertion_side.py:124-126).
        "peg_head_offsets": (3,),
        # [0, center_y, center_z] — box-local offset of the hole center.
        # box_hole_pose = box.pose * box_hole_offsets (line 261) ⇒ consumers
        # compute box.pose = box_hole_pose * box_hole_offsets.inv() to set
        # the box actor correctly (peg_insertion_side.py:128-130).
        "box_hole_offsets": (3,),
    },
}


# Single source of truth for where scene_params live inside the vendor's
# flat obs stream. Both the h5 converter (ms3_replay_handler) and the
# runtime evaluator (build_obs_dict) read from these slices — keeping the
# constants here prevents a silent mismatch if the vendor spec ever drifts.
#
# PegInsertionSide-v1 obs layout (bc_anatomy.md §(c); verified against
# peg_insertion_side.py _get_obs_extra):
#   qpos(9) + qvel(9) + tcp_pose(7) + peg_pose(7) + peg_half_size(3) +
#   box_hole_pose(7) + box_hole_radius(1) = 43
PEG_INS_OBS_PEG_HALF_SIZE_SLICE: Tuple[int, int] = (32, 35)


# Per-task obs-slice layout. Each entry maps a stored npz field to the
# corresponding slice within the h5 ``obs`` array (shape (T+1, obs_dim)).
# Fields NOT stored in our npz (is_grasped, tcp_to_obj_pos, peg_half_size,
# box_hole_radius, etc.) are derivations re-computable at dataset load time
# and are intentionally omitted.
#
# Citation: ``sg/01-ms3-baselines-vendor-and-demos/bc_anatomy.md`` §(c)
# (transcribed from the vendor `_get_obs_state_dict` dict insertion order
# — see ``mani_skill/envs/sapien_env.py:535-551`` +
# ``mani_skill/envs/tasks/tabletop/{push_cube,pick_cube,stack_cube}.py`` for
# Push/Pick/Stack and ``peg_insertion_side.py`` for PegIns).
OBS_SLICES: Dict[str, Dict[str, Tuple[int, int]]] = {
    # PushCube-v1 (35,) — qpos(9)+qvel(9)+tcp_pose(7)+goal_pos(3)+obj_pose(7)
    "PushCube-v1": {
        "qpos":                  (0, 9),
        "qvel":                  (9, 18),
        "tcp_pose":              (18, 25),
        "goal/goal_pos":         (25, 28),
        "obj_poses/obj_pose":    (28, 35),
    },
    # PickCube-v1 (42,) — qpos(9)+qvel(9)+is_grasped(1)+tcp_pose(7)+goal_pos(3)+
    #                    obj_pose(7)+tcp_to_obj_pos(3)+obj_to_goal_pos(3)
    "PickCube-v1": {
        "qpos":                  (0, 9),
        "qvel":                  (9, 18),
        # 18:19 = is_grasped (bool, omitted)
        "tcp_pose":              (19, 26),
        "goal/goal_pos":         (26, 29),
        "obj_poses/obj_pose":    (29, 36),
        # 36:39 = tcp_to_obj_pos (omitted)
        # 39:42 = obj_to_goal_pos (omitted)
    },
    # StackCube-v1 (48,) — qpos(9)+qvel(9)+tcp_pose(7)+cubeA_pose(7)+cubeB_pose(7)+
    #                      tcp_to_cubeA_pos(3)+tcp_to_cubeB_pos(3)+cubeA_to_cubeB_pos(3)
    "StackCube-v1": {
        "qpos":                  (0, 9),
        "qvel":                  (9, 18),
        "tcp_pose":              (18, 25),
        "obj_poses/cubeA_pose":  (25, 32),
        "obj_poses/cubeB_pose":  (32, 39),
        # 39:42 = tcp_to_cubeA_pos (omitted)
        # 42:45 = tcp_to_cubeB_pos (omitted)
        # 45:48 = cubeA_to_cubeB_pos (omitted)
    },
    # PegInsertionSide-v1 (43,) — qpos(9)+qvel(9)+tcp_pose(7)+peg_pose(7)+
    #                              peg_half_size(3)+box_hole_pose(7)+box_hole_radius(1)
    "PegInsertionSide-v1": {
        "qpos":                  (0, 9),
        "qvel":                  (9, 18),
        "tcp_pose":              (18, 25),
        "obj_poses/peg_pose":    (25, 32),
        # 32:35 = peg_half_size (omitted)
        "obj_poses/box_hole_pose": (35, 42),
        # 42:43 = box_hole_radius (omitted)
    },
}

# Back-compat alias for call sites that still assume the 4-dim canary.
ACTION_DIM: int = 4


@dataclass
class EpisodeSchema:
    """Authoritative documentation-only schema. See module docstring."""

    # Per-state (length T+1)
    qpos: np.ndarray = field(default_factory=lambda: np.zeros((0, QPOS_DIM), np.float32))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros((0, QPOS_DIM), np.float32))
    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros((0, TCP_POSE_DIM), np.float32))

    # Per-state object pose dict; each value has shape (T+1, 7) float32.
    obj_poses: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-state goal dict; each value has shape (T+1, D) float32 (D task-specific).
    goal: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-episode scene geometry params (schema v3+). Shape-per-key is frozen in
    # ``TASK_SCENE_PARAM_FIELDS``. Empty dict for tasks without per-env rand.
    scene_params: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-transition (length T). Action_dim is 4 (pd_ee_delta_pos) or 7
    # (pd_ee_delta_pose) depending on ``meta["control_mode"]``.
    action: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    success: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    terminated: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    truncated: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))

    # Metadata dict: {seed:int, env_id:str, control_mode:str, action_dim:int,
    # episode_id:int}. ``control_mode`` + ``action_dim`` are now load-bearing
    # for downstream consumers (dataset classes, evaluators) that need to
    # adapt to per-task action space.
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Save / load helpers
# -----------------------------------------------------------------------------

_SCALAR_META_KEYS = ("seed", "env_id", "control_mode", "episode_id")


def save_episode(path: str, episode: Dict[str, Any]) -> None:
    """Save an episode dict to an .npz file.

    Nested dicts `obj_poses` and `goal` are flattened with slash-separated keys
    (`obj_poses/obj_pose`, `goal/goal_pos`, etc.). The `meta` dict is flattened
    to `meta/<key>` entries; scalars are stored as 0-D numpy arrays.
    """
    to_save: Dict[str, np.ndarray] = {
        "qpos": np.asarray(episode["qpos"], dtype=np.float32),
        "qvel": np.asarray(episode["qvel"], dtype=np.float32),
        "tcp_pose": np.asarray(episode["tcp_pose"], dtype=np.float32),
        "action": np.asarray(episode["action"], dtype=np.float32),
        "success": np.asarray(episode["success"], dtype=bool),
        "terminated": np.asarray(episode["terminated"], dtype=bool),
        "truncated": np.asarray(episode["truncated"], dtype=bool),
    }

    for k, v in episode.get("obj_poses", {}).items():
        to_save[f"obj_poses/{k}"] = np.asarray(v, dtype=np.float32)

    for k, v in episode.get("goal", {}).items():
        to_save[f"goal/{k}"] = np.asarray(v, dtype=np.float32)

    for k, v in episode.get("scene_params", {}).items():
        to_save[f"scene_params/{k}"] = np.asarray(v, dtype=np.float32)

    meta = episode.get("meta", {})
    for mk, mv in meta.items():
        to_save[f"meta/{mk}"] = np.asarray(mv)

    np.savez_compressed(path, **to_save)


def load_episode(path: str) -> Dict[str, Any]:
    """Inverse of `save_episode`. Returns a dict with the same top-level keys."""
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        out: Dict[str, Any] = {
            "qpos": data["qpos"],
            "qvel": data["qvel"],
            "tcp_pose": data["tcp_pose"],
            "action": data["action"],
            "success": data["success"],
            "terminated": data["terminated"],
            "truncated": data["truncated"],
            "obj_poses": {},
            "goal": {},
            "scene_params": {},
            "meta": {},
        }

        for k in keys:
            if k.startswith("obj_poses/"):
                out["obj_poses"][k.split("/", 1)[1]] = data[k]
            elif k.startswith("goal/"):
                out["goal"][k.split("/", 1)[1]] = data[k]
            elif k.startswith("scene_params/"):
                out["scene_params"][k.split("/", 1)[1]] = data[k]
            elif k.startswith("meta/"):
                mk = k.split("/", 1)[1]
                arr = data[k]
                # Decode 0-D arrays to Python scalars for convenience.
                if arr.shape == ():
                    if arr.dtype.kind in ("U", "S"):
                        out["meta"][mk] = str(arr.item())
                    else:
                        out["meta"][mk] = arr.item()
                else:
                    out["meta"][mk] = arr
    return out
