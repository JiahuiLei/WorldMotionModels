"""MS3 replay handler — direct h5-obs-slice converter for MS3 BC demos.

Schema v3 pipeline. Per-state fields (qpos, qvel, tcp_pose, obj_poses/*,
goal/*) are sliced directly out of the vendor-recorded
``h5["traj_X"]["obs"]`` array using the frozen per-task layout in
``schema.OBS_SLICES`` (transcribed from ``bc_anatomy.md`` §(c)).

Action stream is pass-through; action_dim is per-task:

* ``pd_ee_delta_pos``  (Push/Pick/StackCube) — 4D: [dx, dy, dz, gripper]
* ``pd_ee_delta_pose`` (PegInsertionSide)    — 7D: [dx, dy, dz, drx, dry, drz, gripper]

The control mode is auto-detected from the JSON sidecar or can be overridden
explicitly. See ``schema.TASK_CONTROL_MODE`` for the frozen per-task mapping.

v2 → v3 delta: for tasks that randomize scene geometry per-env at
``_load_scene`` (currently PegInsertionSide-v1 only — peg length, peg
radius, and hole in-box offset all sampled from
``_batched_episode_rng.uniform`` per env), each episode's
``meta["scene_params"]`` now records the per-episode geometry dict.

For PegIns, scene_params are extracted by resetting a fresh env with the
episode's stored seed and reading from ``env.unwrapped`` directly:

* ``peg_half_sizes``   = ``env.unwrapped.peg_half_sizes[0]``  (3,)
* ``peg_head_offsets`` = ``env.unwrapped.peg_head_offsets.p[0]``  (3,)
* ``box_hole_offsets`` = ``env.unwrapped.box_hole_offsets.p[0]``  (3,)

This is the ground truth (scene is built once at ``_load_scene`` using
``_batched_episode_rng`` seeded deterministically from the reset seed).
A per-episode env reset costs ~200 ms but only runs once per conversion.

Cross-check: ``peg_half_sizes`` is ALSO in ``h5.obs[:, 32:35]`` (per-frame
constant because the vendor writes scene constants in every step). The
handler verifies ``obs[:, 32:35]`` is constant across the episode; if it
drifts the run aborts. This keeps us honest if the vendor ever changes
what's included in ``_get_obs_extra``. ``box_hole_offsets`` is NOT in
``h5.obs``, so env reset is the ONLY source.

Why schema v2 (h5.obs slicing) instead of v1 (env.set_sim_state →
_get_obs_state_dict): round_trip_diff found v1 introduced a ~1.96 cm drift
on Panda gripper qpos (mimic-joint doubling convention mismatch between
``set_sim_state`` and ``agent.qpos`` accessor) and ~1e-6 m/s spurious qvel
at step 0. v2/v3 are bit-exact to the vendor-recorded obs stream by
construction.

An env is still instantiated ONCE PER TASK at ``save_task_meshes`` to
dump ``robot_embodiment.npz`` + ``object_embodiment.npz`` via
``lib_mani.mesh_utils``. Per-episode scene_params extraction reads from
h5.obs (no env reset) — round_trip_diff.py independently verifies these
values against a fresh env reset per episode.

Reviewer contract: this file must NOT import from or modify any code under
`prepare_data/maniskill/` or `ext/lib_mani/` that would require edits there.
We reuse ``lib_mani.mesh_utils.save_robot_meshes`` / ``save_object_meshes``
— read-only import, no lib_mani edits.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import gymnasium as gym
import h5py
import mani_skill  # noqa: F401  # registers env ids in gym registry
import numpy as np
from tqdm import tqdm

try:
    import mani_skill as _ms
    _MS_VERSION = getattr(_ms, "__version__", "unknown")
except Exception:  # pragma: no cover
    _MS_VERSION = "unknown"

from lib_mani.mesh_utils import save_robot_meshes, save_object_meshes

from . import schema as _schema


def save_task_meshes(env_id: str, task_root: str) -> None:
    """Save ``robot_embodiment.npz`` + ``object_embodiment.npz`` under ``task_root``.

    These files are per-task, NOT per-demo-count — they live at the task
    directory root (``data/ms3/umm_format/<TASK>/``) and are shared by every
    demo-count subdirectory (``100/``, ``1000/``, …).

    Mirrors the old ``prepare_data/maniskill/replay.py:25-30`` pattern but
    targets the ms3 task-root layout. Does nothing if both files already
    exist.

    Args:
        env_id: ManiSkill env id (e.g. ``"PushCube-v1"``).
        task_root: Destination directory (created if missing).
    """
    os.makedirs(task_root, exist_ok=True)
    robot_path = osp.join(task_root, "robot_embodiment.npz")
    object_path = osp.join(task_root, "object_embodiment.npz")
    if osp.exists(robot_path) and osp.exists(object_path):
        return
    env = gym.make(env_id, obs_mode="rgbd")
    try:
        if not osp.exists(robot_path):
            save_robot_meshes(task_root, env)
        if not osp.exists(object_path):
            save_object_meshes(task_root, env)
    finally:
        env.close()


class MS3ReplayHandler:
    """Convert a vendor demo h5 into per-episode .npz files via schema-v2 slicing."""

    def __init__(self, env_id: str):
        self.env_id = env_id
        if env_id not in _schema.TASK_OBJECT_FIELDS:
            raise ValueError(
                f"Unknown env_id {env_id!r}. Known: "
                f"{sorted(_schema.TASK_OBJECT_FIELDS)}"
            )

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def replay_trajectory(
        self,
        traj_h5_path: str,
        output_dir: str,
        count: Optional[int] = None,
        control_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert demos to per-episode npz files under ``output_dir``.

        Args:
            traj_h5_path: Path to vendor ``trajectory.state.<control_mode>.physx_cpu.h5``.
            output_dir: Destination demo-count subdir (e.g. ``.../PushCube-v1/100``).
            count: Number of episodes to convert. None -> all.
            control_mode: Override the control mode. If None, auto-detect from
                the JSON sidecar. Must match the h5's env_kwargs.

        Returns:
            Summary dict (also written as ``output_dir/meta.json``).
        """
        os.makedirs(output_dir, exist_ok=True)

        json_path = traj_h5_path[:-3] + ".json" if traj_h5_path.endswith(".h5") else traj_h5_path + ".json"
        if not osp.exists(json_path):
            raise FileNotFoundError(f"JSON sidecar not found: {json_path}")
        with open(json_path, "r") as f:
            sidecar = json.load(f)

        env_kwargs: Dict[str, Any] = sidecar.get("env_info", {}).get("env_kwargs", {})
        sidecar_mode = env_kwargs.get("control_mode")
        if control_mode is None:
            control_mode = sidecar_mode
        elif sidecar_mode is not None and sidecar_mode != control_mode:
            raise RuntimeError(
                f"control_mode override {control_mode!r} disagrees with "
                f"sidecar env_kwargs.control_mode {sidecar_mode!r} in "
                f"{json_path}"
            )
        if control_mode not in _schema.CONTROL_MODE_ACTION_DIMS:
            raise RuntimeError(
                f"Unsupported control_mode {control_mode!r}; "
                f"known: {sorted(_schema.CONTROL_MODE_ACTION_DIMS)}"
            )
        expected_action_dim = _schema.CONTROL_MODE_ACTION_DIMS[control_mode]
        # Frozen per-task mapping check — catches a cross-wired h5/task pair.
        expected_task_mode = _schema.TASK_CONTROL_MODE.get(self.env_id)
        if expected_task_mode is not None and expected_task_mode != control_mode:
            raise RuntimeError(
                f"Task {self.env_id!r} expects control_mode "
                f"{expected_task_mode!r} (per TASK_CONTROL_MODE), got "
                f"{control_mode!r}."
            )

        seed_map: Dict[int, int] = {}
        for ep in sidecar.get("episodes", []):
            eid = int(ep["episode_id"])
            seed = ep.get("reset_kwargs", {}).get("seed", None)
            if seed is None:
                seed = ep.get("episode_seed", None)
            if seed is None:
                raise RuntimeError(f"Episode {eid} in {json_path} has no seed.")
            seed_map[eid] = int(seed)

        obs_slices = _schema.OBS_SLICES.get(self.env_id)
        if obs_slices is None:
            raise RuntimeError(
                f"No OBS_SLICES registered for {self.env_id}. "
                f"Known: {sorted(_schema.OBS_SLICES)}"
            )

        # Env builder (lazy) — only instantiated if the task has per-episode
        # scene_params that need env-reset extraction.
        scene_param_spec = _schema.TASK_SCENE_PARAM_FIELDS.get(self.env_id, {})
        scene_env = None
        if scene_param_spec:
            scene_env = gym.make(self.env_id, obs_mode="state", sim_backend="physx_cpu")

        converted: List[int] = []
        with h5py.File(traj_h5_path, "r") as hf:
            all_keys = sorted(
                [k for k in hf.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[1]),
            )
            total_in_h5 = len(all_keys)
            keys = all_keys if count is None else all_keys[:count]

            for key in tqdm(keys, desc=f"Convert {self.env_id}"):
                idx = int(key.split("_")[1])
                g = hf[key]

                # Per-transition fields — bit-exact pass-through.
                actions = np.asarray(g["actions"][:], dtype=np.float32)
                terminated = np.asarray(g["terminated"][:], dtype=bool)
                truncated = np.asarray(g["truncated"][:], dtype=bool)
                success = np.asarray(g["success"][:], dtype=bool)

                # Per-state fields — direct slice of the vendor-recorded obs.
                # NO env-reconstruction: we store exactly what the vendor wrote.
                # Schema version 2 vs v1 (env-reconstruction, had gripper drift).
                obs = np.asarray(g["obs"][:], dtype=np.float32)  # (T+1, D)
                T = actions.shape[0]
                if obs.shape[0] != T + 1:
                    raise RuntimeError(
                        f"{key}: obs has {obs.shape[0]} frames but actions "
                        f"has {T}. Expected n_states = T+1."
                    )

                if actions.shape[-1] != expected_action_dim:
                    raise RuntimeError(
                        f"{key}: action dim {actions.shape[-1]} != "
                        f"expected {expected_action_dim} for "
                        f"control_mode={control_mode!r}"
                    )
                n_out = int((np.abs(actions) > 1.0).sum())
                if n_out:
                    raise RuntimeError(
                        f"{key}: {n_out} action scalars outside [-1, 1]; "
                        f"max abs = {float(np.abs(actions).max()):.6f}. "
                        "Refusing to silently clip."
                    )

                # Slice obs into stored fields.
                stored: Dict[str, np.ndarray] = {}
                obj_bufs: Dict[str, np.ndarray] = {}
                goal_bufs: Dict[str, np.ndarray] = {}
                for field_key, (s, e) in obs_slices.items():
                    arr = obs[:, s:e].astype(np.float32, copy=True)
                    if field_key.startswith("obj_poses/"):
                        obj_bufs[field_key.split("/", 1)[1]] = arr
                    elif field_key.startswith("goal/"):
                        goal_bufs[field_key.split("/", 1)[1]] = arr
                    else:
                        stored[field_key] = arr

                # Per-episode scene_params (schema v3). For PegIns, reset
                # a fresh env at the episode's seed and read the ground-
                # truth geometry from env.unwrapped. Cross-verify
                # peg_half_sizes against h5.obs[:, 32:35] (per-frame
                # constant) as belt-and-suspenders.
                seed = seed_map[idx]
                scene_params = self._extract_scene_params(obs, scene_env, seed)
                episode = {
                    "qpos": stored["qpos"],
                    "qvel": stored["qvel"],
                    "tcp_pose": stored["tcp_pose"],
                    "obj_poses": obj_bufs,
                    "goal": goal_bufs,
                    "scene_params": scene_params,
                    "action": actions,
                    "success": success,
                    "terminated": terminated,
                    "truncated": truncated,
                    "meta": {
                        "seed": int(seed),
                        "env_id": self.env_id,
                        "control_mode": control_mode,
                        "action_dim": int(expected_action_dim),
                        "episode_id": int(idx),
                        "schema_version": int(_schema.SCHEMA_VERSION),
                    },
                }
                out_path = osp.join(output_dir, f"traj_{idx:04d}.npz")
                _schema.save_episode(out_path, episode)
                converted.append(idx)

        if scene_env is not None:
            scene_env.close()

        obj_fields = [k.split("/", 1)[1] for k in obs_slices if k.startswith("obj_poses/")]
        goal_fields = [k.split("/", 1)[1] for k in obs_slices if k.startswith("goal/")]
        scene_param_fields = sorted(_schema.TASK_SCENE_PARAM_FIELDS.get(self.env_id, {}))
        summary = {
            "env_id": self.env_id,
            "control_mode": control_mode,
            "action_dim": int(expected_action_dim),
            "schema_version": int(_schema.SCHEMA_VERSION),
            "total_traj_in_h5": int(total_in_h5),
            "converted_count": int(len(converted)),
            "requested_count": int(count) if count is not None else None,
            "per_ep_seeds": {str(i): int(seed_map[i]) for i in converted},
            "mani_skill_version": str(_MS_VERSION),
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "source_h5": osp.abspath(traj_h5_path),
            "obj_fields": obj_fields,
            "goal_fields": goal_fields,
            "scene_param_fields": scene_param_fields,
        }
        with open(osp.join(output_dir, "meta.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ---------------------------------------------------------------
    # scene_params extraction (schema v3)
    # ---------------------------------------------------------------
    def _extract_scene_params(
        self, obs: np.ndarray, scene_env, seed: int,
    ) -> Dict[str, np.ndarray]:
        """Return per-episode scene_params for this task.

        For tasks with per-env geometry randomization (currently PegIns),
        the ground-truth source is a fresh env reset at the episode's
        stored seed — reads ``env.unwrapped.peg_half_sizes[0]``,
        ``peg_head_offsets.p[0]`` and ``box_hole_offsets.p[0]`` directly.

        Args:
            obs:       (T+1, obs_dim) float32 — vendor-recorded obs stream.
                       Used ONLY as a cross-check against the env-reset
                       ground truth (``obs[:, 32:35] == peg_half_sizes``).
            scene_env: the single env instantiated at the top of the
                       conversion loop. ``reset(seed)`` per episode yields
                       per-episode randomized geometry.
            seed:      episode seed from the h5 sidecar.

        Returns:
            Dict ``{field_name: (shape,) float32}`` matching
            ``schema.TASK_SCENE_PARAM_FIELDS[env_id]``. Empty for tasks
            without per-env geometry randomization.
        """
        import torch

        out: Dict[str, np.ndarray] = {}
        spec = _schema.TASK_SCENE_PARAM_FIELDS.get(self.env_id, {})
        if not spec:
            return out

        if self.env_id == "PegInsertionSide-v1":
            if scene_env is None:
                raise RuntimeError(
                    "PegIns scene_params extraction requires a live env "
                    "(scene_env=None). This is a bug in replay_trajectory."
                )
            scene_env.reset(seed=int(seed))
            u = scene_env.unwrapped
            num_envs = int(getattr(u, "num_envs", 1))
            if num_envs != 1:
                raise RuntimeError(
                    f"Expected num_envs=1 for scene_params extraction, "
                    f"got {num_envs}. Indexing [0] would silently pick "
                    f"one env out of many."
                )

            def _as_np3(x) -> np.ndarray:
                if torch.is_tensor(x):
                    x = x.cpu().numpy()
                return np.asarray(x[0], dtype=np.float32)

            peg_half_sizes   = _as_np3(u.peg_half_sizes)         # (3,)
            peg_head_offsets = _as_np3(u.peg_head_offsets.p)     # (3,)
            box_hole_offsets = _as_np3(u.box_hole_offsets.p)     # (3,)

            out["peg_half_sizes"]   = peg_half_sizes
            out["peg_head_offsets"] = peg_head_offsets
            out["box_hole_offsets"] = box_hole_offsets

            # Cross-check #1: peg_half_sizes is in h5.obs[:, 32:35] and is
            # constant across frames. If anything drifts, abort.
            s, e = _schema.PEG_INS_OBS_PEG_HALF_SIZE_SLICE
            peg_half_from_obs = obs[:, s:e]
            if not np.all(peg_half_from_obs == peg_half_from_obs[0:1]):
                max_drift = float(np.abs(peg_half_from_obs - peg_half_from_obs[0:1]).max())
                raise RuntimeError(
                    f"peg_half_sizes not constant across h5.obs frames "
                    f"(max_drift={max_drift:.3e})."
                )
            if not np.allclose(peg_half_from_obs[0], peg_half_sizes, atol=1e-6):
                raise RuntimeError(
                    f"peg_half_sizes mismatch: env-reset "
                    f"{peg_half_sizes.tolist()} vs h5.obs[0, 32:35] "
                    f"{peg_half_from_obs[0].tolist()}. Seed={seed}."
                )

            # Cross-check #2: peg_head_offsets is derivable from peg_half_sizes
            # as [length, 0, 0] (peg_insertion_side.py:124-126). If the
            # vendor ever relocates the peg head, env-reset gives the truth
            # and we catch it here.
            expected_head = np.array(
                [peg_half_sizes[0], 0.0, 0.0], dtype=np.float32,
            )
            if not np.allclose(peg_head_offsets, expected_head, atol=1e-6):
                raise RuntimeError(
                    f"peg_head_offsets unexpected: env {peg_head_offsets.tolist()} "
                    f"vs derived {expected_head.tolist()}. Upstream change?"
                )

            # Shape gate — frozen by schema.TASK_SCENE_PARAM_FIELDS.
            for k, expected_shape in spec.items():
                assert out[k].shape == expected_shape, (
                    f"scene_params[{k!r}] shape {out[k].shape} != "
                    f"expected {expected_shape}"
                )
            return out

        raise RuntimeError(
            f"TASK_SCENE_PARAM_FIELDS has a non-empty spec for "
            f"{self.env_id!r} but _extract_scene_params has no handler. "
            "Add an explicit branch."
        )


# Import-time guard: every task with a non-empty TASK_SCENE_PARAM_FIELDS
# entry must have a matching branch in MS3ReplayHandler._extract_scene_params
# AND round_trip_diff._extract_env_scene_params. We can't import
# round_trip_diff here (circular), but we can verify handler coverage at
# import time so the failure surfaces at module load rather than halfway
# through a 1000-episode data-conversion run.
_SUPPORTED_SCENE_PARAM_TASKS = {"PegInsertionSide-v1"}
for _env_id, _spec in _schema.TASK_SCENE_PARAM_FIELDS.items():
    if _spec and _env_id not in _SUPPORTED_SCENE_PARAM_TASKS:
        raise RuntimeError(
            f"TASK_SCENE_PARAM_FIELDS has non-empty spec for {_env_id!r} "
            f"but MS3ReplayHandler._extract_scene_params has no matching "
            f"branch. Add {_env_id!r} to _SUPPORTED_SCENE_PARAM_TASKS and "
            f"implement the extraction. (Prevents conversion-time failure "
            f"mid-run.)"
        )
del _env_id, _spec
