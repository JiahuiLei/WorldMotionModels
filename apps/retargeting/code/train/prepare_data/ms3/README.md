# prepare_data/ms3

Converter from ManiSkill v3 motion-planning demos (`pd_ee_delta_pos`,
`physx_cpu`) into a flat per-episode WMM format.

**Independence.** This module does NOT import from `prepare_data/maniskill/`,
`src/umm/datasets/umm_v1/maniskill_oracle_dataset.py`, `configs/maniskill/`, or
edit `ext/lib_mani/`. It reuses only `lib_mani.utils.load_h5_recursive` (read-
only import).

## Layout

```
prepare_data/ms3/
  __init__.py
  schema.py              # dataclass + save/load helpers (SPEC only)
  ms3_replay_handler.py  # MS3ReplayHandler class
  replay.py              # CLI: convert h5 -> per-episode npz
  verify_replay.py       # CLI: replay actions, assert >=95% cumulative-success
  README.md
```

Output layout:

```
data/ms3/umm_format/<TASK>/<N>/
  meta.json              # summary: counts, versions, env_kwargs, per-ep seeds
  traj_0000.npz
  traj_0001.npz
  ...
```

## Per-episode fields (`schema.py` is authoritative)

`T+1` states (inclusive of terminal state), `T` transitions.

| Field            | Shape        | dtype   | Notes                                |
|------------------|--------------|---------|--------------------------------------|
| qpos             | (T+1, 9)     | float32 | Panda 7-DoF + 2-DoF gripper          |
| qvel             | (T+1, 9)     | float32 |                                      |
| tcp_pose         | (T+1, 7)     | float32 | [pos(3), quat_wxyz(4)] world frame   |
| obj_poses/*      | (T+1, 7)     | float32 | per-task, see `TASK_OBJECT_FIELDS`   |
| goal/*           | (T+1, D)     | float32 | per-task, e.g. `goal_pos (T+1, 3)`   |
| scene_params/*   | per-key      | float32 | per-EPISODE geometry (v3+, see below)|
| action           | (T, 4) or (T, 7) | float32 | `pd_ee_delta_pos` 4-DOF (Push/Pick/StackCube) or `pd_ee_delta_pose` 7-DOF (PegInsertionSide) ∈ [-1, 1] |
| success          | (T,)         | bool    | raw per-step flag from demo h5       |
| terminated       | (T,)         | bool    |                                      |
| truncated        | (T,)         | bool    |                                      |
| meta/*           | scalar       | -       | seed / env_id / control_mode / action_dim / episode_id / schema_version |

## Usage

```bash
conda activate umm

# Convert
python -m prepare_data.ms3.replay \
    --env_id PushCube-v1 \
    --traj_path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --output_dir data/ms3/umm_format/PushCube-v1/100 \
    --count 100

# Verify
python -m prepare_data.ms3.verify_replay \
    --env_id PushCube-v1 \
    --input_dir data/ms3/umm_format/PushCube-v1/100
```

## Per-episode scene geometry (v3)

Some MS3 tasks randomize scene geometry **per-env** inside `_load_scene`,
so each episode of the same "task" has different object dimensions. The
shared `object_embodiment.npz` captures env 0's meshes only — using it
for every episode gives the wrong geometry (visible gap at insertion
etc.) for every non-env-0 episode.

Schema v3 fixes this by adding `meta["scene_params"]` per episode:

| Task                | scene_params keys                                          |
|---------------------|------------------------------------------------------------|
| PushCube-v1         | `{}` (cube_half_size is a frozen constant)                 |
| PickCube-v1         | `{}`                                                       |
| StackCube-v1        | `{}`                                                       |
| PegInsertionSide-v1 | `peg_half_sizes (3,)`, `peg_head_offsets (3,)`             |

Source of randomization: `peg_insertion_side.py:114-126`:

```
lengths = self._batched_episode_rng.uniform(0.085, 0.125)       # per-env
radii   = self._batched_episode_rng.uniform(0.015, 0.025)       # per-env
peg_half_sizes = np.vstack([lengths, radii, radii]).T           # (N, 3)
peg_head_offsets[:, 0] = peg_half_sizes[:, 0]                   # = length
```

### Dataloader / viz contract (NON-NEGOTIABLE)

**Consumers MUST read `meta["scene_params"]` at load time and rebuild the
reference mesh accordingly for tasks with a non-empty spec.** Assuming a
single fixed mesh per task will produce wrong geometry for PegIns.

For PegIns specifically, the peg mesh must be rebuilt at load time as an
axis-aligned box of `half_size = peg_half_sizes` (peg-local frame; head
along +x, tail along -x). See
`prepare_data/ms3/viser_rollout.py::_override_peg_mesh_from_scene_params`
for a reference implementation (schema v3).

Evaluator obs_dict also carries `scene_params` (non-breaking data-side
addition; existing adapters that ignore the key remain compatible). See
`eval/ms3/policies/base.py::MS3PolicyAdapter.obs_dict_spec`.

### Round-trip bit-exactness

`round_trip_diff.py` independently verifies each stored `scene_params/*`
by re-resetting a fresh env at `meta["seed"]` and reading
`env.unwrapped.peg_half_sizes[0]` / `peg_head_offsets.p[0]`. Pass
criterion: exact float32 equality.

## Demo counts in source h5

| Task                  | Demos in h5 |
|-----------------------|-------------|
| PushCube-v1           | 995         |
| PickCube-v1           | 1000        |
| StackCube-v1          | 990         |
| PegInsertionSide-v1   | 946 under `pd_ee_delta_pose` (54 dropped by vendor pos→pose retarget) |

## Verify pass criterion

Honest replay: `env.reset(seed=stored_seed)`, then step through saved actions
and cumulatively OR `info["success"]`. Episode passes if `success_once` is
true. Per-task pass rate must be >= 95%, except PegInsertionSide-v1 which is
exempt (thin data).
