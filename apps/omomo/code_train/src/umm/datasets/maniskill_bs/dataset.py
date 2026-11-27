"""Flat (obs, action) dataset for the maniskill_bs BC mirror.

Loads ms3 ``umm_format`` episodes (see ``prepare_data/ms3/schema.py``) and
emits per-step ``(obs, action)`` pairs that match the official ManiSkill3
state-obs flatten order exactly (``sapien_env.py:535-544`` →
``common.flatten_state_dict`` → dict insertion order).

**Scope for ** PushCube-v1 only. Other tasks rely on derived fields
(``is_grasped``, ``tcp_to_obj_pos``, ``peg_half_size``, etc.) that are NOT
stored in the ms3 schema yet — a future SG will expand coverage.

No imports from ``umm_v1``. This is a clean sibling dataset version.
"""

import glob
import os
import os.path as osp
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare_data.ms3.schema import load_episode


# PushCube-v1 flat obs: qpos(9) + qvel(9) + tcp_pose(7) + goal_pos(3) + obj_pose(7) = 35.
# Matches the official env _get_obs_state_dict insertion order (agent.*, extra.*).
_PUSHCUBE_STATE_DIM: int = 35
_ACTION_DIM: int = 4


class ManiskillBSDataset(Dataset):
    """Flat step-level BC dataset over ms3 umm_format episodes.

    Each sample is a single ``(obs, action)`` pair drawn from any episode. Obs
    is the pre-transition state (episode index ``t`` for ``t = 0..T-1``); the
    paired action is ``action[t]``.

    Args:
        data_root: Path to ``data/ms3/umm_format/<TASK>/<N>`` containing
            ``traj_*.npz`` files.
        task: Task id, e.g. ``"PushCube-v1"``. **Only PushCube-v1 is supported
            in (canary).** Others raise ``NotImplementedError``.
        split_indices: Optional list of global step indices to restrict this
            dataset to (used by the datamodule for train/val splits). ``None``
            means "use all steps".

    Attributes:
        state_dim (int): 35 for PushCube.
        action_dim (int): 4.
        _episodes (list[dict]): Loaded episode dicts (one per traj_*.npz).
        _step_index (np.ndarray): Shape ``(N_steps, 2)`` int64 array mapping a
            flat step index to ``(episode_idx, within_episode_step_idx)``.
    """

    def __init__(
        self,
        data_root: str,
        task: str,
        split_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if task != "PushCube-v1":
            raise NotImplementedError(
                "The canary is PushCube-v1 only; other tasks (PickCube, "
                "StackCube, PegInsertionSide) need derived state fields "
                "(is_grasped, tcp_to_obj_pos, etc.) that are NOT stored in "
                "the ms3 schema. A future SG will expand coverage."
            )
        self.task = task
        self.state_dim = _PUSHCUBE_STATE_DIM
        self.action_dim = _ACTION_DIM

        if not osp.isdir(data_root):
            raise FileNotFoundError(f"data_root not found: {data_root}")
        traj_paths = sorted(glob.glob(osp.join(data_root, "traj_*.npz")))
        if len(traj_paths) == 0:
            raise FileNotFoundError(f"No traj_*.npz under {data_root}")

        self._episodes: List[dict] = [load_episode(p) for p in traj_paths]

        # Build flat step index. Per episode: T transitions = len(action);
        # obs is taken from the pre-transition state at index t ∈ [0, T-1].
        index_rows = []
        for ei, ep in enumerate(self._episodes):
            T = int(ep["action"].shape[0])
            # Consistency check: qpos/qvel/tcp_pose/obj_pose/goal_pos all have
            # T+1 frames; action has T frames. Guard against malformed data.
            expected_state_frames = T + 1
            for k in ("qpos", "qvel", "tcp_pose"):
                if ep[k].shape[0] != expected_state_frames:
                    raise ValueError(
                        f"{traj_paths[ei]}: {k}.shape[0]={ep[k].shape[0]} "
                        f"but action length is {T}; expected {expected_state_frames}."
                    )
            for k, arr in ep["obj_poses"].items():
                if arr.shape[0] != expected_state_frames:
                    raise ValueError(
                        f"{traj_paths[ei]}: obj_poses[{k}].shape[0]={arr.shape[0]} "
                        f"!= expected {expected_state_frames}."
                    )
            for k, arr in ep["goal"].items():
                if arr.shape[0] != expected_state_frames:
                    raise ValueError(
                        f"{traj_paths[ei]}: goal[{k}].shape[0]={arr.shape[0]} "
                        f"!= expected {expected_state_frames}."
                    )
            for t in range(T):
                index_rows.append((ei, t))

        full_index = np.asarray(index_rows, dtype=np.int64)  # (N_steps, 2)
        if split_indices is None:
            self._step_index = full_index
        else:
            self._step_index = full_index[np.asarray(split_indices, dtype=np.int64)]

    def __len__(self) -> int:
        return int(self._step_index.shape[0])

    def share_split_view(self, split_indices: List[int]) -> "ManiskillBSDataset":
        """Return a new dataset view that shares loaded episodes with ``self``.

        Re-uses the loaded ``_episodes`` list (no re-read from disk) and applies
        a fresh split-index slice. Keeps memory footprint flat when the
        datamodule needs train+val views of the same underlying data.

        Args:
            split_indices: Flat step indices (into the full step index) to
                expose via this view.

        Returns:
            A new ``ManiskillBSDataset`` sharing the same ``_episodes`` list.
        """
        view = self.__class__.__new__(self.__class__)
        view.task = self.task
        view.state_dim = self.state_dim
        view.action_dim = self.action_dim
        view._episodes = self._episodes  # SHARED — no disk re-read
        # Rebuild the full step index from the shared episodes, then slice.
        rows: list[tuple[int, int]] = []
        for ei, ep in enumerate(self._episodes):
            for t in range(int(ep["action"].shape[0])):
                rows.append((ei, t))
        full_index = np.asarray(rows, dtype=np.int64)
        view._step_index = full_index[np.asarray(split_indices, dtype=np.int64)]
        return view

    def total_episode_steps(self) -> int:
        """Total step count across all loaded episodes, ignoring split_indices.

        Useful only for building the initial index (see the datamodule); for
        the length of THIS dataset instance, use ``len(self)``.
        """
        return int(sum(ep["action"].shape[0] for ep in self._episodes))

    def _flatten_obs(self, ep: dict, t: int) -> np.ndarray:
        """Build the PushCube-v1 35-D obs vector for step ``t`` of episode ``ep``.

        Order (insertion order of ``_get_obs_state_dict`` after flatten):
            qpos(9) → qvel(9) → tcp_pose(7) → goal_pos(3) → obj_pose(7) = 35.
        """
        qpos = ep["qpos"][t]                    # (9,) float32
        qvel = ep["qvel"][t]                    # (9,) float32
        tcp = ep["tcp_pose"][t]                 # (7,) float32
        goal = ep["goal"]["goal_pos"][t]        # (3,) float32
        obj = ep["obj_poses"]["obj_pose"][t]    # (7,) float32
        out = np.concatenate([qpos, qvel, tcp, goal, obj], axis=0).astype(np.float32)
        # Shape invariant: (35,) — raise instead of assert (strippable by -O).
        if out.shape != (_PUSHCUBE_STATE_DIM,):
            raise ValueError(
                f"PushCube-v1 obs shape mismatch: got {out.shape}, expected "
                f"({_PUSHCUBE_STATE_DIM},)"
            )
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(obs, action)`` as float32 tensors.

        Args:
            idx: Flat step index into this (possibly split-restricted) dataset.

        Returns:
            ``(obs, action)`` with shapes ``(35,)`` and ``(4,)`` (both float32).
        """
        ei, t = self._step_index[idx]
        ep = self._episodes[int(ei)]
        obs_np = self._flatten_obs(ep, int(t))          # (35,) float32
        act_np = ep["action"][int(t)].astype(np.float32)  # (4,) float32
        obs = torch.from_numpy(obs_np)                  # (35,) float32
        action = torch.from_numpy(act_np)               # (4,) float32
        return obs, action
