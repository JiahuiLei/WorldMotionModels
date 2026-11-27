"""ManiSkill3 state dataset (schema-v3) for the wmm pipeline.

Reads per-episode ``.npz`` files written by ``prepare_data/ms3/schema.py``
and emits :class:`CanonicalDatasetOutput` following the same prompt-build
conventions as ``ManiskillOracleDataset`` in ``umm_v13``.

Schema-v3 key differences from v13 ManiskillOracleDataset:
  - No split JSON: episode list is built by globbing ``<data_root>/traj_*.npz``.
  - States are stored per-key (qpos, tcp_pose, obj_poses/*) rather than in a
    unified state dict keyed by UID.
  - Palm pose is stored directly as ``tcp_pose`` (pos+quat_wxyz, world frame).
  - Finger poses are recovered via PandaFK from qpos.
  - Actions have length T (not T+1); qpos/states have length T+1.
  - Per-episode ``scene_params`` (PegIns only) are stored in the npz.

Per-task finger psid map (DEVIATION from PANDA_LINK_ID_TO_UMM_UID in
dataset_config.py, which was verified only on PushCube-v1):

    task                    psid L  psid R
    -----------------------  ------  ------
    PushCube-v1              12      14      matches dataset_config
    PickCube-v1              12      14      matches dataset_config
    StackCube-v1             12      13      R shifts to 13 (scene-actor order)
    PegInsertionSide-v1      12      13      same as StackCube

This variability is task-specific and is hard-coded here. Do NOT change
``dataset_config.py`` — owns that file and the psid map there is
intentionally PushCube-verified only.

Off-by-one alignment (per v13 line 403 convention):
    actor trace at t=0  = state at t=0  (no action applied)
    actor trace at t=k (k>=1) = target from action at k-1

4-DoF action zero-padding:
    action (T, 4) = [dx, dy, dz, grip]
    → padded (T, 7) = [dx, dy, dz, 0, 0, 0, grip]
    The zero rotation slots mean ``state_action_to_tgt`` applies zero
    delta_euler, leaving palm orientation unchanged — correct for
    pd_ee_delta_pos control mode.

Phase-2 caveat:
    For PegInsertionSide-v1, the object PCL comes from ``object_embodiment.npz``
    which captures env 0's peg geometry only. Per-episode ``peg_half_sizes``
    from ``scene_params`` are recorded in ``extra_dict`` but NOT yet used to
    reconstruct per-episode meshes. This approximation is a known
    limitation.

Text prompt:
    ``text = self.task_name`` verbatim (e.g. ``"PushCube-v1"``).
    No templating.
"""

from __future__ import annotations

import glob
import logging
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lib_io import UnifiedIO
from lib_mani.utils.action_utils import state_action_to_tgt
from lib_mani.utils.panda_fk import PandaFK

from prepare_data.ms3.schema import (
    TASK_OBJECT_FIELDS,
    TASK_SCENE_PARAM_FIELDS,
    load_episode,
)
from eval.ms3.task_protocol import task_action_dim

from umm.datasets.wmm.base_dataset import PromptV0
from umm.datasets.wmm.dataset_config import (
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
    UID_MANISKILL_TABLETOP_ACT,
    UID_MANISKILL_TABLETOP_EE,
    UID_MANISKILL_TABLETOP_LEFT_FINGER,
    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
)
from umm.datasets.wmm.dataset_template import CanonicalDatasetOutput
from umm.datasets.wmm.dataset_utils import sample_list
from umm.datasets.wmm.mask_scheduler import MS_PATTERNS, MaskScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task psid maps — hard-coded in this file, NOT in dataset_config.py
# ---------------------------------------------------------------------------

# Finger psid map: psid -> key in robot_embodiment.npz
# PushCube/PickCube use {12, 14}; StackCube/PegIns use {12, 13}.
# This was discovered empirically — PANDA_LINK_ID_TO_UMM_UID was
# verified only on PushCube-v1 (psid 14 = panda_rightfinger in that scene).
_TASK_FINGER_PSIDS: Dict[str, Dict[str, int]] = {
    "PushCube-v1":         {"left": 12, "right": 14},
    "PickCube-v1":         {"left": 12, "right": 14},
    "StackCube-v1":        {"left": 12, "right": 13},
    "PegInsertionSide-v1": {"left": 12, "right": 13},
}

# Object psid map: list of psids (one per object field, in TASK_OBJECT_FIELDS order).
# psid 18 = first task object, psid 19 = second task object.
# Verified by vertex-count and color inspection of object_embodiment.npz.
_TASK_OBJ_PSIDS: Dict[str, List[int]] = {
    "PushCube-v1":         [18],        # obj_pose
    "PickCube-v1":         [18],        # obj_pose
    "StackCube-v1":        [18, 19],    # cubeA_pose, cubeB_pose
    "PegInsertionSide-v1": [18, 19],    # peg_pose, box_hole_pose
}

# Palm psid in robot_embodiment.npz is always 10 (panda_hand).
_PALM_PSID: int = 10


# ---------------------------------------------------------------------------
# Helper: quat_wxyz → rotation matrix (vectorized numpy, no torch)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternion array to ``(..., 3, 3)`` rotation matrix.

    Args:
        q: ``(..., 4)`` float32 array; last dim = [w, x, y, z].

    Returns:
        ``(..., 3, 3)`` float32 rotation matrix array.
    """
    q = q.astype(np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z
    # safe normalisation — avoid div-by-zero on degenerate quaternions
    n = np.where(n < 1e-12, 1.0, n)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    # Build (..., 3, 3) matrix
    R = np.stack([
        np.stack([1.0 - (yy + zz), xy - wz,        xz + wy       ], axis=-1),
        np.stack([xy + wz,          1.0 - (xx + zz), yz - wx       ], axis=-1),
        np.stack([xz - wy,          yz + wx,         1.0 - (xx + yy)], axis=-1),
    ], axis=-2)  # (..., 3, 3)
    return R.astype(np.float32)


def _pose7_to_mat4_np(pose7: np.ndarray) -> np.ndarray:
    """Convert ``(N, 7)`` or ``(7,)`` [pos, quat_wxyz] → ``(N, 4, 4)`` or ``(4, 4)`` SE(3).

    Args:
        pose7: ``(N, 7)`` or ``(7,)`` float32 array; last 4 dims = [w, x, y, z].

    Returns:
        ``(N, 4, 4)`` or ``(4, 4)`` float32 SE(3) matrix.

    Example::

        T = _pose7_to_mat4_np(np.array([[0,0,0, 1,0,0,0], [1,2,3, 0,0,0,1]]))
        # T.shape == (2, 4, 4)
    """
    single = pose7.ndim == 1
    if single:
        pose7 = pose7[None]  # (1, 7)

    N = pose7.shape[0]
    R = _quat_wxyz_to_rotmat_np(pose7[:, 3:7])  # (N, 3, 3)
    T = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # (N, 4, 4)
    T[:, :3, :3] = R
    T[:, :3, 3] = pose7[:, :3].astype(np.float32)

    if single:
        return T[0]
    return T


# ---------------------------------------------------------------------------
# Ms3StateDataset
# ---------------------------------------------------------------------------

class Ms3StateDataset(Dataset):
    """ManiSkill3 schema-v3 state dataset for the wmm training pipeline.

    Reads per-episode ``.npz`` files from ``<data_root>/traj_*.npz`` and emits
    :class:`CanonicalDatasetOutput` with 6 robot parts (palm / left-finger /
    right-finger / palm-act / left-finger-act / right-finger-act) plus
    per-task object traces.

    Args:
        data_root: Path to the ``<N>/`` directory containing ``traj_*.npz``
            (e.g. ``"./data_wmm/ms3_states/umm_format/PushCube-v1/1000"``).
        task: Task name (e.g. ``"PushCube-v1"``).  If ``None``, inferred from
            the parent directory of ``data_root``.
        base_fps: ManiSkill3 simulation frame rate (default 20 Hz).
        random_flag: Enable stochastic augmentation / sampling.
        safe_flag: If True, silently skip to next sample on errors.
        repeat: Dataset repetition multiplier.
        t_rate: Temporal down-sampling stride.
        max_T: Maximum number of time steps per sample.
        num_pts: Points per point-cloud per entity.
        robot_mesh_fn: Robot embodiment npz filename (at task root).
        object_mesh_fn: Object embodiment npz filename (at task root).
        mask_scheduler: Override mask scheduler.  Default uses ``MS_PATTERNS``
            (same distribution as ManiskillOracleDataset).
    """

    def __init__(
        self,
        data_root: str,
        task: Optional[str] = None,
        base_fps: float = 20.0,
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        t_rate: int = 1,
        max_T: int = 20,
        num_pts: int = 256,
        robot_mesh_fn: str = "robot_embodiment.npz",
        object_mesh_fn: str = "object_embodiment.npz",
        mask_scheduler: Optional[MaskScheduler] = None,
    ) -> None:
        self.data_root = osp.abspath(data_root)
        self.base_fps = base_fps
        self.t_rate = t_rate
        self.random_flag = random_flag
        self.safe_flag = safe_flag
        self.repeat = repeat
        self.max_T = max_T
        self.max_C = 1
        self.num_pts = num_pts

        # --- Task name ---
        # data_root is e.g. .../PushCube-v1/1000  → parent = PushCube-v1 dir
        task_root = osp.dirname(self.data_root)
        if task is None:
            self.task_name = osp.basename(osp.normpath(task_root))
        else:
            self.task_name = task
        logger.info(f"[Ms3StateDataset] task={self.task_name} data_root={self.data_root}")

        # --- Validate task ---
        if self.task_name not in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP:
            raise ValueError(
                f"Task '{self.task_name}' not in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP"
            )
        if self.task_name not in _TASK_FINGER_PSIDS:
            raise ValueError(
                f"Task '{self.task_name}' not in _TASK_FINGER_PSIDS — add it."
            )
        if self.task_name not in TASK_OBJECT_FIELDS:
            raise ValueError(
                f"Task '{self.task_name}' not in TASK_OBJECT_FIELDS"
            )
        if self.task_name not in _TASK_OBJ_PSIDS:
            raise ValueError(
                f"Task '{self.task_name}' not in _TASK_OBJ_PSIDS — add it. "
                "This map is required for object-mesh PCL lookup at __getitem__ time; "
                "leaving it incomplete would raise KeyError silently mid-epoch."
            )
        # Consistency check: obj_psid count must match obj_field count.
        if len(_TASK_OBJ_PSIDS[self.task_name]) != len(TASK_OBJECT_FIELDS[self.task_name]):
            raise ValueError(
                f"Task '{self.task_name}' _TASK_OBJ_PSIDS has "
                f"{len(_TASK_OBJ_PSIDS[self.task_name])} entries but TASK_OBJECT_FIELDS "
                f"has {len(TASK_OBJECT_FIELDS[self.task_name])}; they must be 1-to-1."
            )

        # --- Static per-task config ---
        self.action_dim: int = task_action_dim(self.task_name)
        self.task_obj_uids: List[int] = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[self.task_name]
        self.task_obj_fields: List[str] = TASK_OBJECT_FIELDS[self.task_name]
        self.task_scene_param_keys: List[str] = list(
            TASK_SCENE_PARAM_FIELDS.get(self.task_name, {}).keys()
        )

        # --- IO + embodiment data ---
        self.io = UnifiedIO()
        self.robot_link_data: dict = self.io.load_npzs(
            osp.join(task_root, robot_mesh_fn)
        )
        self.object_link_data: dict = self.io.load_npzs(
            osp.join(task_root, object_mesh_fn)
        )

        # --- FK (cached once) ---
        self.fk = PandaFK()

        # --- Build indices by globbing traj files ---
        traj_paths = sorted(glob.glob(osp.join(self.data_root, "traj_*.npz")))
        if len(traj_paths) == 0:
            raise FileNotFoundError(
                f"No traj_*.npz files found in {self.data_root!r}"
            )

        self.indices: List[Tuple[str, int]] = []
        self.seq_len: Dict[str, int] = {}

        factor = int(self.t_rate)
        for traj_path in traj_paths:
            sid = osp.splitext(osp.basename(traj_path))[0]  # e.g. "traj_0000"
            # Fast probe: load only 'qpos' shape without allowing pickle
            with np.load(traj_path, allow_pickle=False) as npz:
                T_plus_1 = npz["qpos"].shape[0]  # T+1 frames
            self.seq_len[sid] = T_plus_1
            for tid in range(0, T_plus_1, factor):
                self.indices.append((sid, tid))

        # Repeat indices
        if repeat > 1:
            self.indices = self.indices * repeat

        # --- Mask scheduler ---
        self.mask_scheduler = mask_scheduler or MaskScheduler(
            base_random_prob=0.25, patterns=MS_PATTERNS
        )

        logger.info(
            f"[Ms3StateDataset] {self.task_name}: {len(traj_paths)} trajs, "
            f"{len(self.indices)} indices (t_rate={t_rate}, repeat={repeat})"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.safe_flag:
            while True:
                try:
                    return self.__getitem_inner__(idx)
                except Exception as e:
                    logger.error(f"[Ms3StateDataset] Error at idx={idx}: {e}")
                    idx = (idx + 1) % len(self)
        else:
            return self.__getitem_inner__(idx)

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker for single-threaded numpy (signature required by torch DataLoader)."""
        import os
        import threadpoolctl
        _ = worker_id  # signature required by DataLoader; not used internally
        threadpoolctl.threadpool_limits(limits=1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ------------------------------------------------------------------
    # Trace sampling (verbatim copy of v13 ManiskillOracleDataset)
    # ------------------------------------------------------------------

    def __decide_trace_sampling__(
        self,
        seq_id: str,
        current_abs_tid: int,
        dataset_idx: int,
        random_enabled: bool = True,
    ) -> Tuple[int, int, int]:
        """Decide the temporal slice for one training sample.

        Returns:
            ``(current_ith, start_abs_tid, end_abs_tid)``
        """
        factor = int(self.t_rate)
        seq_len = self.seq_len[seq_id]

        if self.random_flag and random_enabled:
            left_len = np.random.randint(0, self.max_T * factor)
        else:
            left_len = dataset_idx % (self.max_T * factor)
        start_abs_tid = current_abs_tid - left_len

        end_abs_tid = start_abs_tid + self.max_T * factor
        start_abs_tid = max(0, start_abs_tid)
        end_abs_tid = min(end_abs_tid, seq_len)

        current_ith = (current_abs_tid - start_abs_tid) // factor
        return current_ith, start_abs_tid, end_abs_tid

    # ------------------------------------------------------------------
    # PCL helpers
    # ------------------------------------------------------------------

    def _load_object_pcl(self, psid: int) -> np.ndarray:
        """Load ``(N, 6)`` pcl (xyz + rgb in [0, 1]) for an OBJECT psid.

        Mirrors v13 ``ManiskillOracleDataset.__add_maniskill_prompt__`` object
        branch: prefer per-link uint8 ``{psid}_c`` (tile to N rows, divide by
        255 if needed); fall back to per-point ``{psid}_pcl_c``.
        """
        link_data = self.object_link_data
        pcl_xyz = link_data[f"{psid}_pcl"]  # (N, 3) float32
        key_c = f"{psid}_c"
        key_pcl_c = f"{psid}_pcl_c"
        if key_c in link_data:
            c = link_data[key_c].astype(np.float32)  # (3,) or (N, 3)
            if c.max() > 1.0:
                c = c / 255.0
            if c.ndim == 1:
                pcl_c = np.tile(c[None, :], (pcl_xyz.shape[0], 1))
            else:
                pcl_c = c
        elif key_pcl_c in link_data:
            pcl_c = link_data[key_pcl_c].astype(np.float32)  # (N, 3)
            if pcl_c.max() > 1.0:
                pcl_c = pcl_c / 255.0
        else:
            raise ValueError(
                f"object psid {psid}: missing both '{key_c}' and '{key_pcl_c}' in object_link_data"
            )
        return np.concatenate([pcl_xyz, pcl_c], axis=1).astype(np.float32)  # (N, 6)

    def _load_robot_pcl(self, psid: int) -> np.ndarray:
        """Load ``(N, 6)`` pcl for a ROBOT psid (uses per-point ``_pcl_c``).

        Mirrors v13 ``__add_maniskill_prompt__`` robot branch verbatim, which
        intentionally uses ``_pcl_c`` (flat grey 0.5 as written by
        ``lib_mani.mesh_utils._save_meshes_to_npz``) rather than the per-link
        uint8 ``_c`` color. This keeps robot PCL color-neutral (link identity
        comes from UID + sem_type, not the RGB channel).
        """
        link_data = self.robot_link_data
        key_pcl = f"{psid}_pcl"
        key_pcl_c = f"{psid}_pcl_c"
        if key_pcl not in link_data or key_pcl_c not in link_data:
            raise ValueError(
                f"robot psid {psid}: missing '{key_pcl}' or '{key_pcl_c}' in robot_link_data"
            )
        pcl_xyz = link_data[key_pcl]            # (N, 3) float32
        pcl_c = link_data[key_pcl_c].astype(np.float32)  # (N, 3)
        if pcl_c.max() > 1.0:
            pcl_c = pcl_c / 255.0
        return np.concatenate([pcl_xyz, pcl_c], axis=1).astype(np.float32)  # (N, 6)

    def _sample_object_pcl(self, psid: int) -> np.ndarray:
        """Load + sample ``num_pts`` points from an OBJECT psid → ``(num_pts, 6)``."""
        pcl = self._load_object_pcl(psid)
        return sample_list(pcl, self.num_pts, random_flag=self.random_flag)

    def _sample_robot_pcl(self, psid: int) -> np.ndarray:
        """Load + sample ``num_pts`` points from a ROBOT psid → ``(num_pts, 6)``."""
        pcl = self._load_robot_pcl(psid)
        return sample_list(pcl, self.num_pts, random_flag=self.random_flag)

    # ------------------------------------------------------------------
    # Core __getitem__
    # ------------------------------------------------------------------

    def __getitem_inner__(self, idx: int):
        sid, current_abs_tid = self.indices[idx]
        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx
        )

        factor = int(self.t_rate)

        # --- Load episode ---
        ep = load_episode(osp.join(self.data_root, sid + ".npz"))

        qpos_np = ep["qpos"]         # (T+1, 9)
        tcp_pose_np = ep["tcp_pose"] # (T+1, 7)
        action_np = ep["action"]     # (T, 4 or 7)

        T_steps = action_np.shape[0]   # T (number of transitions)
        T_plus_1 = qpos_np.shape[0]    # T+1

        # Off-by-one guard
        assert T_steps == T_plus_1 - 1, (
            f"schema-v3 off-by-one: action len={T_steps} != qpos len-1={T_plus_1-1} "
            f"in {sid}"
        )

        # --- Palm SE(3) from tcp_pose (T+1, 7) → (T+1, 4, 4), WORLD frame ---
        palm_T_wl = torch.as_tensor(
            _pose7_to_mat4_np(tcp_pose_np), dtype=torch.float32
        )  # (T+1, 4, 4)

        # --- Finger SE(3) from FK, lifted into the SAME WORLD frame as palm ---
        # ``PandaFK.forward`` returns link poses in the ``panda_link0`` (robot
        # root) frame.  ManiSkill3 tabletop scenes mount ``panda_link0`` at a
        # fixed non-zero world offset (``[-0.615, 0, 0]``), so the raw FK
        # finger poses sit ~0.6 m away from the world-frame ``palm_T_wl`` on
        # the robot-base +x axis.  Without this lift, every finger trace in
        # the batch is corrupted: the visualizer renders fingers off-screen
        # and ``state_action_to_tgt`` bakes the offset into finger targets.
        #
        # Recover the robot-base-to-world transform per timestep from the
        # TCP identity ``tcp_world = T_base_world @ FK.panda_hand_tcp`` →
        # ``T_base_world = tcp_world @ inv(FK.panda_hand_tcp)``, then lift
        # every finger FK output through it.  Identity verified numerically
        # to 1e-7 rad/m across all 4 MS3 tasks.
        qpos_t = torch.as_tensor(qpos_np, dtype=torch.float32)  # (T+1, 9)
        fk_out = self.fk.forward(qpos_t)                         # dict → (T+1, 4, 4)
        tcp_in_base = fk_out["panda_hand_tcp"]                   # (T+1, 4, 4)
        T_base_world = palm_T_wl @ torch.linalg.inv(tcp_in_base) # (T+1, 4, 4)
        left_finger_T_wl = T_base_world @ fk_out["panda_leftfinger"]    # (T+1, 4, 4)
        right_finger_T_wl = T_base_world @ fk_out["panda_rightfinger"]  # (T+1, 4, 4)

        # --- Action → target poses ---
        action_t = torch.as_tensor(action_np, dtype=torch.float32)  # (T, 4 or 7)
        if self.action_dim == 4:
            # pd_ee_delta_pos: zero-pad rotation slots
            action7 = torch.cat(
                [action_t[:, :3], torch.zeros(T_steps, 3), action_t[:, 3:4]], dim=-1
            )  # (T, 7)
        else:
            # pd_ee_delta_pose: straight through
            action7 = action_t  # (T, 7)

        tgt_palm, tgt_L, tgt_R = state_action_to_tgt(
            current_palm_pose=palm_T_wl[:T_steps],         # (T, 4, 4)
            current_left_finger_pose=left_finger_T_wl[:T_steps],
            current_right_finger_pose=right_finger_T_wl[:T_steps],
            action=action7,
            pos_limit=0.1,
            rot_limit=0.1,
        )  # each (T, 4, 4)

        # t+1 alignment: actor at t=0 is the state; at t=k (k>=1) is tgt[k-1].
        # schema-v3: tgt_palm is (T, 4, 4); palm_T_wl is (T+1, 4, 4).
        # cat([state[:1], tgt_all]) gives (T+1, 4, 4) matching state length.
        palm_act_T_wl = torch.cat([palm_T_wl[:1], tgt_palm], dim=0)         # (T+1, 4, 4)
        left_act_T_wl = torch.cat([left_finger_T_wl[:1], tgt_L], dim=0)
        right_act_T_wl = torch.cat([right_finger_T_wl[:1], tgt_R], dim=0)

        # --- Object SE(3): per task field → (T+1, 4, 4) ---
        obj_T_list: List[torch.Tensor] = []
        for field in self.task_obj_fields:
            obj_pose7 = ep["obj_poses"][field]  # (T+1, 7)
            obj_T_list.append(
                torch.as_tensor(_pose7_to_mat4_np(obj_pose7), dtype=torch.float32)
            )  # (T+1, 4, 4)

        # --- Build prompt ---
        prompt = PromptV0(
            P=0,
            T=self.max_T,
            C=self.max_C,
            PCL=self.num_pts,
            PCL_dim=6,
        )

        # --- Objects group ---
        object_gid = prompt.max_group_id + 1
        for i, field in enumerate(self.task_obj_fields):
            obj_T = obj_T_list[i]  # (T+1, 4, 4)
            psid = _TASK_OBJ_PSIDS[self.task_name][i]

            _pcl = self._sample_object_pcl(psid)  # (num_pts, 6) — per-link uint8 color preferred
            _T_trace = obj_T[start_abs_tid:end_abs_tid][::factor]  # (t_sliced, 4, 4)
            # Context is the ABSOLUTE terminal frame of the episode (the
            # success-state pose per schema-v3), NOT the last frame of the
            # current slice. Matches v13 (maniskill_oracle_dataset.py:350).
            _T_ctx = obj_T[-1:].clone()                             # (1, 4, 4)
            t_sliced = _T_trace.shape[0]

            prompt.append_trace(
                trace=_T_trace[None],                              # (1, t_sliced, 4, 4)
                trace_mask=torch.ones(1, t_sliced),
                context=_T_ctx[None],                              # (1, 1, 4, 4)
                context_mask=torch.ones(1, 1),
                group_id=object_gid,
                sem_type=SEM_OBJECT,
                uid=self.task_obj_uids[i],
                local_pcl=torch.as_tensor(_pcl[None], dtype=torch.float32),  # (1, num_pts, 6)
                local_pcl_mask=torch.ones(1),
            )

        # --- Actor group ---
        actor_gid = prompt.max_group_id + 1

        # actor_trace_full: (6, T+1, 4, 4)
        actor_trace_full = torch.stack([
            palm_T_wl,
            left_finger_T_wl,
            right_finger_T_wl,
            palm_act_T_wl,
            left_act_T_wl,
            right_act_T_wl,
        ], dim=0)  # (6, T+1, 4, 4)

        # Actor context = ABSOLUTE terminal frame of the T+1 array (success
        # state), NOT the last frame of the sliced window. Matches v13
        # (maniskill_oracle_dataset.py:422).
        actor_ctx = actor_trace_full[:, -1:].clone()                              # (6, 1, 4, 4)
        actor_trace = actor_trace_full[:, start_abs_tid:end_abs_tid][:, ::factor] # (6, t_sliced, 4, 4)
        actor_t_sliced = actor_trace.shape[1]

        # PCL per actor part: palm/L/R real psids; act parts reuse same pcl
        finger_psids = _TASK_FINGER_PSIDS[self.task_name]
        actor_psids = [
            _PALM_PSID,
            finger_psids["left"],
            finger_psids["right"],
        ]
        actor_pcl_list = []
        for psid in actor_psids:
            _pcl = self._sample_robot_pcl(psid)  # (num_pts, 6) — v13 robot convention: flat 0.5 via _pcl_c
            actor_pcl_list.append(_pcl)
        # act parts duplicate same pcl as real parts (per v13 precedent)
        actor_pcl = np.stack(actor_pcl_list + actor_pcl_list, axis=0)  # (6, num_pts, 6)
        actor_pcl_t = torch.as_tensor(actor_pcl, dtype=torch.float32)

        prompt.append_trace(
            trace=actor_trace,                                         # (6, t_sliced, 4, 4)
            trace_mask=torch.ones(6, actor_t_sliced),
            context=actor_ctx,                                         # (6, 1, 4, 4)
            context_mask=torch.ones(6, 1),
            group_id=actor_gid,
            sem_type=[
                SEM_MANISKILL_TABLETOP_EE,
                SEM_MANISKILL_TABLETOP_LEFT_FINGER,
                SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
                SEM_MANISKILL_TABLETOP_ACT,
                SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
                SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
            ],
            uid=[
                UID_MANISKILL_TABLETOP_EE,
                UID_MANISKILL_TABLETOP_LEFT_FINGER,
                UID_MANISKILL_TABLETOP_RIGHT_FINGER,
                UID_MANISKILL_TABLETOP_ACT,
                UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
                UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
            ],
            local_pcl=actor_pcl_t,                                     # (6, num_pts, 6)
            local_pcl_mask=torch.ones(6),
        )

        # --- Finalise prompt ---
        prompt.catch_P()   # no-op with P=0
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uids = (
            prompt.cat()
        )
        local_pcl, local_pcl_mask = prompt.cat_pcl()

        # --- Time stamps ---
        time_stamp = (torch.arange(self.max_T) - current_ith) * (
            float(self.t_rate) / float(self.base_fps)
        )

        # --- Mask scheduler ---
        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask,
            context_mask=context_mask,
            sem_type=sem_type,
            current_ith=current_ith,
        )

        # --- Text ---
        text = self.task_name

        # --- Extra dict ---
        extra_dict: Dict = {
            "task": self.task_name,
            "action_dim": self.action_dim,
            "task_obj_fields": self.task_obj_fields,
            # scene_params: verbatim from episode (PegIns has 3 keys, others empty)
            "scene_params": ep["scene_params"],
        }

        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace,                         # (P, max_T, 4, 4)
            trace_mask=trace_mask,               # (P, max_T)
            context=context,                     # (P, 1, 4, 4)
            context_mask=context_mask,           # (P, 1)
            known_mask=known_mask,               # (P, max_T)
            context_known_mask=ctx_known_mask,   # (P, 1)
            trace_group_id=group_id,             # (P,)
            trace_rot_mask=rot_mask,             # (P,)
            trace_sem_type=sem_type,             # (P,)
            trace_uid=uids,                      # (P,)
            time=time_stamp,                     # (max_T,)
            text=text,
            local_pcl=local_pcl,                 # (P, num_pts, 6)
            local_pcl_mask=local_pcl_mask,       # (P,)
        )
        return ret, extra_dict
