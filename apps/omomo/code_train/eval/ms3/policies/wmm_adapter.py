"""WmmAdapter — bridges UMM's wmm PLModel (AgentV11) to the MS3 evaluator.

Wraps a live ``pl_module`` (during training, via
:class:`eval.ms3.evaluator.MS3EvalCallback`). Checkpoint auto-load is not
implemented in v1 — upstream callers must hand in a constructed ``PLModel``.

obs_dict keys consumed
----------------------
* ``step`` (int): step counter since last reset; ``step == 0`` triggers
  a per-env reset (history cleared + seeded).
* ``robot.qpos`` ``(9,) float32``: Panda 7 arm + 2 gripper joints;
  forwarded through ``lib_mani.utils.panda_fk.PandaFK`` to recover
  left/right finger SE(3) poses.
* ``robot.ee_pose`` ``(4, 4) float32``: palm SE(3) in world frame.
* ``objects.<name>.pose`` ``(4, 4) float32``: per-task object SE(3)
  poses, where ``<name>`` matches
  ``prepare_data/ms3/schema.py::TASK_OBJECT_FIELDS`` entries with the
  trailing ``_pose`` stripped (``"obj_pose"`` → ``"obj"``).

Action output contract
----------------------
Returns ``np.ndarray`` of shape ``(num_envs, task_action_dim)`` with
dtype ``float32``. Each action is clipped to ``[-1, 1]``. For the 4-DoF
control modes (PushCube / PickCube / StackCube) the returned 4 entries
are ``[dx, dy, dz, gripper]``, sliced from the 7-DoF UMM output via
``ACTION_4DOF_SLICE``. For 7-DoF (PegInsertionSide) all 7 entries
pass through.

Construction patterns (v1)
--------------------------
1. ``WmmAdapter(ckpt_path=None, task=..., pl_module=...)`` — live-module
   wrap. Used by :class:`MS3EvalCallback` via ``policy_factory``.
2. :pymeth:`from_pl_module` — classmethod equivalent of (1).
3. Ckpt auto-load (``ckpt_path != None``) raises
   :class:`NotImplementedError`. v13.8 config convention for wmm is
   not yet finalised; construct the ``PLModel`` externally.

Buffering strategy
------------------
Inference runs in chunks of ``ACTION_CHUNK`` (10) steps. Each env
maintains its own history ring buffer (``HISTORY_LEN`` entries) and
action deque. On ``step == 0`` the env is reset: history cleared and
seeded with 5 copies of the current frame so the model sees a full
window immediately. Subsequent calls append the current frame and
pop one action; when the action deque empties we re-run inference.

Reset detection
---------------
Every call checks ``obs["step"] == 0`` per env. This mirrors the MS3
evaluator reset convention (bc.py / evaluator.py ``ep_step`` resets to
0 on ``env.reset()``).
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from eval.ms3.policies.base import MS3PolicyAdapter
from eval.ms3.task_protocol import task_action_dim
from lib_io import UnifiedIO
from lib_mani.utils.action_utils import tgt_to_state_action
from lib_mani.utils.panda_fk import PandaFK

from prepare_data.ms3.schema import TASK_OBJECT_FIELDS
from umm.datasets.wmm.dataset_config import (
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinned constants (single source of truth for this adapter)
# ---------------------------------------------------------------------------

#: Slice of the 7-DoF UMM action that survives for 4-DoF tasks.
#: Layout: [dx, dy, dz, drx, dry, drz, grip] → keep pos + grip.
ACTION_4DOF_SLICE = (0, 1, 2, 6)

#: Per-step translation limit used by ``state_action_to_tgt`` /
#: ``tgt_to_state_action``. Matches v13 ManiskillOracleDataset default.
POS_LIMIT = 0.1

#: Per-step rotation limit (axis-angle, radians).
ROT_LIMIT = 0.1

#: Number of past frames stored in the rolling history buffer and fed
#: to the model as "known" trace tokens.
HISTORY_LEN = 5

#: Number of future actions we extract from each inference call before
#: the action deque is exhausted and we re-run inference.
ACTION_CHUNK = 10

#: Number of denoising steps used during each inference call.
INFERENCE_N_STEPS = 10

#: Full model window (history + future). Matches v13.8 config
#: ``dataset.configs[*].max_T`` and ``model.*.window``.
WINDOW_T = 20

#: MS3 native simulation FPS (Hz). Must match ``Ms3StateDataset.base_fps``
#: so ``real_timestep`` at deployment aligns with the distribution the
#: model saw during training. ``src/umm/datasets/wmm/ms3_state_dataset.py``
#: defaults this to 20.0. The adapter asserts the dataset config matches
#: at ``__init__`` time.
BASE_FPS = 20.0

#: Dataset temporal stride. Matches the ``t_rate`` field on every
#: wmm ms3 dataset config.
T_RATE = 1


# ---------------------------------------------------------------------------
# Per-task local_uid → psid (Panda link id) mapping for robot parts.
# ---------------------------------------------------------------------------
# The palm is always psid 10; fingers are per-task (see _TASK_FINGER_PSIDS
# in ms3_state_dataset.py — PushCube/PickCube use {12, 14}, StackCube /
# PegInsertionSide use {12, 13}).  The adapter mirrors those values here.
_TASK_FINGER_PSIDS: Dict[str, Dict[str, int]] = {
    "PushCube-v1":         {"left": 12, "right": 14},
    "PickCube-v1":         {"left": 12, "right": 14},
    "StackCube-v1":        {"left": 12, "right": 13},
    "PegInsertionSide-v1": {"left": 12, "right": 13},
}

#: Palm psid in robot_embodiment.npz (``panda_hand``).
_PALM_PSID: int = 10

#: Object psids per task — matches _TASK_OBJ_PSIDS in ms3_state_dataset.py.
_TASK_OBJ_PSIDS: Dict[str, List[int]] = {
    "PushCube-v1":         [18],
    "PickCube-v1":         [18],
    "StackCube-v1":        [18, 19],
    "PegInsertionSide-v1": [18, 19],
}

#: Per-(task, field) obs_dict key override. Most ``TASK_OBJECT_FIELDS``
#: entries go to ``objects.<stripped>.pose`` (MS3 evaluator kind=obj_pose
#: — see ``eval/ms3/evaluator.py::unflatten_raw_state`` lines 346-357).
#: PegInsertionSide-v1's ``box_hole_pose`` is routed to
#: ``goal.box_hole_pose`` by the evaluator (kind=box_hole_pose_goal,
#: see lines 358-360). The adapter must follow that routing at runtime
#: or the first real PegIns obs will raise KeyError.
_OBJ_FIELD_OBS_KEY_OVERRIDE: Dict[tuple, str] = {
    ("PegInsertionSide-v1", "box_hole_pose"): "goal.box_hole_pose",
}


def _resolve_obs_key_for_field(task: str, field: str) -> str:
    """Map a ``TASK_OBJECT_FIELDS[task]`` entry to its runtime obs key.

    The MS3 evaluator's ``unflatten_raw_state`` routes fields based on
    ``TaskSpec.kind``. Most ``obj_pose`` fields become
    ``objects.<stripped>.pose``; ``PegInsertionSide-v1.box_hole_pose`` is
    the one deviation (routed to ``goal.box_hole_pose``). Overrides live
    in :data:`_OBJ_FIELD_OBS_KEY_OVERRIDE`.

    Args:
        task: MS3 task id (e.g. ``"PegInsertionSide-v1"``).
        field: Entry from ``TASK_OBJECT_FIELDS[task]`` (e.g. ``"peg_pose"``
            or ``"box_hole_pose"``). Must end in ``_pose``.

    Returns:
        The canonical obs_dict key that the MS3 evaluator will populate.
    """
    override = _OBJ_FIELD_OBS_KEY_OVERRIDE.get((task, field))
    if override is not None:
        return override
    if not field.endswith("_pose"):
        raise ValueError(
            f"_resolve_obs_key_for_field: field {field!r} for task {task!r} "
            f"does not end in '_pose'; cannot derive obs key. Extend "
            f"_OBJ_FIELD_OBS_KEY_OVERRIDE if routing differs."
        )
    name = field[: -len("_pose")]
    return f"objects.{name}.pose"


class WmmAdapter(MS3PolicyAdapter):
    """MS3 adapter for a UMM ``wmm`` (v13-style) ``PLModel``.

    Args:
        ckpt_path: Not supported in v1 (auto-load from ckpt is not yet
            implemented — wmm config convention is not finalised). If
            non-None and ``pl_module`` is None, raises
            :class:`NotImplementedError`.
        task: MS3 task id, e.g. ``"PushCube-v1"``.
        pl_module: The live ``umm.models.wmm.model.PLModel``. Required for
            the in-training / live-wrap construction path.
        cfg_path: Unused in v1 — reserved for future ckpt-auto-load.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        task: str,
        pl_module: Any = None,
        cfg_path: Optional[str] = None,
    ):
        super().__init__(ckpt_path=ckpt_path, task=task)

        if pl_module is None:
            if ckpt_path is not None:
                raise NotImplementedError(
                    "WmmAdapter ckpt auto-load not yet implemented; "
                    "pass pl_module=..."
                )
            raise ValueError(
                "WmmAdapter: pl_module is required for the live-wrap path. "
                "Either pass pl_module=... (live-module) or construct the "
                "PLModel externally and hand it in. Auto-load from ckpt is "
                "not yet implemented."
            )

        # wmm PLModel exposes .model (AgentV11), NOT .agent.
        if not hasattr(pl_module, "model"):
            raise AttributeError(
                "WmmAdapter: pl_module has no 'model' attribute. Expected "
                "umm.models.wmm.model.PLModel with attribute `model` "
                "(umm.models.wmm.agent.AgentV11). Check that pl_module is a "
                "wmm PLModel and not an older variant exposing `.agent`."
            )

        self.pl_module = pl_module
        self._agent = pl_module.model
        # Respect caller's device placement — do NOT auto-migrate.
        self._device: torch.device = torch.device(pl_module.device)

        # Compute per-task action dimension (4 for pd_ee_delta_pos,
        # 7 for pd_ee_delta_pose). Raises KeyError for unknown tasks.
        self._task_action_dim: int = task_action_dim(task)

        # PandaFK cache — built once. qpos column order matches schema-v3.
        self._fk: PandaFK = PandaFK()

        # --- Load embodiment npzs from cfg. ---
        # wmm PLModel cfg layout (see configs/v13.8/*.yaml + ms3_state block):
        #   cfg["model"]["ms3_state"]["model_path"]   robot_embodiment.npz
        #   cfg["model"]["ms3_state"]["object_path"]  object_embodiment.npz
        model_cfg = pl_module.cfg.get("model", {})
        ms3_state_cfg = model_cfg.get("ms3_state", {})
        robot_fn = ms3_state_cfg.get("model_path")
        object_fn = ms3_state_cfg.get("object_path")
        if not robot_fn or not object_fn:
            raise KeyError(
                "WmmAdapter: cfg['model']['ms3_state'] must provide "
                "'model_path' (robot_embodiment.npz) and 'object_path' "
                "(object_embodiment.npz) for task "
                f"{task!r}. Got model_path={robot_fn!r}, object_path={object_fn!r}."
            )
        self._num_pts: int = int(ms3_state_cfg.get("num_pts", 256))

        self._io = UnifiedIO()
        self._robot_link_data: dict = self._io.load_npzs(robot_fn)
        self._object_link_data: dict = self._io.load_npzs(object_fn)

        # --- Per-task static state. ---
        if task not in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP:
            raise KeyError(
                f"WmmAdapter: task {task!r} not in "
                f"MANISKILL_TABLETOP_TASK_OBJ_UID_MAP."
            )
        if task not in TASK_OBJECT_FIELDS:
            raise KeyError(
                f"WmmAdapter: task {task!r} not in TASK_OBJECT_FIELDS."
            )
        if task not in _TASK_FINGER_PSIDS:
            raise KeyError(
                f"WmmAdapter: task {task!r} not in _TASK_FINGER_PSIDS. "
                "Extend the per-task finger psid map."
            )
        if task not in _TASK_OBJ_PSIDS:
            raise KeyError(
                f"WmmAdapter: task {task!r} not in _TASK_OBJ_PSIDS. "
                "Extend the per-task object psid map."
            )

        self._task_obj_uids: List[int] = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[task]
        self._task_object_fields: List[str] = TASK_OBJECT_FIELDS[task]
        self._task_obj_psids: List[int] = _TASK_OBJ_PSIDS[task]
        self._finger_psids: Dict[str, int] = _TASK_FINGER_PSIDS[task]
        #: Number of object slots this task exposes (drives part ordering).
        self._num_objects: int = len(self._task_object_fields)
        #: Pre-resolved obs key per TASK_OBJECT_FIELDS entry, respecting
        #: the MS3 evaluator's ``unflatten_raw_state`` routing. Most fields
        #: map to ``objects.<stripped>.pose`` (kind=obj_pose). PegInsertion-
        #: Side-v1 ``box_hole_pose`` goes to ``goal.box_hole_pose`` (kind=
        #: box_hole_pose_goal). See eval/ms3/evaluator.py:234-244, 346-360.
        self._task_obj_obs_keys: List[str] = [
            _resolve_obs_key_for_field(task, field)
            for field in self._task_object_fields
        ]

        # --- Pre-sample per-slot PCLs (one-time cost). ---
        # Objects: one per TASK_OBJECT_FIELDS entry (psid 18, 19, ...).
        self._obj_pcls_by_local_idx: Dict[int, np.ndarray] = {}
        for i, psid in enumerate(self._task_obj_psids):
            self._obj_pcls_by_local_idx[i] = self._load_pcl(
                self._object_link_data, psid
            )

        # Robot state PCLs: palm / L / R — reused for act slots.
        self._palm_pcl = self._load_pcl(self._robot_link_data, _PALM_PSID)
        self._left_pcl = self._load_pcl(
            self._robot_link_data, self._finger_psids["left"]
        )
        self._right_pcl = self._load_pcl(
            self._robot_link_data, self._finger_psids["right"]
        )

        # --- Per-env runtime buffers. ---
        # History: list of frame dicts (palm / L / R / objects).
        self._history_buf: Dict[int, List[Dict[str, Any]]] = {}
        # Action deque: popleft on every __call__, replenished in chunks.
        self._action_buf: Dict[int, "collections.deque[np.ndarray]"] = {}

        # --- Hallucinated-goal snapshot for SBS viz (SG-R) ---
        # Stores the most recent ret_context[0, 0] for env 0 only. Shape
        # (P, 4, 4), CPU float32. Consumed by MS3Evaluator when SBS viz is
        # enabled. Non-None only after the first inference call for env 0;
        # cleared by _reset_env(env_idx=0).
        self.last_hallucinated_context: Optional[torch.Tensor] = None

        # --- Assert dataset-config base_fps / t_rate match deployment ---
        # ``real_timestep`` at deployment is ``(arange(T) - 4) * (t_rate /
        # base_fps)``; the LMM was trained on that exact distribution. If
        # (or future configs) change either value, the adapter must
        # be updated in lock-step or the time-conditioning will silently
        # drift. Fail loud at construction rather than in eval logs.
        dataset_cfg_list = pl_module.cfg.get("dataset", {}).get("configs", [])
        for idx, dcfg in enumerate(dataset_cfg_list):
            if dcfg.get("type") != "ms3_state":
                continue
            ds_base_fps = float(dcfg.get("base_fps", BASE_FPS))
            ds_t_rate = int(dcfg.get("t_rate", T_RATE))
            if abs(ds_base_fps - BASE_FPS) > 1e-6 or ds_t_rate != T_RATE:
                raise ValueError(
                    f"WmmAdapter: dataset.configs[{idx}] has "
                    f"base_fps={ds_base_fps}, t_rate={ds_t_rate}; adapter "
                    f"hardcodes BASE_FPS={BASE_FPS}, T_RATE={T_RATE}. "
                    f"Either update the config to match or update the "
                    f"adapter module constants (and re-train)."
                )

        logger.info(
            "WmmAdapter: wrapped pl_module on device=%s for task=%s "
            "(task_action_dim=%d, num_objects=%d, robot_link_keys=%d, "
            "object_link_keys=%d, num_pts=%d)",
            self._device, task, self._task_action_dim, self._num_objects,
            len(self._robot_link_data), len(self._object_link_data),
            self._num_pts,
        )

    # ------------------------------------------------------------------
    # Factory classmethod
    # ------------------------------------------------------------------
    @classmethod
    def from_pl_module(cls, pl_module: Any, task: str) -> "WmmAdapter":
        """Wrap a live ``PLModel`` for in-training MS3 eval.

        Args:
            pl_module: Live :class:`umm.models.wmm.model.PLModel`.
            task: MS3 task id.

        Returns:
            A new :class:`WmmAdapter` sharing ``pl_module``'s weights.
        """
        return cls(ckpt_path=None, task=task, pl_module=pl_module)

    # ------------------------------------------------------------------
    # PCL helper (matches Ms3StateDataset._load_pcl semantics)
    # ------------------------------------------------------------------
    def _load_pcl(self, link_data: dict, psid: int) -> np.ndarray:
        """Load and down-sample the (num_pts, 6) PCL for ``psid``.

        Uses deterministic first-``num_pts`` sampling (no randomness).

        Args:
            link_data: Parsed npz dict.
            psid: SAPIEN per-scene-id of the link or object.

        Returns:
            ``(num_pts, 6)`` float32 numpy array (xyz + rgb in [0, 1]).
        """
        pcl_xyz = link_data[f"{psid}_pcl"]  # (N, 3)
        key_c = f"{psid}_c"
        key_pcl_c = f"{psid}_pcl_c"
        if key_c in link_data:
            c = link_data[key_c].astype(np.float32)
            if c.max() > 1.0:
                c = c / 255.0
            if c.ndim == 1:
                pcl_c = np.tile(c[None, :], (pcl_xyz.shape[0], 1))
            else:
                pcl_c = c
        elif key_pcl_c in link_data:
            pcl_c = link_data[key_pcl_c].astype(np.float32)
            if pcl_c.max() > 1.0:
                pcl_c = pcl_c / 255.0
        else:
            raise ValueError(
                f"psid {psid}: missing both {key_c!r} and {key_pcl_c!r} "
                "in link_data"
            )
        pcl = np.concatenate([pcl_xyz, pcl_c], axis=1).astype(np.float32)
        # Deterministic first-num_pts: tile if too few, take first N if enough.
        N = pcl.shape[0]
        if N >= self._num_pts:
            return pcl[: self._num_pts]
        repeat_times = (self._num_pts + N - 1) // N
        tiled = np.tile(pcl, (repeat_times, 1))
        return tiled[: self._num_pts]

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _build_frame_from_obs(self, obs: dict) -> Dict[str, Any]:
        """Build a single-frame dict (palm / L / R / objects) from an obs.

        Args:
            obs: One env's obs dict (see adapter obs_dict_spec / base.py).

        Returns:
            Dict with keys ``palm`` / ``L`` / ``R`` — each ``(4, 4)`` float
            tensors on CPU — and ``objects`` — dict keyed by local_uid
            (int, index into ``TASK_OBJECT_FIELDS[task]``) with ``(4, 4)``
            float tensors.
        """
        # --- Palm from tcp_pose (already (4, 4) per evaluator.py:340), WORLD. ---
        palm_np = np.asarray(obs["robot.ee_pose"], dtype=np.float32)
        palm = torch.as_tensor(palm_np, dtype=torch.float32)

        # --- Finger poses via PandaFK, lifted into the SAME WORLD frame as
        # ``palm``.  FK returns finger poses in the ``panda_link0`` (robot
        # root) frame; ManiSkill3 tabletop scenes mount the robot at a fixed
        # world offset (typically ``[-0.615, 0, 0]``) so the raw FK finger
        # poses sit ~0.6 m off from ``palm``.  Must match the training-data
        # convention applied in
        # ``src/umm/datasets/wmm/ms3_state_dataset.py::__getitem_inner__`` —
        # otherwise the model sees one frame semantic in training and
        # another at deployment, producing silently wrong rollouts.
        #
        # Recover ``T_base_world = palm @ inv(FK.panda_hand_tcp)`` from the
        # TCP identity (FK exposes ``panda_hand_tcp`` as an auxiliary link
        # matching the MS3 ``tcp_pose`` semantic within 1e-7).
        qpos_np = np.asarray(obs["robot.qpos"], dtype=np.float32)
        qpos_t = torch.as_tensor(qpos_np, dtype=torch.float32).unsqueeze(0)  # (1, 9)
        fk_out = self._fk.forward(qpos_t)
        tcp_in_base = fk_out["panda_hand_tcp"][0]                 # (4, 4)
        T_base_world = palm @ torch.linalg.inv(tcp_in_base)        # (4, 4)
        left = T_base_world @ fk_out["panda_leftfinger"][0]        # (4, 4)
        right = T_base_world @ fk_out["panda_rightfinger"][0]      # (4, 4)

        # --- Objects: look up via pre-computed TASK_OBJECT_FIELDS → obs_key
        # map. The MS3 evaluator's ``unflatten_raw_state`` routes fields by
        # ``TaskSpec.kind``: most go to ``objects.<stripped>.pose``, but
        # PegInsertionSide-v1's ``box_hole_pose`` is routed to
        # ``goal.box_hole_pose`` (kind=box_hole_pose_goal). See
        # ``eval/ms3/evaluator.py`` lines 234-244 (PEG_INSERTION_SIDE_SPEC)
        # and 346-360 (unflatten_raw_state routing). We pre-resolve this map
        # once in __init__ (self._task_obj_obs_keys) and just look it up here.
        objects: Dict[int, torch.Tensor] = {}
        for local_idx, obs_key in enumerate(self._task_obj_obs_keys):
            if obs_key not in obs:
                raise KeyError(
                    f"WmmAdapter: expected obs key {obs_key!r} for task "
                    f"{self.task!r} (field "
                    f"{self._task_object_fields[local_idx]!r}) not found. "
                    f"Available keys starting with 'objects.': "
                    f"{[k for k in obs if k.startswith('objects.')]} "
                    f"| starting with 'goal.': "
                    f"{[k for k in obs if k.startswith('goal.')]}"
                )
            pose_np = np.asarray(obs[obs_key], dtype=np.float32)
            # Verify that the local_idx's psid has a cached pcl.
            if local_idx not in self._obj_pcls_by_local_idx:
                raise KeyError(
                    f"WmmAdapter: no cached PCL for local_idx={local_idx} "
                    f"(field {self._task_object_fields[local_idx]!r}, psid "
                    f"{self._task_obj_psids[local_idx]}). Check "
                    f"object_embodiment.npz."
                )
            objects[local_idx] = torch.as_tensor(pose_np, dtype=torch.float32)

        return {
            "palm": palm.contiguous(),
            "L": left.contiguous(),
            "R": right.contiguous(),
            "objects": objects,
        }

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _reset_env(self, env_idx: int, obs: dict) -> None:
        """Clear per-env buffers and seed history with the current frame.

        Seeds HISTORY_LEN copies of the first frame so the model sees
        a full past window immediately (deployment convention — matches
        maniskill_evaluator.py).
        """
        frame = self._build_frame_from_obs(obs)
        # Use a list comprehension (NOT ``[frame] * HISTORY_LEN``) so the
        # HISTORY_LEN slots reference the same dict — that's fine for
        # read-only history but would become a silent corruption footgun if
        # any downstream op ever mutated a frame in place.
        self._history_buf[env_idx] = [frame for _ in range(HISTORY_LEN)]
        # Unbounded deque: replenishment is explicitly gated by the
        # ``if not self._action_buf.get(env_idx):`` check in ``__call__``,
        # so we never legitimately overfill. ``maxlen=ACTION_CHUNK`` would
        # silently drop the leftmost action on any accidental double-
        # replenishment — a trap with no upside here.
        self._action_buf[env_idx] = collections.deque()
        # Clear hallucinated-goal snapshot on env-0 reset so the next eval
        # cycle starts clean.
        if env_idx == 0:
            self.last_hallucinated_context = None

    def _push_history(self, env_idx: int, obs: dict) -> None:
        """Append current frame to env history; drop oldest if over capacity."""
        if env_idx not in self._history_buf:
            # Defensive — shouldn't happen if the evaluator honours step==0
            # reset, but be explicit rather than silently producing a short
            # history.
            self._reset_env(env_idx, obs)
            return
        frame = self._build_frame_from_obs(obs)
        self._history_buf[env_idx].append(frame)
        if len(self._history_buf[env_idx]) > HISTORY_LEN:
            self._history_buf[env_idx] = self._history_buf[env_idx][-HISTORY_LEN:]

    # ------------------------------------------------------------------
    # Inference batch builder (single-env, B = 1)
    # ------------------------------------------------------------------
    def _build_inference_inputs(self, env_idx: int) -> Dict[str, torch.Tensor]:
        """Assemble the single-env (B=1) batch for ``self._agent.inference``.

        Returns a dict of CPU tensors with keys matching AgentV11.inference
        kwargs. Caller is responsible for moving them to the model device.
        """
        history = self._history_buf[env_idx]
        if len(history) != HISTORY_LEN:
            raise RuntimeError(
                f"WmmAdapter: history for env {env_idx} has {len(history)} "
                f"frames, expected {HISTORY_LEN}. Buffer management bug."
            )

        num_obj = self._num_objects
        P = num_obj + 6  # objects + palm/L/R + palm_act/L_act/R_act

        # --- Build trace: (1, WINDOW_T, P, 4, 4), history[:5] + eye4 padding ---
        # Initialise as eye(4) so padding for t in [5..19] is the identity.
        trace = (
            torch.eye(4, dtype=torch.float32)
            .view(1, 1, 1, 4, 4)
            .expand(1, WINDOW_T, P, 4, 4)
            .clone()
        )
        # Fill the history portion.
        for t_idx, frame in enumerate(history):  # 0..4
            palm_T = frame["palm"]
            left_T = frame["L"]
            right_T = frame["R"]
            # Object slots (0..num_obj-1)
            for local_idx, obj_T in frame["objects"].items():
                if local_idx >= num_obj:
                    raise ValueError(
                        f"WmmAdapter: frame object local_idx={local_idx} "
                        f">= num_objects={num_obj}"
                    )
                trace[0, t_idx, local_idx] = obj_T
            # Actor state slots: palm, L, R
            trace[0, t_idx, num_obj + 0] = palm_T
            trace[0, t_idx, num_obj + 1] = left_T
            trace[0, t_idx, num_obj + 2] = right_T
            # Actor act slots (history): copy of state (deployment convention,
            # matches maniskill_evaluator.py:606-613).
            trace[0, t_idx, num_obj + 3] = palm_T.clone()
            trace[0, t_idx, num_obj + 4] = left_T.clone()
            trace[0, t_idx, num_obj + 5] = right_T.clone()

        # --- Masks. ---
        # trace_known_mask: history True, future False.
        trace_known_mask = torch.zeros(1, WINDOW_T, P, dtype=torch.bool)
        trace_known_mask[:, :HISTORY_LEN, :] = True
        # trace_mask: validity — all True (no padding entities).
        trace_mask = torch.ones(1, WINDOW_T, P, dtype=torch.bool)

        # --- Context: (1, 1, P, 4, 4) eye(4), context_known_mask = zeros. ---
        # Training context is the TERMINAL episode frame (goal anchor;
        # ``ms3_state_dataset.py:511,539``) with ``ctx_known_mask ≈ ones``
        # under ``MS_PATTERNS``. At deployment we DO NOT have the terminal
        # frame — the safe canonical v13 recipe (used by
        # ``src/umm/sims/maniskill_evaluator.py:727-731``) is:
        #     context = eye(4) everywhere
        #     context_mask = True (padding valid)
        #     context_known_mask = False  (goal unknown; let model generate)
        # Seeding context with the CURRENT state + ctx_known=True would
        # silently signal "goal == current pose", a distributional error.
        context = (
            torch.eye(4, dtype=torch.float32)
            .view(1, 1, 1, 4, 4)
            .expand(1, 1, P, 4, 4)
            .clone()
        )
        context_mask = torch.ones(1, 1, P, dtype=torch.bool)
        context_known_mask = torch.zeros(1, 1, P, dtype=torch.bool)

        # --- real_timestep: (1, WINDOW_T) float. ---
        # Training uses ``(arange(max_T) - current_ith) * (t_rate/base_fps)``
        # (``src/umm/datasets/wmm/ms3_state_dataset.py:592-594``). At
        # deployment we align ``current_ith = HISTORY_LEN - 1 = 4`` so
        # history is non-positive, future positive. ``BASE_FPS`` and
        # ``T_RATE`` are pinned module constants; ``__init__`` asserts the
        # dataset config matches so mismatches fail loud.
        time_unit = float(T_RATE) / BASE_FPS
        real_timestep = (
            torch.arange(WINDOW_T, dtype=torch.float32) - (HISTORY_LEN - 1)
        ) * time_unit
        real_timestep = real_timestep.unsqueeze(0)  # (1, WINDOW_T)

        # --- group_id: (1, P). Matches dataset convention exactly. ---
        # ``ms3_state_dataset.py::__add_maniskill_prompt__`` uses
        # ``PromptV0.max_group_id + 1`` which starts at ``-1`` for an empty
        # prompt (see ``base_dataset.py::max_group_id``). First call (objects)
        # yields ``gid=0``; second call (actor group) yields ``gid=1``.
        # We MUST match that convention exactly — the group embedding table
        # was trained with {0, 1} only; using {1, 2} would look up unseen
        # slots and silently degrade the policy.
        group_id = torch.zeros(1, P, dtype=torch.long)
        group_id[0, :num_obj] = 0
        group_id[0, num_obj:] = 1

        # --- sem_type: (1, P). ---
        sem_list: List[int] = [SEM_OBJECT] * num_obj + [
            SEM_MANISKILL_TABLETOP_EE,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
            SEM_MANISKILL_TABLETOP_ACT,
            SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]
        sem_type = torch.tensor([sem_list], dtype=torch.long)  # (1, P)

        # --- rot_mask: all entities have rotation. ---
        rot_mask = torch.ones(1, P, dtype=torch.bool)

        # --- local_pcl: (1, P, num_pts, 6). ---
        pcl_parts: List[np.ndarray] = []
        for local_idx in range(num_obj):
            pcl_parts.append(self._obj_pcls_by_local_idx[local_idx])
        # Actor state + act slots share palm/L/R pcls.
        actor_pcls = [self._palm_pcl, self._left_pcl, self._right_pcl]
        pcl_parts.extend(actor_pcls)        # state palm/L/R
        pcl_parts.extend(actor_pcls)        # act palm/L/R
        local_pcl_np = np.stack(pcl_parts, axis=0)  # (P, num_pts, 6)
        local_pcl = torch.as_tensor(local_pcl_np, dtype=torch.float32).unsqueeze(0)
        local_pcl_mask = torch.ones(1, P, dtype=torch.bool)

        return {
            "trace": trace,
            "context": context,
            "trace_known_mask": trace_known_mask,
            "context_known_mask": context_known_mask,
            "trace_mask": trace_mask,
            "context_mask": context_mask,
            "real_timestep": real_timestep,
            "group_id": group_id,
            "sem_type": sem_type,
            "rot_mask": rot_mask,
            "local_pcl": local_pcl,
            "local_pcl_mask": local_pcl_mask,
        }

    # ------------------------------------------------------------------
    # Inference + action decode
    # ------------------------------------------------------------------
    def _replenish_action_buf(self, env_idx: int) -> None:
        """Run inference for env ``env_idx`` and fill its action deque."""
        inputs = self._build_inference_inputs(env_idx)
        # Move tensors to model device.
        device = self._device
        moved = {k: v.to(device) for k, v in inputs.items()}

        # NOTE: do NOT call ``self._agent.eval()`` here. ``agent.inference()``
        # saves its own ``self.training`` flag at entry and restores it at
        # exit (see ``agent.py::inference`` lines 1953-1955 / 2306-2307).
        # An external ``.eval()`` would flip the flag to ``False`` BEFORE
        # ``inference()`` reads it, causing the agent to stay in eval mode
        # after we return — breaking BN/dropout in subsequent training steps
        # within the same epoch.
        with torch.no_grad():
            ret_context, ret_trace = self._agent.inference(
                trace=moved["trace"],
                context=moved["context"],
                trace_known_mask=moved["trace_known_mask"],
                context_known_mask=moved["context_known_mask"],
                trace_mask=moved["trace_mask"],
                context_mask=moved["context_mask"],
                real_timestep=moved["real_timestep"],
                group_id=moved["group_id"],
                sem_type=moved["sem_type"],
                rot_mask=moved["rot_mask"],
                local_pcl=moved["local_pcl"],
                local_pcl_mask=moved["local_pcl_mask"],
                text=[self.task],
                seed=42 + env_idx,
                n_steps=INFERENCE_N_STEPS,
                cfg_scale=1.0,
                verbose=False,
            )
        # Snapshot hallucinated goal for env 0 only (SBS viz, SG-R). Shape
        # from agent.inference: (B=1, C=1, P, 4, 4); we store (P, 4, 4).
        if env_idx == 0:
            self.last_hallucinated_context = (
                ret_context[0, 0].detach().cpu().clone()
            )
        # ret_trace: (1, WINDOW_T, P, 4, 4). Future slice = [HISTORY_LEN:HISTORY_LEN+ACTION_CHUNK]
        future = ret_trace[0, HISTORY_LEN : HISTORY_LEN + ACTION_CHUNK]  # (ACTION_CHUNK, P, 4, 4)
        future = future.detach().cpu()

        num_obj = self._num_objects
        P_idx_palm = num_obj + 0
        P_idx_palm_act = num_obj + 3
        P_idx_L_act = num_obj + 4
        P_idx_R_act = num_obj + 5

        # --------------------------------------------------------------
        # Decode pairing — MUST match training convention.
        #
        # Training (``ms3_state_dataset.py:480-484``) encodes:
        #     palm_act[0] = palm[0]                               # (sentinel)
        #     palm_act[k] = state_action_to_tgt(palm[k-1], action[k-1])   for k >= 1
        # So the inverse mapping for action[k] is:
        #     action[k] = tgt_to_state_action(palm[k], palm_act[k+1], ...)
        #
        # Concretely, future starts at t=HISTORY_LEN=5; the last known
        # history state is palm[t=4] = ``history[-1]["palm"]``. We want
        # to emit 10 future actions ``action[t=4..13]``. Each pairing is
        # (palm[t], palm_act[t+1]):
        #     action[t=4]  : palm[4]   = history[-1].palm      , palm_act[5]  = future[0].palm_act
        #     action[t=5]  : palm[5]   = future[0].palm         , palm_act[6]  = future[1].palm_act
        #     ...
        #     action[t=13] : palm[13]  = future[8].palm         , palm_act[14] = future[9].palm_act
        #
        # Pairing same-k state & act (naive) yields action ≈ 0 when the
        # model tracks well (``palm[k] ≈ palm_act[k]``) — robot would be
        # silently stationary at inference. See v13 canonical decode at
        # ``src/umm/sims/maniskill_evaluator.py:639-644``.
        # --------------------------------------------------------------
        history_last_palm = self._history_buf[env_idx][-1]["palm"]       # (4, 4), CPU
        # Curr-palm stack: [history_last.palm, future[0].palm, ..., future[8].palm]
        # Length ACTION_CHUNK; shape (ACTION_CHUNK, 4, 4).
        curr_palm_stack = torch.cat(
            [history_last_palm.unsqueeze(0), future[:-1, P_idx_palm]],
            dim=0,
        )

        deque_buf: "collections.deque[np.ndarray]" = self._action_buf[env_idx]
        for k in range(future.shape[0]):
            curr_palm = curr_palm_stack[k].unsqueeze(0)         # (1, 4, 4)
            tgt_palm = future[k, P_idx_palm_act].unsqueeze(0)
            tgt_L = future[k, P_idx_L_act].unsqueeze(0)
            tgt_R = future[k, P_idx_R_act].unsqueeze(0)

            action_7d = tgt_to_state_action(
                current_palm_pose=curr_palm,
                target_palm_pose=tgt_palm,
                target_left_finger_pose=tgt_L,
                target_right_finger_pose=tgt_R,
                pos_limit=POS_LIMIT,
                rot_limit=ROT_LIMIT,
            ).squeeze(0)  # (7,)

            action_7d_np = action_7d.numpy().astype(np.float32)
            if self._task_action_dim == 4:
                action_out = action_7d_np[list(ACTION_4DOF_SLICE)]
            elif self._task_action_dim == 7:
                action_out = action_7d_np
            else:
                raise RuntimeError(
                    f"WmmAdapter: unexpected task_action_dim="
                    f"{self._task_action_dim} for task {self.task!r}"
                )
            action_out = np.clip(action_out, -1.0, 1.0).astype(np.float32)
            deque_buf.append(action_out)

    # ------------------------------------------------------------------
    # MS3PolicyAdapter contract
    # ------------------------------------------------------------------
    def __call__(self, obs_dict_batch: List[dict]) -> np.ndarray:
        """Map a batch of per-env obs dicts to a numpy action batch.

        Args:
            obs_dict_batch: length-``num_envs`` list of observation dicts.

        Returns:
            ``np.ndarray`` of shape ``(num_envs, task_action_dim)`` with
            dtype ``float32`` and values in ``[-1, 1]``.
        """
        actions_out: List[np.ndarray] = []

        for env_idx, obs in enumerate(obs_dict_batch):
            if obs["step"] == 0:
                self._reset_env(env_idx, obs)
            else:
                self._push_history(env_idx, obs)

            # Replenish if empty (dict.get returns None if missing, deque
            # evaluates falsy when empty — both trigger re-inference).
            if not self._action_buf.get(env_idx):
                self._replenish_action_buf(env_idx)

            action = self._action_buf[env_idx].popleft()
            actions_out.append(action)

        return np.stack(actions_out, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # SBS viz helpers (SG-R) — used by Ms3SbsRolloutRenderer
    # ------------------------------------------------------------------
    def get_viz_local_pcl(self) -> torch.Tensor:
        """Return (P, num_pts, 6) float32 local PCL matching part ordering.

        Mirrors :pymeth:`_build_inference_inputs` local_pcl construction but
        without the batch dimension. Used by Ms3SbsRolloutRenderer so the
        mesh-rendering path can derive per-part colors consistently.
        """
        num_obj = self._num_objects
        parts: List[np.ndarray] = []
        for local_idx in range(num_obj):
            parts.append(self._obj_pcls_by_local_idx[local_idx])
        actor_pcls = [self._palm_pcl, self._left_pcl, self._right_pcl]
        parts.extend(actor_pcls)
        parts.extend(actor_pcls)
        return torch.as_tensor(
            np.stack(parts, axis=0), dtype=torch.float32,
        )

    def get_viz_prompt_types(self) -> torch.Tensor:
        """Return (P,) long tensor of UIDs matching adapter part ordering.

        Ordering: [objects(num_obj), palm, L, R, palm_act, L_act, R_act].
        Only the FIRST ``self._num_objects`` entries of
        ``MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[task]`` are used — this mirrors
        the dataset convention in
        ``src/umm/datasets/wmm/ms3_state_dataset.py`` which assigns
        ``task_obj_uids[i]`` for ``i in range(len(TASK_OBJECT_FIELDS[task]))``
        (see ``ms3_state_dataset.py`` line ~281). Using all 40 UIDs from
        the map would produce a length-(40+6) tensor that does not line
        up with the length-``P`` ``local_pcl`` or the P-shaped
        ``ret_context`` the model actually produces, and would index out
        of bounds inside ``Ms3StateVisualizer.render_viz_frames``.
        """
        from umm.datasets.wmm.dataset_config import (
            MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
            UID_MANISKILL_TABLETOP_EE,
            UID_MANISKILL_TABLETOP_ACT,
            UID_MANISKILL_TABLETOP_LEFT_FINGER,
            UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        )
        all_obj_uids = list(MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[self.task])
        obj_uids = all_obj_uids[: self._num_objects]
        part_uids = obj_uids + [
            UID_MANISKILL_TABLETOP_EE,
            UID_MANISKILL_TABLETOP_LEFT_FINGER,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER,
            UID_MANISKILL_TABLETOP_ACT,
            UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
            UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
        ]
        return torch.as_tensor(part_uids, dtype=torch.long)
