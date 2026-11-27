"""HandObject dataset — canonical bimanual MANO + rigid object interaction.

Per-episode ``.npz`` files written by the text2hoi adapter (canonical schema
defined in :mod:`umm.eval.handobject.canonical`). Mirrors :class:`TacoDataset`'s
windowing and PromptV0 structure but emits 3 entities only: left wrist SE(3),
right wrist SE(3) and object SE(3).

Example::

    dataset = HandObjectDataset(
        data_root="data/handobject/umm/grab",
        split_fn="data/handobject/umm/grab/splits/train.json",
        base_fps=30,
        state_dirname="train",
    )
    sample, extras = dataset[0]
"""

import os.path as osp
import json
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy

from umm.datasets.umm_v1.base_dataset import BaseDataset, PromptV0
from umm.datasets.umm_v1.dataset_template import CanonicalDatasetOutput
from umm.datasets.umm_v1.dataset_config import (
    SEM_LEFT_MANO,
    SEM_RIGHT_MANO,
    SEM_OBJECT,
    UID_MANO_LEFT_HAND,
    UID_MANO_RIGHT_HAND,
    UID_TACO_TOOL,
)
from umm.eval.handobject.canonical import load_canonical_npz


def _axis_angle_to_rotmat(aa):  # (T, 3) float32 -> (T, 3, 3) float32
    """Convert a batch of axis-angle rotations to rotation matrices.

    Args:
        aa: Axis-angle rotations of shape ``(T, 3)``.

    Returns:
        Rotation matrices of shape ``(T, 3, 3)`` as ``float32`` numpy array.
    """
    # R_scipy.from_rotvec handles zero vectors (identity); no special casing needed.
    return R_scipy.from_rotvec(np.asarray(aa, dtype=np.float64)).as_matrix().astype(np.float32)


def _build_se3_trace(rot_mat, trans):  # (T, 3, 3), (T, 3) -> (T, 4, 4)
    """Stack rotation + translation into a homogeneous SE(3) trace.

    Args:
        rot_mat: Rotation matrices, shape ``(T, 3, 3)`` float32.
        trans:   Translations, shape ``(T, 3)`` float32.

    Returns:
        SE(3) matrices of shape ``(T, 4, 4)`` float32.
    """
    T = rot_mat.shape[0]
    out = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()  # (T, 4, 4)
    out[:, :3, :3] = rot_mat
    out[:, :3, 3] = trans
    return out


class HandObjectDataset(BaseDataset):
    """Dataset for canonical hand-object interaction episodes.

    Reads per-episode canonical ``.npz`` files (see
    :class:`umm.eval.handobject.canonical.CanonicalSample`) and emits samples
    with 3 SE(3) trajectory entities: left wrist, right wrist, and object.
    Structurally mirrors :class:`TacoDataset` windowing/context sampling so
    the same training pipeline can consume both domains.

    Args:
        data_root: Root directory containing per-split episode directories
            and ``splits/<split>.json`` index files.
        split_fn: Absolute path to ``<data_root>/splits/<split>.json``. JSON
            content is ``[[episode_stem, T_valid], ...]`` where
            ``episode_stem = "NNNNNN"`` (no ``.npz``).
        base_fps: Native recording rate (30 for GRAB and ARCTIC).
        random_flag: Enable stochastic windowing / context sampling.
        safe_flag: If True, skip bad samples silently (wrapped by BaseDataset).
        repeat: Dataset repetition multiplier.
        t_rate: Temporal stride for windowing.
        language_fn: Unused; text lives inside each episode npz. Accepted
            for signature parity with :class:`TacoDataset`.
        max_T: Max time steps per sample.
        max_C: Max context steps per sample.
        state_dirname: Sub-directory under ``data_root`` that holds the
            per-episode ``NNNNNN.npz`` files (typically ``"train"``, ``"val"``,
            or ``"test"``).

    Example::

        ds = HandObjectDataset(
            data_root="data/handobject/umm/grab",
            split_fn="data/handobject/umm/grab/splits/train.json",
            base_fps=30,
            state_dirname="train",
        )
        sample, extras = ds[0]
    """

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps,  # each dataset must set manually
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        t_rate: int = 1,
        language_fn: Optional[str] = None,  # unused; kept for TacoDataset parity
        max_T: int = 60,
        max_C: int = 8,
        state_dirname: str = "train",
    ):
        """Initialize the HandObject dataset and cache per-split config.

        Args:
            data_root: Root directory of the canonical dataset.
            split_fn: Absolute path to the split JSON.
            base_fps: Native frame rate.
            random_flag: Stochastic sampling.
            safe_flag: Safe ``__getitem__`` wrapper.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            language_fn: Unused; accepted only for signature parity.
            max_T: Max time steps per sample.
            max_C: Max context steps per sample.
            state_dirname: Sub-directory containing ``NNNNNN.npz`` files.
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn, random_flag, safe_flag, repeat
        )

        self.max_T = int(max_T)
        self.max_C = int(max_C)
        self.t_rate = int(t_rate)
        self.state_dirname = str(state_dirname)

        # Semantic choices (documented for reviewers):
        #   - SEM_LEFT_MANO / SEM_RIGHT_MANO for wrists (same as TacoDataset).
        #   - SEM_OBJECT for the single rigid object.
        #   - UID_MANO_LEFT_HAND[0] / UID_MANO_RIGHT_HAND[0] are the wrist uids
        #     only (this dataset streams a single wrist pose, not 16 MANO
        #     joints — the wrist is the canonical MANO root).
        #   - UID_TACO_TOOL is reused as the object uid. `dataset_config.py`
        #     exposes no dedicated UID_HANDOBJ constant; UID_TACO_TOOL is the
        #     existing "single generic tool" token (confirmed by grep for
        #     ``UID_HANDOBJ`` yielding no hits, 2026-04-20).
        self.left_wrist_uid = UID_MANO_LEFT_HAND[0]
        self.right_wrist_uid = UID_MANO_RIGHT_HAND[0]
        self.object_uid = UID_TACO_TOOL

        # Cross-check base_fps against the stored per-episode fps.
        if len(self.indices) > 0:
            sid0, _ = self.indices[0]
            path0 = osp.join(self.data_root, self.state_dirname, f"{sid0}.npz")
            with np.load(path0, allow_pickle=False) as raw0:
                fps0 = int(raw0["fps"])
            if fps0 != int(base_fps):
                raise ValueError(
                    f"HandObjectDataset: base_fps={base_fps} does not match "
                    f"on-disk fps={fps0} in {path0}"
                )
        return

    # ------------------------------------------------------------------
    # Windowing — copied verbatim from TacoDataset.__decide_trace_sampling__
    # ------------------------------------------------------------------
    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled: bool = True
    ):
        """Decide the temporal window for a training sample.

        Args:
            seq_id: Episode identifier (file stem).
            current_abs_tid: Absolute anchor frame.
            dataset_idx: Global index (deterministic offset).
            random_enabled: Allow random jitter.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)``.
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
    # Context sampling — copied verbatim from TacoDataset.
    # ------------------------------------------------------------------
    def __decide_context_sampling__(self, idx):
        """Sample context sequence IDs and absolute time steps.

        Args:
            idx: Current dataset index (used for deterministic spacing).

        Returns:
            Tuple ``(ret_sid, ret_abs_tid)`` — lists of length ``max_C``.
        """
        if self.random_flag:
            choice = np.random.choice(len(self), self.max_C, replace=True)
        else:
            step = max(1, len(self) // (1 + self.max_C))
            start = idx % step
            choice = np.arange(start, len(self), step)
            choice = choice[: self.max_C]
        ret_sid, ret_abs_tid = [], []
        for i in choice:
            sid, current_abs_tid = self.indices[i]
            ret_sid.append(sid)
            ret_abs_tid.append(current_abs_tid)
        return ret_sid, ret_abs_tid

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def __add_hand_object_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx: int,  # anchor index inside the slice
        start_tid: int,    # global slice start
        end_tid: int,      # global slice end
        seq_id: str,
        ctx_sids: List[str],
        ctx_abs_tids: List[int],
    ) -> Tuple[PromptV0, Dict[str, Any]]:
        """Populate a PromptV0 with bimanual wrist + object SE(3) traces.

        Loads the canonical episode, composes SE(3) matrices for each of the
        three entities (L wrist, R wrist, object) from the per-frame rotations
        and translations, pulls one frame per context sample, and appends all
        three traces to the given PromptV0.

        Args:
            prompt: Prompt accumulator to mutate.
            cur_rel_idx: Anchor index within the returned slice (unused here,
                kept for signature parity with TacoDataset).
            start_tid: Absolute start frame (inclusive).
            end_tid: Absolute end frame (exclusive).
            seq_id: Current episode file stem.
            ctx_sids: Context episode file stems.
            ctx_abs_tids: Context absolute frame indices.

        Returns:
            Tuple ``(prompt, extras)`` where ``extras`` carries fields used by
            downstream unit tests (``obj_name``, ``text``, ``is_lhand``,
            ``is_rhand``, ``left_hand_joints``, ``right_hand_joints``).
        """
        factor = int(self.t_rate)

        # Load current episode with a single file open.
        ep_path = osp.join(self.data_root, self.state_dirname, f"{seq_id}.npz")
        with np.load(ep_path, allow_pickle=False) as raw:
            if "is_lhand" not in raw.files or "is_rhand" not in raw.files:
                raise KeyError(
                    f"HandObjectDataset: episode {ep_path!r} missing required key(s) "
                    f"'is_lhand'/'is_rhand'. Adapter must include these keys for GID "
                    f"remap to be correct."
                )
            is_l = int(raw["is_lhand"])
            is_r = int(raw["is_rhand"])
            left_hand_joints = raw["left_hand_joints"].copy()   # (T, 21, 3)
            right_hand_joints = raw["right_hand_joints"].copy() # (T, 21, 3)
            left_hand_rot = raw["left_hand_rot"].copy()         # (T, 16, 3)
            right_hand_rot = raw["right_hand_rot"].copy()       # (T, 16, 3)
            obj_translation = raw["obj_translation"].copy()     # (T, 3)
            obj_rotation = raw["obj_rotation"].copy()           # (T, 3, 3)
            valid_mask = raw["valid_mask"].copy()               # (T,)
            obj_name = str(raw["obj_name"])
            text = str(raw["text"])

        # -------------------- current window -----------------------------
        # Axis-angle (T, 3) for wrist (index 0 of the 16 MANO-style rotations),
        # then compose with world-frame wrist position *_hand_joints[:, 0].
        left_aa = left_hand_rot[start_tid:end_tid:factor, 0]          # (T, 3)
        right_aa = right_hand_rot[start_tid:end_tid:factor, 0]        # (T, 3)
        left_trans = left_hand_joints[start_tid:end_tid:factor, 0]    # (T, 3)
        right_trans = right_hand_joints[start_tid:end_tid:factor, 0]  # (T, 3)
        obj_rot_mat = obj_rotation[start_tid:end_tid:factor]          # (T, 3, 3)
        obj_trans = obj_translation[start_tid:end_tid:factor]         # (T, 3)
        valid = valid_mask[start_tid:end_tid:factor]                  # (T,)

        T_eff = int(left_aa.shape[0])
        assert right_aa.shape[0] == T_eff == obj_trans.shape[0] == valid.shape[0]

        left_rotmat = _axis_angle_to_rotmat(left_aa)    # (T, 3, 3)
        right_rotmat = _axis_angle_to_rotmat(right_aa)  # (T, 3, 3)

        left_T_w = _build_se3_trace(left_rotmat, left_trans)      # (T, 4, 4)
        right_T_w = _build_se3_trace(right_rotmat, right_trans)   # (T, 4, 4)
        obj_T_w = _build_se3_trace(obj_rot_mat, obj_trans)        # (T, 4, 4)

        left_trace = torch.as_tensor(left_T_w).unsqueeze(0)    # (1, T, 4, 4)
        right_trace = torch.as_tensor(right_T_w).unsqueeze(0)  # (1, T, 4, 4)
        obj_trace = torch.as_tensor(obj_T_w).unsqueeze(0)      # (1, T, 4, 4)

        # -------------------- context ------------------------------------
        # For each context (sid, abs_tid), extract single-frame SE(3) for each
        # of the three entities. Cache per-sid loads to avoid re-reading when
        # multiple context picks share a sid.
        ctx_cache: Dict[str, Any] = {}

        def _get_ctx_ep(sid: str):
            if sid not in ctx_cache:
                ctx_cache[sid] = load_canonical_npz(
                    osp.join(self.data_root, self.state_dirname, f"{sid}.npz")
                )
            return ctx_cache[sid]

        ctx_left_list: List[np.ndarray] = []
        ctx_right_list: List[np.ndarray] = []
        ctx_obj_list: List[np.ndarray] = []
        for sid, abs_tid in zip(ctx_sids, ctx_abs_tids):
            ctx_ep = _get_ctx_ep(sid)
            T_ctx = ctx_ep.left_hand_joints.shape[0]  # type: ignore[union-attr]
            t = max(0, min(int(abs_tid), T_ctx - 1))

            l_aa = ctx_ep.left_hand_rot[t:t + 1, 0]       # (1, 3)
            r_aa = ctx_ep.right_hand_rot[t:t + 1, 0]      # (1, 3)
            l_tr = ctx_ep.left_hand_joints[t:t + 1, 0]    # (1, 3)
            r_tr = ctx_ep.right_hand_joints[t:t + 1, 0]   # (1, 3)
            o_rot = ctx_ep.obj_rotation[t:t + 1]          # (1, 3, 3)
            o_tr = ctx_ep.obj_translation[t:t + 1]        # (1, 3)

            l_mat = _axis_angle_to_rotmat(l_aa)           # (1, 3, 3)
            r_mat = _axis_angle_to_rotmat(r_aa)           # (1, 3, 3)

            ctx_left_list.append(_build_se3_trace(l_mat, l_tr)[0])   # (4, 4)
            ctx_right_list.append(_build_se3_trace(r_mat, r_tr)[0])  # (4, 4)
            ctx_obj_list.append(_build_se3_trace(o_rot, o_tr)[0])    # (4, 4)

        # (C, 4, 4) -> (1, C, 4, 4)
        ctx_left = torch.as_tensor(np.stack(ctx_left_list, axis=0)).unsqueeze(0)
        ctx_right = torch.as_tensor(np.stack(ctx_right_list, axis=0)).unsqueeze(0)
        ctx_obj = torch.as_tensor(np.stack(ctx_obj_list, axis=0)).unsqueeze(0)

        # -------------------- masks --------------------------------------
        # trace_mask: (1, T_eff) float; zero out entirely for inactive hands.
        valid_mask = torch.as_tensor(valid.astype(np.float32)).unsqueeze(0)  # (1, T_eff)
        left_mask = valid_mask if is_l else torch.zeros_like(valid_mask)
        right_mask = valid_mask if is_r else torch.zeros_like(valid_mask)
        obj_mask = valid_mask  # object is always valid-by-canonical-mask

        ctx_ones = torch.ones(1, ctx_left.shape[1])
        ctx_left_mask = ctx_ones if is_l else torch.zeros_like(ctx_ones)
        ctx_right_mask = ctx_ones if is_r else torch.zeros_like(ctx_ones)
        ctx_obj_mask = torch.ones(1, ctx_obj.shape[1])

        # -------------------- group_id remapping --------------------------
        # Remap active-entity group_ids to be contiguous from 0.
        # The collate function asserts unique active-entity GIDs == [0,1,...,K-1].
        # An inactive hand has trace_mask=all-zeros so collate excludes it from
        # the GID assertion; its GID value is irrelevant (set to 0 as safe default).
        gid = 0
        left_gid = gid
        if is_l:
            gid += 1
        else:
            left_gid = 0  # inactive; excluded from assertion by zero trace_mask
        right_gid = gid
        if is_r:
            gid += 1
        else:
            right_gid = 0  # inactive; excluded from assertion by zero trace_mask
        obj_gid = gid  # object always active

        # -------------------- append traces -----------------------------
        prompt.append_trace(
            trace=left_trace,
            trace_mask=left_mask,
            context=ctx_left,
            context_mask=ctx_left_mask,
            group_id=left_gid,
            sem_type=SEM_LEFT_MANO,
            uid=[self.left_wrist_uid],
        )
        prompt.append_trace(
            trace=right_trace,
            trace_mask=right_mask,
            context=ctx_right,
            context_mask=ctx_right_mask,
            group_id=right_gid,
            sem_type=SEM_RIGHT_MANO,
            uid=[self.right_wrist_uid],
        )
        prompt.append_trace(
            trace=obj_trace,
            trace_mask=obj_mask,
            context=ctx_obj,
            context_mask=ctx_obj_mask,
            group_id=obj_gid,
            sem_type=SEM_OBJECT,
            uid=[self.object_uid],
        )

        extras: Dict[str, Any] = {
            "obj_name": obj_name,
            "text": text,
            "is_lhand": is_l,
            "is_rhand": is_r,
            "left_hand_joints": left_hand_joints,   # (T, 21, 3) float32
            "right_hand_joints": right_hand_joints, # (T, 21, 3) float32
        }
        return prompt, extras

    # ------------------------------------------------------------------
    # Main sample builder
    # ------------------------------------------------------------------
    def __getitem_inner__(self, idx):
        """Build a single training sample (mirrors TacoDataset structure).

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extras)``. ``extras`` is passed
            to :class:`CanonicalDatasetOutput.collate` as the per-sample
            free-form dict consumed by tests and viz code.
        """
        idx = idx % len(self)
        prompt = PromptV0(P=3, T=self.max_T, C=self.max_C)

        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )
        ctx_sids, ctx_abs_tids = self.__decide_context_sampling__(idx)

        prompt, extras = self.__add_hand_object_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid, ctx_sids, ctx_abs_tids
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        text = extras["text"]

        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace,
            trace_mask=trace_mask,
            context=context,
            context_mask=context_mask,
            trace_group_id=group_id,
            trace_rot_mask=rot_mask,
            trace_sem_type=sem_type,
            trace_uid=uid,
            time=time_stamp,
            text=text,
        )
        return ret, extras
