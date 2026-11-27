"""GRAB hand-object interaction dataset for UMM wmm scaffold.

Reads the 30 fps WORLD-frame SE(3) cache embedded in canonical
handobject_new.npz files (reference v0.8 lines 304-313). The cache keys
``Q_l_world_30fps`` / ``Q_r_world_30fps`` / ``obj_se3_world_30fps`` are
pre-aligned so that **canonical t=49 == world_grasp_raw_idx** (the
DiffH2O grasp keyframe in the 120 fps raw timeline) by construction.

No MANO forward at __getitem__. No stride-4 subsampling. The dataset
truncates / pads each sequence's variable T_30 to ``max_T=200`` starting
at cache t=0; because the cache is grasp-anchored at t=49, this
preserves the eval contract `eval_keyframe → t=49 = grasp` for every
sequence (no longer just ~10% as in the prior stride-4 implementation).

Missing cache keys raise ``KeyError`` loudly — this dataset does not
fall back to the 120 fps path. Run
``prepare_data/handobject_new/build_grab_30fps_world_se3.py`` to
augment any missing canonical .npz files before training.

Mirror handling (per reference line 313): the cached SE(3) is in the
un-mirrored frame; only the object mesh (and 2048-pt ``obj_pcl``)
is x-flipped at consumer time when ``world_is_mirrored=True``. The
``_build_pcl`` helper applies that flip here.

Known caveat — cache tail-clipping (review LATENT-RISK):
``build_grab_30fps_world_se3.py`` clips the raw-index map to
``[0, T_raw - 1]`` on both ends when computing the stride-4 anchor.
Sequences whose grasp lands near a temporal boundary therefore have
*repeated* end frames in ``Q_*_world_30fps`` / ``obj_se3_world_30fps``
(the cache "freezes" at the boundary). The dataset's ``trace_mask`` is
not currently aware of this — those frozen frames are flagged as valid.
For typical GRAB seqs (``T_raw`` 600–2000, grasp roughly centered) the
effect is small (<5 frames per side). Mitigation either at build
(propagate ``n_clipped_start/end`` keys) or in this loader (compute
valid range as ``world_grasp_raw_idx ± floor((T_raw - 1 - grasp) / 4)``)
is deferred — neither path is correct without coordinated edits,
which are outside scope.
"""

import logging
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from umm.datasets.wmm.base_dataset import PromptV0
from umm.datasets.wmm.dataset_template import CanonicalDatasetOutput
from umm.datasets.wmm.dataset_utils import sample_list
from umm.datasets.wmm.dataset_config import (
    SEM_LEFT_MANO,
    SEM_RIGHT_MANO,
    SEM_OBJECT,
    UID_MANO_LEFT_HAND,
    UID_MANO_RIGHT_HAND,
    UID_HUMOTO_OBJECT_GROUP,
)
from umm.datasets.wmm.mask_scheduler import MaskScheduler, GRAB_PATTERNS

_LOG = logging.getLogger(__name__)

_T_DEFAULT = 200
_KEYFRAME_T = 49
_NUM_HAND_JOINTS = 16


class GrabDataset(Dataset):
    """GRAB canonical-v0.8 dataset for the wmm hand-object protocol.

    Reads the 30 fps grasp-anchored cache. Each sequence's
    ``Q_l_world_30fps``/``Q_r_world_30fps``/``obj_se3_world_30fps`` is
    laid out so that canonical t=49 corresponds to ``world_grasp_raw_idx``
    in the 120 fps raw timeline. The dataset preserves this contract by
    starting at cache t=0 and truncating/padding to ``max_T``.

    Args:
        data_root: e.g. ``data_wmm/handobject_new/canonical``.
        split: one of ``s1_train``, ``s1_test``, ``s2_train``, ``s2_test``.
            Resolves to ``<data_root>/<s{1,2}>/<{train,test}>/*.npz``.
        random_flag: stochastic PCL subsampling.
        safe_flag: skip-on-error (bounded retry).
        repeat: index repetition multiplier.
        num_pts: PCL points (default 256).
        max_T: time-window length at 30 fps (default 200).
        max_C: ctx slots (default 1).
        text_required: True -> fail loud on missing text_detailed.
        mask_scheduler: optional MaskScheduler; defaults to GRAB_PATTERNS
            with base_random_prob=0.0.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "s1_train",
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        num_pts: int = 256,
        max_T: int = _T_DEFAULT,
        max_C: int = 1,
        text_required: bool = True,
        skip_mirrored: bool = True,
        require_cache: bool = True,
        min_coverage_frac: float = 0.0,
        mask_scheduler: Optional[MaskScheduler] = None,
    ):
        super().__init__()
        valid_splits = {"s1_train", "s1_test", "s2_train", "s2_test"}
        if split not in valid_splits:
            raise ValueError(
                f"GrabDataset: split must be one of {valid_splits}, got {split!r}"
            )
        s_tag, phase = split.split("_")
        self.split = split
        self.npz_dir = osp.join(data_root, s_tag, phase)
        if not osp.isdir(self.npz_dir):
            raise FileNotFoundError(f"GrabDataset: missing {self.npz_dir}")

        self.data_root = data_root
        self.random_flag = bool(random_flag)
        self.safe_flag = bool(safe_flag)
        self.repeat = max(1, int(repeat))
        self.num_pts = int(num_pts)
        self.max_T = int(max_T)
        self.max_C = int(max_C)
        self.text_required = bool(text_required)
        if not (1 <= self.max_T <= 1024):
            raise ValueError(f"GrabDataset: max_T={self.max_T} out of range")
        if self.max_T <= _KEYFRAME_T:
            # eval_keyframe pattern reveals t=49; if max_T <= 49, that column
            # never exists, the pattern silently degrades to the
            # `_enforce_min_known` fallback (random tokens), and the eval
            # contract semantics break. Fail loud instead.
            raise ValueError(
                f"GrabDataset: max_T={self.max_T} must be > _KEYFRAME_T={_KEYFRAME_T} "
                f"(otherwise eval_keyframe pattern has no valid column to reveal)"
            )
        if self.max_C < 1:
            raise ValueError(f"GrabDataset: max_C={self.max_C} must be >= 1")

        files = sorted(f for f in os.listdir(self.npz_dir) if f.endswith(".npz"))
        if not files:
            raise FileNotFoundError(f"GrabDataset: no .npz in {self.npz_dir}")

        # Filter 1: drop M-prefix mirrored seqs by default. Per reference line 23
        # ("M entries inherit the SAME raw GRAB params as their non-mirrored
        # counterparts") and line 313 ("cached SE(3) is in the un-mirrored
        # frame"), M-seqs in our world-frame pipeline produce IDENTICAL output
        # to their non-M counterparts after the obj_pcl flip-undo (see
        # _build_pcl). Including them as separate samples doubles training
        # compute with no novel signal. Set ``skip_mirrored=False`` to keep
        # them (e.g. if downstream code introduces genuine mirror augmentation).
        self.skip_mirrored = bool(skip_mirrored)
        if self.skip_mirrored:
            n_before = len(files)
            files = [f for f in files if not f.startswith("M")]
            n_dropped_m = n_before - len(files)
            if n_dropped_m > 0:
                _LOG.info(
                    "GrabDataset(%s): skip_mirrored=True dropped %d M-seqs (%d remaining)",
                    split, n_dropped_m, len(files),
                )

        # Filter 2: keep only seqs with the 30-fps cache keys present.
        # augments incrementally; partial coverage is expected during
        # the run. Filtering at __init__ (rather than skipping at __getitem__
        # via safe_flag retry) avoids the O(n) consecutive-miss runtime crash
        # mode (s1/train can have ~1900 consecutive un-augmented entries
        # which would exceed any reasonable retry budget).
        self.require_cache = bool(require_cache)
        if self.require_cache:
            REQUIRED = (
                "Q_l_world_30fps", "Q_r_world_30fps",
                "obj_se3_world_30fps", "world_grasp_raw_idx",
            )
            kept: List[str] = []
            n_missing = 0
            n_unreadable = 0
            for f in files:
                try:
                    with np.load(osp.join(self.npz_dir, f), allow_pickle=False) as d:
                        present = set(d.files)
                    if all(k in present for k in REQUIRED):
                        kept.append(f)
                    else:
                        n_missing += 1
                except (OSError, ValueError, KeyError) as e:
                    n_unreadable += 1
                    _LOG.warning(
                        "GrabDataset(%s): unreadable npz %s: %s", split, f, e,
                    )
            if n_missing or n_unreadable:
                _LOG.warning(
                    "GrabDataset(%s): cache filter dropped %d (missing-keys=%d, "
                    "unreadable=%d) of %d files. Run "
                    "prepare_data/handobject_new/build_grab_30fps_world_se3.py "
                    "to augment the canonical .npz files.",
                    split, n_missing + n_unreadable, n_missing, n_unreadable,
                    n_missing + n_unreadable + len(kept),
                )
            # Aggregate-coverage gate (review FIX-RECOMMENDED #2).
            # Per-file WARNINGs are not surfaced by the training launcher;
            # without an aggregate floor, a partially-augmented split could
            # silently train on (e.g.) 20/2252 entries × repeat. Production
            # launchers should set ``min_coverage_frac >= 0.95``; default 0.0
            # preserves existing behaviour for dev / smoke flows.
            # Coverage denominator (n_total) is post-skip_mirrored: it is the
            # count of entries that the dataset would consider eligible
            # ignoring cache presence. With ``skip_mirrored=True`` (default),
            # M-seqs are excluded from both numerator and denominator, so the
            # ratio reflects coverage of the working (non-M) population. If a
            # consumer sets ``skip_mirrored=False``, the denominator includes
            # M-seqs and the ratio reflects raw cache coverage.
            self.min_coverage_frac = float(min_coverage_frac)
            n_total = n_missing + n_unreadable + len(kept)
            coverage = (len(kept) / n_total) if n_total > 0 else 0.0
            if self.min_coverage_frac > 0.0 and coverage < self.min_coverage_frac:
                raise RuntimeError(
                    f"GrabDataset({split}): cache coverage {coverage:.1%} "
                    f"({len(kept)}/{n_total}) below required min_coverage_frac="
                    f"{self.min_coverage_frac:.1%}. Wait for build to "
                    f"finish or run "
                    f"prepare_data/handobject_new/build_grab_30fps_world_se3.py."
                )

            files = kept
            if not files:
                raise FileNotFoundError(
                    f"GrabDataset: no cache-augmented .npz in {self.npz_dir}. "
                    f"Run prepare_data/handobject_new/build_grab_30fps_world_se3.py "
                    f"to populate the 30 fps WORLD-frame cache before training."
                )

        self.entries: List[str] = files
        self.indices: List[int] = list(range(len(self.entries))) * self.repeat

        # Constants kept for downstream interface compat
        self.base_fps = 30
        self.t_rate = 1

        # Default scheduler — base_random_prob=0.0 keeps the eval_keyframe
        # pattern dominant (4:1:1:1 weights, ~57% of samples reveal exactly
        # t=49). Random masks would dilute that regime.
        self.mask_scheduler = mask_scheduler or MaskScheduler(
            base_random_prob=0.0, patterns=GRAB_PATTERNS,
        )

        _LOG.info(
            "GrabDataset(%s): %d entries, repeat=%d, max_T=%d, num_pts=%d",
            split, len(self.entries), self.repeat, self.max_T, self.num_pts,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.safe_flag:
            orig_idx = idx
            tries = 0
            while True:
                try:
                    return self.__getitem_inner__(idx)
                except Exception as e:
                    logging.error(
                        f"GrabDataset: error at idx={idx} (orig={orig_idx}, try={tries+1}): {e}"
                    )
                    idx = (idx + 1) % len(self)
                    tries += 1
                    if tries > 64:
                        raise RuntimeError(
                            f"GrabDataset: 64 consecutive errors starting at orig_idx={orig_idx}"
                        ) from e
        return self.__getitem_inner__(idx)

    @staticmethod
    def worker_init_fn(worker_id):
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def _build_pcl(self, obj_pcl_raw: np.ndarray, is_mirrored: bool) -> torch.Tensor:
        """Subsample 2048-pt obj_pcl -> (num_pts, 6) xyz+rgb. RGB = zeros.

        Mirror: x-flip when world_is_mirrored. obj_pcl is in object-local
        frame (verified in canonical .npz schema).
        """
        pcl = np.asarray(obj_pcl_raw, dtype=np.float32)
        if is_mirrored:
            pcl = pcl.copy()
            pcl[:, 0] *= -1.0
        sampled = sample_list(pcl, self.num_pts, self.random_flag)  # (num_pts, 3)
        rgb = np.zeros_like(sampled, dtype=np.float32)
        return torch.as_tensor(np.concatenate([sampled, rgb], axis=1))

    def __getitem_inner__(self, idx: int) -> Tuple[CanonicalDatasetOutput, Dict]:
        idx = self.indices[idx % len(self.indices)]
        entry = self.entries[idx]
        npz_path = osp.join(self.npz_dir, entry)
        if not osp.exists(npz_path):
            raise FileNotFoundError(f"GrabDataset: missing {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            # 30 fps grasp-anchored cache keys (reference v0.8 lines 304-313).
            # No silent fallback — if the cache is missing, the augmentation
            # script (prepare_data/handobject_new/build_grab_30fps_world_se3.py)
            # has not yet processed this file. Raise loudly so under-augmented
            # data is detected at training launch, not silently degraded.
            for required in (
                "Q_l_world_30fps", "Q_r_world_30fps", "obj_se3_world_30fps",
                "world_grasp_raw_idx",
                "obj_pcl", "world_is_mirrored", "seq_index", "data_id",
                "obj_name", "subject_id",
            ):
                if required not in data.files:
                    raise KeyError(
                        f"GrabDataset: {entry} missing cache field "
                        f"{required!r}. Run "
                        f"prepare_data/handobject_new/build_grab_30fps_world_se3.py "
                        f"to augment canonical .npz files before training."
                    )
            Q_l_30 = data["Q_l_world_30fps"].astype(np.float32)         # (T_30, 16, 4, 4)
            Q_r_30 = data["Q_r_world_30fps"].astype(np.float32)         # (T_30, 16, 4, 4)
            obj_se3_30 = data["obj_se3_world_30fps"].astype(np.float32)  # (T_30, 4, 4)
            world_grasp_raw_idx = int(data["world_grasp_raw_idx"])
            obj_pcl_raw = data["obj_pcl"]                                # (2048, 3)
            world_is_mirrored = bool(data["world_is_mirrored"])
            seq_index = str(data["seq_index"])
            data_id = str(data["data_id"])
            obj_name = str(data["obj_name"])
            subject_id = str(data["subject_id"])
            text_detailed = (
                str(data["text_detailed"]) if "text_detailed" in data.files else ""
            )
            world_lhand_vtemp_path = (
                str(data["world_lhand_vtemp_path"])
                if "world_lhand_vtemp_path" in data.files else ""
            )
            world_rhand_vtemp_path = (
                str(data["world_rhand_vtemp_path"])
                if "world_rhand_vtemp_path" in data.files else ""
            )
            world_obj_mesh_path = (
                str(data["world_obj_mesh_path"])
                if "world_obj_mesh_path" in data.files else ""
            )

        if self.text_required and not text_detailed:
            raise ValueError(
                f"GrabDataset: missing text_detailed in {entry} (text_required=True)"
            )

        # Shape sanity — cache T axis must match across the three arrays.
        T_30_cache = Q_l_30.shape[0]
        if Q_l_30.shape != (T_30_cache, _NUM_HAND_JOINTS, 4, 4):
            raise ValueError(f"GrabDataset: bad Q_l_world_30fps shape {Q_l_30.shape}")
        if Q_r_30.shape != (T_30_cache, _NUM_HAND_JOINTS, 4, 4):
            raise ValueError(f"GrabDataset: bad Q_r_world_30fps shape {Q_r_30.shape}")
        if obj_se3_30.shape != (T_30_cache, 4, 4):
            raise ValueError(
                f"GrabDataset: bad obj_se3_world_30fps shape {obj_se3_30.shape}; "
                f"expected ({T_30_cache}, 4, 4)"
            )
        if obj_pcl_raw.shape != (2048, 3):
            raise ValueError(f"GrabDataset: bad obj_pcl shape {obj_pcl_raw.shape}")
        # The eval contract requires t=49 to be the grasp keyframe. cache
        # is grasp-anchored at index 49, so T_30_cache MUST be > 49 for the
        # cache to contain that index. Empirically all 290 s1/test seqs have
        # T_30 >= 88; any seq below that threshold is a build-time bug or a
        # pathologically short raw GRAB clip — fail loud rather than serve
        # identity-padded grasp poses.
        if T_30_cache <= _KEYFRAME_T:
            raise ValueError(
                f"GrabDataset: {entry} T_30_cache={T_30_cache} <= {_KEYFRAME_T} "
                f"(cache must contain a real grasp pose at t={_KEYFRAME_T}). "
                f"Investigate the build for this seq."
            )

        # Cache is grasp-anchored: canonical t=49 == world_grasp_raw_idx in raw
        # 120 fps timeline. We start at t=0 of the cache (NO random crop start)
        # so the eval_keyframe pattern reveal at t=49 always coincides with the
        # GT grasp pose. Truncating long seqs by [:max_T] preserves t=49 because
        # cache layout puts grasp at index 49.
        T = self.max_T
        valid_T = min(T_30_cache, T)

        def _pad_se3(arr: np.ndarray) -> np.ndarray:
            """Pad along axis 0 to length T using identity SE(3)."""
            if arr.shape[0] >= T:
                return arr[:T]
            pad_T = T - arr.shape[0]
            pad_shape = (pad_T,) + arr.shape[1:]
            pad = np.broadcast_to(np.eye(4, dtype=np.float32), pad_shape).copy()
            return np.concatenate([arr, pad], axis=0)

        Q_l_30 = _pad_se3(Q_l_30)            # (T, 16, 4, 4)
        Q_r_30 = _pad_se3(Q_r_30)
        obj_se3_30 = _pad_se3(obj_se3_30)    # (T, 4, 4)

        # Permute Q to (16, T, 4, 4) — wmm convention
        Q_l = torch.from_numpy(Q_l_30).permute(1, 0, 2, 3).contiguous()  # (16, T, 4, 4)
        Q_r = torch.from_numpy(Q_r_30).permute(1, 0, 2, 3).contiguous()
        obj_se3 = torch.from_numpy(obj_se3_30).unsqueeze(0).contiguous()  # (1, T, 4, 4)

        # Trace masks: True for [0, valid_T), False on padded tail
        hand_trace_mask = torch.zeros(_NUM_HAND_JOINTS, T, dtype=torch.bool)
        hand_trace_mask[:, :valid_T] = True
        obj_trace_mask = torch.zeros(1, T, dtype=torch.bool)
        obj_trace_mask[:, :valid_T] = True

        # Build prompt: 16 (L) + 16 (R) + 1 (obj) = 33 entities
        prompt = PromptV0(
            P=_NUM_HAND_JOINTS * 2 + 1,
            T=T,
            C=self.max_C,
            PCL=self.num_pts,
            PCL_dim=6,
        )
        # Hand ctx: identity SE(3) per joint (cross-sequence; no canonical pose
        # available cheaply)
        hand_ctx = torch.eye(4, dtype=torch.float32).expand(
            _NUM_HAND_JOINTS, self.max_C, 4, 4,
        ).contiguous()
        ctx_mask_full = torch.ones(_NUM_HAND_JOINTS, self.max_C)

        prompt.append_trace(
            trace=Q_l,
            trace_mask=hand_trace_mask,
            context=hand_ctx,
            context_mask=ctx_mask_full,
            group_id=0,
            sem_type=SEM_LEFT_MANO,
            uid=UID_MANO_LEFT_HAND,
        )
        prompt.append_trace(
            trace=Q_r,
            trace_mask=hand_trace_mask.clone(),
            context=hand_ctx.clone(),
            context_mask=ctx_mask_full.clone(),
            group_id=1,
            sem_type=SEM_RIGHT_MANO,
            uid=UID_MANO_RIGHT_HAND,
        )

        # Object
        obj_ctx = torch.eye(4, dtype=torch.float32).expand(1, self.max_C, 4, 4).contiguous()
        local_pcl = self._build_pcl(obj_pcl_raw, world_is_mirrored).unsqueeze(0)  # (1, num_pts, 6)
        prompt.append_trace(
            trace=obj_se3,
            trace_mask=obj_trace_mask,
            context=obj_ctx,
            context_mask=torch.ones(1, self.max_C),
            group_id=2,
            sem_type=SEM_OBJECT,
            uid=int(UID_HUMOTO_OBJECT_GROUP[0]),
            local_pcl=local_pcl,
            local_pcl_mask=torch.ones(1),
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        # Mask scheduling — pass keyframe_t=49 (eval_keyframe pattern uses it)
        # current_ith=49 keeps future_with_goal etc. semantically consistent if
        # downstream patterns ever use it.
        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask,
            context_mask=context_mask,
            sem_type=sem_type,
            current_ith=_KEYFRAME_T,
            keyframe_t=_KEYFRAME_T,
        )

        # time_stamp anchored at t=49 = grasp keyframe by cache
        # construction. "t=0 sec" in the model's clock is therefore the grasp
        # moment for every sequence (eval contract preserved across the
        # whole dataset, not the prior ~10% under stride-4).
        time_unit = float(self.t_rate) / float(self.base_fps)
        time_stamp = (torch.arange(T) - _KEYFRAME_T) * time_unit

        local_pcl_full, local_pcl_mask_full = prompt.cat_pcl()

        ret = CanonicalDatasetOutput(
            seq_name=data_id,
            trace=trace,
            trace_mask=trace_mask,
            context=context,
            context_mask=context_mask,
            known_mask=known_mask,
            context_known_mask=ctx_known_mask,
            trace_group_id=group_id,
            trace_rot_mask=rot_mask,
            trace_sem_type=sem_type,
            trace_uid=uid,
            time=time_stamp,
            text=text_detailed,
            local_pcl=local_pcl_full,
            local_pcl_mask=local_pcl_mask_full,
        )
        grab_extra = {
            "seq_index": seq_index,
            "data_id": data_id,
            "subject_id": subject_id,
            "obj_name": obj_name,
            "world_is_mirrored": world_is_mirrored,
            "world_grasp_raw_idx": int(world_grasp_raw_idx),
            "T_30_cache": int(T_30_cache),
            "valid_T": int(valid_T),
            "world_lhand_vtemp_path": world_lhand_vtemp_path,
            "world_rhand_vtemp_path": world_rhand_vtemp_path,
            "world_obj_mesh_path": world_obj_mesh_path,
            "split": self.split,
            # viz integration: GrabVisualizer (grab_viz.py) renders from
            # world-frame SE(3) tensors keyed `Q_l_world`, `Q_r_world`,
            # `obj_se3_world` (per-sample, shape (T,16,4,4)/(T,4,4) — same
            # layout as the un-permuted padded cache). Without these
            # the viz routine hits actual_len=0 and silently no-ops every
            # train/val viz fire. Emit them here so grab_viz can find them
            # via batch["extra"][key][bid] (after CanonicalDatasetOutput
            # collate). Independent of trace because trace gets canonicalized
            # downstream; viz needs the raw world frame.
            "Q_l_world": torch.from_numpy(Q_l_30).contiguous(),         # (T, 16, 4, 4)
            "Q_r_world": torch.from_numpy(Q_r_30).contiguous(),         # (T, 16, 4, 4)
            "obj_se3_world": torch.from_numpy(obj_se3_30).contiguous(),  # (T, 4, 4)
        }
        return ret, grab_extra
