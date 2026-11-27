"""HUMOTO human-object interaction dataset for UMM V13.

Provides 65-joint Mixamo body trajectories alongside rigid object SE(3) traces
with local point-cloud conditioning for the HUMOTO benchmark.  Follows the
V10+ CanonicalDatasetOutput contract with MaskScheduler integration.

Each sample contains 65 body joint SE(3) traces (group 0) and up to
``max_objects`` rigid object SE(3) traces (groups 1..max_objects), each with
a local point cloud sampled from the object mesh.

Example::

    dataset = HumotoDataset(
        data_root="data/hoi_all/humoto_processed",
        split_fn="data/hoi_all/humoto_processed/splits/train.json",
        base_fps=30,
        max_T=30,
        max_C=1,
        max_objects=4,
        max_P_per_object=256,
    )
    sample, extra = dataset[0]
"""

# Example YAML config entry:
# - type: humoto
#   data_root: ./data/hoi_all/humoto_processed
#   splits:
#     train: ./data/hoi_all/humoto_processed/splits/train.json
#   t_rate: 1
#   max_T: 30
#   max_C: 1
#   max_objects: 4
#   max_P_per_object: 256

import os
import os.path as osp
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from umm.datasets.umm_v13.base_dataset import (
    BaseDataset,
    PromptV0,
)
from umm.datasets.umm_v13.dataset_template import CanonicalDatasetOutput
from umm.datasets.umm_v13.dataset_utils import sample_list
from umm.datasets.umm_v13.dataset_config import (
    UID_HUMOTO_BODY_GROUP,
    UID_HUMOTO_OBJECT_GROUP,
    SEM_HUMAN,
    SEM_HUMAN_GLOBAL,
    SEM_OBJECT,
)
from umm.datasets.umm_v13.mask_scheduler import MaskScheduler, HUMOTO_PATTERNS

_LOG = logging.getLogger(__name__)

# Number of Mixamo body joints in the HUMOTO skeleton
_NUM_BODY_JOINTS = 65


class HumotoDataset(BaseDataset):
    """Dataset for HUMOTO human-object interaction sequences for UMM V13.

    Each sample contains 65 Mixamo body joint SE(3) traces and up to
    ``max_objects`` rigid object SE(3) traces with local point-cloud
    conditioning.  MaskScheduler integration provides ``known_mask``
    and ``context_known_mask`` tensors.

    Args:
        data_root: Root directory of the HUMOTO processed dataset.
        split_fn: JSON split file path (``[[seq_name, T], ...]``).
        base_fps: Recording frame rate (always 30 for HUMOTO).
        random_flag: Stochastic augmentation flag.
        safe_flag: Error-handling wrapper.
        repeat: Repetition multiplier.
        t_rate: Temporal stride.
        language_fn: Not used for HUMOTO; pass None.
        max_T: Maximum time steps per sample.
        max_C: Maximum context steps (1 for HUMOTO canonical context).
        max_objects: Maximum number of rigid objects per sample.
        max_P_per_object: Points per object point cloud.
        mask_scheduler: Optional MaskScheduler instance. If None, a default
            scheduler with ``base_random_prob=0.25`` and HUMOTO_PATTERNS is
            used.

    Example::

        ds = HumotoDataset(
            data_root="data/hoi_all/humoto_processed",
            split_fn="data/hoi_all/humoto_processed/splits/train.json",
            base_fps=30,
        )
        sample, extra = ds[0]
    """

    def __init__(
        self,
        data_root: str,
        split_fn: str,
        base_fps: int,
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        t_rate: int = 1,
        language_fn: Optional[str] = None,
        max_T: int = 30,
        max_C: int = 1,
        max_objects: int = 4,
        max_P_per_object: int = 256,
        mask_scheduler: Optional[MaskScheduler] = None,
    ):
        """Initialize the HUMOTO V13 dataset, pre-load object point clouds and text.

        Args:
            data_root: Root data directory (humoto_processed/).
            split_fn: JSON split file path.
            base_fps: Frame rate (30).
            random_flag: Stochastic sampling.
            safe_flag: Safe ``__getitem__``.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            language_fn: Not used for HUMOTO.
            max_T: Max time steps.
            max_C: Max context steps.
            max_objects: Max rigid objects.
            max_P_per_object: Points per object cloud.
            mask_scheduler: Optional MaskScheduler instance.
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn,
            random_flag, safe_flag, repeat, t_rate,
        )

        self.max_T = max_T
        self.max_C = max_C
        self.max_objects = max_objects
        self.max_P_per_object = max_P_per_object

        self.body_uids = UID_HUMOTO_BODY_GROUP  # list of 65 UIDs
        self.object_uids = UID_HUMOTO_OBJECT_GROUP  # list of 4 UIDs
        assert len(self.body_uids) == _NUM_BODY_JOINTS
        assert len(self.object_uids) >= self.max_objects

        self.mask_scheduler = mask_scheduler or MaskScheduler(
            base_random_prob=0.25, patterns=HUMOTO_PATTERNS,
        )

        # Pre-load object point clouds: {obj_name: (16384, 6) float32}
        self.object_pcl_cache: Dict[str, np.ndarray] = {}
        obj_dir = Path(data_root) / "objects"
        if obj_dir.exists():
            for entry in sorted(obj_dir.iterdir()):
                if entry.is_dir():
                    pcl_path = entry / "pcl.npz"
                    if pcl_path.exists():
                        pcl = np.load(str(pcl_path))
                        pts = pcl["points"].astype(np.float32)    # (16384, 3)
                        cols = pcl["colors"].astype(np.float32)   # (16384, 3)
                        # Replace NaN colors with zeros (HUMOTO meshes have no vertex colors)
                        np.nan_to_num(cols, copy=False, nan=0.0)
                        # xyz_rgb format: (16384, 6)
                        self.object_pcl_cache[entry.name] = np.concatenate(
                            [pts, cols], axis=1,
                        )
            _LOG.info(
                "HumotoDataset: loaded %d object point clouds from %s",
                len(self.object_pcl_cache), obj_dir,
            )
        else:
            _LOG.warning("HumotoDataset: object directory %s not found", obj_dir)

        # Pre-load text metadata + rig_hash: {seq_name: str}
        self.text_cache: Dict[str, str] = {}
        self.rig_hash_cache: Dict[str, str] = {}
        seq_dir = Path(data_root) / "sequences"
        for sid, _ in self.indices:
            if sid in self.text_cache:
                continue
            meta_path = seq_dir / (sid + ".yaml")
            if meta_path.exists():
                with open(str(meta_path), "r") as f:
                    meta = yaml.safe_load(f)
                self.text_cache[sid] = meta.get("text_short", "")
            else:
                self.text_cache[sid] = ""
            # Load rig_hash from npz for T-pose context
            npz_path = seq_dir / (sid + ".npz")
            if npz_path.exists():
                with np.load(str(npz_path), allow_pickle=True) as d:
                    rh = d["rig_hash"]
                    self.rig_hash_cache[sid] = rh.item() if hasattr(rh, "item") else str(rh)

        # Pre-load T-pose contexts per rig_hash: {rig_hash: (65, 4, 4) float32}
        # T-pose = inv(inv_bind), with rotation columns normalized to unit norm
        self.tpose_cache: Dict[str, np.ndarray] = {}
        template_dir = Path(data_root) / "template"
        unique_rigs = set(self.rig_hash_cache.values())
        for rh in unique_rigs:
            tpl_path = template_dir / rh / "template.npz"
            if tpl_path.exists():
                tpl = np.load(str(tpl_path))
                inv_bind = tpl["inv_bind"].astype(np.float64)  # (65, 4, 4)
                tpose = np.linalg.inv(inv_bind)  # (65, 4, 4) -- 0.01-scale rotations
                # Normalize rotation columns to unit norm (same as body_T_w fix)
                R = tpose[:, :3, :3]
                col_norm = np.linalg.norm(R[:, :, 0:1], axis=1, keepdims=True)  # (65, 1, 1)
                col_norm = np.maximum(col_norm, 1e-10)
                tpose[:, :3, :3] = R / col_norm
                tpose[:, :3, 3] -= tpose[0:1, :3, 3]  # pelvis (joint 0) → origin; others relative
                self.tpose_cache[rh] = tpose.astype(np.float32)
        _LOG.info(
            "HumotoDataset: loaded %d T-pose templates for context",
            len(self.tpose_cache),
        )

        _LOG.info(
            "HumotoDataset(%s): %d indices, %d sequences, %d objects cached, "
            "max_T=%d, max_C=%d, max_objects=%d, max_P_per_object=%d",
            split_fn, len(self.indices), len(self.seq_len),
            len(self.object_pcl_cache), max_T, max_C, max_objects, max_P_per_object,
        )

    def __decide_trace_sampling__(
        self,
        seq_id: str,
        current_abs_tid: int,
        dataset_idx: int,
        random_enabled: bool = True,
    ) -> Tuple[int, int, int]:
        """Decide the temporal window for a training sample.

        Args:
            seq_id: Sequence identifier.
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

    def __add_humoto_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx: int,
        start_tid: int,
        end_tid: int,
        seq_id: str,
    ) -> Tuple[PromptV0, Dict]:
        """Populate a PromptV0 with body joints and object traces.

        Loads 65 body joint SE(3) poses and per-object rigid SE(3) poses
        from the sequence npz, slices the temporal window, and appends
        them to the prompt with appropriate UIDs, semantic types, and
        group IDs.

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index within the slice.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.

        Returns:
            Tuple ``(prompt, humoto_dict)`` with updated prompt and extras.
        """
        humoto_dict: Dict = {}
        factor = int(self.t_rate)

        # Load sequence data
        seq_path = osp.join(self.data_root, "sequences", seq_id + ".npz")
        seq_data = np.load(seq_path, allow_pickle=True)

        # Validate schema
        schema = int(seq_data.get("schema_version", 0))
        assert schema == 3, (
            f"HumotoDataset: seq {seq_id} has schema_version={schema}, expected 3"
        )

        # --- Body joints ---
        # body_T_w: (T_full, 65, 4, 4) float64 -> slice and stride
        body_T_w = seq_data["body_T_w"][start_tid:end_tid:factor].copy()  # (T_slice, 65, 4, 4)
        # FIX: HUMOTO body rotation matrices are uniformly scaled by ~0.01
        # (GLB converter artifact). Normalize rotation columns to unit norm.
        R = body_T_w[:, :, :3, :3]  # (T_slice, 65, 3, 3)
        col_norm = np.linalg.norm(R[:, :, :, 0:1], axis=2, keepdims=True)  # (T_slice, 65, 1, 1)
        col_norm = np.maximum(col_norm, 1e-10)
        body_T_w[:, :, :3, :3] = R / col_norm
        body_T_w = torch.as_tensor(body_T_w, dtype=torch.float32)  # (T_slice, 65, 4, 4)
        body_T_w = body_T_w.permute(1, 0, 2, 3).contiguous()  # (65, T_slice, 4, 4)

        # Context for body: canonical T-pose from per-rig template, C=1
        rig_hash = self.rig_hash_cache.get(seq_id, "")
        if rig_hash and rig_hash in self.tpose_cache:
            tpose = self.tpose_cache[rig_hash]  # (65, 4, 4) float32, unit-norm rotations
            body_ctx = torch.from_numpy(tpose).unsqueeze(1).expand(
                -1, self.max_C, -1, -1,
            ).contiguous()  # (65, C, 4, 4)
        else:
            _LOG.warning("HumotoDataset: no T-pose for seq %s (rig=%s), using identity", seq_id, rig_hash)
            body_ctx = torch.eye(4, dtype=torch.float32)[None, None].expand(
                _NUM_BODY_JOINTS, self.max_C, -1, -1,
            ).contiguous()  # (65, C, 4, 4)

        # Semantic types: joint 0 (Hips/pelvis) = SEM_HUMAN_GLOBAL, rest = SEM_HUMAN
        body_sem = np.full((_NUM_BODY_JOINTS,), SEM_HUMAN, dtype=np.int64)
        body_sem[0] = SEM_HUMAN_GLOBAL

        T_slice = body_T_w.shape[1]

        # Append body joints (group 0)
        prompt.append_trace(
            trace=body_T_w,                                            # (65, T_slice, 4, 4)
            trace_mask=torch.ones(_NUM_BODY_JOINTS, T_slice),          # (65, T_slice)
            context=body_ctx,                                          # (65, C, 4, 4)
            context_mask=torch.ones(_NUM_BODY_JOINTS, self.max_C),     # (65, C)
            group_id=0,
            sem_type=body_sem,
            uid=self.body_uids,
        )

        # --- Objects ---
        if "object_names" not in seq_data.files:
            _LOG.warning(
                "HumotoDataset: seq %s has no object_names key -- body-only sample",
                seq_id,
            )
            object_names = []
        else:
            object_names = seq_data["object_names"]
            if isinstance(object_names, np.ndarray):
                object_names = object_names.tolist()
        humoto_dict["object_names"] = list(object_names)

        n_objects = len(object_names)
        for obj_idx in range(n_objects):
            if obj_idx >= self.max_objects:
                break
            obj_name = str(object_names[obj_idx])

            # Object pose: (T_full, 4, 4) -> slice and stride
            obj_key = f"obj_{obj_name}_T_w"
            obj_T_w = seq_data[obj_key][start_tid:end_tid:factor]  # (T_slice, 4, 4)
            obj_se3 = torch.as_tensor(
                obj_T_w, dtype=torch.float32,
            ).unsqueeze(0)  # (1, T_slice, 4, 4)

            # Context: identity SE(3)
            obj_ctx = torch.eye(4, dtype=torch.float32)[None, None].expand(
                1, self.max_C, -1, -1,
            ).contiguous()  # (1, C, 4, 4)

            # Local point cloud: subsample from pre-loaded cache
            if obj_name in self.object_pcl_cache:
                pcl_raw = self.object_pcl_cache[obj_name]  # (16384, 6)
                pcl_sampled = sample_list(
                    pcl_raw, self.max_P_per_object, self.random_flag,
                )  # (max_P_per_object, 6)
                local_pcl = torch.as_tensor(
                    pcl_sampled, dtype=torch.float32,
                ).unsqueeze(0)  # (1, PCL, 6)
            else:
                _LOG.warning(
                    "HumotoDataset: object %r not in pcl cache for seq %s",
                    obj_name, seq_id,
                )
                local_pcl = torch.zeros(
                    1, self.max_P_per_object, 6, dtype=torch.float32,
                )

            local_pcl_mask = torch.ones(1)

            # Append object: group_id = obj_idx + 1 (1-based, body is 0)
            prompt.append_trace(
                trace=obj_se3,                                         # (1, T_slice, 4, 4)
                trace_mask=torch.ones(1, T_slice),                     # (1, T_slice)
                context=obj_ctx,                                       # (1, C, 4, 4)
                context_mask=torch.ones(1, self.max_C),                # (1, C)
                group_id=obj_idx + 1,
                sem_type=SEM_OBJECT,
                uid=self.object_uids[obj_idx],
                local_pcl=local_pcl,                                   # (1, PCL, 6)
                local_pcl_mask=local_pcl_mask,                         # (1,)
            )

        # Pad unused object slots (no trace data, just padding via catch_P)
        # catch_P will handle the remaining padding to reach P total

        return prompt, humoto_dict

    def __getitem_inner__(self, idx: int) -> Tuple[CanonicalDatasetOutput, Dict]:
        """Build a single HUMOTO V13 training sample.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, humoto_dict)``.
        """
        idx = idx % len(self)

        # P = 65 body joints + max_objects rigid objects
        P = _NUM_BODY_JOINTS + self.max_objects

        prompt = PromptV0(
            P=P,
            T=self.max_T,
            C=self.max_C,
            PCL=self.max_P_per_object,
            PCL_dim=6,
        )

        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag,
        )

        prompt, humoto_dict = self.__add_humoto_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid,
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask,
            context_mask=context_mask,
            sem_type=sem_type,
            current_ith=current_ith,
        )

        time_unit = float(self.t_rate) / float(self.base_fps)
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        text = self.text_cache.get(sid, "")

        local_pcl, local_pcl_mask = prompt.cat_pcl()

        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace,                     # (P, T, 4, 4)
            trace_mask=trace_mask,           # (P, T)
            context=context,                 # (P, C, 4, 4)
            context_mask=context_mask,       # (P, C)
            known_mask=known_mask,           # (P, T)
            context_known_mask=ctx_known_mask,  # (P, C)
            trace_group_id=group_id,         # (P,)
            trace_rot_mask=rot_mask,         # (P,)
            trace_sem_type=sem_type,         # (P,)
            trace_uid=uid,                   # (P,)
            time=time_stamp,                 # (T,)
            text=text,
            local_pcl=local_pcl,             # (P, PCL, 6)
            local_pcl_mask=local_pcl_mask,   # (P,)
        )
        return ret, humoto_dict


if __name__ == "__main__":
    from torch.utils.data import DataLoader, ConcatDataset
    from tqdm import tqdm

    dataset = HumotoDataset(
        data_root="../../data/hoi_all/humoto_processed",
        split_fn=osp.abspath(
            "../../data/hoi_all/humoto_processed/splits/train.json",
        ),
        base_fps=30,
        safe_flag=False,
        max_T=30,
        max_C=1,
        max_objects=4,
        max_P_per_object=256,
    )

    sample, extra = dataset[0]
    print(f"trace: {sample.trace.shape}")
    print(f"trace_mask: {sample.trace_mask.shape}")
    print(f"context: {sample.context.shape}")
    print(f"local_pcl: {sample.local_pcl.shape}")
    print(f"local_pcl_mask: {sample.local_pcl_mask.shape}")
    print(f"sem_type: {sample.trace_sem_type}")
    print(f"uid: {sample.trace_uid}")
    print(f"group_id: {sample.trace_group_id}")
    print(f"text: {sample.text}")
    print(f"extra: {extra}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)
        if i >= 2:
            break

    print(f"Dataset length: {len(dataset)}")
