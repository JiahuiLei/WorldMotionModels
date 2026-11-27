"""OMOMO human-object interaction dataset for UMM v1.

Each sample is one 120-frame canonical window from the OMOMO benchmark.
The converter (``prepare_data/omomo/convert.py``) produces the .npz files
consumed here.

Entities per sample (4 total):
  - 0: Root SE3 trace (root_orient_aa + root_trans)       UID_OMOMO_ROOT
  - 1: Left-wrist position trace (position-only)          UID_OMOMO_LHAND
  - 2: Right-wrist position trace (position-only)         UID_OMOMO_RHAND
  - 3: Object SE3 trace (obj_rot + obj_trans)             UID_OMOMO_OBJECT

``extra`` dict keys:
  seq_name, betas, gender, obj_mesh_path, obj_bps, text, actual_len,
  waypoint_indices, first_pose, pose_body_aa, root_orient_aa, root_trans.

Example::

    ds = OmomoDataset(
        data_root="data/omomo/umm/test",
        split_fn="data_splits/omomo/omomo_test.json",
        base_fps=30,
        safe_flag=False,
    )
    sample, extra = ds[0]
    print(sample.trace.shape)   # (4, 120, 4, 4)
"""

import logging
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import pytorch3d.transforms as p3d_transforms

from umm.datasets.umm_v1.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v1.dataset_config import SEM_HUMAN, SEM_OBJECT

# ---------------------------------------------------------------------------
# OMOMO-local UIDs (500-503, not in dataset_config.py)
# ---------------------------------------------------------------------------

UID_OMOMO_ROOT = 500    # SMPL root joint SE3
UID_OMOMO_LHAND = 501   # Left wrist position-only
UID_OMOMO_RHAND = 502   # Right wrist position-only
UID_OMOMO_OBJECT = 503  # Held object SE3

_logger = logging.getLogger(__name__)


def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix.

    Args:
        aa: Axis-angle array, shape ``(..., 3)``.

    Returns:
        Rotation matrices, shape ``(..., 3, 3)``.
    """
    t = torch.from_numpy(aa).float()
    mat = p3d_transforms.axis_angle_to_matrix(t)
    return mat.numpy()


def _build_se3(rot_mat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Pack rotation matrix + translation into 4x4 SE3 matrices.

    Args:
        rot_mat: Rotation matrices, shape ``(T, 3, 3)``.
        trans: Translation vectors, shape ``(T, 3)``.

    Returns:
        SE3 matrices, shape ``(T, 4, 4)``.
    """
    T = rot_mat.shape[0]
    se3 = np.eye(4, dtype=np.float32)[None].repeat(T, axis=0)  # (T, 4, 4)
    se3[:, :3, :3] = rot_mat
    se3[:, :3, 3] = trans
    return se3


class OmomoDataset(BaseDataset):
    """OMOMO human-object interaction dataset.

    Loads pre-converted .npz windows from ``data/omomo/umm/{split}/``.
    Each window is treated as a single fixed-length (T=120) sequence.

    Args:
        data_root: Directory containing ``{stem}.npz`` files.
        split_fn: JSON file listing ``[[stem, actual_len], ...]`` pairs.
        base_fps: Native frame rate (30 for OMOMO).
        random_flag: Enable stochastic augmentation.
        safe_flag: Skip bad samples silently when True.
        repeat: Repetition multiplier for the index list.
        t_rate: Temporal stride.
        language_fn: Optional language annotation file.
        max_T: Maximum time steps (must be >= 120 for OMOMO).
        max_C: Maximum context steps.
        max_P_per_object: Points per object (unused; kept for API parity).

    Example::

        ds = OmomoDataset(
            data_root="data/omomo/umm/test",
            split_fn="data_splits/omomo/omomo_test.json",
            base_fps=30,
        )
        sample, extra = ds[0]
        print(sample.trace.shape)  # (4, 120, 4, 4)
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
        max_T: int = 120,
        max_C: int = 8,
        max_P_per_object: int = 32,
        use_omomo_native: bool = False,
    ) -> None:
        """Initialize OmomoDataset.

        Args:
            data_root: Root directory containing .npz windows.
            split_fn: JSON split file with ``[[stem, actual_len], ...]`` pairs.
            base_fps: Native recording frame rate.
            random_flag: Apply random temporal augmentation.
            safe_flag: Silently skip errors in __getitem__.
            repeat: Repeat the index list this many times.
            t_rate: Temporal stride for index expansion.
            language_fn: Optional external language annotation file.
            max_T: Max time steps; pads/crops to this length.
            max_C: Max context steps.
            max_P_per_object: Points per object (API compatibility only).
            use_omomo_native: If True, signals the caller expects the OMOMO-native
                protocol (15 objects, 951 test / 15976 train windows). Reserved
                for future multi-protocol support; in the current v2 build the
                on-disk data is CHOIS-Table-1 protocol (13 objects, 482 / 3873).
                When ``True`` the caller must also point ``data_root`` +
                ``split_fn`` at OMOMO-native outputs (regenerate via
                ``--protocol native``). Default False (CHOIS Table 1).
        """
        self.use_omomo_native = use_omomo_native
        super().__init__(
            data_root=data_root,
            split_fn=split_fn,
            base_fps=base_fps,
            language_fn=language_fn,
            random_flag=random_flag,
            safe_flag=safe_flag,
            repeat=repeat,
            t_rate=t_rate,
        )
        self.max_T = max_T
        self.max_C = max_C
        self.max_P_per_object = max_P_per_object

        # Collapse frame-level indices (one per timestep from load_indices) into
        # window-level indices (one per .npz). Each OMOMO .npz IS one fixed-T
        # window; there is no intra-window sampling. len(ds) = unique windows.
        unique_stems: List[str] = []
        seen = set()
        for sid, _tid in self.indices:
            if sid not in seen:
                unique_stems.append(sid)
                seen.add(sid)
        self.indices = [(s, 0) for s in unique_stems]
        self.indices = self.indices * max(1, int(self.repeat))

    def __getitem_inner__(self, idx: int) -> Tuple[CanonicalDatasetOutput, Dict]:
        """Build one OMOMO training sample.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of ``(CanonicalDatasetOutput, extra_dict)``.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)  # (4, 120, 4, 4)
        """
        idx = idx % len(self)
        stem, _ = self.indices[idx]
        npz_path = osp.join(self.data_root, f"{stem}.npz")
        npz = np.load(npz_path, allow_pickle=True)

        T = int(npz["T"])
        actual_len = int(npz["actual_len"])

        # ---- load padded time-varying arrays -------------------------
        root_trans = npz["root_trans"].astype(np.float32)          # (T, 3)
        root_orient_aa = npz["root_orient_aa"].astype(np.float32)  # (T, 3)
        pose_body_aa = npz["pose_body_aa"].astype(np.float32)      # (T, 21, 3)
        lhand_wrist = npz["lhand_wrist_jpos"].astype(np.float32)   # (T, 3)
        rhand_wrist = npz["rhand_wrist_jpos"].astype(np.float32)   # (T, 3)
        obj_trans = npz["obj_trans"].astype(np.float32)            # (T, 3)
        obj_rot = npz["obj_rot"].astype(np.float32)                # (T, 3, 3)

        # ---- validity mask (1 for valid frames, 0 for padding) ------
        mask = np.zeros(T, dtype=np.float32)
        mask[:actual_len] = 1.0

        # ---- build SE3 for root (entity 0) ---------------------------
        root_rotmat = _aa_to_rotmat(root_orient_aa)  # (T, 3, 3)
        root_se3 = _build_se3(root_rotmat, root_trans)  # (T, 4, 4)

        # ---- build SE3 for object (entity 3) -------------------------
        obj_se3 = _build_se3(obj_rot, obj_trans)  # (T, 4, 4)

        # ---- convert to (n, T, ...) for PromptV0.append_trace --------
        # root SE3: (1, T, 4, 4)
        root_trace = torch.as_tensor(root_se3).unsqueeze(0)          # (1, T, 4, 4)
        root_mask = torch.as_tensor(mask).unsqueeze(0)                # (1, T)
        root_ctx = root_trace[:, :self.max_C]                         # (1, C, 4, 4)
        root_ctx_mask = root_mask[:, :self.max_C]                     # (1, C)

        # lhand position-only: (1, T, 3)
        lhand_trace = torch.as_tensor(lhand_wrist).unsqueeze(0)       # (1, T, 3)
        lhand_mask = torch.as_tensor(mask).unsqueeze(0)               # (1, T)
        lhand_ctx = lhand_trace[:, :self.max_C]                       # (1, C, 3)
        lhand_ctx_mask = lhand_mask[:, :self.max_C]                   # (1, C)

        # rhand position-only: (1, T, 3)
        rhand_trace = torch.as_tensor(rhand_wrist).unsqueeze(0)       # (1, T, 3)
        rhand_mask = torch.as_tensor(mask).unsqueeze(0)               # (1, T)
        rhand_ctx = rhand_trace[:, :self.max_C]                       # (1, C, 3)
        rhand_ctx_mask = rhand_mask[:, :self.max_C]                   # (1, C)

        # object SE3: (1, T, 4, 4)
        obj_trace = torch.as_tensor(obj_se3).unsqueeze(0)             # (1, T, 4, 4)
        obj_mask = torch.as_tensor(mask).unsqueeze(0)                 # (1, T)
        obj_ctx = obj_trace[:, :self.max_C]                           # (1, C, 4, 4)
        obj_ctx_mask = obj_mask[:, :self.max_C]                       # (1, C)

        # ---- build prompt -------------------------------------------
        prompt = PromptV0(P=4, T=self.max_T, C=self.max_C)

        prompt.append_trace(
            trace=root_trace,
            trace_mask=root_mask,
            context=root_ctx,
            context_mask=root_ctx_mask,
            group_id=0,
            sem_type=SEM_HUMAN,
            uid=UID_OMOMO_ROOT,
        )
        prompt.append_trace(
            trace=lhand_trace,
            trace_mask=lhand_mask,
            context=lhand_ctx,
            context_mask=lhand_ctx_mask,
            group_id=1,
            sem_type=SEM_HUMAN,
            uid=UID_OMOMO_LHAND,
        )
        prompt.append_trace(
            trace=rhand_trace,
            trace_mask=rhand_mask,
            context=rhand_ctx,
            context_mask=rhand_ctx_mask,
            group_id=2,
            sem_type=SEM_HUMAN,
            uid=UID_OMOMO_RHAND,
        )
        prompt.append_trace(
            trace=obj_trace,
            trace_mask=obj_mask,
            context=obj_ctx,
            context_mask=obj_ctx_mask,
            group_id=3,
            sem_type=SEM_OBJECT,
            uid=UID_OMOMO_OBJECT,
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        # ---- time stamp (relative to frame 0 as anchor) --------------
        time_unit = 1.0 / float(self.base_fps)
        time_stamp = torch.arange(self.max_T).float() * time_unit  # (T,)

        # ---- text ----------------------------------------------------
        if self.language_data is not None:
            text = self.language_data.get(stem, "")
        else:
            text = str(npz["text"].item())

        # ---- static / extra -----------------------------------------
        betas = npz["betas"].astype(np.float32)              # (16,)
        obj_bps = torch.as_tensor(npz["obj_bps"].astype(np.float32))  # (1024, 3)
        waypoint_indices = npz["waypoint_indices"]            # (5,) int32
        gender = str(npz["gender"].item())
        obj_mesh_path = str(npz["obj_mesh_path"].item())
        seq_name_str = str(npz["seq_name"].item())

        first_pose = {
            "root_trans": npz["first_root_trans"].astype(np.float32),          # (3,)
            "root_orient_aa": npz["first_root_orient_aa"].astype(np.float32),  # (3,)
            "pose_body_aa": npz["first_pose_body_aa"].astype(np.float32),      # (21, 3)
        }

        extra = {
            "seq_name": stem,
            "betas": betas,
            "gender": gender,
            "obj_mesh_path": obj_mesh_path,
            "obj_bps": obj_bps,
            "text": text,
            "actual_len": actual_len,
            "waypoint_indices": waypoint_indices,
            "first_pose": first_pose,
            "pose_body_aa": pose_body_aa,       # (T, 21, 3) padded
            "root_orient_aa": root_orient_aa,   # (T, 3) padded
            "root_trans": root_trans,           # (T, 3) padded
        }

        ret = CanonicalDatasetOutput(
            seq_name=stem,
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
        return ret, extra


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    ds = OmomoDataset(
        data_root="data/omomo/umm/test",
        split_fn="data_splits/omomo/omomo_test.json",
        base_fps=30,
        safe_flag=False,
    )
    print(f"Dataset length: {len(ds)}")
    sample, extra = ds[0]
    print(f"trace shape: {sample.trace.shape}")
    print(f"trace_mask shape: {sample.trace_mask.shape}")
    print(f"context shape: {sample.context.shape}")
    print(f"rot_mask: {sample.trace_rot_mask}")
    print(f"sem_type: {sample.trace_sem_type}")
    print(f"uid: {sample.trace_uid}")
    print(f"actual_len: {extra['actual_len']}")
    print(f"gender: {extra['gender']}")
    print(f"text: {extra['text'][:80]!r}")
    print(f"obj_bps shape: {extra['obj_bps'].shape}")
