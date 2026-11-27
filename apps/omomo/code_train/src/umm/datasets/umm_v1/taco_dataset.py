"""TACO hand-object interaction dataset.

Provides bimanual MANO hand joint trajectories alongside tool and target
object point-cloud traces for the TACO 3-D benchmark.

Example::

    dataset = TacoDataset(
        data_root="data/taco_3d",
        split_fn="data_splits/taco/taco_3d_train.json",
        base_fps=30,
        model_fn="object_models.npz",
        state_dirname="hoi_states",
    )
    sample, extra = dataset[0]
"""

################################################################################
# GeneMoma Dataset - Unified dataset for general motion modeling
################################################################################

import os
import os.path as osp
import sys
import json
import zipfile
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from copy import deepcopy

from umm.datasets.umm_v1.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v1.dataset_utils import sample_list
from umm.datasets.umm_v1.dataset_config import *


class TacoDataset(BaseDataset):
    """Dataset for TACO bimanual hand-object interaction sequences.

    Each sample contains 16 left-MANO and 16 right-MANO joint traces,
    plus tool and target object point-cloud traces sampled from meshes.

    Args:
        data_root: Root directory of the TACO dataset.
        split_fn: JSON split file.
        base_fps: Recording frame rate.
        random_flag: Stochastic augmentation flag.
        safe_flag: Error-handling wrapper.
        repeat: Repetition multiplier.
        t_rate: Temporal stride.
        language_fn: Optional language annotation file.
        max_T: Maximum time steps.
        max_C: Maximum context steps.
        max_P_per_object: Points per object cloud.
        model_fn: Object meshes archive filename.
        state_dirname: HOI states sub-directory.
        xyz_bias: 3-D translation bias applied to all poses.

    Example::

        ds = TacoDataset(
            data_root="data/taco_3d",
            split_fn="splits/taco_3d_train.json",
            base_fps=30,
            model_fn="object_models.npz",
            state_dirname="hoi_states",
        )
        sample, extra = ds[0]
    """

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps,  # ecah dataset must set manually
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
        language_fn: str | None = None,
        #############################################################
        max_T: int = 30,  # num of time step
        max_C: int = 8,  # num of context steps
        max_P_per_object: int = 32,  # num of prompts per object
        model_fn: str | None = None,
        state_dirname: str | None = None,
        xyz_bias=[0.0, 0.0, -0.4],
    ):
        """Initialize the TACO dataset and load object meshes.

        Args:
            data_root: Root data directory.
            split_fn: JSON split file path.
            base_fps: Frame rate.
            random_flag: Stochastic sampling.
            safe_flag: Safe ``__getitem__``.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            language_fn: Language annotation file.
            max_T: Max time steps.
            max_C: Max context steps.
            max_P_per_object: Points per object.
            model_fn: Mesh archive filename.
            state_dirname: States sub-directory.
            xyz_bias: 3-D bias vector ``[x, y, z]``.

        Example::

            ds = TacoDataset(
                data_root="data/taco_3d",
                split_fn="splits/train.json",
                base_fps=30,
                model_fn="object_models.npz",
                state_dirname="hoi_states",
            )
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn, random_flag, safe_flag, repeat
        )

        self.max_T = max_T
        self.max_C = max_C
        self.max_P_per_object = max_P_per_object
        self.model_fn = model_fn
        self.state_dirname = state_dirname
        self.xyz_bias = torch.as_tensor(xyz_bias).float()
        assert len(self.xyz_bias) == 3

        mesh = np.load(osp.join(self.data_root, model_fn))
        self.mesh = {k: mesh[k] for k in mesh.files}
        self.left_mano_uids = UID_MANO_LEFT_HAND
        self.right_mano_uids = UID_MANO_RIGHT_HAND
        self.tool_uid = UID_TACO_TOOL
        self.target_uid = UID_TACO_TARGET
        self.all_uids = (
            self.left_mano_uids
            + self.right_mano_uids
            + [self.tool_uid]
            + [self.target_uid]
        )
        self.sems = [SEM_LEFT_MANO, SEM_RIGHT_MANO, SEM_OBJECT]

        return

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal window for a training sample.

        Args:
            seq_id: Sequence identifier.
            current_abs_tid: Absolute anchor frame.
            dataset_idx: Global index (deterministic offset).
            random_enabled: Allow random jitter.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)``.

        Example::

            cur_i, s, e = ds.__decide_trace_sampling__("seq_0", 15, 0)
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

    def __decide_context_sampling__(self, idx):
        """Sample context sequence IDs and time steps.

        Args:
            idx: Current dataset index (for deterministic spacing).

        Returns:
            Tuple ``(ret_sid, ret_abs_tid)`` of lists with length ``max_C``.

        Example::

            ctx_sids, ctx_tids = ds.__decide_context_sampling__(0)
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

    def __add_taco_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
        ctx_sids,
        ctx_abs_tids,
    ):
        """Populate a PromptV0 with MANO hand joints and object traces.

        Loads left/right MANO joint poses, tool and target object meshes,
        constructs world-space point-cloud curves, and appends them.

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index within the slice.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.
            ctx_sids: Context sequence IDs.
            ctx_abs_tids: Context absolute time indices.

        Returns:
            Tuple ``(prompt, taco_dict)`` with updated prompt and extras
            (tool_id, target_id, betas).

        Example::

            prompt = PromptV0(P=96, T=30, C=8)
            prompt, extra = ds.__add_taco_prompt__(
                prompt, 5, 0, 30, "seq_0", ctx_sids, ctx_tids
            )
        """
        taco_dict = {}
        assert self.state_dirname is not None
        taco_data = np.load(
            osp.join(self.data_root, self.state_dirname, seq_id + ".npz")
        )
        factor = int(self.t_rate)

        Q_l = taco_data["Q_l"][start_tid:end_tid][::factor]  # T,K,4,4
        Q_r = taco_data["Q_r"][start_tid:end_tid][::factor]  # T,K,4,4
        t_l = taco_data["trans_l"][start_tid:end_tid][::factor]  # T,3
        t_r = taco_data["trans_r"][start_tid:end_tid][::factor]  # T,3
        tool_T_wl = taco_data["tool_T_wl"][start_tid:end_tid][::factor]
        target_T_wl = taco_data["target_T_wl"][start_tid:end_tid][::factor]

        tool_id = taco_data["tool_id"]
        tool_pcl = self.mesh[f"{tool_id}_pcl"]
        target_id = taco_data["target_id"]
        target_pcl = self.mesh[f"{target_id}_pcl"]
        taco_dict["tool_id"] = str(tool_id)
        taco_dict["target_id"] = str(target_id)

        Q_l[:, :, :3, 3] = Q_l[:, :, :3, 3] + t_l[:, None]  # T,16,4,4
        Q_r[:, :, :3, 3] = Q_r[:, :, :3, 3] + t_r[:, None]  # T,16,4,4
        Q_l = torch.as_tensor(Q_l).permute(1, 0, 2, 3).contiguous()  # 16,T,4,4
        Q_r = torch.as_tensor(Q_r).permute(1, 0, 2, 3).contiguous()  # 16,T,4,4

        taco_dict["beta_l"] = taco_data["beta_l"]
        taco_dict["beta_r"] = taco_data["beta_r"]

        # load context hand pose
        ctx_Q_l, ctx_Q_r = [], []
        for sid, abs_tid in zip(ctx_sids, ctx_abs_tids):
            ctx_data = np.load(
                osp.join(self.data_root, self.state_dirname, sid + ".npz")
            )
            _ctx_Q_l = ctx_data["Q_l"][abs_tid]  # K,4,4
            _ctx_t_l = ctx_data["trans_l"][abs_tid]  # 3
            _ctx_Q_r = ctx_data["Q_r"][abs_tid]  # K,4,4
            _ctx_t_r = ctx_data["trans_r"][abs_tid]  # 3
            _ctx_Q_l[:, :3, 3] = _ctx_Q_l[:, :3, 3] + _ctx_t_l[None]
            _ctx_Q_r[:, :3, 3] = _ctx_Q_r[:, :3, 3] + _ctx_t_r[None]
            ctx_Q_l.append(_ctx_Q_l)  # K,4,4
            ctx_Q_r.append(_ctx_Q_r)  # K,4,4
        ctx_Q_l = (
            torch.as_tensor(np.stack(ctx_Q_l, axis=0)).permute(1, 0, 2, 3).contiguous()
        )  # 16,C,4,4
        ctx_Q_r = (
            torch.as_tensor(np.stack(ctx_Q_r, axis=0)).permute(1, 0, 2, 3).contiguous()
        )  # 16,C,4,4

        Q_l[:, :, :3, 3] = Q_l[:, :, :3, 3] + self.xyz_bias[None, None]
        Q_r[:, :, :3, 3] = Q_r[:, :, :3, 3] + self.xyz_bias[None, None]
        ctx_Q_l[:, :, :3, 3] = ctx_Q_l[:, :, :3, 3] + self.xyz_bias[None, None]
        ctx_Q_r[:, :, :3, 3] = ctx_Q_r[:, :, :3, 3] + self.xyz_bias[None, None]

        # append hands
        prompt.append_trace(
            trace=Q_l,
            trace_mask=torch.ones(Q_l.shape[:2]),
            context=ctx_Q_l,
            context_mask=torch.ones(ctx_Q_l.shape[:2]),
            group_id=0,
            sem_type=SEM_LEFT_MANO,
            uid=self.left_mano_uids,
        )
        prompt.append_trace(
            trace=Q_r,
            trace_mask=torch.ones(Q_r.shape[:2]),
            context=ctx_Q_r,
            context_mask=torch.ones(ctx_Q_r.shape[:2]),
            group_id=1,
            sem_type=SEM_RIGHT_MANO,
            uid=self.right_mano_uids,
        )

        # add tool and target
        tool_sampled = sample_list(
            self.mesh[f"{taco_data['tool_id']}_pcl"],
            self.max_P_per_object,
            self.random_flag,
        )
        target_sampled = sample_list(
            self.mesh[f"{taco_data['target_id']}_pcl"],
            self.max_P_per_object,
            self.random_flag,
        )
        tool_curve_xyz = (
            np.einsum("tij,nj->tni", tool_T_wl[:, :3, :3], tool_sampled)
            + tool_T_wl[:, :3, 3][:, None]
        )
        target_curve_xyz = (
            np.einsum("tij,nj->tni", target_T_wl[:, :3, :3], target_sampled)
            + target_T_wl[:, :3, 3][:, None]
        )
        tool_sampled = torch.as_tensor(tool_sampled)[:, None].expand(
            -1, self.max_C, -1
        )  # P,C,3
        target_sampled = torch.as_tensor(target_sampled)[:, None].expand(
            -1, self.max_C, -1
        )  # P,C,3
        tool_curve_xyz = torch.as_tensor(tool_curve_xyz).permute(1, 0, 2)  # C,P,3
        target_curve_xyz = torch.as_tensor(target_curve_xyz).permute(1, 0, 2)  # C,P,3

        tool_curve_xyz = tool_curve_xyz + self.xyz_bias[None, None]
        target_curve_xyz = target_curve_xyz + self.xyz_bias[None, None]

        prompt.append_trace(
            trace=tool_curve_xyz,
            trace_mask=torch.ones(tool_curve_xyz.shape[:2]),
            context=tool_sampled,
            context_mask=torch.ones(tool_sampled.shape[:2]),
            group_id=2,
            sem_type=SEM_OBJECT,
            uid=self.tool_uid,
        )
        prompt.append_trace(
            trace=target_curve_xyz,
            trace_mask=torch.ones(target_curve_xyz.shape[:2]),
            context=target_sampled,
            context_mask=torch.ones(target_sampled.shape[:2]),
            group_id=3,
            sem_type=SEM_OBJECT,
            uid=self.target_uid,
        )
        return prompt, taco_dict

    def __getitem_inner__(self, idx):
        """Build a single TACO training sample.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, taco_dict)``.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)
        """

        idx = idx % len(self)
        prompt = PromptV0(
            P=16 + 16 + 2 * self.max_P_per_object, T=self.max_T, C=self.max_C
        )
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )
        ctx_sids, ctx_abs_tids = self.__decide_context_sampling__(idx)

        prompt, taco_dict = self.__add_taco_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid, ctx_sids, ctx_abs_tids
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        if self.language_data is not None:
            text = self.language_data[sid]
        else:
            text = ""

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
        return ret, taco_dict


if __name__ == "__main__":

    dataset = TacoDataset(
        data_root="../../data/taco_3d",
        split_fn=osp.abspath("../../data_splits/taco/taco_3d_train.json"),
        base_fps=30,
        # random_flag=False,
        safe_flag=False,
        max_T=30,
        max_C=8,
        max_P_per_object=32,
        model_fn="object_models.npz",
        state_dirname="hoi_states",
    )

    dataset[10]

    from torch.utils.data import DataLoader

    dataset2 = TacoDataset(
        data_root="../../data/taco_3d",
        split_fn=osp.abspath("../../data_splits/taco/taco_3d_train.json"),
        base_fps=30,
        # random_flag=False,
        safe_flag=False,
        max_T=20,
        max_C=5,
        max_P_per_object=17,
        model_fn="object_models.npz",
        state_dirname="hoi_states",
    )

    # TODO: check with two different dataset

    cat_dataset = ConcatDataset([dataset, dataset2])

    dataloader = DataLoader(
        cat_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)

    dataset.print_profiling_summary()
    print(f"Dataset length: {len(dataset)}")
