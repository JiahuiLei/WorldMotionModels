"""Motion-planning dataset for multi-robot collision-free trajectory generation.

Caches robot link poses and box-obstacle data in RAM.  Supports per-link
group IDs, optional point-cloud representations for links, obstacle pose
tracking, and configurable scale normalization.

Example::

    dataset = MotionPlanningDataset(
        data_root="data/pyroki_mp/move_across_moving_box2/",
        split_fn="data_splits/mp_move_across_moving_box_v1/train.json",
        base_fps=30,
        max_T=50,
        max_C=8,
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
import logging

from umm.datasets.umm_v9.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v9.dataset_utils import sample_list
from umm.datasets.umm_v9.dataset_config import *


class MotionPlanningDataset(BaseDataset):
    """Dataset for robot motion-planning trajectories with obstacles.

    Caches all per-robot data (link poses, obstacle poses/pcls) in RAM at
    init.  Supports per-link group IDs, link-level point clouds, optional
    obstacle pose tracking, neutral-pose context, and scale normalization.

    Args:
        data_root: Root directory of the motion-planning dataset.
        split_fn: JSON split file.
        base_fps: Recording frame rate.
        random_flag: Stochastic augmentation.
        safe_flag: Error-handling wrapper.
        repeat: Repetition multiplier.
        t_rate: Temporal stride.
        language_fn: Optional language annotation file.
        max_T: Maximum time steps.
        max_C: Maximum context steps.
        max_P_per_object: Obstacle point-cloud size.
        link_per_group_flag: Assign unique group ID per link.
        n_pcl_per_link: If >0, add explicit link point-cloud traces.
        local_pcl_N: Points per local link point cloud (0 = disabled).
        obst_pose_flag: Use SE(3) obstacle pose instead of point curves.
        scale_normalize_range: ``[lo, hi]`` random scale range; disabled
            if ``lo <= 0``.
        first_ctx_neutral_flag: Replace the first context with neutral pose.

    Example::

        ds = MotionPlanningDataset(
            data_root="data/pyroki_mp/move_across_moving_box2/",
            split_fn="splits/train.json",
            base_fps=30,
            max_T=50,
            max_C=8,
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
        link_per_group_flag=False,
        n_pcl_per_link: int = -1,
        local_pcl_N: int = 128,
        obst_pose_flag: bool = False,
        #
        scale_normalize_range=[-1, -1],
        #
        first_ctx_neutral_flag=False,
    ):
        """Initialize the motion-planning dataset and cache robot data to RAM.

        Reads all per-robot ``.npz`` archives and converts to torch tensors.

        Args:
            data_root: Root data directory.
            split_fn: JSON split file path.
            base_fps: Frame rate.
            random_flag: Stochastic sampling.
            safe_flag: Error-handling wrapper.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            language_fn: Language annotation file.
            max_T: Max time steps.
            max_C: Max context steps.
            max_P_per_object: Obstacle cloud size.
            link_per_group_flag: Per-link group IDs.
            n_pcl_per_link: Link PCL trace size.
            local_pcl_N: Local PCL points.
            obst_pose_flag: Obstacle pose mode.
            scale_normalize_range: Scale range ``[lo, hi]``.
            first_ctx_neutral_flag: Neutral first context.

        Example::

            ds = MotionPlanningDataset(
                data_root="data/pyroki_mp/task/",
                split_fn="splits/train.json",
                base_fps=30,
                max_T=50,
            )
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn, random_flag, safe_flag, repeat
        )

        self.max_T = max_T
        self.max_C = max_C
        self.max_P_per_object = max_P_per_object
        self.n_pcl_per_link = n_pcl_per_link
        self.local_pcl_N = local_pcl_N
        if self.local_pcl_N > 0:
            logging.info(f"local_pcl_N is set to {self.local_pcl_N}")
        if self.n_pcl_per_link > 0:
            logging.warning(f"n_pcl_per_link is set to {self.n_pcl_per_link}")
        self.link_per_group_flag = link_per_group_flag
        if self.link_per_group_flag:
            logging.warning(
                "link_per_group_flag is set to True, assign unique GID to each link"
            )
        self.first_ctx_neutral_flag = first_ctx_neutral_flag
        if self.first_ctx_neutral_flag:
            logging.warning(
                "first_ctx_neutral_flag is set to True, add neutral context at the first step"
            )
        self.obst_pose_flag = obst_pose_flag
        if self.obst_pose_flag:
            logging.warning("obst_pose_flag=True")

        # cache sub robot list
        self.robot_sub_indices = {}
        for i, it in enumerate(self.indices):
            robot = it[0].split("___")[0]
            if robot not in self.robot_sub_indices.keys():
                self.robot_sub_indices[robot] = [it]
            else:
                self.robot_sub_indices[robot].append(it)
        for name in self.robot_sub_indices.keys():
            logging.info(f"{name}: {len(self.robot_sub_indices[name])}")

        # directly cache to RAM, and warning if missing robot
        self.data = {}
        self.uids = {}
        print("Loading MP data, caching into RAM ...")
        for robot in tqdm(self.robot_sub_indices.keys()):
            fn = osp.join(self.data_root, robot + ".npz")
            # ! for now must require this file
            assert osp.exists(fn), f"Missing robot: {robot}"
            _data = np.load(fn, allow_pickle=True)
            data = {}
            for file in _data.files:
                d = _data[file]
                if file == "part_names":
                    data[file] = d.tolist()
                elif file in ["mesh_vs", "mesh_fs", "mesh_colors", "mesh_pcls"]:
                    data[file] = [
                        torch.as_tensor(_d.astype(np.float32)).float() for _d in d
                    ]
                elif isinstance(d, np.ndarray):
                    d = d.astype(np.float32)
                    data[file] = torch.as_tensor(d).float()
                else:
                    data[file] = d
            self.data[robot] = data
            self.uids[robot] = UID_MANIPULATOR_DICT[robot]
        self.obstacle_uid = UID_MOTIONPLANNING_BOX_OBSTACLE

        self.scale_normalize_range = scale_normalize_range
        self.scale_normalize_flag = scale_normalize_range[0] > 0
        if self.scale_normalize_flag:
            logging.warning(
                f"Scale normalize flag is set to True, range: {self.scale_normalize_range}"
            )

        return

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal window for a sample.

        If the full sequence fits in ``max_T * factor`` frames the entire
        sequence is returned. Otherwise a random or deterministic window
        is selected around ``current_abs_tid``.

        Args:
            seq_id: Sequence identifier.
            current_abs_tid: Absolute anchor frame.
            dataset_idx: Global index for deterministic offset.
            random_enabled: Allow random jitter.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)``.

        Example::

            cur_i, s, e = ds.__decide_trace_sampling__("panda___0", 10, 0)
        """
        factor = int(self.t_rate)
        seq_len = self.seq_len[seq_id]

        if self.max_T * factor >= seq_len:
            # return fixed sampling
            # ! here no matter where the current tid is, always sample the full seq
            return current_abs_tid // factor, 0, seq_len

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
        """Sample context sequences from the same robot type.

        Args:
            idx: Current dataset index.

        Returns:
            Tuple ``(ret_sid, ret_abs_tid)`` of lists with length ``max_C``.

        Example::

            ctx_sids, ctx_tids = ds.__decide_context_sampling__(0)
        """
        robot = self.indices[idx][0].split("___")[0]
        robot_sublist = self.robot_sub_indices[robot]

        # ! must sample inside the same robot!!
        if self.random_flag:
            choice = np.random.choice(len(robot_sublist), self.max_C, replace=True)
        else:
            step = max(1, len(robot_sublist) // (1 + self.max_C))
            start = idx % step
            choice = np.arange(start, len(robot_sublist), step)
            choice = choice[: self.max_C]
        ret_sid, ret_abs_tid = [], []
        for i in choice:
            sid, current_abs_tid = robot_sublist[i]
            ret_sid.append(sid)
            ret_abs_tid.append(current_abs_tid)

        if self.first_ctx_neutral_flag:
            ret_sid[0] = None
            ret_abs_tid[0] = None

        return ret_sid, ret_abs_tid

    def __load_from_cache__(self, sid):
        """Load robot link poses, obstacle pose, and obstacle PCL from RAM cache.

        Args:
            sid: Sequence identifier in ``<robot>___<index>`` format.

        Returns:
            Tuple ``(r_Q, o_Q, o_pcl)`` of tensors for robot links
            ``(T, M, 4, 4)``, obstacle ``(T, 4, 4)``, and obstacle
            cloud ``(N, 3)``.

        Example::

            r_Q, o_Q, o_pcl = ds.__load_from_cache__("panda___0")
        """
        robot = sid.split("___")[0]
        ind = int(sid.split("___")[1])
        part_names = self.data[robot]["part_names"]
        # todo: later can also load the PCL of the parts when the bandwith is large
        r_Q = self.data[robot]["link_poses"][ind]  # T,M,4,4
        o_Q = self.data[robot]["obstacle_poses"][ind]  # T,4,4
        o_pcl = self.data[robot]["obstacle_pcls"][ind]  # N,3
        return r_Q, o_Q, o_pcl

    def __add_mp_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
        ctx_sids,
        ctx_abs_tids,
    ):
        """Populate a PromptV0 with manipulator link and obstacle traces.

        Loads cached data, constructs robot link traces (optionally with
        per-link PCL curves), and appends the obstacle as either an SE(3)
        pose or a point-cloud curve.

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.
            ctx_sids: Context sequence IDs.
            ctx_abs_tids: Context time indices.

        Returns:
            Tuple ``(prompt, extra_dict)`` with ``manipulator_type``.

        Example::

            prompt = PromptV0(P=1, T=50, C=8, PCL=128)
            prompt, extra = ds.__add_mp_prompt__(
                prompt, 5, 0, 50, "panda___0", ctx_sids, ctx_tids
            )
        """

        robot = seq_id.split("___")[0]
        extra_dict = {"manipulator_type": robot}

        factor = int(self.t_rate)

        Q_robot, Q_obstacle, pcl_obstacle = self.__load_from_cache__(seq_id)
        Q_robot = Q_robot[start_tid:end_tid][::factor]  # T,M,4,4
        Q_obstacle = Q_obstacle[start_tid:end_tid][::factor]  # T,4,4

        ctx_Q_robot = []
        for seq, abs_tid in zip(ctx_sids, ctx_abs_tids):
            if seq is None:
                _Q_r = self.data[robot]["neutral_link_poses"]
            else:
                _Q_r, _, _ = self.__load_from_cache__(seq)
                _Q_r = _Q_r[abs_tid]
            ctx_Q_robot.append(_Q_r)
        ctx_Q_robot = torch.stack(ctx_Q_robot, dim=0).permute(1, 0, 2, 3)  # C,M,4,4

        Q_robot = Q_robot.permute(1, 0, 2, 3)
        if self.link_per_group_flag:
            group_id = torch.arange(Q_robot.shape[0]) + prompt.max_group_id + 1
        else:
            group_id = prompt.max_group_id + 1

        # get link pcl
        if self.local_pcl_N > 0:
            local_pcl = []
            for lid in range(Q_robot.shape[0]):
                _link_pcl = self.data[robot]["mesh_pcls"][lid].clone()
                _pcl = sample_list(
                    _link_pcl,
                    self.local_pcl_N,
                    self.random_flag,
                )  # PCL,3
                local_pcl.append(_pcl)
            local_pcl = torch.stack(local_pcl, dim=0)  # L,PCL,3
        else:
            local_pcl = torch.zeros(len(Q_robot), 0, 3)

        prompt.append_trace(
            trace=Q_robot,
            trace_mask=torch.ones(Q_robot.shape[:2]),
            context=ctx_Q_robot,
            context_mask=torch.ones(ctx_Q_robot.shape[:2]),
            group_id=prompt.max_group_id + 1,
            sem_type=SEM_MANIPULATOR,
            uid=self.uids[robot],
            local_pcl=local_pcl,
            local_pcl_mask=torch.ones(len(local_pcl)),
        )

        # add the link pcl if necessary
        if self.n_pcl_per_link > 0:
            link_ctx, link_trace = [], []
            link_group, link_uid = [], []
            for lid in range(Q_robot.shape[0]):
                _T = Q_robot[lid]  # T,4,4
                _link_pcl = self.data[robot]["mesh_pcls"][lid].clone()
                _pcl = sample_list(
                    _link_pcl,
                    self.n_pcl_per_link,
                    self.random_flag,
                )  # P,3
                link_trace.append(
                    torch.einsum("tij,pj->tpi", _T[:, :3, :3], _pcl)
                    + _T[:, :3, 3][:, None]
                )
                if self.link_per_group_flag:
                    link_group.append(torch.full((self.n_pcl_per_link,), group_id[lid]))
                else:
                    link_group.append(torch.full((self.n_pcl_per_link,), group_id))
                link_uid.append(
                    torch.full((self.n_pcl_per_link,), self.uids[robot][lid])
                )

                _ctx_T = ctx_Q_robot[lid]  # T,4,4
                link_ctx.append(
                    torch.einsum("tij,pj->tpi", _ctx_T[:, :3, :3], _pcl)
                    + _ctx_T[:, :3, 3][:, None]
                )

            link_trace = torch.cat(link_trace, dim=1)  # N,T,3
            link_ctx = torch.cat(link_ctx, dim=1)  # N,C,3
            link_group = torch.cat(link_group, dim=0)  # N
            link_uid = torch.cat(link_uid, dim=0)  # N

            link_trace = link_trace.permute(1, 0, 2)  # N,T,3
            link_ctx = link_ctx.permute(1, 0, 2)  # N,C,3

            prompt.append_trace(
                trace=link_trace,
                trace_mask=torch.ones(link_trace.shape[:2]),
                context=link_ctx,
                context_mask=torch.ones(link_ctx.shape[:2]),
                group_id=link_group,
                sem_type=SEM_MANIPULATOR,
                uid=link_uid,
            )

        # * obs
        if self.obst_pose_flag:
            # 1,P,3
            local_pcl = sample_list(
                pcl_obstacle,
                self.local_pcl_N,
                self.random_flag,
            )[None]
            prompt.append_trace(
                trace=Q_obstacle[None],  # 1,T,4,4
                trace_mask=torch.ones(1, len(Q_obstacle)),
                context=torch.eye(4)[None, None].expand(-1, self.max_C, -1, -1),
                context_mask=torch.ones(1, self.max_C),
                group_id=prompt.max_group_id + 1,
                sem_type=SEM_OBJECT,
                uid=self.obstacle_uid,
                local_pcl=local_pcl,
                local_pcl_mask=torch.ones(1),
            )
        else:
            obs_pcl = sample_list(
                pcl_obstacle,
                self.max_P_per_object,
                self.random_flag,
            )  # P,3
            obs_curve = (
                torch.einsum("tij,pj->tpi", Q_obstacle[:, :3, :3], obs_pcl)
                + Q_obstacle[:, :3, 3][:, None]
            )  # T,P,3
            obs_curve = obs_curve.permute(1, 0, 2)  # P,T,3
            obs_ctx = obs_pcl[:, None].repeat(1, self.max_C, 1)  # P,C,3
            prompt.append_trace(
                trace=obs_curve,
                trace_mask=torch.ones(obs_curve.shape[:2]),
                context=obs_ctx,
                context_mask=torch.ones(obs_ctx.shape[:2]),
                group_id=prompt.max_group_id + 1,
                sem_type=SEM_OBJECT,
                uid=self.obstacle_uid,
            )
        return prompt, extra_dict

    def __getitem_inner__(self, idx):
        """Build a single motion-planning training sample.

        Optionally applies scale normalization to the final traces.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_dict)``.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)
        """

        idx = idx % len(self)
        prompt = PromptV0(P=1, T=self.max_T, C=self.max_C, PCL=self.local_pcl_N)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )
        ctx_sids, ctx_abs_tids = self.__decide_context_sampling__(idx)

        prompt, extra_dict = self.__add_mp_prompt__(
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

        pcl, pcl_mask = prompt.cat_pcl()

        # * handle the scale
        if self.scale_normalize_flag:
            if self.random_flag:
                scale = (
                    torch.rand(1)
                    * (self.scale_normalize_range[1] - self.scale_normalize_range[0])
                    + self.scale_normalize_range[0]
                )
            else:
                scale = (
                    self.scale_normalize_range[1] + self.scale_normalize_range[0]
                ) / 2
            cur_pos = trace[:, current_ith, :3, 3]
            cur_pos_mean = cur_pos.mean(dim=0)
            cur_pos_radius = (cur_pos - cur_pos_mean[None]).norm(dim=-1)
            cur_radius = float(cur_pos_radius.median())
            scale_factor = float(scale / cur_radius)

            trace[..., :3, 3] = trace[..., :3, 3] * scale_factor
            context[..., :3, 3] = context[..., :3, 3] * scale_factor

            pcl = pcl * scale_factor
        else:
            scale_factor = 1.0
        extra_dict["scale_factor"] = scale_factor

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
            local_pcl=pcl,
            local_pcl_mask=pcl_mask,
        )
        return ret, extra_dict


if __name__ == "__main__":

    dataset = MotionPlanningDataset(
        data_root="../../data/pyroki_mp/move_across_moving_box2/",
        split_fn=osp.abspath(
            "../../data_splits/mp_move_across_moving_box_v1/train.json"
        ),
        base_fps=30,
        # random_flag=False,
        safe_flag=False,
        max_T=50,
        max_C=8,
        max_P_per_object=32,
        link_per_group_flag=True,
        n_pcl_per_link=16,
        #
        scale_normalize_range=[0.8, 1.2],
        first_ctx_neutral_flag=True,
        obst_pose_flag=True,
    )

    dataset[10]

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)

    # dataset.print_profiling_summary()
    print(f"Dataset length: {len(dataset)}")
