"""Language-Table oracle dataset with ground-truth poses and point clouds.

Provides oracle (GT) SE(3) link poses and pre-computed mesh point clouds for
the Language-Table planar manipulation benchmark.  Object visibility is
assumed to be complete (all objects always included).

Example::

    dataset = LangTableOracleDataset(
        data_root="data/langtable_oracle_10k_v2",
        split_fn="data_splits/langtable/10k_v2_train.json",
        max_T=20,
    )
    sample, extra = dataset[0]
"""

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
import time

from umm.datasets.umm_v1.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v1.dataset_utils import sample_list
from umm.datasets.umm_v1.dataset_config import (
    UID_LANGTABLE_OBJECT_GROUP,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
    SEM_OBJECT,
    SEM_LANGTABLE_EE,
    SEM_LANGTABLE_ACT,
)

from lib_io import UnifiedIO


class LangTableOracleDataset(BaseDataset):
    """Language-Table dataset using oracle (ground-truth) poses and meshes.

    All objects are assumed visible at every time step.  Context is the
    final (goal) state of the episode.  Optionally supports profiling of
    I/O and processing stages.

    Args:
        data_root: Root directory of the Language-Table dataset.
        split_fn: Path to the JSON split file.
        base_fps: Recording frame rate (default 10 Hz).
        random_flag: Enable stochastic augmentation.
        safe_flag: Wrap ``__getitem__`` with error handling.
        repeat: Dataset repetition multiplier.
        t_rate: Temporal down-sampling stride.
        max_T: Maximum number of time steps per sample.
        num_pts: Points to sample per point cloud.
        robot_mesh_fn: Robot mesh archive filename.
        object_mesh_fn: Object mesh archive filename.
        state_dirname: States sub-directory name.
        image_dirname: Images sub-directory name.
        depth_dirname: Depths sub-directory name.
        inst_dirname: Instance segmentation sub-directory name.
        cam_dirname: Cameras sub-directory name.
        text_dirname: Text instructions sub-directory name.
        is_explore_dataset: If True, return empty text.
        profile: Enable per-stage profiling timers.

    Example::

        ds = LangTableOracleDataset(
            data_root="data/langtable_oracle_10k_v2",
            split_fn="data_splits/langtable/10k_v2_train.json",
            max_T=20,
        )
        sample, extra = ds[0]
    """

    ##########################################################
    # * oracle mode: know the GT pose and the PCL
    # * all visible
    ##########################################################

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps=10.0,  # ecah dataset must set manually
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
        #############################################################
        max_T: int = 30,  # num of time step
        num_pts: int = 256,  # num of points per trace
        ############################################################
        robot_mesh_fn: str = "robot_embodiment.npz",
        object_mesh_fn: str = "object_embodiment.npz",
        state_dirname: str = "states",
        image_dirname: str = "images",
        depth_dirname: str = "depths",
        inst_dirname: str = "instances",
        cam_dirname: str = "cameras",
        text_dirname: str = "text",
        #
        is_explore_dataset: bool = False,
        ############################################################
        # pcl_ref_choice_mode: str = "random",
        # patch_query_p_mode: str = "mean",
        # patch_query_r_mode: str = "random",
        # return_details: bool = True,
        profile: bool = False,
    ):
        """Initialize the Language-Table oracle dataset.

        Loads robot and object mesh archives into memory and sets up
        profiling counters when ``profile=True``.

        Args:
            data_root: Root directory of the dataset.
            split_fn: Path to the JSON split file.
            base_fps: Recording frame rate.
            random_flag: Enable stochastic sampling.
            safe_flag: Wrap ``__getitem__`` with error handling.
            repeat: Dataset repetition multiplier.
            t_rate: Temporal down-sampling stride.
            max_T: Maximum time steps per sample.
            num_pts: Points per point cloud.
            robot_mesh_fn: Robot mesh archive filename.
            object_mesh_fn: Object mesh archive filename.
            state_dirname: States sub-directory.
            image_dirname: Images sub-directory.
            depth_dirname: Depths sub-directory.
            inst_dirname: Instance segmentation sub-directory.
            cam_dirname: Cameras sub-directory.
            text_dirname: Text sub-directory.
            is_explore_dataset: If True, return empty text.
            profile: Enable per-stage profiling.

        Example::

            ds = LangTableOracleDataset(
                data_root="data/langtable_oracle_10k_v2",
                split_fn="splits/10k_v2_train.json",
                max_T=20,
                profile=True,
            )
        """
        super().__init__(
            data_root, split_fn, base_fps, None, random_flag, safe_flag, repeat
        )

        self.max_T = max_T
        self.max_C = 1  # ! always use the succ goal state as the context
        self.num_pts = num_pts

        self.state_dirname = state_dirname
        self.depth_dirname = depth_dirname
        self.inst_dirname = inst_dirname
        self.cam_dirname = cam_dirname
        self.text_dirname = text_dirname
        self.image_dirname = image_dirname
        self.packed_dirname = "packed"

        self.robot_mesh_fn = robot_mesh_fn

        self.io: UnifiedIO = UnifiedIO()
        self.robot_link_data = self.io.load_npzs(
            osp.join(self.data_root, robot_mesh_fn)
        )

        self.object_link_data = self.io.load_npzs(
            osp.join(self.data_root, object_mesh_fn)
        )

        self.is_explore_dataset = is_explore_dataset
        logging.info(
            f"[LangTableOracleDataset]: is_explore_dataset: {self.is_explore_dataset}"
        )

        # self.return_details = return_details
        self.profile = profile
        self.profiling_stats = {
            "load_time_rgbs": 0.0,
            "load_time_deps": 0.0,
            "load_time_insts": 0.0,
            "load_time_cam": 0.0,
            "load_time_state": 0.0,
            "process_obj_time": 0.0,
            "process_ee_time": 0.0,
            "total_time": 0.0,
            "count": 0,
        }
        return

    def print_profiling_summary(self):
        """Print average per-sample timing for each profiled stage.

        Does nothing if no samples have been profiled yet.

        Example::

            dataset.print_profiling_summary()
        """
        if self.profiling_stats["count"] == 0:
            print("No profiling data collected.")
            return

        cnt = self.profiling_stats["count"]
        print(f"\n[LangTableDataset Profiling Report] (N={cnt})")
        print(f"  Load RGBs:    {self.profiling_stats['load_time_rgbs'] / cnt:.4f} s")
        print(f"  Load Depths:  {self.profiling_stats['load_time_deps'] / cnt:.4f} s")
        print(f"  Load Insts:   {self.profiling_stats['load_time_insts'] / cnt:.4f} s")
        print(f"  Load Cam:     {self.profiling_stats['load_time_cam'] / cnt:.4f} s")
        print(f"  Load State:   {self.profiling_stats['load_time_state'] / cnt:.4f} s")
        print(f"  Proc Object:  {self.profiling_stats['process_obj_time'] / cnt:.4f} s")
        print(f"  Proc EE/Act:  {self.profiling_stats['process_ee_time'] / cnt:.4f} s")
        print(f"  Total w/ Ovhead: {self.profiling_stats['total_time'] / cnt:.4f} s")
        print("-" * 40)

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal window for a training sample.

        Selects a contiguous window of ``max_T * t_rate`` frames around
        ``current_abs_tid``, clamped to the sequence bounds.

        Args:
            seq_id: Sequence identifier for length lookup.
            current_abs_tid: Absolute anchor frame index.
            dataset_idx: Global dataset index (deterministic offset).
            random_enabled: Allow random jitter of the window start.

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

    def __get_T_wq__(self, world_pcl):
        """Compute a query frame T_wq from a world-space point cloud.

        Position is derived from the cloud centroid (or a random point).
        Rotation is either identity or a random z-axis rotation.

        Args:
            world_pcl: ``(N, 3)`` array of world-frame 3-D points.

        Returns:
            ``(4, 4)`` homogeneous transform T_wq.

        Example::

            T_wq = ds.__get_T_wq__(world_pcl)
        """
        # get the query frame and warp it with the known transform
        if self.patch_query_p_mode == "mean":
            _query_pos = np.mean(world_pcl, axis=0)
        elif self.patch_query_p_mode == "random":
            _query_pos = world_pcl[np.random.randint(len(world_pcl))]
        else:
            raise ValueError(f"Unknown patch_query_p_mode: {self.patch_query_p_mode}")
        if self.patch_query_r_mode == "random":
            # random rot around z
            _query_rot = np.random.uniform(0, 2 * np.pi)
            _query_rot = np.array(
                [
                    [np.cos(_query_rot), -np.sin(_query_rot), 0],
                    [np.sin(_query_rot), np.cos(_query_rot), 0],
                    [0, 0, 1],
                ]
            )
        elif self.patch_query_r_mode == "id":
            _query_rot = np.eye(3)
        else:
            raise ValueError(f"Unknown patch_query_r_mode: {self.patch_query_r_mode}")
        T_wq = np.eye(4)
        T_wq[:3, :3] = _query_rot
        T_wq[:3, 3] = _query_pos
        return T_wq

    def __warp_frame__(self, T_wq, q_at_ith, T_wl):
        """Re-express a pose trajectory relative to a query frame at a reference time.

        Computes ``T_wl[t] @ inv(T_wl[q_at_ith]) @ T_wq`` for every t.

        Args:
            T_wq: ``(4, 4)`` query frame in world coordinates.
            q_at_ith: Index of the reference time step in ``T_wl``.
            T_wl: ``(N, 4, 4)`` trajectory of world-to-link transforms.

        Returns:
            ``(N, 4, 4)`` warped trajectory.

        Example::

            warped = ds.__warp_frame__(T_wq, 5, T_wl)
        """
        # T_wq: 4,4; q_at_ith: int, T_wl: N,4,4
        T_ref_query = np.linalg.inv(T_wl[q_at_ith]) @ T_wq
        T = np.einsum("tij,jk->tik", T_wl, T_ref_query)
        return T

    def __add_langtable_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
    ):
        """Populate a PromptV0 with Language-Table object and EE traces.

        Loads camera, state, and action data, appends object traces
        (with oracle mesh point clouds) and end-effector / action traces.

        Args:
            prompt: Prompt accumulator to populate.
            cur_rel_idx: Relative anchor index within the slice.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.

        Returns:
            Tuple ``(prompt, extra_dict)`` with the updated prompt and
            auxiliary data (cameras, actions, UIDs, etc.).

        Example::

            prompt = PromptV0(P=0, T=20, C=1, PCL=256, PCL_dim=6)
            prompt, extra = ds.__add_langtable_prompt__(
                prompt, 5, 0, 20, "seq_0"
            )
        """
        extra_dict = {}

        factor = int(self.t_rate)
        ctx_sids = [seq_id]
        ctx_abs_tids = [self.seq_len[seq_id] - 1]

        if self.profile:
            t0 = time.perf_counter()
            t_last = t0

        # * unpack data
        # load frames to prepare the local pcl or later dense pcl prompt
        # rgbs = np.asarray(
        #     self.io.load_rgbs(
        #         osp.join(self.data_root, self.image_dirname, seq_id + ".zip"),
        #         indices=range(start_tid, end_tid, factor),
        #     )
        # )
        extra_dict["rgbs_path"] = osp.join(
            self.data_root, self.image_dirname, seq_id + ".zip"
        )
        extra_dict["rgbs_indices"] = [t for t in range(start_tid, end_tid, factor)]

        camera_data = self.io.load_npzs(
            osp.join(self.data_root, self.cam_dirname, seq_id + ".npz"),
        )
        cam_T_wc = camera_data["T_wc"][start_tid:end_tid][::factor]
        cam_K = camera_data["K"][start_tid:end_tid][::factor]
        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["load_time_cam"] += t_now - t_last
            t_last = t_now

        state_data = self.io.load_npzs(
            osp.join(self.data_root, self.state_dirname, seq_id + ".npz"),
        )
        all_uids = state_data["all_uids"].tolist()
        sem_type_uid = state_data["sem_type_uid"].item()
        ee_uid = sem_type_uid["endeffector"]
        assert len(ee_uid) == 1, "Langtable has more than 1 endeffector"
        ee_uid = ee_uid[0]
        states_all = state_data["states"].item()
        actions_all = state_data["actions"]
        # for failure max step reaching, the action should be one shorter
        if len(actions_all) == len(states_all[ee_uid]):
            actions_all = actions_all[
                :-1
            ]  # ! for hard cut the last obs is not appended (fix this bug later in the maker)
        assert len(states_all[ee_uid]) == len(actions_all) + 1
        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["load_time_state"] += t_now - t_last
            t_last = t_now

        # * record to extra now
        extra_dict["cam_T_wc"] = cam_T_wc.copy()
        extra_dict["cam_K"] = cam_K.copy()
        extra_dict["actions"] = actions_all.copy()[start_tid:end_tid][::factor]
        extra_dict["all_uids"] = all_uids
        extra_dict["sem_type_uid"] = sem_type_uid
        # if self.return_details:
        #     extra_dict["rgbs"] = rgbs

        # *
        object_gid = prompt.max_group_id + 1
        for uid in sem_type_uid["objects"]:
            _T_oracle = states_all[uid][start_tid:end_tid][::factor]
            _T_ctx = states_all[uid][-1][None]
            _T_wl = _T_oracle
            _T_wl = _T_wl[None]
            _T_ctx = _T_ctx[None]

            _local_pcl = self.object_link_data[f"{uid}_pcl"]
            _local_pcl_c = self.object_link_data[f"{uid}_pcl_c"]
            _local_pcl = np.concatenate([_local_pcl, _local_pcl_c], axis=1)
            _pcl = sample_list(_local_pcl, self.num_pts, random_flag=self.random_flag)
            prompt.append_trace(
                trace=_T_wl,
                trace_mask=torch.ones(_T_wl.shape[:2]),
                context=_T_ctx,
                context_mask=torch.ones(_T_ctx.shape[:2]),
                group_id=object_gid,
                sem_type=SEM_OBJECT,
                uid=UID_LANGTABLE_OBJECT_GROUP[uid],
                local_pcl=_pcl[None],
                local_pcl_mask=torch.ones(1),
            )

        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["process_obj_time"] += t_now - t_last
            t_last = t_now

        # * End Effector
        ee_group_id = prompt.max_group_id + 1
        all_ee_states = states_all[ee_uid]
        all_act_states = all_ee_states.copy()
        all_act_states[1:, :2, 3] = all_act_states[:-1, :2, 3].copy() + actions_all

        ee_T_wl = all_ee_states[start_tid:end_tid][::factor]
        ee_T_ctx = all_ee_states[-1][None]
        act_T_wl = all_act_states[start_tid:end_tid][::factor]
        act_T_ctx = all_act_states[-1][None]

        ee_pcl_all = self.robot_link_data[f"{ee_uid}_pcl"]

        ee_pcl = sample_list(ee_pcl_all, self.num_pts, random_flag=self.random_flag)
        ee_pcl_color = np.ones_like(ee_pcl).astype(np.float32)
        ee_pcl = np.concatenate([ee_pcl, ee_pcl_color], axis=1)

        ee_T_wl = ee_T_wl[None]
        ee_T_ctx = ee_T_ctx[None]
        act_T_wl = act_T_wl[None]
        act_T_ctx = act_T_ctx[None]

        prompt.append_trace(
            trace=ee_T_wl,
            trace_mask=torch.ones(ee_T_wl.shape[:2]),
            context=ee_T_ctx,
            context_mask=torch.ones(ee_T_ctx.shape[:2]),
            group_id=ee_group_id,
            sem_type=SEM_LANGTABLE_EE,
            uid=UID_LANGTABLE_EE,
            local_pcl=ee_pcl[None],
            local_pcl_mask=torch.ones(1),
        )

        prompt.append_trace(
            trace=act_T_wl,
            trace_mask=torch.ones(act_T_wl.shape[:2]),
            context=act_T_ctx,
            context_mask=torch.ones(act_T_ctx.shape[:2]),
            group_id=ee_group_id,
            sem_type=SEM_LANGTABLE_ACT,
            uid=UID_LANGTABLE_ACT,
            local_pcl=ee_pcl[None],
            local_pcl_mask=torch.ones(1),
        )

        if self.profile:
            t_end = time.perf_counter()
            self.profiling_stats["process_ee_time"] += t_end - t_last
            self.profiling_stats["total_time"] += t_end - t0
            self.profiling_stats["count"] += 1

            if self.profiling_stats["count"] % 100 == 0:
                self.print_profiling_summary()

        return prompt, extra_dict

    def __getitem_inner__(self, idx):
        """Build a single training sample.

        Combines trace sampling, prompt construction, text loading, and
        timestamp generation into a ``CanonicalDatasetOutput``.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_dict)``.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)
        """

        idx = idx % len(self)
        prompt = PromptV0(P=0, T=self.max_T, C=1, PCL=self.num_pts, PCL_dim=6)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )

        prompt, extra_dict = self.__add_langtable_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid
        )
        extra_dict["is_explore"] = self.is_explore_dataset

        if self.profile:
            # End timing of __add_langtable_prompt__ logic that was split
            pass

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )
        pcl, pcl_mask = prompt.cat_pcl()

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        # load instruction
        if self.is_explore_dataset:
            text = ""
        else:
            text = self.io.load_text(
                osp.join(self.data_root, self.text_dirname, sid + ".txt")
            )
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
            local_pcl=pcl,
            local_pcl_mask=pcl_mask,
            time=time_stamp,
            text=text,
        )
        return ret, extra_dict


if __name__ == "__main__":
    PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
    print(PROJECT_ROOT)

    dataset = LangTableOracleDataset(
        data_root=osp.join(PROJECT_ROOT, "data/langtable_oracle_10k_v2"),
        split_fn=osp.join(PROJECT_ROOT, "data_splits/langtable/10k_v2_train.json"),
        safe_flag=False,
        max_T=20,
        profile=False,
        is_explore_dataset=True,
    )
    dataset2 = LangTableOracleDataset(
        data_root=osp.join(PROJECT_ROOT, "data/langtable_oracle_10k_v2"),
        split_fn=osp.join(PROJECT_ROOT, "data_splits/langtable/10k_v2_train.json"),
        safe_flag=False,
        max_T=20,
        profile=False,
    )

    dataset[10]

    from torch.utils.data import DataLoader

    cat_dataset = ConcatDataset([dataset, dataset2])

    dataloader = DataLoader(
        cat_dataset,
        batch_size=128,  # 48,
        num_workers=8,  # 8,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)

    dataset.print_profiling_summary()
    print(f"Dataset length: {len(dataset)}")
