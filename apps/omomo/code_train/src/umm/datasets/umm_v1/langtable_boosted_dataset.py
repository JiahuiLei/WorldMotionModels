"""Language-Table dataset with depth-based point cloud backprojection.

Extends the Language-Table setup with visibility-aware object PCL loading
from packed depth/instance segments, random z-axis augmentation, and
configurable reference frame selection for the object query pose.

Example::

    dataset = LangTableDataset(
        data_root="data/langtable_debug_oracle",
        split_fn="data_splits/langtable/debug_train.json",
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


class LangTableDataset(BaseDataset):
    """Language-Table dataset with packed depth backprojection for object PCLs.

    Visibility-aware: only objects observed up to the current time step are
    included.  Object point clouds are backprojected from depth/instance
    segments at a chosen reference frame, then canonicalized via a query
    frame with optional random z-axis rotation.

    Args:
        data_root: Root directory of the Language-Table dataset.
        split_fn: JSON split file listing sequences.
        base_fps: Recording frame rate (default 10 Hz).
        random_flag: Enable stochastic augmentation.
        safe_flag: Wrap ``__getitem__`` with error handling.
        repeat: Dataset repetition multiplier.
        t_rate: Temporal down-sampling stride.
        max_T: Maximum time steps per sample.
        num_pts: Points per object point cloud.
        robot_mesh_fn: Robot mesh archive filename.
        state_dirname: States sub-directory.
        image_dirname: Images sub-directory.
        depth_dirname: Depths sub-directory.
        inst_dirname: Instance segmentation sub-directory.
        cam_dirname: Cameras sub-directory.
        text_dirname: Text instructions sub-directory.
        pcl_ref_choice_mode: How to pick the reference frame for PCL
            (``"random"`` | ``"latest"`` | ``"max_cnt"``).
        patch_query_p_mode: Query position mode (``"mean"`` | ``"random"``).
        patch_query_r_mode: Query rotation mode (``"random"`` | ``"id"``).
        return_details: Store extra arrays (RGBs, etc.) in ``extra_dict``.
        profile: Enable per-stage profiling timers.

    Example::

        ds = LangTableDataset(
            data_root="data/langtable_debug_oracle",
            split_fn="data_splits/langtable/debug_train.json",
            max_T=20,
        )
        sample, extra = ds[0]
    """

    ##########################################################
    # * notes:
    # * each object trace is randomly augmented along the z axis rotation; or canonicalize to current frame has ID w.r.p to the world (robot will tell)
    # * the object local pcl is refered at the most visibel time upto t=0 , if upto current time, the object is not visible, it is not included in the prompt at all. ! be very careful during inference, the observation should be exactly the same as the data loader
    # * the EE is always in ID rotation and the xy position is the key to use

    ##########################################################
    # todo: the depth and the instance can be randomly loaded, not all at the begining...
    # ! should have random starting point and random augment the sequence length, becuase the rollout is so short!

    # ! here the RGB and text are as augmented loaded data, is in the list, not cat properly

    # todo: randomly augment the len
    # todo: deal with the scale

    #############
    # * can boost the frame loading
    # * the backprojection

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
        state_dirname: str = "states",
        image_dirname: str = "images",
        depth_dirname: str = "depths",
        inst_dirname: str = "instances",
        cam_dirname: str = "cameras",
        text_dirname: str = "text",
        ############################################################
        pcl_ref_choice_mode: str = "random",
        patch_query_p_mode: str = "mean",
        patch_query_r_mode: str = "random",
        return_details: bool = True,
        profile: bool = False,
    ):
        """Initialize the boosted Language-Table dataset.

        Loads the robot mesh archive and configures PCL reference-frame
        selection, query-frame modes, and profiling.

        Args:
            data_root: Root data directory.
            split_fn: JSON split file path.
            base_fps: Recording frame rate.
            random_flag: Stochastic sampling flag.
            safe_flag: Error-handling wrapper flag.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            max_T: Max time steps.
            num_pts: Points per cloud.
            robot_mesh_fn: Robot mesh filename.
            state_dirname: States sub-directory.
            image_dirname: Images sub-directory.
            depth_dirname: Depths sub-directory.
            inst_dirname: Instance segmentation sub-directory.
            cam_dirname: Cameras sub-directory.
            text_dirname: Text sub-directory.
            pcl_ref_choice_mode: Reference frame selection strategy.
            patch_query_p_mode: Query position strategy.
            patch_query_r_mode: Query rotation strategy.
            return_details: Store extra data in output dict.
            profile: Enable profiling.

        Example::

            ds = LangTableDataset(
                data_root="data/langtable_debug_oracle",
                split_fn="splits/debug_train.json",
                max_T=20,
                pcl_ref_choice_mode="random",
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
            osp.join(self.data_root, self.robot_mesh_fn)
        )

        self.pcl_ref_choice_mode = pcl_ref_choice_mode
        assert self.pcl_ref_choice_mode in ["latest", "max_cnt", "random"]
        self.patch_query_p_mode = patch_query_p_mode
        assert self.patch_query_p_mode in ["mean", "random"]
        self.patch_query_r_mode = patch_query_r_mode
        assert self.patch_query_r_mode in ["random", "id"]
        logging.info(
            f"[LangTableDataset]: pcl_ref_choice_mode: {self.pcl_ref_choice_mode}, patch_query_p_mode: {self.patch_query_p_mode}, patch_query_r_mode: {self.patch_query_r_mode}"
        )
        self.return_details = return_details
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

        Does nothing if no profiling data has been collected.

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

    def __get_world_pcl__(self, depth, rgb, mask, cam_T_wc, cam_K):
        """Backproject masked pixels into a world-frame point cloud.

        Args:
            depth: ``(H, W)`` depth array.
            rgb: ``(H, W, 3)`` colour array.
            mask: ``(H, W)`` instance mask; non-zero pixels are used.
            cam_T_wc: ``(4, 4)`` camera-to-world transform.
            cam_K: ``(3, 3)`` intrinsic matrix.

        Returns:
            Tuple ``(world_xyz, colors)`` each ``(N, 3)`` float arrays.

        Example::

            xyz, c = ds.__get_world_pcl__(depth, rgb, mask, T_wc, K)
        """
        # depth: H,W; rgb: H,W,3; mask: H,W; cam_T_wc: 4,4; cam_K: 3,3
        mask = mask > 0
        d = depth[mask]
        c = rgb[mask]

        if len(d) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        uv = np.stack(
            np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0])),
            axis=-1,
        )
        uv = uv[mask]
        K_inv = np.linalg.inv(cam_K)

        # Backprojection ~ vectorize
        uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=-1)
        cam_xyz = (uv1 @ K_inv.T) * d[:, None]

        # Camera to World
        world_xyz = cam_xyz @ cam_T_wc[:3, :3].T + cam_T_wc[:3, 3]

        return world_xyz, c

    def __get_world_pcl_packed__(self, depth, rgb, uv, cam_T_wc, cam_K):
        """Backproject pre-extracted packed pixels into world-frame coordinates.

        Unlike ``__get_world_pcl__`` this accepts already-flattened arrays
        from the packed segment loader (no image-shaped mask).

        Args:
            depth: ``(N,)`` depth values.
            rgb: ``(N, 3)`` colour values.
            uv: ``(N, 2)`` pixel coordinates.
            cam_T_wc: ``(4, 4)`` camera-to-world transform.
            cam_K: ``(3, 3)`` intrinsic matrix.

        Returns:
            Tuple ``(world_xyz, colors)`` each ``(N, 3)`` float arrays.

        Example::

            xyz, c = ds.__get_world_pcl_packed__(d, rgb, uv, T_wc, K)
        """
        # depth: N; rgb: N,3; uv: N,2; cam_T_wc: 4,4; cam_K: 3,3
        d = depth
        c = rgb

        if len(d) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        K_inv = np.linalg.inv(cam_K)

        # Backprojection ~ vectorize
        uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=-1)
        cam_xyz = (uv1 @ K_inv.T) * d[:, None]

        # Camera to World
        world_xyz = cam_xyz @ cam_T_wc[:3, :3].T + cam_T_wc[:3, 3]

        return world_xyz, c

    def __get_local_pcl__(self, world_xyz, T_wl):
        """Transform world-frame points into a local link frame.

        Args:
            world_xyz: ``(N, 3)`` world-frame points.
            T_wl: ``(4, 4)`` world-to-link transform.

        Returns:
            ``(N, 3)`` points in the local frame.

        Example::

            local = ds.__get_local_pcl__(world_xyz, T_wl)
        """
        # World to Local
        # world_xyz: N,3; T_wl: 4,4
        T_lw = np.linalg.inv(T_wl)
        local_xyz = world_xyz @ T_lw[:3, :3].T + T_lw[:3, 3]
        return local_xyz

    def __get_T_wq__(self, world_pcl):
        """Compute a query frame T_wq from a world-space point cloud.

        Position is the cloud centroid or a random point; rotation is
        identity or random around z, controlled by config flags.

        Args:
            world_pcl: ``(N, 3)`` world-frame 3-D points.

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
        """Re-express a trajectory relative to a query frame at a reference time.

        Args:
            T_wq: ``(4, 4)`` query frame.
            q_at_ith: Reference time index in ``T_wl``.
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
        """Populate a PromptV0 with visibility-aware object and EE traces.

        Loads packed depth segments for visible objects, backprojects to
        point clouds, computes query frames, and appends object and
        end-effector / action traces to the prompt.

        Args:
            prompt: Prompt accumulator to populate.
            cur_rel_idx: Relative anchor index within the slice.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.

        Returns:
            Tuple ``(prompt, extra_dict)`` with the updated prompt and
            auxiliary data.

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
        # rgbs = rgbs.astype(np.float32) / 255.0
        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["load_time_rgbs"] += t_now - t_last
            t_last = t_now

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

        # * Packed Meta for visibility
        packed_zip_path = osp.join(self.data_root, self.packed_dirname, seq_id + ".zip")
        packed_meta = self.io.load_packed_meta(packed_zip_path)
        # packed_meta is list of dict {uid: cnt}, length is full seq len?
        # UnifiedIO load_packed_meta returns List[Dict]. The list index is time.
        # But `packed_meta` in file is for the WHOLE sequence.
        # We need to slice it to match our time window if needed, or index it correctly with `_ref_ith`.
        # Note: `packed_meta` corresponds to ALL frames in the zip.
        # The zip usually covers the whole sequence.

        # * Object Traces
        object_ref_ith = []
        segments_to_load = []  # (time, uid)

        # First pass: decide which time instant to load for each object
        obj_ref_info = {}  # uid -> (ref_ith, ref_abs_tid)

        for uid in sem_type_uid["objects"]:
            _T_oracle = states_all[uid][start_tid:end_tid][::factor]

            # Find visible times in the current window
            # Window is `range(start_tid, end_tid, factor)`
            # `packed_meta` is indexed by absolute time if it matches the generated data structure
            # Let's assume packed_meta aligns with RGBS (0 to N-1).

            _ref_time_choice = []
            for i, abs_tid in enumerate(range(start_tid, end_tid, factor)):
                # i is relative index in the window
                # abs_tid is absolute index in metadata
                # Check if this time step is within the allowed context window (<= cur_rel_idx)
                if (
                    i <= cur_rel_idx
                    and abs_tid < len(packed_meta)
                    and uid in packed_meta[abs_tid].keys()
                ):
                    _ref_time_choice.append(i)

            if len(_ref_time_choice) == 0:
                # logging.debug(
                #     f"Object {uid} is not visible in the current slice, skip."
                # )
                continue

            # use the latest time visible as the reference
            if self.pcl_ref_choice_mode == "latest":
                _ref_ith = _ref_time_choice[-1]
            elif self.pcl_ref_choice_mode == "max_cnt":
                # Need counts from meta
                _ref_time_cnt = []
                for i in _ref_time_choice:
                    abs_tid = start_tid + i * factor
                    _ref_time_cnt.append(packed_meta[abs_tid][uid])
                _ref_ith = _ref_time_choice[np.argmax(_ref_time_cnt)]
            elif self.pcl_ref_choice_mode == "random":
                _ref_ith = np.random.choice(_ref_time_choice)
            else:
                raise ValueError(
                    f"Unknown pcl_ref_choice_mode: {self.pcl_ref_choice_mode}"
                )

            object_ref_ith.append(_ref_ith)

            # Schedule for loading
            # ref_ith is relative index. calculate absolute time for loading
            abs_load_time = start_tid + _ref_ith * factor
            segments_to_load.append((abs_load_time, uid))
            obj_ref_info[uid] = (_ref_ith, abs_load_time)

        # Batch load segments
        if self.profile:
            t_start_load = time.perf_counter()

        loaded_segments = self.io.load_packed_segments(
            packed_zip_path, query_list=segments_to_load
        )

        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["load_time_deps"] += (
                t_now - t_start_load
            )  # Reuse depth bucket for packed loading
            t_last = t_now

        # Process objects
        object_gid = prompt.max_group_id + 1
        for uid in sem_type_uid["objects"]:
            if uid not in obj_ref_info:
                continue

            _ref_ith, abs_load_time = obj_ref_info[uid]

            if (
                abs_load_time not in loaded_segments
                or uid not in loaded_segments[abs_load_time]
            ):
                logging.warning(
                    f"Segment {uid} at {abs_load_time} missing in packed zip."
                )
                continue

            seg_data = loaded_segments[abs_load_time][uid]

            _T_oracle = states_all[uid][start_tid:end_tid][::factor]
            _T_oracle = np.concatenate([_T_oracle, states_all[uid][-1][None]], axis=0)

            # Get PCL from packed data
            # seg_data keys: 'rgb', 'depth', 'uv'
            # Note: rgb is unit8, depth is float16 (or as saved), uv is int16

            _world_pcl, _pcl_colors = self.__get_world_pcl_packed__(
                seg_data["depth"].astype(np.float32),
                seg_data["rgb"],
                seg_data["uv"].astype(np.float32),
                cam_T_wc[_ref_ith],
                cam_K[_ref_ith],
            )

            _T_wq = self.__get_T_wq__(_world_pcl)
            _local_pcl = self.__get_local_pcl__(_world_pcl, _T_wq)
            _T_wl = self.__warp_frame__(_T_wq, _ref_ith, _T_oracle)
            _T_ctx = _T_wl[-1:]
            _T_wl = _T_wl[:-1]

            _T_wl = _T_wl[None]
            _T_ctx = _T_ctx[None]
            _pcl = np.concatenate(
                [_local_pcl, _pcl_colors.astype(np.float32) / 255.0], axis=1
            )
            _pcl = sample_list(_pcl, self.num_pts, random_flag=self.random_flag)
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

        extra_dict["object_ref_ith"] = object_ref_ith
        if self.profile:
            t_now = time.perf_counter()
            self.profiling_stats["process_obj_time"] += t_now - t_last
            t_last = t_now

        # * End Effector
        ee_group_id = prompt.max_group_id + 1
        all_ee_states = states_all[ee_uid]
        all_act_states = all_ee_states.copy()
        all_act_states[1:, :2, 3] = all_act_states[:-1, :2, 3] + actions_all

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
        """Build a single training sample with backprojected object clouds.

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
    PROJECT_ROOT = osp.dirname(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    )

    dataset = LangTableDataset(
        data_root=osp.join(PROJECT_ROOT, "data/langtable_debug_oracle"),
        split_fn=osp.join(PROJECT_ROOT, "data_splits/langtable/debug_train.json"),
        safe_flag=False,
        max_T=20,
        profile=False,
        return_details=True,
    )

    dataset[10]

    from torch.utils.data import DataLoader

    cat_dataset = ConcatDataset([dataset, dataset])

    dataloader = DataLoader(
        cat_dataset,
        batch_size=48,  # 48,
        num_workers=8,  # 8,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)

    dataset.print_profiling_summary()
    print(f"Dataset length: {len(dataset)}")
