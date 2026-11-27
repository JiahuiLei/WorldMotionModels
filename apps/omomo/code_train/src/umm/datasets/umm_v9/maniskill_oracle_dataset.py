"""ManiSkill oracle dataset for tabletop manipulation tasks.

Provides ground-truth SE(3) poses and pre-computed point clouds for a
two-finger-gripper robot and task objects from the ManiSkill simulator.
Only the single-robot, two-finger gripper setup is supported.

Example::

    dataset = ManiskillOracleDataset(
        data_root="data/maniskill_v2/proc_v1/PickCube-v1",
        split_fn="data_splits/maniskill_v2/proc_v1/PickCube-v1_train.json",
        max_T=20,
    )
    sample, extra = dataset[0]
"""

# ! this only support the two finger gripper single robot task for now!

import os.path as osp
import logging
import numpy as np
import torch
from tqdm import tqdm

from umm.datasets.umm_v9.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v9.dataset_utils import sample_list
from umm.datasets.umm_v9.dataset_config import (
    UID_MANISKILL_TABLETOP_EE,
    UID_MANISKILL_TABLETOP_ACT,
    UID_MANISKILL_TABLETOP_LEFT_FINGER,
    UID_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER,
    UID_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    MANISKILL_TABLETOP_TASK_OBJ_UID_MAP,
    ######
    SEM_MANISKILL_TABLETOP_EE,
    SEM_MANISKILL_TABLETOP_ACT,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER,
    SEM_MANISKILL_TABLETOP_LEFT_FINGER_ACT,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER,
    SEM_MANISKILL_TABLETOP_RIGHT_FINGER_ACT,
    SEM_OBJECT,
)

from lib_mani.utils.action_utils import state_action_to_tgt
from lib_io import UnifiedIO


class ManiskillOracleDataset(BaseDataset):
    """Dataset for ManiSkill tabletop manipulation with oracle (GT) observations.

    Loads ground-truth SE(3) link poses and pre-sampled point clouds for a
    two-finger-gripper robot and its task objects.  Context is always the
    final (success) state of the episode.

    Args:
        data_root: Path to the task directory (e.g. ``PickCube-v1/``).
        split_fn: Path to the JSON split file listing sequences.
        base_fps: Recording frame rate (default 10 Hz).
        random_flag: Enable stochastic augmentation / sampling.
        safe_flag: Wrap ``__getitem__`` with error handling.
        repeat: Dataset repetition multiplier.
        t_rate: Temporal down-sampling stride.
        max_T: Maximum number of time steps per sample.
        num_pts: Points to sample per link/object point cloud.
        robot_mesh_fn: Filename for robot embodiment meshes.
        object_mesh_fn: Filename for object embodiment meshes.
        state_dirname: Sub-directory name for state ``.npz`` files.
        image_dirname: Sub-directory name for RGB zip archives.
        depth_dirname: Sub-directory name for depth zip archives.
        inst_dirname: Sub-directory name for instance-seg zip archives.
        cam_dirname: Sub-directory name for camera ``.npz`` files.
        text_dirname: Sub-directory name for text instruction files.
        is_explore_dataset: If True, omit text instructions (not supported).

    Example::

        ds = ManiskillOracleDataset(
            data_root="data/maniskill_v2/proc_v1/PickCube-v1",
            split_fn="data_splits/maniskill_v2/proc_v1/PickCube-v1_train.json",
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
    ):
        """Initialize the ManiSkill oracle dataset.

        Loads robot and object mesh data, resolves the task name from
        ``data_root``, and maps object UIDs to the global allocation pool.

        Args:
            data_root: Path to the task directory.
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
            is_explore_dataset: Exploration mode flag (unsupported).

        Example::

            ds = ManiskillOracleDataset(
                data_root="data/maniskill_v2/proc_v1/PickCube-v1",
                split_fn="splits/PickCube-v1_train.json",
                max_T=20,
                num_pts=256,
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

        # Maniskill tasks have object embodiment in the task folder
        self.object_link_data = self.io.load_npzs(
            osp.join(self.data_root, object_mesh_fn)
        )  # ! here the uids are local, should translate to global uids!!!

        self.is_explore_dataset = is_explore_dataset
        assert (
            not self.is_explore_dataset
        ), "ManiskillOracleDataset does not support explore mode"

        # Infer task name from data_root
        # data_root is expected to be like ".../PickCube-v1" or ".../PickCube-v1/"
        self.task_name = osp.basename(osp.normpath(data_root))
        logging.info(
            f"[ManiskillOracleDataset]: task_name: {self.task_name}, is_explore_dataset: {self.is_explore_dataset}"
        )

        # Get allocated UIDs for this task
        if self.task_name in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP:
            self.task_obj_uids = MANISKILL_TABLETOP_TASK_OBJ_UID_MAP[self.task_name]
        else:
            raise ValueError(
                f"Task {self.task_name} not found in MANISKILL_TABLETOP_TASK_OBJ_UID_MAP"
            )

        return

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal slice for a single training sample.

        Selects a contiguous window of length ``max_T * t_rate`` around the
        anchor time ``current_abs_tid``, randomly or deterministically.

        Args:
            seq_id: Sequence identifier used to look up total length.
            current_abs_tid: Absolute time index of the anchor frame.
            dataset_idx: Global dataset index (used for deterministic offset).
            random_enabled: Allow random jitter of the window start.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)`` where
            ``current_ith`` is the relative index of the anchor inside the
            sampled window.

        Example::

            cur_i, s, e = dataset.__decide_trace_sampling__("ep_0", 25, 0)
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

    def __add_maniskill_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
    ):
        """Populate a PromptV0 with ManiSkill object and robot traces.

        Loads camera, state, and action data for the given sequence slice,
        appends object traces (with per-object point clouds) and the
        end-effector / finger / action traces to the prompt.

        Args:
            prompt: The prompt accumulator to populate.
            cur_rel_idx: Relative time index of the anchor within the slice.
            start_tid: Absolute start frame of the slice.
            end_tid: Absolute end frame of the slice.
            seq_id: Sequence identifier.

        Returns:
            Tuple ``(prompt, extra_dict)`` with the updated prompt and a dict
            of auxiliary data (RGB paths, cameras, actions, UIDs, etc.).

        Example::

            prompt = PromptV0(P=0, T=20, C=1, PCL=256, PCL_dim=6)
            prompt, extra = dataset.__add_maniskill_prompt__(
                prompt, 5, 0, 20, "ep_0"
            )
        """
        # * the end effecto has 3 parts, the ee, the left and right finger

        extra_dict = {}

        factor = int(self.t_rate)

        # * unpack data
        extra_dict["rgbs_path"] = osp.join(
            self.data_root, self.image_dirname, seq_id + ".zip"
        )
        extra_dict["rgbs_indices"] = [t for t in range(start_tid, end_tid, factor)]

        camera_data = self.io.load_npzs(
            osp.join(self.data_root, self.cam_dirname, seq_id + ".npz"),
        )
        cam_T_wc = camera_data["T_wc"][start_tid:end_tid][::factor]
        cam_K = camera_data["K"][start_tid:end_tid][::factor]

        state_data = self.io.load_npzs(
            osp.join(self.data_root, self.state_dirname, seq_id + ".npz"),
        )
        all_uids = state_data["all_uids"].tolist()
        sem_type_uid = state_data["sem_type_uid"].item()  #! local uid

        # Extract UIDs
        ee_uid = sem_type_uid["endeffector"][0]
        left_finger_uid = sem_type_uid["gripper_left_finger"][0]
        right_finger_uid = sem_type_uid["gripper_right_finger"][0]
        object_uids = sem_type_uid["objects"]

        states_all = state_data["states"].item()
        actions_all = state_data["actions"]

        # for failure max step reaching, the action should be one shorter
        assert len(actions_all) == len(
            states_all[str(ee_uid)]
        ), "in Maniskill, the action should have same len with the state!"

        # * record to extra now
        extra_dict["task"] = self.task_name
        extra_dict["cam_T_wc"] = cam_T_wc.copy()
        extra_dict["cam_K"] = cam_K.copy()
        extra_dict["actions"] = actions_all.copy()[start_tid:end_tid][::factor]
        extra_dict["all_uids"] = all_uids
        extra_dict["sem_type_uid"] = sem_type_uid

        # * Objects
        # Ensure we don't exceed allocated pool
        if len(object_uids) > len(self.task_obj_uids):
            logging.warning(
                f"Task {self.task_name} has {len(object_uids)} objects but only {len(self.task_obj_uids)} allocated UIDs. Truncating."
            )
            raise ValueError(
                f"Task {self.task_name} has {len(object_uids)} objects but only {len(self.task_obj_uids)} allocated UIDs. Truncating."
            )
        object_gid = prompt.max_group_id + 1
        for _uid in object_uids:
            _T_oracle = states_all[str(_uid)][start_tid:end_tid][::factor]
            _T_ctx = states_all[str(_uid)][-1][None]  # Goal state (last frame)
            _T_wl = _T_oracle
            _T_wl = _T_wl[None]
            _T_ctx = _T_ctx[None]

            # Point cloud loading
            _local_pcl = self.object_link_data[f"{_uid}_pcl"]
            if f"{_uid}_c" in self.object_link_data:
                _c = self.object_link_data[f"{_uid}_c"]
                # Normalize if needed (heuristic: if max > 1.0 assume 0-255)
                if _c.max() > 1.0:
                    _c = _c / 255.0
                _local_pcl_c = np.tile(_c[None, :], (_local_pcl.shape[0], 1)).astype(
                    np.float32
                )
            elif f"{_uid}_pcl_c" in self.object_link_data:
                _local_pcl_c = self.object_link_data[f"{_uid}_pcl_c"]
            else:
                raise ValueError(f"Object {_uid} missing color data")

            _local_pcl = np.concatenate([_local_pcl, _local_pcl_c], axis=1)

            _pcl = sample_list(_local_pcl, self.num_pts, random_flag=self.random_flag)
            prompt.append_trace(
                trace=_T_wl,
                trace_mask=torch.ones(_T_wl.shape[:2]),
                context=_T_ctx,
                context_mask=torch.ones(_T_ctx.shape[:2]),
                group_id=object_gid,
                sem_type=SEM_OBJECT,
                uid=self.task_obj_uids[int(_uid)],
                local_pcl=_pcl[None],
                local_pcl_mask=torch.ones(1),
            )

        # * End Effector & Action
        actor_group_id = prompt.max_group_id + 1

        palm_T_wl = torch.as_tensor(states_all[str(ee_uid)])
        left_finger_T_wl = torch.as_tensor(states_all[str(left_finger_uid)])
        right_finger_T_wl = torch.as_tensor(states_all[str(right_finger_uid)])
        palm_act_T_wl, left_finger_act_T_wl, right_finger_act_T_wl = (
            state_action_to_tgt(
                current_palm_pose=palm_T_wl,
                current_left_finger_pose=left_finger_T_wl,
                current_right_finger_pose=right_finger_T_wl,
                action=torch.as_tensor(actions_all),
                pos_limit=0.1,
                rot_limit=0.1,
            )
        )

        # ! the action is aligned with the t+1!
        palm_act_T_wl = torch.cat([palm_T_wl[:1], palm_act_T_wl[:-1]], dim=0)
        left_finger_act_T_wl = torch.cat(
            [left_finger_T_wl[:1], left_finger_act_T_wl[:-1]], dim=0
        )
        right_finger_act_T_wl = torch.cat(
            [right_finger_T_wl[:1], right_finger_act_T_wl[:-1]], dim=0
        )

        actor_trace = torch.stack(
            [
                palm_T_wl,
                left_finger_T_wl,
                right_finger_T_wl,
                palm_act_T_wl,
                left_finger_act_T_wl,
                right_finger_act_T_wl,
            ],
            dim=0,
        )  # 6,T,4,4
        actor_ctx = actor_trace[:, -1:].clone()
        actor_trace = actor_trace[:, start_tid:end_tid][:, ::factor]

        # EE PCL
        actor_uids = [ee_uid, left_finger_uid, right_finger_uid]
        actor_pcl = []

        for _uid in actor_uids:
            _pcl_all = self.robot_link_data[f"{_uid}_pcl"]
            _pcl_c = self.robot_link_data[f"{_uid}_pcl_c"]
            _pcl_all = np.concatenate([_pcl_all, _pcl_c], axis=1)
            _pcl = sample_list(_pcl_all, self.num_pts, random_flag=self.random_flag)
            actor_pcl.append(_pcl)
        actor_pcl = np.stack(actor_pcl + actor_pcl, axis=0)
        actor_pcl = torch.as_tensor(actor_pcl, dtype=torch.float32)

        prompt.append_trace(
            trace=actor_trace,
            trace_mask=torch.ones(actor_trace.shape[:2]),
            context=actor_ctx,
            context_mask=torch.ones(actor_ctx.shape[:2]),
            group_id=actor_group_id,
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
            local_pcl=actor_pcl,
            local_pcl_mask=torch.ones(6),
        )

        return prompt, extra_dict

    def __getitem_inner__(self, idx):
        """Build a single training sample from the dataset.

        Combines trace sampling, prompt construction, and text instruction
        loading into a ``CanonicalDatasetOutput``.

        Args:
            idx: Raw dataset index (will be wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_dict)`` ready for the
            dataloader collate function.

        Example::

            ret, extra = dataset.__getitem_inner__(0)
            print(ret.trace.shape)  # (P, T, 4, 4)
        """

        idx = idx % len(self)
        prompt = PromptV0(P=0, T=self.max_T, C=1, PCL=self.num_pts, PCL_dim=6)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )

        prompt, extra_dict = self.__add_maniskill_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid
        )
        extra_dict["is_explore"] = self.is_explore_dataset

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
            # ! use
            text = self.task_name
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

    # Path to Maniskill data (PickCube-v1)
    data_root = osp.join(PROJECT_ROOT, "data/maniskill_v2/proc_v1/PickCube-v1")

    dataset = ManiskillOracleDataset(
        data_root=data_root,
        split_fn=osp.join(
            PROJECT_ROOT, "data_splits/maniskill_v2/proc_v1/PickCube-v1_train.json"
        ),
        safe_flag=False,
        max_T=20,
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        ret, extra = dataset[0]
        print("Sample 0 trace shape:", ret.trace.shape)
        print("Sample 0 uid:", ret.trace_uid)
        print("Sample 0 sem:", ret.trace_sem_type)

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
        if i > 2:
            break
