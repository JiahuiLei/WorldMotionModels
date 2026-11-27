"""BEHAVIOR-1K real-robot dataset for the UMM framework.

Loads multi-view RGB-D observations, instance segmentation, kinematic robot
states, and text annotations from the BEHAVIOR-1K data pipeline.  Each sample
produces a :class:`~umm.datasets.umm_v6_all.dataset_template.CanonicalDatasetOutput`
containing SE(3) traces for objects and robot links, per-entity point clouds
back-projected from depth, and natural-language task descriptions.

Example::

    dataset = BehaviorDataset(
        data_root="/data/behavior_proc",
        split_fn="splits/train.json",
        max_T=30,
        views=["head"],
    )
    sample, extra = dataset[0]
"""

import os.path as osp
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import logging


@contextmanager
def _null_ts(_name: str):
    """No-op timing context used when profiling is disabled."""
    yield


from umm.datasets.umm_v6_all.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_v6_all.dataset_utils import DatasetProfiler, sample_list
from umm.datasets.umm_v6_all.dataset_config import (
    UID_BEHAVIOR_OBJECT,
    UID_R1PRO_ZED,
    UID_R1PRO_LEFT_GRIPPER,
    UID_R1PRO_LEFT_FINGER1,
    UID_R1PRO_LEFT_FINGER2,
    UID_R1PRO_RIGHT_GRIPPER,
    UID_R1PRO_RIGHT_FINGER1,
    UID_R1PRO_RIGHT_FINGER2,
    UID_R1PRO_ZED_ACT,
    UID_R1PRO_LEFT_GRIPPER_ACT,
    UID_R1PRO_LEFT_FINGER1_ACT,
    UID_R1PRO_LEFT_FINGER2_ACT,
    UID_R1PRO_RIGHT_GRIPPER_ACT,
    UID_R1PRO_RIGHT_FINGER1_ACT,
    UID_R1PRO_RIGHT_FINGER2_ACT,
    SEM_OBJECT,
    SEM_CAMERA,
    SEM_MANIPULATOR,
    SEM_BEHAVIOR_ACT,
)
from lib_io import UnifiedIO


class BehaviorDataset(BaseDataset):
    """BEHAVIOR-1K real-robot dataset with multi-view RGB-D and language.

    Loads per-sequence kinematic robot states, multi-view depth/RGB/instance
    segmentation, and text annotations.  Produces SE(3) traces for scene
    objects and robot links together with back-projected point clouds.

    Args:
        data_root: Root directory of the processed BEHAVIOR-1K data.
        split_fn: Path to the JSON split file.
        base_fps: Native recording frame rate.
        random_flag: Enable stochastic sampling and augmentation.
        safe_flag: Silently skip bad samples when True.
        repeat: Dataset repetition multiplier.
        t_rate: Temporal stride factor.
        max_T: Maximum number of time steps per sample.
        num_pts: Number of points to sample per entity point cloud.
        max_scene_P: Maximum number of scene-object entities to include.
        robot_mesh_fn: Filename (relative to *data_root*) of the robot
            link mesh archive (``*.npz``).
        state_dirname: Subdirectory name for state ``*.npz`` files.
        image_dirname: Subdirectory name for RGB image ZIP archives.
        depth_dirname: Subdirectory name for depth image ZIP archives.
        inst_dirname: Subdirectory name for instance segmentation ZIPs.
        cam_dirname: Subdirectory name for camera parameter ``*.npz`` files.
        text_dirname: Subdirectory name for text annotation JSON files.
        return_details: If True, include padded RGB/depth/mask arrays in
            the extra dict for visualisation.
        profile: Enable dataset-level I/O and compute profiling.
        views: Camera view names to load (e.g. ``["head"]``).
        max_range: Maximum XY distance (metres) from robot centre for
            point-cloud filtering; 0 disables filtering.
        uv_oversample: Oversampling factor for UV pixel selection per UID.
        text_mode: Text instruction granularity. One of ``"full"``,
            ``"task"``, ``"activity"``, ``"skill"``.

    Example::

        ds = BehaviorDataset(
            data_root="/data/behavior_proc",
            split_fn="splits/train.json",
            max_T=30,
            views=["head"],
            profile=True,
        )
        sample, extra = ds[0]
    """

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps=30.0,
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
        max_T: int = 30,
        num_pts: int = 256,
        max_scene_P: int = 16,
        robot_mesh_fn: str = "robot_embodiment.npz",
        state_dirname: str = "states",
        image_dirname: str = "images",
        depth_dirname: str = "depths",
        inst_dirname: str = "instances",
        cam_dirname: str = "cameras",
        text_dirname: str = "text",
        return_details: bool = False,
        profile: bool = False,
        views: list[str] = ["head", "left_wrist", "right_wrist"],
        max_range: float = 5.0,
        uv_oversample: int = 4,
        text_mode: str = "activity",
    ):
        """Initialize the BEHAVIOR-1K dataset.

        Validates *text_mode*, loads the robot link mesh archive, and
        initialises the profiler and UV-grid cache.

        Args:
            data_root: Root data directory.
            split_fn: JSON split file path.
            base_fps: Recording frame rate.
            random_flag: Stochastic sampling flag.
            safe_flag: Error-recovery flag.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            max_T: Maximum time steps.
            num_pts: Point-cloud size per entity.
            max_scene_P: Max scene objects.
            robot_mesh_fn: Robot mesh filename.
            state_dirname: State subdirectory.
            image_dirname: RGB subdirectory.
            depth_dirname: Depth subdirectory.
            inst_dirname: Instance-seg subdirectory.
            cam_dirname: Camera subdirectory.
            text_dirname: Text subdirectory.
            return_details: Return visualisation arrays.
            profile: Enable profiling.
            views: Camera view names.
            max_range: XY range filter (metres).
            uv_oversample: UV oversampling factor.
            text_mode: Text granularity mode.

        Raises:
            ValueError: If *text_mode* is not in
                ``{"full", "task", "activity", "skill"}``.

        Example::

            ds = BehaviorDataset(
                data_root="/data/behavior_proc",
                split_fn="splits/train.json",
                max_T=30,
            )
        """
        _valid_text_modes = {"full", "task", "activity", "skill"}
        if text_mode not in _valid_text_modes:
            raise ValueError(
                f"text_mode must be one of {_valid_text_modes}, got {text_mode!r}"
            )
        super().__init__(
            data_root, split_fn, base_fps, None, random_flag, safe_flag, repeat
        )
        self.text_mode = text_mode

        self.max_T = max_T
        self.max_C = 1
        self.num_pts = num_pts
        self.max_scene_P = max_scene_P
        self.max_range = max_range
        self._uv_sample_budget = num_pts * uv_oversample

        self.state_dirname = state_dirname
        self.depth_dirname = depth_dirname
        self.inst_dirname = inst_dirname
        self.cam_dirname = cam_dirname
        self.text_dirname = text_dirname
        self.image_dirname = image_dirname

        self.robot_mesh_fn = robot_mesh_fn

        self.io: UnifiedIO = UnifiedIO()
        self.robot_link_data = self.io.load_npzs(
            osp.join(self.data_root, self.robot_mesh_fn)
        )

        self.return_details = return_details
        self.profile = profile
        self._profiler = DatasetProfiler() if self.profile else None

        self.views = views
        # UV grid cache keyed by (H, W) — built once per unique resolution,
        # reused across all samples and views of the same shape.
        self._uv_cache: dict[tuple[int, int], np.ndarray] = {}
        logging.info(f"Using views: {self.views}")

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal window and the relative current-frame index.

        Randomly (or deterministically) places a window of length
        ``max_T * factor`` around *current_abs_tid*, clamped to the
        sequence boundaries.

        Args:
            seq_id: Sequence identifier (used to look up total length).
            current_abs_tid: Absolute frame index of the anchor time step.
            dataset_idx: Global dataset index (used for deterministic offset).
            random_enabled: If False, use deterministic placement regardless
                of ``self.random_flag``.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)`` where
            *current_ith* is the index of the anchor frame inside the
            sampled window.

        Example::

            cur_i, s, e = self.__decide_trace_sampling__("seq_001", 42, 0)
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

    def __get_or_build_uv_grid__(self, H: int, W: int) -> np.ndarray:
        """Return a cached (H, W, 2) integer UV pixel-coordinate grid.

        The grid is built once per unique (H, W) and reused across all
        samples and views that share the same resolution.

        Example::

            uv = self.__get_or_build_uv_grid__(480, 640)  # shape (480, 640, 2)
        """
        key = (H, W)
        if key not in self._uv_cache:
            self._uv_cache[key] = np.stack(
                np.meshgrid(np.arange(W), np.arange(H)), axis=-1
            )
        return self._uv_cache[key]

    def __list_visible_uids__(
        self,
        seq_id: str,
        frame_idx: int,
    ) -> set[int]:
        """Query all UIDs visible across views at a given frame, without decoding masks.

        Iterates over ``self.views`` and reads only the JSON metadata from each
        instance-segmentation ZIP, collecting the union of all ``"id"`` fields.

        Args:
            seq_id: Sequence identifier.
            frame_idx: Absolute frame index to query.

        Returns:
            Union of integer UIDs present in any view at ``frame_idx``.

        Example::

            visible = self.__list_visible_uids__("seq_001", 15)
        """
        visible: set[int] = set()
        for view in self.views:
            inst_path = osp.join(
                self.data_root, self.inst_dirname, f"{seq_id}_{view}.zip"
            )
            if not osp.exists(inst_path):
                continue
            visible.update(self.io.list_inst_uids(inst_path, frame_idx))
        return visible

    def __decide_target_uids__(
        self,
        task_objects: list[int],
        static_background: list[int],
        visible_uids: set[int],
    ) -> list[int]:
        """Select object UIDs to include, filtered by what is actually visible.

        Only UIDs present in ``visible_uids`` (from instance segmentation at the
        current frame) are considered.  Task objects are prioritised; remaining
        budget is filled with static background objects, capped at ``max_scene_P``.

        Args:
            task_objects: Full list of task-relevant object UIDs from metadata.
            static_background: Full list of background object UIDs from metadata.
            visible_uids: Set of UIDs actually present in the current views.

        Returns:
            List of selected integer UIDs (at most ``self.max_scene_P``).

        Example::

            uids = self.__decide_target_uids__([1, 2, 3], [10, 20], {1, 3, 20})
        """
        task_objects = [u for u in task_objects if u in visible_uids]
        static_background = [u for u in static_background if u in visible_uids]

        budget = self.max_scene_P - len(task_objects)
        if budget > 0:
            static_background = self._subsample_list(static_background, budget)
        else:
            static_background = []

        target_uids = task_objects + static_background
        if len(target_uids) > self.max_scene_P:
            target_uids = self._subsample_list(target_uids, self.max_scene_P)
        return target_uids

    def _subsample_list(self, items: list, n: int) -> list:
        """Subsample a plain Python list to at most *n* elements.

        Uses random choice when ``self.random_flag`` is True, otherwise
        takes evenly spaced elements.

        Args:
            items: Input list to subsample.
            n: Maximum number of elements to return.

        Returns:
            A list with at most *n* elements from *items*.

        Example::

            out = self._subsample_list([1, 2, 3, 4, 5], 3)
        """
        if len(items) <= n:
            return items
        if self.random_flag:
            return np.random.choice(items, n, replace=False).tolist()
        step = max(1, len(items) // n)
        return items[::step][:n]

    def __get_world_pcl__(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        mask: np.ndarray,
        cam_T_wc: np.ndarray,
        K_inv: np.ndarray,
        uv_grid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backproject masked pixels into world-frame point cloud.

        Args:
            depth: (H, W) depth array.
            rgb: (H, W, 3) colour array.
            mask: (H, W) boolean or int mask; non-zero pixels are used.
            cam_T_wc: (4, 4) camera-to-world transform.
            K_inv: (3, 3) precomputed inverse intrinsic (pass ``np.linalg.inv(K)``).
            uv_grid: Optional cached (H, W, 2) UV grid; built internally if None.

        Returns:
            Tuple of ``(world_xyz, colors)`` each ``(N, 3)`` float arrays.

        Example::

            K_inv = np.linalg.inv(K)
            uv = dataset.__get_or_build_uv_grid__(H, W)
            xyz, c = dataset.__get_world_pcl__(depth, rgb, mask, T_wc, K_inv, uv)
        """
        mask = mask > 0
        d = depth[mask]
        c = rgb[mask]

        if len(d) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        if uv_grid is None:
            uv_grid = self.__get_or_build_uv_grid__(*depth.shape[:2])
        uv = uv_grid[mask]

        # Backprojection (OpenCV convention: X right, Y down, Z forward)
        uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=-1)
        cam_xyz = (uv1 @ K_inv.T) * d[:, None]

        # T_wc baked in convert.py, apply directly.
        world_xyz = cam_xyz @ cam_T_wc[:3, :3].T + cam_T_wc[:3, 3]

        return world_xyz, c

    def __precompute_view_world_pcl__(
        self,
        depth: np.ndarray,
        cam_T_wc: np.ndarray,
        cam_K: np.ndarray,
        uv_grid: np.ndarray,
    ) -> np.ndarray:
        """Backproject every pixel of a view into world-frame coordinates.

        Called once per view; the returned (H, W, 3) array is then indexed
        per-UID with a boolean mask, avoiding repeated K_inv and matmul work.

        Args:
            depth: (H, W) float32 depth array.
            cam_T_wc: (4, 4) camera-to-world transform.
            cam_K: (3, 3) intrinsic matrix.
            uv_grid: (H, W, 2) integer UV grid (from ``__get_or_build_uv_grid__``).

        Returns:
            ``world_xyz``: (H, W, 3) float64 world-frame XYZ for every pixel.

        Example::

            uv = self.__get_or_build_uv_grid__(H, W)
            world_xyz = self.__precompute_view_world_pcl__(depth, T_wc, K, uv)
            uid_xyz = world_xyz[uid_mask]  # (N, 3)
        """
        H, W = depth.shape
        K_inv = np.linalg.inv(cam_K)

        # Flatten to (H*W, 2) and append homogeneous coordinate
        uv_flat = uv_grid.reshape(-1, 2)
        uv1 = np.concatenate(
            [uv_flat, np.ones((H * W, 1), dtype=uv_flat.dtype)], axis=-1
        )
        # Backproject to camera space, scale by depth
        cam_xyz = (uv1 @ K_inv.T) * depth.reshape(-1, 1)
        # Transform to world space (OpenCV convention, T_wc baked in convert.py)
        world_xyz = cam_xyz @ cam_T_wc[:3, :3].T + cam_T_wc[:3, 3]

        return world_xyz.reshape(H, W, 3)

    def __get_local_pcl__(self, world_xyz, T_wl):
        """Transform world-frame points into the local frame of an entity.

        Args:
            world_xyz: ``(N, 3)`` world-frame point positions.
            T_wl: ``(4, 4)`` world-to-local SE(3) transform (i.e. the entity
                pose in world frame; its inverse is applied).

        Returns:
            ``(N, 3)`` points expressed in the entity-local frame.

        Example::

            local = self.__get_local_pcl__(world_pts, T_wl)
        """
        T_lw = np.linalg.inv(T_wl)
        local_xyz = world_xyz @ T_lw[:3, :3].T + T_lw[:3, 3]
        return local_xyz

    def __get_T_wq__(self, world_pcl):
        """Compute a query frame centred at the mean of a world point cloud.

        The rotation component is set to identity (no random yaw for now).

        Args:
            world_pcl: ``(N, 3)`` world-frame point positions.

        Returns:
            ``(4, 4)`` homogeneous transform placing the query frame at the
            centroid of *world_pcl* with identity rotation.

        Example::

            T_wq = self.__get_T_wq__(world_pcl)
        """
        _query_pos = np.mean(world_pcl, axis=0)

        # ! FOR NOW DO NTO RANDOMIZE THE INIT QUERY, USE ID
        # if self.random_flag:
        #     _query_rot_z = np.random.uniform(0, 2 * np.pi)
        #     _query_rot = np.array(
        #         [
        #             [np.cos(_query_rot_z), -np.sin(_query_rot_z), 0],
        #             [np.sin(_query_rot_z), np.cos(_query_rot_z), 0],
        #             [0, 0, 1],
        #         ]
        #     )
        # else:
        _query_rot = np.eye(3)

        T_wq = np.eye(4)
        T_wq[:3, :3] = _query_rot
        T_wq[:3, 3] = _query_pos
        return T_wq

    def __warp_frame__(self, T_wq, q_at_ith, T_wl):
        """Warp a time-sequence of poses into a query-centred frame.

        Computes the relative transform between the anchor-time entity pose
        and the query frame, then applies it to every time step so that the
        entity at the anchor frame coincides with *T_wq*.

        Args:
            T_wq: ``(4, 4)`` query-frame transform (world frame).
            q_at_ith: Integer index of the anchor time step within *T_wl*.
            T_wl: ``(T, 4, 4)`` time-series of entity poses in world frame.

        Returns:
            ``(T, 4, 4)`` warped pose sequence aligned to *T_wq*.

        Example::

            T_warped = self.__warp_frame__(T_wq, cur_idx, T_wl_seq)
        """
        T_ref_query = np.linalg.inv(T_wl[q_at_ith]) @ T_wq
        T = np.einsum("tij,jk->tik", T_wl, T_ref_query)
        return T

    def __add_behavior_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,
        start_tid,
        end_tid,
        seq_id,
    ):
        """Populate a prompt with object and robot-link traces for one sample.

        Loads kinematic states, multi-view RGB-D, and instance segmentation
        for the time window ``[start_tid, end_tid)``.  Back-projects per-UID
        point clouds from depth, constructs SE(3) traces for visible objects
        and each robot link, and appends them to *prompt*.

        Args:
            prompt: Mutable :class:`PromptV0` accumulator to extend.
            cur_rel_idx: Relative index of the anchor frame within the window.
            start_tid: Absolute start frame (inclusive).
            end_tid: Absolute end frame (exclusive).
            seq_id: Sequence identifier string.

        Returns:
            Tuple ``(prompt, extra_dict)`` where *extra_dict* contains
            visualisation arrays, RGB paths, camera poses, and the robot
            centre XY used for centering.

        Example::

            prompt, extra = self.__add_behavior_prompt__(
                prompt, cur_idx, 10, 40, "seq_001"
            )
        """
        extra_dict = {}
        factor = int(self.t_rate)

        # Store per-view RGB paths and sequence frame indices for viz
        extra_dict["rgbs_paths"] = [
            osp.join(self.data_root, self.image_dirname, f"{seq_id}_{view}.zip")
            for view in self.views
        ]
        extra_dict["rgbs_views"] = list(self.views)
        extra_dict["rgbs_indices"] = [t for t in range(start_tid, end_tid, factor)]

        # We need data for the current time step which is current_abs_tid
        current_abs_tid = start_tid + cur_rel_idx * factor

        _prof = self._profiler
        _ts = _prof.time_section if _prof else _null_ts

        with _ts("io/state_npz"):
            state_data = self.io.load_npzs(
                osp.join(self.data_root, self.state_dirname, seq_id + ".npz"),
            )
        sem_type_uid = state_data["sem_type_uid"].item()
        states_all = state_data["states"].item()
        kinematic_states = state_data.get("kinematic_states", {}).item()

        # Note: behavior states are keyed by stringified UIDs
        # In behavior dataset, task_objects and static_background are lists of ints.
        task_objects = sem_type_uid.get("task_objects", [])
        static_background = sem_type_uid.get("static_background", [])

        # Compute robot center in XY from kinematic link positions at the
        # current timestep.  Used later to discard depth beyond max_range.
        _robot_positions: list[np.ndarray] = []
        for _link_states in kinematic_states.values():
            if current_abs_tid < len(_link_states):
                _robot_positions.append(_link_states[current_abs_tid][:3, 3])
        _center_xy = (
            np.mean(_robot_positions, axis=0)[:2].astype(np.float32)
            if _robot_positions
            else np.zeros(2, dtype=np.float32)
        )

        with _ts("io/list_inst_uids"):
            visible_uids = self.__list_visible_uids__(seq_id, current_abs_tid)

        target_uids = self.__decide_target_uids__(
            task_objects, static_background, visible_uids
        )
        target_uids_set = set(target_uids)

        rgbs_list = []
        depths_list = []
        masks_list = []
        cam_T_wc_list = []
        K_inv_list = []

        for view in self.views:
            rgb_path = osp.join(
                self.data_root, self.image_dirname, f"{seq_id}_{view}.zip"
            )
            dep_path = osp.join(
                self.data_root, self.depth_dirname, f"{seq_id}_{view}.zip"
            )
            inst_path = osp.join(
                self.data_root, self.inst_dirname, f"{seq_id}_{view}.zip"
            )
            cam_path = osp.join(
                self.data_root, self.cam_dirname, f"{seq_id}_{view}.npz"
            )

            if not osp.exists(rgb_path):
                continue

            with _ts(f"io/rgb_view_{view}"):
                rgbs = self.io.load_rgbs(rgb_path, indices=[current_abs_tid])
            with _ts(f"io/depth_view_{view}"):
                deps = self.io.load_depths(dep_path, indices=[current_abs_tid])
            with _ts(f"io/inst_view_{view}"):
                insts = self.io.load_inst_segs(
                    inst_path,
                    indices=[current_abs_tid],
                    target_uids=target_uids_set,
                )
            depth_frame = np.squeeze(deps[0]).astype(np.float32)
            rgbs_list.append(rgbs[0])
            depths_list.append(depth_frame)
            masks_list.append(np.squeeze(insts[0]))

            with _ts(f"io/cam_npz_view_{view}"):
                cam_data = self.io.load_npzs(cam_path)
            T_wc = cam_data["T_wc"][current_abs_tid]
            K = cam_data["K"][current_abs_tid]
            cam_T_wc_list.append(T_wc)
            K_inv_list.append(np.linalg.inv(K))

        if self.return_details:
            if rgbs_list:
                max_h = max(r.shape[0] for r in rgbs_list)
                max_w = max(r.shape[1] for r in rgbs_list)

                p_rgbs, p_deps, p_masks = [], [], []
                with _ts("compute/img_padding"):
                    for r, d, m in zip(rgbs_list, depths_list, masks_list):
                        pad_h = max_h - r.shape[0]
                        pad_w = max_w - r.shape[1]

                        p_rgbs.append(np.pad(r, ((0, pad_h), (0, pad_w), (0, 0))))
                        p_deps.append(np.pad(d, ((0, pad_h), (0, pad_w))))
                        p_masks.append(np.pad(m, ((0, pad_h), (0, pad_w))))

                    extra_dict["rgbs"] = np.stack(p_rgbs)
                    extra_dict["depths"] = np.stack(p_deps)
                    extra_dict["masks"] = np.stack(p_masks)
            else:
                extra_dict["rgbs"] = np.zeros((0, 3))
                extra_dict["depths"] = np.zeros((0))
                extra_dict["masks"] = np.zeros((0))

        uid_pcl_per_view: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        all_visible: set[int] = set()

        with _ts("compute/view_precompute"):
            flat_depth = [d.ravel() for d in depths_list]
            flat_rgb = [r.reshape(-1, 3) for r in rgbs_list]

            for v_idx in range(len(masks_list)):
                flat_inst = masks_list[v_idx].ravel()
                v_visible = set[Any](np.unique(flat_inst).tolist()) - {0}

                uid_pcl: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                if v_visible:
                    uid_order: list[int] = []
                    uid_idx_chunks: list[np.ndarray] = []
                    boundaries: list[tuple[int, int]] = []
                    offset = 0
                    for uid_val in v_visible:
                        idx = np.flatnonzero(flat_inst == uid_val)
                        if len(idx) > self._uv_sample_budget:
                            if self.random_flag:
                                idx = np.random.choice(
                                    idx, self._uv_sample_budget, replace=False
                                )
                            else:
                                step = len(idx) // self._uv_sample_budget
                                idx = idx[::step][: self._uv_sample_budget]
                        uid_order.append(uid_val)
                        uid_idx_chunks.append(idx)
                        boundaries.append((offset, offset + len(idx)))
                        offset += len(idx)

                    all_idx = np.concatenate(uid_idx_chunks)

                    H, W = masks_list[v_idx].shape
                    u = (all_idx % W).astype(np.float32)
                    v = (all_idx // W).astype(np.float32)
                    d_all = flat_depth[v_idx][all_idx]
                    K_inv_f32 = K_inv_list[v_idx].astype(np.float32)
                    T_wc_f32 = cam_T_wc_list[v_idx].astype(np.float32)

                    uv_f32 = np.stack([u, v], axis=-1)
                    cam_xy = uv_f32 @ K_inv_f32[:2, :2].T + K_inv_f32[:2, 2]
                    cam_xyz_all = np.concatenate(
                        [cam_xy * d_all[:, None], d_all[:, None]], axis=-1
                    )
                    world_xyz_all = cam_xyz_all @ T_wc_f32[:3, :3].T + T_wc_f32[:3, 3]
                    rgb_all = flat_rgb[v_idx][all_idx]

                    for uid_val, (lo, hi) in zip(uid_order, boundaries):
                        chunk_xyz = world_xyz_all[lo:hi]
                        chunk_rgb = rgb_all[lo:hi]
                        if self.max_range > 0:
                            dist_xy = np.sqrt(
                                (chunk_xyz[:, 0] - _center_xy[0]) ** 2
                                + (chunk_xyz[:, 1] - _center_xy[1]) ** 2
                            )
                            valid = dist_xy <= self.max_range
                            chunk_xyz = chunk_xyz[valid]
                            chunk_rgb = chunk_rgb[valid]
                        if len(chunk_xyz) > 0:
                            uid_pcl[uid_val] = (chunk_xyz, chunk_rgb)

                uid_pcl_per_view.append(uid_pcl)
                all_visible.update(uid_pcl.keys())

        for uid in target_uids:
            if uid not in all_visible:
                continue
            uid_str = str(uid)
            if uid_str not in states_all:
                continue

            _T_oracle = states_all[uid_str][start_tid:end_tid][::factor]
            _T_oracle = np.concatenate(
                [_T_oracle, states_all[uid_str][-1][None]], axis=0
            )

            # Per-uid PCL: just look up precomputed slices — no equality scan or matmul.
            all_world_xyz = []
            all_colors = []

            with _ts("compute/backproject"):
                for v_pcl in uid_pcl_per_view:
                    entry = v_pcl.get(uid)
                    if entry is None:
                        continue
                    all_world_xyz.append(entry[0])
                    all_colors.append(entry[1])

            if not all_world_xyz:
                continue

            with _ts("compute/pcl_concat"):
                _world_pcl = np.concatenate(all_world_xyz, axis=0)
                _pcl_colors = np.concatenate(all_colors, axis=0)

            _T_wq = self.__get_T_wq__(_world_pcl)
            _local_pcl = self.__get_local_pcl__(_world_pcl, _T_wq)
            _T_wl = self.__warp_frame__(_T_wq, cur_rel_idx, _T_oracle)
            _T_ctx = _T_wl[-1:]
            _T_wl = _T_wl[:-1]

            _T_wl = _T_wl[None]
            _T_ctx = _T_ctx[None]
            with _ts("compute/pcl_sample"):
                _pcl = np.concatenate(
                    [_local_pcl, _pcl_colors.astype(np.float32) / 255.0], axis=1
                )
                _pcl = sample_list(_pcl, self.num_pts, random_flag=self.random_flag)

            group_id = prompt.max_group_id + 1

            with _ts("compute/trace_append"):
                prompt.append_trace(
                    trace=_T_wl,
                    trace_mask=torch.ones(_T_wl.shape[:2]),
                    context=_T_ctx,
                    context_mask=torch.ones(_T_ctx.shape[:2]),
                    group_id=group_id,
                    sem_type=SEM_OBJECT,
                    uid=UID_BEHAVIOR_OBJECT,
                    local_pcl=_pcl[None],
                    local_pcl_mask=torch.ones(1),
                )

        # * Robot Links
        robot_links_to_add = [
            ("zed_link", UID_R1PRO_ZED, UID_R1PRO_ZED_ACT, SEM_CAMERA),
            (
                "left_gripper_link",
                UID_R1PRO_LEFT_GRIPPER,
                UID_R1PRO_LEFT_GRIPPER_ACT,
                SEM_MANIPULATOR,
            ),
            (
                "left_gripper_finger_link1",
                UID_R1PRO_LEFT_FINGER1,
                UID_R1PRO_LEFT_FINGER1_ACT,
                SEM_MANIPULATOR,
            ),
            (
                "left_gripper_finger_link2",
                UID_R1PRO_LEFT_FINGER2,
                UID_R1PRO_LEFT_FINGER2_ACT,
                SEM_MANIPULATOR,
            ),
            (
                "right_gripper_link",
                UID_R1PRO_RIGHT_GRIPPER,
                UID_R1PRO_RIGHT_GRIPPER_ACT,
                SEM_MANIPULATOR,
            ),
            (
                "right_gripper_finger_link1",
                UID_R1PRO_RIGHT_FINGER1,
                UID_R1PRO_RIGHT_FINGER1_ACT,
                SEM_MANIPULATOR,
            ),
            (
                "right_gripper_finger_link2",
                UID_R1PRO_RIGHT_FINGER2,
                UID_R1PRO_RIGHT_FINGER2_ACT,
                SEM_MANIPULATOR,
            ),
        ]

        for link_name, uid, uid_act, sem_type in robot_links_to_add:
            if link_name not in kinematic_states:
                continue

            _T_oracle = kinematic_states[link_name][start_tid:end_tid][::factor]
            _T_oracle = np.concatenate(
                [_T_oracle, kinematic_states[link_name][-1][None]], axis=0
            )

            # for the action, we want t+factor
            _T_oracle_act = []
            for t_idx in range(start_tid, end_tid, factor):
                next_t_idx = min(t_idx + factor, len(kinematic_states[link_name]) - 1)
                _T_oracle_act.append(kinematic_states[link_name][next_t_idx])
            _T_oracle_act = np.stack(_T_oracle_act, axis=0)
            _T_oracle_act = np.concatenate(
                [_T_oracle_act, _T_oracle_act[-1][None]], axis=0
            )

            # load GT PCL from embodiment
            pcl_key = f"{link_name}_pcl"
            if pcl_key not in self.robot_link_data:
                continue

            gt_pcl = self.robot_link_data[pcl_key]

            _T_ctx = _T_oracle[-1:]
            _T_wl = _T_oracle[:-1]

            _T_wl = _T_wl[None]
            _T_ctx = _T_ctx[None]

            _T_ctx_act = _T_oracle_act[-1:]
            _T_wl_act = _T_oracle_act[:-1]

            _T_wl_act = _T_wl_act[None]
            _T_ctx_act = _T_ctx_act[None]

            # Colors for robot
            c_key = f"{link_name}_pcl_c"
            if c_key in self.robot_link_data:
                gt_c = self.robot_link_data[c_key]
            else:
                gt_c = np.ones_like(gt_pcl)

            with _ts("compute/pcl_sample"):
                _pcl = np.concatenate([gt_pcl, gt_c.astype(np.float32)], axis=1)
                _pcl = sample_list(_pcl, self.num_pts, random_flag=self.random_flag)

            group_id = prompt.max_group_id + 1

            with _ts("compute/trace_append"):
                prompt.append_trace(
                    trace=_T_wl,
                    trace_mask=torch.ones(_T_wl.shape[:2]),
                    context=_T_ctx,
                    context_mask=torch.ones(_T_ctx.shape[:2]),
                    group_id=group_id,
                    sem_type=sem_type,
                    uid=uid,
                    local_pcl=_pcl[None],
                    local_pcl_mask=torch.ones(1),
                )

                prompt.append_trace(
                    trace=_T_wl_act,
                    trace_mask=torch.ones(_T_wl_act.shape[:2]),
                    context=_T_ctx_act,
                    context_mask=torch.ones(_T_ctx_act.shape[:2]),
                    group_id=group_id,
                    sem_type=SEM_BEHAVIOR_ACT,
                    uid=uid_act,
                    local_pcl=_pcl[None],
                    local_pcl_mask=torch.ones(1),
                )

        extra_dict["_center_xy"] = _center_xy
        # Pass camera (zed_link) world pose for global yaw alignment.
        if "zed_link" in kinematic_states and current_abs_tid < len(
            kinematic_states["zed_link"]
        ):
            extra_dict["_cam_T_wc"] = kinematic_states["zed_link"][current_abs_tid]
        return prompt, extra_dict

    def __extract_text_instruction__(
        self,
        seq_id: str,
        start_abs_tid: int,
        end_abs_tid: int,
    ) -> str:
        """Extract text instruction for a sequence, filtering skills by time window.

        Combines task instruction, activity name, and skill descriptions that
        overlap the ``[start_abs_tid, end_abs_tid)`` sampling window into a
        single period-separated string.

        Args:
            seq_id: Sequence identifier used to locate the text JSON file.
            start_abs_tid: Start of the sampling window (inclusive).
            end_abs_tid: End of the sampling window (exclusive).

        Returns:
            A period-joined instruction string, or ``""`` if no text is available.

        Example::

            text = self.__extract_text_instruction__("seq_001", 10, 40)
        """
        text_data = self.io.load_text(
            osp.join(self.data_root, self.text_dirname, seq_id + ".json")
        )
        parts: list[str] = []
        instr = text_data.get("task_instruction", "")
        activity = text_data.get("activity_name", "")

        if self.text_mode == "task":
            if instr:
                parts.append(instr)
        elif self.text_mode == "activity":
            if activity:
                parts.append(activity)
        elif self.text_mode == "skill":
            self._append_skill_parts(parts, text_data, start_abs_tid, end_abs_tid)
        else:  # "full"
            if instr:
                parts.append(instr)
            if activity and activity != instr:
                parts.append(activity)
            self._append_skill_parts(parts, text_data, start_abs_tid, end_abs_tid)

        return ". ".join(parts) if parts else ""

    @staticmethod
    def _append_skill_parts(
        parts: list[str],
        text_data: dict,
        start_abs_tid: int,
        end_abs_tid: int,
    ) -> None:
        """Append time-filtered skill descriptions to *parts* in-place.

        Args:
            parts: Accumulator list to extend with matching skill strings.
            text_data: Parsed text JSON dict containing a ``"skills"`` list.
            start_abs_tid: Start of the sampling window (inclusive).
            end_abs_tid: End of the sampling window (exclusive).

        Example::

            parts: list[str] = []
            BehaviorDataset._append_skill_parts(parts, text_data, 10, 40)
        """
        for sk in text_data.get("skills", []):
            sk_start = sk.get("start", 0)
            sk_end = sk.get("end", float("inf"))
            if sk_start >= end_abs_tid or sk_end <= start_abs_tid:
                continue
            desc = sk.get("description", "")
            raw_objs = sk.get("objects", [])
            objs: list[str] = []
            for o in raw_objs:
                if isinstance(o, list):
                    objs.extend(str(x) for x in o)
                else:
                    objs.append(str(o))
            if desc and objs:
                parts.append(f"{desc} {', '.join(objs)}")
            elif desc:
                parts.append(desc)

    def __getitem_inner__(self, idx):
        """Build one training sample from the BEHAVIOR-1K dataset.

        Performs temporal sampling, constructs object and robot-link traces
        with back-projected point clouds, centres and yaw-aligns all poses
        to the robot camera frame, and extracts the text instruction.

        Args:
            idx: Dataset index (will be wrapped via modulo).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_dict)`` ready for
            collation by :meth:`CanonicalDatasetOutput.collate`.

        Example::

            sample, extra = dataset.__getitem_inner__(10)
        """
        idx = idx % len(self)
        prompt = PromptV0(P=0, T=self.max_T, C=1, PCL=self.num_pts, PCL_dim=6)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )

        prompt, extra_dict = self.__add_behavior_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid
        )

        _ts_item = self._profiler.time_section if self._profiler else _null_ts

        prompt.catch_P()
        with _ts_item("compute/prompt_cat"):
            (
                trace,
                trace_mask,
                context,
                context_mask,
                rot_mask,
                group_id,
                sem_type,
                uid,
            ) = prompt.cat()
            pcl, pcl_mask = prompt.cat_pcl()

        # Shift all poses so the robot center sits at the XY origin.
        _center_xy = extra_dict.pop("_center_xy")
        _shift = torch.zeros(3)
        _shift[:2] = torch.from_numpy(_center_xy)
        trace[..., :3, 3] -= _shift
        context[..., :3, 3] -= _shift

        # Rotate around Z so the camera optical axis (+Z_cam) projects onto +X
        # and camera X axis projects onto -Y in the world XY plane.
        _cam_T_wc = extra_dict.pop("_cam_T_wc", None)
        if _cam_T_wc is not None:
            cam_z_xy = _cam_T_wc[:2, 2]  # optical axis projected to XY
            yaw = float(np.arctan2(cam_z_xy[1], cam_z_xy[0]))
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            R_z = torch.eye(4, dtype=trace.dtype)
            R_z[0, 0] = c
            R_z[0, 1] = s
            R_z[1, 0] = -s
            R_z[1, 1] = c
            trace = R_z @ trace
            context = R_z @ context

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        with _ts_item("io/text"):
            text = self.__extract_text_instruction__(sid, start_abs_tid, end_abs_tid)

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

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """DataLoader worker initialiser that pins each worker to 1 CPU thread.

        Pass this to ``DataLoader(worker_init_fn=BehaviorDataset.worker_init_fn)``
        so that numpy/OpenBLAS/MKL operations inside each worker process do not
        spawn additional threads, preventing CPU over-subscription when multiple
        workers run concurrently.

        Example::

            loader = DataLoader(dataset, num_workers=4,
                                worker_init_fn=BehaviorDataset.worker_init_fn)
        """
        import threadpoolctl, os

        threadpoolctl.threadpool_limits(limits=1)

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        return

    def get_profiler_summary(self) -> dict | None:
        """Return profiling stats as a JSON-serializable dict, or None if profiling is off.

        Usage example::

            dataset = BehaviorDataset(..., profile=True)
            for i in range(10):
                dataset[i]
            summary = dataset.get_profiler_summary()
        """
        if self._profiler is None:
            return None
        return self._profiler.get_summary_dict()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    PROJECT_ROOT = osp.dirname(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    )

    dataset = BehaviorDataset(
        data_root="/datasets/pointw-internal/current/behavior_proc_debug",
        split_fn=osp.join(PROJECT_ROOT, "data_splits/behavior/debug_train.json"),
        safe_flag=False,
        max_T=30,
        profile=True,
        return_details=True,
        views=["head"],
        t_rate=1,
    )

    if len(dataset) > 0:
        batch, extra = dataset[10]
        print("Trace shape:", batch.trace.shape)
        print("Context shape:", batch.context.shape)
        print("PCL shape:", batch.local_pcl.shape)
        if "rgbs" in extra:
            print("RGBs shape:", extra["rgbs"].shape)
            print("Depths shape:", extra["depths"].shape)
            print("Masks shape:", extra["masks"].shape)
    else:
        print("Dataset is empty. Check split_fn or data_root.")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=12,
        collate_fn=CanonicalDatasetOutput.collate,
        worker_init_fn=BehaviorDataset.worker_init_fn,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        pass
        # print(batch["trace"].shape)
        # if i > 10:
        #     break
    print(dataset.get_profiler_summary())
