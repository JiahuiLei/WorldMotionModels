"""BEHAVIOR-1K v2 dataset with pixel UV sampling for BH-2.0.

Inherits from the v1 :class:`BehaviorDataset` and overrides the prompt
construction to use pixel UV sampling instead of centroid-based queries.

Key differences from v1:
1. **Pixel UV sampling**: For each visible UID, sample
   ``num_pixel_queries_per_uid`` random pixel locations from its segmentation
   mask. Each sampled pixel becomes a separate query (separate P in output).
2. **Query SE(3)**: For each sampled pixel, backproject to world XYZ and
   create T_wq = (pixel_xyz, identity_rotation). Warp the same object's
   trajectory to this query frame.
3. **Degenerate local PCL for objects**: Object pixel queries use zeros
   for local_pcl (local_pcl_mask=False). The model will use DINO image
   features instead of PointNet.
4. **Real mesh PCL for robot links**: Robot links keep their real mesh
   point clouds (local_pcl_mask=True), same as V1.
5. **RGB images**: Load and resize current-time RGB images for all views
   to ``image_size x image_size``. Return as (V, 3, H, W) float32 in
   extra_dict["images"].
6. **Pixel UV tracking**: extra_dict stores pixel_uv_normalized (P, 2),
   pixel_view_idx (P,), and pixel_query_mask (P,) for DINO feature lookup.

Example::

    dataset = BehaviorV2Dataset(
        data_root="/data/behavior_proc",
        split_fn="splits/train.json",
        max_T=30,
        views=["head"],
        num_pixel_queries_per_uid=4,
        image_size=224,
    )
    sample, extra = dataset[0]
"""

import os.path as osp
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import logging

from umm.datasets.umm_v6_all.behavior_dataset import BehaviorDataset
from umm.datasets.umm_v6_all.base_dataset import PromptV0, CanonicalDatasetOutput
from umm.datasets.umm_v6_all.dataset_utils import sample_list
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


@contextmanager
def _null_ts(_name: str):
    """No-op timing context used when profiling is disabled."""
    yield


class BehaviorV2Dataset(BehaviorDataset):
    """BEHAVIOR-1K v2 dataset with pixel UV sampling and RGB images.

    Extends the v1 BehaviorDataset with:
    - Per-pixel query sampling from instance segmentation masks
    - Degenerate (zero) local point clouds
    - Resized RGB images for DINO encoding

    Args:
        num_pixel_queries_per_uid: Number of pixel UV queries to sample
            per visible UID (default 4). Each becomes a separate part P.
        image_size: Target size for RGB image resize (default 224).
            Images are resized to (image_size, image_size).
        **kwargs: All other arguments forwarded to :class:`BehaviorDataset`.

    Example::

        ds = BehaviorV2Dataset(
            data_root="/data/behavior_proc",
            split_fn="splits/train.json",
            max_T=30,
            views=["head"],
            num_pixel_queries_per_uid=4,
            image_size=224,
        )
        sample, extra = ds[0]
        # extra["images"] has shape (V, 3, 224, 224) float32
    """

    # Number of robot link parts: 7 links × 2 (state + action)
    NUM_ROBOT_PARTS = 14

    def __init__(
        self,
        *args,
        num_pixel_queries_per_uid: int = 4,
        fixed_P: int = 0,
        image_size: int = 224,
        **kwargs,
    ):
        """Initialize BehaviorV2Dataset with pixel query and image params.

        Args:
            num_pixel_queries_per_uid: Pixel queries per UID (legacy, used
                when fixed_P=0).
            fixed_P: If >0, total parts per sample (fixed). Overrides
                num_pixel_queries_per_uid. Pixel queries = fixed_P - 14.
            image_size: RGB resize target (square).
            *args: Positional args for BehaviorDataset.
            **kwargs: Keyword args for BehaviorDataset.
        """
        super().__init__(*args, **kwargs)
        self.num_pixel_queries_per_uid = num_pixel_queries_per_uid
        self.fixed_P = fixed_P
        self.image_size = image_size
        if fixed_P > 0:
            self.fixed_pixel_P = fixed_P - self.NUM_ROBOT_PARTS
            assert (
                self.fixed_pixel_P > 0
            ), f"fixed_P={fixed_P} too small for {self.NUM_ROBOT_PARTS} robot parts"

    def __add_behavior_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,
        start_tid,
        end_tid,
        seq_id,
    ):
        """Populate prompt with pixel UV sampled queries and RGB images.

        Instead of centroid queries, samples ``num_pixel_queries_per_uid``
        random pixel locations per UID from the segmentation mask, backprojects
        each to world XYZ, and creates a separate query for each pixel.

        Local point clouds are set to zeros (degenerate) since the model
        will use DINO features from the RGB images instead.

        RGB images for all views are resized to ``image_size x image_size``
        and stored in extra_dict["images"] as (V, 3, H, W) float32 [0, 1].

        Args:
            prompt: Mutable PromptV0 accumulator to extend.
            cur_rel_idx: Relative index of the anchor frame within the window.
            start_tid: Absolute start frame (inclusive).
            end_tid: Absolute end frame (exclusive).
            seq_id: Sequence identifier string.

        Returns:
            Tuple (prompt, extra_dict).
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

        current_abs_tid = start_tid + cur_rel_idx * factor

        _prof = self._profiler
        _ts = _prof.time_section if _prof else _null_ts

        # ── Load states ──────────────────────────────────────────────
        with _ts("io/state_npz"):
            state_data = self.io.load_npzs(
                osp.join(self.data_root, self.state_dirname, seq_id + ".npz"),
            )
        sem_type_uid = state_data["sem_type_uid"].item()
        states_all = state_data["states"].item()
        kinematic_states = state_data.get("kinematic_states", {}).item()

        task_objects = sem_type_uid.get("task_objects", [])
        static_background = sem_type_uid.get("static_background", [])

        # Robot center for XY range filtering
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

        # ── Load views: RGB, depth, instance seg, camera ─────────────
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

        # ── RGB images for DINO: resize to image_size x image_size ───
        images_for_dino = []
        with _ts("compute/rgb_resize"):
            for rgb in rgbs_list:
                # rgb is (H, W, 3) uint8
                resized = cv2.resize(
                    rgb,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )  # (image_size, image_size, 3) uint8
                # Normalize to [0, 1] float32 and transpose to CHW
                resized_f = resized.astype(np.float32) / 255.0  # (H, W, 3) float32
                resized_chw = resized_f.transpose(2, 0, 1)  # (3, H, W) float32
                images_for_dino.append(resized_chw)

        if images_for_dino:
            extra_dict["images"] = np.stack(images_for_dino, axis=0)
            # (V, 3, image_size, image_size) float32
        else:
            extra_dict["images"] = np.zeros(
                (0, 3, self.image_size, self.image_size), dtype=np.float32
            )

        # ── Precompute per-view world XYZ + normalized UV for backprojection ──
        # Build per-view, per-UID: (chunk_xyz, chunk_uv_normalized)
        uid_pcl_per_view: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        all_visible: set[int] = set()

        with _ts("compute/view_precompute"):
            flat_depth = [d.ravel() for d in depths_list]

            for v_idx in range(len(masks_list)):
                flat_inst = masks_list[v_idx].ravel()
                v_visible = set(np.unique(flat_inst).tolist()) - {0}

                uid_data_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                if v_visible:
                    uid_order: list[int] = []
                    uid_idx_chunks: list[np.ndarray] = []
                    boundaries: list[tuple[int, int]] = []
                    offset = 0
                    for uid_val in v_visible:
                        idx = np.flatnonzero(flat_inst == uid_val)
                        # For V2 we don't need _uv_sample_budget since
                        # we only sample num_pixel_queries_per_uid pixels.
                        # But we still cap to avoid huge arrays for large masks.
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

                    uv_f32 = np.stack([u, v], axis=-1)  # (N, 2)
                    cam_xy = uv_f32 @ K_inv_f32[:2, :2].T + K_inv_f32[:2, 2]
                    cam_xyz_all = np.concatenate(
                        [cam_xy * d_all[:, None], d_all[:, None]], axis=-1
                    )  # (N, 3)
                    world_xyz_all = (
                        cam_xyz_all @ T_wc_f32[:3, :3].T + T_wc_f32[:3, 3]
                    )  # (N, 3) float32

                    # Normalized UV: u / (W-1), v / (H-1) -> [0, 1]
                    W_denom = max(W - 1, 1)
                    H_denom = max(H - 1, 1)
                    uv_normalized_all = np.stack(
                        [u / W_denom, v / H_denom], axis=-1
                    )  # (N, 2) float32

                    for uid_val, (lo, hi) in zip(uid_order, boundaries):
                        chunk_xyz = world_xyz_all[lo:hi]
                        chunk_uv = uv_normalized_all[lo:hi]  # (M, 2)
                        if self.max_range > 0:
                            dist_xy = np.sqrt(
                                (chunk_xyz[:, 0] - _center_xy[0]) ** 2
                                + (chunk_xyz[:, 1] - _center_xy[1]) ** 2
                            )
                            valid = dist_xy <= self.max_range
                            chunk_xyz = chunk_xyz[valid]
                            chunk_uv = chunk_uv[valid]
                        if len(chunk_xyz) > 0:
                            uid_data_dict[uid_val] = (chunk_xyz, chunk_uv)

                uid_pcl_per_view.append(uid_data_dict)
                all_visible.update(uid_data_dict.keys())

        # ── Per-part pixel UV tracking ────────────────────────────────
        per_part_uv: list[np.ndarray] = []  # (2,) normalized UV per part
        per_part_view: list[int] = []  # view index per part
        per_part_query_mask: list[bool] = []  # True for object pixel queries
        per_part_viz_pcl: list[np.ndarray] = []  # (num_pts, 6) query-local PCL per part

        # ── Precompute per-UID trajectory + per-UID pixel data ────────
        uid_trajectory: dict[int, np.ndarray] = {}  # uid -> (T+1, 4, 4)
        uid_world_pcl: dict[int, np.ndarray] = {}  # uid -> (N, 3)
        uid_uv_norm: dict[int, np.ndarray] = {}  # uid -> (N, 2)
        uid_view_idx: dict[int, np.ndarray] = {}  # uid -> (N,)

        for uid in target_uids:
            if uid not in all_visible:
                continue
            uid_str = str(uid)
            if uid_str not in states_all:
                continue

            _T_oracle = states_all[uid_str][start_tid:end_tid][::factor]
            _T_oracle = np.concatenate(
                [_T_oracle, states_all[uid_str][-1][None]], axis=0
            )  # (T+1, 4, 4)
            uid_trajectory[uid] = _T_oracle

            # Gather all world XYZ + UV for this UID across views
            _xyz_parts = []
            _uv_parts = []
            _vi_parts = []
            with _ts("compute/backproject"):
                for v_idx, v_pcl in enumerate(uid_pcl_per_view):
                    entry = v_pcl.get(uid)
                    if entry is None:
                        continue
                    chunk_xyz, chunk_uv = entry  # (M, 3), (M, 2)
                    _xyz_parts.append(chunk_xyz)
                    _uv_parts.append(chunk_uv)
                    _vi_parts.append(np.full(len(chunk_xyz), v_idx, dtype=np.int64))

            if not _xyz_parts:
                continue

            with _ts("compute/pcl_concat"):
                uid_world_pcl[uid] = np.concatenate(_xyz_parts, axis=0)
                uid_uv_norm[uid] = np.concatenate(_uv_parts, axis=0)
                uid_view_idx[uid] = np.concatenate(_vi_parts, axis=0)

        # ── Build candidate pool and sample pixels ────────────────────
        if self.fixed_P > 0:
            # Fixed-P mode: pool ALL candidates across UIDs, sample exactly
            # fixed_pixel_P points uniformly.
            pool_uid = []  # uid for each candidate
            pool_xyz = []  # world xyz
            pool_uv = []  # normalized uv
            pool_view = []  # view index
            pool_local = []  # index within uid arrays

            for uid in uid_world_pcl:
                n = len(uid_world_pcl[uid])
                pool_uid.append(np.full(n, uid, dtype=np.int64))
                pool_xyz.append(uid_world_pcl[uid])
                pool_uv.append(uid_uv_norm[uid])
                pool_view.append(uid_view_idx[uid])
                pool_local.append(np.arange(n, dtype=np.int64))

            if pool_xyz:
                pool_uid_arr = np.concatenate(pool_uid)
                pool_xyz_arr = np.concatenate(pool_xyz)
                pool_uv_arr = np.concatenate(pool_uv)
                pool_view_arr = np.concatenate(pool_view)

                n_total = len(pool_xyz_arr)
                replace = n_total < self.fixed_pixel_P
                if self.random_flag:
                    chosen = np.random.choice(
                        n_total, size=self.fixed_pixel_P, replace=replace
                    )
                else:
                    if n_total >= self.fixed_pixel_P:
                        step = max(1, n_total // self.fixed_pixel_P)
                        chosen = np.arange(0, n_total, step)[: self.fixed_pixel_P]
                        if len(chosen) < self.fixed_pixel_P:
                            chosen = np.concatenate(
                                [
                                    chosen,
                                    np.zeros(
                                        self.fixed_pixel_P - len(chosen), dtype=np.int64
                                    ),
                                ]
                            )
                    else:
                        chosen = np.arange(n_total)
                        extra = np.zeros(self.fixed_pixel_P - n_total, dtype=np.int64)
                        chosen = np.concatenate([chosen, extra])

                # Build per-query data from pooled selection
                selected_uids = pool_uid_arr[chosen]
                selected_xyz = pool_xyz_arr[chosen]
                selected_uv = pool_uv_arr[chosen]
                selected_view = pool_view_arr[chosen]
            else:
                selected_uids = np.array([], dtype=np.int64)
                selected_xyz = np.zeros((0, 3), dtype=np.float32)
                selected_uv = np.zeros((0, 2), dtype=np.float32)
                selected_view = np.array([], dtype=np.int64)
        else:
            # Legacy per-UID mode: sample num_pixel_queries_per_uid per UID
            selected_uids_list = []
            selected_xyz_list = []
            selected_uv_list = []
            selected_view_list = []

            for uid in uid_world_pcl:
                _world_pcl = uid_world_pcl[uid]
                _all_uvs = uid_uv_norm[uid]
                _all_views = uid_view_idx[uid]
                n_avail = len(_world_pcl)
                n_queries = min(self.num_pixel_queries_per_uid, n_avail)

                if self.random_flag:
                    idx = np.random.choice(n_avail, n_queries, replace=False)
                else:
                    step = max(1, n_avail // n_queries)
                    idx = np.arange(0, n_avail, step)[:n_queries]

                selected_uids_list.append(np.full(n_queries, uid, dtype=np.int64))
                selected_xyz_list.append(_world_pcl[idx])
                selected_uv_list.append(_all_uvs[idx])
                selected_view_list.append(_all_views[idx])

            if selected_xyz_list:
                selected_uids = np.concatenate(selected_uids_list)
                selected_xyz = np.concatenate(selected_xyz_list)
                selected_uv = np.concatenate(selected_uv_list)
                selected_view = np.concatenate(selected_view_list)
            else:
                selected_uids = np.array([], dtype=np.int64)
                selected_xyz = np.zeros((0, 3), dtype=np.float32)
                selected_uv = np.zeros((0, 2), dtype=np.float32)
                selected_view = np.array([], dtype=np.int64)

        # ── Create PromptParts for selected pixel queries ─────────────
        with _ts("compute/pixel_queries"):
            for qi in range(len(selected_uids)):
                uid = int(selected_uids[qi])
                pixel_xyz = selected_xyz[qi]
                pixel_uv = selected_uv[qi]
                pixel_view = int(selected_view[qi])

                _T_oracle = uid_trajectory[uid]
                group_id = prompt.max_group_id + 1

                # Create T_wq at this pixel's world position, identity rotation
                T_wq = np.eye(4, dtype=np.float32)
                T_wq[:3, 3] = pixel_xyz  # (3,)

                # Warp object trajectory to this query frame
                _T_wl = self.__warp_frame__(T_wq, cur_rel_idx, _T_oracle)
                _T_ctx = _T_wl[-1:]  # (1, 4, 4)
                _T_wl = _T_wl[:-1]  # (T, 4, 4)

                _T_wl = _T_wl[None]  # (1, T, 4, 4)
                _T_ctx = _T_ctx[None]  # (1, 1, 4, 4)

                # Degenerate local PCL: zeros
                _pcl = np.zeros(
                    (1, self.num_pts, 6), dtype=np.float32
                )  # (1, num_pts, 6)

                with _ts("compute/trace_append"):
                    prompt.append_trace(
                        trace=_T_wl,
                        trace_mask=torch.ones(_T_wl.shape[:2]),
                        context=_T_ctx,
                        context_mask=torch.ones(_T_ctx.shape[:2]),
                        group_id=group_id,
                        sem_type=SEM_OBJECT,
                        uid=UID_BEHAVIOR_OBJECT,
                        local_pcl=_pcl,
                        local_pcl_mask=torch.zeros(1),
                    )

                # Track pixel UV info for this object query part
                per_part_uv.append(pixel_uv.astype(np.float32))
                per_part_view.append(pixel_view)
                per_part_query_mask.append(True)

                # Store world PCL in query-local frame for viz
                _uid_pcl = uid_world_pcl[uid]
                _uid_views = uid_view_idx[uid]
                _viz_pcl_xyz = _uid_pcl - pixel_xyz[None, :]  # (N, 3)

                # Sample RGB from image at backprojected pixel locations
                _rgb_img = rgbs_list[pixel_view]  # (H, W, 3) uint8
                _H_img, _W_img = _rgb_img.shape[:2]
                _uid_uvs = uid_uv_norm[uid]
                _pix_u = np.clip(
                    (_uid_uvs[:, 0] * (_W_img - 1)).astype(int), 0, _W_img - 1
                )
                _pix_v = np.clip(
                    (_uid_uvs[:, 1] * (_H_img - 1)).astype(int), 0, _H_img - 1
                )
                _same_view_mask = _uid_views == pixel_view
                _viz_rgb = (
                    np.ones((len(_uid_pcl), 3), dtype=np.float32) * 0.5
                )  # default gray
                _viz_rgb[_same_view_mask] = (
                    _rgb_img[_pix_v[_same_view_mask], _pix_u[_same_view_mask]].astype(
                        np.float32
                    )
                    / 255.0
                )  # (N_view, 3) float [0,1]

                _viz_pcl_full = np.concatenate(
                    [_viz_pcl_xyz, _viz_rgb], axis=1
                )  # (N, 6) xyz_local + rgb

                _viz_pcl_sampled = sample_list(
                    _viz_pcl_full, self.num_pts, random_flag=self.random_flag
                )  # (num_pts, 6)
                per_part_viz_pcl.append(_viz_pcl_sampled)

        # ── Robot links (real mesh PCL, same as V1) ─────────────────
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

            # Action: t+factor
            _T_oracle_act = []
            for t_idx in range(start_tid, end_tid, factor):
                next_t_idx = min(t_idx + factor, len(kinematic_states[link_name]) - 1)
                _T_oracle_act.append(kinematic_states[link_name][next_t_idx])
            _T_oracle_act = np.stack(_T_oracle_act, axis=0)
            _T_oracle_act = np.concatenate(
                [_T_oracle_act, _T_oracle_act[-1][None]], axis=0
            )

            # Load GT PCL from embodiment (same as V1)
            pcl_key = f"{link_name}_pcl"
            if pcl_key not in self.robot_link_data:
                continue

            gt_pcl = self.robot_link_data[pcl_key]

            _T_ctx = _T_oracle[-1:]
            _T_wl = _T_oracle[:-1]
            _T_wl = _T_wl[None]  # (1, T, 4, 4)
            _T_ctx = _T_ctx[None]  # (1, 1, 4, 4)

            _T_ctx_act = _T_oracle_act[-1:]
            _T_wl_act = _T_oracle_act[:-1]
            _T_wl_act = _T_wl_act[None]  # (1, T, 4, 4)
            _T_ctx_act = _T_ctx_act[None]  # (1, 1, 4, 4)

            # Real mesh PCL with colors (same as V1)
            c_key = f"{link_name}_pcl_c"
            if c_key in self.robot_link_data:
                gt_c = self.robot_link_data[c_key]
            else:
                gt_c = np.ones_like(gt_pcl)

            with _ts("compute/pcl_sample"):
                _pcl = np.concatenate(
                    [gt_pcl, gt_c.astype(np.float32)], axis=1
                )  # (N, 6) xyz+rgb
                _pcl = sample_list(
                    _pcl, self.num_pts, random_flag=self.random_flag
                )  # (num_pts, 6)

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
                    local_pcl=_pcl[None],  # (1, num_pts, 6)
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
                    local_pcl=_pcl[None],  # (1, num_pts, 6)
                    local_pcl_mask=torch.ones(1),
                )

            # Dummy UV data for robot link parts (2 parts: state + action)
            for _ in range(2):
                per_part_uv.append(np.zeros(2, dtype=np.float32))
                per_part_view.append(0)
                per_part_query_mask.append(False)
                per_part_viz_pcl.append(np.zeros((self.num_pts, 6), dtype=np.float32))

        # ── Store pixel UV data in extra_dict ─────────────────────────
        n_parts = len(per_part_uv)
        if n_parts > 0:
            extra_dict["pixel_uv_normalized"] = np.stack(
                per_part_uv, axis=0
            )  # (P, 2) float32
            extra_dict["pixel_view_idx"] = np.array(
                per_part_view, dtype=np.int64
            )  # (P,)
            extra_dict["pixel_query_mask"] = np.array(
                per_part_query_mask, dtype=bool
            )  # (P,)
        else:
            extra_dict["pixel_uv_normalized"] = np.zeros((0, 2), dtype=np.float32)
            extra_dict["pixel_view_idx"] = np.zeros(0, dtype=np.int64)
            extra_dict["pixel_query_mask"] = np.zeros(0, dtype=bool)

        extra_dict["_center_xy"] = _center_xy
        if per_part_viz_pcl:
            extra_dict["viz_obj_pcl"] = np.stack(
                per_part_viz_pcl, axis=0
            )  # (P, num_pts, 6) float32 — query-local xyz + rgb
        else:
            extra_dict["viz_obj_pcl"] = np.zeros((0, self.num_pts, 6), dtype=np.float32)
        # Pass camera (zed_link) world pose for global yaw alignment.
        if "zed_link" in kinematic_states and current_abs_tid < len(
            kinematic_states["zed_link"]
        ):
            extra_dict["_cam_T_wc"] = kinematic_states["zed_link"][current_abs_tid]
        return prompt, extra_dict
