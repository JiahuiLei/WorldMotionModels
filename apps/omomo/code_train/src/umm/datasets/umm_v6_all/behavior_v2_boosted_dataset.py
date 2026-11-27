"""Boosted BEHAVIOR-1K v2 dataset — vectorized pixel query construction.

Drop-in replacement for :class:`BehaviorV2Dataset` that replaces the
per-pixel-query loop (82% of __getitem__ time) with batched numpy operations.

v7 optimizations over v6:
- Completely bypasses PromptV0 — all arrays built in numpy, single torch
  conversion at the end. Eliminates ~14ms of torch.as_tensor/torch.cat overhead.
- Persistent ThreadPoolExecutor — lazy init, reuse across __getitem__ calls.
- Center shift + yaw rotation done in numpy before torch conversion.
- Padding done in numpy (avoids torch.eye expansion).

Same output, same semantics as BehaviorV2Dataset.
"""

import os.path as osp
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from umm.datasets.umm_v6_all.behavior_dataset import BehaviorDataset
from umm.datasets.umm_v6_all.base_dataset import CanonicalDatasetOutput
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

# Pre-allocated identity matrices (avoid repeated allocation)
_EYE4_F32 = np.eye(4, dtype=np.float32)


@contextmanager
def _null_ts(_name: str):
    """No-op timing context used when profiling is disabled."""
    yield


# ── Robot link table (class-level constant) ───────────────────────────
_ROBOT_LINKS = [
    ("zed_link", UID_R1PRO_ZED, UID_R1PRO_ZED_ACT, SEM_CAMERA),
    ("left_gripper_link", UID_R1PRO_LEFT_GRIPPER, UID_R1PRO_LEFT_GRIPPER_ACT, SEM_MANIPULATOR),
    ("left_gripper_finger_link1", UID_R1PRO_LEFT_FINGER1, UID_R1PRO_LEFT_FINGER1_ACT, SEM_MANIPULATOR),
    ("left_gripper_finger_link2", UID_R1PRO_LEFT_FINGER2, UID_R1PRO_LEFT_FINGER2_ACT, SEM_MANIPULATOR),
    ("right_gripper_link", UID_R1PRO_RIGHT_GRIPPER, UID_R1PRO_RIGHT_GRIPPER_ACT, SEM_MANIPULATOR),
    ("right_gripper_finger_link1", UID_R1PRO_RIGHT_FINGER1, UID_R1PRO_RIGHT_FINGER1_ACT, SEM_MANIPULATOR),
    ("right_gripper_finger_link2", UID_R1PRO_RIGHT_FINGER2, UID_R1PRO_RIGHT_FINGER2_ACT, SEM_MANIPULATOR),
]


class BehaviorV2BoostedDataset(BehaviorDataset):
    """Boosted BEHAVIOR-1K v2 dataset with vectorized pixel query construction.

    v7: Completely bypasses PromptV0. All trajectory data accumulated as numpy
    arrays, with a single torch.from_numpy conversion at the end of __getitem__.

    Args:
        Same as BehaviorV2Dataset.
    """

    NUM_ROBOT_PARTS = 14

    def __init__(
        self,
        *args,
        num_pixel_queries_per_uid: int = 4,
        fixed_P: int = 0,
        image_size: int = 224,
        npz_cache_size: int = 128,
        mask_downsample: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_pixel_queries_per_uid = num_pixel_queries_per_uid
        self.fixed_P = fixed_P
        self.image_size = image_size
        self.mask_downsample = mask_downsample
        if fixed_P > 0:
            self.fixed_pixel_P = fixed_P - self.NUM_ROBOT_PARTS
            assert (
                self.fixed_pixel_P > 0
            ), f"fixed_P={fixed_P} too small for {self.NUM_ROBOT_PARTS} robot parts"

        # LRU cache for state NPZ files (shared across samples from same seq)
        self._npz_cache_size = npz_cache_size
        self._state_cache: dict = {}
        self._cam_cache: dict = {}
        # Persistent thread pool (lazy init — survives across __getitem__ calls
        # since DataLoader uses persistent_workers=True)
        self._thread_pool = None

    def _get_thread_pool(self):
        """Get or create a persistent ThreadPoolExecutor."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=3)
        return self._thread_pool

    def _load_state_cached(self, path: str) -> dict:
        """Load state NPZ with per-worker LRU cache."""
        if path in self._state_cache:
            return self._state_cache[path]
        data = self.io.load_npzs(path)
        if len(self._state_cache) >= self._npz_cache_size:
            oldest = next(iter(self._state_cache))
            del self._state_cache[oldest]
        self._state_cache[path] = data
        return data

    def _load_cam_cached(self, path: str) -> dict:
        """Load camera NPZ with per-worker LRU cache."""
        if path in self._cam_cache:
            return self._cam_cache[path]
        data = self.io.load_npzs(path)
        if len(self._cam_cache) >= self._npz_cache_size:
            oldest = next(iter(self._cam_cache))
            del self._cam_cache[oldest]
        self._cam_cache[path] = data
        return data

    def __add_behavior_prompt__(
        self,
        cur_rel_idx,
        start_tid,
        end_tid,
        seq_id,
    ):
        """Build all prompt arrays in numpy — no PromptV0 involved.

        Returns:
            Tuple of (arrays_dict, extra_dict) where arrays_dict contains
            all SE3 trace data as numpy arrays ready for torch conversion.
        """
        extra_dict = {}
        factor = int(self.t_rate)
        T_max = self.max_T
        PCL = self.num_pts

        extra_dict["rgbs_paths"] = [
            osp.join(self.data_root, self.image_dirname, f"{seq_id}_{view}.zip")
            for view in self.views
        ]
        extra_dict["rgbs_views"] = list(self.views)
        extra_dict["rgbs_indices"] = [t for t in range(start_tid, end_tid, factor)]

        current_abs_tid = start_tid + cur_rel_idx * factor

        _prof = self._profiler
        _ts = _prof.time_section if _prof else _null_ts

        # Accumulator lists for all parts (pixel queries + robot links)
        # These are all numpy arrays — no torch until __getitem_inner__
        trace_parts = []         # list of (n_i, T_max, 4, 4) float32
        trace_mask_parts = []    # list of (n_i, T_max) float32
        context_parts = []       # list of (n_i, 1, 4, 4) float32
        context_mask_parts = []  # list of (n_i, 1) float32
        rot_mask_parts = []      # list of (n_i,) bool
        group_id_parts = []      # list of (n_i,) int64
        sem_type_parts = []      # list of (n_i,) int64
        uid_parts = []           # list of (n_i,) int64
        pcl_parts = []           # list of (n_i, PCL, 6) float32
        pcl_mask_parts = []      # list of (n_i,) float32
        next_group_id = 0

        # ── Load states ──────────────────────────────────────────────
        with _ts("io/state_npz"):
            state_data = self._load_state_cached(
                osp.join(self.data_root, self.state_dirname, seq_id + ".npz"),
            )
        sem_type_uid = state_data["sem_type_uid"].item()
        states_all = state_data["states"].item()
        kinematic_states = state_data.get("kinematic_states", {}).item()

        task_objects = sem_type_uid.get("task_objects", [])
        static_background = sem_type_uid.get("static_background", [])

        _robot_positions = []
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

        executor = self._get_thread_pool()

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

            # Parallel IO: load RGB, depth, inst seg concurrently
            with _ts(f"io/parallel_view_{view}"):
                rgb_future = executor.submit(
                    self.io.load_rgbs, rgb_path, [current_abs_tid]
                )
                dep_future = executor.submit(
                    self.io.load_depths, dep_path, [current_abs_tid]
                )
                inst_future = executor.submit(
                    self.io.load_inst_segs, inst_path,
                    1, [current_abs_tid], target_uids_set,
                )
                rgbs = rgb_future.result()
                deps = dep_future.result()
                insts = inst_future.result()

            depth_frame = np.squeeze(deps[0]).astype(np.float32)
            rgbs_list.append(rgbs[0])
            depths_list.append(depth_frame)
            masks_list.append(np.squeeze(insts[0]))

            with _ts(f"io/cam_npz_view_{view}"):
                cam_data = self._load_cam_cached(cam_path)
            T_wc = cam_data["T_wc"][current_abs_tid]
            K = cam_data["K"][current_abs_tid]
            cam_T_wc_list.append(T_wc)
            K_inv_list.append(np.linalg.inv(K))

        # ── RGB images for DINO ───────────────────────────────────────
        with _ts("compute/rgb_resize"):
            if rgbs_list:
                images_for_dino = np.empty(
                    (len(rgbs_list), 3, self.image_size, self.image_size),
                    dtype=np.float32,
                )
                for i, rgb in enumerate(rgbs_list):
                    resized = cv2.resize(
                        rgb,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    images_for_dino[i] = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
                extra_dict["images"] = images_for_dino
            else:
                extra_dict["images"] = np.zeros(
                    (0, 3, self.image_size, self.image_size), dtype=np.float32
                )

        # ── Precompute per-view world XYZ + normalized UV ─────────────
        uid_pcl_per_view = []
        all_visible = set()

        with _ts("compute/view_precompute"):
            for v_idx in range(len(masks_list)):
                H, W = masks_list[v_idx].shape
                # Downsample mask + depth for faster UID grouping
                # Nearest-neighbor preserves integer UID labels
                ds = self.mask_downsample
                mask_ds = masks_list[v_idx][::ds, ::ds]
                depth_ds = depths_list[v_idx][::ds, ::ds]
                H_ds, W_ds = mask_ds.shape
                flat_inst = mask_ds.ravel()
                flat_d = depth_ds.ravel()

                v_visible = set(np.unique(flat_inst).tolist()) - {0}

                uid_data_dict = {}
                if v_visible:
                    uid_order = []
                    uid_idx_chunks = []
                    chunk_boundaries = []
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
                        chunk_boundaries.append((offset, offset + len(idx)))
                        offset += len(idx)

                    # Batch backproject — UV in downsampled coords, map to original
                    all_idx = np.concatenate(uid_idx_chunks)
                    u_ds = (all_idx % W_ds).astype(np.float32)
                    v_ds = (all_idx // W_ds).astype(np.float32)
                    # Map to original resolution for accurate backprojection
                    u = u_ds * ds
                    v = v_ds * ds
                    d_all = flat_d[all_idx]
                    K_inv_f32 = K_inv_list[v_idx].astype(np.float32)
                    T_wc_f32 = cam_T_wc_list[v_idx].astype(np.float32)

                    uv_f32 = np.stack([u, v], axis=-1)
                    cam_xy = uv_f32 @ K_inv_f32[:2, :2].T + K_inv_f32[:2, 2]
                    cam_xyz_all = np.concatenate(
                        [cam_xy * d_all[:, None], d_all[:, None]], axis=-1
                    )
                    world_xyz_all = (
                        cam_xyz_all @ T_wc_f32[:3, :3].T + T_wc_f32[:3, 3]
                    )
                    W_denom = max(W - 1, 1)
                    H_denom = max(H - 1, 1)
                    uv_normalized_all = np.stack(
                        [u / W_denom, v / H_denom], axis=-1
                    )

                    for uid_val, (lo, hi) in zip(uid_order, chunk_boundaries):
                        chunk_xyz = world_xyz_all[lo:hi]
                        chunk_uv = uv_normalized_all[lo:hi]
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

        # ── Precompute per-UID trajectory + per-UID pixel data ────────
        uid_trajectory = {}
        uid_world_pcl = {}
        uid_uv_norm = {}
        uid_view_idx = {}

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
            uid_trajectory[uid] = _T_oracle

            _xyz_parts = []
            _uv_parts = []
            _vi_parts = []
            for v_idx, v_pcl in enumerate(uid_pcl_per_view):
                entry = v_pcl.get(uid)
                if entry is None:
                    continue
                chunk_xyz, chunk_uv = entry
                _xyz_parts.append(chunk_xyz)
                _uv_parts.append(chunk_uv)
                _vi_parts.append(np.full(len(chunk_xyz), v_idx, dtype=np.int64))

            if not _xyz_parts:
                continue

            uid_world_pcl[uid] = np.concatenate(_xyz_parts, axis=0)
            uid_uv_norm[uid] = np.concatenate(_uv_parts, axis=0)
            uid_view_idx[uid] = np.concatenate(_vi_parts, axis=0)

        # ── Build candidate pool and sample pixels ────────────────────
        if self.fixed_P > 0:
            pool_uid = []
            pool_xyz = []
            pool_uv = []
            pool_view = []

            for uid in uid_world_pcl:
                n = len(uid_world_pcl[uid])
                pool_uid.append(np.full(n, uid, dtype=np.int64))
                pool_xyz.append(uid_world_pcl[uid])
                pool_uv.append(uid_uv_norm[uid])
                pool_view.append(uid_view_idx[uid])

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
                                [chosen, np.zeros(self.fixed_pixel_P - len(chosen), dtype=np.int64)]
                            )
                    else:
                        chosen = np.arange(n_total)
                        chosen = np.concatenate(
                            [chosen, np.zeros(self.fixed_pixel_P - n_total, dtype=np.int64)]
                        )

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

        # ══════════════════════════════════════════════════════════════
        # VECTORIZED pixel query construction — pure numpy, no PromptV0
        # ══════════════════════════════════════════════════════════════
        Q = len(selected_uids)
        per_part_uv = []
        per_part_view = []
        per_part_query_mask = []
        viz_pcl_parts = []  # collected viz PCL arrays for extra_dict

        with _ts("compute/pixel_queries"):
            if Q > 0:
                unique_uids_sel = np.unique(selected_uids)
                T_len = uid_trajectory[int(unique_uids_sel[0])].shape[0]  # t+1
                t = T_len - 1  # trace length
                all_warped = np.empty((Q, T_len, 4, 4), dtype=np.float32)

                for uid_val in unique_uids_sel:
                    uid_val = int(uid_val)
                    q_indices = np.flatnonzero(selected_uids == uid_val)
                    n_q = len(q_indices)

                    _T_oracle = uid_trajectory[uid_val].astype(np.float32)

                    # T_wq batch: identity with pixel XYZ positions
                    T_wq_batch = np.broadcast_to(
                        _EYE4_F32, (n_q, 4, 4)
                    ).copy()
                    T_wq_batch[:, :3, 3] = selected_xyz[q_indices]

                    # T_ref = inv(T_oracle[anchor]) @ T_wq
                    T_anchor_inv = np.linalg.inv(
                        _T_oracle[cur_rel_idx]
                    ).astype(np.float32)
                    T_ref_queries = np.einsum(
                        "ij,Qjk->Qik", T_anchor_inv, T_wq_batch
                    )

                    # warped = T_oracle @ T_ref → (T+1, n_q, 4, 4)
                    warped = np.einsum(
                        "tij,Qjk->tQik", _T_oracle, T_ref_queries
                    )
                    all_warped[q_indices] = warped.transpose(1, 0, 2, 3)

                # Split into trace (t steps) and context (1 step)
                pix_trace = all_warped[:, :-1, :, :].astype(np.float32)  # (Q, t, 4, 4)
                pix_context = all_warped[:, -1:, :, :].astype(np.float32)  # (Q, 1, 4, 4)

                # Pad trace to T_max
                pix_trace_mask = np.ones((Q, t), dtype=np.float32)
                pix_trace, pix_trace_mask = _pad_trace_np(pix_trace, pix_trace_mask, T_max)
                pix_context_mask = np.ones((Q, 1), dtype=np.float32)

                # Group IDs: each pixel query gets unique group
                pix_group_ids = np.arange(next_group_id, next_group_id + Q, dtype=np.int64)
                next_group_id += Q

                # Append pixel query arrays
                trace_parts.append(pix_trace)
                trace_mask_parts.append(pix_trace_mask)
                context_parts.append(pix_context)
                context_mask_parts.append(pix_context_mask)
                rot_mask_parts.append(np.ones(Q, dtype=bool))
                group_id_parts.append(pix_group_ids)
                sem_type_parts.append(np.full(Q, SEM_OBJECT, dtype=np.int64))
                uid_parts.append(np.full(Q, UID_BEHAVIOR_OBJECT, dtype=np.int64))
                pcl_parts.append(np.zeros((Q, PCL, 6), dtype=np.float32))
                pcl_mask_parts.append(np.zeros(Q, dtype=np.float32))

                # Pixel UV tracking
                per_part_uv.append(selected_uv.astype(np.float32))
                per_part_view.append(selected_view)
                per_part_query_mask.append(np.ones(Q, dtype=bool))

                # ── Vectorized viz PCL construction ────────────────────
                # Pre-compute per-UID RGB (done once per UID, shared by all queries)
                uid_viz_rgb_cache = {}
                uid_viz_xyz_cache = {}
                for uid_val in unique_uids_sel:
                    uid_val = int(uid_val)
                    _uid_pcl = uid_world_pcl[uid_val]
                    _uid_views = uid_view_idx[uid_val]
                    _uid_uvs = uid_uv_norm[uid_val]
                    n_pts_uid = len(_uid_pcl)

                    per_view_rgb = np.full((n_pts_uid, 3), 0.5, dtype=np.float32)
                    for v_i, rgb_img in enumerate(rgbs_list):
                        _H_img, _W_img = rgb_img.shape[:2]
                        view_mask = _uid_views == v_i
                        if not np.any(view_mask):
                            continue
                        _pix_u = np.clip(
                            (_uid_uvs[view_mask, 0] * (_W_img - 1)).astype(int),
                            0, _W_img - 1,
                        )
                        _pix_v = np.clip(
                            (_uid_uvs[view_mask, 1] * (_H_img - 1)).astype(int),
                            0, _H_img - 1,
                        )
                        per_view_rgb[view_mask] = (
                            rgb_img[_pix_v, _pix_u].astype(np.float32) / 255.0
                        )
                    uid_viz_rgb_cache[uid_val] = per_view_rgb
                    uid_viz_xyz_cache[uid_val] = _uid_pcl

                # Batch viz PCL per UID
                all_viz_pcl = np.zeros((Q, PCL, 6), dtype=np.float32)
                for uid_val in unique_uids_sel:
                    uid_val = int(uid_val)
                    q_indices = np.flatnonzero(selected_uids == uid_val)
                    n_q = len(q_indices)

                    _uid_pcl = uid_viz_xyz_cache[uid_val]
                    _viz_rgb = uid_viz_rgb_cache[uid_val]
                    N = len(_uid_pcl)
                    if N == 0:
                        continue

                    M = PCL
                    if self.random_flag:
                        sample_idx = np.random.randint(0, N, size=(n_q, M))
                    else:
                        if N >= M:
                            step = max(1, N // M)
                            base = np.arange(0, N, step)[:M]
                            if len(base) < M:
                                base = np.concatenate([base, np.zeros(M - len(base), dtype=np.intp)])
                            sample_idx = np.broadcast_to(base[None, :], (n_q, M)).copy()
                        else:
                            repeat_times = (M + N - 1) // N
                            base = np.tile(np.arange(N), repeat_times)[:M]
                            sample_idx = np.broadcast_to(base[None, :], (n_q, M)).copy()

                    flat_idx = sample_idx.ravel()
                    sampled_xyz = _uid_pcl[flat_idx].reshape(n_q, M, 3)
                    sampled_rgb = _viz_rgb[flat_idx].reshape(n_q, M, 3)

                    pixel_xyzs = selected_xyz[q_indices]
                    sampled_xyz = sampled_xyz - pixel_xyzs[:, None, :]

                    all_viz_pcl[q_indices] = np.concatenate(
                        [sampled_xyz, sampled_rgb], axis=2
                    )

                viz_pcl_parts.append(all_viz_pcl)

        # ── Robot links — build numpy arrays directly ─────────────────
        for link_name, uid, uid_act, sem_type in _ROBOT_LINKS:
            if link_name not in kinematic_states:
                continue

            _T_oracle = kinematic_states[link_name][start_tid:end_tid][::factor]
            _T_oracle = np.concatenate(
                [_T_oracle, kinematic_states[link_name][-1][None]], axis=0
            )
            t_link = len(_T_oracle) - 1

            _T_oracle_act_list = []
            for t_idx in range(start_tid, end_tid, factor):
                next_t_idx = min(t_idx + factor, len(kinematic_states[link_name]) - 1)
                _T_oracle_act_list.append(kinematic_states[link_name][next_t_idx])
            _T_oracle_act = np.stack(_T_oracle_act_list, axis=0)
            _T_oracle_act = np.concatenate(
                [_T_oracle_act, _T_oracle_act[-1][None]], axis=0
            )

            pcl_key = f"{link_name}_pcl"
            if pcl_key not in self.robot_link_data:
                continue

            gt_pcl = self.robot_link_data[pcl_key]
            c_key = f"{link_name}_pcl_c"
            gt_c = self.robot_link_data.get(c_key, np.ones_like(gt_pcl))

            _pcl = np.concatenate(
                [gt_pcl, gt_c.astype(np.float32)], axis=1
            )
            _pcl = sample_list(_pcl, PCL, random_flag=self.random_flag)

            gid = next_group_id
            next_group_id += 1

            # Position trace: (1, t, 4, 4)
            _T_wl = _T_oracle[:-1][None].astype(np.float32)
            _T_ctx = _T_oracle[-1:][None].astype(np.float32)
            _mask = np.ones((1, t_link), dtype=np.float32)
            _T_wl_padded, _mask_padded = _pad_trace_np(_T_wl, _mask, T_max)
            _ctx_mask = np.ones((1, 1), dtype=np.float32)

            trace_parts.append(_T_wl_padded)
            trace_mask_parts.append(_mask_padded)
            context_parts.append(_T_ctx)
            context_mask_parts.append(_ctx_mask)
            rot_mask_parts.append(np.ones(1, dtype=bool))
            group_id_parts.append(np.array([gid], dtype=np.int64))
            sem_type_parts.append(np.array([sem_type], dtype=np.int64))
            uid_parts.append(np.array([uid], dtype=np.int64))
            pcl_parts.append(_pcl[None].astype(np.float32))
            pcl_mask_parts.append(np.ones(1, dtype=np.float32))

            # Action trace: (1, t, 4, 4)
            _T_wl_act = _T_oracle_act[:-1][None].astype(np.float32)
            _T_ctx_act = _T_oracle_act[-1:][None].astype(np.float32)
            _T_wl_act_padded, _mask_act_padded = _pad_trace_np(
                _T_wl_act, np.ones((1, t_link), dtype=np.float32), T_max
            )

            trace_parts.append(_T_wl_act_padded)
            trace_mask_parts.append(_mask_act_padded)
            context_parts.append(_T_ctx_act)
            context_mask_parts.append(np.ones((1, 1), dtype=np.float32))
            rot_mask_parts.append(np.ones(1, dtype=bool))
            group_id_parts.append(np.array([gid], dtype=np.int64))
            sem_type_parts.append(np.array([SEM_BEHAVIOR_ACT], dtype=np.int64))
            uid_parts.append(np.array([uid_act], dtype=np.int64))
            pcl_parts.append(_pcl[None].astype(np.float32))
            pcl_mask_parts.append(np.ones(1, dtype=np.float32))

            # UV tracking + viz PCL entries for robot links (not pixel queries)
            per_part_uv.append(np.zeros((2, 2), dtype=np.float32))
            per_part_view.append(np.zeros(2, dtype=np.int64))
            per_part_query_mask.append(np.zeros(2, dtype=bool))
            viz_pcl_parts.append(np.zeros((2, PCL, 6), dtype=np.float32))

        # ── Store pixel UV data in extra_dict ─────────────────────────
        if per_part_uv:
            extra_dict["pixel_uv_normalized"] = np.concatenate(per_part_uv, axis=0)
            extra_dict["pixel_view_idx"] = np.concatenate(per_part_view, axis=0)
            extra_dict["pixel_query_mask"] = np.concatenate(per_part_query_mask, axis=0)
        else:
            extra_dict["pixel_uv_normalized"] = np.zeros((0, 2), dtype=np.float32)
            extra_dict["pixel_view_idx"] = np.zeros(0, dtype=np.int64)
            extra_dict["pixel_query_mask"] = np.zeros(0, dtype=bool)

        extra_dict["_center_xy"] = _center_xy

        if viz_pcl_parts:
            extra_dict["viz_obj_pcl"] = np.concatenate(viz_pcl_parts, axis=0)
        else:
            extra_dict["viz_obj_pcl"] = np.zeros((0, PCL, 6), dtype=np.float32)

        if "zed_link" in kinematic_states and current_abs_tid < len(
            kinematic_states["zed_link"]
        ):
            extra_dict["_cam_T_wc"] = kinematic_states["zed_link"][current_abs_tid]

        # ── Concatenate all parts into final arrays ──────────────────
        if trace_parts:
            arrays = {
                "trace": np.concatenate(trace_parts, axis=0),
                "trace_mask": np.concatenate(trace_mask_parts, axis=0),
                "context": np.concatenate(context_parts, axis=0),
                "context_mask": np.concatenate(context_mask_parts, axis=0),
                "rot_mask": np.concatenate(rot_mask_parts, axis=0),
                "group_id": np.concatenate(group_id_parts, axis=0),
                "sem_type": np.concatenate(sem_type_parts, axis=0),
                "uid": np.concatenate(uid_parts, axis=0),
                "pcl": np.concatenate(pcl_parts, axis=0),
                "pcl_mask": np.concatenate(pcl_mask_parts, axis=0),
            }
        else:
            # Edge case: no visible objects and no robot links
            arrays = {
                "trace": np.broadcast_to(_EYE4_F32, (0, T_max, 4, 4)).copy(),
                "trace_mask": np.zeros((0, T_max), dtype=np.float32),
                "context": np.broadcast_to(_EYE4_F32, (0, 1, 4, 4)).copy(),
                "context_mask": np.zeros((0, 1), dtype=np.float32),
                "rot_mask": np.zeros(0, dtype=bool),
                "group_id": np.zeros(0, dtype=np.int64),
                "sem_type": np.zeros(0, dtype=np.int64),
                "uid": np.zeros(0, dtype=np.int64),
                "pcl": np.zeros((0, PCL, 6), dtype=np.float32),
                "pcl_mask": np.zeros(0, dtype=np.float32),
            }

        return arrays, extra_dict

    def __getitem_inner__(self, idx):
        """Boosted __getitem_inner__ — pure numpy pipeline with single torch conversion.

        Bypasses PromptV0 entirely. All data is built as numpy arrays by
        __add_behavior_prompt__ and converted to torch tensors exactly once here.
        """
        idx = idx % len(self)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )

        _ts = self._profiler.time_section if self._profiler else _null_ts

        arrays, extra_dict = self.__add_behavior_prompt__(
            current_ith, start_abs_tid, end_abs_tid, sid
        )

        # ── Apply center shift + yaw rotation in numpy ────────────────
        with _ts("compute/transforms"):
            _center_xy = extra_dict.pop("_center_xy")
            _cam_T_wc = extra_dict.pop("_cam_T_wc", None)

            trace_np = arrays["trace"]    # (P, T, 4, 4)
            context_np = arrays["context"]  # (P, C, 4, 4)

            # Shift XY to robot center
            trace_np[:, :, :2, 3] -= _center_xy[None, None, :]
            context_np[:, :, :2, 3] -= _center_xy[None, None, :]

            # Rotate around Z for camera alignment — only rows 0,1 change
            if _cam_T_wc is not None:
                cam_z_xy = _cam_T_wc[:2, 2]
                yaw = float(np.arctan2(cam_z_xy[1], cam_z_xy[0]))
                c = np.float32(np.cos(yaw))
                s = np.float32(np.sin(yaw))
                # R_z @ M: row0_new = c*row0 + s*row1, row1_new = -s*row0 + c*row1
                for arr in (trace_np, context_np):
                    r0 = arr[:, :, 0, :].copy()
                    r1 = arr[:, :, 1, :].copy()
                    arr[:, :, 0, :] = c * r0 + s * r1
                    arr[:, :, 1, :] = -s * r0 + c * r1

        # ── Single torch conversion ───────────────────────────────────
        with _ts("compute/torch_convert"):
            trace = torch.from_numpy(trace_np).float()
            trace_mask = torch.from_numpy(arrays["trace_mask"]).float()
            context = torch.from_numpy(context_np).float()
            context_mask = torch.from_numpy(arrays["context_mask"]).float()
            rot_mask = torch.from_numpy(arrays["rot_mask"]).bool()
            group_id = torch.from_numpy(arrays["group_id"]).long()
            sem_type = torch.from_numpy(arrays["sem_type"]).long()
            uid = torch.from_numpy(arrays["uid"]).long()
            pcl = torch.from_numpy(arrays["pcl"]).float()
            pcl_mask = torch.from_numpy(arrays["pcl_mask"]).bool()

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        with _ts("io/text"):
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


def _pad_trace_np(trace_np, mask_np, T_max):
    """Pad trace (n, t, 4, 4) and mask (n, t) to T_max in numpy."""
    t = trace_np.shape[1]
    if t >= T_max:
        return trace_np, mask_np
    n = trace_np.shape[0]
    eye_pad = np.broadcast_to(_EYE4_F32, (n, T_max - t, 4, 4))
    trace_padded = np.concatenate([trace_np, eye_pad], axis=1)
    mask_padded = np.concatenate(
        [mask_np, np.zeros((n, T_max - t), dtype=np.float32)], axis=1
    )
    return trace_padded, mask_padded
