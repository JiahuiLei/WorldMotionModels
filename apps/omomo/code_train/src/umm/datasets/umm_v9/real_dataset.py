"""Real-world point-tracking dataset for the UMM framework.

Loads 3D point tracks with local-frame rotations from real-world video
sequences (DAVIS, Bridge, etc.).  Each sample produces a
:class:`~umm.datasets.umm_v9.dataset_template.CanonicalDatasetOutput`
containing SE(3) traces for scene pixel queries and an anchor-frame DINO image.

All P parts are scene pixel queries derived from 3D point tracks with
estimated local rotations. No robot parts.

Data layout::

    {data_root}/{seq_id}/
        cameras.npz        K (T,3,3), T_wc (T,4,4), gravity (3,)
        depths.zip         not used (tracks provide 3D positions directly)
        images.zip         JPEG frames  000000.jpg, 000001.jpg, ...
        instances.zip      COCO-RLE JSON per frame (for UID lookup)
        meta.json          sequence metadata
        tracks.npz         tracks (N, T, 6) = [u, v, x, y, z, vis]

    {localframes_root}/{seq_id}.npz
        rotations (N, T, 3)   axis-angle local rotations
        rot_valid (N, T)      bool validity mask for rotations

Example::

    ds = RealDataset(
        data_root="/data/davis_v1.2",
        localframes_root="/data/davis_v1.2_localframes",
        split_fn="data_splits/davis/v1_train.json",
        max_T=30,
        fixed_P=512,
    )
    sample, extra = ds[0]
"""

import logging
import os.path as osp
import zipfile
import json
from typing import Optional, Dict

import cv2
import numpy as np
import torch

from lib_io import UnifiedIO
from umm.datasets.umm_v9.base_dataset import BaseDataset, CanonicalDatasetOutput
from umm.datasets.umm_v9.dataset_config import (
    UID_REAL_SCENE,
    UID_REAL_CAMERA,
    SEM_REAL_SCENE,
    SEM_REAL_CAMERA,
)

logger = logging.getLogger(__name__)

# Pre-allocated identity matrix (avoid repeated allocation)
_EYE4_F32 = np.eye(4, dtype=np.float32)


def _axis_angle_to_rotation_matrix_batch(axis_angles: np.ndarray) -> np.ndarray:
    """Convert batch of axis-angle vectors to rotation matrices.

    Uses vectorized Rodrigues formula for efficiency.

    Args:
        axis_angles: (N, 3) axis-angle rotation vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = axis_angles.shape[0]
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)  # (N, 1)
    # Handle zero-angle case: use identity
    small = angles.squeeze(-1) < 1e-8  # (N,)
    safe_angles = np.where(angles < 1e-8, 1.0, angles)  # avoid div by zero
    axes = axis_angles / safe_angles  # (N, 3) unit axes

    # Rodrigues: R = I + sin(θ)K + (1 - cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    sin_a = np.sin(angles)  # (N, 1)
    cos_a = np.cos(angles)  # (N, 1)

    # Skew-symmetric components
    kx = axes[:, 0]  # (N,)
    ky = axes[:, 1]  # (N,)
    kz = axes[:, 2]  # (N,)

    # K matrix elements (skew-symmetric cross-product matrix)
    # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
    # K² = [[-(ky²+kz²), kx*ky, kx*kz],
    #        [kx*ky, -(kx²+kz²), ky*kz],
    #        [kx*kz, ky*kz, -(kx²+ky²)]]

    R = np.zeros((N, 3, 3), dtype=np.float32)

    s = sin_a.squeeze(-1)   # (N,)
    c = cos_a.squeeze(-1)   # (N,)
    t = 1.0 - c             # (N,)

    R[:, 0, 0] = t * kx * kx + c
    R[:, 0, 1] = t * kx * ky - s * kz
    R[:, 0, 2] = t * kx * kz + s * ky
    R[:, 1, 0] = t * kx * ky + s * kz
    R[:, 1, 1] = t * ky * ky + c
    R[:, 1, 2] = t * ky * kz - s * kx
    R[:, 2, 0] = t * kx * kz - s * ky
    R[:, 2, 1] = t * ky * kz + s * kx
    R[:, 2, 2] = t * kz * kz + c

    # For zero-angle entries, set to identity
    R[small] = np.eye(3, dtype=np.float32)

    return R


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


class RealDataset(BaseDataset):
    """Real-world point-tracking dataset with SE(3) traces from 3D tracks.

    All P parts are scene pixel queries (no robot). Tracks provide 3D XYZ
    positions, local-frame rotations provide SO(3), combined into SE(3).

    Args:
        data_root: Root directory containing per-sequence data folders.
        localframes_root: Directory containing per-sequence localframe NPZ files.
        split_fn: Path to JSON split file (list of [seq_id, frame_count]).
        base_fps: Native frame rate of the dataset.
        random_flag: Enable random sampling and augmentation.
        safe_flag: If True, silently skip to next sample on errors.
        repeat: Number of times to repeat the index list.
        t_rate: Temporal stride for index expansion.
        max_T: Maximum number of time steps per trace window.
        fixed_P: Number of scene pixel queries (all P are pixel queries).
        image_size: Size for DINO image (resized to image_size x image_size).
        max_range: Maximum XY distance from center for spatial filtering.
        add_camera_flag: If True, add camera trajectory as an extra part.
        ctx_mode: Context frame selection. "current" uses the anchor frame
            (default, DAVIS behavior). "last" uses the last frame of the
            entire sequence (Bridge behavior — goal-conditioned).
        scene_align: Scene alignment mode. "yaw" (default) centers XY on
            visible tracks and aligns camera forward to +X via Z-axis
            rotation. "camera" transforms everything into the anchor
            camera frame (full rigid transform via T_cw = inv(T_wc)).
        npz_cache_size: Max number of cached NPZ files per worker.
    """

    def __init__(
        self,
        data_root: str,
        localframes_root: str,
        split_fn: str,
        base_fps: float = 30.0,
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        t_rate: int = 1,
        max_T: int = 30,
        fixed_P: int = 512,
        image_size = (288, 160),  # (W, H) — both must be divisible by DINO patch_size (16)
        max_range: float = 50.0,
        add_camera_flag: bool = False,
        reference_depth_median: float = -1.0,
        ctx_mode: str = "current",
        scene_align: str = "yaw",
        npz_cache_size: int = 128,
    ):
        super().__init__(
            data_root, split_fn, base_fps,
            language_fn=None,
            random_flag=random_flag,
            safe_flag=safe_flag,
            repeat=repeat,
            t_rate=t_rate,
        )
        self.localframes_root = localframes_root
        assert ctx_mode in ("current", "last"), f"ctx_mode must be 'current' or 'last', got '{ctx_mode}'"
        self.ctx_mode = ctx_mode
        assert scene_align in ("yaw", "camera"), f"scene_align must be 'yaw' or 'camera', got '{scene_align}'"
        self.scene_align = scene_align
        self.max_T = max_T
        self.max_C = 1
        self.fixed_P = fixed_P
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)  # (W, H)
        self.max_range = max_range
        self.add_camera_flag = add_camera_flag
        self.reference_depth_median = reference_depth_median

        # LRU caches for NPZ files (shared across samples from same seq)
        self._npz_cache_size = npz_cache_size
        self._tracks_cache: Dict[str, dict] = {}
        self._localframes_cache: Dict[str, dict] = {}
        self._cameras_cache: Dict[str, dict] = {}

        self.io = UnifiedIO()

        # Load language annotations from meta.json per sequence
        self.seq_text: Dict[str, str] = {}
        for sid in self.seq_len:
            meta_fn = osp.join(data_root, sid, "meta.json")
            try:
                with open(meta_fn, "r") as f:
                    meta = json.load(f)
                # Prefer "language" field, fall back to "task_name", then sid
                text = meta.get("language") or meta.get("task_name") or sid
                self.seq_text[sid] = text
            except (FileNotFoundError, json.JSONDecodeError):
                self.seq_text[sid] = sid

        logger.info(
            f"RealDataset: {len(self.indices)} indices, "
            f"fixed_P={fixed_P}, max_T={max_T}, image_size={image_size}"
        )

    # ── Cache helpers ─────────────────────────────────────────────────

    def _load_tracks_cached(self, path: str) -> dict:
        """Load tracks NPZ with per-worker LRU cache."""
        if path in self._tracks_cache:
            return self._tracks_cache[path]
        data = dict(np.load(path, allow_pickle=False))
        if len(self._tracks_cache) >= self._npz_cache_size:
            oldest = next(iter(self._tracks_cache))
            del self._tracks_cache[oldest]
        self._tracks_cache[path] = data
        return data

    def _load_localframes_cached(self, path: str) -> dict:
        """Load localframes NPZ with per-worker LRU cache."""
        if path in self._localframes_cache:
            return self._localframes_cache[path]
        data = dict(np.load(path, allow_pickle=False))
        if len(self._localframes_cache) >= self._npz_cache_size:
            oldest = next(iter(self._localframes_cache))
            del self._localframes_cache[oldest]
        self._localframes_cache[path] = data
        return data

    def _load_cameras_cached(self, path: str) -> dict:
        """Load cameras NPZ with per-worker LRU cache."""
        if path in self._cameras_cache:
            return self._cameras_cache[path]
        data = dict(np.load(path, allow_pickle=False))
        if len(self._cameras_cache) >= self._npz_cache_size:
            oldest = next(iter(self._cameras_cache))
            del self._cameras_cache[oldest]
        self._cameras_cache[path] = data
        return data

    # ── Trace sampling ────────────────────────────────────────────────

    def __decide_trace_sampling__(
        self, seq_id: str, current_abs_tid: int, dataset_idx: int,
        random_enabled: bool = True,
    ):
        """Decide temporal window and relative anchor index.

        Same logic as BehaviorDataset.__decide_trace_sampling__: randomly
        (or deterministically) places a window of length max_T * factor
        around current_abs_tid, clamped to sequence boundaries.

        Returns:
            Tuple (current_ith, start_abs_tid, end_abs_tid)
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

    # ── UID lookup from instance segmentation ─────────────────────────

    def _lookup_uids_at_anchor(
        self,
        seq_id: str,
        anchor_frame: int,
        pixel_uv: np.ndarray,
    ) -> np.ndarray:
        """Look up instance segmentation UIDs at pixel locations.

        Args:
            seq_id: Sequence identifier.
            anchor_frame: Absolute frame index for the anchor.
            pixel_uv: (P, 2) pixel coordinates [u, v] in original image space.

        Returns:
            (P,) int64 array of UIDs (0 for background / not covered).
        """
        inst_path = osp.join(self.data_root, seq_id, "instances.zip")
        P = pixel_uv.shape[0]
        uids = np.zeros(P, dtype=np.int64)

        if not osp.exists(inst_path):
            return uids

        try:
            uid_frames, _ = self.io.load_inst_segs(
                inst_path, 1, [anchor_frame], target_uids=None
            )
            if uid_frames.size == 0:
                return uids
            seg_map = uid_frames[0]  # (H, W) int64
            H, W = seg_map.shape

            u = np.clip(pixel_uv[:, 0].astype(np.int32), 0, W - 1)
            v = np.clip(pixel_uv[:, 1].astype(np.int32), 0, H - 1)
            uids = seg_map[v, u]
        except Exception as e:
            logger.debug(f"Failed to load instance seg for {seq_id}: {e}")

        return uids.astype(np.int64)

    # ── Main getitem ──────────────────────────────────────────────────

    def __getitem_inner__(self, idx):
        """Build one sample: SE(3) traces from 3D tracks + local rotations.

        Returns:
            Tuple of (CanonicalDatasetOutput, extra_dict).
        """
        idx = idx % len(self)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, current_abs_tid, idx, random_enabled=self.random_flag
        )

        factor = int(self.t_rate)
        T_max = self.max_T

        # Frame indices for the trace window
        frame_indices = list(range(start_abs_tid, end_abs_tid, factor))
        t_len = len(frame_indices)  # actual time steps in window
        anchor_frame = start_abs_tid + current_ith * factor

        # ── Load tracks and localframes ───────────────────────────────
        tracks_path = osp.join(self.data_root, sid, "tracks.npz")
        localframes_path = osp.join(self.localframes_root, sid + ".npz")
        cameras_path = osp.join(self.data_root, sid, "cameras.npz")

        tracks_data = self._load_tracks_cached(tracks_path)
        tracks_all = tracks_data["tracks"]  # (N, T_total, 6) = [u, v, x, y, z, vis]
        N_total, T_total, _ = tracks_all.shape

        lf_data = self._load_localframes_cached(localframes_path)
        rotations_all = lf_data["rotations"]  # (N, T_total, 3) axis-angle
        rot_valid_all = lf_data["rot_valid"]   # (N, T_total) bool

        cam_data = self._load_cameras_cached(cameras_path)
        K_all = cam_data["K"]       # (T_total, 3, 3) float64
        T_wc_all = cam_data["T_wc"]  # (T_total, 4, 4) float64

        # ── Sample P tracks visible at current time ────────────────────
        P = self.fixed_P

        # Only sample from tracks visible & rotation-valid at current_abs_tid
        vis_at_cur = tracks_all[:, current_abs_tid, 5]       # (N,)
        rv_at_cur = rot_valid_all[:, current_abs_tid]         # (N,)
        valid_pool = np.where((vis_at_cur > 0.5) & rv_at_cur)[0]

        # Fallback: if fewer valid tracks than needed, use all tracks
        pool = valid_pool if len(valid_pool) > 0 else np.arange(N_total)

        if self.random_flag:
            track_indices = np.random.choice(
                pool, P, replace=(len(pool) < P)
            )
        else:
            if len(pool) >= P:
                step = max(1, len(pool) // P)
                track_indices = pool[np.arange(0, len(pool), step)[:P]]
                if len(track_indices) < P:
                    track_indices = np.concatenate([
                        track_indices,
                        np.full(P - len(track_indices), pool[0], dtype=np.int64),
                    ])
            else:
                repeat_times = (P + len(pool) - 1) // len(pool)
                track_indices = np.tile(pool, repeat_times)[:P]

        # ── Extract temporal window for selected tracks ───────────────
        fi = np.array(frame_indices, dtype=np.int64)  # (t_len,)

        # tracks_sel: (P, t_len, 6)
        tracks_sel = tracks_all[track_indices][:, fi, :]
        positions = tracks_sel[:, :, 2:5].astype(np.float32)  # (P, t_len, 3) xyz
        vis = tracks_sel[:, :, 5]  # (P, t_len) visibility
        uv_raw = tracks_sel[:, :, 0:2]  # (P, t_len, 2) pixel coords

        # rotations_sel: (P, t_len, 3) axis-angle
        rotations_sel = rotations_all[track_indices][:, fi, :]
        rot_valid_sel = rot_valid_all[track_indices][:, fi]  # (P, t_len) bool

        # ── Build SE(3) from axis-angle + XYZ ─────────────────────────
        # Reshape for batch conversion: (P*t_len, 3)
        rot_flat = rotations_sel.reshape(-1, 3).astype(np.float32)
        R_flat = _axis_angle_to_rotation_matrix_batch(rot_flat)  # (P*t_len, 3, 3)
        R_all = R_flat.reshape(P, t_len, 3, 3)  # (P, t_len, 3, 3)

        # Build 4x4 SE(3) matrices
        se3 = np.broadcast_to(_EYE4_F32, (P, t_len, 4, 4)).copy()
        se3[:, :, :3, :3] = R_all
        se3[:, :, :3, 3] = positions

        # ── Build trace_mask = (vis > 0.5) & rot_valid ────────────────
        trace_mask = ((vis > 0.5) & rot_valid_sel).astype(np.float32)  # (P, t_len)

        # Replace invalid entries with identity to prevent NaN propagation
        # through attention (invalid parts still carry raw track data that
        # may contain NaN/Inf; model processes all P parts)
        invalid = trace_mask < 0.5  # (P, t_len)
        se3[invalid] = _EYE4_F32

        # ── Context frame selection ────────────────────────────────
        # "current": use anchor frame (DAVIS behavior)
        # "last": use last frame of entire sequence (Bridge — goal state)
        if self.ctx_mode == "last":
            ctx_frame = self.seq_len[sid] - 1
        else:
            ctx_frame = current_abs_tid
        ctx_positions = tracks_all[track_indices, ctx_frame, 2:5].astype(np.float32)  # (P, 3)
        ctx_rotations = rotations_all[track_indices, ctx_frame, :].astype(np.float32)  # (P, 3)
        ctx_rot_valid = rot_valid_all[track_indices, ctx_frame]  # (P,) bool
        ctx_vis = tracks_all[track_indices, ctx_frame, 5]  # (P,) vis

        ctx_R = _axis_angle_to_rotation_matrix_batch(ctx_rotations)  # (P, 3, 3)
        context = np.broadcast_to(_EYE4_F32, (P, 1, 4, 4)).copy()
        context[:, 0, :3, :3] = ctx_R
        context[:, 0, :3, 3] = ctx_positions
        context_mask = ((ctx_vis > 0.5) & ctx_rot_valid).astype(np.float32)[:, None]  # (P, 1)
        # Replace invalid context entries with identity
        ctx_invalid = context_mask[:, 0] < 0.5  # (P,)
        context[ctx_invalid, 0] = _EYE4_F32

        # ── Scene alignment: compute T_align (4x4) ────────────────────
        # Both modes produce a single rigid transform T_align such that:
        #   se3_aligned = T_align @ se3_world
        # Viz undoes this with inv(T_align).
        cam_T_wc = T_wc_all[anchor_frame].astype(np.float32)  # (4, 4)

        if self.scene_align == "camera":
            # Camera mode: R_reorient @ inv(T_wc)
            # Reorients so Z=up, X=-forward, Y=right (scene lies horizontal).
            #   X_new = -Z_cam, Y_new = X_cam, Z_new = -Y_cam
            R_reorient = np.array([
                [0, 0, -1, 0],
                [1, 0,  0, 0],
                [0, -1, 0, 0],
                [0, 0,  0, 1],
            ], dtype=np.float32)
            T_align = (R_reorient @ np.linalg.inv(cam_T_wc)).astype(np.float32)
        else:
            # Yaw mode: T_align = R_z(-yaw) @ Translate(-center_xy, 0)
            anchor_positions = positions[:, current_ith, :]  # (P, 3)
            anchor_vis = vis[:, current_ith]  # (P,)
            visible_at_anchor = anchor_vis > 0.5
            if visible_at_anchor.any():
                center_xyz = anchor_positions[visible_at_anchor].mean(axis=0)  # (3,)
            else:
                center_xyz = anchor_positions.mean(axis=0)  # (3,)
            center_xy = center_xyz[:2].astype(np.float32)  # (2,)

            cam_z_xy = cam_T_wc[:2, 2]  # forward direction projected to XY
            yaw = float(np.arctan2(cam_z_xy[1], cam_z_xy[0]))
            c = np.float32(np.cos(yaw))
            s = np.float32(np.sin(yaw))

            T_align = np.eye(4, dtype=np.float32)
            T_align[0, 0], T_align[0, 1] = c, s
            T_align[1, 0], T_align[1, 1] = -s, c
            T_align[0, 3] = -c * center_xy[0] - s * center_xy[1]
            T_align[1, 3] = s * center_xy[0] - c * center_xy[1]

        # Apply unified transform
        se3 = np.einsum('ij,ptjk->ptik', T_align, se3)
        context = np.einsum('ij,pcjk->pcik', T_align, context)

        # ── Per-scene depth rescaling ────────────────────────────────
        # Scale all 3D positions so median depth at anchor = reference_depth_median
        scene_scale = 1.0
        if self.reference_depth_median > 0:
            # Depth = distance from camera origin in transformed frame
            # After center_shift + yaw, anchor camera is at the origin looking forward
            # Use anchor positions (already center-shifted + yaw-rotated)
            anchor_pos_t = se3[:, current_ith, :3, 3]  # (P, 3) transformed
            anchor_valid = trace_mask[:, current_ith] > 0.5  # (P,)
            if anchor_valid.any():
                depths = np.linalg.norm(anchor_pos_t[anchor_valid], axis=-1)  # (N,)
                med = float(np.median(depths))
                if med > 1e-6:
                    scene_scale = self.reference_depth_median / med
                    # Scale translation columns of SE3 matrices
                    se3[:, :, :3, 3] *= scene_scale
                    context[:, :, :3, 3] *= scene_scale

        # ── Pad trace to max_T ────────────────────────────────────────
        trace_padded, trace_mask_padded = _pad_trace_np(se3, trace_mask, T_max)

        # ── Group IDs, sem types, UIDs ────────────────────────────────
        # Group IDs must be contiguous 0..N-1 among foreground (valid) parts.
        # Parts with no valid trace frames (trace_mask all zeros) get gid=0
        # (they won't be counted as foreground by the collate function).
        fg_mask_per_part = trace_mask.max(axis=1) > 0  # (P,) which parts have valid frames
        group_ids = np.zeros(P, dtype=np.int64)
        group_ids[fg_mask_per_part] = np.arange(fg_mask_per_part.sum(), dtype=np.int64)
        sem_types = np.full(P, SEM_REAL_SCENE, dtype=np.int64)
        uid_arr = np.full(P, UID_REAL_SCENE, dtype=np.int64)
        rot_mask = np.ones(P, dtype=bool)

        # ── Local PCL: zeros (DINO features used instead via pixel UV) ──
        pcl = np.zeros((P, 0, 6), dtype=np.float32)
        pcl_mask = np.zeros(P, dtype=np.float32)

        # ── Load anchor frame image for DINO ──────────────────────────
        target_W, target_H = int(self.image_size[0]), int(self.image_size[1])
        images_path = osp.join(self.data_root, sid, "images.zip")
        rgbs = self.io.load_rgbs(images_path, [anchor_frame])
        if rgbs:
            rgb = rgbs[0]  # (H, W, 3) uint8
            img_H, img_W = rgb.shape[:2]  # original image dims
            resized = cv2.resize(
                rgb, (target_W, target_H),
                interpolation=cv2.INTER_AREA,
            )
            image_dino = resized.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, target_H, target_W)
            images_for_dino = image_dino[None, :, :, :]  # (1, 3, target_H, target_W)
            scale_x = target_W / img_W  # resize scale factor x
            scale_y = target_H / img_H  # resize scale factor y
        else:
            images_for_dino = np.zeros(
                (1, 3, target_H, target_W), dtype=np.float32
            )
            img_H, img_W = target_H, target_W
            scale_x, scale_y = 1.0, 1.0

        # ── Pixel UV normalized + view idx ────────────────────────────
        # UV at anchor frame in original pixel space → scale to resized pixel space → normalize
        anchor_uv = uv_raw[:, current_ith, :].astype(np.float32)  # (P, 2) [u, v] original
        anchor_uv_resized = anchor_uv.copy()
        anchor_uv_resized[:, 0] *= scale_x  # u in resized pixel space
        anchor_uv_resized[:, 1] *= scale_y  # v in resized pixel space
        pixel_uv_normalized = np.stack([
            anchor_uv_resized[:, 0] / max(target_W - 1, 1),
            anchor_uv_resized[:, 1] / max(target_H - 1, 1),
        ], axis=-1).astype(np.float32)  # (P, 2) normalized [0, 1]

        pixel_view_idx = np.zeros(P, dtype=np.int64)  # single view, all zeros
        pixel_query_mask = np.ones(P, dtype=bool)  # all are pixel queries

        # ── UID lookup from instance seg ──────────────────────────────
        # Look up instance seg UIDs at anchor frame pixel locations
        inst_uids = self._lookup_uids_at_anchor(sid, anchor_frame, anchor_uv)
        # Map instance UIDs to visualization: use UID_REAL_SCENE for all
        # (instance IDs are large arbitrary integers, not in our UID registry)
        # Keep unique instance IDs in extra dict for viz coloring
        # But for trace_uid, use UID_REAL_SCENE uniformly
        # Store per-pixel instance IDs separately for viz if needed

        # ── Add camera trajectory if requested ────────────────────────
        next_group_id = int(fg_mask_per_part.sum())
        if self.add_camera_flag:
            # Camera trajectory: T_wc at each frame in the window + context
            cam_trace = np.stack(
                [T_wc_all[fi_idx].astype(np.float32) for fi_idx in frame_indices],
                axis=0,
            )  # (t_len, 4, 4)
            cam_ctx = T_wc_all[ctx_frame].astype(np.float32)[None, :, :]  # (1, 4, 4)

            # Apply unified T_align to camera trace
            cam_trace = np.einsum('ij,tjk->tik', T_align, cam_trace)
            cam_ctx = np.einsum('ij,cjk->cik', T_align, cam_ctx)

            # Apply per-scene depth rescaling to camera translations
            if scene_scale != 1.0:
                cam_trace[:, :3, 3] *= scene_scale
                cam_ctx[:, :3, 3] *= scene_scale

            cam_trace_4d = cam_trace[None, :, :, :]  # (1, t_len, 4, 4)
            cam_mask = np.ones((1, t_len), dtype=np.float32)
            cam_trace_padded, cam_mask_padded = _pad_trace_np(
                cam_trace_4d, cam_mask, T_max
            )
            cam_ctx_4d = cam_ctx[None, :, :, :]  # (1, 1, 4, 4)
            cam_ctx_mask = np.ones((1, 1), dtype=np.float32)

            # Concatenate camera to main arrays
            trace_padded = np.concatenate([trace_padded, cam_trace_padded], axis=0)
            trace_mask_padded = np.concatenate(
                [trace_mask_padded, cam_mask_padded], axis=0
            )
            context = np.concatenate([context, cam_ctx_4d], axis=0)
            context_mask = np.concatenate([context_mask, cam_ctx_mask], axis=0)
            group_ids = np.concatenate(
                [group_ids, np.array([next_group_id], dtype=np.int64)]
            )
            sem_types = np.concatenate(
                [sem_types, np.array([SEM_REAL_CAMERA], dtype=np.int64)]
            )
            uid_arr = np.concatenate(
                [uid_arr, np.array([UID_REAL_CAMERA], dtype=np.int64)]
            )
            rot_mask = np.concatenate([rot_mask, np.array([True])])
            pcl = np.concatenate(
                [pcl, np.zeros((1, 0, 6), dtype=np.float32)], axis=0
            )
            pcl_mask = np.concatenate(
                [pcl_mask, np.zeros(1, dtype=np.float32)]
            )
            # Extend pixel tracking arrays for the camera part (not a pixel query)
            pixel_uv_normalized = np.concatenate(
                [pixel_uv_normalized, np.zeros((1, 2), dtype=np.float32)], axis=0
            )
            pixel_view_idx = np.concatenate(
                [pixel_view_idx, np.zeros(1, dtype=np.int64)]
            )
            pixel_query_mask = np.concatenate(
                [pixel_query_mask, np.array([False])]
            )

        # ── Single torch conversion ───────────────────────────────────
        trace_t = torch.from_numpy(trace_padded).float()
        trace_mask_t = torch.from_numpy(trace_mask_padded).float()
        context_t = torch.from_numpy(context).float()
        context_mask_t = torch.from_numpy(context_mask).float()
        rot_mask_t = torch.from_numpy(rot_mask).bool()
        group_id_t = torch.from_numpy(group_ids).long()
        sem_type_t = torch.from_numpy(sem_types).long()
        uid_t = torch.from_numpy(uid_arr).long()

        # PCL: need at least 1 point for collate compatibility
        if pcl.shape[1] == 0:
            # Create minimal dummy PCL with 1 point
            total_P = trace_padded.shape[0]
            pcl_t = torch.zeros((total_P, 1, 6), dtype=torch.float32)
            pcl_mask_t = torch.zeros(total_P, dtype=torch.bool)
        else:
            pcl_t = torch.from_numpy(pcl).float()
            pcl_mask_t = torch.from_numpy(pcl_mask).bool()

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(T_max) - current_ith) * time_unit

        # Text: language annotation from meta.json (or fallback)
        text = self.seq_text.get(sid, sid)

        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace_t,
            trace_mask=trace_mask_t,
            context=context_t,
            context_mask=context_mask_t,
            trace_group_id=group_id_t,
            trace_rot_mask=rot_mask_t,
            trace_sem_type=sem_type_t,
            trace_uid=uid_t,
            local_pcl=pcl_t,
            local_pcl_mask=pcl_mask_t,
            time=time_stamp,
            text=text,
        )

        # ── Scale anchor K intrinsics for the resize ──────────────────
        K_anchor = K_all[anchor_frame].astype(np.float32).copy()  # (3, 3) original
        K_anchor[0, 0] *= scale_x  # fx
        K_anchor[0, 2] *= scale_x  # cx
        K_anchor[1, 1] *= scale_y  # fy
        K_anchor[1, 2] *= scale_y  # cy

        extra_dict = {
            "images": images_for_dino,  # (1, 3, target_H, target_W) float32 [0,1]
            "pixel_uv_normalized": pixel_uv_normalized,  # (P, 2) float32 [0,1]
            "pixel_view_idx": pixel_view_idx,  # (P,) int64
            "pixel_query_mask": pixel_query_mask,  # (P,) bool
            "viz_obj_pcl": np.zeros((trace_padded.shape[0], 1, 6), dtype=np.float32),
            "rgbs_paths": [images_path],
            "rgbs_views": ["head"],
            "rgbs_indices": frame_indices,
            "_T_align": T_align,  # (4, 4) unified scene alignment transform
            "_cam_T_wc": cam_T_wc,  # (4, 4) original anchor cam pose (world-from-camera)
            "_inst_uids": inst_uids,  # (P,) per-pixel instance UIDs for viz
            # ── Image transform info for viz ──
            "_K_resized": K_anchor,  # (3, 3) scaled intrinsics for resized image
            "_img_resize_wh": np.array([target_W, target_H], dtype=np.int32),
            "_img_orig_wh": np.array([img_W, img_H], dtype=np.int32),
            "_scale_xy": np.array([scale_x, scale_y], dtype=np.float32),
            "_cam_K_all": K_all.astype(np.float32),  # (T_total, 3, 3) original K per frame
            "_cam_T_wc_all": T_wc_all.astype(np.float32),  # (T_total, 4, 4) camera poses
            "_frame_indices": np.array(frame_indices, dtype=np.int64),  # which frames in window
            "_anchor_frame": anchor_frame,  # anchor frame abs index
            "_anchor_ith": current_ith,  # anchor index within the window (0-based)
            "_data_root": self.data_root,  # for on-demand depth loading in viz
            "_seq_id": sid,  # sequence ID for depth loading
            "_scene_scale": np.float32(scene_scale),  # per-scene depth rescaling factor
        }

        return ret, extra_dict
