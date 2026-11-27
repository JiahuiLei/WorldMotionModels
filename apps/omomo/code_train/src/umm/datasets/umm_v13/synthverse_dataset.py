"""SynthVerse synthetic point-tracking dataset for the UMM framework.

Loads dense 3D point tracks with local-frame rotations from the SynthVerse
synthetic dataset.  Each sample produces a
:class:`~umm.datasets.umm_v13.dataset_template.CanonicalDatasetOutput`
containing SE(3) traces for scene pixel queries and an anchor-frame RGB image.

SynthVerse data layout (extracted)::

    {data_root}/{category}/{seq_id}/
        {seq_id}.npy    dict with:
            traj_3d     (T, N, 3) float32 — world-space positions
            coords      (T, N, 2) float32 — pixel coords (garbage when occluded)
            occluded    (T, N) uint8 — 0=visible, 1=occluded
            intrinsics  (T, 3, 3) float32 — camera intrinsics
            matrix_world (T, 4, 4) float32 — camera-to-world (T_wc)
            depth_range  (2,) float32
        frames/
            NNNN.png         RGB frames (512x512)
            NNNN_depth.png   depth frames
            NNNN_sem.png     semantic frames

    {localframes_root}/{category}/{seq_id}.npz
        rotations (N, T, 3)   axis-angle local rotations
        rot_valid (N, T)      bool validity mask

Split JSON format: list of ["{category}/{seq_id}", frame_count].

Example::

    ds = SynthVerseDataset(
        data_root="/data/SynthVerse_extracted",
        localframes_root="/data/SynthVerse_localframes",
        split_fn="data_splits/synthverse/train.json",
        max_T=30,
        fixed_P=512,
    )
    sample, extra = ds[0]
"""

# Example YAML config entry:
# - type: synthverse
#   data_root: ./data/SynthVerse_extracted
#   localframes_root: ./data/SynthVerse_localframes
#   splits:
#     train: ./data_splits/synthverse/train.json
#   t_rate: 1
#   max_T: 30
#   fixed_P: 512

import logging
import os.path as osp
from typing import Dict

import cv2
import numpy as np
import torch

from umm.datasets.umm_v13.base_dataset import BaseDataset, CanonicalDatasetOutput
from umm.datasets.umm_v13.mask_scheduler import MaskScheduler, SCENE_PATTERNS
from umm.datasets.umm_v13.dataset_config import (
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

    Uses vectorized Rodrigues formula.

    Args:
        axis_angles: (N, 3) axis-angle rotation vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = axis_angles.shape[0]
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)  # (N, 1)
    small = angles.squeeze(-1) < 1e-8  # (N,)
    safe_angles = np.where(angles < 1e-8, 1.0, angles)
    axes = axis_angles / safe_angles  # (N, 3) unit axes

    sin_a = np.sin(angles)  # (N, 1)
    cos_a = np.cos(angles)  # (N, 1)

    kx, ky, kz = axes[:, 0], axes[:, 1], axes[:, 2]

    R = np.zeros((N, 3, 3), dtype=np.float32)
    s = sin_a.squeeze(-1)
    c = cos_a.squeeze(-1)
    t = 1.0 - c

    R[:, 0, 0] = t * kx * kx + c
    R[:, 0, 1] = t * kx * ky - s * kz
    R[:, 0, 2] = t * kx * kz + s * ky
    R[:, 1, 0] = t * kx * ky + s * kz
    R[:, 1, 1] = t * ky * ky + c
    R[:, 1, 2] = t * ky * kz - s * kx
    R[:, 2, 0] = t * kx * kz - s * ky
    R[:, 2, 1] = t * ky * kz + s * kx
    R[:, 2, 2] = t * kz * kz + c

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


def _synthesize_bg_queries_from_depth(
    data_root: str,
    sid: str,
    anchor_frame: int,
    depth_range: np.ndarray,
    intrinsics_anchor: np.ndarray,
    T_wc: np.ndarray,
    P_bg: int,
    random_flag: bool,
) -> tuple:
    """Synthesize static background queries from depth maps.

    Samples random UV locations across the full depth map and backprojects
    to world-space 3D coordinates. All pixels in SynthVerse depth maps
    have valid depth (no background sentinel). Image resolution varies
    per category (e.g. 512x512, 1280x720, 2048x858).

    Args:
        data_root: SynthVerse data root.
        sid: Sequence ID "{category}/{seq_id}".
        anchor_frame: Anchor frame index.
        depth_range: (2,) float32 [near, far] depth bounds.
        intrinsics_anchor: (3, 3) float32 camera intrinsics at anchor.
        T_wc: (4, 4) float32 camera-to-world transform at anchor.
        P_bg: Number of BG queries to synthesize.
        random_flag: If True, sample randomly; else deterministic.

    Returns:
        Tuple of:
            bg_xyz_world: (P_bg, 3) float32 world-space positions.
            bg_uv: (P_bg, 2) float32 pixel coordinates (u, v).
        Returns None if depth map unavailable.
    """
    depth_path = osp.join(data_root, sid, "frames", f"{anchor_frame:04d}_depth.png")
    if not osp.exists(depth_path):
        return None

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16 (H, W)
    if depth_raw is None or depth_raw.dtype != np.uint16:
        return None

    H, W = depth_raw.shape[:2]  # actual dims — varies per category
    n_pixels = H * W

    # Decode depth: Z = near + (far - near) * uint16 / 65535
    # All pixels have valid depth in SynthVerse (no background sentinel)

    # Sample pixel locations from full frame
    if random_flag:
        chosen = np.random.choice(n_pixels, P_bg, replace=(n_pixels < P_bg))
    else:
        step = max(1, n_pixels // P_bg)
        chosen = np.arange(0, n_pixels, step)[:P_bg]
        if len(chosen) < P_bg:
            chosen = np.concatenate([chosen, np.full(P_bg - len(chosen), chosen[0])])

    us = (chosen % W).astype(np.float32)  # (P_bg,) pixel u
    vs = (chosen // W).astype(np.float32)  # (P_bg,) pixel v
    depth_uint16 = depth_raw.ravel()[chosen].astype(np.float32)

    # Convert to metric depth (camera-space Z)
    dr_near, dr_far = float(depth_range[0]), float(depth_range[1])
    depth_Z = dr_near + (dr_far - dr_near) * depth_uint16 / 65535.0  # (P_bg,)

    # Backproject to camera space
    fx = intrinsics_anchor[0, 0]
    fy = intrinsics_anchor[1, 1]
    cx = intrinsics_anchor[0, 2]
    cy = intrinsics_anchor[1, 2]

    x_cam = (us - cx) / fx * depth_Z  # (P_bg,)
    y_cam = (vs - cy) / fy * depth_Z  # (P_bg,)
    z_cam = depth_Z                    # (P_bg,)

    # Transform to world space: xyz_world = T_wc @ [x, y, z, 1]^T
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)  # (P_bg, 4)
    pts_world = (T_wc @ pts_cam.T).T[:, :3].astype(np.float32)  # (P_bg, 3)

    bg_uv = np.stack([us, vs], axis=-1).astype(np.float32)  # (P_bg, 2)

    return pts_world, bg_uv


class SynthVerseDataset(BaseDataset):
    """SynthVerse synthetic point-tracking dataset with SE(3) traces.

    All P parts are scene pixel queries (no robot). Tracks provide 3D XYZ
    positions, local-frame rotations provide SO(3), combined into SE(3).

    Args:
        data_root: Root directory containing extracted SynthVerse data.
        localframes_root: Directory containing per-sequence localframe NPZ files.
        split_fn: Path to JSON split file (list of ["{cat}/{seq_id}", frame_count]).
        base_fps: Frame rate of the dataset (default 24 for Blender renders).
        random_flag: Enable random sampling and augmentation.
        safe_flag: If True, silently skip to next sample on errors.
        repeat: Number of times to repeat the index list.
        t_rate: Temporal stride for index expansion.
        max_T: Maximum number of time steps per trace window.
        fixed_P: Number of scene pixel queries.
        image_size: Size for DINO image (resized to image_size x image_size).
        add_camera_flag: If True, add camera trajectory as an extra part.
        reference_depth_median: If > 0, rescale scene depth to this median.
        scene_align: Scene alignment mode ("yaw" or "camera").
        npz_cache_size: Max number of cached NPZ files per worker.
    """

    def __init__(
        self,
        data_root: str,
        localframes_root: str,
        split_fn: str,
        base_fps: float = 24.0,
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        t_rate: int = 1,
        max_T: int = 30,
        fixed_P: int = 512,
        image_size=(288, 160),  # (W, H) — must be divisible by DINO patch_size (16)
        add_camera_flag: bool = False,
        reference_depth_median: float = -1.0,
        scene_align: str = "yaw",
        npz_cache_size: int = 4096,
        bg_query_ratio: float = 0.0,
        mask_scheduler: "MaskScheduler | None" = None,
    ):
        # Expand ~ in paths
        data_root = osp.expanduser(data_root)
        localframes_root = osp.expanduser(localframes_root)
        super().__init__(
            data_root, split_fn, base_fps,
            language_fn=None,
            random_flag=random_flag,
            safe_flag=safe_flag,
            repeat=repeat,
            t_rate=t_rate,
        )
        self.localframes_root = localframes_root
        assert scene_align in ("yaw", "camera"), f"scene_align must be 'yaw' or 'camera', got '{scene_align}'"
        self.scene_align = scene_align
        self.max_T = max_T
        self.max_C = 1
        self.fixed_P = fixed_P
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)  # (W, H)
        self.add_camera_flag = add_camera_flag
        self.reference_depth_median = reference_depth_median
        self.bg_query_ratio = bg_query_ratio

        self.mask_scheduler = mask_scheduler or MaskScheduler(
            base_random_prob=0.25, patterns=SCENE_PATTERNS
        )

        # LRU caches
        self._npz_cache_size = npz_cache_size
        self._npy_cache: Dict[str, dict] = {}
        self._localframes_cache: Dict[str, dict] = {}

        logger.info(
            "SynthVerseDataset: %d indices, fixed_P=%d, max_T=%d, image_size=%s, bg_query_ratio=%.2f",
            len(self.indices), fixed_P, max_T, image_size, bg_query_ratio,
        )

        # Pre-load ALL unique scene data in __init__ (main process) so that
        # forked workers share the data via copy-on-write.  Without this,
        # each worker loads multi-GB hand sequences from NFS on every access
        # (10-49 s per file), stalling training to < 0.2 it/s.
        unique_sids = sorted(set(sid for sid, _ in self.indices))
        logger.info("Pre-loading %d unique scenes into cache...", len(unique_sids))
        for i, sid in enumerate(unique_sids):
            seq_basename = osp.basename(sid)
            npy_key = osp.join(data_root, sid, seq_basename + ".npy")
            lf_path = osp.join(localframes_root, sid + ".npz")
            # Pre-load track data
            self._load_npy_cached(npy_key)
            # Pre-load local frames
            if osp.exists(lf_path):
                self._load_localframes_cached(lf_path)
            if (i + 1) % 50 == 0 or i + 1 == len(unique_sids):
                logger.info("  pre-loaded %d/%d scenes", i + 1, len(unique_sids))
        logger.info("Pre-load complete. Cache: npy=%d, lf=%d entries",
                     len(self._npy_cache), len(self._localframes_cache))

    # ── Cache helpers ─────────────────────────────────────────────────

    def _load_npy_cached(self, path: str) -> dict:
        """Load SynthVerse data with per-worker LRU cache.

        Tries .npz first (backwards-compatible format), then .npy.
        All files are cached regardless of size because the pre-load in
        __init__ fills the cache in the main process and workers share it
        via copy-on-write after fork.
        """
        if path in self._npy_cache:
            return self._npy_cache[path]
        # Try .npz first (backwards-compatible arrays)
        npz_path = path.replace(".npy", ".npz")
        if osp.exists(npz_path):
            data = dict(np.load(npz_path, allow_pickle=False))
        else:
            data = np.load(path, allow_pickle=True).item()
        if len(self._npy_cache) >= self._npz_cache_size:
            oldest = next(iter(self._npy_cache))
            del self._npy_cache[oldest]
        self._npy_cache[path] = data
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

    # ── Trace sampling ────────────────────────────────────────────────

    def __decide_trace_sampling__(
        self, seq_id: str, current_abs_tid: int, dataset_idx: int,
        random_enabled: bool = True,
    ):
        """Decide temporal window and relative anchor index."""
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
        t_len = len(frame_indices)
        anchor_frame = start_abs_tid + current_ith * factor

        # ── Load data ────────────────────────────────────────────────
        # sid is "{category}/{seq_id}" e.g. "animal/0000"
        seq_basename = osp.basename(sid)
        npy_path = osp.join(self.data_root, sid, seq_basename + ".npy")
        npz_path = osp.join(self.data_root, sid, seq_basename + ".npz")
        if not osp.exists(npy_path) and not osp.exists(npz_path):
            raise FileNotFoundError(
                f"Expected SynthVerse data at {npy_path} or {npz_path}. "
                f"Check that data_root and sid '{sid}' match the expected layout: "
                f"{{data_root}}/{{category}}/{{seq_id}}/{{seq_id}}.npy"
            )

        sv_data = self._load_npy_cached(npy_path)
        traj_3d = sv_data["traj_3d"]      # (T_total, N_total, 3)
        coords = sv_data["coords"]        # (T_total, N_total, 2)
        occluded = sv_data["occluded"]     # (T_total, N_total)
        intrinsics = sv_data["intrinsics"]  # (T_total, 3, 3)
        matrix_world = sv_data["matrix_world"]  # (T_total, 4, 4) T_wc

        T_total, N_total, _ = traj_3d.shape

        # Load local frames (stored as (N, T, 3) axis-angle)
        # sid is "category/seq_id", localframes at "{root}/{category}/{seq_id}.npz"
        lf_path = osp.join(self.localframes_root, sid + ".npz")
        lf_data = self._load_localframes_cached(lf_path)
        # rotations: (N_total, T_total, 3) axis-angle
        # rot_valid: (N_total, T_total) bool
        rotations_all = lf_data["rotations"]  # (N_total, T_total, 3)
        rot_valid_all = lf_data["rot_valid"]   # (N_total, T_total)

        # ── Consistency check: N and T must match between data and localframes
        if rotations_all.shape[0] != N_total:
            raise ValueError(
                f"N mismatch for {sid}: traj_3d has N={N_total}, "
                f"localframes has N={rotations_all.shape[0]}. "
                f"Recompute local frames for this sequence."
            )
        if rotations_all.shape[1] != T_total:
            raise ValueError(
                f"T mismatch for {sid}: traj_3d has T={T_total}, "
                f"localframes has T={rotations_all.shape[1]}. "
                f"Recompute local frames for this sequence."
            )

        # ── Visibility: SynthVerse uses (T, N) layout ─────────────────
        vis_TN = (occluded == 0).astype(np.float32)  # (T_total, N_total) 1=visible

        # ── Sample P queries: FG tracks + optional BG from depth ──────
        P = self.fixed_P
        P_bg = int(P * self.bg_query_ratio) if self.bg_query_ratio > 0 else 0
        P_fg = P - P_bg
        if self.bg_query_ratio > 0 and P_bg == 0:
            logger.debug("bg_query_ratio=%.2f but P_bg=0 (P=%d too small)", self.bg_query_ratio, P)

        vis_at_cur = vis_TN[current_abs_tid]          # (N_total,) visibility at anchor
        rv_at_cur = rot_valid_all[:, current_abs_tid]  # (N_total,) rotation valid at anchor
        valid_pool = np.where((vis_at_cur > 0.5) & rv_at_cur)[0]

        pool = valid_pool if len(valid_pool) > 0 else np.arange(N_total)

        if self.random_flag:
            fg_indices = np.random.choice(pool, P_fg, replace=(len(pool) < P_fg))
        else:
            if len(pool) >= P_fg:
                step = max(1, len(pool) // P_fg)
                fg_indices = pool[np.arange(0, len(pool), step)[:P_fg]]
                if len(fg_indices) < P_fg:
                    fg_indices = np.concatenate([
                        fg_indices,
                        np.full(P_fg - len(fg_indices), pool[0], dtype=np.int64),
                    ])
            else:
                repeat_times = (P_fg + len(pool) - 1) // len(pool)
                fg_indices = np.tile(pool, repeat_times)[:P_fg]

        # ── Extract FG temporal window ────────────────────────────────
        fi = np.array(frame_indices, dtype=np.int64)  # (t_len,)

        fg_positions = traj_3d[fi][:, fg_indices, :].transpose(1, 0, 2).astype(np.float32)  # (P_fg, t_len, 3)
        fg_vis = vis_TN[fi][:, fg_indices].T.astype(np.float32)  # (P_fg, t_len)
        fg_uv = coords[fi][:, fg_indices, :].transpose(1, 0, 2).astype(np.float32)  # (P_fg, t_len, 2)
        fg_rotations = rotations_all[fg_indices][:, fi, :]  # (P_fg, t_len, 3) axis-angle
        fg_rot_valid = rot_valid_all[fg_indices][:, fi]      # (P_fg, t_len) bool

        # ── Synthesize BG queries from depth maps ─────────────────────
        bg_result = None
        if P_bg > 0:
            depth_range = sv_data.get("depth_range", None)
            if depth_range is not None:
                cam_T_wc_anchor = matrix_world[anchor_frame].astype(np.float32)
                K_anchor_raw = intrinsics[anchor_frame].astype(np.float32)
                bg_result = _synthesize_bg_queries_from_depth(
                    self.data_root, sid, anchor_frame, depth_range,
                    K_anchor_raw, cam_T_wc_anchor, P_bg, self.random_flag,
                )

        if bg_result is not None:
            bg_xyz_world, bg_uv_anchor = bg_result  # (P_bg, 3), (P_bg, 2)
            # BG queries: static across all frames, identity rotation
            bg_positions = np.broadcast_to(
                bg_xyz_world[:, None, :], (P_bg, t_len, 3)
            ).copy().astype(np.float32)
            bg_vis = np.ones((P_bg, t_len), dtype=np.float32)
            bg_uv = np.broadcast_to(
                bg_uv_anchor[:, None, :], (P_bg, t_len, 2)
            ).copy().astype(np.float32)
            bg_rotations = np.zeros((P_bg, t_len, 3), dtype=np.float32)  # identity
            bg_rot_valid = np.ones((P_bg, t_len), dtype=bool)

            # Concatenate FG + BG
            positions = np.concatenate([fg_positions, bg_positions], axis=0)
            vis = np.concatenate([fg_vis, bg_vis], axis=0)
            uv_raw = np.concatenate([fg_uv, bg_uv], axis=0)
            rotations_sel = np.concatenate([fg_rotations, bg_rotations], axis=0)
            rot_valid_sel = np.concatenate([fg_rot_valid, bg_rot_valid], axis=0)
        else:
            # No BG (depth unavailable or bg_query_ratio=0): fill with extra FG
            if P_bg > 0:
                # Fallback: sample more FG to fill P
                extra = np.random.choice(pool, P_bg, replace=True) if self.random_flag else \
                    np.tile(pool, (P_bg + len(pool) - 1) // len(pool))[:P_bg]
                extra_pos = traj_3d[fi][:, extra, :].transpose(1, 0, 2).astype(np.float32)
                extra_vis = vis_TN[fi][:, extra].T.astype(np.float32)
                extra_uv = coords[fi][:, extra, :].transpose(1, 0, 2).astype(np.float32)
                extra_rot = rotations_all[extra][:, fi, :]
                extra_rv = rot_valid_all[extra][:, fi]
                fg_indices = np.concatenate([fg_indices, extra])
                positions = np.concatenate([fg_positions, extra_pos], axis=0)
                vis = np.concatenate([fg_vis, extra_vis], axis=0)
                uv_raw = np.concatenate([fg_uv, extra_uv], axis=0)
                rotations_sel = np.concatenate([fg_rotations, extra_rot], axis=0)
                rot_valid_sel = np.concatenate([fg_rot_valid, extra_rv], axis=0)
            else:
                positions = fg_positions
                vis = fg_vis
                uv_raw = fg_uv
                rotations_sel = fg_rotations
                rot_valid_sel = fg_rot_valid

        # ── Build SE(3) from axis-angle + XYZ ─────────────────────────
        rot_flat = rotations_sel.reshape(-1, 3).astype(np.float32)
        R_flat = _axis_angle_to_rotation_matrix_batch(rot_flat)  # (P*t_len, 3, 3)
        R_all = R_flat.reshape(P, t_len, 3, 3)

        se3 = np.broadcast_to(_EYE4_F32, (P, t_len, 4, 4)).copy()
        se3[:, :, :3, :3] = R_all
        se3[:, :, :3, 3] = positions

        # ── Build trace_mask = (vis > 0.5) & rot_valid ────────────────
        trace_mask = ((vis > 0.5) & rot_valid_sel).astype(np.float32)  # (P, t_len)

        # Replace invalid entries with identity
        invalid = trace_mask < 0.5
        se3[invalid] = _EYE4_F32

        # ── Context frame: use anchor frame (DAVIS-like behavior) ─────
        ctx_frame = current_abs_tid
        # FG context from tracks
        fg_ctx_positions = traj_3d[ctx_frame, fg_indices, :].astype(np.float32)
        fg_ctx_rotations = rotations_all[fg_indices, ctx_frame, :].astype(np.float32)
        fg_ctx_rot_valid = rot_valid_all[fg_indices, ctx_frame]
        fg_ctx_vis = vis_TN[ctx_frame, fg_indices]

        if bg_result is not None:
            # BG context: static position, identity rotation, always valid
            bg_ctx_positions = bg_result[0].astype(np.float32)  # (P_bg, 3)
            ctx_all_positions = np.concatenate([fg_ctx_positions, bg_ctx_positions], axis=0)
            ctx_all_rotations = np.concatenate([
                fg_ctx_rotations,
                np.zeros((P_bg, 3), dtype=np.float32),
            ], axis=0)
            ctx_all_rot_valid = np.concatenate([
                fg_ctx_rot_valid,
                np.ones(P_bg, dtype=bool),
            ])
            ctx_all_vis = np.concatenate([
                fg_ctx_vis,
                np.ones(P_bg, dtype=np.float32),
            ])
        else:
            ctx_all_positions = fg_ctx_positions
            ctx_all_rotations = fg_ctx_rotations
            ctx_all_rot_valid = fg_ctx_rot_valid
            ctx_all_vis = fg_ctx_vis

        ctx_R = _axis_angle_to_rotation_matrix_batch(ctx_all_rotations)  # (P, 3, 3)
        context = np.broadcast_to(_EYE4_F32, (P, 1, 4, 4)).copy()
        context[:, 0, :3, :3] = ctx_R
        context[:, 0, :3, 3] = ctx_all_positions
        context_mask = ((ctx_all_vis > 0.5) & ctx_all_rot_valid).astype(np.float32)[:, None]  # (P, 1)
        ctx_invalid = context_mask[:, 0] < 0.5
        context[ctx_invalid, 0] = _EYE4_F32

        # ── Scene alignment ──────────────────────────────────────────
        cam_T_wc = matrix_world[anchor_frame].astype(np.float32)  # (4, 4)

        if self.scene_align == "camera":
            R_reorient = np.array([
                [0, 0, -1, 0],
                [1, 0,  0, 0],
                [0, -1, 0, 0],
                [0, 0,  0, 1],
            ], dtype=np.float32)
            T_align = (R_reorient @ np.linalg.inv(cam_T_wc)).astype(np.float32)
        else:
            # Yaw mode
            anchor_positions = positions[:, current_ith, :]  # (P, 3)
            anchor_vis = vis[:, current_ith]
            visible_at_anchor = anchor_vis > 0.5
            if visible_at_anchor.any():
                center_xyz = anchor_positions[visible_at_anchor].mean(axis=0)
            else:
                center_xyz = anchor_positions.mean(axis=0)
            center_xy = center_xyz[:2].astype(np.float32)

            cam_z_xy = cam_T_wc[:2, 2]
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

        # ── Per-scene depth rescaling (camera-space Z at current frame) ──
        scene_scale = 1.0
        if self.reference_depth_median > 0:
            # Use camera-space Z depth, not radial distance from aligned origin.
            # Transform anchor-frame positions to camera space and take Z.
            T_cw = np.linalg.inv(cam_T_wc).astype(np.float32)  # (4, 4) world-to-camera
            anchor_pos_world = positions[:, current_ith, :]  # (P, 3) WORLD positions at anchor
            anchor_valid = trace_mask[:, current_ith] > 0.5  # (P,)
            if anchor_valid.any():
                valid_pos = anchor_pos_world[anchor_valid]  # (M, 3)
                # Project to camera space: p_cam = R_cw @ p_world + t_cw
                pts_cam = (T_cw[:3, :3] @ valid_pos.T + T_cw[:3, 3:4]).T  # (M, 3)
                z_depths = pts_cam[:, 2]  # camera-space Z
                valid_z = z_depths[z_depths > 1e-6]
                if len(valid_z) > 0:
                    med = float(np.median(valid_z))
                    scene_scale = self.reference_depth_median / med
                    se3[:, :, :3, 3] *= scene_scale
                    context[:, :, :3, 3] *= scene_scale

        # ── Pad trace to max_T ────────────────────────────────────────
        trace_padded, trace_mask_padded = _pad_trace_np(se3, trace_mask, T_max)

        # ── Group IDs, sem types, UIDs ────────────────────────────────
        # Per-query individual groups — no grouping assumed during training
        group_ids = np.arange(P, dtype=np.int64)
        sem_types = np.full(P, SEM_REAL_SCENE, dtype=np.int64)
        uid_arr = np.full(P, UID_REAL_SCENE, dtype=np.int64)
        rot_mask = np.ones(P, dtype=bool)

        # ── Local PCL: zeros ──────────────────────────────────────────
        pcl = np.zeros((P, 0, 6), dtype=np.float32)
        pcl_mask = np.zeros(P, dtype=np.float32)

        # ── Load anchor frame image for DINO ──────────────────────────
        target_W, target_H = int(self.image_size[0]), int(self.image_size[1])

        # SynthVerse images: frames/{NNNN}.png
        img_path = osp.join(self.data_root, sid, "frames", f"{anchor_frame:04d}.png")
        if osp.exists(img_path):
            rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
                img_H, img_W = rgb.shape[:2]
                resized = cv2.resize(rgb, (target_W, target_H), interpolation=cv2.INTER_AREA)
                image_dino = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
                images_for_dino = image_dino[None, :, :, :]  # (1, 3, H, W)
                scale_x = target_W / img_W
                scale_y = target_H / img_H
            else:
                images_for_dino = np.zeros((1, 3, target_H, target_W), dtype=np.float32)
                img_H, img_W = 512, 512
                scale_x, scale_y = target_W / 512, target_H / 512
        else:
            images_for_dino = np.zeros((1, 3, target_H, target_W), dtype=np.float32)
            img_H, img_W = 512, 512
            scale_x, scale_y = target_W / 512, target_H / 512

        # ── Pixel UV normalized + view idx ────────────────────────────
        # UV at anchor frame: for occluded tracks, use 0
        anchor_uv = uv_raw[:, current_ith, :].copy()  # (P, 2)
        # Clean up garbage coords for invisible points
        anchor_vis = vis[:, current_ith] > 0.5  # (P,)
        anchor_uv[~anchor_vis] = 0.0

        anchor_uv_resized = anchor_uv.copy()
        anchor_uv_resized[:, 0] *= scale_x
        anchor_uv_resized[:, 1] *= scale_y
        pixel_uv_normalized = np.stack([
            anchor_uv_resized[:, 0] / max(target_W - 1, 1),
            anchor_uv_resized[:, 1] / max(target_H - 1, 1),
        ], axis=-1).astype(np.float32)  # (P, 2) normalized [0, 1]
        # Clamp to [0, 1]
        pixel_uv_normalized = np.clip(pixel_uv_normalized, 0.0, 1.0)

        pixel_view_idx = np.zeros(P, dtype=np.int64)
        pixel_query_mask = np.ones(P, dtype=bool)

        # ── Add camera trajectory if requested ────────────────────────
        next_group_id = P
        if self.add_camera_flag:
            cam_trace = np.stack(
                [matrix_world[fi_idx].astype(np.float32) for fi_idx in frame_indices],
                axis=0,
            )  # (t_len, 4, 4)
            cam_ctx = matrix_world[ctx_frame].astype(np.float32)[None, :, :]

            cam_trace = np.einsum('ij,tjk->tik', T_align, cam_trace)
            cam_ctx = np.einsum('ij,cjk->cik', T_align, cam_ctx)

            if scene_scale != 1.0:
                cam_trace[:, :3, 3] *= scene_scale
                cam_ctx[:, :3, 3] *= scene_scale

            cam_trace_4d = cam_trace[None, :, :, :]
            cam_mask = np.ones((1, t_len), dtype=np.float32)
            cam_trace_padded, cam_mask_padded = _pad_trace_np(cam_trace_4d, cam_mask, T_max)
            cam_ctx_4d = cam_ctx[None, :, :, :]
            cam_ctx_mask = np.ones((1, 1), dtype=np.float32)

            trace_padded = np.concatenate([trace_padded, cam_trace_padded], axis=0)
            trace_mask_padded = np.concatenate([trace_mask_padded, cam_mask_padded], axis=0)
            context = np.concatenate([context, cam_ctx_4d], axis=0)
            context_mask = np.concatenate([context_mask, cam_ctx_mask], axis=0)
            group_ids = np.concatenate([group_ids, np.array([next_group_id], dtype=np.int64)])
            sem_types = np.concatenate([sem_types, np.array([SEM_REAL_CAMERA], dtype=np.int64)])
            uid_arr = np.concatenate([uid_arr, np.array([UID_REAL_CAMERA], dtype=np.int64)])
            rot_mask = np.concatenate([rot_mask, np.array([True])])
            pcl = np.concatenate([pcl, np.zeros((1, 0, 6), dtype=np.float32)], axis=0)
            pcl_mask = np.concatenate([pcl_mask, np.zeros(1, dtype=np.float32)])
            pixel_uv_normalized = np.concatenate(
                [pixel_uv_normalized, np.zeros((1, 2), dtype=np.float32)], axis=0
            )
            pixel_view_idx = np.concatenate([pixel_view_idx, np.zeros(1, dtype=np.int64)])
            pixel_query_mask = np.concatenate([pixel_query_mask, np.array([False])])

        # ── Single torch conversion ───────────────────────────────────
        trace_t = torch.from_numpy(trace_padded).float()
        trace_mask_t = torch.from_numpy(trace_mask_padded).float()
        context_t = torch.from_numpy(context).float()
        context_mask_t = torch.from_numpy(context_mask).float()
        rot_mask_t = torch.from_numpy(rot_mask).bool()
        group_id_t = torch.from_numpy(group_ids).long()
        sem_type_t = torch.from_numpy(sem_types).long()
        uid_t = torch.from_numpy(uid_arr).long()

        # PCL: need at least 1 point for collate
        total_P = trace_padded.shape[0]
        pcl_t = torch.zeros((total_P, 1, 6), dtype=torch.float32)
        pcl_mask_t = torch.zeros(total_P, dtype=torch.bool)

        time_unit = float(self.t_rate) / float(self.base_fps)
        time_stamp = (torch.arange(T_max) - current_ith) * time_unit

        # Text: use category/seq_id as text
        text = sid

        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask_t,
            context_mask=context_mask_t,
            sem_type=sem_type_t,
            current_ith=current_ith,
        )

        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace_t,
            trace_mask=trace_mask_t,
            context=context_t,
            context_mask=context_mask_t,
            known_mask=known_mask,
            context_known_mask=ctx_known_mask,
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
        K_anchor = intrinsics[anchor_frame].astype(np.float32).copy()
        K_anchor[0, 0] *= scale_x
        K_anchor[0, 2] *= scale_x
        K_anchor[1, 1] *= scale_y
        K_anchor[1, 2] *= scale_y

        extra_dict = {
            "images": images_for_dino,
            "pixel_uv_normalized": pixel_uv_normalized,
            "pixel_view_idx": pixel_view_idx,
            "pixel_query_mask": pixel_query_mask,
            "viz_obj_pcl": np.zeros((total_P, 1, 6), dtype=np.float32),
            "rgbs_paths": [osp.join(self.data_root, sid, "frames")],
            "rgbs_views": ["head"],
            "rgbs_indices": frame_indices,
            "_T_align": T_align,
            "_cam_T_wc": cam_T_wc,
            "_K_resized": K_anchor,
            "_img_resize_wh": np.array([target_W, target_H], dtype=np.int32),
            "_img_orig_wh": np.array([img_W, img_H], dtype=np.int32),
            "_scale_xy": np.array([scale_x, scale_y], dtype=np.float32),
            "_cam_K_all": intrinsics.astype(np.float32),
            "_cam_T_wc_all": matrix_world.astype(np.float32),
            "_frame_indices": np.array(frame_indices, dtype=np.int64),
            "_anchor_frame": anchor_frame,
            "_anchor_ith": current_ith,
            "_data_root": self.data_root,
            "_seq_id": sid,
            "_scene_scale": np.float32(scene_scale),
        }

        return ret, extra_dict
