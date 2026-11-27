"""SynthVerse trajectory-sharded dataset — lazy loads per-shard npz files.

Complements :class:`SynthVerseChunkedDataset` (which slices by TIME) with an
alternative IO strategy that slices by TRAJECTORY: each npz shard contains
``N_shard <= 2048`` trajectories for the FULL temporal extent of its source
scene.  ``__getitem__`` loads one shard, samples ``fixed_P`` trajectories from
its ``N_shard``, and returns the same ``(CanonicalDatasetOutput, extra_dict)``
tuple as the chunked dataset.

Data layout::

    {data_root}/{cat}/{sid_basename}/
        index.json                                  per-scene shard manifest
        shard_0000_0000.npz                         main shards (timechunk_shardidx)
        ...

    {localframes_root}/{cat}/{sid_basename}/
        shard_0000_0000.npz                         localframes shards
        ...

    {frames_root}/{cat}/{sid_basename}/frames/NNNN.png   original PNG frames

Each main shard npz contains:
    traj_3d:       (T_scene, N_shard, 3) float32
    coords:        (T_scene, N_shard, 2) float32
    occluded:      (T_scene, N_shard)   uint8
    intrinsics:    (T_scene, 3, 3)      float32   (scene-level, same for all shards)
    matrix_world:  (T_scene, 4, 4)      float32   (scene-level, same for all shards)
    depth_range:   (2,)                 float32
    n_trajectories: scalar int           (number of trajectories in this shard)

Each localframes shard npz contains:
    rotations:  (N_shard, T_scene, 3) float32   axis-angle
    rot_valid:  (N_shard, T_scene)    bool
"""

import json
import logging
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from umm.datasets.wmm.base_dataset import BaseDataset, CanonicalDatasetOutput
from umm.datasets.wmm.mask_scheduler import MaskScheduler, SCENE_PATTERNS
from umm.datasets.wmm.dataset_config import (
    UID_REAL_SCENE,
    UID_REAL_CAMERA,
    SEM_REAL_SCENE,
    SEM_REAL_CAMERA,
)
from umm.datasets.wmm.synthverse_chunked_dataset import (
    ChunkCorruptionError,
    _axis_angle_to_rotation_matrix_batch,
    _pad_trace_np,
    _synthesize_bg_queries_from_depth,
)

logger = logging.getLogger(__name__)

# Pre-allocated identity matrix (avoid repeated allocation)
_EYE4_F32 = np.eye(4, dtype=np.float32)


class SynthVerseShardedDataset(BaseDataset):
    """Lazy trajectory-sharded SynthVerse point-tracking dataset.

    Unlike :class:`SynthVerseChunkedDataset` (which slices scenes by time),
    this class slices by trajectory: each npz shard holds up to 2048
    trajectories covering the full temporal extent of its source scene.
    ``__getitem__`` loads one shard and samples ``fixed_P`` trajectories from
    the shard's ``N_shard``, keeping all frames.  This eliminates the RAM
    pre-load problem while preserving the full temporal signal per trajectory.

    The output is identical in format to :class:`SynthVerseChunkedDataset`.

    Args:
        data_root: Root of traj-sharded npz data (e.g.
            ``data/SynthVerse_traj_sharded``).
        localframes_root: Root of traj-sharded localframes npz data.
        frames_root: Root of the ORIGINAL extracted scenes for PNG image
            loading (e.g. ``data/SynthVerse_extracted``).
        split_fn: Path to JSON split file
            (list of ``["{cat}/{sid}", orig_T]``).
        base_fps: Native frame rate of the dataset.
        random_flag: Enable random sampling and augmentation.
        safe_flag: If True, silently skip to next sample on errors.
        repeat: Number of times to repeat the index list.
        t_rate: Temporal stride for index expansion (applied to full scene T).
        max_T: Maximum number of time steps per trace window.
        fixed_P: Number of scene pixel queries sampled per shard.
        image_size: ``(W, H)`` for the DINO image resize
            (both must be divisible by the DINO patch size, 16).
        add_camera_flag: If True, add camera trajectory as an extra part.
        reference_depth_median: If > 0, rescale scene depth to this median.
        scene_align: Scene alignment mode (``"yaw"`` or ``"camera"``).
        bg_query_ratio: Fraction of ``fixed_P`` reserved for BG queries
            synthesized from the anchor-frame depth map.
        mask_scheduler: Optional :class:`MaskScheduler` for known-mask
            sampling.
    """

    def __init__(
        self,
        data_root: str,
        localframes_root: str,
        frames_root: str,
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
        bg_query_ratio: float = 0.0,
        mask_scheduler: Optional["MaskScheduler"] = None,
    ):
        # Expand ~ in all paths up-front.
        data_root = osp.expanduser(data_root)
        localframes_root = osp.expanduser(localframes_root)
        frames_root = osp.expanduser(frames_root)

        # BaseDataset.__init__ sets self.data_root, language_data, IO handle,
        # and populates self.indices / self.seq_len from the ORIGINAL split.
        # We overwrite self.indices / self.seq_len below with shard-aware
        # versions built from the per-scene index.json manifests.
        super().__init__(
            data_root, split_fn, base_fps,
            language_fn=None,
            random_flag=random_flag,
            safe_flag=safe_flag,
            repeat=repeat,
            t_rate=t_rate,
        )
        self.localframes_root = localframes_root
        self.frames_root = frames_root
        assert scene_align in ("yaw", "camera"), (
            f"scene_align must be 'yaw' or 'camera', got '{scene_align}'"
        )
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

        # ── Rebuild indices from per-scene index.json files ──────────────
        # The indices produced by BaseDataset.__init__ were keyed on the
        # ORIGINAL scene ids. We replace them with shard-aware indices:
        #   indices:  [(shard_key, tid_in_chunk), ...]
        #   seq_len:  {shard_key: n_frames}
        # where shard_key = "{sid}::tc{chunk_idx}_ts{shard_idx}" (:: avoids '/' collision).
        # Each shard covers n_frames of its time chunk (chunk-local indexing).
        original_scene_sids = sorted({sid for sid, _ in self.indices})

        self.shard_info: Dict[str, Dict] = {}
        new_indices: List[Tuple[str, int]] = []
        new_seq_len: Dict[str, int] = {}

        missing_scenes: List[str] = []
        for sid in original_scene_sids:
            scene_dir = osp.join(data_root, sid)
            idx_json = osp.join(scene_dir, "index.json")
            if not osp.exists(idx_json):
                missing_scenes.append(sid)
                continue
            with open(idx_json, "r") as f:
                scene_meta = json.load(f)
            original_T = int(scene_meta["original_T"])
            for s in scene_meta["shards"]:
                shard_idx = int(s["shard_idx"])
                n_trajs = int(s["n_trajectories"])
                time_chunk_idx = int(s["time_chunk_idx"])
                shard_fname = f"shard_{time_chunk_idx:04d}_{shard_idx:04d}.npz"
                main_path = osp.join(scene_dir, shard_fname)
                lf_path = osp.join(localframes_root, sid, shard_fname)
                chunk_start = int(s["chunk_start"])
                n_frames = int(s["n_frames"])
                shard_key = f"{sid}::tc{time_chunk_idx:02d}_ts{shard_idx:04d}"
                self.shard_info[shard_key] = {
                    "sid": sid,
                    "shard_idx": shard_idx,
                    "n_trajs": n_trajs,
                    "chunk_start": chunk_start,
                    "n_frames": n_frames,
                    "original_T": original_T,
                    "main_path": main_path,
                    "lf_path": lf_path,
                }
                # Shard covers n_frames of its time chunk (chunk-local indexing).
                new_seq_len[shard_key] = n_frames
                # Expand across chunk time at stride t_rate (chunk-local tids).
                for tid in range(0, n_frames, int(t_rate)):
                    new_indices.append((shard_key, tid))

        # Apply repeat explicitly — load_indices does not, and we're
        # bypassing it here since we built the indices manually.
        if repeat > 1:
            new_indices = new_indices * int(repeat)

        self.indices = new_indices
        self.seq_len = new_seq_len

        # Fail LOUD if any scene from the split is missing its index.json.
        if missing_scenes:
            preview = ", ".join(missing_scenes[:5])
            if len(missing_scenes) > 5:
                preview += f", ... (+{len(missing_scenes) - 5} more)"
            raise RuntimeError(
                f"SynthVerseShardedDataset: {len(missing_scenes)}/"
                f"{len(original_scene_sids)} scenes from split '{split_fn}' "
                f"are missing their `index.json` under {data_root}. "
                f"The traj-shard script must complete for every scene before "
                f"this dataset is used. Missing: {preview}"
            )
        if len(self.indices) == 0:
            raise RuntimeError(
                f"SynthVerseShardedDataset: no shards found under {data_root}. "
                f"Run the trajectory-shard preparation script first."
            )

        logger.info(
            "SynthVerseShardedDataset: %d shards, %d total indices, "
            "fixed_P=%d, max_T=%d, image_size=%s, bg_query_ratio=%.2f",
            len(self.shard_info), len(self.indices), fixed_P, max_T,
            image_size, bg_query_ratio,
        )
        logger.info("SynthVerseShardedDataset: lazy-load mode (no pre-load cache)")

    # ── Trace sampling ────────────────────────────────────────────────

    def __decide_trace_sampling__(
        self, seq_id: str, current_abs_tid: int, dataset_idx: int,
        random_enabled: bool = True,
    ):
        """Decide temporal window and relative anchor index.

        Operates on chunk-local time (0..n_frames-1 for this shard's time chunk).
        ``seq_id`` is the shard_key; ``current_abs_tid`` is the chunk-local
        frame index.  Windows clip at chunk boundaries.

        Args:
            seq_id: Shard key (used to look up seq_len).
            current_abs_tid: Current frame index in the scene.
            dataset_idx: Global dataset index (for deterministic mode).
            random_enabled: If True, use random window placement.

        Returns:
            Tuple of (current_ith, start_abs_tid, end_abs_tid).
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

    # ── Main getitem ──────────────────────────────────────────────────

    def __getitem_inner__(self, idx: int):
        """Build one sample: SE(3) traces from trajectory-sharded 3D tracks.

        Loads one shard (up to 2048 trajectories, full scene T), samples
        ``fixed_P`` trajectories, builds SE(3) traces + context, applies
        scene alignment, loads anchor-frame DINO image, and assembles the
        canonical output.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of ``(CanonicalDatasetOutput, extra_dict)``.
        """
        idx = idx % len(self)
        shard_key, current_tid = self.indices[idx]  # current_tid is chunk-local
        meta = self.shard_info[shard_key]
        sid = meta["sid"]
        n_frames = meta["n_frames"]
        chunk_start = meta["chunk_start"]
        original_T = meta["original_T"]
        main_path = meta["main_path"]
        lf_path = meta["lf_path"]

        current_ith, start_tid, end_tid = self.__decide_trace_sampling__(
            shard_key, current_tid, idx, random_enabled=self.random_flag
        )

        factor = int(self.t_rate)
        T_max = self.max_T

        # Frame indices for the trace window — chunk-local (0-based).
        frame_indices = list(range(start_tid, end_tid, factor))
        t_len = len(frame_indices)
        anchor_frame = start_tid + current_ith * factor  # chunk-local anchor
        scene_anchor = chunk_start + anchor_frame  # scene-level for PNG paths

        # ── Load data — NO cache, NO pre-load ────────────────────────
        if not osp.exists(main_path):
            raise FileNotFoundError(
                f"SynthVerseShardedDataset: missing main shard file {main_path}. "
                f"Run the trajectory-shard preparation script to regenerate."
            )
        if not osp.exists(lf_path):
            raise FileNotFoundError(
                f"SynthVerseShardedDataset: missing localframes shard file {lf_path}. "
                f"Run the trajectory-shard preparation script to regenerate."
            )

        sv_data = dict(np.load(main_path, allow_pickle=False))
        traj_3d = sv_data["traj_3d"]          # (T_scene, N_shard, 3)
        coords = sv_data["coords"]             # (T_scene, N_shard, 2)
        occluded = sv_data["occluded"]         # (T_scene, N_shard) uint8
        intrinsics = sv_data["intrinsics"]     # (T_scene, 3, 3)
        matrix_world = sv_data["matrix_world"]  # (T_scene, 4, 4) T_wc
        depth_range = sv_data["depth_range"]   # (2,) float32
        n_trajs_stored = int(sv_data["n_trajectories"])

        T_scene, N_shard, _ = traj_3d.shape

        # Guard: n_trajs in npz must match index.json.
        expected_n_trajs = int(meta["n_trajs"])
        if n_trajs_stored != expected_n_trajs:
            raise ChunkCorruptionError(
                f"SynthVerseShardedDataset: shard {shard_key} has "
                f"n_trajs={n_trajs_stored} but index.json reports "
                f"n_trajs={expected_n_trajs}. File {main_path} is "
                f"out-of-sync with its index.json — re-run the shard script."
            )
        # Guard: T must match n_frames in index.json.
        if T_scene != n_frames:
            raise ChunkCorruptionError(
                f"SynthVerseShardedDataset: shard {shard_key} has "
                f"traj_3d.shape[0]={T_scene} but index.json reports "
                f"n_frames={n_frames}. File {main_path} is "
                f"out-of-sync with its index.json — re-run the shard script."
            )

        lf_data = dict(np.load(lf_path, allow_pickle=False))
        rotations_all = lf_data["rotations"]   # (N_shard, T_scene, 3)
        rot_valid_all = lf_data["rot_valid"]    # (N_shard, T_scene) bool

        # ── Consistency check: N and T must match between main and lf ─
        if rotations_all.shape[0] != N_shard:
            raise ValueError(
                f"N mismatch for {shard_key}: traj_3d has N_shard={N_shard}, "
                f"localframes has N={rotations_all.shape[0]}. "
                f"Recompute local frames for this shard."
            )
        if rotations_all.shape[1] != T_scene:
            raise ValueError(
                f"T mismatch for {shard_key}: traj_3d has T={T_scene}, "
                f"localframes has T={rotations_all.shape[1]}. "
                f"Recompute local frames for this shard."
            )

        # ── Visibility: (T, N) layout ─────────────────────────────────
        vis_TN = (occluded == 0).astype(np.float32)  # (T_scene, N_shard) 1=visible

        # ── Sample P queries: FG tracks + optional BG from depth ──────
        P = self.fixed_P
        P_bg = int(P * self.bg_query_ratio) if self.bg_query_ratio > 0 else 0
        P_fg = P - P_bg
        if self.bg_query_ratio > 0 and P_bg == 0:
            logger.debug(
                "bg_query_ratio=%.2f but P_bg=0 (P=%d too small)",
                self.bg_query_ratio, P,
            )

        vis_at_cur = vis_TN[current_tid]              # (N_shard,)
        rv_at_cur = rot_valid_all[:, current_tid]      # (N_shard,)
        valid_pool = np.where((vis_at_cur > 0.5) & rv_at_cur)[0]

        pool = valid_pool if len(valid_pool) > 0 else np.arange(N_shard)

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
        fi = np.array(frame_indices, dtype=np.int64)  # (t_len,) chunk-local

        fg_positions = traj_3d[fi][:, fg_indices, :].transpose(1, 0, 2).astype(np.float32)  # (P_fg, t_len, 3)
        fg_vis = vis_TN[fi][:, fg_indices].T.astype(np.float32)  # (P_fg, t_len)
        fg_uv = coords[fi][:, fg_indices, :].transpose(1, 0, 2).astype(np.float32)  # (P_fg, t_len, 2)
        fg_rotations = rotations_all[fg_indices][:, fi, :]  # (P_fg, t_len, 3) axis-angle
        fg_rot_valid = rot_valid_all[fg_indices][:, fi]      # (P_fg, t_len) bool

        # ── Synthesize BG queries from depth maps ─────────────────────
        # Depth PNGs live at the ORIGINAL extracted root, keyed by the
        # scene-level PNG frame index (chunk_start + chunk-local anchor).
        bg_result = None
        if P_bg > 0:
            cam_T_wc_anchor = matrix_world[anchor_frame].astype(np.float32)
            K_anchor_raw = intrinsics[anchor_frame].astype(np.float32)
            bg_result = _synthesize_bg_queries_from_depth(
                self.frames_root, sid, scene_anchor, depth_range,
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
                extra = (
                    np.random.choice(pool, P_bg, replace=True)
                    if self.random_flag
                    else np.tile(pool, (P_bg + len(pool) - 1) // len(pool))[:P_bg]
                )
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
        ctx_frame = current_tid  # chunk-local
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
        context_mask = (
            (ctx_all_vis > 0.5) & ctx_all_rot_valid
        ).astype(np.float32)[:, None]  # (P, 1)
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
            anchor_vis_at_ith = vis[:, current_ith]
            visible_at_anchor = anchor_vis_at_ith > 0.5
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
            T_cw = np.linalg.inv(cam_T_wc).astype(np.float32)  # (4, 4) world-to-camera
            anchor_pos_world = positions[:, current_ith, :]  # (P, 3) WORLD positions at anchor
            anchor_valid = trace_mask[:, current_ith] > 0.5  # (P,)
            if anchor_valid.any():
                valid_pos = anchor_pos_world[anchor_valid]  # (M, 3)
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
        group_ids = np.arange(P, dtype=np.int64)
        sem_types = np.full(P, SEM_REAL_SCENE, dtype=np.int64)
        uid_arr = np.full(P, UID_REAL_SCENE, dtype=np.int64)
        rot_mask = np.ones(P, dtype=bool)

        # ── Local PCL: zeros ──────────────────────────────────────────
        pcl = np.zeros((P, 0, 6), dtype=np.float32)
        pcl_mask = np.zeros(P, dtype=np.float32)

        # ── Load anchor frame image for DINO ──────────────────────────
        # PNGs live at the ORIGINAL extracted root, keyed by scene_anchor.
        target_W, target_H = int(self.image_size[0]), int(self.image_size[1])

        img_path = osp.join(
            self.frames_root, sid, "frames", f"{scene_anchor:04d}.png"
        )
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
        anchor_uv = uv_raw[:, current_ith, :].copy()  # (P, 2)
        anchor_vis_uv = vis[:, current_ith] > 0.5  # (P,)
        anchor_uv[~anchor_vis_uv] = 0.0

        anchor_uv_resized = anchor_uv.copy()
        anchor_uv_resized[:, 0] *= scale_x
        anchor_uv_resized[:, 1] *= scale_y
        pixel_uv_normalized = np.stack([
            anchor_uv_resized[:, 0] / max(target_W - 1, 1),
            anchor_uv_resized[:, 1] / max(target_H - 1, 1),
        ], axis=-1).astype(np.float32)  # (P, 2) normalized [0, 1]
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

        # Text: use original scene id (preserves training semantics).
        text = sid

        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask_t,
            context_mask=context_mask_t,
            sem_type=sem_type_t,
            current_ith=current_ith,
        )

        ret = CanonicalDatasetOutput(
            seq_name=shard_key,  # shard-aware for debug disambiguation
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

        # Viz consumer contract (matches synthverse_chunked_dataset.py:880-894):
        # `_frame_indices` and `_anchor_frame` are scene-level (for PNG paths).
        # `_cam_K_all` / `_cam_T_wc_all` are padded to (original_T, ...) so
        # scene-level indices work for both camera lookups and PNG paths.
        scene_frame_indices = [chunk_start + fi_val for fi_val in frame_indices]
        rgbs_indices_original = scene_frame_indices

        K_full = np.zeros((original_T, 3, 3), dtype=np.float32)
        T_wc_full = np.broadcast_to(_EYE4_F32, (original_T, 4, 4)).copy()
        K_full[chunk_start:chunk_start + T_scene] = intrinsics.astype(np.float32)
        T_wc_full[chunk_start:chunk_start + T_scene] = matrix_world.astype(np.float32)

        extra_dict = {
            "images": images_for_dino,
            "pixel_uv_normalized": pixel_uv_normalized,
            "pixel_view_idx": pixel_view_idx,
            "pixel_query_mask": pixel_query_mask,
            "viz_obj_pcl": np.zeros((total_P, 1, 6), dtype=np.float32),
            "rgbs_paths": [osp.join(self.frames_root, sid, "frames")],
            "rgbs_views": ["head"],
            "rgbs_indices": rgbs_indices_original,
            "_T_align": T_align,
            "_cam_T_wc": cam_T_wc,
            "_K_resized": K_anchor,
            "_img_resize_wh": np.array([target_W, target_H], dtype=np.int32),
            "_img_orig_wh": np.array([img_W, img_H], dtype=np.int32),
            "_scale_xy": np.array([scale_x, scale_y], dtype=np.float32),
            "_cam_K_all": K_full,                                        # (original_T, 3, 3) padded
            "_cam_T_wc_all": T_wc_full,                                  # (original_T, 4, 4) padded
            "_frame_indices": np.array(rgbs_indices_original, dtype=np.int64),  # scene-level
            "_anchor_frame": int(scene_anchor),  # scene-level for PNG path
            "_anchor_ith": current_ith,
            "_data_root": self.frames_root,  # where PNGs actually live
            "_seq_id": sid,                    # original scene id, not shard_key
            "_scene_scale": np.float32(scene_scale),
            "_depth_range": depth_range.astype(np.float32),  # (2,) near/far — NEW
        }

        return ret, extra_dict
