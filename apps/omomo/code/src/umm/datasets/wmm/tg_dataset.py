"""TraceGen (TG) dataset for WMM — Droid + EpicKitchen 3D point-tracking.

Emits per-sample dicts matching the schema pinned in the dispatch
( 20260422-0257). Absolute-position training (Q9): trajectory positions
are stored as absolute SE(3) with identity rotation (Q10, `w_R=0`).

Pipeline mirrors authors' ``data_wmm/tracegen/repo/dataio/datasets.py`` with
byte-copied helpers (``fill_traj_with_last_valid``, ``compute_movement_bool``,
``_prepare_gt_tensors``) and the same ordering:

    raw traj (400, 128, 3, with -inf padding)
        │
        ▼  trajectory_mask = (traj != -inf)  # (400, 128, 3) per-coord
        ▼  fill_traj_with_last_valid(traj)   # per-kp last valid carry-forward
        ▼  compute_movement_bool(post-fill traj, pre-fill mask, orig_size)
        ▼  resize 384 + keypoint/traj x,y rescale (z unchanged, 3D env)
        ▼  normalize x,y to [0,1] via /384 (= /orig after rescale)
        ▼  _prepare_gt_tensors (K→400, T→33, per-coord mask)
        ▼  SE(3) build: R = I_3, t = abs_xyz (absolute, no delta conversion)

The per-coord ``trace_mask`` shape ``(400, 33, 3)`` is LOAD-BEARING — it
exactly matches authors' ``trajectory_mask`` at the metric site so 's
bit-identical metric computation lands on the same tensors.

Schema (authoritative for Wave-3 consumers):

+------------------+-----------------------------------+
| key              | shape / dtype                     |
+==================+===================================+
| ``image``        | ``(1, 3, 384, 384)`` float32      |
| ``text``         | ``str``                           |
| ``trace``        | ``(400, 33, 4, 4)`` float32 SE(3) |
| ``trace_mask``   | ``(400, 33, 3)`` bool             |
| ``first_keypoint`` | ``(400, 2)`` float32 [0,1]      |
| ``gt_mask``      | ``(400,)`` bool                   |
| ``movement_bool``| ``(num_kps,)`` bool               |
| ``valid_steps``  | ``()`` int64                      |
| ``depth_raw``    | ``(H, W)`` float32 or ``None``    |
| ``env_name``     | ``str``                           |
| ``episode_id``   | ``str``                           |
| ``frame_id``     | ``str``                           |
+------------------+-----------------------------------+

Author attribution: helpers copied from
``data_wmm/tracegen/repo/dataio/datasets.py`` (Jung, Yoon-Kyo et al.,
TraceGen CVPR 2026). No structural reuse outside the byte-copied helpers.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as _ScipyRotation
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


DEFAULT_TEXT = "something movable"


@dataclass(frozen=True)
class _EnvLayout:
    root_subdir: str
    train_splits: Tuple[str, ...]
    test_splits: Tuple[str, ...]


_ENV_LAYOUT: Dict[str, _EnvLayout] = {
    "droid": _EnvLayout(
        root_subdir="Droid",
        train_splits=("droid",),
        test_splits=("droid_test",),
    ),
    "epic": _EnvLayout(
        root_subdir="Epickitchen",
        train_splits=("epic_1", "epic_2", "epic_3"),
        test_splits=("epic_test",),
    ),
}


def _fill_traj_with_last_valid(trajectories: np.ndarray) -> np.ndarray:
    # Adapted from data_wmm/tracegen/repo/dataio/datasets.py:628-640
    # (TraceGen authors). Per-keypoint: detect last finite first-coord index
    # (via pre-fill mask computed on the INPUT) and repeat that entry for
    # all subsequent time steps. Writes into the local ``filled`` copy so
    # the input ``trajectories`` is NOT mutated — authors' version has a
    # latent mutation side-effect we fix here; behavior of the return value
    # is identical.
    filled = trajectories.copy()
    K = trajectories.shape[0]

    for k in range(K):
        traj = filled[k]  # view into the copy — no mutation of the input
        mask = np.isfinite(traj[:, 0])  # first coordinate based on valid check
        if not mask.any():
            continue  # skip if all -inf
        last_idx = np.where(mask)[0][-1]  # last valid index
        traj[last_idx + 1:] = traj[last_idx]  # fill with last value
    return filled


def _compute_movement_bool(
    trajectories,
    trajectory_mask,
    orig_size,
    threshold: float = 0.1,
):
    # Copied from data_wmm/tracegen/repo/dataio/datasets.py:424-442
    # (TraceGen authors). Movement = sum of absolute per-step deltas on
    # valid steps, divided by image dimensions (normalized deltas).
    # trajectories: (..., T, 2 or 3)  numpy or torch
    # trajectory_mask: (..., T, 2 or 3)  numpy or torch (bool or 0/1)
    # orig_size: (W, H)
    traj_t = torch.as_tensor(trajectories, dtype=torch.float32)
    mask_t = torch.as_tensor(trajectory_mask, dtype=torch.float32)

    W, H = float(orig_size[0]), float(orig_size[1])

    dx = (traj_t[:, 1:, 0] / W - traj_t[:, :-1, 0] / W) * mask_t[:, 1:, 0]
    dy = (traj_t[:, 1:, 1] / H - traj_t[:, :-1, 1] / H) * mask_t[:, 1:, 1]

    movement_sum_x = dx.abs().sum(dim=1)           # shape: [batch]
    movement_sum_y = dy.abs().sum(dim=1)           # shape: [batch]
    movement_bool = (movement_sum_x + movement_sum_y) > threshold  # bool tensor

    return movement_bool


def _prepare_gt_tensors(
    keypoints_normalized: np.ndarray,
    trajectory_normalized: np.ndarray,
    trajectory_mask: np.ndarray,
    num_kps: int,
    trajectory_horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Adapted from data_wmm/tracegen/repo/dataio/datasets.py:642-683
    # (TraceGen authors). Pad keypoint count to num_kps and trajectory
    # length to trajectory_horizon+1. Returns per-coord mask with trailing
    # axis size 3 (load-bearing for metric bit-identity).
    num_keypoints = len(keypoints_normalized)

    gt_xy = torch.zeros(num_kps, 2, dtype=torch.float32)
    gt_mask = torch.zeros(num_kps, dtype=torch.bool)
    gt_trajectory = torch.zeros(num_kps, trajectory_horizon + 1, 3, dtype=torch.float32)
    gt_trajectory_mask = torch.zeros(num_kps, trajectory_horizon + 1, 3, dtype=torch.bool)

    num_to_fill = min(num_keypoints, num_kps)
    max_seq_length = min(trajectory_normalized.shape[1], trajectory_horizon + 1)
    if num_to_fill > 0:
        gt_xy[:num_to_fill] = torch.from_numpy(keypoints_normalized[:num_to_fill])
        gt_mask[:num_to_fill] = True
        gt_trajectory[:num_to_fill, :max_seq_length] = torch.from_numpy(
            trajectory_normalized[:num_to_fill, :max_seq_length]
        )
        gt_trajectory_mask[:num_to_fill, :max_seq_length] = torch.from_numpy(
            trajectory_mask[:num_to_fill, :max_seq_length]
        )
    return gt_xy, gt_mask, gt_trajectory, gt_trajectory_mask


class TracegenDataset(Dataset):
    """TraceGen dataset emitting UMM-WMM per-sample dicts.

    Args:
        data_root: Root containing ``Droid/`` and ``Epickitchen/`` subtrees.
        env_name: ``"droid"`` or ``"epic"``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        trajectory_horizon: Number of future frames (authors default 32).
            Output trace has ``trajectory_horizon + 1`` timesteps.
        num_kps: Keypoint pad count (authors default 400).
        image_size: Resize target (authors default 384).
        val_split: Fraction of train episodes held out for val (seeded).
        random_seed: Seed for train/val episode shuffle (authors default 42).
        max_episodes: Optional cap for dev; ``None`` to use all.
        safe_flag: If True, silently skip to next sample on load error.
        base_fps: Native frame rate used to convert per-step indices into
            seconds for the LMM time axis. Mirrors LangTable / Bridge / DAVIS
            (wmm/real_dataset.py:683). Default 30.0.
        t_rate: Temporal stride. Default 1.
        dummy_id_rot: When True, treat every track as having a "valid"
            identity rotation (placeholder). Downstream effect: the
            ``_TGAdapterDataset`` emits ``trace_rot_mask = ones(P, bool)``
            instead of zeros, so rotation channels enter the 9D
            representation, get noised + denoised by diffusion, and are
            supervised against identity. The trace already stores R=I_3 in
            the SE(3) tensor, so no value change is needed here. Default
            False = legacy behavior (rot_mask all-zero, rotation channels
            zeroed in 9D). When ``localframe_root`` is set, this flag is
            ignored — per-track rot_mask is driven by sidecar validity.
        localframe_root: w1.6.0 — Optional path to a sidecar tree of
            precomputed local-frame rotations (see
            ``prepare_data/tracegen/compute_local_frames.py``). When
            non-None, rotations are loaded per sample, re-anchored so the
            current frame (t=0) is identity, and written into the trace
            SE(3) matrix. Per-frame rot_valid is folded into
            ``trace_mask`` (α-mode: invalid-rot frames become fully
            invisible, losing position supervision at those frames too —
            mirrors ``real_dataset.py:449``). Per-track validity becomes
            the ``trace_rot_mask`` in the adapter. Default None = legacy
            w1.5.0 behavior (no localframe loaded).
    """

    def __init__(
        self,
        data_root: str,
        env_name: str,
        split: str,
        trajectory_horizon: int = 32,
        num_kps: int = 400,
        image_size: int = 384,
        val_split: float = 0.01,
        random_seed: int = 42,
        max_episodes: Optional[int] = None,
        safe_flag: bool = True,
        base_fps: float = 30.0,
        t_rate: int = 1,
        dummy_id_rot: bool = False,
        localframe_root: Optional[str] = None,
    ):
        if env_name not in _ENV_LAYOUT:
            raise ValueError(
                f"env_name must be one of {list(_ENV_LAYOUT.keys())}, got {env_name!r}"
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.data_root = Path(data_root)
        self.env_name = env_name
        self.split = split
        self.trajectory_horizon = int(trajectory_horizon)
        self.num_kps = int(num_kps)
        self.image_size = int(image_size)
        self.val_split = float(val_split)
        self.random_seed = int(random_seed)
        self.safe_flag = bool(safe_flag)
        self.base_fps = float(base_fps)
        self.t_rate = int(t_rate)
        self.dummy_id_rot = bool(dummy_id_rot)
        self.localframe_root: Optional[Path] = (
            Path(localframe_root) if localframe_root else None
        )
        # Per-instance miss tracking (w1.6.0). Reset per process; not shared
        # across DataLoader workers (each worker has its own copy via the
        # worker_init_fn — torch.utils.data spawn semantics).
        self._lf_missed_dirs: Set[str] = set()
        self._lf_miss_count: int = 0
        self._lf_load_count: int = 0

        layout = _ENV_LAYOUT[env_name]
        env_root = self.data_root / layout.root_subdir
        if not env_root.exists():
            raise FileNotFoundError(f"env root not found: {env_root}")

        if split == "test":
            split_dirs = [env_root / s for s in layout.test_splits]
        else:
            split_dirs = [env_root / s for s in layout.train_splits]

        for sd in split_dirs:
            if not sd.exists():
                raise FileNotFoundError(f"split dir not found: {sd}")

        # Collect episodes, shuffle once with a stable seed, then split
        # 1% to val. Matches authors' 10% default behavior but tighter
        # (per Q14: UMM val = 1% of train).
        all_eps: List[Path] = []
        for sd in split_dirs:
            eps = sorted([p for p in sd.iterdir() if p.is_dir()])
            all_eps.extend(eps)

        if max_episodes is not None:
            all_eps = all_eps[:max_episodes]

        if split == "test":
            self.episodes = all_eps
        else:
            rng = random.Random(self.random_seed)
            shuffled = list(all_eps)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * self.val_split))
            if split == "train":
                self.episodes = shuffled[:-n_val]
            else:  # val
                self.episodes = shuffled[-n_val:]

        # Episode metadata: (episode_dir, stem, img_path, npz_path, depth_path|None)
        self.samples: List[Tuple[Path, str, Path, Path, Optional[Path]]] = []
        self._episode_text_cache: Dict[str, List[str]] = {}
        for ep_dir in self.episodes:
            images_dir = ep_dir / "images"
            samples_dir = ep_dir / "samples"
            depth_dir = ep_dir / "depth"
            if not images_dir.exists() or not samples_dir.exists():
                logger.warning(f"episode missing images/ or samples/: {ep_dir}")
                continue
            for img_path in sorted(images_dir.glob("*.png")):
                stem = img_path.stem
                npz_path = samples_dir / f"{stem}.npz"
                if not npz_path.exists():
                    continue
                depth_path: Optional[Path] = None
                if depth_dir.exists():
                    cand = depth_dir / f"{stem}_raw.npz"
                    if cand.exists():
                        depth_path = cand
                self.samples.append(
                    (ep_dir, stem, img_path, npz_path, depth_path)
                )
            self._episode_text_cache[ep_dir.name] = self._load_instructions(ep_dir)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"TracegenDataset({env_name}, {split}): no samples "
                f"found under {split_dirs}"
            )

        logger.info(
            f"TracegenDataset env={env_name} split={split}: "
            f"{len(self.episodes)} episodes, {len(self.samples)} samples"
        )

        # w1.6.0: init-time coverage probe. Per user instruction the dataset
        # is "robust to missing localframe file" (per-sample fallback to
        # identity rot, rot_mask=False), so partial coverage is allowed by
        # design. We still surface the coverage prominently:
        #   * coverage < 50%  → ValueError: almost certainly a misconfig
        #     (wrong path, sidecar tree pointed at an empty dir, etc.)
        #   * coverage in [50%, 99%) → WARNING with the percentage and a few
        #     example missing paths. Training proceeds with mixed distribution
        #     (rot-supervised tracks for samples with sidecar; legacy identity
        #     for samples without) — exactly what the user asked for.
        #   * coverage >= 99% → INFO.
        if self.localframe_root is not None:
            n_probe = min(500, len(self.samples))
            rng = np.random.default_rng(0)
            probe_idx = rng.choice(
                len(self.samples), size=n_probe, replace=False
            )
            hits = 0
            misses: List[Path] = []
            for i in probe_idx:
                _, _, _, npz_path, _ = self.samples[int(i)]
                lf = self._resolve_localframe_path(npz_path)
                if lf.exists():
                    hits += 1
                else:
                    misses.append(lf)
            coverage = hits / n_probe
            if coverage < 0.50:
                example = "\n  ".join(str(p) for p in misses[:5])
                raise ValueError(
                    f"TracegenDataset({env_name}, {split}): localframe "
                    f"coverage {coverage:.1%} ({hits}/{n_probe}) is below "
                    f"the 50% sanity floor — likely a misconfigured "
                    f"localframe_root ({self.localframe_root}). "
                    f"Example missing paths:\n  {example}"
                )
            if coverage < 0.99:
                example = "\n  ".join(str(p) for p in misses[:3])
                logger.warning(
                    f"TracegenDataset({env_name}, {split}): localframe "
                    f"coverage {coverage:.1%} ({hits}/{n_probe} probed). "
                    f"Samples without a sidecar will train as identity-rot "
                    f"(legacy w1.5.0 semantics) — mixed distribution. "
                    f"Example missing paths:\n  {example}"
                )
            else:
                logger.info(
                    f"TracegenDataset env={env_name} split={split}: "
                    f"localframe coverage {coverage:.1%} "
                    f"({hits}/{n_probe} probed)"
                )

    # ── Localframe sidecar helpers (w1.6.0) ────────────────────────────

    def _resolve_localframe_path(self, npz_path: Path) -> Path:
        """Map source npz path to the localframe sidecar path.

        Source layout: ``<data_root>/<env>/<split>/<episode>/samples/<stem>.npz``
        Sidecar:       ``<localframe_root>/<env>/<split>/<episode>/<stem>.npz``
        (matches ``compute_local_frames.py:_mirror_path`` — only the immediate
        ``samples`` parent is collapsed.)
        """
        assert self.localframe_root is not None
        rel = npz_path.relative_to(self.data_root)
        parts = list(rel.parts)
        if not (len(parts) >= 2 and parts[-2] == "samples"):
            # Layout violation — fail loud rather than fall through to a
            # silently-failing .exists() check on a malformed path.
            raise ValueError(
                f"TracegenDataset: source npz_path={npz_path} does not have "
                f"'samples' as immediate parent (parts={parts}); cannot "
                f"resolve sidecar path."
            )
        parts = parts[:-2] + parts[-1:]
        return self.localframe_root.joinpath(*parts)

    def _load_and_align_localframe(
        self,
        lf_path: Path,
        perm: Optional[np.ndarray],
        K_raw: int,
        T_target: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a localframe sidecar, slice/pad, re-anchor at t=0.

        Args:
            lf_path: sidecar npz path (must exist; caller checks).
            perm: K-permutation applied to the source traj when
                K_raw > num_kps (else None). Same perm is applied to
                rotations + rot_valid here.
            K_raw: keypoint count in the source npz BEFORE permutation.
                Used for the post-perm assertion.
            T_target: trajectory_horizon + 1.

        Returns:
            R_aligned: (num_kps, T_target, 3, 3) float32 — re-anchored
                rotations. Tracks with track_rot_valid=False are clamped
                to identity at every time step. All tracks have
                R_aligned[:, 0] = I (load-bearing invariant).
            rv_PT: (num_kps, T_target) bool — per-frame validity, padded
                to num_kps with False on pad rows.
            track_rot_valid: (num_kps,) bool — per-track validity =
                rv_PT[:, 0].
        """
        with np.load(lf_path, allow_pickle=False) as d:
            aa = d["rotations"].astype(np.float32).copy()  # (K_lf, T_lf, 3)
            rv = d["rot_valid"].astype(bool).copy()        # (K_lf, T_lf)

        # K dimension must match the source npz before any permutation.
        if aa.shape[0] != K_raw or rv.shape[0] != K_raw:
            raise ValueError(
                f"TracegenDataset: localframe K mismatch at {lf_path}: "
                f"sidecar K_raw={aa.shape[0]} vs source K_raw={K_raw}. "
                f"Sidecar tree is out of sync with source data."
            )

        # Apply same K-permutation as the source traj.
        if perm is not None:
            aa = aa[perm]
            rv = rv[perm]

        # Slice T to target. If sidecar T is shorter than target, pad.
        T_have = aa.shape[1]
        if T_have >= T_target:
            aa = aa[:, :T_target]
            rv = rv[:, :T_target]
        else:
            pad_T = T_target - T_have
            aa = np.concatenate(
                [aa, np.zeros((aa.shape[0], pad_T, 3), dtype=np.float32)],
                axis=1,
            )
            rv = np.concatenate(
                [rv, np.zeros((rv.shape[0], pad_T), dtype=bool)],
                axis=1,
            )

        # Pad K to num_kps (pad rows: zero aa = identity, rv = False).
        K_cur = aa.shape[0]
        if K_cur < self.num_kps:
            pad_K = self.num_kps - K_cur
            aa = np.concatenate(
                [aa, np.zeros((pad_K, T_target, 3), dtype=np.float32)],
                axis=0,
            )
            rv = np.concatenate(
                [rv, np.zeros((pad_K, T_target), dtype=bool)],
                axis=0,
            )
        elif K_cur > self.num_kps:
            # Should never happen — perm logic should have already cut to
            # num_kps. Defensive truncate.
            aa = aa[: self.num_kps]
            rv = rv[: self.num_kps]

        # Convert axis-angle (K, T, 3) -> rotation matrix (K, T, 3, 3).
        # Invalid frames have aa = 0 → R = I (compute_local_frames invariant).
        aa_flat = aa.reshape(-1, 3)
        R_flat = _ScipyRotation.from_rotvec(aa_flat).as_matrix().astype(np.float32)
        R = R_flat.reshape(self.num_kps, T_target, 3, 3)

        track_rot_valid = rv[:, 0].copy()  # (num_kps,) bool

        # Re-anchor: R_aligned[k, t] = R[k, t] @ R[k, 0].T
        # einsum index map:
        #   first operand  R     has indices (k, t, i, j)  — shape (K, T, 3, 3)
        #   second operand R[:,0] has indices (k, m, j)    — shape (K, 3, 3)
        #     (we label its second axis 'm' and third axis 'j' to express
        #      transpose-of-second-operand inside the contraction)
        #   output: out[k, t, i, m] = sum_j R[k, t, i, j] * R[k, 0, m, j]
        #         = (R[k, t] @ R[k, 0].T)[i, m]
        R_aligned = np.einsum("ktij,kmj->ktim", R, R[:, 0])

        # Invariant clamps:
        #  (a) tracks with rot_valid[k, 0]=False → entire time slice = I
        invalid_tracks = ~track_rot_valid
        if invalid_tracks.any():
            R_aligned[invalid_tracks] = np.eye(3, dtype=np.float32)
        #  (b) every track's t=0 = I (defense in depth; einsum already gives
        #      this for valid tracks, identity overwrite for invalid)
        R_aligned[:, 0] = np.eye(3, dtype=np.float32)

        # Load-bearing invariant for "context = identity always".
        if not np.allclose(R_aligned[:, 0], np.eye(3), atol=1e-5):
            raise RuntimeError(
                "TracegenDataset: re-anchor invariant R_aligned[:, 0] == I "
                "violated; this should be impossible."
            )

        return R_aligned, rv, track_rot_valid

    @staticmethod
    def _load_instructions(episode_dir: Path) -> List[str]:
        text_file = episode_dir / "three_instructions.json"
        if not text_file.exists():
            return []
        try:
            with open(text_file, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"failed to parse {text_file}: {e}")
            return []
        out: List[str] = []
        for key in ("instruction_1", "instruction_2", "instruction_3"):
            if key in data:
                v = str(data[key]).strip()
                if v:
                    out.append(v)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    # ── Main getitem ──────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.safe_flag:
            # bounded loop avoids infinite recursion if dataset is totally broken
            n = len(self.samples)
            for _ in range(n):
                try:
                    return self._getitem_inner(idx)
                except Exception as e:  # noqa: BLE001 — error logged + advanced
                    logger.warning(
                        f"TracegenDataset: failed idx={idx}: {e!r}; advancing"
                    )
                    idx = (idx + 1) % n
            raise RuntimeError("TracegenDataset: all samples failed to load")
        return self._getitem_inner(idx)

    def _getitem_inner(self, idx: int) -> Dict[str, Any]:
        ep_dir, stem, img_path, npz_path, depth_path = self.samples[idx]

        # ── Load image, get orig size ──────────
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            orig_w, orig_h = pil_img.size  # PIL (W, H)
            resized = pil_img.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            image_np = np.asarray(resized, dtype=np.uint8)  # (H,W,3)

        # Float [0,1], (1, 3, 384, 384). DINO encoder internally applies
        # ImageNet mean/std, so we do NOT normalize here (matches real_dataset).
        image_tensor = torch.from_numpy(
            image_np.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0).contiguous()  # (1, 3, 384, 384)

        # ── Load npz: keypoints, trajectory (raw absolute), valid_steps ──
        # Context-managed load: release mmap handle before we leave the
        # function so heavy multi-worker DataLoader usage does not leak fds.
        with np.load(npz_path) as npz:
            kp_raw = npz["keypoints"].astype(np.float32).copy()  # (400, 2) pixel
            traj_raw = npz["traj"].astype(np.float32).copy()      # (400, 128, 3 or 2)
            valid_steps = torch.as_tensor(
                int(npz["valid_steps"].sum()), dtype=torch.int64
            )

        # trajectory_mask from RAW trajectory BEFORE fill — matches authors
        # (trajectories != -inf → per-coord bool). Shape (400, 128, 3).
        trajectory_mask = (traj_raw != -np.inf)

        # Fill -inf with last valid value per keypoint.
        traj_filled = _fill_traj_with_last_valid(traj_raw)

        # movement_bool uses POST-FILL traj + PRE-FILL mask (byte-match to
        # authors' datasets.py:545-547). Passing pre-fill traj here would
        # produce -inf * 0 = NaN and silently collapse movement_bool to all
        # False for partially-observed keypoints.
        movement_bool = _compute_movement_bool(
            traj_filled, trajectory_mask, (orig_w, orig_h), threshold=0.1
        ).bool()

        # Random sub-sample of keypoints when num_keypoints > num_kps.
        # Earlier behavior took the FIRST num_kps raw keypoints, which biased
        # K_train < 400 viz frames to the upper portion of the image (raw
        # keypoint ordering follows TraceGen's deterministic sampling pattern,
        # heavy in the upper image rows). User decision (qa.md Q12): randomly
        # sample num_kps from the available num_keypoints. The SAME permutation
        # is applied to ALL keypoint-keyed fields per sample so they stay
        # aligned: kp_raw, traj_filled, trajectory_mask, movement_bool. Once
        # subsampled, K_raw == self.num_kps and the subsequent alignment +
        # _prepare_gt_tensors reduce to identity / no-op for these fields.
        # K=400 case (num_keypoints == num_kps) is byte-identical: no permute.
        K_raw = int(kp_raw.shape[0])
        perm: Optional[np.ndarray] = None
        if K_raw > self.num_kps:
            # Per-call non-deterministic RNG (per-epoch randomness is an
            # accepted training-time signal; reproducibility not required
            # per qa.md Q12 / dispatch).
            perm = np.random.default_rng().permutation(K_raw)[: self.num_kps]
            kp_raw = kp_raw[perm]
            traj_filled = traj_filled[perm]
            trajectory_mask = trajectory_mask[perm]
            movement_bool = movement_bool[torch.from_numpy(perm).long()]

        # Align movement_bool to num_kps (post-subsample). After the optional
        # permutation above movement_bool.shape[0] is min(K_raw, num_kps), so
        # this is identity for K=128 (subsampled) and slice [:400] of a
        # (400,) for K=400 — both byte-identical to the pre-K=400 path.
        K_curr = int(movement_bool.shape[0])
        n_take = min(K_curr, self.num_kps)
        mb_aligned = torch.zeros(self.num_kps, dtype=torch.bool)
        mb_aligned[:n_take] = movement_bool[:n_take]
        movement_bool = mb_aligned

        # ── Rescale keypoints + traj x,y to 384 pixel space, then [0,1] ──
        target = float(self.image_size)
        scale_x = target / float(orig_w)
        scale_y = target / float(orig_h)

        kp_norm = kp_raw.copy()
        kp_norm[:, 0] *= scale_x
        kp_norm[:, 1] *= scale_y
        # normalize to [0,1] by /target and clamp (matches authors' NormalizeTransform)
        kp_norm[:, 0] /= target
        kp_norm[:, 1] /= target
        kp_norm = np.clip(kp_norm, 0.0, 1.0)

        traj_norm = traj_filled.copy()
        traj_norm[:, :, 0] *= scale_x
        traj_norm[:, :, 1] *= scale_y
        traj_norm[:, :, 0] /= target
        traj_norm[:, :, 1] /= target
        # Only clamp x,y to [0,1] — z is raw meters (3D env), unclamped.
        traj_norm[:, :, 0] = np.clip(traj_norm[:, :, 0], 0.0, 1.0)
        traj_norm[:, :, 1] = np.clip(traj_norm[:, :, 1], 0.0, 1.0)

        # Sanity: if source trajectory was 2D (not expected for Droid/Epic),
        # pad a z=0 channel + False mask for the z axis (defensive).
        if traj_norm.shape[-1] == 2:
            traj_norm = np.concatenate(
                [traj_norm, np.zeros((traj_norm.shape[0], traj_norm.shape[1], 1),
                                     dtype=np.float32)],
                axis=-1,
            )
            trajectory_mask = np.concatenate(
                [trajectory_mask, np.zeros((trajectory_mask.shape[0],
                                            trajectory_mask.shape[1], 1),
                                           dtype=bool)],
                axis=-1,
            )

        # ── Pad K→num_kps, cut T→horizon+1, produce per-coord mask ───
        gt_xy, gt_mask, gt_trajectory, gt_trajectory_mask = _prepare_gt_tensors(
            kp_norm,
            traj_norm.astype(np.float32),
            trajectory_mask,
            num_kps=self.num_kps,
            trajectory_horizon=self.trajectory_horizon,
        )
        # Shapes: gt_xy (400,2), gt_mask (400,), gt_trajectory (400,33,3),
        # gt_trajectory_mask (400,33,3) — per-coord mask, load-bearing.

        # ── Build SE(3) trace: R = I_3, t = abs_xyz (absolute, Q9) ───
        T = self.trajectory_horizon + 1
        trace = torch.zeros(self.num_kps, T, 4, 4, dtype=torch.float32)
        eye3 = torch.eye(3, dtype=torch.float32)
        trace[:, :, :3, :3] = eye3.view(1, 1, 3, 3).expand(self.num_kps, T, 3, 3)
        trace[:, :, :3, 3] = gt_trajectory            # absolute xyz
        trace[:, :, 3, 3] = 1.0                        # SE(3) homogeneous

        # ── w1.6.0: localframe rotation overlay ────────────────────────
        # Default: track_rot_valid all-False (legacy semantics — equivalent
        # to dummy_id_rot=False at the adapter level).
        track_rot_valid = torch.zeros(self.num_kps, dtype=torch.bool)
        if self.localframe_root is not None:
            lf_path = self._resolve_localframe_path(npz_path)
            if lf_path.exists():
                try:
                    R_aligned, rv_PT, track_rv = self._load_and_align_localframe(
                        lf_path, perm=perm, K_raw=K_raw,
                        T_target=T,
                    )
                    trace[:, :, :3, :3] = torch.from_numpy(R_aligned)
                    track_rot_valid = torch.from_numpy(track_rv).bool()
                    # α: fold per-frame rot_valid into gt_trajectory_mask
                    # ONLY for tracks marked rot-valid. Tracks with
                    # track_rot_valid=False keep their position-coord
                    # trace_mask intact (no rot supervision; position
                    # supervision unchanged).
                    keep = (~track_rot_valid[:, None]) | torch.from_numpy(rv_PT)
                    gt_trajectory_mask = gt_trajectory_mask & keep[..., None]
                    self._lf_load_count += 1
                except Exception as e:  # noqa: BLE001 — logged + advanced
                    logger.warning(
                        f"TracegenDataset: localframe load failed for "
                        f"{lf_path}: {e!r}; falling back to identity rot "
                        f"for this sample"
                    )
                    self._lf_miss_count += 1
                    # leave trace at identity, track_rot_valid all-False
            else:
                # Missing sidecar — log warn-once per parent dir.
                parent = str(lf_path.parent)
                if parent not in self._lf_missed_dirs:
                    self._lf_missed_dirs.add(parent)
                    logger.warning(
                        f"TracegenDataset: missing localframe sidecar at "
                        f"{lf_path}; falling back to identity rot. Will not "
                        f"warn again for this dir."
                    )
                self._lf_miss_count += 1
        # Post-condition: trace[:, 0, :3, :3] is identity for every track,
        # because (a) branch not taken: trace[:, :, :3, :3] = I
        #         (b) branch taken, valid track: re-anchor gives R[:,0]=I
        #         (c) branch taken, invalid track: clamp in
        #             _load_and_align_localframe gave R = I

        # ── Text (random of 3 @train, instruction_1 @val/test, fallback) ──
        instructions = self._episode_text_cache.get(ep_dir.name, [])
        if len(instructions) == 0:
            text = DEFAULT_TEXT
        elif self.split == "train":
            text = random.choice(instructions)
        else:
            text = instructions[0]

        # ── Depth (raw, optional) ──────────────────────────────────────
        depth_tensor: Optional[torch.Tensor] = None
        if depth_path is not None:
            try:
                with np.load(depth_path) as dp:
                    depth_tensor = torch.from_numpy(
                        dp["depth"].astype(np.float32)
                    )  # (H, W)
            except (OSError, KeyError) as e:
                logger.warning(f"depth load failed {depth_path}: {e}")
                depth_tensor = None

        return {
            "image": image_tensor,                         # (1,3,384,384)
            "text": text,                                  # str
            "trace": trace,                                # (400,33,4,4)
            "trace_mask": gt_trajectory_mask,              # (400,33,3) bool
            "first_keypoint": gt_xy,                       # (400,2) float32
            "gt_mask": gt_mask,                            # (400,) bool
            "movement_bool": movement_bool,                # (num_kps,) bool
            "valid_steps": valid_steps,                    # () int64
            "depth_raw": depth_tensor,                     # (H,W) float32 or None
            "track_rot_valid": track_rot_valid,            # (num_kps,) bool — w1.6.0
            "env_name": self.env_name,                     # str
            "episode_id": ep_dir.name,                     # str
            "frame_id": stem,                              # str
        }
