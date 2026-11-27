"""TraceGen (TG) dataset for WMM — Droid + EpicKitchen 3D point-tracking.

Emits per-sample dicts matching the schema pinned in the dispatch
. Absolute-position training (Q9): trajectory positions
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
exactly matches authors' ``trajectory_mask`` at the metric site so 
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
| ``movement_bool``| ``(400,)`` bool                   |
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
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
        # (val = 1% of train).
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

        # ── Load image (PNG per finding), get orig size ──────────
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
            "movement_bool": movement_bool,                # (400,) bool
            "valid_steps": valid_steps,                    # () int64
            "depth_raw": depth_tensor,                     # (H,W) float32 or None
            "env_name": self.env_name,                     # str
            "episode_id": ep_dir.name,                     # str
            "frame_id": stem,                              # str
        }
