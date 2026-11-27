"""Train the hand-object action classifier on a Text2HOI dataset (GRAB or ARCTIC).

Usage (illustrative — launched by the caller, script does not auto-run):
    python -m umm.eval.handobject.train_classifier \
        --dataset grab \
        --data_root <path> \
        --bin_map <path> \
        --labels_json .../assets/action_labels.json \
        --out_dir data/handobject/imos_ckpts/classifier/grab \
        --epochs 100 --batch_size 64 --lr 1e-3 --max_t 120 --seed 42 \
        --split 0.8,0.1,0.1 --device cuda

Pipeline:
    1. Read labels JSON, build vocab (29 GRAB / 11 ARCTIC).
    2. Iterate all GT episodes via `iter_gt_episodes` -> CanonicalSample.
    3. For each sample:
        - Build (T, 43, 3) points via `canonical_to_points` on valid frames only.
        - Pad/sub-sample to MAX_T via `pad_or_subsample`.
        - Map text -> class index via `text_to_label_idx`. Fail loud (do not drop)
          if the verb is unexpectedly missing from the vocab.
    4. Deterministic random train/val/test split (seed=42).
    5. AdamW (no weight decay) + CosineAnnealingLR.
    6. CE loss. Log top-1 and top-3 val accuracy per epoch.
    7. Save best.pth (best val top-1), last.pth, config.json, split_indices.json,
       val_acc_curve.png.

IMoS uses multilabel sigmoid+BCE (action_rec.py:81); we use softmax+CE because
every hand-object episode has exactly one verb.

GPU rules:
    - The script prints os.environ['CUDA_VISIBLE_DEVICES'] BEFORE importing torch
      (enforced at module import: see the top of this file — if you import
      this module in a notebook, set CUDA_VISIBLE_DEVICES first).
    - `--device` selects cpu vs cuda explicitly; no hardcoded device.
    - No torch.compile, no gradient checkpointing, no fast_dev_run hacks.

References:
    - IMoS action_rec.py:14-82 (architecture being ported).
    - canonical.py (sample schema).
    - gt_loader.py (iter_gt_episodes).
    - assets/action_labels.json (frozen vocab).
"""

# --- GPU visibility echo BEFORE any torch import (per UMM GPU rules) ---
import os as _os_early
import sys as _sys_early

_cvd_at_import = _os_early.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
print(f"[train_classifier] CUDA_VISIBLE_DEVICES={_cvd_at_import}", file=_sys_early.stderr)

import argparse
import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: must be set before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

from umm.eval.handobject.action_label import (
    load_action_labels,
    text_to_label_idx,
)
from umm.eval.handobject.classifier import HandObjectActionClassifier, Text2HOIActionClassifier
from umm.eval.handobject.gt_loader import iter_gt_episodes
from umm.eval.handobject.motion_feats import MAX_T, canonical_to_tensor, canonical_to_text2hoi_input


logger = logging.getLogger("train_classifier")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HandObjectClassifierDataset(Dataset):
    """In-memory dataset of (points, length, label_idx) tuples.

    Pre-extracts all episodes from the Text2HOI npz at construction time. Keeps
    everything in CPU numpy until `__getitem__`, which returns torch tensors.

    All samples share shape (max_t, 43, 3). `lengths` vary within [1, max_t].

    Fields:
        points_np:  (N, max_t, 43, 3) float32
        lengths_np: (N,) int64
        labels_np:  (N,) int64
    """

    def __init__(
        self,
        dataset_name: str,
        bin_map_path: str,
        npz_path: str,
        text_json_path: str,
        label_to_idx: Dict[str, int],
        max_t: int,
        max_episodes: Optional[int] = None,
    ) -> None:
        """Build the in-memory arrays.

        Args:
            dataset_name: "grab" or "arctic".
            bin_map_path: Path to bin_map_{grab,arctic}.json.
            npz_path:     Path to Text2HOI data.npz.
            text_json_path: Path to Text2HOI text.json.
            label_to_idx:  Verb -> class index.
            max_t:         Fixed sequence length.
            max_episodes:  Optional cap for debug.

        Raises:
            KeyError: If any episode's verb is missing from the vocab. The
                action_labels.json is built from the full dataset; a missing
                verb indicates a vocabulary/data mismatch and must fail loud.
        """
        super().__init__()
        pts_list: List[np.ndarray] = []                         # each (max_t, 43, 3)
        len_list: List[int] = []                                # scalar int
        lbl_list: List[int] = []                                # scalar int

        iterator = iter_gt_episodes(
            dataset=dataset_name,
            bin_map_path=bin_map_path,
            npz_path=npz_path,
            text_json_path=text_json_path,
            max_episodes=max_episodes,
        )
        for i, sample in enumerate(iterator):
            pts, length = canonical_to_tensor(sample, max_t=max_t)   # (max_t, 43, 3), int
            # Fail loud if verb missing — this is a vocab/data mismatch, not a skip condition.
            idx = text_to_label_idx(sample.text, label_to_idx)       # int
            pts_list.append(pts)
            len_list.append(length)
            lbl_list.append(idx)

        if len(pts_list) == 0:
            raise ValueError(
                f"HandObjectClassifierDataset: zero episodes loaded from {npz_path!r}"
            )

        self.points_np: np.ndarray = np.stack(pts_list, axis=0)          # (N, max_t, 43, 3)
        self.lengths_np: np.ndarray = np.asarray(len_list, dtype=np.int64)  # (N,)
        self.labels_np: np.ndarray = np.asarray(lbl_list, dtype=np.int64)   # (N,)
        self.max_t: int = int(max_t)

        logger.info(
            "Loaded %d episodes from %s; points %s, lengths range [%d, %d]",
            self.points_np.shape[0], npz_path, self.points_np.shape,
            int(self.lengths_np.min()), int(self.lengths_np.max()),
        )

    def __len__(self) -> int:
        return int(self.points_np.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "points":  torch.from_numpy(self.points_np[idx]),   # (max_t, 43, 3) float32
            "length":  torch.tensor(int(self.lengths_np[idx]), dtype=torch.int64),  # ()
            "label":   torch.tensor(int(self.labels_np[idx]),   dtype=torch.int64),  # ()
        }


class DataAugDataset(Dataset):
    """Wraps HandObjectClassifierDataset and applies on-the-fly augmentation.

    Augmentations applied per __getitem__ call (i.e., per-batch during training):
      1. Gaussian joint noise (sigma=0.003 m) on joints 0-41 (hand + obj translation).
         Joint indices follow canonical_to_tensor: [0..20] = left hand, [21..41] = right hand,
         [42] = obj translation. We apply noise to all 43 points for simplicity.
      2. Temporal flip: 50% probability, reverse frame order.
      3. Left-right hand swap: 50% probability, swap left (joints 0-20) and right (21-41)
         joint arrays. Only applied to verbs that are handedness-symmetric (see safe list below).

    Args:
        base_ds:      An iterable dataset (typically a torch.utils.data.Subset of
                      HandObjectClassifierDataset).
        label_to_idx: Verb -> class index mapping. Used to identify safe-swap verb indices.
        rng_seed:     Seed for the augmentation RNG.
        noise_sigma:  Gaussian noise std in metres. Default 0.003.
    """

    # Verbs where swapping left/right hand is semantically neutral.
    # Asymmetric verbs (e.g. "hand over", "pass") are NOT in this list.
    _SYMMETRIC_VERBS = {
        "grab", "grasp", "hold", "lift", "place", "put", "set", "pick", "take",
        "squeeze", "pour", "eat", "drink", "open", "close", "flip", "move",
        "carry", "drop", "release",
    }

    def __init__(
        self,
        base_ds: Dataset,
        label_to_idx: Dict[str, int],
        rng_seed: int = 42,
        noise_sigma: float = 0.003,
    ) -> None:
        super().__init__()
        self.base_ds = base_ds
        self.noise_sigma = float(noise_sigma)
        # Build a set of label indices where left-right swap is safe.
        self._symmetric_label_indices = {
            idx for verb, idx in label_to_idx.items()
            if any(sv in verb.lower() for sv in self._SYMMETRIC_VERBS)
        }
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.base_ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_ds[idx]
        points = item["points"].clone()          # (max_t, 43, 3) float32
        length = item["length"]                  # () int64
        label = item["label"]                    # () int64

        T, P, D = points.shape                   # (max_t, 43, 3)
        T_eff = int(length.item())               # actual valid frames

        # --- Augmentation 1: Gaussian noise on all 43 joints ---
        noise = torch.from_numpy(
            self._rng.normal(0.0, self.noise_sigma, size=(T, P, D)).astype(np.float32)
        )
        points = points + noise                  # (max_t, 43, 3)

        # --- Augmentation 2: Temporal flip (50%) ---
        if self._rng.random() < 0.5:
            # Flip only the valid frames; leave padding untouched.
            points[:T_eff] = points[:T_eff].flip(dims=[0])

        # --- Augmentation 3: Left-right hand swap (50%, symmetric verbs only) ---
        label_int = int(label.item())
        if (label_int in self._symmetric_label_indices) and (self._rng.random() < 0.5):
            # Joints 0-20 = left hand, 21-41 = right hand.
            left = points[:, :21, :].clone()
            right = points[:, 21:42, :].clone()
            points[:, :21, :] = right
            points[:, 21:42, :] = left

        return {"points": points, "length": length, "label": label}


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-sample dicts into a mini-batch.

    Args:
        items: List of dicts from HandObjectClassifierDataset.__getitem__.

    Returns:
        Dict with:
            points:  (B, max_t, 43, 3) float32
            lengths: (B,) int64
            labels:  (B,) int64
    """
    points = torch.stack([it["points"] for it in items], dim=0)         # (B, T, 43, 3)
    lengths = torch.stack([it["length"] for it in items], dim=0)        # (B,)
    labels = torch.stack([it["label"] for it in items], dim=0)          # (B,)
    return {"points": points, "lengths": lengths, "labels": labels}


class Text2HOIClassifierDataset(Dataset):
    """In-memory dataset of (points, label_idx) tuples for the Text2HOI arch.

    Parallel to HandObjectClassifierDataset but emits (15, 42, 3) tensors
    via canonical_to_text2hoi_input with deterministic=True (linspace frame
    selection). No lengths stored — Text2HOI classifier uses fixed num_frames=15.

    Decision: We always use deterministic=True (test/eval mode linspace sampling)
    for both train and val splits. Text2HOI uses random within-group sampling for
    training, but deterministic is simpler and still provides reasonable samples.
    Augmentation (via DataAugDataset) can be applied on top if desired.

    Fields:
        points_np: (N, 15, 42, 3) float32
        labels_np: (N,) int64
    """

    def __init__(
        self,
        dataset_name: str,
        bin_map_path: str,
        npz_path: str,
        text_json_path: str,
        label_to_idx: Dict[str, int],
        num_frames: int = 15,
        max_episodes: Optional[int] = None,
    ) -> None:
        """Build the in-memory arrays.

        Args:
            dataset_name:   "grab" or "arctic".
            bin_map_path:   Path to bin_map_{grab,arctic}.json.
            npz_path:       Path to Text2HOI data.npz.
            text_json_path: Path to Text2HOI text.json.
            label_to_idx:   Verb -> class index.
            num_frames:     Fixed sequence length (default 15).
            max_episodes:   Optional cap for debug.

        Raises:
            KeyError: If any episode's verb is missing from the vocab.
        """
        super().__init__()
        pts_list: List[np.ndarray] = []                              # each (15, 42, 3)
        lbl_list: List[int] = []                                     # scalar int

        iterator = iter_gt_episodes(
            dataset=dataset_name,
            bin_map_path=bin_map_path,
            npz_path=npz_path,
            text_json_path=text_json_path,
            max_episodes=max_episodes,
        )
        for sample in iterator:
            # deterministic=True: linspace frame selection (test/eval mode).
            pts = canonical_to_text2hoi_input(
                sample, num_frames=num_frames, deterministic=True
            )                                                        # (15, 42, 3) float32
            idx = text_to_label_idx(sample.text, label_to_idx)      # int
            pts_list.append(pts)
            lbl_list.append(idx)

        if len(pts_list) == 0:
            raise ValueError(
                f"Text2HOIClassifierDataset: zero episodes loaded from {npz_path!r}"
            )

        self.points_np: np.ndarray = np.stack(pts_list, axis=0)          # (N, 15, 42, 3)
        self.labels_np: np.ndarray = np.asarray(lbl_list, dtype=np.int64)  # (N,)
        self.num_frames: int = int(num_frames)

        logger.info(
            "Text2HOIClassifierDataset: loaded %d episodes from %s; points %s",
            self.points_np.shape[0], npz_path, self.points_np.shape,
        )

    def __len__(self) -> int:
        return int(self.points_np.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "points": torch.from_numpy(self.points_np[idx]),   # (15, 42, 3) float32
            "label":  torch.tensor(int(self.labels_np[idx]), dtype=torch.int64),  # ()
        }


def collate_batch_text2hoi(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-sample dicts into a mini-batch for the text2hoi42 arch.

    Args:
        items: List of dicts from Text2HOIClassifierDataset.__getitem__.

    Returns:
        Dict with:
            points: (B, 15, 42, 3) float32
            labels: (B,) int64
    """
    points = torch.stack([it["points"] for it in items], dim=0)     # (B, 15, 42, 3)
    labels = torch.stack([it["label"] for it in items], dim=0)      # (B,)
    return {"points": points, "labels": labels}


def _forward_model(
    model: "Any",
    batch: Dict[str, torch.Tensor],
    arch: str,
    device: "torch.device",
) -> Dict[str, torch.Tensor]:
    """Dispatch model forward based on arch, handling the points/lengths signature difference.

    Args:
        model:  HandObjectActionClassifier or Text2HOIActionClassifier.
        batch:  Collated batch dict (already moved to device by run_epoch).
        arch:   "imos43" or "text2hoi42".
        device: Torch device (unused here; batch is already on device).

    Returns:
        Model output dict with "logits" and "features".
    """
    points = batch["points"]
    if arch == "imos43":
        lengths = batch["lengths"]
        return model(points, lengths)
    elif arch == "text2hoi42":
        # lengths is not stored for text2hoi42; pass None (model ignores it).
        return model(points, None)
    else:
        raise ValueError(f"_forward_model: unknown arch {arch!r}")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def deterministic_split(
    n: int, fractions: Tuple[float, float, float], seed: int,
) -> Dict[str, List[int]]:
    """Shuffle [0..n) with the given seed and split by fractions.

    Args:
        n: Total number of items.
        fractions: (train, val, test) fractions; must sum to ~1.0.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys train/val/test -> List[int] of indices into the full set.
    """
    tr, va, te = fractions
    total = tr + va + te
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"deterministic_split: fractions must sum to 1.0, got {total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()                                  # List[int] len=n
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    # Absorb rounding residue into the test set to preserve the exact total.
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_va]
    test_idx = perm[n_tr + n_va:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Top-k accuracy for a batch, as a Python float.

    Args:
        logits: (B, n_labels) float.
        labels: (B,) int64.
        k: Top-k cutoff.

    Returns:
        Scalar float accuracy in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"topk_accuracy: k must be positive, got {k}")
    k_eff = min(k, logits.shape[-1])
    topk = logits.topk(k_eff, dim=-1).indices                            # (B, k_eff)
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1).float()         # (B,)
    return float(correct.mean().item())


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: "Any",
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[AdamW],
    criterion: nn.Module,
    arch: str = "imos43",
    n_labels: Optional[int] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Run a single epoch (training if optimizer is not None, else eval).

    Loss semantics follow the arch:
      - imos43    → softmax + CE on single-label targets.
      - text2hoi42 → sigmoid + BCE on one-hot multi-label targets, matching
                    IMoS `LSTM_Action_Classifier.forward` (action_rec.py:82).

    Args:
        model:     classifier module.
        loader:    DataLoader.
        device:    torch device.
        optimizer: AdamW for training; None for eval.
        criterion: CrossEntropyLoss (imos43) or BCELoss (text2hoi42).
        arch:      "imos43" or "text2hoi42".
        n_labels:  required when arch=='text2hoi42' to build one-hot targets.
        grad_clip: if not None, clip grad norm to this value (IMoS uses 1.0).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct_1 = 0
    total_correct_3 = 0
    total_n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch_on_device: Dict[str, torch.Tensor] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device, non_blocking=True)
                else:
                    batch_on_device[k] = v
            labels = batch_on_device["labels"]  # (B,) int64

            out = _forward_model(model, batch_on_device, arch=arch, device=device)
            logits = out["logits"]  # (B, n_labels)

            if arch == "text2hoi42":
                if n_labels is None:
                    raise ValueError("run_epoch: n_labels required for text2hoi42 BCE loss")
                probs = torch.sigmoid(logits)
                one_hot = torch.zeros(
                    logits.shape[0], n_labels, device=logits.device, dtype=probs.dtype
                )
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                loss = criterion(probs, one_hot)
                # For top-k we rank by sigmoid score (probs) — same ordering as logits,
                # but makes the multilabel intent explicit.
                score = probs
            else:
                loss = criterion(logits, labels)
                score = logits

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            B = labels.shape[0]
            total_loss += float(loss.item()) * B
            total_n += B

            # Multilabel top-k: the target is the one true class; top-k correct iff
            # its rank is in the top-k highest scores. This is the standard IMoS eval.
            topk = score.topk(min(3, score.shape[-1]), dim=-1).indices  # (B, k)
            correct_1 = int((topk[:, :1] == labels.unsqueeze(-1)).any(dim=-1).sum().item())
            correct_3 = int((topk == labels.unsqueeze(-1)).any(dim=-1).sum().item())
            total_correct_1 += correct_1
            total_correct_3 += correct_3

    if total_n == 0:
        raise RuntimeError("run_epoch: loader yielded zero samples")
    return {
        "loss":  total_loss / total_n,
        "top1":  total_correct_1 / total_n,
        "top3":  total_correct_3 / total_n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hand-object action classifier.")
    p.add_argument("--dataset", type=str, required=True, choices=["grab", "arctic"])
    p.add_argument("--data_root", type=str, required=True,
                   help="Directory containing data.npz and text.json (Text2HOI raw root).")
    p.add_argument("--bin_map", type=str, required=True,
                   help="Path to bin_map_{grab,arctic}.json.")
    p.add_argument("--labels_json", type=str, required=True,
                   help="Path to action_labels.json produced by the data audit.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory (will be created).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_t", type=int, default=MAX_T)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="0.8,0.1,0.1",
                   help="train,val,test fractions (comma-separated, sum=1.0).")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. 0 = single-process (data is in memory).")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="Cap episodes (debug). 0 = no cap.")
    p.add_argument("--augmentation", action="store_true", default=False,
                   help="Enable data augmentation on training split: Gaussian joint noise "
                        "(sigma=3mm), temporal flip (p=0.5), LR hand swap (p=0.5, symmetric verbs only).")
    p.add_argument(
        "--classifier_arch",
        type=str,
        default="imos43",
        choices=["imos43", "text2hoi42"],
        help=(
            "Classifier architecture to train. "
            "'imos43': existing IMoS-adapted 2-stream LSTM on 43 points (21L+21R+1obj), "
            "XY/XZ projections, input (B, max_t, 43, 3). "
            "'text2hoi42': IMoS-adapted 2-stream XY/XZ LSTM on 42 joints (21L+21R), "
            "joints_dim=84, sigmoid+BCE multilabel, fixed 15 frames, input (B, 15, 42, 3). "
            "Default: imos43 (backward-compatible)."
        ),
    )
    p.add_argument(
        "--full_dataset_train",
        action="store_true",
        default=False,
        help=(
            "TRAIN-ON-EVERYTHING mode (IMoS issue #10 leakage protocol). "
            "When set, override split fractions to (1.0, 0.0, 0.0): ALL samples "
            "become training data, no held-out val/test. best.pth is picked by "
            "max train top-1. "
            "This matches the protocol admitted by the IMoS author and (by citation) "
            "inherited by Text2HOI — it is the only way to reach paper-reported "
            "FID/div/mm numbers on Text2HOI's released sample set. "
            "The resulting classifier has seen every GT sample during training, so "
            "accuracy/FID on GT is meaningless (top-1=1.0 by construction). Only "
            "use it to reproduce paper tables; the honest clean-split variant "
            "should be reported alongside."
        ),
    )
    args = p.parse_args()
    _warn_if_leakage(args)
    return args


def _warn_if_leakage(args: argparse.Namespace) -> None:
    """Loudly warn when the leakage protocol is selected.

    The flag is no longer banned outright — user order 2026-04-21 re-enabled
    it for paper-apples-to-apples reproduction. But the reproduction pipeline
    should always ALSO run the clean-split variant, and every resulting
    checkpoint / metric JSON must carry the ``full_dataset_train`` flag so
    downstream reports can label honest vs leaked numbers separately.
    """
    if getattr(args, "full_dataset_train", False):
        logger.warning(
            "LEAKAGE PROTOCOL ENABLED (--full_dataset_train). Train+val+test "
            "merged per IMoS issue #10. Classifier will memorize every GT "
            "sample — accuracy/FID on GT are meaningless under this protocol. "
            "Use only to match paper tables; report honest clean-split "
            "numbers alongside."
        )


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    # Loud warning for programmatic callers that bypass parse_args().
    _warn_if_leakage(args)

    # Device selection — no hardcoding; honor --device and CUDA_VISIBLE_DEVICES.
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "train_classifier: --device=cuda but torch.cuda.is_available() is False. "
                "Set CUDA_VISIBLE_DEVICES to a free GPU and retry, or pass --device=cpu."
            )
        torch.cuda.set_device(0)  # first GPU in the CUDA_VISIBLE_DEVICES subset
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    seed_all(args.seed)

    # ---- Paths ----
    npz_path = os.path.join(args.data_root, "data.npz")
    text_json_path = os.path.join(args.data_root, "text.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"train_classifier: data.npz not found: {npz_path}")
    if not os.path.exists(text_json_path):
        raise FileNotFoundError(f"train_classifier: text.json not found: {text_json_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Vocabulary ----
    labels_spec = load_action_labels(args.dataset, args.labels_json)
    label_to_idx: Dict[str, int] = labels_spec["label_to_idx"]
    labels_list: List[str] = labels_spec["labels"]
    n_labels = len(labels_list)
    logger.info("Vocabulary size for %s: %d labels", args.dataset, n_labels)

    # ---- Arch selection ----
    arch = args.classifier_arch  # "imos43" or "text2hoi42"
    logger.info("Classifier arch: %s", arch)

    # ---- Data ----
    max_eps: Optional[int] = args.max_episodes if args.max_episodes > 0 else None

    if arch == "imos43":
        ds: Dataset = HandObjectClassifierDataset(
            dataset_name=args.dataset,
            bin_map_path=args.bin_map,
            npz_path=npz_path,
            text_json_path=text_json_path,
            label_to_idx=label_to_idx,
            max_t=args.max_t,
            max_episodes=max_eps,
        )
        _collate_fn = collate_batch
    elif arch == "text2hoi42":
        ds = Text2HOIClassifierDataset(
            dataset_name=args.dataset,
            bin_map_path=args.bin_map,
            npz_path=npz_path,
            text_json_path=text_json_path,
            label_to_idx=label_to_idx,
            num_frames=15,
            max_episodes=max_eps,
        )
        _collate_fn = collate_batch_text2hoi
    else:
        raise ValueError(f"main: unknown classifier_arch {arch!r}")

    # ---- Split ----
    if args.full_dataset_train:
        # IMoS issue #10 leakage protocol: train on ALL data, no val/test split.
        logger.warning(
            "=== TRAIN-ON-EVERYTHING mode (IMoS issue #10 leakage protocol) ==="
        )
        logger.warning(
            "All %d samples assigned to train. No val/test split. "
            "best.pth selected by max train top-1 across epochs. "
            "Metrics on GT samples will be artificially inflated.",
            len(ds),  # type: ignore[arg-type]
        )
        all_indices = list(range(len(ds)))  # type: ignore[arg-type]
        split_indices = {"train": all_indices, "val": [], "test": []}
        fractions = (1.0, 0.0, 0.0)
    else:
        frac_parts = args.split.split(",")
        if len(frac_parts) != 3:
            raise ValueError(f"--split must be 'tr,va,te' (3 values); got {args.split!r}")
        fractions = (float(frac_parts[0]), float(frac_parts[1]), float(frac_parts[2]))
        split_indices = deterministic_split(len(ds), fractions, seed=args.seed)  # type: ignore[arg-type]

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    logger.info(
        "Split sizes: train=%d val=%d test=%d (fractions=%s, seed=%d)",
        len(train_idx), len(val_idx), len(test_idx), fractions, args.seed,
    )
    with open(os.path.join(args.out_dir, "split_indices.json"), "w") as f:
        json.dump(
            {"train": train_idx, "val": val_idx, "test": test_idx,
             "seed": args.seed, "fractions": list(fractions),
             "full_dataset_train": args.full_dataset_train},
            f, indent=2,
        )

    train_ds_raw = Subset(ds, train_idx)

    if args.augmentation and arch == "imos43":
        # DataAugDataset is designed for the imos43 (43-point) format.
        # text2hoi42 augmentation not yet implemented; skip with a warning.
        train_ds: Dataset = DataAugDataset(
            base_ds=train_ds_raw,
            label_to_idx=label_to_idx,
            rng_seed=args.seed,
            noise_sigma=0.003,
        )
        logger.info(
            "Augmentation ENABLED: Gaussian noise sigma=0.003m, temporal flip p=0.5, "
            "LR hand swap p=0.5 (symmetric verbs only). Symmetric label indices: %d/%d",
            len(train_ds._symmetric_label_indices), n_labels,  # type: ignore[attr-defined]
        )
    elif args.augmentation and arch == "text2hoi42":
        logger.warning(
            "Augmentation requested but not implemented for arch=text2hoi42; "
            "training without augmentation."
        )
        train_ds = train_ds_raw
    else:
        train_ds = train_ds_raw

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate_fn, drop_last=False,
    )

    # Build val/test loaders only when splits are non-empty.
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    if val_idx:
        val_ds = Subset(ds, val_idx)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate_fn, drop_last=False,
        )
    if test_idx:
        test_ds = Subset(ds, test_idx)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate_fn, drop_last=False,
        )

    # ---- Model ----
    if arch == "imos43":
        model: "Any" = HandObjectActionClassifier(
            n_points=43, hidden_dim=128, n_labels=n_labels
        ).to(device)
        model_name = "HandObjectActionClassifier"
    elif arch == "text2hoi42":
        model = Text2HOIActionClassifier(n_labels=n_labels).to(device)
        model_name = "Text2HOIActionClassifier"
    else:
        raise ValueError(f"main: unknown arch {arch!r}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, %.2fK params", model_name, n_params / 1e3)

    # Loss + optimizer setup depends on arch.
    # - imos43: existing softmax+CE + AdamW + CosineAnnealingLR.
    # - text2hoi42: IMoS-faithful sigmoid+BCE + Adam + ReduceLROnPlateau
    #   (IMoS defaults: factor=0.999, patience=3, threshold=0.05, min_lr=2e-7;
    #    grad_clip=1.0 in run_epoch).
    if arch == "text2hoi42":
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.999, patience=3, threshold=0.05,
            min_lr=2e-7,
        )
        grad_clip_value: Optional[float] = 1.0
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        grad_clip_value = None

    # ---- Train loop ----
    history: List[Dict[str, Any]] = []
    best_val_top1 = -1.0
    best_val_top3 = -1.0
    best_train_top1 = -1.0   # used in full_dataset_train mode
    best_epoch = -1
    best_ckpt = os.path.join(args.out_dir, "best.pth")
    last_ckpt = os.path.join(args.out_dir, "last.pth")

    for epoch in range(1, args.epochs + 1):
        tr_stats = run_epoch(
            model, train_loader, device, optimizer, criterion,
            arch=arch, n_labels=n_labels, grad_clip=grad_clip_value,
        )

        if val_loader is not None:
            va_stats = run_epoch(
                model, val_loader, device, None, criterion,
                arch=arch, n_labels=n_labels,
            )
        else:
            va_stats = {"loss": float("nan"), "top1": float("nan"), "top3": float("nan")}

        # ReduceLROnPlateau (arch='text2hoi42') needs the monitored scalar;
        # other schedulers step unconditionally.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Step on train loss (no val in full-dataset mode) — matches IMoS
            # train_action_recognition.py:210 `self.scheduler_pose.step(train_loss)`.
            scheduler.step(tr_stats["loss"])
        else:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d | train: loss=%.4f top1=%.4f top3=%.4f | val: loss=%.4f top1=%.4f top3=%.4f | lr=%.2e",
            epoch, args.epochs,
            tr_stats["loss"], tr_stats["top1"], tr_stats["top3"],
            va_stats["loss"], va_stats["top1"], va_stats["top3"],
            current_lr,
        )
        history.append({
            "epoch": epoch, "lr": current_lr,
            "train": tr_stats, "val": va_stats,
        })

        if args.full_dataset_train:
            # Full-dataset mode: select best.pth by max train top-1.
            # Rationale: there is no held-out val set; train top-1 is the only available
            # convergence signal. We save the epoch with the highest train top-1.
            if tr_stats["top1"] > best_train_top1:
                best_train_top1 = tr_stats["top1"]
                best_val_top1 = tr_stats["top1"]   # record train top-1 in both for compat
                best_val_top3 = tr_stats["top3"]
                best_epoch = epoch
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "n_labels": n_labels,
                    "labels": labels_list,
                    "max_t": args.max_t,
                    "arch": arch,
                    "full_dataset_train": True,
                }, best_ckpt)
        else:
            # Standard mode: select best.pth by val top-3.
            # FIX-B mandates val_top3 (was val_top1 in v1/v2).
            # Rationale: paper reports top-3 accuracy; val_top3 directly selects the ckpt
            # that will score best on the published metric.
            if va_stats["top3"] > best_val_top3:
                best_val_top1 = va_stats["top1"]
                best_val_top3 = va_stats["top3"]
                best_epoch = epoch
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "n_labels": n_labels,
                    "labels": labels_list,
                    "max_t": args.max_t,
                    "arch": arch,
                    "full_dataset_train": False,
                }, best_ckpt)

    # Save last-epoch checkpoint.
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": args.epochs,
        "n_labels": n_labels,
        "labels": labels_list,
        "max_t": args.max_t,
        "arch": arch,
        "full_dataset_train": args.full_dataset_train,
    }, last_ckpt)

    # ---- Final test eval (load best) ----
    best_state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(best_state["state_dict"])

    if test_loader is not None:
        te_stats = run_epoch(
            model, test_loader, device, None, criterion,
            arch=arch, n_labels=n_labels,
        )
        logger.info(
            "Test (best ckpt epoch=%d): loss=%.4f top1=%.4f top3=%.4f",
            best_epoch, te_stats["loss"], te_stats["top1"], te_stats["top3"],
        )
    else:
        te_stats = {"loss": float("nan"), "top1": float("nan"), "top3": float("nan")}
        if args.full_dataset_train:
            logger.info(
                "full_dataset_train=True: no test split. "
                "best ckpt epoch=%d (train top-1=%.4f)",
                best_epoch, best_train_top1,
            )

    # ---- Persist config + history ----
    config = {
        "args": vars(args),
        "n_labels": n_labels,
        "labels": labels_list,
        "classifier_arch": arch,
        "full_dataset_train": args.full_dataset_train,
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "best_epoch": best_epoch,
        "test": te_stats,
        "split_sizes": {
            "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)
        },
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- Plot accuracy curve ----
    epochs_arr = np.array([h["epoch"] for h in history], dtype=np.int64)
    train_top1_arr = np.array([h["train"]["top1"] for h in history], dtype=np.float32)
    train_top3_arr = np.array([h["train"]["top3"] for h in history], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs_arr, train_top1_arr, label="train top-1", linewidth=2, linestyle="--")
    ax.plot(epochs_arr, train_top3_arr, label="train top-3", linewidth=2, linestyle="--")

    if not args.full_dataset_train:
        val_top1_arr = np.array([h["val"]["top1"] for h in history], dtype=np.float32)
        val_top3_arr = np.array([h["val"]["top3"] for h in history], dtype=np.float32)
        ax.plot(epochs_arr, val_top1_arr, label="val top-1", linewidth=2)
        ax.plot(epochs_arr, val_top3_arr, label="val top-3", linewidth=2)
        title = (
            f"{args.dataset} {arch} — val acc "
            f"(best top1={best_val_top1:.3f} @ ep {best_epoch})"
        )
    else:
        title = (
            f"{args.dataset} {arch} — train acc [full-dataset] "
            f"(best top1={best_train_top1:.3f} @ ep {best_epoch})"
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "val_acc_curve.png"), dpi=120)
    plt.close(fig)

    if args.full_dataset_train:
        logger.info(
            "Done [full_dataset_train]. Best train top-1 = %.4f (top-3 = %.4f) at epoch %d.",
            best_train_top1, best_val_top3, best_epoch,
        )
    else:
        logger.info(
            "Done. Best val top-1 = %.4f (top-3 = %.4f) at epoch %d. Test top-1 = %.4f.",
            best_val_top1, best_val_top3, best_epoch, te_stats["top1"],
        )


if __name__ == "__main__":
    main()
