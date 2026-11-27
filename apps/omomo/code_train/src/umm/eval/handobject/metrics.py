"""IMoS-adapted evaluation metrics for hand-object interaction.

6 metrics (5 reported + FID variants):
  1. accuracy_top1   -- top-1 action classification accuracy (IMoS literal).
  2. accuracy_top3   -- top-3 action classification accuracy (Text2HOI Table 1 header).
  3. fid_corpus      -- corpus-level Frechet Inception Distance on 128-dim features.
  4. fid_per_prompt  -- per-prompt-grouped FID (IMoS-literal: group by text, mean over prompts).
  5. fid_asym        -- asymmetric: GT corpus-level vs pred per-prompt.
  6. diversity       -- mean L2 distance across random pairs (200 pairs, IMoS-literal).
  7. multimodality   -- per-label rejection-sampled pairwise L2 distance (IMoS-literal).
  8. physical_realism -- contact-and-no-penetration fraction (implemented in physical_realism.py).

References:
  - IMoS evaluate_statistical_metrics.py (full file, anindita127/IMoS).
  - Text2HOI Table 1 / Section 4.3 / arxiv 2404.00562.
  - HumanML3D Guo et al. corpus-level FID convention.

FID adaptation note:
  IMoS's `evaluate_fid` (evaluate_statistical_metrics.py:5-16) groups sequences by text
  prompt and computes (mu, sigma) PER TEXT GROUP (N_seq_per_prompt samples) then averages
  Frechet distances across groups. Our hand-object classifier returns ONE 128-dim
  episode-level feature (lstm_out[-1] after tanh*0.1), so per-episode statistics are
  degenerate (N=1 -> singular covariance). We therefore provide THREE FID variants:
    - compute_fid (corpus-level, backward-compat alias for compute_fid_corpus)
    - compute_fid_per_prompt (IMoS-literal per-prompt grouping)
    - compute_fid_asym (GT corpus vs pred per-prompt, handles sparse GT groups)
  The 'fid_per_prompt' variant is the closest to IMoS paper intent.

Diversity note:
  IMoS calculate_diversity_ uses diversity_times=200 pairs (evaluate_statistical_metrics.py:38).
  FIX-1 aligns our default n_pairs to 200 to match IMoS exactly.

Multimodality note:
  IMoS calculate_multimodality_ uses rejection-sampling while-loop with divisor
  n_per_label * num_labels_total (= 20 * 29 for GRAB). FIX-2 rewrites compute_multimodality
  to use this exact algorithm. `num_labels_total` must be passed by the caller.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import linalg

from umm.eval.handobject.canonical import CanonicalSample
from umm.eval.handobject.classifier import HandObjectActionClassifier, Text2HOIActionClassifier
from umm.eval.handobject.motion_feats import canonical_to_tensor, canonical_to_text2hoi_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def batch_forward(
    classifier: HandObjectActionClassifier,
    samples: List[CanonicalSample],
    max_t: int = 120,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Run classifier over all samples in mini-batches, returning collected tensors.

    Args:
        classifier:  HandObjectActionClassifier in eval mode, already on `device`.
        samples:     List of CanonicalSample (validated upstream).
        max_t:       Max sequence length passed to canonical_to_tensor.
        batch_size:  Samples per inference batch.
        device:      Torch device string.

    Returns:
        Dict with:
            "logits":   (N, n_labels) float32.
            "features": (N, 128) float32.
            "lengths":  (N,) int64.

    Raises:
        ValueError: If samples list is empty.
        RuntimeError: On CUDA OOM or shape mismatch.
    """
    if len(samples) == 0:
        raise ValueError("batch_forward: samples list is empty")

    all_logits: List[torch.Tensor] = []
    all_features: List[torch.Tensor] = []
    all_lengths: List[int] = []

    classifier.eval()
    with torch.no_grad():
        start = 0
        while start < len(samples):
            end = min(start + batch_size, len(samples))
            batch_samples = samples[start:end]
            B = len(batch_samples)

            points_list: List[np.ndarray] = []
            length_list: List[int] = []

            for s in batch_samples:
                pts, length = canonical_to_tensor(s, max_t=max_t)  # (max_t, 43, 3), int
                points_list.append(pts)
                length_list.append(length)

            # Stack: (B, max_t, 43, 3)
            points_np = np.stack(points_list, axis=0).astype(np.float32)
            points_t = torch.from_numpy(points_np).to(device)        # (B, max_t, 43, 3)
            lengths_t = torch.tensor(length_list, dtype=torch.int64, device=device)  # (B,)

            out = classifier.forward(points_t, lengths_t)
            all_logits.append(out["logits"].cpu())                    # (B, n_labels)
            all_features.append(out["features"].cpu())                # (B, 128)
            all_lengths.extend(length_list)

            start = end

    logits = torch.cat(all_logits, dim=0)       # (N, n_labels)
    features = torch.cat(all_features, dim=0)   # (N, 128)
    lengths = torch.tensor(all_lengths, dtype=torch.int64)  # (N,)

    return {"logits": logits, "features": features, "lengths": lengths}


def batch_forward_text2hoi(
    classifier: Text2HOIActionClassifier,
    samples: List[CanonicalSample],
    num_frames: int = 15,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Run Text2HOIActionClassifier over all samples in mini-batches.

    Uses canonical_to_text2hoi_input (deterministic=True, linspace frame selection)
    to build (num_frames, 42, 3) inputs for each sample. No variable-length handling —
    Text2HOI classifier uses a fixed sequence length.

    Args:
        classifier:  Text2HOIActionClassifier in eval mode, already on `device`.
        samples:     List of CanonicalSample (validated upstream).
        num_frames:  Fixed sequence length (default 15). Must match classifier.num_frames.
        batch_size:  Samples per inference batch.
        device:      Torch device string.

    Returns:
        Dict with:
            "logits":   (N, n_labels) float32.
            "features": (N, 128) float32.

    Raises:
        ValueError: If samples list is empty or num_frames mismatches classifier.
    """
    if len(samples) == 0:
        raise ValueError("batch_forward_text2hoi: samples list is empty")
    if num_frames != classifier.num_frames:
        raise ValueError(
            f"batch_forward_text2hoi: num_frames={num_frames} does not match "
            f"classifier.num_frames={classifier.num_frames}"
        )

    all_logits: List[torch.Tensor] = []
    all_features: List[torch.Tensor] = []

    classifier.eval()
    with torch.no_grad():
        start = 0
        while start < len(samples):
            end = min(start + batch_size, len(samples))
            batch_samples = samples[start:end]

            points_list: List[np.ndarray] = []
            for s in batch_samples:
                # deterministic=True: linspace frame selection (test/eval mode).
                pts = canonical_to_text2hoi_input(
                    s, num_frames=num_frames, deterministic=True
                )                                                   # (num_frames, 42, 3)
                points_list.append(pts)

            # Stack: (B, num_frames, 42, 3)
            points_np = np.stack(points_list, axis=0).astype(np.float32)
            points_t = torch.from_numpy(points_np).to(device)      # (B, num_frames, 42, 3)

            out = classifier.forward(points_t)
            all_logits.append(out["logits"].cpu())                  # (B, n_labels)
            all_features.append(out["features"].cpu())              # (B, 128)

            start = end

    logits = torch.cat(all_logits, dim=0)       # (N, n_labels)
    features = torch.cat(all_features, dim=0)   # (N, 128)

    return {"logits": logits, "features": features}


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------

def compute_accuracy_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 3,
) -> float:
    """Top-k accuracy: fraction of samples where the true label is in the top-k predictions.

    Adaptation of IMoS's top-1 confusion-matrix accuracy
    (evaluate_statistical_metrics.py:25-35) generalised to top-k via torch.topk.
    IMoS computed accuracy = trace(confusion) / N (sum of diagonal = top-1 hits).
    We generalise to topk: a sample is correct if gt_label in top-k predicted indices.

    For k=1 this reduces to standard top-1 accuracy.
    For k=3 (dispatch §2 spec): correct if any of the 3 highest-logit classes equals
    the ground-truth label.

    Args:
        logits: (N, n_labels) float — raw classifier outputs (pre-softmax).
        labels: (N,) int64 — ground-truth class indices.
        k:      Number of top predictions to check.

    Returns:
        Float in [0, 1]: fraction of correct samples.

    Raises:
        ValueError: On shape mismatch or k < 1.
    """
    if logits.ndim != 2:
        raise ValueError(f"compute_accuracy_topk: logits must be 2D (N, C), got {logits.shape}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"compute_accuracy_topk: labels shape {labels.shape} must be (N={logits.shape[0]},)"
        )
    if k < 1:
        raise ValueError(f"compute_accuracy_topk: k must be >= 1, got {k}")

    N = logits.shape[0]
    if N == 0:
        raise ValueError(
            "compute_accuracy_topk: cannot compute accuracy over empty batch "
            "(logits has 0 samples)"
        )
    k_eff = min(k, logits.shape[1])

    # torch.topk returns (values, indices); indices shape (N, k).
    _, top_indices = torch.topk(logits, k=k_eff, dim=1)  # (N, k_eff)
    labels_expanded = labels.unsqueeze(1).expand_as(top_indices)  # (N, k_eff)
    correct = top_indices.eq(labels_expanded).any(dim=1)           # (N,) bool
    return float(correct.sum().item()) / N


def compute_accuracy_top1(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Top-1 accuracy: IMoS-literal implementation.

    FIX-3 (2026-04-20): Added top-1 variant alongside top-3. IMoS
    calculate_accuracy (evaluate_statistical_metrics.py:25-35) uses:
      `batch_pred = y_pred.max(dim=1).indices`  -- top-1 argmax.
    Text2HOI Table 1 header says "Accuracy (top-3)" but the IMoS code is TOP-1.
    Both are reported to resolve the ambiguity.

    Source: evaluate_statistical_metrics.py:25-35.

    Args:
        logits: (N, n_labels) float -- raw classifier outputs.
        labels: (N,) int64 -- ground-truth class indices.

    Returns:
        Float in [0, 1]: fraction of samples where argmax(logits) == label.

    Raises:
        ValueError: On shape mismatch.
    """
    return compute_accuracy_topk(logits, labels, k=1)


# ---------------------------------------------------------------------------
# FID: corpus-level (Heusel et al. / Guo et al. HumanML3D convention)
# ---------------------------------------------------------------------------

def _calculate_activation_statistics(
    features: torch.Tensor,
) -> tuple:
    """Compute (mu, sigma) over a set of 128-dim features.

    Mirrors IMoS evaluate_statistical_metrics.py:19-23 but applied to the
    full corpus (all N episodes) rather than per-episode frames.

    Args:
        features: (N, 128) float32.

    Returns:
        (mu, sigma): mu is (128,) ndarray, sigma is (128, 128) ndarray.

    Raises:
        ValueError: If N < 2 (covariance undefined for N=1).
    """
    if features.shape[0] < 2:
        raise ValueError(
            f"_calculate_activation_statistics: need >= 2 samples for covariance, "
            f"got {features.shape[0]}"
        )
    acts = features.cpu().numpy().astype(np.float64)   # (N, 128)
    mu = np.mean(acts, axis=0)                         # (128,)
    sigma = np.cov(acts, rowvar=False)                 # (128, 128)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Numpy implementation of the Frechet Distance.

    Copied verbatim from IMoS evaluate_statistical_metrics.py:90-132 with attribution.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Source: evaluate_statistical_metrics.py:90-132.

    Args:
        mu1:    (D,) ndarray -- mean of distribution 1 (generated samples).
        sigma1: (D, D) ndarray -- covariance of distribution 1.
        mu2:    (D,) ndarray -- mean of distribution 2 (reference).
        sigma2: (D, D) ndarray -- covariance of distribution 2.
        eps:    Small constant added to diagonal for numerical stability.

    Returns:
        Frechet distance (float >= 0).

    Raises:
        ValueError: If mu/sigma shapes are mismatched or imaginary component is large.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def compute_fid(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
) -> float:
    """Corpus-level FID between GT and predicted episode feature sets.

    Implements the Heusel et al. / Guo et al. HumanML3D / Text2HOI Table 1
    corpus-level FID convention. Each input is a set of N episode-level 128-dim
    features; a single (mu, sigma) is computed for each set and the Frechet
    distance between the two resulting Gaussians is returned.

    IMoS adaptation note:
      IMoS evaluate_statistical_metrics.py:5-16 computes FID PER EPISODE, iterating
      over frame-level activations where each episode has its own population of frames.
      Our hand-object classifier produces ONE feature per episode (lstm_out[-1] after
      tanh*0.1 -- see classifier.py:196), so per-episode statistics would be computed
      from a single data point (degenerate covariance). Corpus-level FID is the
      standard alternative used in Text2HOI Table 1 and is numerically well-defined.
      See also: evaluate_statistical_metrics.py:19-23 for the activation_statistics formula.

    Args:
        gt_features:   (N, 128) float32 -- episode features from GT samples.
        pred_features: (M, 128) float32 -- episode features from predicted samples.

    Returns:
        FID (float >= 0, lower is better).

    Raises:
        ValueError: If either set has fewer than 2 samples or feature dim mismatches.
    """
    if gt_features.shape[1] != pred_features.shape[1]:
        raise ValueError(
            f"compute_fid: feature dim mismatch: gt={gt_features.shape[1]}, "
            f"pred={pred_features.shape[1]}"
        )

    mu1, sigma1 = _calculate_activation_statistics(gt_features)
    mu2, sigma2 = _calculate_activation_statistics(pred_features)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


# Keep old name as alias for backward compatibility and dispatch schema.
compute_fid_imos = compute_fid
# corpus-level alias: fid_corpus in metrics.json
compute_fid_corpus = compute_fid


# ---------------------------------------------------------------------------
# FID: per-prompt grouping (FIX-4) -- IMoS-literal per evaluate_fid loop
# ---------------------------------------------------------------------------

def _group_features_by_text(
    features: torch.Tensor,
    texts: List[str],
) -> Dict[str, torch.Tensor]:
    """Group feature rows by text prompt string.

    Args:
        features: (N, D) float.
        texts:    (N,) list of text strings, aligned with features.

    Returns:
        Dict mapping each unique text -> (K, D) tensor (K >= 1 samples for that text).

    Raises:
        ValueError: If features and texts lengths differ.
    """
    N = features.shape[0]
    if len(texts) != N:
        raise ValueError(
            f"_group_features_by_text: features has {N} rows but texts has {len(texts)} items"
        )
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(texts):
        groups.setdefault(t, []).append(i)
    return {t: features[idx] for t, idx in groups.items()}


def compute_fid_per_prompt(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
    gt_texts: List[str],
    pred_texts: List[str],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Per-prompt-grouped FID, IMoS evaluate_fid-literal.

    FIX-4 (2026-04-20). Source: IMoS evaluate_statistical_metrics.py:5-16
    evaluate_fid: loops over idx, computes (mu, sigma) per text group from
    gt and pred features, then Frechet distance, then averages.

    Requires >= 2 GT samples AND >= 2 pred samples per prompt to compute a
    non-degenerate covariance. Prompts with fewer samples on either side are
    skipped (counted in n_skipped).

    Args:
        gt_features:   (N_gt, D) float -- GT episode features.
        pred_features: (N_pred, D) float -- predicted episode features.
        gt_texts:      (N_gt,) list -- text prompts aligned with gt_features.
        pred_texts:    (N_pred,) list -- text prompts aligned with pred_features.
        eps:           Regularization eps for Frechet distance covariance.

    Returns:
        Dict with:
          'mean':      mean FID over valid prompts.
          'n_prompts': number of prompts used (>= 2 samples on both sides).
          'n_skipped': number of prompts skipped (insufficient samples).

    Raises:
        ValueError: If no prompt has sufficient samples on both sides.
    """
    if gt_features.shape[1] != pred_features.shape[1]:
        raise ValueError(
            f"compute_fid_per_prompt: feature dim mismatch: "
            f"gt={gt_features.shape[1]}, pred={pred_features.shape[1]}"
        )

    gt_groups = _group_features_by_text(gt_features, gt_texts)
    pred_groups = _group_features_by_text(pred_features, pred_texts)

    common_texts = set(gt_groups.keys()) & set(pred_groups.keys())

    fid_values: List[float] = []
    n_skipped = 0

    for text in sorted(common_texts):
        gt_group = gt_groups[text]    # (K_gt, D)
        pred_group = pred_groups[text]  # (K_pred, D)

        if gt_group.shape[0] < 2 or pred_group.shape[0] < 2:
            n_skipped += 1
            logger.debug(
                "compute_fid_per_prompt: skip prompt %r "
                "(gt_count=%d, pred_count=%d, need >=2 each)",
                text, gt_group.shape[0], pred_group.shape[0],
            )
            continue

        try:
            mu1, sigma1 = _calculate_activation_statistics(gt_group)
            mu2, sigma2 = _calculate_activation_statistics(pred_group)
            fid_val = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
            fid_values.append(fid_val)
        except Exception as exc:
            logger.warning(
                "compute_fid_per_prompt: skip prompt %r due to error: %s",
                text, exc,
            )
            n_skipped += 1

    # Also count prompts that appear only in gt or only in pred.
    gt_only = len(set(gt_groups.keys()) - set(pred_groups.keys()))
    pred_only = len(set(pred_groups.keys()) - set(gt_groups.keys()))
    n_skipped += gt_only + pred_only

    if len(fid_values) == 0:
        raise ValueError(
            "compute_fid_per_prompt: no prompt had >= 2 samples on both sides. "
            f"Total prompts: gt={len(gt_groups)}, pred={len(pred_groups)}, "
            f"common={len(common_texts)}, skipped={n_skipped}."
        )

    return {
        "mean": float(np.mean(fid_values)),
        "n_prompts": len(fid_values),
        "n_skipped": n_skipped,
    }


def compute_fid_asym(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
    gt_texts: List[str],
    pred_texts: List[str],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Asymmetric FID: GT corpus-level vs pred per-prompt groups.

    FIX-4 (2026-04-20). Motivation: GT often has < 2 samples per prompt
    (GRAB GT test set has ~133 samples over 29 classes -> ~4.6 GT/class on
    average, but many individual text prompts have only 1 GT). Using the full
    GT corpus to compute (mu_gt, sigma_gt) ensures numerically stable GT
    statistics. The pred side is grouped per prompt (as in compute_fid_per_prompt),
    and the same GT corpus statistics are used for all prompt comparisons.

    Procedure:
      1. Compute (mu_gt, sigma_gt) from ALL gt_features (corpus-level).
      2. For each unique pred prompt with >= 2 pred samples:
         Compute (mu_pred, sigma_pred) from that group's pred features.
         Compute Frechet distance (mu_gt, sigma_gt) vs (mu_pred, sigma_pred).
      3. Return mean over valid pred groups.

    Args:
        gt_features:   (N_gt, D) float -- GT episode features.
        pred_features: (N_pred, D) float -- predicted episode features.
        gt_texts:      (N_gt,) list -- text prompts aligned with gt_features.
        pred_texts:    (N_pred,) list -- text prompts aligned with pred_features.
        eps:           Regularization eps.

    Returns:
        Dict with:
          'mean':      mean asymmetric FID over valid pred groups.
          'n_prompts': number of pred prompts used.
          'n_skipped': number of pred prompts skipped (< 2 pred samples).

    Raises:
        ValueError: If gt_features has < 2 samples or no pred group has >= 2 samples.
    """
    if gt_features.shape[1] != pred_features.shape[1]:
        raise ValueError(
            f"compute_fid_asym: feature dim mismatch: "
            f"gt={gt_features.shape[1]}, pred={pred_features.shape[1]}"
        )

    # GT corpus statistics (used for ALL pred group comparisons).
    mu_gt, sigma_gt = _calculate_activation_statistics(gt_features)

    pred_groups = _group_features_by_text(pred_features, pred_texts)

    fid_values: List[float] = []
    n_skipped = 0

    for text in sorted(pred_groups.keys()):
        pred_group = pred_groups[text]  # (K_pred, D)
        if pred_group.shape[0] < 2:
            n_skipped += 1
            continue
        try:
            mu_pred, sigma_pred = _calculate_activation_statistics(pred_group)
            fid_val = calculate_frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred, eps=eps)
            fid_values.append(fid_val)
        except Exception as exc:
            logger.warning(
                "compute_fid_asym: skip pred prompt %r due to error: %s",
                text, exc,
            )
            n_skipped += 1

    if len(fid_values) == 0:
        raise ValueError(
            "compute_fid_asym: no pred prompt had >= 2 samples. "
            f"Total pred prompts: {len(pred_groups)}, skipped: {n_skipped}."
        )

    return {
        "mean": float(np.mean(fid_values)),
        "n_prompts": len(fid_values),
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def compute_diversity(
    features: torch.Tensor,
    n_pairs: int = 200,
    seed: int = 42,
    normalize: bool = False,
) -> float:
    """Mean L2 distance across n_pairs random pairs of episode features.

    IMoS-literal implementation of calculate_diversity_ from
    evaluate_statistical_metrics.py:37-48. IMoS uses diversity_times=200 pairs.
    FIX-1 (2026-04-20): changed default n_pairs from 300 to 200 to match IMoS exactly.

    Uses numpy.random.RandomState(seed) -- never the global RNG -- for reproducibility
    under --seed in the CLI.

    Args:
        features: (N, D) float -- episode feature vectors.
        n_pairs:  Number of random pairs to sample. Default 200 (IMoS-literal).
                  Source: evaluate_statistical_metrics.py:38 diversity_times=200.
        seed:     RNG seed for pair sampling. Must be set for reproducibility.
        normalize: If True, L2-normalize each feature to unit norm before computing
                   distances. This neutralizes classifier-dependent feature-space scale
                   so reported diversity is a pure geometric spread on the unit sphere.
                   Useful when paper's classifier features may have different magnitude
                   than ours (e.g., Text2HOI ARCTIC diversity gap: +74% under L2 raw,
                   expected to drop with normalization).

    Returns:
        Mean L2 distance (float >= 0, higher = more diverse).

    Raises:
        ValueError: If N < 2 or n_pairs < 1.
    """
    N = features.shape[0]
    if N < 2:
        raise ValueError(f"compute_diversity: need >= 2 samples, got {N}")
    if n_pairs < 1:
        raise ValueError(f"compute_diversity: n_pairs must be >= 1, got {n_pairs}")

    rng = np.random.RandomState(seed)
    first_indices = rng.randint(0, N, n_pairs)   # (n_pairs,)
    second_indices = rng.randint(0, N, n_pairs)  # (n_pairs,)

    feat_np = features.cpu().float().numpy()     # (N, D) float32
    if normalize:
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feat_np = feat_np / norms

    diversity = 0.0
    for fi, si in zip(first_indices, second_indices):
        diff = feat_np[fi] - feat_np[si]         # (D,)
        diversity += float(np.linalg.norm(diff))
    diversity /= n_pairs
    return diversity


# ---------------------------------------------------------------------------
# Multimodality
# ---------------------------------------------------------------------------

def compute_multimodality(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_labels_total: int,
    n_per_label: int = 20,
    seed: int = 42,
    max_iter: int = 10000,
    normalize: bool = False,
) -> float:
    """Per-label pairwise L2 distance of episode features, IMoS-literal implementation.

    FIX-2 (2026-04-20): Rewrote to match IMoS calculate_multimodality_ exactly.
    Source: evaluate_statistical_metrics.py:50-73.

    Key differences vs old implementation:
      - Uses rejection-sampling while-loop (draw random first_idx; if its label's
        quota is 0 skip; draw second_idx; retry second until same label as first;
        decrement quota; accumulate distance). This is the IMoS exact algorithm.
      - Divisor = n_per_label * num_labels_total (= 20 * 29 for GRAB, 20 * 11 for
        ARCTIC). NOT the count of active labels. This matches IMoS line:
        `multimodality /= (multimodality_times * num_labels)`.
      - Uses seeded numpy.random.RandomState instead of IMoS's global RNG.
      - max_iter safety cap (IMoS: `count <= 10000`).

    If a class has too few samples to fill its quota, the while-loop exhausts
    max_iter and the remaining quota stays unfulfilled. The denominator is still
    n_per_label * num_labels_total, so sparse classes cause under-estimation
    (matching IMoS's documented behavior per evidence_log.md §4).

    Args:
        features:         (N, D) float -- episode feature vectors.
        labels:           (N,) int64 -- ground-truth class indices (0-indexed).
        num_labels_total: Vocabulary size (29 for GRAB, 11 for ARCTIC). This is
                          len(label_to_idx), NOT the count of active labels in this
                          sample set. Used as divisor: sum / (n_per_label * num_labels_total).
        n_per_label:      Target pair quota per label. Default 20 (IMoS: multimodality_times).
        seed:             RNG seed for reproducibility (replaces IMoS's global RNG).
        max_iter:         Safety cap on while-loop iterations. Default 10000 (IMoS).

    Returns:
        Multimodality (float >= 0, higher = more within-class diversity).

    Raises:
        ValueError: If N < 2, n_per_label < 1, or num_labels_total < 1.
    """
    N = features.shape[0]
    if N < 2:
        raise ValueError(f"compute_multimodality: need >= 2 samples, got {N}")
    if n_per_label < 1:
        raise ValueError(f"compute_multimodality: n_per_label must be >= 1, got {n_per_label}")
    if num_labels_total < 1:
        raise ValueError(
            f"compute_multimodality: num_labels_total must be >= 1, got {num_labels_total}"
        )

    rng = np.random.RandomState(seed)
    feat_np = features.cpu().float().numpy()    # (N, D)
    labels_np = labels.cpu().numpy()            # (N,)
    if normalize:
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feat_np = feat_np / norms

    # Per-label quota: how many more pairs to collect for each label.
    # IMoS: labal_quotas = np.repeat(multimodality_times, num_labels)
    label_quotas = np.full(num_labels_total, n_per_label, dtype=np.int64)

    multimodality = 0.0
    count = 0

    while np.any(label_quotas > 0) and count < max_iter:
        count += 1

        # Draw first sample; skip if its label quota is exhausted.
        first_idx = rng.randint(0, N)
        first_label = int(labels_np[first_idx])
        if first_label < 0 or first_label >= num_labels_total:
            # Out-of-range label -- skip silently.
            continue
        if label_quotas[first_label] <= 0:
            continue

        # Draw second sample; reject until same label as first.
        # Inner while has no iteration cap (mirrors IMoS exactly); the outer
        # count <= max_iter provides the safety net.
        second_idx = rng.randint(0, N)
        second_label = int(labels_np[second_idx])
        inner_tries = 0
        while second_label != first_label and inner_tries < max_iter:
            second_idx = rng.randint(0, N)
            second_label = int(labels_np[second_idx])
            inner_tries += 1

        if second_label != first_label:
            # Could not find a matching pair in inner_tries; continue.
            continue

        label_quotas[first_label] -= 1
        diff = feat_np[first_idx] - feat_np[second_idx]  # (D,)
        multimodality += float(np.linalg.norm(diff))

    # IMoS divisor: multimodality_times * num_labels (fixed, not active count).
    divisor = float(n_per_label * num_labels_total)
    return multimodality / divisor
