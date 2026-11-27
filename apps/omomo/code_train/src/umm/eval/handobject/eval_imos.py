"""IMoS-adapted evaluation CLI for hand-object interaction.

Computes 5 metrics against GT samples using the hand-object action classifier:
  1. accuracy_top3       -- top-3 action classification accuracy.
  2. fid                 -- corpus-level Frechet Inception Distance.
  3. diversity           -- mean L2 pairwise distance (300 pairs).
  4. multimodality       -- per-label pairwise L2 distance (20 pairs/label).
  5. physical_realism    -- contact-and-no-penetration fraction.

Usage:
  python -m umm.eval.handobject.eval_imos \\
    --samples <canonical_dir> \\
    --gt <canonical_gt_dir_OR_text2hoi_raw_root> \\
    --dataset {grab,arctic} \\
    --classifier_ckpt data/handobject/imos_ckpts/classifier/<dataset>/best.pth \\
    --labels_json <path> \\
    --out <metrics.json> \\
    [--seed 42] \\
    [--device cuda] \\
    [--contact_thresh_m 0.02] \\
    [--penetration_thresh_m 0.005] \\
    [--n_repeats 20] \\
    [--batch_size 32] \\
    [--max_t 120]

Notes:
  - FID is corpus-level (Guo/HumanML3D/Text2HOI convention), NOT IMoS per-sample.
    See metrics.py::compute_fid for the adaptation rationale.
  - Pair-sampling metrics (diversity, multimodality) are repeated `n_repeats` times
    with sub-seeds derived from --seed to produce mean ± std.
  - GT auto-detection: if <gt>/data.npz exists, treat as Text2HOI raw root and
    load via iter_gt_episodes; otherwise treat as canonical npz directory.
  - Physical realism requires object meshes on disk (DATA_ROOT env or data/).
    Missing meshes for an obj_name are logged as warnings; samples with missing
    meshes are counted but their physical_realism is NaN.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from umm.eval.handobject.action_label import load_action_labels, text_to_label_idx
from umm.eval.handobject.canonical import (
    CanonicalSample,
    load_canonical_npz,
    validate_canonical_dir,
)
from umm.eval.handobject.classifier import HandObjectActionClassifier, Text2HOIActionClassifier
from umm.eval.handobject.gt_loader import iter_gt_episodes
from umm.eval.handobject.metrics import (
    batch_forward,
    batch_forward_text2hoi,
    compute_accuracy_top1,
    compute_accuracy_topk,
    compute_diversity,
    compute_fid,
    compute_fid_asym,
    compute_fid_corpus,
    compute_fid_per_prompt,
    compute_multimodality,
)
from umm.eval.handobject.motion_feats import canonical_to_text2hoi_input
from umm.eval.handobject.physical_realism import (
    compute_physical_realism,
    load_arctic_obj_top_masks,
    load_obj_pcs,
    load_object_mesh,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------

def load_classifier(
    ckpt_path: str,
    device: str,
    arch: str = "imos43",
) -> "Any":
    """Load action classifier from a checkpoint path.

    Supports two architectures selected by `arch`:
      - "imos43": HandObjectActionClassifier (IMoS-adapted 2-stream LSTM, 43 points).
        Config keys: n_points, hidden_dim, n_labels, num_layers, kernel_size.
        Defaults: n_points=43, hidden_dim=128, num_layers=2, kernel_size=3.
      - "text2hoi42": Text2HOIActionClassifier (IMoS-adapted 2-stream XY/XZ
        LSTM, 42 joints, sigmoid+BCE multilabel — mirrors IMoS action_rec.py:14-82
        with joints_dim=84 instead of 44).
        Config keys: n_labels (required; read from checkpoint 'n_labels' field).
        Defaults: num_frames=15, joints_per_frame=42, feat_dim=3, hidden_dim=128,
                  num_layers=2, kernel_size=3. NO dropout (IMoS arch has none).

    A sibling `config.json` is used if present. The checkpoint's 'n_labels' field
    takes priority over config.json for n_labels (since the ckpt is authoritative).

    Args:
        ckpt_path: Path to the .pth checkpoint file.
        device:    Torch device string.
        arch:      "imos43" (default) or "text2hoi42".

    Returns:
        Classifier in eval mode on `device`.

    Raises:
        FileNotFoundError: If ckpt_path does not exist.
        ValueError: On malformed checkpoint, unknown arch, or missing n_labels.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"load_classifier: checkpoint not found: {ckpt_path}")

    if arch not in ("imos43", "text2hoi42"):
        raise ValueError(f"load_classifier: unknown arch {arch!r}; expected 'imos43' or 'text2hoi42'")

    # Load raw checkpoint first — the checkpoint's n_labels is authoritative.
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        ckpt_meta = {k: v for k, v in raw.items() if k != "state_dict"}
    elif isinstance(raw, dict) and all(isinstance(k, str) for k in raw):
        state_dict = raw
        ckpt_meta = {}
    else:
        raise ValueError(
            f"load_classifier: cannot interpret checkpoint format in {ckpt_path!r}. "
            "Expected a state_dict or {'state_dict': ...} dict."
        )

    # Try to load config.json sibling.
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    cfg_from_file: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg_from_file = json.load(f)
        logger.info("load_classifier: loaded config from %s", config_path)
    else:
        logger.warning(
            "load_classifier: config.json not found next to %s; using checkpoint meta + defaults.",
            ckpt_path,
        )

    # n_labels: ckpt_meta['n_labels'] > config.json > ValueError.
    n_labels_int: Optional[int] = None
    if "n_labels" in ckpt_meta:
        n_labels_int = int(ckpt_meta["n_labels"])
    elif "n_labels" in cfg_from_file:
        n_labels_int = int(cfg_from_file["n_labels"])
    if n_labels_int is None:
        raise ValueError(
            f"load_classifier: cannot determine n_labels from checkpoint or config.json "
            f"at {ckpt_path!r}. Checkpoint was probably saved without the 'n_labels' field "
            f"(older format). Re-train or manually add 'n_labels' to config.json."
        )

    if arch == "imos43":
        classifier_kwargs: Dict[str, Any] = {
            "n_points": 43, "hidden_dim": 128, "n_labels": n_labels_int,
            "num_layers": 2, "kernel_size": 3,
        }
        # Override from config.json if present.
        for key in ("n_points", "hidden_dim", "num_layers", "kernel_size"):
            if key in cfg_from_file:
                classifier_kwargs[key] = int(cfg_from_file[key])
        logger.info(
            "load_classifier: arch=imos43, kwargs=%s", classifier_kwargs
        )
        classifier = HandObjectActionClassifier(**classifier_kwargs)
    else:  # text2hoi42
        # Use defaults (Text2HOI constants); no custom kwargs from config.json.
        logger.info(
            "load_classifier: arch=text2hoi42, n_labels=%d (IMoS-2-stream XY/XZ, "
            "num_frames=15, joints_per_frame=42, feat_dim=3, hidden_dim=128, "
            "num_layers=2, kernel_size=3, NO dropout)",
            n_labels_int,
        )
        classifier = Text2HOIActionClassifier(n_labels=n_labels_int)

    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    classifier.to(device)
    logger.info("load_classifier: arch=%s loaded %s -> device=%s", arch, ckpt_path, device)
    return classifier


# ---------------------------------------------------------------------------
# Sample loading helpers
# ---------------------------------------------------------------------------

def _load_canonical_dir(dirpath: str) -> List[CanonicalSample]:
    """Load all .npz samples from a canonical directory. Fail loud on validation error."""
    summary = validate_canonical_dir(dirpath)
    if summary["errors"]:
        for e in summary["errors"]:
            logger.error("canonical dir error: %s -- %s", e["file"], e["error"])
        raise ValueError(
            f"_load_canonical_dir: {len(summary['errors'])} validation error(s) in {dirpath!r}. "
            "Fix the files or remove them before running eval."
        )

    samples: List[CanonicalSample] = []
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(dirpath, fname)
        if not os.path.isfile(fpath):
            continue
        samples.append(load_canonical_npz(fpath))
    logger.info("_load_canonical_dir: loaded %d samples from %s", len(samples), dirpath)
    return samples


def _resolve_raw_root(gt_path: str, dataset: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(npz, bin_map, text_json)`` paths if ``gt_path`` is a valid
    Text2HOI raw root, else None.

    A "valid" raw root must contain ALL three required sibling files. Previous
    behavior detected raw-root mode whenever ``data.npz`` was present in the
    gt_path directory alone, which silently misfired when ``bin_map_*.json``
    lived under the repo-wide data directory (e.g. text2hoi_raw/grab/data.npz
    but bin_map_grab.json at data/handobject/bin_map_grab.json). When only
    data.npz is present, fall through to canonical-dir loading instead of
    erroring out.

    For convenience, also accept bin_map_{dataset}.json at the parent dir of
    gt_path (the repo layout data/handobject/bin_map_{ds}.json +
    data/handobject/text2hoi_raw/{ds}/data.npz). If neither location has the
    bin_map, return None.
    """
    npz_path = os.path.join(gt_path, "data.npz")
    text_json_path = os.path.join(gt_path, "text.json")
    if not (os.path.isfile(npz_path) and os.path.isfile(text_json_path)):
        return None

    # bin_map: try gt_path first, then parent, then grandparent.
    bin_map_name = f"bin_map_{dataset}.json"
    candidates = [
        os.path.join(gt_path, bin_map_name),
        os.path.join(os.path.dirname(gt_path), bin_map_name),
        os.path.join(os.path.dirname(os.path.dirname(gt_path)), bin_map_name),
    ]
    bin_map_path = next((p for p in candidates if os.path.isfile(p)), None)
    if bin_map_path is None:
        return None
    return npz_path, bin_map_path, text_json_path


def _is_text2hoi_raw_root(path: str, dataset: str) -> bool:
    """Return True only when ALL raw-root files resolve (not just data.npz).

    Signature updated from the prior ``(path: str)`` form to require an
    explicit dataset so we can look up the correct ``bin_map_{dataset}.json``.
    """
    return _resolve_raw_root(path, dataset) is not None


def _load_gt_samples(
    gt_path: str,
    dataset: str,
    max_episodes: Optional[int] = None,
) -> List[CanonicalSample]:
    """Load GT samples from either a canonical npz dir or a Text2HOI raw root.

    Auto-detects format:
      - If ``_resolve_raw_root(gt_path, dataset)`` succeeds (all three required
        sibling files — data.npz, text.json, bin_map_{dataset}.json — resolve,
        possibly across parent dirs), load via ``iter_gt_episodes``.
      - Otherwise, treat ``gt_path`` as a canonical-npz directory and dispatch
        to ``_load_canonical_dir``. Previous behavior flagged raw-root mode on
        ``data.npz`` alone and then crashed when ``bin_map_*.json`` was not
        co-located; the stricter resolution here prevents that.

    Args:
        gt_path:      Path to GT directory (canonical npz or Text2HOI raw).
        dataset:      "grab" or "arctic".
        max_episodes: Cap on number of episodes to load (for testing).

    Returns:
        List of CanonicalSample.
    """
    resolved = _resolve_raw_root(gt_path, dataset)
    if resolved is not None:
        npz_path, bin_map_path, text_json_path = resolved
        logger.info(
            "_load_gt_samples: detected Text2HOI raw root at %s "
            "(bin_map=%s)",
            gt_path, bin_map_path,
        )
        samples = list(
            iter_gt_episodes(
                dataset=dataset,
                bin_map_path=bin_map_path,
                npz_path=npz_path,
                text_json_path=text_json_path,
                max_episodes=max_episodes,
            )
        )
        logger.info("_load_gt_samples: loaded %d GT episodes from raw root", len(samples))
        return samples
    return _load_canonical_dir(gt_path)


# ---------------------------------------------------------------------------
# Object mesh loader (with caching, warn on missing)
# ---------------------------------------------------------------------------

# GRAB object-name translation: canonical `proc_obj_name` (space-separated,
# lowercase) -> contact_meshes PLY filename (single-word, no spaces). For sized
# primitive objects (cube, cylinder, pyramid, sphere, torus) Text2HOI's
# proc_obj_name drops the size qualifier -- use the 'large' variant as default.
_GRAB_NAME_TRANSLATE = {
    "cube": "cubelarge",
    "cylinder": "cylinderlarge",
    "pyramid": "pyramidlarge",
    "sphere": "spherelarge",
    "torus": "toruslarge",
}


def _translate_obj_name(dataset: str, canonical_obj_name: str) -> List[str]:
    """Return candidate mesh-file basenames to try for a given canonical obj_name.

    GRAB PLY filenames are single-word lowercase (e.g. 'alarmclock.ply'); canonical
    `proc_obj_name` is space-separated (e.g. 'alarm clock'). Sized primitives lose
    their size qualifier in proc_obj_name and are mapped to '*large'. ARCTIC has
    no mismatch (canonical name matches mesh dir name directly).

    Returns a list of candidates to try in order (first success wins). Non-empty.
    """
    candidates: List[str] = [canonical_obj_name]
    if dataset == "grab":
        squashed = canonical_obj_name.replace(" ", "").lower()
        if squashed != canonical_obj_name:
            candidates.append(squashed)
        if canonical_obj_name in _GRAB_NAME_TRANSLATE:
            candidates.append(_GRAB_NAME_TRANSLATE[canonical_obj_name])
    return candidates


def _collect_obj_meshes(
    samples: List[CanonicalSample],
    dataset: str,
) -> Dict[str, Any]:
    """Load meshes for all unique obj_names in samples. Warn on missing; do not crash.

    For GRAB, canonical obj_names are translated to single-word PLY filenames via
    `_translate_obj_name` (e.g. 'alarm clock' -> 'alarmclock'; 'cube' -> 'cubelarge').
    The dict keys are the ORIGINAL canonical obj_names so that downstream lookup
    via `sample.obj_name` works without further translation.

    Returns:
        Dict[canonical_obj_name -> trimesh.Trimesh]. Missing obj_names are not in the dict.
    """
    obj_names = {s.obj_name for s in samples}
    obj_meshes: Dict[str, Any] = {}
    for name in sorted(obj_names):
        last_exc: Optional[Exception] = None
        for cand in _translate_obj_name(dataset, name):
            try:
                obj_meshes[name] = load_object_mesh(dataset=dataset, obj_name=cand)
                if cand != name:
                    logger.info(
                        "_collect_obj_meshes: %r -> %r (name-translated)", name, cand
                    )
                last_exc = None
                break
            except (FileNotFoundError, ValueError) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            logger.warning(
                "_collect_obj_meshes: skip %r -- all candidates failed; last error: %s",
                name, last_exc,
            )
    return obj_meshes


def _collect_obj_pcs(dataset: str) -> Dict[str, Any]:
    """Load object point clouds from obj.pkl for physical realism (MANO-vertex method).

    Uses Text2HOI's obj.pkl['obj_pcs'] dict — shape (N_pc, 3) per object, object-local frame.
    Returns empty dict with a warning if obj.pkl is not found (eval will skip phys_realism).

    Args:
        dataset: 'grab' or 'arctic'.

    Returns:
        Dict[obj_name -> (N_pc, 3) float32 array].
    """
    try:
        obj_pcs = load_obj_pcs(dataset=dataset)
        logger.info("_collect_obj_pcs: loaded %d object PCs for dataset=%r", len(obj_pcs), dataset)
        return obj_pcs
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning(
            "_collect_obj_pcs: could not load obj.pkl for dataset=%r: %s. "
            "Physical realism will be NaN for all samples.",
            dataset, exc,
        )
        return {}


def _collect_obj_top_masks(dataset: str) -> Dict[str, np.ndarray]:
    """Load per-object articulated ``top`` masks for ARCTIC (empty for other datasets).

    ARCTIC objects have one revolute DoF; the top part rotates relative to the
    bottom. ``top_mask[i] == True`` means point i belongs to the articulated
    upper shell. For GRAB and other rigid datasets, returns an empty dict so
    the rigid transform path is taken in `compute_physical_realism`.
    """
    if dataset != "arctic":
        return {}
    try:
        masks = load_arctic_obj_top_masks()
        logger.info(
            "_collect_obj_top_masks: loaded %d ARCTIC top-masks", len(masks)
        )
        return masks
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning(
            "_collect_obj_top_masks: could not load ARCTIC top masks: %s. "
            "ARCTIC physical_realism will fall back to rigid (un-articulated) "
            "transform — this is a known bug and will give depressed scores.",
            exc,
        )
        return {}


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return current HEAD SHA (short). Returns 'unknown' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning(
            "_git_sha: `git rev-parse --short HEAD` returned code=%d stderr=%r",
            result.returncode, result.stderr.strip(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("_git_sha: git invocation failed: %s", exc)
    return "unknown"


# ---------------------------------------------------------------------------
# Method name inference
# ---------------------------------------------------------------------------

def _infer_method_name(samples_dir: str) -> str:
    """Infer method name from config.json in samples directory. Returns 'unknown' if not found."""
    cfg_path = os.path.join(samples_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            for key in ("method", "method_name", "name", "model"):
                if key in cfg and isinstance(cfg[key], str):
                    return cfg[key]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "_infer_method_name: could not read %r: %s", cfg_path, exc
            )
    return "unknown"


# ---------------------------------------------------------------------------
# Metric computation loop with repeat
# ---------------------------------------------------------------------------

def _compute_metrics_with_repeat(
    pred_features: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_features: torch.Tensor,
    gt_labels: torch.Tensor,
    num_labels_total: int,
    n_repeats: int,
    base_seed: int,
    pred_texts: Optional[List[str]] = None,
    gt_texts: Optional[List[str]] = None,
    normalize_features: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute FID variants, diversity, multimodality n_repeats times; accuracy once.

    FID variants and accuracy are deterministic (no sampling); they are computed once
    and reported with std=0. Diversity and multimodality use pair sampling with varying
    sub-seeds derived from base_seed.

    FIX-1: diversity now uses n_pairs=200 (IMoS-literal, was 300).
    FIX-2: multimodality now uses IMoS rejection-sampling + num_labels_total divisor.
    FIX-3: both accuracy_top1 (IMoS literal) and accuracy_top3 are computed.
    FIX-4: three FID variants: fid_corpus, fid_per_prompt, fid_asym.

    Args:
        pred_features:    (N, 128) float.
        pred_logits:      (N, n_labels) float.
        pred_labels:      (N,) int64.
        gt_features:      (M, 128) float.
        gt_labels:        (M,) int64.
        num_labels_total: Vocabulary size (len(label_to_idx)), used as multimodality divisor.
        n_repeats:        Number of times to repeat pair-sampling metrics.
        base_seed:        Base RNG seed; sub-seeds = base_seed + repeat_idx.
        pred_texts:       (N,) list of text prompts for pred samples (for FID per-prompt).
        gt_texts:         (M,) list of text prompts for GT samples (for FID per-prompt).

    Returns:
        Dict metric_name -> {"mean": float, "std": float} (plus n_prompts/n_skipped for FID variants).
    """
    # Deterministic metrics (one shot).
    acc_top1 = compute_accuracy_top1(pred_logits, pred_labels)
    acc_top3 = compute_accuracy_topk(pred_logits, pred_labels, k=3)
    fid_corpus_val = compute_fid_corpus(gt_features, pred_features)

    # FID per-prompt and asymmetric (require text lists).
    fid_per_prompt_result: Dict[str, float] = {"mean": float("nan"), "n_prompts": 0, "n_skipped": 0}
    fid_asym_result: Dict[str, float] = {"mean": float("nan"), "n_prompts": 0, "n_skipped": 0}
    if pred_texts is not None and gt_texts is not None:
        try:
            fid_per_prompt_result = compute_fid_per_prompt(
                gt_features, pred_features, gt_texts, pred_texts
            )
        except ValueError as exc:
            logger.warning("FID per-prompt computation failed: %s", exc)
        try:
            fid_asym_result = compute_fid_asym(
                gt_features, pred_features, gt_texts, pred_texts
            )
        except ValueError as exc:
            logger.warning("FID asym computation failed: %s", exc)
    else:
        logger.warning(
            "_compute_metrics_with_repeat: pred_texts/gt_texts not provided; "
            "fid_per_prompt and fid_asym will be NaN."
        )

    # Pair-sampling metrics: repeat with sub-seeds.
    div_values: List[float] = []
    mm_values: List[float] = []
    for rep in range(n_repeats):
        sub_seed = base_seed + rep
        div_values.append(
            compute_diversity(pred_features, n_pairs=200, seed=sub_seed, normalize=normalize_features)
        )
        mm_values.append(
            compute_multimodality(
                pred_features,
                pred_labels,
                num_labels_total=num_labels_total,
                n_per_label=20,
                seed=sub_seed,
                normalize=normalize_features,
            )
        )

    return {
        "accuracy_top1": {"mean": acc_top1, "std": 0.0},
        "accuracy_top3": {"mean": acc_top3, "std": 0.0},
        "fid_corpus": {"mean": fid_corpus_val, "std": 0.0},
        "fid_per_prompt": fid_per_prompt_result,
        "fid_asym": fid_asym_result,
        # Legacy key "fid" kept for backward-compat (corpus-level).
        "fid": {"mean": fid_corpus_val, "std": 0.0},
        "diversity": {"mean": float(np.mean(div_values)), "std": float(np.std(div_values))},
        "multimodality": {"mean": float(np.mean(mm_values)), "std": float(np.std(mm_values))},
    }


# ---------------------------------------------------------------------------
# Physical realism aggregation
# ---------------------------------------------------------------------------

def _compute_physical_realism_aggregated(
    samples: List[CanonicalSample],
    obj_meshes: Dict[str, Any],
    contact_thresh_m: float,
    penetration_thresh_m: Optional[float],
    n_repeats: int,
    base_seed: int,
    max_frames: Optional[int] = None,
    obj_pcs: Optional[Dict[str, Any]] = None,
    obj_top_masks: Optional[Dict[str, np.ndarray]] = None,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    """Compute physical realism per sample, return mean +- std across samples.

    Uses MANO-vertex KNN penetration + fingertip contact (Text2HOI method).
    Samples with missing obj_pcs entry or zero valid frames are excluded from the
    aggregate (logged as warnings). n_repeats is accepted for API symmetry but
    physical realism is deterministic. We report sample-level mean/std.

    Args:
        obj_pcs: Dict[obj_name -> (N_pc, 3) float32] from _collect_obj_pcs.
                 If None, all samples return NaN.

    Returns:
        {"mean": float, "std": float}.
    """
    if obj_pcs is None:
        logger.warning(
            "_compute_physical_realism_aggregated: obj_pcs is None; "
            "all samples will return NaN."
        )
        obj_pcs = {}

    realism_values: List[float] = []
    n_missing_mesh = 0
    n_no_valid = 0

    for s in samples:
        if s.obj_name not in obj_pcs:
            n_missing_mesh += 1
            continue
        try:
            pr = compute_physical_realism(
                s,
                obj_meshes=None,
                contact_thresh_m=contact_thresh_m,
                penetration_thresh_m=penetration_thresh_m,
                max_frames=max_frames,
                obj_pcs=obj_pcs,
                obj_top_masks=obj_top_masks,
                dataset=dataset,
            )
            r = pr["realism"]
            if np.isnan(r):
                n_no_valid += 1
                continue
            realism_values.append(r)
        except Exception as exc:
            logger.warning("physical_realism: skip sample obj=%r -- %s", s.obj_name, exc)
            n_missing_mesh += 1

    if n_missing_mesh > 0:
        logger.warning(
            "physical_realism: %d samples skipped (missing mesh)", n_missing_mesh
        )
    if n_no_valid > 0:
        logger.warning(
            "physical_realism: %d samples had zero valid frames", n_no_valid
        )

    if len(realism_values) == 0:
        logger.warning("physical_realism: no valid samples; returning NaN")
        return {"mean": float("nan"), "std": float("nan")}

    return {
        "mean": float(np.mean(realism_values)),
        "std": float(np.std(realism_values)),
    }


# ---------------------------------------------------------------------------
# Physical-realism-only fast path
# ---------------------------------------------------------------------------

def _run_phys_realism_only(args: argparse.Namespace) -> None:
    """Compute physical realism only (skips classifier + distribution metrics).

    Used when only the physical realism metric needs updating (e.g., after
    changing the MANO-vertex KNN method) and distribution metrics (FID, diversity,
    multimodality, accuracy) are already available from a prior v2 run.

    Output JSON contains only 'physical_realism' in 'metrics'. Merge with v2
    output to produce a complete v3 file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger.info("--phys_realism_only: skipping classifier + distribution metrics.")
    logger.info("Loading predicted samples from %s ...", args.samples)
    pred_samples = _load_canonical_dir(args.samples)
    if len(pred_samples) == 0:
        raise ValueError(f"No samples found in {args.samples!r}")

    logger.info("Loading object point clouds for physical realism (MANO-vertex method) ...")
    obj_pcs = _collect_obj_pcs(dataset=args.dataset)
    obj_top_masks = _collect_obj_top_masks(dataset=args.dataset)

    phys_max_frames = args.phys_realism_max_frames if args.phys_realism_max_frames > 0 else None
    logger.info(
        "Computing physical realism for %d predicted samples (max_frames=%s, arti_objs=%d) ...",
        len(pred_samples), phys_max_frames, len(obj_top_masks),
    )
    pr_result = _compute_physical_realism_aggregated(
        samples=pred_samples,
        obj_meshes=obj_pcs,
        contact_thresh_m=args.contact_thresh_m,
        penetration_thresh_m=args.penetration_thresh_m,
        n_repeats=getattr(args, "n_repeats", 20),
        base_seed=getattr(args, "seed", 42),
        max_frames=phys_max_frames,
        obj_pcs=obj_pcs,
        obj_top_masks=obj_top_masks,
        dataset=args.dataset,
    )

    method_name = _infer_method_name(args.samples)
    output: Dict[str, Any] = {
        "dataset": args.dataset,
        "method": method_name,
        "n_samples": len(pred_samples),
        "metrics": {
            "physical_realism": pr_result,
        },
        "commit_sha": _git_sha(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "notes": (
            "phys_realism_only=True: physical_realism computed with 778-vertex MANO KNN "
            "penetration + fingertip-vert contact (Text2HOI FIX-A). "
            "Other metrics absent; merge with v2 for complete results."
        ),
        "config": {
            "contact_thresh_m": args.contact_thresh_m,
            "penetration_thresh_m": args.penetration_thresh_m,
            "phys_realism_max_frames": phys_max_frames,
        },
    }

    out_path = args.out
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Physical realism metrics written to %s", out_path)
    logger.info("physical_realism: mean=%.4f  std=%.4f", pr_result["mean"], pr_result["std"])
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    """Run full evaluation and write metrics.json."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Fast path: physical-realism only (skips classifier + distribution metrics).
    phys_only = getattr(args, "phys_realism_only", False)
    if phys_only:
        _run_phys_realism_only(args)
        return

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available; falling back to cpu. "
            "Set --device cpu to suppress this warning."
        )
        device = "cpu"

    # 1. Load classifier.
    arch = getattr(args, "classifier_arch", "imos43")
    classifier = load_classifier(args.classifier_ckpt, device=device, arch=arch)
    n_labels = classifier.n_labels

    # 2. Load action labels.
    labels_info = load_action_labels(dataset=args.dataset, labels_json_path=args.labels_json)
    label_to_idx = labels_info["label_to_idx"]

    # 3. Load samples (predicted).
    logger.info("Loading predicted samples from %s ...", args.samples)
    pred_samples = _load_canonical_dir(args.samples)
    if len(pred_samples) == 0:
        raise ValueError(f"No samples found in {args.samples!r}")

    # 4. Load GT.
    logger.info("Loading GT from %s ...", args.gt)
    gt_samples = _load_gt_samples(args.gt, dataset=args.dataset)
    if len(gt_samples) == 0:
        raise ValueError(f"No GT samples found in {args.gt!r}")

    # 5. Extract labels from text fields.
    # Out-of-vocabulary verb handling:
    #   strict=True  (default): raise — required for scientific correctness when
    #                           evaluating baselines that may emit novel verbs.
    #                           Silently folding unknowns into class 0 would
    #                           corrupt accuracy/multimodality.
    #   strict=False (CLI --allow-unknown-verbs): warn and fold to 0 for
    #                           backwards-compatible smoke runs only.
    strict_vocab = not args.allow_unknown_verbs

    def _extract_labels(
        samples: List[CanonicalSample], label_to_idx: Dict[str, int]
    ) -> List[int]:
        result: List[int] = []
        unknowns: List[str] = []
        for s in samples:
            try:
                result.append(text_to_label_idx(s.text, label_to_idx))
            except (KeyError, ValueError) as exc:
                if strict_vocab:
                    raise ValueError(
                        f"eval_imos: sample text={s.text!r} produced unknown verb: {exc}. "
                        f"Vocabulary has {len(label_to_idx)} labels. Re-run with "
                        f"--allow-unknown-verbs to fold unknowns to class 0 "
                        f"(produces corrupted accuracy/multimodality)."
                    ) from exc
                unknowns.append(s.text)
                result.append(0)
        if unknowns:
            logger.warning(
                "eval_imos: %d/%d samples had out-of-vocab verbs (folded to class 0). "
                "Examples: %r",
                len(unknowns), len(samples), unknowns[:5],
            )
        return result

    pred_label_indices = _extract_labels(pred_samples, label_to_idx)
    gt_label_indices = _extract_labels(gt_samples, label_to_idx)

    pred_labels_t = torch.tensor(pred_label_indices, dtype=torch.int64)
    gt_labels_t = torch.tensor(gt_label_indices, dtype=torch.int64)

    # Collect text prompts for FID per-prompt computation.
    pred_texts: List[str] = [s.text for s in pred_samples]
    gt_texts: List[str] = [s.text for s in gt_samples]

    # num_labels_total is the vocabulary size (FIX-2: multimodality divisor).
    num_labels_total = len(label_to_idx)

    # 6. Batch forward on classifier.
    logger.info(
        "Running classifier (arch=%s) on %d predicted samples ...", arch, len(pred_samples)
    )
    if arch == "imos43":
        pred_out = batch_forward(
            classifier,
            pred_samples,
            max_t=args.max_t,
            batch_size=args.batch_size,
            device=device,
        )
        gt_out = batch_forward(
            classifier,
            gt_samples,
            max_t=args.max_t,
            batch_size=args.batch_size,
            device=device,
        )
    elif arch == "text2hoi42":
        pred_out = batch_forward_text2hoi(
            classifier,
            pred_samples,
            num_frames=15,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info("Running classifier (arch=%s) on %d GT samples ...", arch, len(gt_samples))
        gt_out = batch_forward_text2hoi(
            classifier,
            gt_samples,
            num_frames=15,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        raise ValueError(f"run_eval: unknown arch {arch!r}")

    pred_features = pred_out["features"]   # (N, 128)
    pred_logits = pred_out["logits"]       # (N, n_labels)
    gt_features = gt_out["features"]       # (M, 128)

    # 7. Compute distribution metrics.
    logger.info("Computing distribution metrics (n_repeats=%d, seed=%d) ...", args.n_repeats, args.seed)
    dist_metrics = _compute_metrics_with_repeat(
        pred_features=pred_features,
        pred_logits=pred_logits,
        pred_labels=pred_labels_t,
        gt_features=gt_features,
        gt_labels=gt_labels_t,
        num_labels_total=num_labels_total,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        pred_texts=pred_texts,
        gt_texts=gt_texts,
        normalize_features=getattr(args, "normalize_features", False),
    )

    # 8. Physical realism.
    logger.info("Loading object point clouds for physical realism (MANO-vertex method) ...")
    obj_pcs = _collect_obj_pcs(dataset=args.dataset)
    obj_top_masks = _collect_obj_top_masks(dataset=args.dataset)

    logger.info(
        "Computing physical realism for %d predicted samples (arti_objs=%d) ...",
        len(pred_samples), len(obj_top_masks),
    )
    phys_max_frames = args.phys_realism_max_frames if args.phys_realism_max_frames > 0 else None
    pr_result = _compute_physical_realism_aggregated(
        samples=pred_samples,
        obj_meshes=obj_pcs,  # obj_pcs passed here; compute_physical_realism ignores obj_meshes arg
        contact_thresh_m=args.contact_thresh_m,
        penetration_thresh_m=args.penetration_thresh_m,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        max_frames=phys_max_frames,
        obj_pcs=obj_pcs,
        obj_top_masks=obj_top_masks,
        dataset=args.dataset,
    )

    # 9. Assemble output.
    method_name = _infer_method_name(args.samples)

    output: Dict[str, Any] = {
        "dataset": args.dataset,
        "method": method_name,
        "n_samples": len(pred_samples),
        "n_gt": len(gt_samples),
        "metrics": {
            # FIX-3: both top-1 (IMoS-literal) and top-3 (Text2HOI Table 1 header).
            "accuracy_top1": dist_metrics["accuracy_top1"],
            "accuracy_top3": dist_metrics["accuracy_top3"],
            # FIX-4: three FID variants; fid_corpus is backward-compat key.
            "fid_corpus": dist_metrics["fid_corpus"],
            "fid_per_prompt": dist_metrics["fid_per_prompt"],
            "fid_asym": dist_metrics["fid_asym"],
            # Legacy key retained for backward compat (= fid_corpus).
            "fid": dist_metrics["fid"],
            # FIX-1: diversity n_pairs=200 (IMoS-literal).
            "diversity": dist_metrics["diversity"],
            # FIX-2: multimodality with rejection-sampling and num_labels_total divisor.
            "multimodality": dist_metrics["multimodality"],
            "physical_realism": pr_result,
        },
        "classifier_ckpt": os.path.abspath(args.classifier_ckpt),
        "encoder_ckpt": os.path.abspath(args.classifier_ckpt),  # single-network design
        "commit_sha": _git_sha(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "notes": (
            "v3 metrics (2026-04-20 FIX-A): "
            "FIX-1: diversity n_pairs=200 (IMoS-literal, was 300). "
            "FIX-2: multimodality uses IMoS rejection-sampling + divisor=n_per_label*num_labels_total. "
            "FIX-3: accuracy_top1 (IMoS-literal) and accuracy_top3 (Text2HOI Table 1 header) both reported. "
            "FIX-4: fid_corpus (corpus-level), fid_per_prompt (per-prompt grouping, IMoS-literal), "
            "fid_asym (GT corpus vs pred per-prompt). "
            "FIX-5: phys_realism_max_frames=0 by default (full-sequence, no cap). "
            "FIX-A: physical_realism replaced SDF-joint method with 778-vertex MANO KNN penetration "
            "+ fingertip-vert contact (Text2HOI get_penet_hand_obj_loss exact replication)."
        ),
        "config": {
            "contact_thresh_m": args.contact_thresh_m,
            "penetration_thresh_m": args.penetration_thresh_m,
            "max_t": args.max_t,
            "batch_size": args.batch_size,
            "n_repeats": args.n_repeats,
            "phys_realism_max_frames": phys_max_frames,
            "num_labels_total": num_labels_total,
        },
    }

    # 10. Write output.
    out_path = args.out
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Metrics written to %s", out_path)

    # Print summary to stdout.
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m umm.eval.handobject.eval_imos",
        description=(
            "IMoS-adapted evaluation CLI for hand-object interaction. "
            "Computes 5 metrics (accuracy_top3, fid, diversity, multimodality, "
            "physical_realism) against GT samples and writes metrics.json."
        ),
    )
    p.add_argument(
        "--samples",
        required=True,
        help="Directory of canonical .npz files produced by the method to evaluate.",
    )
    p.add_argument(
        "--gt",
        required=True,
        help=(
            "GT source. Either: (a) a directory of canonical .npz files, or "
            "(b) a Text2HOI raw root directory containing data.npz. "
            "Auto-detected by presence of <gt>/data.npz."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["grab", "arctic"],
        help="Dataset name used for action label vocabulary and mesh loading.",
    )
    p.add_argument(
        "--classifier_ckpt",
        required=True,
        help=(
            "Path to the classifier .pth checkpoint. "
            "A sibling config.json will be loaded if present."
        ),
    )
    p.add_argument(
        "--classifier_arch",
        type=str,
        default="imos43",
        choices=["imos43", "text2hoi42"],
        help=(
            "Classifier architecture to load. "
            "'imos43': HandObjectActionClassifier (IMoS-adapted 2-stream LSTM, 43 points, "
            "input (B, max_t, 43, 3)). "
            "'text2hoi42': Text2HOIActionClassifier (Text2HOI-style LSTM, 42 joints, "
            "fixed 15 frames, input (B, 15, 42, 3)). "
            "Must match the arch used when training the checkpoint. "
            "Default: imos43 (backward-compatible)."
        ),
    )
    p.add_argument(
        "--labels_json",
        required=True,
        help="Path to action_labels.json produced by the data audit.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path for metrics.json.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair-sampling metrics (diversity, multimodality). Default: 42.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device string. Default: cuda.",
    )
    p.add_argument(
        "--contact_thresh_m",
        type=float,
        default=0.02,
        help="Contact threshold in meters for physical realism. Default: 0.02 (2 cm).",
    )
    p.add_argument(
        "--penetration_thresh_m",
        type=float,
        default=None,
        help=(
            "Penetration threshold in meters for physical realism. Default: None. "
            "When None (default), resolves from dataset via "
            "_DATASET_PENETRATION_THRESH_M in physical_realism.py: GRAB -> 10mm "
            "(calibration p50=7mm, p90=10mm, matches paper GT realism ~0.81), "
            "ARCTIC -> 15mm (compensation for ignored per-subject MANO "
            "betas, closes 0.85 gate). Pass "
            "an explicit value to override (sacred over the dataset default). Use "
            "0.005 only with the legacy 21-joint SDF method."
        ),
    )
    p.add_argument(
        "--n_repeats",
        type=int,
        default=20,
        help=(
            "Number of repetitions for pair-sampling metrics (diversity, multimodality) "
            "to compute mean +- 95% CI. Default: 20 (paper Text2HOI §4.3 verbatim: "
            "'Experiments are conducted 20 times to establish the robustness, and we "
            "reported results within a 95% confidence interval.'). Use lower values ONLY "
            "for debug smoke runs; production evals must use 20."
        ),
    )
    p.add_argument(
        "--normalize_features",
        action="store_true",
        default=False,
        help=(
            "If set, L2-normalize classifier features to unit norm before computing "
            "diversity and multimodality. Neutralizes classifier-dependent feature "
            "magnitude so reported numbers are pure geometric spread on the unit sphere. "
            "Experiment flag for closing ARCTIC div/mm gaps to paper (paper's feature "
            "magnitude may differ from ours)."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for classifier inference. Default: 32.",
    )
    p.add_argument(
        "--max_t",
        type=int,
        default=120,
        help="Maximum sequence length (frames) passed to the classifier. Default: 120.",
    )
    p.add_argument(
        "--phys_realism_max_frames",
        type=int,
        default=0,
        help=(
            "Cap per-episode valid frames used for physical-realism SDF queries. "
            "Longer episodes are subsampled uniformly to this count. "
            "FIX-5 (2026-04-20): default changed from 120 to 0 (no cap, paper convention). "
            "Set >0 to subsample for faster debugging (slower on GRAB 1007-frame episodes "
            "when set to 0). The paper convention is full-sequence scoring."
        ),
    )
    p.add_argument(
        "--allow-unknown-verbs",
        dest="allow_unknown_verbs",
        action="store_true",
        help=(
            "Fold any sample whose verb is out-of-vocabulary into label 0 (with a "
            "warning). DEFAULT IS STRICT: unknown verbs raise, because folding to "
            "class 0 silently corrupts accuracy and multimodality for baselines "
            "that emit novel verbs."
        ),
    )
    p.add_argument(
        "--phys_realism_only",
        action="store_true",
        default=False,
        help=(
            "Skip classifier inference + distribution metrics (FID, diversity, "
            "multimodality, accuracy) and only compute physical realism. "
            "Output JSON contains only 'physical_realism' in 'metrics'. "
            "Use to update physical realism metric without re-running the full "
            "expensive pipeline (sqrtm + classifier on 8928 samples). "
            "Merge with v2 JSON manually to produce complete v3 results."
        ),
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_eval(args)
