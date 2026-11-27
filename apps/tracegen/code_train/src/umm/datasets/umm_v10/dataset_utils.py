"""Shared utility functions for UMM V10 dataset loading and preprocessing.

Self-contained copy — NO imports from V9 (module independence).

Provides helpers for:
- Array resampling (:func:`sample_list`)
- Dataset I/O (:func:`load_indices`, :func:`load_language`)
"""

import json
import logging
import numpy as np

import cv2
cv2.setNumThreads(0)


def sample_list(x, M, random_flag=True):
    """Resample an array to exactly ``M`` elements along the first axis.

    If ``len(x) < M``, elements are duplicated. If ``len(x) > M``, elements
    are subsampled. If ``len(x) == M``, returned as-is.

    Args:
        x: Input array of shape ``(N, ...)``.
        M: Target number of elements.
        random_flag: If True, use random sampling; otherwise deterministic.

    Returns:
        Array of shape ``(M, ...)`` with the same dtype as *x*.
    """
    len_x = len(x)
    if len_x < M:
        if random_flag:
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            repeat_times = (M + len_x - 1) // len_x
            repeated = np.tile(x, (repeat_times,) + (1,) * (x.ndim - 1))
            return repeated[:M]
    elif len_x > M:
        if random_flag:
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            indices = np.linspace(0, len_x - 1, M, dtype=np.int32)
            return x[indices]
    else:
        return x


def load_indices(idx_fn, t_rate):
    """Load dataset split indices from a JSON file and expand by temporal rate.

    Args:
        idx_fn: Path to a JSON file with entries ``[[sid, length], ...]``.
        t_rate: Temporal stride; every ``t_rate``-th frame becomes an index.

    Returns:
        Tuple of ``(indices, seq_len)`` where ``indices`` is a list of
        ``(sid, tid)`` tuples and ``seq_len`` maps sid -> total frame count.
    """
    raw_indices = json.load(open(idx_fn, "r"))
    indices, seq_len = [], {}
    for sid, length in raw_indices:
        length = int(length)
        seq_len[sid] = length
        for tid in range(0, length, t_rate):
            indices.append((sid, tid))
    return indices, seq_len


def load_language(language_fn):
    """Load language annotations from a JSON file.

    Supports ARM4R format (list of dicts) and simple format (dict).

    Args:
        language_fn: Path to JSON file, or None to skip loading.

    Returns:
        Dict mapping sequence IDs to instruction strings, or None.
    """
    if language_fn is None:
        return None
    language_data = json.load(open(language_fn, "r"))
    try:
        language_data = {it["clip_id"]: it["instruction"] for it in language_data}
    except Exception:
        language_data = language_data
    logging.info(f"Loaded language data from {language_fn}")
    return language_data
