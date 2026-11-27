"""Shared utility functions for UMM dataset loading and preprocessing.

Provides helpers for:
- Array resampling (:func:`sample_list`)
- Image operations (:func:`backproject`, :func:`resize_to_target_with_crop`, :func:`fetch_uid`)
- Dataset I/O (:func:`load_indices`, :func:`load_language`)
- Performance profiling (:class:`DatasetProfiler`)
"""

import numpy as np
import cv2
import time
import json
import logging

cv2.setNumThreads(0)


def sample_list(x, M, random_flag=True):
    """Resample an array to exactly ``M`` elements along the first axis.

    If ``len(x) < M``, elements are duplicated (random with replacement or
    tiled deterministically).  If ``len(x) > M``, elements are subsampled
    (random or evenly spaced).  If ``len(x) == M``, ``x`` is returned as-is.

    Args:
        x: Input array of shape ``(N, ...)``.
        M: Target number of elements.
        random_flag: If True, use random sampling; otherwise deterministic.

    Returns:
        Array of shape ``(M, ...)`` with the same dtype as *x*.

    Example::

        pts = sample_list(np.random.randn(100, 3), 64, random_flag=True)
        assert pts.shape == (64, 3)
    """
    # x: N,C
    # M: int
    len_x = len(x)
    if len_x < M:
        if random_flag:
            # Random sample with replacement
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            # Repeat to reach M samples
            repeat_times = (M + len_x - 1) // len_x  # ceil division
            repeated = np.tile(x, (repeat_times,) + (1,) * (x.ndim - 1))
            return repeated[:M]
    elif len_x > M:
        if random_flag:
            # Random sample with replacement
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            # Uniform sampling
            indices = np.linspace(0, len_x - 1, M, dtype=np.int32)
            return x[indices]
    else:
        return x


def fetch_uid(uv, uv_mask, uid_buffer):
    """Look up unique IDs from an instance segmentation buffer at UV coordinates.

    Args:
        uv: Pixel coordinates of shape ``(N, 2)`` in ``(u=col, v=row)`` order.
        uv_mask: Boolean validity mask of shape ``(N,)``.
        uid_buffer: Instance segmentation map of shape ``(H, W)``.

    Returns:
        Tuple of ``(uids, valid_mask)`` where ``uids`` is ``(N,)`` integer
        array of fetched IDs (0 for invalid) and ``valid_mask`` is the
        updated boolean mask.

    Example::

        uids, mask = fetch_uid(uv_coords, mask, inst_seg_map)
    """
    # uv: N,2
    # uv_mask: N
    # uid_buffer: H,W
    H, W = uid_buffer.shape
    uv_valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv_mask = uv_mask.copy() & uv_valid_mask
    uv = uv.astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)

    buffer = uid_buffer.reshape(-1)
    index = uv[:, 1] * W + uv[:, 0]
    ret = buffer[index]
    ret[~uv_mask] = 0
    return ret, uv_mask


def backproject(depth, cam_K):
    """Backproject a depth map to 3D points in camera coordinates.

    Args:
        depth: Depth image of shape ``(H, W)``.
        cam_K: Camera intrinsic matrix of shape ``(3, 3)``.

    Returns:
        Point cloud of shape ``(H, W, 3)`` in camera frame (X right, Y down, Z forward).

    Example::

        xyz_cam = backproject(depth_map, K)  # shape (H, W, 3)
    """
    # depth: H,W
    # cam_K: 3,3
    # return: H,W,3
    H, W = depth.shape
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Backproject to 3D
    z = depth
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    # Stack to get 3D points
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz


def resize_to_target_with_crop(
    image_ndarray, H, W, K=None, uv=None, resize_mode="nearest"
):
    """
    Resize image to fit target size and crop to exact target dimensions.

    Args:
        image_ndarray: Input image of shape (H_orig, W_orig, C)
        H: Target height
        W: Target width
        K: Camera intrinsic matrix of shape (3, 3) or None
        uv: Pixel coordinates of shape (..., 2) in OpenCV convention (u=col, v=row) or None
        resize_mode: Resize interpolation mode ("bilinear" or "nearest")

    Returns:
        resized_cropped_image: Image of shape (H, W, C)
        adjusted_K: Adjusted camera matrix or None
        adjusted_uv: Adjusted pixel coordinates or None
    """
    # Get original dimensions
    H_orig, W_orig = image_ndarray.shape[:2]

    # Calculate scale factor to fit the target size (maintain aspect ratio)
    scale_h = H / H_orig
    scale_w = W / W_orig
    scale = max(scale_h, scale_w)  # Use max to ensure image covers target area

    # Calculate new dimensions after scaling
    new_H = int(H_orig * scale)
    new_W = int(W_orig * scale)

    # Resize image
    if resize_mode == "bilinear":
        print("Warning, DO NOT USE ANY INTERP OTHER THAN NEAREST!!!")
        interpolation = cv2.INTER_LINEAR
    elif resize_mode == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        print("Warning, DO NOT USE ANY INTERP OTHER THAN NEAREST!!!")
        interpolation = cv2.INTER_LINEAR

    resized_image = cv2.resize(
        image_ndarray.copy(), (new_W, new_H), interpolation=interpolation
    )

    # Calculate crop offsets to center the crop
    crop_x = (new_W - W) // 2
    crop_y = (new_H - H) // 2

    # Crop to target size
    cropped_image = resized_image[crop_y : crop_y + H, crop_x : crop_x + W]

    # Adjust camera matrix K if provided
    adjusted_K = None
    if K is not None:
        adjusted_K = K.copy()
        # Scale the intrinsic parameters
        adjusted_K[0, 0] *= scale  # fx
        adjusted_K[1, 1] *= scale  # fy
        adjusted_K[0, 2] = adjusted_K[0, 2] * scale - crop_x  # cx
        adjusted_K[1, 2] = adjusted_K[1, 2] * scale - crop_y  # cy

    # Adjust pixel coordinates uv if provided
    adjusted_uv = None
    if uv is not None:
        adjusted_uv = uv.copy()
        # Scale coordinates
        adjusted_uv[..., 0] = adjusted_uv[..., 0] * scale - crop_x  # u (horizontal)
        adjusted_uv[..., 1] = adjusted_uv[..., 1] * scale - crop_y  # v (vertical)

    return cropped_image, adjusted_K, adjusted_uv


class DatasetProfiler:
    """Profiler for dataset loading operations with per-section timing stats.

    Usage example::

        profiler = DatasetProfiler()
        with profiler.time_section("io/rgb"):
            img = load_rgb(path)
        profiler.print_summary()
        summary = profiler.get_summary_dict()
    """

    IO_PREFIXES = ("io/",)
    COMPUTE_PREFIXES = ("compute/",)

    def __init__(self):
        """Initialize empty timing dictionaries."""
        self.times: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def time_section(self, section_name: str) -> "_TimingContext":
        """Return a context manager that records elapsed time for *section_name*."""
        return self._TimingContext(self, section_name)

    class _TimingContext:
        """Context manager that records wall-clock time for a named section."""

        def __init__(self, profiler: "DatasetProfiler", section_name: str):
            """Initialize with parent profiler and section name."""
            self.profiler = profiler
            self.section_name = section_name
            self.start_time: float | None = None

        def __enter__(self) -> "DatasetProfiler._TimingContext":
            """Record start time."""
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Record elapsed time into the profiler's accumulator."""
            elapsed = time.time() - (self.start_time or 0)
            if self.section_name not in self.profiler.times:
                self.profiler.times[self.section_name] = []
                self.profiler.counts[self.section_name] = 0
            self.profiler.times[self.section_name].append(elapsed)
            self.profiler.counts[self.section_name] += 1

    def _section_stats(self, times_list: list[float]) -> dict:
        """Compute aggregate statistics for a list of recorded times.

        Args:
            times_list: List of elapsed times in seconds.

        Returns:
            Dict with ``total``, ``avg``, ``min``, ``max``, ``std``, ``count``.
        """
        arr = np.array(times_list)
        return {
            "total": float(np.sum(arr)),
            "avg": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
            "count": len(arr),
        }

    def print_summary(self) -> None:
        """Print a detailed profiling summary with min/max/std and % of total."""
        grand_total = sum(np.sum(t) for t in self.times.values())

        io_total = 0.0
        compute_total = 0.0

        print("\n" + "=" * 80)
        print("DATASET LOADING PROFILER SUMMARY")
        print("=" * 80)
        header = f"{'Section':<35} {'avg':>7} {'min':>7} {'max':>7} {'std':>7} {'total':>9} {'%':>6} {'calls':>6}"
        print(header)
        print("-" * 80)

        for section in sorted(self.times.keys()):
            s = self._section_stats(self.times[section])
            pct = (s["total"] / grand_total * 100) if grand_total > 0 else 0.0
            row = (
                f"{section:<35} "
                f"{s['avg']*1000:>6.1f}ms "
                f"{s['min']*1000:>6.1f}ms "
                f"{s['max']*1000:>6.1f}ms "
                f"{s['std']*1000:>6.1f}ms "
                f"{s['total']*1000:>8.1f}ms "
                f"{pct:>5.1f}% "
                f"{s['count']:>6d}"
            )
            print(row)
            if any(section.startswith(p) for p in self.IO_PREFIXES):
                io_total += s["total"]
            if any(section.startswith(p) for p in self.COMPUTE_PREFIXES):
                compute_total += s["total"]

        print("-" * 80)
        io_pct = (io_total / grand_total * 100) if grand_total > 0 else 0.0
        compute_pct = (compute_total / grand_total * 100) if grand_total > 0 else 0.0
        print(f"{'IO total':<35} {io_total*1000:>8.1f}ms  {io_pct:>5.1f}%")
        print(
            f"{'Compute total':<35} {compute_total*1000:>8.1f}ms  {compute_pct:>5.1f}%"
        )
        print(f"{'GRAND TOTAL':<35} {grand_total*1000:>8.1f}ms  100.0%")
        print("=" * 80 + "\n")

    def get_summary_dict(self) -> dict:
        """Return a JSON-serializable dict of all profiling stats.

        Returns a dict with keys:
          - ``sections``: per-section stats (total/avg/min/max/std/count/pct)
          - ``io_total_ms``, ``compute_total_ms``, ``grand_total_ms``
        """
        grand_total = sum(np.sum(t) for t in self.times.values())
        sections = {}
        io_total = 0.0
        compute_total = 0.0

        for section, times_list in self.times.items():
            s = self._section_stats(times_list)
            pct = (s["total"] / grand_total * 100) if grand_total > 0 else 0.0
            sections[section] = {
                k: v * 1000 if k not in ("count",) else v for k, v in s.items()
            }
            sections[section]["pct"] = float(pct)
            if any(section.startswith(p) for p in self.IO_PREFIXES):
                io_total += s["total"]
            if any(section.startswith(p) for p in self.COMPUTE_PREFIXES):
                compute_total += s["total"]

        return {
            "sections": sections,
            "io_total_ms": io_total * 1000,
            "compute_total_ms": compute_total * 1000,
            "grand_total_ms": grand_total * 1000,
        }

    def reset(self) -> None:
        """Reset all timing data."""
        self.times = {}
        self.counts = {}


def load_indices(idx_fn, t_rate):
    """Load dataset split indices from a JSON file and expand by temporal rate.

    The JSON file contains a list of ``[sequence_id, length]`` pairs.  This
    function generates ``(seq_id, time_idx)`` tuples stepping through each
    sequence at intervals of ``t_rate``.

    Args:
        idx_fn: Path to a JSON file with entries ``[[sid, length], ...]``.
        t_rate: Temporal stride; every ``t_rate``-th frame becomes an index.

    Returns:
        Tuple of ``(indices, seq_len)`` where ``indices`` is a list of
        ``(sid, tid)`` tuples and ``seq_len`` is a dict mapping sequence
        IDs to their total frame count.

    Example::

        indices, seq_len = load_indices("splits/train.json", t_rate=3)
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

    Supports two formats:
    - ARM4R format: list of dicts with ``"clip_id"`` and ``"instruction"`` keys.
    - Simple format: dict mapping sequence IDs to text strings.

    Args:
        language_fn: Path to JSON file, or None to skip loading.

    Returns:
        Dict mapping sequence IDs to instruction strings, or None if
        ``language_fn`` is None.

    Example::

        lang = load_language("data/language.json")
        text = lang["seq_001"]  # "pick up the cube"
    """
    if language_fn is None:
        return None
    language_data = json.load(open(language_fn, "r"))
    try:  # arm4r format
        language_data = {it["clip_id"]: it["instruction"] for it in language_data}
    except Exception:  # {sid: text}
        language_data = language_data
    logging.info(f"Loaded language data from {language_fn}")
    return language_data
