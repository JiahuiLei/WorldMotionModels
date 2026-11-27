#!/usr/bin/env python3
"""Compute local SE(3) frames for TraceGen sequences.

Walks ``data_wmm/tracegen/data/<env>/<split>/<episode>/samples/*.npz``
and writes a sidecar tree under ``data_wmm/tracegen_localframes/``
(or any --out-root). The TraceGen source tree is never touched.

Per source clip, packs into the (K, T, 6) format
``[u, v, x, y, z, vis]`` and runs
``prepare_data.local_frames.compute_local_frames``.

TraceGen schema (per clip, K=400, T=128):
    image_path  : (1,)  str   relative path to first-frame PNG
    frame_index : (1,)  int64
    keypoints   : (K, 2)      first-frame pixel UV
    traj        : (K, T, 3)   per-frame (u, v, depth) — pixel u,v + depth_m;
                              -inf padded outside ``valid_steps``.
                              NB: u,v are CLAMPED to image edges for off-screen
                              points, with depth sampled at the clamped pixel
                              (a property of the TraceGen data).
    traj_2d     : (K, 8, 2)   keyframe pixel UV (unused here)
    valid_steps : (T,)  bool  per-frame validity (NO per-track signal)

Visibility is fabricated as
    vis[k, t] = isfinite(traj[k, t]).all() & valid_steps[t]
The Procrustes / Kabsch math in ``compute_local_frames`` gates neighbor
selection, co-visibility, and the identity-anchor by ``vis`` — so off-screen
garbage positions are excluded automatically.

Coordinate frame: TraceGen has NO camera extrinsics, so the resulting R_t
is in the same camera-relative ``(u_norm, v_norm, depth_m)`` frame the model
already trains translations in. Ego-motion and object motion are entangled in
R_t; this is the same dataset noise floor that already affects translation
supervision. Document this; do not try to "fix" by guessing extrinsics.

Output (sidecar mirroring source layout):
    {out_root}/<env>/<split>/<episode>/<stem>.npz
        rotations  : (K, T, 3) float32   axis-angle, identity at first vis
        rot_valid  : (K, T)    bool      Procrustes had >= min_neighbors co-vis
        nn_indices : (K, N)    int32     neighbor indices (debug / viewer)
        method     : ()        str       'svd' | 'irls' | 'windowed'
        params     : ()        str       JSON-encoded {K_spatial, N_final, ...}

Usage (from release_data/tracegen/L3_train/):
    # Single clip (debug)
    python <repo>/apps/tracegen/prepare_data/compute_local_frames.py \\
        --src-root data_wmm/tracegen/data \\
        --out-root data_wmm/tracegen_localframes \\
        --single data_wmm/tracegen/data/Droid/droid_test/episode_9344/samples/episode_9344_14.npz

    # All Droid + Epic, every split (default), 16 workers
    python <repo>/apps/tracegen/prepare_data/compute_local_frames.py \\
        --src-root data_wmm/tracegen/data \\
        --out-root data_wmm/tracegen_localframes \\
        --workers 16

    # Specific envs/splits
    python <repo>/apps/tracegen/prepare_data/compute_local_frames.py \\
        --src-root data_wmm/tracegen/data \\
        --out-root data_wmm/tracegen_localframes \\
        --envs Droid \\
        --splits droid_test \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Ensure repo root on path so we can import prepare_data.local_frames
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prepare_data.local_frames import compute_local_frames  # noqa: E402

logger = logging.getLogger("tracegen_localframes")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# Min number of valid frames in a clip for local-frame computation to be
# meaningful. compute_local_frames sets min_covis = max(3, T // 10) internally;
# below ~5 valid frames there is essentially no signal.
_MIN_VALID_FRAMES = 5

# Internal sentinel used in the (rel, ok, info) tuple to flag "skipped"
# clips. Must not appear in any user-facing error message — exception
# messages are emitted unmodified by _worker_safe.
_SKIP = "<skip>:"


def _load_image_size(ep_dir: Path, image_rel: str) -> Tuple[int, int]:
    """Return (orig_w, orig_h) for the clip's first-frame PNG.

    image_rel is the npz field ``image_path`` (e.g. ``images/episode_9344_14.png``).
    Path is resolved relative to ``ep_dir``.
    """
    img_path = ep_dir / image_rel
    if not img_path.exists():
        # Fallback: bare filename under images/
        img_path = ep_dir / "images" / Path(image_rel).name
    with Image.open(img_path) as im:
        return im.size  # PIL returns (W, H)


def tracegen_npz_to_packed(
    npz_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Load a TraceGen clip npz and pack into (K, T, 6) tracks + (K, T) vis.

    Returns (tracks, vis, T_valid) or (None, None, 0) if the clip has too few
    valid frames for local-frame computation.

    tracks columns: [u_norm, v_norm, x_norm, y_norm, depth, vis_float]
        u_norm, v_norm: pixel UV normalized to [0, 1] by orig image dims.
        x_norm, y_norm: same as u_norm, v_norm — passed as the "world xy"
            consumed by compute_local_frames. TraceGen has no proper backproject;
            this hybrid (u_norm, v_norm, depth_m) is the same frame the model
            trains translations in (see tg_dataset.py: _prepare_gt_tensors).
        depth: raw meters from the npz, unchanged.
    """
    ep_dir = npz_path.parent.parent  # samples/<stem>.npz -> episode dir
    with np.load(npz_path, allow_pickle=False) as d:
        traj = d["traj"].astype(np.float32)              # (K, T, 3) -inf padded
        valid_steps = d["valid_steps"].astype(bool)      # (T,)
        image_rel = str(d["image_path"][0])

    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(
            f"{npz_path}: expected traj shape (K, T, 3); got {traj.shape}. "
            "TraceGen 2D-only fallback (last dim == 2) is not supported here — "
            "local frames need 3D positions."
        )

    T_valid = int(valid_steps.sum())
    if T_valid < _MIN_VALID_FRAMES:
        return None, None, T_valid

    # Per-track per-frame visibility: finite traj AND inside valid_steps window.
    # TraceGen has no per-track signal — vis_per_track varies only via -inf check
    # which (see the note above) is perfectly correlated with valid_steps.
    vis = np.isfinite(traj).all(axis=-1) & valid_steps[None, :]  # (K, T) bool

    # Normalize pixel UV by ORIGINAL image dims so x,y are in [0, 1] —
    # matches the frame `tg_dataset.py:436-440` puts gt_trajectory in.
    orig_w, orig_h = _load_image_size(ep_dir, image_rel)

    traj_norm = traj.copy()
    traj_norm[..., 0] /= float(orig_w)
    traj_norm[..., 1] /= float(orig_h)
    # depth (col 2) stays raw meters — also matches tg_dataset.py behavior.

    # Zero out -inf positions so SVD never sees non-finite values. The vis mask
    # ensures these are never used as neighbors or in Procrustes anyway.
    traj_clean = np.where(vis[..., None], traj_norm, 0.0).astype(np.float32)

    # Pack as (K, T, 6) format.
    # u, v cols (0-1) are not used by compute_local_frames math (only carried
    # for color sampling in viewers). Reuse normalized x, y here.
    uv = traj_clean[..., :2]                              # (K, T, 2)
    xyz = traj_clean                                       # (K, T, 3)
    vis_f = vis[..., None].astype(np.float32)              # (K, T, 1)
    tracks = np.concatenate([uv, xyz, vis_f], axis=-1).astype(np.float32)
    return tracks, vis, T_valid


def _matrices_to_axisangle(
    rot_mats: np.ndarray, rot_valid: np.ndarray
) -> np.ndarray:
    """Convert (K, T, 3, 3) rotations to (K, T, 3) axis-angle, zero on invalid."""
    K, T = rot_valid.shape
    aa = np.zeros((K, T, 3), dtype=np.float32)
    flat_valid = rot_valid.ravel()
    if not flat_valid.any():
        return aa
    flat_R = rot_mats.reshape(-1, 3, 3)
    valid_aa = R.from_matrix(flat_R[flat_valid]).as_rotvec().astype(np.float32)
    aa.reshape(-1, 3)[flat_valid] = valid_aa
    return aa


# ---------------------------------------------------------------------------
# Per-clip worker
# ---------------------------------------------------------------------------


def _mirror_path(src: Path, src_root: Path, out_root: Path) -> Path:
    """Map source <src_root>/<env>/<split>/<ep>/samples/<stem>.npz
    to       <out_root>/<env>/<split>/<ep>/<stem>.npz.

    Only the *immediate parent* segment named ``'samples'`` is collapsed
    (i.e. parts[-2]). Other segments named 'samples' (env / split / episode
    named 'samples') are preserved to avoid silent path collisions.
    """
    rel = src.relative_to(src_root)
    parts = list(rel.parts)
    if len(parts) >= 2 and parts[-2] == "samples":
        parts = parts[:-2] + parts[-1:]
    return out_root.joinpath(*parts)


def process_clip(
    npz_path: Path,
    src_root: Path,
    out_root: Path,
    method: str,
    K_spatial: int,
    N_final: int,
    min_neighbors: int,
    force: bool,
    **method_kwargs,
) -> Tuple[str, bool, str]:
    """Compute local frames for one TraceGen clip.

    Returns (rel_path, success, info-or-error).
    """
    rel = str(npz_path.relative_to(src_root))
    out_path = _mirror_path(npz_path, src_root, out_root)

    if out_path.exists() and not force:
        return (rel, True, _SKIP + "exists")

    try:
        t0 = time.time()
        tracks, _, T_valid = tracegen_npz_to_packed(npz_path)
        if tracks is None:
            return (rel, True, _SKIP + f"T_valid={T_valid} < {_MIN_VALID_FRAMES}")

        K, T = tracks.shape[:2]
        result = compute_local_frames(
            tracks,
            method=method,
            K_spatial=K_spatial,
            N_final=N_final,
            min_neighbors=min_neighbors,
            **method_kwargs,
        )

        rot_aa = _matrices_to_axisangle(result.rotations, result.rot_valid)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            rotations=rot_aa,                              # (K, T, 3) f32
            rot_valid=result.rot_valid,                    # (K, T) bool
            nn_indices=result.nn_indices,                  # (K, N_final) i32
            method=np.array(result.method),                # str
            params=np.array(json.dumps(result.params)),    # str
        )

        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / 1024 / 1024
        valid_frac = float(result.rot_valid.mean())
        return (
            rel,
            True,
            f"K={K} T={T} T_valid={T_valid} valid={valid_frac:.2f} "
            f"{elapsed:.1f}s {size_mb:.2f}MB",
        )
    except Exception as e:  # pragma: no cover — surface in CLI
        return (rel, False, f"{e.__class__.__name__}: {e}")


def _worker_safe(
    npz_path: Path, **kwargs
) -> Tuple[str, bool, str]:
    try:
        return process_clip(npz_path, **kwargs)
    except Exception as e:  # pragma: no cover
        return (str(npz_path), False, f"{e.__class__.__name__}: {e}")


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def discover_clips(
    src_root: Path,
    envs: Optional[List[str]],
    splits: Optional[List[str]],
) -> List[Path]:
    """Walk src_root and return all per-clip sample npz paths.

    Pattern: ``<src_root>/<env>/<split>/<episode>/samples/*.npz``.
    Excludes ``depth/*.npz`` and any path not matching the pattern.
    """
    clips: List[Path] = []
    if not src_root.exists():
        raise FileNotFoundError(f"src_root does not exist: {src_root}")

    env_dirs = (
        [src_root / e for e in envs]
        if envs
        else sorted(d for d in src_root.iterdir() if d.is_dir())
    )

    for env_dir in env_dirs:
        if not env_dir.is_dir():
            logger.warning("env dir missing: %s", env_dir)
            continue

        split_dirs = (
            [env_dir / s for s in splits]
            if splits
            else sorted(d for d in env_dir.iterdir() if d.is_dir())
        )

        for split_dir in split_dirs:
            if not split_dir.is_dir():
                continue
            for ep_dir in sorted(split_dir.iterdir()):
                samples_dir = ep_dir / "samples"
                if not samples_dir.is_dir():
                    continue
                for npz in sorted(samples_dir.glob("*.npz")):
                    clips.append(npz)

    return clips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute local SE(3) frames for TraceGen clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="TraceGen source data root, e.g. data_wmm/tracegen/data",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output sidecar root, e.g. data_wmm/tracegen_localframes",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Process only this single npz path (debug). Overrides --envs/--splits.",
    )
    parser.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help="Envs to process (e.g. Droid Epickitchen). Default: all dirs under src-root.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to process (e.g. droid droid_test). Default: all subdirs of each env.",
    )

    parser.add_argument("--method", "-m", default="svd",
                        choices=["svd", "irls", "windowed"])
    parser.add_argument("--K", "--K-spatial", dest="K_spatial",
                        type=int, default=32,
                        help="Initial spatial neighbor candidates (default: 32)")
    parser.add_argument("--N", "--N-final", dest="N_final",
                        type=int, default=8,
                        help="Final neighbor count (default: 8)")
    parser.add_argument("--min-neighbors", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=3,
                        help="IRLS iterations (method=irls). Must be >= 1.")
    parser.add_argument("--half-window", type=int, default=2,
                        help="Temporal half-window (method=windowed). Must be >= 1.")

    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output exists")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    method_kwargs = {}
    if args.method == "irls":
        if args.n_iter < 1:
            parser.error("--n-iter must be >= 1 (upstream IRLS does not handle 0)")
        method_kwargs["n_iter"] = args.n_iter
    elif args.method == "windowed":
        if args.half_window < 1:
            parser.error("--half-window must be >= 1")
        method_kwargs["half_window"] = args.half_window

    if args.single:
        clips = [Path(args.single).expanduser().resolve()]
    else:
        clips = discover_clips(src_root, args.envs, args.splits)
    logger.info("src_root=%s out_root=%s", src_root, out_root)
    logger.info("Method=%s K_spatial=%d N_final=%d min_neighbors=%d",
                args.method, args.K_spatial, args.N_final, args.min_neighbors)
    logger.info("Discovered %d clips, workers=%d", len(clips), args.workers)
    if not clips:
        logger.warning("No clips found — check --src-root, --envs, --splits.")
        return

    common_kwargs = dict(
        src_root=src_root,
        out_root=out_root,
        method=args.method,
        K_spatial=args.K_spatial,
        N_final=args.N_final,
        min_neighbors=args.min_neighbors,
        force=args.force,
        **method_kwargs,
    )
    worker_fn = partial(_worker_safe, **common_kwargs)

    succeeded = 0
    skipped = 0
    failed = 0
    failures: List[Tuple[str, str]] = []
    t0 = time.time()

    def _consume(rel: str, ok: bool, info: str) -> None:
        nonlocal succeeded, skipped, failed
        # Order matters: a successful clip whose msg happens to contain "skip"
        # must NOT be miscounted, and an exception with a "<skip>:" substring
        # in its message must NOT either. Decide by `ok` first, then sentinel.
        if not ok:
            failed += 1
            failures.append((rel, info))
            logger.error("FAIL %s: %s", rel, info)
            return
        if info.startswith(_SKIP):
            skipped += 1
            logger.info("SKIP %s: %s", rel, info[len(_SKIP):].lstrip())
        else:
            succeeded += 1
            logger.info("OK   %s: %s", rel, info)

    if args.workers <= 1:
        for npz in clips:
            _consume(*worker_fn(npz))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for rel, ok, info in pool.imap_unordered(worker_fn, clips):
                _consume(rel, ok, info)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Total: %d ok, %d skip, %d fail in %.1fs (%d clips, %d workers)",
        succeeded, skipped, failed, elapsed, len(clips), args.workers,
    )
    if failures:
        logger.info("Failures (first 20):")
        for rel, err in failures[:20]:
            logger.info("  %s: %s", rel, err)


if __name__ == "__main__":
    main()
