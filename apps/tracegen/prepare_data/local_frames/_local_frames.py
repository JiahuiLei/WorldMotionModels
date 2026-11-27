"""Compute local SE(3) frames from dense 3D trajectory neighborhoods.

Given packed tracks (N, T, 6) with [u, v, world_x, world_y, world_z, vis],
computes per-track per-frame rotation R_t in SO(3) by solving Procrustes
on local neighborhoods. The translation is the world-space xyz itself.

Three methods available:
  - svd: Batch Kabsch (standard SVD, fastest)
  - irls: IRLS-weighted Kabsch (Huber loss, robust to outlier neighbors)
  - windowed: Windowed Kabsch (temporal window, smooth + accurate)

All computation is CPU-only, fully vectorized with numpy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LocalFrameResult:
    """Output of local frame computation."""
    rotations: np.ndarray       # (N, T, 3, 3) float32 — absolute world-frame rotations
    rot_valid: np.ndarray       # (N, T) bool — True where rotation is valid
    nn_indices: np.ndarray      # (N, N_final) int32 — neighbor indices per track
    method: str                 # method name used
    params: dict                # parameters used


# ---------------------------------------------------------------------------
# Neighborhood computation
# ---------------------------------------------------------------------------

def compute_neighborhoods(
    xyz: np.ndarray,
    vis: np.ndarray,
    K_spatial: int = 64,
    N_final: int = 16,
) -> np.ndarray:
    """Find N_final nearest neighbors per track by curve distance.

    Two-phase approach:
    1. At a reference frame, find K_spatial spatial candidates via KD-tree
    2. Refine to N_final by curve distance (max Euclidean distance over
       co-visible frames)

    Args:
        xyz: (N, T, 3) float32 — world-space positions (NaN where invisible)
        vis: (N, T) bool — visibility mask
        K_spatial: initial spatial candidate count
        N_final: final neighbor count

    Returns:
        nn_indices: (N, N_final) int32 — neighbor indices per track.
                    -1 where fewer than N_final neighbors found.
    """
    N, T, _ = xyz.shape
    t0 = time.time()

    # Pick reference frame with most visible tracks
    ref_frame = int(vis.sum(axis=0).argmax())
    logger.info("Neighborhood: ref_frame=%d (%d visible tracks)", ref_frame, vis[:, ref_frame].sum())

    # Build position array: use ref_frame if visible, else first visible frame
    xyz_ref = xyz[:, ref_frame, :].copy()  # (N, 3)
    vis_ref = vis[:, ref_frame]

    first_vis = np.argmax(vis, axis=1)  # (N,)
    ever_visible = vis.any(axis=1)  # (N,)
    needs_fallback = ~vis_ref & ever_visible
    fallback_idx = np.where(needs_fallback)[0]
    for i in fallback_idx:
        xyz_ref[i] = xyz[i, first_vis[i]]

    # Phase 1: KD-tree spatial candidates
    valid_mask = ever_visible
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return np.full((N, N_final), -1, dtype=np.int32)

    tree = cKDTree(xyz_ref[valid_idx])
    K_query = min(K_spatial + 1, len(valid_idx))
    _, spatial_nn_local = tree.query(xyz_ref[valid_idx], k=K_query)
    # Map to global indices
    spatial_nn_global = valid_idx[spatial_nn_local]  # (n_valid, K_query)

    logger.info("Phase 1 (spatial KD-tree): %.2fs", time.time() - t0)
    t1 = time.time()

    # Phase 2: curve distance refinement — vectorized per candidate
    nn_result = np.full((N, N_final), -1, dtype=np.int32)
    min_covis = max(3, T // 10)

    for li, gi in enumerate(valid_idx):
        cands = spatial_nn_global[li]
        cands = cands[cands != gi]  # remove self
        if len(cands) == 0:
            continue

        # Co-visibility
        co_vis = vis[gi][None, :] & vis[cands]  # (K_cand, T)
        co_count = co_vis.sum(axis=1)

        # Curve distance = max distance over co-visible frames
        diff = xyz[cands] - xyz[gi][None, :]  # (K_cand, T, 3)
        dist_sq = (diff ** 2).sum(axis=-1)  # (K_cand, T)
        dist_sq = np.where(co_vis, dist_sq, 0.0)
        curve_dist = np.sqrt(dist_sq.max(axis=1))
        curve_dist[co_count < min_covis] = np.inf

        sorted_idx = np.argsort(curve_dist)
        n_take = min(N_final, int((curve_dist[sorted_idx] < np.inf).sum()))
        nn_result[gi, :n_take] = cands[sorted_idx[:n_take]]

    logger.info("Phase 2 (curve distance): %.2fs", time.time() - t1)
    logger.info("Neighborhood total: %.2fs", time.time() - t0)
    return nn_result


# ---------------------------------------------------------------------------
# Core: build per-frame cross-covariance matrices (vectorized)
# ---------------------------------------------------------------------------

def _build_frame_pair_H(
    xyz: np.ndarray,
    vis: np.ndarray,
    nn_indices: np.ndarray,
    t0: int,
    t1: int,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cross-covariance H for all tracks between frames t0 and t1.

    Returns:
        H: (N, 3, 3) float64 — cross-covariance per track
        valid: (N,) bool — True if track has enough co-visible neighbors
    """
    N = xyz.shape[0]
    K = nn_indices.shape[1]

    # Gather neighbor positions at t0 and t1: (N, K, 3)
    # Handle -1 indices by clamping to 0, then masking
    nn_clamped = np.maximum(nn_indices, 0)  # (N, K)
    nn_valid_mask = nn_indices >= 0  # (N, K)

    nb_xyz_t0 = xyz[nn_clamped, t0, :]  # (N, K, 3)
    nb_xyz_t1 = xyz[nn_clamped, t1, :]  # (N, K, 3)
    nb_vis_t0 = vis[nn_clamped, t0]     # (N, K)
    nb_vis_t1 = vis[nn_clamped, t1]     # (N, K)

    # Joint mask: valid neighbor AND visible at both frames
    joint = nn_valid_mask & nb_vis_t0 & nb_vis_t1  # (N, K)
    n_valid = joint.sum(axis=1)  # (N,)
    valid = n_valid >= min_neighbors  # (N,)

    # Zero out invalid positions
    joint_f = joint.astype(np.float32)  # (N, K)
    P = np.where(joint[:, :, None], nb_xyz_t0, 0.0)  # (N, K, 3)
    Q = np.where(joint[:, :, None], nb_xyz_t1, 0.0)  # (N, K, 3)

    # Compute means (only over valid neighbors)
    n_safe = np.maximum(n_valid, 1).astype(np.float32)[:, None, None]  # (N, 1, 1)
    P_mean = P.sum(axis=1, keepdims=True) / n_safe  # (N, 1, 3)
    Q_mean = Q.sum(axis=1, keepdims=True) / n_safe  # (N, 1, 3)

    # Center
    P_c = np.where(joint[:, :, None], P - P_mean, 0.0)  # (N, K, 3)
    Q_c = np.where(joint[:, :, None], Q - Q_mean, 0.0)  # (N, K, 3)

    # Cross-covariance: H[n] = P_c[n].T @ Q_c[n]
    H = np.einsum('nki,nkj->nij', P_c, Q_c)  # (N, 3, 3)

    return H, valid


def _batch_svd_to_rotation(H: np.ndarray) -> np.ndarray:
    """SVD on batch of 3x3 matrices → rotation matrices.

    Args:
        H: (B, 3, 3) — cross-covariance matrices

    Returns:
        R: (B, 3, 3) float32 — proper rotation matrices
    """
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(np.einsum('bij,bkj->bik', Vt, U))  # det(Vt.T @ U.T)
    D = np.ones((len(d), 3), dtype=H.dtype)
    D[:, -1] = np.sign(d)
    R = np.einsum('bji,bj,bkj->bik', Vt, D, U)  # (B, 3, 3)
    return R.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame-pair iteration with cumulation (shared by all methods)
# ---------------------------------------------------------------------------

def _cumulate_deltas(
    delta_R: np.ndarray,
    delta_valid: np.ndarray,
    vis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulate frame-pair delta rotations into absolute rotations.

    Args:
        delta_R: (N, T-1, 3, 3) — delta rotation from frame t to t+1
        delta_valid: (N, T-1) bool — whether the delta is valid
        vis: (N, T) bool — track visibility

    Returns:
        rotations: (N, T, 3, 3) float32 — absolute rotations
        rot_valid: (N, T) bool
    """
    N, T = vis.shape

    rotations = np.zeros((N, T, 3, 3), dtype=np.float32)
    rot_valid = np.zeros((N, T), dtype=bool)

    # First visible frame per track gets identity
    # Find first visible frame for each track
    has_any = vis.any(axis=1)  # (N,)
    first_vis = np.argmax(vis, axis=1)  # (N,) — index of first True

    valid_tracks = np.where(has_any)[0]
    for i in valid_tracks:
        fv = first_vis[i]
        rotations[i, fv] = np.eye(3, dtype=np.float32)
        rot_valid[i, fv] = True

    # Forward cumulation: for each track, walk through visible frames
    # and multiply deltas. This must handle gaps (invisible frames).
    #
    # Strategy: process frame-by-frame. For each frame t, tracks that
    # are visible at t and have a valid rotation at some previous frame
    # get their rotation updated.
    #
    # We need to handle the case where visibility has gaps — if track i
    # is visible at t=5, invisible at t=6,7, visible at t=8, we need
    # the delta from t=5→8. But our deltas are only t→t+1.
    #
    # Simple approach: cumulate through consecutive frames. If a track
    # is invisible at t, carry the last rotation forward (but mark invalid).
    # When it becomes visible again and delta_valid, update.

    # Actually: the deltas are computed for EVERY consecutive frame pair,
    # even if the track is invisible. The delta uses neighbor co-visibility,
    # not the track's own visibility. So we just need:
    # R[t+1] = delta_R[t] @ R[t]  when delta_valid[t] and the track has
    # a valid R at some frame <= t.

    # Vectorized forward pass
    R_current = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))  # (N, 3, 3)
    has_started = np.zeros(N, dtype=bool)

    for t in range(T):
        # Tracks starting at this frame
        starting = has_any & (first_vis == t)
        has_started |= starting
        R_current[starting] = np.eye(3, dtype=np.float32)

        # Set rotation for tracks that have started and are visible
        active = has_started & vis[:, t]
        rotations[active, t] = R_current[active]
        rot_valid[active, t] = True

        # Apply delta for next frame
        if t < T - 1:
            can_update = has_started & delta_valid[:, t]
            if can_update.any():
                idx = np.where(can_update)[0]
                R_current[idx] = np.einsum(
                    'bij,bjk->bik',
                    delta_R[idx, t],
                    R_current[idx],
                )

    return rotations, rot_valid


# ---------------------------------------------------------------------------
# Method 1: Batch SVD Kabsch
# ---------------------------------------------------------------------------

def _solve_svd(
    xyz: np.ndarray,
    vis: np.ndarray,
    nn_indices: np.ndarray,
    min_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard batch Kabsch — fastest method.

    Computes delta rotations between all consecutive frame pairs
    in a fully vectorized manner, then cumulates to absolute.
    """
    N, T, _ = xyz.shape

    delta_R = np.zeros((N, T - 1, 3, 3), dtype=np.float32)
    delta_valid = np.zeros((N, T - 1), dtype=bool)

    for t in range(T - 1):
        H, valid = _build_frame_pair_H(xyz, vis, nn_indices, t, t + 1, min_neighbors)
        if valid.any():
            idx = np.where(valid)[0]
            R = _batch_svd_to_rotation(H[idx])
            delta_R[idx, t] = R
            delta_valid[idx, t] = True

    return _cumulate_deltas(delta_R, delta_valid, vis)


# ---------------------------------------------------------------------------
# Method 2: IRLS-weighted Kabsch
# ---------------------------------------------------------------------------

def _solve_irls(
    xyz: np.ndarray,
    vis: np.ndarray,
    nn_indices: np.ndarray,
    min_neighbors: int = 4,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """IRLS-weighted Kabsch — robust to outlier neighbors.

    For each frame pair, runs n_iter iterations of weighted Kabsch,
    downweighting correspondences with large residuals (Huber loss).
    """
    N, T, _ = xyz.shape
    K = nn_indices.shape[1]

    delta_R = np.zeros((N, T - 1, 3, 3), dtype=np.float32)
    delta_valid = np.zeros((N, T - 1), dtype=bool)

    nn_clamped = np.maximum(nn_indices, 0)
    nn_valid_mask = nn_indices >= 0

    for t in range(T - 1):
        # Gather neighbor data for this frame pair
        nb_xyz_t0 = xyz[nn_clamped, t, :]     # (N, K, 3)
        nb_xyz_t1 = xyz[nn_clamped, t + 1, :]
        nb_vis_t0 = vis[nn_clamped, t]
        nb_vis_t1 = vis[nn_clamped, t + 1]

        joint = nn_valid_mask & nb_vis_t0 & nb_vis_t1  # (N, K)
        n_valid = joint.sum(axis=1)
        valid = n_valid >= min_neighbors

        if not valid.any():
            continue

        idx = np.where(valid)[0]
        B = len(idx)

        P_full = nb_xyz_t0[idx]  # (B, K, 3)
        Q_full = nb_xyz_t1[idx]
        mask = joint[idx]         # (B, K)

        w = mask.astype(np.float32)  # initial weights = 1 where valid

        for _it in range(n_iter):
            w_sum = w.sum(axis=1, keepdims=True).clip(1.0)  # (B, 1)

            # Weighted means
            P_w = P_full * w[:, :, None]  # (B, K, 3)
            Q_w = Q_full * w[:, :, None]
            P_mean = P_w.sum(axis=1, keepdims=True) / w_sum[:, :, None]  # (B, 1, 3)
            Q_mean = Q_w.sum(axis=1, keepdims=True) / w_sum[:, :, None]

            P_c = np.where(mask[..., None], P_full - P_mean, 0.0)  # (B, K, 3)
            Q_c = np.where(mask[..., None], Q_full - Q_mean, 0.0)

            # Weighted cross-covariance
            H = np.einsum('bki,bk,bkj->bij', P_c, w, Q_c)  # (B, 3, 3)
            R = _batch_svd_to_rotation(H)

            # Residuals
            Q_pred = np.einsum('bij,bkj->bki', R, P_c)  # (B, K, 3)
            residuals = np.linalg.norm(Q_pred - Q_c, axis=-1)  # (B, K)
            residuals = np.where(mask, residuals, 0.0)

            # Huber weights
            med_r = np.median(
                np.where(mask, residuals, np.nan), axis=1
            )  # (B,) — may have NaN from numpy median, but we masked
            # Use nanmedian for safety
            med_r = np.nanmedian(
                np.where(mask, residuals, np.nan), axis=1
            )
            delta_h = np.maximum(med_r * 1.5, 1e-6)[:, None]  # (B, 1)
            w_new = np.where(residuals < delta_h, 1.0, delta_h / np.maximum(residuals, 1e-8))
            w = np.where(mask, w_new, 0.0).astype(np.float32)

        delta_R[idx, t] = R
        delta_valid[idx, t] = True

    return _cumulate_deltas(delta_R, delta_valid, vis)


# ---------------------------------------------------------------------------
# Method 3: Windowed Kabsch
# ---------------------------------------------------------------------------

def _solve_windowed(
    xyz: np.ndarray,
    vis: np.ndarray,
    nn_indices: np.ndarray,
    min_neighbors: int = 4,
    half_window: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed Kabsch — accumulate covariance over temporal window.

    For each frame pair (t, t+1), accumulates cross-covariance from
    shifted pairs (t+w, t+1+w) for w in [-W, W], providing more
    data points and implicit temporal smoothness.
    """
    N, T, _ = xyz.shape
    W = half_window

    delta_R = np.zeros((N, T - 1, 3, 3), dtype=np.float32)
    delta_valid = np.zeros((N, T - 1), dtype=bool)

    for t in range(T - 1):
        # Accumulate H over window offsets
        H_acc = np.zeros((N, 3, 3), dtype=np.float64)
        total_valid = np.zeros(N, dtype=np.int32)

        for w in range(-W, W + 1):
            ta = t + w
            tb = t + 1 + w
            if ta < 0 or ta >= T or tb < 0 or tb >= T:
                continue

            H_w, valid_w = _build_frame_pair_H(
                xyz, vis, nn_indices, ta, tb, min_neighbors=2
            )
            # Only accumulate for tracks with at least 2 neighbors at this offset
            H_acc[valid_w] += H_w[valid_w]
            total_valid[valid_w] += 1

        # Require enough total pairs
        valid = total_valid >= 1  # at least one offset contributed
        if valid.any():
            idx = np.where(valid)[0]
            R = _batch_svd_to_rotation(H_acc[idx].astype(np.float32))
            delta_R[idx, t] = R
            delta_valid[idx, t] = True

    return _cumulate_deltas(delta_R, delta_valid, vis)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

METHODS = {
    "svd": _solve_svd,
    "irls": _solve_irls,
    "windowed": _solve_windowed,
}


def compute_local_frames(
    tracks: np.ndarray,
    method: str = "svd",
    K_spatial: int = 64,
    N_final: int = 16,
    min_neighbors: int = 4,
    **method_kwargs,
) -> LocalFrameResult:
    """Compute local frames for all tracks.

    Args:
        tracks: (N, T, 6) float32 — packed [u, v, x, y, z, vis]
        method: one of 'svd', 'irls', 'windowed'
        K_spatial: initial spatial candidate count for neighborhood
        N_final: final neighbor count after curve distance refinement
        min_neighbors: minimum co-visible neighbors for valid rotation
        **method_kwargs: extra kwargs passed to the method

    Returns:
        LocalFrameResult with rotations, validity, neighbors, metadata
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")

    N, T, C = tracks.shape
    assert C == 6, f"Expected 6 columns [u,v,x,y,z,vis], got {C}"

    xyz = tracks[:, :, 2:5].copy()  # (N, T, 3)
    vis = tracks[:, :, 5] > 0.5     # (N, T) bool

    # Replace NaN with 0 for computation (masked out by vis)
    xyz = np.nan_to_num(xyz, nan=0.0)

    logger.info("Computing local frames: N=%d, T=%d, method=%s", N, T, method)
    t_start = time.time()

    # Step 1: neighborhoods
    nn_indices = compute_neighborhoods(xyz, vis, K_spatial=K_spatial, N_final=N_final)

    # Step 2: solve rotations
    solve_fn = METHODS[method]
    rotations, rot_valid = solve_fn(
        xyz, vis, nn_indices, min_neighbors=min_neighbors, **method_kwargs
    )

    elapsed = time.time() - t_start
    logger.info("Local frames done: %.2fs, valid_frac=%.3f", elapsed, rot_valid.mean())

    # Sanity check: det(R) ≈ 1 for valid rotations
    if rot_valid.any():
        R_valid = rotations[rot_valid]
        dets = np.linalg.det(R_valid)
        det_err = np.abs(dets - 1.0).max()
        if det_err > 1e-3:
            logger.warning("Rotation det error: max=%.6f", det_err)
        else:
            logger.info("Rotation sanity: max det error = %.6f", det_err)

    params = dict(
        K_spatial=K_spatial,
        N_final=N_final,
        min_neighbors=min_neighbors,
        **method_kwargs,
    )

    return LocalFrameResult(
        rotations=rotations,
        rot_valid=rot_valid,
        nn_indices=nn_indices,
        method=method,
        params=params,
    )
