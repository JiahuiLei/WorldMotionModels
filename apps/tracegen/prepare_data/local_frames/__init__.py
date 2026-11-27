"""Local SE(3) frame computation from dense 3D track neighborhoods.

The core algorithm (`compute_local_frames`) is a self-contained module,
vendored unchanged.

Public API:
    compute_local_frames(tracks, method, K_spatial, N_final, min_neighbors, **kwargs)
        tracks: (N, T, 6) float32 packed [u, v, x, y, z, vis]
        returns LocalFrameResult with .rotations (N,T,3,3), .rot_valid (N,T),
                .nn_indices (N, N_final), .method, .params
"""

from ._local_frames import LocalFrameResult, compute_local_frames

__all__ = ["LocalFrameResult", "compute_local_frames"]
