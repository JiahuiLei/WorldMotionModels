"""SO(3) utilities for V11.3 compact LMM.

Provides geodesic mean rotation computation from wing-vector parameterization.
Used by P-layer dual spatial RoPE to compute mean rotation per entity.

Compile safety: The main function ``compute_entity_mean_9d`` is decorated with
``@torch.compiler.disable`` because it uses ``torch.linalg.svd`` and
``torch.linalg.det`` which are compile-unsafe.
"""

import torch


@torch.compiler.disable
def compute_entity_mean_9d(
    x_9d: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-entity mean 9D position over valid frames.

    Position (first 3 dims): Euclidean mean over valid frames.
    Rotation (last 6 dims, wing vectors): SO(3) geodesic mean via SVD projection.

    The wing vectors are the first two columns of a rotation matrix scaled by
    wing_len. Recovery procedure:
        1. Normalize each wing by L2 norm -> unit columns
        2. Third column = cross product of normalized wings
        3. R = [w1_hat, w2_hat, col3]  (3x3 rotation matrix)
        4. Mean: M = sum(R_i) / n_valid, then SVD project to SO(3)
        5. Convert back: first two columns of R_mean, scaled by mean wing_len

    Detached (no gradients). Invalid entities get identity rotation + zero pos.

    Compile audit: Uses torch.linalg.svd, torch.linalg.det, and conditional
    logic on mask values. All compile-unsafe. Entire function disabled.

    AMP safety: SVD and det require float32. Function disables autocast and
    casts inputs to float32, then casts output back to original dtype.

    Args:
        x_9d: (B, S, P, 9) raw 9D positions [pos(3), wing1(3), wing2(3)].
        mask: (B, S, P) bool mask, True = valid frame.

    Returns:
        mean_9d: (B, P, 9) per-entity mean 9D position.
    """
    B, S, P, _ = x_9d.shape
    device = x_9d.device
    dtype = x_9d.dtype

    # Disable AMP autocast -- SVD/det require float32, autocast would recast
    with torch.amp.autocast("cuda", enabled=False):
        # Detach + cast to float32
        x_9d = x_9d.detach().float()

        # Expand mask for broadcasting: (B, S, P, 1)
        mask_f = mask.float().unsqueeze(-1)  # (B, S, P, 1)
        raw_count = mask_f.sum(dim=1)  # (B, P, 1) -- raw count before clamp
        n_valid = raw_count.clamp(min=1.0)  # (B, P, 1) -- clamped for safe division

        # === Position mean (Euclidean) ===
        pos = x_9d[..., :3]  # (B, S, P, 3)
        pos_mean = (pos * mask_f).sum(dim=1) / n_valid  # (B, P, 3)

        # === Rotation mean (SO(3) geodesic via SVD) ===
        w1 = x_9d[..., 3:6]  # (B, S, P, 3) wing1
        w2 = x_9d[..., 6:9]  # (B, S, P, 3) wing2

        # Wing lengths for scaling back
        w1_len = w1.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, S, P, 1)
        w2_len = w2.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, S, P, 1)

        # Mean wing length over valid frames
        mean_w1_len = (w1_len * mask_f).sum(dim=1) / n_valid  # (B, P, 1)
        mean_w2_len = (w2_len * mask_f).sum(dim=1) / n_valid  # (B, P, 1)

        # Normalize wings to unit vectors
        w1_hat = w1 / w1_len  # (B, S, P, 3)
        w2_hat = w2 / w2_len  # (B, S, P, 3)

        # Third column = cross product of normalized wings
        col3 = torch.cross(w1_hat, w2_hat, dim=-1)  # (B, S, P, 3)

        # Build rotation matrices R = [w1_hat, w2_hat, col3]: (B, S, P, 3, 3)
        R = torch.stack([w1_hat, w2_hat, col3], dim=-1)  # (B, S, P, 3, 3)

        # Mean rotation matrix: M = sum(R_i * valid) / n_valid
        R_masked = R * mask_f.unsqueeze(-1)  # (B, S, P, 3, 3)
        M = R_masked.sum(dim=1) / n_valid.unsqueeze(-1)  # (B, P, 3, 3)

        # SVD projection to SO(3): R_mean = U @ diag(1, 1, det(U@Vt)) @ Vt
        U, _S, Vt = torch.linalg.svd(M)  # U: (B, P, 3, 3), Vt: (B, P, 3, 3)
        UVt = U @ Vt  # (B, P, 3, 3)
        det_UVt = torch.linalg.det(UVt)  # (B, P)

        # Correction matrix to ensure det = +1 (proper rotation)
        correction = torch.ones(B, P, 3, device=device, dtype=torch.float32)
        correction[..., 2] = det_UVt.sign()  # flip last singular value if needed
        R_mean = U * correction.unsqueeze(-2) @ Vt  # (B, P, 3, 3)

        # Convert back to wings: first two columns, scaled by mean wing_len
        w1_mean = R_mean[..., 0] * mean_w1_len  # (B, P, 3)
        w2_mean = R_mean[..., 1] * mean_w2_len  # (B, P, 3)

        # === Handle fully-invalid entities ===
        # Entities with zero valid frames get identity rotation + zero position
        all_invalid = (raw_count.squeeze(-1) < 0.5)  # (B, P)
        pos_mean[all_invalid] = 0.0
        # Identity rotation wings: [wing_len, 0, 0] and [0, wing_len, 0]
        w1_mean[all_invalid] = torch.tensor(
            [0.2, 0.0, 0.0], device=device, dtype=torch.float32
        )
        w2_mean[all_invalid] = torch.tensor(
            [0.0, 0.2, 0.0], device=device, dtype=torch.float32
        )

        # Assemble output: (B, P, 9)
        mean_9d = torch.cat([pos_mean, w1_mean, w2_mean], dim=-1)  # (B, P, 9)

    return mean_9d.to(dtype)
