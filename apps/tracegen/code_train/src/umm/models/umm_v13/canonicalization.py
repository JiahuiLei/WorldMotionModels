"""Per-sample SIM(3) canonicalization for V11.

Computes a SIM(3) transform from known+valid tokens that normalizes each
training sample to a canonical frame:
  - Mean position → origin
  - Position std → target (configurable, default 0.5)
  - Mean rotation → identity (via SVD)

Applied BEFORE noise addition.  Loss computed in canonicalized space.
De-canonicalization inverts the transform for inference output.

All functions are ``@torch.compiler.disable`` because they use dynamic
masking (boolean indexing on per-sample known_mask).

Example::

    canon = Canonicalizer(target_pos_std=0.5)
    canon_poses, canon_params = canon.canonicalize(poses_9d, known_mask, valid_mask)
    original_poses = canon.decanonicalize(canon_poses, canon_params)
"""

import logging
from dataclasses import dataclass

import torch

from umm.models.umm_v13.rotation_utils import sixd_to_rotmat, rotmat_to_6d

logger = logging.getLogger(__name__)


@dataclass
class CanonParams:
    """Parameters of a SIM(3) canonicalization transform.

    Stores everything needed to invert the canonicalization at inference time.

    Attributes:
        mean_pos: Mean position of known+valid tokens, shape ``(B, 3)``.
        scale: Scale factor applied, shape ``(B, 1)``.
        R_canon: Rotation applied (to align mean rotation → identity), ``(B, 3, 3)``.
    """
    mean_pos: torch.Tensor   # (B, 3)
    scale: torch.Tensor      # (B, 1)
    R_canon: torch.Tensor    # (B, 3, 3)


@torch.compiler.disable
def compute_canon_params(
    poses_9d: torch.Tensor,
    known_valid_mask: torch.Tensor,
    target_pos_std: float = 0.5,
) -> CanonParams:
    """Compute SIM(3) canonicalization parameters from known+valid tokens.

    Args:
        poses_9d: 9D poses, shape ``(B, N, 9)`` where N = T*P or (C+T)*P
            flattened.  Format: [pos(3), rot6d(6)].
        known_valid_mask: Boolean mask of known AND valid tokens, ``(B, N)``.
        target_pos_std: Target std for position after canonicalization.

    Returns:
        CanonParams with mean_pos, scale, R_canon for each sample.

    Raises:
        RuntimeError: If any sample has < 2 known+valid tokens.
    """
    B = poses_9d.shape[0]
    device = poses_9d.device
    dtype = poses_9d.dtype

    # Disable AMP autocast — SVD/det don't support BFloat16
    with torch.autocast(device_type=device.type, enabled=False):
        poses_f32 = poses_9d.float()

        mean_pos = torch.zeros(B, 3, device=device, dtype=torch.float32)
        scale = torch.ones(B, 1, device=device, dtype=torch.float32)
        R_canon = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()

        # Upstream finite-check: if ANY input is NaN/Inf, entire batch goes identity
        # and we log loud. This catches bad dataset samples before they poison the LMM.
        if not torch.isfinite(poses_f32).all():
            n_bad = int((~torch.isfinite(poses_f32)).sum().item())
            logger.warning(
                f"canonicalizer: {n_bad} NaN/Inf elements in input poses; "
                f"using identity canonicalization for ALL samples in this batch. "
                f"Check dataset upstream."
            )
            return CanonParams(
                mean_pos=mean_pos.to(dtype),
                scale=scale.to(dtype),
                R_canon=R_canon.to(dtype),
            )

        # POS_STD_MIN floor: the only upper bound on scale is implicit via
        # this floor (scale = target_pos_std / pos_std, so scale_max =
        # target_pos_std / POS_STD_MIN). Originally 0.01, giving scale_max=50
        # for target_pos_std=0.5. NO explicit SCALE_MAX clamp — close-up
        # scenes and small-pos_std samples legitimately need large scale.
        POS_STD_MIN = 0.01

        for b in range(B):
            mask_b = known_valid_mask[b]  # (N,)
            n_known = int(mask_b.sum().item())
            if n_known < 2:
                # Defensive fallback: identity canonicalization (mean=0, scale=1, R=I).
                # The training_step's loss is masked by all_mask, so a 0-known sample
                # contributes 0 loss/grad regardless of canon params.
                logger.warning(
                    f"canonicalizer: sample {b} has only {n_known} known+valid "
                    f"tokens; using identity canonicalization (loss masked out)."
                )
                continue

            known_poses = poses_f32[b, mask_b]  # (K, 9)
            known_pos = known_poses[:, :3]     # (K, 3)
            known_rot6d = known_poses[:, 3:]   # (K, 6)

            # 1. Mean position
            mean_p = known_pos.mean(dim=0)  # (3,)
            if not torch.isfinite(mean_p).all():
                logger.warning(
                    f"canonicalizer: sample {b} mean_pos is non-finite; identity fallback."
                )
                continue
            mean_pos[b] = mean_p

            # 2. Position std → scale. Handle NaN (rare: e.g. K=1 single-point case
            # where std is well-defined but could underflow). max() with a NaN in
            # Python returns the FIRST arg (NaN), not the second — we must guard
            # explicitly.
            centered = known_pos - mean_p.unsqueeze(0)  # (K, 3)
            pos_std_raw = centered.std().item()
            if not (pos_std_raw == pos_std_raw):  # NaN check without math import
                pos_std_raw = POS_STD_MIN
            pos_std = max(pos_std_raw, POS_STD_MIN)
            s = target_pos_std / pos_std
            scale[b, 0] = s

            # 3. Mean rotation via SVD. Safe against:
            #   - All-zero R_sum (regularization: add eps*I)
            #   - SVD numerical failure (try/except → identity fallback)
            #   - NaN in output (finite check → identity fallback)
            R_matrices = sixd_to_rotmat(known_rot6d)  # (K, 3, 3)
            if not torch.isfinite(R_matrices).all():
                logger.warning(
                    f"canonicalizer: sample {b} rot6d produced non-finite R; identity."
                )
                continue
            R_sum = R_matrices.sum(dim=0)  # (3, 3)
            # Regularize to prevent SVD on rank-deficient / zero matrices.
            R_sum_reg = R_sum + 1e-4 * torch.eye(
                3, device=device, dtype=torch.float32
            )
            try:
                U, _, Vh = torch.linalg.svd(R_sum_reg)
                det = torch.det(U @ Vh)
                diag_vec = torch.ones(3, device=device, dtype=torch.float32)
                diag_vec[2] = det.sign()
                R_mean = U @ torch.diag(diag_vec) @ Vh  # (3, 3)
                if not torch.isfinite(R_mean).all():
                    raise RuntimeError("SVD produced non-finite R_mean")
                # Canon rotation: R_canon @ R_mean = I → R_canon = R_mean^T
                R_canon[b] = R_mean.T
            except RuntimeError as e:
                # Covers LinAlgError (subclass of RuntimeError) and other numerical failures.
                logger.warning(
                    f"canonicalizer: sample {b} SVD failed ({type(e).__name__}: {e}); "
                    f"identity rotation fallback."
                )
                # R_canon[b] stays as identity (initialized above)

        # Cast back to original dtype
        mean_pos = mean_pos.to(dtype)
        scale = scale.to(dtype)
        R_canon = R_canon.to(dtype)

    return CanonParams(mean_pos=mean_pos, scale=scale, R_canon=R_canon)


@torch.compiler.disable
def canonicalize(
    poses_9d: torch.Tensor,
    params: CanonParams,
) -> torch.Tensor:
    """Apply SIM(3) canonicalization to all poses.

    Transform: pos' = scale * R_canon @ (pos - mean_pos)
               rot' = R_canon @ rot

    Args:
        poses_9d: 9D poses, shape ``(B, *, 9)``. Can be any shape with
            batch dim first and 9D last.
        params: CanonParams from :func:`compute_canon_params`.

    Returns:
        Canonicalized 9D poses, same shape as input.
    """
    orig_shape = poses_9d.shape
    B = orig_shape[0]
    # Flatten middle dims
    poses_flat = poses_9d.reshape(B, -1, 9)  # (B, N, 9)
    N = poses_flat.shape[1]

    pos = poses_flat[..., :3]    # (B, N, 3)
    rot6d = poses_flat[..., 3:]  # (B, N, 6)

    # Position: translate, rotate, scale
    # pos_centered = pos - mean_pos[:, None, :]  (B, N, 3)
    pos_centered = pos - params.mean_pos[:, None, :]
    # Rotate: (B, 3, 3) @ (B, N, 3).T -> use einsum
    pos_rotated = torch.einsum("bij,bnj->bni", params.R_canon, pos_centered)  # (B, N, 3)
    pos_scaled = pos_rotated * params.scale[:, :, None].expand(-1, N, 3)  # (B, N, 3)

    # Rotation: R_canon @ R_i
    R = sixd_to_rotmat(rot6d)  # (B, N, 3, 3)
    R_canon_expanded = params.R_canon[:, None, :, :]  # (B, 1, 3, 3)
    R_canonicalized = R_canon_expanded @ R  # (B, N, 3, 3)
    rot6d_canon = rotmat_to_6d(R_canonicalized)  # (B, N, 6)

    out = torch.cat([pos_scaled, rot6d_canon], dim=-1)  # (B, N, 9)
    return out.reshape(orig_shape)


@torch.compiler.disable
def decanonicalize(
    poses_9d: torch.Tensor,
    params: CanonParams,
) -> torch.Tensor:
    """Invert SIM(3) canonicalization.

    Inverse: pos = R_canon^T @ (pos' / scale) + mean_pos
             rot = R_canon^T @ rot'

    Args:
        poses_9d: Canonicalized 9D poses, shape ``(B, *, 9)``.
        params: CanonParams from :func:`compute_canon_params`.

    Returns:
        De-canonicalized 9D poses in original space.
    """
    orig_shape = poses_9d.shape
    B = orig_shape[0]
    poses_flat = poses_9d.reshape(B, -1, 9)
    N = poses_flat.shape[1]

    pos = poses_flat[..., :3]    # (B, N, 3)
    rot6d = poses_flat[..., 3:]  # (B, N, 6)

    # Inverse position: unscale, unrotate, untranslate
    R_canon_T = params.R_canon.transpose(-1, -2)  # (B, 3, 3)
    pos_unscaled = pos / (params.scale[:, :, None].expand(-1, N, 3) + 1e-8)
    pos_unrotated = torch.einsum("bij,bnj->bni", R_canon_T, pos_unscaled)  # (B, N, 3)
    pos_orig = pos_unrotated + params.mean_pos[:, None, :]

    # Inverse rotation: R_canon^T @ R_canonicalized
    R = sixd_to_rotmat(rot6d)  # (B, N, 3, 3)
    R_canon_T_expanded = R_canon_T[:, None, :, :]  # (B, 1, 3, 3)
    R_orig = R_canon_T_expanded @ R  # (B, N, 3, 3)
    rot6d_orig = rotmat_to_6d(R_orig)  # (B, N, 6)

    out = torch.cat([pos_orig, rot6d_orig], dim=-1)
    return out.reshape(orig_shape)


class Canonicalizer:
    """Convenience wrapper for SIM(3) canonicalization.

    Args:
        target_pos_std: Target standard deviation for positions. Default 0.5.

    Example::

        canon = Canonicalizer(target_pos_std=0.5)
        x_canon, params = canon.forward(poses_9d, known_valid_mask)
        x_orig = canon.inverse(x_canon, params)
    """

    def __init__(self, target_pos_std: float = 0.5):
        self.target_pos_std = target_pos_std

    def forward(
        self,
        poses_9d: torch.Tensor,
        known_valid_mask: torch.Tensor,
    ) -> tuple:
        """Canonicalize poses using known+valid tokens.

        Args:
            poses_9d: (B, N, 9) or (B, *, 9) — all poses to transform.
            known_valid_mask: (B, N) bool — which tokens are known AND valid.

        Returns:
            Tuple of (canonicalized_poses, CanonParams).
        """
        # Flatten to (B, N, 9) for param computation
        B = poses_9d.shape[0]
        poses_flat = poses_9d.reshape(B, -1, 9)
        params = compute_canon_params(
            poses_flat, known_valid_mask, self.target_pos_std
        )
        canon_poses = canonicalize(poses_9d, params)
        return canon_poses, params

    def inverse(
        self,
        poses_9d: torch.Tensor,
        params: CanonParams,
    ) -> torch.Tensor:
        """De-canonicalize poses.

        Args:
            poses_9d: Canonicalized (B, *, 9) poses.
            params: CanonParams from forward().

        Returns:
            De-canonicalized poses in original space.
        """
        return decanonicalize(poses_9d, params)
