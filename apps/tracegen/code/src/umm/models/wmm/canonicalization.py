"""Per-sample SIM(3) canonicalization for V11.

Computes a SIM(3) transform from known+valid tokens that normalizes each
training sample to a canonical frame:
  - Mean position → origin
  - Position std → target (configurable, default 0.5)
  - Mean rotation → identity (via SVD)

Two variants:
  - **Original (`bounded=False`, default):** scalar (uniform) std → uniform scale.
    Output is unbounded (linear in position units).
  - **Bounded (`bounded=True`):** depth-filtered compute set, per-axis std →
    per-axis scale, followed by ``tanh`` so output ∈ (-1, 1) elementwise.
    Inverse uses ``atanh`` after a safety clamp.

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
import math
from dataclasses import dataclass
from typing import Optional

import torch

from umm.models.wmm.rotation_utils import sixd_to_rotmat, rotmat_to_6d

logger = logging.getLogger(__name__)


# Per-axis position std floor. Originally a single scalar floor on the scalar
# pos_std; now applied per axis (x, y, z) for both variants.
POS_STD_MIN = 0.01


@dataclass
class CanonParams:
    """Parameters of a SIM(3) canonicalization transform.

    Stores everything needed to invert the canonicalization at inference time.

    Attributes:
        mean_pos: Mean position of the compute set, shape ``(B, 3)``.
        scale: Per-axis scale factor, shape ``(B, 3)``. For the original
            (``bounded=False``) variant, all 3 entries within a sample are
            identical (scalar broadcast to per-axis), preserving back-compat.
        R_canon: Rotation applied (to align mean rotation → identity), ``(B, 3, 3)``.
        bounded: If True, ``canonicalize`` applied ``tanh`` after the linear
            transform; ``decanonicalize`` must invert via ``atanh`` (with clamp).
        atanh_eps: Safety margin for the pre-atanh clamp: ``clamp(±(1-eps))``.
            Only consulted when ``bounded=True``.
    """
    mean_pos: torch.Tensor   # (B, 3)
    scale: torch.Tensor      # (B, 3)
    R_canon: torch.Tensor    # (B, 3, 3)
    bounded: bool = False
    atanh_eps: float = 1.0e-4
    # Per-sample diagnostic stats.
    # ratio_inside_thr: populated only when bounded=True. Fraction of known
    #   points with z<dep_thr (or 1.0 when fallback fired with topk subset).
    #   Shape (B,).
    # std_axis: populated when bounded=True OR per_axis_std=True. Per-axis std
    #   measured in the std_frame (pre-floor). Shape (B, 3).
    ratio_inside_thr: Optional[torch.Tensor] = None
    std_axis: Optional[torch.Tensor] = None


@torch.compiler.disable
def compute_canon_params(
    poses_9d: torch.Tensor,
    known_valid_mask: torch.Tensor,
    target_pos_std: float = 0.5,
    bounded: bool = False,
    bounded_dep_thr: float = 1.5,
    bounded_topk_frac: float = 0.1,
    bounded_atanh_eps: float = 1.0e-4,
    canon_rotation: bool = True,
    per_axis_std: bool = False,
) -> CanonParams:
    """Compute SIM(3) canonicalization parameters from known+valid tokens.

    Args:
        poses_9d: 9D poses, shape ``(B, N, 9)`` where N = T*P or (C+T)*P
            flattened.  Format: [pos(3), rot6d(6)].
        known_valid_mask: Boolean mask of known AND valid tokens, ``(B, N)``.
        target_pos_std: Target std for position after canonicalization.
        bounded: If True, use the bounded variant (depth-filter + per-axis std
            + tanh). If False (default), use original scalar-std behavior;
            ``scale`` is filled with the broadcast scalar so all callers see
            ``(B, 3)`` uniformly.
        bounded_dep_thr: Depth threshold for the compute set (only consulted
            when ``bounded=True``). Compute set = known_valid & (z < dep_thr).
        bounded_topk_frac: Fallback fraction (only consulted when
            ``bounded=True``). When the depth filter excludes all known
            tokens, fall back to the top-``frac`` smallest-z subset of known.
        bounded_atanh_eps: Stored on the returned ``CanonParams`` for use by
            ``decanonicalize`` (clamp before atanh). Only consulted when
            ``bounded=True``.
        canon_rotation: Hint controlling the per-axis std frame in the bounded
            variant. When True, std is computed in the canon (post-R_canon)
            frame; when False, std is computed in the centered (pre-rotation)
            frame, matching the agent's override that resets R_canon to
            identity. No effect on the original (bounded=False) variant.
        per_axis_std: When True (and ``bounded=False``), compute per-axis std
            from all known points (no depth filter), giving a per-axis scale
            but WITHOUT the bounded variant's tanh squashing. Default False
            preserves the original scalar-std behavior. Ignored when
            ``bounded=True`` (bounded already does per-axis).

    Returns:
        CanonParams with mean_pos (B,3), scale (B,3), R_canon (B,3,3),
        bounded flag, atanh_eps.
    """
    B = poses_9d.shape[0]
    device = poses_9d.device
    dtype = poses_9d.dtype

    # Disable AMP autocast — SVD/det don't support BFloat16
    with torch.autocast(device_type=device.type, enabled=False):
        poses_f32 = poses_9d.float()

        mean_pos = torch.zeros(B, 3, device=device, dtype=torch.float32)
        # scale is (B, 3) for both variants; original variant fills all 3 with
        # the same scalar (back-compat: callers see uniform shape).
        scale = torch.ones(B, 3, device=device, dtype=torch.float32)
        R_canon = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
        # Diagnostic stats — only meaningful for bounded=True.
        ratio_inside_thr = torch.zeros(B, device=device, dtype=torch.float32)
        std_axis_all = torch.zeros(B, 3, device=device, dtype=torch.float32)

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
                bounded=bounded,
                atanh_eps=bounded_atanh_eps,
            )

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

            # ----- Branch on variant for compute-set selection -----
            if bounded:
                z = known_pos[:, 2]
                depth_mask = z < bounded_dep_thr
                n_inside = int(depth_mask.sum().item())
                if n_inside == 0:
                    # Fallback: top-frac smallest-z subset of known points.
                    K = known_pos.shape[0]
                    k = max(1, int(math.ceil(bounded_topk_frac * K)))
                    logger.warning(
                        f"canonicalizer (bounded): sample {b} has 0 known tokens "
                        f"with z<{bounded_dep_thr}; falling back to top-{k}/{K} "
                        f"smallest-z subset (frac={bounded_topk_frac})."
                    )
                    # torch.topk with largest=False returns smallest values.
                    _, idx = torch.topk(z, k=k, largest=False)
                    compute_pos = known_pos[idx]
                    compute_rot6d = known_rot6d[idx]
                    # Stat: 0% of known passed the threshold (fallback mode).
                    ratio_inside_thr[b] = 0.0
                else:
                    compute_pos = known_pos[depth_mask]
                    compute_rot6d = known_rot6d[depth_mask]
                    ratio_inside_thr[b] = float(n_inside) / float(known_pos.shape[0])
            else:
                compute_pos = known_pos
                compute_rot6d = known_rot6d

            # 1. Mean position (compute set)
            mean_p = compute_pos.mean(dim=0)  # (3,)
            if not torch.isfinite(mean_p).all():
                logger.warning(
                    f"canonicalizer: sample {b} mean_pos is non-finite; identity fallback."
                )
                continue
            mean_pos[b] = mean_p

            # 2. Rotation (SVD on compute_rot6d → R_canon).
            # `canon_rotation` controls the std frame: True → std in the
            # post-R_canon (canon) frame; False → std in the centered
            # pre-rotation frame (matches the agent's override that resets
            # R_canon to identity). The R_canon SVD is always computed (cheap
            # vs the full forward) and stored on CanonParams; the caller
            # decides whether to keep it.
            R_matrices = sixd_to_rotmat(compute_rot6d)  # (Kc, 3, 3)
            if not torch.isfinite(R_matrices).all():
                logger.warning(
                    f"canonicalizer: sample {b} rot6d produced non-finite R; identity."
                )
                continue
            R_sum = R_matrices.sum(dim=0)  # (3, 3)
            R_sum_reg = R_sum + 1e-4 * torch.eye(
                3, device=device, dtype=torch.float32
            )
            R_canon_b = torch.eye(3, device=device, dtype=torch.float32)
            try:
                U, _, Vh = torch.linalg.svd(R_sum_reg)
                det = torch.det(U @ Vh)
                diag_vec = torch.ones(3, device=device, dtype=torch.float32)
                diag_vec[2] = det.sign()
                R_mean = U @ torch.diag(diag_vec) @ Vh  # (3, 3)
                if not torch.isfinite(R_mean).all():
                    raise RuntimeError("SVD produced non-finite R_mean")
                # Canon rotation: R_canon @ R_mean = I → R_canon = R_mean^T
                R_canon_b = R_mean.T
            except RuntimeError as e:
                # Covers LinAlgError (subclass of RuntimeError) and other numerical failures.
                logger.warning(
                    f"canonicalizer: sample {b} SVD failed ({type(e).__name__}: {e}); "
                    f"identity rotation fallback."
                )
                # R_canon_b stays identity

            R_canon[b] = R_canon_b

            # 3. Position std → scale.
            centered = compute_pos - mean_p.unsqueeze(0)  # (Kc, 3)

            if bounded or per_axis_std:
                # Per-axis std. Used by:
                # - bounded=True (with depth-filter compute set, then tanh).
                # - per_axis_std=True (no depth-filter, no tanh) — gives a
                #   per-axis linear scale instead of the legacy scalar.
                # Std frame:
                # - canon_rotation=True: scale is applied AFTER rotating
                #   centered by R_canon, so std must be measured post-rotation:
                #   centered_canon = R_canon @ centered, i.e. centered @ R_canon^T.
                # - canon_rotation=False: agent overrides R_canon to identity
                #   and re-applies; scale is therefore applied in the centered
                #   (pre-rotation) frame, so std is measured in `centered`.
                if canon_rotation:
                    std_frame = centered @ R_canon_b.T  # (Kc, 3)
                else:
                    std_frame = centered  # (Kc, 3)
                # If only 1 point, std is undefined (returns NaN/0). Use floor.
                if std_frame.shape[0] < 2:
                    std_axis_t = torch.full(
                        (3,), POS_STD_MIN, device=device, dtype=torch.float32,
                    )
                else:
                    std_axis_t = std_frame.std(dim=0)  # (3,)
                # NaN guard per axis
                std_axis_t = torch.where(
                    torch.isfinite(std_axis_t),
                    std_axis_t,
                    torch.full_like(std_axis_t, POS_STD_MIN),
                )
                # Per-axis floor
                std_axis_t = torch.clamp(std_axis_t, min=POS_STD_MIN)
                scale_axis = target_pos_std / std_axis_t  # (3,)
                scale[b] = scale_axis
                std_axis_all[b] = std_axis_t
            else:
                # Original behavior: scalar std, scalar scale, broadcast to (3,).
                # Std is computed in the centered (pre-rotation) frame; this
                # matches the legacy implementation byte-for-byte.
                pos_std_raw = centered.std().item()
                if not (pos_std_raw == pos_std_raw):  # NaN check without math import
                    pos_std_raw = POS_STD_MIN
                pos_std = max(pos_std_raw, POS_STD_MIN)
                s = target_pos_std / pos_std
                scale[b, 0] = s
                scale[b, 1] = s
                scale[b, 2] = s

        # Cast back to original dtype
        mean_pos = mean_pos.to(dtype)
        scale = scale.to(dtype)
        R_canon = R_canon.to(dtype)

    stats_kw = {}
    if bounded:
        stats_kw = dict(
            ratio_inside_thr=ratio_inside_thr.to(dtype),
            std_axis=std_axis_all.to(dtype),
        )
    elif per_axis_std:
        stats_kw = dict(
            std_axis=std_axis_all.to(dtype),
        )
    return CanonParams(
        mean_pos=mean_pos,
        scale=scale,
        R_canon=R_canon,
        bounded=bounded,
        atanh_eps=bounded_atanh_eps,
        **stats_kw,
    )


@torch.compiler.disable
def canonicalize(
    poses_9d: torch.Tensor,
    params: CanonParams,
) -> torch.Tensor:
    """Apply SIM(3) canonicalization to all poses.

    Original variant: pos' = scale * R_canon @ (pos - mean_pos)
                      rot' = R_canon @ rot

    Bounded variant: pos' = tanh(scale_axis * R_canon @ (pos - mean_pos))
                     rot' = R_canon @ rot   (unchanged)

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

    pos = poses_flat[..., :3]    # (B, N, 3)
    rot6d = poses_flat[..., 3:]  # (B, N, 6)

    # Position: translate, rotate, scale (per-axis).
    pos_centered = pos - params.mean_pos[:, None, :]  # (B, N, 3)
    # Rotate: (B, 3, 3) @ (B, N, 3) along last dim
    pos_rotated = torch.einsum("bij,bnj->bni", params.R_canon, pos_centered)  # (B, N, 3)
    # Per-axis scale: (B, 3) → (B, 1, 3) broadcasts over N.
    pos_scaled = pos_rotated * params.scale[:, None, :]  # (B, N, 3)

    if params.bounded:
        pos_out = torch.tanh(pos_scaled)
    else:
        pos_out = pos_scaled

    # Rotation: R_canon @ R_i (unchanged across variants)
    R = sixd_to_rotmat(rot6d)  # (B, N, 3, 3)
    R_canon_expanded = params.R_canon[:, None, :, :]  # (B, 1, 3, 3)
    R_canonicalized = R_canon_expanded @ R  # (B, N, 3, 3)
    rot6d_canon = rotmat_to_6d(R_canonicalized)  # (B, N, 6)

    out = torch.cat([pos_out, rot6d_canon], dim=-1)  # (B, N, 9)
    return out.reshape(orig_shape)


@torch.compiler.disable
def decanonicalize(
    poses_9d: torch.Tensor,
    params: CanonParams,
) -> torch.Tensor:
    """Invert SIM(3) canonicalization.

    Original variant: pos = R_canon^T @ (pos' / scale) + mean_pos
                      rot = R_canon^T @ rot'

    Bounded variant:  pos = R_canon^T @ (atanh(clamp(pos', ±(1-eps))) / scale_axis) + mean_pos
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

    pos = poses_flat[..., :3]    # (B, N, 3)
    rot6d = poses_flat[..., 3:]  # (B, N, 6)

    if params.bounded:
        # Pre-clamp to a safe region BEFORE atanh. atanh blows up at ±1.
        eps = float(params.atanh_eps)
        bound = 1.0 - eps
        pos_clamped = torch.clamp(pos, min=-bound, max=bound)
        pos_pre = torch.atanh(pos_clamped)  # (B, N, 3) — back in linear scaled space
    else:
        pos_pre = pos

    # Inverse position: unscale (per-axis), unrotate, untranslate.
    R_canon_T = params.R_canon.transpose(-1, -2)  # (B, 3, 3)
    # Per-axis scale: (B, 3) → (B, 1, 3) broadcasts over N.
    # No epsilon needed: scale is initialized to 1.0 and only set to
    # target_pos_std / std_axis, where std_axis is floored at POS_STD_MIN
    # (0.01). So scale ≤ target_pos_std / POS_STD_MIN = 50 — never zero.
    pos_unscaled = pos_pre / params.scale[:, None, :]  # (B, N, 3)
    pos_unrotated = torch.einsum("bij,bnj->bni", R_canon_T, pos_unscaled)  # (B, N, 3)
    pos_orig = pos_unrotated + params.mean_pos[:, None, :]

    # Inverse rotation: R_canon^T @ R_canonicalized (unchanged)
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
        bounded: Use the bounded (depth-filter + per-axis std + tanh) variant.
        bounded_dep_thr: Depth threshold for the bounded compute set.
        bounded_topk_frac: Fallback fraction when depth filter is empty.
        bounded_atanh_eps: Safety margin for the inverse atanh clamp.
        canon_rotation: Whether the agent will keep R_canon (True) or
            override it to identity (False). Used to choose the std frame in
            the bounded variant. Has no effect on the original variant.

    Example::

        canon = Canonicalizer(target_pos_std=0.5)
        x_canon, params = canon.forward(poses_9d, known_valid_mask)
        x_orig = canon.inverse(x_canon, params)
    """

    def __init__(
        self,
        target_pos_std: float = 0.5,
        bounded: bool = False,
        bounded_dep_thr: float = 1.5,
        bounded_topk_frac: float = 0.1,
        bounded_atanh_eps: float = 1.0e-4,
        canon_rotation: bool = True,
        per_axis_std: bool = False,
    ):
        self.target_pos_std = target_pos_std
        self.bounded = bounded
        self.bounded_dep_thr = bounded_dep_thr
        self.bounded_topk_frac = bounded_topk_frac
        self.bounded_atanh_eps = bounded_atanh_eps
        self.canon_rotation = canon_rotation
        self.per_axis_std = per_axis_std

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
            poses_flat,
            known_valid_mask,
            target_pos_std=self.target_pos_std,
            bounded=self.bounded,
            bounded_dep_thr=self.bounded_dep_thr,
            bounded_topk_frac=self.bounded_topk_frac,
            bounded_atanh_eps=self.bounded_atanh_eps,
            canon_rotation=self.canon_rotation,
            per_axis_std=self.per_axis_std,
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
