"""Loss utilities for per-type weighted averaging.

Provides ``type_mean`` (original, compile-unsafe) and ``type_mean_compiled``
(scatter-based, torch.compile-safe) for computing weighted means over unique
semantic types.

Also provides ``type_mean_balanced`` / ``type_mean_balanced_compiled`` which
split elements into two groups (robot+action vs scene) and return separate
per-group losses for balanced reweighting.

Example::

    from umm.models.umm_v9.loss_utils import type_mean_compiled, type_mean_balanced_compiled
    loss = type_mean_compiled(per_element_loss, uid_tensor, {3: 2.0, 5: 0.5}, n_types=1024)
    robot_loss, scene_loss = type_mean_balanced_compiled(
        per_element_loss, uid_tensor, is_scene_mask, {3: 2.0}, n_types=1024
    )
"""

import torch
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Original implementation (compile-unsafe — uses torch.unique + boolean indexing)
# ---------------------------------------------------------------------------

@torch.compiler.disable
def type_mean(x, x_type, weight_modifier={}):
    """Compute weighted per-type mean. NOT torch.compile safe.

    Args:
        x (torch.Tensor): 1-D tensor of values.
        x_type (torch.Tensor): 1-D tensor of type identifiers.
        weight_modifier (dict): Maps type ID (int) -> weight (float).

    Returns:
        Scalar mean as 0-dim tensor.
    """
    masks = []
    weight = []
    with torch.no_grad():
        uids = torch.unique(x_type)
        for uid in uids:
            mask = x_type == uid
            masks.append(mask)
            if int(uid) in weight_modifier:
                weight.append(weight_modifier[int(uid)])
            else:
                weight.append(1.0)
    collected = [x[mask].mean() for mask in masks]
    collected = torch.stack(collected)
    weight = torch.tensor(weight).to(collected)
    ret = collected * weight
    return ret.mean()


@torch.compiler.disable
def type_mean_balanced(x, x_type, is_scene, weight_modifier={}):
    """Compute per-type mean split into robot+action vs scene. NOT compile safe.

    Returns:
        Tuple of (robot_action_loss, scene_loss), each scalar or None.
    """
    robot_mask = ~is_scene
    scene_mask = is_scene

    robot_loss = None
    scene_loss = None

    if robot_mask.sum() > 0:
        robot_loss = type_mean(x[robot_mask], x_type[robot_mask], weight_modifier)
    if scene_mask.sum() > 0:
        scene_loss = type_mean(x[scene_mask], x_type[scene_mask], weight_modifier)

    return robot_loss, scene_loss


# ---------------------------------------------------------------------------
# Compile-safe implementation (scatter-based, no torch.unique / boolean indexing)
# ---------------------------------------------------------------------------

@torch.compiler.disable
def _build_weight_vector(
    weight_modifier: Dict[int, float], n_types: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build weight vector from dict. Runs outside compile boundary."""
    weights = torch.ones(n_types, device=device, dtype=dtype)
    for uid, w in weight_modifier.items():
        if uid < n_types:
            weights[uid] = w
    return weights


def _type_mean_scatter(
    x: torch.Tensor,
    x_type: torch.Tensor,
    weights: torch.Tensor,
    n_types: int,
) -> torch.Tensor:
    """Core scatter computation. Fully compile-safe (no dicts, no loops)."""
    idx = x_type.long()  # (N,)

    # Per-type sum and count via scatter
    sums = torch.zeros(n_types, device=x.device, dtype=x.dtype)
    counts = torch.zeros(n_types, device=x.device, dtype=x.dtype)
    sums.scatter_add_(0, idx, x)
    counts.scatter_add_(0, idx, torch.ones_like(x))

    # Per-type mean (safe division)
    active = counts > 0  # (n_types,) bool
    means = sums / counts.clamp(min=1)  # (n_types,)

    # Weighted mean over active types only
    weighted = means * weights * active
    n_active = active.sum().clamp(min=1)
    return weighted.sum() / n_active


def type_mean_compiled(
    x: torch.Tensor,
    x_type: torch.Tensor,
    weight_modifier: Dict[int, float] = {},
    n_types: int = 1024,
) -> torch.Tensor:
    """Compute weighted per-type mean using scatter ops. torch.compile safe.

    Uses fixed-size scatter buffers instead of torch.unique + boolean indexing.
    Mathematically equivalent to ``type_mean``.

    Args:
        x: 1-D tensor of values.
        x_type: 1-D tensor of integer type identifiers (0 .. n_types-1).
        weight_modifier: Maps type ID (int) -> weight (float).
        n_types: Upper bound on type IDs. Buffer size = n_types.

    Returns:
        Scalar weighted mean as 0-dim tensor.
    """
    weights = _build_weight_vector(weight_modifier, n_types, x.device, x.dtype)
    return _type_mean_scatter(x, x_type, weights, n_types)


@torch.compiler.disable
def type_mean_balanced_compiled(
    x: torch.Tensor,
    x_type: torch.Tensor,
    is_scene: torch.Tensor,
    weight_modifier: Dict[int, float] = {},
    n_types: int = 1024,
) -> tuple:
    """Compute per-type mean split into robot+action vs scene.

    The outer branching (robot_mask.any()) is data-dependent so this wrapper
    runs outside compile. The inner _type_mean_scatter is compile-safe.

    Returns:
        Tuple of (robot_action_loss, scene_loss), each scalar or None.
    """
    robot_mask = ~is_scene
    scene_mask = is_scene

    weights = _build_weight_vector(weight_modifier, n_types, x.device, x.dtype)

    robot_loss = None
    scene_loss = None

    if robot_mask.any():
        robot_loss = _type_mean_scatter(
            x[robot_mask], x_type[robot_mask], weights, n_types
        )
    if scene_mask.any():
        scene_loss = _type_mean_scatter(
            x[scene_mask], x_type[scene_mask], weights, n_types
        )

    return robot_loss, scene_loss
