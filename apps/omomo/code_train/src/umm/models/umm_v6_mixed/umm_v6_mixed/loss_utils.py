"""Loss utilities for per-type weighted averaging.

Provides ``type_mean`` which computes a weighted mean over unique semantic
types, allowing different loss weights per embodiment or part group.

Also provides ``type_mean_balanced`` which splits elements into two groups
(robot+action vs scene) and returns separate per-group losses for balanced
reweighting.

Example::

    from umm.models.umm_v6_mixed.loss_utils import type_mean, type_mean_balanced
    loss = type_mean(per_element_loss, uid_tensor, {3: 2.0, 5: 0.5})
    robot_loss, scene_loss = type_mean_balanced(
        per_element_loss, uid_tensor, is_scene_mask, {3: 2.0}
    )
"""

import torch


def type_mean(x, x_type, weight_modifier={}):
    """
    Computes the mean of 'x' for each unique type in 'x_type', optionally weighting by 'weight_modifier'.

    Args:
        x (torch.Tensor): Input tensor of values.
        x_type (torch.Tensor): Tensor of type identifiers corresponding to 'x'.
        weight_modifier (dict): Dictionary mapping type ID (int) to a weight (float).

    Returns:
        Scalar mean value as a 0-dim tensor.

    Example::

        loss = type_mean(per_element_loss, uid_tensor, {3: 2.0})
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


def type_mean_balanced(x, x_type, is_scene, weight_modifier={}):
    """Compute per-type mean split into robot+action vs scene groups.

    Same logic as ``type_mean`` but partitions UIDs into two groups based on
    the ``is_scene`` boolean mask, then returns per-group means separately.

    Args:
        x (torch.Tensor): 1-D tensor of per-element loss values.
        x_type (torch.Tensor): 1-D tensor of UID integers, same length as *x*.
        is_scene (torch.Tensor): 1-D boolean tensor, True for scene elements.
        weight_modifier (dict): Maps UID (int) -> weight (float).

    Returns:
        Tuple of (robot_action_loss, scene_loss), each a scalar tensor.
        If a group has no elements, its loss is ``None``.

    Example::

        robot_loss, scene_loss = type_mean_balanced(
            loss_flat, uid_flat, is_scene_flat, {3: 2.0}
        )
    """
    robot_mask = ~is_scene  # robot + action elements
    scene_mask = is_scene

    robot_loss = None
    scene_loss = None

    if robot_mask.sum() > 0:
        robot_loss = type_mean(x[robot_mask], x_type[robot_mask], weight_modifier)
    if scene_mask.sum() > 0:
        scene_loss = type_mean(x[scene_mask], x_type[scene_mask], weight_modifier)

    return robot_loss, scene_loss
