"""Loss utilities for per-type weighted averaging.

Provides ``type_mean`` which computes a weighted mean over unique semantic
types, allowing different loss weights per embodiment or part group.

Example::

    from umm.models.umm_v2.loss_utils import type_mean
    loss = type_mean(per_element_loss, uid_tensor, {3: 2.0, 5: 0.5})
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
