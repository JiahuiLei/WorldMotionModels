"""Data utilities for preparing humanoid/human attribute tensors and timeline alignment.

Provides helper functions used by the Agent to set up semantic types,
group IDs, and UIDs for joint human--humanoid motion, as well as
temporal padding to unify asynchronous timelines.

Example::

    from umm.models.umm_v2.data_utils import prepare_attributes, pad_time
    sem, gids, uids, Ph, Pr = prepare_attributes("g1", "smplx", device)
    padded_t, h_mask, r_mask = pad_time(human_time, robot_time, 30.0, device)
"""

import torch
import sys
import os.path as osp
from copy import deepcopy

from umm.datasets.umm_v2.dataset_config import (
    HUMANOID_LINK_GROUP_DICT,
    SEM_HUMAN,
    SEM_LEFT_MANO,
    SEM_RIGHT_MANO,
    SEM_HUMANOID,
    UID_SMPLX55_GROUP,
    UID_HUMANOID_GROUP_DICT,
)


def prepare_attributes(humanoid_type, human_type, device):  # order: [human|robot]
    """
    Prepares attribute tensors for humanoid and human types.

    Args:
        humanoid_type (str): Type of the humanoid/robot (e.g., 'g1').
        human_type (str): Type of the human model (e.g., 'smplx').
        device (torch.device): Device to creating tensors on.

    Returns:
        Tuple of ``(sem_type, group_ids, uids, Ph, Pr)``:
            - sem_type: Semantic type IDs tensor.
            - group_ids: Group IDs (0 for human, 1 for robot).
            - uids: Combined UIDs for human and robot.
            - Ph: Number of human parts.
            - Pr: Number of robot parts.

    Example::

        sem, gids, uids, Ph, Pr = prepare_attributes("g1", "smplx", device)
    """
    assert human_type in ["smplx"], f"Invalid human type: {human_type}"
    robot_link_names = HUMANOID_LINK_GROUP_DICT[humanoid_type]
    robot_uids = UID_HUMANOID_GROUP_DICT[humanoid_type][: len(robot_link_names)]
    human_uids = deepcopy(UID_SMPLX55_GROUP)
    Ph, Pr = len(human_uids), len(robot_uids)
    uids = torch.as_tensor(human_uids + robot_uids, device=device).long()
    group_ids = torch.as_tensor([0] * Ph + [1] * Pr, device=device).long()
    # SMPLX+Robot
    sem_type = (
        [SEM_HUMAN] * 25
        + [SEM_LEFT_MANO] * 15
        + [SEM_RIGHT_MANO] * 15
        + [SEM_HUMANOID] * len(robot_uids)
        # ! warning, if use global factored version need to updat the sem here!!!!
    )
    sem_type = torch.as_tensor(sem_type, device=device).long()
    return sem_type, group_ids, uids, Ph, Pr


def pad_time(human_time, robot_time, fps: float, device):
    """
    Aligns and pads human and robot timelines to a common unified timeline.

    Args:
        human_time (torch.Tensor or None): Time stamps for human motion.
        robot_time (torch.Tensor or None): Time stamps for robot motion.
        fps (float): Frames per second for quantization.
        device (torch.device): Device for output tensors.

    Returns:
        Tuple of ``(padded_T, padded_T_human_mask, padded_T_robot_mask)``:
            - padded_T: Unified timeline tensor.
            - padded_T_human_mask: Bool mask indicating valid human frames.
            - padded_T_robot_mask: Bool mask indicating valid robot frames.

    Example::

        padded_t, h_mask, r_mask = pad_time(human_time, robot_time, 30.0, device)
    """
    # T or None
    # return: padded_T (T'), padded_T_human_mask (T'), padded_T_robot_mask (T')

    assert human_time is not None or robot_time is not None
    min_t, max_t = 1e10, -1e10
    if human_time is not None:
        min_t = min(min_t, human_time.min())
        max_t = max(max_t, human_time.max())
    if robot_time is not None:
        min_t = min(min_t, robot_time.min())
        max_t = max(max_t, robot_time.max())

    if human_time is not None:
        device = human_time.device
    else:
        device = robot_time.device

    # quantization
    min_idx = torch.round(torch.as_tensor(min_t) * fps).long()
    max_idx = torch.round(torch.as_tensor(max_t) * fps).long()

    len_t = max_idx - min_idx + 1

    # new timeline
    # [0, 1, 2, ...] + min_idx
    indices = torch.arange(len_t, device=device) + min_idx
    padded_T = indices.float() / fps  # T'

    # mask
    padded_T_human_mask = torch.zeros(len_t, dtype=torch.bool, device=device)
    padded_T_robot_mask = torch.zeros(len_t, dtype=torch.bool, device=device)

    # populate mask
    if human_time is not None:
        idx = torch.round(human_time * fps).long() - min_idx
        padded_T_human_mask[idx] = True

    if robot_time is not None:
        idx = torch.round(robot_time * fps).long() - min_idx
        padded_T_robot_mask[idx] = True

    return (
        padded_T.to(device),
        padded_T_human_mask.to(device),
        padded_T_robot_mask.to(device),
    )
