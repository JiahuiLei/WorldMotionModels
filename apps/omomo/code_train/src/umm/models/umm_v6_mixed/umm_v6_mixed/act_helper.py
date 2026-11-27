"""Action extraction helpers for converting SE3 trajectories to robot actions.

Provides functions for:
- Depth map backprojection to 3D point clouds.
- SE3 trajectory to delta end-effector + gripper action conversion (ManiSkill).
- Language-table EE delta action extraction.

Example::

    from umm.models.umm_v6_mixed.act_helper import get_act_func, extract_langtable_action
    act_fn = get_act_func(act_cfg)
    action = act_fn(se3traj, se3type)
    lt_act, lt_mask = extract_langtable_action(T_wl, uid, mask)
"""

import logging
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from umm.datasets.umm_v2.dataset_config import UID_LANGTABLE_ACT, UID_LANGTABLE_EE


def backproject_depth(
    depth: torch.Tensor,
    K: torch.Tensor,
    T_wc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Backproject a depth map to 3D points in world coordinates.

    Uses the pinhole camera model with intrinsic matrix K and optional
    camera-to-world extrinsic transform T_wc.

    Args:
        depth: Depth map of shape ``[B, H, W]``.
        K: Camera intrinsic matrix of shape ``[B, 3, 3]``.
        T_wc: Camera-to-world transform of shape ``[B, 4, 4]``.
            Defaults to identity if None.

    Returns:
        3D points in world coordinates of shape ``[B, H, W, 3]``.

    Example::

        points_3d = backproject_depth(depth, K, T_wc)
    """
    B, H, W = depth.shape
    depth = depth.reshape(B, -1)
    K = K.reshape(B, 3, 3)
    if T_wc is None:
        T_wc = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
        T_wc = T_wc.repeat(B, 1, 1)

    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    xs = xs.reshape(1, -1).repeat(B, 1)  # [B, H*W]
    ys = ys.reshape(1, -1).repeat(B, 1)  # [B, H*W]
    depth_flat = depth  # [B, H*W]

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    X = ((xs.float() - cx) * depth_flat) / fx
    Y = ((ys.float() - cy) * depth_flat) / fy
    Z = depth_flat

    cam_points = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # [B, H*W, 4]
    cam_points = cam_points.transpose(1, 2)  # [B, 4, H*W]

    world_points = torch.bmm(T_wc, cam_points)  # [B, 4, H*W]
    world_points = world_points.transpose(1, 2)[..., :3]  # [B, H*W, 3]
    points = world_points.reshape(B, H, W, 3)
    return points


def get_act_func(act_cfg: dict) -> Optional[Callable]:
    """Get the SE3-to-action conversion function based on configuration.

    Supports the following modes via ``act_cfg["action_convertion_mode"]``:
        - ``"maniskill_franka_pd_ee_delta_pos_v2"``: Delta EE position + gripper.
        - ``"maniskill_franka_pd_ee_delta_pos"``: Deprecated, raises ValueError.

    Args:
        act_cfg: Action configuration dictionary with keys like
            ``action_convertion_mode``, ``ee_idx_in_embodiment``, etc.

    Returns:
        A partial function that converts SE3 trajectories to actions,
        or None if no mode is configured.

    Example::

        act_fn = get_act_func({"action_convertion_mode": "maniskill_franka_pd_ee_delta_pos_v2"})
        action = act_fn(se3traj, se3type)
    """
    mode = act_cfg.get("action_convertion_mode", None)

    if mode == "maniskill_franka_pd_ee_delta_pos":
        raise ValueError("maniskill_franka_pd_ee_delta_pos is deprecated")
    elif mode == "maniskill_franka_pd_ee_delta_pos_v2":
        return partial(
            maniskill_franka_pd_ee_delta_pos_se3_to_action_v2,
            ee_idx_in_embodiment=act_cfg.get("ee_idx_in_embodiment", [12, 27]),
            finger0_idx_in_embodiment=act_cfg.get(
                "finger0_idx_in_embodiment", [13, 28]
            ),
            finger1_idx_in_embodiment=act_cfg.get(
                "finger1_idx_in_embodiment", [14, 29]
            ),
            delta_scaler=act_cfg.get("delta_scaler", 10.0),
            soft_sign_factor=act_cfg.get("soft_sign_factor", -1.0),
            detach_state_flag=act_cfg.get("detach_state_flag", False),
        )
    else:
        logging.warning("Action conversion not configured")
        return None


def fetch_se3(
    se3traj: torch.Tensor,
    se3type: torch.Tensor,
    idx: int,
) -> torch.Tensor:
    """Fetch the SE3 trajectory for a specific part type index.

    Selects the single part matching ``idx`` in each batch element and
    returns its trajectory.

    Args:
        se3traj: SE3 trajectory of shape ``[B, T, P, 4, 4]``.
        se3type: Part type IDs of shape ``[B, P]``.
        idx: Type index to extract.

    Returns:
        SE3 trajectory for the matching part, shape ``[B, T, 4, 4]``.

    Raises:
        AssertionError: If any batch element has != 1 matching part.

    Example::

        ee_traj = fetch_se3(se3traj, se3type, idx=12)
    """
    B = se3traj.shape[0]
    mask = se3type == idx
    assert (mask.sum(dim=1) == 1).all()
    s_se3 = se3traj.permute(0, 2, 1, 3, 4)[mask]  # [B, T, 4, 4]
    assert len(s_se3) == B
    return s_se3.clone()


def maniskill_franka_pd_ee_delta_pos_se3_to_action_v2(
    se3traj: torch.Tensor,
    se3type: torch.Tensor,
    ee_idx_in_embodiment: List[int] = [12, 27],
    finger0_idx_in_embodiment: List[int] = [13, 28],
    finger1_idx_in_embodiment: List[int] = [14, 29],
    delta_scaler: float = 10.0,
    soft_sign_factor: float = -1.0,
    detach_state_flag: bool = False,
) -> torch.Tensor:
    """Convert SE3 trajectories to Franka delta-EE + gripper actions (v2).

    Computes the delta between predicted action EE and current state EE
    positions, scaled by ``delta_scaler``. Gripper open/close is derived
    from the relative finger displacements.

    Each ``*_idx_in_embodiment`` is a 2-element list ``[action_uid, state_uid]``
    specifying which UIDs correspond to the action target and current state.

    Args:
        se3traj: SE3 trajectory of shape ``[B, T, P, 4, 4]``.
        se3type: Part type IDs of shape ``[B, P]``.
        ee_idx_in_embodiment: ``[action_ee_uid, state_ee_uid]``.
        finger0_idx_in_embodiment: ``[action_finger0_uid, state_finger0_uid]``.
        finger1_idx_in_embodiment: ``[action_finger1_uid, state_finger1_uid]``.
        delta_scaler: Multiplier for the EE position delta.
        soft_sign_factor: If negative, use hard sign for gripper;
            otherwise apply tanh with this factor.
        detach_state_flag: If True, detach state from computation graph.

    Returns:
        Action tensor of shape ``[B, T, 4]`` — (delta_x, delta_y, delta_z, gripper).

    Example::

        action = maniskill_franka_pd_ee_delta_pos_se3_to_action_v2(
            se3traj, se3type, ee_idx_in_embodiment=[12, 27]
        )
    """
    assert se3traj.ndim == 5
    se3traj = se3traj.clone()
    se3type = se3type.clone()

    state_se3 = fetch_se3(se3traj, se3type, ee_idx_in_embodiment[1])
    action_se3 = fetch_se3(se3traj, se3type, ee_idx_in_embodiment[0])
    state_t = state_se3[..., :3, 3].clone()
    action_t = action_se3[..., :3, 3].clone()
    if detach_state_flag:
        state_t = state_t.detach()
    delta_ee = (action_t - state_t) * delta_scaler

    finger0_se3_state = fetch_se3(se3traj, se3type, finger0_idx_in_embodiment[1])
    if detach_state_flag:
        finger0_se3_state = finger0_se3_state.detach()
    finger0_se3_action = fetch_se3(se3traj, se3type, finger0_idx_in_embodiment[0])
    finger0_T_state_action = torch.linalg.inv(finger0_se3_state) @ finger0_se3_action
    finger0_open = finger0_T_state_action[..., 1, 3]

    finger1_se3_state = fetch_se3(se3traj, se3type, finger1_idx_in_embodiment[1])
    if detach_state_flag:
        finger1_se3_state = finger1_se3_state.detach()
    finger1_se3_action = fetch_se3(se3traj, se3type, finger1_idx_in_embodiment[0])
    finger1_T_state_action = torch.linalg.inv(finger1_se3_state) @ finger1_se3_action
    finger1_open = -finger1_T_state_action[..., 1, 3]
    finger_open = (finger1_open + finger0_open) / 2.0

    if soft_sign_factor < 0.0:
        finger_open = finger_open.sign().float()
    else:
        finger_open = torch.tanh(finger_open * soft_sign_factor)

    action = torch.cat([delta_ee, finger_open[..., None]], dim=-1)
    return action


def extract_langtable_action(
    T_wl: torch.Tensor,
    uid: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract language-table EE delta actions from SE3 trajectories.

    Computes the 2D (x, y) displacement between the action target at time
    t+1 and the EE position at time t.

    Args:
        T_wl: World-to-link transforms of shape ``[B, T, P, 4, 4]``.
        uid: Unique part IDs of shape ``[B, P]``.
        mask: Validity mask of shape ``[B, T, P]``.

    Returns:
        Tuple of:
            - act: Delta actions of shape ``[B, T-1, 2]``.
            - valid_mask: Boolean validity mask of shape ``[B, T-1]``.

    Example::

        act, valid = extract_langtable_action(T_wl, uid, trace_mask)
    """
    with torch.no_grad():
        B, T, P, _, _ = T_wl.shape
        ee_mask = uid == UID_LANGTABLE_EE
        act_mask = uid == UID_LANGTABLE_ACT
        assert (ee_mask.sum(-1) == 1).all()
        assert (act_mask.sum(-1) == 1).all()
        ee_mask = ee_mask[:, None].expand(-1, T, -1)
        act_mask = act_mask[:, None].expand(-1, T, -1)
        ee_valid = mask[ee_mask].reshape(B, T)
        act_valid = mask[act_mask].reshape(B, T)
    ee_T = T_wl[ee_mask].reshape(B, T, 4, 4).clone()
    act_T = T_wl[act_mask].reshape(B, T, 4, 4).clone()
    act = act_T[:, 1:, :2, 3] - ee_T[:, :-1, :2, 3]
    valid_mask = act_valid[:, 1:] * ee_valid[:, :-1]
    return act, valid_mask
