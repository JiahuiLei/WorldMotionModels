import torch
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles


def extract_ee_parts(sem_type_uid):
    """
    Extracts relevant UIDs for End Effector parts from the semantic type mapping.
    """

    def get_first(val):
        if isinstance(val, list):
            return val[0] if val else None
        return val

    palm_uid = get_first(sem_type_uid.get("endeffector"))
    left_finger_uid = get_first(sem_type_uid.get("gripper_left_finger"))
    right_finger_uid = get_first(sem_type_uid.get("gripper_right_finger"))

    return palm_uid, left_finger_uid, right_finger_uid


def state_action_to_tgt(
    current_palm_pose,
    current_left_finger_pose,
    current_right_finger_pose,
    action,
    pos_limit=0.1,
    rot_limit=0.1,
):
    """
    Computes the target SE(3) poses for the End Effector (EE) palm and fingers based on the current state and action.
    """
    # Ensure batch dimension
    is_batched = action.ndim == 2
    if not is_batched:
        action = action.unsqueeze(0)
        current_palm_pose = current_palm_pose.unsqueeze(0)
        current_left_finger_pose = current_left_finger_pose.unsqueeze(0)
        current_right_finger_pose = current_right_finger_pose.unsqueeze(0)

    device = action.device
    B = action.shape[0]

    # 1. Parse Action
    delta_pos_norm = action[:, :3]
    delta_rot_norm = action[:, 3:6]
    gripper_action = action[:, 6]  # [-1, 1]

    # 2. Denormalize Deltas (World Frame)
    delta_pos = delta_pos_norm * pos_limit
    # NOTE(leijh): ManiSkill 1.0/2.0 rotation is often inverted (uses rot_lower which is -0.1)
    # We use -rot_limit here to match the ManiSkill behavior where positive action leads to negative rotation
    delta_euler = delta_rot_norm * (-rot_limit)

    # 3. Compute Target Palm Pose
    curr_palm_p = current_palm_pose[:, :3, 3]
    curr_palm_rot_mat = current_palm_pose[:, :3, :3]

    target_palm_p = curr_palm_p + delta_pos

    # Differentiable Rotation Matrix Construction using PyTorch3D
    delta_rot_mat = euler_angles_to_matrix(delta_euler.flip(-1), "ZYX")

    # Apply Delta Rotation (Left multiply for World Frame)
    target_palm_rot_mat = delta_rot_mat @ curr_palm_rot_mat

    target_palm_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    target_palm_pose[:, :3, :3] = target_palm_rot_mat
    target_palm_pose[:, :3, 3] = target_palm_p

    # 4. Compute Finger Targets (Relative to Target Palm)
    curr_palm_inv = torch.inverse(current_palm_pose)
    T_palm_left = curr_palm_inv @ current_left_finger_pose
    T_palm_right = curr_palm_inv @ current_right_finger_pose

    # Target joint pos: 0.04 * (gripper + 1) / 2
    target_d = 0.04 * (gripper_action + 1) / 2.0  # (B,)

    # Left Finger
    left_y = T_palm_left[:, 1, 3]
    left_sign = torch.sign(left_y)
    left_sign[left_sign == 0] = 1.0  # Default positive

    T_palm_left_target = T_palm_left.clone()
    T_palm_left_target[:, 1, 3] = left_sign * target_d

    # Right Finger
    right_y = T_palm_right[:, 1, 3]
    right_sign = torch.sign(right_y)
    right_sign[right_sign == 0] = -1.0  # Default negative

    T_palm_right_target = T_palm_right.clone()
    T_palm_right_target[:, 1, 3] = right_sign * target_d

    # Transform back to World
    target_left_finger_pose = target_palm_pose @ T_palm_left_target
    target_right_finger_pose = target_palm_pose @ T_palm_right_target

    if not is_batched:
        return (
            target_palm_pose.squeeze(0),
            target_left_finger_pose.squeeze(0),
            target_right_finger_pose.squeeze(0),
        )

    return target_palm_pose, target_left_finger_pose, target_right_finger_pose


def tgt_to_state_action(
    current_palm_pose,
    target_palm_pose,
    target_left_finger_pose,
    target_right_finger_pose,
    pos_limit=0.1,
    rot_limit=0.1,
):
    """
    Inverse function of state_action_to_tgt.
    Recovers the action vector given the current and target poses.
    """
    is_batched = current_palm_pose.ndim == 3
    if not is_batched:
        current_palm_pose = current_palm_pose.unsqueeze(0)
        target_palm_pose = target_palm_pose.unsqueeze(0)
        target_left_finger_pose = target_left_finger_pose.unsqueeze(0)
        target_right_finger_pose = target_right_finger_pose.unsqueeze(0)

    device = current_palm_pose.device
    B = current_palm_pose.shape[0]

    # 1. Delta Position
    curr_p = current_palm_pose[:, :3, 3]
    target_p = target_palm_pose[:, :3, 3]
    
    # 2. Delta Rotation
    curr_rot = current_palm_pose[:, :3, :3]
    target_rot = target_palm_pose[:, :3, :3]

    delta_pos = target_p - curr_p
    delta_pos_norm = delta_pos / pos_limit

    # target = delta @ curr  =>  delta = target @ curr^T
    delta_rot_mat = target_rot @ curr_rot.transpose(1, 2)

    # Convert to euler angles using PyTorch3D
    delta_euler = matrix_to_euler_angles(delta_rot_mat, "ZYX").flip(-1)
    # NOTE(leijh): ManiSkill 1.0/2.0 rotation is often inverted (uses rot_lower which is -0.1)
    # We use -rot_limit here to match the ManiSkill behavior
    delta_rot_norm = delta_euler / (-rot_limit)

    # 3. Gripper Action
    target_palm_inv = torch.inverse(target_palm_pose)
    T_palm_left_target = target_palm_inv @ target_left_finger_pose
    
    target_d = torch.abs(T_palm_left_target[:, 1, 3])
    gripper_action = (target_d * 2.0 / 0.04) - 1.0

    action = torch.cat(
        [delta_pos_norm, delta_rot_norm, gripper_action.unsqueeze(-1)], dim=-1
    )

    if not is_batched:
        return action.squeeze(0)
    return action


def compute_target_pose_for_frame(
    current_states, action, sem_type_uid, pos_limit=0.1, rot_limit=0.1
):
    """
    Wrapper function to compute target poses from numpy state dictionaries.
    """
    palm_uid, left_uid, right_uid = extract_ee_parts(sem_type_uid)
    if palm_uid is None or str(palm_uid) not in current_states:
        return {}

    device = torch.device("cpu")
    to_t = lambda x: torch.from_numpy(x).float().to(device)

    curr_palm = to_t(current_states[str(palm_uid)])
    curr_left = to_t(current_states[str(left_uid)]) if left_uid else torch.eye(4)
    curr_right = to_t(current_states[str(right_uid)]) if right_uid else torch.eye(4)
    act = to_t(action)

    tgt_palm, tgt_left, tgt_right = state_action_to_tgt(
        curr_palm, curr_left, curr_right, act, pos_limit, rot_limit
    )

    res = {palm_uid: tgt_palm.numpy()}
    if left_uid:
        res[left_uid] = tgt_left.numpy()
    if right_uid:
        res[right_uid] = tgt_right.numpy()

    return res
