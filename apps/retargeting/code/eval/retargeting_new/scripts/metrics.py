# NOTE
# fps can be computed by total processing time and total frames.
# succ-rate: average by seq (succ or nto per seq) and per frame (upto failing)
# local and global keypoint error

# ? which keypiont to select?

# we play with more challenging settings to show the advantage.

# FPS, SR-seq, SR-timeframes, local-Err, abs-Err 5 rows

import numpy as np
import torch
from transforms3d.euler import euler2mat


def align_phy_to_kin(phy_Q, kin_Q):
    phy_Q = phy_Q.clone()
    kin_Q = kin_Q.clone()
    phy_center = phy_Q[0, 0, :2, 3].clone()
    phy_Q[:, :, :2, 3] = phy_Q[:, :, :2, 3] - phy_center[None, None, :]

    T_phy_align = torch.eye(4)
    ref_forward = kin_Q[0, 0, :2, 0]
    phy_forward = phy_Q[0, 0, :2, 0]
    angle = np.arctan2(ref_forward[1], ref_forward[0]) - np.arctan2(
        phy_forward[1], phy_forward[0]
    )
    R_phy_align = euler2mat(0.0, 0.0, float(angle), "sxyz")
    # R_phy_align = euler2mat(0.0,0.0,np.pi, "sxyz")
    T_phy_align[:3, :3] = torch.from_numpy(R_phy_align).float()
    phy_Q = torch.einsum("ij,...jk->...ik", T_phy_align, phy_Q)

    exe_center = kin_Q[0, 0, :2, 3].clone()
    phy_Q[:, :, :2, 3] = phy_Q[:, :, :2, 3] + exe_center[None, None, :]
    return phy_Q


def fail_detect(
    human_T,
    kin_T,
    phy_T,
    results: dict,
    fail_N=5,
    human_up_deg_th=45,
    robot_down_deg_th=45,
    robot_down_height_th=0.2,
):
    # ! now the fail is determined by continous fail_N frames if the human pose is upward and phy is lying
    human_T = human_T.clone()
    kin_T = kin_T.clone()
    phy_T = phy_T.clone()
    assert human_T.shape == kin_T.shape == phy_T.shape
    human_up_vec = human_T[:, :3, 1]
    phy_up_vec = phy_T[:, :3, 2]
    phy_height = phy_T[:, 2, 3]
    human_up_deg = np.arccos(human_up_vec @ np.array([0, 0, 1])) * 180 / np.pi
    phy_up_deg = np.arccos(phy_up_vec @ np.array([0, 0, 1])) * 180 / np.pi

    human_up = human_up_deg < human_up_deg_th
    phy_down = (phy_up_deg > robot_down_deg_th) & (phy_height < robot_down_height_th)
    cnt = 0
    ret = []
    for i in range(len(human_up)):
        if human_up[i] and phy_down[i]:
            cnt += 1
        else:
            cnt = 0
        if cnt >= fail_N:
            ret.append(False)
            break
        else:
            ret.append(True)
    ret = ret + [False] * (len(human_up) - len(ret))
    results["standing"] = np.array(ret)
    results["fail_up_th"] = human_up_deg_th
    results["fail_down_th"] = robot_down_deg_th
    results["fail_N"] = fail_N
    return results


def keypoint_track_error(kin_T, phy_T, results: dict):
    kin_T = kin_T.clone()
    phy_T = phy_T.clone()
    assert kin_T.shape == phy_T.shape
    # global
    global_err = (kin_T[..., :3, 3] - phy_T[..., :3, 3]).norm(dim=-1)
    results["global_tracking_err"] = global_err

    kin_wl = kin_T[:, 0]  # T,4,4
    phy_wl = phy_T[:, 0]  # T,4,4
    kin_lw = torch.linalg.inv(kin_wl)
    phy_lw = torch.linalg.inv(phy_wl)
    kin_T_local = torch.einsum("tij,tnjk->tnik", kin_lw, kin_T)
    phy_T_local = torch.einsum("tij,tnjk->tnik", phy_lw, phy_T)

    local_err = (kin_T_local[..., :3, 3] - phy_T_local[..., :3, 3]).norm(dim=-1)
    results["local_tracking_err"] = local_err
    return results


def human_fidelity(kin_T, phy_T, results: dict):
    return
