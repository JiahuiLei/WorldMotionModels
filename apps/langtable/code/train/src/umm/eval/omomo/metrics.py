"""Port of OMOMO's ``evaluation_metrics.compute_metrics`` and helpers.

Source: ``ext/baselines/omomo_release/evaluation_metrics.py``. Numerics are
kept identical to the baseline. Distance metrics are returned in **mm**
(the baseline multiplies the underlying meters by ``1000`` — see ``*1000``
occurrences below).

The only non-trivial semantic point is that ``gt_rot_mat`` / ``pred_rot_mat``
are **body-joint global rotation matrices** of shape ``(T, 22, 3, 3)``:

* ``trainer_full_body_manip_diffusion.py:672-673``: ``global_rot_mat =
  transforms.rotation_6d_to_matrix(...)`` where the input is ``global_rot_6d``
  (22 x 6) — i.e. body-joint global orientations.
* ``trainer_full_body_manip_diffusion.py:758``: ``human_rot_list.append(
  global_rot_mat)``, later stacked (``:803``) and passed to
  ``compute_metrics`` (``:493-495``).
* ``evaluation_metrics.py:332-334``: only root index 0 is used via
  ``pred_rot_mat[:, 0]`` for the ``rot_dist`` ("oroot") metric.

Object rotation is used only by ``compute_collision``, not here.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from sklearn.cluster import DBSCAN


def get_frobenious_norm_rot_only(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Frobenius norm of rotation error matrices.

    Baseline: ``evaluation_metrics.py:16-25``.

    Args:
        x: ``(N, 3, 3)`` predicted rotation matrices.
        y: ``(N, 3, 3)`` ground-truth rotation matrices.

    Returns:
        Scalar mean of ``||I - x @ y^{-1}||_F`` across ``N``.
    """
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, "fro")
    return error / len(x)


def determine_floor_height_and_contacts(
    body_joint_seq: np.ndarray, fps: int
) -> float:
    """Estimate floor height from a sequence of body joints.

    Clusters low-velocity toe heights via DBSCAN and returns the median of
    the lowest-height cluster (the "floor").

    Baseline: ``evaluation_metrics.py:27-96``.

    Args:
        body_joint_seq: ``(N, 22, 3)`` body joint positions. ``z`` is up.
        fps: frames per second (unused in current port; kept for API parity).

    Returns:
        Estimated floor height (scalar, meters).
    """
    FLOOR_VEL_THRESH = 0.005
    # Unused but kept for semantic parity with baseline (floor offset).
    _FLOOR_HEIGHT_OFFSET = 0.01  # noqa: F841

    # compute toe velocities
    root_seq = body_joint_seq[:, 0, :]
    left_toe_seq = body_joint_seq[:, 10, :]
    right_toe_seq = body_joint_seq[:, 11, :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # foot heights (z up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[
        right_toe_vel < FLOOR_VEL_THRESH
    ]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(
        left_static_foot_heights, right_static_foot_heights
    )
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(
            all_static_foot_heights.reshape(-1, 1)
        )
        all_labels = np.unique(clustering.labels_)

        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )

            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = float(min_median)
    else:
        floor_height = 0.0

    return floor_height


def compute_foot_sliding_for_smpl(
    pred_global_jpos: np.ndarray, floor_height: float
) -> float:
    """Foot-sliding metric for SMPL-style skeleton.

    Baseline: ``evaluation_metrics.py:98-138``. The returned value is scaled
    by ``* 1000`` internally (mm).

    Args:
        pred_global_jpos: ``(T, J, 3)`` world-frame joint positions.
            J must include SMPL indices 7/8 (ankles) and 10/11 (toes).
        floor_height: Estimated floor height (meters).

    Returns:
        Scalar foot-sliding score in mm (averaged across 4 foot joints).
    """
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor z = 0 and compute.
    pred_global_jpos[:, :, 2] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :]  # T X 3
    ltoe_pos = pred_global_jpos[:, 10, :]  # T X 3

    rankle_pos = pred_global_jpos[:, 8, :]  # T X 3
    rtoe_pos = pred_global_jpos[:, 11, :]  # T X 3

    H_ankle = 0.08  # meter
    H_toe = 0.04  # meter

    lankle_disp = np.linalg.norm(
        lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis=1
    )
    ltoe_disp = np.linalg.norm(
        ltoe_pos[1:, :2] - ltoe_pos[:-1, :2], axis=1
    )
    rankle_disp = np.linalg.norm(
        rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis=1
    )
    rtoe_disp = np.linalg.norm(
        rtoe_pos[1:, :2] - rtoe_pos[:-1, :2], axis=1
    )

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, -1] < H_toe

    lankle_sliding_stats = np.abs(
        lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1] / H_ankle))
    )[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats) / seq_len * 1000

    ltoe_sliding_stats = np.abs(
        ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, -1] / H_toe))
    )[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats) / seq_len * 1000

    rankle_sliding_stats = np.abs(
        rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1] / H_ankle))
    )[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats) / seq_len * 1000

    rtoe_sliding_stats = np.abs(
        rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, -1] / H_toe))
    )[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats) / seq_len * 1000

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.0
    return float(sliding)


def compute_metrics(
    ori_verts_gt: torch.Tensor,
    ori_verts_pred: torch.Tensor,
    ori_jpos_gt: torch.Tensor,
    ori_jpos_pred: torch.Tensor,
    human_faces: np.ndarray,
    gt_trans: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_rot_mat: torch.Tensor,
    pred_rot_mat: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: np.ndarray,
    actual_len: int,
    use_joints24: bool = True,
    pred_obj_verts: Optional[torch.Tensor] = None,
) -> Tuple[
    float, float, float, float, float, float, float, float, float,
    float, float, float, float, float, float,
]:
    """Port of OMOMO ``compute_metrics`` (``evaluation_metrics.py:278-438``).

    All distance metrics are returned in **millimeters**. The rotation,
    contact precision/recall/F1, and accuracy metrics are unitless.

    Args:
        ori_verts_gt: ``(T, Nv, 3)`` GT mesh verts.
        ori_verts_pred: ``(T, Nv, 3)`` predicted mesh verts.
        ori_jpos_gt: ``(T, J, 3)`` GT joints (J=22 or 24).
        ori_jpos_pred: ``(T, J, 3)`` predicted joints.
        human_faces: ``(Nf, 3)`` SMPL-X face indices (unused in core metric
            computation; kept for signature parity).
        gt_trans: ``(T, 3)`` GT root translation.
        pred_trans: ``(T, 3)`` predicted root translation.
        gt_rot_mat: ``(T, 22, 3, 3)`` GT body-joint global rotation matrices
            (only index 0 = root is used).
        pred_rot_mat: ``(T, 22, 3, 3)`` predicted counterpart.
        obj_verts: ``(T, No, 3)`` world-frame object vertices.
        obj_faces: ``(Nf, 3)`` object faces (unused; kept for signature parity).
        actual_len: Valid frame count.
        use_joints24: Hand indices. Joint-24 layout uses idx 22/23; joint-22
            layout uses idx 20/21. Contact threshold also differs.
        pred_obj_verts: Optional ``(T, No, 3)`` predicted-side object verts.
            When provided, activates CHOIS-style semantics — pred-side
            contact is computed against ``pred_obj_verts``, GT-side
            against ``obj_verts`` (mirrors
            ``ext/baselines/chois_release/evaluation_metrics.py:354-464``).
            When ``None``, both sides use ``obj_verts`` — OMOMO semantics
            (single ``obj_verts`` in ``evaluation_metrics.py:278``).

    Returns:
        Tuple of 15 scalars:
        ``(lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err,
        gt_contact_dist, contact_dist, gt_foot_sliding_jnts, foot_sliding_jnts,
        contact_precision, contact_recall, contact_acc, contact_f1_score)``.
    """
    # Silence "unused" for parity args.
    _ = human_faces
    _ = obj_faces

    ori_verts_gt = ori_verts_gt[:actual_len]
    ori_verts_pred = ori_verts_pred[:actual_len]
    ori_jpos_gt = ori_jpos_gt[:actual_len]
    ori_jpos_pred = ori_jpos_pred[:actual_len]
    gt_trans = gt_trans[:actual_len]
    pred_trans = pred_trans[:actual_len]
    gt_rot_mat = gt_rot_mat[:actual_len]
    pred_rot_mat = pred_rot_mat[:actual_len]
    obj_verts = obj_verts[:actual_len]
    if pred_obj_verts is not None:
        pred_obj_verts = pred_obj_verts[:actual_len]

    # Calculate global hand joint position error
    if use_joints24:
        lhand_idx = 22
        rhand_idx = 23
    else:
        lhand_idx = 20
        rhand_idx = 21
    lhand_jpos_pred = ori_jpos_pred[:, lhand_idx, :].detach().cpu().numpy()
    rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy()
    lhand_jpos_gt = ori_jpos_gt[:, lhand_idx, :].detach().cpu().numpy()
    rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy()
    lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=1).mean() * 1000
    rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=1).mean() * 1000
    hand_jpe = (lhand_jpe + rhand_jpe) / 2.0

    # Calculate MPVPE
    verts_pred = ori_verts_pred - ori_jpos_pred[:, 0:1]
    verts_gt = ori_verts_gt - ori_jpos_gt[:, 0:1]
    verts_pred = verts_pred.detach().cpu().numpy()
    verts_gt = verts_gt.detach().cpu().numpy()
    mpvpe = np.linalg.norm(verts_pred - verts_gt, axis=2).mean() * 1000

    # Calculate MPJPE
    jpos_pred = ori_jpos_pred - ori_jpos_pred[:, 0:1]
    jpos_gt = ori_jpos_gt - ori_jpos_gt[:, 0:1]
    jpos_pred = jpos_pred.detach().cpu().numpy()
    jpos_gt = jpos_gt.detach().cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    # Translation error
    trans_err = (
        np.linalg.norm(
            pred_trans.detach().cpu().numpy() - gt_trans.detach().cpu().numpy(),
            axis=1,
        ).mean()
        * 1000
    )

    # Rotation error (root only)
    rot_mat_pred = pred_rot_mat.detach().cpu().numpy()[:, 0]
    rot_mat_gt = gt_rot_mat.detach().cpu().numpy()[:, 0]
    rot_dist = get_frobenious_norm_rot_only(
        rot_mat_pred.reshape(-1, 3, 3), rot_mat_gt.reshape(-1, 3, 3)
    )

    floor_height = determine_floor_height_and_contacts(
        ori_jpos_pred.detach().cpu().numpy(), fps=30
    )
    gt_floor_height = determine_floor_height_and_contacts(
        ori_jpos_gt.detach().cpu().numpy(), fps=30
    )

    foot_sliding_jnts = compute_foot_sliding_for_smpl(
        ori_jpos_pred.detach().cpu().numpy(), floor_height
    )
    gt_foot_sliding_jnts = compute_foot_sliding_for_smpl(
        ori_jpos_gt.detach().cpu().numpy(), gt_floor_height
    )

    # Contact metrics
    num_obj_verts = obj_verts.shape[1]
    if use_joints24:
        contact_threh = 0.05
    else:
        contact_threh = 0.10

    gt_lhand_jnt = ori_jpos_gt[:, lhand_idx, :]
    gt_rhand_jnt = ori_jpos_gt[:, rhand_idx, :]

    gt_lhand2obj_dist = torch.sqrt(
        (
            (
                gt_lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1)
                - obj_verts.to(gt_lhand_jnt.device)
            )
            ** 2
        ).sum(dim=-1)
    )
    gt_rhand2obj_dist = torch.sqrt(
        (
            (
                gt_rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1)
                - obj_verts.to(gt_rhand_jnt.device)
            )
            ** 2
        ).sum(dim=-1)
    )

    gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0]
    gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0]

    gt_lhand_contact = gt_lhand2obj_dist_min < contact_threh
    gt_rhand_contact = gt_rhand2obj_dist_min < contact_threh

    lhand_jnt = ori_jpos_pred[:, lhand_idx, :]
    rhand_jnt = ori_jpos_pred[:, rhand_idx, :]

    # CHOIS-style parity: when ``pred_obj_verts`` is provided, pred-side
    # contact is computed against the PREDICTED object geometry — a
    # self-consistent "did pred human land on pred object" check.
    # Matches ``ext/baselines/chois_release/evaluation_metrics.py:457-458``.
    # When ``pred_obj_verts`` is None, fall back to OMOMO semantics
    # (pred-side uses GT object, since OMOMO doesn't predict objects).
    pred_side_obj_verts = pred_obj_verts if pred_obj_verts is not None else obj_verts
    pred_side_num_obj_verts = pred_side_obj_verts.shape[1]

    lhand2obj_dist = torch.sqrt(
        (
            (
                lhand_jnt[:, None, :].repeat(1, pred_side_num_obj_verts, 1)
                - pred_side_obj_verts.to(lhand_jnt.device)
            )
            ** 2
        ).sum(dim=-1)
    )
    rhand2obj_dist = torch.sqrt(
        (
            (
                rhand_jnt[:, None, :].repeat(1, pred_side_num_obj_verts, 1)
                - pred_side_obj_verts.to(rhand_jnt.device)
            )
            ** 2
        ).sum(dim=-1)
    )

    lhand2obj_dist_min = lhand2obj_dist.min(dim=1)[0]
    rhand2obj_dist_min = rhand2obj_dist.min(dim=1)[0]

    lhand_contact = lhand2obj_dist_min < contact_threh
    rhand_contact = rhand2obj_dist_min < contact_threh

    num_steps = gt_lhand_contact.shape[0]

    contact_dist = 0.0
    gt_contact_dist = 0.0
    gt_contact_cnt = 0
    contact_dist_acc = torch.tensor(0.0, device=lhand2obj_dist_min.device)
    gt_contact_dist_acc = torch.tensor(0.0, device=lhand2obj_dist_min.device)
    for idx in range(num_steps):
        if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
            gt_contact_cnt += 1
            contact_dist_acc = contact_dist_acc + torch.minimum(
                lhand2obj_dist_min[idx], rhand2obj_dist_min[idx]
            )
            gt_contact_dist_acc = gt_contact_dist_acc + torch.minimum(
                gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx]
            )

    if gt_contact_cnt == 0:
        contact_dist = 0.0
        gt_contact_dist = 0.0
    else:
        contact_dist = float(
            contact_dist_acc.detach().cpu().numpy() / float(gt_contact_cnt)
        )
        gt_contact_dist = float(
            gt_contact_dist_acc.detach().cpu().numpy() / float(gt_contact_cnt)
        )

    # Precision / recall / F1 for contact classification.
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(num_steps):
        gt_in_contact = bool(gt_lhand_contact[idx] or gt_rhand_contact[idx])
        pred_in_contact = bool(lhand_contact[idx] or rhand_contact[idx])
        if gt_in_contact and pred_in_contact:
            TP += 1
        if (not gt_in_contact) and pred_in_contact:
            FP += 1
        if (not gt_in_contact) and (not pred_in_contact):
            TN += 1
        if gt_in_contact and (not pred_in_contact):
            FN += 1

    # Intentional divergence from baseline `evaluation_metrics.py:418`:
    # baseline has no zero-division guard and raises ZeroDivisionError on
    # 0-frame input. Port returns 0.0 instead. bit-parity run on
    # 30 windows × 120 frames never touches this branch (denom always > 0),
    # so the divergence is latent-only for valid inputs.
    contact_acc = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0

    if (TP + FP) == 0:
        contact_precision = 0.0
    else:
        contact_precision = TP / (TP + FP)

    if (TP + FN) == 0:
        contact_recall = 0.0
    else:
        contact_recall = TP / (TP + FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0.0
    else:
        contact_f1_score = (
            2 * (contact_precision * contact_recall)
            / (contact_precision + contact_recall)
        )

    return (
        float(lhand_jpe),
        float(rhand_jpe),
        float(hand_jpe),
        float(mpvpe),
        float(mpjpe),
        float(rot_dist),
        float(trans_err),
        float(gt_contact_dist),
        float(contact_dist),
        float(gt_foot_sliding_jnts),
        float(foot_sliding_jnts),
        float(contact_precision),
        float(contact_recall),
        float(contact_acc),
        float(contact_f1_score),
    )


def compute_object_trans_rot_metrics(
    gt_obj_trans: np.ndarray,
    pred_obj_trans: np.ndarray,
    gt_obj_rot_mat: np.ndarray,
    pred_obj_rot_mat: np.ndarray,
    actual_len: int,
    waypoints_index_list: Optional[list] = None,
) -> dict:
    """Object trajectory + rotation error metrics (CHOIS paper Table 1 cols).

    Mirrors ``ext/baselines/chois_release/evaluation_metrics.py:534-549``.

    All translation distances returned in **cm** (paper unit). ``o_obj`` is a
    unitless Frobenius norm.

    Args:
        gt_obj_trans: ``(T, 3)`` GT object COM translation.
        pred_obj_trans: ``(T, 3)`` predicted counterpart.
        gt_obj_rot_mat: ``(T, 3, 3)`` GT object rotation.
        pred_obj_rot_mat: ``(T, 3, 3)`` predicted counterpart.
        actual_len: Valid frame count (padding truncated).
        waypoints_index_list: Frame indices for T_xy computation. Defaults
            to CHOIS paper's ``[29, 59, 89]`` which is a T/4, T/2, 3T/4
            split for 120-frame windows.

    Returns:
        dict with ``t_s_cm``, ``t_e_cm``, ``t_xy_cm``, ``t_obj_cm``,
        ``o_obj`` (unitless).
    """
    if waypoints_index_list is None:
        waypoints_index_list = [29, 59, 89]

    gt_t = np.asarray(gt_obj_trans)[:actual_len]
    pr_t = np.asarray(pred_obj_trans)[:actual_len]
    gt_r = np.asarray(gt_obj_rot_mat)[:actual_len]
    pr_r = np.asarray(pred_obj_rot_mat)[:actual_len]

    t_obj_cm = float(np.linalg.norm(pr_t - gt_t, axis=1).mean() * 100)
    t_s_cm = float(np.linalg.norm(pr_t[0] - gt_t[0]) * 100)
    end_idx = max(actual_len - 1, 0)
    t_e_cm = float(np.linalg.norm(pr_t[end_idx] - gt_t[end_idx]) * 100)

    # Guard waypoint indices against short sequences.
    valid_wp = [i for i in waypoints_index_list if 0 <= i < actual_len]
    if valid_wp:
        wp_err = np.linalg.norm(
            pr_t[valid_wp, :2] - gt_t[valid_wp, :2], axis=1
        ).mean() * 100
        t_xy_cm = float(wp_err)
    else:
        t_xy_cm = float("nan")

    o_obj = float(
        get_frobenious_norm_rot_only(
            pr_r.reshape(-1, 3, 3), gt_r.reshape(-1, 3, 3)
        )
    )

    return {
        "t_s_cm": t_s_cm,
        "t_e_cm": t_e_cm,
        "t_xy_cm": t_xy_cm,
        "t_obj_cm": t_obj_cm,
        "o_obj": o_obj,
    }


def compute_foot_floor_height(
    pred_jpos: np.ndarray,
    floor_height: float,
    actual_len: int,
) -> float:
    """H_feet — mean |min-foot-z - floor_height| in cm.

    Foot joints in joints24 SMPL-X layout: lankle=7, rankle=8,
    ltoe=10, rtoe=11. For each frame we take the minimum z across
    these four joints and subtract the floor height. Returned in
    centimeters (paper unit). A perfect prediction with feet on floor
    returns 0.

    Args:
        pred_jpos: ``(T, J, 3)`` predicted joints.
        floor_height: scalar floor z in meters (output of
            ``determine_floor_height_and_contacts``).
        actual_len: valid frame count.

    Returns:
        Scalar mean-|foot-floor-gap| in cm.
    """
    jpos = np.asarray(pred_jpos)[:actual_len]
    foot_idxs = [7, 8, 10, 11]
    foot_z = jpos[:, foot_idxs, 2]  # (T, 4)
    min_foot_z_per_frame = foot_z.min(axis=1)  # (T,)
    gap_m = np.abs(min_foot_z_per_frame - floor_height)
    return float(gap_m.mean() * 100)
