"""Port of OMOMO ``compute_collision`` (Pbody / object-human SDF penetration).

Source: ``ext/baselines/omomo_release/evaluation_metrics.py:159-276``.

Differences from baseline:

* SDF folder path is a function argument, defaulting to
  ``./data/omomo/raw/data/rest_object_sdf_256_npy_files`` (the 
  layout; the inner ``raw/data/`` comes from the OMOMO zip extraction).
  Baseline hard-codes ``./data/rest_object_sdf_256_npy_files``.
* Debug-visualization branch (``vis_debug``) is removed.
* All imports live at module top; numerics are unchanged.
"""

import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_collision(
    ori_verts_pred: torch.Tensor,
    human_faces: np.ndarray,
    obj_verts: torch.Tensor,
    obj_faces: np.ndarray,
    obj_name: str,
    obj_scale: torch.Tensor,
    obj_rot_mat: torch.Tensor,
    obj_trans: torch.Tensor,
    actual_len: int,
    sdf_folder: str = "./data/omomo/raw/data/rest_object_sdf_256_npy_files",
) -> Tuple[float, float]:
    """Object-vs-human penetration metrics via rest-pose SDF grid sampling.

    Args:
        ori_verts_pred: ``(T, Nv, 3)`` predicted human mesh verts (world).
        human_faces: ``(Nf, 3)`` SMPL-X faces (unused; parity).
        obj_verts: ``(T, No, 3)`` object verts (unused here; parity).
        obj_faces: ``(Nf, 3)`` object faces (unused; parity).
        obj_name: Base name of object (e.g. ``'largebox'``). SDF files are
            ``<obj_name>_cleaned_simplified.obj.npy`` / ``.json``.
        obj_scale: ``(T,)`` object scale factor.
        obj_rot_mat: ``(T, 3, 3)`` object world rotation matrices.
        obj_trans: ``(T, 3)`` object world translation.
        actual_len: Valid frame count.
        sdf_folder: Folder containing the per-object SDF ``.npy`` + ``.json``.

    Returns:
        ``(pen_percent, pen_loss)`` — fraction of frames with a penetration
        above ``pen_thresh=0.04`` and mean penetration depth on those frames.
    """
    _ = human_faces
    _ = obj_verts
    _ = obj_faces

    sdf_path = os.path.join(
        sdf_folder, obj_name + "_cleaned_simplified.obj.npy"
    )
    sdf_data = np.load(sdf_path)  # 256 X 256 X 256

    # Bring human verts into object rest frame.
    tmp_verts = (ori_verts_pred - obj_trans[:, None, :]) * (
        1 / obj_scale[:, None, None]
    )
    transformed_human_verts = torch.matmul(
        obj_rot_mat.transpose(1, 2), tmp_verts.transpose(1, 2)
    )
    transformed_human_verts = transformed_human_verts.transpose(1, 2)[:actual_len]
    nv = transformed_human_verts.shape[1]

    sdf_json_path = os.path.join(
        sdf_folder, obj_name + "_cleaned_simplified.obj.json"
    )
    with open(sdf_json_path, "r") as fh:
        sdf_json_data = json.load(fh)

    target_device = transformed_human_verts.device
    target_dtype = transformed_human_verts.dtype

    if "coord_center" in sdf_json_data:
        coord_center = np.asarray(sdf_json_data["coord_center"])
        coord_min = sdf_json_data["coord_min"]
        coord_max = sdf_json_data["coord_max"]

        query_human_verts = transformed_human_verts - torch.from_numpy(
            coord_center
        )[None, None, :].to(device=target_device, dtype=target_dtype)
        query_human_verts = (query_human_verts - coord_min) / (
            coord_max - coord_min
        )
        query_human_verts -= 0.5
        query_human_verts *= 2.0
    else:
        sdf_centroid = torch.from_numpy(
            np.asarray(sdf_json_data["centroid"])
        )[None, None, :].to(device=target_device, dtype=target_dtype)
        sdf_extents = np.asarray(sdf_json_data["extents"])
        query_human_verts = (
            (transformed_human_verts - sdf_centroid)
            * 2
            / sdf_extents.max()
        )

    query_human_verts = query_human_verts[:, :, [2, 1, 0]]

    sdf = torch.from_numpy(sdf_data).float().to(target_device)

    pen_thresh = 0.04
    pen_loss = torch.tensor(0.0, device=target_device, dtype=target_dtype)
    pen_cnt = 0

    num_steps = transformed_human_verts.shape[0]
    for t_idx in range(num_steps):
        signed_dists = F.grid_sample(
            sdf.unsqueeze(0).unsqueeze(0),
            query_human_verts[t_idx].reshape(1, nv, 1, 1, 3).float(),
            padding_mode="border",
            align_corners=True,
        )
        signed_dists = signed_dists.squeeze()
        signed_dists = signed_dists * obj_scale[t_idx]

        neg_dists_mask = signed_dists.lt(0).flatten()
        neg_dists = torch.abs(signed_dists[neg_dists_mask])
        if len(neg_dists) != 0:
            pen_mask = neg_dists.gt(pen_thresh).flatten()
            actual_neg_dists = neg_dists[pen_mask]
            if len(actual_neg_dists) > 0:
                pen_loss = pen_loss + actual_neg_dists.mean()
                pen_cnt += 1

    if pen_cnt > 0:
        pen_loss = pen_loss / pen_cnt
    else:
        pen_loss = torch.tensor(0.0, device=target_device, dtype=target_dtype)

    pen_percent = pen_cnt / num_steps if num_steps > 0 else 0.0
    return float(pen_percent), float(pen_loss.item())
