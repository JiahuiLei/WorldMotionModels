"""SDF query utilities for hand-object penetration loss.

UNIT NOTE: data_wmm/omomo_v2/rest_object_sdf_v2/*.npy stores SDF values
in METRES (regenerated from rest_object_geo/*.ply via Phase A0).
Negative = inside, positive = outside. Axis order: (Z, Y, X) per the
bake script's transpose to match PyTorch grid_sample's (D, H, W).
"""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_sdf_asset(stem_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single SDF asset.

    Args:
        stem_path: stem like "data_wmm/omomo_v2/rest_object_sdf_v2/clothesstand"

    Returns:
        sdf: (256, 256, 256) float32, in METRES, axis order (Z, Y, X), negative=inside
        centroid: (3,) float32, AABB centroid in REST FRAME (metres)
        extents: (3,) float32, AABB extents in metres (already includes padding)
    """
    sdf = np.load(stem_path + ".npy").astype(np.float32)
    meta = json.load(open(stem_path + ".json"))
    assert meta.get("axis_order", "ZYX") == "ZYX", f"unexpected axis_order {meta.get('axis_order')}"
    assert meta.get("sign_convention", "negative=inside") == "negative=inside"
    centroid = np.asarray(meta["centroid"], dtype=np.float32)
    extents = np.asarray(meta["extents"], dtype=np.float32)
    return sdf, centroid, extents


def query_sdf_at_world_points(
    pts_world: torch.Tensor,        # (B, T, P, 3) in metres, world frame
    obj_R: torch.Tensor,            # (B, T, 3, 3) world rotation of object
    obj_t: torch.Tensor,            # (B, T, 3) world translation of object
    sdf_grid: torch.Tensor,         # (B, 256, 256, 256) per-sample SDF, axis order (Z, Y, X), metres
    sdf_centroid: torch.Tensor,     # (B, 3) AABB centroid in REST frame, metres
    sdf_extents: torch.Tensor,      # (B, 3) AABB extents in metres
) -> torch.Tensor:
    """Query SDF at world points -> object rest frame -> grid_sample.

    Returns:
        (B, T, P) signed distance in METRES.
    """
    B, T, P, _ = pts_world.shape
    # World -> rest: x_rest = R^T @ (x_world - t)
    pts_centered = pts_world - obj_t[:, :, None, :]
    pts_rest = torch.einsum("btij,btpj->btpi", obj_R.transpose(-1, -2), pts_centered)
    # Rest (X, Y, Z) in metres -> grid coords [-1, 1] for grid_sample (x, y, z) order
    aabb_min = (sdf_centroid - sdf_extents / 2.0)[:, None, None, :]
    extents_b = sdf_extents[:, None, None, :]
    # grid coord in (x, y, z) frame, mapped from physical pos to [-1, 1]
    grid_xyz = (pts_rest - aabb_min) / extents_b * 2.0 - 1.0  # (B, T, P, 3) in (X, Y, Z) order
    # F.grid_sample 3D expects grid (N, D_out, H_out, W_out, 3) with last-dim ordering (x, y, z).
    # Our SDF is stored as (D=Z, H=Y, W=X), so grid order (x, y, z) IS (W, H, D) = pts_rest order.
    grid_query = grid_xyz.reshape(B, 1, T, P, 3)
    sdf_in = sdf_grid[:, None]  # (B, 1, D=256, H=256, W=256)
    sdf_at_pts = F.grid_sample(
        sdf_in, grid_query,
        mode="bilinear", padding_mode="border", align_corners=True,
    )  # (B, 1, 1, T, P)
    return sdf_at_pts.squeeze(1).squeeze(1)  # (B, T, P)
