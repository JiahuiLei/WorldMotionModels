"""Utility functions for visualization helpers.

Provides camera pose construction, image layout, Procrustes alignment,
point-cloud sampling, and caption rendering used across all domain-specific
visualizers.

Example::

    from umm.models.umm_v6_mixed.viz_helpers.utils import make_viz_T_cam, make_square_cat_image
    cam = make_viz_T_cam(viz_cfg, batch)
    grid = make_square_cat_image(frame_list)
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


def move_camera_T_wc(back=0.0, lift=0.0, left=0.0):
    """Build a relative camera translation matrix.

    Constructs a 4x4 homogeneous transform that shifts the camera in its
    own coordinate frame along the X (left), Y (lift), and Z (back) axes.

    Args:
        back (float): Displacement along camera +Z (away from target).
        lift (float): Displacement along camera +Y (upward in camera frame).
        left (float): Displacement along camera +X (leftward in camera frame).

    Returns:
        torch.Tensor: 4x4 relative translation matrix.

    Example::

        T_rel = move_camera_T_wc(back=0.5, lift=0.1)
        T_cam = batch["cam_T_wc"][0, 0] @ T_rel
    """
    T = torch.eye(4)
    # Relative movement in camera frame
    # Assuming: +X right, +Y down, +Z forward (OpenCV) or similar
    # If "back" means pulling away from target -> -Z translation?
    # Or if T_wc represents the camera position in world.
    # If we multiply T_old @ T_rel, we are moving the camera frame relative to itself.
    # For now, implementing simple translation.
    T[0, 3] = left
    T[1, 3] = lift
    T[2, 3] = back
    return T


def make_square_cat_image(image_list):
    """Arrange a list of images into a square grid.

    Images are resized to a common (H, W), laid out in ceil(sqrt(N)) rows
    and columns, and padded with white if the total is not a perfect square.

    Args:
        image_list (list[np.ndarray]): List of images, each (H, W, C).

    Returns:
        np.ndarray: Concatenated grid image (H_grid, W_grid, 3).

    Example::

        grid = make_square_cat_image([frame_0, frame_1, frame_2, frame_3])
        # grid.shape == (2*H, 2*W, 3) for a 2x2 layout
    """
    import cv2

    n = len(image_list)
    if n == 0:
        return np.ones((1, 1, 3), dtype=np.uint8) * 255
    L = int(np.ceil(np.sqrt(n)))
    # Resize all images to same size to avoid concatenation mismatch (e.g. from
    # matplotlib bbox_inches="tight" producing slightly different dimensions)
    H = max(it.shape[0] for it in image_list)
    W = max(it.shape[1] for it in image_list)
    dtype = image_list[0].dtype
    resized = []
    for it in image_list:
        arr = it[..., :3] if it.ndim >= 3 else np.stack([it] * 3, axis=-1)
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
        resized.append(arr.astype(dtype))
    pad = np.ones((H, W, 3), dtype=dtype) * 255
    to_cat = resized + [pad] * (L * L - n)
    ret = []
    for i in range(L):
        ret.append(np.concatenate(to_cat[i * L : (i + 1) * L], axis=1))
    return np.concatenate(ret, axis=0)


def weighted_procrustes(A, B, w=None, eps=1e-16, return_T=True):
    """
    X: torch tensor B x N x 3
    Y: torch tensor B x N x 3
    w: torch tensor B x N
    """
    assert len(A) == len(B)
    if w is not None:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        a_mean = (w_norm * A).sum(dim=1, keepdim=True)
        b_mean = (w_norm * B).sum(dim=1, keepdim=True)

        A_c = A - a_mean
        B_c = B - b_mean

        H = torch.einsum("bni,bnj->bij", A_c, w_norm * B_c)
    else:
        a_mean = A.mean(axis=1, keepdim=True)
        b_mean = B.mean(axis=1, keepdim=True)

        A_c = A - a_mean
        B_c = B - b_mean

        H = torch.einsum("bij,bik->bjk", A_c, B_c)

    U, S, V = torch.svd(H)  # U: B x 3 x 3, S: B x 3, V: B x 3 x 3
    Z = torch.eye(3).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))  # B x 3 x 3
    R = V @ Z @ U.transpose(1, 2)  # B x 3 x 3
    t = b_mean - torch.einsum("bij,bjk->bik", R, a_mean.transpose(-2, -1)).transpose(
        -2, -1
    )
    if return_T:
        T = torch.eye(4).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
        T[:, :3, :3] = R
        T[:, :3, 3] = t.squeeze()
        return T
    return R, t.squeeze()


def get_viz_T_wc(viz_stand_at, viz_look_at):
    """Compute a camera-to-world transform from stand/look-at positions.

    Builds a right-handed camera frame where +Z points from ``viz_stand_at``
    toward ``viz_look_at``, +X is perpendicular to world-down, and +Y
    completes the basis.

    Args:
        viz_stand_at (list or np.ndarray): Camera position in world (3,).
        viz_look_at (list or np.ndarray): Look-at target in world (3,).

    Returns:
        torch.Tensor: 4x4 camera-to-world homogeneous transform.

    Example::

        T_wc = get_viz_T_wc([2.0, 0.0, 1.5], [0.0, 0.0, 0.0])
    """
    viz_from = torch.as_tensor(viz_stand_at, dtype=torch.float32)
    viz_to = torch.as_tensor(viz_look_at, dtype=torch.float32)
    z = nn.functional.normalize(viz_to - viz_from, dim=-1)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    x = torch.cross(down, z)
    if x.norm() < 1e-6:
        x = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x = nn.functional.normalize(x, dim=-1)
    y = torch.cross(z, x)
    y = nn.functional.normalize(y, dim=-1)
    R_wc = torch.stack([x, y, z], dim=1)  # 3,3
    T_wc = torch.eye(4, dtype=torch.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = viz_from
    return T_wc


def make_viz_T_cam(viz_cfg, batch):
    """Construct a visualization camera pose from config and batch data.

    Supports three modes:

    1. ``viz_cam_ith_view >= 0`` -- use the i-th camera from the batch with
       a small offset.
    2. ``viz_stand_at`` is set -- build camera from explicit stand/look-at.
    3. Fallback -- a simple translation offset.

    Args:
        viz_cfg (dict): Visualization config with keys like
            ``viz_cam_ith_view``, ``viz_stand_at``, ``viz_look_at``.
        batch (dict): Training batch (used when ``viz_cam_ith_view >= 0``).

    Returns:
        torch.Tensor: 4x4 camera-to-world transform.

    Example::

        cam = make_viz_T_cam({"viz_stand_at": [2, 0, 1]}, batch)
    """
    viz_cam_ith_view = viz_cfg.get("viz_cam_ith_view", -1)
    viz_stand_at = viz_cfg.get("viz_stand_at", None)
    viz_look_at = viz_cfg.get("viz_look_at", [0.0, 0.0, 0.0])

    if viz_cam_ith_view >= 0:
        viz_T_cam = batch["cam_T_wc"][0, viz_cam_ith_view].cpu() @ move_camera_T_wc(
            back=0.5, lift=0.1, left=-0.0
        )
    elif viz_stand_at is not None:
        viz_T_cam = get_viz_T_wc(viz_stand_at, viz_look_at)
    else:
        viz_T_cam = move_camera_T_wc(back=1.0, lift=0.5, left=-0.0)
    return viz_T_cam


def sample_list(x, M, random_flag=False):
    """
    Sample M elements from x.

    Args:
        x (np.ndarray): Input array [N, C].
        M (int): Number of samples desired.
        random_flag (bool): If True, sample randomly with replacement.
                            If False, sample uniformly (or repeat) to reach M.

    Returns:
        np.ndarray: Sampled array [M, C].
    """
    len_x = len(x)
    if len_x < M:
        if random_flag:
            # Random sample with replacement
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            # Repeat to reach M samples
            repeat_times = (M + len_x - 1) // len_x  # ceil division
            repeated = np.tile(x, (repeat_times,) + (1,) * (x.ndim - 1))
            return repeated[:M]
    elif len_x > M:
        if random_flag:
            # Random sample with replacement
            indices = np.random.choice(len_x, M, replace=True)
            return x[indices]
        else:
            # Uniform sampling
            indices = np.linspace(0, len_x - 1, M, dtype=np.int32)
            return x[indices]
    else:
        return x


def put_caption_at_bottom(image_list, caption):
    """Add a caption bar at the bottom of each frame in image_list.

    Auto-adjusts font size based on frame dimensions and word-wraps the
    caption into multiple lines when it does not fit in a single row.
    All frames in image_list are assumed to have the same (H, W).

    Args:
        image_list: list of numpy arrays, each (H, W, C) uint8.
        caption: string to render at the bottom.

    Returns:
        list of numpy arrays with caption bar appended, each (H + bar_h, W, C).
    """
    import cv2

    if not image_list or not caption:
        return image_list
    image_list = deepcopy(image_list)
    image_list = [np.asarray(it) for it in image_list]

    H, W = image_list[0].shape[:2]
    C = image_list[0].shape[2] if image_list[0].ndim == 3 else 3

    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_width = int(W * 0.9)

    # --- choose font scale proportional to frame height ---
    # target a single line of text being ~4% of frame height
    target_line_h = max(int(H * 0.04), 10)
    lo, hi = 0.2, 5.0
    font_scale = lo
    for _ in range(20):
        mid = (lo + hi) / 2.0
        (_, th), _ = cv2.getTextSize("Ag", font, mid, 1)
        if th < target_line_h:
            font_scale = mid
            lo = mid
        else:
            hi = mid

    thickness = max(1, int(font_scale * 1.5))
    (_, line_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_spacing = int(line_h * 0.4)  # gap between lines

    # --- word-wrap caption into lines that fit max_text_width ---
    # First, split by spaces
    space_words = caption.split(" ")
    split_words = []

    # Then split long properties by comma to allow wrapping between them
    for word in space_words:
        if "," in word:
            subwords = word.split(",")
            for i, sub in enumerate(subwords):
                if i < len(subwords) - 1:
                    split_words.append(sub + ",")
                elif sub:
                    split_words.append(sub)
        else:
            split_words.append(word)

    lines = []
    cur_line = ""
    for word in split_words:
        # Avoid putting space before commas if the previous word ended with comma
        test = (
            f"{cur_line}{word}"
            if cur_line and cur_line.endswith(",")
            else f"{cur_line} {word}".strip()
        )
        if not cur_line:
            test = word

        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)

        if tw <= max_text_width:
            cur_line = test
        else:
            if cur_line:
                lines.append(cur_line)
            cur_line = word

    if cur_line:
        lines.append(cur_line)
    if not lines:
        lines = [caption]

    # --- compute bar height ---
    n_lines = len(lines)
    pad_y = max(int(line_h * 0.5), 4)
    bar_h = n_lines * line_h + (n_lines - 1) * line_spacing + baseline + 2 * pad_y

    # --- render caption bar once (same for every frame) ---
    bar = np.ones((bar_h, W, C), dtype=image_list[0].dtype) * 255
    for i, line in enumerate(lines):
        (lw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = max((W - lw) // 2, 4)
        text_y = pad_y + line_h + i * (line_h + line_spacing)
        cv2.putText(
            bar,
            line,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # --- append bar to each frame ---
    result = []
    for img in image_list:
        frame = img[..., :C].copy() if img.shape[2] >= C else img.copy()
        result.append(np.concatenate([frame, bar], axis=0))
    return result
