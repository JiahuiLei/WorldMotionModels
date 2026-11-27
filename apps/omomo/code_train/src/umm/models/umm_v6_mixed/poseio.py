"""SE3 pose encoder/decoder with group-wise normalization (Version 3 & 4).

Encodes 4x4 SE3 transformation matrices into a compact 9D representation
(xyz + two rotated-offset columns) and decodes them back, applying
per-semantic-group normalization to stabilize diffusion training.

**V3**: ``[xyz, xyz + rot*rot_len]`` → normalize all 9 channels by same mean/std.
**V4**: normalize xyz first → ``[xyz_norm, xyz_norm + rot*rot_len]``. Rotation
offset is applied in *normalized* space, so ``rot_len`` has consistent magnitude
regardless of the dataset's raw spatial scale. This is critical for mixed-domain
training where datasets have wildly different position scales.

Example::

    from umm.models.umm_v6_mixed.poseio import PoseIO_V3, PoseIO_V4
    poseio = PoseIO_V4(sem_group_ids=[[-1]], rot_len=0.2)
    param = poseio.encode(T, sem_ids, stat_mask=None)  # [..., 9]
    T_rec = poseio.decode(param, sem_ids)               # [..., 4, 4]
"""

import numpy as np
import torch, torch.nn as nn
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
import torch.nn.functional as F
import matplotlib.cm as cm
import logging


class PoseIO_V3(nn.Module):
    """SE3 pose encoder/decoder with per-semantic-group normalization.

    Transforms 4x4 homogeneous matrices into a 9-channel representation:
    ``[xyz, xyz + rot_col0 * rot_len, xyz + rot_col1 * rot_len]``, then
    normalizes each channel group by learned mean/std per semantic group.

    Running statistics (mean, variance) are optionally tracked via Welford's
    online algorithm for monitoring data distribution during training.

    Args:
        sem_group_ids: List of lists of semantic IDs per normalization group.
            ``[-1]`` denotes the catch-all default group.
        param_mean: Per-group position means, shape ``[S, 3]``.
        param_std: Per-group position stds, shape ``[S, 3]``.
        std_factor: Per-group divisor applied to ``param_std``.
        state_mean_std_flag: Whether to track running statistics.
        rot_len: Scale factor for rotation column offsets.
        state_only_pos_flag: If True, only track position (first 3 channels)
            in running statistics.

    Example::

        poseio = PoseIO_V3(
            sem_group_ids=[[1, 2], [-1]],
            param_mean=[[0.0, 0.0, 0.9], [0.0, 0.0, 0.0]],
            param_std=[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        )
    """

    def __init__(
        self,
        sem_group_ids=[[-1]],
        param_mean=[[0.0, 0.0, 0.0]],
        param_std=[[1.0, 1.0, 1.0]],
        std_factor=[1.0],
        state_mean_std_flag=False,
        rot_len=1.0,
        state_only_pos_flag=True,
    ):
        """Initialize PoseIO_V3 with normalization parameters and optional running stats.

        Args:
            sem_group_ids: List of lists of semantic IDs per normalization group.
            param_mean: Per-group position means, shape ``[S, 3]``.
            param_std: Per-group position stds, shape ``[S, 3]``.
            std_factor: Per-group divisor applied to ``param_std``.
            state_mean_std_flag: Whether to track running statistics.
            rot_len: Scale factor for rotation column offsets.
            state_only_pos_flag: If True, only track xyz in running statistics.

        Example::

            poseio = PoseIO_V3(sem_group_ids=[[-1]], rot_len=1.0)
        """
        super().__init__()
        self.dim = 9
        self.rot_len = rot_len
        self.state_only_pos_flag = state_only_pos_flag

        self.sem_group_ids = sem_group_ids
        self.marked_sem_ids = []
        for it in self.sem_group_ids:
            if it != [-1]:
                self.marked_sem_ids.extend(it)

        S = len(sem_group_ids)

        if len(param_mean) == 0:
            param_mean = [[0.0, 0.0, 0.0]] * S
        else:
            assert len(param_mean) == S
        if len(param_std) == 0:
            param_std = [[1.0, 1.0, 1.0]] * S
        else:
            assert len(param_std) == S
        if len(std_factor) == 0:
            std_factor = [1.0] * S
        assert len(std_factor) == S
        for i in range(S):
            if not isinstance(std_factor[i], list):
                std_factor[i] = [std_factor[i]] * 3
        param_mean = torch.as_tensor(param_mean)
        param_std = torch.as_tensor(param_std)
        std_factor = torch.as_tensor(std_factor)
        param_std = param_std / std_factor

        self.register_buffer("param_mean", param_mean)
        self.register_buffer("param_std", param_std)

        self.state_mean_std_flag = state_mean_std_flag
        if self.state_mean_std_flag:
            # Initialize running statistics for state tracking
            self.register_buffer("running_mean", torch.zeros_like(self.param_mean))
            self.register_buffer("running_var", torch.ones_like(self.param_std))
            self.register_buffer("running_count", torch.zeros(S, dtype=torch.long))
            self.momentum = 0.1  # momentum for running average update

    def __get_mask__(self, sem_ids, i):
        """Build a boolean mask selecting elements belonging to normalization group ``i``.

        If the group's semantic ID list is ``[-1]`` (wildcard), the mask
        selects all elements *not* claimed by any other group.

        Args:
            sem_ids: Flattened semantic IDs, shape ``[N]``.
            i: Index into ``self.sem_group_ids``.

        Returns:
            Boolean mask of shape ``[N]``.

        Example::

            mask = poseio.__get_mask__(sem_ids_flat, group_idx=0)
        """
        sid = self.sem_group_ids[i]
        if sid == [-1]:
            rev_ids = self.marked_sem_ids
            rev_mask = (
                sem_ids[:, None] == torch.tensor(rev_ids, device=sem_ids.device)[None]
            )
            rev_mask = rev_mask.any(dim=-1)
            mask = ~rev_mask
        else:
            mask = sem_ids[:, None] == torch.tensor(sid, device=sem_ids.device)[None]
            mask = mask.any(dim=-1)
        return mask

    def encode(self, T, sem_ids, stat_mask, **kwargs):
        """
        Encode transformations with group-wise normalization.

        Args:
            T (torch.Tensor): Transformation matrices [..., 4, 4].
            sem_ids (torch.Tensor): Semantic IDs to determine normalization group.
            stat_mask (torch.Tensor): Mask for statistics update.

        Returns:
            torch.Tensor: Encoded parameters [..., 9].
        """

        # * must use this conversion because it handle the rescaling of the 7 channels
        # T: ..., 4,4; sem_ids: ...
        assert T.shape[:-2] == sem_ids.shape
        T = T.clone()
        t = T[..., :3, 3]
        R = T[..., :3, :3]

        R = R[..., :2]  # first two columns
        r1 = R[..., 0]
        r2 = R[..., 1]
        r1 = t + r1 * self.rot_len
        r2 = t + r2 * self.rot_len
        ret = torch.cat([t, r1, r2], dim=-1).clone()
        ret_shape = ret.shape
        ret = ret.reshape(-1, ret.shape[-1])
        sem_ids = sem_ids.reshape(-1)

        # ! for different group!
        for i in range(len(self.sem_group_ids)):
            mask = self.__get_mask__(sem_ids, i)
            if not mask.any():
                continue
            _n = mask.sum()
            mean = self.param_mean[i].clone().to(ret)[None].repeat(_n, 1)
            mean = torch.cat([mean, mean, mean], dim=-1)
            std = self.param_std[i].clone().to(ret)[None].repeat(_n, 1)
            std = torch.cat([std, std, std], dim=-1)
            mean, std = mean.detach(), std.detach()
            _ret = (ret[mask].clone() - mean) / torch.clamp(std, min=1e-6)
            ret[mask] = ret[mask].clone() * 0.0 + _ret
            if (
                self.state_mean_std_flag and ret.numel() > 0 and stat_mask is not None
            ):  # ! only stat the x channels
                stat_mask = stat_mask.reshape(-1)
                _mask = stat_mask & mask
                if _mask.any():
                    if self.state_only_pos_flag:
                        self._update_running_stats(ret[_mask][:, :3], i)
                    else:
                        # ! warning, here the rot len is in the original space!!!
                        to_state = ret[_mask].reshape(-1, 3)
                        self._update_running_stats(to_state, i)

        ret = ret.reshape(*ret_shape)
        return ret

    def decode(self, param, sem_ids, **kwargs):
        """
        Decode parameters with group-wise denormalization.

        Args:
            param (torch.Tensor): Encoded parameters [..., 9].
            sem_ids (torch.Tensor): Semantic IDs.

        Returns:
            torch.Tensor: Transformation matrices [..., 4, 4].
        """

        in_shape = param.shape
        assert in_shape[:-1] == sem_ids.shape
        param = param.clone().reshape(-1, param.shape[-1])
        sem_ids = sem_ids.reshape(-1)

        for i in range(len(self.sem_group_ids)):
            mask = self.__get_mask__(sem_ids, i)
            if not mask.any():
                continue
            _n = mask.sum()
            mean = self.param_mean[i].clone().to(param)[None].repeat(_n, 1)
            mean = torch.cat([mean, mean, mean], dim=-1)
            std = self.param_std[i].clone().to(param)[None].repeat(_n, 1)
            std = torch.cat([std, std, std], dim=-1)
            mean, std = mean.detach(), std.detach()
            _param = (param[mask].clone() * torch.clamp(std, min=1e-6)) + mean
            param[mask] = param[mask].clone() * 0.0 + _param
        old_dtype = param.dtype
        param = param.to(torch.float64)
        t = param[:, :3].clone()  # N,3
        r1 = param[:, 3:6].clone()
        r2 = param[:, 6:9].clone()
        r1 = (r1 - t) / self.rot_len
        r2 = (r2 - t) / self.rot_len
        R = torch.stack([r1, r2], dim=-1)  # [..., 3, 2]
        R = self._rectify_rotation_matrix(R)
        T = torch.cat([R, t[..., None]], dim=-1).to(old_dtype)  # N,3,4
        bottom = torch.Tensor([0, 0, 0, 1]).to(T)
        bottom = bottom[None, None].repeat(T.shape[0], 1, 1)  # N,1,4
        T = torch.cat([T, bottom], dim=-2)  # N,4,4
        return T.reshape(*in_shape[:-1], 4, 4)

    def rot_mask(self, mask):
        """Expand a per-part boolean mask to a 9-channel mask.

        Where ``mask`` is True the full 9D representation is kept; where
        False only the first 3 positional channels are retained (rotation
        channels are zeroed).

        Args:
            mask: Boolean mask of arbitrary leading shape ``[...]``.

        Returns:
            Boolean mask of shape ``[..., 9]``.

        Example::

            rot9 = poseio.rot_mask(rot_flag)  # [B, P] -> [B, P, 9]
        """
        # mask .... return ...., 9
        # mask if true full 9, otherwise only first 3 positional
        sh = mask.shape
        ret_true = torch.ones(sh + (9,), dtype=torch.bool, device=mask.device)
        ret_false = ret_true.clone()
        ret_false[..., 3:] = False
        ret = ret_true * mask[..., None] + ret_false * (~mask)[..., None]
        return ret.bool()

    def nonsigular_empty(self, x, rot_mask):
        """Fill position-only (non-SE3) entries with a non-singular default.

        For elements where ``rot_mask`` is False in the rotation channels,
        replaces those channels with an identity-like 9D vector to avoid
        singular rotation matrices during decoding.

        Args:
            x: Encoded parameters of shape ``[..., 9]``.
            rot_mask: Boolean mask of shape ``[..., 9]`` indicating which
                channels are valid.

        Returns:
            Tensor of shape ``[..., 9]`` with non-singular defaults inserted.

        Example::

            x_safe = poseio.nonsigular_empty(x, rot9_mask)
        """
        # x: ..., 9; rot_mask: ..., 9
        x_sh = x.shape
        empty_param = (
            torch.Tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .to(x.device)[None]
            .expand(x.reshape(-1, 9).shape[0], -1)
        )
        empty_param = empty_param.reshape(x_sh)
        return x * rot_mask.float() + (1.0 - rot_mask.float()) * empty_param

    @torch.no_grad()
    def _update_running_stats(self, in_data, i):
        """Update running mean/variance for group ``i`` using Welford's algorithm.

        Args:
            in_data: Data batch of shape ``[N, 3]`` (or ``[N, D]``).
            i: Normalization group index.

        Example::

            poseio._update_running_stats(pos_data, group_idx=0)
        """
        # data: N, 7
        data = in_data.detach().clone()
        batch_mean = data.mean(dim=0).detach().clone()  # 7
        batch_var = data.var(dim=0, unbiased=False).detach().clone()  # 7
        batch_count = data.shape[0]

        if self.running_count[i] == 0:
            # First batch
            self.running_mean[i].data.copy_(batch_mean)
            self.running_var[i].data.copy_(batch_var)
            self.running_count[i].data.copy_(
                torch.tensor(batch_count, dtype=torch.long, device=data.device)
            )
        else:
            # Update running statistics using Welford's online algorithm
            old_count = self.running_count[i].clone()
            new_count = old_count + batch_count

            # Update mean
            delta = batch_mean - self.running_mean[i]
            self.running_mean[i] += delta * batch_count / new_count

            # Update variance
            delta2 = batch_mean - self.running_mean[i]
            self.running_var[i] = (
                old_count * self.running_var[i]
                + batch_count * batch_var
                + delta * delta2 * old_count * batch_count / new_count
            ) / new_count

            self.running_count[i].data.copy_(new_count)

    def print_stats(self):
        """Log the current running mean and standard deviation for all groups.

        Raises:
            RuntimeError: If ``state_mean_std_flag`` is False.

        Example::

            poseio.print_stats()
        """
        if not self.state_mean_std_flag:
            raise RuntimeError(
                "state_mean_std_flag is False, no statistics are being tracked"
            )
        # Return running mean and std (sqrt of variance)
        for i in range(len(self.sem_group_ids)):
            logging.info(f"POSE-IO-V3 group {self.sem_group_ids[i]}:")
            running_std = torch.sqrt(self.running_var[i] + 1e-8)
            logging.info(f"POSE-IO-V3 running_mean: {self.running_mean[i]}")
            logging.info(f"POSE-IO-V3 running_std: {running_std}")
        return

    def reset_running_stats(self):
        """Reset running mean, variance, and count to initial values.

        Example::

            poseio.reset_running_stats()
        """
        if self.state_mean_std_flag:
            self.running_mean.data.zero_()
            self.running_var.data.fill_(1.0)
            self.running_count.data.zero_()

    def _rectify_rotation_matrix(self, R):
        """Gram-Schmidt orthogonalize a 3x2 matrix into a valid SO(3) rotation.

        Takes the first two column vectors, orthonormalizes them, and
        computes the third via cross product to ensure a proper rotation
        matrix with determinant +1.

        Args:
            R: Tensor of shape ``[N, 3, 2]`` containing two column vectors.

        Returns:
            Orthonormal rotation matrices of shape ``[N, 3, 3]``.

        Example::

            R_rect = poseio._rectify_rotation_matrix(R_approx)
        """
        # todo: can have better rotation error rect
        R_type = R.dtype
        R = R.double()

        # Extract the column vectors from the input matrix
        v1 = R[:, :, 0]  # First column
        v2 = R[:, :, 1]  # Second column

        # Step 1: r1 is just normalized first column
        r1 = F.normalize(v1, dim=1, eps=1e-8)

        # Step 2: r3 is normalized third column
        r3 = torch.cross(v1, v2, dim=1)
        # r3 = torch.cross(r1, F.normalize(v2, dim=1, eps=1e-8), dim=1)
        r3 = F.normalize(r3, dim=1, eps=1e-8)

        # Step 3: r2 is computed from cross product between r1 and r3_temp
        r2 = torch.cross(r3, r1, dim=1)
        r2 = F.normalize(r2, dim=1, eps=1e-8)

        # Construct the rectified rotation matrix, each r is a column vector
        R_rectified = torch.stack([r1, r2, r3], dim=2)  # N, 3, 3

        # assert torch.allclose(
        #     R_rectified.det(),
        #     torch.ones(
        #         R_rectified.shape[0], dtype=torch.float64, device=R_rectified.device
        #     ),
        #     f"Rot failure, the det_min={R_rectified.det().min()}",
        # )

        return R_rectified.to(R_type)


class PoseIO_V4(nn.Module):
    """SE3 pose encoder/decoder — normalize xyz first, then add rotation at fixed length.

    Key difference from V3: rotation columns are added AFTER position normalization,
    so ``rot_len`` is in normalized-space units.  This gives consistent rotation
    signal magnitude across datasets with different spatial scales.

    Encoding:
        1. Extract ``xyz`` and ``R`` from SE3.
        2. Normalize ``xyz``: ``xyz_norm = (xyz - mean) / std``.
        3. Build 9D: ``[xyz_norm, xyz_norm + rot_col0 * rot_len,
           xyz_norm + rot_col1 * rot_len]``.

    Decoding:
        1. Extract ``t_norm = param[:3]``, ``r1 = param[3:6]``, ``r2 = param[6:9]``.
        2. Recover rotation columns: ``rot_col = (r - t_norm) / rot_len``.
        3. Denormalize position: ``xyz = t_norm * std + mean``.
        4. Reconstruct SE3 from ``xyz`` and orthogonalized ``R``.

    Args:
        sem_group_ids: List of lists of semantic IDs per normalization group.
        param_mean: Per-group position means, shape ``[S, 3]``.
        param_std: Per-group position stds, shape ``[S, 3]``.
        std_factor: Per-group divisor applied to ``param_std``.
        state_mean_std_flag: Whether to track running statistics.
        rot_len: Rotation column offset length in *normalized* space.
        state_only_pos_flag: If True, only track position in running statistics.
    """

    def __init__(
        self,
        sem_group_ids=[[-1]],
        param_mean=[[0.0, 0.0, 0.0]],
        param_std=[[1.0, 1.0, 1.0]],
        std_factor=[1.0],
        state_mean_std_flag=False,
        rot_len=0.2,
        state_only_pos_flag=True,
    ):
        super().__init__()
        self.dim = 9
        self.rot_len = rot_len
        self.state_only_pos_flag = state_only_pos_flag

        self.sem_group_ids = sem_group_ids
        self.marked_sem_ids = []
        for it in self.sem_group_ids:
            if it != [-1]:
                self.marked_sem_ids.extend(it)

        S = len(sem_group_ids)

        if len(param_mean) == 0:
            param_mean = [[0.0, 0.0, 0.0]] * S
        else:
            assert len(param_mean) == S
        if len(param_std) == 0:
            param_std = [[1.0, 1.0, 1.0]] * S
        else:
            assert len(param_std) == S
        if len(std_factor) == 0:
            std_factor = [1.0] * S
        assert len(std_factor) == S
        for i in range(S):
            if not isinstance(std_factor[i], list):
                std_factor[i] = [std_factor[i]] * 3
        param_mean = torch.as_tensor(param_mean)
        param_std = torch.as_tensor(param_std)
        std_factor = torch.as_tensor(std_factor)
        param_std = param_std / std_factor

        self.register_buffer("param_mean", param_mean)
        self.register_buffer("param_std", param_std)

        self.state_mean_std_flag = state_mean_std_flag
        if self.state_mean_std_flag:
            self.register_buffer("running_mean", torch.zeros_like(self.param_mean))
            self.register_buffer("running_var", torch.ones_like(self.param_std))
            self.register_buffer("running_count", torch.zeros(S, dtype=torch.long))

    def __get_mask__(self, sem_ids, i):
        """Build boolean mask for normalization group ``i``. Same as V3."""
        sid = self.sem_group_ids[i]
        if sid == [-1]:
            rev_ids = self.marked_sem_ids
            rev_mask = (
                sem_ids[:, None] == torch.tensor(rev_ids, device=sem_ids.device)[None]
            )
            rev_mask = rev_mask.any(dim=-1)
            mask = ~rev_mask
        else:
            mask = sem_ids[:, None] == torch.tensor(sid, device=sem_ids.device)[None]
            mask = mask.any(dim=-1)
        return mask

    def encode(self, T, sem_ids, stat_mask, **kwargs):
        """Encode SE3 → 9D with position-first normalization.

        1. Extract xyz position and first two rotation columns.
        2. Normalize xyz per semantic group: ``xyz_norm = (xyz - mean) / std``.
        3. Add rotation at fixed length: ``[xyz_norm, xyz_norm + rot_col * rot_len]``.

        Args:
            T: Transformation matrices ``[..., 4, 4]``.
            sem_ids: Semantic IDs ``[...]``.
            stat_mask: Mask for running statistics update.

        Returns:
            Encoded parameters ``[..., 9]``.
        """
        assert T.shape[:-2] == sem_ids.shape
        T = T.clone()
        xyz = T[..., :3, 3]  # ..., 3
        R = T[..., :3, :3]
        rot_col0 = R[..., :, 0]  # ..., 3  (first column of rotation matrix)
        rot_col1 = R[..., :, 1]  # ..., 3  (second column)

        # Flatten for per-group processing
        orig_shape = xyz.shape  # ..., 3
        xyz_flat = xyz.reshape(-1, 3)  # N, 3
        rot0_flat = rot_col0.reshape(-1, 3)  # N, 3
        rot1_flat = rot_col1.reshape(-1, 3)  # N, 3
        sem_flat = sem_ids.reshape(-1)  # N

        # Step 1: normalize xyz per semantic group
        xyz_norm = xyz_flat.clone()
        for i in range(len(self.sem_group_ids)):
            mask = self.__get_mask__(sem_flat, i)
            if not mask.any():
                continue
            mean = self.param_mean[i].clone().to(xyz_norm).detach()  # (3,)
            std = torch.clamp(self.param_std[i].clone().to(xyz_norm).detach(), min=1e-6)  # (3,)
            xyz_norm[mask] = (xyz_flat[mask] - mean[None]) / std[None]

            # Update running stats on normalized xyz
            if self.state_mean_std_flag and stat_mask is not None:
                sm_flat = stat_mask.reshape(-1)
                valid = sm_flat & mask
                if valid.any():
                    self._update_running_stats(xyz_norm[valid], i)

        # Step 2: build 9D = [xyz_norm, xyz_norm + rot_col0 * rot_len, xyz_norm + rot_col1 * rot_len]
        r1 = xyz_norm + rot0_flat * self.rot_len
        r2 = xyz_norm + rot1_flat * self.rot_len
        ret = torch.cat([xyz_norm, r1, r2], dim=-1)  # N, 9

        return ret.reshape(*orig_shape[:-1], 9)

    def decode(self, param, sem_ids, **kwargs):
        """Decode 9D → SE3 with position-first denormalization.

        1. Extract ``t_norm``, ``r1``, ``r2`` from 9D.
        2. Recover rotation columns: ``rot_col = (r - t_norm) / rot_len``.
        3. Denormalize position: ``xyz = t_norm * std + mean``.
        4. Reconstruct SE3.

        Args:
            param: Encoded parameters ``[..., 9]``.
            sem_ids: Semantic IDs ``[...]``.

        Returns:
            Transformation matrices ``[..., 4, 4]``.
        """
        in_shape = param.shape
        assert in_shape[:-1] == sem_ids.shape
        param = param.clone().reshape(-1, 9)
        sem_flat = sem_ids.reshape(-1)

        t_norm = param[:, :3].clone()  # N, 3
        r1 = param[:, 3:6].clone()  # N, 3
        r2 = param[:, 6:9].clone()  # N, 3

        # Step 1: recover rotation columns in normalized space
        rot_col0 = (r1 - t_norm) / self.rot_len  # N, 3
        rot_col1 = (r2 - t_norm) / self.rot_len  # N, 3

        # Step 2: denormalize xyz
        xyz = t_norm.clone()
        for i in range(len(self.sem_group_ids)):
            mask = self.__get_mask__(sem_flat, i)
            if not mask.any():
                continue
            mean = self.param_mean[i].clone().to(xyz).detach()
            std = torch.clamp(self.param_std[i].clone().to(xyz).detach(), min=1e-6)
            xyz[mask] = t_norm[mask] * std[None] + mean[None]

        # Step 3: reconstruct SE3
        old_dtype = param.dtype
        rot_col0 = rot_col0.to(torch.float64)
        rot_col1 = rot_col1.to(torch.float64)
        R = torch.stack([rot_col0, rot_col1], dim=-1)  # N, 3, 2
        R = self._rectify_rotation_matrix(R)  # N, 3, 3

        xyz_64 = xyz.to(torch.float64)
        T_out = torch.cat([R, xyz_64[..., None]], dim=-1).to(old_dtype)  # N, 3, 4
        bottom = torch.tensor([0, 0, 0, 1], dtype=old_dtype, device=T_out.device)
        bottom = bottom[None, None].expand(T_out.shape[0], 1, 4)
        T_out = torch.cat([T_out, bottom], dim=-2)  # N, 4, 4

        return T_out.reshape(*in_shape[:-1], 4, 4)

    def rot_mask(self, mask):
        """Expand per-part boolean mask to 9-channel mask. Same as V3."""
        sh = mask.shape
        ret_true = torch.ones(sh + (9,), dtype=torch.bool, device=mask.device)
        ret_false = ret_true.clone()
        ret_false[..., 3:] = False
        ret = ret_true * mask[..., None] + ret_false * (~mask)[..., None]
        return ret.bool()

    def nonsigular_empty(self, x, rot_mask):
        """Fill position-only entries with non-singular default.

        For V4, the default 9D for a position-only point at the origin is:
        ``[0, 0, 0, rot_len, 0, 0, 0, rot_len, 0]`` — identity rotation
        columns at the configured rot_len offset in normalized space.
        """
        x_sh = x.shape
        empty_param = torch.tensor(
            [0.0, 0.0, 0.0, self.rot_len, 0.0, 0.0, 0.0, self.rot_len, 0.0],
            dtype=x.dtype, device=x.device,
        )[None].expand(x.reshape(-1, 9).shape[0], -1)
        empty_param = empty_param.reshape(x_sh)
        return x * rot_mask.float() + (1.0 - rot_mask.float()) * empty_param

    @torch.no_grad()
    def _update_running_stats(self, in_data, i):
        """Update running mean/variance for group ``i`` using Welford's algorithm."""
        data = in_data.detach().clone()
        batch_mean = data.mean(dim=0).detach().clone()
        batch_var = data.var(dim=0, unbiased=False).detach().clone()
        batch_count = data.shape[0]

        if self.running_count[i] == 0:
            self.running_mean[i].data.copy_(batch_mean)
            self.running_var[i].data.copy_(batch_var)
            self.running_count[i].data.copy_(
                torch.tensor(batch_count, dtype=torch.long, device=data.device)
            )
        else:
            old_count = self.running_count[i].clone()
            new_count = old_count + batch_count
            delta = batch_mean - self.running_mean[i]
            self.running_mean[i] += delta * batch_count / new_count
            delta2 = batch_mean - self.running_mean[i]
            self.running_var[i] = (
                old_count * self.running_var[i]
                + batch_count * batch_var
                + delta * delta2 * old_count * batch_count / new_count
            ) / new_count
            self.running_count[i].data.copy_(new_count)

    def print_stats(self):
        """Log current running mean and std for all groups."""
        if not self.state_mean_std_flag:
            raise RuntimeError("state_mean_std_flag is False")
        for i in range(len(self.sem_group_ids)):
            logging.info(f"POSE-IO-V4 group {self.sem_group_ids[i]}:")
            running_std = torch.sqrt(self.running_var[i] + 1e-8)
            logging.info(f"POSE-IO-V4 running_mean: {self.running_mean[i]}")
            logging.info(f"POSE-IO-V4 running_std: {running_std}")

    def reset_running_stats(self):
        """Reset running statistics."""
        if self.state_mean_std_flag:
            self.running_mean.data.zero_()
            self.running_var.data.fill_(1.0)
            self.running_count.data.zero_()

    @staticmethod
    def _rectify_rotation_matrix(R):
        """Gram-Schmidt orthogonalize a 3x2 matrix into valid SO(3). Same as V3."""
        R_type = R.dtype
        R = R.double()
        v1 = R[:, :, 0]
        v2 = R[:, :, 1]
        r1 = F.normalize(v1, dim=1, eps=1e-8)
        r3 = torch.cross(v1, v2, dim=1)
        r3 = F.normalize(r3, dim=1, eps=1e-8)
        r2 = torch.cross(r3, r1, dim=1)
        r2 = F.normalize(r2, dim=1, eps=1e-8)
        R_rectified = torch.stack([r1, r2, r3], dim=2)
        return R_rectified.to(R_type)
