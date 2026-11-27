"""PointNet-based point cloud encoders with ResNet blocks.

Adapted from https://github.com/autonomousvision/occupancy_flow. Provides
several encoder variants that map variable-size point clouds to fixed-size
latent codes via a hierarchy of ResNet FC blocks and pooling operations.

Classes:
    ResnetBlockFC: FC ResNet block (pre-activation, zero-init residual).
    BNanyshape: BatchNorm1d wrapper that handles arbitrary leading dims.
    ResnetBlockFC_BN: FC ResNet block with BatchNorm.
    ResnetPointnet: Standard PointNet encoder with max-pooling.
    ResnetPointnetV2: PointNet encoder with BatchNorm and sigmoid activation.
    AttentionPooling: Learned attention-weighted pooling over point dim.
    ResnetPointnetAttenPooling: PointNet encoder with attention pooling.

Example:
    >>> net = ResnetPointnet(c_dim=128, dim=3, hidden_dim=256)
    >>> pts = torch.randn(4, 1024, 3)
    >>> code = net(pts)  # [4, 128]
"""

import torch
import torch.nn as nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, act_method=nn.LeakyReLU):
        """Initialize the FC ResNet block.

        Args:
            size_in (int): Input dimension.
            size_out (int, optional): Output dimension. Defaults to ``size_in``.
            size_h (int, optional): Hidden dimension. Defaults to
                ``min(size_in, size_out)``.
            act_method: Activation class. Defaults to ``nn.LeakyReLU``.

        Example:
            >>> block = ResnetBlockFC(256, 128)
        """
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = act_method()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        """Apply pre-activation residual block.

        Args:
            x (torch.Tensor): Input of shape ``[..., size_in]``.

        Returns:
            torch.Tensor: Output of shape ``[..., size_out]``.

        Example:
            >>> block = ResnetBlockFC(256, 128)
            >>> out = block(torch.randn(4, 256))  # [4, 128]
        """
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class BNanyshape(nn.Module):
    """BatchNorm1d wrapper that accepts tensors with arbitrary leading dimensions.

    Reshapes input to 2-D for BatchNorm, then restores the original shape.

    Args:
        size_in (int): Feature dimension (last axis).

    Example:
        >>> bn = BNanyshape(64)
        >>> out = bn(torch.randn(2, 10, 64))  # [2, 10, 64]
    """

    def __init__(self, size_in):
        """Initialize with a BatchNorm1d layer.

        Args:
            size_in (int): Feature dimension.

        Example:
            >>> bn = BNanyshape(64)
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(size_in)

    def forward(self, x):
        """Apply batch normalization to the last dimension.

        Args:
            x (torch.Tensor): Input of shape ``[..., size_in]``.

        Returns:
            torch.Tensor: Normalized tensor, same shape as input.

        Example:
            >>> bn = BNanyshape(64)
            >>> out = bn(torch.randn(2, 10, 64))
        """
        xs = x.shape
        return self.bn(x.reshape(-1, xs[-1])).reshape(*xs)


class ResnetBlockFC_BN(nn.Module):
    """Fully connected ResNet block with BatchNorm.

    Same structure as ``ResnetBlockFC`` but with ``BNanyshape`` normalization
    before each activation.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, act_method=nn.LeakyReLU):
        """Initialize the BN-augmented FC ResNet block.

        Args:
            size_in (int): Input dimension.
            size_out (int, optional): Output dimension. Defaults to ``size_in``.
            size_h (int, optional): Hidden dimension. Defaults to
                ``min(size_in, size_out)``.
            act_method: Activation class. Defaults to ``nn.LeakyReLU``.

        Example:
            >>> block = ResnetBlockFC_BN(256, 128)
        """
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = BNanyshape(size_in)
        self.fc_0 = nn.Linear(size_in, size_h)
        self.bn_1 = BNanyshape(size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = act_method()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        """Apply BN + pre-activation residual block.

        Args:
            x (torch.Tensor): Input of shape ``[..., size_in]``.

        Returns:
            torch.Tensor: Output of shape ``[..., size_out]``.

        Example:
            >>> block = ResnetBlockFC_BN(256, 128)
            >>> out = block(torch.randn(4, 256))  # [4, 128]
        """
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    """Perform max-pooling along a given dimension.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        dim (int): Dimension along which to pool. Defaults to -1.
        keepdim (bool): Whether to retain the pooled dimension. Defaults to False.

    Returns:
        torch.Tensor: Max-pooled tensor.

    Example:
        >>> out = maxpool(torch.randn(4, 1024, 128), dim=1)  # [4, 128]
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetPointnet(nn.Module):
    """PointNet-based encoder with ResNet FC blocks and max-pooling.

    Encodes a variable-size point cloud into a fixed-size latent code by
    alternating ResNet blocks with global max-pool aggregation.

    Args:
        c_dim (int): Output latent code dimension. Defaults to 128.
        dim (int): Per-point input dimension. Defaults to 3.
        hidden_dim (int): Hidden dimension of ResNet blocks. Defaults to 512.

    Example:
        >>> net = ResnetPointnet(c_dim=128, dim=3, hidden_dim=256)
        >>> code = net(torch.randn(4, 1024, 3))  # [4, 128]
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        """Initialize position projection, ResNet blocks, and output FC.

        Args:
            c_dim (int): Output latent code dimension.
            dim (int): Per-point input dimension.
            hidden_dim (int): Hidden dimension of ResNet blocks.

        Example:
            >>> net = ResnetPointnet(c_dim=128, dim=6, hidden_dim=256)
        """
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        """Encode a point cloud to a latent code.

        Args:
            p (torch.Tensor): Point cloud of shape ``[B, T, D]``.
            return_unpooled (bool): If True, also return pre-pooled features.

        Returns:
            torch.Tensor: Latent code ``[B, c_dim]``. If ``return_unpooled``
                is True, returns ``(code, unpooled_features)``.

        Example:
            >>> net = ResnetPointnet(c_dim=128, dim=3)
            >>> code = net(torch.randn(4, 512, 3))  # [4, 128]
        """
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


class ResnetPointnetV2(nn.Module):
    """PointNet-based encoder with BatchNorm ResNet blocks and Sigmoid output.

    Similar to ``ResnetPointnet`` but uses ``ResnetBlockFC_BN`` for batch-
    normalized residual blocks and ``nn.Sigmoid`` as the final activation.

    Args:
        c_dim (int): Output latent code dimension. Defaults to 128.
        dim (int): Per-point input dimension. Defaults to 3.
        hidden_dim (int): Hidden dimension of ResNet blocks. Defaults to 512.

    Example:
        >>> net = ResnetPointnetV2(c_dim=128, dim=3, hidden_dim=256)
        >>> code = net(torch.randn(4, 1024, 3))  # [4, 128]
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        """Initialize BN-augmented position projection, ResNet blocks, and output FC.

        Args:
            c_dim (int): Output latent code dimension.
            dim (int): Per-point input dimension.
            hidden_dim (int): Hidden dimension of ResNet blocks.

        Example:
            >>> net = ResnetPointnetV2(c_dim=128, dim=6, hidden_dim=256)
        """
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.bn_0 = BNanyshape(2 * hidden_dim)
        self.block_0 = ResnetBlockFC_BN(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC_BN(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC_BN(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC_BN(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC_BN(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.Sigmoid()
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        """Encode a point cloud to a latent code (with BatchNorm).

        Args:
            p (torch.Tensor): Point cloud of shape ``[B, T, D]``.
            return_unpooled (bool): If True, also return pre-pooled features.

        Returns:
            torch.Tensor: Latent code ``[B, c_dim]``. If ``return_unpooled``
                is True, returns ``(code, unpooled_features)``.

        Example:
            >>> net = ResnetPointnetV2(c_dim=128, dim=3)
            >>> code = net(torch.randn(4, 512, 3))  # [4, 128]
        """
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.bn_0(net)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


class ResnetPointnetV3(nn.Module):
    """V3: per-entity median-radial-length normalization + scale-as-feature.

    Pre-normalizes valid xyz coordinates so that the median radial length is
    1.0 per entity, then runs a V2-style backbone on the normalized cloud, then
    concatenates ``log(scale)`` as +1 channel to the pooled feature.

    External output dim is exactly ``c_dim`` (backbone runs at ``c_dim - 1``).
    Color channels (if ``dim == 6``) are untouched by the normalization.

    Numerical safety:
      - Degenerate entities (``M <= 1`` OR median radial length ``< EPS_DEGEN``)
        get ``scale = 1.0`` (no rescale).
      - ``log_scale`` uses ``scale.clamp(min=EPS_LOG)`` to prevent very negative
        values.

    Args:
        c_dim (int): Output latent code dimension. Defaults to 128.
        dim (int): Per-point input dimension (3 for xyz, 6 for xyz+rgb).
            Defaults to 3.
        hidden_dim (int): Hidden dimension of ResNet blocks. Defaults to 512.

    Example:
        >>> net = ResnetPointnetV3(c_dim=128, dim=6, hidden_dim=256)
        >>> code = net(torch.randn(4, 1024, 6))  # [4, 128]
    """

    EPS_NORM = 1e-12   # for sqrt
    EPS_DEGEN = 1e-6   # below this, treat as degenerate
    EPS_LOG = 1e-4     # clamp before log

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        """Initialize the V2-style backbone at reduced width (``c_dim - 1``).

        Args:
            c_dim (int): External output dimension (>=2). The internal backbone
                is constructed with ``c_dim - 1`` so the concatenated output has
                exactly ``c_dim`` channels.
            dim (int): Per-point input dimension (3 or 6).
            hidden_dim (int): Hidden dimension of ResNet blocks.
        """
        super().__init__()
        assert c_dim >= 2, f"c_dim must be >= 2 for PointNetV3 (got {c_dim})"
        # Internal backbone uses c_dim - 1 so external output is exactly c_dim
        # after log_scale is concatenated.
        self._internal_c_dim = c_dim - 1
        self._external_c_dim = c_dim
        self.c_dim = c_dim
        self.dim = dim
        self.backbone = ResnetPointnetV2(
            c_dim=self._internal_c_dim,
            dim=dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, p, valid_mask=None):
        """Encode a point cloud with per-entity scale normalization.

        Args:
            p (torch.Tensor): Input of shape ``(K, M, dim)``. ``dim`` is 3
                (xyz) or 6 (xyz+rgb). ``K`` is the number of entities, ``M`` is
                points per entity.
            valid_mask (torch.Tensor, optional): Per-point valid mask of
                shape ``(K, M)`` (bool). If provided, the median radial length
                is computed only over valid points per entity, and entities
                with fewer than 2 valid points are treated as degenerate
                (scale = 1.0). When ``None``, all M points are treated as
                valid (current call sites already pass entity-filtered tensors).

        Returns:
            torch.Tensor: Per-entity feature of shape ``(K, c_dim)``, with
                ``log(scale)`` concatenated as the last channel.
        """
        K, M, D = p.shape
        assert D == self.dim, f"input dim {D} != expected {self.dim}"

        xyz = p[..., :3]  # (K, M, 3)

        # Per-point radial length (numerical safety: add EPS to sq norm before sqrt)
        sq_norm = xyz.pow(2).sum(dim=-1)  # (K, M)
        length = (sq_norm + self.EPS_NORM).sqrt()  # (K, M)

        # Per-entity median radial length.
        if valid_mask is None:
            # No mask: use all M points (current codebase behavior).
            scale = length.median(dim=-1).values  # (K,)
            n_valid = torch.full(
                (K,), M, device=length.device, dtype=torch.long
            )
        else:
            # Per-point mask: compute masked median per entity.
            # We replace invalid entries with +inf so they sort to the end,
            # then take median over the first n_valid sorted values per entity.
            # For numerical safety we just iterate per entity (K is small).
            assert valid_mask.shape == (K, M), (
                f"valid_mask shape {tuple(valid_mask.shape)} != ({K},{M})"
            )
            n_valid = valid_mask.sum(dim=-1)  # (K,) long
            scale = torch.ones(K, device=length.device, dtype=length.dtype)
            for k in range(K):
                nv = int(n_valid[k].item())
                if nv >= 1:
                    scale[k] = length[k][valid_mask[k]].median()

        # Degenerate handling: if M (or n_valid) <= 1 or scale < EPS_DEGEN,
        # set scale = 1.0. The single-point case is numerically dangerous
        # (median of one value, then dividing every coord by that one number
        # collapses geometry); the user explicitly asked us to skip rescaling.
        degenerate = (scale < self.EPS_DEGEN) | (n_valid <= 1)
        scale = torch.where(degenerate, torch.ones_like(scale), scale)

        # Normalize xyz (color untouched)
        xyz_norm = xyz / scale[:, None, None]  # (K, M, 3)
        if D > 3:
            p_norm = torch.cat([xyz_norm, p[..., 3:]], dim=-1)
        else:
            p_norm = xyz_norm

        # Backbone forward -> (K, c_dim - 1)
        pooled = self.backbone(p_norm)

        # log scale, clamped to prevent very negative values
        log_scale = torch.log(scale.clamp(min=self.EPS_LOG))  # (K,)

        # Concatenate as +1 channel -> (K, c_dim)
        out = torch.cat([pooled, log_scale[:, None]], dim=-1)
        return out


class AttentionPooling(nn.Module):
    """
    Attention Pooling module.

    Args:
        dim (int): Input dimension.
    """

    def __init__(self, dim) -> None:
        """Initialize attention key network.

        Args:
            dim (int): Input feature dimension.

        Example:
            >>> pool = AttentionPooling(128)
        """

        super().__init__()
        self.dim = dim
        self.K = ResnetBlockFC(dim, dim, dim)

    def forward(self, x, keepdim=True):
        """Compute attention-weighted pooling over the point dimension.

        Args:
            x (torch.Tensor): Input features of shape ``[B, N, C]``.
            keepdim (bool): If True, output has shape ``[B, 1, C]``;
                otherwise ``[B, C]``. Defaults to True.

        Returns:
            torch.Tensor: Pooled features.

        Example:
            >>> pool = AttentionPooling(128)
            >>> out = pool(torch.randn(4, 256, 128))  # [4, 1, 128]
        """
        # B,N,C, pool over dim 1
        assert x.ndim == 3 and x.shape[-1] == self.dim
        k = self.K(x.max(dim=1, keepdim=True).values)  # B,1,C
        alpha = (k * x).sum(dim=-1, keepdim=True)
        weight = torch.softmax(alpha, dim=1)
        pooled = (weight * x).sum(dim=1, keepdim=keepdim)
        return pooled


class ResnetPointnetAttenPooling(nn.Module):
    """PointNet-based encoder with learned attention pooling.

    Replaces the max-pool aggregation in ``ResnetPointnet`` with learned
    ``AttentionPooling`` modules for richer global feature aggregation.

    Args:
        c_dim (int): Output latent code dimension. Defaults to 128.
        dim (int): Per-point input dimension. Defaults to 3.
        hidden_dim (int): Hidden dimension of ResNet blocks. Defaults to 512.

    Example:
        >>> net = ResnetPointnetAttenPooling(c_dim=128, dim=3, hidden_dim=256)
        >>> code = net(torch.randn(4, 1024, 3))  # [4, 128]
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        """Initialize position projection, ResNet blocks, attention pools, and output FC.

        Args:
            c_dim (int): Output latent code dimension.
            dim (int): Per-point input dimension.
            hidden_dim (int): Hidden dimension of ResNet blocks.

        Example:
            >>> net = ResnetPointnetAttenPooling(c_dim=128, dim=3, hidden_dim=256)
        """
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_0 = AttentionPooling(hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_1 = AttentionPooling(hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_2 = AttentionPooling(hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_3 = AttentionPooling(hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_4 = AttentionPooling(hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        """Encode a point cloud to a latent code using attention pooling.

        Args:
            p (torch.Tensor): Point cloud of shape ``[B, T, D]``.
            return_unpooled (bool): If True, also return pre-pooled features.

        Returns:
            torch.Tensor: Latent code ``[B, c_dim]``. If ``return_unpooled``
                is True, returns ``(code, unpooled_features)``.

        Example:
            >>> net = ResnetPointnetAttenPooling(c_dim=128, dim=3)
            >>> code = net(torch.randn(4, 512, 3))  # [4, 128]
        """
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.atten_pool_0(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.atten_pool_1(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.atten_pool_2(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.atten_pool_3(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.atten_pool_4(net, keepdim=False)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


if __name__ == "__main__":
    device = torch.device("cuda")
    B, N = 5, 1024
    net = ResnetPointnetAttenPooling(c_dim=128, dim=3, hidden_dim=256).to(device)
    x = torch.rand(B, N, 3).to(device)
    y = net(x)
    print(x.shape)
    print(y.shape)
