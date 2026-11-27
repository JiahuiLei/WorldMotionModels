"""Shared utility modules and functions for world models.

Contains lightweight building blocks used across the MotionMachine variants:

- ``disable_model_grad``: Freeze all parameters of a model.
- ``build_attention_mask``: Construct boolean attention masks (full/causal)
  with optional left/right padding.
- ``LayerNorm``: LayerNorm with optional bias.
- ``HardLayerNorm``: Parameter-free LayerNorm (normalize only).
- ``SimpleFC``: Multi-layer FC block with HardLayerNorm and dropout.
- ``NerfStylePE``: NeRF-style sinusoidal positional encoding for continuous
  spatial coordinates.

Example:
    >>> mask = build_attention_mask(torch.ones(2, 10).bool(), mode="full")
    >>> pe = NerfStylePE(in_channels=3, num_frequencies=10)
    >>> encoded = pe(torch.randn(4, 3))  # [4, 63]
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def disable_model_grad(model):
    """Freeze all parameters of a model by setting ``requires_grad=False``.

    Args:
        model (nn.Module): The model to freeze.

    Returns:
        nn.Module: The same model with all gradients disabled.

    Example:
        >>> frozen = disable_model_grad(nn.Linear(10, 10))
        >>> assert all(not p.requires_grad for p in frozen.parameters())
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def build_attention_mask(mask, mode="full", block=1, left_pad=0, right_pad=0):
    """Construct a boolean self-attention mask from a 2-D validity mask.

    Supports ``"full"`` (bidirectional) and ``"causal"`` (lower-triangular)
    modes. Optionally pads the sequence on left/right with always-valid
    positions (e.g., for register tokens).

    Args:
        mask (torch.Tensor): Validity mask of shape ``[N, S]`` (bool or 0/1).
        mode (str): ``"full"`` or ``"causal"``. Defaults to ``"full"``.
        block (int): Block expansion factor for causal mask. Defaults to 1.
        left_pad (int): Number of always-valid tokens prepended. Defaults to 0.
        right_pad (int): Number of always-valid tokens appended. Defaults to 0.

    Returns:
        torch.Tensor: Boolean attention mask of shape ``[N, S', S']`` where
            ``S' = left_pad + S + right_pad``. Diagonal is always True.

    Example:
        >>> mask = torch.ones(2, 10).bool()
        >>> attn = build_attention_mask(mask, mode="full", left_pad=4)  # [2, 14, 14]
    """
    # mask: N,S
    # return: N,S,S
    assert mask.dim() == 2, "mask should be 2D"
    assert mode in ["full", "causal"], "mode should be 'full' or 'causal'"
    if left_pad > 0:
        mask = torch.cat([torch.ones(mask.shape[0], left_pad).to(mask), mask], dim=1)
    if right_pad > 0:
        mask = torch.cat([mask, torch.ones(mask.shape[0], right_pad).to(mask)], dim=1)
    N, S = mask.shape  # N: batch size, S: sequence length
    if mode == "full":
        ret = mask.unsqueeze(1) & mask.unsqueeze(2)  # N,S,S
    elif mode == "causal":
        ret = mask.unsqueeze(1) & mask.unsqueeze(2)  # N,S,S
        ret = ret.tril(diagonal=0)  # lower triangular mask
        # Expand ret to match block dimensions
        ret = ret[..., None, None]  # N,S,S,1,1
        ret = ret.repeat(1, 1, 1, block, block)  # N,S,S,factor,factor
        ret = ret.permute(0, 1, 3, 2, 4).reshape(
            N, S * block, S * block
        )  # N,S*factor,S*factor
    else:
        raise ValueError("Invalid mode for attention mask")
    idx = torch.arange(ret.shape[1], device=ret.device)
    ret[:, idx, idx] = True
    return ret.bool()


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias parameter.

    Standard PyTorch ``nn.LayerNorm`` does not support ``bias=False`` cleanly;
    this wrapper makes it explicit.

    Args:
        ndim (int): Normalized shape (last dimension size).
        bias (bool): Whether to include a learnable bias.

    Example:
        >>> ln = LayerNorm(128, bias=False)
        >>> out = ln(torch.randn(4, 128))  # [4, 128]
    """

    def __init__(self, ndim, bias):
        """Initialize weight (and optional bias) parameters.

        Args:
            ndim (int): Feature dimension.
            bias (bool): Whether to include a bias parameter.

        Example:
            >>> ln = LayerNorm(128, bias=True)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """Apply layer normalization.

        Args:
            input (torch.Tensor): Input of shape ``[..., ndim]``.

        Returns:
            torch.Tensor: Normalized tensor, same shape.

        Example:
            >>> ln = LayerNorm(128, bias=True)
            >>> out = ln(torch.randn(4, 128))
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class HardLayerNorm(nn.Module):
    """Parameter-free LayerNorm (normalization only, no learnable affine).

    Args:
        dim (int): Feature dimension to normalize over.

    Example:
        >>> hln = HardLayerNorm(128)
        >>> out = hln(torch.randn(4, 10, 128))  # [4, 10, 128]
    """

    def __init__(self, dim):
        """Initialize with target dimension.

        Args:
            dim (int): Feature dimension.

        Example:
            >>> hln = HardLayerNorm(128)
        """
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """Apply parameter-free layer normalization.

        Args:
            input (torch.Tensor): Input of shape ``[..., dim]``.

        Returns:
            torch.Tensor: Normalized tensor, same shape.

        Example:
            >>> hln = HardLayerNorm(128)
            >>> out = hln(torch.randn(4, 10, 128))
        """
        # just normalize, without any learnable parameters
        # # ! 2026.01.09 overwrite this, need to check whether this ruin the training
        # dtype = input.dtype
        # return F.layer_norm(input, (self.dim,), None, None, 1e-5).to(dtype)
        # ! 2026.01.09 overwrite this, need to check whether this ruin the training
        return F.layer_norm(input, (self.dim,), None, None, 1e-5)


class SimpleFC(nn.Module):
    """Multi-layer fully-connected block with HardLayerNorm and dropout.

    Stacks ``n_layers - 1`` hidden layers (Linear + HardLayerNorm + activation)
    followed by a final Linear + Dropout. All layers preserve dimension.

    Args:
        dim (int): Input and output dimension.
        dropout (float): Dropout probability after the last linear.
        n_layers (int): Total number of linear layers (including output).
        act_class: Activation class. Defaults to ``nn.LeakyReLU``.

    Example:
        >>> fc = SimpleFC(dim=128, dropout=0.1, n_layers=4)
        >>> out = fc(torch.randn(4, 128))  # [4, 128]
    """

    def __init__(self, dim, dropout, n_layers: int = 4, act_class=nn.LeakyReLU):
        """Initialize stacked FC layers.

        Args:
            dim (int): Feature dimension (preserved throughout).
            dropout (float): Dropout probability.
            n_layers (int): Total number of linear layers.
            act_class: Activation class constructor.

        Example:
            >>> fc = SimpleFC(dim=128, dropout=0.1, n_layers=3)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    HardLayerNorm(dim),
                    act_class(),
                )
            )
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        return

    def forward(self, x):
        """Apply the FC block.

        Args:
            x (torch.Tensor): Input of shape ``[..., dim]``.

        Returns:
            torch.Tensor: Output of shape ``[..., dim]``.

        Example:
            >>> fc = SimpleFC(dim=128, dropout=0.1, n_layers=4)
            >>> out = fc(torch.randn(4, 128))  # [4, 128]
        """
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class NerfStylePE(nn.Module):
    """
    NeRF-style Positional Encoding for continuous spatial coordinates.

    Unlike diffusers.Timesteps which uses log-spaced frequencies for integers,
    this uses exponentially increasing frequencies (2^0, 2^1, ..., 2^(L-1))
    to amplify local details in small-range continuous data (e.g., bbox coordinates).

    For each input dimension d and frequency level l:
        γ(x_d) = [sin(2^0 π x_d), cos(2^0 π x_d), ..., sin(2^(L-1) π x_d), cos(2^(L-1) π x_d)]

    Args:
        in_channels: Number of input dimensions
        num_frequencies: Number of frequency bands (L)
        include_input: Whether to concatenate original input to the encoding
    """

    def __init__(
        self, in_channels: int, num_frequencies: int = 10, include_input: bool = True
    ):
        """Initialize frequency bands and compute output dimension.

        Args:
            in_channels (int): Number of input dimensions.
            num_frequencies (int): Number of frequency bands L.
            include_input (bool): Concatenate raw input to encoding.

        Example:
            >>> pe = NerfStylePE(in_channels=3, num_frequencies=10)
            >>> pe.out_channels  # 63 (3 + 3*2*10)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Output dimension: each input dim gets 2*num_frequencies features (sin + cos for each frequency)
        # Plus original input if include_input=True
        self.out_channels = in_channels * 2 * num_frequencies
        if include_input:
            self.out_channels += in_channels

        # Pre-compute frequency bands: [2^0, 2^1, 2^2, ..., 2^(L-1)]
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [..., in_channels] with values typically in [0, 1] or normalized

        Returns:
            Tensor of shape [..., out_channels]
        """
        # x: [..., in_channels]
        # freq_bands: [num_frequencies]

        # Broadcast to [..., in_channels, num_frequencies]
        x_expanded = x[..., :, None] * self.freq_bands[None, :] * torch.pi

        # Compute sin and cos: [..., in_channels, num_frequencies]
        sin_features = torch.sin(x_expanded)
        cos_features = torch.cos(x_expanded)

        # Interleave sin and cos, then flatten: [..., in_channels * 2 * num_frequencies]
        encoded = torch.stack([sin_features, cos_features], dim=-1)
        encoded = encoded.flatten(start_dim=-3)

        # Optionally concatenate original input
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded
