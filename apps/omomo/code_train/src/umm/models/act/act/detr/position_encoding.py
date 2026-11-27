# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Positional encoding modules for the DETR transformer.

Provides sinusoidal and learned positional encoding implementations
generalized to work on 2D image feature maps, as used in the DETR and
ACT architectures.
"""

import math
import torch
from torch import nn

from act.utils import NestedTensor

import IPython

e = IPython.embed


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D positional encoding for image feature maps.

    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Generates fixed sinusoidal positional encodings along the height and width
    dimensions, concatenated to form a 2D positional encoding.

    Args:
        num_pos_feats: Number of positional features per spatial dimension
            (total encoding dimension = 2 * num_pos_feats).
        temperature: Temperature scaling factor for the sinusoidal frequencies.
        normalize: Whether to normalize the cumulative positions to [0, scale].
        scale: Scale factor applied after normalization. Defaults to 2*pi.

    Example:
        >>> pos_enc = PositionEmbeddingSine(128, normalize=True)
        >>> x = torch.randn(2, 256, 7, 7)
        >>> pos = pos_enc(x)
        >>> pos.shape
        torch.Size([1, 256, 7, 7])
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        """Initialize PositionEmbeddingSine.

        Args:
            num_pos_feats: Number of positional features per spatial dimension.
            temperature: Temperature scaling for sinusoidal frequencies.
            normalize: Whether to normalize positions before encoding.
            scale: Scale factor for normalized positions. Must set
                normalize=True if scale is provided.

        Raises:
            ValueError: If scale is provided but normalize is False.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        """Compute sinusoidal 2D positional encoding for the input tensor.

        Args:
            tensor: Input feature tensor of shape (N, C, H, W).

        Returns:
            Positional encoding tensor of shape (1, 2*num_pos_feats, H, W).
        """
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Learned 2D positional encoding for image feature maps.

    Absolute pos embedding, learned.

    Uses separate learnable row and column embedding tables to construct
    a 2D positional encoding. Supports spatial dimensions up to 50x50.

    Args:
        num_pos_feats: Number of positional features per spatial dimension
            (total encoding dimension = 2 * num_pos_feats).

    Example:
        >>> pos_enc = PositionEmbeddingLearned(128)
        >>> nt = NestedTensor(torch.randn(2, 256, 7, 7), mask=None)
        >>> pos = pos_enc(nt)
        >>> pos.shape
        torch.Size([2, 256, 7, 7])
    """

    def __init__(self, num_pos_feats=256):
        """Initialize PositionEmbeddingLearned with row and column embeddings.

        Args:
            num_pos_feats: Number of positional features per spatial dimension.
        """
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset embedding weights to uniform initialization."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """Compute learned 2D positional encoding for the input.

        Args:
            tensor_list: NestedTensor containing image features with a
                ``.tensors`` attribute of shape (N, C, H, W).

        Returns:
            Positional encoding tensor of shape (N, 2*num_pos_feats, H, W).
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


def build_position_encoding(hidden_dim, position_embedding):
    """Build a positional encoding module based on the specified type.

    Args:
        hidden_dim: Total hidden dimension. The number of positional features
            per dimension will be hidden_dim // 2.
        position_embedding: Type of positional encoding. One of
            'v2'/'sine' for sinusoidal or 'v3'/'learned' for learned.

    Returns:
        A positional encoding module (PositionEmbeddingSine or
        PositionEmbeddingLearned).

    Raises:
        ValueError: If position_embedding type is not supported.

    Example:
        >>> pos_enc = build_position_encoding(256, 'sine')
    """
    N_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding
