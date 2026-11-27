# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR Transformer module for the ACT architecture.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

Provides the full Transformer (encoder + decoder), as well as individual
encoder/decoder layer and stack classes used in both the CVAE encoder
and the action decoder.
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython

e = IPython.embed


class Transformer(nn.Module):
    """Full Transformer model with encoder and decoder stacks.

    Modified from the standard PyTorch Transformer to accept external
    positional encodings and to support additional latent/proprioceptive
    input tokens prepended to the encoder input.

    Args:
        d_model: The hidden dimension of the model.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
        activation: Activation function ('relu' or 'gelu').
        normalize_before: If True, use pre-norm (LayerNorm before attention).
        return_intermediate_dec: If True, return intermediate decoder outputs.

    Example:
        >>> transformer = Transformer(d_model=256, nhead=8,
        ...     num_encoder_layers=4, num_decoder_layers=4)
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        """Initialize Transformer with encoder and decoder stacks.

        Args:
            d_model: The hidden dimension of the model.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
            activation: Activation function name ('relu' or 'gelu').
            normalize_before: If True, apply LayerNorm before attention.
            return_intermediate_dec: If True, return all intermediate
                decoder layer outputs stacked.
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Initialize all parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
    ):
        """Forward pass through the full Transformer.

        Handles two input modes:
        1. State-only (src is None): Uses latent_input and proprio_input
           as the only encoder tokens.
        2. Visual (src has shape (N, C, H, W)): Flattens spatial dims and
           prepends latent/proprio tokens.

        Args:
            src: Source features from the backbone, shape (N, C, H, W), or
                None for state-only mode.
            mask: Source key padding mask, or None.
            query_embed: Decoder query embeddings, shape
                (num_queries, hidden_dim).
            pos_embed: Positional encoding for src, shape (N, C, H, W), or
                None for state-only mode.
            latent_input: Latent vector from the CVAE encoder, shape
                (batch, hidden_dim).
            proprio_input: Proprioceptive state embedding, shape
                (batch, hidden_dim).
            additional_pos_embed: Positional embeddings for latent and
                proprio tokens, shape (2, hidden_dim).

        Returns:
            Decoder output tensor of shape
            (num_decoder_layers, batch, num_queries, hidden_dim) if
            return_intermediate_dec is True, otherwise
            (1, batch, num_queries, hidden_dim).
        """
        if src is None:
            bs = proprio_input.shape[0]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            src = torch.stack([latent_input, proprio_input], axis=0)
        # TODO flatten only when input has H and W
        elif len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # HWxNxC
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers.

    Args:
        encoder_layer: An instance of TransformerEncoderLayer to clone.
        num_layers: Number of encoder layers.
        norm: Optional final LayerNorm to apply after all layers.

    Example:
        >>> layer = TransformerEncoderLayer(d_model=256, nhead=8)
        >>> encoder = TransformerEncoder(layer, num_layers=4)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        """Initialize TransformerEncoder by cloning the given layer.

        Args:
            encoder_layer: A single TransformerEncoderLayer to be cloned.
            num_layers: Number of encoder layers in the stack.
            norm: Optional LayerNorm applied after all layers.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass through all encoder layers sequentially.

        Args:
            src: Source sequence tensor of shape (seq_len, batch, d_model).
            mask: Attention mask of shape (seq_len, seq_len), or None.
            src_key_padding_mask: Key padding mask of shape
                (batch, seq_len), or None.
            pos: Positional encoding tensor of shape
                (seq_len, 1_or_batch, d_model), or None.

        Returns:
            Encoded output tensor of shape (seq_len, batch, d_model).
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers with optional intermediate outputs.

    Args:
        decoder_layer: An instance of TransformerDecoderLayer to clone.
        num_layers: Number of decoder layers.
        norm: Optional final LayerNorm to apply after all layers.
        return_intermediate: If True, return outputs from all decoder layers
            stacked along a new dimension.

    Example:
        >>> layer = TransformerDecoderLayer(d_model=256, nhead=8)
        >>> decoder = TransformerDecoder(layer, num_layers=4,
        ...     return_intermediate=True)
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """Initialize TransformerDecoder by cloning the given layer.

        Args:
            decoder_layer: A single TransformerDecoderLayer to be cloned.
            num_layers: Number of decoder layers in the stack.
            norm: Optional LayerNorm applied after all layers.
            return_intermediate: If True, return a stack of all intermediate
                layer outputs.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass through all decoder layers sequentially.

        Args:
            tgt: Target sequence tensor of shape
                (num_queries, batch, d_model).
            memory: Encoder output tensor of shape
                (src_len, batch, d_model).
            tgt_mask: Target attention mask, or None.
            memory_mask: Memory attention mask, or None.
            tgt_key_padding_mask: Target key padding mask, or None.
            memory_key_padding_mask: Memory key padding mask, or None.
            pos: Positional encoding for memory, or None.
            query_pos: Positional encoding for queries, or None.

        Returns:
            If return_intermediate is True, a tensor of shape
            (num_layers, num_queries, batch, d_model) with all intermediate
            outputs. Otherwise, shape (1, num_queries, batch, d_model).
        """
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with self-attention and feedforward.

    Supports both pre-norm and post-norm configurations.

    Args:
        d_model: The hidden dimension of the model.
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
        activation: Activation function name ('relu' or 'gelu').
        normalize_before: If True, apply LayerNorm before attention (pre-norm).

    Example:
        >>> layer = TransformerEncoderLayer(d_model=256, nhead=8)
        >>> src = torch.randn(10, 2, 256)
        >>> out = layer(src)
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        """Initialize TransformerEncoderLayer.

        Args:
            d_model: The hidden dimension of the model.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
            activation: Activation function name ('relu' or 'gelu').
            normalize_before: If True, apply pre-norm ordering.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional encoding to the tensor if provided.

        Args:
            tensor: Input tensor.
            pos: Positional encoding tensor, or None.

        Returns:
            tensor + pos if pos is not None, else tensor unchanged.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Post-norm forward pass (attention then LayerNorm).

        Args:
            src: Source sequence, shape (seq_len, batch, d_model).
            src_mask: Attention mask, or None.
            src_key_padding_mask: Key padding mask, or None.
            pos: Positional encoding, or None.

        Returns:
            Output tensor of shape (seq_len, batch, d_model).
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Pre-norm forward pass (LayerNorm then attention).

        Args:
            src: Source sequence, shape (seq_len, batch, d_model).
            src_mask: Attention mask, or None.
            src_key_padding_mask: Key padding mask, or None.
            pos: Positional encoding, or None.

        Returns:
            Output tensor of shape (seq_len, batch, d_model).
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass dispatching to pre-norm or post-norm.

        Args:
            src: Source sequence, shape (seq_len, batch, d_model).
            src_mask: Attention mask, or None.
            src_key_padding_mask: Key padding mask, or None.
            pos: Positional encoding, or None.

        Returns:
            Output tensor of shape (seq_len, batch, d_model).
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with self-attention, cross-attention, and FFN.

    Supports both pre-norm and post-norm configurations.

    Args:
        d_model: The hidden dimension of the model.
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
        activation: Activation function name ('relu' or 'gelu').
        normalize_before: If True, apply LayerNorm before attention (pre-norm).

    Example:
        >>> layer = TransformerDecoderLayer(d_model=256, nhead=8)
        >>> tgt = torch.randn(100, 2, 256)
        >>> memory = torch.randn(50, 2, 256)
        >>> out = layer(tgt, memory)
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        """Initialize TransformerDecoderLayer.

        Args:
            d_model: The hidden dimension of the model.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
            activation: Activation function name ('relu' or 'gelu').
            normalize_before: If True, apply pre-norm ordering.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional encoding to the tensor if provided.

        Args:
            tensor: Input tensor.
            pos: Positional encoding tensor, or None.

        Returns:
            tensor + pos if pos is not None, else tensor unchanged.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Post-norm forward pass for the decoder layer.

        Args:
            tgt: Target sequence, shape (tgt_len, batch, d_model).
            memory: Encoder output, shape (src_len, batch, d_model).
            tgt_mask: Target self-attention mask, or None.
            memory_mask: Cross-attention mask, or None.
            tgt_key_padding_mask: Target key padding mask, or None.
            memory_key_padding_mask: Memory key padding mask, or None.
            pos: Positional encoding for memory, or None.
            query_pos: Positional encoding for queries, or None.

        Returns:
            Output tensor of shape (tgt_len, batch, d_model).
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Pre-norm forward pass for the decoder layer.

        Args:
            tgt: Target sequence, shape (tgt_len, batch, d_model).
            memory: Encoder output, shape (src_len, batch, d_model).
            tgt_mask: Target self-attention mask, or None.
            memory_mask: Cross-attention mask, or None.
            tgt_key_padding_mask: Target key padding mask, or None.
            memory_key_padding_mask: Memory key padding mask, or None.
            pos: Positional encoding for memory, or None.
            query_pos: Positional encoding for queries, or None.

        Returns:
            Output tensor of shape (tgt_len, batch, d_model).
        """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass dispatching to pre-norm or post-norm.

        Args:
            tgt: Target sequence, shape (tgt_len, batch, d_model).
            memory: Encoder output, shape (src_len, batch, d_model).
            tgt_mask: Target self-attention mask, or None.
            memory_mask: Cross-attention mask, or None.
            tgt_key_padding_mask: Target key padding mask, or None.
            memory_key_padding_mask: Memory key padding mask, or None.
            pos: Positional encoding for memory, or None.
            query_pos: Positional encoding for queries, or None.

        Returns:
            Output tensor of shape (tgt_len, batch, d_model).
        """
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    """Create N deep copies of a module.

    Args:
        module: The module to clone.
        N: Number of clones to create.

    Returns:
        nn.ModuleList containing N independent copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    dim_feedforward=512,
    enc_layers=2,
    dec_layers=4,
    pre_norm=False,
):
    """Build a Transformer model with specified hyperparameters.

    Convenience factory function that constructs a Transformer with
    return_intermediate_dec=True for use in the ACT architecture.

    Args:
        hidden_dim: Model hidden dimension (d_model).
        dropout: Dropout rate.
        nheads: Number of attention heads.
        dim_feedforward: Feedforward network hidden dimension.
        enc_layers: Number of encoder layers.
        dec_layers: Number of decoder layers.
        pre_norm: If True, use pre-norm (LayerNorm before attention).

    Returns:
        A Transformer module with intermediate decoder outputs enabled.

    Example:
        >>> transformer = build_transformer(hidden_dim=256, nheads=8)
    """
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string.

    Args:
        activation: Name of the activation function ('relu', 'gelu', or 'glu').

    Returns:
        The corresponding PyTorch activation function.

    Raises:
        RuntimeError: If the activation name is not supported.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
