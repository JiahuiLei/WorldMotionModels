# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR-VAE model for Action Chunking Transformer (ACT).

Implements the Conditional Variational Autoencoder (CVAE) based on the DETR
architecture for action sequence prediction. Combines visual features from
camera backbones with proprioceptive state to predict action chunks via a
latent-conditioned transformer decoder.
"""

import torch
from torch import nn
from torch.autograd import Variable
from act.detr.transformer import (
    build_transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

import numpy as np

import IPython

e = IPython.embed


def reparametrize(mu, logvar):
    """Apply the reparameterization trick for VAE sampling.

    Samples from N(mu, std) using the reparameterization trick:
    z = mu + std * eps, where eps ~ N(0, 1).

    Args:
        mu: Mean of the latent distribution, shape (batch, latent_dim).
        logvar: Log-variance of the latent distribution, shape (batch, latent_dim).

    Returns:
        Sampled latent vector of shape (batch, latent_dim).

    Example:
        >>> mu = torch.zeros(4, 32)
        >>> logvar = torch.zeros(4, 32)
        >>> z = reparametrize(mu, logvar)
        >>> z.shape
        torch.Size([4, 32])
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """Generate a sinusoidal positional encoding table.

    Creates fixed sinusoidal position embeddings as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        n_position: Number of positions to encode.
        d_hid: Dimensionality of the positional encoding.

    Returns:
        FloatTensor of shape (1, n_position, d_hid) containing the
        sinusoidal encoding table.

    Example:
        >>> table = get_sinusoid_encoding_table(100, 256)
        >>> table.shape
        torch.Size([1, 100, 256])
    """

    def get_position_angle_vec(position):
        """Compute the angle vector for a single position.

        Args:
            position: The position index.

        Returns:
            List of angle values for each hidden dimension.
        """
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """DETR-based Conditional Variational Autoencoder for action prediction.

    Combines a CVAE encoder (for training) with a transformer-based decoder
    to predict action sequences conditioned on visual observations and
    proprioceptive state. During training, the encoder maps state-action
    pairs to a latent distribution. During inference, the latent is sampled
    from a standard normal prior.

    Args:
        backbones: List of backbone modules for visual feature extraction,
            or None for state-only mode.
        transformer: Transformer decoder module.
        encoder: Transformer encoder module for the CVAE encoder.
        state_dim: Dimensionality of the robot state input.
        action_dim: Dimensionality of each action.
        num_queries: Number of action queries (i.e., action chunk length).
        include_depth: Whether the visual input includes depth channels.
        state_maskout: List of state indices to mask out (zero) during inference.

    Example:
        >>> model = DETRVAE(backbones=None, transformer=transformer,
        ...                 encoder=encoder, state_dim=14,
        ...                 action_dim=7, num_queries=100)
    """

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        action_dim,
        num_queries,
        include_depth=True,
        state_maskout=[],
    ):
        """Initialize DETRVAE with encoder, decoder, and projection layers.

        Args:
            backbones: List of backbone modules for visual feature extraction,
                or None for state-only mode.
            transformer: Transformer decoder module for action prediction.
            encoder: Transformer encoder module for the CVAE latent encoding.
            state_dim: Dimensionality of the robot state input.
            action_dim: Dimensionality of each action.
            num_queries: Number of action queries (action chunk length).
            include_depth: Whether the visual input includes depth channels.
            state_maskout: List of state dimension indices to zero out.
        """
        super().__init__()

        self.state_maskout = state_maskout
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.include_depth = include_depth
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_state_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project state to embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, hidden_dim
        )  # project action to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )  # [CLS], state, actions

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for state and proprio

    def forward(self, obs, actions=None):
        """Forward pass through the CVAE encoder (if training) and decoder.

        During training (actions provided), encodes the state-action pair into
        a latent distribution and samples from it. During inference, samples
        from a zero-mean unit-variance prior.

        Args:
            obs: Observation dictionary with keys 'state', 'rgb', and
                optionally 'depth' when backbones are used. When backbones
                are None, obs is the raw state tensor.
            actions: Ground-truth action sequences of shape
                (batch, num_queries, action_dim). None during inference.

        Returns:
            Tuple of (a_hat, [mu, logvar]) where:
                - a_hat: Predicted action sequence of shape
                    (batch, num_queries, action_dim).
                - mu: Latent mean of shape (batch, latent_dim), or None
                    during inference.
                - logvar: Latent log-variance of shape (batch, latent_dim),
                    or None during inference.
        """
        is_training = actions is not None
        state = obs["state"] if self.backbones is not None else obs
        if self.backbones:
            for i in self.state_maskout:
                state[:, i] = state[:, i] * 0.0
        bs = state.shape[0]

        if is_training:
            # project CLS token, state sequence, and action sequence to embedding dim
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            state_embed = self.encoder_state_proj(state)  # (bs, hidden_dim)
            state_embed = torch.unsqueeze(state_embed, axis=1)  # (bs, 1, hidden_dim)
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            # concat them together to form an input to the CVAE encoder
            encoder_input = torch.cat(
                [cls_embed, state_embed, action_embed], axis=1
            )  # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, bs, hidden_dim)
            # no masking is applied to all parts of the CVAE encoder input
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(
                state.device
            )  # False: not a padding
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # query CVAE encoder
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                state.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        # CVAE decoder
        if self.backbones is not None:
            vis_data = obs["rgb"]
            if self.include_depth:
                assert "depth" in obs
                vis_data = torch.cat([vis_data, obs["depth"]], dim=2)
            num_cams = vis_data.shape[1]

            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(num_cams):
                features, pos = self.backbones[0](vis_data[:, cam_id])  # HARDCODED
                features = features[
                    0
                ]  # take the last layer feature # (batch, hidden_dim, H, W)
                pos = pos[0]  # (1, hidden_dim, H, W)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # proprioception features (state)
            proprio_input = self.input_proj_robot_state(state)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)  # (batch, hidden_dim, 4, 8)
            pos = torch.cat(all_cam_pos, axis=3)  # (batch, hidden_dim, 4, 8)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,  # num_queries, hidden_dim
                pos,
                # two additional inputs as tokens
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,  # the pe for latent and proprio
            )[
                0
            ]  # (batch, num_queries, hidden_dim)
        else:
            state = self.input_proj_robot_state(state)
            hs = self.transformer(
                None,
                None,
                self.query_embed.weight,
                None,
                latent_input,
                state,
                self.additional_pos_embed.weight,
            )[0]

        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]


def build_encoder(
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    dim_feedforward=512,
    num_encoder_layers=4,
    normalize_before=False,
    activation="relu",
):
    """Build a transformer encoder for the CVAE encoder.

    Constructs a TransformerEncoder with the specified hyperparameters
    for use as the CVAE encoder in the DETRVAE model.

    Args:
        hidden_dim: Model hidden dimension (d_model).
        dropout: Dropout rate.
        nheads: Number of attention heads.
        dim_feedforward: Feedforward network hidden dimension.
        num_encoder_layers: Number of encoder layers.
        normalize_before: If True, apply layer norm before attention (pre-norm).
        activation: Activation function name ('relu' or 'gelu').

    Returns:
        A TransformerEncoder module.

    Example:
        >>> encoder = build_encoder(hidden_dim=256, nheads=8)
    """
    d_model = hidden_dim  # 256
    dropout = dropout  # 0.1
    nhead = nheads  # 8
    dim_feedforward = dim_feedforward  # 2048
    num_encoder_layers = num_encoder_layers  # 4 # TODO shared with VAE decoder
    normalize_before = normalize_before  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder
