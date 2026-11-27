"""ACT (Action Chunking with Transformers) agent module for visual imitation learning.

This module implements the ActAgent, a CVAE-based policy that combines CNN
backbones for visual feature extraction with a transformer-based CVAE for
action chunk prediction.

Example:
    >>> agent = ActAgent(include_depth=False, state_dim=25, act_dim=4)
    >>> loss_dict = agent.compute_loss(obs, action_seq)
"""

import argparse
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from act.evaluate import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
import sys


from umm.models.act.act.utils import IterationBasedBatchSampler, worker_init_fn
from umm.models.act.act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from umm.models.act.act.detr.backbone import build_backbone
from umm.models.act.act.detr.transformer import build_transformer
from umm.models.act.act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from tqdm import tqdm


class ActAgent(nn.Module):
    """ACT agent that uses a CVAE with CNN backbone and transformer for action prediction.

    Combines a ResNet CNN backbone for visual feature extraction with a DETR-style
    CVAE (encoder + decoder) for predicting action chunks from observations.

    Args:
        include_depth: Whether to include depth images as input.
        state_dim: Dimension of the robot state observation.
        act_dim: Dimension of the action space.
        kl_weight: Weight for the KL divergence loss term.
        predict_horizon: Number of future action steps to predict (action chunk size).
        loss_type: Type of reconstruction loss, either "l1" or "l2".
        state_maskout: List of state indices to mask out (zero) during forward pass.

    Example:
        >>> agent = ActAgent(state_dim=25, act_dim=4, predict_horizon=30)
        >>> obs = {"state": torch.randn(2, 25), "rgb": torch.randint(0, 255, (2, 1, 3, 224, 224))}
        >>> action_seq = torch.randn(2, 30, 4)
        >>> loss_dict = agent.compute_loss(obs, action_seq)
    """

    def __init__(
        self,
        include_depth=False,
        state_dim=25,
        act_dim=4,
        kl_weight=10.0,
        predict_horizon=30,
        loss_type="l1",
        state_maskout=[],
    ):
        """Initialize the ActAgent.

        Args:
            include_depth: Whether to include depth images as input.
            state_dim: Dimension of the robot state observation.
            act_dim: Dimension of the action space.
            kl_weight: Weight for the KL divergence loss term.
            predict_horizon: Number of future action steps to predict.
            loss_type: Type of reconstruction loss, "l1" or "l2".
            state_maskout: List of state indices to mask out during forward pass.

        Example:
            >>> agent = ActAgent(state_dim=25, act_dim=4, kl_weight=10.0)
        """
        super().__init__()
        self.include_depth = include_depth
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.kl_weight = kl_weight
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.predict_horizon = predict_horizon

        self.loss_type = loss_type

        # CNN backbone
        backbones = []
        backbone = build_backbone(
            hidden_dim=256,
            position_embedding="sine",
            lr_backbone=1.0e-05,
            masks_flag=False,
            backbone_type="resnet18",
            include_depth=include_depth,
            dilation=False,
        )
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=512,
            enc_layers=2,
            dec_layers=4,
            pre_norm=False,
        )

        # CVAE encoder
        encoder = build_encoder(
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=512,
            num_encoder_layers=4,
            normalize_before=False,
            activation="relu",
        )

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=self.predict_horizon,
            include_depth=self.include_depth,
            state_maskout=state_maskout,
        )
        return

    def compute_loss(self, obs, action_seq):
        """Compute training loss for the ACT agent.

        Runs a forward pass through the CVAE, computing reconstruction loss
        (L1 or L2) and KL divergence loss.

        Args:
            obs: Dictionary with keys "rgb" (B, V, C, H, W uint8), optionally
                "depth" (B, V, 1, H, W), and "state" (B, state_dim).
            action_seq: Ground truth action sequence (B, predict_horizon, act_dim).

        Returns:
            Dictionary with keys "l_recon", "kl", "loss", "close_acc", "delta_pos_norm".

        Example:
            >>> loss_dict = agent.compute_loss(obs, action_seq)
            >>> loss_dict["loss"].backward()
        """
        # normalize rgb data
        obs["rgb"] = obs["rgb"].float() / 255.0
        obs["rgb"] = self.normalize(obs["rgb"])

        # depth data
        if self.include_depth:
            obs["depth"] = obs["depth"].float()

        # forward pass
        a_hat, (mu, logvar) = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        if self.loss_type == "l1":
            all_l = F.l1_loss(action_seq, a_hat, reduction="none")
        elif self.loss_type == "l2":
            all_l = F.mse_loss(action_seq, a_hat, reduction="none")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        main_loss = all_l.mean()

        # store all loss
        loss_dict = dict()
        loss_dict["l_recon"] = main_loss
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l_recon"] + loss_dict["kl"] * self.kl_weight
        # metrics

        # ! debug
        gt_close = action_seq[..., -1] > 0.0
        pred_close = a_hat[..., -1] > 0.0
        close_acc = (gt_close == pred_close).float().mean()
        loss_dict["close_acc"] = close_acc

        delta_pos = action_seq[..., :3] - a_hat[..., :3]
        delta_pos_norm = delta_pos.norm(dim=-1)
        loss_dict["delta_pos_norm"] = delta_pos_norm.mean()

        return loss_dict

    def get_action(self, obs):
        """Get predicted actions from the agent (inference mode, samples from prior).

        Args:
            obs: Dictionary with keys "rgb" (B, V, C, H, W uint8), optionally
                "depth", and "state" (B, state_dim).

        Returns:
            Predicted action sequence tensor of shape (B, predict_horizon, act_dim).

        Example:
            >>> with torch.no_grad():
            ...     actions = agent.get_action(obs)
        """
        # normalize rgb data
        obs["rgb"] = obs["rgb"].float() / 255.0
        obs["rgb"] = self.normalize(obs["rgb"])

        # depth data
        if self.include_depth:
            obs["depth"] = obs["depth"].float()

        # forward pass
        a_hat, (_, _) = self.model(obs)  # no action, sample from prior

        return a_hat

    def get_optim_groups(self, lr_backbone=1.0e-5):
        """Get optimizer parameter groups with separate learning rate for backbone.

        Args:
            lr_backbone: Learning rate for backbone parameters (typically lower
                than the main learning rate for fine-tuning).

        Returns:
            List of two parameter group dicts: one for non-backbone params
            (using the optimizer's default lr) and one for backbone params
            (using lr_backbone).

        Example:
            >>> optim_groups = agent.get_optim_groups(lr_backbone=1e-5)
            >>> optimizer = torch.optim.AdamW(optim_groups, lr=1e-4)
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]


def kl_divergence(mu, logvar):
    """Compute KL divergence between a Gaussian and the standard normal.

    Computes KL(q(z|x) || p(z)) where q is parameterized by mu and logvar,
    and p is N(0, I).

    Args:
        mu: Mean of the approximate posterior, shape (B, latent_dim).
        logvar: Log-variance of the approximate posterior, shape (B, latent_dim).

    Returns:
        Tuple of (total_kld, dimension_wise_kld, mean_kld):
            - total_kld: Sum over dimensions, mean over batch (1,).
            - dimension_wise_kld: Mean over batch per dimension (latent_dim,).
            - mean_kld: Mean over both dimensions and batch (1,).

    Example:
        >>> mu = torch.randn(8, 32)
        >>> logvar = torch.randn(8, 32)
        >>> total_kld, dim_kld, mean_kld = kl_divergence(mu, logvar)
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
