"""PyTorch Lightning model wrapper for the ACT (Action Chunking Transformer) agent.

Wraps the ActAgent in a LightningModule for training with PyTorch Lightning,
including EMA support, learning rate scheduling, and visualization.

Example:
    >>> model = PLModel(cfg=config_dict)
    >>> trainer = pl.Trainer(max_steps=100000)
    >>> trainer.fit(model, datamodule=dm)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import logging
import numpy as np
import sys, os, os.path as osp
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import time


from umm.models.act.act_agent import ActAgent


def disable_model_grad(model):
    """Disable gradient computation for all parameters in a model.

    Args:
        model: PyTorch nn.Module to freeze.

    Returns:
        The same model with all parameters set to requires_grad=False.

    Example:
        >>> frozen_model = disable_model_grad(my_model)
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


class PLModel(pl.LightningModule):
    """PyTorch Lightning module wrapping the ACT agent for training.

    Handles training step logic, EMA weight management, optimizer/scheduler
    configuration, and optional visualization callbacks.

    Args:
        cfg: Configuration dictionary with "model" and "model.optim" sub-dicts.

    Example:
        >>> cfg = {"model": {"predict_horizon": 30, "optim": {"lr": 6e-4}}}
        >>> model = PLModel(cfg=cfg)
    """

    def __init__(self, cfg=None):
        """Initialize the PLModel with config-driven agent and optional EMA.

        Args:
            cfg: Configuration dictionary containing model and optimizer settings.

        Example:
            >>> model = PLModel(cfg={"model": {"use_ema": True, "optim": {"lr": 1e-4}}})
        """
        super().__init__()
        self.cfg, self.model_cfg = cfg, cfg.get("model", None)
        self.optim_cfg = self.model_cfg.get("optim", None)

        self.use_ema = self.model_cfg.get("use_ema", False)

        # config as act baseline
        self.agent: ActAgent = ActAgent(
            include_depth=self.model_cfg.get("include_depth", False),
            predict_horizon=self.model_cfg.get("predict_horizon", 30),
            kl_weight=self.optim_cfg.get("w_kl", 10.0),
            loss_type=self.model_cfg.get("loss_type", "l1"),
            state_maskout=self.model_cfg.get("state_maskout", []),
        )

        if self.use_ema:
            from diffusers.training_utils import EMAModel

            self.ema = EMAModel(
                parameters=self.agent.parameters(),
                power=self.model_cfg.get("ema_power", 0.75),
            )
            self.ema_agent = disable_model_grad(
                ActAgent(
                    include_depth=self.model_cfg.get("include_depth", False),
                    predict_horizon=self.model_cfg.get("predict_horizon", 30),
                    kl_weight=self.optim_cfg.get("w_kl", 10.0),
                    loss_type=self.model_cfg.get("loss_type", "l1"),
                    state_maskout=self.model_cfg.get("state_maskout", []),
                )
            )

        self.save_hyperparameters()
        return

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA weights after each training step.

        Args:
            outputs: Output from the training step.
            batch: The current batch data.
            batch_idx: Index of the current batch.

        Example:
            >>> # Called automatically by Lightning after each training step.
        """
        # Call after optimizer.step(): Lightning has already stepped the optimizer here.
        # This updates EMA with the *new* weights.
        if self.use_ema:
            # Ensure EMA parameters are on the same device as the model parameters
            if len(self.ema.shadow_params) > 0:
                model_device = next(self.agent.parameters()).device
                if self.ema.shadow_params[0].device != model_device:
                    self.ema.to(model_device)
            self.ema.step(self.agent.parameters())
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        """Copy EMA weights to the shadow agent before saving checkpoint.

        Args:
            checkpoint: The checkpoint dictionary being saved.

        Example:
            >>> # Called automatically by Lightning when saving a checkpoint.
        """
        if self.use_ema:
            self.ema.copy_to(self.ema_agent.parameters())
        super().on_save_checkpoint(checkpoint)

    def training_step(self, batch, batch_idx):
        """Execute one training step.

        Args:
            batch: Dictionary with "states", "current_idx", "rgb255", "dep",
                "converted_actions" keys.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the total loss under key "loss".

        Example:
            >>> # Called automatically by Lightning during training.
        """
        return self.__step__(batch, "train")

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     return self.__step__(batch, "val")

    def __step__(self, batch, phase):
        """Shared step logic for training (and potentially validation).

        Extracts observations and actions from the batch, runs forward pass
        through the agent, and logs all loss metrics.

        Args:
            batch: Dictionary containing "states", "current_idx", "rgb255",
                "dep", and "converted_actions".
            phase: Phase string, e.g. "train" or "val".

        Returns:
            Dictionary with the total "loss" tensor.

        Example:
            >>> ret = model.__step__(batch, "train")
            >>> ret["loss"].backward()
        """
        ret = {}
        current_idx = batch["current_idx"]
        current_state = torch.gather(
            batch["states"],
            1,
            current_idx[:, None, None].expand(-1, -1, batch["states"].shape[-1]),
        ).squeeze(1)
        obs_batch_dict = {
            "state": current_state,
            "rgb": batch["rgb255"].permute(0, 1, 4, 2, 3),  # B,V,3,H,W in 0,255
            "depth": batch["dep"][:, :, None],  # B,V,1,H,W
        }
        act_batch = batch["converted_actions"]  # B,Q,A

        loss_dict = self.agent.compute_loss(
            obs=obs_batch_dict,  # obs_batch_dict['state'] is (B, obs_dim)
            action_seq=act_batch,  # (B, num_queries, act_dim)
        )
        total_loss = loss_dict["loss"]

        ret["loss"] = total_loss

        # Ensure consistent logging for checkpoint monitoring fallback
        self.log(
            f"total_loss/{phase}",
            total_loss,
            on_step=True,
            on_epoch=True,
            batch_size=len(batch["states"]),
            prog_bar=False,
        )

        for k, v in loss_dict.items():
            self.log(
                f"L/{k}",
                v,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch["states"]),
                prog_bar=k == "loss",
            )

        return ret

    def rollout(self, eval_func):
        """Run evaluation rollout using the EMA agent (or main agent if no EMA).

        Temporarily sets the agent to eval mode, runs the provided evaluation
        function, then restores the original training state.

        Args:
            eval_func: Callable that takes (get_action_fn, predict_horizon) and
                returns evaluation results.

        Returns:
            Result from eval_func, typically a dict of evaluation metrics.

        Example:
            >>> results = model.rollout(evaluator.evaluate)
        """
        if self.use_ema:
            self.ema.copy_to(self.ema_agent.parameters())
            eval_agent = self.ema_agent
        else:
            eval_agent = self.agent
        is_training = eval_agent.training
        eval_agent.eval()
        ret = eval_func(eval_agent.get_action, int(eval_agent.predict_horizon))
        if is_training:
            eval_agent.train()
        return ret

    @torch.no_grad()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Visualize RGB observations by logging concatenated camera views.

        Args:
            batch_idx: Index of the current batch.
            batch: Batch dictionary containing "rgb255" of shape (B, V, H, W, 3).
            step_output: Output from the training/validation step (unused).
            phase: Phase string, e.g. "train" or "val".
            **kwargs: Additional keyword arguments (unused).

        Example:
            >>> model.viz(batch_idx=0, batch=batch, step_output=None, phase="train")
        """
        logging.warning(f"Start viz batchidx={batch_idx} in phase={phase} ...")
        rgbs = batch["rgb255"][0]  # V,H,W,3
        rgbs = [i for i in rgbs]
        rgbs = torch.cat(rgbs, 1)  # H,W',3
        self.logger.log_image(
            phase,
            f"rgb_obs_{phase}",
            rgbs.float().permute(2, 0, 1) / 255.0,
            self.global_step,
            batch_idx,
        )
        return

    def configure_optimizers(self):
        """Configure AdamW optimizer and learning rate scheduler.

        Supports two LR schedule modes:
        - "step": StepLR with a single drop at a fraction of total steps.
        - "cosine": Linear warmup followed by cosine annealing to min_lr.

        Returns:
            Dictionary with "optimizer" and "lr_scheduler" keys for Lightning.

        Example:
            >>> # Called automatically by Lightning during trainer.fit().
        """
        self.lr = float(self.optim_cfg.get("lr", 6e-4))
        self.lr_backbone = float(self.optim_cfg.get("lr_backbone", 1.0e-5))
        optim_group = self.agent.get_optim_groups(lr_backbone=self.lr_backbone)
        self.optimizer = torch.optim.AdamW(
            optim_group,
            lr=self.lr,
            betas=self.optim_cfg.get("betas", (0.9, 0.95)),
            eps=self.optim_cfg.get("eps", 1e-8),
            # fused=True,
            weight_decay=self.optim_cfg.get("weight_decay", 1.0e-2),
        )

        self.lr_schedule_mode = self.optim_cfg.get("lr_schedule_mode", "cosine")
        assert self.lr_schedule_mode in ["cosine", "step"]

        if self.lr_schedule_mode == "step":
            total_steps = self.trainer.max_steps
            lr_drop = int(
                total_steps * self.optim_cfg.get("lr_drop_fraction", 2.0 / 3.0)
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)
        elif self.lr_schedule_mode == "cosine":
            self.min_lr = self.optim_cfg.get("min_lr", 6e-5)
            self.lr_decay_iters = self.optim_cfg.get("lr_decay_iters", 600000)
            self.lr_warmup_iters = self.optim_cfg.get("lr_warmup_iters", 2000)

            def lr_func(it):
                """Compute absolute LR at iteration *it* with warmup + cosine decay.

                Args:
                    it: Current training iteration.

                Returns:
                    Absolute learning rate value (divided by ``self.lr`` externally
                    to produce a multiplier for ``LambdaLR``).
                """
                # ! warning, this function returns the factor that multiply with the init lr
                # logging.warning(f"DEBUG: Step: {it}")
                # 1) linear warmup for warmup_iters steps
                if it < self.lr_warmup_iters:
                    ret = self.lr * (it + 1) / (self.lr_warmup_iters + 1.0)
                # 2) if it > lr_decay_iters, return min learning rate
                elif it > self.lr_decay_iters:
                    ret = float(self.min_lr)
                else:
                    # 3) in between, use cosine decay down to min learning rate
                    decay_ratio = (it - self.lr_warmup_iters) / (
                        self.lr_decay_iters - self.lr_warmup_iters
                    )
                    assert 0 <= decay_ratio <= 1
                    coeff = 0.5 * (
                        1.0 + np.cos(np.pi * decay_ratio)
                    )  # coeff ranges 0..1
                    ret = self.min_lr + coeff * (self.lr - self.min_lr)
                factor = ret / self.lr
                return factor

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_func
            )
        else:
            self.scheduler = None

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
