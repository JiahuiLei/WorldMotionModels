"""PyTorch Lightning training module for UMM E11 (equivariant sandwich).

Wraps AgentE11 with training/validation loops, EMA, optimizer, and viz dispatch.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm

from umm.models.umm_e11.agent import AgentE11
from umm.models.umm_e11.viz_helpers.behavior_viz_v2 import BehaviorVisualizer
from umm.models.umm_e11.viz_helpers.real_viz import RealVisualizer
from umm.models.umm_e11.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_e11.viz_helpers.langtable_viz import LangTableVisualizer
from umm.models.umm_e11.viz_helpers.manipulator_viz import ManipulatorVisualizer
from umm.models.umm_e11.viz_helpers.maniptrans_viz import ManipTransVisualizer
from umm.models.umm_e11.viz_helpers.maniskill_viz import ManiskillVisualizer
from umm.models.umm_e11.viz_helpers.taco_viz import TacoVisualizer


def sync_model_buffers(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy all buffers from source to target (EMA only tracks parameters)."""
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source.buffers(), target.buffers()):
            buf_tgt.copy_(buf_src)


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for E11 equivariant sandwich training."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        # Create agent
        self.model = AgentE11(cfg)

        if self.model_cfg.get("compile_flag", False):
            logging.warning("@" * 40)
            logging.warning("Compiling model...")
            self.model = torch.compile(self.model)
            logging.warning("Model compiled.")

        # EMA
        self.use_ema = self.model_cfg.get("use_ema", True)
        if self.use_ema:
            from diffusers.training_utils import EMAModel
            self.ema = EMAModel(
                parameters=self.model.parameters(),
                power=0.75,
                decay=self.model_cfg.get("ema_decay", 0.9999),
            )
            self.ema_model = AgentE11(cfg)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema.copy_to(self.ema_model.parameters())

        # Optimizer config
        self.lr = self.optim_cfg.get("lr", 1e-4)
        self.min_lr = self.optim_cfg.get("min_lr", 1e-5)
        self.lr_warmup_iters = self.optim_cfg.get("lr_warmup_iters", 100)
        self.lr_decay_iters = self.optim_cfg.get("lr_decay_iters", 300000)
        self.optimizer_weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        self.optimizer_type = self.optim_cfg.get("optimizer_type", "adamw")

        # Viz helpers
        self.color_plate = cm.get_cmap("Set3").colors[:10]
        self._init_visualizers()

        n_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"PLModel E11: {n_params:,} params")

    def _init_visualizers(self):
        """Initialize domain-specific visualizers (matching v9/E10 pattern)."""
        self.behavior_visualizer = BehaviorVisualizer(
            self, self.cfg.get("visualize", {}).get("behavior", {})
        )
        self.real_visualizer = RealVisualizer(
            self, self.cfg.get("visualize", {}).get("real", {})
        )
        human_cfg = self.model_cfg.get("human", {})
        humanoid_cfg = self.model_cfg.get("humanoid", {})
        if human_cfg or humanoid_cfg:
            self.human_visualizer = HumanVisualizer(self, human_cfg, humanoid_cfg)
        else:
            self.human_visualizer = None
        langtable_cfg = self.model_cfg.get("langtable", {})
        self.langtable_visualizer = LangTableVisualizer(self, langtable_cfg) if langtable_cfg else None
        manipulator_cfg = self.model_cfg.get("manipulator", {})
        self.manipulator_visualizer = ManipulatorVisualizer(self, manipulator_cfg) if manipulator_cfg else None
        taco_cfg = self.model_cfg.get("taco", {})
        self.taco_visualizer = TacoVisualizer(self, taco_cfg) if taco_cfg else None
        maniskill_cfg = self.model_cfg.get("maniskill", {})
        self.maniskill_visualizer = ManiskillVisualizer(self, maniskill_cfg) if maniskill_cfg else None
        maniptrans_cfg = self.model_cfg.get("maniptrans", {})
        self.maniptrans_visualizer = ManipTransVisualizer(self, maniptrans_cfg) if maniptrans_cfg else None

    def on_fit_start(self):
        """Move EMA shadow params and ema_model to the correct device."""
        if self.use_ema:
            device = self.device
            self.ema.shadow_params = [p.to(device) for p in self.ema.shadow_params]
            self.ema_model = self.ema_model.to(device)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        self.train()
        ret = self.model(batch, "train", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(k, v, on_step=True, on_epoch=True,
                     batch_size=len(batch["trace"]), prog_bar=False)
        return ret

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        ret = self.model(batch, "val", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(k, v, on_step=True, on_epoch=True,
                     batch_size=len(batch["trace"]), prog_bar=False)
        return ret

    def on_before_optimizer_step(self, optimizer):
        grad_clip_val = self.trainer.gradient_clip_val
        if grad_clip_val is not None and grad_clip_val > 0:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if len(parameters) > 0:
                device = parameters[0].device
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]),
                    2.0,
                )
                self.log("grad_norm", total_norm, on_step=True, prog_bar=False)
                if total_norm > grad_clip_val:
                    self.log("grad_clip_count", 1.0, on_step=True, reduce_fx="sum")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema.step(self.model.parameters())
            if next(self.ema_model.parameters()).device != self.device:
                self.ema_model = self.ema_model.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.use_ema and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.ema.copy_to(self.ema_model.parameters())

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Dispatch visualization to domain-specific visualizers."""
        torch.cuda.empty_cache()
        is_training = self.training
        self.eval()
        logging.warning(f"Start viz batchidx={batch_idx} in phase={phase} ...")
        start_time = time.time()

        if self.use_ema:
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            eval_model = self.ema_model
        else:
            eval_model = self.model
        eval_model.eval()

        if self.behavior_visualizer is not None and self.behavior_visualizer.flag:
            self.behavior_visualizer.behavior_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        if self.real_visualizer is not None and self.real_visualizer.flag:
            self.real_visualizer.real_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        if self.langtable_visualizer is not None and getattr(
            self.langtable_visualizer, "flag", False
        ):
            self.langtable_visualizer.langtable_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        if self.manipulator_visualizer is not None and getattr(
            self.manipulator_visualizer, "flag", False
        ):
            self.manipulator_visualizer.manipulator_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        if self.human_visualizer is not None and (
            getattr(self.human_visualizer, "human_flag", False)
            or getattr(self.human_visualizer, "humanoid_flag", False)
        ):
            self.human_visualizer.human_humanoid_viz_routine(
                step_output, batch, phase, batch_idx, eval_model
            )

        if self.taco_visualizer is not None and getattr(
            self.taco_visualizer, "flag", False
        ):
            self.taco_visualizer.taco_viz_routine(batch, phase, batch_idx, eval_model)

        if self.maniptrans_visualizer is not None and getattr(
            self.maniptrans_visualizer, "flag", False
        ):
            self.maniptrans_visualizer.maniptrans_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        if self.maniskill_visualizer is not None and getattr(
            self.maniskill_visualizer, "flag", False
        ):
            self.maniskill_visualizer.maniskill_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        elapsed = time.time() - start_time
        logging.warning(f"Viz done in {elapsed:.1f}s")
        if is_training:
            self.train()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.lr,
                betas=self.optim_cfg.get("betas", (0.9, 0.999)),
                eps=self.optim_cfg.get("eps", 1e-8),
                weight_decay=self.optimizer_weight_decay,
            )
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.lr,
                betas=self.optim_cfg.get("betas", (0.9, 0.999)),
                eps=self.optim_cfg.get("eps", 1e-8),
            )
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_type} not implemented")

        def lr_func(it):
            if it < self.lr_warmup_iters:
                return (it + 1) / (self.lr_warmup_iters + 1.0)
            elif it > self.lr_decay_iters:
                return self.min_lr / self.lr
            else:
                decay_ratio = (it - self.lr_warmup_iters) / (
                    self.lr_decay_iters - self.lr_warmup_iters
                )
                coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                return (self.min_lr / self.lr) + coeff * (1.0 - self.min_lr / self.lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
