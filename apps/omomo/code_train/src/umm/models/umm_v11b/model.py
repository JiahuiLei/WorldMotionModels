"""PyTorch Lightning training module for UMM V11B.

Wraps AgentV11B with EMA, optimizer, gradient monitoring, and viz dispatch.

Example::

    from umm.models.umm_v11b.model import PLModel
    model = PLModel(cfg)
    trainer.fit(model, datamodule)
"""

import json
import logging
import time
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.utilities import rank_zero_only

from umm.models.umm_v11b.agent import AgentV11B
from umm.models.umm_v9.world_models.wm_utils import disable_model_grad


def sync_model_buffers(source_model, target_model):
    """Copy all buffers from source to target (EMA only tracks parameters)."""
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source_model.buffers(), target_model.buffers()):
            buf_tgt.copy_(buf_src)


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for UMM V11B.

    Args:
        cfg: Full configuration dict. Must include ``model.name = "umm_v11b"``.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        self.model: AgentV11B = self._get_agent(
            cfg,
            init_fn=self.model_cfg.get("init_fn", None),
            strict_flag=self.model_cfg.get("init_strict_flag", True),
            lmm_flag=self.model_cfg.get("init_lmm_flag", True),
            input_flag=self.model_cfg.get("init_input_flag", True),
        )

        if self.model_cfg.get("compile_flag", False):
            logging.warning("@" * 40)
            logging.warning("Compiling LMM submodule only...")
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 256
            compile_dynamic = self.model_cfg.get("compile_dynamic", True)
            compile_mode = self.model_cfg.get("compile_mode", None)
            logging.warning(f"compile dynamic={compile_dynamic}, mode={compile_mode}")
            self.model.lmm = torch.compile(
                self.model.lmm, dynamic=compile_dynamic, mode=compile_mode,
            )
            logging.warning("LMM compiled.")

        self.use_ema = self.model_cfg.get("use_ema", True)
        if self.use_ema:
            from diffusers.training_utils import EMAModel

            self.ema = EMAModel(
                parameters=self.model.parameters(),
                power=0.75,
                decay=self.model_cfg.get("ema_decay", 0.9999),
            )
            self.ema_model = disable_model_grad(AgentV11B(cfg))
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

        self.color_plate = cm.get_cmap("Set3").colors[:10]

        # Initialize domain visualizers
        self._init_visualizers()

    def _init_visualizers(self):
        """Initialize domain-specific visualizers for V11.

        Currently supports human+humanoid (HH) domain only.
        """
        human_cfg = self.model_cfg.get("human", {})
        humanoid_cfg = self.model_cfg.get("humanoid", {})
        if human_cfg or humanoid_cfg:
            from umm.models.umm_v11b.viz_helpers.human_viz import HumanVisualizerV11
            self.human_visualizer = HumanVisualizerV11(
                self, human_cfg, humanoid_cfg
            )
        else:
            self.human_visualizer = None

    @torch.inference_mode()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Dispatch visualization to registered domain-specific visualizers."""
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

        if self.human_visualizer is not None and (
            getattr(self.human_visualizer, "human_flag", False)
            or getattr(self.human_visualizer, "humanoid_flag", False)
        ):
            self.human_visualizer.human_humanoid_viz_routine(
                step_output, batch, phase, batch_idx, eval_model
            )

        elapsed = time.time() - start_time
        logging.warning(f"Viz done in {elapsed:.1f}s")

        if is_training:
            self.train()

    @staticmethod
    def _get_agent(
        cfg,
        init_fn=None,
        ckpt_prefix="model.",
        strict_flag=True,
        lmm_flag=True,
        input_flag=False,
    ):
        """Create and optionally initialize AgentV11B from checkpoint."""
        model = AgentV11B(cfg)

        model_cfg = cfg.get("model", {})
        if init_fn is None:
            init_fn = model_cfg.get("init_fn", None)
        if init_fn is not None:
            checkpoint_dict = AgentV11B.filter_ckpt_prefix(
                torch.load(init_fn)["state_dict"], ckpt_prefix
            )
            new_dict = {}
            for k, v in checkpoint_dict.items():
                if k.startswith("_orig_mod."):
                    new_dict[k[len("_orig_mod."):]] = v
                else:
                    new_dict[k] = v
            checkpoint_dict = new_dict

            model.load_from_checkpoint(
                checkpoint_dict=checkpoint_dict,
                lmm_flag=lmm_flag,
                input_flag=input_flag,
                strict=strict_flag,
            )
            logging.info(
                f"Loaded V11 checkpoint from {init_fn} "
                f"Strict={strict_flag} LMM={lmm_flag} Input={input_flag}"
            )
        return model

    @rank_zero_only
    def on_train_start(self):
        """Log full training config to wandb."""
        from umm.training_utils.log_helpers import CustomLogger
        if isinstance(self.logger, CustomLogger) and hasattr(self.logger, "wandb_logger"):
            cfg_plain = json.loads(json.dumps(self.cfg, default=str))
            self.logger.wandb_logger.experiment.config.update(
                cfg_plain, allow_val_change=True
            )
            logging.info("Logged full config to wandb")

    def training_step(self, batch, batch_idx):
        """Single training step."""
        self.train()
        ret = self.model(batch, "train", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k, v,
                on_step=True, on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Single validation step."""
        self.eval()
        ret = self.model(batch, "val", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k, v,
                on_step=True, on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        grad_clip_val = self.trainer.gradient_clip_val
        grad_clip_algorithm = self.trainer.gradient_clip_algorithm

        if grad_clip_val is not None and grad_clip_val > 0:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if len(parameters) > 0:
                device = parameters[0].device
                if grad_clip_algorithm == "norm":
                    total_norm = torch.norm(
                        torch.stack([
                            torch.norm(p.grad.detach(), 2.0).to(device)
                            for p in parameters
                        ]), 2.0,
                    )
                else:
                    total_norm = max(
                        [p.grad.detach().abs().max().item() for p in parameters]
                    )

                self.log(
                    "monitoring/global_grad_norm",
                    total_norm,
                    on_step=True, on_epoch=True,
                    batch_size=1, prog_bar=False,
                )

    def optimizer_step(self, *args, **kwargs):
        """Perform optimizer step and update EMA."""
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            model_device = next(self.model.parameters()).device
            if len(self.ema.shadow_params) > 0:
                if self.ema.shadow_params[0].device != model_device:
                    self.ema.to(model_device)
            ema_model_device = next(self.ema_model.parameters()).device
            if ema_model_device != model_device:
                self.ema_model.to(model_device)
            self.ema.step(self.model.parameters())
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

    def configure_optimizers(self):
        """Configure optimizer with warmup + cosine decay + flat min_lr schedule."""
        weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        lr = self.optim_cfg.get("lr", 1e-4)
        lmm_lr = self.optim_cfg.get("lmm_lr", lr)
        input_layer_lr = self.optim_cfg.get("input_layer_lr", lr)
        betas = self.optim_cfg.get("betas", [0.9, 0.999])

        param_groups = self.model.get_optim_groups(
            weight_decay=weight_decay,
            lmm_lr=lmm_lr,
            input_layer_lr=input_layer_lr,
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=tuple(betas),
        )

        # Warmup + cosine decay + flat min_lr (matches v9x LambdaLR pattern)
        warmup_iters = self.optim_cfg.get("lr_warmup_iters", 100)
        decay_iters = self.optim_cfg.get("lr_decay_iters", 300000)
        min_lr = self.optim_cfg.get("min_lr", 1e-5)

        def make_lr_func(base_lr):
            """Create a cosine-warmup LR lambda for a specific base learning rate."""
            min_lr_ratio = min_lr / lr  # ratio relative to global base lr
            actual_min_lr = base_lr * min_lr_ratio

            def lr_func(it):
                if it < warmup_iters:
                    return (it + 1) / (warmup_iters + 1.0)
                elif it > decay_iters:
                    return actual_min_lr / base_lr
                else:
                    decay_ratio = (it - warmup_iters) / (
                        decay_iters - warmup_iters
                    )
                    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                    ret = actual_min_lr + coeff * (base_lr - actual_min_lr)
                    return ret / base_lr

            return lr_func

        lr_funcs = []
        for group in param_groups:
            group_base_lr = group.get("lr", lr)
            lr_funcs.append(make_lr_func(group_base_lr))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_funcs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state to checkpoint."""
        if self.use_ema:
            checkpoint["ema_state_dict"] = {
                "shadow_params": [p.clone() for p in self.ema.shadow_params],
                "decay": self.ema.decay,
                "optimization_step": self.ema.optimization_step,
            }

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA state from checkpoint."""
        if self.use_ema and "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            for sp, loaded_sp in zip(
                self.ema.shadow_params, ema_state["shadow_params"]
            ):
                sp.data.copy_(loaded_sp.data)
            self.ema.decay = ema_state.get("decay", self.ema.decay)
            self.ema.optimization_step = ema_state.get(
                "optimization_step", 0
            )
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            logging.info("Restored EMA state from checkpoint")
        elif self.use_ema:
            logging.warning(
                "No EMA state in checkpoint — EMA will restart from "
                "current weights"
            )
