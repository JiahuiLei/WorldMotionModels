"""PyTorch Lightning training module for UMM V10.

Wraps ``AgentV10`` (SIM(3) canonicalization + 6D rotation + flow matching)
and handles training/validation loops, EMA, optimizer, gradient monitoring,
checkpoint save/load, and optional LMM-only compilation.

Example::

    from umm.models.umm_v10.model import PLModel
    model = PLModel(cfg)
    trainer.fit(model, datamodule)
"""

import json
import logging
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.utilities import rank_zero_only

from umm.models.umm_v10.agent import AgentV10
from umm.models.umm_v10.viz_helpers.human_viz import V10HumanViz
from umm.models.umm_v10.viz_helpers.taco_viz import V10TacoViz
from umm.models.umm_v10.viz_helpers.langtable_viz import V10LangTableViz
from umm.models.umm_v9.world_models.wm_utils import disable_model_grad


def sync_model_buffers(
    source_model: torch.nn.Module, target_model: torch.nn.Module
) -> None:
    """Copy all buffers from source to target (EMA only tracks parameters)."""
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source_model.buffers(), target_model.buffers()):
            buf_tgt.copy_(buf_src)


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for UMM V10.

    Wraps AgentV10 with EMA, optimizer, gradient monitoring, and viz dispatch.

    Args:
        cfg: Full configuration dict. Must include ``model.name = "umm_v10"``.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        self.model: AgentV10 = self._get_agent(
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
            self.ema_model = disable_model_grad(AgentV10(cfg))
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

        self.color_plate = cm.get_cmap("Set3").colors[:10]
        self._init_visualizers()

    def _init_visualizers(self):
        """Initialize domain-specific V10 visualizers.

        Supports human+humanoid, TACO, and LangTable domains.
        Visualizers are only created when their config section exists.
        """
        # Human + humanoid
        human_cfg = self.model_cfg.get("human", {})
        humanoid_cfg = self.model_cfg.get("humanoid", {})
        if human_cfg or humanoid_cfg:
            self.human_visualizer = V10HumanViz(self, human_cfg, humanoid_cfg)
        else:
            self.human_visualizer = None

        # TACO
        taco_cfg = self.model_cfg.get("taco", {})
        if taco_cfg:
            self.taco_visualizer = V10TacoViz(self, taco_cfg)
        else:
            self.taco_visualizer = None

        # LangTable
        langtable_cfg = self.model_cfg.get("langtable", {})
        if langtable_cfg:
            self.langtable_visualizer = V10LangTableViz(self, langtable_cfg)
        else:
            self.langtable_visualizer = None

    @staticmethod
    def _get_agent(
        cfg,
        init_fn=None,
        ckpt_prefix="model.",
        strict_flag=True,
        lmm_flag=True,
        input_flag=False,
    ):
        """Create and optionally initialize AgentV10 from checkpoint."""
        model = AgentV10(cfg)

        model_cfg = cfg.get("model", {})
        if init_fn is None:
            init_fn = model_cfg.get("init_fn", None)
        if init_fn is not None:
            checkpoint_dict = AgentV10.filter_ckpt_prefix(
                torch.load(init_fn)["state_dict"], ckpt_prefix
            )
            # Handle compiled model checkpoint prefix
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
                f"Loaded V10 checkpoint from {init_fn} "
                f"Strict={strict_flag} LMM={lmm_flag} Input={input_flag}"
            )
        return model

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

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

                total_norm_val = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
                if total_norm_val > grad_clip_val:
                    logging.warning(
                        f"Gradient clipping at step {self.global_step}: "
                        f"grad_norm={total_norm_val:.4f} > clip_val={grad_clip_val:.4f}"
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

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state."""
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Load EMA state. Handles missing keys and shape mismatches from cross-mode loading."""
        if "state_dict" in checkpoint:
            current_state = self.state_dict()
            # Fill missing keys
            for key in current_state:
                if key not in checkpoint["state_dict"]:
                    checkpoint["state_dict"][key] = current_state[key]
                    logging.info(f"Filling missing key from current model: {key}")

            # Handle shape mismatches (e.g. input_merge between v10_3cat and noisy_only)
            keys_to_replace = []
            for key in list(checkpoint["state_dict"].keys()):
                if key in current_state:
                    ck_shape = tuple(checkpoint["state_dict"][key].shape)
                    cur_shape = tuple(current_state[key].shape)
                    if ck_shape != cur_shape:
                        keys_to_replace.append(key)
                        logging.warning(
                            "Shape mismatch for '%s': checkpoint %s vs current %s. "
                            "Replacing with current model value.",
                            key, ck_shape, cur_shape,
                        )
            for key in keys_to_replace:
                checkpoint["state_dict"][key] = current_state[key]

        super().on_load_checkpoint(checkpoint)

        if self.use_ema and "ema_state_dict" in checkpoint:
            current_ema = self.ema.state_dict()
            for key in current_ema:
                if key not in checkpoint["ema_state_dict"]:
                    checkpoint["ema_state_dict"][key] = current_ema[key]
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.ema.copy_to(self.ema_model.parameters())
            logging.info("Loaded EMA weights from checkpoint")

    # ------------------------------------------------------------------
    # Optimizer / LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure optimizer with cosine-warmup LR schedule."""
        self.lr = float(self.optim_cfg.get("lr", 6e-4))
        self.min_lr = self.optim_cfg.get("min_lr", 6e-5)
        self.lr_decay_iters = self.optim_cfg.get("lr_decay_iters", 600000)
        self.lr_warmup_iters = self.optim_cfg.get("lr_warmup_iters", 2000)
        self.optimizer_weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        logging.info(f"Optimizer weight decay: {self.optimizer_weight_decay}")

        lmm_lr = self.optim_cfg.get("lmm_lr", self.lr)
        input_layer_lr = self.optim_cfg.get("input_layer_lr", self.lr)

        optim_group = self.model.get_optim_groups(
            weight_decay=self.optimizer_weight_decay,
            lmm_lr=lmm_lr,
            input_layer_lr=input_layer_lr,
        )

        self.optimizer = torch.optim.AdamW(
            optim_group,
            lr=self.lr,
            betas=self.optim_cfg.get("betas", (0.9, 0.999)),
            eps=self.optim_cfg.get("eps", 1e-8),
        )

        def make_lr_func(base_lr):
            min_lr_ratio = self.min_lr / self.lr
            actual_min_lr = base_lr * min_lr_ratio

            def lr_func(it):
                if it < self.lr_warmup_iters:
                    ret = base_lr * (it + 1) / (self.lr_warmup_iters + 1.0)
                elif it > self.lr_decay_iters:
                    ret = float(actual_min_lr)
                else:
                    decay_ratio = (it - self.lr_warmup_iters) / (
                        self.lr_decay_iters - self.lr_warmup_iters
                    )
                    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                    ret = actual_min_lr + coeff * (base_lr - actual_min_lr)
                return ret / base_lr
            return lr_func

        lr_funcs = [make_lr_func(g.get("lr", self.lr)) for g in optim_group]

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_funcs
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Visualization dispatch — minimal for V10 initial bringup."""
        torch.cuda.empty_cache()
        is_training = self.training
        self.eval()
        logging.warning(f"V10 viz batchidx={batch_idx} phase={phase}")

        if self.use_ema:
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            eval_model = self.ema_model
        else:
            eval_model = self.model
        eval_model.eval()

        viz_try = self.cfg.get("viz_try", False)

        if self.human_visualizer is not None and (
            getattr(self.human_visualizer, "human_flag", False)
            or getattr(self.human_visualizer, "humanoid_flag", False)
        ):
            if viz_try:
                try:
                    self.human_visualizer.viz_routine(
                        batch, phase, batch_idx, eval_model, step_output
                    )
                except Exception as e:
                    logging.error(f"V10 HH viz failed: {e}", exc_info=True)
            else:
                self.human_visualizer.viz_routine(
                    batch, phase, batch_idx, eval_model, step_output
                )

        if self.taco_visualizer is not None and getattr(
            self.taco_visualizer, "flag", False
        ):
            if viz_try:
                try:
                    self.taco_visualizer.viz_routine(
                        batch, phase, batch_idx, eval_model
                    )
                except Exception as e:
                    logging.error(f"V10 TACO viz failed: {e}", exc_info=True)
            else:
                self.taco_visualizer.viz_routine(
                    batch, phase, batch_idx, eval_model
                )

        if self.langtable_visualizer is not None and getattr(
            self.langtable_visualizer, "flag", False
        ):
            if viz_try:
                try:
                    self.langtable_visualizer.viz_routine(
                        batch, phase, batch_idx, eval_model, step_output
                    )
                except Exception as e:
                    logging.error(f"V10 LT viz failed: {e}", exc_info=True)
            else:
                self.langtable_visualizer.viz_routine(
                    batch, phase, batch_idx, eval_model, step_output
                )

        if is_training:
            self.train()
