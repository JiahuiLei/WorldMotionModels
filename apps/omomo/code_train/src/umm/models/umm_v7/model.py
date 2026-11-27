"""PyTorch Lightning training module for UMM V6 All.

Unified training module supporting ALL application domains with the V6
two-scale pooled temporal attention architecture. Optional DINO image
encoder for domains that have images.

Example::

    from umm.models.umm_v7.model import PLModel
    model = PLModel(cfg)
    trainer.fit(model, datamodule)
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm

from umm.models.umm_v7.agent import AgentV7
from umm.models.umm_v7.viz_helpers.behavior_viz_v2 import BehaviorVisualizer
from umm.models.umm_v7.viz_helpers.real_viz import RealVisualizer
from umm.models.umm_v7.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_v7.viz_helpers.langtable_viz import LangTableVisualizer
from umm.models.umm_v7.viz_helpers.manipulator_viz import ManipulatorVisualizer
from umm.models.umm_v7.viz_helpers.maniptrans_viz import ManipTransVisualizer
from umm.models.umm_v7.viz_helpers.maniskill_viz import ManiskillVisualizer
from umm.models.umm_v7.viz_helpers.taco_viz import TacoVisualizer
from umm.models.umm_v7.world_models.wm_utils import disable_model_grad


def sync_model_buffers(
    source_model: torch.nn.Module, target_model: torch.nn.Module
) -> None:
    """Copy all buffers (e.g. BatchNorm running stats) from source to target.

    EMA only tracks parameters; buffers must be synced separately.

    Args:
        source_model: Model to copy buffers from.
        target_model: Model to copy buffers to.
    """
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source_model.buffers(), target_model.buffers()):
            buf_tgt.copy_(buf_src)


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for UMM V6.

    Wraps ``AgentV7`` (with optional frozen DINOv2 encoder, dual-path input layer)
    and handles training/validation loops, EMA weight management, optimizer
    configuration, gradient monitoring, checkpoint save/load, and
    visualization dispatch.

    Args:
        cfg: Full configuration dict. Must include ``model.name = "umm_v6"``
            and optionally ``model.dino_encoder`` sub-dict.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """Initialize PLModel with AgentV7, EMA, and domain visualizers.

        Args:
            cfg: Configuration dictionary. If None, uses empty dict defaults.
        """
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg, self.model_cfg = cfg, cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        self.model: AgentV7 = self._get_agent(
            cfg,
            init_fn=self.model_cfg.get("init_fn", None),
            strict_flag=self.model_cfg.get("init_strict_flag", True),
            lmm_flag=self.model_cfg.get("init_lmm_flag", True),
            input_flag=self.model_cfg.get("init_input_flag", True),
        )
        if self.model_cfg.get("compile_flag", False):
            logging.warning("@" * 40)
            logging.warning("Compiling model...")
            self.model = torch.compile(self.model)
            logging.warning("Model compiled.")

        self.use_ema = self.model_cfg.get("use_ema", True)
        if self.use_ema:
            from diffusers.training_utils import EMAModel

            self.ema = EMAModel(
                parameters=self.model.parameters(),
                power=0.75,
                decay=self.model_cfg.get("ema_decay", 0.9999),
            )
            self.ema_model = disable_model_grad(AgentV7(cfg))
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

        # Initialize visualizers — V5 currently supports behavior domain only
        self._init_visualizers()
        self.color_plate = cm.get_cmap("Set3").colors[:10]

    def _init_visualizers(self):
        """Initialize domain-specific visualizers.

        Supports behavior, real, and human+humanoid domains. Other domain
        visualizers are set to None for now.
        """
        self.behavior_visualizer = BehaviorVisualizer(
            self, self.cfg.get("visualize", {}).get("behavior", {})
        )
        self.real_visualizer = RealVisualizer(
            self, self.cfg.get("visualize", {}).get("real", {})
        )
        # Human + humanoid visualizer
        human_cfg = self.model_cfg.get("human", {})
        humanoid_cfg = self.model_cfg.get("humanoid", {})
        if human_cfg or humanoid_cfg:
            self.human_visualizer = HumanVisualizer(
                self, human_cfg, humanoid_cfg
            )
        else:
            self.human_visualizer = None
        # LangTable visualizer
        langtable_cfg = self.model_cfg.get("langtable", {})
        if langtable_cfg:
            self.langtable_visualizer = LangTableVisualizer(
                self, langtable_cfg
            )
        else:
            self.langtable_visualizer = None
        # Manipulator (motion planning) visualizer
        manipulator_cfg = self.model_cfg.get("manipulator", {})
        if manipulator_cfg:
            self.manipulator_visualizer = ManipulatorVisualizer(
                self, manipulator_cfg
            )
        else:
            self.manipulator_visualizer = None
        # TACO visualizer
        taco_cfg = self.model_cfg.get("taco", {})
        if taco_cfg:
            self.taco_visualizer = TacoVisualizer(self, taco_cfg)
        else:
            self.taco_visualizer = None
        # ManiSkill visualizer
        maniskill_cfg = self.model_cfg.get("maniskill", {})
        if maniskill_cfg:
            self.maniskill_visualizer = ManiskillVisualizer(
                self, maniskill_cfg
            )
        else:
            self.maniskill_visualizer = None
        # ManipTrans (dexterous hand) visualizer
        maniptrans_cfg = self.model_cfg.get("maniptrans", {})
        if maniptrans_cfg:
            self.maniptrans_visualizer = ManipTransVisualizer(
                self, maniptrans_cfg
            )
        else:
            self.maniptrans_visualizer = None

    @staticmethod
    def _get_agent(
        cfg,
        init_fn=None,
        ckpt_prefix="model.",
        strict_flag=True,
        lmm_flag=True,
        input_flag=False,
        compile_flag=False,
    ):
        """Create and optionally initialize an AgentV7 from a checkpoint.

        Args:
            cfg: Full configuration dictionary.
            init_fn: Path to checkpoint file for weight initialization.
            ckpt_prefix: Key prefix to strip from checkpoint state dict.
            strict_flag: Enforce strict key matching on load.
            lmm_flag: Load LMM weights.
            input_flag: Load input layer weights.
            compile_flag: Whether to ``torch.compile`` the model.

        Returns:
            Initialized ``AgentV7`` instance.
        """
        model: AgentV7 = AgentV7(cfg)

        if compile_flag:
            logging.warning("@" * 40)
            logging.warning("Compiling model...")
            model = torch.compile(model)
            logging.warning("Model compiled.")

        model_cfg = cfg.get("model", {})
        if init_fn is None:
            init_fn = model_cfg.get("init_fn", None)
        if init_fn is not None:
            checkpoint_dict = AgentV7.filter_ckpt_prefix(
                torch.load(init_fn)["state_dict"], ckpt_prefix
            )

            # Handle compiled model checkpoint prefix
            new_checkpoint_dict = {}
            for k, v in checkpoint_dict.items():
                if k.startswith("_orig_mod."):
                    new_checkpoint_dict[k[len("_orig_mod.") :]] = v
                else:
                    new_checkpoint_dict[k] = v
            checkpoint_dict = new_checkpoint_dict

            model.load_from_checkpoint(
                checkpoint_dict=checkpoint_dict,
                lmm_flag=lmm_flag,
                input_flag=input_flag,
                strict=strict_flag,
            )
            logging.info(
                f"Loaded checkpoint from {init_fn} Strict={strict_flag} "
                f"LMM={lmm_flag} Input={input_flag}, ckpt_prefix={ckpt_prefix}"
            )
            if hasattr(model, "ctx_record_buffer"):
                ctx_buffer_key = None
                for key in checkpoint_dict.keys():
                    if key.endswith("ctx_record_buffer"):
                        ctx_buffer_key = key
                        break
                if ctx_buffer_key is not None:
                    model.ctx_record_buffer = checkpoint_dict[ctx_buffer_key]
                    logging.info(f"Loaded ctx_record_buffer from key: {ctx_buffer_key}")
                    for k, v in model.ctx_record_buffer.items():
                        logging.info(f"ctx_record_buffer: {k[:3]}...: {len(v)}")
                else:
                    logging.warning(
                        "ctx_record_buffer not found in checkpoint_dict after filtering"
                    )
        return model

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Execute a single training step through the Agent.

        Args:
            batch: Batch data dict.
            batch_idx: Index of this batch in the current epoch.

        Returns:
            Dict containing ``"loss"`` and ``"log_dict"``.
        """
        self.train()
        ret = self.model(batch, "train", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k,
                v,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Execute a single validation step through the Agent.

        Args:
            batch: Batch data dict.
            batch_idx: Index of this batch.
            dataloader_idx: Index of the active validation dataloader.

        Returns:
            Dict containing ``"loss"`` and ``"log_dict"``.
        """
        self.eval()
        ret = self.model(batch, "val", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k,
                v,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms and warn if gradient clipping occurred.

        Args:
            optimizer: The optimizer about to step.
        """
        extra_log_flag = self.cfg.get("debug_logging", False)
        start_time = time.time()
        grad_clip_val = self.trainer.gradient_clip_val
        grad_clip_algorithm = self.trainer.gradient_clip_algorithm

        if grad_clip_val is not None and grad_clip_val > 0:
            parameters = [p for p in self.model.parameters() if p.grad is not None]

            if len(parameters) > 0:
                device = parameters[0].device

                if grad_clip_algorithm == "norm":
                    total_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2.0).to(device)
                                for p in parameters
                            ]
                        ),
                        2.0,
                    )
                elif grad_clip_algorithm == "value":
                    total_norm = max(
                        [p.grad.detach().abs().max().item() for p in parameters]
                    )
                else:
                    total_norm = 0.0

                logging.debug(
                    f"Grad Norm at step {self.global_step}: {total_norm:.4f} / max={grad_clip_val:.4f}"
                )
                self.log(
                    "monitoring/global_grad_norm",
                    total_norm,
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                    prog_bar=False,
                )

                if total_norm > grad_clip_val:
                    logging.warning("@" * 40)
                    logging.warning(
                        f"Gradient clipping occurred at step {self.global_step}: "
                        f"grad_norm={total_norm:.4f} > clip_val={grad_clip_val:.4f} "
                        f"(algorithm={grad_clip_algorithm})"
                    )
                    logging.warning("@" * 40)

                if extra_log_flag:
                    significant_threshold = grad_clip_val * 0.1
                    param_grad_info = []
                    log_grad_info = []
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_norm = torch.norm(p.grad.detach(), 2.0).item()
                            log_grad_info.append((name, grad_norm, p.grad.shape))
                            if grad_norm > significant_threshold:
                                param_grad_info.append((name, grad_norm, p.grad.shape))

                    param_grad_info.sort(key=lambda x: x[1], reverse=True)
                    log_grad_info.sort(key=lambda x: x[1], reverse=True)

                    grad_norm_dict = {
                        f"debug/grad_norm/{name.replace('.', '/')}": gn
                        for name, gn, shape in log_grad_info
                    }
                    self.log_dict(
                        grad_norm_dict,
                        on_step=True,
                        on_epoch=False,
                        batch_size=1,
                        prog_bar=False,
                    )

                    if total_norm > grad_clip_val:
                        logging.warning("@" * 40)
                        logging.warning(
                            f"Gradient clipping occurred at step {self.global_step}: "
                            f"grad_norm={total_norm:.4f} > clip_val={grad_clip_val:.4f} "
                            f"(algorithm={grad_clip_algorithm})"
                        )
                        logging.warning(
                            f"  Parameters with grad_norm > {significant_threshold:.4f} "
                            f"({len(param_grad_info)} out of "
                            f"{len(list(self.model.parameters()))} total):"
                        )
                        for i, (name, gn, shape) in enumerate(param_grad_info):
                            logging.warning(
                                f"    {i+1}. {name} (shape={shape}): {gn:.4f}"
                            )
                        logging.warning("@" * 40)
        logging.debug(
            f"Time taken for gradient checking: {time.time() - start_time:.4f} seconds"
        )

    def optimizer_step(self, *args, **kwargs):
        """Perform optimizer step and update EMA weights.

        Args:
            *args: Forwarded to ``super().optimizer_step``.
            **kwargs: Forwarded to ``super().optimizer_step``.
        """
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
            if hasattr(self.model, "ctx_record_buffer"):
                self.ema_model.ctx_record_buffer = self.model.ctx_record_buffer

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state dict alongside the standard checkpoint.

        Args:
            checkpoint: Checkpoint dictionary to augment.
        """
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA state, ctx_record_buffer, and handle key mismatches.

        Handles missing DINO encoder keys when loading from an older checkpoint
        into V5.

        Args:
            checkpoint: Checkpoint dictionary loaded from disk.
        """
        if "state_dict" in checkpoint:
            current_state = self.state_dict()
            for key in current_state:
                if key not in checkpoint["state_dict"]:
                    if key.startswith("model.dino_encoder.") or key.startswith(
                        "ema_model.dino_encoder."
                    ):
                        checkpoint["state_dict"][key] = current_state[key]
                        logging.info(
                            f"Filling missing DINO key from current model: {key}"
                        )
                    elif key.startswith("human_model."):
                        checkpoint["state_dict"][key] = current_state[key]

        # Extract ctx_record_buffer before super() to avoid key mismatch
        ctx_buffer_data = None
        ctx_buffer_keys_to_remove = []
        if "state_dict" in checkpoint:
            for key in list(checkpoint["state_dict"].keys()):
                if key.endswith("ctx_record_buffer") and (
                    key.startswith("model.") or key.startswith("ema_model.")
                ):
                    if key.startswith("model.") and ctx_buffer_data is None:
                        ctx_buffer_data = checkpoint["state_dict"][key]
                        logging.info(
                            f"Extracted ctx_record_buffer from checkpoint key: {key}"
                        )
                    ctx_buffer_keys_to_remove.append(key)

            for key in ctx_buffer_keys_to_remove:
                del checkpoint["state_dict"][key]

        super().on_load_checkpoint(checkpoint)

        # Manually restore ctx_record_buffer
        if self.model.ctx_record_flag:
            if ctx_buffer_data is not None:
                self.model.ctx_record_buffer = ctx_buffer_data
                logging.info(
                    f"Loaded ctx_record_buffer with {len(ctx_buffer_data)} entries"
                )
            else:
                logging.warning(
                    "ctx_record_buffer not found in checkpoint, initializing empty"
                )

        if self.use_ema and "ema_state_dict" in checkpoint:
            current_ema_state = self.ema.state_dict()
            for key in current_ema_state:
                if (
                    key.startswith("shadow_params")
                    and key not in checkpoint["ema_state_dict"]
                ):
                    checkpoint["ema_state_dict"][key] = current_ema_state[key]

            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.ema.copy_to(self.ema_model.parameters())
            logging.info("Successfully loaded EMA weights from checkpoint")

            if self.model.ctx_record_flag:
                self.ema_model.ctx_record_buffer = self.model.ctx_record_buffer

    # ------------------------------------------------------------------
    # Optimizer / LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure optimizer and cosine-warmup LR scheduler.

        Returns:
            Dict with ``"optimizer"`` and ``"lr_scheduler"`` entries.
        """
        self.lr = float(self.optim_cfg.get("lr", 6e-4))
        self.min_lr = self.optim_cfg.get("min_lr", 6e-5)
        self.lr_decay_iters = self.optim_cfg.get("lr_decay_iters", 600000)
        self.lr_warmup_iters = self.optim_cfg.get("lr_warmup_iters", 2000)
        self.optimizer_weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        logging.info(f"Optimizer weight decay: {self.optimizer_weight_decay}")

        self.optimizer_type = self.optim_cfg.get("type", "adamw").lower()
        if self.optimizer_type == "adamw":
            logging.warning("Use AdamW Optimizer")
            optim_cls = torch.optim.AdamW
        elif self.optimizer_type == "sgd":
            logging.warning("Use SGD Optimizer")
            optim_cls = torch.optim.SGD
        elif self.optimizer_type == "adam":
            logging.warning("Use Adam Optimizer")
            optim_cls = torch.optim.Adam
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_type} not implemented"
            )

        lmm_lr = self.optim_cfg.get("lmm_lr", self.lr)
        input_layer_lr = self.optim_cfg.get("input_layer_lr", self.lr)
        image_encoder_lr = self.optim_cfg.get("image_encoder_lr", self.lr)

        optim_group = self.model.get_optim_groups(
            weight_decay=self.optimizer_weight_decay,
            lmm_lr=lmm_lr,
            input_layer_lr=input_layer_lr,
            image_encoder_lr=image_encoder_lr,
        )

        self.optimizer = optim_cls(
            optim_group,
            lr=self.lr,
            betas=self.optim_cfg.get("betas", (0.9, 0.999)),
            eps=self.optim_cfg.get("eps", 1e-8),
        )

        def make_lr_func(base_lr, min_lr_ratio=None):
            """Create a cosine-warmup LR lambda for a specific base learning rate."""
            if min_lr_ratio is None:
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
                    assert 0 <= decay_ratio <= 1
                    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                    ret = actual_min_lr + coeff * (base_lr - actual_min_lr)
                factor = ret / base_lr
                return factor

            return lr_func

        lr_funcs = []
        for group in optim_group:
            group_base_lr = group.get("lr", self.lr)
            lr_funcs.append(make_lr_func(group_base_lr))

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
    # Rollout / evaluation
    # ------------------------------------------------------------------

    def rollout(self, eval_func):
        """Run a policy rollout using the EMA model (or main model if EMA disabled).

        Args:
            eval_func: Callable accepting ``(get_action_fn, act_horizon,
                temporal_agg=...)`` that executes environment rollout.

        Returns:
            The return value of ``eval_func``.
        """
        is_training = self.training
        self.eval()

        if self.use_ema:
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            eval_agent = self.ema_model
        else:
            eval_agent = self.model
        eval_agent.eval()

        ret = eval_func(
            eval_agent.get_action,
            eval_agent.act_horizon,
            temporal_agg=eval_agent.temporal_agg,
        )
        if is_training:
            self.train()
        return ret

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Dispatch visualization to registered domain-specific visualizers.

        Args:
            batch_idx: Index of the current batch.
            batch: Batch data dictionary.
            step_output: Output dict from the training/validation step.
            phase: ``"train"`` or ``"val"``.
            **kwargs: Additional keyword arguments forwarded to visualizers.
        """
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
