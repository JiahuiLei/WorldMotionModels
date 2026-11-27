"""PyTorch Lightning training module for the Universal Motion Machine.

Wraps the :class:`Agent` diffusion model in a ``pl.LightningModule`` that
handles training/validation loops, EMA weight management, optimizer
configuration (AdamW with cosine-warmup LR), gradient monitoring, checkpoint
save/load, and visualization dispatch.

Example::

    from umm.models.umm_v1.model import PLModel
    model = PLModel(cfg)
    trainer.fit(model, datamodule)
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

import json
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.utilities import rank_zero_only

from umm.models.umm_v1.agent import Agent
from umm.models.umm_v1.viz_helpers.behavior_viz import BehaviorVisualizer
from umm.models.umm_v1.viz_helpers.human_viz import HumanVisualizer
from umm.models.umm_v1.viz_helpers.langtable_viz import LangTableVisualizer
from umm.models.umm_v1.viz_helpers.maniskill_viz import ManiskillVisualizer
from umm.models.umm_v1.world_models.wm_utils import disable_model_grad


def sync_model_buffers(
    source_model: torch.nn.Module, target_model: torch.nn.Module
) -> None:
    """Copy all buffers (e.g. BatchNorm running stats) from source to target.

    EMA only tracks parameters; buffers must be synced separately.

    Args:
        source_model: Model to copy buffers from.
        target_model: Model to copy buffers to.

    Example::

        sync_model_buffers(model, ema_model)
    """
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source_model.buffers(), target_model.buffers()):
            buf_tgt.copy_(buf_src)


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for the Universal Motion Machine.

    Wraps the :class:`Agent` diffusion model and handles:
    - Training/validation forward passes with metric logging.
    - EMA (Exponential Moving Average) weight management.
    - Optimizer configuration (AdamW/Adam/SGD with cosine-warmup LR schedule).
    - Gradient norm monitoring and clipping diagnostics.
    - Checkpoint save/load with ctx_record_buffer and EMA state handling.
    - Visualization dispatch to domain-specific visualizers.

    Args:
        cfg: Full configuration dict with ``model``, ``optim``, ``visualize``
            sub-dicts. If None, initializes with empty config.

    Example::

        model = PLModel(cfg)
        trainer = pl.Trainer(...)
        trainer.fit(model, datamodule)
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """Initialize PLModel with Agent, EMA, and domain visualizers.

        Args:
            cfg: Configuration dictionary. If None, uses empty dict defaults.
        """
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg, self.model_cfg = cfg, cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        self.model: Agent = self.get_agent(
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
                power=0.75,  # actually this is not used because use_ema_warmup is False
                decay=self.model_cfg.get("ema_decay", 0.9999),
            )
            self.ema_model = disable_model_grad(Agent(cfg))
            # CRITICAL: Copy both parameters AND buffers (e.g., BatchNorm running stats)!
            # EMA only tracks parameters, not buffers, so we must sync buffers manually
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

        try:
            from umm.models.umm_v1.viz_helpers.taco_viz import TacoVisualizer

            self.taco_visualizer = TacoVisualizer(self, self.model_cfg.get("taco", {}))
        except (ImportError, ModuleNotFoundError) as e:
            logging.warning(f"TacoVisualizer not found: {e}")
            self.taco_visualizer = None
        self.human_visualizer = HumanVisualizer(
            self,
            self.model_cfg.get("human", {}),
            self.model_cfg.get("humanoid", {}),
        )

        try:
            from umm.models.umm_v1.viz_helpers.manipulator_viz import (
                ManipulatorVisualizer,
            )

            self.manipulator_visualizer = ManipulatorVisualizer(
                self, self.model_cfg.get("manipulator", {})
            )
        except (ImportError, ModuleNotFoundError) as e:
            logging.warning(f"ManipulatorVisualizer not found: {e}")
            self.manipulator_visualizer = None

        try:
            from umm.models.umm_v1.viz_helpers.maniptrans_viz import (
                ManipTransVisualizer,
            )

            self.maniptrans_visualizer = ManipTransVisualizer(
                self, self.model_cfg.get("maniptrans", {})
            )
        except (ImportError, ModuleNotFoundError) as e:
            logging.warning(f"ManipTransVisualizer not found: {e}")
            self.maniptrans_visualizer = None

        self.langtable_visualizer = LangTableVisualizer(
            self, self.model_cfg.get("langtable", {})
        )
        self.maniskill_visualizer = ManiskillVisualizer(
            self, self.model_cfg.get("maniskill", {})
        )
        self.behavior_visualizer = BehaviorVisualizer(
            self, self.cfg.get("visualize", {}).get("behavior", {})
        )
        self.color_plate = cm.get_cmap("Set3").colors[:10]
        return

    @staticmethod
    def get_agent(
        cfg,
        # init
        init_fn=None,
        ckpt_prefix="model.",
        strict_flag=True,
        lmm_flag=True,
        input_flag=False,
        compile_flag=False,
    ):
        """Create and optionally initialize an Agent from a checkpoint.

        Builds an ``Agent`` instance, optionally compiles it with
        ``torch.compile``, and loads pretrained weights from ``init_fn``
        (if provided), handling both compiled and non-compiled checkpoint
        key prefixes.

        Args:
            cfg: Full configuration dictionary.
            init_fn: Path to a checkpoint file to initialize from.
            ckpt_prefix: Key prefix to strip from checkpoint state dict.
            strict_flag: Whether to enforce strict key matching on load.
            lmm_flag: Whether to load the LMM (world model) weights.
            input_flag: Whether to load the input layer weights.
            compile_flag: Whether to ``torch.compile`` the model.

        Returns:
            Initialized ``Agent`` instance.

        Example::

            agent = PLModel.get_agent(cfg, init_fn="ckpt.pt")
        """
        model: Agent = Agent(cfg)

        ###########################################
        ###########################################
        if compile_flag:
            logging.warning("@" * 40)
            logging.warning("Compiling model...")
            model = torch.compile(model)
            logging.warning("Model compiled.")
        ###########################################
        ###########################################

        model_cfg = cfg.get("model", {})
        if init_fn is None:
            init_fn = model_cfg.get("init_fn", None)
        if init_fn is not None:
            checkpoint_dict = Agent.filter_ckpt_prefix(
                torch.load(init_fn)["state_dict"], ckpt_prefix
            )

            ###########################################
            ###########################################
            # ! Handle the compiled model checkpoint prefix
            new_checkpoint_dict = {}
            for k, v in checkpoint_dict.items():
                if k.startswith("_orig_mod."):
                    new_checkpoint_dict[k[len("_orig_mod.") :]] = v
                else:
                    new_checkpoint_dict[k] = v
            checkpoint_dict = new_checkpoint_dict
            ###########################################
            ###########################################

            model.load_from_checkpoint(
                checkpoint_dict=checkpoint_dict,
                lmm_flag=lmm_flag,
                input_flag=input_flag,
                strict=strict_flag,
            )
            logging.info(
                f"Loaded checkpoint from {init_fn} Strict={strict_flag} LMM={lmm_flag} Input={input_flag}, ckpt_prefix={ckpt_prefix}"
            )
            if hasattr(model, "ctx_record_buffer"):
                # Handle both compiled (_orig_mod.ctx_record_buffer) and non-compiled (ctx_record_buffer) cases
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

    @rank_zero_only
    def on_train_start(self):
        """Log full training config to wandb at the start of training."""
        from umm.training_utils.log_helpers import CustomLogger  # deferred: avoid circular import
        if isinstance(self.logger, CustomLogger) and hasattr(self.logger, "wandb_logger"):
            cfg_plain = json.loads(json.dumps(self.cfg, default=str))
            self.logger.wandb_logger.experiment.config.update(
                cfg_plain, allow_val_change=True
            )
            logging.info("Logged full config to wandb")

    def training_step(self, batch, batch_idx):
        """Execute a single training step through the Agent.

        Args:
            batch: Batch data dict with keys ``trace``, ``context``,
                ``trace_mask``, etc.
            batch_idx: Index of this batch in the current epoch.

        Returns:
            Dict containing ``"loss"`` and ``"log_dict"`` with all
            logged training metrics.

        Example::

            ret = model.training_step(batch, batch_idx=0)
        """

        self.train()
        ret = self.model(batch, "train", global_step=self.global_step)
        # Log all metrics
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
            batch: Batch data dict with keys ``trace``, ``context``,
                ``trace_mask``, etc.
            batch_idx: Index of this batch.
            dataloader_idx: Index of the active validation dataloader.

        Returns:
            Dict containing ``"loss"`` and ``"log_dict"`` with all
            logged validation metrics.

        Example::

            ret = model.validation_step(batch, batch_idx=0)
        """

        self.eval()
        ret = self.model(batch, "val", global_step=self.global_step)
        # Log all metrics
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

        Called by Lightning after gradient clipping but before the
        optimizer step. Computes the total gradient norm (L2 or max-abs),
        logs it to ``monitoring/global_grad_norm``, and optionally logs
        per-parameter gradient norms when ``debug_logging`` is enabled.

        Args:
            optimizer: The optimizer about to step.

        Example::

            # Called automatically by the Lightning Trainer.
            model.on_before_optimizer_step(optimizer)
        """
        # Get gradient clipping config from trainer
        extra_log_flag = self.cfg.get("debug_logging", False)
        start_time = time.time()
        grad_clip_val = self.trainer.gradient_clip_val
        grad_clip_algorithm = self.trainer.gradient_clip_algorithm

        if grad_clip_val is not None and grad_clip_val > 0:
            # Compute current gradient norm
            parameters = [p for p in self.model.parameters() if p.grad is not None]

            if len(parameters) > 0:
                device = parameters[0].device

                if grad_clip_algorithm == "norm":
                    # Compute total gradient norm (L2 norm)
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
                    # Compute max absolute gradient value
                    total_norm = max(
                        [p.grad.detach().abs().max().item() for p in parameters]
                    )
                else:
                    total_norm = 0.0

                # Log tensor directly — no CPU sync needed
                self.log(
                    "monitoring/global_grad_norm",
                    total_norm,
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                    prog_bar=False,
                )

                # Single .item() for all CPU-side checks (debug log + clip warning)
                total_norm_val = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

                logging.debug(
                    f"Grad Norm at step {self.global_step}: {total_norm_val:.4f} / max={grad_clip_val:.4f}"
                )

                if total_norm_val > grad_clip_val:
                    logging.warning("@" * 40)
                    logging.warning(
                        f"Gradient clipping occurred at step {self.global_step}: "
                        f"grad_norm={total_norm_val:.4f} > clip_val={grad_clip_val:.4f} "
                        f"(algorithm={grad_clip_algorithm})"
                    )
                    logging.warning("@" * 40)

                if extra_log_flag:
                    # Log only parameters with large gradient norms for debugging
                    # Use threshold to filter out small/zero gradients
                    significant_threshold = grad_clip_val * 0.1  # 10% of clip value
                    param_grad_info = []
                    log_grad_info = []
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_norm = torch.norm(p.grad.detach(), 2.0).item()
                            log_grad_info.append((name, grad_norm, p.grad.shape))
                            if grad_norm > significant_threshold:
                                param_grad_info.append((name, grad_norm, p.grad.shape))

                    # Sort by gradient norm (largest first)
                    param_grad_info.sort(key=lambda x: x[1], reverse=True)
                    log_grad_info.sort(key=lambda x: x[1], reverse=True)

                    # Log all parameter gradient norms to debug namespace in a single call
                    grad_norm_dict = {
                        f"debug/grad_norm/{name.replace('.', '/')}": grad_norm
                        for name, grad_norm, shape in log_grad_info
                    }
                    self.log_dict(
                        grad_norm_dict,
                        on_step=True,
                        on_epoch=False,
                        batch_size=1,
                        prog_bar=False,
                    )

                    if total_norm_val > grad_clip_val:
                        logging.warning("@" * 40)
                        logging.warning(
                            f"Gradient clipping occurred at step {self.global_step}: "
                            f"grad_norm={total_norm_val:.4f} > clip_val={grad_clip_val:.4f} "
                            f"(algorithm={grad_clip_algorithm})"
                        )
                        logging.warning(
                            f"  Parameters with grad_norm > {significant_threshold:.4f} "
                            f"({len(param_grad_info)} out of {len(list(self.model.parameters()))} total):"
                        )
                        for i, (name, grad_norm, shape) in enumerate(param_grad_info):
                            logging.warning(
                                f"    {i+1}. {name} (shape={shape}): {grad_norm:.4f}"
                            )
                        logging.warning("@" * 40)
        logging.debug(
            f"Time taken for gradient checking: {time.time() - start_time:.4f} seconds"
        )
        return

    def optimizer_step(self, *args, **kwargs):
        """Perform optimizer step and update EMA weights.

        Calls the parent optimizer step, then updates the EMA shadow
        parameters and syncs the ``ctx_record_buffer`` from the main
        model to the EMA model.

        Args:
            *args: Positional arguments forwarded to ``super().optimizer_step``.
            **kwargs: Keyword arguments forwarded to ``super().optimizer_step``.

        Example::

            # Called automatically by the Lightning Trainer.
            model.optimizer_step(optimizer=optimizer)
        """
        # Call super() FIRST to ensure this runs before all callbacks
        # Callbacks are executed after the LightningModule's optimizer_step
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            # Ensure EMA parameters are on the same device as the model parameters
            model_device = next(self.model.parameters()).device
            if len(self.ema.shadow_params) > 0:
                if self.ema.shadow_params[0].device != model_device:
                    self.ema.to(model_device)
            ema_model_device = next(self.ema_model.parameters()).device
            if ema_model_device != model_device:
                self.ema_model.to(model_device)

            # Update EMA parameters
            self.ema.step(self.model.parameters())
            if hasattr(self.model, "ctx_record_buffer"):
                # the ema always follow the main model
                self.ema_model.ctx_record_buffer = self.model.ctx_record_buffer
        return

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state dict alongside the standard checkpoint.

        Args:
            checkpoint: Checkpoint dictionary to augment.

        Example::

            # Called automatically by Lightning during checkpointing.
            model.on_save_checkpoint(checkpoint)
        """
        if self.use_ema:
            # Save the EMA state dict to the checkpoint
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA state, ctx_record_buffer, and handle key mismatches.

        Fills in missing ``human_model.*`` parameters from the current
        model, extracts ``ctx_record_buffer`` before standard loading to
        avoid compiled/non-compiled key conflicts, restores EMA weights,
        and syncs buffers.

        Args:
            checkpoint: Checkpoint dictionary loaded from disk.

        Example::

            # Called automatically by Lightning during checkpoint loading.
            model.on_load_checkpoint(checkpoint)
        """
        # Fill in missing human_model.* parameters from current model if they're missing from checkpoint
        if "state_dict" in checkpoint:
            current_state = self.state_dict()
            for key in current_state:
                if (
                    key.startswith("human_model.")
                    and key not in checkpoint["state_dict"]
                ):
                    checkpoint["state_dict"][key] = current_state[key]

        # Extract ctx_record_buffer before super().on_load_checkpoint() to avoid key mismatch issues
        # between compiled (model._orig_mod.ctx_record_buffer) and non-compiled (model.ctx_record_buffer) keys
        ctx_buffer_data = None
        ctx_buffer_keys_to_remove = []
        if "state_dict" in checkpoint:
            for key in list(checkpoint["state_dict"].keys()):
                if key.endswith("ctx_record_buffer") and (
                    key.startswith("model.") or key.startswith("ema_model.")
                ):
                    if key.startswith("model.") and ctx_buffer_data is None:
                        # Save the first model ctx_record_buffer we find
                        ctx_buffer_data = checkpoint["state_dict"][key]
                        logging.info(
                            f"Extracted ctx_record_buffer from checkpoint key: {key}"
                        )
                    # Mark all ctx_record_buffer keys for removal to avoid loading conflicts
                    ctx_buffer_keys_to_remove.append(key)

            # Remove ctx_record_buffer keys from state_dict before loading
            for key in ctx_buffer_keys_to_remove:
                del checkpoint["state_dict"][key]

        super().on_load_checkpoint(checkpoint)

        # Manually restore ctx_record_buffer after super().on_load_checkpoint()
        if self.model.ctx_record_flag:
            if ctx_buffer_data is not None:
                self.model.ctx_record_buffer = ctx_buffer_data
                logging.info(
                    f"Loaded ctx_record_buffer with {len(ctx_buffer_data)} entries"
                )
            else:
                logging.warning(
                    "ctx_record_buffer not found in checkpoint, initializing with empty buffer"
                )
        if self.use_ema and "ema_state_dict" in checkpoint:
            # Fill in missing human_model.* parameters in EMA state dict as well
            current_ema_state = self.ema.state_dict()
            for key in current_ema_state:
                if (
                    key.startswith("shadow_params")
                    and key not in checkpoint["ema_state_dict"]
                ):
                    checkpoint["ema_state_dict"][key] = current_ema_state[key]

            # Restore the EMA state from the checkpoint
            # Try to load EMA state dict
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

            # Copy EMA weights to the ema_model for evaluation
            # This may fail if there are shape mismatches between checkpoint and current model
            self.ema.copy_to(self.ema_model.parameters())
            logging.info("Successfully loaded EMA weights from checkpoint")

            if self.model.ctx_record_flag:
                self.ema_model.ctx_record_buffer = self.model.ctx_record_buffer

    def configure_optimizers(self):
        """Configure optimizer and cosine-warmup LR scheduler.

        Builds an AdamW/Adam/SGD optimizer with per-component learning
        rates (LMM, input layer, image encoder) and a ``LambdaLR``
        scheduler implementing linear warmup followed by cosine decay.

        Returns:
            Dict with ``"optimizer"`` and ``"lr_scheduler"`` entries
            for Lightning.

        Example::

            opt_cfg = model.configure_optimizers()
        """

        # the GPT2 use warmup and then cosine decay
        self.lr = float(self.optim_cfg.get("lr", 6e-4))
        self.min_lr = self.optim_cfg.get("min_lr", 6e-5)
        self.lr_decay_iters = self.optim_cfg.get("lr_decay_iters", 600000)
        self.lr_warmup_iters = self.optim_cfg.get("lr_warmup_iters", 2000)
        self.optimizer_weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        logging.info(f"Optimizer weight decay: {self.optimizer_weight_decay}")

        self.optimizer_type = self.optim_cfg.get("type", "adamw").lower()
        if self.optimizer_type == "adamw":
            logging.warning(f"Use AdamW Optimizer")
            optim_cls = torch.optim.AdamW
        elif self.optimizer_type == "sgd":
            logging.warning(f"Use SGD Optimizer")
            optim_cls = torch.optim.SGD
        elif self.optimizer_type == "adam":
            logging.warning(f"Use Adam Optimizer")
            optim_cls = torch.optim.Adam
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_type} not implemented"
            )

        # Get learning rate configurations for different components
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
            betas=self.optim_cfg.get("betas", (0.9, 0.95)),
            eps=self.optim_cfg.get("eps", 1e-8),
            # fused=True,
        )

        def make_lr_func(base_lr, min_lr_ratio=None):
            """Create a cosine-warmup LR lambda for a specific base learning rate.

            Args:
                base_lr: Peak learning rate for this parameter group.
                min_lr_ratio: Ratio of minimum LR to base LR. If None,
                    uses the global ``min_lr / lr`` ratio.

            Returns:
                A callable ``lr_func(it) -> float`` returning the LR
                multiplier at iteration ``it``.

            Example::

                lr_fn = make_lr_func(6e-4, min_lr_ratio=0.1)
            """
            if min_lr_ratio is None:
                min_lr_ratio = (
                    self.min_lr / self.lr
                )  # Use same ratio as global settings
            actual_min_lr = base_lr * min_lr_ratio

            def lr_func(it):
                """Compute learning rate multiplier at iteration *it*.

                Args:
                    it: Current training iteration.

                Returns:
                    Learning rate value (not a multiplier — absolute LR).
                """
                if it < self.lr_warmup_iters:
                    ret = base_lr * (it + 1) / (self.lr_warmup_iters + 1.0)
                elif it > self.lr_decay_iters:
                    ret = float(actual_min_lr)
                else:
                    decay_ratio = (it - self.lr_warmup_iters) / (
                        self.lr_decay_iters - self.lr_warmup_iters
                    )
                    assert 0 <= decay_ratio <= 1
                    coeff = 0.5 * (
                        1.0 + np.cos(np.pi * decay_ratio)
                    )  # coeff ranges 0..1
                    ret = actual_min_lr + coeff * (base_lr - actual_min_lr)
                factor = ret / base_lr
                return factor

            return lr_func

        # Create learning rate functions for each parameter group
        lr_funcs = []
        for group in optim_group:
            group_base_lr = group.get("lr", self.lr)  # Use group-specific lr or default
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

    def rollout(self, eval_func):
        """Run a policy rollout using the EMA model (or main model if EMA disabled).

        Switches to eval mode, syncs EMA weights, calls ``eval_func``
        with the agent's ``get_action`` method, then restores the
        original training state.

        Args:
            eval_func: Callable accepting ``(get_action_fn, act_horizon,
                temporal_agg=...)`` that executes environment rollout.

        Returns:
            The return value of ``eval_func``.

        Example::

            results = pl_model.rollout(sim_callback.evaluate)
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

    ####################################################################
    # viz routines
    ####################################################################

    @torch.inference_mode()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Dispatch visualization to all registered domain-specific visualizers.

        Switches to eval mode, syncs EMA weights, then sequentially
        invokes each domain visualizer (langtable, behavior, manipulator,
        human/humanoid, taco, maniptrans, maniskill) if enabled.

        Args:
            batch_idx: Index of the current batch.
            batch: Batch data dictionary.
            step_output: Output dict from the training/validation step.
            phase: ``"train"`` or ``"val"``.
            **kwargs: Additional keyword arguments forwarded to visualizers.

        Example::

            pl_model.viz(batch_idx=0, batch=batch, step_output=out, phase="val")
        """
        torch.cuda.empty_cache()
        is_training = self.training
        self.eval()
        logging.warning(f"Start viz batchidx={batch_idx} in phase={phase} ...")
        # print(batch["seq_id"], batch["rgb255"].shape[1], batch["prompt_type"].shape)
        start_time = time.time()

        if self.use_ema:
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            eval_model = self.ema_model
        else:
            eval_model = self.model
        eval_model.eval()

        if self.langtable_visualizer is not None and self.langtable_visualizer.flag:
            self.langtable_visualizer.langtable_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        if self.behavior_visualizer is not None and self.behavior_visualizer.flag:
            self.behavior_visualizer.behavior_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        if self.manipulator_visualizer is not None and self.manipulator_visualizer.flag:
            self.manipulator_visualizer.manipulator_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        if self.human_visualizer is not None and (
            self.human_visualizer.human_flag or self.human_visualizer.humanoid_flag
        ):
            self.human_visualizer.human_humanoid_viz_routine(
                step_output, batch, phase, batch_idx, eval_model
            )

        if self.taco_visualizer is not None and self.taco_visualizer.flag:
            self.taco_visualizer.taco_viz_routine(batch, phase, batch_idx, eval_model)

        if self.maniptrans_visualizer is not None and self.maniptrans_visualizer.flag:
            self.maniptrans_visualizer.maniptrans_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        if self.maniskill_visualizer is not None and self.maniskill_visualizer.flag:
            self.maniskill_visualizer.maniskill_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        return
