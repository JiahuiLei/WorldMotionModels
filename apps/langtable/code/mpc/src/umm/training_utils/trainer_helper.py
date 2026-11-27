"""PyTorch Lightning ``Trainer`` factory.

Assembles all callbacks (checkpointing, printing, visualization, learning-rate
monitoring, model summary) and configures the ``Trainer`` from a unified
``Config`` object and a ``CustomLogger`` instance.

Example:
    >>> from umm.training_utils import get_config, get_logger, get_trainer
    >>> cfg = get_config()
    >>> logger = get_logger(cfg)
    >>> trainer = get_trainer(cfg, logger)
"""

import torch
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    # DeviceStatsMonitor,
    # ModelCheckpoint,
    LearningRateMonitor,
    OnExceptionCheckpoint,
    RichProgressBar,
    ProgressBar,
)
import os.path as osp
import sys
from .callbacks import PrintingCallback, VizCallback, RichModelSummaryLoggingCallback
from .callbacks.model_checkpoint_verbose import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from umm.sims.robot_sim_callbacks import get_eval_callback


def _resolve_precision(cfg_precision):
    """Resolve config precision to PL Trainer precision string.

    Default: bf16-mixed (better dynamic range than fp16).
    Falls back to 16-mixed (fp16) if bf16 is not supported, with a warning.
    """
    if cfg_precision != 16:
        return 32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logging.info("Using bf16-mixed precision (default for precision=16)")
        return "bf16-mixed"
    else:
        logging.warning(
            "bf16 not supported on this GPU — falling back to fp16 (16-mixed). "
            "fp16 has limited dynamic range and may cause NaN/overflow after "
            "long training. Consider using a GPU with bf16 support."
        )
        return "16-mixed"


def get_trainer(cfg, logger, use_distributed_sampler=True):
    """Build and return a fully-configured ``pytorch_lightning.Trainer``.

    Constructs all standard callbacks (model checkpoint, printing, viz,
    learning-rate monitor, exception checkpoint, model summary) and
    instantiates the ``Trainer`` with settings derived from *cfg*.

    Args:
        cfg: A ``Config`` (dict-like) object containing training hyper-
            parameters. Key fields include ``max_steps``, ``max_epochs``,
            ``precision``, ``accumulate_grad_batches``, ``gradient_clip_val``,
            ``ckpt_monitor``, ``ckpt_every_n_train_steps``, and many more
            (see ``config_helpers.config_parser`` for the full list).
        logger: A ``CustomLogger`` instance (TensorBoard + optional WandB)
            whose ``log_dir`` determines checkpoint save paths.
        use_distributed_sampler: Whether the ``Trainer`` should attach its own
            distributed sampler to DataLoaders. Set to ``False`` when the
            DataModule provides its own. Defaults to ``True``.

    Returns:
        pytorch_lightning.Trainer: A ready-to-use trainer.

    Example:
        >>> trainer = get_trainer(cfg, logger)
        >>> trainer.fit(model, datamodule=dm)
    """

    if cfg.get("xcloud", False) and False:  # ! because have async upload
        from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

        # use dataflux checkpoint plugin
        dataflux_ckpt = DatafluxLightningCheckpoint(project_name="xcloud-gcp-prod")
        plugins = [dataflux_ckpt]
        # model_ckpt_callback_dir = osp.join(logger.log_dir, "checkpoints")
        # assert model_ckpt_callback_dir.startswith("/gcs/")
        # model_ckpt_callback_dir = "gs://" + model_ckpt_callback_dir[5:]
        model_ckpt_callback_dir = (
            f"gs://{logger.gcs_bucket_name}/{logger.gcs_log_dir}/checkpoints"
        )

    else:
        plugins = []
        model_ckpt_callback_dir = osp.join(logger.log_dir, "checkpoints")
    logging.warning(f"model_ckpt_callback_dir: {model_ckpt_callback_dir}")

    # assert use_distributed_sampler, "must use distributed sampler"
    model_callback = ModelCheckpoint(
        dirpath=model_ckpt_callback_dir,
        filename="{epoch:03d}_{step:010d}",
        save_last=True,
        verbose=True,
        monitor=cfg.ckpt_monitor,
        mode=cfg.ckpt_monitor_mode,
        save_top_k=cfg.ckpt_save_top_k,
        every_n_train_steps=cfg.ckpt_every_n_train_steps,
        every_n_epochs=cfg.ckpt_every_n_epochs,
        save_on_train_epoch_end=cfg.ckpt_save_on_train_epoch_end,
        ckpt_warmup_steps=getattr(cfg, "ckpt_warmup_steps", 0),
    )
    # progbar_callback = RichProgressBar(
    #     refresh_rate=cfg.progress_bar_refresh_rate,
    #     leave=True,
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="yellow1",
    #         batch_progress="gree_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #     ),
    # )

    if cfg.get("do_profile", False):
        from pytorch_lightning.profilers import AdvancedProfiler

        profiler = AdvancedProfiler(dirpath=logger.log_dir, filename="perf_logs")
    else:
        profiler = None

    logging.info(f"Trainer accumulate_grad_batches: {cfg.accumulate_grad_batches}")

    callbacks = [
        # progbar_callback,
        PrintingCallback(cfg),
        VizCallback(cfg),
        # DeviceStatsMonitor(),
        model_callback,
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        OnExceptionCheckpoint(
            dirpath=osp.join(logger.log_dir, "checkpoints"), filename="last"
        ),
        RichModelSummaryLoggingCallback(max_depth=1),
    ]
    if cfg.get("model_summary_detailed", True):
        callbacks.append(
            RichModelSummaryLoggingCallback(
                max_depth=cfg.get("model_summary_detailed_level", -1)
            )
        )
    callbacks.extend(get_eval_callback(cfg, osp.join(logger.log_dir, "rollout_videos")))

    trainer = Trainer(
        max_steps=cfg.max_steps,
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.get("fast_dev_run", 0),
        logger=logger,
        use_distributed_sampler=use_distributed_sampler,  # some datamodule has its own distributed sampler, when defining them, set this attr to False
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=_resolve_precision(cfg.precision),
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,  # ,  check whether device slow down
        plugins=plugins,
        detect_anomaly=cfg.enable_anomaly,
        # benchmark=True,
        val_check_interval=cfg.val_every_n_iter,
        check_val_every_n_epoch=cfg.val_every_n_epoch,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        # deterministic=True, # some pytorch native function, like cumsum does not support this now...
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        devices=cfg.get("num_gpu_per_node", torch.cuda.device_count()),
        strategy=(
            "ddp_find_unused_parameters_true"
            if cfg.get("ddp_find_unused_parameters", False)
            else "auto"
        ),
        profiler=profiler,
        #
        # compile=True,
    )
    return trainer
