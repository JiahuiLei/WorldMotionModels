"""Time-throttled console metric printing callback for PyTorch Lightning.

Logs training and evaluation metrics to the console at configurable
intervals, along with GPU memory statistics.  Also clears the CUDA
cache around validation loops.  Derived from the Lightning Bolts 0.6.0
source.

Example:
    >>> cb = PrintingCallback(cfg)
    >>> trainer = pl.Trainer(callbacks=[cb])
"""

# Lighting bolts is not stable, dev from 0.6.0 source code
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
import time, torch
import logging
from pprint import pformat
import sys
from pytorch_lightning.utilities import rank_zero_only

# TODO: have to check the distributed mode


@rank_zero_only
def print_step(trainer):
    """Log the current training metrics and GPU memory to the console.

    Collects all ``trainer.callback_metrics`` (skipping ``_step``/``_epoch``
    suffixed and debug/lr- prefixed keys), formats them with ``pprint``,
    and logs via the ``logging`` module.

    Args:
        trainer: The PyTorch Lightning ``Trainer`` instance.

    Example:
        >>> print_step(trainer)
    """
    global_iters = trainer.global_step

    # # phase
    # if trainer.running_sanity_check:
    #     phase = "sanity_check"
    # elif trainer.running_validation:
    #     phase = "validation"
    # elif trainer.running_test:
    #     phase = "test"
    # else:
    #     phase = "train"
    # logging.info(f"Phase: {phase}")

    batch_metrics = {}
    for k, v in trainer.callback_metrics.items():
        if k.endswith("_step") or k.endswith("_epoch"):
            # skip, because created when set on_epoch=True
            continue
        if k.startswith("debug") or k.startswith("lr-"):
            # skip debug metrics
            continue
        if isinstance(v, torch.Tensor):
            batch_metrics[k] = float(v.detach().cpu())
        else:
            batch_metrics[k] = float(v)
    logging.info("." * 40)
    logging.info("Global iters: {}".format(global_iters))
    try:
        free, total = torch.cuda.mem_get_info()
        free = float(free) / 1024.0 / 1024.0 / 1024.0
        total = float(total) / 1024.0 / 1024.0 / 1024.0
        logging.info(f"# GPU {free:.2f}GB/{total:.2f}GB free #")
    except (AttributeError, RuntimeError, AssertionError) as e:
        pass  # may not on cuda gpu
    logging.info("Metric:\n{}".format(pformat(batch_metrics, indent=2, compact=True)))
    return


class PrintingCallback(Callback):
    """Lightning callback that prints metrics to the console at timed intervals.

    During training, metrics are printed no more often than every
    ``train_print_freq_seconds`` seconds.  During validation, test, and
    predict phases, printing occurs either at the same interval or every
    step if ``nontrain_print_every_step`` is set.  Additionally clears the
    CUDA cache before and after validation.

    Args:
        cfg: Config dict-like object with optional keys
            ``train_print_freq_seconds`` (float, default 2) and
            ``nontrain_print_every_step`` (bool, default False).

    Example:
        >>> cb = PrintingCallback(cfg)
        >>> trainer = pl.Trainer(callbacks=[cb])
    """

    def __init__(self, cfg) -> None:
        """Initialize PrintingCallback with timing and per-step flags.

        Args:
            cfg: Config dict-like object containing
                ``train_print_freq_seconds`` and ``nontrain_print_every_step``.
        """
        self.train_print_freq_seconds = cfg.get("train_print_freq_seconds", 2)
        logging.info(
            "PrintingCallback: train_print_freq_seconds={}".format(
                self.train_print_freq_seconds
            )
        )
        self.last_print_time = time.time()

        self.nontrain_print_every_step = cfg.get("nontrain_print_every_step", False)
        logging.info(
            f"PrintingCallback: nontrain_print_every_step={self.nontrain_print_every_step}"
        )
        return

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Print metrics if enough time has elapsed since the last print.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule being trained.
            outputs: Step outputs from the training step.
            batch: The current batch data.
            batch_idx: Index of the current batch.
        """
        # logging.warning(f"pl_module: global_step {pl_module.global_step}")
        # logging.warning(f"trainer: global_step {trainer.global_step}")
        if time.time() - self.last_print_time > self.train_print_freq_seconds:
            self.last_print_time = time.time()
            print_step(trainer)
        return

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Print metrics during validation if time threshold is met.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
            outputs: Validation step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        # logging.warning(f"pl_module: global_step {pl_module.global_step}")
        # logging.warning(f"trainer: global_step {trainer.global_step}")
        if (
            time.time() - self.last_print_time > self.train_print_freq_seconds
            or self.nontrain_print_every_step
        ):
            self.last_print_time = time.time()
            print_step(trainer)
        return

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Print metrics during testing if time threshold is met.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
            outputs: Test step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        if (
            time.time() - self.last_print_time > self.train_print_freq_seconds
            or self.nontrain_print_every_step
        ):
            self.last_print_time = time.time()
            print_step(trainer)
        return

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Print metrics during prediction if time threshold is met.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
            outputs: Predict step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        if (
            time.time() - self.last_print_time > self.train_print_freq_seconds
            or self.nontrain_print_every_step
        ):
            self.last_print_time = time.time()
            print_step(trainer)
        return

    # ! also handle the empty cache
    def on_validation_start(self, trainer, pl_module):
        """Empty the CUDA cache before validation to reduce memory fragmentation.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
        """
        logging.info(f"Empty cache before validation")
        torch.cuda.empty_cache()
        return

    def on_validation_end(self, trainer, pl_module):
        """Empty the CUDA cache after validation to reclaim memory.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
        """
        logging.info(f"Empty cache after validation")
        torch.cuda.empty_cache()
        return

    # def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     metrics_dict = copy.copy(trainer.callback_metrics)
    #     self.metrics.append(metrics_dict)
    #     rank_zero_info(dicts_to_table(self.metrics))
