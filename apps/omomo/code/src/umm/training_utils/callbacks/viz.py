"""Visualization callback for PyTorch Lightning.

Provides VizCallback that periodically calls pl_module.viz() during
training, validation, and test loops at configurable step intervals.

Example:
    >>> callback = VizCallback(cfg)
    >>> trainer = Trainer(callbacks=[callback])
"""

# provide batch step end viz callback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging
from pytorch_lightning.utilities import rank_zero_only


class VizCallback(Callback):
    """Callback that invokes pl_module.viz() at configurable intervals.

    Triggers visualization during training (by global step), validation
    (by batch index), and test (by batch index).  When viz_try is set,
    exceptions in the viz function are caught and logged as warnings instead
    of crashing the run.

    Args:
        cfg: Experiment Config with viz_every_n_train_steps,
            viz_every_n_val_steps, viz_every_n_test_steps, and
            viz_try.

    Example:
        >>> cb = VizCallback(cfg)
    """

    def __init__(self, cfg) -> None:
        """Initialize the visualization callback.

        Args:
            cfg: Experiment config with viz frequency settings and
                viz_try flag.
        """
        # if set to -1, does not log here
        self.viz_per_n_train_steps = cfg.viz_every_n_train_steps
        self.viz_per_n_val_steps = cfg.viz_every_n_val_steps
        self.viz_per_n_test_steps = cfg.viz_every_n_test_steps
        logging.info(
            f"VizCallback: viz_per_n_train_steps: {self.viz_per_n_train_steps}; viz_per_n_val_steps: {self.viz_per_n_val_steps}, viz_per_n_test_steps: {self.viz_per_n_test_steps}"
        )
        self.viz_try = cfg.viz_try
        logging.info(f"VizCallback: viz_try: {self.viz_try}")

        self.vized_train_steps = (
            []
        )  # ! this is to avoid multiple viz when acc gradient where the same step is called multiple times
        return

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Call pl_module.viz() if the global step matches the viz interval.

        Deduplicates calls when gradient accumulation causes the same global
        step to trigger multiple batch-end hooks.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule (must have a viz method).
            outputs: Training step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
        """
        # trainer.strategy.barrier()
        if self.viz_per_n_train_steps <= 0:
            return
        # ! here training use global step to determine viz, but val and test use batch_idx!!!!
        if (
            pl_module.global_step % self.viz_per_n_train_steps == 0
            and pl_module.global_step > 0
        ):
            if pl_module.global_step in self.vized_train_steps:
                return
            self.vized_train_steps.append(pl_module.global_step)
            logging.info(
                f"Viz train batch {batch_idx} at global step {pl_module.global_step}"
            )

            if hasattr(pl_module, "viz") and callable(pl_module.viz):
                pl_module.freeze()
                if self.viz_try:
                    try:
                        pl_module.viz(
                            batch_idx=batch_idx,
                            batch=batch,
                            step_output=outputs,
                            phase="train",
                        )
                    except Exception as e:
                        logging.warning(
                            f"VizCallback: viz train batch {batch_idx} failed: {e}, continue"
                        )
                else:
                    pl_module.viz(
                        batch_idx=batch_idx,
                        batch=batch,
                        step_output=outputs,
                        phase="train",
                    )
                pl_module.unfreeze()
        return

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Call pl_module.viz() during validation at the configured interval.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
            outputs: Validation step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        if self.viz_per_n_val_steps < 0:
            return
        if (batch_idx + 1) % self.viz_per_n_val_steps == 0:
            logging.info(f"Viz val batch {batch_idx}")
            if hasattr(pl_module, "viz") and callable(pl_module.viz):
                try:
                    pl_module.viz(
                        batch_idx=batch_idx,
                        batch=batch,
                        step_output=outputs,
                        phase="val",
                    )
                except Exception as e:
                    if self.viz_try:
                        logging.warning(
                            f"VizCallback: viz val batch {batch_idx} failed: {e}, continue"
                        )
                    else:
                        raise e
        return

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Call pl_module.viz() during testing at the configured interval.

        Args:
            trainer: The active Trainer.
            pl_module: The LightningModule.
            outputs: Test step outputs.
            batch: The current batch data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        if self.viz_per_n_test_steps < 0:
            return
        if (batch_idx + 1) % self.viz_per_n_test_steps == 0:
            logging.info(f"Viz test batch {batch_idx}")
            if hasattr(pl_module, "viz") and callable(pl_module.viz):
                try:
                    pl_module.viz(
                        batch_idx=batch_idx,
                        batch=batch,
                        step_output=outputs,
                        phase="test",
                    )
                except Exception as e:
                    if self.viz_try:
                        logging.warning(
                            f"VizCallback: viz test batch {batch_idx} failed: {e}, continue"
                        )
                    else:
                        raise e
        return
