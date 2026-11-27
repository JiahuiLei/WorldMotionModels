"""Training utilities for the UMM (Universal Motion Machine) project.

This package provides configuration parsing, logging setup, training
initialization, and PyTorch Lightning trainer construction. It exposes four
main entry points used by the top-level ``train.py`` script:

- ``get_config``: Parse CLI arguments and YAML config files into a unified
  ``Config`` object.
- ``init_setup``: Set precision, random seeds, and other global state.
- ``get_logger``: Create a ``CustomLogger`` (TensorBoard + optional WandB).
- ``get_trainer``: Build a fully-configured ``pytorch_lightning.Trainer``.

Example:
    >>> from umm.training_utils import get_config, init_setup, get_logger, get_trainer
    >>> cfg = get_config()
    >>> init_setup(cfg)
    >>> logger = get_logger(cfg)
    >>> trainer = get_trainer(cfg, logger)
"""

from .config_helpers import get_config
from .setup_helpers import init_setup
from .log_helpers import get_logger
from .trainer_helper import get_trainer
