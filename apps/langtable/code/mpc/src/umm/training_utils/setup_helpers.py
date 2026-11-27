"""Global training setup utilities (precision, seeding, determinism).

Called early in the training script to configure PyTorch precision settings
and optionally seed all random number generators for reproducibility.

Example:
    >>> from umm.training_utils.setup_helpers import init_setup
    >>> init_setup(cfg)
"""

import torch
import numpy as np
import random, logging

from pytorch_lightning import seed_everything


def setup_seed(seed: int):
    """Seed all random number generators and enable deterministic cuDNN.

    Sets seeds for ``torch``, ``torch.cuda``, ``numpy``, and ``random``,
    and forces cuDNN into deterministic mode.

    Args:
        seed: The integer seed value. String values are cast to ``int``.

    Example:
        >>> setup_seed(42)
    """
    if isinstance(seed, str):
        seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_setup(cfg):
    """Perform one-time global setup before training starts.

    Configures float32 matrix-multiplication precision (for Tensor Core
    acceleration) and optionally seeds all RNGs via PyTorch Lightning's
    ``seed_everything``.

    Args:
        cfg: A ``Config`` (dict-like) object with at least the following keys:

            - ``precision`` (int): Floating-point precision (16 or 32).
            - ``matmul_precision`` (str): One of ``"highest"``, ``"high"``,
              ``"medium"`` — passed to
              ``torch.set_float32_matmul_precision``.
            - ``init_seed`` (int | None): If not ``None``, seed all RNGs.

    Example:
        >>> from umm.training_utils.config_helpers import Config
        >>> cfg = Config({"precision": 32, "matmul_precision": "highest", "init_seed": 42})
        >>> init_setup(cfg)
    """
    logging.info(f"Precision: {cfg.precision}; Matmul: {cfg.matmul_precision}")
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.init_seed is not None:
        logging.info(f"Init seed: {cfg.init_seed}")
        # seed_everything handles torch, numpy, random, and sets PL_GLOBAL_SEED
        # workers=True ensures proper worker seeding in DataLoaders
        seed_everything(cfg.init_seed, workers=True)

        # Ensure cudnn deterministic behavior for reproducibility
        # This matches previous behavior where setup_seed forced this
        torch.backends.cudnn.deterministic = True

    return
