"""Dataset module for Universal Motion Machine (UMM).

Provides a factory function :func:`get_data_module` that dynamically loads
a PyTorch Lightning ``LightningDataModule`` based on the dataset name in config.

Example::

    from umm.datasets import get_data_module
    dm = get_data_module(cfg)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
"""

import importlib.util
import os.path as osp
import sys

from lib_io import UnifiedIO


def get_data_module(cfg):
    """Dynamically load and instantiate a PyTorch Lightning data module.

    Locates the ``datamodule.py`` file under the subdirectory named by
    ``cfg.dataset.name`` and imports the ``PLData`` class from it.

    Args:
        cfg: Configuration namespace with ``cfg.dataset.name`` pointing to
            the dataset subdirectory (e.g. ``"umm_v1"``).

    Returns:
        An instance of ``PLData`` (a ``LightningDataModule``) initialised
        with the provided config.

    Example::

        dm = get_data_module(cfg)
        dm.setup("fit")
    """
    spec = importlib.util.spec_from_file_location(
        "PLData",
        osp.join(osp.abspath(osp.dirname(__file__)), cfg.dataset.name, "datamodule.py"),
    )
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    dmclass = getattr(foo, "PLData")
    return dmclass(cfg)
