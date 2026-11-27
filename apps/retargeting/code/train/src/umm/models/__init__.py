"""Model registry for the Universal Motion Machine.

Dynamically loads a ``PLModel`` class from a versioned sub-package
(e.g. ``umm_v1/model.py``) based on the ``cfg.model.name`` field.

Example::

    from umm.models import get_model
    model = get_model(cfg)  # cfg.model.name == "umm_v1"
"""

import importlib
import os, os.path as osp
import importlib.util
import sys


def get_model(cfg):
    """Instantiate a PLModel from the versioned sub-package specified by config.

    Dynamically imports ``<models_dir>/<cfg.model.name>/model.py`` and
    returns an instance of the ``PLModel`` class defined therein.

    Args:
        cfg: Configuration namespace/dict with a ``model.name`` attribute
            identifying the model sub-package (e.g. ``"umm_v1"``).

    Returns:
        An instantiated ``PLModel`` for the requested model version.

    Example::

        from umm.models import get_model
        model = get_model(cfg)
    """
    spec = importlib.util.spec_from_file_location(
        "PLModel",
        osp.join(osp.abspath(osp.dirname(__file__)), cfg.model.name, "model.py"),
    )
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    model_class = getattr(foo, "PLModel")
    return model_class(cfg)
