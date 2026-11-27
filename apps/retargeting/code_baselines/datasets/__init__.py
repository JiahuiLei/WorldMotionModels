import importlib
import os, os.path as osp
import importlib.util
import sys


def get_data_module(cfg):
    spec = importlib.util.spec_from_file_location(
        "PLData", osp.join(osp.abspath(osp.dirname(__file__)), cfg.dataset.name, "datamodule.py")
    )
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    dmclass = getattr(foo, "PLData")
    return dmclass(cfg)
