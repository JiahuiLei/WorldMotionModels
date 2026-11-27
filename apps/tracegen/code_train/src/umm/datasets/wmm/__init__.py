"""UMM wmm dataset module — fork of umm_v13 for the new MS3 pipeline."""

# Core exports. Tolerate Ms3StateDataset *not yet existing* (optional module).
# Narrow to ModuleNotFoundError so that a broken-but-present ms3_state_dataset.py
# (e.g. SyntaxError, NameError, or a transitive ImportError from torch/etc.)
# propagates instead of being silenced into a None sentinel.
try:
    from umm.datasets.wmm.ms3_state_dataset import Ms3StateDataset  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when the module is missing
    if exc.name not in ("umm.datasets.wmm.ms3_state_dataset",):
        raise  # a transitive dep is missing; do NOT silence that
    Ms3StateDataset = None  # populated when the module exists

from umm.datasets.wmm.humoto_dataset import HumotoDataset  # pyright: ignore[reportMissingImports]

# OMOMO dataset (CHOIS-aligned protocol)
from umm.datasets.wmm.omomo_dataset import OmomoDataset  # noqa: E402

# TG (Tracegen) dataset
from umm.datasets.wmm.tg_dataset import TracegenDataset  # noqa: E402

__all__ = ["Ms3StateDataset", "HumotoDataset", "OmomoDataset", "TracegenDataset"]
