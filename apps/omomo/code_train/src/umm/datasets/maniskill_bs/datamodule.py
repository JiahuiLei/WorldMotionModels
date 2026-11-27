"""LightningDataModule for the maniskill_bs BC mirror.

Registry contract: ``src/umm/datasets/__init__.py::get_data_module`` loads
``datamodule.py::PLData``.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from umm.datasets.maniskill_bs.dataset import ManiskillBSDataset


class PLData(pl.LightningDataModule):
    """Train/val datamodule with a deterministic 90/10 step-level split.

    Config keys read:
        - ``cfg.dataset.data_root`` (str): path to
          ``data/ms3/umm_format/<TASK>/<N>``.
        - ``cfg.dataset.task`` (str): task id (e.g. ``"PushCube-v1"``).
        - ``cfg.train_batchsize`` (int): train DataLoader batch size (1024 per bc.py:54).
        - ``cfg.eval_batchsize`` (int): val DataLoader batch size.
        - ``cfg.workers`` (int, default 0): num DataLoader workers. 0 for smoke
          runs to avoid multiprocessing issues.

    The split is reproducible: step indices are shuffled with a fixed seed
    (``42``), the first 90% go to train, the last 10% to val. No leakage
    between splits.
    """

    _SPLIT_SEED: int = 42
    _VAL_FRACTION: float = 0.10

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._train_dataset: Optional[ManiskillBSDataset] = None
        self._val_dataset: Optional[ManiskillBSDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build train/val datasets with a deterministic step-level split.

        Loads the full dataset ONCE (paying the npz I/O and RAM cost a single
        time), then constructs two lightweight views that share the underlying
        episode list via ``share_split_view``. Scales to 10k+ demos without
        triple-loading.

        Args:
            stage: Lightning stage ("fit", "validate", "test", "predict", or
                None). All stages build the same splits.
        """
        data_root = str(self.cfg.dataset.data_root)
        task = str(self.cfg.dataset.task)

        full = ManiskillBSDataset(data_root=data_root, task=task, split_indices=None)
        n_total = len(full)

        rng = np.random.default_rng(self._SPLIT_SEED)
        perm = rng.permutation(n_total)
        n_val = max(1, int(round(n_total * self._VAL_FRACTION)))
        train_idx = sorted(perm[:-n_val].tolist())
        val_idx = sorted(perm[-n_val:].tolist())

        self._train_dataset = full.share_split_view(train_idx)
        self._val_dataset = full.share_split_view(val_idx)

    def _workers(self) -> int:
        return int(self.cfg.get("workers", 0))

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "setup() must be called before train_dataloader"
        nw = self._workers()
        return DataLoader(
            self._train_dataset,
            batch_size=int(self.cfg.train_batchsize),
            shuffle=True,
            drop_last=True,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "setup() must be called before val_dataloader"
        nw = self._workers()
        # eval_batchsize falls back to train_batchsize if missing.
        bs = int(self.cfg.get("eval_batchsize", self.cfg.train_batchsize))
        return DataLoader(
            self._val_dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )
