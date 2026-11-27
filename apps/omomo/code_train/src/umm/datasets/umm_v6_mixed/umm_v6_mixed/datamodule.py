"""PyTorch Lightning data module for mixed-domain training (v6_mixed).

Supports multiple datasets (bridge, davis, behavior) in the SAME batch.
Each sample carries a ``dataset_id`` so the model can apply per-dataset
PoseIO normalization.  Images of different spatial resolutions are padded
to the batch-maximum size in the collate function.

Uses :class:`BalancedSampler` with configurable per-dataset weights
(default 1:1:1) regardless of dataset size.
"""

import logging
from typing import Iterator, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from umm.datasets.umm_v6_mixed.base_dataset import BaseDataset, CanonicalDatasetOutput
from umm.datasets.umm_v6_mixed.behavior_v2_dataset import BehaviorV2Dataset
from umm.datasets.umm_v6_mixed.behavior_v2_boosted_dataset import BehaviorV2BoostedDataset
from umm.datasets.umm_v6_mixed.real_dataset import RealDataset


class BalancedSampler(Sampler):
    """Weighted hierarchical sampler for multi-dataset training.

    Performs two-level sampling: first selects a sub-dataset with probability
    proportional to ``weights``, then uniformly samples an index within that
    sub-dataset.

    Args:
        dataset_lengths: Number of samples in each sub-dataset.
        num_samples: Total samples per epoch.  Defaults to sum of all lengths.
        weights: Per-dataset sampling weights (uniform if None).
    """

    def __init__(
        self,
        dataset_lengths: List[int],
        num_samples: Optional[int] = None,
        weights: Optional[List[float]] = None,
    ):
        self.dataset_lengths = dataset_lengths
        self.num_samples = (
            num_samples if num_samples is not None else sum(dataset_lengths)
        )
        self.offsets = torch.tensor([0] + dataset_lengths[:-1]).cumsum(dim=0).tolist()
        if weights is None:
            self.weights = torch.ones(len(dataset_lengths))
        else:
            assert len(weights) == len(dataset_lengths), (
                f"weights length {len(weights)} must match "
                f"dataset length {len(dataset_lengths)}"
            )
            self.weights = torch.tensor(weights)
        logging.info(
            f"Initialized len_balanced sampler with dataset sizes: "
            f"{dataset_lengths} and weights: {self.weights}"
        )

    def __iter__(self) -> Iterator[int]:
        """Yield global indices via hierarchical weighted sampling."""
        for _ in range(self.num_samples):
            d_idx = torch.multinomial(self.weights, 1).item()
            local_idx = torch.randint(0, self.dataset_lengths[d_idx], (1,)).item()
            yield self.offsets[d_idx] + local_idx

    def __len__(self) -> int:
        return self.num_samples


class DatasetWithID(torch.utils.data.Dataset):
    """Wrapper that injects ``dataset_id`` into the extra dict of each sample.

    Args:
        dataset: Underlying dataset (BaseDataset subclass).
        dataset_id: Integer ID to attach to every sample from this dataset.
    """

    def __init__(self, dataset: BaseDataset, dataset_id: int):
        self.dataset = dataset
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        canonical, extra = self.dataset[idx]
        extra["dataset_id"] = self.dataset_id
        return canonical, extra

    @property
    def worker_init_fn(self):
        return self.dataset.worker_init_fn


class PLData(LightningDataModule):
    """PyTorch Lightning data module for mixed-domain v6_mixed training.

    Each sub-dataset config in ``cfg.dataset.configs`` gets a sequential
    ``dataset_id`` (0, 1, 2, ...) that flows through to the model for
    per-dataset PoseIO normalization.

    Sampling is balanced by default (``sampler_mode: len_balanced``)
    with equal weights unless overridden.

    Args:
        cfg: Full experiment config.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg

        self.workers = self.cfg.get("workers", 0)
        logging.info(f"works={self.workers}")
        self.training_batchsize = self.cfg.get("train_batchsize", 32)
        self.eval_batchsize = self.cfg.get("eval_batchsize", 4)

        self.d_cfg = cfg.get("dataset", {})
        self.dataloader_drop_last = self.d_cfg.get("dataloader_drop_last", True)
        self.persistent_workers_flag = self.d_cfg.get("persistent_workers_flag", True)
        if self.persistent_workers_flag:
            logging.info("Persistent workers flag is enabled")

        # Default to balanced sampling for mixed-domain training
        self.sampler_mode = self.d_cfg.get("sampler_mode", "len_balanced")
        self.sampling_weights = self.d_cfg.get("sampling_weights", None)

        self.save_hyperparameters()

    def __make_dataset__(self, cfg: dict, phase: str, dataset_id: int) -> DatasetWithID:
        """Instantiate a single dataset wrapped with its dataset_id.

        Args:
            cfg: Domain config dict with ``"type"`` and ``"splits"`` keys.
            phase: One of ``"train"``, ``"val"``, ``"test"``.
            dataset_id: Integer ID for per-dataset PoseIO normalization.

        Returns:
            A :class:`DatasetWithID` wrapping the domain-specific dataset.

        Raises:
            ValueError: If ``cfg["type"]`` is not a recognised dataset name.
        """
        random_flag = phase == "train"
        if cfg["type"] == "behavior_boosted":
            ds = BehaviorV2BoostedDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                num_pts=cfg.get("num_pts", 256),
                max_scene_P=cfg.get("max_scene_P", 16),
                max_range=cfg.get("max_range", 5.0),
                return_details=False,
                views=cfg.get("views", ["head", "left_wrist", "right_wrist"]),
                uv_oversample=cfg.get("uv_oversample", 4),
                text_mode=cfg.get("text_mode", "activity"),
                num_pixel_queries_per_uid=cfg.get("num_pixel_queries_per_uid", 4),
                fixed_P=cfg.get("fixed_P", 0),
                image_size=cfg.get("image_size", 224),
                mask_downsample=cfg.get("mask_downsample", 2),
                state_dirname=cfg.get("state_dirname", "states"),
            )
        elif cfg["type"] == "behavior":
            ds = BehaviorV2Dataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                num_pts=cfg.get("num_pts", 256),
                max_scene_P=cfg.get("max_scene_P", 16),
                max_range=cfg.get("max_range", 5.0),
                return_details=False,
                views=cfg.get("views", ["head", "left_wrist", "right_wrist"]),
                uv_oversample=cfg.get("uv_oversample", 4),
                text_mode=cfg.get("text_mode", "activity"),
                num_pixel_queries_per_uid=cfg.get("num_pixel_queries_per_uid", 4),
                fixed_P=cfg.get("fixed_P", 0),
                image_size=cfg.get("image_size", 224),
                state_dirname=cfg.get("state_dirname", "states"),
            )
        elif cfg["type"] == "real":
            ds = RealDataset(
                data_root=cfg["data_root"],
                localframes_root=cfg["localframes_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                fixed_P=cfg.get("fixed_P", 512),
                image_size=cfg.get("image_size", 224),
                max_range=cfg.get("max_range", 50.0),
                add_camera_flag=cfg.get("add_camera_flag", False),
                reference_depth_median=cfg.get("reference_depth_median", -1.0),
                ctx_mode=cfg.get("ctx_mode", "current"),
                scene_align=cfg.get("scene_align", "yaw"),
            )
        else:
            raise ValueError(
                f"Unknown dataset type for umm_v6_mixed: {cfg['type']}. "
                f"Supported: behavior, behavior_boosted, real"
            )
        self.worker_init_fn = ds.worker_init_fn
        return DatasetWithID(ds, dataset_id)

    def setup(self, stage) -> None:
        """Build train/val/test ``ConcatDataset`` instances and optional sampler.

        Args:
            stage: Lightning stage string (``"fit"``, ``"validate"``, ``"test"``).
        """
        train_list, val_list, test_list = [], [], []
        for dataset_id, cfg in enumerate(self.d_cfg["configs"]):
            if "train" in cfg["splits"]:
                train_list.append(self.__make_dataset__(cfg, "train", dataset_id))
            if "val" in cfg["splits"]:
                val_list.append(self.__make_dataset__(cfg, "val", dataset_id))
            if "test" in cfg["splits"]:
                test_list.append(self.__make_dataset__(cfg, "test", dataset_id))
        self.train_dataset = ConcatDataset(train_list) if len(train_list) > 0 else None
        self.val_dataset = ConcatDataset(val_list) if len(val_list) > 0 else None
        self.test_dataset = ConcatDataset(test_list) if len(test_list) > 0 else None

        if self.sampler_mode is None:
            self.train_sampler = None
        elif self.sampler_mode == "len_balanced":
            if len(train_list) > 0:
                n_list = [len(d) for d in train_list]
                self.train_sampler = BalancedSampler(
                    n_list, weights=self.sampling_weights
                )
                logging.info(
                    f"Initialized len_balanced sampler with dataset sizes: "
                    f"{n_list} and weights: {self.sampling_weights}"
                )
        else:
            raise ValueError(f"Unknown sampler mode: {self.sampler_mode}")

    def __make_dataloader__(self, dataset, eval_flag) -> DataLoader:
        """Wrap a dataset in a ``DataLoader`` with the canonical collate function.

        Args:
            dataset: A ``ConcatDataset`` or None.
            eval_flag: If True, use eval batch size and disable shuffling.

        Returns:
            A ``DataLoader`` instance, or ``[]`` if *dataset* is falsy.
        """
        if not dataset:
            return []
        sampler = None
        if not eval_flag:
            sampler = self.train_sampler
        ret = DataLoader(
            dataset,
            batch_size=(self.eval_batchsize if eval_flag else self.training_batchsize),
            num_workers=self.workers,
            pin_memory=False,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=self.persistent_workers_flag and self.workers > 0,
            drop_last=self.dataloader_drop_last,
            shuffle=not eval_flag and sampler is None,
            collate_fn=CanonicalDatasetOutput.collate,
            sampler=sampler,
        )
        return ret

    def train_dataloader(self):
        return self.__make_dataloader__(self.train_dataset, False)

    def val_dataloader(self):
        return self.__make_dataloader__(self.val_dataset, True)

    def test_dataloader(self):
        return self.__make_dataloader__(self.test_dataset, True)

    def teardown(self, stage) -> None:
        return
