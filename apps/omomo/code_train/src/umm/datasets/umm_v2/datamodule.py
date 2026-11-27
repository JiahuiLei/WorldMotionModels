"""PyTorch Lightning data module and balanced sampler for UMM v2 training.

Defines :class:`PLData` (the ``LightningDataModule`` entry-point) which reads
the dataset config, instantiates per-domain datasets, and builds train/val/test
``DataLoader`` s with the canonical collate function.

V2 currently supports behavior and real-world domains (with pixel UV sampling and
DINO image features).
"""

import logging
from typing import Iterator, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from umm.datasets.umm_v2.base_dataset import BaseDataset, CanonicalDatasetOutput
from umm.datasets.umm_v2.behavior_v2_dataset import BehaviorV2Dataset
from umm.datasets.umm_v2.behavior_v2_boosted_dataset import BehaviorV2BoostedDataset
from umm.datasets.umm_v2.real_dataset import RealDataset


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


class PLData(LightningDataModule):
    """PyTorch Lightning data module for UMM v2 datasets.

    Reads ``cfg.dataset.configs``, builds per-domain datasets, and wraps
    them in ``DataLoader`` s.

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

        self.sampler_mode = self.d_cfg.get("sampler_mode", None)
        self.sampling_weights = self.d_cfg.get("sampling_weights", None)

        self.save_hyperparameters()

    def __make_dataset__(self, cfg: dict, phase: str) -> BaseDataset:
        """Instantiate a single dataset from a per-domain config dict.

        Args:
            cfg: Domain config dict with ``"type"`` and ``"splits"`` keys.
            phase: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A :class:`BaseDataset` subclass instance.

        Raises:
            ValueError: If ``cfg["type"]`` is not a recognised dataset name.
        """
        random_flag = phase == "train"
        if cfg["type"] == "behavior_boosted":
            ret = BehaviorV2BoostedDataset(
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
            ret = BehaviorV2Dataset(
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
                # V2-specific: pixel UV sampling + DINO image size
                num_pixel_queries_per_uid=cfg.get("num_pixel_queries_per_uid", 4),
                fixed_P=cfg.get("fixed_P", 0),
                image_size=cfg.get("image_size", 224),
                state_dirname=cfg.get("state_dirname", "states"),
            )
        elif cfg["type"] == "real":
            ret = RealDataset(
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
                f"Unknown dataset type for umm_v2: {cfg['type']}. "
                f"V2 currently supports: behavior, real"
            )
        self.worker_init_fn = ret.worker_init_fn
        return ret

    def setup(self, stage) -> None:
        """Build train/val/test ``ConcatDataset`` instances and optional sampler.

        Args:
            stage: Lightning stage string (``"fit"``, ``"validate"``, ``"test"``).
        """
        train_list, val_list, test_list = [], [], []
        for cfg in self.d_cfg["configs"]:
            if "train" in cfg["splits"]:
                train_list.append(self.__make_dataset__(cfg, "train"))
            if "val" in cfg["splits"]:
                val_list.append(self.__make_dataset__(cfg, "val"))
            if "test" in cfg["splits"]:
                test_list.append(self.__make_dataset__(cfg, "test"))
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
