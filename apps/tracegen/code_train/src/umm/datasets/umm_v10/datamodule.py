"""PyTorch Lightning data module for UMM V10 — with mask scheduler support.

Integrates all V10 dataset types with mask scheduling for training-time
conditioning mask generation.

Example::

    dm = PLData(cfg)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
"""

import logging
from typing import Iterator, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from umm.datasets.umm_v10.base_dataset import BaseDataset
from umm.datasets.umm_v10.dataset_template import CanonicalDatasetOutput
from umm.datasets.umm_v10.mask_scheduler import (
    MaskScheduler,
    HH_PATTERNS,
    TACO_PATTERNS,
    LANGTABLE_PATTERNS,
)
from umm.datasets.umm_v10.hh_dataset import MultiHHDataset
from umm.datasets.umm_v10.taco_dataset import TacoDataset
from umm.datasets.umm_v10.langtable_oracle_dataset import LangTableOracleDataset


class BalancedSampler(Sampler):
    """Weighted hierarchical sampler for multi-dataset training.

    Performs two-level sampling: first selects a sub-dataset with probability
    proportional to ``weights``, then uniformly samples an index within that
    sub-dataset.
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
            assert len(weights) == len(dataset_lengths)
            self.weights = torch.tensor(weights)
        logging.info(
            f"Initialized len_balanced sampler with dataset sizes: "
            f"{dataset_lengths} and weights: {self.weights}"
        )

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            d_idx = torch.multinomial(self.weights, 1).item()
            local_idx = torch.randint(0, self.dataset_lengths[d_idx], (1,)).item()
            yield self.offsets[d_idx] + local_idx

    def __len__(self) -> int:
        return self.num_samples


# Default mask scheduler configs per dataset type
DEFAULT_MASK_CONFIGS = {
    "multi_humanhumanoid": {
        "base_random_prob": 0.25,
        "patterns": HH_PATTERNS,
    },
    "taco": {
        "base_random_prob": 0.25,
        "patterns": TACO_PATTERNS,
    },
    "langtable_oracle": {
        "base_random_prob": 0.25,
        "patterns": LANGTABLE_PATTERNS,
    },
}


class PLData(LightningDataModule):
    """PyTorch Lightning data module for UMM V10 — all apps with mask scheduler.

    Reads ``cfg.dataset.configs``, builds per-domain datasets, and wraps
    them in ``DataLoader``s. Each dataset gets a MaskScheduler based on
    its domain type.

    Supported dataset types:
    - multi_humanhumanoid: MultiHHDataset with HH mask patterns
    - taco: TacoDataset with Taco mask patterns
    - langtable_oracle: LangTableOracleDataset with LangTable mask patterns

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

    def _make_mask_scheduler(self, cfg: dict) -> MaskScheduler:
        """Create a MaskScheduler from dataset config.

        Uses per-dataset mask_scheduler config if present, otherwise falls
        back to domain defaults.
        """
        dtype = cfg["type"]
        mask_cfg = cfg.get("mask_scheduler", {})
        base_random_prob = mask_cfg.get("base_random_prob", 0.25)

        default_cfg = DEFAULT_MASK_CONFIGS.get(dtype, {})
        patterns = default_cfg.get("patterns", [])

        return MaskScheduler(
            base_random_prob=base_random_prob,
            patterns=patterns,
        )

    def __make_dataset__(self, cfg: dict, phase: str):
        """Instantiate a single dataset from a per-domain config dict.

        Args:
            cfg: Domain config dict with ``"type"`` and ``"splits"`` keys.
            phase: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A dataset instance for the requested domain.
        """
        random_flag = phase == "train"
        dtype = cfg["type"]
        mask_scheduler = self._make_mask_scheduler(cfg) if random_flag else None

        if dtype == "taco":
            ret = TacoDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 3),
                language_fn=cfg.get("language_fn", None),
                max_T=cfg.get("max_T", 30),
                max_C=cfg.get("max_C", 8),
                max_P_per_object=cfg.get("max_P_per_object", 32),
                model_fn=cfg.get("model_fn", None),
                state_dirname=cfg.get("state_dirname", "state"),
                xyz_bias=cfg.get("xyz_bias", [0.0, 0.0, -0.4]),
                wrist_local_coords=cfg.get("wrist_local_coords", False),
                wrist_sem_as_body=cfg.get("wrist_sem_as_body", False),
                rigid_object_pose=cfg.get("rigid_object_pose", False),
                pcl_with_rotation=cfg.get("pcl_with_rotation", False),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "multi_humanhumanoid":
            ret = MultiHHDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 3),
                language_fn=cfg.get("language_fn", None),
                max_T=cfg.get("max_T", 30),
                max_C=cfg.get("max_C", 8),
                max_P=cfg.get("max_P", 4),
                max_G=cfg.get("max_G", 8),
                embodiment_dirs=cfg.get(
                    "embodiment_dirs",
                    {"smplx": "human_states", "g1": "g1_states"},
                ),
                center_self_flag=cfg.get("center_self_flag", False),
                center_ctx_flag=cfg.get("center_ctx_flag", False),
                center_real_flag=cfg.get("center_real_flag", False),
                human_sample_bias_prob=cfg.get("human_sample_bias_prob", -1.0),
                local_global_factor_flag=cfg.get("local_global_factor_flag", False),
                local_global_trans_factor_flag=cfg.get(
                    "local_global_trans_factor_flag", False
                ),
                ctx_fixed_location_interval=cfg.get(
                    "ctx_fixed_location_interval", -1.0
                ),
                ctx_fixed_location_bias=cfg.get("ctx_fixed_location_bias", 0.0),
                max_robot_per_person=cfg.get("max_robot_per_person", -1),
                hand_attached_to_wrist_flag=cfg.get(
                    "hand_attached_to_wrist_flag", False
                ),
                remap_hands_to_mano_sem=cfg.get("remap_hands_to_mano_sem", False),
                wrist_local_coords=cfg.get("wrist_local_coords", False),
                z_rot_aug=cfg.get("z_rot_aug", False),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "langtable_oracle":
            ret = LangTableOracleDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 10),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                num_pts=cfg.get("num_pts", 256),
                is_explore_dataset=cfg.get("is_explore_dataset", False),
                mask_scheduler=mask_scheduler,
            )
        else:
            raise ValueError(
                f"Unknown dataset type for umm_v10: {dtype}. "
                f"Supported: taco, multi_humanhumanoid, langtable_oracle"
            )
        self.worker_init_fn = ret.worker_init_fn
        return ret

    def setup(self, stage) -> None:
        """Build train/val/test ConcatDataset instances and optional sampler."""
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

        global_max_T = max(cfg.get("max_T", 30) for cfg in self.d_cfg["configs"])
        if self.cfg.get("model", {}).get("compile_flag", False):
            CanonicalDatasetOutput._compile_fixed_T = global_max_T
            logging.info(
                f"compile_flag=True: padding all batches to fixed T={global_max_T}"
            )

        if self.sampler_mode is None:
            self.train_sampler = None
        elif self.sampler_mode == "len_balanced":
            if len(train_list) > 0:
                n_list = [len(d) for d in train_list]
                self.train_sampler = BalancedSampler(
                    n_list, weights=self.sampling_weights
                )
        else:
            raise ValueError(f"Unknown sampler mode: {self.sampler_mode}")

    def __make_dataloader__(self, dataset, eval_flag) -> DataLoader:
        """Wrap a dataset in a DataLoader with the canonical collate function."""
        if not dataset:
            return []
        sampler = None
        if not eval_flag:
            sampler = self.train_sampler
        dl_kwargs = dict(
            dataset=dataset,
            batch_size=(self.eval_batchsize if eval_flag else self.training_batchsize),
            num_workers=self.workers,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=self.persistent_workers_flag and self.workers > 0,
            drop_last=self.dataloader_drop_last,
            shuffle=not eval_flag and sampler is None,
            collate_fn=CanonicalDatasetOutput.collate,
            sampler=sampler,
        )
        if self.workers > 0:
            dl_kwargs["prefetch_factor"] = self.d_cfg.get("prefetch_factor", 4)
        ret = DataLoader(**dl_kwargs)
        return ret

    def train_dataloader(self):
        return self.__make_dataloader__(self.train_dataset, False)

    def val_dataloader(self):
        return self.__make_dataloader__(self.val_dataset, True)

    def test_dataloader(self):
        return self.__make_dataloader__(self.test_dataset, True)

    def teardown(self, stage) -> None:
        return

    def print_mask_stats(self):
        """Print mask scheduler statistics for all training datasets."""
        if self.train_dataset is None:
            logging.info("No training datasets")
            return
        for ds in self.train_dataset.datasets:
            if hasattr(ds, "mask_scheduler") and ds.mask_scheduler is not None:
                logging.info(f"--- {ds.__class__.__name__} ---")
                ds.mask_scheduler.print_stats()
