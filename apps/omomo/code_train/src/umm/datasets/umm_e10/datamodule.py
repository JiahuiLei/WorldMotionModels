"""PyTorch Lightning data module for UMM v6_all — unified dataset for all apps.

Integrates ALL dataset types from umm_v1 (old apps: hh, taco, mp, langtable,
maniskill, maniptrans) and umm_v2 (behavior, real/bridge/davis) into a single
data module.

Uses the same CanonicalDatasetOutput collate from v1 (which is compatible
with v2 datasets that also produce CanonicalDatasetOutput).

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

# Base dataset and collate function — all from v6_all (self-contained)
from umm.datasets.umm_e10.base_dataset import BaseDataset as BaseDatasetV1
from umm.datasets.umm_e10.base_dataset import CanonicalDatasetOutput as CanonicalV1
from umm.datasets.umm_e10.dataset_template import CanonicalDatasetOutput as CanonicalV6All

# All dataset classes — from v6_all (self-contained, no v1/v2 cross-references)
from umm.datasets.umm_e10.taco_dataset import TacoDataset
from umm.datasets.umm_e10.mp_dataset import MotionPlanningDataset
from umm.datasets.umm_e10.maniptrans_dataset import ManipTransDataset
from umm.datasets.umm_e10.hh_dataset import MultiHHDataset
from umm.datasets.umm_e10.langtable_oracle_dataset import LangTableOracleDataset
from umm.datasets.umm_e10.maniskill_oracle_dataset import ManiskillOracleDataset
from umm.datasets.umm_e10.behavior_v2_dataset import BehaviorV2Dataset
from umm.datasets.umm_e10.behavior_v2_boosted_dataset import BehaviorV2BoostedDataset
from umm.datasets.umm_e10.real_dataset import RealDataset


class BalancedSampler(Sampler):
    """Weighted hierarchical sampler for multi-dataset training.

    Performs two-level sampling: first selects a sub-dataset with probability
    proportional to ``weights``, then uniformly samples an index within that
    sub-dataset.

    Args:
        dataset_lengths: Number of samples in each sub-dataset.
        num_samples: Total samples per epoch. Defaults to sum of all lengths.
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
    """PyTorch Lightning data module for UMM v6_all — all apps unified.

    Reads ``cfg.dataset.configs``, builds per-domain datasets from both v1 and
    v2 dataset classes, and wraps them in ``DataLoader`` s.

    Supported dataset types:
    - From v1: taco, multi_humanhumanoid, motionplanning, maniptrans,
               langtable_oracle, maniskill_oracle
    - From v2: behavior, behavior_boosted, real

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

    def __make_dataset__(self, cfg: dict, phase: str):
        """Instantiate a single dataset from a per-domain config dict.

        Supports all v1 and v2 dataset types.

        Args:
            cfg: Domain config dict with ``"type"`` and ``"splits"`` keys.
            phase: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A dataset instance for the requested domain.

        Raises:
            ValueError: If ``cfg["type"]`` is not a recognised dataset name.
        """
        random_flag = phase == "train"
        dtype = cfg["type"]

        # ── V1 dataset types (old applications) ──────────────────────
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
                z_rot_aug=cfg.get("z_rot_aug", False),
            )
        elif dtype == "motionplanning":
            ret = MotionPlanningDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 3),
                language_fn=cfg.get("language_fn", None),
                max_T=cfg.get("max_T", 50),
                max_C=cfg.get("max_C", 8),
                max_P_per_object=cfg.get("max_P_per_object", 32),
                n_pcl_per_link=cfg.get("n_pcl_per_link", -1),
                local_pcl_N=cfg.get("local_pcl_N", 128),
                link_per_group_flag=cfg.get("link_per_group_flag", False),
                scale_normalize_range=cfg.get("scale_normalize_range", [-1, -1]),
                first_ctx_neutral_flag=cfg.get("first_ctx_neutral_flag", False),
                obst_pose_flag=cfg.get("obst_pose_flag", False),
            )
        elif dtype == "maniptrans":
            ret = ManipTransDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 60),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 3),
                language_fn=cfg.get("language_fn", None),
                max_T=cfg.get("max_T", 30),
                max_C=cfg.get("max_C", 8),
                max_P_per_object=cfg.get("max_P_per_object", 32),
                max_P_per_background=cfg.get("max_P_per_background", 10),
                cache_all_data=cfg.get("cache_all_data", False),
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
            )
        elif dtype == "maniskill_oracle":
            ret = ManiskillOracleDataset(
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
            )
        # ── V2 dataset types (new applications) ─────────────────────
        elif dtype == "behavior_boosted":
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
        elif dtype == "behavior":
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
                num_pixel_queries_per_uid=cfg.get("num_pixel_queries_per_uid", 4),
                fixed_P=cfg.get("fixed_P", 0),
                image_size=cfg.get("image_size", 224),
                state_dirname=cfg.get("state_dirname", "states"),
            )
        elif dtype == "real":
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
                f"Unknown dataset type for umm_e10: {dtype}. "
                f"Supported: taco, multi_humanhumanoid, motionplanning, maniptrans, "
                f"langtable_oracle, maniskill_oracle, behavior, behavior_boosted, real"
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
            collate_fn=CanonicalV6All.collate,
            sampler=sampler,
        )
        return ret

    def train_dataloader(self):
        """Return the training ``DataLoader``."""
        return self.__make_dataloader__(self.train_dataset, False)

    def val_dataloader(self):
        """Return the validation ``DataLoader``."""
        return self.__make_dataloader__(self.val_dataset, True)

    def test_dataloader(self):
        """Return the test ``DataLoader``."""
        return self.__make_dataloader__(self.test_dataset, True)

    def teardown(self, stage) -> None:
        """Clean up after fit or test."""
        return
