"""PyTorch Lightning data module for WMM V11 — all datasets with mask scheduler.

Integrates ALL dataset types with mask scheduling for training-time
conditioning mask generation. Merges v9's full dataset support with v10's
MaskScheduler.

Example::

    dm = PLData(cfg)
    dm.setup("fit")
    train_loader = dm.train_dataloader
"""

import logging
from typing import Iterator, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from umm.datasets.wmm.base_dataset import BaseDataset
from umm.datasets.wmm.dataset_template import CanonicalDatasetOutput
from umm.datasets.wmm.mask_scheduler import (
    MaskScheduler,
    HH_PATTERNS,
    TACO_PATTERNS,
    LANGTABLE_PATTERNS,
    MP_PATTERNS,
    SCENE_PATTERNS,
    HUMOTO_PATTERNS,
    MS3_NO_CTX_PATTERNS,
    one_step_future,
)
from umm.datasets.wmm.hh_dataset import MultiHHDataset
from umm.datasets.wmm.taco_dataset import TacoDataset
from umm.datasets.wmm.langtable_oracle_dataset import LangTableOracleDataset
from umm.datasets.wmm.mp_dataset import MotionPlanningDataset
from umm.datasets.wmm.maniptrans_dataset import ManipTransDataset
from umm.datasets.wmm.maniskill_oracle_dataset import ManiskillOracleDataset
from umm.datasets.wmm.behavior_dataset import BehaviorDataset
from umm.datasets.wmm.behavior_v2_boosted_dataset import BehaviorV2BoostedDataset
from umm.datasets.wmm.real_dataset import RealDataset
from umm.datasets.wmm.synthverse_dataset import SynthVerseDataset
from umm.datasets.wmm.synthverse_chunked_dataset import SynthVerseChunkedDataset
from umm.datasets.wmm.synthverse_traj_sharded_dataset import SynthVerseShardedDataset
from umm.datasets.wmm.humoto_dataset import HumotoDataset
from umm.datasets.wmm.mask_scheduler import OMOMO_PATTERNS
from umm.datasets.wmm.omomo_dataset import OmomoDataset
from umm.datasets.wmm.mask_scheduler import GRAB_PATTERNS
from umm.datasets.wmm.grab_dataset import GrabDataset

# TG: deterministic one_step_future (base_random_prob=0.0 → always picks this pattern).
# Known = t=0 only; context fully known. Matches TraceGen semantic exactly.
# ``one_step_future`` is the wmm-fork binding imported above from
# ``umm.datasets.wmm.mask_scheduler``; the wmm and umm_v13 implementations
# are bit-identical. Using the wmm binding keeps the package self-contained
# (per project Module Independence rule and the
# ``test_no_umm_v13_residual_imports_in_datamodule`` master-side guard).
TG_PATTERNS = [("one_step_future", one_step_future, 1.0)]


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


class _TGAdapterDataset(Dataset):
    """Adapter: wraps TracegenDataset (raw dict output) -> emits
    ``(CanonicalDatasetOutput, extra_dict)`` tuples with mask_scheduler
    applied. Kept in datamodule.py so `tg_dataset.py` remains a pure
    data emitter (scope).

    The mask scheduler (typically ``one_step_future`` via ``TG_PATTERNS``)
    sets ``known[:, 0] = True`` for valid keypoints and
    ``ctx_known = context_mask`` (fully known).
    """

    def __init__(self, tg_dataset, mask_scheduler, sem_type_val):
        self.tg_dataset = tg_dataset
        self.mask_scheduler = mask_scheduler
        self.sem_type_val = int(sem_type_val)
        # Reuse the standard threadpool-limiting worker_init_fn from BaseDataset
        # so mixed-dataset configs that pass this dataset last via
        # `self.worker_init_fn = ret.worker_init_fn` still get NumPy/OMP thread
        # limits applied in workers. TracegenDataset is a plain torch Dataset
        # (no worker_init_fn of its own).
        self.worker_init_fn = BaseDataset.worker_init_fn

    def __len__(self):
        return len(self.tg_dataset)

    def __getitem__(self, idx):
        item = self.tg_dataset[idx]
        # item keys (per schema):
        #   image (1,3,384,384) f32
        #   text str
        #   trace (400,33,4,4) f32 SE(3) — (P, T, 4, 4)
        #   trace_mask (400,33,3) bool — per-coord
        #   first_keypoint (400,2) f32 [0,1]
        #   gt_mask (400,) bool
        #   movement_bool (num_kps,) bool
        #   valid_steps () int64
        #   depth_raw (H,W) f32 or None
        #   env_name, episode_id, frame_id str

        trace = item["trace"]                              # (P, T, 4, 4)
        trace_mask_per_coord = item["trace_mask"]          # (P, T, 3) bool
        trace_mask = trace_mask_per_coord.any(dim=-1).bool()  # (P, T) attention-gating

        P, T = trace.shape[:2]

        # Context = duplicated first-frame trace; fully known
        context = trace[:, :1].contiguous()                # (P, 1, 4, 4)
        gt_mask = item["gt_mask"].bool()                   # (P,)
        context_mask = gt_mask.unsqueeze(-1)               # (P, 1)

        # Semantic + group + rot metadata
        sem_type = torch.full((P,), self.sem_type_val, dtype=torch.long)
        group_id = torch.zeros(P, dtype=torch.long)
        # rot_mask precedence (w1.6.0):
        #   1. localframe_root set → per-track validity from sidecar (overrides
        #      dummy_id_rot per Q2 in plan)
        #   2. dummy_id_rot=True   → all-True (legacy w1.5.0 dummy identity)
        #   3. dummy_id_rot=False  → all-False (legacy rot-zeroed)
        if self.tg_dataset.localframe_root is not None:
            rot_mask = item["track_rot_valid"]                     # (P,) bool
        elif self.tg_dataset.dummy_id_rot:
            rot_mask = torch.ones(P, dtype=torch.bool)
        else:
            rot_mask = torch.zeros(P, dtype=torch.bool)   # identity rot, w_R=0
        uid = torch.zeros(P, dtype=torch.long)

        # time = (arange(T) - anchor) * t_rate / base_fps  — mirrors
        # LangTable / Bridge / DAVIS pattern (wmm/real_dataset.py:683).
        # TG anchor is index 0 (t=0 is the only known frame), so
        # current_ith=0 → time_stamp = arange(T) * t_rate / base_fps.
        # Without this the LMM's t-axis RoPE + t_pe collapse to a single
        # phase across all 33 trace rows and the model cannot distinguish
        # time steps. base_fps + t_rate sourced from the underlying
        # TracegenDataset (configurable via YAML).
        time_unit = float(self.tg_dataset.t_rate) / float(self.tg_dataset.base_fps)
        time_stamp = torch.arange(T, dtype=torch.float32) * time_unit

        # Mask-scheduler yields known / ctx_known. With TG_PATTERNS this
        # is deterministically `one_step_future` (t=0 known, ctx fully known).
        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask,
            context_mask=context_mask,
            sem_type=sem_type,
        )

        seq_name = f"{item['env_name']}/{item['episode_id']}/{item['frame_id']}"

        canonical = CanonicalDatasetOutput(
            seq_name=seq_name,
            trace=trace,
            trace_mask=trace_mask,
            context=context,
            context_mask=context_mask,
            known_mask=known_mask,
            context_known_mask=ctx_known_mask,
            trace_group_id=group_id,
            trace_rot_mask=rot_mask,
            trace_sem_type=sem_type,
            trace_uid=uid,
            time=time_stamp,
            text=item["text"],
            local_pcl=None,
            local_pcl_mask=None,
        )

        # Store per-coord mask permuted to (T, P, 3) so post-collate
        # stacking yields (B, T, P, 3) matching the (B, T, P) trace_mask
        # convention.
        per_coord_tp = trace_mask_per_coord.permute(1, 0, 2).contiguous()  # (T, P, 3)

        extra = {
            "trace_valid_mask_per_coord": per_coord_tp,   # (T, P, 3) -> stacked to (B,T,P,3)
            "images": item["image"],                      # (1, 3, 384, 384)
            "pixel_uv_normalized": item["first_keypoint"], # (P, 2) [0,1]
            "pixel_view_idx": torch.zeros(P, dtype=torch.long),
            "pixel_query_mask": gt_mask,                  # (P,) bool
            "first_keypoint": item["first_keypoint"],     # duplicate for clarity
            "gt_mask": gt_mask,                           # (P,) bool (metric filter)
            "movement_bool": item["movement_bool"].bool(), # (P,) bool (metric filter)
            "valid_steps": item["valid_steps"],            # () int64
            "depth_raw": item["depth_raw"],                # (H,W) or None — for viz
            "env_name": item["env_name"],
            "episode_id": item["episode_id"],
            "frame_id": item["frame_id"],
            "cur_frame_idx": torch.zeros(P, dtype=torch.long),  # w1.4.0: TG anchors at trace t=0
        }
        return canonical, extra


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
    "motionplanning": {
        "base_random_prob": 0.25,
        "patterns": MP_PATTERNS,
    },
    "maniptrans": {
        "base_random_prob": 0.25,
        "patterns": TACO_PATTERNS,
    },
    "maniskill_oracle": {
        "base_random_prob": 0.25,
        "patterns": LANGTABLE_PATTERNS,
    },
    "ms3_state": {
        # User order 2026-04-23: MS3 training must NEVER mark ctx as known.
        # Deployment inference hallucinates the goal (ctx_known=False), so
        # training MUST match that regime. Set base_random_prob=0 (the random
        # path otherwise sets ctx_known ~50% of slots) + use MS3_NO_CTX_PATTERNS
        # (all patterns return ctx_known_mask all-False). Result: every
        # training sample trains the model in deployment-style generation mode.
        "base_random_prob": 0.0,
        "patterns": MS3_NO_CTX_PATTERNS,
    },
    "behavior_boosted": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "behavior": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "real": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "synthverse": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "synthverse_chunked": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "synthverse_traj_sharded": {
        "base_random_prob": 0.25,
        "patterns": SCENE_PATTERNS,
    },
    "humoto": {
        "base_random_prob": 0.25,
        "patterns": HUMOTO_PATTERNS,
    },
    "omomo": {
        "base_random_prob": 0.25,
        "patterns": OMOMO_PATTERNS,
    },
    "grab": {
        "base_random_prob": 0.0,
        "patterns": GRAB_PATTERNS,
    },
    "tracegen": {
        "base_random_prob": 0.0,
        "patterns": TG_PATTERNS,
    },
}


def _tg_stack_collate(in_list):
    """Collate that wraps CanonicalDatasetOutput.collate and stacks the TG
    per-coord mask list -> tensor ``(B, T, P, 3)``.

    Behaviour by batch composition:
    - Non-TG batch: ``extra["trace_valid_mask_per_coord"]`` absent -> no-op.
    - Pure TG batch: all entries tensors -> stacked to ``(B, T, P, 3)``.
    - Mixed batch (some TG, some non-TG): at least one entry is ``None``.
      We raise ``ValueError`` immediately with a clear message rather than
      silently leaving a heterogeneous list that would break far downstream
      at ``.to(device)`` / ``.shape`` calls. (reviewer finding #1 —
      latent-risk guard.)

    The TG ConcatDataset in this never mixes with non-TG, so this
    guard only trips if someone later wires a mixed config.
    """
    batch = CanonicalDatasetOutput.collate(in_list)
    extra = batch.get("extra", {})
    key = "trace_valid_mask_per_coord"
    if key in extra:
        entries = extra[key]
        if len(entries) == 0:
            return batch
        if all(isinstance(x, torch.Tensor) for x in entries):
            extra[key] = torch.stack(entries, dim=0)  # (B, T, P, 3)
        elif all(x is None for x in entries):
            # key leaked in from a schema that never populated it; drop.
            extra.pop(key)
        else:
            raise ValueError(
                "trace_valid_mask_per_coord collate error: mixed "
                "tensor/None entries in batch — this happens when a "
                "ConcatDataset mixes TG with a non-TG dataset. Separate "
                "them into their own dataloaders."
            )
    return batch


class PLData(LightningDataModule):
    """PyTorch Lightning data module for WMM V11 — all apps with mask scheduler.

    Reads ``cfg.dataset.configs``, builds per-domain datasets, and wraps
    them in ``DataLoader``s. Each dataset gets a MaskScheduler based on
    its domain type.

    Supported dataset types:
    - multi_humanhumanoid: MultiHHDataset with HH mask patterns
    - taco: TacoDataset with Taco mask patterns
    - langtable_oracle: LangTableOracleDataset with LangTable mask patterns
    - motionplanning: MotionPlanningDataset
    - maniptrans: ManipTransDataset
    - maniskill_oracle: ManiskillOracleDataset
    - ms3_state: Ms3StateDataset (wmm MS3 schema-v3; lazy-imported)
    - behavior_boosted: BehaviorV2BoostedDataset
    - behavior: BehaviorV2Dataset
    - real: RealDataset
    - synthverse: SynthVerseDataset
    - synthverse_chunked: SynthVerseChunkedDataset (lazy-load chunked version)
    - synthverse_traj_sharded: SynthVerseShardedDataset (trajectory-sharded version)
    - humoto: HumotoDataset with HUMOTO mask patterns

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
        back to the type-specific ``DEFAULT_MASK_CONFIGS`` entry (so e.g.
        ``tracegen`` gets ``base_random_prob=0.0`` and not the hardcoded
        0.25). Final fallback is 0.25 for any unregistered dtype.

        YAML may override per-pattern weights via the ``pattern_weights``
        block (a ``{name: weight}`` map). Each name MUST appear in the
        type's default patterns list — unknown names raise ``KeyError``.
        Names omitted from ``pattern_weights`` keep their default weight,
        so ``pattern_weights`` is purely additive overrides. When the
        ``pattern_weights`` block is absent, the patterns list is bit-
        identical to ``DEFAULT_MASK_CONFIGS[dtype]["patterns"]``.
        """
        dtype = cfg["type"]
        mask_cfg = cfg.get("mask_scheduler", {})
        default_cfg = DEFAULT_MASK_CONFIGS.get(dtype, {})
        base_random_prob = mask_cfg.get(
            "base_random_prob",
            default_cfg.get("base_random_prob", 0.25),
        )
        default_patterns = default_cfg.get("patterns", [])
        yaml_weights = mask_cfg.get("pattern_weights", None)
        if yaml_weights is None:
            patterns = default_patterns
        else:
            default_names = {p[0] for p in default_patterns}
            unknown = set(yaml_weights) - default_names
            if unknown:
                raise KeyError(
                    f"_make_mask_scheduler({dtype}): unknown pattern names in "
                    f"YAML pattern_weights: {sorted(unknown)}. "
                    f"Valid names: {sorted(default_names)}"
                )
            patterns = [
                (name, fn, yaml_weights.get(name, default_w))
                for (name, fn, default_w) in default_patterns
            ]

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
                local_pcl_N=cfg.get("local_pcl_N", 0),
                mesh_npz_paths=cfg.get("mesh_npz_paths", {}),
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
                mask_scheduler=mask_scheduler,
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
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "ms3_state":
            # Lazy import: creates this file in parallel.
            # Keeping the import inside the branch body lets `datamodule.py`
            # stay importable while `ms3_state_dataset.py` has not yet landed.
            from umm.datasets.wmm.ms3_state_dataset import Ms3StateDataset  # noqa: E501  pyright: ignore[reportMissingImports]

            ret = Ms3StateDataset(
                data_root=cfg["data_root"],
                task=cfg["task"],
                # MS3 Panda sim runs at 20 Hz (agent.controller step; see
                # WmmAdapter.BASE_FPS). A default of 10 would cause a silent
                # 2× timestamp drift between training and deployment.
                base_fps=cfg.get("base_fps", 20),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 20),
                num_pts=cfg.get("num_pts", 256),
                mask_scheduler=mask_scheduler,
            )
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
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "behavior":
            ret = BehaviorDataset(
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
                state_dirname=cfg.get("state_dirname", "states"),
                mask_scheduler=mask_scheduler,
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
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "synthverse":
            ret = SynthVerseDataset(
                data_root=cfg["data_root"],
                localframes_root=cfg["localframes_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 24),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                fixed_P=cfg.get("fixed_P", 512),
                image_size=cfg.get("image_size", 224),
                add_camera_flag=cfg.get("add_camera_flag", False),
                reference_depth_median=cfg.get("reference_depth_median", -1.0),
                scene_align=cfg.get("scene_align", "yaw"),
                bg_query_ratio=cfg.get("bg_query_ratio", 0.0),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "synthverse_chunked":
            ret = SynthVerseChunkedDataset(
                data_root=cfg["data_root"],
                localframes_root=cfg["localframes_root"],
                frames_root=cfg["frames_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 24),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                fixed_P=cfg.get("fixed_P", 512),
                image_size=cfg.get("image_size", 224),
                add_camera_flag=cfg.get("add_camera_flag", False),
                reference_depth_median=cfg.get("reference_depth_median", -1.0),
                scene_align=cfg.get("scene_align", "yaw"),
                bg_query_ratio=cfg.get("bg_query_ratio", 0.0),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "synthverse_traj_sharded":
            ret = SynthVerseShardedDataset(
                data_root=cfg["data_root"],
                localframes_root=cfg["localframes_root"],
                frames_root=cfg["frames_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 24),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                max_T=cfg.get("max_T", 30),
                fixed_P=cfg.get("fixed_P", 512),
                image_size=cfg.get("image_size", 224),
                add_camera_flag=cfg.get("add_camera_flag", False),
                reference_depth_median=cfg.get("reference_depth_median", -1.0),
                scene_align=cfg.get("scene_align", "yaw"),
                bg_query_ratio=cfg.get("bg_query_ratio", 0.0),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "humoto":
            ret = HumotoDataset(
                data_root=cfg["data_root"],
                split_fn=cfg["splits"][phase],
                base_fps=cfg.get("base_fps", 30),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                t_rate=cfg.get("t_rate", 1),
                language_fn=cfg.get("language_fn", None),
                max_T=cfg.get("max_T", 30),
                max_C=cfg.get("max_C", 1),
                max_objects=cfg.get("max_objects", 4),
                max_P_per_object=cfg.get("max_P_per_object", 256),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "omomo":
            # OMOMO uses a single positional split key (train|test|val) instead
            # of a split_fn JSON path; mirrors humoto's contract for everything
            # else (mask_scheduler, safe_flag, repeat).
            split_key = phase if phase != "val" else "test"
            ret = OmomoDataset(
                data_root=cfg["data_root"],
                split=cfg.get("splits", {}).get(phase, split_key),
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                num_pts=cfg.get("num_pts", 256),
                text_required=cfg.get("text_required", True),
                max_T=cfg.get("max_T", 120),
                max_C=cfg.get("max_C", 1),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "grab":
            split_key = phase if phase != "val" else "test"
            split_name = cfg.get("splits", {}).get(phase, f"s1_{split_key}")
            ret = GrabDataset(
                data_root=cfg["data_root"],
                split=split_name,
                random_flag=random_flag,
                safe_flag=cfg.get("safe_flag", True),
                repeat=cfg.get("repeat", 1),
                num_pts=cfg.get("num_pts", 256),
                max_T=cfg.get("max_T", 200),
                max_C=cfg.get("max_C", 1),
                text_required=cfg.get("text_required", True),
                # Filter knobs forwarded from YAML so production launchers can
                # gate on cache coverage and toggle M-seq deduplication.
                # Defaults (skip_mirrored=True, require_cache=True,
                # min_coverage_frac=0.0) match GrabDataset's safe defaults.
                skip_mirrored=cfg.get("skip_mirrored", True),
                require_cache=cfg.get("require_cache", True),
                min_coverage_frac=cfg.get("min_coverage_frac", 0.0),
                mask_scheduler=mask_scheduler,
            )
        elif dtype == "tracegen":
            # Lazy-import TG dataset to avoid loading at module init.
            # SEM_REAL_SCENE comes from the wmm fork's dataset_config —
            # ``umm.datasets.wmm.dataset_config`` is the single source of truth
            # for wmm semantic IDs (the master-side rename-guard test forbids
            # any ``umm_v13`` imports from this module).
            from umm.datasets.wmm.tg_dataset import TracegenDataset
            from umm.datasets.wmm.dataset_config import SEM_REAL_SCENE

            tg = TracegenDataset(
                data_root=cfg["data_root"],
                env_name=cfg["env_name"],
                split=phase,
                trajectory_horizon=cfg.get("trajectory_horizon", 32),
                num_kps=cfg.get("num_kps", 400),
                image_size=cfg.get("image_size", 384),
                val_split=cfg.get("val_split", 0.01),
                random_seed=cfg.get("random_seed", 42),
                max_episodes=cfg.get("max_episodes", None),
                safe_flag=cfg.get("safe_flag", True),
                base_fps=cfg.get("base_fps", 30.0),
                t_rate=cfg.get("t_rate", 1),
                dummy_id_rot=cfg.get("dummy_id_rot", False),
                localframe_root=cfg.get("localframe_root", None),  # w1.6.0
            )
            if mask_scheduler is None:
                # eval/val phase — reuse `_make_mask_scheduler` so the
                # DEFAULT_MASK_CONFIGS["tracegen"] entry is the single
                # source of truth for both train and eval scheduler config.
                # (reviewer finding #3 — no inline duplication.)
                mask_scheduler = self._make_mask_scheduler(cfg)
            ret = _TGAdapterDataset(tg, mask_scheduler, SEM_REAL_SCENE)
        else:
            raise ValueError(
                f"Unknown dataset type for wmm: {dtype}. "
                f"Supported: taco, multi_humanhumanoid, motionplanning, maniptrans, "
                f"langtable_oracle, maniskill_oracle, ms3_state, behavior, "
                f"behavior_boosted, real, synthverse, synthverse_chunked, "
                f"synthverse_traj_sharded, humoto, omomo, grab, tracegen"
            )
        self.worker_init_fn = ret.worker_init_fn
        return ret

    def setup(self, stage) -> None:
        """Build train/val/test ConcatDataset instances and optional sampler.

        Dataset-type conventions:
        * Legacy (oracle / behavior / real / synthverse / ...) types require
          ``cfg["splits"]`` to enumerate the phases they own.
        * ``ms3_state`` resolves its own split internally from
          ``data_root`` + ``task`` and supplies no ``splits`` key. Default to
          a train-only phase set for such configs so we do not KeyError.
        """
        train_list, val_list, test_list = [], [], []
        for cfg in self.d_cfg["configs"]:
            splits = cfg.get("splits", {"train": None})
            if "train" in splits:
                train_list.append(self.__make_dataset__(cfg, "train"))
            if "val" in splits:
                val_list.append(self.__make_dataset__(cfg, "val"))
            if "test" in splits:
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
            collate_fn=_tg_stack_collate,   # was: CanonicalDatasetOutput.collate
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
