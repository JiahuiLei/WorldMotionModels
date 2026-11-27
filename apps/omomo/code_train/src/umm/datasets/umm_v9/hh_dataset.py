"""Multi-person human-humanoid dataset for motion retargeting and generation.

Supports multiple humans and flexible embodiment choices (SMPL-X, G1, H12,
T1 humanoids).  Each person can have multiple embodiment representations
that are jointly sampled.  Includes centering, local/global factorisation,
and fixed-grid context placement options.

Example::

    dataset = MultiHHDataset(
        data_root="data/embody3d_phuma_processed/hand_duo_subset/",
        split_fn="data_splits/embody3d_humanoid_v2/debug_train.json",
        base_fps=30,
        max_T=30,
        max_C=5,
        embodiment_dirs={"smplx": "human_states", "g1": "g1_states"},
    )
    sample, extra = dataset[0]
"""

################################################################################
# GeneMoma Dataset - Unified dataset for general motion modeling
################################################################################
# ! from humanhumanoid: 1. multiple human 2. flexible choice of robot or human. 3. have a flag to constrain the paired human-robot setting 4. deal with the alignment because there is global pose, the human is always the reference. and the alignmetn is per person
# this already include single human, set max human to 1 and set a flag to canonicalize to each local human!!

import os
import os.path as osp
import sys
import json
import zipfile
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from copy import deepcopy
import random

from umm.datasets.umm_v9.base_dataset import (
    BaseDataset,
    PromptV0,
)
from umm.datasets.umm_v9.dataset_template import CanonicalDatasetOutput
from umm.datasets.umm_v9.dataset_utils import sample_list
from umm.datasets.umm_v9.dataset_config import *


def load_indices(raw_indices, t_rate):
    """Build per-frame index list and sequence-length dict from raw split data.

    Args:
        raw_indices: List of ``(sid, length_str)`` tuples from the JSON split.
        t_rate: Temporal stride between sampled frames.

    Returns:
        Tuple ``(indices, seq_len)`` where ``indices`` is a list of
        ``(sid, tid)`` and ``seq_len`` maps sid to total length.

    Example::

        indices, seq_len = load_indices([("seq_0", "100")], t_rate=2)
    """
    indices, seq_len = [], {}
    for sid, length in raw_indices:
        length = int(length)
        seq_len[sid] = length
        for tid in range(0, length, t_rate):
            indices.append((sid, tid))
    return indices, seq_len


def id_p_split(fn):
    """Split a filename into scene ID and person ID by 10-underscore delimiter.

    Args:
        fn: Filename string (e.g. ``"scene01__________person0"``).

    Returns:
        Tuple ``(sid, pid)`` where ``pid`` is ``"SINGLETON"`` if no
        delimiter is found.

    Example::

        sid, pid = id_p_split("scene01__________person0")
        # ("scene01", "person0")
    """
    r = fn.split("_" * 10)
    if len(r) == 2:
        return r[0], r[1]
    elif len(r) == 1:
        return r[0], "SINGLETON"
    else:
        raise ValueError(f"Invalid filename: {fn}")


def get_fn(sid, pid):
    """Reconstruct a filename from scene ID and person ID.

    Args:
        sid: Scene identifier.
        pid: Person identifier (``"SINGLETON"`` omits the delimiter).

    Returns:
        Filename string.

    Example::

        fn = get_fn("scene01", "person0")
        # "scene01__________person0"
    """
    if pid == "SINGLETON":
        return f"{sid}"
    else:
        return f"{sid}__________{pid}"


def determine_ctx_fixed_grid_size(n):
    """Map person count to a 2-D grid size ``(rows, cols)`` for fixed-location context.

    Args:
        n: Number of persons (1--4).

    Returns:
        Tuple ``(rows, cols)``.

    Example::

        rows, cols = determine_ctx_fixed_grid_size(3)  # (2, 2)
    """
    return {
        1: (1, 1),  # x,y
        2: (1, 2),  # x,y
        3: (2, 2),
        4: (2, 2),
    }[n]


class MultiHHDataset(BaseDataset):
    """Multi-person human-humanoid dataset with flexible embodiment sampling.

    Each sequence may contain multiple persons, each with one or more
    embodiment representations (SMPL-X human, G1/H12/T1 humanoid).
    Supports centering, local/global factorisation, hand attachment to
    wrist, biased human sampling, and fixed-grid context placement.

    Args:
        data_root: Root directory.
        split_fn: JSON split file.
        base_fps: Recording frame rate.
        random_flag: Stochastic augmentation.
        safe_flag: Error-handling wrapper.
        repeat: Repetition multiplier.
        t_rate: Temporal stride.
        language_fn: Optional language annotation file.
        max_T: Max time steps.
        max_C: Max context steps.
        max_P: Max persons per sample.
        max_G: Max groups (embodiments) across all persons.
        embodiment_dirs: Mapping from embodiment key to state sub-directory.
        center_self_flag: Center both context and trace on self.
        center_ctx_flag: Center context on pelvis at current time.
        center_real_flag: Center trace on pelvis at current time.
        human_sample_bias_prob: Prob of prioritising human embodiments.
        max_robot_per_person: Cap non-human embodiments per person.
        local_global_factor_flag: Factor first joint as global pose.
        local_global_trans_factor_flag: Factor first joint translation.
        hand_attached_to_wrist_flag: Express hands in wrist-local frame.
        remap_hands_to_mano_sem: Remap HH wrist/finger joints to TACO-compatible
            semantic IDs (SEM_LEFT/RIGHT_MANO_WRIST/FINGERS).
        wrist_local_coords: Express finger joints in wrist-local frame with
            TACO-compatible semantics. Requires remap_hands_to_mano_sem=True.
        ctx_fixed_location_interval: Fixed-grid spacing (<=0 to disable).
        ctx_fixed_location_bias: X-axis bias for the grid.
        z_rot_aug: Random z-axis rotation augmentation (only when random_flag=True).

    Example::

        ds = MultiHHDataset(
            data_root="data/embody3d/",
            split_fn="splits/train.json",
            base_fps=30,
            max_T=30,
            max_C=5,
            embodiment_dirs={"smplx": "human_states", "g1": "g1_states"},
        )
        sample, extra = ds[0]
    """

    # SID -> PERSON -> EMBODIMENT
    # todo: can sample less human to support more htan 2 emb per person???
    def __init__(
        self,
        data_root,
        split_fn,
        base_fps,
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
        language_fn: str | None = None,
        #############################################################
        max_T: int = 30,  # num of time step
        max_C: int = 8,  # num of context steps
        z_rot_aug: bool = False,  # random z-rotation augmentation
        # Embodiment    # config different embodiments
        max_P: int = 4,  # num of persons
        max_G: int = 8,  # num of groups, if each person has two group, e.g. human+robot then the max_G should be  max_P*2 = 8
        embodiment_dirs: dict[str, str] = {"smplx": "human_states", "g1": "g1_states"},
        # todo: add more flags
        center_self_flag: bool = False,
        center_ctx_flag: bool = False,  # ctx is centered if center_ctx_flag|center_self_flag
        center_real_flag: bool = False,  # real is centered if center_real_flag|center_self_flag
        human_sample_bias_prob: float = -1.0,
        max_robot_per_person: int = -1,
        #######################################################################
        local_global_factor_flag: bool = False,
        local_global_trans_factor_flag: bool = False,
        # if set 1. change the sem 2. always use first Q as the global pose and all others are canonicalized in this local frame.
        # ! NOTE THE LOCAL CANONICALIZATION IS PER GROUP!
        hand_attached_to_wrist_flag: bool = False,  # if so the hand is on the local frame of the body
        remap_hands_to_mano_sem: bool = False,  # remap HH hand joints to TACO-compatible sem IDs
        wrist_local_coords: bool = False,  # express finger joints in wrist-local frame (requires remap_hands_to_mano_sem)
        #######################################################################
        ctx_fixed_location_interval: float = -1.0,
        ctx_fixed_location_bias: float = 0.0,
        #######################################################################
    ):
        """Initialize the multi-person HH dataset.

        Scans the split file to group sequences by scene and person,
        discovers available embodiments per person, and builds indices.

        Args:
            data_root: Root data directory.
            split_fn: JSON split file path.
            base_fps: Frame rate.
            random_flag: Stochastic sampling.
            safe_flag: Error-handling wrapper.
            repeat: Repetition multiplier.
            t_rate: Temporal stride.
            language_fn: Language annotation file.
            max_T: Max time steps.
            max_C: Max context steps.
            max_P: Max persons.
            max_G: Max groups.
            embodiment_dirs: Embodiment key to directory mapping.
            center_self_flag: Center both ctx and trace.
            center_ctx_flag: Center context.
            center_real_flag: Center trace.
            human_sample_bias_prob: Human-bias probability.
            max_robot_per_person: Max non-human embodiments.
            local_global_factor_flag: Global-pose factorisation.
            local_global_trans_factor_flag: Translation factorisation.
            hand_attached_to_wrist_flag: Wrist-local hands.
            ctx_fixed_location_interval: Grid spacing.
            ctx_fixed_location_bias: Grid x-bias.
            z_rot_aug: Apply random z-axis rotation augmentation.

        Example::

            ds = MultiHHDataset(
                data_root="data/embody3d/",
                split_fn="splits/train.json",
                base_fps=30,
                embodiment_dirs={"smplx": "human_states"},
            )
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn, random_flag, safe_flag, repeat
        )

        self.max_T = max_T
        self.max_C = max_C
        self.max_P = max_P
        self.max_G = max_G
        self.z_rot_aug = z_rot_aug

        self.center_ctx_flag = center_ctx_flag | center_self_flag
        self.center_real_flag = center_real_flag | center_self_flag

        self.human_sample_bias_prob = human_sample_bias_prob
        self.max_robot_per_person = max_robot_per_person

        self.hand_attached_to_wrist_flag = hand_attached_to_wrist_flag
        if self.hand_attached_to_wrist_flag:
            logging.warning(
                "hand_attached_to_wrist_flag is set to True, which means the hand is on the local frame of the body."
            )
        self.remap_hands_to_mano_sem = remap_hands_to_mano_sem
        self.wrist_local_coords = wrist_local_coords
        if self.wrist_local_coords:
            assert self.remap_hands_to_mano_sem, (
                "wrist_local_coords requires remap_hands_to_mano_sem=True"
            )
            assert not self.hand_attached_to_wrist_flag, (
                "wrist_local_coords and hand_attached_to_wrist_flag are mutually exclusive — "
                "wrist_local_coords includes hand localization with TACO-compatible semantics"
            )
            assert not local_global_factor_flag, (
                "wrist_local_coords and local_global_factor_flag are mutually exclusive"
            )
            assert not local_global_trans_factor_flag, (
                "wrist_local_coords and local_global_trans_factor_flag are mutually exclusive"
            )
            logging.info(
                "HH wrist_local_coords=True — fingers expressed in wrist-local frame "
                "with TACO-compatible MANO semantic IDs"
            )
        elif self.remap_hands_to_mano_sem:
            assert not self.hand_attached_to_wrist_flag, (
                "hand_attached_to_wrist_flag and remap_hands_to_mano_sem are mutually exclusive — "
                "use wrist_local_coords=True instead for localization with TACO semantics"
            )
            logging.info(
                "HH remap_hands_to_mano_sem=True — wrist/finger joints remapped to "
                "SEM_LEFT/RIGHT_MANO_WRIST/FINGERS for shared PoseIO with TACO"
            )
        self.local_global_factor_flag = local_global_factor_flag
        if self.local_global_factor_flag:
            assert (
                not self.hand_attached_to_wrist_flag
            ), "local_global_factor_flag and hand_attached_to_wrist_flag cannot be set at the same time."
            logging.warning(
                "local_global_factor_flag is set to True, which means the first Q is used as the global pose and all others are canonicalized in this local frame."
            )
        self.local_global_trans_factor_flag = local_global_trans_factor_flag
        if self.local_global_trans_factor_flag:
            assert (
                not self.hand_attached_to_wrist_flag
            ), "local_global_trans_factor_flag and hand_attached_to_wrist_flag cannot be set at the same time."
            logging.warning(
                "local_global_trans_factor_flag is set to True, which means the first T is used as the global translation and all others are canonicalized in this local frame."
            )
        assert not (
            self.local_global_factor_flag and self.local_global_trans_factor_flag
        ), "local_global_factor_flag and local_global_trans_factor_flag cannot be set at the same time."
        self.ctx_fixed_location_interval = ctx_fixed_location_interval
        if self.ctx_fixed_location_interval > 0:
            logging.warning(
                "ctx_fixed_location_L is set to {}".format(
                    self.ctx_fixed_location_interval
                )
            )
            # ! overwrite the datset
            grid = determine_ctx_fixed_grid_size(self.max_P)
            self.ctx_loc = []
            for i in range(grid[0]):
                for j in range(grid[1]):
                    self.ctx_loc.append([i, j])
            self.ctx_loc = torch.tensor(self.ctx_loc) * self.ctx_fixed_location_interval
            self.ctx_loc[:, 0] = self.ctx_loc[:, 0] - self.ctx_loc[:, 0].mean()
            self.ctx_loc[:, 1] = self.ctx_loc[:, 1] - self.ctx_loc[:, 1].mean()
            # ! this is the x bias!
            self.ctx_loc[:, 0] = self.ctx_loc[:, 0] + float(ctx_fixed_location_bias)
            assert (
                not self.center_ctx_flag
            ), "center_ctx_flag is not supported when ctx_fixed_location_interval is set."

        # config different embodiments
        self.embodiment_cfg = {}
        for k, d in embodiment_dirs.items():
            if k in ["smplx"]:  # HUMAN
                # todo: add support of smpl24
                sem = deepcopy(
                    [SEM_HUMAN] * 25 + [SEM_LEFT_MANO] * 15 + [SEM_RIGHT_MANO] * 15
                )
                if self.local_global_factor_flag or self.local_global_trans_factor_flag:
                    sem[0] = SEM_HUMAN_GLOBAL  # fix
                if self.remap_hands_to_mano_sem:
                    # Remap finger joints to TACO-compatible semantic IDs
                    # SMPL-X 55: 0-24 body (incl. wrists at 20,21), 25-39 left MANO, 40-54 right MANO
                    # Wrists (idx 20,21) stay SEM_HUMAN — shared PoseIO with body
                    # Fingers get separate SEM for shared PoseIO with TACO fingers
                    for i in range(25, 40):
                        sem[i] = SEM_LEFT_MANO_FINGERS
                    for i in range(40, 55):
                        sem[i] = SEM_RIGHT_MANO_FINGERS
                self.embodiment_cfg[k] = {
                    "type": "human",
                    "state_dirname": d,
                    "uid": UID_SMPLX55_GROUP,
                    "sem": sem,
                }
            elif k in ["g1", "h12", "t1"]:
                sem = deepcopy([SEM_HUMANOID] * len(HUMANOID_LINK_GROUP_DICT[k]))
                if self.local_global_factor_flag or self.local_global_trans_factor_flag:
                    sem[0] = SEM_HUMANOID_GLOBAL
                self.embodiment_cfg[k] = {
                    "type": "humanoid",
                    "state_dirname": d,
                    "link_names": HUMANOID_LINK_GROUP_DICT[k],
                    "uid": UID_HUMANOID_GROUP_DICT[k],
                    "sem": sem,
                }
        logging.info(f"HH dataset has embodiment:\n{self.embodiment_cfg}")

        # group the sequence
        # todo: support more index

        raw_indices = json.load(open(split_fn, "r"))
        grouped_indices = {}
        # for each sequence, which person it has, which embodiment it has
        for name, length in tqdm(raw_indices):
            length = int(length)
            sid, pid = id_p_split(name)
            ava_emb = []
            for k, cfg in self.embodiment_cfg.items():
                fn = osp.join(self.data_root, cfg["state_dirname"], name + ".npz")
                if osp.exists(fn):
                    ava_emb.append(k)
            ava_emb = set(ava_emb)
            if len(ava_emb) == 0:
                continue
            if sid in grouped_indices:
                grouped_indices[sid][0][pid] = ava_emb
            else:
                grouped_indices[sid] = [{pid: ava_emb}, length]
        self.grouped_indices = grouped_indices
        self.__summarize_grouped_indices__()

        self.sid_len_dict = {k: v[1] for k, v in grouped_indices.items()}
        self.indices, self.seq_len = load_indices(
            [(k, str(v[1])) for k, v in grouped_indices.items()], self.t_rate
        )
        return

    def __summarize_grouped_indices__(self, width=50):
        """Log a summary of grouped indices (persons and embodiments per crop).

        Args:
            width: Character width for the log separator lines.

        Example::

            ds.__summarize_grouped_indices__()
        """
        num_frames = [v[1] for v in self.grouped_indices.values()]
        total_frames = sum(num_frames)
        logging.info("=" * width)
        logging.info(
            f"HH dataset has {len(self.grouped_indices)} crops, total {total_frames} unique time steps"
        )
        logging.info("-" * width)
        n_humans = [len(v[0]) for v in self.grouped_indices.values()]
        unique_n, counts = np.unique(n_humans, return_counts=True)
        for n, c in zip(unique_n, counts):
            logging.info(
                f"  - {c}/{len(self.grouped_indices)} ({float(c) / len(self.grouped_indices) * 100:.2f}%) crops have {n} persons"
            )
        logging.info("-" * width)
        n_embodiments = []
        for v in self.grouped_indices.values():
            for vv in v[0].values():
                n_embodiments.append(len(vv))
        unique_n, counts = np.unique(n_embodiments, return_counts=True)
        for n, c in zip(unique_n, counts):
            logging.info(
                f"  - {c}/{len(n_embodiments)} ({float(c) / len(n_embodiments) * 100:.2f}%) trajs have {n} embodiments"
            )
        logging.info("=" * width)
        return

    def __decide_sample_trajs__(self, seq_id, dataset_idx):
        """Decide which persons and embodiments to include in a sample.

        Applies ``max_robot_per_person``, ``max_P``, ``max_G`` caps and
        optional human-biased sampling.

        Args:
            seq_id: Scene/sequence identifier.
            dataset_idx: Global index for deterministic fallback.

        Returns:
            Dict ``{pid: set(embodiment_keys)}`` for each selected person.

        Example::

            pid_emb = ds.__decide_sample_trajs__("scene01", 0)
        """
        # decide which person and embodiment to sample
        grouped = deepcopy(self.grouped_indices[seq_id][0])

        # ! constrain the grouped
        # todo: now this is still random if hte dataset is nto random
        if self.max_robot_per_person > 0:
            for k, v in grouped.items():
                new_v = deepcopy(v)
                new_v.discard("smplx")
                new_v.discard("smpl")
                if len(new_v) > self.max_robot_per_person:
                    humanset = grouped[k] - new_v
                    new_v = np.random.choice(
                        list(new_v), self.max_robot_per_person, replace=True
                    )
                    grouped[k] = set(new_v) | humanset

        person_pool = list(grouped.keys())
        if len(person_pool) > self.max_P:
            if self.random_flag:
                person_pool = np.random.choice(person_pool, self.max_P, replace=False)
            else:
                step = max(1, len(person_pool) // (1 + self.max_P))
                start = dataset_idx % step
                person_pool = person_pool[start::step]

        # human biased sampling, there are some chance the dataset will first gaurentee to sample all smplx, then try to sample other robots
        seed = (
            np.random.rand()
        )  # ! FOR NOW THIS WILL BREAK THE DETERMINISTIC OF NON-RANDOM DATASET
        if seed < self.human_sample_bias_prob:
            human_pool = [
                (pid, emb)
                for pid in person_pool
                for emb in grouped[pid]
                if emb in ["smplx"]
            ]
            other_pool = [
                (pid, emb)
                for pid in person_pool
                for emb in grouped[pid]
                if emb not in ["smplx"]
            ]
            if len(other_pool) > self.max_G - len(human_pool):
                if self.random_flag:
                    choice = np.random.choice(
                        len(other_pool), self.max_G - len(human_pool), replace=False
                    )
                    _pool = [other_pool[i] for i in choice]
                    other_pool = _pool
                else:
                    step = max(1, len(other_pool) // (1 + self.max_G - len(human_pool)))
                    start = dataset_idx % step
                    other_pool = other_pool[start::step]
            pool = human_pool + other_pool
        else:
            # fully random sampling
            pool = [(pid, emb) for pid in person_pool for emb in grouped[pid]]
            if len(pool) > self.max_G:
                if self.random_flag:
                    choice = np.random.choice(len(pool), self.max_G, replace=False)
                    _pool = [pool[i] for i in choice]
                    pool = _pool
                else:
                    step = max(1, len(pool) // (1 + self.max_G))
                    start = dataset_idx % step
                    pool = pool[start::step]

        # get to structure list
        ret = {}
        for it in pool:
            pid, emb = it
            if pid in ret:
                ret[pid].add(emb)
            else:
                ret[pid] = set([emb])
        return ret

    def __decide_sample_time__(self, seq_id, current_abs_tid, dataset_idx):
        """Decide the temporal window for a sample.

        Args:
            seq_id: Sequence identifier.
            current_abs_tid: Absolute anchor frame.
            dataset_idx: Global index for deterministic offset.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)``.

        Example::

            cur_i, s, e = ds.__decide_sample_time__("scene01", 15, 0)
        """
        factor = int(self.t_rate)
        seq_len = self.seq_len[seq_id]

        if self.random_flag:
            left_len = np.random.randint(0, self.max_T * factor)
        else:
            left_len = dataset_idx % (self.max_T * factor)
        start_abs_tid = current_abs_tid - left_len

        end_abs_tid = start_abs_tid + self.max_T * factor
        start_abs_tid = max(0, start_abs_tid)
        end_abs_tid = min(end_abs_tid, seq_len)

        current_ith = (current_abs_tid - start_abs_tid) // factor
        return current_ith, start_abs_tid, end_abs_tid

    def __find_sid_person_with_emb__(self, emb_set, idx, N):
        """Find up to N (sid, pid) pairs whose embodiment set covers ``emb_set``.

        Args:
            emb_set: Required set of embodiment keys.
            idx: Dataset index (unused, reserved for deterministic mode).
            N: Maximum number of results.

        Returns:
            List of ``(sid, pid)`` tuples (length <= N).

        Example::

            pool = ds.__find_sid_person_with_emb__({"smplx", "g1"}, 0, 5)
        """
        ret = []
        sids = list(self.grouped_indices.keys())
        if self.random_flag:
            np.random.shuffle(sids)
        else:
            # ! for now still shuffle this!!!!! the eval dataset is still random for ctx!!!!
            np.random.shuffle(sids)
            # for sid, it in self.grouped_indices.items():
        for sid in sids:
            it = self.grouped_indices[sid]
            for pid, emb in it[0].items():
                # if emb_set is a subset of emb
                if emb_set.issubset(emb):
                    ret.append((sid, pid))
                    if len(ret) >= N:
                        return ret
        return ret

    def __decide_context_sampling__(self, pid_emb_dict, idx):
        """Sample context sequences for each person/embodiment group.

        For each person, finds sequences that share the same embodiment set
        and samples time steps.

        Args:
            pid_emb_dict: Dict ``{pid: set(embodiment_keys)}``.
            idx: Dataset index.

        Returns:
            Tuple ``(ret_sid_pid, ret_abs_tid)`` where each is a list of
            lists (one per person in ``pid_emb_dict``).

        Example::

            ctx_sp, ctx_t = ds.__decide_context_sampling__(pid_emb, 0)
        """
        # return should be the same len as pid_emb_dict, each element has len C
        ret_sid_pid, ret_abs_tid = [], []

        for pid, emb_set in pid_emb_dict.items():
            pool = self.__find_sid_person_with_emb__(emb_set, idx, self.max_C)
            time_step = []
            for sid, _ in pool:
                length = self.sid_len_dict[sid]
                if self.random_flag:
                    time_step.append(np.random.randint(0, length))
                else:
                    time_step.append(idx % length)
            ret_sid_pid.append(pool)
            ret_abs_tid.append(time_step)
        return ret_sid_pid, ret_abs_tid

    def __load_human__(
        self, human_data, start_tid, end_tid, cur_rel_idx, factor, center_flag
    ):
        """Load SMPL-X human joint poses from a numpy archive.

        Adds global translation to joint rotations and optionally centers
        around the pelvis (joint 0) at the anchor frame.

        Args:
            human_data: Loaded ``.npz`` dict with ``Q``, ``global_t``, etc.
            start_tid: Start frame (absolute).
            end_tid: End frame (absolute).
            cur_rel_idx: Relative anchor index for centering.
            factor: Temporal stride.
            center_flag: If True, center XY on pelvis at ``cur_rel_idx``.

        Returns:
            Tuple ``(Q, pose, beta, model_type)`` where ``Q`` is
            ``(T, 55, 4, 4)``.

        Example::

            Q, pose, beta, mt = ds.__load_human__(data, 0, 30, 5, 1, True)
        """
        human_mask = human_data["valid_mask"][:, start_tid:end_tid][:, ::factor]
        Q = human_data["Q"][:, start_tid:end_tid][:, ::factor]  # H,T,K,4,4
        global_t = human_data["global_t"][:, start_tid:end_tid][:, ::factor]
        global_t = global_t[:, :, None].repeat(Q.shape[2], axis=2)
        Q[:, :, :, :3, 3] = Q[:, :, :, :3, 3] + global_t
        # ! center with 0 joint at current time
        if center_flag:
            cur_t = Q[:, cur_rel_idx, 0, :2, 3]
            Q[..., :2, 3] = Q[..., :2, 3] - cur_t[:, None, None, :]
        pose = human_data["pose"][:, start_tid:end_tid][:, ::factor]
        beta = human_data["betas"]
        model_type = str(human_data["model_type"])
        # ! only support single human in one file, this is the old left behavor, now mult human are in different files
        assert len(Q) == 1
        Q = Q[0]  # T,N,4,4
        pose = pose[0]
        beta = beta[0]
        return Q, pose, beta, model_type

    @staticmethod
    def localize_hand(Q: torch.Tensor) -> torch.Tensor:
        """Express MANO hand joints in their respective wrist-local frames.

        Left hand joints (indices 25-39) are expressed relative to left
        wrist (index 20), right hand (40-54) relative to right wrist (21).

        Args:
            Q: ``(55, T, 4, 4)`` SMPL-X joint transforms.

        Returns:
            ``(55, T, 4, 4)`` tensor with hand joints localized.

        Example::

            Q_local = MultiHHDataset.localize_hand(Q)
        """
        # Q: N,T,4,4
        assert len(Q) == 55
        Qd = Q.dtype
        Q = Q.clone().float()
        l_wrist = Q[20].clone()
        r_wrist = Q[21].clone()
        l_wrist_inv = torch.inverse(l_wrist)
        r_wrist_inv = torch.inverse(r_wrist)
        l_hand = Q[25:40].clone()
        r_hand = Q[40:55].clone()
        l_hand_local = torch.einsum("tij,ntjk->ntik", l_wrist_inv, l_hand)
        r_hand_local = torch.einsum("tij,ntjk->ntik", r_wrist_inv, r_hand)
        Q[25:40] = l_hand_local.clone()
        Q[40:55] = r_hand_local.clone()
        return Q.to(dtype=Qd)

    def __add_human_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
        ctx_sids,
        ctx_abs_tids,
        emb,
        ith_human,  # for ctx fixed location
    ):
        """Append a human (SMPL-X) embodiment trace to the prompt.

        Handles centering, local/global factorisation, hand localisation,
        and fixed-grid context placement.

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Person-level sequence filename (without extension).
            ctx_sids: Context sequence IDs.
            ctx_abs_tids: Context absolute time indices.
            emb: Embodiment key (e.g. ``"smplx"``).
            ith_human: Person index for fixed-grid placement.

        Returns:
            Tuple ``(prompt, extra_dict, group_id)`` where ``group_id``
            is -1 if the data file is missing.

        Example::

            prompt, extra, gid = ds.__add_human_prompt__(
                prompt, 5, 0, 30, "scene01__________p0",
                ctx_sids, ctx_tids, "smplx", 0,
            )
        """
        # ! ASSUME THAT THE HUMAN WILL ALWAYS EXSIST, ONLY HUMANOID WILL MISS
        extra_dict = {}

        emb_cfg = self.embodiment_cfg[emb]
        state_dir = emb_cfg["state_dirname"]
        human_uids = emb_cfg["uid"]
        human_sem = emb_cfg["sem"]

        data_fn = osp.join(self.data_root, state_dir, seq_id + ".npz")
        if not osp.exists(data_fn):
            return prompt, {"beta": np.zeros((1, 10)), "human_uids": human_uids}, -1

        human_data = np.load(data_fn)
        factor = int(self.t_rate)
        Q, pose, beta, model_type = self.__load_human__(
            human_data, start_tid, end_tid, cur_rel_idx, factor, self.center_real_flag
        )  # T,55,4,4; T,164; 10
        Q = torch.as_tensor(Q).permute(1, 0, 2, 3).contiguous()  # 55,T,4,4
        # Z-rotation must apply BEFORE localize_hand (all joints in global frame)
        if self._z_rot_Rz is not None:
            Q = self.apply_z_rotation(Q, self._z_rot_Rz)
        if self.local_global_factor_flag:
            _ref_Q = Q[:1].float()
            _ref_Q_inv = torch.inverse(_ref_Q)
            _local_Q = Q[1:].float()
            _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q_inv, _local_Q)
            Q = torch.cat([_ref_Q, _local_Q], dim=0)
        elif self.local_global_trans_factor_flag:
            _ref_Q = Q[:1].float()
            _ref_Q_t = _ref_Q[..., :3, 3]
            _local_Q = Q[1:].float()
            _local_Q[..., :3, 3] = _local_Q[..., :3, 3] - _ref_Q_t
            Q = torch.cat([_ref_Q, _local_Q], dim=0)
        if self.hand_attached_to_wrist_flag or self.wrist_local_coords:
            # Q: 55,T,4,4 → hands become wrist-local
            Q = self.localize_hand(Q.clone())

        extra_dict["pose"] = pose
        extra_dict["beta"] = beta
        extra_dict["human_model_type"] = model_type

        ctx_Q = []
        for sid, abs_tid in zip(ctx_sids, ctx_abs_tids):
            if not osp.exists(osp.join(self.data_root, state_dir, sid + ".npz")):
                continue
            ctx_data = np.load(osp.join(self.data_root, state_dir, sid + ".npz"))
            _Q, _, _, _ = self.__load_human__(
                ctx_data,
                abs_tid,
                abs_tid + 1,
                cur_rel_idx=0,
                factor=1,
                center_flag=self.center_ctx_flag,
            )
            ctx_Q.append(torch.as_tensor(_Q))
        assert len(ctx_Q) > 0
        ctx_Q = torch.cat(ctx_Q, dim=0).permute(1, 0, 2, 3).contiguous()  # 24,C,4,4

        if self.ctx_fixed_location_interval > 0:
            x_loc = self.ctx_loc[ith_human]
            ctx_Q[..., :2, 3] = (
                ctx_Q[..., :2, 3]
                - ctx_Q[:1, :, :2, 3]
                + torch.as_tensor(x_loc)[None, None]
            )

        # Z-rotation must apply BEFORE localize_hand (all joints in global frame)
        if self._z_rot_Rz is not None:
            ctx_Q = self.apply_z_rotation(ctx_Q, self._z_rot_Rz)
        if self.local_global_factor_flag:
            _ref_Q = ctx_Q[:1].float()
            _ref_Q_inv = torch.inverse(_ref_Q)
            _local_Q = ctx_Q[1:].float()
            _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q_inv, _local_Q)
            ctx_Q = torch.cat([_ref_Q, _local_Q], dim=0)
        elif self.local_global_trans_factor_flag:
            _ref_Q = ctx_Q[:1].float()
            _ref_Q_t = _ref_Q[..., :3, 3]
            _local_Q = ctx_Q[1:].float()
            _local_Q[..., :3, 3] = _local_Q[..., :3, 3] - _ref_Q_t
            ctx_Q = torch.cat([_ref_Q, _local_Q], dim=0)
        if self.hand_attached_to_wrist_flag or self.wrist_local_coords:
            ctx_Q = self.localize_hand(ctx_Q)

        group_id = prompt.max_group_id + 1
        prompt.append_trace(
            trace=Q,
            trace_mask=torch.ones(Q.shape[:2], dtype=torch.bool),
            context=ctx_Q,
            context_mask=torch.ones(ctx_Q.shape[:2], dtype=torch.bool),
            group_id=group_id,
            sem_type=human_sem,
            uid=human_uids,
        )
        extra_dict["human_uids"] = human_uids
        return prompt, extra_dict, group_id

    def __load_humanoid__(
        self,
        g1_data,
        start_tid,
        end_tid,
        cur_rel_idx,
        factor,
        humanoid_link_names,
        center_flag,
    ):
        """Load humanoid link poses from a numpy archive.

        Selects only the links listed in ``humanoid_link_names`` and
        optionally centers XY around the root at ``cur_rel_idx``.

        Args:
            g1_data: Loaded ``.npz`` dict with ``Q`` and ``names``.
            start_tid: Start frame (absolute).
            end_tid: End frame (absolute).
            cur_rel_idx: Anchor index for centering.
            factor: Temporal stride.
            humanoid_link_names: Ordered list of link names to select.
            center_flag: Center XY on root link at anchor.

        Returns:
            ``(T, M, 4, 4)`` numpy array of link poses.

        Example::

            Q = ds.__load_humanoid__(data, 0, 30, 5, 1, names, True)
        """
        Q = g1_data["Q"][start_tid:end_tid][::factor]
        names = g1_data["names"].tolist()
        sample_ids = [names.index(name) + 1 for name in humanoid_link_names]
        # robot has the first link always equal to root!
        Q = Q[:, sample_ids]
        if center_flag:
            cur_trans = Q[cur_rel_idx, 0, :2, 3]
            Q[:, :, :2, 3] = Q[:, :, :2, 3] - cur_trans[None, None]
        return Q

    def __add_humanoid_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,  # this is inside the slice seq
        start_tid,  # this is global how to slice
        end_tid,  # this is global how to slice
        seq_id,
        ctx_sids,
        ctx_abs_tids,
        emb,
        ith_human,  # for ctx fixed location
    ):
        """Append a humanoid embodiment trace to the prompt.

        If the data file is missing, appends identity poses with a zero
        mask so the model can still produce output (hallucination mode).

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Person-level sequence filename (without extension).
            ctx_sids: Context sequence IDs.
            ctx_abs_tids: Context absolute time indices.
            emb: Embodiment key (e.g. ``"g1"``).
            ith_human: Person index for fixed-grid placement.

        Returns:
            Tuple ``(prompt, extra_dict, group_id)``.

        Example::

            prompt, extra, gid = ds.__add_humanoid_prompt__(
                prompt, 5, 0, 30, "scene01__________p0",
                ctx_sids, ctx_tids, "g1", 0,
            )
        """
        # ! EVEN IF MISSING, STILL APPEND, CAN HALLUCINATE!!!!
        # todo: the current sampling will nto sample empty humanoid!! can enable this later!!!! to see the behavior on failure retargeting of oracle

        emb_cfg = self.embodiment_cfg[emb]
        state_dir = emb_cfg["state_dirname"]
        humanoid_link_names = emb_cfg["link_names"]
        humanoid_uids = emb_cfg["uid"]
        humanoid_sem = emb_cfg["sem"]

        extra_dict = {}

        data_fn = osp.join(self.data_root, state_dir, seq_id + ".npz")

        factor = int(self.t_rate)
        if osp.exists(data_fn):
            r_data = np.load(data_fn)
            Q = self.__load_humanoid__(
                r_data,
                start_tid,
                end_tid,
                cur_rel_idx,
                factor,
                humanoid_link_names,
                self.center_real_flag,
            )
            Q = torch.as_tensor(Q).permute(1, 0, 2, 3).contiguous()  # N,T,4,4
            if self.local_global_factor_flag:
                _ref_Q = Q[:1].float()
                _ref_Q_inv = torch.inverse(_ref_Q)
                _local_Q = Q[1:].float()
                _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q_inv, _local_Q)
                Q = torch.cat([_ref_Q, _local_Q], dim=0)
            elif self.local_global_trans_factor_flag:
                _ref_Q = Q[:1].float()
                _ref_Q_t = _ref_Q[..., :3, 3]
                _local_Q = Q[1:].float()
                _local_Q[..., :3, 3] = _local_Q[..., :3, 3] - _ref_Q_t
                Q = torch.cat([_ref_Q, _local_Q], dim=0)
            # Z-rotation for humanoid trace (all global frame, no localize_hand)
            if self._z_rot_Rz is not None:
                Q = self.apply_z_rotation(Q, self._z_rot_Rz)
            trace_mask = torch.ones(Q.shape[:2], dtype=torch.bool)
        else:
            Q = torch.eye(4, dtype=torch.float32)[None, None].expand(
                len(humanoid_link_names), end_tid - start_tid, -1, -1
            )
            trace_mask = torch.zeros(Q.shape[:2], dtype=torch.bool)

        ctx_Q = []
        for sid, abs_tid in zip(ctx_sids, ctx_abs_tids):
            if not osp.exists(osp.join(self.data_root, state_dir, sid + ".npz")):
                continue
            ctx_data = np.load(osp.join(self.data_root, state_dir, sid + ".npz"))
            _Q = self.__load_humanoid__(
                ctx_data,
                abs_tid,
                abs_tid + 1,
                cur_rel_idx=0,
                factor=1,
                humanoid_link_names=humanoid_link_names,
                center_flag=self.center_ctx_flag,
            )
            ctx_Q.append(torch.as_tensor(_Q))

        if len(ctx_Q) > 0:
            ctx_Q = torch.cat(ctx_Q, dim=0).permute(1, 0, 2, 3).contiguous()  # 24,C,4,4
            if self.ctx_fixed_location_interval > 0:
                x_loc = self.ctx_loc[ith_human]
                ctx_Q[..., :2, 3] = (
                    ctx_Q[..., :2, 3]
                    - ctx_Q[:1, :, :2, 3]
                    + torch.as_tensor(x_loc)[None, None]
                )

            ctx_mask = torch.ones(ctx_Q.shape[:2], dtype=torch.bool)
            if self.local_global_factor_flag:
                _ref_Q = ctx_Q[:1].float()
                _ref_Q_inv = torch.inverse(_ref_Q)
                _local_Q = ctx_Q[1:].float()
                _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q_inv, _local_Q)
                ctx_Q = torch.cat([_ref_Q, _local_Q], dim=0)
            elif self.local_global_trans_factor_flag:
                _ref_Q = ctx_Q[:1].float()
                _ref_Q_t = _ref_Q[..., :3, 3]
                _local_Q = ctx_Q[1:].float()
                _local_Q[..., :3, 3] = _local_Q[..., :3, 3] - _ref_Q_t
                ctx_Q = torch.cat([_ref_Q, _local_Q], dim=0)
            # Z-rotation for humanoid context
            if self._z_rot_Rz is not None:
                ctx_Q = self.apply_z_rotation(ctx_Q, self._z_rot_Rz)
            ctx_mask = torch.ones(ctx_Q.shape[:2], dtype=torch.bool)
        else:
            ctx_Q = torch.eye(4, dtype=torch.float32)[None, None].expand(
                len(humanoid_link_names), self.max_C, -1, -1
            )
            ctx_mask = torch.zeros(ctx_Q.shape[:2], dtype=torch.bool)

        group_id = prompt.max_group_id + 1
        prompt.append_trace(
            trace=Q,
            trace_mask=trace_mask,
            context=ctx_Q,
            context_mask=ctx_mask,
            group_id=group_id,
            sem_type=humanoid_sem,
            uid=humanoid_uids[: len(Q)],
        )
        extra_dict["humanoid_link_names"] = humanoid_link_names
        extra_dict["humanoid_uids"] = humanoid_uids[: len(Q)]
        extra_dict["humanoid_type"] = emb
        return prompt, extra_dict, group_id

    @staticmethod
    def make_z_rotation_matrix(theta: float) -> torch.Tensor:
        """Build a 4x4 z-axis rotation matrix.

        Args:
            theta: Rotation angle in radians.

        Returns:
            ``(4, 4)`` float32 homogeneous rotation matrix.
        """
        Rz = torch.eye(4, dtype=torch.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Rz[0, 0] = cos_t
        Rz[0, 1] = -sin_t
        Rz[1, 0] = sin_t
        Rz[1, 1] = cos_t
        return Rz

    @staticmethod
    def apply_z_rotation(Q: torch.Tensor, Rz: torch.Tensor) -> torch.Tensor:
        """Apply z-rotation to SE(3) matrices in global frame.

        Applied BEFORE localize_hand so all joints are still in global frame.

        Args:
            Q: SE(3) poses, shape ``(..., 4, 4)``.
            Rz: ``(4, 4)`` rotation matrix.

        Returns:
            Rotated poses, same shape and dtype as input.
        """
        orig_dtype = Q.dtype
        return torch.einsum("ij,...jk->...ik", Rz, Q.float()).to(orig_dtype)

    def __getitem_inner__(self, idx):
        """Build a single multi-person HH training sample.

        Samples persons, embodiments, time window, and context; then
        appends human and humanoid prompts for each selected embodiment.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_dict_all)`` where
            ``extra_dict_all`` maps group IDs to per-embodiment extras.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)
        """

        idx = idx % len(self)
        # Pre-compute z-rotation matrix for this sample (shared across all bodies)
        if self.z_rot_aug and self.random_flag:
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            self._z_rot_Rz = self.make_z_rotation_matrix(theta)
        else:
            self._z_rot_Rz = None
        prompt = PromptV0(P=1, T=self.max_T, C=self.max_C)
        sid, current_abs_tid = self.indices[idx]

        current_ith, start_abs_tid, end_abs_tid = self.__decide_sample_time__(
            sid, current_abs_tid, idx
        )
        pid_emb_dict = self.__decide_sample_trajs__(sid, idx)
        ctx_sid_pid_list, ctx_abs_tid_list = self.__decide_context_sampling__(
            pid_emb_dict, idx
        )
        all_pids = list(set(pid_emb_dict.keys()))
        if len(all_pids) < self.max_P:
            all_pids = all_pids + [None] * (self.max_P - len(all_pids))
        if self.random_flag:
            random.shuffle(all_pids)  # the ith is random

        # ! use same retargeting and sync to id which is which
        extra_dict_all = {"gid": []}  # gid: extra
        for pid, ctx_sid_pid, ctx_abs_tids in zip(
            pid_emb_dict.keys(), ctx_sid_pid_list, ctx_abs_tid_list
        ):
            ctx_sids = [get_fn(*it) for it in ctx_sid_pid]
            ith_human = all_pids.index(pid)
            for emb in pid_emb_dict[pid]:
                if emb in ["smplx"]:
                    prompt, extra_dict, gid = self.__add_human_prompt__(
                        prompt,
                        current_ith,
                        start_abs_tid,
                        end_abs_tid,
                        get_fn(sid, pid),
                        ctx_sids,
                        ctx_abs_tids,
                        emb,
                        ith_human,
                    )

                else:
                    prompt, extra_dict, gid = self.__add_humanoid_prompt__(
                        prompt,
                        current_ith,
                        start_abs_tid,
                        end_abs_tid,
                        get_fn(sid, pid),
                        ctx_sids,
                        ctx_abs_tids,
                        emb,
                        ith_human,
                    )
                extra_dict_all["gid"].append(gid)
                for key in extra_dict.keys():
                    if key not in extra_dict_all:
                        extra_dict_all[key] = [None] * len(extra_dict_all["gid"])
                        extra_dict_all[key][-1] = extra_dict[key]
                    else:
                        extra_dict_all[key].append(extra_dict[key])
                for key in extra_dict_all.keys():
                    if len(extra_dict_all[key]) != len(extra_dict_all["gid"]):
                        extra_dict_all[key].append(None)

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit
        if self.language_data is not None:
            text = self.language_data[sid]
        else:
            text = ""

        dummy_pcl, dummy_pcl_mask = prompt.cat_pcl()
        ret = CanonicalDatasetOutput(
            seq_name=sid,
            trace=trace,
            trace_mask=trace_mask,
            context=context,
            context_mask=context_mask,
            trace_group_id=group_id,
            trace_rot_mask=rot_mask,
            trace_sem_type=sem_type,
            trace_uid=uid,
            time=time_stamp,
            text=text,
            local_pcl=dummy_pcl,
            local_pcl_mask=dummy_pcl_mask,
        )

        return ret, extra_dict_all


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # dataset = MultiHHDataset(
    #     data_root="../../data/embody3d_phuma_processed/hand_duo_subset/",
    #     split_fn=osp.abspath("../../data_splits/embody3d_humanoid_v2/debug_train.json"),
    #     base_fps=30,
    #     # random_flag=False,
    #     safe_flag=False,
    #     max_T=30,
    #     max_C=5,
    #     max_P=4,
    #     max_G=8,
    #     embodiment_dirs={
    #         "smplx": "human_states",
    #         "g1": "g1_states",
    #         "h12": "h12_states",
    #         "t1": "t1_states",
    #     },
    # )

    dataset = MultiHHDataset(
        # data_root="../../data/embody3d_phuma_processed/g1",
        # split_fn=osp.abspath("../../data_splits/embody3d_humanoid_v2/full_train.json"),
        data_root="../../../../data/embody3d_phuma_processed/hand_duo_subset/",
        split_fn=osp.abspath(
            "../../../../data_splits/embody3d_humanoid_v2/debug_train.json"
        ),
        base_fps=30,
        # random_flag=False,
        safe_flag=False,
        max_T=30,
        max_C=5,
        max_P=4,
        max_G=8,
        embodiment_dirs={
            "smplx": "human_states",
            "g1": "g1_states",
            # "h12": "h12_states",
            # "t1": "t1_states",
        },
        human_sample_bias_prob=1.0,
        max_robot_per_person=1,
        # local_global_factor_flag=True,
        local_global_trans_factor_flag=False,
        ctx_fixed_location_interval=1.0,
        hand_attached_to_wrist_flag=True,
    )

    # dataset = HumanHumanoidDataset(
    #     data_root="../../data/phuma_lmm_dataset_h1_2",
    #     split_fn=osp.abspath("../../data_splits/h1_2/phuma_debug_train.json"),
    #     base_fps=30,
    #     # random_flag=False,
    #     safe_flag=False,
    #     max_T=30,
    #     max_C=5,
    #     humanoid_state_dirname="h12_states",
    #     humanoid_type="h12",
    # )

    # dataset = HumanHumanoidDataset(
    #     data_root="../../data/phuma_lmm_dataset",
    #     split_fn=osp.abspath("../../data_splits/g1/phuma_train.json"),
    #     base_fps=30,
    #     # random_flag=False,
    #     safe_flag=False,
    #     max_T=20,
    #     max_C=5,
    # )

    dataset[10]

    from torch.utils.data import DataLoader

    # dataset2 = HumanHumanoidDataset(
    #     data_root="../../data/phuma_lmm_dataset",
    #     split_fn=osp.abspath("../../data_splits/g1/phuma_train.json"),
    #     base_fps=30,
    #     # random_flag=False,
    #     safe_flag=False,
    #     max_T=20,
    #     max_C=5,
    # )

    # TODO: check with two different dataset

    cat_dataset = ConcatDataset([dataset, dataset])

    dataloader = DataLoader(
        cat_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        collate_fn=CanonicalDatasetOutput.collate,
    )

    for i, batch in tqdm(enumerate(dataloader)):
        print(batch["trace"].shape)
        # print(batch["extra"]["humanoid_link_names"])

    # dataset.print_profiling_summary()
    print(f"Dataset length: {len(dataset)}")
