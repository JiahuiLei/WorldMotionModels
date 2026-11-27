"""ManipTrans dexterous manipulation dataset.

Provides multi-link hand poses, manipulated-object trajectories, and
background scene objects from the ManipTrans benchmark.  Meshes are loaded
from a single ``meshes.npz`` archive and cached lazily.

Example::

    dataset = ManipTransDataset(
        data_root="data/maniptrans_favor",
        split_fn="data_splits/maniptrans_favor/train.json",
        base_fps=30,
        max_T=30,
        max_C=5,
    )
    sample, extra = dataset[0]
"""

################################################################################
# GeneMoma Dataset - Unified dataset for general motion modeling
################################################################################

import os
import os.path as osp
import sys
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import trimesh
from copy import deepcopy

from umm.datasets.umm_e10.base_dataset import (
    BaseDataset,
    PromptV0,
    CanonicalDatasetOutput,
)
from umm.datasets.umm_e10.dataset_utils import sample_list
from umm.datasets.umm_e10.dataset_config import *


class ManipTransDataset(BaseDataset):
    """Dataset for ManipTrans dexterous hand manipulation sequences.

    Loads multi-link hand poses (left/right Inspire hands), manipulated
    objects, and static scene objects.  Meshes are lazily read from a
    single ``meshes.npz`` archive and cached per mesh ID.

    Args:
        data_root: Root directory of the ManipTrans dataset.
        split_fn: JSON split file.
        base_fps: Recording frame rate.
        random_flag: Stochastic augmentation.
        safe_flag: Error-handling wrapper.
        repeat: Repetition multiplier.
        t_rate: Temporal stride.
        language_fn: Optional language annotation file.
        max_T: Maximum time steps.
        max_C: Maximum context steps.
        max_P_per_object: Points per manipulated-object cloud.
        max_P_per_background: Points per background-object cloud.
        cache_all_data: Pre-load all sequences and meshes into RAM.

    Example::

        ds = ManipTransDataset(
            data_root="data/maniptrans_favor",
            split_fn="splits/train.json",
            base_fps=30,
            max_T=30,
            max_C=5,
        )
        sample, extra = ds[0]
    """

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps,  # each dataset must set manually
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
        language_fn: str | None = None,
        #############################################################
        max_T: int = 30,  # num of time step
        max_C: int = 8,  # num of context steps
        max_P_per_object: int = 32,  # num of prompts per object
        max_P_per_background: int = 10,  # num of prompts per background
        #############################################################
        cache_all_data: bool = False,
    ):
        """Initialize the ManipTrans dataset.

        Optionally caches all sequences and meshes to RAM.

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
            max_P_per_object: Points per object.
            max_P_per_background: Points per background object.
            cache_all_data: Pre-load everything to RAM.

        Example::

            ds = ManipTransDataset(
                data_root="data/maniptrans_favor",
                split_fn="splits/train.json",
                base_fps=30,
                cache_all_data=True,
            )
        """
        super().__init__(
            data_root, split_fn, base_fps, language_fn, random_flag, safe_flag, repeat
        )

        self.max_T = max_T
        self.max_C = max_C
        self.max_P_per_object = max_P_per_object
        self.max_P_per_background = max_P_per_background

        self.cache_all_data = cache_all_data
        self.cached_sequences = {}

        # Load mesh archive
        self.mesh_archive = None
        self.mesh_registry = {}

        if self.cache_all_data:
            print("Caching all data to RAM...")
            self.init_mesh_archive()
            # Cache all sequences
            for sid, length in tqdm(self.seq_len.items(), desc="Caching sequences"):
                data_path = os.path.join(self.data_root, sid)
                self.cached_sequences[sid] = dict(np.load(data_path))
            print("Done caching.")
        else:
            # Lazy init will happen in get_mesh or __getitem__
            pass

        return

    def init_mesh_archive(self):
        """Load the ``meshes.npz`` archive into RAM and clear the per-ID cache.

        Called automatically on first ``get_mesh`` access or during
        ``__init__`` when ``cache_all_data=True``.

        Example::

            ds.init_mesh_archive()
        """
        mpath = os.path.join(self.data_root, "meshes.npz")
        # User requested to ALWAYS cache meshes in RAM as a deep copy (dict)
        if os.path.exists(mpath):
            print(f"Loading Mesh Archive to RAM: {mpath}")
            # Load everything into dict immediately
            self.mesh_archive = dict(np.load(mpath))
            self.mesh_registry = {}  # Clear cache
        else:
            print("Warning: meshes.npz not found")

    def get_mesh(self, idx):
        """Retrieve a mesh dict by integer index, with lazy loading and caching.

        Args:
            idx: Mesh index as stored in the sequence ``.npz``.

        Returns:
            Dict ``{"vertices": ndarray, "faces": ndarray}`` or ``None``
            if the mesh is unavailable.

        Example::

            mesh = ds.get_mesh(42)
            if mesh is not None:
                verts = mesh["vertices"]
        """
        if idx in self.mesh_registry:
            return self.mesh_registry[idx]

        if self.mesh_archive is None:
            self.init_mesh_archive()  # Lazy init

        if self.mesh_archive is None:
            return None

        try:
            # Lazy Read
            v = self.mesh_archive[f"mesh_{idx}_vertices"]
            f = self.mesh_archive[f"mesh_{idx}_faces"]
            # Create a simple struct or trimesh
            # We just need vertices for sampling usually
            m = {"vertices": v, "faces": f}
            self.mesh_registry[idx] = m
            return m
        except KeyError:
            print(f"Mesh {idx} key missing in archive!")
            return None

    def __decide_trace_sampling__(
        self, seq_id, current_abs_tid, dataset_idx, random_enabled=True
    ):
        """Decide the temporal window for a sample.

        If the sequence is shorter than ``max_T * factor`` the full
        sequence is used. ``current_ith`` is always 0 (future prediction).

        Args:
            seq_id: Sequence identifier.
            current_abs_tid: Absolute anchor frame (unused beyond fallback).
            dataset_idx: Global index for deterministic offset.
            random_enabled: Allow random start.

        Returns:
            Tuple ``(current_ith, start_abs_tid, end_abs_tid)``.

        Example::

            cur_i, s, e = ds.__decide_trace_sampling__("seq.npz", 0, 0)
        """
        factor = int(self.t_rate)

        seq_len = self.seq_len[seq_id]

        if self.max_T * factor >= seq_len:

            start_abs_tid = 0
        else:
            if self.random_flag and random_enabled:
                # Random start
                # end can be at most seq_len
                # start can be at most seq_len - max_T*factor
                start_abs_tid = np.random.randint(0, seq_len - self.max_T * factor)
            else:
                # Sequential or fixed
                # wrap around or stride
                start_abs_tid = (dataset_idx * factor) % (
                    max(1, seq_len - self.max_T * factor)
                )

        end_abs_tid = start_abs_tid + self.max_T * factor
        # Clamp just in case
        end_abs_tid = min(end_abs_tid, seq_len)

        current_ith = 0  # Relative to the slice, where is "current"? usually 0 for future prediction

        return current_ith, start_abs_tid, end_abs_tid

    def __decide_context_sampling__(self, idx):
        """Sample context sequence IDs and time steps.

        Args:
            idx: Current dataset index.

        Returns:
            Tuple ``(ret_sid, ret_abs_tid)`` of lists with length ``max_C``.

        Example::

            ctx_sids, ctx_tids = ds.__decide_context_sampling__(0)
        """
        if self.random_flag:
            choice = np.random.choice(len(self), self.max_C, replace=True)
        else:
            step = max(1, len(self) // (1 + self.max_C))
            start = idx % step
            choice = np.arange(start, len(self), step)
            choice = choice[: self.max_C]
        ret_sid, ret_abs_tid = [], []
        for i in choice:
            sid, current_abs_tid = self.indices[i]
            ret_sid.append(sid)
            ret_abs_tid.append(current_abs_tid)

        return ret_sid, ret_abs_tid

    def __get_data__(self, seq_id):
        """Load or retrieve a cached sequence ``.npz`` dict.

        Args:
            seq_id: Sequence identifier (includes ``.npz`` extension).

        Returns:
            Dict-like with keys such as ``hand_rh_poses``,
            ``obj_rh_poses``, etc.

        Example::

            data = ds.__get_data__("seq_001.npz")
            rh_poses = data["hand_rh_poses"]
        """
        data_path = os.path.join(self.data_root, seq_id)  # seq_id already includes .npz
        if seq_id in self.cached_sequences:
            data = deepcopy(self.cached_sequences[seq_id])
        else:
            data = np.load(data_path)
        return data

    def __add_maniptrans_prompt__(
        self,
        prompt: PromptV0,
        cur_rel_idx,
        start_tid,
        end_tid,
        seq_id,
        ctx_sids,
        ctx_abs_tids,
    ):
        """Populate a PromptV0 with hand, object, and scene traces.

        Processes left/right dexterous hand link poses, manipulated-object
        point-cloud curves, and background scene objects.

        Args:
            prompt: Prompt accumulator.
            cur_rel_idx: Relative anchor index.
            start_tid: Absolute start frame.
            end_tid: Absolute end frame.
            seq_id: Sequence identifier.
            ctx_sids: Context sequence IDs.
            ctx_abs_tids: Context time indices.

        Returns:
            Tuple ``(prompt, extra_info)`` with mesh pointers in extras.

        Example::

            prompt = PromptV0(P=1, T=30, C=5)
            prompt, extra = ds.__add_maniptrans_prompt__(
                prompt, 0, 0, 30, "seq.npz", ctx_sids, ctx_tids
            )
        """
        data = self.__get_data__(seq_id)

        factor = int(self.t_rate)

        extra_info = {}
        extra_info["mesh_pointers"] = {}  # Store mesh ID pointers

        # Helper to convert pose matrix + mesh to points
        def get_points(mesh_id, poses, num_points):
            """Sample mesh vertices and transform them by per-frame poses.

            Args:
                mesh_id: Integer mesh index or list of indices (multi-link).
                poses: ``(T, 4, 4)`` for a single object or ``(T, M, 4, 4)``
                    for multi-link meshes.
                num_points: Number of vertices to sample per mesh.

            Returns:
                ``(T, P, 3)`` for single objects or ``(T, M, P, 3)`` for
                multi-link, or ``None`` if the mesh is not found.
            """
            # poses: T, 4, 4 or T, M, 4, 4
            # mesh_id: int or list of ints

            # Case 1: Single Object
            if isinstance(mesh_id, (int, np.integer)):
                mesh = self.get_mesh(mesh_id)
                if mesh is None:
                    return None
                pcl = sample_list(
                    mesh["vertices"], num_points, self.random_flag
                )  # P, 3
                # Transform
                # poses: T, 4, 4
                # result: T, P, 3
                return (
                    np.einsum("tij,pj->tpi", poses[:, :3, :3], pcl)
                    + poses[:, :3, 3][:, None]
                )

            # Case 2: Multi-link (Hand/Robot)
            elif isinstance(mesh_id, (list, np.ndarray)):
                # poses: T, M, 4, 4
                # mesh_id: M
                all_link_points = []
                for i, mid in enumerate(mesh_id):
                    mesh = self.get_mesh(mid)
                    if mesh is None:
                        # fallback / padding?
                        pass
                    # Assuming we want points for each link? Or aggregate?
                    # Usually for hands we want aggregate or per-link?
                    # Gemm protocol usually allows multi-group.
                    # Let's return list of (points, link_idx)
                    pcl = sample_list(mesh["vertices"], num_points, self.random_flag)
                    t_pj = (
                        np.einsum("tij,pj->tpi", poses[:, i, :3, :3], pcl)
                        + poses[:, i, :3, 3][:, None]
                    )
                    all_link_points.append(t_pj)
                return np.stack(all_link_points, axis=1)  # T, M, P, 3
            return None

        # 1. Hands
        # Check for left/right hands
        for side, sem, group_uid_base in [
            ("rh", SEM_RIGHT_DEX, UID_MANIPTRANS_INSPIRE_RH),
            (
                "lh",
                SEM_LEFT_DEX,
                UID_MANIPTRANS_INSPIRE_LH,
            ),  # Check config if these exists, assumed from user prompt
        ]:
            key_pose = f"hand_{side}_poses"
            key_mesh = f"hand_{side}_mesh_ids"

            if key_pose in data:
                poses = data[key_pose][start_tid:end_tid:factor]  # T, M, 4, 4
                mesh_ids = data[key_mesh]

                extra_info["mesh_pointers"][f"hand_{side}"] = mesh_ids.tolist()

                # We need context poses too
                # Collect context poses
                ctx_poses_list = []
                for csid, catid in zip(ctx_sids, ctx_abs_tids):
                    # Assuming context is from same file for simplicity here as strictly enforced in decision
                    ctx_data = self.__get_data__(csid)
                    if key_pose in ctx_data:
                        c_poses = ctx_data[key_pose][catid]  # M, 4, 4
                        ctx_poses_list.append(c_poses)
                ctx_poses = np.stack(ctx_poses_list)  # C, M, 4, 4

                num_links = poses.shape[1]

                poses_torch = torch.from_numpy(poses).float()
                ctx_poses_torch = torch.from_numpy(ctx_poses).float()

                poses_torch = poses_torch.permute(1, 0, 2, 3)  # M, T, 4, 4
                ctx_poses_torch = ctx_poses_torch.permute(1, 0, 2, 3)  # M, C, 4, 4

                # UIDs
                # We'll generate dynamic UIDs or use zeros if not critical for now,
                # but better to use unique ones.
                # Using a large offset for safety if not in config.
                if side == "rh":
                    uids = UID_MANIPTRANS_INSPIRE_RH
                elif side == "lh":
                    uids = UID_MANIPTRANS_INSPIRE_LH
                else:
                    raise ValueError(f"Unknown side: {side}")

                prompt.append_trace(
                    trace=poses_torch,
                    trace_mask=torch.ones(poses_torch.shape[:2]),
                    context=ctx_poses_torch,
                    context_mask=torch.ones(ctx_poses_torch.shape[:2]),
                    group_id=prompt.max_group_id + 1,
                    sem_type=sem,
                    uid=uids,
                )

        # 2. Manipulated Objects
        # Keys like obj_rh_poses, obj_rh_mesh_id
        extra_info["mesh_pointers"]["objects"] = {}
        for side, uid in zip(
            ["rh", "lh"],
            [
                UID_MANIPTRANS_OBJECT_0,
                UID_MANIPTRANS_OBJECT_1,
            ],
        ):
            key_pose = f"obj_{side}_poses"
            key_mesh = f"obj_{side}_mesh_id"
            if key_pose in data:
                poses = data[key_pose][start_tid:end_tid:factor]  # T, 4, 4
                mesh_id = int(data[key_mesh])
                extra_info["mesh_pointers"]["objects"][uid] = mesh_id

                # Sample points for object
                # Trace points: T, P, 3
                # Context points: C, P, 3

                # Get trace points
                # Need points in locally to transform? OR just transform all
                mesh = self.get_mesh(mesh_id)
                if mesh:
                    pcl = sample_list(
                        mesh["vertices"], self.max_P_per_object, self.random_flag
                    )  # P, 3

                    # Trace
                    trace_pts = (
                        np.einsum("tij,pj->tpi", poses[:, :3, :3], pcl)
                        + poses[:, :3, 3][:, None]
                    )
                    trace_pts = torch.from_numpy(trace_pts).float()  # T, P, 3

                    # Context
                    ctx_pts = torch.from_numpy(pcl)[None].expand(
                        self.max_C, -1, -1
                    )  # C, P, 3

                    # Permute for append_trace: P, T, 3
                    trace_pts = trace_pts.permute(1, 0, 2)
                    ctx_pts = ctx_pts.permute(1, 0, 2)

                    prompt.append_trace(
                        trace=trace_pts,
                        trace_mask=torch.ones(trace_pts.shape[:2]),
                        context=ctx_pts,
                        context_mask=torch.ones(ctx_pts.shape[:2]),
                        group_id=prompt.max_group_id + 1,
                        sem_type=SEM_OBJECT,
                        uid=uid,  # Or specific
                    )

        # 3. Scene Objects (Background)
        extra_info["mesh_pointers"]["scene"] = {}
        if "scene_object_poses" in data:
            poses = data["scene_object_poses"]  # T, N_objs, 4, 4
            mesh_ids = data["scene_object_mesh_ids"]
            names = data.get("scene_object_names", [])

            # Slice time
            poses_slice = poses[start_tid:end_tid:factor]  # T, N, 4, 4

            # For each scene object
            for i, mid in enumerate(mesh_ids):
                mesh = self.get_mesh(mid)
                if mesh is None:
                    continue

                pcl = sample_list(
                    mesh["vertices"], self.max_P_per_object, self.random_flag
                )

                # Trace
                # poses_slice[:, i] is T, 4, 4
                obj_poses = poses_slice[:, i]
                trace_pts = (
                    np.einsum("tij,pj->tpi", obj_poses[:, :3, :3], pcl)
                    + obj_poses[:, :3, 3][:, None]
                )
                trace_pts = torch.from_numpy(trace_pts).float()  # T, P, 3

                # Context
                ctx_pts = torch.from_numpy(pcl)[None].expand(
                    self.max_C, -1, -1
                )  # C, P, 3

                # Permute
                trace_pts = trace_pts.permute(1, 0, 2)
                ctx_pts = ctx_pts.permute(1, 0, 2)

                prompt.append_trace(
                    trace=trace_pts,
                    trace_mask=torch.ones(trace_pts.shape[:2]),
                    context=ctx_pts,
                    context_mask=torch.ones(ctx_pts.shape[:2]),
                    group_id=prompt.max_group_id + 1,
                    sem_type=SEM_BACKGROUND,
                    uid=UID_MANIPTRANS_BACKGROUND_GROUP[i],
                )
                extra_info["mesh_pointers"]["scene"][
                    UID_MANIPTRANS_BACKGROUND_GROUP[i]
                ] = mid

        return prompt, extra_info

    def __getitem_inner__(self, idx):
        """Build a single ManipTrans training sample.

        Args:
            idx: Raw dataset index (wrapped by ``len(self)``).

        Returns:
            Tuple ``(CanonicalDatasetOutput, extra_info)``.

        Example::

            ret, extra = ds.__getitem_inner__(0)
            print(ret.trace.shape)
        """
        idx = idx % len(self)
        prompt = PromptV0(P=1, T=self.max_T, C=self.max_C)  # P auto-expands

        sid, _ = self.indices[idx]

        # Decide sampling
        current_ith, start_abs_tid, end_abs_tid = self.__decide_trace_sampling__(
            sid, 0, idx, random_enabled=self.random_flag
        )
        ctx_sids, ctx_abs_tids = self.__decide_context_sampling__(idx)

        prompt, extra_info = self.__add_maniptrans_prompt__(
            prompt, current_ith, start_abs_tid, end_abs_tid, sid, ctx_sids, ctx_abs_tids
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        time_unit = 1.0 / (float(self.base_fps) * float(self.t_rate))
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

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
            text="",  # No text
        )
        return ret, extra_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = ManipTransDataset(
        data_root="data/maniptrans_favor",
        split_fn="data_splits/maniptrans_favor/train.json",
        base_fps=30,
        random_flag=False,
        safe_flag=False,
        max_T=30,
        max_C=5,
        max_P_per_object=16,
        max_P_per_background=10,
        cache_all_data=False,  # Test false first
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        ret, extra = dataset[0]
        print("Sample 0 keys:", vars(ret).keys())
        print("Trace shape:", ret.trace.shape)
        print("Extra info keys:", extra.keys())
        print("Mesh pointers:", extra.get("mesh_pointers"))

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            shuffle=True,
            collate_fn=CanonicalDatasetOutput.collate,
        )

        print("Testing DataLoader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} trace shape: {batch['trace'].shape}")
            # if i >= 2:
            #     break
