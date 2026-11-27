"""Base dataset class and prompt builder for WMM V10 datasets.

Provides :class:`BaseDataset` (common loading logic) and :class:`PromptV0`
(SE3 trace accumulator with padding/concatenation).

Self-contained — NO imports from V9 (module independence).
"""

import logging
from torch.utils.data import Dataset
import torch
import numpy as np

from lib_io import UnifiedIO
from umm.datasets.umm_v11.dataset_utils import load_indices, load_language
from umm.datasets.umm_v11.dataset_template import CanonicalDatasetOutput


class BaseDataset(Dataset):
    """Abstract base class for all WMM V10 datasets.

    Handles split loading, safe error recovery, language annotation loading,
    and DataLoader worker initialisation. Subclasses must override
    :meth:`__getitem_inner__` to produce a :class:`CanonicalDatasetOutput`.

    Args:
        data_root: Root directory containing the dataset files.
        split_fn: Path to JSON split file (list of ``[sid, length]``).
        base_fps: Native frame rate of this dataset.
        language_fn: Optional path to language annotation JSON.
        random_flag: Enable random augmentations / sampling.
        safe_flag: If True, silently skip to next sample on errors.
        repeat: Number of times to repeat the index list.
        t_rate: Temporal stride for index expansion.
    """

    def __init__(
        self,
        data_root,
        split_fn,
        base_fps,
        language_fn=None,
        random_flag=True,
        safe_flag=True,
        repeat=1,
        t_rate=1,
    ):
        self.data_root = data_root
        self.split_fn = split_fn
        self.base_fps = base_fps
        self.t_rate = t_rate
        self.random_flag = random_flag
        self.safe_flag = safe_flag
        self.repeat = repeat

        self.indices, self.seq_len = load_indices(split_fn, self.t_rate)
        self.io = UnifiedIO()
        self.language_data = load_language(language_fn)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.safe_flag:
            while True:
                try:
                    return self.__getitem_inner__(idx)
                except Exception as e:
                    logging.error(f"Error in loading {idx}: {e}")
                    idx = (idx + 1) % len(self)
        else:
            return self.__getitem_inner__(idx)

    def __getitem_inner__(self, idx):
        """Subclass hook: produce one sample.  Must be overridden."""
        raise NotImplementedError

    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize worker to use single thread for NumPy operations."""
        import threadpoolctl
        import os

        threadpoolctl.threadpool_limits(limits=1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"


class PromptV0:
    """Accumulator for SE3 trajectory traces with automatic padding.

    Collects multiple groups of SE3 traces (each potentially a different
    number of entities) and pads them to uniform sizes ``(P, T, 4, 4)``
    and ``(P, C, 4, 4)`` for batching.

    Args:
        P: Maximum number of trajectory entities (0 means auto-grow).
        T: Maximum number of time steps per trace.
        C: Maximum number of context steps.
        PCL: Number of local point cloud points per entity.
        PCL_dim: Dimensionality of each point (typically 3 or 6 for xyz+rgb).
    """

    def __init__(self, P, T, C, PCL=0, PCL_dim=3):
        self.se3_trace = []
        self.se3_trace_mask = []
        self.se3_context = []
        self.se3_context_mask = []

        self.group_id = []
        self.rot_mask = []
        self.sem_type = []
        self.uids = []

        self.local_pcl = []
        self.local_pcl_mask = []

        self.T = T
        self.C = C
        self.P = P
        self.PCL = PCL
        self.PCL_dim = PCL_dim

    def append_trace(
        self,
        trace,
        trace_mask,
        context,
        context_mask,
        group_id,
        sem_type,
        uid,
        local_pcl=None,
        local_pcl_mask=None,
    ):
        """Append a batch of trajectory entities to this prompt.

        Handles both SE3 ``(n, T, 4, 4)`` and position-only ``(n, T, 3)``
        traces, auto-converting the latter to homogeneous form with identity
        rotation.  Time and context dimensions are zero-padded to ``self.T``
        and ``self.C`` respectively.

        Args:
            trace: Trajectory data, shape ``(n, t, 4, 4)`` or ``(n, t, 3)``.
            trace_mask: Validity mask, shape ``(n, t)``.
            context: Context data, shape ``(n, c, 4, 4)`` or ``(n, c, 3)``.
            context_mask: Context validity, shape ``(n, c)``.
            group_id: Group assignment, int or array of shape ``(n,)``.
            sem_type: Semantic type, int or array of shape ``(n,)``.
            uid: Unique ID, int or array of shape ``(n,)``.
            local_pcl: Optional local point cloud, shape ``(n, PCL, PCL_dim)``.
            local_pcl_mask: Optional PCL validity, shape ``(n,)``.
        """
        trace = torch.as_tensor(trace)
        trace_mask = torch.as_tensor(trace_mask)
        context = torch.as_tensor(context)
        context_mask = torch.as_tensor(context_mask)
        if local_pcl is not None:
            local_pcl = torch.as_tensor(local_pcl)
        if local_pcl_mask is not None:
            local_pcl_mask = torch.as_tensor(local_pcl_mask)

        n = len(trace)
        t = trace.shape[1]
        c = context.shape[1]
        assert t <= self.T and c <= self.C

        if trace.ndim == 4:
            assert trace.shape == (n, t, 4, 4)
            rot_mask = torch.ones(n).bool()
        else:
            assert trace.shape == (n, t, 3)
            rot_mask = torch.zeros(n).bool()
            _trace = torch.eye(4)[None, None].repeat(n, t, 1, 1)
            _trace[:, :, :3, 3] = trace.float()
            trace = _trace

        if context.ndim == 4:
            assert context.shape == (n, c, 4, 4)
        else:
            assert context.shape == (n, c, 3)
            _context = torch.eye(4)[None, None].repeat(n, c, 1, 1)
            _context[:, :, :3, 3] = context.float()
            context = _context

        assert trace_mask.shape == (n, t)
        assert context_mask.shape == (n, c)

        if isinstance(group_id, int):
            group_id = np.full((n,), group_id)
        elif isinstance(group_id, list):
            group_id = np.array(group_id)
        if isinstance(sem_type, int):
            sem_type = np.full((n,), sem_type)
        elif isinstance(sem_type, list):
            sem_type = np.array(sem_type)
        if isinstance(uid, int):
            uid = np.full((n,), uid)
        elif isinstance(uid, list):
            uid = np.array(uid)
        assert group_id.shape == (n,)
        assert sem_type.shape == (n,)
        assert uid.shape == (n,)

        # Pad time dimension
        if t < self.T:
            trace = torch.cat(
                [trace, torch.eye(4)[None, None].expand(n, self.T - t, -1, -1)], 1
            )
            trace_mask = torch.cat([trace_mask, torch.zeros(n, self.T - t)], 1)
        if c < self.C:
            context = torch.cat(
                [context, torch.eye(4)[None, None].expand(n, self.C - c, -1, -1)], 1
            )
            context_mask = torch.cat([context_mask, torch.zeros(n, self.C - c)], 1)

        self.se3_trace.append(trace)
        self.se3_trace_mask.append(trace_mask)
        self.se3_context.append(context)
        self.se3_context_mask.append(context_mask)

        self.rot_mask.append(torch.as_tensor(rot_mask))
        self.group_id.append(torch.as_tensor(group_id))
        self.sem_type.append(torch.as_tensor(sem_type))
        self.uids.append(torch.as_tensor(uid))

        if local_pcl is None:
            local_pcl = torch.zeros((n, self.PCL, self.PCL_dim))
            local_pcl_mask = torch.zeros(n)
        else:
            assert local_pcl.shape == (n, self.PCL, self.PCL_dim)
            assert local_pcl_mask.shape == (n,)
        self.local_pcl.append(local_pcl)
        self.local_pcl_mask.append(local_pcl_mask)

    def cat(self):
        """Concatenate all appended traces into single tensors.

        Returns:
            Tuple of 8 tensors: ``(trace, trace_mask, context, context_mask,
            rot_mask, group_id, sem_type, uids)`` with first dim ``P``.
        """
        return (
            torch.cat(self.se3_trace, 0),  # (P, T, 4, 4)
            torch.cat(self.se3_trace_mask, 0),  # (P, T)
            torch.cat(self.se3_context, 0),  # (P, C, 4, 4)
            torch.cat(self.se3_context_mask, 0),  # (P, C)
            torch.cat(self.rot_mask, 0),  # (P,)
            torch.cat(self.group_id, 0),  # (P,)
            torch.cat(self.sem_type, 0),  # (P,)
            torch.cat(self.uids, 0),  # (P,)
        )

    def cat_pcl(self):
        """Concatenate local point clouds from all appended traces.

        Returns:
            Tuple of ``(pcl, pcl_mask)`` with shapes ``(P, PCL, PCL_dim)``
            and ``(P,)`` respectively.
        """
        return torch.cat(self.local_pcl, 0), torch.cat(self.local_pcl_mask, 0).bool()

    def __len__(self):
        return sum([len(x) for x in self.uids])

    def catch_P(self, P=None):
        """Pad to at least ``P`` entities with identity-pose placeholders."""
        if P is None:
            P = self.P
        if len(self) < P:
            n = P - len(self)
            self.se3_trace.append(torch.eye(4)[None, None].expand(n, self.T, -1, -1))
            self.se3_trace_mask.append(torch.zeros(n, self.T))
            self.se3_context.append(torch.eye(4)[None, None].expand(n, self.C, -1, -1))
            self.se3_context_mask.append(torch.zeros(n, self.C))

            self.rot_mask.append(torch.zeros(n))
            self.group_id.append(torch.zeros(n))
            self.sem_type.append(torch.zeros(n))
            self.uids.append(torch.zeros(n))

            self.local_pcl.append(torch.zeros((n, self.PCL, self.PCL_dim)))
            self.local_pcl_mask.append(torch.zeros(n))
        assert len(self) >= P

    @property
    def max_group_id(self):
        if len(self.group_id) == 0:
            return -1
        return int(max([max(x) for x in self.group_id]))
