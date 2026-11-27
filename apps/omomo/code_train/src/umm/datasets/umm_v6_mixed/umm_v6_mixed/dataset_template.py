"""Canonical dataset output format with dynamic-size batched collation.

Defines :class:`CanonicalDatasetOutput`, the standard data container that
all UMM datasets produce.  Includes a custom ``collate`` classmethod that
handles variable ``P`` (entity count), ``T`` (time steps), and ``C``
(context steps) across samples in a batch by zero-padding to the per-batch
maximum.
"""

import torch
import numpy as np
from dataclasses import dataclass, fields
import time


@dataclass
class CanonicalDatasetOutput:
    """Standard output container for all UMM v1 datasets.

    All datasets produce instances of this dataclass.  The custom
    :meth:`collate` classmethod pads variable-size fields (P, T, C, PCL)
    across a batch and randomly permutes group IDs for training robustness.

    Attributes:
        seq_name: Sequence identifier string.
        trace: SE3 trajectory poses, shape ``(P, T, 4, 4)``.
        trace_mask: Validity mask for trace, shape ``(P, T)``.
        context: Context poses, shape ``(P, C, 4, 4)``.
        context_mask: Context validity, shape ``(P, C)``.
        trace_group_id: Group assignment per entity, shape ``(P,)``.
        trace_rot_mask: Whether entity has rotation (vs. position-only), ``(P,)``.
        trace_sem_type: Semantic type ID per entity, shape ``(P,)``.
        trace_uid: Unique ID per entity (for visualization), shape ``(P,)``.
        time: Relative time stamps, shape ``(T,)``.
        text: Language instruction string.
        local_pcl: Local point cloud per entity, shape ``(P, PCL, 3+)``.
        local_pcl_mask: PCL validity per entity, shape ``(P,)``.
    """

    # P the num of trajectories
    # T the num of time steps
    # C the num of context steps

    seq_name: str

    trace: np.ndarray  # T,P,4,4
    trace_mask: np.ndarray  # T,P # some intermediate steps may be invalid
    context: np.ndarray  # C,P,4,4 # the context state
    context_mask: np.ndarray  # C,P # some context steps may be invalid

    trace_group_id: np.ndarray  # P # !
    trace_rot_mask: np.ndarray  # P # some prompt do not have rotation term
    trace_sem_type: np.ndarray  # P # the semantic type of the prompt
    # # TODO: later this can be collected from image current time step features
    # P # the unique id of the prompt # ! not used, just to be compatible with the old version viz
    trace_uid: np.ndarray  # P
    time: np.ndarray  # T # the time list
    text: str  # the text prompt

    # PCL the num of local points
    local_pcl: np.ndarray = None  # P,PCL,3
    local_pcl_mask: np.ndarray = None  # P,PCL

    def check_shape(self):
        """Validate shapes and types of all fields.

        Raises:
            AssertionError: If any field shape is inconsistent.
        """
        assert isinstance(self.seq_name, str)
        assert (
            self.trace.ndim == 4
            and self.trace.shape[-1] == 4
            and self.trace.shape[-2] == 4
        )
        T, P = self.trace.shape[:2]
        assert self.trace_mask.ndim == 2 and self.trace_mask.shape == (T, P)
        assert (
            self.context.ndim == 4
            and self.context.shape[-1] == 4
            and self.context.shape[-2] == 4
        )
        assert self.context.shape[1] == P
        C = self.context.shape[0]
        assert self.context_mask.ndim == 2 and self.context_mask.shape == (C, P)
        assert self.trace_group_id.ndim == 1 and self.trace_group_id.shape == (P,)
        assert self.trace_rot_mask.ndim == 1 and self.trace_rot_mask.shape == (P,)
        assert self.trace_sem_type.ndim == 1 and self.trace_sem_type.shape == (P,)
        assert self.trace_uid.ndim == 1 and self.trace_uid.shape == (P,)
        assert self.time_idx.ndim == 1 and self.time_idx.shape == (T,)
        assert isinstance(self.current_idx, int)
        assert isinstance(self.time_unit, float)
        assert isinstance(self.text, str)

        assert self.local_pcl.ndim == 3 and self.local_pcl.shape[-1] == 3
        assert self.local_pcl.shape[1] == P
        PCL = self.local_pcl.shape[0]
        assert self.local_pcl_mask.ndim == 1 and self.local_pcl_mask.shape == (P,)

    def dict(self):
        """Convert all fields to a plain dict after shape validation.

        Returns:
            Dict mapping field names to their values.
        """
        self.check_shape()
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def collate(batch):
        """Collate function placeholder (overridden by classmethod below).

        Raises:
            NotImplementedError: Always — see the classmethod ``collate`` instead.
        """
        # this deal with various T,P,C size and pad them to form a batch
        raise NotImplementedError

    @property
    def P(self):
        """Number of entities (parts/points)."""
        return self.trace.shape[0]

    @property
    def T(self):
        """Number of time-steps in the trace."""
        return self.trace.shape[1]

    @property
    def C(self):
        """Number of context frames."""
        return self.context.shape[1]

    @property
    def PCL(self):
        """Number of local point-cloud points per entity."""
        return self.local_pcl.shape[1]

    @property
    def PCL_dim(self):
        """Dimensionality of each point-cloud point (3 or 6)."""
        return self.local_pcl.shape[2]

    @classmethod
    def collate(cls, in_list):
        """Custom collate that pads variable-length T, P, C, PCL across a batch.

        Args:
            in_list: List of ``(CanonicalDatasetOutput, extra_dict)`` tuples
                from individual dataset ``__getitem__`` calls.

        Returns:
            Tuple of ``(batched_dict, extra_data_list)`` where the batched dict
            has zero-padded arrays with ``max_*`` dimensions.
        """
        # start_t = time.time()

        canonical_data: list[cls] = [it[0] for it in in_list]
        extra_data: list[dict] = [it[1] for it in in_list]
        P_list = [it.P for it in canonical_data]
        T_list = [it.T for it in canonical_data]
        C_list = [it.C for it in canonical_data]
        max_P = max(P_list)
        max_T = max(T_list)
        max_C = max(C_list)
        max_PCL = max(
            [it.PCL for it in canonical_data]
        )  # if less, resample with replacement!
        max_PCL_dim = max([it.PCL_dim for it in canonical_data])

        ret = {}
        # main data
        B = len(canonical_data)
        trace_list, trace_mask_list, context_list, context_mask_list = [], [], [], []
        group_id_list, rot_mask_list, sem_type_list, uid_list = [], [], [], []
        pcl_list, pcl_mask_list = [], []
        time_list, text_list = [], []
        seq_name_list = []
        for bid in range(B):
            seq_name_list.append(canonical_data[bid].seq_name)
            _trace = canonical_data[bid].trace  # (p,t,4,4)
            _trace_mask = canonical_data[bid].trace_mask  # (p,t)
            _context = canonical_data[bid].context  # (p,c,4,4)
            _context_mask = canonical_data[bid].context_mask  # (p,c)
            _group_id = canonical_data[bid].trace_group_id  # (p)
            _rot_mask = canonical_data[bid].trace_rot_mask  # (p)
            _sem_type = canonical_data[bid].trace_sem_type  # (p)
            _uid = canonical_data[bid].trace_uid  # (p)
            _time = canonical_data[bid].time  # (t)
            _text = canonical_data[bid].text

            _p, _t = _trace.shape[:2]
            _c = _context.shape[1]

            _pcl = canonical_data[bid].local_pcl  # (p,pcl,3)
            _pcl_mask = canonical_data[bid].local_pcl_mask  # (p)
            if _pcl.shape[1] < max_PCL:
                # duplicate the pcl on the PCL dim
                n_dup = int(np.ceil((max_PCL - _pcl.shape[1]) / _pcl.shape[1]))
                _pcl = torch.cat([_pcl] * n_dup, 1)[:, :max_PCL]
            if _pcl.shape[2] < max_PCL_dim:
                _pcl = torch.cat(
                    [
                        _pcl,
                        torch.zeros(_p, max_PCL, max_PCL_dim - _pcl.shape[2]),
                    ],
                    2,
                )

            if _t < max_T:
                _trace = torch.cat(
                    [
                        _trace,
                        torch.eye(4)[None, None].repeat(_p, max_T - _t, 1, 1),
                    ],
                    1,
                )
                _trace_mask = torch.cat([_trace_mask, torch.zeros(_p, max_T - _t)], 1)
                _time = torch.cat([_time, torch.zeros(max_T - _t)], 0)
            if _c < max_C:
                _context = torch.cat(
                    [
                        _context,
                        torch.eye(4)[None, None].repeat(_p, max_C - _c, 1, 1),
                    ],
                    1,
                )
                _context_mask = torch.cat(
                    [_context_mask, torch.zeros(_p, max_C - _c)], 1
                )
            if _p < max_P:
                _trace = torch.cat(
                    [_trace, torch.eye(4)[None, None].repeat(max_P - _p, max_T, 1, 1)],
                    0,
                )
                _trace_mask = torch.cat(
                    [_trace_mask, torch.zeros(max_P - _p, max_T)], 0
                )
                _context = torch.cat(
                    [
                        _context,
                        torch.eye(4)[None, None].repeat(max_P - _p, max_C, 1, 1),
                    ],
                    0,
                )
                _context_mask = torch.cat(
                    [_context_mask, torch.zeros(max_P - _p, max_C)], 0
                )
                _group_id = torch.cat([_group_id, torch.zeros(max_P - _p)], 0)
                _rot_mask = torch.cat([_rot_mask, torch.zeros(max_P - _p)], 0)
                _sem_type = torch.cat([_sem_type, torch.zeros(max_P - _p)], 0)
                _uid = torch.cat([_uid, torch.zeros(max_P - _p)], 0)

                _pcl = torch.cat(
                    [_pcl, torch.zeros(max_P - _p, max_PCL, max_PCL_dim)], 0
                )
                _pcl_mask = torch.cat([_pcl_mask, torch.zeros(max_P - _p)], 0)

            trace_list.append(_trace)
            trace_mask_list.append(_trace_mask)
            context_list.append(_context)
            context_mask_list.append(_context_mask)

            # ! random permute the group_id, there is a new mapping between the old and new valid groups!
            _group_id = _group_id.long()
            fg_mask = _trace_mask.any(dim=1)  # n
            old_gid = _group_id[fg_mask].clone()
            old_fg_gid = old_gid.unique()
            assert (
                old_fg_gid.sort()[0] == torch.arange(len(old_fg_gid))
            ).all(), "GID invalid, check dataset"
            new_fg_gid = old_fg_gid[torch.randperm(len(old_fg_gid))]
            new_gid = torch.gather(new_fg_gid, 0, old_gid)
            _group_id[fg_mask] = new_gid
            # print(_group_id)

            if "gid" in extra_data[bid]:
                # ! must permute the order of gid also
                old_extra_gid = extra_data[bid]["gid"]
                new_extra_gid = [int(new_fg_gid[i]) for i in old_extra_gid]
                extra_data[bid]["gid"] = new_extra_gid

            group_id_list.append(_group_id.contiguous())
            rot_mask_list.append(_rot_mask)
            sem_type_list.append(_sem_type)
            uid_list.append(_uid)
            time_list.append(_time)
            text_list.append(_text)

            pcl_list.append(_pcl)
            pcl_mask_list.append(_pcl_mask)

        ret = {
            "seq_name": seq_name_list,
            "trace": torch.stack(trace_list, 0).float().permute(0, 2, 1, 3, 4),
            "trace_mask": torch.stack(trace_mask_list, 0).bool().permute(0, 2, 1),
            "context": torch.stack(context_list, 0).float().permute(0, 2, 1, 3, 4),
            "context_mask": torch.stack(context_mask_list, 0).bool().permute(0, 2, 1),
            "trace_group_id": torch.stack(group_id_list, 0).int(),
            "trace_rot_mask": torch.stack(rot_mask_list, 0).bool(),
            "trace_sem_type": torch.stack(sem_type_list, 0).int(),
            "trace_uid": torch.stack(uid_list, 0).int(),
            "time": torch.stack(time_list, 0).float(),
            "text": text_list,
            "local_pcl": torch.stack(pcl_list, 0).float(),  # B,P,PCL,3
            "local_pcl_mask": torch.stack(pcl_mask_list, 0).bool(),  # B,P
        }

        # pad keys
        all_keys = set.union(*[set(it.keys()) for it in extra_data])

        # start_t = time.time()
        extra = {k: [] for k in all_keys}
        for k in all_keys:
            for it in extra_data:
                if k in it:
                    extra[k].append(it[k])
                else:
                    extra[k].append(None)

        ret["extra"] = extra
        # print("Overhead: ", time.time() - start_t)
        return ret
