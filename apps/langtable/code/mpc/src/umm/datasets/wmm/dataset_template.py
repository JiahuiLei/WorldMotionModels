"""Canonical dataset output format with known_mask and dynamic-size batched collation.

V10 addition: ``known_mask`` and ``context_known_mask`` fields for training-time
mask conditioning. The model conditions on known tokens and predicts unknown ones.

Defines :class:`CanonicalDatasetOutput`, the standard data container that
all UMM V10 datasets produce.  Includes a custom ``collate`` classmethod that
handles variable ``P`` (entity count), ``T`` (time steps), and ``C``
(context steps) across samples in a batch by zero-padding to the per-batch
maximum.  Padded positions in known_mask are set to False (not known).
"""

import torch
import numpy as np
from dataclasses import dataclass, fields
from typing import ClassVar, Optional


@dataclass
class CanonicalDatasetOutput:
    """Standard output container for all UMM V10 datasets.

    All datasets produce instances of this dataclass. The custom
    :meth:`collate` classmethod pads variable-size fields (P, T, C, PCL)
    across a batch and randomly permutes group IDs for training robustness.

    Attributes:
        seq_name: Sequence identifier string.
        trace: SE3 trajectory poses, shape ``(P, T, 4, 4)``.
        trace_mask: Validity mask for trace, shape ``(P, T)``.
        context: Context poses, shape ``(P, C, 4, 4)``.
        context_mask: Context validity, shape ``(P, C)``.
        known_mask: Which trace tokens are known (conditioning input), ``(P, T)``.
        context_known_mask: Which context tokens are known, ``(P, C)``.
        trace_group_id: Group assignment per entity, shape ``(P,)``.
        trace_rot_mask: Whether entity has rotation (vs. position-only), ``(P,)``.
        trace_sem_type: Semantic type ID per entity, shape ``(P,)``.
        trace_uid: Unique ID per entity (for visualization), shape ``(P,)``.
        time: Relative time stamps, shape ``(T,)``.
        text: Language instruction string.
        local_pcl: Local point cloud per entity, shape ``(P, PCL, 3+)``.
        local_pcl_mask: PCL validity per entity, shape ``(P,)``.
    """

    # When set, collate pads T to this fixed value to prevent torch.compile
    # recompilation from variable sequence lengths across batches.
    _compile_fixed_T: ClassVar[Optional[int]] = None

    seq_name: str

    trace: torch.Tensor  # (P, T, 4, 4)
    trace_mask: torch.Tensor  # (P, T) bool
    context: torch.Tensor  # (P, C, 4, 4)
    context_mask: torch.Tensor  # (P, C) bool

    # V10 NEW: mask scheduler outputs
    known_mask: torch.Tensor  # (P, T) bool — which trace tokens are known
    context_known_mask: torch.Tensor  # (P, C) bool — which context tokens are known

    trace_group_id: torch.Tensor  # (P,) int
    trace_rot_mask: torch.Tensor  # (P,) bool
    trace_sem_type: torch.Tensor  # (P,) int
    trace_uid: torch.Tensor  # (P,) int
    time: torch.Tensor  # (T,) float
    text: str

    local_pcl: Optional[torch.Tensor] = None  # (P, PCL, 3+)
    local_pcl_mask: Optional[torch.Tensor] = None  # (P,) bool

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
        if self.local_pcl is None:
            return 0
        return self.local_pcl.shape[1]

    @property
    def PCL_dim(self):
        """Dimensionality of each point-cloud point (3 or 6)."""
        if self.local_pcl is None:
            return 3
        return self.local_pcl.shape[2]

    @classmethod
    def collate(cls, in_list):
        """Custom collate that pads variable-length T, P, C, PCL across a batch.

        Padded positions get:
        - trace/context: identity matrix
        - trace_mask/context_mask: False
        - known_mask/context_known_mask: False (padded = NOT known)
        - group_id/sem_type/uid: 0

        Group IDs are randomly permuted within each sample for training robustness.

        Args:
            in_list: List of ``(CanonicalDatasetOutput, extra_dict)`` tuples.

        Returns:
            Tuple of ``(batched_dict, extra_data)`` where batched_dict has
            zero-padded tensors with ``max_*`` dimensions.
            Output trace shape: ``(B, T, P, 4, 4)`` (time-major, entity-minor).
        """
        canonical_data = [it[0] for it in in_list]
        extra_data = [it[1] for it in in_list]
        P_list = [it.P for it in canonical_data]
        T_list = [it.T for it in canonical_data]
        C_list = [it.C for it in canonical_data]
        max_P = max(P_list)
        max_T = max(T_list)
        if cls._compile_fixed_T is not None:
            max_T = max(max_T, cls._compile_fixed_T)
        max_C = max(C_list)
        pcl_sizes = [
            it.local_pcl.shape[1] if it.local_pcl is not None else 0
            for it in canonical_data
        ]
        max_PCL = max(pcl_sizes) if any(s > 0 for s in pcl_sizes) else 0
        pcl_dims = [
            it.local_pcl.shape[2] if it.local_pcl is not None else 3
            for it in canonical_data
        ]
        max_PCL_dim = max(pcl_dims)

        B = len(canonical_data)
        trace_list, trace_mask_list = [], []
        context_list, context_mask_list = [], []
        known_mask_list, ctx_known_mask_list = [], []
        group_id_list, rot_mask_list, sem_type_list, uid_list = [], [], [], []
        pcl_list, pcl_mask_list = [], []
        time_list, text_list, seq_name_list = [], [], []

        for bid in range(B):
            seq_name_list.append(canonical_data[bid].seq_name)
            _trace = canonical_data[bid].trace  # (P, T, 4, 4)
            _trace_mask = canonical_data[bid].trace_mask  # (P, T)
            _context = canonical_data[bid].context  # (P, C, 4, 4)
            _context_mask = canonical_data[bid].context_mask  # (P, C)
            _known_mask = canonical_data[bid].known_mask  # (P, T)
            _ctx_known_mask = canonical_data[bid].context_known_mask  # (P, C)
            _group_id = canonical_data[bid].trace_group_id  # (P,)
            _rot_mask = canonical_data[bid].trace_rot_mask  # (P,)
            _sem_type = canonical_data[bid].trace_sem_type  # (P,)
            _uid = canonical_data[bid].trace_uid  # (P,)
            _time = canonical_data[bid].time  # (T,)
            _text = canonical_data[bid].text

            _p, _t = _trace.shape[:2]
            _c = _context.shape[1]

            # --- PCL handling ---
            _pcl = canonical_data[bid].local_pcl
            _pcl_mask = canonical_data[bid].local_pcl_mask
            if _pcl is None or max_PCL == 0:
                _pcl = torch.zeros(_p, max(max_PCL, 1), max_PCL_dim)
                _pcl_mask = torch.zeros(_p)
            else:
                if _pcl.shape[1] < max_PCL:
                    n_dup = int(np.ceil((max_PCL - _pcl.shape[1]) / _pcl.shape[1]))
                    _pcl = torch.cat([_pcl] * (n_dup + 1), 1)[:, :max_PCL]
                if _pcl.shape[2] < max_PCL_dim:
                    _pcl = torch.cat(
                        [_pcl, torch.zeros(_p, max_PCL, max_PCL_dim - _pcl.shape[2])],
                        2,
                    )

            # --- Pad T dimension ---
            if _t < max_T:
                _trace = torch.cat(
                    [_trace, torch.eye(4)[None, None].repeat(_p, max_T - _t, 1, 1)],
                    1,
                )
                _trace_mask = torch.cat(
                    [_trace_mask, torch.zeros(_p, max_T - _t)], 1
                )
                _known_mask = torch.cat(
                    [_known_mask, torch.zeros(_p, max_T - _t, dtype=torch.bool)], 1
                )
                _time = torch.cat([_time, torch.zeros(max_T - _t)], 0)

            # --- Pad C dimension ---
            if _c < max_C:
                _context = torch.cat(
                    [_context, torch.eye(4)[None, None].repeat(_p, max_C - _c, 1, 1)],
                    1,
                )
                _context_mask = torch.cat(
                    [_context_mask, torch.zeros(_p, max_C - _c)], 1
                )
                _ctx_known_mask = torch.cat(
                    [_ctx_known_mask, torch.zeros(_p, max_C - _c, dtype=torch.bool)], 1
                )

            # --- Pad P dimension ---
            if _p < max_P:
                pad_p = max_P - _p
                _trace = torch.cat(
                    [_trace, torch.eye(4)[None, None].repeat(pad_p, max_T, 1, 1)], 0
                )
                _trace_mask = torch.cat(
                    [_trace_mask, torch.zeros(pad_p, max_T)], 0
                )
                _known_mask = torch.cat(
                    [_known_mask, torch.zeros(pad_p, max_T, dtype=torch.bool)], 0
                )
                _context = torch.cat(
                    [_context, torch.eye(4)[None, None].repeat(pad_p, max_C, 1, 1)], 0
                )
                _context_mask = torch.cat(
                    [_context_mask, torch.zeros(pad_p, max_C)], 0
                )
                _ctx_known_mask = torch.cat(
                    [_ctx_known_mask, torch.zeros(pad_p, max_C, dtype=torch.bool)], 0
                )
                _group_id = torch.cat([_group_id, torch.zeros(pad_p)], 0)
                _rot_mask = torch.cat([_rot_mask, torch.zeros(pad_p)], 0)
                _sem_type = torch.cat([_sem_type, torch.zeros(pad_p)], 0)
                _uid = torch.cat([_uid, torch.zeros(pad_p)], 0)
                _pcl = torch.cat(
                    [_pcl, torch.zeros(pad_p, max(_pcl.shape[1], 1), max_PCL_dim)], 0
                )
                _pcl_mask = torch.cat([_pcl_mask, torch.zeros(pad_p)], 0)

            # --- Random permute group IDs ---
            _group_id = _group_id.long()
            fg_mask = _trace_mask.any(dim=1)  # (P,)
            old_gid = _group_id[fg_mask].clone()
            old_fg_gid = old_gid.unique()

            sorted_fg_gid, _ = old_fg_gid.sort()
            n_fg = len(sorted_fg_gid)
            if n_fg > 0 and (sorted_fg_gid != torch.arange(n_fg)).any():
                remap = torch.zeros(int(sorted_fg_gid.max()) + 1, dtype=torch.long)
                remap[sorted_fg_gid] = torch.arange(n_fg)
                old_gid = remap[old_gid]
                _group_id[fg_mask] = old_gid
                old_fg_gid = torch.arange(n_fg)

            new_fg_gid = old_fg_gid[torch.randperm(len(old_fg_gid))]
            if n_fg > 0:
                new_gid = torch.gather(new_fg_gid, 0, old_gid)
                _group_id[fg_mask] = new_gid

            if "gid" in extra_data[bid]:
                old_extra_gid = extra_data[bid]["gid"]
                if n_fg > 0 and (sorted_fg_gid != torch.arange(n_fg)).any():
                    full_remap = torch.zeros(
                        int(sorted_fg_gid.max()) + 1, dtype=torch.long
                    )
                    full_remap[sorted_fg_gid] = new_fg_gid
                    new_extra_gid = [int(full_remap[i]) for i in old_extra_gid]
                else:
                    new_extra_gid = [
                        int(new_fg_gid[i]) if i < len(new_fg_gid) else i
                        for i in old_extra_gid
                    ]
                extra_data[bid]["gid"] = new_extra_gid

            trace_list.append(_trace)
            trace_mask_list.append(_trace_mask)
            context_list.append(_context)
            context_mask_list.append(_context_mask)
            known_mask_list.append(_known_mask)
            ctx_known_mask_list.append(_ctx_known_mask)
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
            # Permute (B, P, T, ...) -> (B, T, P, ...) for model consumption
            "trace": torch.stack(trace_list, 0).float().permute(0, 2, 1, 3, 4),
            "trace_mask": torch.stack(trace_mask_list, 0).bool().permute(0, 2, 1),
            "context": torch.stack(context_list, 0).float().permute(0, 2, 1, 3, 4),
            "context_mask": torch.stack(context_mask_list, 0).bool().permute(0, 2, 1),
            # V10 NEW: known_mask follows same permute as trace_mask
            "known_mask": torch.stack(known_mask_list, 0).bool().permute(0, 2, 1),
            "context_known_mask": (
                torch.stack(ctx_known_mask_list, 0).bool().permute(0, 2, 1)
            ),
            "trace_group_id": torch.stack(group_id_list, 0).int(),
            "trace_rot_mask": torch.stack(rot_mask_list, 0).bool(),
            "trace_sem_type": torch.stack(sem_type_list, 0).int(),
            "trace_uid": torch.stack(uid_list, 0).int(),
            "time": torch.stack(time_list, 0).float(),
            "text": text_list,
            "local_pcl": torch.stack(pcl_list, 0).float(),  # (B, P, PCL, 3+)
            "local_pcl_mask": torch.stack(pcl_mask_list, 0).bool(),  # (B, P)
        }

        # Merge extra data
        all_keys = set.union(*[set(it.keys()) for it in extra_data])
        extra = {k: [] for k in all_keys}
        for k in all_keys:
            for it in extra_data:
                extra[k].append(it.get(k, None))
        ret["extra"] = extra
        return ret
