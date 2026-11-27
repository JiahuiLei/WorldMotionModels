"""OMOMO human-object interaction dataset for UMM wmm scaffold.

CHOIS-aligned protocol: SMPL-X 22-joint body trajectories alongside a single
rigid object SE(3) trace with local point-cloud conditioning.  Follows the
V10+ ``CanonicalDatasetOutput`` contract with ``MaskScheduler`` integration.

Each sample contains 22 body joint SE(3) traces (group 0) and 1 rigid object
SE(3) trace (group 1) with a local point cloud sampled from the CHOIS
rest-pose object geometry.

Joint encoding — UMM real world-frame SE(3) convention
-----------------------------------------------------
The dataset reads the pre-computed ``Q`` field from v2 npz files
(``(T, 22, 4, 4)`` float32).  Each entry ``Q[t, k]`` is the **world-frame
SE(3)** of SMPL-X joint ``k`` at frame ``t``:

* ``Q[t, k, :3, :3]`` = world joint orientation (from SMPL-X FK).
* ``Q[t, k, :3, 3]``  = world joint position   (from SMPL-X FK).
* ``Q[t, k, 3, :]``   = ``[0, 0, 0, 1]``.

This matches the UMM per-joint real-SE(3) convention used across all wmm
datasets — no axis-angle, no in-dataset FK, no Rodrigues.  The Agent
consumes full world poses directly.

The ``Q`` field is generated at convert time from pickle fields via
``prepare_data/omomo/build_q.py::build_q_from_pickle_fields`` (SMPL-X FK on
extracted root_trans + root_orient_aa + pose_body_aa).  The inverse map
``build_q.se3_to_axis_angle`` is a lossless round-trip (verified at
1.19e-07 max err in ``test/wmm/test_omomo_pickle_path_roundtrip.py``) and
is used by the eval callback to write pred npz in CHOIS schema.

The axis-angle fields ``root_trans``/``root_orient_aa``/``pose_body_aa``
are retained in the v2 npz for eval-pipeline compatibility but are **not**
read by this dataset.

Object encoding
---------------
Single object trace from ``obj_rot`` (3x3) + ``obj_trans`` (per-frame).
``obj_scale`` is intentionally ignored — CHOIS rest-pose convention is
``obj_rot @ rest + obj_com_pos`` with NO scale.

Local PCL: 256 xyz_rgb points sampled once per sequence from
``rest_object_geo/<obj_category>.ply``.  PLYs without vertex colors get
zero-filled RGB (HUMOTO PointNet NaN-grad lesson).

Example
-------
>>> ds = OmomoDataset(data_root='./data_wmm/omomo_v2', split='test')
>>> len(ds)
482
>>> sample, extra = ds[0]
>>> sample.trace.shape
torch.Size([23, 120, 4, 4])
>>> extra["betas"].shape, extra["gender"]
((16,), 'male')
"""

import json
import logging
import os
import os.path as osp
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from umm.datasets.wmm.base_dataset import PromptV0
from umm.datasets.wmm.dataset_template import CanonicalDatasetOutput
from umm.datasets.wmm.dataset_utils import sample_list
from umm.datasets.wmm.dataset_config import (
    SEM_HUMAN,
    SEM_HUMAN_GLOBAL,
    SEM_OBJECT,
    UID_HUMOTO_OBJECT_GROUP,
    UID_SMPLX55_GROUP,
)
from umm.datasets.wmm.mask_scheduler import (
    MaskScheduler,
    OMOMO_PATTERNS,
)

_LOG = logging.getLogger(__name__)

# OMOMO uses SMPL-X body joints 0..21 (root + 21 body joints)
_NUM_BODY_JOINTS = 22
# Fixed window length per CHOIS protocol
_T_FIXED = 120
# Strip suffix from obj_mesh_path to get object category name
_MESH_PATH_SUFFIX_RE = re.compile(r"_cleaned_simplified\.obj$")


def _ply_to_pcl(ply_path: str) -> np.ndarray:
    """Load a PLY mesh and return ``(N, 6)`` xyz+rgb point array.

    RGB is rescaled to ``[-0.5, 0.5]`` matching the wmm pcl convention
    (see ``hh_dataset._precomputed_pcl``).  Missing vertex colors are
    filled with zeros (HUMOTO lesson — never NaN, kills PointNet grads).

    Args:
        ply_path: Path to a PLY file.

    Returns:
        ``(N, 6)`` float32 array: ``xyz`` (object-local frame) + ``rgb``.
    """
    if not osp.exists(ply_path):
        raise FileNotFoundError(f"OmomoDataset: rest-object PLY missing: {ply_path}")
    mesh = trimesh.load(ply_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)  # (N, 3)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        cols = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)  # (N, 4)
        cols = cols[:, :3]  # drop alpha
    else:
        cols = np.zeros((verts.shape[0], 3), dtype=np.float32)
    # Sanitize NaN BEFORE rescale (HUMOTO PCL NaN bug)
    np.nan_to_num(cols, copy=False, nan=0.0)
    cols = (cols / 255.0) - 0.5  # [-0.5, 0.5]
    pcl = np.concatenate([verts, cols.astype(np.float32)], axis=1)  # (N, 6)
    if not np.isfinite(pcl).all():
        raise ValueError(f"OmomoDataset: non-finite values in PCL {ply_path}")
    return pcl


def _obj_category_from_mesh_path(mesh_path: str) -> str:
    """Map ``"<cat>_cleaned_simplified.obj"`` → ``"<cat>"``.

    Fails loud on any unexpected pattern.
    """
    base = osp.basename(mesh_path)
    match = _MESH_PATH_SUFFIX_RE.search(base)
    if match is None:
        raise ValueError(
            f"OmomoDataset: obj_mesh_path {mesh_path!r} does not match "
            f"'<category>_cleaned_simplified.obj' pattern"
        )
    return base[: match.start()]


class OmomoDataset(Dataset):
    """OMOMO Q12 .npz dataset for the wmm CHOIS-aligned protocol.

    Args:
        data_root: Root directory holding ``test_npz/``, ``train_npz/``,
            ``rest_object_geo/``, ``omomo_text_anno/{train,test}/``,
            ``splits.json``.
        split: One of ``"train"``, ``"test"``, or ``"val"`` (alias of test).
        random_flag: Random PCL sub-sampling.
        safe_flag: If True, skip on per-sample errors; if False, raise.
        repeat: Index repetition multiplier (used to lengthen short splits).
        num_pts: Local PCL points per sample (default 256).
        text_required: If True, missing text raises (default True — fail loud).
        max_T: Effective time-window length the dataset returns (default
            120 == ``_T_FIXED``). Setting ``max_T < 120`` crops trace,
            mask, and time arrays to the first ``max_T`` frames; padded
            waypoints beyond ``max_T`` are dropped, and ``actual_len`` is
            clipped. The on-disk NPZ always holds 120 frames; this knob
            only changes what the dataset emits.
        max_C: Number of context slots replicated from the (per-joint or
            object-rest-pose) single-frame ctx (default 1). The ctx
            tensor is ``.expand``'d across the C-axis, so all slots share
            the same content.
        mask_scheduler: Optional explicit ``MaskScheduler``; defaults to
            ``OMOMO_PATTERNS`` with ``base_random_prob=0.25``.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        random_flag: bool = True,
        safe_flag: bool = True,
        repeat: int = 1,
        num_pts: int = 256,
        text_required: bool = True,
        max_T: int = 120,
        max_C: int = 1,
        mask_scheduler: Optional[MaskScheduler] = None,
    ):
        super().__init__()
        if split == "val":
            split = "test"
        if split not in ("train", "test"):
            raise ValueError(
                f"OmomoDataset: split must be 'train'|'test'|'val', got {split!r}"
            )
        self.data_root = data_root
        self.split = split
        self.random_flag = random_flag
        self.safe_flag = safe_flag
        self.repeat = max(1, int(repeat))
        self.num_pts = int(num_pts)
        self.text_required = bool(text_required)

        # NPZ + text directories
        self.npz_dir = osp.join(data_root, f"{split}_npz")
        self.text_dir = osp.join(data_root, "omomo_text_anno", split)
        self.geo_dir = osp.join(data_root, "rest_object_geo")
        # SMPL-X model dir for DA-pose ctx (standing-body prior in body_ctx).
        # omomo_v2 symlinks smplx -> ../omomo/smplx which holds SMPLX_{GENDER}.npz.
        self._omomo_smplx_dir = osp.join(data_root, "smplx")
        for d in (self.npz_dir, self.text_dir, self.geo_dir):
            if not osp.isdir(d):
                raise FileNotFoundError(f"OmomoDataset: missing directory {d}")

        # Load split list — each entry is e.g. "sub16_clothesstand_000_w0"
        splits_path = osp.join(data_root, "splits.json")
        with open(splits_path, "r") as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(
                f"OmomoDataset: splits.json has no '{split}' key (keys={list(splits)})"
            )
        self.entries: List[str] = list(splits[split])
        self.indices = list(range(len(self.entries))) * self.repeat
        # base_fps + t_rate kept for downstream compat (OMOMO is fixed T=120)
        self.base_fps = 30
        self.t_rate = 1
        self.max_T = int(max_T)
        self.max_C = int(max_C)
        assert 1 <= self.max_T <= _T_FIXED, (
            f"OmomoDataset: max_T={self.max_T} must be in [1, {_T_FIXED}]"
        )
        assert self.max_C >= 1, (
            f"OmomoDataset: max_C={self.max_C} must be >= 1"
        )

        # Object category → cached (N,6) PCL
        self._pcl_cache: Dict[str, np.ndarray] = {}

        # Mask scheduler — domain default uses OMOMO_PATTERNS
        self.mask_scheduler = mask_scheduler or MaskScheduler(
            base_random_prob=0.25, patterns=OMOMO_PATTERNS,
        )

        _LOG.info(
            "OmomoDataset(%s): %d entries, repeat=%d, num_pts=%d",
            split, len(self.entries), self.repeat, self.num_pts,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.safe_flag:
            tries = 0
            while True:
                try:
                    return self.__getitem_inner__(idx)
                except Exception as e:
                    logging.error(f"OmomoDataset: error at idx={idx}: {e}")
                    idx = (idx + 1) % len(self)
                    tries += 1
                    if tries > 64:
                        raise RuntimeError(
                            f"OmomoDataset: 64 consecutive errors starting at idx={idx}"
                        ) from e
        return self.__getitem_inner__(idx)

    @staticmethod
    def worker_init_fn(worker_id):
        """Single-thread numpy/openblas inside dataloader workers."""
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entry_to_seq_base(self, entry: str) -> str:
        """``"sub16_clothesstand_000_w0"`` → ``"sub16_clothesstand_000"``.

        The trailing ``_w<int>`` is the window index; text annotations are
        keyed by the window-stripped seq name.  Fails loud if no ``_w`` suffix.
        """
        match = re.search(r"_w\d+$", entry)
        if match is None:
            raise ValueError(
                f"OmomoDataset: entry {entry!r} missing '_w<int>' suffix"
            )
        return entry[: match.start()]

    def _load_text(self, npz_data, entry: str) -> str:
        """Read motion description from ``omomo_text_anno/{split}/<seq_base>.json``.

        Falls through to the ``text`` field of the npz only if the JSON file
        is missing AND ``text_required`` is False.  Fails loud otherwise.
        """
        seq_base = self._entry_to_seq_base(entry)
        json_path = osp.join(self.text_dir, f"{seq_base}.json")
        if osp.exists(json_path):
            with open(json_path, "r") as f:
                d = json.load(f)
            if seq_base in d:
                return str(d[seq_base])
            # Some files may have the description as the sole value
            vals = list(d.values())
            if len(vals) == 1 and isinstance(vals[0], str):
                return vals[0]
            raise KeyError(
                f"OmomoDataset: text JSON {json_path} has no key {seq_base!r} "
                f"(keys={list(d)[:5]})"
            )
        # Fall-through: no JSON file — try npz "text" field
        if "text" in npz_data.files:
            t = npz_data["text"].item()
            if isinstance(t, str) and t:
                return t
        if self.text_required:
            raise FileNotFoundError(
                f"OmomoDataset: text JSON missing and npz['text'] empty for {entry}"
            )
        return ""

    def _get_pcl(self, obj_category: str) -> np.ndarray:
        """Load + cache the (N,6) rest-pose PCL for an object category."""
        if obj_category not in self._pcl_cache:
            ply_path = osp.join(self.geo_dir, f"{obj_category}.ply")
            self._pcl_cache[obj_category] = _ply_to_pcl(ply_path)
        return self._pcl_cache[obj_category]

    def _build_object_se3(
        self,
        obj_rot: np.ndarray,    # (T, 3, 3)
        obj_trans: np.ndarray,  # (T, 3)
    ) -> torch.Tensor:
        """Construct ``(1, T, 4, 4)`` SE(3) for the rigid object."""
        rot_t = torch.as_tensor(obj_rot, dtype=torch.float32)
        trans_t = torch.as_tensor(obj_trans, dtype=torch.float32)
        T = rot_t.shape[0]
        assert rot_t.shape == (T, 3, 3)
        assert trans_t.shape == (T, 3)
        Q = torch.eye(4, dtype=torch.float32).expand(T, 4, 4).contiguous()
        Q[:, :3, :3] = rot_t
        Q[:, :3, 3] = trans_t
        return Q.unsqueeze(0)  # (1, T, 4, 4)

    # ------------------------------------------------------------------
    # Inner getitem
    # ------------------------------------------------------------------

    def __getitem_inner__(self, idx: int) -> Tuple[CanonicalDatasetOutput, Dict]:
        idx = self.indices[idx % len(self.indices)]
        entry = self.entries[idx]
        npz_path = osp.join(self.npz_dir, f"{entry}.npz")
        if not osp.exists(npz_path):
            raise FileNotFoundError(f"OmomoDataset: missing npz {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            # --- Body: v2 schema requires the pre-computed Q field (world-frame
            # per-joint SE(3)). Raise loud on v1 npz — no AA → SE(3) fallback.
            if "Q" not in data.files:
                raise KeyError(
                    f"OmomoDataset requires v2 npz (has 'Q' field). "
                    f"Regenerate with prepare_data/omomo/convert.py "
                    f"--output_subdir_suffix=_v2. Got v1 npz at: {npz_path}"
                )
            Q_np = data["Q"]                                # (120,22,4,4) float32
            # betas + gender plumbed to viz/eval callback via extra
            betas = np.asarray(data["betas"], dtype=np.float32)      # (16,)
            # Wrist positions for w1.6+ hand-object loss (must read inside `with` block)
            lhand_wrist_jpos = np.asarray(data["lhand_wrist_jpos"], dtype=np.float32)  # (T, 3)
            rhand_wrist_jpos = np.asarray(data["rhand_wrist_jpos"], dtype=np.float32)  # (T, 3)
            gender = str(data["gender"].item())                      # 'male'|'female'|'neutral'
            # obj_rot is the schema name in the actual npz files (the reference
            # text uses "obj_rot_mat" but the on-disk key is "obj_rot").
            if "obj_rot" in data.files:
                obj_rot = data["obj_rot"]                  # (120,3,3)
            elif "obj_rot_mat" in data.files:
                obj_rot = data["obj_rot_mat"]
            else:
                raise KeyError(
                    f"OmomoDataset: {entry} has neither 'obj_rot' nor 'obj_rot_mat'"
                )
            obj_trans = data["obj_trans"]                  # (120,3)
            obj_mesh_path = str(data["obj_mesh_path"].item())
            seq_name = str(data["seq_name"].item()) if "seq_name" in data.files else entry
            waypoint_indices = (
                data["waypoint_indices"].astype(np.int64).tolist()
                if "waypoint_indices" in data.files else []
            )
            # actual_len < T means the window is at a sequence boundary and
            # frames [actual_len, T) are zero-padded. We MUST mask them out;
            # otherwise the model trains on degenerate "teleport-to-origin"
            # trajectories with a non-orthogonal R=zeros rotation matrix
            #.
            if "actual_len" in data.files:
                actual_len = int(data["actual_len"].item())
            else:
                actual_len = int(Q_np.shape[0])
            text = self._load_text(data, entry)

        T = int(Q_np.shape[0])
        if T != _T_FIXED:
            raise ValueError(f"OmomoDataset: {entry} has T={T}, expected {_T_FIXED}")
        if Q_np.shape[1:] != (_NUM_BODY_JOINTS, 4, 4):
            raise ValueError(
                f"OmomoDataset: {entry} Q has shape {Q_np.shape}, "
                f"expected ({T}, {_NUM_BODY_JOINTS}, 4, 4)"
            )
        if not 1 <= actual_len <= T:
            raise ValueError(
                f"OmomoDataset: {entry} has actual_len={actual_len}, "
                f"expected 1..{T}"
            )

        # Drop padded waypoints so chois_obj_waypoints_to_all does NOT
        # reveal a zero-pad frame as a known waypoint.
        waypoint_indices = [int(w) for w in waypoint_indices if 0 <= int(w) < actual_len]
        # Defensive identity-padding guard. The v2 convert pipeline already
        # identity-pads Q[actual_len:] and obj_rot[actual_len:] at write time
        # (see prepare_data/omomo/convert.py), so this is a NO-OP on v2 npz.
        # trace_mask is the authoritative source of truth for valid frames.
        if actual_len < T:
            obj_rot = np.asarray(obj_rot).copy()
            obj_trans = np.asarray(obj_trans).copy()
            obj_rot[actual_len:] = np.eye(3, dtype=obj_rot.dtype)
            obj_trans[actual_len:] = 0.0

        # max_T crop: when the dataset is configured to emit a shorter
        # window than the on-disk 120 frames, crop time-axis arrays to the
        # first `max_T` frames. Done AFTER identity-padding (which used
        # the full-T arrays). Waypoints beyond `max_T` are dropped, and
        # `actual_len` is clipped. When max_T == _T_FIXED this is a no-op.
        if self.max_T < T:
            waypoint_indices = [w for w in waypoint_indices if w < self.max_T]
            actual_len = min(actual_len, self.max_T)
            T = self.max_T

        # ------------------------------------------------------------------
        # Build prompt: 22 body joints (group 0) + 1 object (group 1)
        # ------------------------------------------------------------------
        prompt = PromptV0(
            P=_NUM_BODY_JOINTS + 1,
            T=self.max_T,
            C=self.max_C,
            PCL=self.num_pts,
            PCL_dim=6,
        )

        # Per-token trace_mask: True for [0, actual_len), False for padded tail.
        body_trace_mask = torch.zeros(_NUM_BODY_JOINTS, T, dtype=torch.bool)
        body_trace_mask[:, :actual_len] = True
        obj_trace_mask = torch.zeros(1, T, dtype=torch.bool)
        obj_trace_mask[:, :actual_len] = True

        # ---- Body
        # Q_np: (T_full, 22, 4, 4) world-frame SE(3) per joint, already built at
        # convert time by SMPL-X FK (prepare_data/omomo/build_q.py).
        # Permute to the wmm trace convention (P, T_full, 4, 4) and crop to
        # `max_T` along the time axis (no-op when max_T == _T_FIXED).
        body_se3 = torch.as_tensor(Q_np, dtype=torch.float32).permute(
            1, 0, 2, 3,
        ).contiguous()[:, :self.max_T]                                  # (22,T,4,4)
        # Body ctx: DA-pose world-frame SE(3) per joint — a canonical STANDING
        # reference that gives the model (and the Ctx visualizer) a meaningful
        # prior. Replaces identity-SE(3) per joint which rendered as a
        # "collapsed humanoid blob" at origin under the real-SE(3) convention.
        # Shape-independent approximation (neutral gender, zeros betas) so the
        # ctx is a single cached pose shared across all samples — body mesh
        # viz/eval still uses per-subject LBS via Q at runtime, so the shape
        # approximation here is inconsequential to mesh fidelity.
        from prepare_data.omomo.build_q import da_pose_q  # local: avoid circularity
        _da_q = da_pose_q(
            gender="neutral",
            betas=np.zeros(16, dtype=np.float32),
            smplx_dir=self._omomo_smplx_dir,
        )                                                                # (22, 4, 4) float32
        body_ctx = torch.as_tensor(_da_q, dtype=torch.float32)[:, None].expand(
            _NUM_BODY_JOINTS, self.max_C, 4, 4,
        ).contiguous()
        assert body_ctx.shape == (_NUM_BODY_JOINTS, self.max_C, 4, 4), body_ctx.shape
        body_sem = np.full((_NUM_BODY_JOINTS,), SEM_HUMAN, dtype=np.int64)
        body_sem[0] = SEM_HUMAN_GLOBAL
        body_uids = np.asarray(UID_SMPLX55_GROUP[:_NUM_BODY_JOINTS], dtype=np.int64)
        prompt.append_trace(
            trace=body_se3,                                          # (22,T,4,4)
            trace_mask=body_trace_mask,                              # (22,T)
            context=body_ctx,                                        # (22,C,4,4)
            context_mask=torch.ones(_NUM_BODY_JOINTS, self.max_C),   # (22,C)
            group_id=0,
            sem_type=body_sem,
            uid=body_uids,
        )

        # ---- Object (single)
        obj_category = _obj_category_from_mesh_path(obj_mesh_path)
        # Crop obj_se3 to `max_T` along time axis (no-op when max_T==_T_FIXED).
        obj_se3 = self._build_object_se3(obj_rot, obj_trans)[:, :self.max_T]
        obj_ctx = torch.eye(4, dtype=torch.float32).expand(1, self.max_C, 4, 4).contiguous()
        assert obj_ctx.shape == (1, self.max_C, 4, 4), obj_ctx.shape
        pcl_raw = self._get_pcl(obj_category)                        # (N_full, 6)
        pcl_sampled = sample_list(pcl_raw, self.num_pts, self.random_flag)  # (num_pts,6)
        local_pcl = torch.as_tensor(pcl_sampled, dtype=torch.float32).unsqueeze(0)
        prompt.append_trace(
            trace=obj_se3,                                           # (1,T,4,4)
            trace_mask=obj_trace_mask,                               # (1,T)
            context=obj_ctx,                                         # (1,C,4,4)
            context_mask=torch.ones(1, self.max_C),                  # (1,C)
            group_id=1,
            sem_type=SEM_OBJECT,
            uid=int(UID_HUMOTO_OBJECT_GROUP[0]),
            local_pcl=local_pcl,                                     # (1,num_pts,6)
            local_pcl_mask=torch.ones(1),                            # (1,)
        )

        prompt.catch_P()
        trace, trace_mask, context, context_mask, rot_mask, group_id, sem_type, uid = (
            prompt.cat()
        )

        # current_ith for OMOMO == 0 (window starts at the anchor frame)
        current_ith = 0
        known_mask, ctx_known_mask = self.mask_scheduler.sample(
            trace_mask=trace_mask,
            context_mask=context_mask,
            sem_type=sem_type,
            current_ith=current_ith,
            waypoint_indices=waypoint_indices,
            reveal_body_t0=True,
        )

        time_unit = float(self.t_rate) / float(self.base_fps)
        time_stamp = (torch.arange(self.max_T) - current_ith) * time_unit

        local_pcl_full, local_pcl_mask_full = prompt.cat_pcl()

        ret = CanonicalDatasetOutput(
            seq_name=seq_name,
            trace=trace,                          # (P, T, 4, 4)
            trace_mask=trace_mask,                # (P, T)
            context=context,                      # (P, C, 4, 4)
            context_mask=context_mask,            # (P, C)
            known_mask=known_mask,                # (P, T)
            context_known_mask=ctx_known_mask,    # (P, C)
            trace_group_id=group_id,              # (P,)
            trace_rot_mask=rot_mask,              # (P,)
            trace_sem_type=sem_type,              # (P,)
            trace_uid=uid,                        # (P,)
            time=time_stamp,                      # (T,)
            text=text,
            local_pcl=local_pcl_full,             # (P, num_pts, 6)
            local_pcl_mask=local_pcl_mask_full,   # (P,)
        )
        omomo_extra = {
            "obj_category": obj_category,
            "obj_mesh_path": obj_mesh_path,
            "waypoint_indices": waypoint_indices,
            "actual_len": actual_len,
            "entry": entry,
            "betas": betas,      # (16,) float32 — per-sample SMPL-X shape
            "gender": gender,    # 'male'|'female'|'neutral' — SMPL-X model key
            "lhand_wrist_jpos": lhand_wrist_jpos[:self.max_T],  # (T, 3) read inside `with` block above
            "rhand_wrist_jpos": rhand_wrist_jpos[:self.max_T],  # (T, 3) read inside `with` block above
        }
        return ret, omomo_extra
