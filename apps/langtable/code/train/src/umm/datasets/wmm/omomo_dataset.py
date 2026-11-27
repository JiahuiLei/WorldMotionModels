"""OMOMO human-object interaction dataset for UMM wmm scaffold.

CHOIS-aligned protocol: SMPL-X 22-joint body trajectories alongside a single
rigid object SE(3) trace with local point-cloud conditioning.  Follows the
V10+ ``CanonicalDatasetOutput`` contract with ``MaskScheduler`` integration.

Each sample contains 22 body joint SE(3) traces (group 0) and 1 rigid object
SE(3) trace (group 1) with a local point cloud sampled from the CHOIS
rest-pose object geometry.

Joint encoding
--------------
We do NOT run SMPL-X FK in the dataset.  Instead each body joint is encoded
as an SE(3) carrying the SMPL-X axis-angle parameter directly:

* Joint 0 (pelvis): rotation = ``axis_angle_to_matrix(root_orient_aa[t])``,
  translation = ``root_trans[t]`` (world frame).  Semantic ``SEM_HUMAN_GLOBAL``.
* Joints 1..21: rotation = ``axis_angle_to_matrix(pose_body_aa[t, j-1])``,
  translation = 0 (local-rotation only).  Semantic ``SEM_HUMAN``.

This preserves the OMOMO Q12 npz fields verbatim and matches the SMPL-X
parameterization used by CHOIS during training.  The wmm Agent learns the
forward-kinematics relationship from these per-joint local rotations.

Object encoding
---------------
Single object trace from ``obj_rot`` (3x3) + ``obj_trans`` (per-frame).
``obj_scale`` is intentionally ignored — CHOIS rest-pose convention is
``obj_rot @ rest + obj_com_pos`` with NO scale (protocol fix #2).

Local PCL: 256 xyz_rgb points sampled once per sequence from
``rest_object_geo/<obj_category>.ply``.  PLYs without vertex colors get
zero-filled RGB (HUMOTO PointNet NaN-grad lesson).

Example
-------
>>> ds = OmomoDataset(data_root='./data_wmm/omomo', split='test')
>>> len(ds)
482
>>> sample, extra = ds[0]
>>> sample.trace.shape
torch.Size([23, 120, 4, 4])
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


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues axis-angle → 3x3 rotation matrix (batched).

    Args:
        aa: ``(..., 3)`` axis-angle tensor (angle = magnitude, axis = direction).

    Returns:
        ``(..., 3, 3)`` rotation matrix tensor (same dtype/device as ``aa``).

    For ``angle < eps`` (1e-6) we fall back to the identity rotation; this
    avoids the Rodrigues axis-normalization singularity at zero. SMPL-X
    body-joint axis-angles are in the order of 0.1-2.0 rad in practice,
    so the eps cutoff never triggers on real OMOMO data.
    """
    eps = 1.0e-6
    angle = aa.norm(dim=-1, keepdim=True)        # (..., 1)
    is_small = angle < eps                        # (..., 1) bool
    angle_safe = torch.where(is_small, torch.ones_like(angle), angle)
    axis = aa / angle_safe                        # (..., 3); garbage where is_small
    ax, ay, az = axis.unbind(dim=-1)              # each (...,)
    zero = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zero, -az, ay], dim=-1),
            torch.stack([az, zero, -ax], dim=-1),
            torch.stack([-ay, ax, zero], dim=-1),
        ],
        dim=-2,
    )                                             # (..., 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    sin_a = angle.sin().unsqueeze(-1)             # (..., 1, 1)
    cos_a = angle.cos().unsqueeze(-1)
    R = eye + sin_a * K + (1 - cos_a) * (K @ K)
    # Where angle was below eps, force identity rotation.
    R = torch.where(is_small.unsqueeze(-1), eye, R)
    return R


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
        self.max_T = _T_FIXED
        self.max_C = 1

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

    def _build_body_se3(
        self,
        root_trans: np.ndarray,    # (T, 3)
        root_orient_aa: np.ndarray,  # (T, 3)
        pose_body_aa: np.ndarray,    # (T, 21, 3)
    ) -> torch.Tensor:
        """Construct ``(22, T, 4, 4)`` per-joint SE(3) tensor.

        Joint 0 carries world-frame ``(R(root_orient_aa), root_trans)``.
        Joints 1..21 carry local-frame ``(R(pose_body_aa[:, j-1]), 0)``.
        """
        root_t = torch.as_tensor(root_trans, dtype=torch.float32)        # (T,3)
        root_aa = torch.as_tensor(root_orient_aa, dtype=torch.float32)   # (T,3)
        body_aa = torch.as_tensor(pose_body_aa, dtype=torch.float32)     # (T,21,3)
        T = root_t.shape[0]
        assert root_aa.shape == (T, 3)
        assert body_aa.shape == (T, 21, 3)

        # All 22 joints' axis-angle stacked: (T, 22, 3)
        all_aa = torch.cat([root_aa.unsqueeze(1), body_aa], dim=1)       # (T,22,3)
        rot = _axis_angle_to_matrix(all_aa)                              # (T,22,3,3)

        Q = torch.eye(4, dtype=torch.float32).expand(T, 22, 4, 4).contiguous()
        Q[..., :3, :3] = rot
        # World translation only on pelvis (joint 0); others stay 0
        Q[:, 0, :3, 3] = root_t
        # Permute to (P=22, T, 4, 4) per Prompt convention
        return Q.permute(1, 0, 2, 3).contiguous()

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
            root_trans = data["root_trans"]                # (120,3)
            root_orient_aa = data["root_orient_aa"]        # (120,3)
            pose_body_aa = data["pose_body_aa"]            # (120,21,3)
            # obj_rot is the schema name in the actual npz files (the protocol
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
            # (~46% of train npz have actual_len < 120).
            if "actual_len" in data.files:
                actual_len = int(data["actual_len"].item())
            else:
                actual_len = int(root_trans.shape[0])
            text = self._load_text(data, entry)

        T = int(root_trans.shape[0])
        if T != _T_FIXED:
            raise ValueError(f"OmomoDataset: {entry} has T={T}, expected {_T_FIXED}")
        if not 1 <= actual_len <= T:
            raise ValueError(
                f"OmomoDataset: {entry} has actual_len={actual_len}, "
                f"expected 1..{T}"
            )

        # Drop padded waypoints so chois_obj_waypoints_to_all does NOT
        # reveal a zero-pad frame as a known waypoint.
        waypoint_indices = [int(w) for w in waypoint_indices if 0 <= int(w) < actual_len]
        # Replace zero-padded trailing trace data with identity so any
        # downstream code that ignores trace_mask (matrix-log, quat) cannot
        # NaN. Mask is still the source of truth.
        if actual_len < T:
            root_trans = np.asarray(root_trans).copy()
            root_orient_aa = np.asarray(root_orient_aa).copy()
            pose_body_aa = np.asarray(pose_body_aa).copy()
            obj_rot = np.asarray(obj_rot).copy()
            obj_trans = np.asarray(obj_trans).copy()
            root_trans[actual_len:] = 0.0
            root_orient_aa[actual_len:] = 0.0     # axis-angle 0 → identity rot
            pose_body_aa[actual_len:] = 0.0
            obj_rot[actual_len:] = np.eye(3, dtype=obj_rot.dtype)
            obj_trans[actual_len:] = 0.0

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
        body_se3 = self._build_body_se3(root_trans, root_orient_aa, pose_body_aa)
        body_ctx = torch.eye(4, dtype=torch.float32).expand(
            _NUM_BODY_JOINTS, self.max_C, 4, 4,
        ).contiguous()
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
        obj_se3 = self._build_object_se3(obj_rot, obj_trans)
        obj_ctx = torch.eye(4, dtype=torch.float32).expand(1, self.max_C, 4, 4).contiguous()
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
        }
        return ret, omomo_extra
