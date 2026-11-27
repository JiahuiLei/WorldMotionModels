"""Physical realism metric for hand-object interaction evaluation.

Implements the contact-and-no-penetration criterion from Text2HOI
`get_penet_hand_obj_loss` / `get_penet_hand_obj_loss` (loss.py:201),
using 778-vertex MANO hand meshes and per-object point clouds (obj.pkl).

A frame is physically realistic iff (for the union of both hands):
  (1) At least one MANO fingertip vertex is within `contact_thresh_m`
      of the nearest object point-cloud point (KNN distance).
  (2) No object point-cloud point penetrates the hand mesh by more than
      `penetration_thresh_m`.  Penetration follows Text2HOI exactly:
        nn_dist, nn_idx = get_NN(obj_pc_world, hand_verts)
        interior = get_interior(hand_normals, hand_verts, obj_pc_world, nn_idx)
        penet_depth = sqrt(nn_dist[interior]).max()  # 0.0 if none interior
        no_penetration = (penet_depth < penetration_thresh_m)

Both conditions must hold simultaneously for a frame to be ``realistic``.
Both hands are evaluated jointly:
  - contact := OR across left and right hands
  - no_penetration := AND across left and right hands

MANO decode (world frame):
  Canonical sample provides:
    left_hand_joints  (T, 21, 3) — world-frame MANO joints + tips; joint 0 = wrist
    left_hand_rot     (T, 16, 3) — axis-angle: rot[:,0,:] = global_orient (1),
                                   rot[:,1:16,:] = hand_pose (15 joints x 3 AA)

  Procedure per frame t:
    1. Call smplx.MANO(global_orient=rot[t,0,:], hand_pose=rot[t,1:16,:].reshape(45))
       with betas=zeros(10), transl=zeros(3).
    2. MANO returns verts_canonical (778, 3) and joints_canonical (16, 3).
       joint[0] = wrist in the MANO canonical frame (NOT at origin — MANO has
       a fixed offset). To shift to world frame:
         offset = joint_canonical[0] - j_hand_world[t, 0, :]
       BUT: actually we translate so that MANO's joint[0] aligns with the
       world wrist position from the canonical sample:
         verts_world = verts_canonical - joint_canonical[0] + wrist_world
       where wrist_world = j_hand[t, 0, :].
    3. Compute trimesh vertex normals from (verts_world, MANO_faces).

Fingertip vertex indices (from Text2HOI lib/models/mano.py line 92-96):
  thumb=745, index=317, middle=445, ring=556, pinky=673

Object point cloud:
  Loaded from obj.pkl['obj_pcs'][obj_name] — shape (N_pc, 3), object-local frame.
  Transformed per frame: obj_pc_world = (R @ obj_pc.T).T + t
  using sample.obj_rotation[t] (3,3) and sample.obj_translation[t] (3,).

DATA_ROOT resolution (first match wins):
  1. DATA_ROOT environment variable if set and non-empty.
  2. Walk up from this file until data/handobject/raw_registered found.
  3. Fallback to data/ under project root.
"""

import logging
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import smplx
import torch
import trimesh

from umm.eval.handobject.canonical import CanonicalSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MANO fingertip vertex indices — from Text2HOI lib/models/mano.py lines 92-96.
MANO_FINGERTIP_VERT_IDX: List[int] = [745, 317, 445, 556, 673]  # thumb, index, middle, ring, pinky

_DEFAULT_CONTACT_THRESH_M: float = 0.02    # 2 cm
_DEFAULT_PENETRATION_THRESH_M: float = 0.010  # 10 mm (GRAB-calibrated legacy default)
# NOTE: Text2HOI paper does not publish a threshold number.
# Calibrated from 120 GT frames (4 GRAB episodes): GT penet depths are in [0, 15mm],
# p90=9.9mm. At 5mm threshold GT realism = 0.15 (far below paper 0.808). At 10mm,
# GT realism = 0.90, consistent with paper. 10mm is the correct threshold FOR GRAB.
#
# diagnostic (
# zero, is too tight for ARCTIC, which has per-subject non-zero betas that our
# _decode_mano_world_batch currently ignores (betas=zeros in _decode_mano_world_batch).
# ARCTIC GT penet p90=12.67mm, p95=14.79mm — driven by wrong-sized MANO mesh rather
# than true grasp penetration (deepest_at_fingertip_frac=0.001 confirms palm/body,
# not fingertip). At 10mm threshold ARCTIC GT realism = 0.733; at 15mm it is 0.857.
# Path B (beta plumbing through CanonicalSample + adapters + MANO decode) is the
# we apply Path A — a dataset-aware default threshold mirroring 's
# _DATASET_FLAT_HAND_MEAN pattern. Explicit caller-passed values are always sacred.
_DATASET_PENETRATION_THRESH_M: Dict[str, float] = {
    "grab":   0.010,   # GRAB-calibrated (paper-matching, betas-are-zero dataset).
    "arctic": 0.015,   # ARCTIC compensation for ignored per-subject MANO betas.
}

_MANO_MODEL_PATH: str = (
    "data/handobject/raw_registered/mano_unzip/mano_v1_2/models/"
)

# ---------------------------------------------------------------------------
# Data-root resolution
# ---------------------------------------------------------------------------

def _data_root() -> str:
    """Return the data root directory.

    Resolution order (first match wins):
      1. DATA_ROOT environment variable if set and non-empty.
      2. Walk up from this file, looking for data/handobject/raw_registered/.
      3. Fallback to data/ under project root.
    """
    env = os.environ.get("DATA_ROOT", "")
    if env:
        return env

    here = os.path.abspath(__file__)
    candidate = here
    for _ in range(10):
        candidate = os.path.dirname(candidate)
        probe = os.path.join(candidate, "data", "handobject", "raw_registered")
        if os.path.isdir(probe):
            return os.path.join(candidate, "data")

    # Final fallback: data/ relative to project root (5 levels up from this file).
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(here)
                )
            )
        )
    )
    return os.path.join(project_root, "data")


# ---------------------------------------------------------------------------
# Mesh loading (kept for API compatibility with eval_imos.py)
# ---------------------------------------------------------------------------

_GRAB_MESH_DIR_RPATH = "handobject/raw_registered/grab_unzip/contact_meshes"
_ARCTIC_MESH_DIR_RPATH = (
    "handobject/raw_registered/arctic_unzip/meta/object_vtemplates"
)

_DATASET_NAMES = {"grab", "arctic"}


def load_object_mesh(dataset: str, obj_name: str) -> trimesh.Trimesh:
    """Load an object mesh for the given dataset and object name.

    Args:
        dataset:  One of 'grab' or 'arctic' (case-sensitive).
        obj_name: Lowercase object name as it appears in CanonicalSample.obj_name.

    Returns:
        A trimesh.Trimesh with vertices in the canonical object-local frame.

    Raises:
        ValueError: If dataset is not 'grab' or 'arctic'.
        FileNotFoundError: If the mesh file does not exist on disk.
    """
    if dataset not in _DATASET_NAMES:
        raise ValueError(
            f"load_object_mesh: dataset must be one of {sorted(_DATASET_NAMES)}, "
            f"got {dataset!r}"
        )

    data_root = _data_root()

    if dataset == "grab":
        mesh_path = os.path.join(
            data_root, _GRAB_MESH_DIR_RPATH, f"{obj_name}.ply"
        )
    else:  # arctic
        mesh_path = os.path.join(
            data_root, _ARCTIC_MESH_DIR_RPATH, obj_name, "mesh.obj"
        )

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            f"load_object_mesh: mesh file not found for dataset={dataset!r}, "
            f"obj_name={obj_name!r}. Expected path: {mesh_path}"
        )

    loaded = trimesh.load(mesh_path, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(
                f"load_object_mesh: loaded Scene for {mesh_path!r} "
                "contains no geometries."
            )
        loaded = trimesh.util.concatenate(geometries)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"load_object_mesh: expected trimesh.Trimesh after loading "
            f"{mesh_path!r}, got {type(loaded).__name__}"
        )

    # Unit harmonization: ARCTIC vtemplate meshes are millimeters.
    max_extent = float(loaded.extents.max())
    if max_extent > 10.0:
        loaded.apply_scale(1.0 / 1000.0)
        warnings.warn(
            f"load_object_mesh: mesh for dataset={dataset!r}, obj_name={obj_name!r} "
            f"had max extent {max_extent:.1f} (likely mm); rescaled to meters.",
            stacklevel=2,
        )

    return loaded


# ---------------------------------------------------------------------------
# Object point cloud loading
# ---------------------------------------------------------------------------

_OBJ_PKL_RPATH_GRAB = "handobject/text2hoi_raw/grab/obj.pkl"
_OBJ_PKL_RPATH_ARCTIC = "handobject/text2hoi_raw/arctic/obj.pkl"


def load_obj_pcs(dataset: str) -> Dict[str, np.ndarray]:
    """Load the object point cloud dict from obj.pkl.

    Args:
        dataset: 'grab' or 'arctic'.

    Returns:
        Dict[obj_name -> (N_pc, 3) float32 array in object-local frame].

    Raises:
        ValueError: If dataset is unknown.
        FileNotFoundError: If obj.pkl not found.
        KeyError: If 'obj_pcs' key missing in pkl.
    """
    if dataset not in _DATASET_NAMES:
        raise ValueError(f"load_obj_pcs: unknown dataset {dataset!r}")

    data_root = _data_root()
    rpath = _OBJ_PKL_RPATH_GRAB if dataset == "grab" else _OBJ_PKL_RPATH_ARCTIC
    pkl_path = os.path.join(data_root, rpath)

    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"load_obj_pcs: obj.pkl not found at {pkl_path}. "
            "Set DATA_ROOT or check data layout."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "obj_pcs" not in data:
        raise KeyError(
            f"load_obj_pcs: 'obj_pcs' key not found in {pkl_path}. "
            f"Available keys: {list(data.keys())}"
        )

    # Convert to float32 numpy arrays.
    obj_pcs: Dict[str, np.ndarray] = {}
    raw = data["obj_pcs"]
    if isinstance(raw, dict):
        for k, v in raw.items():
            obj_pcs[k] = np.asarray(v, dtype=np.float32)
    else:
        raise ValueError(
            f"load_obj_pcs: expected 'obj_pcs' to be a dict, got {type(raw).__name__}"
        )

    return obj_pcs


def load_arctic_obj_top_masks() -> Dict[str, np.ndarray]:
    """Load ARCTIC per-object point-cloud ``top`` masks from obj.pkl.

    obj.pkl stores ``obj_pc_top[obj_name]`` as int64 with the convention from
    Text2HOI preprocessing (`preprocessing_arctic.py`): ``1 - parts[point_set]``,
    where ``parts`` is 0 for top and 1 for bottom. Thus ``obj_pc_top == 1`` marks
    the articulated top part.

    Returns:
        Dict[obj_name -> (N_pc,) bool array]. True = articulated top vertex.

    Raises:
        FileNotFoundError: If ARCTIC obj.pkl not found.
        KeyError:          If 'obj_pc_top' key missing.
        ValueError:        If top mask is not a dict.
    """
    data_root = _data_root()
    pkl_path = os.path.join(data_root, _OBJ_PKL_RPATH_ARCTIC)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"load_arctic_obj_top_masks: obj.pkl not found at {pkl_path}."
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "obj_pc_top" not in data:
        raise KeyError(
            f"load_arctic_obj_top_masks: 'obj_pc_top' not in {pkl_path}. "
            f"Available keys: {list(data.keys())}"
        )
    raw = data["obj_pc_top"]
    if not isinstance(raw, dict):
        raise ValueError(
            f"load_arctic_obj_top_masks: expected dict, got {type(raw).__name__}"
        )
    return {k: np.asarray(v, dtype=bool) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# ARCTIC per-frame articulation (axis (0,0,-1) about object-local origin)
# ---------------------------------------------------------------------------

def articulate_arctic_obj_pc(
    obj_pc: np.ndarray,            # (N, 3) object-local, meters
    top_mask: np.ndarray,          # (N,) bool, True = articulated top part
    x_obj_angle: np.ndarray,       # (F,) float32 radians
    obj_R: np.ndarray,             # (F, 3, 3)
    obj_t: np.ndarray,             # (F, 3)
) -> np.ndarray:
    """Articulate ARCTIC object point cloud and transform to world frame.

    Mirrors Text2HOI `get_transformed_obj_pc` (lib/utils/proc_output.py) for
    dataset='arctic': per-frame revolute rotation about object-local axis
    (0, 0, -1) applied ONLY to ``top_mask`` points, then global rotation and
    translation to world frame.

    Args:
        obj_pc:      (N, 3) object-local point cloud in meters.
        top_mask:    (N,) bool mask — True for articulated top part.
        x_obj_angle: (F,) revolute joint angle in radians, one per frame.
        obj_R:       (F, 3, 3) world-frame rotation matrix per frame.
        obj_t:       (F, 3)    world-frame translation per frame.

    Returns:
        (F, N, 3) float32 object point cloud in world frame.
    """
    if obj_pc.ndim != 2 or obj_pc.shape[1] != 3:
        raise ValueError(
            f"articulate_arctic_obj_pc: obj_pc must be (N, 3), got {obj_pc.shape}"
        )
    if top_mask.shape != (obj_pc.shape[0],):
        raise ValueError(
            f"articulate_arctic_obj_pc: top_mask must be (N,) with N={obj_pc.shape[0]}, "
            f"got {top_mask.shape}"
        )
    F = int(x_obj_angle.shape[0])
    if obj_R.shape != (F, 3, 3):
        raise ValueError(
            f"articulate_arctic_obj_pc: obj_R must be (F={F}, 3, 3), got {obj_R.shape}"
        )
    if obj_t.shape != (F, 3):
        raise ValueError(
            f"articulate_arctic_obj_pc: obj_t must be (F={F}, 3), got {obj_t.shape}"
        )

    # Per-frame articulation rotation: axis = (0, 0, -1), angle = x_obj_angle[f].
    # Rodrigues closed form with k = [0, 0, -1] yields rotation by angle θ about -z:
    #   R_arti = [[ cos θ,  sin θ, 0],
    #             [-sin θ,  cos θ, 0],
    #             [     0,      0, 1]]
    angle = np.asarray(x_obj_angle, dtype=np.float64)  # (F,)
    c = np.cos(angle)
    s = np.sin(angle)
    zero = np.zeros_like(c)
    one = np.ones_like(c)
    arti_R = np.stack([
        np.stack([c,  s,    zero], axis=-1),
        np.stack([-s, c,    zero], axis=-1),
        np.stack([zero, zero, one], axis=-1),
    ], axis=-2).astype(np.float64)  # (F, 3, 3)

    obj_pc_d = obj_pc.astype(np.float64)  # (N, 3)
    N = obj_pc_d.shape[0]

    # Apply arti_R to the full pc (broadcast over frames): (F, N, 3).
    pc_arti = np.einsum("fij,nj->fni", arti_R, obj_pc_d)  # (F, N, 3)

    # Frame-specific pc: broadcast-copy obj_pc, overwrite TOP points with arti result.
    pc_local = np.broadcast_to(obj_pc_d[None, :, :], (F, N, 3)).copy()  # (F, N, 3)
    pc_local[:, top_mask, :] = pc_arti[:, top_mask, :]

    # Global rotation + translation: world = R @ pc_local.T + t, broadcast over F.
    obj_R_d = obj_R.astype(np.float64)
    obj_t_d = obj_t.astype(np.float64)
    pc_world = (
        np.einsum("fij,fnj->fni", obj_R_d, pc_local) + obj_t_d[:, None, :]
    ).astype(np.float32)  # (F, N, 3)
    return pc_world


# ---------------------------------------------------------------------------
# MANO model cache
# ---------------------------------------------------------------------------

_mano_cache: Dict[Tuple[str, bool], Any] = {}  # key: (hand, flat_hand_mean) -> smplx.MANO

# Dataset -> Text2HOI MANO flat_hand setting (see /tmp/Text2HOI/configs/dataset/*.yaml
# and /tmp/Text2HOI/lib/models/mano.py::build_mano_aa). This governs whether the
# per-joint hand_pose axis-angles are added on top of MANO's learned hand-mean pose
# (flat_hand_mean=False) or on top of a flat zero-pose (flat_hand_mean=True).
# Using the wrong setting shifts every MANO vertex by ~1-6 cm across all finger
# joints, which makes the fingertip-contact check spuriously fail even on GT frames.
_DATASET_FLAT_HAND_MEAN: Dict[str, bool] = {
    "grab":   True,    # /tmp/Text2HOI/configs/dataset/grab.yaml -> flat_hand: True
    "arctic": False,   # /tmp/Text2HOI/configs/dataset/arctic.yaml -> flat_hand: False
}


def _get_mano(hand: str, flat_hand_mean: bool = True) -> Any:
    """Return a cached smplx.MANO model for the given hand.

    Args:
        hand: 'left' or 'right'.
        flat_hand_mean: Whether to apply MANO's flat-hand mean subtraction
            (True) or use the learned hand-mean pose (False). Must match the
            value used at data-generation time; mismatches manifest as 1-6 cm
            systematic finger offsets that silently break contact checks.
            Dataset defaults are exposed via ``_DATASET_FLAT_HAND_MEAN``.

    Returns:
        smplx.MANO instance (CPU, eval mode).

    Raises:
        ValueError: If hand is not 'left' or 'right'.
        RuntimeError: If MANO model path is not found.
    """
    if hand not in ("left", "right"):
        raise ValueError(f"_get_mano: hand must be 'left' or 'right', got {hand!r}")

    cache_key = (hand, bool(flat_hand_mean))
    if cache_key in _mano_cache:
        return _mano_cache[cache_key]

    mano_path = os.environ.get("MANO_MODEL_PATH", _MANO_MODEL_PATH)
    if not os.path.isdir(mano_path):
        # Try resolving relative to data root.
        data_root = _data_root()
        mano_path = os.path.join(
            data_root,
            "handobject/raw_registered/mano_unzip/mano_v1_2/models/",
        )
        if not os.path.isdir(mano_path):
            raise RuntimeError(
                f"_get_mano: MANO model directory not found. Tried: "
                f"{_MANO_MODEL_PATH!r} and {mano_path!r}. "
                "Set MANO_MODEL_PATH env var or verify data layout."
            )

    is_rhand = (hand == "right")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smplx.MANO(
            model_path=mano_path,
            is_rhand=is_rhand,
            use_pca=False,
            flat_hand_mean=bool(flat_hand_mean),
        )
    model.eval()
    _mano_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# MANO decode: world-frame verts + vertex normals (BATCHED over frames)
# ---------------------------------------------------------------------------

def _decode_mano_world(
    hand: str,
    rot_aa: np.ndarray,          # (16, 3) axis-angle: [global_orient(1), hand_pose(15)]
    wrist_world: np.ndarray,     # (3,)    world-frame wrist position
    flat_hand_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one frame of MANO to world-frame vertices + vertex normals.

    Single-frame convenience wrapper around _decode_mano_world_batch.

    Args:
        hand:        'left' or 'right'.
        rot_aa:      (16, 3) axis-angle rotation array.
        wrist_world: (3,) world-frame wrist position.
        flat_hand_mean: MANO build-time flag. Must match the value used at
            data-generation time (GRAB=True, ARCTIC=False).

    Returns:
        verts_world:    (778, 3) float32 — hand vertices in world frame.
        vertex_normals: (778, 3) float32 — per-vertex outward normals.
    """
    verts_batch, normals_batch = _decode_mano_world_batch(
        hand,
        rot_aa[np.newaxis],       # (1, 16, 3)
        wrist_world[np.newaxis],  # (1, 3)
        flat_hand_mean=flat_hand_mean,
    )
    return verts_batch[0], normals_batch[0]


def _decode_mano_world_batch(
    hand: str,
    rot_aa: np.ndarray,    # (F, 16, 3)
    wrist_world: np.ndarray,  # (F, 3)
    flat_hand_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode F frames of MANO in a single batched forward pass.

    Procedure per batch:
      1. Batch MANO forward with global_orient=(F,3), hand_pose=(F,45),
         betas=zeros(F,10), transl=zeros(F,3).
      2. verts_canonical: (F, 778, 3), joint0_canonical: (F, 3).
      3. World frame per frame f:
           verts_world[f] = verts_canonical[f] - joint0_canonical[f] + wrist_world[f]
      4. Compute vertex normals for each frame via trimesh (vectorized over frames).

    Args:
        hand:        'left' or 'right'.
        rot_aa:      (F, 16, 3) axis-angle rotations.
        wrist_world: (F, 3) world-frame wrist positions.

    Returns:
        verts_world:    (F, 778, 3) float32 — hand vertices in world frame.
        vertex_normals: (F, 778, 3) float32 — per-vertex outward normals.

    Raises:
        RuntimeError: If MANO model fails to load.
        ValueError: If rot_aa shape is wrong.
    """
    F = rot_aa.shape[0]
    if rot_aa.shape != (F, 16, 3):
        raise ValueError(
            f"_decode_mano_world_batch: rot_aa must be (F, 16, 3), got {rot_aa.shape}"
        )

    model = _get_mano(hand, flat_hand_mean=flat_hand_mean)

    global_orient = torch.tensor(rot_aa[:, 0, :], dtype=torch.float32)    # (F, 3)
    hand_pose = torch.tensor(
        rot_aa[:, 1:16, :].reshape(F, 45), dtype=torch.float32
    )  # (F, 45)
    betas = torch.zeros(F, 10, dtype=torch.float32)
    transl = torch.zeros(F, 3, dtype=torch.float32)

    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            transl=transl,
        )

    # verts_canonical: (F, 778, 3), joint0_canonical: (F, 3)
    verts_canonical = out.vertices.numpy()    # (F, 778, 3)
    joint0_canonical = out.joints[:, 0, :].numpy()  # (F, 3)

    # Translate: verts_world[f] = verts_canonical[f] - joint0_canonical[f] + wrist_world[f]
    # Shapes: (F, 778, 3) - (F, 1, 3) + (F, 1, 3)
    verts_world = (
        verts_canonical
        - joint0_canonical[:, np.newaxis, :]
        + wrist_world[:, np.newaxis, :]
    ).astype(np.float32)  # (F, 778, 3)

    # Compute vertex normals for all frames at once using vectorized numpy.
    # This avoids F trimesh construction calls which dominate runtime.
    faces = model.faces.astype(np.int32)  # (1538, 3)
    vertex_normals = _compute_vertex_normals_batch(verts_world, faces)  # (F, 778, 3)

    return verts_world, vertex_normals


def _compute_vertex_normals_batch(
    verts: np.ndarray,   # (F, V, 3)
    faces: np.ndarray,   # (Nf, 3) int32
) -> np.ndarray:
    """Compute per-vertex normals for F frames in vectorized numpy.

    Algorithm:
      1. Compute per-face normals as cross product of face edges.
      2. Accumulate face normals into vertex normal sums (area-weighted by cross product magnitude).
      3. L2-normalize each vertex normal.

    Args:
        verts: (F, V, 3) float32 vertex positions.
        faces: (Nf, 3) int32 face indices.

    Returns:
        vertex_normals: (F, V, 3) float32 — outward unit normals.
    """
    F, V = verts.shape[:2]
    Nf = faces.shape[0]

    # Face vertex positions: (F, Nf, 3) for each of the 3 corners.
    v0 = verts[:, faces[:, 0], :]   # (F, Nf, 3)
    v1 = verts[:, faces[:, 1], :]   # (F, Nf, 3)
    v2 = verts[:, faces[:, 2], :]   # (F, Nf, 3)

    # Face cross products (not normalized — includes area weighting).
    e1 = v1 - v0   # (F, Nf, 3)
    e2 = v2 - v0   # (F, Nf, 3)
    face_normals = np.cross(e1, e2)   # (F, Nf, 3)

    # Accumulate face normals into per-vertex sums using torch scatter_add for speed.
    # This is the fastest approach for batched sparse accumulation.
    face_normals_t = torch.from_numpy(face_normals.astype(np.float32))   # (F, Nf, 3)
    faces_t = torch.from_numpy(faces.astype(np.int64))   # (Nf, 3) int64

    vertex_normals_t = torch.zeros(F, V, 3, dtype=torch.float32)
    for corner in range(3):
        # idx: (F, Nf, 1) -> expand to (F, Nf, 3)
        idx = faces_t[:, corner].view(1, Nf, 1).expand(F, Nf, 3)  # (F, Nf, 3)
        vertex_normals_t.scatter_add_(1, idx, face_normals_t)

    vertex_normals = vertex_normals_t.numpy().astype(np.float32)  # (F, V, 3)

    # L2-normalize.
    norms = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)  # (F, V, 1)
    norms = np.maximum(norms, 1e-8)   # avoid divide by zero
    vertex_normals = vertex_normals / norms

    return vertex_normals


# ---------------------------------------------------------------------------
# Text2HOI KNN + interior primitives (numpy, no pytorch3d)
# ---------------------------------------------------------------------------

def _get_nn(src_xyz: np.ndarray, trg_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each src point, find nearest trg point.

    Replicates Text2HOI `get_NN(src, trg, k=1)` but in numpy.
    Uses scipy cKDTree for O(N log M) complexity.

    Args:
        src_xyz: (N, 3) source points (obj pc in Text2HOI usage).
        trg_xyz: (M, 3) target points (hand verts in Text2HOI usage).

    Returns:
        nn_dists: (N,) float32 — squared distances to nearest trg point.
        nn_idx:   (N,) int32   — index into trg_xyz for each src point.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 — lazy import for eval only
    tree = cKDTree(trg_xyz.astype(np.float64))
    # query returns (distance, index); distance here is Euclidean (not squared).
    dist, idx = tree.query(src_xyz.astype(np.float64), k=1)
    # Text2HOI `get_NN` returns SQUARED distances (knn_points returns dists^2).
    nn_dists = (dist ** 2).astype(np.float32)
    nn_idx = idx.astype(np.int32)
    return nn_dists, nn_idx


def _get_interior(
    hand_normals: np.ndarray,   # (778, 3)
    hand_verts: np.ndarray,     # (778, 3)
    obj_pc: np.ndarray,         # (N, 3)
    nn_idx: np.ndarray,         # (N,)  index into hand_verts
) -> np.ndarray:
    """Determine which obj points are inside the hand mesh.

    Replicates Text2HOI `get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx)`.

    For each obj point p_i:
      NN in hand verts: h = hand_verts[nn_idx[i]]
      NN_vector = h - p_i                    (vector from obj point to NN hand vert)
      NN_normal = hand_normals[nn_idx[i]]
      interior[i] = dot(NN_vector, NN_normal) > 0

    Args:
        hand_normals: (778, 3) vertex normals (outward).
        hand_verts:   (778, 3) hand vertices in world frame.
        obj_pc:       (N, 3)   object point cloud in world frame.
        nn_idx:       (N,)     index of nearest hand vert for each obj point.

    Returns:
        interior: (N,) bool — True if obj point is inside the hand.
    """
    # Gather nearest hand vert and normal for each obj point.
    nn_hand_verts = hand_verts[nn_idx]      # (N, 3)
    nn_hand_normals = hand_normals[nn_idx]  # (N, 3)

    # Vector from obj point to its nearest hand vert.
    nn_vector = nn_hand_verts - obj_pc      # (N, 3)

    # Dot product: positive means obj point is on the interior side.
    dot = np.sum(nn_vector * nn_hand_normals, axis=-1)  # (N,)
    return dot > 0


# ---------------------------------------------------------------------------
# Per-episode physical realism (public API)
# ---------------------------------------------------------------------------

def compute_physical_realism(
    sample: CanonicalSample,
    obj_meshes: Optional[Dict[str, Any]] = None,  # kept for API compat; not used
    contact_thresh_m: float = _DEFAULT_CONTACT_THRESH_M,
    penetration_thresh_m: Optional[float] = None,
    fingertip_idx: Optional[List[int]] = None,     # kept for API compat; ignored
    max_frames: Optional[int] = None,
    obj_pcs: Optional[Dict[str, np.ndarray]] = None,
    obj_top_masks: Optional[Dict[str, np.ndarray]] = None,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    """Compute physical realism metrics for one episode.

    Uses 778-vertex MANO mesh (decoded from canonical rot arrays) and
    per-object point clouds (from obj.pkl) following Text2HOI exactly.

    Physical realism (per frame):
      - contact: min fingertip-vert to nearest obj-pc distance <= contact_thresh_m
      - no_penetration: max interior obj-pc penetration depth < penetration_thresh_m
      - realistic = contact AND no_penetration

    Contact is OR across hands; no_penetration is AND (both hands must be clean).
    Episode realism = fraction of valid frames that are realistic.

    Args:
        sample:               CanonicalSample with hand rot + joints and object pose.
        obj_meshes:           Ignored (kept for API compatibility with eval_imos.py).
        contact_thresh_m:     Fingertip-to-nearest-obj-point distance threshold (m).
        penetration_thresh_m: Interior penetration depth threshold (m). When
                              ``None`` (default), resolves from
                              ``_DATASET_PENETRATION_THRESH_M[dataset]`` if
                              ``dataset`` is known, otherwise falls back to the
                              GRAB-calibrated legacy default
                              ``_DEFAULT_PENETRATION_THRESH_M`` (0.010 m).
                              Explicit caller-passed values are sacred — the
                              dataset default applies only when the caller did
                              not specify one.
        fingertip_idx:        Ignored (fingertip vertex indices are fixed per Text2HOI).
        max_frames:           Cap on valid frames to evaluate (uniform subsample).
        obj_pcs:              Dict[obj_name -> (N_pc, 3) float32] in object-local frame.
                              If None, caller must set obj_pcs or this will raise.
        obj_top_masks:        Dict[obj_name -> (N_pc,) bool]. For ARCTIC, True marks
                              the articulated top part. When this is provided AND
                              `sample.obj_articulation_angle` is not None, the object
                              pc is articulated per-frame before global rot+trans
                              (matches Text2HOI `get_transformed_obj_pc`). When None
                              (e.g. GRAB), the rigid vectorized path is used.
        dataset:              Optional dataset name ('grab' or 'arctic'). Selects
                              MANO ``flat_hand_mean`` to match Text2HOI preprocessing:
                              GRAB=True, ARCTIC=False. When ``None``, defaults to
                              True (legacy behavior; correct for GRAB, WRONG for ARCTIC).
                              Passing the wrong value produces a ~1-6 cm systematic
                              offset across all finger joints and silently nukes
                              contact-rate (ARCTIC GT drops from ~0.90 to ~0.09 when
                              decoded with flat_hand_mean=True).

    Returns:
        Dict with:
          'realism'           -- fraction of valid frames that are realistic.
          'contact_rate'      -- fraction of valid frames with contact.
          'penetration_rate'  -- fraction of valid frames with penetration.
          'n_valid_frames'    -- int count of evaluated valid frames.

    Raises:
        KeyError: If sample.obj_name not in obj_pcs.
        ValueError: If hand rot arrays have wrong shape.
        RuntimeError: If MANO model fails to load or decode raises.
    """
    if obj_pcs is None:
        raise RuntimeError(
            "compute_physical_realism: obj_pcs must be provided (not None). "
            "Call load_obj_pcs(dataset) and pass the result. "
            "The old trimesh-SDF path (obj_meshes) is no longer used."
        )

    if sample.obj_name not in obj_pcs:
        raise KeyError(
            f"compute_physical_realism: obj_name={sample.obj_name!r} not in obj_pcs. "
            f"Available: {sorted(obj_pcs.keys())}"
        )

    obj_pc_local = obj_pcs[sample.obj_name].astype(np.float32)  # (N, 3) object-local

    left_rot = sample.left_hand_rot     # (T, 16, 3)
    right_rot = sample.right_hand_rot   # (T, 16, 3)
    left_joints = sample.left_hand_joints   # (T, 21, 3) — joint 0 = wrist
    right_joints = sample.right_hand_joints  # (T, 21, 3)
    obj_R = sample.obj_rotation         # (T, 3, 3)
    obj_t = sample.obj_translation      # (T, 3)
    valid_mask = sample.valid_mask      # (T,)

    T = left_rot.shape[0]

    if left_rot.shape != (T, 16, 3):
        raise ValueError(
            f"compute_physical_realism: left_hand_rot shape {left_rot.shape}, "
            "expected (T, 16, 3)"
        )
    if right_rot.shape != (T, 16, 3):
        raise ValueError(
            f"compute_physical_realism: right_hand_rot shape {right_rot.shape}, "
            "expected (T, 16, 3)"
        )

    valid_idx = np.where(valid_mask)[0]
    n_valid = int(len(valid_idx))

    if n_valid == 0:
        return {
            "realism": float("nan"),
            "contact_rate": float("nan"),
            "penetration_rate": float("nan"),
            "n_valid_frames": 0,
        }

    # Subsample valid frames when max_frames is set.
    if max_frames is not None and n_valid > max_frames:
        sel = np.linspace(0, n_valid - 1, num=max_frames).round().astype(np.int64)
        valid_idx = valid_idx[sel]
        n_valid = int(len(valid_idx))

    tip_idx = MANO_FINGERTIP_VERT_IDX  # [745, 317, 445, 556, 673]
    tip_idx_arr = np.array(tip_idx, dtype=np.int32)

    # Gather valid-frame arrays. (F, ...)
    F = n_valid
    l_rot_f = left_rot[valid_idx]        # (F, 16, 3)
    r_rot_f = right_rot[valid_idx]       # (F, 16, 3)
    l_wrist_f = left_joints[valid_idx, 0, :]   # (F, 3)
    r_wrist_f = right_joints[valid_idx, 0, :]  # (F, 3)
    obj_R_f = obj_R[valid_idx]           # (F, 3, 3)
    obj_t_f = obj_t[valid_idx]           # (F, 3)

    # Transform obj pc to world frame for all F frames at once.
    # ARCTIC: apply per-frame revolute articulation to TOP points before global
    #         rotation/translation (matches Text2HOI `get_transformed_obj_pc`).
    # Other datasets (e.g. GRAB): rigid — vectorized global transform only.
    has_arti = (
        obj_top_masks is not None
        and sample.obj_name in obj_top_masks
        and sample.obj_articulation_angle is not None
    )
    if has_arti:
        top_mask = np.asarray(obj_top_masks[sample.obj_name], dtype=bool)
        if top_mask.shape != (obj_pc_local.shape[0],):
            raise ValueError(
                f"compute_physical_realism: obj_top_masks[{sample.obj_name!r}] shape "
                f"{top_mask.shape} != obj_pc shape ({obj_pc_local.shape[0]},)"
            )
        angle_f = np.asarray(
            sample.obj_articulation_angle[valid_idx], dtype=np.float32
        )  # (F,)
        obj_pc_world_f = articulate_arctic_obj_pc(
            obj_pc=obj_pc_local,
            top_mask=top_mask,
            x_obj_angle=angle_f,
            obj_R=obj_R_f,
            obj_t=obj_t_f,
        )  # (F, N, 3) float32
    else:
        # Rigid global transform: obj_pc_world[f] = obj_R_f[f] @ obj_pc_local.T + obj_t_f[f]
        obj_pc_local_d = obj_pc_local.astype(np.float64)  # (N, 3)
        obj_R_d = obj_R_f.astype(np.float64)               # (F, 3, 3)
        obj_t_d = obj_t_f.astype(np.float64)               # (F, 3)
        obj_pc_world_f = (
            np.einsum("ni,fji->fnj", obj_pc_local_d, obj_R_d)
            + obj_t_d[:, np.newaxis, :]
        ).astype(np.float32)  # (F, N, 3)

    # Resolve flat_hand_mean from dataset (GRAB=True, ARCTIC=False).
    # Legacy callers (dataset=None) keep the historical True default; this is
    # correct for GRAB but silently wrong for ARCTIC -- see dataclass docstring.
    if dataset is None:
        flat_hand_mean = True
    else:
        if dataset not in _DATASET_FLAT_HAND_MEAN:
            raise ValueError(
                f"compute_physical_realism: unknown dataset={dataset!r}; "
                f"expected one of {sorted(_DATASET_FLAT_HAND_MEAN.keys())}"
            )
        flat_hand_mean = _DATASET_FLAT_HAND_MEAN[dataset]

    # Resolve penetration_thresh_m. Explicit caller-passed values are sacred.
    # Only when penetration_thresh_m is None do we pick from the dataset table
    # (falling back to the GRAB-calibrated legacy default when dataset is None).
    # Rationale in constants section: ARCTIC's 10mm threshold is too tight due
    # to missing beta plumbing — see _DATASET_PENETRATION_THRESH_M and 
    # Unknown-dataset handling mirrors flat_hand_mean above: if the caller
    # named a dataset, it MUST exist in the threshold table too — silent
    # fallback here would hide a missing-entry bug when a new dataset is added
    # to _DATASET_FLAT_HAND_MEAN but not to _DATASET_PENETRATION_THRESH_M.
    if penetration_thresh_m is None:
        if dataset is None:
            penetration_thresh_m = _DEFAULT_PENETRATION_THRESH_M
        else:
            if dataset not in _DATASET_PENETRATION_THRESH_M:
                raise ValueError(
                    f"compute_physical_realism: unknown dataset={dataset!r} "
                    f"for penetration threshold; expected one of "
                    f"{sorted(_DATASET_PENETRATION_THRESH_M.keys())}. "
                    f"Add the entry to _DATASET_PENETRATION_THRESH_M or pass "
                    f"penetration_thresh_m explicitly."
                )
            penetration_thresh_m = _DATASET_PENETRATION_THRESH_M[dataset]

    # gate MANO decode on is_lhand / is_rhand. Raw Text2HOI does not
    # zero idle-hand streams (body-rest finger positions remain in j_lhand even
    # when is_lhand=0), so an unconditional decode feeds idle-hand vertices
    # into contact/penetration scoring. Gating here matches Text2HOI's own
    # "zero the missing-hand stream" convention — idle hand contributes no
    # contact and no penetration. Backward compat: CanonicalSample defaults
    # to (True, True) when legacy files lack the field (load_canonical_npz).
    use_l = bool(sample.is_lhand)
    use_r = bool(sample.is_rhand)
    if not (use_l or use_r):
        raise ValueError(
            "compute_physical_realism: sample has is_lhand=False and "
            "is_rhand=False — at least one hand must be active."
        )

    # Batch MANO decode for all valid frames (1 forward pass per active hand).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_l:
            l_verts_f, l_norms_f = _decode_mano_world_batch(
                "left", l_rot_f, l_wrist_f, flat_hand_mean=flat_hand_mean,
            )
        else:
            l_verts_f = l_norms_f = None
        if use_r:
            r_verts_f, r_norms_f = _decode_mano_world_batch(
                "right", r_rot_f, r_wrist_f, flat_hand_mean=flat_hand_mean,
            )
        else:
            r_verts_f = r_norms_f = None
    # Active-hand arrays: (F, 778, 3) each; inactive-hand arrays are None.

    from scipy.spatial import cKDTree  # noqa: PLC0415

    is_contact = np.zeros(F, dtype=bool)
    has_penetration = np.zeros(F, dtype=bool)

    for frame_i in range(F):
        obj_pc = obj_pc_world_f[frame_i]   # (N, 3)
        obj_tree = cKDTree(obj_pc.astype(np.float64))

        # --- Left hand ---
        if use_l:
            l_verts = l_verts_f[frame_i]       # (778, 3)
            l_norms = l_norms_f[frame_i]       # (778, 3)

            l_tip_verts = l_verts[tip_idx_arr]  # (5, 3)
            l_tip_dists, _ = obj_tree.query(l_tip_verts.astype(np.float64), k=1)
            l_contact = bool(np.any(l_tip_dists <= contact_thresh_m))

            l_nn_dists_sq, l_nn_idx = _get_nn(obj_pc, l_verts)
            l_interior = _get_interior(l_norms, l_verts, obj_pc, l_nn_idx)
            if l_interior.any():
                l_max_penet = float(np.sqrt(l_nn_dists_sq[l_interior]).max())
            else:
                l_max_penet = 0.0
        else:
            # Idle hand: no contact, no penetration contribution (Text2HOI convention).
            l_contact = False
            l_max_penet = 0.0

        # --- Right hand ---
        if use_r:
            r_verts = r_verts_f[frame_i]       # (778, 3)
            r_norms = r_norms_f[frame_i]       # (778, 3)

            r_tip_verts = r_verts[tip_idx_arr]  # (5, 3)
            r_tip_dists, _ = obj_tree.query(r_tip_verts.astype(np.float64), k=1)
            r_contact = bool(np.any(r_tip_dists <= contact_thresh_m))

            r_nn_dists_sq, r_nn_idx = _get_nn(obj_pc, r_verts)
            r_interior = _get_interior(r_norms, r_verts, obj_pc, r_nn_idx)
            if r_interior.any():
                r_max_penet = float(np.sqrt(r_nn_dists_sq[r_interior]).max())
            else:
                r_max_penet = 0.0
        else:
            r_contact = False
            r_max_penet = 0.0

        # Frame result: contact is OR across active hands; no-penetration is
        # AND (both active hands must be clean — idle hand contributes 0).
        is_contact[frame_i] = l_contact or r_contact
        has_penetration[frame_i] = (
            (l_max_penet >= penetration_thresh_m) or (r_max_penet >= penetration_thresh_m)
        )

    is_realistic = is_contact & ~has_penetration

    return {
        "realism": float(is_realistic.mean()),
        "contact_rate": float(is_contact.mean()),
        "penetration_rate": float(has_penetration.mean()),
        "n_valid_frames": n_valid,
    }


# ---------------------------------------------------------------------------
# Per-episode physical realism — raw per-frame output (for threshold sweep)
# ---------------------------------------------------------------------------

def compute_physical_realism_raw(
    sample: CanonicalSample,
    obj_pcs: Dict[str, np.ndarray],
    dataset: Optional[str] = None,
    obj_top_masks: Optional[Dict[str, np.ndarray]] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute per-frame raw physical realism signals for one episode.

    Mirrors compute_physical_realism exactly (same MANO decode, same obj
    transform, same KNN + interior test) but instead of applying thresholds
    and returning aggregate scalars, returns per-frame arrays so that any
    (contact_thresh_m, penetration_thresh_m) pair can be swept without
    re-running the expensive MANO + KNN step.

    Args:
        sample:        CanonicalSample with hand rot + joints and object pose.
        obj_pcs:       Dict[obj_name -> (N_pc, 3) float32] in object-local frame.
        dataset:       'grab' or 'arctic'. Selects MANO flat_hand_mean and
                       articulation path. When None, defaults to True (GRAB legacy).
        obj_top_masks: Dict[obj_name -> (N_pc,) bool]. For ARCTIC articulation.
                       None disables articulation (rigid path).
        max_frames:    Cap on valid frames (uniform subsample), same as
                       compute_physical_realism.

    Returns:
        Dict with:
          'n_valid_frames': int            -- number of evaluated frames F.
          'l_tip_min_dist': (F,) float32   -- per-frame min left-fingertip distance to obj (m).
                                             When is_lhand=False, filled with +inf so no
                                             contact-threshold sweep ever registers contact.
          'r_tip_min_dist': (F,) float32   -- per-frame min right-fingertip distance to obj (m).
                                             When is_rhand=False, filled with +inf (idle).
          'l_max_penet':    (F,) float32   -- per-frame left max penetration depth (m); 0.0
                                             when no interior points OR when is_lhand=False.
          'r_max_penet':    (F,) float32   -- per-frame right max penetration depth (m); 0.0
                                             when no interior points OR when is_rhand=False.

    Raises:
        RuntimeError: If obj_pcs is None.
        KeyError:     If sample.obj_name not in obj_pcs.
        ValueError:   If hand rot arrays have wrong shape, dataset unknown, or
                      both is_lhand and is_rhand are False.
        RuntimeError: If MANO model fails to load.
    """
    if obj_pcs is None:
        raise RuntimeError(
            "compute_physical_realism_raw: obj_pcs must be provided (not None). "
            "Call load_obj_pcs(dataset) and pass the result."
        )

    if sample.obj_name not in obj_pcs:
        raise KeyError(
            f"compute_physical_realism_raw: obj_name={sample.obj_name!r} not in obj_pcs. "
            f"Available: {sorted(obj_pcs.keys())}"
        )

    obj_pc_local = obj_pcs[sample.obj_name].astype(np.float32)  # (N, 3) object-local

    left_rot = sample.left_hand_rot     # (T, 16, 3)
    right_rot = sample.right_hand_rot   # (T, 16, 3)
    left_joints = sample.left_hand_joints   # (T, 21, 3)
    right_joints = sample.right_hand_joints  # (T, 21, 3)
    obj_R = sample.obj_rotation         # (T, 3, 3)
    obj_t = sample.obj_translation      # (T, 3)
    valid_mask = sample.valid_mask      # (T,)

    T = left_rot.shape[0]

    if left_rot.shape != (T, 16, 3):
        raise ValueError(
            f"compute_physical_realism_raw: left_hand_rot shape {left_rot.shape}, "
            "expected (T, 16, 3)"
        )
    if right_rot.shape != (T, 16, 3):
        raise ValueError(
            f"compute_physical_realism_raw: right_hand_rot shape {right_rot.shape}, "
            "expected (T, 16, 3)"
        )

    valid_idx = np.where(valid_mask)[0]
    n_valid = int(len(valid_idx))

    if n_valid == 0:
        return {
            "n_valid_frames": 0,
            "l_tip_min_dist": np.zeros(0, dtype=np.float32),
            "r_tip_min_dist": np.zeros(0, dtype=np.float32),
            "l_max_penet": np.zeros(0, dtype=np.float32),
            "r_max_penet": np.zeros(0, dtype=np.float32),
        }

    # Subsample valid frames when max_frames is set — identical to compute_physical_realism.
    if max_frames is not None and n_valid > max_frames:
        sel = np.linspace(0, n_valid - 1, num=max_frames).round().astype(np.int64)
        valid_idx = valid_idx[sel]
        n_valid = int(len(valid_idx))

    tip_idx_arr = np.array(MANO_FINGERTIP_VERT_IDX, dtype=np.int32)

    # Gather valid-frame arrays.
    F = n_valid
    l_rot_f = left_rot[valid_idx]        # (F, 16, 3)
    r_rot_f = right_rot[valid_idx]       # (F, 16, 3)
    l_wrist_f = left_joints[valid_idx, 0, :]   # (F, 3)
    r_wrist_f = right_joints[valid_idx, 0, :]  # (F, 3)
    obj_R_f = obj_R[valid_idx]           # (F, 3, 3)
    obj_t_f = obj_t[valid_idx]           # (F, 3)

    # Object pc world transform — same logic as compute_physical_realism.
    has_arti = (
        obj_top_masks is not None
        and sample.obj_name in obj_top_masks
        and sample.obj_articulation_angle is not None
    )
    if has_arti:
        top_mask = np.asarray(obj_top_masks[sample.obj_name], dtype=bool)
        if top_mask.shape != (obj_pc_local.shape[0],):
            raise ValueError(
                f"compute_physical_realism_raw: obj_top_masks[{sample.obj_name!r}] shape "
                f"{top_mask.shape} != obj_pc shape ({obj_pc_local.shape[0]},)"
            )
        angle_f = np.asarray(
            sample.obj_articulation_angle[valid_idx], dtype=np.float32
        )  # (F,)
        obj_pc_world_f = articulate_arctic_obj_pc(
            obj_pc=obj_pc_local,
            top_mask=top_mask,
            x_obj_angle=angle_f,
            obj_R=obj_R_f,
            obj_t=obj_t_f,
        )  # (F, N, 3) float32
    else:
        obj_pc_local_d = obj_pc_local.astype(np.float64)
        obj_R_d = obj_R_f.astype(np.float64)
        obj_t_d = obj_t_f.astype(np.float64)
        obj_pc_world_f = (
            np.einsum("ni,fji->fnj", obj_pc_local_d, obj_R_d)
            + obj_t_d[:, np.newaxis, :]
        ).astype(np.float32)  # (F, N, 3)

    # Resolve flat_hand_mean — same logic as compute_physical_realism.
    if dataset is None:
        flat_hand_mean = True
    else:
        if dataset not in _DATASET_FLAT_HAND_MEAN:
            raise ValueError(
                f"compute_physical_realism_raw: unknown dataset={dataset!r}; "
                f"expected one of {sorted(_DATASET_FLAT_HAND_MEAN.keys())}"
            )
        flat_hand_mean = _DATASET_FLAT_HAND_MEAN[dataset]

    # propagation: gate MANO decode on is_lhand / is_rhand, mirroring
    # compute_physical_realism. An idle hand must not contribute contact
    # (fill tip_min_dist with +inf so it never satisfies tip <= c_thresh) and
    # must not contribute penetration (fill max_penet with 0.0 so it never
    # satisfies penet >= p_thresh). This keeps raw-cache threshold-sweep
    # semantics identical to the aggregate path.
    use_l = bool(sample.is_lhand)
    use_r = bool(sample.is_rhand)
    if not (use_l or use_r):
        raise ValueError(
            "compute_physical_realism_raw: sample has is_lhand=False and "
            "is_rhand=False — at least one hand must be active."
        )

    # Batch MANO decode — only for active hands.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_l:
            l_verts_f, l_norms_f = _decode_mano_world_batch(
                "left", l_rot_f, l_wrist_f, flat_hand_mean=flat_hand_mean,
            )
        else:
            l_verts_f = l_norms_f = None
        if use_r:
            r_verts_f, r_norms_f = _decode_mano_world_batch(
                "right", r_rot_f, r_wrist_f, flat_hand_mean=flat_hand_mean,
            )
        else:
            r_verts_f = r_norms_f = None
    # Active-hand arrays: (F, 778, 3) each; inactive-hand arrays are None.

    from scipy.spatial import cKDTree  # noqa: PLC0415

    # Sentinel init: idle hand → tip_min_dist=+inf, max_penet=0.0. Active
    # hand → overwritten per-frame below.
    l_tip_min_dist = np.full(F, np.inf, dtype=np.float32) if not use_l else np.zeros(F, dtype=np.float32)
    r_tip_min_dist = np.full(F, np.inf, dtype=np.float32) if not use_r else np.zeros(F, dtype=np.float32)
    l_max_penet_arr = np.zeros(F, dtype=np.float32)
    r_max_penet_arr = np.zeros(F, dtype=np.float32)

    for frame_i in range(F):
        obj_pc = obj_pc_world_f[frame_i]   # (N, 3)
        obj_tree = cKDTree(obj_pc.astype(np.float64))

        # --- Left hand ---
        if use_l:
            l_verts = l_verts_f[frame_i]       # (778, 3)
            l_norms = l_norms_f[frame_i]       # (778, 3)

            l_tip_verts = l_verts[tip_idx_arr]  # (5, 3)
            l_tip_dists, _ = obj_tree.query(l_tip_verts.astype(np.float64), k=1)
            l_tip_min_dist[frame_i] = float(l_tip_dists.min())

            l_nn_dists_sq, l_nn_idx = _get_nn(obj_pc, l_verts)
            l_interior = _get_interior(l_norms, l_verts, obj_pc, l_nn_idx)
            if l_interior.any():
                l_max_penet_arr[frame_i] = float(np.sqrt(l_nn_dists_sq[l_interior]).max())
            # else already 0.0

        # --- Right hand ---
        if use_r:
            r_verts = r_verts_f[frame_i]       # (778, 3)
            r_norms = r_norms_f[frame_i]       # (778, 3)

            r_tip_verts = r_verts[tip_idx_arr]  # (5, 3)
            r_tip_dists, _ = obj_tree.query(r_tip_verts.astype(np.float64), k=1)
            r_tip_min_dist[frame_i] = float(r_tip_dists.min())

            r_nn_dists_sq, r_nn_idx = _get_nn(obj_pc, r_verts)
            r_interior = _get_interior(r_norms, r_verts, obj_pc, r_nn_idx)
            if r_interior.any():
                r_max_penet_arr[frame_i] = float(np.sqrt(r_nn_dists_sq[r_interior]).max())
            # else already 0.0

    return {
        "n_valid_frames": n_valid,
        "l_tip_min_dist": l_tip_min_dist,
        "r_tip_min_dist": r_tip_min_dist,
        "l_max_penet": l_max_penet_arr,
        "r_max_penet": r_max_penet_arr,
    }
