"""GT episode loader — reads Text2HOI raw NPZ and yields CanonicalSample per episode.

Uses bin_map_{grab,arctic}.json as single source of truth for bin indices.
All array slices are derived from the bin_map at runtime — no hardcoded index
literals in Python. This is the canonical adapter that downstream /code should follow.

21-joint world-frame positions are taken directly from j_lhand/j_rhand (already
MANO-decoded at Text2HOI preprocessing time — see /tmp/Text2HOI/lib/utils/proc.py:83-87).
rot6d -> axis-angle conversion for left_hand_rot/right_hand_rot (Zhou et al. continuity).
rot6d -> (3,3) rotation matrix for obj_rotation.

Hand-presence semantics:
    `left_hand_joints` and `left_hand_rot` are meaningful only if the episode had a
    left hand (Text2HOI's `is_lhand` flag). Same for right. Canonical `valid_mask`
    tracks only temporal padding, not hand presence. Downstream consumers must
    consult a separate source for hand presence if needed — this loader preserves
    raw Text2HOI data as-is and does NOT zero out inactive hands.

Text-key composition (mirrored from Text2HOI):
    f"{action_name} {obj_name} with {left|right|both} hand[s].".capitalize()
    Verified against:
      - /tmp/Text2HOI/lib/utils/proc_grab.py:8-14 (process_text for GRAB)
      - /tmp/Text2HOI/lib/utils/proc_arctic.py:20-26 (process_text for ARCTIC)
    Quote: `text = f"{action_name} {obj_name} with both hands."` ... `text_key = text.capitalize()`
    For ARCTIC the raw key uses `object_name` (not a secondary processed name).
    We pick the FIRST paraphrase (deterministic) — Text2HOI runtime picks randomly.

allow_pickle note:
    The raw Text2HOI npz contains ragged object arrays (lcf_idx, lcov_idx, etc.)
    which require `allow_pickle=True` at load. Canonical samples never use ragged
    arrays — `load_canonical_npz` uses `allow_pickle=False` for safety.

Note on MANO joint-0 reference (`lhand_org`, `rhand_org`):
    - GRAB stores joint-0 PRE-translation (before applying MANO `transl`).
    - ARCTIC stores joint-0 POST-translation (after `transl`).
    CanonicalSample uses `j_lhand`/`j_rhand` directly for joint positions (always
    world-frame meters), so downstream consumers should NOT read `lhand_org`
    without first accounting for this per-dataset convention.
    Source: audit_report.md §B.4 + §Surprises item 6.
"""

import json
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from umm.eval.handobject.canonical import CanonicalSample


# ---------------------------------------------------------------------------
# bin_map loader
# ---------------------------------------------------------------------------

def load_bin_map(path: str) -> Dict[str, Any]:
    """Load and sanity-check a bin_map_{grab,arctic}.json file.

    Checks that expected top-level keys exist and that per-feature bin slices
    are lists of length 2 with int dim_total. Raises ValueError on any failure.

    Args:
        path: Absolute path to bin_map_*.json.

    Returns:
        The parsed dict, unmodified.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"bin_map: file not found: {path}")
    with open(path, "r") as f:
        bm = json.load(f)

    required_top = ["dataset", "fps", "per_frame_features", "episode_metadata"]
    for k in required_top:
        if k not in bm:
            raise ValueError(f"bin_map {path!r}: missing top-level key '{k}'")

    pff = bm["per_frame_features"]
    required_feats = ["x_lhand", "x_rhand", "x_obj"]
    for feat in required_feats:
        if feat not in pff:
            raise ValueError(f"bin_map {path!r}: missing per_frame_features.{feat}")
        feat_spec = pff[feat]
        if not isinstance(feat_spec.get("dim_total"), int):
            raise ValueError(
                f"bin_map {path!r}: per_frame_features.{feat}.dim_total must be int"
            )
        bins = feat_spec.get("bins", {})
        if not isinstance(bins, dict) or len(bins) == 0:
            raise ValueError(
                f"bin_map {path!r}: per_frame_features.{feat}.bins must be a non-empty dict"
            )
        for bname, bspec in bins.items():
            sl = bspec.get("slice")
            if not (isinstance(sl, list) and len(sl) == 2 and
                    isinstance(sl[0], int) and isinstance(sl[1], int)):
                raise ValueError(
                    f"bin_map {path!r}: bins.{feat}.{bname}.slice must be [int,int], got {sl!r}"
                )
    return bm


# ---------------------------------------------------------------------------
# rot6d conversions (Zhou et al.)
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert Zhou et al. 6D rotation to 3x3 rotation matrix.

    Text2HOI convention: rot6d is packed as rotmat[:, :, :2].reshape(N, -1)
    (row-major), giving interleaved layout [R[0,0], R[0,1], R[1,0], R[1,1],
    R[2,0], R[2,1]]. Reshape to (..., 3, 2): column 0 = b1, column 1 = b2.

    Gram-Schmidt on the two 3-vector columns:
        c1 = normalize(b1); c2 = normalize(b2 - (c1.b2) c1); c3 = c1 x c2
        R = [c1 | c2 | c3]  (column-vector convention)

    Degenerate inputs (all-zero 6D, common on Text2HOI zero-padded tail frames) are
    mapped to identity — downstream consumers must mask those frames via valid_mask.

    Args:
        rot6d: (..., 6) float — Text2HOI interleaved rot6d (not block-major).

    Returns:
        (..., 3, 3) float32 rotation matrices.
    """
    rot6d = np.asarray(rot6d)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d_to_rotmat: last dim must be 6, got {rot6d.shape}")
    # Text2HOI packs rot6d as rotmat[:, :, :2].reshape(N, -1) — row-major interleaved.
    # Decoded layout: [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]].
    # Reshape (..., 6) -> (..., 3, 2): column 0 = b1, column 1 = b2.
    m = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    b1 = m[..., :, 0]   # (..., 3) — first column of the rotation matrix
    b2 = m[..., :, 1]   # (..., 3) — second column

    eps = 1e-8
    c1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    c2_raw_norm = np.linalg.norm(b2, axis=-1, keepdims=True)
    c1 = b1 / np.clip(c1_norm, eps, None)

    # project out c1 from b2
    dot = np.sum(c1 * b2, axis=-1, keepdims=True)
    b2_perp = b2 - dot * c1
    c2_norm = np.linalg.norm(b2_perp, axis=-1, keepdims=True)
    c2 = b2_perp / np.clip(c2_norm, eps, None)

    c3 = np.cross(c1, c2)

    # Stack as columns: R[..., :, 0] = c1 etc.
    R = np.stack([c1, c2, c3], axis=-1)  # (..., 3, 3)

    # Degenerate mask: either b1 or b2 was zero-vector -> rotation undefined.
    # Substitute identity. Applies to padded tail frames that are all-zero.
    degenerate = (
        (c1_norm < eps) | (c2_raw_norm < eps) | (c2_norm < eps)
    ).squeeze(-1)  # (...,)
    if degenerate.any():
        eye = np.broadcast_to(np.eye(3, dtype=R.dtype), R.shape).copy()
        R = np.where(degenerate[..., None, None], eye, R)
    return R.astype(np.float32, copy=False)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert Zhou et al. 6D rotation to axis-angle (rotvec).

    Vectorized across all leading dimensions via scipy. No per-frame Python loop.
    Handles identity (->(0,0,0)) robustly.

    Args:
        rot6d: (..., 6) float.

    Returns:
        (..., 3) float32 axis-angle.
    """
    rot6d = np.asarray(rot6d)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d_to_axis_angle: last dim must be 6, got {rot6d.shape}")
    lead_shape = rot6d.shape[:-1]
    R = rot6d_to_rotmat(rot6d).reshape(-1, 3, 3).astype(np.float64, copy=False)
    # scipy.from_matrix handles batched (N,3,3) -> as_rotvec -> (N,3)
    rotvec = R_scipy.from_matrix(R).as_rotvec()
    return rotvec.reshape(*lead_shape, 3).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Text key composition (mirrors Text2HOI process_text)
# ---------------------------------------------------------------------------

def compose_text_key(action_name: str, obj_name: str,
                     is_lhand: int, is_rhand: int) -> str:
    """Compose the exact text lookup key used by Text2HOI's process_text.

    This is the canonical text-key composition rule for both GRAB and ARCTIC datasets.
    All adapters and metric-port readers must call this function rather
    than reimplementing the rule independently, to guarantee text-key alignment.

    Verified against:
      - /tmp/Text2HOI/lib/utils/proc_grab.py:8-14 (process_text for GRAB)
      - /tmp/Text2HOI/lib/utils/proc_arctic.py:20-26 (process_text for ARCTIC)

    Quotes from proc_grab.py:8-14:
        if is_lhand and is_rhand: text = f"{action_name} {obj_name} with both hands."
        elif is_lhand:            text = f"{action_name} {obj_name} with left hand."
        elif is_rhand:            text = f"{action_name} {obj_name} with right hand."
        text_key = text.capitalize()
    """
    if not action_name:
        raise ValueError("_compose_text_key: empty action_name")
    if not obj_name:
        raise ValueError("_compose_text_key: empty obj_name")
    if is_lhand and is_rhand:
        text = f"{action_name} {obj_name} with both hands."
    elif is_lhand:
        text = f"{action_name} {obj_name} with left hand."
    elif is_rhand:
        text = f"{action_name} {obj_name} with right hand."
    else:
        raise ValueError(
            f"_compose_text_key: neither hand active for action={action_name!r}, "
            f"obj={obj_name!r}"
        )
    return text.capitalize()


# ---------------------------------------------------------------------------
# Main iterator
# ---------------------------------------------------------------------------

def iter_gt_episodes(
    dataset: str,
    bin_map_path: str,
    npz_path: str,
    text_json_path: str,
    max_episodes: Optional[int] = None,
) -> Iterator[CanonicalSample]:
    """Yield CanonicalSample per episode from a Text2HOI raw NPZ.

    Args:
        dataset: "grab" or "arctic".
        bin_map_path: Path to the matching bin_map_*.json.
        npz_path: Path to Text2HOI data.npz.
        text_json_path: Path to Text2HOI text.json (dict[str, List[str]]).
        max_episodes: Cap iteration at this many episodes (for tests). None = all.

    Yields:
        CanonicalSample, one per raw episode.

    Raises:
        ValueError: On bad dataset literal, malformed npz, or missing text key.
    """
    if dataset not in ("grab", "arctic"):
        raise ValueError(f"iter_gt_episodes: dataset must be 'grab' or 'arctic', got {dataset!r}")

    bm = load_bin_map(bin_map_path)  # single source of truth for all slice indices and fps

    # --- Read fps from bin_map (not hardcoded) ---
    fps = int(bm["fps"])

    # --- Extract slice indices from bin_map ---
    pff = bm["per_frame_features"]

    # Hand rot6d: slice [lh_rot6d_root.start : lh_rot6d_finger_joints.end]
    # = rot6d_global (1 joint) + rot6d_j1..j15 (15 joints) = 16 joints total.
    # Both GRAB and ARCTIC have identical x_lhand/x_rhand layout.
    lh_bins = pff["x_lhand"]["bins"]
    lh_rot_start = lh_bins["lh_rot6d_root"]["slice"][0]
    lh_rot_end   = lh_bins["lh_rot6d_finger_joints"]["slice"][1]
    lh_trans_s, lh_trans_e = lh_bins["lh_trans"]["slice"]

    rh_bins = pff["x_rhand"]["bins"]
    rh_rot_start = rh_bins["rh_rot6d_root"]["slice"][0]
    rh_rot_end   = rh_bins["rh_rot6d_finger_joints"]["slice"][1]

    obj_bins = pff["x_obj"]["bins"]
    obj_trans_s, obj_trans_e = obj_bins["obj_trans"]["slice"]
    obj_rot_s,   obj_rot_e   = obj_bins["obj_rot6d"]["slice"]

    # Consistency check: bin_map dim_total must match the expected slice span
    # for the rot6d block (root + finger joints = 16 joints * 6 = 96 dims).
    n_rot_joints = (lh_rot_end - lh_rot_start) // 6
    if n_rot_joints != 16:
        raise ValueError(
            f"bin_map {bin_map_path!r}: expected 16 rot joints "
            f"from slice [{lh_rot_start}:{lh_rot_end}], got {n_rot_joints}"
        )

    with open(text_json_path, "r") as f:
        text_json: Dict[str, Any] = json.load(f)

    data = np.load(npz_path, allow_pickle=True)

    x_lhand = data["x_lhand"]          # (N, T, 99) float32
    x_rhand = data["x_rhand"]          # (N, T, 99) float32
    j_lhand = data["j_lhand"]          # (N, T, 21, 3) float32
    j_rhand = data["j_rhand"]          # (N, T, 21, 3) float32
    x_obj   = data["x_obj"]            # (N, T, 9)  float32
    nframes = data["nframes"]          # (N,) int64
    is_lhand_arr = data["is_lhand"]    # (N,) int64
    is_rhand_arr = data["is_rhand"]    # (N,) int64
    action_name_arr = data["action_name"]  # (N,) str

    if dataset == "grab":
        # fix: obj_pcs (from obj.pkl) is keyed by CamelCase/size-suffixed
        # ``obj_name`` ("alarmclock", "cubelarge"), not space-separated
        # ``proc_obj_name`` ("alarm clock", "cube"). 530/1335 GRAB rows (~40%)
        # have obj_name != proc_obj_name, so using proc_obj_name for the
        # canonical ``obj_name`` field breaks downstream obj_pcs lookup in
        # compute_physical_realism. proc_obj_name is needed ONLY for the
        # text.json key (per /tmp/Text2HOI/lib/datasets/grab.py:289-297).
        obj_name_arr = data["obj_name"]
        proc_obj_name_arr = data["proc_obj_name"]
        x_obj_angle_all = None
    else:
        # ARCTIC has a single object-name column ``object_name`` (no CamelCase
        # vs space-separated split like GRAB) — confirmed by inspecting
        # /tmp/Text2HOI/prepare_data/arctic/proc_arctic.py (writes only
        # ``object_name``) and text.json keys which are literal ``object_name``
        # values (e.g. "microwave", "laptop"). Aliasing is intentional; if a
        # future ARCTIC preprocessing ever introduces a separate proc column,
        # replace this alias with the appropriate column.
        obj_name_arr = data["object_name"]
        proc_obj_name_arr = obj_name_arr
        # Revolute articulation angle (radians) per frame — only present for ARCTIC.
        # Shape: (N, T_padded, 1) float32. Padded frames are 0.0.
        if "x_obj_angle" not in data.files:
            raise ValueError(
                f"iter_gt_episodes: dataset='arctic' expects 'x_obj_angle' in "
                f"{npz_path!r}, not found. Keys: {sorted(data.files)}"
            )
        x_obj_angle_all = data["x_obj_angle"]  # (N, T_padded, 1) float32

    N_total, T_padded = x_lhand.shape[0], x_lhand.shape[1]

    # Fail-loud consistency check: bin_map dim_total must match loaded array width.
    if x_lhand.shape[2] != pff["x_lhand"]["dim_total"]:
        raise ValueError(
            f"x_lhand dim mismatch: npz has {x_lhand.shape[2]}, "
            f"bin_map says {pff['x_lhand']['dim_total']}"
        )
    if x_obj.shape[2] != pff["x_obj"]["dim_total"]:
        raise ValueError(
            f"x_obj dim mismatch: npz has {x_obj.shape[2]}, "
            f"bin_map says {pff['x_obj']['dim_total']}"
        )

    n_ep = N_total if max_episodes is None else min(max_episodes, N_total)

    for i in range(n_ep):
        T_valid = int(nframes[i])
        if T_valid <= 0:
            raise ValueError(f"episode {i}: nframes={T_valid} (non-positive)")

        # valid_mask over padded length
        valid_mask = np.arange(T_padded) < T_valid  # (T_padded,) bool

        # Hand joints — already MANO-decoded, world-frame meters, 21 joints w/ tip
        lhj = np.asarray(j_lhand[i], dtype=np.float32)  # (T_padded, 21, 3)
        rhj = np.asarray(j_rhand[i], dtype=np.float32)

        # slice [lh_rot6d_root.start : lh_rot6d_finger_joints.end]
        # = rot6d_global + rot6d_j1..j15 = 16 joints total.
        # reshape (T, 96) -> (T, 16, 6) -> (T, 16, 3)
        lh_rot6d = x_lhand[i, :, lh_rot_start:lh_rot_end].reshape(T_padded, n_rot_joints, 6)
        rh_rot6d = x_rhand[i, :, rh_rot_start:rh_rot_end].reshape(T_padded, n_rot_joints, 6)
        lh_aa = rot6d_to_axis_angle(lh_rot6d).astype(np.float32, copy=False)
        rh_aa = rot6d_to_axis_angle(rh_rot6d).astype(np.float32, copy=False)

        # Object translation and rotation matrix (slices from bin_map)
        obj_trans = np.asarray(x_obj[i, :, obj_trans_s:obj_trans_e], dtype=np.float32)  # (T, 3)
        obj_rot6d = x_obj[i, :, obj_rot_s:obj_rot_e].reshape(T_padded, 6)
        obj_rotmat = rot6d_to_rotmat(obj_rot6d).astype(np.float32, copy=False)  # (T, 3, 3)

        # Metadata
        action_name = str(action_name_arr[i])
        obj_name = str(obj_name_arr[i])            # obj.pkl key (CamelCase for GRAB)
        proc_obj_name = str(proc_obj_name_arr[i])  # text.json key (space-separated for GRAB)
        is_l = int(is_lhand_arr[i])
        is_r = int(is_rhand_arr[i])

        # Text lookup — deterministic, first paraphrase
        key = compose_text_key(action_name, proc_obj_name, is_l, is_r)
        if key not in text_json:
            raise ValueError(
                f"episode {i}: text key not found in {text_json_path!r}: {key!r}"
            )
        paraphrases = text_json[key]
        if not isinstance(paraphrases, list) or len(paraphrases) == 0:
            raise ValueError(
                f"episode {i}: text.json[{key!r}] not a non-empty list (got {type(paraphrases).__name__})"
            )
        text = str(paraphrases[0])

        obj_arti = None
        if x_obj_angle_all is not None:
            # (T_padded,) float32. Radians. Padded frames are 0.0.
            obj_arti = np.asarray(x_obj_angle_all[i, :, 0], dtype=np.float32)

        sample = CanonicalSample(
            left_hand_joints=lhj,
            right_hand_joints=rhj,
            left_hand_rot=lh_aa,
            right_hand_rot=rh_aa,
            obj_translation=obj_trans,
            obj_rotation=obj_rotmat,
            obj_name=obj_name,
            valid_mask=valid_mask,
            fps=fps,
            text=text,
            obj_articulation_angle=obj_arti,
            is_lhand=bool(is_l),
            is_rhand=bool(is_r),
        )
        yield sample


# ---------------------------------------------------------------------------
# ARCTIC articulation helper (not part of canonical schema)
# ---------------------------------------------------------------------------

def iter_arctic_articulation(npz_path: str) -> Iterator[np.ndarray]:
    """Yield ARCTIC articulation angle (rad) per episode.

    ARCTIC objects each have exactly one hinge/revolute articulation DoF (laptop
    lid, box lid, scissors, mixer dial, etc.). Canonical schema deliberately omits
    this to stay dataset-agnostic; downstream /code may use this helper
    when articulation is needed.

    Yields in the same episode index order as `iter_gt_episodes(dataset='arctic',
    npz_path=same_path, ...)`. Zip the two iterators for aligned
    (canonical_sample, articulation_angle) pairs. Iterates ALL episodes; use
    `itertools.islice` or `zip(iter_gt_episodes(..., max_episodes=M),
    iter_arctic_articulation(...))` for truncation.

    Args:
        npz_path: Path to the ARCTIC Text2HOI data.npz.

    Yields:
        (T_padded,) float32 articulation angle per episode. Padded frames are 0.0.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "x_obj_angle" not in data.files:
        raise ValueError(
            f"iter_arctic_articulation: 'x_obj_angle' not in {npz_path!r} — "
            f"only ARCTIC has this field"
        )
    x_obj_angle = data["x_obj_angle"]  # (N, T, 1) float32
    for i in range(x_obj_angle.shape[0]):
        yield np.asarray(x_obj_angle[i, :, 0], dtype=np.float32)
