"""IMoS raw output -> canonical .npz adapter.

IMoS (Ghosh et al., 2023) generates full-body SMPL-X motion with object
interaction. The canonical eval pipeline needs only hand joints and object
pose. This adapter extracts those from the IMoS synthesized output by running
SMPL-X forward kinematics on the saved pose parameters.

Input: one file per sample saved via `torch.save(output, outfname)` where
output is:

    {
        'obj_p': {
            'transl':        Tensor (T, 3)  -- object translation (world frame)
            'global_orient': Tensor (T, 3)  -- object rotation as axis-angle
        },
        'sbj_p': {
            # From batch_parms_6D2smplx_params(output_fullpose, transl)
            # -> fullpose2smplx_params_rotmat with shape (T, 55, 3, 3)
            'global_orient':   Tensor (T, 1, 3, 3)
            'body_pose':       Tensor (T, 21, 3, 3)
            'jaw_pose':        Tensor (T, 1, 3, 3)
            'leye_pose':       Tensor (T, 1, 3, 3)
            'reye_pose':       Tensor (T, 1, 3, 3)
            'left_hand_pose':  Tensor (T, 15, 3, 3)  -- finger joints [25..39]
            'right_hand_pose': Tensor (T, 15, 3, 3)  -- finger joints [40..54]
            'transl':          Tensor (T, 3)
        },
        'sequence_name': str,      -- e.g. "s10/airplane_fly_1.npz"
        'sbj_vtemp':     Tensor (1, V, 3),  -- subject v_template
        'gender':        int,      -- 1=male, 2=female (dataloader convention)
        'obj_name':      str       -- GRAB raw name (e.g. "airplane", "alarmclock")
    }

Output: canonical CanonicalSample with per-frame hand joints (21 per hand) and
object pose, plus metadata fields. Loaded via smplx.SMPLXLayer forward pass to
compute world-frame joint positions.

Joint extraction (from /tmp/IMoS/src/test/test_synthesis.py:35-41):
    left_hand_finger_joints  = [25..39]  (SMPL-X indices for left fingers)
    right_hand_finger_joints = [40..54]  (SMPL-X indices for right fingers)
    left_wrist               = joint 20
    right_wrist              = joint 21
    SMPL-X total joints from SMPLXLayer output: 127 (but we use only 0..54)

MANO 21-joint convention (canonical schema):
    joints 0:  wrist (SMPL-X joint 20 or 21)
    joints 1-15: 15 finger joints (SMPL-X left_hand_finger_joints / right_hand_finger_joints)
    joints 16-20: 5 fingertips -- SMPL-X has no tip joints; we pad by duplicating
                  the last joint of each finger chain (distal phalanx). Documented
                  as an approximation: the eval metrics only use joints 0..15 for
                  distance-based metrics (wrist + knuckle skeleton).

Hand axis-angle (canonical left_hand_rot / right_hand_rot), shape (T, 16, 3):
    index 0:    wrist axis-angle (from SMPL-X joint 20/21 in fullpose as rotmat -> aa)
    index 1-15: 15 finger joint axis-angles (from sbj_p['left_hand_pose'] /
                sbj_p['right_hand_pose'])

CLI:
    python -m umm.eval.handobject.baseline_adapters.imos_to_canonical \\
        --raw-dir data/handobject/baseline_raw/imos_grab \\
        --out-dir data/handobject/baseline_samples/imos_grab \\
        --smplx-model-root data/handobject/imos_deps/smplx/models \\
        --smplx-corr-root data/handobject/imos_deps/tools/smplx_correspondence
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as R_scipy

from umm.eval.handobject.baseline_adapters.text2hoi_to_canonical import (
    _translate_obj_name_grab,
)
from umm.eval.handobject.canonical import (
    CanonicalSample,
    validate_canonical_sample,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL-X joint index constants (from /tmp/IMoS/src/test/test_synthesis.py:35-41)
# ---------------------------------------------------------------------------

# SMPL-X indices for the left and right finger joints (0-indexed into the 55-joint
# fullpose layout used by IMoS's fullpose2smplx_params_rotmat).
_LEFT_WRIST_IDX: int = 20
_RIGHT_WRIST_IDX: int = 21
_LEFT_FINGER_SMPLX: list = list(range(25, 40))   # 15 joints
_RIGHT_FINGER_SMPLX: list = list(range(40, 55))  # 15 joints

# SMPL-X output joints (from SMPLXLayer) order for the hand skeleton.
# SMPLXLayer.joints has shape (T, 127, 3). The first 55 entries match the
# fullpose joint ordering (see /tmp/IMoS/src/evaluate/eval.py:31-33).
# Wrists = joints 20 and 21; finger joints = 25-39 (left), 40-54 (right).
_SMPLX_LEFT_HAND_JOINTS: list = [_LEFT_WRIST_IDX] + _LEFT_FINGER_SMPLX   # 16 joints
_SMPLX_RIGHT_HAND_JOINTS: list = [_RIGHT_WRIST_IDX] + _RIGHT_FINGER_SMPLX  # 16 joints

# Intent -> verb text mapping for text generation.
# Derived from /tmp/IMoS/src/data_loader/dataset_preprocess.py:32 and plural_dict:48+
# Maps intent string (from sequence_name part [1]) to readable verb phrase.
# map IMoS intent-string to a GRAB-vocab first-word verb so eval_imos's
# text-to-label parser (`action_label.text_to_label_idx`) finds a known verb.
# GRAB vocab (first word of text) has 29 verbs; see action_labels.json:grab.labels.
# Intents NOT in GRAB vocab (wear, use, offhand) mapped to closest match:
#   - wear  -> "play" (paper groups "wear eyeglasses" under inspect/play)
#   - use   -> "inspect" (generic interaction)
#   - offhand -> "hand" (hand-over motion; "hand" IS in GRAB vocab)
_INTENT_TO_VERB: dict = {
    "offhand": "hand",
    "pass": "pass",
    "lift": "lift",
    "drink": "drink",
    "brush": "brush",
    "eat": "eat",
    "peel": "peel",
    "takepicture": "take",
    "see": "see",
    "wear": "play",       # NOT in GRAB vocab -> fallback
    "play": "play",
    "clean": "clean",
    "browse": "browse",
    "inspect": "inspect",
    "pour": "pour",
    "use": "inspect",     # NOT in GRAB vocab -> fallback
    "switchON": "switch",
    "cook": "cook",
    "toast": "toast",
    "staple": "staple",
    "squeeze": "squeeze",
    "set": "set",
    "open": "open",
    "chop": "chop",
    "screw": "screw",
    "call": "call",
    "shake": "shake",
    "fly": "fly",
    "stamp": "stamp",
}


# ---------------------------------------------------------------------------
# SMPL-X model cache (avoid reloading per sample)
# ---------------------------------------------------------------------------

_SMPLX_MODEL_CACHE: dict = {}


def _get_smplx_layer(model_path: str, gender: str):
    """Return a cached smplx.SMPLXLayer for the given (model_path, gender) pair.

    The smplx library is loaded lazily to avoid hard-importing at module level
    (it is not always installed in the `umm` conda env).

    Args:
        model_path: Path to directory containing SMPL-X model .npz files
                    (e.g. .../smplx/models/smplx/).
        gender:     "male" or "female".

    Returns:
        smplx.SMPLXLayer instance (CPU).
    """
    import torch
    try:
        import smplx
    except ImportError as exc:
        raise ImportError(
            "smplx is not installed in this environment. "
            "Activate the `imos` conda env or install with: "
            "pip install smplx[all]"
        ) from exc

    key = (os.path.abspath(model_path), gender)
    if key not in _SMPLX_MODEL_CACHE:
        logger.info("_get_smplx_layer: loading SMPL-X model gender=%s from %s", gender, model_path)
        layer = smplx.SMPLXLayer(
            model_path=model_path,
            gender=gender,
            num_pca_comps=45,
            flat_hand_mean=True,
        )
        layer.eval()
        _SMPLX_MODEL_CACHE[key] = layer
    return _SMPLX_MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotmat_to_axis_angle(rotmat: ndarray) -> ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 3) axis-angle (rotvec).

    Args:
        rotmat: (..., 3, 3) float.

    Returns:
        (..., 3) float32 axis-angle.
    """
    lead = rotmat.shape[:-2]
    R_flat = rotmat.reshape(-1, 3, 3).astype(np.float64)
    aa = R_scipy.from_matrix(R_flat).as_rotvec()
    return aa.reshape(*lead, 3).astype(np.float32)


def _aa_to_rotmat(aa: ndarray) -> ndarray:
    """Convert (..., 3) axis-angle to (..., 3, 3) rotation matrices.

    Args:
        aa: (..., 3) float.

    Returns:
        (..., 3, 3) float32.
    """
    lead = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3).astype(np.float64)
    R_flat = R_scipy.from_rotvec(aa_flat).as_matrix()
    return R_flat.reshape(*lead, 3, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Gender decoding
# ---------------------------------------------------------------------------

def _decode_gender(gender_raw) -> str:
    """Decode IMoS gender encoding to SMPL-X string.

    IMoS dataloader converts gender strings to ints:
        1 = male, 2 = female
    (see /tmp/IMoS/src/data_loader/dataloader.py:82-85)

    Also accepts string "male"/"female" in case the file stores the string form.

    Args:
        gender_raw: int (1 or 2) or str ("male" / "female").

    Returns:
        "male" or "female".

    Raises:
        ValueError: If gender cannot be decoded.
    """
    if isinstance(gender_raw, (int, np.integer)):
        g = int(gender_raw)
        if g == 1:
            return "male"
        elif g == 2:
            return "female"
        else:
            raise ValueError(
                f"_decode_gender: unexpected int gender value {g!r}. "
                "Expected 1=male, 2=female."
            )
    if isinstance(gender_raw, (str, bytes)):
        s = gender_raw.decode() if isinstance(gender_raw, bytes) else gender_raw
        s_lower = s.strip().lower()
        if s_lower in ("male", "m"):
            return "male"
        elif s_lower in ("female", "f"):
            return "female"
        else:
            raise ValueError(
                f"_decode_gender: unexpected string gender value {s!r}. "
                "Expected 'male' or 'female'."
            )
    # Tensor with a single value
    try:
        v = int(gender_raw)
        return _decode_gender(v)
    except Exception:
        pass
    raise ValueError(
        f"_decode_gender: cannot decode gender from {gender_raw!r} (type {type(gender_raw).__name__})"
    )


# ---------------------------------------------------------------------------
# Text generation from sequence_name
# ---------------------------------------------------------------------------

def _sequence_name_to_text(sequence_name: str, obj_name_canonical: str) -> str:
    """Build a human-readable text annotation from the IMoS sequence name.

    IMoS sequence names follow GRAB convention:
        "<subject>/<object>_<intent>_<trial>.npz"
    e.g. "s10/airplane_fly_1.npz"

    The intent is extracted from the filename and mapped to a verb phrase.
    Sentence format: "The person <verb> the <obj>."

    If the intent cannot be decoded (unknown or absent), falls back to:
        "The person interacts with the <obj>."

    Args:
        sequence_name:    Raw sequence path string from the IMoS output dict.
        obj_name_canonical: Canonical space-separated object name (already translated).

    Returns:
        str text annotation.
    """
    # FIX-REQUIRED from .
    # IMoS `sequence_name` is underscore-joined like "s10_apple_eat_1.npz" OR
    # "data/post_processed/test/s10_apple_eat_1.npz". After basename+split("_"):
    #   parts = ["s10", "apple", "eat", "1"]
    #   parts[0] = subject, parts[1] = OBJECT, parts[2] = INTENT, parts[3] = trial.
    # Previous code read parts[1] as intent -> all 25 samples fell back to
    # "inspect <obj>." regardless of real intent, collapsing accuracy to 1/25.
    #
    # `eval_imos.action_label.text_to_label_idx` parses the FIRST word of text
    # as the verb. So text is imperative: "<verb> <obj>.".
    fname = os.path.basename(sequence_name)
    stem = fname.replace(".npz", "")
    parts = stem.split("_")
    # Find the intent. Typical forms seen in IMoS output:
    #   "s10_apple_eat_1"         -> parts[2] = "eat"
    #   "s10_apple_eat_Retake_1"  -> parts[2] = "eat" still (Retake is suffix)
    intent = None
    if len(parts) >= 3 and parts[0].startswith("s"):
        # subject-prefixed form
        intent = parts[2]
    elif len(parts) >= 2:
        # legacy fallback: <obj>_<intent>_<trial>
        intent = parts[1]
    if intent is not None:
        verb = _INTENT_TO_VERB.get(intent)
        if verb is not None:
            return f"{verb} {obj_name_canonical}."
    # Fallback: "inspect" is in GRAB vocab.
    return f"inspect {obj_name_canonical}."


# ---------------------------------------------------------------------------
# Core single-sample conversion
# ---------------------------------------------------------------------------

# Cache for MANO->SMPL-X vertex ID correspondence arrays.
# rhand_smplx_ids.npy / lhand_smplx_ids.npy: (778,) int — for each of MANO's 778
# vertices, the corresponding SMPL-X body-mesh vertex index. Used to extract
# real fingertip positions from SMPL-X output vertices (per MANO tip IDs).
_SMPLX_CORR_CACHE: dict = {}

# MANO convention fingertip vertex IDs (within MANO's 778-vertex right/left hand mesh).
# FIX-REQUIRED #1 from : the tip concatenation ORDER in
# Text2HOI's MANO_C wrapper is [index, middle, pinky, ring, thumb]
# (/tmp/Text2HOI/lib/models/mano.py:99-102). GT samples from gt_loader.py follow
# the same order. Previous iter-2 used [thumb, index, middle, ring, pinky] which
# produced a systematic per-finger mislabeling vs GT → FID/MM inflation.
# Vertex IDs (MANO convention): thumb=745, index=317, middle=445, ring=556, pinky=673.
_MANO_TIP_IDS_RIGHT: list = [317, 445, 673, 556, 745]  # [index, middle, pinky, ring, thumb]
_MANO_TIP_IDS_LEFT: list = [317, 445, 673, 556, 745]   # same convention (symmetric)


def _load_smplx_corr(smplx_corr_root: str):
    """Load (rhand, lhand) MANO->SMPLX vertex ID arrays. Returns (r_ids, l_ids) tuple
    of int ndarrays shape (778,), or (None, None) if root path empty.
    """
    if not smplx_corr_root:
        return None, None
    if smplx_corr_root in _SMPLX_CORR_CACHE:
        return _SMPLX_CORR_CACHE[smplx_corr_root]
    rpath = os.path.join(smplx_corr_root, "rhand_smplx_ids.npy")
    lpath = os.path.join(smplx_corr_root, "lhand_smplx_ids.npy")
    if not (os.path.exists(rpath) and os.path.exists(lpath)):
        raise FileNotFoundError(
            f"_load_smplx_corr: missing {rpath!r} or {lpath!r}. "
            "Smplx correspondence required for real fingertip extraction."
        )
    r_ids = np.load(rpath).astype(np.int64)  # (778,)
    l_ids = np.load(lpath).astype(np.int64)
    _SMPLX_CORR_CACHE[smplx_corr_root] = (r_ids, l_ids)
    return r_ids, l_ids


def _convert_single(
    raw_path: str,
    smplx_model_root: str,
    smplx_corr_root: str = "",
) -> CanonicalSample:
    """Convert one IMoS torch-saved file to a CanonicalSample.

    Strategy:
    1. Load raw dict via torch.load.
    2. Decode gender, obj_name, sequence_name.
    3. Run SMPL-X FK using sbj_p + sbj_vtemp to obtain world-frame joints.
    4. Extract 21 hand joints per hand from SMPL-X joints[0..54].
       -- 16 skeletal joints (wrist + 15 fingers) + 5 approximated tips
          (duplicate the last joint per finger chain as a lossy approximation;
          documented here since eval metrics use wrist+finger distance, not tips).
    5. Extract hand axis-angle rotations from sbj_p rotmats.
    6. Extract object pose (transl + aa->rotmat).
    7. Build and validate CanonicalSample.

    Fingertip approximation rationale:
        SMPL-X with 55 joints has no separate tip joints. The MANO convention
        expects 21 joints (16 skeletal + 5 tips). We replicate the distal
        phalanx joint for each finger as a proxy. For downstream metrics this
        means tip-distance errors will be underestimated by ~1-2cm (fingertip
        vs distal joint). This is acceptable for the baseline reproduction task
        where all methods are compared consistently.

        Finger chain layout in SMPL-X (left-hand example, 0-indexed in sbj_p):
          thumb:   joints [0, 1, 2]  (LF_tip approx = joint 2)
          index:   joints [3, 4, 5]  (LF_tip approx = joint 5)
          middle:  joints [6, 7, 8]  (LF_tip approx = joint 8)
          ring:    joints [9, 10, 11](LF_tip approx = joint 11)
          pinky:   joints [12,13,14] (LF_tip approx = joint 14)

    Args:
        raw_path:         Path to a torch-saved IMoS output file (.npz extension).
        smplx_model_root: Directory containing SMPL-X model .npz files.
                          Passed as `model_path` to smplx.SMPLXLayer.

    Returns:
        Validated CanonicalSample.

    Raises:
        KeyError:   If a required key is missing from the raw file.
        ValueError: If shapes are wrong or gender/obj_name cannot be decoded.
    """
    import torch

    # --- Load raw dict ---
    raw = torch.load(raw_path, map_location="cpu")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "_convert_single: loaded %s, top-level keys: %s",
            raw_path, list(raw.keys())
        )

    # --- Validate top-level keys ---
    for key in ("obj_p", "sbj_p", "sequence_name", "sbj_vtemp", "gender", "obj_name"):
        if key not in raw:
            raise KeyError(
                f"_convert_single: missing top-level key {key!r} in {raw_path!r}. "
                f"Present keys: {list(raw.keys())}"
            )

    for sub in ("transl", "global_orient"):
        if sub not in raw["obj_p"]:
            raise KeyError(
                f"_convert_single: missing obj_p[{sub!r}] in {raw_path!r}. "
                f"obj_p keys: {list(raw['obj_p'].keys())}"
            )

    # --- Decode metadata ---
    gender_str = _decode_gender(raw["gender"])
    obj_name_raw = str(raw["obj_name"])
    obj_name_canonical = _translate_obj_name_grab(obj_name_raw)
    sequence_name = str(raw["sequence_name"])
    text = _sequence_name_to_text(sequence_name, obj_name_canonical)

    logger.info(
        "_convert_single: gender=%s obj=%r -> %r seq=%s",
        gender_str, obj_name_raw, obj_name_canonical, sequence_name,
    )

    # --- Extract sbj_p arrays ---
    # sbj_p comes from fullpose2smplx_params_rotmat which produces rotmats.
    # transl shape: (T, 3)
    # global_orient: (T, 1, 3, 3) or possibly (T, 3, 3)
    # left_hand_pose: (T, 15, 3, 3)
    # right_hand_pose: (T, 15, 3, 3)
    sbj_p = raw["sbj_p"]
    transl_tensor = sbj_p["transl"]  # (T, 3)

    # Infer T from transl.
    if transl_tensor.ndim == 1:
        # Degenerate case: single frame stored as (3,). Expand.
        transl_tensor = transl_tensor.unsqueeze(0)
    T = int(transl_tensor.shape[0])
    logger.info("_convert_single: T=%d frames", T)

    # --- SMPL-X FK forward pass ---
    # We need world-frame joint positions. Use sbj_vtemp as v_template for
    # subject-specific shape personalization (IMoS stores the pre-computed mesh).
    smplx_layer = _get_smplx_layer(smplx_model_root, gender_str)

    # Build sbj_p dict for smplx forward pass.
    # All params must be float tensors with batch dim.
    def _to_float(t) -> "torch.Tensor":
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        return t.float()

    smplx_kwargs: dict = {
        "transl": _to_float(transl_tensor),  # (T, 3)
        # expand creates a non-contiguous view; .contiguous() makes it safe for smplx.
        "v_template": _to_float(raw["sbj_vtemp"]).squeeze(0).unsqueeze(0).expand(T, -1, -1).contiguous(),  # (T, V, 3)
    }

    # This SMPLXLayer variant (imos env) expects rotation matrices (not axis-angle)
    # for all pose parameters. It calls lbs() with pose2rot=False and concatenates
    # all pose components as (B, K, 3, 3) rotmats internally.
    # See smplx/body_models.py lines 1387-1427.
    #
    # fullpose2smplx_params_rotmat (IMoS source) already produces rotmats:
    #   global_orient: (T, 1, 3, 3)
    #   body_pose:     (T, 21, 3, 3)
    #   left_hand_pose:(T, 15, 3, 3)
    #   right_hand_pose:(T, 15, 3, 3)
    # We pass them directly as rotmats.

    def _ensure_batch(t, expected_ndim: int):
        """Add leading batch dim if missing. Returns float contiguous tensor."""
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        t = t.float()
        while t.ndim < expected_ndim:
            t = t.unsqueeze(0)
        return t.contiguous()

    # global_orient: (T, 1, 3, 3) rotmat.
    g_orient = _ensure_batch(sbj_p["global_orient"], 4)  # (T, 1, 3, 3)
    smplx_kwargs["global_orient"] = g_orient  # (T, 1, 3, 3)

    # body_pose: (T, 21, 3, 3) rotmat.
    if "body_pose" in sbj_p:
        smplx_kwargs["body_pose"] = _ensure_batch(sbj_p["body_pose"], 4)  # (T, 21, 3, 3)

    # jaw/eye poses: identity rotmats (face not needed for hands).
    eye3 = torch.eye(3, dtype=torch.float32)
    for pk in ("jaw_pose", "leye_pose", "reye_pose"):
        smplx_kwargs[pk] = eye3.view(1, 1, 3, 3).expand(T, 1, 3, 3).contiguous()

    # left_hand_pose: (T, 15, 3, 3) rotmat — store for aa extraction.
    left_hand_rotmat = None
    if "left_hand_pose" in sbj_p:
        lhp = _ensure_batch(sbj_p["left_hand_pose"], 4)  # (T, 15, 3, 3)
        left_hand_rotmat = lhp.detach().cpu().numpy().astype(np.float32)  # save for aa extraction
        smplx_kwargs["left_hand_pose"] = lhp  # (T, 15, 3, 3)
    else:
        smplx_kwargs["left_hand_pose"] = eye3.view(1, 1, 3, 3).expand(T, 15, 3, 3).contiguous()
        left_hand_rotmat = np.tile(np.eye(3, dtype=np.float32)[None, None], (T, 15, 1, 1))

    right_hand_rotmat = None
    if "right_hand_pose" in sbj_p:
        rhp = _ensure_batch(sbj_p["right_hand_pose"], 4)  # (T, 15, 3, 3)
        right_hand_rotmat = rhp.detach().cpu().numpy().astype(np.float32)
        smplx_kwargs["right_hand_pose"] = rhp  # (T, 15, 3, 3)
    else:
        smplx_kwargs["right_hand_pose"] = eye3.view(1, 1, 3, 3).expand(T, 15, 3, 3).contiguous()
        right_hand_rotmat = np.tile(np.eye(3, dtype=np.float32)[None, None], (T, 15, 1, 1))

    # Ensure all tensors are contiguous before calling smplx.
    # smplx's lbs.py (imos env version) uses .view() on pose slices which fails on
    # non-contiguous tensors produced by internal concatenation. The lbs.py in the imos
    # env has been patched to call .contiguous() before .view() (one-line fix).
    # We also call .contiguous() here as belt-and-suspenders.
    smplx_kwargs = {k: v.contiguous() for k, v in smplx_kwargs.items()}

    # Run SMPL-X forward pass.
    with torch.no_grad():
        smplx_out = smplx_layer(**smplx_kwargs)

    # joints shape: (T, 127, 3) — first 55 match the fullpose ordering.
    joints_all = smplx_out.joints.detach().cpu().numpy().astype(np.float32)  # (T, 127, 3)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("_convert_single: smplx joints shape=%s", joints_all.shape)

    # --- Extract hand joint positions (16 skeletal per hand) ---
    # SMPL-X joint index to position mapping.
    left_wrist_pos = joints_all[:, _LEFT_WRIST_IDX, :]    # (T, 3)
    right_wrist_pos = joints_all[:, _RIGHT_WRIST_IDX, :]  # (T, 3)
    left_finger_pos = joints_all[:, _LEFT_FINGER_SMPLX, :]   # (T, 15, 3)
    right_finger_pos = joints_all[:, _RIGHT_FINGER_SMPLX, :] # (T, 15, 3)

    # Build 16-joint arrays: [wrist (1), fingers (15)] -> (T, 16, 3)
    left_hand_16 = np.concatenate(
        [left_wrist_pos[:, None, :], left_finger_pos], axis=1
    )   # (T, 16, 3)
    right_hand_16 = np.concatenate(
        [right_wrist_pos[:, None, :], right_finger_pos], axis=1
    )   # (T, 16, 3)

    # FIX-REQUIRED #2 from .
    # Previous iter duplicated last finger joint as tips. That feeds into
    # motion_feats.canonical_to_points -> classifier feature encoder, shifting
    # distribution vs GT (real MANO tips). Fix: use SMPL-X vertex output +
    # MANO->SMPLX correspondence to extract ACTUAL fingertip vertex positions.
    #: silent fallback removed.
    # smplx_corr_root is now required at the CLI level and must produce valid
    # correspondence arrays. Any failure raises immediately rather than silently
    # degrading to duplicated-phalanx tips.
    r_corr, l_corr = _load_smplx_corr(smplx_corr_root)
    if r_corr is None or l_corr is None:
        raise ValueError(
            "_convert_single: smplx_corr_root is required for real MANO tip extraction; "
            "silent fallback to duplicated distal phalanx was removed in iter-4."
        )
    # smplx_out.vertices shape: (T, 10475, 3) for SMPL-X.
    vertices_all = smplx_out.vertices.detach().cpu().numpy().astype(np.float32)
    # MANO tip IDs → SMPL-X vertex IDs via correspondence.
    r_tip_smplx_ids = r_corr[_MANO_TIP_IDS_RIGHT]   # (5,)
    l_tip_smplx_ids = l_corr[_MANO_TIP_IDS_LEFT]    # (5,)
    right_tips = vertices_all[:, r_tip_smplx_ids, :]  # (T, 5, 3)
    left_tips = vertices_all[:, l_tip_smplx_ids, :]   # (T, 5, 3)

    # Concatenate: (T, 16, 3) skeletal + (T, 5, 3) tips = (T, 21, 3)
    left_hand_joints = np.concatenate([left_hand_16, left_tips], axis=1).astype(np.float32)
    right_hand_joints = np.concatenate([right_hand_16, right_tips], axis=1).astype(np.float32)
    # Shape: (T, 21, 3) float32

    # --- Extract hand axis-angle rotations (T, 16, 3) ---
    # Index 0: wrist (from global fullpose joint 20/21 as rotmat -> aa).
    # Indices 1..15: 15 finger joints (from sbj_p['left_hand_pose'] rotmats -> aa).

    # Wrist rotmats: extract from sbj_p['global_orient'] composed with parent chain
    # is complex. Instead, we use the wrist joint rotation directly from the 55-joint
    # fullpose, which is stored in sbj_p as the arm wrist joint (index 20/21 in
    # fullpose -> corresponding key in sbj_p).
    #
    # However sbj_p only has: global_orient(0), body_pose(1-21), jaw(22),
    # leye(23), reye(24), left_hand(25-39), right_hand(40-54).
    # The wrist rotation is joint 20 (left_wrist) and 21 (right_wrist) —
    # these are stored within body_pose as indices [19] and [20] respectively
    # (body_pose covers joints 1-21 -> indices 0-20 within body_pose array).
    #
    # body_pose[..., 19, :, :] = joint 20 (left wrist)
    # body_pose[..., 20, :, :] = joint 21 (right wrist)
    #
    # We extract these from sbj_p['body_pose'] as local rotmats, then convert to aa.

    if "body_pose" in sbj_p:
        bp_rotmat = _ensure_batch(sbj_p["body_pose"], 4).detach().cpu().numpy()  # (T, 21, 3, 3)
        # joint 20 (left wrist) = body_pose index 19
        # joint 21 (right wrist) = body_pose index 20
        left_wrist_rotmat = bp_rotmat[:, 19, :, :]   # (T, 3, 3) local
        right_wrist_rotmat = bp_rotmat[:, 20, :, :]  # (T, 3, 3) local
    else:
        # Fallback: identity (zero aa).
        left_wrist_rotmat = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))
        right_wrist_rotmat = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))

    # Convert wrist rotmats to axis-angle.
    left_wrist_aa = _rotmat_to_axis_angle(left_wrist_rotmat)    # (T, 3)
    right_wrist_aa = _rotmat_to_axis_angle(right_wrist_rotmat)  # (T, 3)

    # Convert 15 finger rotmats to axis-angle.
    left_finger_aa = _rotmat_to_axis_angle(
        left_hand_rotmat.reshape(T, 15, 3, 3)
    )   # (T, 15, 3)
    right_finger_aa = _rotmat_to_axis_angle(
        right_hand_rotmat.reshape(T, 15, 3, 3)
    )   # (T, 15, 3)

    # Concatenate: [wrist (T,1,3), fingers (T,15,3)] -> (T, 16, 3)
    left_hand_rot = np.concatenate(
        [left_wrist_aa[:, None, :], left_finger_aa], axis=1
    ).astype(np.float32)
    right_hand_rot = np.concatenate(
        [right_wrist_aa[:, None, :], right_finger_aa], axis=1
    ).astype(np.float32)

    # --- Object pose ---
    obj_transl_t = raw["obj_p"]["transl"]  # Tensor (T, 3)
    obj_orient_aa_t = raw["obj_p"]["global_orient"]  # Tensor (T, 3) axis-angle

    if isinstance(obj_transl_t, np.ndarray):
        obj_transl_np = obj_transl_t.astype(np.float32)
        obj_orient_aa_np = obj_orient_aa_t.astype(np.float32)
    else:
        obj_transl_np = obj_transl_t.detach().cpu().numpy().astype(np.float32)
        obj_orient_aa_np = obj_orient_aa_t.detach().cpu().numpy().astype(np.float32)

    obj_translation = obj_transl_np                      # (T, 3) float32
    obj_rotation = _aa_to_rotmat(obj_orient_aa_np)       # (T, 3, 3) float32

    # --- Valid mask (all frames valid — IMoS synthesizes exactly T frames) ---
    valid_mask = np.ones((T,), dtype=np.bool_)

    # --- Build CanonicalSample ---
    sample = CanonicalSample(
        left_hand_joints=left_hand_joints,    # (T, 21, 3) float32
        right_hand_joints=right_hand_joints,  # (T, 21, 3) float32
        left_hand_rot=left_hand_rot,          # (T, 16, 3) float32
        right_hand_rot=right_hand_rot,        # (T, 16, 3) float32
        obj_translation=obj_translation,      # (T, 3)     float32
        obj_rotation=obj_rotation,            # (T, 3, 3)  float32
        obj_name=obj_name_canonical,
        valid_mask=valid_mask,                # (T,)       bool
        fps=30,
        text=text,
    )

    validate_canonical_sample(sample, strict=True)
    return sample


# ---------------------------------------------------------------------------
# SHA256 file hash
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Git SHA
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return current HEAD SHA. Returns 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning(
            "_git_sha: git rev-parse HEAD returned code=%d stderr=%r",
            result.returncode, result.stderr.strip(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("_git_sha: git invocation failed: %s", exc)
    return "unknown"


# ---------------------------------------------------------------------------
# UTC timestamp
# ---------------------------------------------------------------------------

def _utc_timestamp() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Save canonical + extras
# ---------------------------------------------------------------------------

def _save_canonical(sample: CanonicalSample, path: str, strict: bool = True) -> None:
    """Validate and save CanonicalSample to a compressed .npz file.

    Args:
        sample: The sample to save.
        path:   Destination file path.
        strict: If True, re-validates orthonormality/det>0 of obj_rotation.
    """
    validate_canonical_sample(sample, strict=strict)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    np.savez_compressed(
        path,
        left_hand_joints=sample.left_hand_joints,
        right_hand_joints=sample.right_hand_joints,
        left_hand_rot=sample.left_hand_rot,
        right_hand_rot=sample.right_hand_rot,
        obj_translation=sample.obj_translation,
        obj_rotation=sample.obj_rotation,
        obj_name=np.array(sample.obj_name),
        valid_mask=sample.valid_mask,
        fps=np.array(int(sample.fps)),
        text=np.array(sample.text),
    )


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert(
    raw_dir: str,
    out_dir: str,
    smplx_model_root: str = "",
    smplx_corr_root: str = "",
    max_samples: Optional[int] = None,
) -> None:
    """Convert all IMoS raw files in raw_dir to canonical .npz format.

    Scans raw_dir for files saved by torch.save (accepts any extension, including
    .npz which is what IMoS uses). Runs SMPL-X forward kinematics to produce
    world-frame joint positions. Writes output to <out_dir>/samples/NNNNNN.npz.

    Also writes:
      - <out_dir>/MANIFEST.sha256
      - <out_dir>/config.json
      - <out_dir>/README.md

    Args:
        raw_dir:          Directory of IMoS torch-saved output files.
        out_dir:          Output directory. Created if missing.
        smplx_model_root: Path passed to smplx.SMPLXLayer as model_path.
                          Should point to the directory containing SMPL-X .npz
                          model files (e.g. .../smplx/models/).
        smplx_corr_root:  Path to smplx_correspondence directory (rhand_smplx_ids.npy
                          etc.). Currently unused in this adapter — kept for CLI
                          compatibility with future fingertip enhancement.
        max_samples:      Optional cap on the number of raw files to convert.

    Raises:
        FileNotFoundError: If raw_dir does not exist or contains no matching files.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"convert: raw_dir not found: {raw_dir!r}")

    # Collect raw files (torch.save outputs; IMoS uses .npz extension).
    raw_files = sorted(
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
        and not f.startswith(".")
        and not f.endswith(".json")
        and not f.endswith(".md")
        and not f.endswith(".sha256")
    )
    if not raw_files:
        raise FileNotFoundError(
            f"convert: no raw files found in raw_dir: {raw_dir!r}"
        )

    if max_samples is not None:
        raw_files = raw_files[:max_samples]

    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    manifest: dict = {}
    n_converted = 0
    n_failed = 0

    for i, raw_fname in enumerate(raw_files):
        raw_path = os.path.join(raw_dir, raw_fname)
        out_stem = f"{i:06d}"
        out_path = os.path.join(samples_dir, f"{out_stem}.npz")

        logger.info(
            "convert: [%d/%d] %s -> %s",
            i + 1, len(raw_files), raw_fname, out_stem
        )

        try:
            sample = _convert_single(raw_path, smplx_model_root=smplx_model_root,
                                     smplx_corr_root=smplx_corr_root)
        except Exception as exc:
            logger.warning(
                "convert: FAILED %s: %s: %s",
                raw_fname, type(exc).__name__, exc
            )
            n_failed += 1
            continue

        _save_canonical(sample, out_path, strict=True)
        manifest[f"samples/{out_stem}.npz"] = _sha256_file(out_path)
        n_converted += 1

    if n_converted == 0:
        raise RuntimeError(
            f"convert: all {len(raw_files)} files failed to convert. "
            "Check logs for per-file errors."
        )

    # --- Write manifest ---
    manifest_path = os.path.join(out_dir, "MANIFEST.sha256")
    with open(manifest_path, "w") as f:
        for stem, digest in sorted(manifest.items()):
            f.write(f"{digest}  {stem}\n")
    logger.info("convert: wrote MANIFEST.sha256 (%d entries)", len(manifest))

    # --- Write config.json ---
    commit_sha = _git_sha()
    timestamp = _utc_timestamp()
    config = {
        "adapter_version": 1,
        "source_raw_dir": os.path.abspath(raw_dir),
        "n_samples_converted": n_converted,
        "n_failed": n_failed,
        "commit_sha": commit_sha,
        "timestamp": timestamp,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("convert: wrote config.json")

    # --- Write README.md ---
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            "# Canonical Samples -- IMoS GRAB\n\n"
            "**DO NOT DELETE THIS DIRECTORY.** These canonical .npz files are the\n"
            "scored output for baseline reproduction. Deleting them requires re-running\n"
            "the full IMoS inference pipeline.\n\n"
            "## Provenance\n\n"
            f"- Source: `{os.path.abspath(raw_dir)}`\n"
            f"- Adapter version: 1\n"
            f"- Commit: `{commit_sha}`\n"
            f"- Timestamp: `{timestamp}`\n"
            f"- Samples converted: {n_converted}\n"
            f"- Samples failed: {n_failed}\n\n"
            "See `config.json` for full provenance and `MANIFEST.sha256` for file hashes.\n"
        )
    logger.info("convert: wrote README.md")

    print(
        f"[imos_to_canonical] DONE: {n_converted} samples -> {os.path.abspath(samples_dir)}"
        + (f" ({n_failed} failed)" if n_failed > 0 else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m umm.eval.handobject.baseline_adapters.imos_to_canonical",
        description=(
            "Convert IMoS raw torch-saved output to canonical schema. "
            "Writes <out-dir>/samples/NNNNNN.npz + MANIFEST.sha256 + config.json + README.md."
        ),
    )
    p.add_argument(
        "--raw-dir",
        required=True,
        help="Directory of IMoS raw files (torch.save format, typically .npz extension).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. Created if missing.",
    )
    p.add_argument(
        "--smplx-model-root",
        default="",
        help=(
            "Directory containing SMPL-X model .npz files. "
            "Passed as model_path to smplx.SMPLXLayer. "
            "Default: '' (will use smplx library's default search path)."
        ),
    )
    p.add_argument(
        "--smplx-corr-root",
        required=True,
        help=(
            "Directory containing smplx_correspondence files "
            "(rhand_smplx_ids.npy and lhand_smplx_ids.npy, both shape (778,)). "
            "Used to extract real MANO fingertip vertex positions from SMPL-X "
            "output vertices. Required — silent fallback to duplicated distal "
            "phalanx (lossy) is disabled."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on number of raw files to convert (for debugging).",
    )
    return p


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args()
    convert(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        smplx_model_root=args.smplx_model_root,
        smplx_corr_root=args.smplx_corr_root,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
