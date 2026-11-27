"""Motion feature extraction from CanonicalSample for the hand-object classifier.

The hand-object-adapted IMoS classifier (see `classifier.py`) takes raw 3D
per-frame points and performs 2D projections (XY, XZ) inside its forward
method to mirror IMoS action_rec.py:51-52. This module prepares the
`(B, T, 43, 3)` points tensor + per-sample length:

    43 points per frame = 21 left-hand joints + 21 right-hand joints
                          + 1 object translation center.

Canonical schema (see canonical.py) uses meters, world frame, float32.

Sequence handling:
    - Fixed-length padding to MAX_T (default 120).
    - If T > MAX_T: uniform subsample indices np.linspace(0, T-1, MAX_T).
    - If T <= MAX_T: right-pad with zeros, report original length.

References:
    - IMoS action_rec.py:14-82 (two-stream XY/XZ LSTM design).
    - assets/data_audit.md (episode counts; T distribution per dataset).
"""

from typing import Optional, Tuple

import numpy as np

from umm.eval.handobject.canonical import CanonicalSample


# Default max sequence length. Covers the expected majority of GRAB+ARCTIC
# clips; longer episodes are uniformly sub-sampled to MAX_T frames.
MAX_T: int = 120


def canonical_to_points(sample: CanonicalSample) -> np.ndarray:
    """Build the (T, 43, 3) point array from a CanonicalSample.

    Layout along the point axis:
        [0:21]   left_hand_joints   (T, 21, 3)
        [21:42]  right_hand_joints  (T, 21, 3)
        [42:43]  obj_translation    (T, 1, 3)  (added center point)

    Args:
        sample: A validated CanonicalSample (call validate_canonical_sample first if unsure).

    Returns:
        np.ndarray of shape (T, 43, 3), float32. World-frame meters, one frame
        per row. Padded tail frames (per sample.valid_mask) are preserved as-is —
        the caller is responsible for masking or subsampling.

    Raises:
        ValueError: If any of the three input arrays has an unexpected shape.
    """
    lhj = np.asarray(sample.left_hand_joints)   # (T, 21, 3) float32
    rhj = np.asarray(sample.right_hand_joints)  # (T, 21, 3) float32
    ot  = np.asarray(sample.obj_translation)    # (T, 3)     float32

    if lhj.ndim != 3 or lhj.shape[1:] != (21, 3):
        raise ValueError(
            f"canonical_to_points: left_hand_joints expected (T, 21, 3), got {lhj.shape}"
        )
    if rhj.ndim != 3 or rhj.shape[1:] != (21, 3):
        raise ValueError(
            f"canonical_to_points: right_hand_joints expected (T, 21, 3), got {rhj.shape}"
        )
    if ot.ndim != 2 or ot.shape[1] != 3:
        raise ValueError(
            f"canonical_to_points: obj_translation expected (T, 3), got {ot.shape}"
        )
    if not (lhj.shape[0] == rhj.shape[0] == ot.shape[0]):
        raise ValueError(
            f"canonical_to_points: T mismatch — lhj={lhj.shape[0]}, "
            f"rhj={rhj.shape[0]}, obj_trans={ot.shape[0]}"
        )

    T = lhj.shape[0]
    obj_center = ot.reshape(T, 1, 3)                             # (T, 1, 3) — treat object center as 1 "joint"
    points = np.concatenate([lhj, rhj, obj_center], axis=1)      # (T, 43, 3)
    return points.astype(np.float32, copy=False)                 # (T, 43, 3) float32


def pad_or_subsample(
    points: np.ndarray, max_t: int = MAX_T
) -> Tuple[np.ndarray, int]:
    """Pad or uniform-subsample a (T, 43, 3) point array to (max_t, 43, 3).

    If T >  max_t: uniformly sub-sample to max_t frames via np.linspace indices;
                   the reported length is max_t.
    If T <= max_t: zero-pad the tail; the reported length is the original T.

    Args:
        points: (T, 43, 3) float32.
        max_t:  Target fixed sequence length.

    Returns:
        (points_padded, length):
            points_padded: (max_t, 43, 3) float32.
            length:        int in [1, max_t] — actual valid frames (for LSTM indexing).

    Raises:
        ValueError: If points has unexpected shape or T is zero or max_t <= 0.
    """
    if points.ndim != 3 or points.shape[1:] != (43, 3):
        raise ValueError(
            f"pad_or_subsample: expected (T, 43, 3), got {points.shape}"
        )
    if max_t <= 0:
        raise ValueError(f"pad_or_subsample: max_t must be positive, got {max_t}")
    T = points.shape[0]
    if T <= 0:
        raise ValueError(f"pad_or_subsample: input has T={T}; must be positive")

    if T > max_t:
        # Uniform sub-sampling. np.linspace gives endpoints T[0] and T[T-1] inclusive.
        idx = np.linspace(0, T - 1, max_t).round().astype(np.int64)  # (max_t,) int64
        out = points[idx]                                            # (max_t, 43, 3)
        length = max_t
        return out.astype(np.float32, copy=False), length

    # T <= max_t: pad tail with zeros.
    out = np.zeros((max_t, 43, 3), dtype=np.float32)                 # (max_t, 43, 3)
    out[:T] = points                                                 # first T frames
    return out, int(T)


def canonical_to_tensor(
    sample: CanonicalSample, max_t: int = MAX_T
) -> Tuple[np.ndarray, int]:
    """Convenience: canonical_to_points + pad_or_subsample, respecting valid_mask.

    Only frames with `sample.valid_mask[t] == True` are kept before padding/subsampling.
    If all frames are invalid, raises ValueError (fail-loud).

    Args:
        sample: CanonicalSample (validated upstream).
        max_t:  Target fixed length.

    Returns:
        (points_padded, length) as in pad_or_subsample.

    Raises:
        ValueError: If valid_mask has zero True entries.
    """
    vm = np.asarray(sample.valid_mask)                               # (T_padded,) bool
    if vm.ndim != 1:
        raise ValueError(
            f"canonical_to_tensor: valid_mask expected 1D, got shape {vm.shape}"
        )
    if not vm.any():
        raise ValueError(
            "canonical_to_tensor: sample has no valid frames (valid_mask all False)"
        )

    points = canonical_to_points(sample)                             # (T_padded, 43, 3)
    # Keep only valid frames (strips Text2HOI zero-padded tails).
    points_valid = points[vm]                                        # (T_valid, 43, 3)
    return pad_or_subsample(points_valid, max_t=max_t)


# ---------------------------------------------------------------------------
# Text2HOI-style classifier input — (num_frames, 42, 3)
# ---------------------------------------------------------------------------

def _select_from_groups_indices(V: int, num_groups: int, rng: "np.random.Generator") -> np.ndarray:
    """Select one frame index per group to mirror Text2HOI's train-mode sampling.

    Implements Text2HOI proc_output.py:select_from_groups (test=False branch):
        ave_duration, remainder = divmod(V, num_groups)
        offset1 = arange(remainder) * (ave_duration + 1) + randint(0, ave_duration+1, size=remainder)
        offset2 = arange(remainder, num_groups) * ave_duration + remainder
                  + randint(0, ave_duration, size=num_groups-remainder)
        final_offsets = concat([offset1, offset2])

    If ave_duration == 0 (V < num_groups), the randint(0, ave_duration) call in offset2 would be
    randint(0, 0) which is undefined. In that case we fall back to linspace (deterministic).
    If V == 0, raises ValueError.

    Args:
        V:          Number of valid frames (>0).
        num_groups: Number of output indices (= num_frames, typically 15).
        rng:        np.random.Generator for reproducible random sampling.

    Returns:
        (num_groups,) int64 array of frame indices in [0, V-1].

    Raises:
        ValueError: If V == 0.
    """
    if V == 0:
        raise ValueError("_select_from_groups_indices: V must be >0, got 0")

    ave_duration, remainder = divmod(V, num_groups)

    # If ave_duration == 0, cannot sample without replacement within groups.
    # Fall back to linspace TRUNCATION — matches the deterministic=True path
    # (Text2HOI proc_output.py:22 uses `astype(np.int32)` which is floor-cast).
    # Using `.round()` here would silently produce different indices from the
    # deterministic branch for the same short clip, which the reviewer flagged.
    if ave_duration == 0:
        return np.linspace(0, V - 1, num_groups).astype(np.int64)

    # Groups with an extra frame (first `remainder` groups get ave_duration+1 frames).
    offset1 = (
        np.arange(remainder) * (ave_duration + 1)
        + rng.integers(0, ave_duration + 1, size=remainder)
    )
    # Remaining groups (each gets ave_duration frames).
    offset2 = (
        np.arange(remainder, num_groups) * ave_duration
        + remainder
        + rng.integers(0, ave_duration, size=num_groups - remainder)
    )
    return np.concatenate([offset1, offset2]).astype(np.int64)


def canonical_to_text2hoi_input(
    sample: CanonicalSample,
    num_frames: int = 15,
    deterministic: bool = True,
    rng: "Optional[np.random.Generator]" = None,
) -> np.ndarray:
    """Build Text2HOI-exact classifier input (num_frames, 42, 3) from a CanonicalSample.

    Mirrors /tmp/Text2HOI/lib/utils/proc_output.py:get_action_classifier_input exactly:
      1. Select `num_frames` indices from valid frames:
          - deterministic=True:  np.linspace(0, V-1, num_frames).round().astype(int)
            (Text2HOI test=True path, proc_output.py:22)
          - deterministic=False: random within-group (Text2HOI train mode; see
            _select_from_groups_indices above).
      2. Extract 21 left-hand + 21 right-hand joints from the selected frames.
      3. Wrist-centering (Text2HOI proc_output.py:86-96):
            if is_lhand: hand_org = lhand_joints[:, 0, 0].clone()
            if is_rhand: hand_org = rhand_joints[:, 0, 0].clone()
         Both branches execute in sequence, so for a bimanual episode the
         SECOND assignment wins — `hand_org` = first-frame right wrist. For
         left-only episodes (`is_rhand=False`), `hand_org` = first-frame
         left wrist.
      4. Per-hand gate (Text2HOI proc_output.py:98-106):
            if is_lhand: lhand_joints_ra = lhand_joints - hand_org
            else:        lhand_joints_ra = zeros_like(lhand_joints)
            if is_rhand: rhand_joints_ra = rhand_joints - hand_org
            else:        rhand_joints_ra = zeros_like(rhand_joints)
      5. Concat: np.concatenate([left_ra, right_ra], axis=1) -> (num_frames, 42, 3).
      6. Return float32.

    propagation: CanonicalSample now carries `is_lhand`/`is_rhand` bool
    fields. Text2HOI raw GRAB/ARCTIC samples include unimanual episodes
    (is_lhand=False OR is_rhand=False) where the idle hand's joints still
    contain body-rest pose data. Without per-hand gating, those body-rest
    verts flow into the classifier as if they were active contact motion,
    biasing accuracy / FID / diversity / multimodality.

    Args:
        sample:        Validated CanonicalSample.
        num_frames:    Default 15 (Text2HOI constant).
        deterministic: True = test/eval mode (linspace), False = train mode (within-group).
        rng:           np.random.Generator for deterministic=False. Required when
                       deterministic=False; ignored when deterministic=True.

    Returns:
        (num_frames, 42, 3) float32.

    Raises:
        ValueError: If valid_mask has zero True entries.
        ValueError: If deterministic=False and rng is None.
    """
    if not deterministic and rng is None:
        raise ValueError(
            "canonical_to_text2hoi_input: rng must be provided when deterministic=False"
        )

    vm = np.asarray(sample.valid_mask)                               # (T_padded,) bool
    if vm.ndim != 1:
        raise ValueError(
            f"canonical_to_text2hoi_input: valid_mask expected 1D, got shape {vm.shape}"
        )
    if not vm.any():
        raise ValueError(
            "canonical_to_text2hoi_input: sample has no valid frames (valid_mask all False)"
        )

    lhj = np.asarray(sample.left_hand_joints)                        # (T_padded, 21, 3)
    rhj = np.asarray(sample.right_hand_joints)                        # (T_padded, 21, 3)

    if lhj.ndim != 3 or lhj.shape[1:] != (21, 3):
        raise ValueError(
            f"canonical_to_text2hoi_input: left_hand_joints expected (T, 21, 3), "
            f"got {lhj.shape}"
        )
    if rhj.ndim != 3 or rhj.shape[1:] != (21, 3):
        raise ValueError(
            f"canonical_to_text2hoi_input: right_hand_joints expected (T, 21, 3), "
            f"got {rhj.shape}"
        )

    # Keep only valid frames.
    lhj_valid = lhj[vm]                                               # (V, 21, 3)
    rhj_valid = rhj[vm]                                               # (V, 21, 3)
    V = lhj_valid.shape[0]

    # --- Frame selection ---
    if deterministic:
        # Text2HOI test=True path (proc_output.py:22) — floor-cast via astype(int32).
        # We use int64 indexing but match the TRUNCATION (not rounding) so the
        # selected indices are bit-exact to Text2HOI.
        idx = np.linspace(0, V - 1, num_frames).astype(np.int64)   # (num_frames,)
    else:
        idx = _select_from_groups_indices(V, num_groups=num_frames, rng=rng)  # (num_frames,)

    lhj_sel = lhj_valid[idx]                                          # (num_frames, 21, 3)
    rhj_sel = rhj_valid[idx]                                          # (num_frames, 21, 3)

    # read per-hand flags from the canonical sample. Legacy samples
    # without the field default to (True, True) via CanonicalSample defaults,
    # which matches the historical "both hands always" assumption.
    is_lhand = bool(sample.is_lhand)
    is_rhand = bool(sample.is_rhand)
    if not (is_lhand or is_rhand):
        raise ValueError(
            "canonical_to_text2hoi_input: sample has is_lhand=False and "
            "is_rhand=False — at least one hand must be active."
        )

    # --- Wrist-centering (Text2HOI proc_output.py:86-96) ---
    # Text2HOI runs BOTH conditionals in sequence:
    #     if is_lhand: hand_org = lhand_joints[:, 0, 0].clone()
    #     if is_rhand: hand_org = rhand_joints[:, 0, 0].clone()
    # Right-hand assignment wins when both are set. Left-only when is_rhand=False.
    if is_rhand:
        hand_org = rhj_sel[0, 0].copy()  # (3,) — bimanual or right-only
    else:
        hand_org = lhj_sel[0, 0].copy()  # (3,) — left-only (is_lhand=True, is_rhand=False)

    # --- Per-hand gate (Text2HOI proc_output.py:98-106) ---
    if is_lhand:
        lhj_ra = lhj_sel - hand_org                                   # (num_frames, 21, 3)
    else:
        lhj_ra = np.zeros_like(lhj_sel)                               # idle left → zeros
    if is_rhand:
        rhj_ra = rhj_sel - hand_org                                   # (num_frames, 21, 3)
    else:
        rhj_ra = np.zeros_like(rhj_sel)                               # idle right → zeros

    # --- Concat left + right -> (num_frames, 42, 3) ---
    out = np.concatenate([lhj_ra, rhj_ra], axis=1)                    # (num_frames, 42, 3)
    return out.astype(np.float32, copy=False)
