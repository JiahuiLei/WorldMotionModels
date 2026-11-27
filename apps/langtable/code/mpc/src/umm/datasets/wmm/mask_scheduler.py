"""Mask scheduler for V10 training-time conditioning mask generation.

Generates ``known_mask`` (P, T) and ``context_known_mask`` (P, C) tensors
that determine which tokens are given as conditioning input vs. predicted
by the model during training.

Architecture:
    25% base random: sample N tokens from [2, total_valid], mark as known
    75% domain-specific: weighted sampling from dataset-declared patterns

Guarantees:
    - At least 2 known tokens (hard constraint for SIM(3) canonicalization)
    - All known positions must be valid: known_mask &= trace_mask
"""

import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch

from umm.datasets.wmm.dataset_config import (
    SEM_HUMAN,
    SEM_HUMAN_GLOBAL,
    SEM_HUMANOID,
    SEM_HUMANOID_GLOBAL,
    SEM_LEFT_MANO,
    SEM_LEFT_MANO_FINGERS,
    SEM_LEFT_MANO_WRIST,
    SEM_OBJECT,
    SEM_RIGHT_MANO,
    SEM_RIGHT_MANO_FINGERS,
    SEM_RIGHT_MANO_WRIST,
)

logger = logging.getLogger(__name__)

# Type alias for pattern functions
# Args: trace_mask (P, T), context_mask (P, C), sem_type (P,), **kwargs
# Returns: (known_mask (P, T), context_known_mask (P, C))
PatternFn = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


# ──────────────────────────────────────────────────────────────────────
# Temporal Patterns
# ──────────────────────────────────────────────────────────────────────

def half_future(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Known = first T/2 of valid trace. Context fully known."""
    P, T = trace_mask.shape
    known = trace_mask.clone().bool()
    known[:, T // 2:] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def random_prefix_future(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Known = first k frames where k ~ Uniform{1, ..., T-1}; rest unknown.

    Generalizes ``half_future`` (which is the k=T/2 special case) to
    arbitrary prefix lengths, so the model learns to continue from ANY
    seen-history length rather than only the halfway point. Context
    fully known.
    """
    P, T = trace_mask.shape
    if T <= 1:
        return trace_mask.clone().bool(), context_mask.clone().bool()
    k = int(torch.randint(1, T, (1,)).item())  # k in [1, T-1]
    known = trace_mask.clone().bool()
    known[:, k:] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def one_step_future(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Known = only t=0 for all valid entities. Context fully known."""
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    known[:, 0] = trace_mask[:, 0].bool()
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def infilling(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Known = first + last frame only. Context fully known."""
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    known[:, 0] = trace_mask[:, 0].bool()
    known[:, -1] = trace_mask[:, -1].bool()
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def unconditional(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All unknown — the >=2 known enforcement will kick in after."""
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


# ──────────────────────────────────────────────────────────────────────
# Entity Patterns (HH domain)
# ──────────────────────────────────────────────────────────────────────

def _is_human_sem(sem_type: torch.Tensor) -> torch.Tensor:
    """Return bool mask for human semantic types. Shape: (P,)."""
    return (
        (sem_type == SEM_HUMAN)
        | (sem_type == SEM_HUMAN_GLOBAL)
    )


def _is_robot_sem(sem_type: torch.Tensor) -> torch.Tensor:
    """Return bool mask for robot/humanoid semantic types. Shape: (P,)."""
    return (
        (sem_type == SEM_HUMANOID)
        | (sem_type == SEM_HUMANOID_GLOBAL)
    )


def h2r(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Human-to-Robot: human entities known, robot entities unknown."""
    known = trace_mask.clone().bool()
    robot_mask = _is_robot_sem(sem_type)  # (P,)
    known[robot_mask] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def r2h(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robot-to-Human: robot entities known, human entities unknown."""
    known = trace_mask.clone().bool()
    human_mask = _is_human_sem(sem_type)  # (P,)
    known[human_mask] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


# ──────────────────────────────────────────────────────────────────────
# Entity Patterns (Taco domain)
# ──────────────────────────────────────────────────────────────────────

def _is_hand_sem(sem_type: torch.Tensor) -> torch.Tensor:
    """Return bool mask for hand semantic types. Shape: (P,)."""
    return (
        (sem_type == SEM_LEFT_MANO)
        | (sem_type == SEM_RIGHT_MANO)
        | (sem_type == SEM_LEFT_MANO_WRIST)
        | (sem_type == SEM_LEFT_MANO_FINGERS)
        | (sem_type == SEM_RIGHT_MANO_WRIST)
        | (sem_type == SEM_RIGHT_MANO_FINGERS)
    )


def _is_object_sem(sem_type: torch.Tensor) -> torch.Tensor:
    """Return bool mask for object semantic types. Shape: (P,)."""
    return sem_type == SEM_OBJECT


def o2h(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Object-to-Hand: objects known, hands unknown."""
    known = trace_mask.clone().bool()
    hand_mask = _is_hand_sem(sem_type)  # (P,)
    known[hand_mask] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def h2o(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hand-to-Object: hands known, objects unknown."""
    known = trace_mask.clone().bool()
    obj_mask = _is_object_sem(sem_type)  # (P,)
    known[obj_mask] = False
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


# ──────────────────────────────────────────────────────────────────────
# Context Patterns (LangTable domain)
# ──────────────────────────────────────────────────────────────────────

def future_with_goal(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    current_ith=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Future prediction with goal context known.

    Semantics: "known" = all frames at or before the current frame (time <= 0).

    Args:
        trace_mask: (P, T) bool — valid trace tokens.
        context_mask: (P, C) bool — valid context tokens.
        current_ith: int or None. If provided, positions ``[0..current_ith]``
            inclusive are known. If None, fall back to legacy behavior
            (first 1-3 positions known, sampled randomly).

    Returns:
        Tuple of ``(known_mask, ctx_known_mask)`` where ``ctx_known_mask``
        is the full context (goal IS known).
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    if current_ith is None:
        # Legacy fallback: first 1-3 steps (sample randomly for diversity)
        n_known_steps = random.randint(1, max(1, min(3, T // 2)))
        for t in range(n_known_steps):
            known[:, t] = trace_mask[:, t].bool()
    else:
        known[:, :current_ith + 1] = trace_mask[:, :current_ith + 1].bool()
    ctx_known = context_mask.clone().bool()  # goal IS known
    return known, ctx_known


def future_without_goal(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    current_ith=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Future prediction WITHOUT goal context.

    Semantics: "known" = all frames at or before the current frame (time <= 0).

    Args:
        trace_mask: (P, T) bool — valid trace tokens.
        context_mask: (P, C) bool — valid context tokens.
        current_ith: int or None. If provided, positions ``[0..current_ith]``
            inclusive are known. If None, fall back to legacy behavior
            (first 1-3 positions known, sampled randomly).

    Returns:
        Tuple of ``(known_mask, ctx_known_mask)`` where ``ctx_known_mask``
        is fully zero (goal is NOT known).
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    if current_ith is None:
        # Legacy fallback: first 1-3 steps (sample randomly for diversity)
        n_known_steps = random.randint(1, max(1, min(3, T // 2)))
        for t in range(n_known_steps):
            known[:, t] = trace_mask[:, t].bool()
    else:
        known[:, :current_ith + 1] = trace_mask[:, :current_ith + 1].bool()
    # Context is NOT known
    ctx_known = torch.zeros_like(context_mask, dtype=torch.bool)
    return known, ctx_known


def cur_only(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    current_ith=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Only the current frame is known. Context fully known.

    Semantics: exactly one trace position — the current frame — is marked
    known. This is a narrower version of ``future_with_goal`` that does
    not reveal any past history.

    Args:
        trace_mask: (P, T) bool — valid trace tokens.
        context_mask: (P, C) bool — valid context tokens.
        current_ith: int or None. If provided, only position ``current_ith``
            is known. If None, fall back to legacy ``one_step_future``
            behavior (position 0 is known).

    Returns:
        Tuple of ``(known_mask, ctx_known_mask)`` where ``ctx_known_mask``
        is the full context (goal IS known).
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    if current_ith is None:
        known[:, 0] = trace_mask[:, 0].bool()
    else:
        known[:, current_ith] = trace_mask[:, current_ith].bool()
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


# ──────────────────────────────────────────────────────────────────────
# V13 LangTable variants — one_step_future / cur_only WITHOUT goal.
# Used only by LANGTABLE_PATTERNS. All other pattern sets (HH, Taco,
# SCENE, MP) continue to use the goal-known versions above.
#
# Motivation: in langtable training we want MOST samples to have an
# unknown goal context so the model learns to generate the goal from
# text. Only the `future_with_goal` pattern reveals the goal.
# ──────────────────────────────────────────────────────────────────────

def lt_one_step_future_no_ctx(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Known = only t=0 for all valid entities. Context UNKNOWN.

    LangTable-V13 variant of ``one_step_future`` that zeros
    ``ctx_known_mask`` so the model must learn to generate the goal.
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    known[:, 0] = trace_mask[:, 0].bool()
    ctx_known = torch.zeros_like(context_mask, dtype=torch.bool)
    return known, ctx_known


def lt_cur_only_no_ctx(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    current_ith=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Only the current frame is known. Context UNKNOWN.

    LangTable-V13 variant of ``cur_only`` that zeros ``ctx_known_mask``
    so the model must learn to generate the goal.
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    if current_ith is None:
        known[:, 0] = trace_mask[:, 0].bool()
    else:
        known[:, current_ith] = trace_mask[:, current_ith].bool()
    ctx_known = torch.zeros_like(context_mask, dtype=torch.bool)
    return known, ctx_known


# ──────────────────────────────────────────────────────────────────────
# Preset pattern sets
# ──────────────────────────────────────────────────────────────────────

# Each entry: (name, pattern_fn, weight)
HH_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("half_future", half_future, 2.0),
    ("one_step_future", one_step_future, 1.0),
    ("infilling", infilling, 1.0),
    ("h2r", h2r, 2.0),
    ("r2h", r2h, 2.0),
]

TACO_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("half_future", half_future, 2.0),
    ("one_step_future", one_step_future, 1.0),
    ("o2h", o2h, 2.0),
    ("h2o", h2o, 2.0),
]

LANGTABLE_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    # V13: only `future_with_goal` reveals the ctx (goal). The other three
    # hide it so the model is forced to learn goal generation from text.
    ("future_with_goal", future_with_goal, 1.0),
    ("future_without_goal", future_without_goal, 2.0),
    ("lt_one_step_future_no_ctx", lt_one_step_future_no_ctx, 0.2),
    ("lt_cur_only_no_ctx", lt_cur_only_no_ctx, 0.2),
]

# V13: ManiSkill pattern set. Mirrors the original v11 LANGTABLE_PATTERNS
# (BEFORE the v13 lt edit) so ManiSkill keeps the goal-context distribution
# it had in v11. Used by `maniskill_oracle_dataset.py`.
#
#   future_with_goal      (w=1.0): goal IS known    (~22% of all samples)
#   future_without_goal   (w=2.0): goal hidden       (~44%)
#   one_step_future       (w=0.2): goal IS known    (~4.4%)
#   cur_only              (w=0.2): goal IS known    (~4.4%)
# Plus base_random (25% of all samples) which always sets goal known.
# → ~56% of ms samples have ctx_known_mask = full at the dataset level.
MS_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("future_with_goal", future_with_goal, 1.0),
    ("future_without_goal", future_without_goal, 2.0),
    ("one_step_future", one_step_future, 0.2),
    ("cur_only", cur_only, 0.2),
]

# MS3 pattern set — ctx_known_mask is NEVER set to True. User order
# 2026-04-23: MS3 deployment inference must hallucinate the goal; therefore
# training must NEVER reveal the terminal success frame as a known anchor.
# All 3 entries use the `*_no_ctx` variants which zero ctx_known_mask.
# Paired with base_random_prob=0.0 in DEFAULT_MASK_CONFIGS["ms3_state"]
# (datamodule.py) so the `_base_random` path — which sets ctx_known ~50%
# of the time — is never invoked.
#
#   future_without_goal          (w=2.0): all frames at or before current known, ctx UNKNOWN.
#   lt_one_step_future_no_ctx    (w=1.0): only t=0 known, ctx UNKNOWN.
#   lt_cur_only_no_ctx           (w=0.5): only current frame known, ctx UNKNOWN.
MS3_NO_CTX_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("future_without_goal", future_without_goal, 2.0),
    ("lt_one_step_future_no_ctx", lt_one_step_future_no_ctx, 1.0),
    ("lt_cur_only_no_ctx", lt_cur_only_no_ctx, 0.5),
]

MP_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("half_future", half_future, 2.0),
    ("one_step_future", one_step_future, 1.0),
    ("infilling", infilling, 2.0),
]

SCENE_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("future_with_goal", future_with_goal, 2.0),
    ("future_without_goal", future_without_goal, 1.0),
    ("one_step_future", one_step_future, 1.0),
    ("cur_only", cur_only, 0.2),
]

HUMOTO_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("half_future", half_future, 2.0),
    ("one_step_future", one_step_future, 1.0),
    ("humoto_o2h", r2h, 2.0),    # objects known, human unknown (r2h masks SEM_HUMAN)
    ("humoto_h2o", h2o, 2.0),    # human known, objects unknown (h2o masks SEM_OBJECT)
]


# ──────────────────────────────────────────────────────────────────────
# OMOMO / CHOIS-aligned pattern
# ──────────────────────────────────────────────────────────────────────
# CHOIS conditions on: full object pose at frame 0 (start), full object pose
# at frame T-1 (end), and the XY component of N_waypoints sparse intermediate
# frames. The model predicts everything else (full body trajectory, full
# object trajectory between waypoints).
#
# Implementation note: we cannot mask sub-DOFs of an SE(3) token (the
# mask is per-token boolean). To approximate "XY-only known" for waypoint
# frames we mark those frames as known on the OBJECT token; the dataset
# already supplies the full SE(3) but the model conditions on it as a
# coarse XY-style waypoint. This matches CHOIS's sparse-waypoint protocol
# in spirit while staying inside the wmm token contract.

def chois_obj_waypoints_to_all(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    current_ith=None,
    waypoint_indices: Optional[List[int]] = None,
    n_waypoints: int = 4,
    reveal_body_t0: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CHOIS-style mask: object tokens revealed at caller-supplied waypoints.

    Object tokens reveal exactly the frames in ``waypoint_indices`` (or
    ``n_waypoints`` evenly-spaced frames including frame 0 and T-1 if
    ``waypoint_indices`` is empty). All other object frames and ALL human
    frames are unknown unless ``reveal_body_t0=True``. Context (text/ctx
    tokens) fully known.

    Args:
        trace_mask: (P, T) bool — valid trace tokens.
        context_mask: (P, C) bool — valid context tokens.
        sem_type: (P,) int — token semantic IDs.
        current_ith: ignored (signature kept for dispatcher uniformity).
        waypoint_indices: optional list of int frame indices supplied by
            the dataset (overrides ``n_waypoints``).
        n_waypoints: number of evenly-spaced intermediate waypoints if
            ``waypoint_indices`` is empty (default 4).
        reveal_body_t0: if True, reveal frame 0 for body tokens
            (SEM_HUMAN | SEM_HUMAN_GLOBAL) in addition to the object
            anchors. Mirrors CHOIS ``input_first_human_pose=True``
            (released script ``test_chois_single_window.sh:7``,
            applied at ``trainer_chois.py:1112-1114``).

    Returns:
        Tuple of ``(known_mask, ctx_known_mask)``.
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    obj_mask = _is_object_sem(sem_type)  # (P,)
    if obj_mask.any():
        # Choose frames: 0, T-1, plus waypoints
        if waypoint_indices:
            frames = sorted({int(w) for w in waypoint_indices if 0 <= int(w) < T})
        else:
            inner = []
            if n_waypoints > 0 and T > 2:
                # Even spacing in (0, T-1) exclusive
                step = T / (n_waypoints + 1)
                inner = [int(round((k + 1) * step)) for k in range(n_waypoints)]
            frames = sorted({0, T - 1, *[f for f in inner if 0 < f < T - 1]})
        # Build per-frame index tensor and reveal those object frames
        for t in frames:
            known[obj_mask, t] = trace_mask[obj_mask, t].bool()
    if reveal_body_t0:
        body_mask = _is_human_sem(sem_type)  # (P,) — SEM_HUMAN | SEM_HUMAN_GLOBAL
        if body_mask.any():
            known[body_mask, 0] = trace_mask[body_mask, 0].bool()
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


def chois_strict(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    sem_type: torch.Tensor,
    current_ith=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CHOIS-strict mask — initial state only (mirrors eval-time M1).

    Marks ONLY frame 0 as known on both obj and body (full SE(3) at the
    initial state); intermediate waypoints (29/59/89) and the end frame
    are LEFT UNKNOWN. The future-waypoint xy / xyz signal — which CHOIS
    legitimately conditions on — is delivered at inference time via the
    M1 (``chois_mode='replace'``) channel-replace sampler hook in
    ``AgentV11.inference()``. This keeps training and the CHOIS-faithful
    eval protocol aligned at the per-token-mask level.

    Compare to ``chois_obj_waypoints_to_all`` (legacy): that pattern
    leaks FULL SE(3) at all 5 anchors (z + 6D rot at 29/59/89/end),
    which is the long-standing eval bug fixed by the strict protocol.

    Body frame 0 is always revealed (same convention as CHOIS's released
    ``input_first_human_pose=True`` evaluator). Other arguments —
    ``waypoint_indices`` and ``reveal_body_t0`` — supplied by the caller
    via ``**kwargs`` are intentionally ignored; this pattern is by
    construction "frame-0 only".

    Args/Returns: same shape contract as :func:`chois_obj_waypoints_to_all`.
    """
    return chois_obj_waypoints_to_all(
        trace_mask=trace_mask,
        context_mask=context_mask,
        sem_type=sem_type,
        waypoint_indices=[0],
        reveal_body_t0=True,
    )


# OMOMO pattern mix (oriented to improve CHOIS Table 1 benchmark performance). Dominant weight (8.0) on chois_obj_waypoints_to_all
# to strictly mirror CHOIS Table-1 eval protocol (~60% of all training samples
# after 0.25 base_random share); supporting patterns add diversity so the
# model also generalizes to half-future/one-step and the body↔object swaps.
#
# IMPORTANT: OMOMO has NO MANO/hand tokens (just SMPL-X 22 body joints +
# 1 rigid object). The bare `o2h` pattern (which masks `_is_hand_sem`)
# would be a no-op here and silently mark every token as known. We mirror
# the HUMOTO convention instead:
#   - `omomo_h2o` reuses `h2o`  → masks SEM_OBJECT (body known, object hidden)
#   - `omomo_o2b` reuses `r2h` → masks SEM_HUMAN[+_GLOBAL] (object known, body hidden)
#
# w1.8.x family adds ``chois_strict`` at default weight 1.0 — older configs
# that do not list it in ``pattern_weights`` will pick up this new pattern at
# ~15% per-sample fire rate (post 25% base_random share). Users who want the
# old (no-strict) distribution must pin ``chois_strict: 0.0`` in their YAML
# or stay on the historical configs already on disk.
OMOMO_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("chois_obj_waypoints_to_all", chois_obj_waypoints_to_all, 8.0),
    ("chois_strict", chois_strict, 1.0),
    # random_prefix_future generalizes half_future: k ~ U{1..T-1} each
    # sample so the model learns continuation from ANY seen-history
    # length (including k=1 which replaces the old one_step_future).
    ("random_prefix_future", random_prefix_future, 1.0),
    ("omomo_h2o", h2o, 0.5),    # body known, object unknown
    ("omomo_o2b", r2h, 0.5),    # object known, body unknown
]


# ──────────────────────────────────────────────────────────────────────
# GRAB / handobject_new pattern — eval_keyframe + GRAB_PATTERNS
# ──────────────────────────────────────────────────────────────────────
# At inference time the canonical eval (eval/handobject_new/canonical_eval.py)
# only knows L hand + R hand + obj at t=49. The eval_keyframe pattern mimics
# this exactly. Dominant weight (4.0) so the model sees this regime in
# ~57% of training samples (at base_random_prob=0.0); the other three
# patterns (o2h/h2o/random_prefix_future) provide diversity.

def eval_keyframe(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    keyframe_t: int = 49,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reveal exactly one frame (default t=49) for ALL valid trajectories.

    Mimics the GRAB canonical-eval input contract: L hand + R hand + obj SE(3)
    known only at the grasp keyframe. Context fully known. If ``keyframe_t``
    is out of range or its column has no valid tokens, returns an all-zero
    known mask (the MaskScheduler ``_enforce_min_known`` step will then add
    >=2 valid tokens to satisfy the SIM(3) canonicalization constraint).

    Args:
        trace_mask: (P, T) bool — valid trace tokens.
        context_mask: (P, C) bool — valid context tokens.
        keyframe_t: integer frame index to reveal (default 49 per protocol).
        **kwargs: ignored (sem_type, current_ith, etc.).

    Returns:
        Tuple of ``(known_mask, ctx_known_mask)``.
    """
    P, T = trace_mask.shape
    known = torch.zeros(P, T, dtype=torch.bool)
    if 0 <= keyframe_t < T:
        known[:, keyframe_t] = trace_mask[:, keyframe_t].bool()
    ctx_known = context_mask.clone().bool()
    return known, ctx_known


GRAB_PATTERNS: List[Tuple[str, PatternFn, float]] = [
    ("eval_keyframe", eval_keyframe, 4.0),
    ("o2h", o2h, 1.0),
    ("h2o", h2o, 1.0),
    ("random_prefix_future", random_prefix_future, 1.0),
]


# ──────────────────────────────────────────────────────────────────────
# MaskScheduler
# ──────────────────────────────────────────────────────────────────────

class MaskScheduler:
    """Training-time mask generator.

    Generates ``known_mask`` and ``context_known_mask`` tensors that tell
    the model which tokens are known conditioning inputs.

    Args:
        base_random_prob: Probability of using base random mask (default 0.25).
        patterns: List of ``(name, pattern_fn, weight)`` tuples for
            domain-specific patterns. The remaining ``1 - base_random_prob``
            probability is split across these patterns by weight.

    Example::

        scheduler = MaskScheduler(
            base_random_prob=0.25,
            patterns=HH_PATTERNS,
        )
        known, ctx_known = scheduler.sample(trace_mask, context_mask, sem_type=sem)
    """

    def __init__(
        self,
        base_random_prob: float = 0.25,
        patterns: Optional[List[Tuple[str, PatternFn, float]]] = None,
    ):
        self.base_random_prob = base_random_prob
        if patterns is None:
            patterns = []
        self.pattern_names = [p[0] for p in patterns]
        self.pattern_fns = [p[1] for p in patterns]
        self.pattern_weights = [p[2] for p in patterns]
        total_w = sum(self.pattern_weights)
        if total_w > 0:
            self.pattern_probs = [w / total_w for w in self.pattern_weights]
        else:
            self.pattern_probs = []

        self._stats: Dict[str, int] = {"base_random": 0}
        for name in self.pattern_names:
            self._stats[name] = 0
        self._total_calls = 0
        self._last_pattern_name = "base_random"

    def sample(
        self,
        trace_mask: torch.Tensor,
        context_mask: torch.Tensor,
        sem_type: Optional[torch.Tensor] = None,
        current_ith: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate known_mask and context_known_mask for one training sample.

        Args:
            trace_mask: (P, T) bool — valid tokens.
            context_mask: (P, C) bool — valid context tokens.
            sem_type: (P,) int — semantic types (needed for entity patterns).
            current_ith: int or None. Relative index of the current frame
                inside the trace window. Passed through to patterns that
                need it (``future_with_goal``, ``future_without_goal``,
                ``cur_only``). Patterns that do not use it ignore via
                ``**kwargs``.
            **kwargs: Extra args passed to pattern functions.

        Returns:
            Tuple of ``(known_mask, context_known_mask)``:
            - known_mask: (P, T) bool
            - context_known_mask: (P, C) bool

        Guarantees:
            - known_mask &= trace_mask (all known are valid)
            - context_known_mask &= context_mask
            - At least 2 known tokens across trace (hard constraint)
        """
        self._total_calls += 1
        trace_mask = trace_mask.bool()
        context_mask = context_mask.bool()

        if random.random() < self.base_random_prob or len(self.pattern_fns) == 0:
            # Base random mask
            known_mask = self._base_random(trace_mask)
            ctx_known_mask = torch.rand_like(context_mask, dtype=torch.float) < 0.5
            self._stats["base_random"] += 1
            self._last_pattern_name = "base_random"
        else:
            # Domain-specific pattern
            idx = self._weighted_choice()
            pattern_fn = self.pattern_fns[idx]
            pattern_name = self.pattern_names[idx]
            known_mask, ctx_known_mask = pattern_fn(
                trace_mask=trace_mask,
                context_mask=context_mask,
                sem_type=sem_type,
                current_ith=current_ith,
                **kwargs,
            )
            self._stats[pattern_name] += 1
            self._last_pattern_name = pattern_name

        # Enforce: known must be valid
        known_mask = known_mask & trace_mask
        ctx_known_mask = ctx_known_mask & context_mask

        # Enforce: at least 2 known tokens in trace
        known_mask = self._enforce_min_known(known_mask, trace_mask, min_known=2)

        return known_mask, ctx_known_mask

    def _base_random(self, trace_mask: torch.Tensor) -> torch.Tensor:
        """Generate a base random known_mask.

        Randomly selects N tokens from [2, total_valid] valid positions.
        """
        P, T = trace_mask.shape
        valid_positions = trace_mask.nonzero(as_tuple=False)  # (N_valid, 2)
        n_valid = len(valid_positions)

        if n_valid <= 2:
            # All valid tokens are known
            return trace_mask.clone().bool()

        n_known = random.randint(2, n_valid)
        selected_indices = random.sample(range(n_valid), n_known)

        known_mask = torch.zeros(P, T, dtype=torch.bool)
        for i in selected_indices:
            p, t = valid_positions[i]
            known_mask[p, t] = True
        return known_mask

    def _enforce_min_known(
        self,
        known_mask: torch.Tensor,
        trace_mask: torch.Tensor,
        min_known: int = 2,
    ) -> torch.Tensor:
        """Ensure at least ``min_known`` tokens are known.

        If fewer than ``min_known`` tokens are known, randomly add valid tokens
        until the minimum is reached.
        """
        n_known = known_mask.sum().item()
        if n_known >= min_known:
            return known_mask

        # Find valid but not-yet-known positions
        candidates = (trace_mask & ~known_mask).nonzero(as_tuple=False)
        if len(candidates) == 0:
            # All valid positions already known
            return known_mask

        need = min_known - int(n_known)
        need = min(need, len(candidates))
        selected = random.sample(range(len(candidates)), need)
        for i in selected:
            p, t = candidates[i]
            known_mask[p, t] = True
        return known_mask

    def _weighted_choice(self) -> int:
        """Sample a pattern index by weight."""
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(self.pattern_probs):
            cumulative += prob
            if r < cumulative:
                return i
        return len(self.pattern_probs) - 1

    def get_stats(self) -> Dict[str, float]:
        """Return mask pattern distribution statistics.

        Returns:
            Dict mapping pattern names to their usage fraction.
        """
        if self._total_calls == 0:
            return {}
        return {
            name: count / self._total_calls
            for name, count in self._stats.items()
        }

    def print_stats(self):
        """Print mask pattern distribution statistics."""
        stats = self.get_stats()
        if not stats:
            logger.info("MaskScheduler: no samples yet")
            return
        logger.info(f"MaskScheduler stats ({self._total_calls} samples):")
        for name, frac in sorted(stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {name:25s}: {frac:6.1%} ({self._stats[name]:d})")

    def reset_stats(self):
        """Reset usage statistics counters."""
        for k in self._stats:
            self._stats[k] = 0
        self._total_calls = 0
