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

from umm.datasets.umm_v10.dataset_config import (
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
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Future prediction with goal context known. First few steps known."""
    P, T = trace_mask.shape
    # known = first 1-3 steps (sample randomly for diversity)
    n_known_steps = random.randint(1, max(1, min(3, T // 2)))
    known = torch.zeros(P, T, dtype=torch.bool)
    for t in range(n_known_steps):
        known[:, t] = trace_mask[:, t].bool()
    ctx_known = context_mask.clone().bool()  # goal IS known
    return known, ctx_known


def future_without_goal(
    trace_mask: torch.Tensor,
    context_mask: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Future prediction WITHOUT goal context. First few steps known."""
    P, T = trace_mask.shape
    n_known_steps = random.randint(1, max(1, min(3, T // 2)))
    known = torch.zeros(P, T, dtype=torch.bool)
    for t in range(n_known_steps):
        known[:, t] = trace_mask[:, t].bool()
    # Context is NOT known
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
    ("future_with_goal", future_with_goal, 1.0),
    ("future_without_goal", future_without_goal, 1.0),
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

    def sample(
        self,
        trace_mask: torch.Tensor,
        context_mask: torch.Tensor,
        sem_type: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate known_mask and context_known_mask for one training sample.

        Args:
            trace_mask: (P, T) bool — valid tokens.
            context_mask: (P, C) bool — valid context tokens.
            sem_type: (P,) int — semantic types (needed for entity patterns).
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
            ctx_known_mask = context_mask.clone()
            self._stats["base_random"] += 1
        else:
            # Domain-specific pattern
            idx = self._weighted_choice()
            pattern_fn = self.pattern_fns[idx]
            pattern_name = self.pattern_names[idx]
            known_mask, ctx_known_mask = pattern_fn(
                trace_mask=trace_mask,
                context_mask=context_mask,
                sem_type=sem_type,
                **kwargs,
            )
            self._stats[pattern_name] += 1

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
