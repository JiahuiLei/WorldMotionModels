"""Base visualization utilities for V10.

Provides shared inference caller and mask construction helpers used by
all domain-specific V10 visualizers.

V10 inference is simpler than V9: no local_pcl, no images, no DINO.
The model uses SIM(3) canonicalization + 6D rotation + flow matching/DDPM.
Conditioning is via known_mask (which tokens are observed).

Example::

    gen_ctx, gen_trace = run_v10_inference(
        eval_model, batch, bid=0,
        trace_known_mask=mask, context_known_mask=ctx_mask,
    )
"""

import logging
from typing import Optional

import torch

from umm.models.umm_v10.agent import AgentV10

logger = logging.getLogger(__name__)


def run_v10_inference(
    eval_model: AgentV10,
    batch: dict,
    bid: int,
    trace_known_mask: torch.Tensor,
    context_known_mask: torch.Tensor,
    seed: int = 42,
    n_steps: Optional[int] = None,
    verbose: bool = False,
    return_intermediates: bool = False,
):
    """Run V10 inference on a single batch item.

    Constructs the full inference call from batch + masks.

    Args:
        eval_model: AgentV10 (or EMA copy).
        batch: Full batch dict from V10 datamodule.
        bid: Batch item index.
        trace_known_mask: (1, T, P) bool — which trace tokens are known.
        context_known_mask: (1, C, P) bool — which context tokens are known.
        seed: Random seed for reproducibility.
        n_steps: Override number of denoising steps.
        verbose: Print progress bar.
        return_intermediates: If True, return dict with intermediate X0
            predictions at 25/50/75/100% of denoising steps.

    Returns:
        If return_intermediates=False: tuple (gen_context, gen_trace).
        If return_intermediates=True: dict with "context", "trace",
        "intermediates" keys.
    """
    result = eval_model.inference(
        trace=batch["trace"][bid:bid + 1],
        context=batch["context"][bid:bid + 1],
        trace_known_mask=trace_known_mask,
        context_known_mask=context_known_mask,
        trace_mask=batch["trace_mask"][bid:bid + 1],
        context_mask=batch["context_mask"][bid:bid + 1],
        real_timestep=batch["time"][bid:bid + 1],
        group_id=batch["trace_group_id"][bid:bid + 1],
        sem_type=batch["trace_sem_type"][bid:bid + 1],
        rot_mask=batch["trace_rot_mask"][bid:bid + 1],
        text=[batch["text"][bid]],
        seed=seed,
        n_steps=n_steps,
        verbose=verbose,
        return_intermediates=return_intermediates,
    )
    if return_intermediates:
        return result  # dict with context, trace, intermediates
    return result  # tuple (gen_context, gen_trace)


def make_entity_known_mask(
    trace_mask: torch.Tensor,
    sem_type: torch.Tensor,
    entity_sem_ids: list,
    invert: bool = False,
) -> torch.Tensor:
    """Construct known_mask for entity-based patterns (H2R, R2H, O2H, H2O).

    Args:
        trace_mask: (T, P) or (B, T, P) validity mask.
        sem_type: (P,) or (B, P) semantic type IDs.
        entity_sem_ids: List of semantic type IDs for the KNOWN entities.
        invert: If True, mark entity_sem_ids as UNKNOWN instead.

    Returns:
        known_mask: same shape as trace_mask, bool.
    """
    if sem_type.dim() == 1:
        # (P,) → entity mask
        entity_mask = torch.zeros(sem_type.shape[0], dtype=torch.bool,
                                  device=sem_type.device)
        for sid in entity_sem_ids:
            entity_mask |= (sem_type == sid)

        if invert:
            entity_mask = ~entity_mask

        # Expand to match trace_mask
        if trace_mask.dim() == 2:
            # (T, P) → known where entity AND valid
            known = trace_mask.clone().bool()
            known[:, ~entity_mask] = False
        else:
            # (B, T, P)
            known = trace_mask.clone().bool()
            known[:, :, ~entity_mask] = False
    else:
        # (B, P)
        B = sem_type.shape[0]
        known = trace_mask.clone().bool()
        for b in range(B):
            entity_mask = torch.zeros(sem_type.shape[1], dtype=torch.bool,
                                      device=sem_type.device)
            for sid in entity_sem_ids:
                entity_mask |= (sem_type[b] == sid)
            if invert:
                entity_mask = ~entity_mask
            known[b, :, ~entity_mask] = False

    return known
