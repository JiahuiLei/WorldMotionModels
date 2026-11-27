"""M7 R1+R2+R3 inference protocol for OMOMO.

Three-round refinement on top of M6:

R1: identical to M6 R1 — synth keyframes (z = lerp(z[0], z[end]); rot = copy
    of obj rot at f0) at obj waypoints {29, 59, 89}; all 5 obj keyframes
    {0, 29, 59, 89, 119} marked KNOWN. Intermediate ones get a K%-noised
    clamp; f0 + f119 stay clean clamped (GT). Body and rest of obj denoise
    freely. Returns ret_trace_r1.

R2 (NEW vs M6): keep EVERYTHING else known and only regenerate the obj
    keyframes that were either synthetic in R1 or artificially clamped to
    GT beyond what CHOIS-strict allows — i.e. obj[{29, 59, 89, 119}]. f0
    stays GT-known (the single legitimate CHOIS-strict obj anchor). Body
    [0..119] (body[0] GT + body[1..119] from R1) and obj at non-keyframes
    {1..28, 30..58, 60..88, 90..118} (from R1) are all KNOWN. Plain
    inference, no hooks.  Returns ret_trace_r2.

R3 (NEW vs M6): keep obj fully KNOWN (R2's regenerated keyframes +
    R1's non-keyframe obj + GT f0). body[0] GT KNOWN, body[1..119]
    UNKNOWN — regenerate body conditioned on the cleaned obj trajectory.
    Plain inference, no hooks. Returns ret_trace_r3 (FINAL).

Default K_pct = 0 (M6 winner). Same R1 hooks/kwargs as M6 — see
``m6_runner.py`` docstring for details and the rationale of marking
intermediate keyframes KNOWN.

Goal: fix the residual keyframe rotation trembling observed in M6
output. M6 R2 leaves obj at non-keyframes from R1 KNOWN but regenerates
all 5 obj keyframes — when R1's f0-rotation-bias propagates to non-kf
neighbours, those neighbours become OOD anchors for R2 keyframe
regeneration. M7 R2 keeps the same gap pattern but only regenerates the
4 frames whose values cannot be trusted (synth or artificial); R3 then
re-derives a body trajectory consistent with the cleaned obj.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

OBJ_IDX_OMOMO = 22
KEYFRAMES_DEFAULT: Tuple[int, ...] = (0, 29, 59, 89, 119)
INTERMEDIATE_KF_DEFAULT: Tuple[int, ...] = (29, 59, 89)
# R2 unknown set: synth intermediates + artificially-clamped end frame.
# f0 is excluded — it is the single CHOIS-strict legitimate obj anchor.
R2_UNKNOWN_KF_DEFAULT: Tuple[int, ...] = (29, 59, 89, 119)


def run_m7_inference(
    agent,
    batch: Dict[str, torch.Tensor],
    *,
    seed: int,
    n_steps: Optional[int],
    scheduler_mode: str,
    obj_idx: int = OBJ_IDX_OMOMO,
    keyframes: Tuple[int, ...] = KEYFRAMES_DEFAULT,
    intermediate_kf: Tuple[int, ...] = INTERMEDIATE_KF_DEFAULT,
    r2_unknown_kf: Tuple[int, ...] = R2_UNKNOWN_KF_DEFAULT,
    K_pct: float = 0.0,
    r2_fuzzy_K_pct: Optional[float] = 3.0,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Three-round M7 inference. Returns final R3 ret_trace (B, T, P, 4, 4).

    Args mirror M6's ``run_m6_inference`` plus:
      * ``r2_unknown_kf`` (default ``(29, 59, 89, 119)``): the obj keyframes
        regenerated in R2.
      * ``r2_fuzzy_K_pct`` (default 3.0, since 2026-05-02): R2 fuzzy-clamp
        K%. body[1..119] (from R1) and obj[non_kf] (from R1) are clamped at
        q_K-noise level while DDPM step ``t > K_step``, then DROPPED from
        the known set for ``t <= K_step`` (denoise jointly with the
        unknowns). f0 anchor (body[0] + obj[0]) stays clean clamped
        throughout. obj at ``r2_unknown_kf`` are UNKNOWN throughout.
        Set to ``None`` (or 0) to disable and recover M7-v1 R2 behavior
        (clean clamp on all known positions).

        K=3 was selected from the K∈{1..50} sweep on w1.8.1.1 165k
        sub16_clothesstand_000_w0: K=3 hits the tightest f89 dx (2.1 cm
        vs M7-v1 3.7 cm) and f89 drot (9.5° vs M7-v1 16.8°). Cliff at K=4
        (mean stays flat but max-jump jumps from ~7.6 cm to ~11.4 cm).
    """
    B = batch["context"].shape[0]
    C = batch["context"].shape[1]
    T = batch["trace"].shape[1]
    P = batch["trace"].shape[2]
    device = batch["trace"].device

    # K_pct -> alpha (same as M6).
    num_train_timesteps = agent.diffusion.num_train_timesteps
    t_step = round(K_pct / 100.0 * num_train_timesteps)
    t_step = max(0, min(num_train_timesteps - 1, t_step))
    alpha_K = float(
        agent.diffusion.noise_scheduler.alphas_cumprod[t_step].item()
    )

    # =========== R1: identical to M6 R1 ============================
    trace_r1 = batch["trace"].clone()
    z0 = trace_r1[:, 0, obj_idx, 2, 3]
    zN = trace_r1[:, T - 1, obj_idx, 2, 3]
    rot0 = trace_r1[:, 0, obj_idx, :3, :3].clone()
    for kf in intermediate_kf:
        if not (0 <= kf < T):
            continue
        frac = float(kf) / float(T - 1)
        trace_r1[:, kf, obj_idx, 2, 3] = (1.0 - frac) * z0 + frac * zN
        trace_r1[:, kf, obj_idx, :3, :3] = rot0

    known_mask_r1 = batch["known_mask"].clone()
    for kf in keyframes:
        if 0 <= kf < T:
            known_mask_r1[:, kf, obj_idx] = True

    m7_clamp_mask = torch.zeros(B, C + T, P, dtype=torch.bool, device=device)
    for kf in intermediate_kf:
        if 0 <= kf < T:
            m7_clamp_mask[:, C + kf, obj_idx] = True

    with torch.amp.autocast("cuda", dtype=autocast_dtype):
        _, ret_trace_r1 = agent.inference(
            trace=trace_r1, context=batch["context"],
            trace_known_mask=known_mask_r1,
            context_known_mask=batch["context_known_mask"],
            trace_mask=batch["trace_mask"], context_mask=batch["context_mask"],
            real_timestep=batch["time"], group_id=batch["trace_group_id"],
            sem_type=batch["trace_sem_type"], rot_mask=batch["trace_rot_mask"],
            text=batch["text"], local_pcl=batch["local_pcl"],
            local_pcl_mask=batch["local_pcl_mask"],
            seed=seed, n_steps=n_steps, verbose=False,
            scheduler_mode=scheduler_mode,
            m6_known_clamp_alpha=alpha_K,
            m6_known_clamp_mask=m7_clamp_mask,
        )

    # =========== R2: regen ONLY obj at r2_unknown_kf ==================
    # Everything else is KNOWN: body[0..119] (body[0] GT + body[1..119]
    # from R1), obj[non_kf] from R1, obj[0] GT.
    trace_r2 = batch["trace"].clone()
    # Body[1..119] <- R1 (body[0] stays GT). All P except obj_idx.
    if P > 1:
        # Slice replacement is faster than per-(t, p) loop.
        for tt in range(1, T):
            trace_r2[:, tt, :obj_idx, :, :] = ret_trace_r1[:, tt, :obj_idx, :, :]
            if obj_idx + 1 < P:
                trace_r2[:, tt, obj_idx + 1:, :, :] = (
                    ret_trace_r1[:, tt, obj_idx + 1:, :, :]
                )
    # Obj non-keyframes <- R1 (obj[0] and obj[119] stay GT in trace).
    kf_set = set(keyframes)
    non_kf = [t for t in range(T) if t not in kf_set]
    for tt in non_kf:
        trace_r2[:, tt, obj_idx, :, :] = ret_trace_r1[:, tt, obj_idx, :, :]

    known_mask_r2 = batch["known_mask"].clone()
    # Body all known.
    if P > 1:
        if obj_idx > 0:
            known_mask_r2[:, :, :obj_idx] = True
        if obj_idx + 1 < P:
            known_mask_r2[:, :, obj_idx + 1:] = True
    # Obj non-keyframes known.
    for tt in non_kf:
        known_mask_r2[:, tt, obj_idx] = True
    # Obj f0 known (GT) — single legitimate CHOIS-strict anchor.
    known_mask_r2[:, 0, obj_idx] = True
    # Obj at r2_unknown_kf left UNKNOWN — model regenerates these.
    r2_unknown_set = set(r2_unknown_kf)
    for kf in r2_unknown_set:
        if 0 <= kf < T:
            known_mask_r2[:, kf, obj_idx] = False

    # Fuzzy-R2 hook (default ON at K=3%). Build fuzzy_mask covering
    # body[1..119] + obj[non_kf except f0]; pass alpha_K and release_t.
    # When r2_fuzzy_K_pct is None or <= 0 the kwargs default to no-op
    # (M7-v1 clean-clamp R2 behavior).
    _fuzzy_kwargs: Dict[str, object] = {}
    if r2_fuzzy_K_pct is not None and float(r2_fuzzy_K_pct) > 0.0:
        t_step_fuzzy = round(float(r2_fuzzy_K_pct) / 100.0 * num_train_timesteps)
        t_step_fuzzy = max(0, min(num_train_timesteps - 1, t_step_fuzzy))
        alpha_fuzzy = float(
            agent.diffusion.noise_scheduler.alphas_cumprod[t_step_fuzzy].item()
        )
        fuzzy_mask = torch.zeros(
            B, C + T, P, dtype=torch.bool, device=device,
        )
        # Body[1..T-1]: all P except obj_idx, frames 1..T-1.
        if obj_idx > 0:
            fuzzy_mask[:, C + 1: C + T, :obj_idx] = True
        if obj_idx + 1 < P:
            fuzzy_mask[:, C + 1: C + T, obj_idx + 1:] = True
        # Obj non-keyframes (skip f0; obj[0] is the clean anchor).
        for tt in non_kf:
            if tt == 0:
                continue
            fuzzy_mask[:, C + tt, obj_idx] = True
        _fuzzy_kwargs = dict(
            m6_known_clamp_alpha=alpha_fuzzy,
            m6_known_clamp_mask=fuzzy_mask,
            m7_release_t_int=int(t_step_fuzzy),
        )

    with torch.amp.autocast("cuda", dtype=autocast_dtype):
        _, ret_trace_r2 = agent.inference(
            trace=trace_r2, context=batch["context"],
            trace_known_mask=known_mask_r2,
            context_known_mask=batch["context_known_mask"],
            trace_mask=batch["trace_mask"], context_mask=batch["context_mask"],
            real_timestep=batch["time"], group_id=batch["trace_group_id"],
            sem_type=batch["trace_sem_type"], rot_mask=batch["trace_rot_mask"],
            text=batch["text"], local_pcl=batch["local_pcl"],
            local_pcl_mask=batch["local_pcl_mask"],
            seed=seed, n_steps=n_steps, verbose=False,
            scheduler_mode=scheduler_mode,
            **_fuzzy_kwargs,
        )

    # =========== R3: regen body[1..119] with full obj KNOWN ===========
    # Final obj trajectory: GT f0 (already in batch.trace) + R2's regen
    # at r2_unknown_kf + R1's obj at non-keyframes. Pulling the entire
    # obj column from ret_trace_r2 captures all three (R2 leaves the
    # known slots untouched at their input values).
    trace_r3 = batch["trace"].clone()
    trace_r3[:, :, obj_idx, :, :] = ret_trace_r2[:, :, obj_idx, :, :]
    # Body[1..119] inputs are irrelevant (will be unknown). Keep GT body[0].

    known_mask_r3 = batch["known_mask"].clone()
    # body[0] already True via batch.known_mask under chois_strict;
    # body[1..119] left as-is (False).
    # Obj fully known across all T frames.
    known_mask_r3[:, :, obj_idx] = True

    with torch.amp.autocast("cuda", dtype=autocast_dtype):
        _, ret_trace_r3 = agent.inference(
            trace=trace_r3, context=batch["context"],
            trace_known_mask=known_mask_r3,
            context_known_mask=batch["context_known_mask"],
            trace_mask=batch["trace_mask"], context_mask=batch["context_mask"],
            real_timestep=batch["time"], group_id=batch["trace_group_id"],
            sem_type=batch["trace_sem_type"], rot_mask=batch["trace_rot_mask"],
            text=batch["text"], local_pcl=batch["local_pcl"],
            local_pcl_mask=batch["local_pcl_mask"],
            seed=seed, n_steps=n_steps, verbose=False,
            scheduler_mode=scheduler_mode,
        )

    return ret_trace_r3
