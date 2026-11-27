"""Unified diffusion schedulers for V11.

Supports three noise paradigms in one model:
1. **Flow Matching**: Linear interpolation x_t = (1-t)*x_0 + t*eps.
   Model always outputs x0 prediction, converted to velocity target internally.
2. **DDPM**: Standard DDPM forward process via diffusers.DDPMScheduler.
   Model always outputs x0 prediction, converted to eps/vel target internally.
3. **MeanFlow**: Average velocity with (r, t) conditioning.
   Model outputs native average velocity — NO converter (the sole exception).

Key V11 design decisions:
- x0net + converter: model ALWAYS outputs x0, converter produces target-space output
- Single source of truth for inference steps: gen_timesteps
- Loss scaling: linear (s_linear) not squared (s_squared)
- Noise level scheduler: per-sample branching (clean-known vs async)
- Known tokens get x_level=0 during inference (THE critical v10 bug fix)
- All operations use torch tensors (never python float for t.clamp)

Example::

    from umm.models.wmm.diffusion import build_diffusion_scheduler
    scheduler = build_diffusion_scheduler(cfg)
    noisy_x, t_levels, noise, target = scheduler.add_noise(x_0, valid_mask=mask)
"""

import logging
from typing import Optional, Tuple

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from umm.models.umm_v9.world_models.wm_utils import safe_sqrt

logger = logging.getLogger(__name__)


class FlowMatchingScheduler:
    """Flow matching noise scheduler.

    V11: model outputs x0. Converter produces velocity internally.
    Target returned is always x0 (the native output).

    Args:
        sync_noise_prob: Probability of syncing all noised tokens to a single
            randomly picked t. 0.0 = fully async (per-token independent t),
            1.0 = always sync. Values in between give a mixture. Ported from
            umm_v13.
    """

    def __init__(self, sync_noise_prob: float = 0.0):
        self.sync_noise_prob = float(sync_noise_prob)
        self.noise_type = "flow_matching"
        self._last_clean_known_branch = None  # V13.3.4: set by add_noise

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add FM noise with per-token async t and optional sync.

        Per-sample branching (from V11 spec):
          1. ALL slots → per-token random t from [0,1] (async)
          2. When known_mask provided: branch=Bernoulli(0.5) per sample;
             branch=1 → known slots get t=0, unknown → random
          3. BOTH branches: invalid slots → t=1.0 (always)
          4. Clean known tokens ARE included in loss

        Args:
            x_0: Clean data ``(B, N, D)``.
            noise: Optional pre-sampled noise.
            valid_mask: Optional ``(B, N)`` bool. Invalid → t=1.0.
            known_mask: Optional ``(B, N)`` bool. Known tokens.

        Returns:
            (noisy_x, t_levels, noise, target) where target = x_0 (native x0 output).
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample random t values — always per-token async
        t_random = torch.rand(B, N, device=device, dtype=x_0.dtype)

        t_levels = t_random.clone()

        # V13.3.4: reset branch flag (None when known_mask not provided)
        self._last_clean_known_branch = None

        # Per-sample 50/50 branching when known_mask is provided
        if known_mask is not None:
            branch = torch.rand(B, 1, device=device) < 0.5  # (B, 1) per sample
            # V13.3.4: store branch flag for conditional mask training
            self._last_clean_known_branch = branch
            # Branch 1 (clean-known): known→t=0, unknown→random
            clean_t = torch.zeros(B, N, device=device, dtype=x_0.dtype)
            clean_t = torch.where(known_mask, clean_t, t_random)
            # Branch 0 (async): ALL slots → random (already t_random)
            t_levels = torch.where(branch, clean_t, t_random)

        # Synced noise: probabilistically sync all noised tokens to one picked t.
        #
        # BUG FIX 2026-04-30: previously ``noised_mask = t_levels > 0`` was
        # used, but for FM this is essentially equivalent to "all valid"
        # (continuous t, prob of exact 0 ≈ 0); for DDPM (integer sampler)
        # the same predicate wrongly EXCLUDES tokens that randomly rolled
        # ``t_random == 0`` from the noised set, conflating "intentional clean
        # (known)" with "rolled clean by chance". That left ~1% of noised
        # tokens at t=0 instead of at the synced t.
        #
        # Correct semantics: a token is "noised" (i.e., in the prediction
        # set, should share the synced t) iff:
        #   - branch=clean-known (per-sample): valid AND not in known_mask
        #   - branch=async-everywhere:        valid (no semantic known)
        # ``self._last_clean_known_branch`` (set above) carries the per-sample
        # branch flag. Falls back to "all valid → noised" when known_mask is
        # not provided (legacy callers).
        if self.sync_noise_prob > 0.0:
            do_sync = torch.rand(B, 1, device=device) < self.sync_noise_prob
            valid_bool = (
                valid_mask
                if valid_mask is not None
                else torch.ones_like(t_levels, dtype=torch.bool)
            )
            clean_branch = self._last_clean_known_branch  # (B, 1) bool or None
            if known_mask is not None and clean_branch is not None:
                noised_clean = valid_bool & ~known_mask          # (B, N)
                noised_async = valid_bool                        # (B, N)
                noised_mask = torch.where(
                    clean_branch.expand_as(t_levels), noised_clean, noised_async
                )
            else:
                noised_mask = valid_bool
            rand_weights = torch.rand(B, N, device=device) * noised_mask.float()
            picked_idx = rand_weights.argmax(dim=1, keepdim=True)
            picked_t = t_levels.gather(1, picked_idx)
            synced_t = torch.where(noised_mask, picked_t.expand_as(t_levels), t_levels)
            t_levels = torch.where(do_sync, synced_t, t_levels)

        # Invalid → always max noise
        if valid_mask is not None:
            t_levels = torch.where(~valid_mask, torch.ones_like(t_levels), t_levels)

        t_expanded = t_levels.unsqueeze(-1)
        noisy_x = (1.0 - t_expanded) * x_0 + t_expanded * noise

        return noisy_x, t_levels, noise, x_0  # target is x0

    def x0_to_output(self, x0_pred: torch.Tensor, noisy_x: torch.Tensor,
                     t_levels: torch.Tensor) -> torch.Tensor:
        """Convert x0 prediction to velocity output for Euler inference.

        v = (x_0 - x_t) / t  (unstable near t=0)
        Instead: use direct x0 scheduling at inference.
        This method exists for training loss compatibility only.
        """
        # FM velocity: v = eps - x_0 = (x_t - x_0 * (1-t) - x_0 * t) / t
        # Actually: v = eps - x_0. But we have x0_pred, not eps.
        # From x_t = (1-t)*x_0 + t*eps → eps = (x_t - (1-t)*x0_pred)/t
        # v = eps - x_0 = (x_t - (1-t)*x0)/t - x0 = (x_t - x0)/t
        _EPS = 1e-4
        t = t_levels.unsqueeze(-1).clamp(min=_EPS)
        return (noisy_x - x0_pred) / t  # velocity


class DDPMWrapper:
    """DDPM scheduler with x0net design.

    V11: model outputs x0. Converter converts to the scheduler's prediction_type
    internally so DDIM scheduler.step() works correctly.

    Args:
        num_train_timesteps: Number of training timesteps.
        beta_schedule: Beta schedule.
        prediction_type: Internal DDPM prediction type for scheduler.step().
        gen_timesteps: Number of DDIM inference steps (single source of truth).
        sync_noise_prob: Probability of syncing all noised tokens to a single
            randomly picked t. 0.0 = fully async (default), 1.0 = always sync.
            Ported from umm_v13.
    """

    def __init__(
        self,
        num_train_timesteps: int = 200,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        gen_timesteps: int = 20,
        sync_noise_prob: float = 0.0,
    ):
        self.sync_noise_prob = float(sync_noise_prob)
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.noise_type = "ddpm"
        self._last_clean_known_branch = None  # V13.3.4: set by add_noise

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            timestep_spacing="trailing",
            clip_sample=False,
        )
        self.gen_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            timestep_spacing="trailing",
            clip_sample=False,
        )
        # Single source of truth for inference steps
        self.gen_scheduler.set_timesteps(gen_timesteps)
        # full-DDPM generation path used by the
        # CHOIS-Table-1 on-the-fly eval callback. Strictly additive — no existing
        # line changed. Accessed only via ``get_gen_scheduler("ddpm_1000")`` so
        # existing DDIM callers are byte-identical.
        self.ddpm_gen_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            timestep_spacing="trailing",
            clip_sample=False,
        )

    def get_gen_scheduler(self, mode: str):
        """Return the inference scheduler for ``mode`` (additive).

        * ``"ddim_gen"`` — existing DDIMScheduler at ``gen_timesteps`` (default).
        * ``"ddpm_full"`` — DDPMScheduler using all ``num_train_timesteps`` steps
          (preferred name; clearly communicates step count follows training config,
          NOT necessarily 1000 steps).
        * ``"ddpm_1000"`` — alias for ``"ddpm_full"`` (legacy; kept for backward
          compatibility — does NOT guarantee 1000 steps; actual count equals
          ``num_train_timesteps`` regardless of the name).
          Caller must pass ``n_steps=None`` to ``inference()`` to avoid a
          capping conflict with ``set_timesteps``.

        Raises ``ValueError`` on any other mode — fail loud per project rules.
        """
        if mode == "ddim_gen":
            return self.gen_scheduler
        if mode in ("ddpm_full", "ddpm_1000"):
            # Do NOT call set_timesteps — scheduler defaults to all
            # num_train_timesteps steps (set in __init__).
            # "ddpm_full" is the canonical name; "ddpm_1000" is a legacy alias.
            return self.ddpm_gen_scheduler
        raise ValueError(
            f"DDPMWrapper.get_gen_scheduler: unknown mode={mode!r}. "
            "Use 'ddim_gen' (default), 'ddpm_full', or 'ddpm_1000' (alias)."
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add DDPM noise with per-token async t and optional sync.

        Target is always x_0 since the model outputs x0 natively.
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample random t values — always per-token async
        t_random = torch.randint(
            0, self.num_train_timesteps, (B, N), device=device
        ).long()

        timesteps = t_random.clone()
        max_ts = self.num_train_timesteps - 1

        # V13.3.4: reset branch flag (None when known_mask not provided)
        self._last_clean_known_branch = None

        # Per-sample 50/50 branching when known_mask is provided
        if known_mask is not None:
            branch = torch.rand(B, 1, device=device) < 0.5  # (B, 1) per sample
            # V13.3.4: store branch flag for conditional mask training
            self._last_clean_known_branch = branch
            # Branch 1 (clean-known): known→t=0, unknown→random
            clean_t = torch.zeros(B, N, device=device, dtype=torch.long)
            clean_t = torch.where(known_mask, clean_t, t_random)
            # Branch 0 (async): ALL slots → random (already t_random)
            timesteps = torch.where(branch, clean_t, t_random)

        # Synced noise: probabilistically sync all noised tokens to one picked t.
        #
        # BUG FIX 2026-04-30: previously ``noised_mask = timesteps > 0`` was
        # used, but the integer DDPM sampler ``randint(0, num_train_timesteps)``
        # rolls exactly 0 for ~1/num_train_timesteps of tokens. Those random
        # zeros were wrongly EXCLUDED from sync (filter assumes t==0 means
        # "intentional clean / known"). Result: ~1% of noised tokens stayed
        # at t=0 instead of the synced t — visible as random clean drops in
        # the noise-level viz over a uniform synced background.
        #
        # Correct semantics: a token is "noised" (in the prediction set,
        # should share the synced t) iff:
        #   - branch=clean-known (per-sample): valid AND not in known_mask
        #   - branch=async-everywhere:        valid (no semantic known)
        # ``self._last_clean_known_branch`` (set above) carries the per-sample
        # branch flag. Falls back to "all valid → noised" when known_mask is
        # not provided (legacy callers).
        if self.sync_noise_prob > 0.0:
            do_sync = torch.rand(B, 1, device=device) < self.sync_noise_prob
            valid_bool = (
                valid_mask
                if valid_mask is not None
                else torch.ones_like(timesteps, dtype=torch.bool)
            )
            clean_branch = self._last_clean_known_branch  # (B, 1) bool or None
            if known_mask is not None and clean_branch is not None:
                noised_clean = valid_bool & ~known_mask          # (B, N)
                noised_async = valid_bool                        # (B, N)
                noised_mask = torch.where(
                    clean_branch.expand_as(timesteps), noised_clean, noised_async
                )
            else:
                noised_mask = valid_bool
            rand_weights = torch.rand(B, N, device=device) * noised_mask.float()
            picked_idx = rand_weights.argmax(dim=1, keepdim=True)
            picked_t = timesteps.gather(1, picked_idx)
            synced_t = torch.where(noised_mask, picked_t.expand_as(timesteps), timesteps)
            timesteps = torch.where(do_sync, synced_t, timesteps)

        # Invalid → always max noise
        if valid_mask is not None:
            timesteps = torch.where(~valid_mask, torch.full_like(timesteps, max_ts), timesteps)

        noisy_x = self.noise_scheduler.add_noise(
            original_samples=x_0.reshape(B * N, D),
            noise=noise.reshape(B * N, D),
            timesteps=timesteps.reshape(B * N),
        ).reshape(B, N, D)

        return noisy_x, timesteps.float(), noise, x_0  # target is x0

    def x0_to_output(self, x0_pred: torch.Tensor, noisy_x: torch.Tensor,
                     t_levels: torch.Tensor) -> torch.Tensor:
        """Convert x0 prediction to the scheduler's prediction_type output.

        This converter runs INSIDE the model so the output matches what
        scheduler.step() expects.
        """
        _EPS = 1e-6
        device = x0_pred.device
        with torch.autocast(device_type=device.type, enabled=False):
            x0_f = x0_pred.float()
            noisy_f = noisy_x.float()
            ts_flat = t_levels.reshape(-1).long()
            alphas_cp = self.noise_scheduler.alphas_cumprod.to(device).float()
            alpha_t = alphas_cp[ts_flat]
            sqrt_alpha = safe_sqrt(alpha_t).reshape_as(x0_f)
            sqrt_1m_alpha = safe_sqrt(1.0 - alpha_t).reshape_as(x0_f).clamp(min=_EPS)

            if self.prediction_type == "epsilon":
                # x0 → eps: eps = (x_t - sqrt(a)*x0) / sqrt(1-a)
                out = (noisy_f - sqrt_alpha * x0_f) / sqrt_1m_alpha
            elif self.prediction_type == "v_prediction":
                # x0 → v: v = sqrt(a)*x_t - sqrt(1-a)*x0) / sqrt(1-a)
                out = (sqrt_alpha * noisy_f - x0_f) / sqrt_1m_alpha
            elif self.prediction_type == "sample":
                out = x0_f  # already x0
            else:
                raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        return out.to(x0_pred.dtype)

    def compute_snr_weights(
        self,
        timesteps: torch.Tensor,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """Compute min-SNR-gamma loss weights."""
        device = timesteps.device
        _EPS = 1e-6

        with torch.autocast(device_type=device.type, enabled=False):
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device).float()
            alpha_t = alphas_cumprod[timesteps.long()]
            snr = alpha_t / (1.0 - alpha_t).clamp(min=_EPS)
            snr = snr.clamp(min=_EPS, max=1e6)
            weights = torch.clamp_max(snr, gamma) / snr

        return weights.to(timesteps.device)


class MeanFlowScheduler:
    """MeanFlow scheduler — WIP/DEFERRED in V11.

    DO NOT USE for training or inference until properly implemented.
    Missing pieces:
    - r_levels passed to LMM for interval-length conditioning
    - Correct avg-velocity loss space
    - Inference path (ODE solver for mean velocity)

    The forward process IS implemented (same as FM with (r,t) sub-interval),
    but the agent forward/inference paths raise NotImplementedError.

    MeanFlow is the sole exception: model outputs average velocity directly,
    no x0net converter. The forward process is the same as FM but with
    (r, t) two-timestep conditioning.

    Args:
        r_neq_t_ratio: Fraction of samples where r != t.
        sync_noise_prob: Probability of syncing all noised tokens to a single
            randomly picked t. 0.0 = fully async (default), 1.0 = always sync.
            Ported from umm_v13. Synced t propagates to r through the existing
            r_levels = t_levels.clone() path.
    """

    def __init__(
        self,
        r_neq_t_ratio: float = 0.25,
        sync_noise_prob: float = 0.0,
    ):
        self.r_neq_t_ratio = r_neq_t_ratio
        self.sync_noise_prob = float(sync_noise_prob)
        self._last_clean_known_branch = None  # V13.3.4: set by add_noise
        self.noise_type = "meanflow"

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add MeanFlow noise with per-token async t and optional sync.

        Returns (noisy_x, t_levels, r_levels, noise, u_target).
        """
        B, N, D = x_0.shape
        device = x_0.device
        dtype = x_0.dtype

        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample random t values — always per-token async
        t_random = torch.rand(B, N, device=device, dtype=dtype)

        t_levels = t_random.clone()

        # V13.3.4: reset branch flag (None when known_mask not provided)
        self._last_clean_known_branch = None

        # Per-sample 50/50 branching when known_mask is provided
        if known_mask is not None:
            branch = torch.rand(B, 1, device=device) < 0.5  # (B, 1) per sample
            # V13.3.4: store branch flag for conditional mask training
            self._last_clean_known_branch = branch
            # Branch 1 (clean-known): known→t=0, unknown→random
            clean_t = torch.zeros(B, N, device=device, dtype=dtype)
            clean_t = torch.where(known_mask, clean_t, t_random)
            # Branch 0 (async): ALL slots → random
            t_levels = torch.where(branch, clean_t, t_random)

        # Synced noise: probabilistically sync all noised tokens to one picked t.
        # Applied BEFORE r_levels derivation so synced t propagates correctly
        # into r_levels=t_levels.clone() below.
        #
        # BUG FIX 2026-04-30: see FlowMatchingScheduler.add_noise / DDPMWrapper
        # .add_noise comments above. ``noised_mask = t_levels > 0`` wrongly
        # excludes random t==0 rolls. Use known_mask + branch flag instead.
        if self.sync_noise_prob > 0.0:
            do_sync = torch.rand(B, 1, device=device) < self.sync_noise_prob
            valid_bool = (
                valid_mask
                if valid_mask is not None
                else torch.ones_like(t_levels, dtype=torch.bool)
            )
            clean_branch = self._last_clean_known_branch  # (B, 1) bool or None
            if known_mask is not None and clean_branch is not None:
                noised_clean = valid_bool & ~known_mask          # (B, N)
                noised_async = valid_bool                        # (B, N)
                noised_mask = torch.where(
                    clean_branch.expand_as(t_levels), noised_clean, noised_async
                )
            else:
                noised_mask = valid_bool
            rand_weights = torch.rand(B, N, device=device) * noised_mask.float()
            picked_idx = rand_weights.argmax(dim=1, keepdim=True)
            picked_t = t_levels.gather(1, picked_idx)
            synced_t = torch.where(noised_mask, picked_t.expand_as(t_levels), t_levels)
            t_levels = torch.where(do_sync, synced_t, t_levels)

        # Compute r levels (sub-interval for mean velocity)
        r_neq_t_mask = torch.rand(B, N, device=device) < self.r_neq_t_ratio
        r_levels = t_levels.clone()
        if r_neq_t_mask.any():
            r_rand = torch.rand(B, N, device=device, dtype=dtype)
            r_levels = torch.where(r_neq_t_mask, r_rand * t_levels, r_levels)

        # Invalid → always max noise
        if valid_mask is not None:
            t_levels = torch.where(~valid_mask, torch.ones_like(t_levels), t_levels)
            r_levels = torch.where(~valid_mask, torch.zeros_like(r_levels), r_levels)

        t_expanded = t_levels.unsqueeze(-1)
        noisy_x = (1.0 - t_expanded) * x_0 + t_expanded * noise

        r_expanded = r_levels.unsqueeze(-1)
        z_r = (1.0 - r_expanded) * x_0 + r_expanded * noise

        dt = (t_levels - r_levels).clamp(min=1e-6)
        u_target = (noisy_x - z_r) / dt.unsqueeze(-1)
        u_target = u_target.float()

        return noisy_x, t_levels, r_levels, noise, u_target

    # MeanFlow has NO converter — model outputs average velocity natively


def compute_x0(
    scheduler,
    pred: torch.Tensor,
    time_steps: torch.Tensor,
    noisy_x: torch.Tensor,
) -> torch.Tensor:
    """Compute x0 from DDPM model output — fp16-safe.

    Used only for DDPM when computing x0 from scheduler-format predictions
    (eps/vel) for metrics. In x0net mode, this is not needed — the model
    already outputs x0.
    """
    if pred.shape != noisy_x.shape:
        raise ValueError(
            f"pred and noisy_x must have same shape. Got {pred.shape} and {noisy_x.shape}"
        )
    if pred.shape[:-1] != time_steps.shape:
        raise ValueError(
            f"pred/time_steps shape mismatch. pred: {pred.shape}, time_steps: {time_steps.shape}"
        )

    device = pred.device
    original_shape = pred.shape
    _EPS = 1e-6

    with torch.autocast(device_type=device.type, enabled=False):
        pred_flat = pred.float().reshape(-1, pred.shape[-1])
        noisy_flat = noisy_x.float().reshape(-1, pred.shape[-1])
        ts_flat = time_steps.reshape(-1)

        alphas_cumprod = scheduler.alphas_cumprod.to(device).float()
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

        if prediction_type == "epsilon":
            sqrt_alpha = safe_sqrt(alphas_cumprod[ts_flat]).unsqueeze(-1)
            sqrt_1m_alpha = safe_sqrt(1.0 - alphas_cumprod[ts_flat]).unsqueeze(-1)
            x0_flat = (noisy_flat - sqrt_1m_alpha * pred_flat) / sqrt_alpha.clamp(min=_EPS)
        elif prediction_type == "sample":
            x0_flat = pred_flat
        elif prediction_type == "v_prediction":
            sqrt_alpha = safe_sqrt(alphas_cumprod[ts_flat]).unsqueeze(-1)
            sqrt_1m_alpha = safe_sqrt(1.0 - alphas_cumprod[ts_flat]).unsqueeze(-1)
            x0_flat = sqrt_alpha * noisy_flat - sqrt_1m_alpha * pred_flat
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    return x0_flat.to(pred.dtype).reshape(original_shape)


def build_diffusion_scheduler(cfg: dict):
    """Build a diffusion scheduler from config.

    Args:
        cfg: Dict with keys:
            - ``noise_type``: ``"flow_matching"`` (default), ``"ddpm"``, or ``"meanflow"``
            - ``num_train_timesteps``: int (DDPM, default 200)
            - ``beta_schedule``: str (DDPM, default "squaredcos_cap_v2")
            - ``prediction_type``: str (DDPM, default "epsilon")
            - ``gen_timesteps``: int (default 20)
            - ``r_neq_t_ratio``: float (MeanFlow, default 0.25)

    Returns:
        Scheduler instance.
    """
    noise_type = cfg.get("noise_type", "flow_matching")
    sync_noise_prob = float(cfg.get("sync_noise_prob", 0.0))

    _BETA_ALIASES = {"cosine": "squaredcos_cap_v2"}

    if noise_type == "flow_matching":
        scheduler = FlowMatchingScheduler(sync_noise_prob=sync_noise_prob)
        logger.info(
            f"V11 FM scheduler: x0net mode, sync_noise_prob={sync_noise_prob}"
        )
        return scheduler
    elif noise_type == "meanflow":
        r_neq_t_ratio = cfg.get("r_neq_t_ratio", 0.25)
        scheduler = MeanFlowScheduler(
            r_neq_t_ratio=r_neq_t_ratio,
            sync_noise_prob=sync_noise_prob,
        )
        logger.info(
            f"V11 MeanFlow scheduler: r_neq_t_ratio={r_neq_t_ratio}, "
            f"sync_noise_prob={sync_noise_prob}, native avg-vel output"
        )
        return scheduler
    elif noise_type == "ddpm":
        raw_beta = cfg.get("beta_schedule", "squaredcos_cap_v2")
        beta_schedule = _BETA_ALIASES.get(raw_beta, raw_beta)
        gen_timesteps = cfg.get("gen_timesteps", 20)
        scheduler = DDPMWrapper(
            num_train_timesteps=cfg.get("num_train_timesteps", 200),
            beta_schedule=beta_schedule,
            prediction_type=cfg.get("prediction_type", "epsilon"),
            gen_timesteps=gen_timesteps,
            sync_noise_prob=sync_noise_prob,
        )
        logger.info(
            f"V11 DDPM scheduler: timesteps={scheduler.num_train_timesteps}, "
            f"pred={scheduler.prediction_type}, gen_steps={gen_timesteps}, "
            f"sync_noise_prob={sync_noise_prob}, x0net mode"
        )
        return scheduler
    else:
        raise ValueError(
            f"Unknown noise_type: {noise_type}. "
            "Use 'flow_matching', 'ddpm', or 'meanflow'."
        )
