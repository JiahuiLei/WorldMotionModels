"""Flow matching and DDPM noise schedules for V10.

Supports two noise paradigms:

1. **Flow matching** (default): Linear interpolation ``x_t = (1-t)*x_0 + t*eps``.
   Velocity prediction target ``v = eps - x_0``. Euler ODE integration for inference.
   Async noise: each token gets independent ``t_i ~ U(0,1)``.
   Sync noise: all tokens share ``t ~ U(0,1)``.

2. **DDPM**: Standard DDPM forward process via ``diffusers.DDPMScheduler``.
   DDIM sampling for inference.

Example::

    fm = FlowMatchingScheduler(async_noise=True)
    noisy_x, t_levels, noise, v_target = fm.add_noise(x_0)
    loss = (v_pred - v_target).pow(2).mean()  # per-element

    # Inference
    x_hat = fm.sample(model_fn, shape, device, n_steps=50)
"""

import logging
from typing import Optional, Tuple

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

logger = logging.getLogger(__name__)


class FlowMatchingScheduler:
    """Flow matching noise scheduler with async/sync noise support.

    Args:
        async_noise: If True (default), each token gets an independent
            ``t_i ~ U(0,1)``. If False, all tokens share one ``t``.
    """

    def __init__(self, async_noise: bool = True):
        self.async_noise = async_noise

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add flow matching noise to clean data.

        Forward process: ``x_t = (1 - t) * x_0 + t * eps``.
        Velocity target: ``v = eps - x_0``.

        Args:
            x_0: Clean data, shape ``(B, N, D)`` where N = (C+T)*P or any
                flattened token count.
            noise: Optional pre-sampled noise, same shape as x_0. If None,
                sampled from N(0, I).
            valid_mask: Optional ``(B, N)`` bool. Invalid tokens get t=1.0
                (fully noised).

        Returns:
            Tuple of:
            - noisy_x: ``(B, N, D)`` noised samples.
            - t_levels: ``(B, N)`` noise levels in [0, 1].
            - noise: ``(B, N, D)`` the noise used.
            - v_target: ``(B, N, D)`` velocity targets (eps - x_0).
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample time levels
        if self.async_noise:
            t_levels = torch.rand(B, N, device=device, dtype=x_0.dtype)  # (B, N)
        else:
            # Sync: one t per sample, broadcast to all tokens
            t_per_sample = torch.rand(B, 1, device=device, dtype=x_0.dtype)
            t_levels = t_per_sample.expand(B, N)  # (B, N)

        # Invalid tokens → fully noised (t=1)
        if valid_mask is not None:
            t_levels = t_levels.clone()
            t_levels[~valid_mask] = 1.0

        # x_t = (1 - t) * x_0 + t * eps
        t_expanded = t_levels.unsqueeze(-1)  # (B, N, 1)
        noisy_x = (1.0 - t_expanded) * x_0 + t_expanded * noise  # (B, N, D)

        # Velocity target: v = eps - x_0
        v_target = noise - x_0  # (B, N, D)

        return noisy_x, t_levels, noise, v_target

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        x_init: torch.Tensor,
        n_steps: int = 50,
        known_mask: Optional[torch.Tensor] = None,
        known_x: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Euler ODE integration from t=1 to t=0.

        At each step, known tokens are clamped to their clean values.

        Args:
            model_fn: Callable ``(x_t, t_levels, **kwargs) -> v_pred``.
                Takes noisy input and time levels, returns predicted velocity.
            x_init: Starting point (pure noise), shape ``(B, N, D)``.
            n_steps: Number of Euler steps. Default 50.
            known_mask: ``(B, N)`` bool. Known tokens are clamped each step.
            known_x: ``(B, N, D)`` clean values for known tokens.
            **model_kwargs: Additional kwargs passed to model_fn.

        Returns:
            Denoised output, shape ``(B, N, D)``.
        """
        B, N, D = x_init.shape
        device = x_init.device
        dt = 1.0 / n_steps

        x_t = x_init.clone()

        for step_idx in range(n_steps):
            # Current time: goes from 1.0 down to near 0.0
            t = 1.0 - step_idx * dt
            t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)

            # Clamp known tokens
            if known_mask is not None and known_x is not None:
                # Known tokens at time t: interpolate between x_0 and noise
                # But at inference we just clamp to x_0 (clean)
                x_t = torch.where(
                    known_mask.unsqueeze(-1).expand_as(x_t),
                    known_x,
                    x_t,
                )

            # Model predicts velocity
            v_pred = model_fn(x_t, t_levels, **model_kwargs)

            # Euler step: x_{t-dt} = x_t - dt * v_pred
            # v = dx/dt = eps - x_0, and we integrate backward (t→0)
            x_t = x_t - dt * v_pred

        # Final clamping of known tokens
        if known_mask is not None and known_x is not None:
            x_t = torch.where(
                known_mask.unsqueeze(-1).expand_as(x_t),
                known_x,
                x_t,
            )

        return x_t


class DDPMWrapper:
    """Wrapper around diffusers DDPM/DDIM schedulers for V10.

    Provides the same interface as FlowMatchingScheduler but using DDPM
    forward process and DDIM inference.

    Args:
        num_train_timesteps: Number of diffusion timesteps. Default 100.
        beta_schedule: Beta schedule type. Default "squaredcos_cap_v2".
        prediction_type: "epsilon", "sample", or "v_prediction".
        gen_timesteps: Number of DDIM inference steps. Default 25.
        async_noise: If True, per-token timesteps. If False, per-sample.
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        gen_timesteps: int = 25,
        async_noise: bool = True,
    ):
        self.async_noise = async_noise
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps

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
        self.gen_scheduler.set_timesteps(gen_timesteps)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add DDPM noise.

        Args:
            x_0: Clean data, shape ``(B, N, D)``.
            noise: Optional pre-sampled noise.
            valid_mask: ``(B, N)`` bool. Invalid → max timestep.

        Returns:
            Tuple of (noisy_x, timesteps, noise, target):
            - noisy_x: ``(B, N, D)``
            - timesteps: ``(B, N)`` int
            - noise: ``(B, N, D)``
            - target: ``(B, N, D)`` — depends on prediction_type
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample timesteps
        if self.async_noise:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (B, N), device=device
            ).long()
        else:
            t_per_sample = torch.randint(
                0, self.num_train_timesteps, (B, 1), device=device
            ).long()
            timesteps = t_per_sample.expand(B, N).clone()

        # Invalid → max timestep
        max_ts = self.num_train_timesteps - 1
        if valid_mask is not None:
            timesteps[~valid_mask] = max_ts

        # Apply noise
        noisy_x = self.noise_scheduler.add_noise(
            original_samples=x_0.reshape(B * N, D),
            noise=noise.reshape(B * N, D),
            timesteps=timesteps.reshape(B * N),
        ).reshape(B, N, D)

        # Compute target
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x_0
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                x_0.reshape(B * N, D),
                noise.reshape(B * N, D),
                timesteps.reshape(B * N),
            ).reshape(B, N, D)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        return noisy_x, timesteps, noise, target

    def compute_snr_weights(
        self,
        timesteps: torch.Tensor,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """Compute min-SNR-gamma loss weights (Hang et al. 2023).

        Formula: weight = min(SNR(t), gamma) / SNR(t)
        where SNR(t) = alpha_t / (1 - alpha_t).

        Forces fp32 for numerical safety — SNR(t) → inf at t=0, → 0 at t=T.

        Args:
            timesteps: Integer timesteps, shape ``(...)``.
            gamma: Min-SNR clipping threshold. Default 5.0.

        Returns:
            Weights, shape ``(...)`` in [0, 1].
        """
        device = timesteps.device
        _EPS = 1e-6

        with torch.autocast(device_type=device.type, enabled=False):
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device).float()
            alpha_t = alphas_cumprod[timesteps.long()]  # (...)

            # SNR = alpha_t / (1 - alpha_t), clamped for fp16 safety
            snr = alpha_t / (1.0 - alpha_t).clamp(min=_EPS)  # (...)
            snr = snr.clamp(min=_EPS, max=1e6)  # avoid inf

            # min-SNR-gamma weight
            weights = torch.clamp_max(snr, gamma) / snr  # (...)

        return weights.to(timesteps.device)

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        x_init: torch.Tensor,
        n_steps: Optional[int] = None,
        known_mask: Optional[torch.Tensor] = None,
        known_x: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """DDIM sampling.

        Args:
            model_fn: ``(x_t, timesteps, **kwargs) -> pred``.
            x_init: Starting noise, ``(B, N, D)``.
            n_steps: Ignored (uses gen_scheduler timesteps).
            known_mask: ``(B, N)`` bool for clamping.
            known_x: ``(B, N, D)`` clean values for known tokens.
            **model_kwargs: Passed to model_fn.

        Returns:
            Denoised output, ``(B, N, D)``.
        """
        B, N, D = x_init.shape
        device = x_init.device

        x_t = x_init.clone()

        for t in self.gen_scheduler.timesteps:
            t_tensor = torch.full((B, N), int(t), device=device, dtype=torch.long)

            # Clamp known tokens
            if known_mask is not None and known_x is not None:
                x_t = torch.where(
                    known_mask.unsqueeze(-1).expand_as(x_t),
                    known_x,
                    x_t,
                )

            pred = model_fn(x_t, t_tensor, **model_kwargs)

            # Step — apply per-element since each token may have different t
            # For simplicity with DDIM (which expects scalar t), we use uniform t
            prev_x = self.gen_scheduler.step(
                pred.reshape(B * N, D), int(t), x_t.reshape(B * N, D)
            ).prev_sample.reshape(B, N, D)

            # Only update unknown tokens
            if known_mask is not None:
                x_t = torch.where(
                    known_mask.unsqueeze(-1).expand_as(x_t),
                    x_t,
                    prev_x,
                )
            else:
                x_t = prev_x

        # Final clamping
        if known_mask is not None and known_x is not None:
            x_t = torch.where(
                known_mask.unsqueeze(-1).expand_as(x_t),
                known_x,
                x_t,
            )

        return x_t


def compute_x0(
    scheduler,
    pred: torch.Tensor,
    time_steps: torch.Tensor,
    noisy_x: torch.Tensor,
) -> torch.Tensor:
    """Compute x0 from diffusion model output — fp16-safe version.

    Forces fp32 computation to avoid division-by-zero when sqrt(alpha_t)→0
    at high timesteps (epsilon mode) or sqrt(1-alpha_t)→0 at low timesteps.

    Args:
        scheduler: DDPMScheduler or DDIMScheduler with ``alphas_cumprod``
            and ``config.prediction_type``.
        pred: Model predictions, shape ``(..., D)``.
        time_steps: Integer timesteps, shape ``(...)``.
        noisy_x: Noisy samples, shape ``(..., D)``.

    Returns:
        x0_pred: Predicted original samples, shape ``(..., D)``.
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
    _EPS = 1e-6  # fp16-safe minimum (1e-8 rounds to 0 in fp16)

    # Force fp32 for numerical safety — sqrt/division unsafe in fp16
    with torch.autocast(device_type=device.type, enabled=False):
        pred_flat = pred.float().reshape(-1, pred.shape[-1])        # (N, D)
        noisy_flat = noisy_x.float().reshape(-1, noisy_x.shape[-1])  # (N, D)
        ts_flat = time_steps.reshape(-1)                             # (N,)

        alphas_cumprod = scheduler.alphas_cumprod.to(device).float()
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

        if prediction_type == "epsilon":
            # x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            sqrt_alpha = torch.sqrt(alphas_cumprod[ts_flat]).unsqueeze(-1)     # (N, 1)
            sqrt_1m_alpha = torch.sqrt(1.0 - alphas_cumprod[ts_flat]).unsqueeze(-1)
            x0_flat = (noisy_flat - sqrt_1m_alpha * pred_flat) / sqrt_alpha.clamp(min=_EPS)

        elif prediction_type == "sample":
            x0_flat = pred_flat

        elif prediction_type == "v_prediction":
            # v = sqrt(alpha_t)*eps - sqrt(1-alpha_t)*x0
            # x0 = sqrt(alpha_t)*x_t - sqrt(1-alpha_t)*v
            sqrt_alpha = torch.sqrt(alphas_cumprod[ts_flat]).unsqueeze(-1)
            sqrt_1m_alpha = torch.sqrt(1.0 - alphas_cumprod[ts_flat]).unsqueeze(-1)
            x0_flat = sqrt_alpha * noisy_flat - sqrt_1m_alpha * pred_flat

        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    return x0_flat.to(pred.dtype).reshape(original_shape)


class MeanFlowScheduler:
    """MeanFlow scheduler — average velocity with (r, t) conditioning.

    Implements the MeanFlow framework (Geng et al. 2025, arxiv 2505.13447).

    MeanFlow models **average velocity** u(z_t, r, t) instead of instantaneous
    velocity v(z_t, t) used in Flow Matching. The MeanFlow Identity relates them:

        u(z_t, r, t) = v(z_t, t) - (t - r) * du/dt

    Full MeanFlow training requires computing du/dt via JVP (Jacobian-vector
    product) through the network, which is incompatible with torch.compile.

    This implementation provides the **FM-equivalent variant** (no JVP):
    - Same forward process as FM: z_t = (1-t)*x_0 + t*eps
    - Same velocity target: v = eps - x_0 (conditional velocity is constant
      along straight-line paths, so average vel = instantaneous vel)
    - Also samples r levels for future JVP support
    - Model is conditioned on t and dt = t - r (two time variables)

    When r = t for all samples, this reduces exactly to FlowMatchingScheduler.
    The r != t samples don't change the target without JVP but provide correct
    scheduling infrastructure for future JVP implementation.

    Args:
        async_noise: Per-token independent t (default True).
        r_neq_t_ratio: Fraction of samples where r != t. Default 0.25
            (paper's ablation shows 25% is optimal).
    """

    def __init__(self, async_noise: bool = True, r_neq_t_ratio: float = 0.25):
        self.async_noise = async_noise
        self.r_neq_t_ratio = r_neq_t_ratio

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add MeanFlow noise — samples both r and t levels.

        Forward process: ``z_t = (1 - t) * x_0 + t * eps`` (same as FM).

        For r != t fraction: r is sampled uniformly in [0, t].
        For r == t fraction: r = t (reduces to standard FM).

        Args:
            x_0: Clean data ``(B, N, D)``.
            noise: Optional pre-sampled noise.
            valid_mask: Optional ``(B, N)`` bool. Invalid → t=1, r=0.

        Returns:
            Tuple of:
            - noisy_x: ``(B, N, D)`` noised samples at time t.
            - t_levels: ``(B, N)`` end times in [0, 1].
            - r_levels: ``(B, N)`` start times in [0, t].
            - noise: ``(B, N, D)`` the noise used.
            - u_target: ``(B, N, D)`` average velocity targets.
        """
        B, N, D = x_0.shape
        device = x_0.device
        dtype = x_0.dtype

        if noise is None:
            noise = torch.randn_like(x_0)

        # --- Sample t levels ---
        if self.async_noise:
            t_levels = torch.rand(B, N, device=device, dtype=dtype)
        else:
            t_per_sample = torch.rand(B, 1, device=device, dtype=dtype)
            t_levels = t_per_sample.expand(B, N)

        # --- Sample r levels: r in [0, t] for r!=t fraction, r=t for rest ---
        r_neq_t_mask = torch.rand(B, N, device=device) < self.r_neq_t_ratio
        r_levels = t_levels.clone()  # Default: r = t (FM)
        if r_neq_t_mask.any():
            r_rand = torch.rand(B, N, device=device, dtype=dtype)
            r_levels = torch.where(
                r_neq_t_mask,
                r_rand * t_levels,  # r in [0, t]
                r_levels,
            )

        # --- Invalid tokens: t=1 (fully noised), r=0 ---
        if valid_mask is not None:
            t_levels = t_levels.clone()
            t_levels[~valid_mask] = 1.0
            r_levels = r_levels.clone()
            r_levels[~valid_mask] = 0.0

        # --- Forward process: z_t = (1 - t) * x_0 + t * eps ---
        t_expanded = t_levels.unsqueeze(-1)  # (B, N, 1)
        noisy_x = (1.0 - t_expanded) * x_0 + t_expanded * noise  # (B, N, D)

        # --- Average velocity target: u = (z_t - z_r) / (t - r) ---
        # Compute z_r = (1 - r) * x_0 + r * eps (state at start time r)
        r_expanded = r_levels.unsqueeze(-1)  # (B, N, 1)
        z_r = (1.0 - r_expanded) * x_0 + r_expanded * noise  # (B, N, D)

        # Numerical stability: clamp dt = t - r to avoid division by zero
        dt = (t_levels - r_levels).clamp(min=1e-6)  # (B, N)
        u_target = (noisy_x - z_r) / dt.unsqueeze(-1)  # (B, N, D)

        # Force fp32 for numerical safety in the division
        u_target = u_target.float()

        return noisy_x, t_levels, r_levels, noise, u_target

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        x_init: torch.Tensor,
        n_steps: int = 50,
        known_mask: Optional[torch.Tensor] = None,
        known_x: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Euler ODE integration from t=1 to t=0.

        Identical to FlowMatchingScheduler.sample() — multi-step Euler.
        For 1-step MeanFlow inference (future): x = eps - u(eps, 0, 1).

        Args:
            model_fn: ``(x_t, t_levels, r_levels=None, **kwargs) -> v_pred``.
            x_init: Starting noise ``(B, N, D)``.
            n_steps: Euler steps. Default 50.
            known_mask: ``(B, N)`` bool for clamping.
            known_x: ``(B, N, D)`` clean values.
            **model_kwargs: Passed to model_fn.

        Returns:
            Denoised output ``(B, N, D)``.
        """
        B, N, D = x_init.shape
        device = x_init.device
        dt = 1.0 / n_steps

        x_t = x_init.clone()

        for step_idx in range(n_steps):
            t = 1.0 - step_idx * dt
            t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)
            r_levels = torch.full((B, N), t - dt, device=device, dtype=x_t.dtype)

            # Clamp known tokens
            if known_mask is not None and known_x is not None:
                x_t = torch.where(
                    known_mask.unsqueeze(-1).expand_as(x_t),
                    known_x,
                    x_t,
                )

            # Model predicts velocity
            v_pred = model_fn(x_t, t_levels, r_levels=r_levels, **model_kwargs)

            # Euler step: x_{t-dt} = x_t - dt * v_pred
            x_t = x_t - dt * v_pred

        # Final clamping
        if known_mask is not None and known_x is not None:
            x_t = torch.where(
                known_mask.unsqueeze(-1).expand_as(x_t),
                known_x,
                x_t,
            )

        return x_t


def build_diffusion_scheduler(cfg: dict):
    """Build a diffusion scheduler from config.

    Args:
        cfg: Dict with keys:
            - ``noise_type``: ``"flow_matching"`` (default) or ``"ddpm"``
            - ``async_noise``: bool (default True)
            - ``num_train_timesteps``: int (DDPM only, default 100)
            - ``beta_schedule``: str (DDPM only)
            - ``prediction_type``: str (DDPM only)
            - ``gen_timesteps``: int (default 10/50)
            - ``n_inference_steps``: int (flow matching only, default 50)

    Returns:
        FlowMatchingScheduler or DDPMWrapper.
    """
    noise_type = cfg.get("noise_type", "flow_matching")
    async_noise = cfg.get("async_noise", True)

    # Map friendly beta_schedule aliases to diffusers names
    _BETA_ALIASES = {
        "cosine": "squaredcos_cap_v2",
    }

    if noise_type == "flow_matching":
        scheduler = FlowMatchingScheduler(async_noise=async_noise)
        logger.info(f"Flow matching scheduler: async={async_noise}")
        return scheduler
    elif noise_type == "meanflow":
        r_neq_t_ratio = cfg.get("r_neq_t_ratio", 0.25)
        scheduler = MeanFlowScheduler(
            async_noise=async_noise,
            r_neq_t_ratio=r_neq_t_ratio,
        )
        logger.info(
            f"MeanFlow scheduler: async={async_noise}, "
            f"r_neq_t_ratio={r_neq_t_ratio} (FM-equivalent, no JVP)"
        )
        return scheduler
    elif noise_type == "ddpm":
        raw_beta = cfg.get("beta_schedule", "squaredcos_cap_v2")
        beta_schedule = _BETA_ALIASES.get(raw_beta, raw_beta)
        scheduler = DDPMWrapper(
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=beta_schedule,
            prediction_type=cfg.get("prediction_type", "epsilon"),
            gen_timesteps=cfg.get("gen_timesteps", 25),
            async_noise=async_noise,
        )
        logger.info(
            f"DDPM scheduler: timesteps={scheduler.num_train_timesteps}, "
            f"pred={scheduler.prediction_type}, async={async_noise}"
        )
        return scheduler
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Use 'flow_matching' or 'ddpm'.")
