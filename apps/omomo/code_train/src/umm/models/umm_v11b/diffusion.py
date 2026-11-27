"""Diffusion schedulers for V11B: Flow Matching and DDPM (x0 prediction).

Two noise paradigms:
- Flow Matching: Linear interpolation x_t = (1-t)*x_0 + t*eps.
- DDPM: alpha_cumprod-based schedule via diffusers.DDPMScheduler.
  Model always outputs x0 prediction in both cases.

Key V11B design decisions:
- x0net: model ALWAYS outputs x0
- Synced noise only (async_noise=False): one t per sample
- Clean-known mode always:
    known+valid -> t=0 (clean)
    unknown+valid -> global t (noised)
    invalid -> t=1.0 (FM) or t=num_train_timesteps-1 (DDPM)
- Known tokens ARE included in loss
- All operations use torch tensors (never python float for t.clamp)

Example::

    from umm.models.umm_v11b.diffusion import build_diffusion_scheduler
    scheduler = build_diffusion_scheduler(cfg)
    noisy_x, t_levels, noise, target = scheduler.add_noise(x_0, valid_mask=mask)
"""

import logging
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class FlowMatchingScheduler:
    """Flow matching noise scheduler.

    V11B: model outputs x0. Synced noise, clean-known mode only.

    Always uses:
    - async_noise=False: one t per sample (synced)
    - Clean-known mode: known+valid->t=0, unknown+valid->global t, invalid->t=1.0
    """

    def __init__(self):
        self.async_noise = False
        self.noise_type = "flow_matching"

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add FM noise -- synced, clean-known mode only.

        - known+valid -> t=0 (clean)
        - unknown+valid -> global t (noised)
        - invalid -> t=1.0

        Args:
            x_0: Clean data ``(B, N, D)``.
            noise: Optional pre-sampled noise.
            valid_mask: Optional ``(B, N)`` bool. Invalid -> t=1.0.
            known_mask: Optional ``(B, N)`` bool. Known+valid -> t=0.

        Returns:
            (noisy_x, t_levels, noise, target) where target = x_0.
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # One t per sample (synced)
        t = torch.rand(B, 1, device=device, dtype=x_0.dtype)  # (B, 1)

        # Build per-token t_levels
        t_levels = t.expand(B, N).clone()  # (B, N)

        if known_mask is not None:
            # Known+valid -> t=0
            known_valid = known_mask & (valid_mask if valid_mask is not None else torch.ones_like(known_mask))
            t_levels = torch.where(known_valid, torch.zeros_like(t_levels), t_levels)

        if valid_mask is not None:
            # Invalid -> t=1.0
            t_levels = torch.where(~valid_mask, torch.ones_like(t_levels), t_levels)

        # FM interpolation: x_t = (1-t)*x_0 + t*eps
        t_exp = t_levels.unsqueeze(-1)  # (B, N, 1)
        noisy_x = (1.0 - t_exp) * x_0 + t_exp * noise

        return noisy_x, t_levels, noise, x_0  # target = x_0


class DDPMX0Scheduler:
    """DDPM noise scheduler with x0 prediction target.

    Uses diffusers.DDPMScheduler for alpha_cumprod-based noise schedule.
    Synced noise only (one timestep per sample). Clean-known mode:
    - known+valid -> t=0 (clean, no noise added)
    - unknown+valid -> global t (noised)
    - invalid -> t=max_timestep (pure noise)
    """

    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "squaredcos_cap_v2"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as DiffusersDDPM
        self.noise_type = "ddpm"
        self.async_noise = False
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.ddpm = DiffusersDDPM(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="sample",  # x0 prediction
            clip_sample=False,  # pose data exceeds [-1, 1]
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add DDPM noise -- synced, clean-known mode.

        - known+valid -> t=0 (clean)
        - unknown+valid -> global t (noised via alpha_cumprod)
        - invalid -> t=num_train_timesteps-1 (maximum noise)

        Returns:
            (noisy_x, t_levels, noise, target) where target = x_0.
            t_levels are INTEGER timesteps as float tensors.
        """
        B, N, D = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        # One integer timestep per sample (synced)
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)  # (B,)

        # Use diffusers to compute noisy_x: sqrt(alpha_t)*x_0 + sqrt(1-alpha_t)*noise
        # DDPMScheduler.add_noise expects: original_samples, noise, timesteps
        # It works on arbitrary shapes as long as timesteps broadcasts correctly
        noisy_x = self.ddpm.add_noise(x_0, noise, t)  # (B, N, D)

        # Build per-token t_levels (integer timesteps as float)
        t_levels = t[:, None].expand(B, N).clone().float()  # (B, N)

        if known_mask is not None:
            known_valid = known_mask & (valid_mask if valid_mask is not None else torch.ones_like(known_mask))
            t_levels = torch.where(known_valid, torch.zeros_like(t_levels), t_levels)
            # Known+valid tokens get clean x_0 (no noise)
            noisy_x = torch.where(
                known_valid.unsqueeze(-1).expand_as(noisy_x),
                x_0,
                noisy_x,
            )

        if valid_mask is not None:
            # Invalid -> maximum timestep (pure noise)
            t_levels = torch.where(~valid_mask, torch.full_like(t_levels, float(self.num_train_timesteps - 1)), t_levels)
            # Invalid tokens get pure noise
            noisy_x = torch.where(
                (~valid_mask).unsqueeze(-1).expand_as(noisy_x),
                noise,
                noisy_x,
            )

        return noisy_x, t_levels, noise, x_0  # target = x_0


def build_diffusion_scheduler(cfg: dict) -> Union[FlowMatchingScheduler, DDPMX0Scheduler]:
    """Build diffusion scheduler based on noise_type config."""
    noise_type = cfg.get("noise_type", "flow_matching")
    if noise_type == "ddpm":
        return DDPMX0Scheduler(
            num_train_timesteps=cfg.get("num_train_timesteps", 1000),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        )
    else:
        return FlowMatchingScheduler()
