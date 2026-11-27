"""Diffusion-process helper functions for computing x0 and SNR.

Provides vectorized routines for recovering the predicted original sample
(x0) from diffusion model outputs and computing the signal-to-noise ratio
at arbitrary timesteps.

Example::

    from umm.models.umm_v6_mixed.df_helpers import compute_x0, compute_snr
    x0 = compute_x0(scheduler, pred, timesteps, noisy_x)
    snr = compute_snr(step, scheduler)
"""

# diffusion helpers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch


def compute_x0(
    scheduler: DDPMScheduler | DDIMScheduler,
    pred: torch.Tensor,
    time_steps: torch.Tensor,
    noisy_x: torch.Tensor,
):
    """
    Efficiently compute x0 (original sample) from noise predictions for batched operations
    with varying timesteps. This function replaces the inefficient loop-based approach
    that calls scheduler.step() individually for each element.

    Performance Benefits:
    - Vectorized operations using PyTorch tensors
    - Eliminates Python loops over batch elements
    - Leverages GPU parallelization for large batches
    - Reduces memory allocation overhead
    - Can be 10-100x faster depending on batch size

    Args:
        scheduler: DDPMScheduler or DDIMScheduler instance
        pred: Model predictions of shape [..., channels] - can be noise or x0 depending on prediction_type
        time_steps: Timesteps tensor of shape [...] - can have varying values across batch
        noisy_x: Noisy samples of shape [..., channels]

    Returns:
        x0_pred: Predicted original samples of shape [..., channels]

    Example:
        >>> scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
        >>> pred = torch.randn(4, 16, 32, 7)  # B, T, P, channels
        >>> time_steps = torch.randint(0, 1000, (4, 16, 32))
        >>> noisy_x = torch.randn(4, 16, 32, 7)
        >>> x0 = compute_x0(scheduler, pred, time_steps, noisy_x)
    """
    # Input validation
    if pred.shape != noisy_x.shape:
        raise ValueError(
            f"pred and noisy_x must have the same shape. Got {pred.shape} and {noisy_x.shape}"
        )

    if pred.shape[:-1] != time_steps.shape:
        raise ValueError(
            f"pred and time_steps shape mismatch. pred: {pred.shape}, time_steps: {time_steps.shape}"
        )

    device = pred.device
    dtype = pred.dtype
    original_shape = pred.shape

    # Flatten all dimensions except the last (channel) dimension
    pred_flat = pred.reshape(-1, pred.shape[-1])  # [N, channels]
    time_steps_flat = time_steps.reshape(-1)  # [N]
    noisy_x_flat = noisy_x.reshape(-1, noisy_x.shape[-1])  # [N, channels]

    batch_size = pred_flat.shape[0]

    # Get scheduler parameters - these are precomputed arrays
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Handle different prediction types
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

    if prediction_type == "epsilon":
        # pred is noise, compute x0: x0 = (x_t - sqrt(1-alpha_t) * epsilon) / sqrt(alpha_t)
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[time_steps_flat])  # [N]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1.0 - alphas_cumprod[time_steps_flat]
        )  # [N]

        # Reshape for broadcasting: [N, 1] to broadcast with [N, channels]
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        x0_pred_flat = (
            noisy_x_flat - sqrt_one_minus_alpha_cumprod * pred_flat
        ) / sqrt_alpha_cumprod

    elif prediction_type == "sample":
        # pred is already x0
        x0_pred_flat = pred_flat

    elif prediction_type == "v_prediction":
        # v-parameterization: v = sqrt(alpha_t) * epsilon - sqrt(1-alpha_t) * x0
        # Solve for x0: x0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[time_steps_flat]).unsqueeze(
            -1
        )  # [N, 1]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1.0 - alphas_cumprod[time_steps_flat]
        ).unsqueeze(
            -1
        )  # [N, 1]

        x0_pred_flat = (
            sqrt_alpha_cumprod * noisy_x_flat - sqrt_one_minus_alpha_cumprod * pred_flat
        )

    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    # Reshape back to original shape
    x0_pred = x0_pred_flat.reshape(original_shape)

    return x0_pred


def compute_snr(step, scheduler: DDPMScheduler):
    """
    Computes Signal-to-Noise Ratio (SNR) for a given step using the scheduler's alpha cumprod.

    Args:
        step (torch.Tensor or int): Timestep(s).
        scheduler (DDPMScheduler): The diffusion scheduler.

    Returns:
        torch.Tensor: SNR values with the same shape as ``step``.

    Example::

        snr = compute_snr(torch.tensor([50, 100]), scheduler)
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(step.device)
    snr = alphas_cumprod[step] / (1 - alphas_cumprod[step])
    return snr
