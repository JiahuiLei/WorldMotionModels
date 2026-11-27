"""UMM v6 Non-Diffusion-Forcing model.

Replaces diffusion forcing (per-element async noise) with masked conditioned
DDPM: single global noise level, explicit condition tensor + mask, dual
conditioning path (input concatenation + AdaLN addition).

Key differences from umm_v6_all:
- ONE global noise level ``t`` per sample (standard DDPM)
- Three tensors: ``X`` (all noised), ``cond_X`` (known=real, unknown=learnable), ``mask`` (1=known)
- Dual conditioning: input concat ``[X; cond_X; mask]`` -> 19 channels + AdaLN per-token
- Loss on ALL valid positions (gated by trace_mask and ctx_mask)
- Standard DDIM inference with one global t
"""
