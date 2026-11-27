"""UMM v6_all model components — unified model for ALL applications.

Based on the v6 two-scale pooled temporal attention architecture, extended to
support all application domains with optional DINO image encoder and variable P.

- ``AgentV6All`` -- Agent with optional DINO, variable P, all domains.
- ``PLModel`` -- PyTorch Lightning module wrapping AgentV6All.
- ``MotionMachineV6`` -- Two-scale pooled temporal attention denoiser.

When ``model.dino_encoder`` is absent or null in config, the DINO encoder
is not instantiated and image-related code paths are skipped entirely.
This enables the same model architecture to serve image-based domains
(behavior, bridge, langtable, maniskill) and non-image domains
(hh, taco, mp, dex) without code changes.
"""
