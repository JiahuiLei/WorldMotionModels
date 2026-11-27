"""UMM v1 model components.

This package contains the core model architecture for UMM v1, including:

- ``PLModel`` -- PyTorch Lightning module that orchestrates training/inference.
- ``Agent`` -- the main ``nn.Module`` combining MotionMachine and PoseIO.
- ``MotionMachineV2`` / ``MotionMachineV4`` -- DiT-based denoising backbones.
- ``PoseIO_V3`` -- SE(3) pose encoder/decoder (xyz + 6-D rotation).
- Visualization helpers for rollout rendering.
"""
