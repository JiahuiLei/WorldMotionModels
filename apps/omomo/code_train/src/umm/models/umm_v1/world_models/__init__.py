"""World models for UMM (Universal Motion Machine).

This package contains the core neural network components for motion prediction:

- ``dit_atten``: DiT (Diffusion Transformer) attention blocks with RoPE
  positional encoding.
- ``language``: T5 and CLIP language encoders for text conditioning.
- ``moma_v2``: Primary MotionMachine (V2) with factored temporal/spatial/all
  attention layers and IdentifyLayerV2 for embodiment encoding.
- ``moma_v4``: All-attention MotionMachine variant (V4) that uses only global
  ("a") attention layers with an OutputHead.
- ``pointnet``: PointNet-based point cloud encoders (ResNet variants).
- ``wm_utils``: Shared utilities (LayerNorm variants, attention masks,
  positional encodings, etc.).
- ``moma_v1``, ``moma_v21``, ``moma_v3``: Archived/deprecated earlier
  iterations (entirely commented out).
"""
