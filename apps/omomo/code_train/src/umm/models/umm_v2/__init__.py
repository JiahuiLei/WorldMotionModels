"""UMM v2 model components.

This package contains the model architecture for UMM v2, which extends v1 with:

- ``DINOImageEncoder`` -- frozen DINOv2 encoder for RGB images with view position
  embeddings and spatial feature map for pixel queries.
- ``IdentifyLayerV2Dual`` -- dual-path input layer: PointNet for robot links,
  DINO pixel query for objects.
- ``AgentV2`` -- Agent subclass that encodes images via DINO spatial features,
  queries per-pixel geo_emb for objects, and concatenates flat DINO tokens with
  language tokens as ``add_tokens`` for cross-attention.
- ``PLModel`` -- PyTorch Lightning module that swaps in AgentV2.

All components are self-contained within this package (no cross-version imports).
"""
