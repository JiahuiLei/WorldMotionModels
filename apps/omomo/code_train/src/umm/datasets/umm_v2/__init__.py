"""UMM v2 dataset implementations for BH-2.0.

This package provides a modified BEHAVIOR dataset that uses pixel UV sampling
instead of centroid-based queries, and returns RGB images for DINO encoding.

Key differences from v1:
- Pixel UV sampling: uniformly sample pixel locations from segmentation masks
  instead of computing centroid queries.
- Degenerate local PCL: local point clouds are zeroed out (model uses DINO
  image features instead of PointNet).
- RGB images: current-time RGB images are resized and returned in the batch
  for the model's DINO encoder.
"""
