"""UMM v6_mixed dataset implementations for mixed-domain training.

Supports multiple datasets (bridge, davis, behavior) in the SAME batch.
Each sample carries a dataset_id for per-dataset PoseIO normalization.
Images of different spatial resolutions are handled by the model via
zero-padding to batch-maximum dimensions.
"""
