"""UMM V6 Mixed: Unified model for mixed-domain training (bridge+davis+behavior).

Key differences from umm_v6:
- Per-dataset PoseIO normalization (separate mean/std per dataset_id)
- Variable image resolution handling (pad to batch max in DINO forward)
- dataset_id flows through batch for routing normalization

All components are self-contained within this package (no cross-version imports).
"""
