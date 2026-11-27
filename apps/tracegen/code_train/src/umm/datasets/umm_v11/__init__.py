"""UMM v10 dataset module — with mask scheduler and known_mask support.

Key additions over V9:
- MaskScheduler: training-time mask generation (25% random, 75% domain-specific)
- known_mask / context_known_mask fields in CanonicalDatasetOutput
- Collate handles known_mask (padded positions -> not known)
"""
