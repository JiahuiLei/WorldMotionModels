"""UMM v1 dataset implementations.

This package provides domain-specific dataset classes for the Universal Motion
Machine (UMM) framework.  Each dataset converts raw per-domain data (robot
arms, humanoids, dexterous hands, humans, etc.) into the canonical
:class:`~umm.datasets.umm_v1.dataset_template.CanonicalDatasetOutput` format
consumed by the UMM training loop.

Subpackage contents:

- ``base_dataset`` -- abstract :class:`BaseDataset` and :class:`PromptV0`
- ``dataset_config`` -- semantic-ID / UID constants
- ``dataset_utils`` -- shared I/O and array helpers
- ``behavior_dataset`` -- BEHAVIOR-1K real-robot dataset
- ``hh_dataset`` -- human--humanoid retargeting dataset
- ``maniptrans_dataset`` -- ManipTrans dexterous manipulation dataset
- ``mp_dataset`` -- motion-planning obstacle-avoidance dataset
- ``maniskill_oracle_dataset`` -- ManiSkill simulation dataset
- ``langtable_dataset`` -- Language-Table dataset
- ``taco_dataset`` -- TACO bimanual manipulation dataset
"""
