"""UMM v6_all dataset module — unified dataset infrastructure for all applications.

Integrates datasets from all application domains:
- Human+humanoid (hh) — from umm_v1
- TACO HOI — from umm_v1
- Manipulator motion planning (mp) — from umm_v1
- LangTable — from umm_v1
- ManiSkill — from umm_v1
- ManipTrans/dex — from umm_v1
- Behavior — from umm_v2
- Real-world (Bridge, DAVIS) — from umm_v2

All dataset classes use the shared CanonicalDatasetOutput format.
"""
