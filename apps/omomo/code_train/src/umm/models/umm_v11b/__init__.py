"""UMM V11B model: MLP input/output + P-layers only at C_P=512.

Key changes from V11:
- FM-only (no DDPM, no MeanFlow)
- Synced noise (one t per sample, no async, no 50/50 branching)
- MLP-based world model: flatten per-entity -> P-layer attention -> decode
- Loss on ALL valid slots (known + unknown)
- No network_output_space conversion (always x0)
"""
