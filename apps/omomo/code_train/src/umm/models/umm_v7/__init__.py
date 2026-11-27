"""UMM v7 model — register-based temporal attention with structured P-layer.

Based on v6.10.7 architecture, with key changes:
- 3 register types (reg-c, reg-x, reg-a) in t-layer with structured attention mask
- Structured P-layer: sub-P-c (half-width) + sub-P-x (half-width) + main P
- Layer-0 input mixing via learned linear for ctx+x tokens
- C_T=128 (narrow), C_P=512 (wide), fc_factor_t=4

- ``AgentV7`` -- Agent with register-based architecture.
- ``PLModel`` -- PyTorch Lightning module wrapping AgentV7.
- ``MotionMachineV7`` -- Register-based temporal attention denoiser.
"""
