"""UMM E11 — Equivariant Sandwich Architecture.

Architecture: [et, ep, t, p, t, p, t, dp, dt]
- et/ep: Equivariant VN boundary layers (canonicalization via learned registers + inner products)
- t, p: Free (unconstrained) axial transformers with full AdaLN
- dp/dt: Equivariant decoder (scalar weights × equivariant registers)

Based on Kaba et al. (ICML 2023) universal approximation theorem:
equivariant boundary + unconstrained middle = universal approximator of equivariant functions.
"""
