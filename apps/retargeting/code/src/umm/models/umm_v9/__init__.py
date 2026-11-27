"""WMM v9 model — clean v7.4 architecture for HH with improvements.

Key changes from v7:
- Separate n_reg_c/x/a (supports zero to skip)
- Per-register-type AdaLN conditioning (ctx-mean, x-mean, all-mean)
- Per-layer learnable register tokens (additive with prev output, random init)
- Time embedding binding (cat + project) when RoPE is off
- Fixed AdaLN MLP bug (pop last activation when ada_n_layers > 1)
- Dead args cleaned from moma constructor
- Compile-optimized (CPU-GPU sync ops removed)
"""
