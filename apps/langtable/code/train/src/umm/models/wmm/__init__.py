"""UMM V11 model: compact and dense LMM variants with unified noise schedulers.

Key changes from V10:
- x0net + converter: model ALWAYS outputs x0, converter produces target-space output
- 50/50 per-sample noise branching (clean-known vs async)
- Known tokens get x_level=0 during inference (THE critical v10 bug fix)
- Linear loss scaling (s_linear not s_squared)
- Configurable registers (n_reg_c/x/a, default 2/2/2, 0=skip)
- Supports DDPM/FM/MeanFlow in one model
- Removes 3-cat input (uses simple projection instead)
- All operations use torch tensors (never python float for t.clamp)

LMM variants:
- Compact: MotionMachineV7 (v9x) with hardcoded t-p alternation
- Dense: MotionMachineV11Dense with string-based t/p/a layer config
"""
