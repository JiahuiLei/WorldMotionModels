"""MotionMachine V11 MoE -- Mixture of Experts variant of Super.

Replaces all T/P/A DiT blocks with DiTBlockMoE (3 expert pathways per block).
Token routing is hard-coded by position: X/trace -> expert 0, ctx+add_tokens -> expert 1,
registers -> expert 2.

Subclasses MotionMachineV11Super to inherit all grid logic, conditioning, skip connections.
Only overrides: __init__ (replace blocks), forward (store N_CTX), _step_regular (add token_type).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from umm.models.wmm.world_models.moma_v11_final_super import (
    MotionMachineV11Super,
)
from umm.models.wmm.world_models.dit_atten import DiTBlockMoE

logger = logging.getLogger(__name__)


class MotionMachineV11Moe(MotionMachineV11Super):
    """MoE variant of the Super LMM with 3 hard-routed experts per DiT block."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Capture params needed for block re-creation
        n_heads = kwargs.get("n_heads", 16)
        fc_factor = kwargs.get("fc_factor", 4)
        fc_n_layers = kwargs.get("fc_n_layers", 2)
        ada_scale_bias_bound = kwargs.get("ada_scale_bias_bound", 1.0)
        gate_bound = kwargs.get("gate_bound", 2.0)
        ada_n_layers = kwargs.get("ada_n_layers", 1)

        _dit_kw = dict(
            num_heads=n_heads, num_registers=0,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            qk_norm=True,
        )

        # Replace T/P/A blocks with DiTBlockMoE. S/Q blocks (if any) stay as-is.
        new_blocks = nn.ModuleList()
        for i, c in enumerate(self.layer_type_string):
            c_feat = self.cond_d2_eff if c in "sq" else self.cond_d1_eff
            if c in "tpa":
                d = self.D1
                new_blocks.append(DiTBlockMoE(
                    C=d, mlp_hidden_dims=[d * fc_factor] * fc_n_layers,
                    C_feat=c_feat, **_dit_kw,
                ))
            else:  # s, q -- keep original block
                new_blocks.append(self.blocks[i])
        self.blocks = new_blocks

        # Guard: ctx_unidirectional_mask is not supported
        # by the MoE override of _step_regular (which always uses symmetric masks).
        assert not self.ctx_unidirectional_mask, (
            "ctx_unidirectional_mask=True is not supported by MotionMachineV11Moe. "
            "The MoE _step_regular override uses symmetric _mask_to_attn only."
        )

        # Internal state for _step_regular (set by forward)
        self._n_ctx = 0
        self._moe_logged_layers = set()

        logger.info(
            f"LMM V11 MoE: replaced T/P/A blocks with DiTBlockMoE. "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def forward(
        self,
        x, x_mask, x_time, x_level, feat,
        ctx, ctx_mask, ctx_level,
        **kwargs,
    ):
        """Store N_CTX for _step_regular, then delegate to Super."""
        self._n_ctx = ctx.shape[1]
        return super().forward(
            x, x_mask, x_time, x_level, feat,
            ctx, ctx_mask, ctx_level,
            **kwargs,
        )

    @torch.compiler.disable
    def _log_moe_counts(
        self, lt: str, layer_idx: int, tt: torch.Tensor, total: int, extra: str = "",
    ):
        """Log MoE token counts once per layer. Entire method outside compile boundary."""
        key = (lt, layer_idx)
        if key not in self._moe_logged_layers:
            self._moe_logged_layers.add(key)
            n0 = int((tt == 0).sum())
            n1 = int((tt == 1).sum())
            n2 = int((tt == 2).sum())
            logger.info(
                f"MoE {lt.upper()}-layer {layer_idx}: X={n0} ctx={n1} reg={n2} "
                f"(total={total}){extra}"
            )

    @torch.compiler.disable  # ModuleDict key is dynamic (f"layer_{layer_idx}") -- not traceable
    def _step_regular(
        self,
        lt, ra, rt, rp, X, C, R, M, layer_idx,
        add_tokens, add_mask, add_cond,
        R_e=None,
        N_CTX=0,
        unknown_x_mask=None,  # V13.3.4: accepted but ignored (MoE uses symmetric masks only)
    ):
        """Override: compute token_type and pass to DiTBlockMoE blocks."""
        Rt, Rp, D = self.Rt, self.Rp, self.D1
        D_cond = C.shape[-1]
        Z = self._assemble_grid(ra, rt, rp, X)
        B, n_rows, n_cols = Z.shape[0], Z.shape[1], Z.shape[2]
        # Use N_CTX from parent when passed (> 0), else fall back to self._n_ctx.
        N_CTX = N_CTX if N_CTX > 0 else self._n_ctx
        device = Z.device

        if lt == "t":  # column-wise
            seq = Z.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D)
            cond = C.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D_cond)
            valid = M.permute(0, 2, 1).reshape(B * n_cols, n_rows)
            S = n_rows - Rt
            t_rope = R[:, Rt:, :].permute(0, 2, 1).reshape(B * n_cols, S)

            # token_type: 2D grid (n_cols, n_rows) — register cols differ from entity cols.
            # After permute, batch dim = B*n_cols with cols interleaved:
            #   batch positions [b*n_cols + col] for col in [0, n_cols).
            # Register cols (col < Rp): all rows are register data → type 2.
            # Entity cols (col >= Rp): rows [0,Rt)=reg(2), [Rt,Rt+N_CTX)=ctx(1), rest=trace(0).
            tt_grid = torch.zeros(n_cols, n_rows, device=device, dtype=torch.long)
            tt_grid[:, :Rt] = 2                       # all cols: register rows
            tt_grid[:Rp, Rt:] = 2                     # register cols: all data rows are reg
            tt_grid[Rp:, Rt:Rt + N_CTX] = 1           # entity cols: ctx rows
            # tt_grid[Rp:, Rt+N_CTX:] = 0             # entity cols: trace rows (already 0)
            tt = tt_grid.unsqueeze(0).expand(B, -1, -1).reshape(B * n_cols, n_rows)

            attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq, cond=cond, t=t_rope,
                num_external_regs=Rt,
                attn_mask=attn_mask,
                token_type=tt,
            )
            out = out.reshape(B, n_cols, n_rows, D).permute(0, 2, 1, 3)

            # Log once per layer — pass tensor to @torch.compiler.disable method
            entity_col = min(Rp, n_cols - 1)
            self._log_moe_counts(
                "t", layer_idx, tt_grid[entity_col], n_rows,
                extra=f", reg_cols={Rp} all-type2",
            )

        elif lt == "p":  # row-wise
            seq = Z.reshape(B * n_rows, n_cols, D)
            cond = C.reshape(B * n_rows, n_cols, D_cond)
            valid = M.reshape(B * n_rows, n_cols)

            # token_type: (n_rows, n_cols)
            tt_grid = torch.zeros(n_rows, n_cols, device=device, dtype=torch.long)
            tt_grid[:, :Rp] = 2               # register columns always type 2
            tt_grid[:Rt, Rp:] = 2             # register rows, entity columns
            tt_grid[Rt:Rt + N_CTX, Rp:] = 1  # ctx rows, entity columns
            # tt_grid[Rt+N_CTX:, Rp:] = 0    # trace rows (already 0)
            tt = tt_grid.unsqueeze(0).expand(B, -1, -1).reshape(B * n_rows, n_cols)

            attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq, cond=cond, t=None,
                num_external_regs=0,
                attn_mask=attn_mask,
                token_type=tt,
            )
            out = out.reshape(B, n_rows, n_cols, D)

            trace_row = min(Rt + N_CTX, n_rows - 1)
            self._log_moe_counts("p", layer_idx, tt_grid[trace_row], n_cols)

        else:  # "a" -- global flatten + add tokens
            N = n_rows * n_cols
            seq = Z.reshape(B, N, D)
            cond_flat = C.reshape(B, N, D_cond)
            valid = M.reshape(B, N)
            t_rope = R.reshape(B, N)
            e_rope = R_e.reshape(B, N) if R_e is not None else None
            A = add_tokens.shape[1]
            tok = self.add_token_projs[f"layer_{layer_idx}"](add_tokens)
            seq = torch.cat([tok, seq], dim=1)
            cond_flat = torch.cat([add_cond, cond_flat], dim=1)
            valid = torch.cat([add_mask, valid], dim=1)

            # token_type for grid
            tt_grid = torch.zeros(n_rows, n_cols, device=device, dtype=torch.long)
            tt_grid[:, :Rp] = 2                  # all register columns
            tt_grid[:Rt, Rp:] = 2                # register rows, entity columns
            tt_grid[Rt:Rt + N_CTX, Rp:] = 1     # ctx entities
            # tt_grid[Rt+N_CTX:, Rp:] = 0       # trace entities (already 0)
            tt_flat = tt_grid.reshape(N)

            # add_tokens = type 1 (context: language, DINO image tokens)
            tt_add = torch.ones(A, device=device, dtype=torch.long)
            tt = torch.cat([tt_add, tt_flat]).unsqueeze(0).expand(B, -1)

            attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq, cond=cond_flat, t=t_rope,
                num_external_regs=A,
                attn_mask=attn_mask,
                e=e_rope,
                token_type=tt,
            )
            out = out[:, A:].reshape(B, n_rows, n_cols, D)

            self._log_moe_counts("a", layer_idx, tt[0], A + N)

        return self._disassemble_grid(out)
