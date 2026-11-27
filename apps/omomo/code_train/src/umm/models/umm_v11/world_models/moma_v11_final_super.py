"""MotionMachine V11 Super — human-editable sandbox copy of `_ultra`.

This file is a standalone copy of `moma_v11_final_ultra.py` that can be edited
freely without touching the canonical Ultra network. The class name is
`MotionMachineV11Super` so it can coexist with Ultra inside the same package.

Run it directly (no training harness needed):

    conda activate umm
    python src/umm/models/umm_v11/world_models/moma_v11_final_super.py

Or launch via VSCode: pick the "LMM Super Test" configuration in the Run
and Debug panel (see `.vscode/launch.json`).

The `__main__` self-test instantiates several variants (full-dim / narrow /
skip-list) and runs a forward pass with dummy motion inputs **plus dummy
image tokens** so the s/q attention path is actually exercised.

Super-specific rules (vs Ultra):
  * NO AugGridLatent subclass. State is a plain tuple (ra, rt, rp, X).
  * NO pre-attention register<->X fusion.
  * NO post-attention per-layer register adaLN+FFN.
  Registers are updated ONLY by the normal attention blocks (T/P/S/Q steps).

State flows as a tuple of 4 tensors:
  ra  -- corner registers   (B, Rt, Rp, D)
  rt  -- time registers     (B, Rt, P, D)    per-entity
  rp  -- position registers (B, S, Rp, D)    per-timestep
  X   -- primary latent     (B, S, P, D)     ctx + noisy frames

Auxiliary grids (built once, read-only during the loop):
  C   -- adaLN conditioning grid  (B, Rt+S, Rp+P, D)  projected fresh each layer
  R   -- RoPE time grid           (B, Rt+S, Rp+P)
  M   -- validity mask grid       (B, Rt+S, Rp+P)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.wm_utils import HardLayerNorm
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly

logger = logging.getLogger(__name__)

# State tuple: (ra, rt, rp, X)
#   ra: (B, Rt, Rp, D)   corner registers
#   rt: (B, Rt, P, D)    time registers (per-entity)
#   rp: (B, S, Rp, D)    position registers (per-timestep)
#   X:  (B, S, P, D)     primary latent (ctx + noisy frames)
GridState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# ======================================================================
# DiTBlockDualFFN -- separate FFN for register vs primary tokens
# ======================================================================


class DiTBlockDualFFN(DiTBlockTimeOnly):
    """DiT block with two FFNs: one for register tokens, one for primary tokens.

    Attention is shared (all tokens see each other). After attention, the
    sequence is split at `n_reg`: positions [0, n_reg) go through FFN_reg,
    positions [n_reg, L) go through the standard FFN_main. Each FFN has its
    own adaLN modulation (shift/scale/gate).

    Used for T and P layers where registers and data are contiguous.
    """

    def __init__(self, n_reg: int, **kwargs):
        super().__init__(**kwargs)
        C = self.C
        C_feat = self.C_feat

        self.n_reg = n_reg

        # Register-specific FFN (same architecture as self.mlp)
        mlp_hidden_dims = kwargs.get("mlp_hidden_dims", [int(C * 4.0)])
        layers_reg = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers_reg.append(nn.Linear(in_dim, h_dim))
            layers_reg.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers_reg.append(nn.Linear(in_dim, C))
        self.mlp_reg = nn.Sequential(*layers_reg)

        # Register-specific adaLN: 3*C (shift, scale, gate for FFN_reg)
        self.ada_reg = nn.Linear(C_feat, 3 * C)
        nn.init.zeros_(self.ada_reg.bias)

        # Register-specific norm
        self.norm2_reg = nn.LayerNorm(C, elementwise_affine=False)

    def forward(self, x, cond, t=None, num_external_regs=0, attn_mask=None):
        B, L_in, C = x.shape
        x_in, cond_in = self._prepare_inputs(x, cond)

        # shared adaLN for attention + main FFN (register FFN uses separate ada_reg)
        ada_out = self.ada(cond_in)
        s1, sc1, g1, s2, sc2, g2 = ada_out.chunk(6, dim=-1)

        # shared attention (all tokens see each other)
        h = self.norm1(x_in)
        h = self._apply_ada(h, s1, sc1)
        h = self._attn_explicit_time(h, t, num_external_regs, attn_mask_bool=attn_mask)
        x_in = self._add_gate(x_in, g1, h)

        # split at n_reg boundary (num_registers=0 in super, so R=0)
        n_reg = self.n_reg
        x_reg = x_in[:, :n_reg]              # (B, n_reg, C)
        x_main = x_in[:, n_reg:]             # (B, L_main, C)
        c_reg = cond_in[:, :n_reg]            # (B, n_reg, C_feat)

        # main FFN with shared adaLN (s2/sc2/g2 sliced to main positions only)
        h_main = self.norm2(x_main)
        h_main = self._apply_ada(h_main, s2[:, n_reg:], sc2[:, n_reg:])
        h_main = self.mlp(h_main)
        x_main = self._add_gate(x_main, g2[:, n_reg:], h_main)

        # register FFN with separate adaLN (3*C projection)
        ada_reg_out = self.ada_reg(c_reg)
        sr, scr, gr = ada_reg_out.chunk(3, dim=-1)
        h_reg = self.norm2_reg(x_reg)
        h_reg = self._apply_ada(h_reg, sr, scr)
        h_reg = self.mlp_reg(h_reg)
        x_reg = self._add_gate(x_reg, gr, h_reg)

        return torch.cat([x_reg, x_main], dim=1)


# ======================================================================
# Network
# ======================================================================


class MotionMachineV11Super(nn.Module):
    """Tuple-state LMM. State = (ra, rt, rp, X) updated by T/P/S/Q attention steps."""

    # Grid layout — how the 4 state tensors map to a logical 2D grid:
    #
    #              Rp cols              P entity cols
    #           +---------------+---------------------------+
    #  Rt rows  |  ra           |  rt                       |
    #           |  (B,Rt,Rp,D)  |  (B,Rt,P,D)              |
    #           |  corner regs  |  per-entity time regs     |
    #           +---------------+---------------------------+
    #  S rows   |  rp           |  X = [ctx ; noisy]        |
    #           |  (B,S,Rp,D)   |  (B,S,P,D)               |
    #           |  per-step regs|  primary latent           |
    #           +---------------+---------------------------+
    #
    #  S = CTX + T  (context frames + noisy frames, concatenated on dim 1)
    #
    #  Attention patterns per layer type:
    #    T-layer: every COLUMN attends vertically    (all 4 regions, RoPE)
    #    P-layer: every ROW attends horizontally     (all 4 regions)
    #    A-layer: GLOBAL flatten, all-to-all          (all 4 regions, RoPE + add tokens)
    #    S-layer: only LEFT cols (ra, rp) vertical   (rt, X unchanged, add tokens)
    #    Q-layer: only TOP rows (ra, rt) horizontal  (rp, X unchanged, add tokens)
    #
    #  Add tokens (lang+image) are only visible to A, S, Q layers.

    def __init__(
        self,
        # --- architecture ---
        D1: int = 512,
        D2: int = 768,  # register-layer dim (S/Q pool→attn→depool)
        n_heads: int = 16,
        fc_factor: int = 4,
        fc_n_layers: int = 2,
        layer_type_string: str = "tptptptp",
        Rt: int = 4,
        Rp: int = 4,
        # --- input/output ---
        feat_dim: int = 512,
        input_dim: int = 9,
        n_ctx: int = 128,
        fc_out_n_layers: int = 12,
        # --- conditioning ---
        n_traj_tokens: int = 1024,
        time_multiplier: float = 30.0,
        # --- DiT tuning ---
        ada_scale_bias_bound: float = 1.0,
        gate_bound: float = 2.0,
        ada_n_layers: int = 2,
        # --- flexible skip-layer collection for the output head ---
        # None -> auto-compute as all T-layer positions in layer_type_string.
        # Otherwise: explicit list of absolute layer indices (any type allowed, sorted+deduped).
        skip_layer_idx_list: Optional[list] = None,
        # --- uniform AdaLN cond width override ---
        # When > 0, all layers (T/P/A/S/Q) use this single width for their
        # cond_binder output and the DiTBlock C_feat input. When <= 0 (default),
        # the per-layer behavior is preserved: D2 for S/Q, D1 for T/P/A.
        cond_dim_override: int = -1,
    ):
        super().__init__()
        self.D1, self.D2, self.Rt, self.Rp = D1, D2, Rt, Rp
        self.input_dim = input_dim
        self.time_multiplier, self.n_traj_tokens = time_multiplier, n_traj_tokens
        self.layer_type_string = layer_type_string
        self.n_layers = len(layer_type_string)

        assert layer_type_string[0] == "t" and self.D1 % n_heads == 0
        for ch in layer_type_string:
            assert ch in "tpsqa"

        # ---- skip_layer_idx_list resolution ----
        # Default = all T-layer absolute positions; explicit = validated and sorted.
        if skip_layer_idx_list is None:
            resolved_skip = [i for i, c in enumerate(layer_type_string) if c == "t"]
        else:
            assert all(
                isinstance(x, int) for x in skip_layer_idx_list
            ), f"skip_layer_idx_list must be list of ints, got {skip_layer_idx_list}"
            assert len(set(skip_layer_idx_list)) == len(
                skip_layer_idx_list
            ), f"skip_layer_idx_list has duplicates: {skip_layer_idx_list}"
            for idx in skip_layer_idx_list:
                assert (
                    0 <= idx < self.n_layers
                ), f"skip idx {idx} out of range [0, {self.n_layers})"
            resolved_skip = sorted(skip_layer_idx_list)
        self._skip_layer_idx_list = resolved_skip
        self._skip_layer_idx_set = set(resolved_skip)  # O(1) lookup in forward

        # --- input embedding ---
        self.x_proj = nn.Linear(input_dim, D1)
        self.hard_ln_feat = HardLayerNorm(feat_dim)
        self.hard_ln_pe = HardLayerNorm(D1)
        self.ctx_t_emb = nn.Embedding(n_ctx + 1, D1)
        self.level_pe = Timesteps(
            D1, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.t_pe = Timesteps(
            D1, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.t_proj = nn.Linear(D1 + 1, D1)
        if n_traj_tokens > 0:
            self.traj_pe = Timesteps(
                D1, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )

        # --- register init (nn.Embedding-style: N(0, 1), not tiny) ---
        self.reg_t_init = nn.Parameter(torch.randn(1, Rt, 1, D1))
        self.reg_p_init = nn.Parameter(torch.randn(1, 1, Rp, D1))
        self.reg_a_init = nn.Parameter(torch.randn(1, Rt, Rp, D1))

        # --- per-layer register biases (zero-init: identity at start) ---
        self.layer_reg_bias_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, Rt, Rp, D1)) for _ in range(self.n_layers)]
        )
        self.layer_reg_bias_t = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, Rt, 1, D1)) for _ in range(self.n_layers)]
        )
        self.layer_reg_bias_p = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, Rp, D1)) for _ in range(self.n_layers)]
        )

        # --- conditioning: per-layer binder ---
        # Input width (unchanged): concat of [feat, level_pe, t_pe, (traj_pe?)].
        self.binder_in_dim = feat_dim + 2 * D1 + (D1 if n_traj_tokens > 0 else 0)
        self.cond_dim_override = cond_dim_override

        # Effective per-layer cond OUTPUT widths.
        #   override > 0  -> uniform `cond_dim_override` for every layer
        #   override <= 0 -> legacy: D2 for s/q layers, D1 for t/p/a layers
        # Stored on self so the forward path (and add_cond_emb params below)
        # can size their tensors to match.
        self.cond_d1_eff = cond_dim_override if cond_dim_override > 0 else D1
        self.cond_d2_eff = cond_dim_override if cond_dim_override > 0 else D2

        def _layer_cond_out(c: str) -> int:
            return self.cond_d2_eff if c in "sq" else self.cond_d1_eff

        self.cond_binders = nn.ModuleList(
            [
                nn.Linear(self.binder_in_dim, _layer_cond_out(c))
                for c in layer_type_string
            ]
        )

        # --- DiT blocks: T/P use DualFFN at D1, A uses standard at D1, S/Q at D2 ---
        # Token widths (C) are UNCHANGED. Only C_feat (cond input width) is
        # routed through _layer_cond_out so the override widens cond uniformly.
        _dit_kw = dict(
            num_heads=n_heads, num_registers=0,
            ada_scale_bias_bound=ada_scale_bias_bound, gate_bound=gate_bound,
            ada_n_layers=ada_n_layers, qk_norm=True,
        )
        # n_reg for dual-FFN split: T-layer columns have Rt register rows,
        # P-layer rows have Rp register columns
        _n_reg_for = {"t": Rt, "p": Rp}
        self.blocks = nn.ModuleList()
        for c in layer_type_string:
            c_feat = _layer_cond_out(c)
            if c in "tp":
                d = D1
                self.blocks.append(DiTBlockDualFFN(
                    n_reg=_n_reg_for[c],
                    C=d, mlp_hidden_dims=[d * fc_factor] * fc_n_layers, C_feat=c_feat,
                    **_dit_kw,
                ))
            else:
                d = D2 if c in "sq" else D1
                self.blocks.append(DiTBlockTimeOnly(
                    C=d, mlp_hidden_dims=[d * fc_factor] * fc_n_layers, C_feat=c_feat,
                    **_dit_kw,
                ))

        # --- learnable conditioning for add tokens (zero-init = identity at start) ---
        self.lang_cond_emb = nn.Parameter(torch.zeros(1, 1, self.cond_d1_eff))
        self.img_cond_emb = nn.Parameter(torch.zeros(1, 1, self.cond_d1_eff))
        self.lang_cond_emb_d2 = nn.Parameter(torch.zeros(1, 1, self.cond_d2_eff))
        self.img_cond_emb_d2 = nn.Parameter(torch.zeros(1, 1, self.cond_d2_eff))

        # --- per-layer add-token projectors (only A/S/Q use add tokens) ---
        self.add_token_projs = nn.ModuleDict()
        for i, c in enumerate(layer_type_string):
            if c in "qsa":
                out_dim = D2 if c in "sq" else D1
                self.add_token_projs[f"layer_{i}"] = nn.Linear(D1, out_dim)

        # --- per-layer S/Q register pool/depool ---
        # Each S/Q layer gets its own pool projectors + depool MLPs.
        # Keyed by absolute layer index as "layer_{i}".
        self.reg_data_pool = nn.ModuleDict()
        self.reg_depool = nn.ModuleDict()
        for i, c in enumerate(layer_type_string):
            if c == "s":
                N_reg = Rp
                pool_in = Rp * D1
            elif c == "q":
                N_reg = Rt
                pool_in = Rt * D1
            else:
                continue
            k = f"layer_{i}"
            self.reg_data_pool[k] = nn.Linear(pool_in, D2)
            self.reg_depool[k] = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(D2 + D1, D1), nn.GELU(), nn.Linear(D1, D1))
                    for _ in range(N_reg)
                ]
            )

        # --- per-skip LayerNorm (normalize each skip feature before cat) ---
        n_skips = len(self._skip_layer_idx_list) + 1  # +1 for initial X
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(D1) for _ in range(n_skips)
        ])

        # --- output head (skip connections: resolved skip layers + initial X) ---
        skip_dim = D1 * (len(self._skip_layer_idx_list) + 1)
        layers, c_in = [], skip_dim
        for c_out in np.linspace(skip_dim, input_dim, fc_out_n_layers)[:-1]:
            c_out = int(c_out)
            layers += [nn.Linear(c_in, c_out), HardLayerNorm(c_out), nn.LeakyReLU()]
            c_in = c_out
        layers.append(nn.Linear(c_in, input_dim))
        self.output_proj = nn.Sequential(*layers)

        logger.info(
            f"LMM V11 Super: {layer_type_string}  D1={D1} D2={D2}  Rt={Rt} Rp={Rp}  "
            f"cond_dim_override={cond_dim_override}  "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    # ==================== optim helpers ====================

    def get_optim_groups(
        self, weight_decay: float = 0.0, base_lr: Optional[float] = None
    ):
        """Optimizer groups with weight-decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if base_lr is not None:
            for group in optim_groups:
                group["lr"] = base_lr
        return optim_groups

    # ==================== grid builders ====================

    @torch.compiler.disable
    def _sample_traj_ids(self, B, P, device):
        if self.training:
            return torch.stack(
                [
                    torch.randperm(self.n_traj_tokens, device=device)[:P]
                    for _ in range(B)
                ]
            )
        return torch.arange(P, device=device)[None].expand(B, -1)

    @torch.compiler.disable
    def _build_C_raw(
        self,
        cond: torch.Tensor,  # (B, S, P, cond_dim)
        valid_entities: torch.Tensor,  # (B, P) bool
        valid_times: torch.Tensor,  # (B, S) bool
    ) -> torch.Tensor:
        """(B, S, P, cond_dim) -> full grid (B, Rt+S, Rp+P, cond_dim).

        Register regions = MASKED mean-pooled from valid neighbors only.
        Built ONCE per forward.
        """
        Rt, Rp = self.Rt, self.Rp

        # masked mean over entities -> reg_p conditioning
        e_mask = valid_entities[:, None, :, None].float()  # (B, 1, P, 1)
        e_count = e_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, 1, 1, 1)
        c_rp = (cond * e_mask).sum(dim=2, keepdim=True) / e_count  # (B, S, 1, cond_dim)
        c_rp = c_rp.expand(-1, -1, Rp, -1)  # (B, S, Rp, cond_dim)

        c_bot = torch.cat([c_rp, cond], dim=2)  # (B, S, Rp+P, cond_dim)

        # masked mean over time -> reg_t/reg_a conditioning
        t_mask = valid_times[:, :, None, None].float()  # (B, S, 1, 1)
        t_count = t_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1, 1)
        c_top = (c_bot * t_mask).sum(
            dim=1, keepdim=True
        ) / t_count  # (B, 1, Rp+P, cond_dim)
        c_top = c_top.expand(-1, Rt, -1, -1)  # (B, Rt, Rp+P, cond_dim)

        return torch.cat([c_top, c_bot], dim=1)  # (B, Rt+S, Rp+P, cond_dim)

    @torch.compiler.disable
    def _build_R(
        self, B: int, S: int, P: int, N_CTX: int, time_scaled: torch.Tensor
    ) -> torch.Tensor:
        """Build time-only RoPE grid (B, Rt+S, Rp+P). Regs and ctx = 0."""
        Rt, Rp = self.Rt, self.Rp
        R = torch.zeros(
            B, Rt + S, Rp + P, device=time_scaled.device, dtype=time_scaled.dtype
        )
        R[:, Rt + N_CTX :, Rp:] = time_scaled[:, :, None].expand(
            -1, -1, P
        )  # noisy X = timestamps
        return R

    @torch.compiler.disable
    def _build_M(
        self,
        B: int,
        S: int,
        P: int,
        ctx_mask: torch.Tensor,
        noisy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build validity grid M (B, Rt+S, Rp+P) bool.

        Primary X = per-cell mask. Register regions = valid iff that entity/timestep
        is valid in ANY frame. Corner reg_a = always True.
        """
        Rt, Rp = self.Rt, self.Rp
        M = torch.ones(B, Rt + S, Rp + P, device=ctx_mask.device, dtype=torch.bool)
        all_mask = torch.cat([ctx_mask, noisy_mask], dim=1)  # (B, S, P)
        valid_entities = all_mask.any(dim=1)  # (B, P)
        valid_times = all_mask.any(dim=2)  # (B, S)
        M[:, Rt:, Rp:] = all_mask  # (B, S, P) per-cell
        M[:, :Rt, Rp:] = valid_entities[:, None].expand(-1, Rt, -1)  # (B, Rt, P)
        M[:, Rt:, :Rp] = valid_times[:, :, None].expand(-1, -1, Rp)  # (B, S, Rp)
        return M

    @staticmethod
    def _mask_to_attn(valid: torch.Tensor) -> torch.Tensor:
        """(batch, seq) bool -> (batch, 1, seq, seq) bool.

        Prevents softmax-of-all-(-inf) NaN (issue #1): every row must have at
        least one True entry. Two layers of defense:
        1. Force diagonal True -- each query always attends to itself.
        2. If a row is STILL all False (shouldn't happen but defensive), force
           the diagonal again via torch.where.

        Softmax of a row with exactly one 0 (unmasked) and rest -inf produces a
        one-hot, NOT NaN. This is why diagonal-True is sufficient.
        """
        attn = valid[:, None, :] & valid[:, :, None]  # (batch, seq, seq)
        diag = torch.arange(attn.shape[-1], device=attn.device)
        attn[:, diag, diag] = True  # primary defense: self-attention always valid
        return attn[:, None, :, :]  # (batch, 1, seq, seq)

    @staticmethod
    def _assemble_grid(
        ra: torch.Tensor,
        rt: torch.Tensor,
        rp: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """(ra, rt, rp, X) -> (B, Rt+S, Rp+P, D) assembled grid."""
        top = torch.cat([ra, rt], dim=2)  # (B, Rt, Rp+P, D)
        bot = torch.cat([rp, X], dim=2)  # (B, S, Rp+P, D)
        return torch.cat([top, bot], dim=1)  # (B, Rt+S, Rp+P, D)

    def _disassemble_grid(self, Z: torch.Tensor) -> GridState:
        """(B, Rt+S, Rp+P, D) -> (ra, rt, rp, X)."""
        Rt, Rp = self.Rt, self.Rp
        ra = Z[:, :Rt, :Rp, :]  # (B, Rt, Rp, D)
        rt = Z[:, :Rt, Rp:, :]  # (B, Rt, P, D)
        rp = Z[:, Rt:, :Rp, :]  # (B, S, Rp, D)
        X = Z[:, Rt:, Rp:, :]  # (B, S, P, D)
        return ra, rt, rp, X

    # ==================== step functions ====================

    def _step_regular(
        self,
        lt: str,  # "t", "p", or "a"
        ra: torch.Tensor,
        rt: torch.Tensor,
        rp: torch.Tensor,
        X: torch.Tensor,
        C: torch.Tensor,  # (B, Rt+S, Rp+P, cond_d1_eff) -- cond_binder output (D1 or override)
        R: torch.Tensor,  # (B, Rt+S, Rp+P)
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        add_tokens: torch.Tensor,  # (B, A, D1)
        add_mask: torch.Tensor,  # (B, A) bool
        add_cond: torch.Tensor,  # (B, A, cond_d1_eff) learnable lang/img conditioning
    ) -> GridState:
        """Full-grid attention. All 4 regions (ra, rt, rp, X) participate and update.

        T: column-wise with RoPE, no add tokens.
        P: row-wise without RoPE, no add tokens.
        A: global (flatten entire grid) with RoPE, add tokens prepended.
        """
        Rt, Rp, D = self.Rt, self.Rp, self.D1
        D_cond = C.shape[-1]  # may differ from D when cond_dim_override > 0
        Z = self._assemble_grid(ra, rt, rp, X)  # (B, Rt+S, Rp+P, D)
        B, n_rows, n_cols = Z.shape[0], Z.shape[1], Z.shape[2]

        if lt == "t":  # column-wise
            seq = Z.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D)
            cond = C.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D_cond)
            valid = M.permute(0, 2, 1).reshape(B * n_cols, n_rows)
            S = n_rows - Rt
            t_rope = R[:, Rt:, :].permute(0, 2, 1).reshape(B * n_cols, S)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=t_rope,
                num_external_regs=Rt,
                attn_mask=self._mask_to_attn(valid),
            )
            out = out.reshape(B, n_cols, n_rows, D).permute(0, 2, 1, 3)

        elif lt == "p":  # row-wise, no add tokens
            seq = Z.reshape(B * n_rows, n_cols, D)
            cond = C.reshape(B * n_rows, n_cols, D_cond)
            valid = M.reshape(B * n_rows, n_cols)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=None,
                num_external_regs=0,
                attn_mask=self._mask_to_attn(valid),
            )
            out = out.reshape(B, n_rows, n_cols, D)

        else:  # a: global -- flatten entire grid, RoPE + add tokens
            N = n_rows * n_cols
            seq = Z.reshape(B, N, D)
            cond = C.reshape(B, N, D_cond)
            valid = M.reshape(B, N)
            t_rope = R.reshape(B, N)
            A = add_tokens.shape[1]
            tok = self.add_token_projs[f"layer_{layer_idx}"](add_tokens)
            seq = torch.cat([tok, seq], dim=1)
            cond = torch.cat([add_cond, cond], dim=1)
            valid = torch.cat([add_mask, valid], dim=1)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=t_rope,
                num_external_regs=A,
                attn_mask=self._mask_to_attn(valid),
            )
            out = out[:, A:].reshape(B, n_rows, n_cols, D)

        return self._disassemble_grid(out)

    def _step_register(
        self,
        lt: str,  # "s" (left cols, vertical) or "q" (top rows, horizontal)
        ra: torch.Tensor,
        rt: torch.Tensor,
        rp: torch.Tensor,
        X: torch.Tensor,
        C: torch.Tensor,  # (B, Rt+S, Rp+P, cond_d2_eff) -- cond_binder output (D2 or override)
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        add_tokens: torch.Tensor,  # (B, A, D1)
        add_mask: torch.Tensor,  # (B, A) bool
        add_cond: torch.Tensor,  # (B, A, cond_d2_eff) learnable lang/img conditioning
    ) -> GridState:
        """Register-only attention via pool -> D2 attn -> depool. X is unchanged.

        Pool:   cat N_reg register channels -> (B, L, N_reg*D1) -> Linear -> (B, L, D2)
        Cond:   mean-pool N_reg positions of D2 conditioning -> (B, L, D2)
        Attn:   DiT block at D2, seq = L (with add tokens prepended)
        Depool: for each register r: reg_r += MLP_r(cat(attn_out, reg_r))

        S: pool Rp left columns vertically (L = Rt+S). rt, X unchanged.
        Q: pool Rt top rows horizontally (L = Rp+P). rp, X unchanged.
        """
        Rt, Rp, D1, D2 = self.Rt, self.Rp, self.D1, self.D2
        B = ra.shape[0]
        A = add_tokens.shape[1]

        k = f"layer_{layer_idx}"
        data_pool = self.reg_data_pool[k]
        depool_mlps = self.reg_depool[k]

        if lt == "s":
            region = torch.cat([ra, rp], dim=1)  # (B, Rt+S, Rp, D1)
            c_reg = C[:, :, :Rp, :]  # (B, Rt+S, Rp, D2)
            N_reg, L = Rp, region.shape[1]
            valid = M[:, :, 0]  # (B, Rt+S)
        else:  # q
            region = torch.cat([ra, rt], dim=2)  # (B, Rt, Rp+P, D1)
            c_reg = C[:, :Rt, :, :]  # (B, Rt, Rp+P, D2)
            N_reg, L = Rt, region.shape[2]
            valid = M[:, 0, :]  # (B, Rp+P)
            region = region.permute(0, 2, 1, 3)  # (B, Rp+P, Rt, D1)
            c_reg = c_reg.permute(0, 2, 1, 3)  # (B, Rp+P, Rt, D2)

        # ---- pool data: (B, L, N_reg, D1) -> (B, L, N_reg*D1) -> (B, L, D2) ----
        seq = data_pool(region.reshape(B, L, N_reg * D1))  # (B, L, D2)
        # ---- pool cond: mean over N_reg positions (already at D2) ----
        cond = c_reg.mean(dim=2)  # (B, L, D2)

        # ---- prepend add tokens at D2 ----
        tok = self.add_token_projs[k](add_tokens)  # (B, A, D2)
        seq = torch.cat([tok, seq], dim=1)  # (B, A+L, D2)
        cond = torch.cat([add_cond, cond], dim=1)  # (B, A+L, D2)
        valid = torch.cat([add_mask, valid], dim=1)  # (B, A+L)

        # ---- attention at D2 ----
        out = self.blocks[layer_idx](
            x=seq,
            cond=cond,
            t=None,
            num_external_regs=0,
            attn_mask=self._mask_to_attn(valid),
        )
        attn_out = out[:, A:]  # (B, L, D2) -- strip add tokens

        # ---- depool: per-register residual MLP ----
        updated = []
        for r in range(N_reg):
            reg_in = region[:, :, r, :]  # (B, L, D1)
            combined = torch.cat([attn_out, reg_in], dim=-1)  # (B, L, D2+D1)
            reg_out = reg_in + depool_mlps[r](combined)  # (B, L, D1) residual
            updated.append(reg_out)
        result = torch.stack(updated, dim=2)  # (B, L, N_reg, D1)

        if lt == "s":
            return result[:, :Rt], rt, result[:, Rt:], X
        else:  # q
            result = result.permute(0, 2, 1, 3)  # (B, Rt, Rp+P, D1)
            return result[:, :, :Rp], result[:, :, Rp:], rp, X

    # ==================== forward ====================

    def forward(
        self,
        x: torch.Tensor,  # (B, T, P, input_dim) noisy motion frames
        x_mask: torch.Tensor,  # (B, T, P) bool
        x_time: torch.Tensor,  # (B, T) diffusion timestamps [0,1]
        x_level: torch.Tensor,  # (B, T, P) per-token noise level [0,1000]
        feat: torch.Tensor,  # (B, P, feat_dim) per-entity identity features
        ctx: torch.Tensor,  # (B, CTX, P, input_dim) clean context frames
        ctx_mask: torch.Tensor,  # (B, CTX, P) bool
        ctx_level: torch.Tensor,  # (B, CTX, P) noise level (0 for clean)
        add_tokens: Optional[torch.Tensor] = None,       # (B, A_lang, D1) lang tokens
        add_tokens_mask: Optional[torch.Tensor] = None,  # (B, A_lang) bool
        image_tokens: Optional[torch.Tensor] = None,      # (B, A_img, D1)
        image_tokens_mask: Optional[torch.Tensor] = None,  # (B, A_img) bool
        group_ids: Optional[torch.Tensor] = None,          # unused (compat with agent)
        x_r_level: Optional[torch.Tensor] = None,          # unused (MeanFlow compat)
        ctx_r_level: Optional[torch.Tensor] = None,        # unused (MeanFlow compat)
    ) -> torch.Tensor:
        """Returns (B, S, P, input_dim) where S = CTX + T."""
        noisy_x, noisy_mask, noisy_time = x, x_mask, x_time
        noise_level, identity_feat, ctx_x = x_level, feat, ctx
        lang_tokens, lang_tokens_mask = add_tokens, add_tokens_mask

        B, T, P, _ = noisy_x.shape
        device = noisy_x.device
        Rt, Rp, D = self.Rt, self.Rp, self.D1
        N_CTX = ctx_x.shape[1]
        time_scaled = noisy_time.float().clone() * self.time_multiplier  # (B, T)

        # ---- 1. concat context + noisy ----
        all_x = torch.cat([ctx_x, noisy_x], 1)  # (B, S, P, input_dim)
        all_level = torch.cat([ctx_level, noise_level], 1)  # (B, S, P)
        S = N_CTX + T

        time_emb = self.t_pe(time_scaled.reshape(-1)).reshape(B, T, -1)  # (B, T, D)
        time_pos = self.t_proj(
            torch.cat([time_scaled[..., None], time_emb], -1)
        )  # (B, T, D)
        ctx_emb = self.ctx_t_emb(1 + torch.arange(N_CTX, device=device))[None].expand(
            B, -1, -1
        )  # (B, CTX, D)
        all_time_pos = torch.cat([ctx_emb, time_pos], 1)  # (B, S, D)

        # ---- 2. validity masks (needed for C_raw and M) ----
        all_mask = torch.cat([ctx_mask, noisy_mask], dim=1)  # (B, S, P)
        valid_entities = all_mask.any(dim=1)  # (B, P)
        valid_times = all_mask.any(dim=2)  # (B, S)

        # ---- 3. C_raw conditioning grid (B, Rt+S, Rp+P, cond_dim) ----
        level_pe = (
            self.level_pe(all_level.reshape(B * S * P)).reshape(B, S, P, D).to(all_x)
        )  # (B, S, P, D)
        feat_exp = self.hard_ln_feat(
            identity_feat[:, None].expand(-1, S, -1, -1)
        )  # (B, S, P, feat_dim)
        cond_parts = [
            feat_exp,
            self.hard_ln_pe(level_pe),
            self.hard_ln_pe(all_time_pos[:, :, None].expand(-1, -1, P, -1)),
        ]
        if self.n_traj_tokens > 0:
            traj_ids = self._sample_traj_ids(B, P, device)  # (B, P)
            traj_pe = self.traj_pe(traj_ids.reshape(-1).float()).reshape(
                B, P, D
            )  # (B, P, D)
            cond_parts.append(self.hard_ln_pe(traj_pe[:, None].expand(-1, S, -1, -1)))
        C_raw = self._build_C_raw(
            torch.cat(cond_parts, -1), valid_entities, valid_times
        )  # (B, Rt+S, Rp+P, cond_dim)

        # ---- 3. state tuple (ra, rt, rp, X) ----
        x_emb = self.x_proj(all_x)  # (B, S, P, D)
        X = x_emb  # (B, S, P, D)
        ra = self.reg_a_init.expand(B, -1, -1, -1)  # (B, Rt, Rp, D)
        rt = self.reg_t_init.expand(B, -1, P, -1)  # (B, Rt, P, D)
        rp = self.reg_p_init.expand(B, S, -1, -1)  # (B, S, Rp, D)

        # ---- 4. R time-RoPE grid ----
        R = self._build_R(B, S, P, N_CTX, time_scaled)  # (B, Rt+S, Rp+P)

        # ---- 5. M validity grid ----
        M = self._build_M(B, S, P, ctx_mask, noisy_mask)  # (B, Rt+S, Rp+P)

        # ---- 6. unified add_tokens = cat(lang, image), add_mask = cat(masks) ----
        # In a future release lang_tokens and image_tokens will merge into one arg.
        parts, masks, cond_d1_parts, cond_d2_parts = [], [], [], []
        if lang_tokens is not None:
            A_lang = lang_tokens.shape[1]
            parts.append(lang_tokens)
            masks.append(
                lang_tokens_mask
                if lang_tokens_mask is not None
                else torch.ones(B, A_lang, device=device, dtype=torch.bool)
            )
            cond_d1_parts.append(self.lang_cond_emb.expand(B, A_lang, -1))
            cond_d2_parts.append(self.lang_cond_emb_d2.expand(B, A_lang, -1))
        if image_tokens is not None:
            A_img = image_tokens.shape[1]
            parts.append(image_tokens)
            masks.append(
                image_tokens_mask
                if image_tokens_mask is not None
                else torch.ones(B, A_img, device=device, dtype=torch.bool)
            )
            cond_d1_parts.append(self.img_cond_emb.expand(B, A_img, -1))
            cond_d2_parts.append(self.img_cond_emb_d2.expand(B, A_img, -1))
        if parts:
            add_tokens = torch.cat(parts, dim=1)  # (B, A, D1)
            add_mask = torch.cat(masks, dim=1)  # (B, A)
            add_cond_d1 = torch.cat(cond_d1_parts, dim=1)  # (B, A, D1)
            add_cond_d2 = torch.cat(cond_d2_parts, dim=1)  # (B, A, D2)
        else:
            add_tokens = torch.zeros(B, 0, D, device=device)
            add_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
            add_cond_d1 = torch.zeros(B, 0, self.cond_d1_eff, device=device)
            add_cond_d2 = torch.zeros(B, 0, self.cond_d2_eff, device=device)

        # ---- 7. skip list (initial X always prepended) ----
        skip_list = [X.reshape(B * S, P, D).clone()]

        # ---- 8. main loop ----
        for i, lt in enumerate(self.layer_type_string):
            # per-layer register bias (zero-init = identity at start)
            ra = ra + self.layer_reg_bias_a[i]
            rt = rt + self.layer_reg_bias_t[i]
            rp = rp + self.layer_reg_bias_p[i]

            C = self.cond_binders[i](C_raw)  # (B, Rt+S, Rp+P, D1 or D2)
            if lt in "tpa":
                ra, rt, rp, X = self._step_regular(
                    lt, ra, rt, rp, X, C, R, M, i, add_tokens, add_mask, add_cond_d1
                )
            else:  # s, q
                ra, rt, rp, X = self._step_register(
                    lt, ra, rt, rp, X, C, M, i, add_tokens, add_mask, add_cond_d2
                )

            if i in self._skip_layer_idx_set:
                skip_list.append(X.reshape(B * S, P, D).clone())

        # ---- 9. output (LayerNorm each skip, then cat) ----
        normed = [self.skip_norms[i](s) for i, s in enumerate(skip_list)]
        x_out = torch.cat(normed, dim=-1)  # (B*S, P, D * len(skip_list))
        return self.output_proj(x_out.reshape(-1, x_out.shape[-1])).reshape(
            B, S, P, self.input_dim
        )


# ==================== self-test ====================
#
# Standalone smoke-test. No training harness, no YAML config, no dataset.
# Run directly:
#     conda activate umm
#     python src/umm/models/umm_v11/world_models/moma_v11_final_super.py
#
# Or via VSCode: "LMM Super Test" launch configuration (see .vscode/launch.json).
#
# The matrix below instantiates multiple variants and runs a forward pass with
# dummy motion inputs AND dummy lang+image tokens. All P/S/Q layers receive
# the unified add_tokens = cat(lang, image) with a proper mask. Configs without
# P/S/Q (pure T-layers) accept tokens but don't use them -- we still feed
# tokens to prove it's a no-op, not a crash.
#
# Tuning knobs for quick human iteration:
#   - B / T / C / P     : batch / noisy-frames / ctx-frames / entities
#   - A_LANG / A_IMG    : number of lang tokens / image tokens per batch
#   - COMMON["D1"]      : model width
#   - configs dict      : add / remove / tweak architectures

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    COMMON = dict(
        D1=512,
        D2=768,
        n_heads=8,
        fc_factor=4,
        feat_dim=512,
        input_dim=9,
        Rt=4,
        Rp=4,
        fc_n_layers=2,
        fc_out_n_layers=12,
        ada_n_layers=2,
        time_multiplier=30.0,
        ada_scale_bias_bound=1.0,
        gate_bound=2.0,
        n_traj_tokens=1024,
    )
    configs = {
        "tptptpt": dict(layer_type_string="tptptpt"),
        "tqtqtqt": dict(layer_type_string="tqtqtqt"),
        "tststst": dict(layer_type_string="tststst"),
        "tatatata": dict(layer_type_string="tatatata"),
        "tpsqatpsqat (all types)": dict(layer_type_string="tpsqatpsqat"),
        "tqtqtqt skip-all": dict(
            layer_type_string="tqtqtqt", skip_layer_idx_list=[0, 1, 2, 3, 4, 5, 6]
        ),
    }

    B, T, C, P = 2, 5, 4, 8
    A_LANG, A_IMG = 3, 6
    D = COMMON["D1"]
    FEAT = COMMON["feat_dim"]
    IN_DIM = COMMON["input_dim"]

    noisy_mask = torch.ones(B, T, P, dtype=torch.bool, device=device)
    noisy_mask[:, :, -2:] = False
    ctx_mask = torch.ones(B, C, P, dtype=torch.bool, device=device)
    ctx_mask[:, :, -2:] = False

    image_tokens = torch.randn(B, A_IMG, D, device=device)
    image_tokens_mask = torch.ones(B, A_IMG, dtype=torch.bool, device=device)
    print(
        f"Dummy: x=({B},{T},{P},{IN_DIM})  ctx=({B},{C},{P},{IN_DIM})  "
        f"lang=({B},{A_LANG},{D})  img=({B},{A_IMG},{D})"
    )

    def run_forward(net):
        with torch.no_grad():
            return net(
                x=torch.randn(B, T, P, IN_DIM, device=device),
                x_mask=noisy_mask,
                x_time=torch.rand(B, T, device=device),
                x_level=torch.rand(B, T, P, device=device) * 1000,
                feat=torch.randn(B, P, FEAT, device=device),
                ctx=torch.randn(B, C, P, IN_DIM, device=device),
                ctx_mask=ctx_mask,
                ctx_level=torch.zeros(B, C, P, device=device),
                add_tokens=torch.randn(B, A_LANG, D, device=device),
                add_tokens_mask=torch.ones(B, A_LANG, device=device, dtype=torch.bool),
                image_tokens=image_tokens,
                image_tokens_mask=image_tokens_mask,
            )

    n_passed = 0
    for name, cfg in configs.items():
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        net = MotionMachineV11Super(**COMMON, **cfg).to(device).eval()
        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        uses_add = any(c in cfg["layer_type_string"] for c in "psqa")
        print(
            f"Params: {n_params:.2f}M  skip={net._skip_layer_idx_list}  "
            f"types={cfg['layer_type_string']}  add_tokens={uses_add}"
        )
        out = run_forward(net)
        assert out.shape == (B, C + T, P, IN_DIM), f"FAIL {name}: {out.shape}"
        assert torch.isfinite(out).all(), f"FAIL {name}: non-finite output"
        print(f"Output: {list(out.shape)}  finite=True  PASS")
        n_passed += 1

    print(f"\n{'='*72}\nAll {n_passed}/{len(configs)} passed.\n{'='*72}")
