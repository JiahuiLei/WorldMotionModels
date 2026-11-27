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
# V13: use the local dit_atten fork so we can ship 2D RoPE without touching
# shared umm_v9 code.
from umm.models.umm_v13.world_models.dit_atten import DiTBlockTimeOnly

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
        # --- V13: RoPE mode for "a" layers ---
        # "1d_time" (default) = current behavior, time-only RoPE.
        # "2d_time_trajid" = 2D RoPE: head-dim split half/half, first half
        #     rotated by time phase, second half rotated by random-permuted
        #     traj_id. Forces attention to distinguish entities at the same
        #     time step. Requires head_dim % 4 == 0.
        rope_mode: str = "1d_time",
        # Multiplier applied to traj_id before it becomes a RoPE phase. The
        # default `0.0` means "use the formula default": 16.0 / n_traj_tokens
        # so the entity phase covers roughly the same range as time phase.
        # Set explicitly (>0) to override.
        entity_rope_multiplier: float = 0.0,
        # --- uniform AdaLN cond width override ---
        # When > 0, all layers (T/P/A/S/Q) use this single width for their
        # cond_binder output and the DiTBlock C_feat input. When <= 0 (default),
        # the per-layer behavior is preserved: D2 for S/Q, D1 for T/P/A.
        cond_dim_override: int = -1,
        # --- V13.2 LT explore flags. Defaults preserve OLD behavior. ---
        # V2: if True, the learnable traj_pe is not used in the cond
        # concat (replaced by a zero tensor of identical shape), and
        # traj_pe.weight is frozen. binder_in_dim is UNCHANGED so checkpoint-
        # load compat holds. Requires rope_mode == "2d_time_trajid" (else the
        # model loses ALL entity-id signal).
        disable_trajpe: bool = False,
        # V3: how `_sample_traj_ids` samples training-mode ids.
        #   "permN" (default) = existing behavior: randperm(n_traj_tokens)[:P]
        #   "permP"           = torch.randperm(P) per sample (entity phase in [0,P))
        traj_id_sampler: str = "permN",
        # V7: how `self.traj_pe.weight` is initialized.
        #   "same" (default) = existing behavior: copy slot 0 into all slots
        #   "std"            = nn.init.normal_(weight, mean=0.0, std=0.02)
        traj_pe_init: str = "same",
        # V5: per-sample dynamic ctx/reg pseudo positions.
        # False (default): keep the pre-V5 phase layout in _build_R /
        #   _build_R_entity (back-compat, bit-identical).
        # True: apply the V5 per-sample phase table (see _build_R /
        #   _build_R_entity docstrings). Rt + CTX register rows get unique
        #   time phases strictly outside the real time range; Rp register
        #   cols get unique entity phases strictly outside the real entity
        #   range. Trace rows' Rp register cols also propagate row time
        #   (fixes a pre-existing collision at register cols on trace rows).
        ctx_reg_unique_positions: bool = False,
        # V13.2 explore 11: force _sample_traj_ids to return all-zeros.
        # All entities then share traj_id=0 -> (a) identical traj_pe slot,
        # (b) zero entity phase under rope_mode=2d_time_trajid. Intentional
        # dual ablation -- do NOT try to disentangle the two effects.
        constant_traj_id: bool = False,
        # V13.2 explore 13/16: per-layer feat fusion MLP. When True,
        # after each attention step, broadcast feat (B,P,feat_dim) to (B,S,P,feat_dim),
        # concat with X (B,S,P,D1), run a 2-layer MLP, add residual to X only.
        # Registers (ra, rt, rp) are untouched. Adds ~9.4M params @ D1=512, feat_dim=1024.
        feat_fusion_mlp: bool = False,
        # V13.3: unidirectional ctx attention mask. When True, ctx queries
        # cannot attend to trace-X keys (masked out), but X queries CAN attend
        # to ctx keys. ctx still attends to registers and add_tokens.
        ctx_unidirectional_mask: bool = False,
        # V13.7: extend add_token projector registration + injection to
        # additional layer types beyond the default {q, s, a}. Only "t" and "p"
        # are supported (both project to D1, same as a-branch). Default ""
        # preserves byte-identical behavior. Each character in the string
        # activates that layer-type branch to consume add_tokens via its own
        # projector entry in `add_token_projs`.
        extra_add_token_layers: str = "",
        # --- V13.8 Feature A ---
        # When N > 0, project the first N language add_tokens and CONCATENATE
        # the result into cond_parts before _build_C_raw. Mirrors lt2.8 umm_v9
        # moma_v2.MotionMachine (L295-307 init, L508-523 forward). Default -1
        # = OFF preserves byte-identical behavior for all pre-v13.8 configs.
        add_first_n_add_token_to_feat: int = -1,
    ):
        super().__init__()
        self.D1, self.D2, self.Rt, self.Rp = D1, D2, Rt, Rp
        self.input_dim = input_dim
        self.time_multiplier, self.n_traj_tokens = time_multiplier, n_traj_tokens
        self.layer_type_string = layer_type_string
        self.n_layers = len(layer_type_string)
        # V13.2 V5: stored per-instance so forward builders can branch.
        self.ctx_reg_unique_positions = bool(ctx_reg_unique_positions)
        # V13.3: unidirectional ctx attention mask.
        self.ctx_unidirectional_mask = bool(ctx_unidirectional_mask)
        # V13.7: extra add-token layer types (subset of "tp").
        self.extra_add_token_layers = extra_add_token_layers or ""
        assert all(c in "tp" for c in self.extra_add_token_layers), (
            f"extra_add_token_layers must only contain chars in 'tp', "
            f"got {self.extra_add_token_layers!r}"
        )

        # V13: removed the `layer_type_string[0] == "t"` constraint so pure-"a"
        # schedules (mimicking lt5.4's "aaaaaaaaa") can be used.
        assert self.D1 % n_heads == 0, (
            f"D1={self.D1} must be divisible by n_heads={n_heads}"
        )
        for ch in layer_type_string:
            assert ch in "tpsqa"

        # ---- V13 RoPE mode ----
        assert rope_mode in ("1d_time", "2d_time_trajid"), (
            f"rope_mode must be '1d_time' or '2d_time_trajid', got {rope_mode!r}"
        )
        self.rope_mode = rope_mode
        if rope_mode == "2d_time_trajid":
            _head_dim = self.D1 // n_heads
            assert _head_dim % 4 == 0, (
                f"2d_time_trajid RoPE needs head_dim={_head_dim} divisible by 4"
            )
            assert n_traj_tokens > 0, (
                "2d_time_trajid requires n_traj_tokens > 0 (entity-id range)"
            )
        # Default formula: spread entity phase over ~16 radians (similar to
        # the typical time phase range for langtable). User can override.
        if entity_rope_multiplier and entity_rope_multiplier > 0.0:
            self.entity_rope_multiplier = float(entity_rope_multiplier)
        else:
            self.entity_rope_multiplier = (
                16.0 / max(int(n_traj_tokens), 1)
                if n_traj_tokens > 0 else 0.0
            )
        if rope_mode == "2d_time_trajid":
            logger.warning(
                f"V13 2D RoPE enabled (rope_mode={rope_mode}, "
                f"entity_rope_multiplier={self.entity_rope_multiplier:.6f}, "
                f"n_traj_tokens={n_traj_tokens}, head_dim={self.D1 // n_heads})"
            )

        # ---- V13.2 LT explore flag validation + storage ----
        assert traj_id_sampler in ("permN", "permP"), (
            f"traj_id_sampler must be 'permN' or 'permP', got {traj_id_sampler!r}"
        )
        assert traj_pe_init in ("same", "std"), (
            f"traj_pe_init must be 'same' or 'std', got {traj_pe_init!r}"
        )
        self.traj_id_sampler = traj_id_sampler
        self.traj_pe_init = traj_pe_init
        self.disable_trajpe = bool(disable_trajpe)
        self.constant_traj_id = bool(constant_traj_id)
        self.feat_fusion_mlp = bool(feat_fusion_mlp)
        if self.constant_traj_id:
            # V13.2 explore 11: the flag only has a meaningful effect
            # when `_sample_traj_ids` is actually called (n_traj_tokens > 0 and
            # no externally-supplied learned_traj_ids). Guard at init to catch
            # conflicting-flag config errors early.
            assert n_traj_tokens > 0, (
                "constant_traj_id=True requires n_traj_tokens > 0 "
                "(else _sample_traj_ids is never called and the flag is a no-op)"
            )
            logger.warning(
                "V13.2 explore 11: constant_traj_id=True -- all entities "
                "share traj_id=0. Dual ablation: identical traj_pe slot + zero "
                "entity RoPE phase (under rope_mode=2d_time_trajid)."
            )
        if self.disable_trajpe:
            assert rope_mode == "2d_time_trajid", (
                "disable_trajpe=True requires rope_mode='2d_time_trajid' "
                "(else the model loses all entity-id signal)"
            )
            assert n_traj_tokens > 0, (
                "disable_trajpe=True requires n_traj_tokens > 0 "
                "(so the Embedding still exists for shape consistency)"
            )
            logger.warning(
                "V13.2 V2: disable_trajpe=True — traj_pe contribution "
                "replaced by zero tensor, traj_pe.weight frozen."
            )

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
            # V13: learnable trajectory-ID embedding (lt5.4 `num_random_trace_id_pe`
            # recipe). Default init = all slots = slot 0 (`traj_pe_init="same"`),
            # forcing the model to break symmetry via language + semantics.
            # V13.2 V7: `traj_pe_init="std"` uses a standard small-normal
            # init instead (N(0, 0.02)) — a DiT-style init (NOT Embedding's
            # default N(0,1), which is too large at D1=512).
            self.traj_pe = nn.Embedding(n_traj_tokens, D1)
            if self.traj_pe_init == "same":
                with torch.no_grad():
                    self.traj_pe.weight.copy_(
                        self.traj_pe.weight[0].expand_as(self.traj_pe.weight)
                    )
                logger.warning(
                    f"traj_pe: learnable Embedding({n_traj_tokens}, {D1}) "
                    f"with all-same init (V13 default)"
                )
            else:  # "std"
                nn.init.normal_(self.traj_pe.weight, mean=0.0, std=0.02)
                logger.warning(
                    f"traj_pe: learnable Embedding({n_traj_tokens}, {D1}) "
                    f"with N(0, 0.02) std init (V13.2 V7)"
                )
            # V13.2 V2: freeze traj_pe.weight when disable_trajpe is on.
            if self.disable_trajpe:
                self.traj_pe.weight.requires_grad = False

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

        # V13.8 Feature A: project first N lang add_tokens and append
        # to cond_parts (widens binder input by D1). Stored unconditionally so
        # forward can branch safely; modules built only when the flag is on.
        self.add_first_n_add_token_to_feat = add_first_n_add_token_to_feat
        if self.add_first_n_add_token_to_feat > 0:
            self.add_to_feat_fc1 = nn.Sequential(
                nn.Linear(D1, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            self.add_to_feat_fc2 = nn.Linear(
                64 * self.add_first_n_add_token_to_feat, D1
            )
            self.binder_in_dim += D1

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

        # --- per-layer add-token projectors (A/S/Q use add tokens by default; ---
        # V13.7: extra_add_token_layers can extend this to t/p layers).
        self.add_token_projs = nn.ModuleDict()
        extra_chars = set(self.extra_add_token_layers)
        effective_chars = set("qsa") | extra_chars
        for i, c in enumerate(layer_type_string):
            if c in effective_chars:
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

        # V13.2 explore 13/16: per-layer feat fusion MLPs (optional).
        # Last Linear weight initialized with N(0, 0.02) -- DO NOT zero-init
        # (kills upstream gradients).
        if self.feat_fusion_mlp:
            self.feat_fusion_mlps = nn.ModuleList()
            for _ in range(self.n_layers):
                last_linear = nn.Linear(D1, D1)
                nn.init.normal_(last_linear.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last_linear.bias)
                self.feat_fusion_mlps.append(nn.Sequential(
                    nn.Linear(D1 + feat_dim, D1),
                    nn.LeakyReLU(0.1),
                    last_linear,
                ))

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
        if self.constant_traj_id:
            return torch.zeros(B, P, dtype=torch.long, device=device)
        if self.training:
            # V13.2 V3: choose sampling strategy.
            if self.traj_id_sampler == "permP":
                # Entity phase lives in [0, P) rather than [0, n_traj_tokens).
                return torch.stack(
                    [torch.randperm(P, device=device) for _ in range(B)]
                )
            # Default permN: randperm(n_traj_tokens)[:P]
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
        """Build time-only RoPE grid (B, Rt+S, Rp+P).

        Default layout (``ctx_reg_unique_positions=False``, pre-V5):
            Register and CTX rows = 0. Trace rows (rows Rt+N_CTX..Rt+S-1)
            on entity cols = ``time_scaled[:, j]``; register cols on trace
            rows also stayed 0 (a pre-existing collision).

        V13.2 V5 layout (``ctx_reg_unique_positions=True``, per-sample):
            - Rt register rows (k ∈ [0, Rt)), all cols:
                ``t_min[b] - (Rt - k) - N_CTX``
            - CTX rows (k_ctx ∈ [0, N_CTX), grid row Rt+k_ctx), all cols:
                ``t_min[b] - (N_CTX - k_ctx)``
            - Trace rows (grid row Rt+N_CTX+j, j ∈ [0, T)), all cols:
                ``time_scaled[b, j]`` — propagates row time into BOTH the
                Rp register cols and the entity cols, fixing the
                pre-existing register-col collision.

            where ``t_min[b] = time_scaled[b].min()``. All phases are
            computed per-sample; each batch element has its own shift.
            Rt + CTX register rows are offset strictly below the real time
            range, so register rows can never collide with real time steps.
        """
        Rt, Rp = self.Rt, self.Rp
        device = time_scaled.device
        dtype = time_scaled.dtype
        R = torch.zeros(B, Rt + S, Rp + P, device=device, dtype=dtype)

        if not self.ctx_reg_unique_positions:
            # Pre-V5 back-compat path: only trace rows on entity cols carry
            # time; register rows / register cols / ctx rows stay zero.
            R[:, Rt + N_CTX:, Rp:] = time_scaled[:, :, None].expand(-1, -1, P)
            return R

        # V5 path. time_scaled: (B, T)  where T = S - N_CTX.
        t_min = time_scaled.min(dim=1).values  # (B,)

        # Rt register rows: time phase = t_min - (Rt - k) - N_CTX.
        # k = Rt-1 -> t_min - 1 - N_CTX ; k = 0 -> t_min - Rt - N_CTX.
        # Broadcast across all Rp+P cols.
        for k in range(Rt):
            row_phase = t_min - float(Rt - k) - float(N_CTX)  # (B,)
            R[:, k, :] = row_phase[:, None]

        # CTX rows: grid row index = Rt + k_ctx, k_ctx in [0, N_CTX).
        # time phase = t_min - (N_CTX - k_ctx). k_ctx = N_CTX-1 -> t_min-1.
        for k_ctx in range(N_CTX):
            row_phase = t_min - float(N_CTX - k_ctx)  # (B,)
            R[:, Rt + k_ctx, :] = row_phase[:, None]

        # Trace rows: grid row index = Rt + N_CTX + j, j in [0, T).
        # ALL cols (Rp register cols + entity cols) get row time.
        R[:, Rt + N_CTX:, :] = time_scaled[:, :, None].expand(-1, -1, Rp + P)

        return R

    @torch.compiler.disable
    def _build_R_entity(
        self, B: int, S: int, P: int, traj_ids: torch.Tensor
    ) -> torch.Tensor:
        """V13: build the entity-id RoPE phase grid (B, Rt+S, Rp+P).

        Layout — phases by grid region:
            - register columns (cols 0..Rp-1, all rows):           phase 0
              i.e. ``ra`` corner and ``rp`` strip get phase 0.
            - entity columns (cols Rp..Rp+P-1, all rows):
                * context + trace rows: ``traj_ids[b, p] * scale``
                * ``rt`` register rows (rows 0..Rt-1):
                  ALSO ``traj_ids[b, p] * scale`` (broadcast across rt rows).
                  This is intentional: it lets ``rt`` registers see entity-p's
                  phase so register-to-entity attention encodes per-entity
                  identity. Only the ``ra/rp`` register *columns* are phase 0.

        Args:
            traj_ids: (B, P) long or float — random-permuted trajectory IDs
                from ``_sample_traj_ids`` (long) or learned traj_ids from
                ``input_xattn_v2`` (float in [0, K]).

        Returns:
            (B, Rt+S, Rp+P) float — phases for the second RoPE axis, already
            scaled by ``self.entity_rope_multiplier``.
        """
        Rt, Rp = self.Rt, self.Rp
        device = traj_ids.device
        dtype = torch.float32
        scale = float(self.entity_rope_multiplier)
        R_e = torch.zeros(B, Rt + S, Rp + P, device=device, dtype=dtype)
        # ent_phase: (B, P).  Broadcast across ALL (Rt+S) rows of the entity
        # columns. Context AND trace rows share the same per-entity phase so
        # attention can bind goal↔trace through matching entity-id. The Rt
        # register rows also receive entity phase intentionally — see docstring.
        # V5 INTENTIONALLY preserves this entity binding on CTX/Rt/trace rows;
        # only the Rp REGISTER COLUMNS get unique out-of-range phases.
        ent_phase = traj_ids.float() * scale
        R_e[:, :, Rp:] = ent_phase[:, None, :].expand(-1, Rt + S, -1)

        if not self.ctx_reg_unique_positions:
            return R_e

        # V5 path: Rp register cols get unique per-sample entity phases
        # strictly outside the real entity range.
        #   col k in [0, Rp) -> entity phase = e_max[b] + (k+1) * scale
        # e_max is per-sample max of the ent_phase tensor above.
        e_max = ent_phase.max(dim=1).values  # (B,)
        for k in range(Rp):
            col_phase = e_max + float(k + 1) * scale  # (B,)
            R_e[:, :, k] = col_phase[:, None]  # broadcast across all (Rt+S) rows

        return R_e

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
    def _mask_to_attn_unidirectional(
        valid: torch.Tensor,
        is_ctx: torch.Tensor,
        is_trace_x: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric attention mask: ctx queries cannot attend to trace-X keys.

        Args:
            valid: (batch, seq) bool -- per-token validity
            is_ctx: (batch, seq) bool -- True for ctx positions
            is_trace_x: (batch, seq) bool -- True for trace-X positions

        Returns:
            (batch, 1, seq, seq) bool attention mask (True=valid, False=excluded)
        """
        attn = valid[:, None, :] & valid[:, :, None]  # (batch, seq, seq)
        # Block: ctx queries attending to trace-X keys
        block_mask = is_ctx[:, :, None] & is_trace_x[:, None, :]  # (B, seq, seq)
        attn = attn & ~block_mask
        # Preserve diagonal: self-attention always valid
        diag = torch.arange(attn.shape[-1], device=attn.device)
        attn[:, diag, diag] = True
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
        R: torch.Tensor,  # (B, Rt+S, Rp+P)  -- time RoPE phase
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        add_tokens: torch.Tensor,  # (B, A, D1)
        add_mask: torch.Tensor,  # (B, A) bool
        add_cond: torch.Tensor,  # (B, A, cond_d1_eff) learnable lang/img conditioning
        R_e: Optional[torch.Tensor] = None,  # V13: (B, Rt+S, Rp+P) entity RoPE phase
        N_CTX: int = 0,  # V13.3: number of ctx timesteps (for unidirectional mask)
        unknown_x_mask: Optional[torch.Tensor] = None,  # V13.3.4: (B, S, P) dynamic mask
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
        S = n_rows - Rt  # V13.3.4: computed once for all layer types
        P = n_cols - Rp

        # V13: 2D RoPE entity phase (R_e) only benefits A-layers (global).
        # T-layers attend per-column: all tokens share the same entity, so
        # entity RoPE adds a constant phase offset (no effect on attention).
        # P-layers use no RoPE at all (t=None), so entity RoPE is N/A.

        if lt == "t":  # column-wise
            seq = Z.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D)       # (B*n_cols, n_rows, D1)
            cond = C.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, D_cond)  # (B*n_cols, n_rows, D_cond)
            valid = M.permute(0, 2, 1).reshape(B * n_cols, n_rows)            # (B*n_cols, n_rows)
            t_rope = R[:, Rt:, :].permute(0, 2, 1).reshape(B * n_cols, S)     # (B*n_cols, S)

            # V13.7: optional add_token injection for t-layer. add_tokens
            # are INSERTED AFTER the Rt register rows (positions [Rt:Rt+A]),
            # NOT prepended at the front. This preserves DiTBlockDualFFN's FFN
            # split: the block splits at self.n_reg=Rt to route register rows
            # through mlp_reg. Prepending at the front would shift that split
            # and send add_tokens (not registers) through the register FFN --
            # silent semantic corruption.
            # Resulting layout per column: [Rt regs, A add_tokens, S trace].
            # num_external_regs = Rt + A so RoPE is applied only to the S trace
            # positions (t_rope.shape = (B*n_cols, S)).
            inject_add_t = f"layer_{layer_idx}" in self.add_token_projs
            if inject_add_t:
                A = add_tokens.shape[1]
                tok = self.add_token_projs[f"layer_{layer_idx}"](add_tokens)   # (B, A, D1)
                tok_bc = tok.unsqueeze(1).expand(B, n_cols, A, D).reshape(B * n_cols, A, D)                      # (B*n_cols, A, D1)
                cond_bc = add_cond.unsqueeze(1).expand(B, n_cols, A, D_cond).reshape(B * n_cols, A, D_cond)      # (B*n_cols, A, D_cond)
                mask_bc = add_mask.unsqueeze(1).expand(B, n_cols, A).reshape(B * n_cols, A)                      # (B*n_cols, A)
                seq = torch.cat([seq[:, :Rt], tok_bc, seq[:, Rt:]], dim=1)         # (B*n_cols, Rt+A+S, D1)
                cond = torch.cat([cond[:, :Rt], cond_bc, cond[:, Rt:]], dim=1)      # (B*n_cols, Rt+A+S, D_cond)
                valid = torch.cat([valid[:, :Rt], mask_bc, valid[:, Rt:]], dim=1)   # (B*n_cols, Rt+A+S)
                num_ext = A + Rt  # Rt register rows + A add_tokens (all non-RoPE)
            else:
                A = 0
                num_ext = Rt

            if unknown_x_mask is not None and N_CTX > 0:
                # V13.3.4: Dynamic conditional mask in column space.
                # unknown_x_mask (B, S, P) -> grid (B, n_rows, n_cols) -> column-wise
                is_unknown = torch.zeros(B, n_rows, n_cols, dtype=torch.bool, device=valid.device)
                is_unknown[:, Rt:Rt + S, Rp:Rp + P] = unknown_x_mask  # (B, S, P)
                is_unknown_col = is_unknown.permute(0, 2, 1).reshape(B * n_cols, n_rows)   # (B*n_cols, n_rows)
                # V13.3.4: ctx indicator (same as static path)
                col_is_entity = torch.arange(n_cols, device=valid.device) >= Rp  # (n_cols,)
                col_mask = col_is_entity[None, :].expand(B, -1).reshape(B * n_cols)  # (B*n_cols,)
                is_ctx = torch.zeros(B * n_cols, n_rows, dtype=torch.bool, device=valid.device)
                is_ctx[:, Rt:Rt + N_CTX] = True
                is_ctx = is_ctx & col_mask[:, None]  # zero out register columns
                # V13.7: insert A False entries at position Rt so mask aligns
                # with the [Rt regs, A add_tokens, S trace] seq layout.
                if A > 0:
                    add_zeros = torch.zeros(B * n_cols, A, dtype=torch.bool, device=valid.device)
                    is_unknown_col = torch.cat([is_unknown_col[:, :Rt], add_zeros, is_unknown_col[:, Rt:]], dim=1)  # (B*n_cols, Rt+A+S)
                    is_ctx = torch.cat([is_ctx[:, :Rt], add_zeros, is_ctx[:, Rt:]], dim=1)                          # (B*n_cols, Rt+A+S)
                attn_mask = self._mask_to_attn_unidirectional(valid, is_ctx, is_unknown_col)
            elif self.ctx_unidirectional_mask and N_CTX > 0:
                # T-layer column: [Rt regs, N_CTX ctx, T trace] per column.
                # Only entity columns (>= Rp) get ctx/trace labels.
                # Register columns (< Rp) stay neutral (symmetric attention).
                col_is_entity = torch.arange(n_cols, device=valid.device) >= Rp  # (n_cols,)
                col_mask = col_is_entity[None, :].expand(B, -1).reshape(B * n_cols)  # (B*n_cols,)
                is_ctx = torch.zeros(B * n_cols, n_rows, dtype=torch.bool, device=valid.device)
                is_ctx[:, Rt:Rt + N_CTX] = True
                is_ctx = is_ctx & col_mask[:, None]  # zero out register columns
                is_trace_x = torch.zeros(B * n_cols, n_rows, dtype=torch.bool, device=valid.device)
                is_trace_x[:, Rt + N_CTX:] = True
                is_trace_x = is_trace_x & col_mask[:, None]  # zero out register columns
                # V13.7: insert A False entries at position Rt so mask aligns
                # with the [Rt regs, A add_tokens, S trace] seq layout.
                if A > 0:
                    add_zeros = torch.zeros(B * n_cols, A, dtype=torch.bool, device=valid.device)
                    is_ctx = torch.cat([is_ctx[:, :Rt], add_zeros, is_ctx[:, Rt:]], dim=1)                # (B*n_cols, Rt+A+S)
                    is_trace_x = torch.cat([is_trace_x[:, :Rt], add_zeros, is_trace_x[:, Rt:]], dim=1)    # (B*n_cols, Rt+A+S)
                attn_mask = self._mask_to_attn_unidirectional(valid, is_ctx, is_trace_x)
            else:
                attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=t_rope,
                num_external_regs=num_ext,
                attn_mask=attn_mask,
            )
            # V13.7: splice out the A add_token slice inserted at [Rt:Rt+A]
            # so the output matches the pre-patch (B*n_cols, Rt+S, D) shape.
            if A > 0:
                out = torch.cat([out[:, :Rt], out[:, Rt + A:]], dim=1)  # (B*n_cols, Rt+S, D1)
            out = out.reshape(B, n_cols, n_rows, D).permute(0, 2, 1, 3)

        elif lt == "p":  # row-wise, no add tokens by default
            seq = Z.reshape(B * n_rows, n_cols, D)        # (B*n_rows, n_cols, D1)
            cond = C.reshape(B * n_rows, n_cols, D_cond)  # (B*n_rows, n_cols, D_cond)
            valid = M.reshape(B * n_rows, n_cols)         # (B*n_rows, n_cols)

            # V13.7: optional add_token injection for p-layer. add_tokens
            # are INSERTED AFTER the Rp register cols (positions [Rp:Rp+A]),
            # NOT prepended at the front. This preserves DiTBlockDualFFN's FFN
            # split at self.n_reg=Rp -- register cols go through mlp_reg.
            # Resulting layout per row: [Rp regs, A add_tokens, P entities].
            inject_add_p = f"layer_{layer_idx}" in self.add_token_projs
            if inject_add_p:
                A = add_tokens.shape[1]
                tok = self.add_token_projs[f"layer_{layer_idx}"](add_tokens)   # (B, A, D1)
                tok_bc = tok.unsqueeze(1).expand(B, n_rows, A, D).reshape(B * n_rows, A, D)                  # (B*n_rows, A, D1)
                cond_bc = add_cond.unsqueeze(1).expand(B, n_rows, A, D_cond).reshape(B * n_rows, A, D_cond)  # (B*n_rows, A, D_cond)
                mask_bc = add_mask.unsqueeze(1).expand(B, n_rows, A).reshape(B * n_rows, A)                  # (B*n_rows, A)
                seq = torch.cat([seq[:, :Rp], tok_bc, seq[:, Rp:]], dim=1)         # (B*n_rows, Rp+A+P, D1)
                cond = torch.cat([cond[:, :Rp], cond_bc, cond[:, Rp:]], dim=1)      # (B*n_rows, Rp+A+P, D_cond)
                valid = torch.cat([valid[:, :Rp], mask_bc, valid[:, Rp:]], dim=1)   # (B*n_rows, Rp+A+P)
                num_ext = Rp + A  # Rp register cols + A add_tokens (both non-RoPE; irrelevant since t=None)
            else:
                A = 0
                num_ext = 0

            # NOTE: P-layer is exempt from unidirectional masking. Each row is
            # a separate attention computation. Within a single row, all entity
            # tokens share the same type (all ctx OR all trace), so block_mask
            # (ctx_query & trace_key) is always all-False -- a no-op.
            attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=None,
                num_external_regs=num_ext,
                attn_mask=attn_mask,
            )
            # V13.7: splice out the A add_token slice inserted at [Rp:Rp+A]
            # so the output matches the pre-patch (B*n_rows, Rp+P, D) shape.
            if A > 0:
                out = torch.cat([out[:, :Rp], out[:, Rp + A:]], dim=1)  # (B*n_rows, Rp+P, D1)
            out = out.reshape(B, n_rows, n_cols, D)

        else:  # a: global -- flatten entire grid, RoPE + add tokens
            N = n_rows * n_cols
            seq = Z.reshape(B, N, D)
            cond = C.reshape(B, N, D_cond)
            valid = M.reshape(B, N)
            t_rope = R.reshape(B, N)
            # V13: optional second-axis RoPE (entity / traj_id phase).
            e_rope = R_e.reshape(B, N) if R_e is not None else None
            A = add_tokens.shape[1]
            tok = self.add_token_projs[f"layer_{layer_idx}"](add_tokens)
            seq = torch.cat([tok, seq], dim=1)
            cond = torch.cat([add_cond, cond], dim=1)
            valid = torch.cat([add_mask, valid], dim=1)
            if unknown_x_mask is not None and N_CTX > 0:
                # V13.3.4: Dynamic conditional mask in flattened global space.
                # unknown_x_mask (B, S, P) -> grid (B, n_rows, n_cols) -> flatten (B, N)
                is_unknown_grid = torch.zeros(B, n_rows, n_cols, dtype=torch.bool, device=valid.device)
                is_unknown_grid[:, Rt:Rt + S, Rp:Rp + P] = unknown_x_mask  # (B, S, P)
                is_unknown_flat = is_unknown_grid.reshape(B, N)  # (B, N)
                # V13.3.4: prepend zeros for add_tokens (neutral)
                add_zeros = torch.zeros(B, A, dtype=torch.bool, device=valid.device)
                is_unknown_x = torch.cat([add_zeros, is_unknown_flat], dim=1)  # (B, A+N)
                # V13.3.4: ctx indicator (same grid logic as static path)
                row_idx = torch.arange(n_rows, device=valid.device)
                col_idx = torch.arange(n_cols, device=valid.device)
                row_is_ctx = (row_idx >= Rt) & (row_idx < Rt + N_CTX)  # (n_rows,)
                col_is_entity = col_idx >= Rp  # (n_cols,)
                grid_is_ctx = (row_is_ctx[:, None] & col_is_entity[None, :]).reshape(N)
                is_ctx = torch.cat([torch.zeros(A, device=valid.device, dtype=torch.bool), grid_is_ctx]).unsqueeze(0).expand(B, -1)  # (B, A+N)
                attn_mask = self._mask_to_attn_unidirectional(valid, is_ctx, is_unknown_x)
            elif self.ctx_unidirectional_mask and N_CTX > 0:
                # A-layer global: flattened grid with add_tokens prepended.
                # Grid layout (row-major): (Rt+S) rows x (Rp+P) cols flattened to N.
                # add_tokens (first A positions) are neutral.
                # Grid cells:
                #   row < Rt or col < Rp -> register (neutral)
                #   row in [Rt, Rt+N_CTX) & col >= Rp -> ctx
                #   row >= Rt+N_CTX & col >= Rp -> trace
                row_idx = torch.arange(n_rows, device=valid.device)
                col_idx = torch.arange(n_cols, device=valid.device)
                row_is_ctx = (row_idx >= Rt) & (row_idx < Rt + N_CTX)  # (n_rows,)
                row_is_trace = row_idx >= Rt + N_CTX  # (n_rows,)
                col_is_entity = col_idx >= Rp  # (n_cols,)
                # Build grid indicators: (n_rows, n_cols) -> flatten to (N,)
                grid_is_ctx = (row_is_ctx[:, None] & col_is_entity[None, :]).reshape(N)
                grid_is_trace = (row_is_trace[:, None] & col_is_entity[None, :]).reshape(N)
                # Prepend A zeros for add_tokens (neutral)
                add_zeros = torch.zeros(A, device=valid.device, dtype=torch.bool)
                is_ctx = torch.cat([add_zeros, grid_is_ctx]).unsqueeze(0).expand(B, -1)  # (B, A+N)
                is_trace_x = torch.cat([add_zeros, grid_is_trace]).unsqueeze(0).expand(B, -1)  # (B, A+N)
                attn_mask = self._mask_to_attn_unidirectional(valid, is_ctx, is_trace_x)
            else:
                attn_mask = self._mask_to_attn(valid)
            out = self.blocks[layer_idx](
                x=seq,
                cond=cond,
                t=t_rope,
                num_external_regs=A,
                attn_mask=attn_mask,
                e=e_rope,
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
        learned_traj_ids: Optional[torch.Tensor] = None,   # V13.2 Explore 10: (B, P) float
        unknown_x_mask: Optional[torch.Tensor] = None,     # V13.3.4: (B, S, P) dynamic mask
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
        # V13: sample traj_ids ONCE (shared between learnable traj_pe cond
        # and the optional 2D RoPE entity-phase axis). When n_traj_tokens=0
        # both paths are skipped.
        traj_ids = None
        if self.n_traj_tokens > 0:
            if learned_traj_ids is not None:
                # V13.2 Explore 10: use externally-provided learned traj_ids
                # (float). These come from input_xattn_v2's sigmoid*K head.
                # Skip _sample_traj_ids() -- learned_traj_ids already has the
                # right values.
                # V13.2: constant_traj_id=True is incompatible with
                # learned_traj_ids (_sample_traj_ids is never called, so the
                # flag is a silent no-op -- a config error).
                assert not self.constant_traj_id, (
                    "constant_traj_id=True is incompatible with learned_traj_ids "
                    "(the flag is a silent no-op because _sample_traj_ids is bypassed)"
                )
                traj_ids = learned_traj_ids  # (B, P) float
            else:
                traj_ids = self._sample_traj_ids(B, P, device)  # (B, P) long

            if self.disable_trajpe:
                traj_pe = torch.zeros(
                    B, P, self.D1, device=device, dtype=feat_exp.dtype
                )
            else:
                # Can't use Embedding with float traj_ids
                assert learned_traj_ids is None, (
                    "learned_traj_ids requires disable_trajpe=True "
                    "(can't index Embedding with floats)"
                )
                traj_pe = self.traj_pe(traj_ids)  # (B, P, D1)
            cond_parts.append(self.hard_ln_pe(traj_pe[:, None].expand(-1, S, -1, -1)))

        # V13.8 Feature A: project first N lang tokens and append to
        # cond_parts. Mirrors lt2.8 umm_v9 moma_v2.MotionMachine L508-523.
        # Silently skipping when lang_tokens is None would narrow cond_parts
        # and break the downstream cond_binders matmul (they are sized to the
        # widened binder_in_dim). Fail loud instead.
        if self.add_first_n_add_token_to_feat > 0:
            assert lang_tokens is not None, (
                "add_first_n_add_token_to_feat > 0 requires lang_tokens "
                "(add_tokens arg) to be provided every forward call; got None"
            )
            N_add = self.add_first_n_add_token_to_feat
            assert lang_tokens.shape[1] >= N_add, (
                f"add_first_n_add_token_to_feat={N_add} but lang_tokens "
                f"has only {lang_tokens.shape[1]} tokens"
            )
            _add_feat = lang_tokens[:, :N_add].contiguous()  # (B, N, D1)
            _add_feat = self.add_to_feat_fc1(_add_feat)      # (B, N, 64)
            _add_feat = self.add_to_feat_fc2(
                _add_feat.reshape(B, -1)
            )                                                # (B, D1)
            _add_feat = _add_feat[:, None, None].expand(
                -1, S, P, -1
            ).contiguous()                                   # (B, S, P, D1)
            cond_parts.append(self.hard_ln_pe(_add_feat))    # V13.8 Feature A

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

        # ---- 4b. V13 entity RoPE grid (only when 2D RoPE is enabled) ----
        # If traj_ids is None (n_traj_tokens=0) we cannot do 2D RoPE; the
        # init-time assert ensures rope_mode==2d_time_trajid implies n_traj_tokens>0.
        R_e = None
        if self.rope_mode == "2d_time_trajid" and traj_ids is not None:
            R_e = self._build_R_entity(B, S, P, traj_ids)  # (B, Rt+S, Rp+P)

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
                    lt, ra, rt, rp, X, C, R, M, i,
                    add_tokens, add_mask, add_cond_d1,
                    R_e=R_e,  # V13: 2D-RoPE entity phase (None when 1d_time)
                    N_CTX=N_CTX,  # V13.3: for unidirectional mask
                    unknown_x_mask=unknown_x_mask,  # V13.3.4: dynamic mask
                )
            else:  # s, q
                ra, rt, rp, X = self._step_register(
                    lt, ra, rt, rp, X, C, M, i, add_tokens, add_mask, add_cond_d2
                )

            # V13.2 explore 13/16: optional per-layer feat fusion.
            # Residual on X only; registers (ra, rt, rp) untouched.
            if self.feat_fusion_mlp:
                feat_b = feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, feat_dim)
                x_with_feat = torch.cat([X, feat_b], dim=-1)   # (B, S, P, D+feat_dim)
                X = X + self.feat_fusion_mlps[i](x_with_feat)

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
