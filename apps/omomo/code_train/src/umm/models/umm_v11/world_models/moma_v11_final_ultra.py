"""MotionMachine V11 Final (Manual).

Core: AugGridLatent (torch.Tensor subclass) of shape (B, Rt+S, Rp+P, D).
Four grids flow through the network:
  Z   -- latent state grid, updated by each layer
  C   -- adaLN conditioning grid, projected from C_raw + enriched per layer
  R   -- RoPE time grid, built once, time-only (no spatial RoPE)
  M   -- validity mask grid, built once

Per-layer pre-attention fusion:
  F   -- bidirectional register<->X fusion grid (per layer, not stored)
  Z   = Z + fc(cat([Z, Z0, F]))   -- Z0 highway + register info injected
  C   = C + fc(cat([C, F]))        -- adaLN enriched with register state
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.wm_utils import HardLayerNorm
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly

logger = logging.getLogger(__name__)


# ======================================================================
# AugGridLatent -- torch.Tensor subclass, native value = Z grid
# ======================================================================


class AugGridLatent(torch.Tensor):
    """Grid tensor (B, Rt+S, Rp+P, D) with named region accessors.

      Grid layout (registers top-left, primary X bottom-right):

                Rp cols          P entity cols
              +-------------+--------------------+
    Rt rows   | reg_a       | reg_t              |   [:, :Rt, :Rp]   [:, :Rt, Rp:]
              | (B,Rt,Rp,D) | (B,Rt,P,D)         |
              +-------------+--------------------+
    S rows    | reg_p       | X = cat(ctx,x)     |   [:, Rt:, :Rp]   [:, Rt:, Rp:]
              | (B,S,Rp,D)  | (B,S,P,D)          |
              +-------------+--------------------+
    """

    Rt: int = 0
    Rp: int = 0
    CTX: int = 0

    @staticmethod
    def __new__(cls, data: torch.Tensor, Rt: int, Rp: int, CTX: int):
        obj = data.as_subclass(cls)
        obj.Rt = Rt
        obj.Rp = Rp
        obj.CTX = CTX
        return obj

    def __deepcopy__(self, memo):
        """Deepcopy returns a plain torch.Tensor (drops subclass metadata).

        Required because PyTorch Lightning deepcopies callback_metrics; if a
        subclassed tensor leaks into metrics, default deepcopy raises.
        """
        return self.detach().clone().as_subclass(torch.Tensor)

    @classmethod
    @torch.compiler.disable
    def build(cls, ctx_feat, x_feat, reg_t_feat, reg_p_feat, reg_a_feat, Rt, Rp):
        """Assemble Z grid from 5 named components."""
        CTX = ctx_feat.shape[1]
        X = torch.cat([ctx_feat, x_feat], dim=1)  # (B, S, P, D)
        top = torch.cat([reg_a_feat, reg_t_feat], dim=2)  # (B, Rt, Rp+P, D)
        bot = torch.cat([reg_p_feat, X], dim=2)  # (B, S, Rp+P, D)
        Z = torch.cat([top, bot], dim=1)  # (B, Rt+S, Rp+P, D)
        return cls(Z, Rt=Rt, Rp=Rp, CTX=CTX)

    # --- named region accessors (read-only slices) ---
    @property
    def reg_a(self):
        return self[:, : self.Rt, : self.Rp, :]  # (B, Rt, Rp, D)

    @property
    def reg_t(self):
        return self[:, : self.Rt, self.Rp :, :]  # (B, Rt, P, D)

    @property
    def reg_p(self):
        return self[:, self.Rt :, : self.Rp, :]  # (B, S, Rp, D)

    @property
    def X(self):
        return self[:, self.Rt :, self.Rp :, :]  # (B, S, P, D)

    @property
    def ctx(self):
        return self.X[:, : self.CTX]  # (B, CTX, P, D)

    @property
    def x(self):
        return self.X[:, self.CTX :]  # (B, T, P, D)

    # --- derived dims ---
    @property
    def B(self):
        return self.shape[0]

    @property
    def RtS(self):
        return self.shape[1]  # Rt + S

    @property
    def RpP(self):
        return self.shape[2]  # Rp + P

    @property
    def D(self):
        return self.shape[3]

    @property
    def S(self):
        return self.RtS - self.Rt

    @property
    def P(self):
        return self.RpP - self.Rp

    @property
    def T(self):
        return self.S - self.CTX


# ======================================================================
# Register adaLN+FFN (ported from moma_v11_final.py)
# ======================================================================


class _RegisterAdaLNFFN(nn.Module):
    """adaLN-Zero + FFN for register tokens. One per register region per layer.

    Pattern: norm -> modulate(scale,shift) -> FFN(GELU) -> gate -> residual.
    Zero-init adaLN so output = identity at initialization.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_proj = nn.Linear(dim, 3 * dim)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x, cond: (..., dim). Returns (..., dim) with residual."""
        scale, shift, gate = self.ada_proj(cond).chunk(3, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.ffn(h)
        return x + gate * h


# ======================================================================
# Network
# ======================================================================


class MotionMachineV11Ultra(nn.Module):
    """Unified-grid LMM with per-layer register<->X fusion."""

    def __init__(
        self,
        # --- architecture ---
        latent_dim: int = 512,
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
        # --- narrow T-layer (for tqtq scaling to huge P) ---
        # When t_layer_dim is -1 or == latent_dim: inactive (no projectors, T-blocks at full dim).
        # When t_layer_dim < latent_dim: each T-layer wraps its DiT block in
        #   down_proj(latent_dim -> t_layer_dim) -> narrow DiT -> up_proj(t_layer_dim -> latent_dim)
        #   with a residual add to the post-fusion Z. Z and C each have their own down-projector.
        t_layer_dim: int = -1,
        t_layer_fc_factor: int = -1,        # narrow-block MLP expansion; -1 -> follow fc_factor
        t_layer_fc_n_layers: int = -1,      # narrow-block MLP depth; -1 -> follow fc_n_layers
        t_layer_ada_n_layers: int = -1,     # narrow-block adaLN depth; -1 -> follow ada_n_layers
        # --- flexible skip-layer collection for the output head ---
        # None -> auto-compute as all T-layer positions in layer_type_string.
        # Otherwise: explicit list of absolute layer indices (any type allowed, sorted+deduped).
        skip_layer_idx_list: Optional[list] = None,
        # --- toggles ---
        use_reg_post_ffn: bool = False,  # DEPRECATED post-attention register adaLN+FFN (default off)
        enable_pre_attention_fusion: bool = True,  # When False: disable fusion block + _pre_attention (ablation)
    ):
        super().__init__()
        self.D, self.Rt, self.Rp = latent_dim, Rt, Rp
        self.n_ctx, self.input_dim, self.feat_dim = n_ctx, input_dim, feat_dim
        self.time_multiplier, self.n_traj_tokens = time_multiplier, n_traj_tokens
        self.layer_type_string = layer_type_string
        self.n_layers = len(layer_type_string)
        self.use_reg_post_ffn = use_reg_post_ffn
        self.enable_pre_attention_fusion = enable_pre_attention_fusion

        assert layer_type_string[0] == "t" and self.D % n_heads == 0
        for ch in layer_type_string:
            assert ch in "tpsq"

        self.n_t = sum(1 for c in layer_type_string if c == "t")

        # ---- narrow T-layer resolution ----
        # Inactive iff t_layer_dim == -1 or == latent_dim. When active, T-layer DiT
        # blocks live at t_layer_dim; per-T-layer down/up projectors wrap them.
        self.t_layer_dim = t_layer_dim
        self._narrow_active = (t_layer_dim != -1) and (t_layer_dim != self.D)
        # Resolve narrow-block params unconditionally (harmless when inactive).
        _t_fc_factor = fc_factor if t_layer_fc_factor == -1 else t_layer_fc_factor
        _t_fc_n_layers = fc_n_layers if t_layer_fc_n_layers == -1 else t_layer_fc_n_layers
        _t_ada_n_layers = ada_n_layers if t_layer_ada_n_layers == -1 else t_layer_ada_n_layers
        if self._narrow_active:
            assert 0 < t_layer_dim < self.D, (
                f"t_layer_dim ({t_layer_dim}) must be in (0, latent_dim={self.D})"
            )
            assert t_layer_dim % n_heads == 0, (
                f"t_layer_dim ({t_layer_dim}) must be divisible by n_heads ({n_heads})"
            )

        # ---- skip_layer_idx_list resolution ----
        # Default = all T-layer absolute positions; explicit = validated and sorted.
        if skip_layer_idx_list is None:
            resolved_skip = [i for i, c in enumerate(layer_type_string) if c == "t"]
        else:
            assert all(isinstance(x, int) for x in skip_layer_idx_list), (
                f"skip_layer_idx_list must be list of ints, got {skip_layer_idx_list}"
            )
            assert len(set(skip_layer_idx_list)) == len(skip_layer_idx_list), (
                f"skip_layer_idx_list has duplicates: {skip_layer_idx_list}"
            )
            for idx in skip_layer_idx_list:
                assert 0 <= idx < self.n_layers, (
                    f"skip idx {idx} out of range [0, {self.n_layers})"
                )
            resolved_skip = sorted(skip_layer_idx_list)
        self._skip_layer_idx_list = resolved_skip
        self._skip_layer_idx_set = set(resolved_skip)  # O(1) lookup in forward

        # --- input embedding ---
        self.x_proj = nn.Linear(input_dim, self.D)
        self.hard_ln_feat = HardLayerNorm(feat_dim)
        self.hard_ln_pe = HardLayerNorm(self.D)
        self.ctx_t_emb = nn.Embedding(n_ctx + 1, self.D)
        self.level_pe = Timesteps(
            self.D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.t_pe = Timesteps(
            self.D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.t_proj = nn.Linear(self.D + 1, self.D)
        if n_traj_tokens > 0:
            self.traj_pe = Timesteps(
                self.D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )

        # --- register init (nn.Embedding-style: N(0, 1), not tiny) ---
        self.reg_t_init = nn.Parameter(torch.randn(1, Rt, 1, self.D))
        self.reg_p_init = nn.Parameter(torch.randn(1, 1, Rp, self.D))
        self.reg_a_init = nn.Parameter(torch.randn(1, Rt, Rp, self.D))

        # --- conditioning: one binder per layer, projects C_raw -> self.D ---
        self.cond_dim = feat_dim + 2 * self.D + (self.D if n_traj_tokens > 0 else 0)
        self.cond_binders = nn.ModuleList(
            [nn.Linear(self.cond_dim, self.D) for _ in range(self.n_layers)]
        )

        # --- DiT blocks: one per layer. Heterogeneous when narrow: T-layer entries
        # live at t_layer_dim, all other entries (P/Q/S) live at latent_dim.
        full_dit_kw = dict(
            C=self.D,
            num_heads=n_heads,
            num_registers=0,
            mlp_hidden_dims=[self.D * fc_factor] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.D,
            qk_norm=True,
        )
        # narrow_dit_kw is only built (and used) when narrow mode is active.
        # Guarding prevents an invalid C=-1 kwarg dict from existing in the inactive case.
        if self._narrow_active:
            narrow_dit_kw = dict(
                C=t_layer_dim,
                num_heads=n_heads,
                num_registers=0,
                mlp_hidden_dims=[t_layer_dim * _t_fc_factor] * _t_fc_n_layers,
                ada_scale_bias_bound=ada_scale_bias_bound,
                gate_bound=gate_bound,
                ada_n_layers=_t_ada_n_layers,
                C_feat=t_layer_dim,
                qk_norm=True,
            )
            self.blocks = nn.ModuleList([
                DiTBlockTimeOnly(**(narrow_dit_kw if c == "t" else full_dit_kw))
                for c in layer_type_string
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlockTimeOnly(**full_dit_kw) for _ in layer_type_string
            ])

        # --- per-T-layer down/up projectors (only when narrow active) ---
        # Keyed by absolute layer idx as "layer_{i}" inside a ModuleDict.
        # Each entry is itself a ModuleDict with three Linears: z_down, c_down, up.
        self.t_narrow_projs = nn.ModuleDict()
        if self._narrow_active:
            for i, c in enumerate(layer_type_string):
                if c == "t":
                    self.t_narrow_projs[f"layer_{i}"] = nn.ModuleDict({
                        "z_down": nn.Linear(self.D, t_layer_dim),
                        "c_down": nn.Linear(self.D, t_layer_dim),
                        "up":     nn.Linear(t_layer_dim, self.D),
                    })

        # --- per-TYPE add-token (lang) projectors ---
        # All P-layers share one Linear; same for Q and S. T-layers do NOT see
        # lang tokens (they live on a per-column temporal axis where lang adds
        # no value AND would inflate the B*(Rp+P) batch x A cost at huge P).
        # Only types that actually appear in layer_type_string get a projector.
        _present_types = set(c for c in layer_type_string if c in "pqs")
        self.add_token_projs = nn.ModuleDict({
            c: nn.Linear(self.D, self.D) for c in _present_types
        })

        # --- per-layer register<->X fusion + pre-attention (toggleable) ---
        if self.enable_pre_attention_fusion:
            self.fusions = nn.ModuleList(
                [self._make_fusion_block(self.D, Rt, Rp) for _ in range(self.n_layers)]
            )
        else:
            self.fusions = None

        # --- per-layer register adaLN+FFN (applied AFTER each attention step) ---
        # Toggled by use_reg_post_ffn. When False, no params are created.
        if self.use_reg_post_ffn:
            reg_hidden = self.D * fc_factor
            self.reg_rt_blocks = nn.ModuleList(
                [_RegisterAdaLNFFN(self.D, reg_hidden) for _ in range(self.n_layers)]
            )
            self.reg_rp_blocks = nn.ModuleList(
                [_RegisterAdaLNFFN(self.D, reg_hidden) for _ in range(self.n_layers)]
            )
            self.reg_corner_blocks = nn.ModuleList(
                [_RegisterAdaLNFFN(self.D, reg_hidden) for _ in range(self.n_layers)]
            )
            # per-layer learnable register conditioning (zero-init)
            self.ada_rt_embeds = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, Rt, self.D)) for _ in range(self.n_layers)]
            )
            self.ada_rp_embeds = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, Rp, self.D)) for _ in range(self.n_layers)]
            )
            self.ada_corner_embeds = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, Rt * Rp, self.D)) for _ in range(self.n_layers)]
            )
            logging.warning("MotionMachineV11Ultra: use_reg_post_ffn=True is DEPRECATED and will be removed in a future release. This enables per-layer register adaLN+FFN blocks (matching old code) but adds significant params and compute. Default is False (off).")

        # --- output head (skip connections: resolved skip layers + initial X) ---
        skip_dim = self.D * (len(self._skip_layer_idx_list) + 1)
        layers, c_in = [], skip_dim
        for c_out in np.linspace(skip_dim, input_dim, fc_out_n_layers)[:-1]:
            c_out = int(c_out)
            layers += [nn.Linear(c_in, c_out), HardLayerNorm(c_out), nn.LeakyReLU()]
            c_in = c_out
        layers.append(nn.Linear(c_in, input_dim))
        self.output_proj = nn.Sequential(*layers)

        n_fusion_params = (
            sum(p.numel() for f in self.fusions for p in f.parameters())
            if self.fusions is not None
            else 0
        )
        logger.info(
            f"LMM V11 Ultra: {layer_type_string}  D={self.D}  Rt={Rt} Rp={Rp}  "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.2f}M  "
            f"(fusion={n_fusion_params / 1e6:.2f}M, enabled={self.enable_pre_attention_fusion})"
        )

    # ==================== optim helpers ====================

    def get_optim_groups(
        self, weight_decay: float = 0.0, base_lr: Optional[float] = None
    ):
        """Optimizer groups with weight-decay separation."""
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
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

    # ==================== init helpers ====================

    @staticmethod
    def _small_linear(in_dim: int, out_dim: int) -> nn.Linear:
        """Linear with small weight init (std=0.02), zero bias."""
        fc = nn.Linear(in_dim, out_dim)
        nn.init.normal_(fc.weight, std=0.02)
        nn.init.zeros_(fc.bias)
        return fc

    def _make_fusion_block(self, D: int, Rt: int, Rp: int) -> nn.ModuleDict:
        """Per-layer FCs for register<->X exchange and pre-attention fusion.

        Init: all intermediate FCs default (kaiming). Only the final residual
        gates (`fuse_z2`, `fuse_c2`) are small-init (std=0.02) so initial
        fusion contribution to Z and C is tiny but NOT exactly zero —
        preserves gradient flow for upstream FCs.
        """
        sl = self._small_linear
        return nn.ModuleDict({
            # Register -> X: compress regs to summaries (default init)
            'ra1': nn.Linear(Rp * D, D),    # reg_a: compress Rp -> (B, Rt, D)
            'ra2': nn.Linear(Rt * D, D),    # reg_a: compress Rt -> (B, D) global
            'rt': nn.Linear(Rt * D, D),     # reg_t: compress Rt -> (B, P, D) per-entity
            'rp': nn.Linear(Rp * D, D),     # reg_p: compress Rp -> (B, S, D) per-timestep
            'outer': nn.Linear(3 * D, D),   # fuse reg summaries -> (B, S, P, D)
            # X -> Register: pointnet pool into reg regions (default init)
            'pool_t': nn.Linear(3 * D, D),  # pool X over S -> (B, P, D) -> reg_t
            'pool_p': nn.Linear(3 * D, D),  # pool X over P -> (B, S, D) -> reg_p
            'pool_a': nn.Linear(3 * D, D),  # pool X over S&P -> (B, D) -> reg_a
            # Pre-attention: 2-layer FC for Z0 highway + fusion residual
            'fuse_z1': nn.Linear(3 * D, D),  # cat(Z, Z0, F) -> hidden (default init)
            'fuse_z2': sl(D, D),             # hidden -> Z residual (SMALL, adds to Z)
            'fuse_c1': nn.Linear(2 * D, D),  # cat(C, F) -> hidden (default init)
            'fuse_c2': sl(D, D),             # hidden -> C residual (SMALL, adds to C)
        })

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
        cond: torch.Tensor,    # (B, S, P, cond_dim)
        valid_entities: torch.Tensor,  # (B, P) bool
        valid_times: torch.Tensor,     # (B, S) bool
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
        c_top = (c_bot * t_mask).sum(dim=1, keepdim=True) / t_count  # (B, 1, Rp+P, cond_dim)
        c_top = c_top.expand(-1, Rt, -1, -1)  # (B, Rt, Rp+P, cond_dim)

        return torch.cat([c_top, c_bot], dim=1)  # (B, Rt+S, Rp+P, cond_dim)

    @torch.compiler.disable
    def _build_R(self, Z: AugGridLatent, time_scaled: torch.Tensor) -> torch.Tensor:
        """Build time-only RoPE grid (B, Rt+S, Rp+P). Regs and ctx = 0."""
        R = torch.zeros(
            Z.B, Z.RtS, Z.RpP, device=time_scaled.device, dtype=time_scaled.dtype
        )
        R[:, Z.Rt + Z.CTX :, Z.Rp :] = time_scaled[:, :, None].expand(
            -1, -1, Z.P
        )  # noisy X = timestamps
        return R

    @torch.compiler.disable
    def _build_M(
        self,
        Z: AugGridLatent,
        ctx_mask: torch.Tensor,
        noisy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build validity grid M (B, Rt+S, Rp+P) bool.

        Primary X = per-cell mask. Register regions = valid iff that entity/timestep
        is valid in ANY frame. Corner reg_a = always True.
        """
        M = torch.ones(Z.B, Z.RtS, Z.RpP, device=ctx_mask.device, dtype=torch.bool)
        all_mask = torch.cat([ctx_mask, noisy_mask], dim=1)  # (B, S, P)
        valid_entities = all_mask.any(dim=1)  # (B, P)
        valid_times = all_mask.any(dim=2)  # (B, S)
        M[:, Z.Rt :, Z.Rp :] = all_mask  # (B, S, P) per-cell
        M[:, : Z.Rt, Z.Rp :] = valid_entities[:, None].expand(
            -1, Z.Rt, -1
        )  # (B, Rt, P)
        M[:, Z.Rt :, : Z.Rp] = valid_times[:, :, None].expand(
            -1, -1, Z.Rp
        )  # (B, S, Rp)
        return M

    @staticmethod
    def _mask_to_attn(valid: torch.Tensor) -> torch.Tensor:
        """(batch, seq) bool -> (batch, 1, seq, seq) bool.

        Prevents softmax-of-all-(-inf) NaN (issue #1): every row must have at
        least one True entry. Two layers of defense:
        1. Force diagonal True — each query always attends to itself.
        2. If a row is STILL all False (shouldn't happen but defensive), force
           the diagonal again via torch.where.

        Softmax of a row with exactly one 0 (unmasked) and rest -inf produces a
        one-hot, NOT NaN. This is why diagonal-True is sufficient.
        """
        attn = valid[:, None, :] & valid[:, :, None]  # (batch, seq, seq)
        diag = torch.arange(attn.shape[-1], device=attn.device)
        attn[:, diag, diag] = True  # primary defense: self-attention always valid
        return attn[:, None, :, :]  # (batch, 1, seq, seq)

    @torch.compiler.disable
    def _wrap_grid(self, t: torch.Tensor, Rt: int, Rp: int, CTX: int) -> AugGridLatent:
        """Wrap plain tensor as AugGridLatent. Compile-disabled."""
        return AugGridLatent(t, Rt=Rt, Rp=Rp, CTX=CTX)

    # ==================== masked pooling ====================

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """Masked min/max/mean pool along dim. Returns cat(min, max, mean) on last dim.

        Args:
            x:    (..., D) feature tensor
            mask: (...) bool, same shape as x except last dim
            dim:  dimension to reduce

        Returns: (..., 3*D) with pooled dim squeezed out
        """
        mask_f = mask.unsqueeze(-1)  # (..., 1) bool
        any_valid = mask.any(dim=dim, keepdim=True).unsqueeze(-1)  # (..., 1) keepdim

        # mean: zero invalid, sum, divide by count
        count = mask_f.float().sum(dim=dim, keepdim=True).clamp(min=1)  # (..., 1)
        x_mean = (x * mask_f.float()).sum(dim=dim, keepdim=True) / count  # (..., D)

        # min: +inf for invalid -> min ignores them
        x_min = x.masked_fill(~mask_f, float('inf'))
        x_min = x_min.min(dim=dim, keepdim=True).values  # (..., D)
        x_min = torch.where(any_valid, x_min, torch.zeros_like(x_min))

        # max: -inf for invalid -> max ignores them
        x_max = x.masked_fill(~mask_f, float('-inf'))
        x_max = x_max.max(dim=dim, keepdim=True).values  # (..., D)
        x_max = torch.where(any_valid, x_max, torch.zeros_like(x_max))

        return torch.cat([x_min, x_max, x_mean], dim=-1).squeeze(dim)  # (..., 3*D)

    # ==================== register<->X fusion ====================

    @torch.compiler.disable
    def _get_reg_outer(
        self,
        Z: AugGridLatent,  # (B, Rt+S, Rp+P, D)
        M: torch.Tensor,   # (B, Rt+S, Rp+P) validity mask
        layer_idx: int,
    ) -> torch.Tensor:
        """Bidirectional register<->X fusion. Returns (B, Rt+S, Rp+P, D) fusion grid.

        Register -> X: compress regs to summaries, outer-product expand to X region.
        X -> Register: masked pointnet pool X, fc, expand to register regions.
        Order: fc on compact shape first, expand after.
        """
        fc = self.fusions[layer_idx]
        B, Rt, Rp, D = Z.B, Z.Rt, Z.Rp, Z.D
        S, P = Z.S, Z.P
        act = F_fn.leaky_relu

        # ---- Register -> X direction (mask out padded entries before compression) ----
        # reg_t mask: zero out columns for padded entities
        rt_mask = M[:, :Rt, Rp:].unsqueeze(-1).float()  # (B, Rt, P, 1)
        # reg_p mask: zero out rows for padded timesteps
        rp_mask = M[:, Rt:, :Rp].unsqueeze(-1).float()  # (B, S, Rp, 1)

        # reg_a: (B, Rt, Rp, D) -> compress Rp -> compress Rt -> (B, D) global
        ra = act(fc['ra1'](Z.reg_a.reshape(B, Rt, Rp * D)))  # (B, Rt, D)
        ra = act(fc['ra2'](ra.reshape(B, Rt * D)))  # (B, D)

        # reg_t: (B, Rt, P, D) -> mask padded entities -> compress Rt -> (B, P, D)
        reg_t_masked = Z.reg_t * rt_mask  # (B, Rt, P, D) zero out padded
        rt = reg_t_masked.permute(0, 2, 1, 3).reshape(B * P, Rt * D)  # (B*P, Rt*D)
        rt = act(fc['rt'](rt)).reshape(B, P, D)  # (B, P, D)

        # reg_p: (B, S, Rp, D) -> mask padded timesteps -> compress Rp -> (B, S, D)
        reg_p_masked = Z.reg_p * rp_mask  # (B, S, Rp, D) zero out padded
        rp = act(fc['rp'](reg_p_masked.reshape(B * S, Rp * D)))  # (B*S, D)
        rp = rp.reshape(B, S, D)  # (B, S, D)

        # outer product: expand summaries -> concat -> fc -> (B, S, P, D)
        reg_outer = torch.cat([
            ra[:, None, None, :].expand(-1, S, P, -1),  # (B, S, P, D)
            rt[:, None, :, :].expand(-1, S, -1, -1),    # (B, S, P, D)
            rp[:, :, None, :].expand(-1, -1, P, -1),    # (B, S, P, D)
        ], dim=-1)  # (B, S, P, 3D)
        x_from_reg = fc['outer'](act(reg_outer))  # (B, S, P, D)

        # ---- X -> Register direction (masked pointnet) ----
        x = Z.X                   # (B, S, P, D)
        x_mask = M[:, Rt:, Rp:]   # (B, S, P)

        # pool over S (time) -> per-entity -> fc -> expand to reg_t
        pool_s = self._masked_pool(x, x_mask, dim=1)  # (B, P, 3D)
        reg_t_fx = fc['pool_t'](act(pool_s))  # (B, P, D)
        reg_t_fx = reg_t_fx[:, None, :, :].expand(-1, Rt, -1, -1)  # (B, Rt, P, D)

        # pool over P (entity) -> per-timestep -> fc -> expand to reg_p
        pool_p = self._masked_pool(x, x_mask, dim=2)  # (B, S, 3D)
        reg_p_fx = fc['pool_p'](act(pool_p))  # (B, S, D)
        reg_p_fx = reg_p_fx[:, :, None, :].expand(-1, -1, Rp, -1)  # (B, S, Rp, D)

        # pool over both S & P -> global -> fc -> expand to reg_a
        x_flat = x.reshape(B, S * P, D)      # (B, S*P, D)
        m_flat = x_mask.reshape(B, S * P)     # (B, S*P)
        pool_sp = self._masked_pool(x_flat, m_flat, dim=1)  # (B, 3D)
        reg_a_fx = fc['pool_a'](act(pool_sp))  # (B, D)
        reg_a_fx = reg_a_fx[:, None, None, :].expand(-1, Rt, Rp, -1)  # (B, Rt, Rp, D)

        # ---- assemble fusion grid ----
        F_top = torch.cat([reg_a_fx, reg_t_fx], dim=2)   # (B, Rt, Rp+P, D)
        F_bot = torch.cat([reg_p_fx, x_from_reg], dim=2)  # (B, S, Rp+P, D)
        return torch.cat([F_top, F_bot], dim=1)  # (B, Rt+S, Rp+P, D)

    @torch.compiler.disable
    def _pre_attention(
        self,
        Z: AugGridLatent,  # (B, Rt+S, Rp+P, D) current latent
        Z0: torch.Tensor,  # (B, Rt+S, Rp+P, D) frozen initial
        C: torch.Tensor,   # (B, Rt+S, Rp+P, D) projected conditioning
        M: torch.Tensor,   # (B, Rt+S, Rp+P) validity mask
        layer_idx: int,
    ):
        """Per-layer pre-attention: Z0 highway + register<->X fusion + adaLN enrichment.

        Returns (Z_updated, C_updated) as plain tensors.
        """
        fc = self.fusions[layer_idx]
        F = self._get_reg_outer(Z, M, layer_idx)  # (B, Rt+S, Rp+P, D)

        # Z = Z + fc2(act(fc1(cat(Z, Z0, F))))
        Z_out = Z + fc['fuse_z2'](F_fn.leaky_relu(
            fc['fuse_z1'](torch.cat([Z, Z0, F], dim=-1))  # (B, Rt+S, Rp+P, 3D) -> D
        ))  # (B, Rt+S, Rp+P, D)

        # C = C + fc2(act(fc1(cat(C, F))))
        C_out = C + fc['fuse_c2'](F_fn.leaky_relu(
            fc['fuse_c1'](torch.cat([C, F], dim=-1))  # (B, Rt+S, Rp+P, 2D) -> D
        ))  # (B, Rt+S, Rp+P, D)

        return Z_out, C_out

    # ==================== register FFN (post-attention) ====================

    def _apply_reg_ffn(
        self,
        Z: torch.Tensor,  # (B, Rt+S, Rp+P, D) plain tensor from step function
        layer_idx: int,
        B: int,
        S: int,
        P: int,
    ) -> torch.Tensor:
        """Apply separate adaLN+FFN to each register region (reg_t, reg_p, reg_a).

        Called AFTER each attention step, BEFORE wrap(). Matches old code's
        _RegisterAdaLNFFN pattern: independent nonlinear transform per region.
        """
        Rt, Rp, D = self.Rt, self.Rp, self.D

        # reg_t: (B, Rt, P, D) -> (B*P, Rt, D) -> FFN -> reshape back
        rt = Z[:, :Rt, Rp:, :].permute(0, 2, 1, 3).reshape(B * P, Rt, D)
        ada_rt = self.ada_rt_embeds[layer_idx].expand(B * P, -1, -1)  # (B*P, Rt, D)
        rt = self.reg_rt_blocks[layer_idx](rt, ada_rt)
        rt = rt.reshape(B, P, Rt, D).permute(0, 2, 1, 3)  # (B, Rt, P, D)

        # reg_p: (B, S, Rp, D) -> (B*S, Rp, D) -> FFN -> reshape back
        rp = Z[:, Rt:, :Rp, :].reshape(B * S, Rp, D)
        ada_rp = self.ada_rp_embeds[layer_idx].expand(B * S, -1, -1)  # (B*S, Rp, D)
        rp = self.reg_rp_blocks[layer_idx](rp, ada_rp)
        rp = rp.reshape(B, S, Rp, D)  # (B, S, Rp, D)

        # reg_a: (B, Rt, Rp, D) -> (B, Rt*Rp, D) -> FFN -> reshape back
        ra = Z[:, :Rt, :Rp, :].reshape(B, Rt * Rp, D)
        ada_ra = self.ada_corner_embeds[layer_idx].expand(B, -1, -1)  # (B, Rt*Rp, D)
        ra = self.reg_corner_blocks[layer_idx](ra, ada_ra)
        ra = ra.reshape(B, Rt, Rp, D)  # (B, Rt, Rp, D)

        # X region unchanged
        x_region = Z[:, Rt:, Rp:, :]  # (B, S, P, D)

        # reassemble grid
        top = torch.cat([ra, rt], dim=2)       # (B, Rt, Rp+P, D)
        bot = torch.cat([rp, x_region], dim=2)  # (B, S, Rp+P, D)
        return torch.cat([top, bot], dim=1)     # (B, Rt+S, Rp+P, D)

    # ==================== step functions ====================

    def _step_t(
        self,
        Z: torch.Tensor,  # (B, Rt+S, Rp+P, D) latent grid
        C: torch.Tensor,  # (B, Rt+S, Rp+P, D) conditioning grid
        R: torch.Tensor,  # (B, Rt+S, Rp+P)    RoPE time grid
        M: torch.Tensor,  # (B, Rt+S, Rp+P)    validity grid
        layer_idx: int,
        N_CTX: int,
    ) -> torch.Tensor:
        """T-layer: every column attends vertically (seq = Rt+S).

                    Rp cols       P cols
                  +-----------+------------+
        Rt rows   |  |  |  |  |  |  |  |  |
                  |  v  v  v  |  v  v  v  v|   batch = B * (Rp+P)
        S rows    |  |  |  |  |  |  |  |  |
                  |  v  v  v  |  v  v  v  v|   seq = Rt+S
                  +-----------+------------+

        Narrow mode (t_layer_dim < latent_dim):
            Z_in (full) -> z_down -> narrow DiT -> up -> residual add to Z_in.
            C (full)   -> c_down (fed to narrow DiT adaLN)
            Returns Z_in + up(narrow_block(z_down(Z_in), c_down(C))) at full dim.
        """
        B, RS, RpP, D = Z.shape
        Rt, Rp = self.Rt, self.Rp
        T = RS - Rt - N_CTX

        if self._narrow_active:
            projs = self.t_narrow_projs[f"layer_{layer_idx}"]
            td = self.t_layer_dim

            # Down-project Z and C (B, RS, RpP, D) -> (B, RS, RpP, td).
            # nn.Linear applies to last dim, so we can pass 4D directly.
            Z_nar = projs["z_down"](Z)
            C_nar = projs["c_down"](C)

            # Per-column narrow DiT (seq = RS, batch = B*RpP, dim = td)
            cols_n = Z_nar.permute(0, 2, 1, 3).reshape(B * RpP, RS, td)
            cond_n = C_nar.permute(0, 2, 1, 3).reshape(B * RpP, RS, td)
            t_rope = (
                R[:, Rt + N_CTX :, :].permute(0, 2, 1).reshape(B * RpP, T)
            )  # (B*RpP, T)
            col_valid = M.permute(0, 2, 1).reshape(B * RpP, RS)  # (B*RpP, RS)
            attn_mask = self._mask_to_attn(col_valid)  # (B*RpP, 1, RS, RS)

            out_n = self.blocks[layer_idx](
                x=cols_n,
                cond=cond_n,
                t=t_rope,
                num_external_regs=Rt + N_CTX,
                attn_mask=attn_mask,
            )  # (B*RpP, RS, td)

            # Un-permute back to grid at narrow dim
            out_n_grid = out_n.reshape(B, RpP, RS, td).permute(0, 2, 1, 3)  # (B, RS, RpP, td)

            # Up-project to full dim and residual add to INPUT Z
            out_full = projs["up"](out_n_grid)  # (B, RS, RpP, D)
            return Z + out_full  # residual: post-fusion Z + narrow-block output

        # ---- full-dim path (unchanged) ----
        # per-column: each column is a seq of RS tokens
        cols = Z.permute(0, 2, 1, 3).reshape(B * RpP, RS, D)  # (B*RpP, RS, D)
        cond = C.permute(0, 2, 1, 3).reshape(B * RpP, RS, D)  # (B*RpP, RS, D)
        t_rope = (
            R[:, Rt + N_CTX :, :].permute(0, 2, 1).reshape(B * RpP, T)
        )  # (B*RpP, T)

        # attention mask: per-column validity
        col_valid = M.permute(0, 2, 1).reshape(B * RpP, RS)  # (B*RpP, RS)
        attn_mask = self._mask_to_attn(col_valid)  # (B*RpP, 1, RS, RS)

        out = self.blocks[layer_idx](
            x=cols,
            cond=cond,
            t=t_rope,
            num_external_regs=Rt + N_CTX,
            attn_mask=attn_mask,
        )  # (B*RpP, RS, D)

        return out.reshape(B, RpP, RS, D).permute(0, 2, 1, 3)  # (B, RS, RpP, D)

    def _step_p(
        self,
        Z: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        C: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        lang_tokens: torch.Tensor,  # (B, A, D)
    ) -> torch.Tensor:
        """P-layer: every row attends horizontally (seq = A+Rp+P).

                    Rp cols       P cols
                  +-----------+------------+
        Rt rows   | -------->>|---------->>|   batch = B * (Rt+S)
                  +-----------+------------+
        S rows    | -------->>|---------->>|   seq = A+Rp+P
                  +-----------+------------+
        """
        B, RS, RpP, D = Z.shape
        Rp = self.Rp
        A = lang_tokens.shape[1]

        rows = Z.reshape(B * RS, RpP, D)  # (B*RS, RpP, D)

        # Per-TYPE shared lang projector (one Linear shared across all P-layers)
        lang_proj = self.add_token_projs["p"](lang_tokens)  # (B, A, D)
        lang_tiled = (
            lang_proj[:, None].expand(-1, RS, -1, -1).reshape(B * RS, A, D)
        )  # (B*RS, A, D)
        seq = torch.cat([lang_tiled, rows], dim=1)  # (B*RS, A+RpP, D)

        cond_rows = C.reshape(B * RS, RpP, D)  # (B*RS, RpP, D)
        cond_lang = torch.zeros(
            B * RS, A, D, device=Z.device, dtype=Z.dtype
        )  # (B*RS, A, D)
        cond_seq = torch.cat([cond_lang, cond_rows], dim=1)  # (B*RS, A+RpP, D)

        # attention mask: lang always valid
        row_valid = M.reshape(B * RS, RpP)  # (B*RS, RpP)
        lang_valid = torch.ones(
            B * RS, A, device=Z.device, dtype=torch.bool
        )  # (B*RS, A)
        seq_valid = torch.cat([lang_valid, row_valid], dim=1)  # (B*RS, A+RpP)
        attn_mask = self._mask_to_attn(seq_valid)  # (B*RS, 1, A+RpP, A+RpP)

        out = self.blocks[layer_idx](
            x=seq,
            cond=cond_seq,
            t=None,
            num_external_regs=A + Rp,
            attn_mask=attn_mask,
        )  # (B*RS, A+RpP, D)

        return out[:, A:].reshape(B, RS, RpP, D)  # (B, RS, RpP, D)

    def _step_s(
        self,
        Z: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        C: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        lang_tokens: torch.Tensor,  # (B, A, D)
    ) -> torch.Tensor:
        """S-layer: LEFT register columns attend vertically (seq = A+Rt+S).

                    Rp cols       P cols
                  +-----------+------------+
        Rt rows   |  |  |  |  |            |
                  |  v  v  v  |  unchanged |   batch = B * Rp
        S rows    |  |  |  |  |            |
                  |  v  v  v  |  unchanged |   seq = A+Rt+S
                  +-----------+------------+
        Lang tokens prepended to each column's sequence (shared per-S-type
        projector). T-layer is the only step that does NOT see lang tokens.
        """
        B, RS, RpP, D = Z.shape
        Rp = self.Rp
        A = lang_tokens.shape[1]

        cols = (
            Z[:, :, :Rp, :].permute(0, 2, 1, 3).reshape(B * Rp, RS, D)
        )  # (B*Rp, RS, D)
        cond = (
            C[:, :, :Rp, :].permute(0, 2, 1, 3).reshape(B * Rp, RS, D)
        )  # (B*Rp, RS, D)

        # Per-TYPE shared lang projector (one Linear shared across all S-layers)
        lang_proj = self.add_token_projs["s"](lang_tokens)  # (B, A, D)
        lang_tiled = (
            lang_proj[:, None].expand(-1, Rp, -1, -1).reshape(B * Rp, A, D)
        )  # (B*Rp, A, D)
        seq = torch.cat([lang_tiled, cols], dim=1)  # (B*Rp, A+RS, D)

        cond_lang = torch.zeros(
            B * Rp, A, D, device=Z.device, dtype=Z.dtype
        )  # (B*Rp, A, D)
        cond_seq = torch.cat([cond_lang, cond], dim=1)  # (B*Rp, A+RS, D)

        # attention mask: lang always valid
        col_valid = M[:, :, :Rp].permute(0, 2, 1).reshape(B * Rp, RS)  # (B*Rp, RS)
        lang_valid = torch.ones(
            B * Rp, A, device=Z.device, dtype=torch.bool
        )  # (B*Rp, A)
        seq_valid = torch.cat([lang_valid, col_valid], dim=1)  # (B*Rp, A+RS)
        attn_mask = self._mask_to_attn(seq_valid)  # (B*Rp, 1, A+RS, A+RS)

        out = self.blocks[layer_idx](
            x=seq,
            cond=cond_seq,
            t=None,
            num_external_regs=A + self.Rt,
            attn_mask=attn_mask,
        )  # (B*Rp, A+RS, D)

        # Strip lang tokens before reshaping back to grid
        updated = out[:, A:].reshape(B, Rp, RS, D).permute(0, 2, 1, 3)  # (B, RS, Rp, D)
        return torch.cat([updated, Z[:, :, Rp:, :]], dim=2)  # (B, RS, RpP, D)

    def _step_q(
        self,
        Z: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        C: torch.Tensor,  # (B, Rt+S, Rp+P, D)
        M: torch.Tensor,  # (B, Rt+S, Rp+P)
        layer_idx: int,
        lang_tokens: torch.Tensor,  # (B, A, D)
    ) -> torch.Tensor:
        """Q-layer: TOP register rows attend horizontally (seq = A+Rp+P).

                    Rp cols       P cols
                  +-----------+------------+
        Rt rows   | -------->>|---------->>|   batch = B * Rt
                  +-----------+------------+
        S rows    |       unchanged        |   seq = A+Rp+P
                  +------------------------+
        Lang tokens prepended to each row's sequence (shared per-Q-type
        projector). T-layer is the only step that does NOT see lang tokens.
        """
        B, RS, RpP, D = Z.shape
        Rt, Rp = self.Rt, self.Rp
        A = lang_tokens.shape[1]

        rows = Z[:, :Rt, :, :].reshape(B * Rt, RpP, D)  # (B*Rt, RpP, D)
        cond = C[:, :Rt, :, :].reshape(B * Rt, RpP, D)  # (B*Rt, RpP, D)

        # Per-TYPE shared lang projector (one Linear shared across all Q-layers)
        lang_proj = self.add_token_projs["q"](lang_tokens)  # (B, A, D)
        lang_tiled = (
            lang_proj[:, None].expand(-1, Rt, -1, -1).reshape(B * Rt, A, D)
        )  # (B*Rt, A, D)
        seq = torch.cat([lang_tiled, rows], dim=1)  # (B*Rt, A+RpP, D)

        cond_lang = torch.zeros(
            B * Rt, A, D, device=Z.device, dtype=Z.dtype
        )  # (B*Rt, A, D)
        cond_seq = torch.cat([cond_lang, cond], dim=1)  # (B*Rt, A+RpP, D)

        # attention mask: lang always valid
        row_valid = M[:, :Rt, :].reshape(B * Rt, RpP)  # (B*Rt, RpP)
        lang_valid = torch.ones(
            B * Rt, A, device=Z.device, dtype=torch.bool
        )  # (B*Rt, A)
        seq_valid = torch.cat([lang_valid, row_valid], dim=1)  # (B*Rt, A+RpP)
        attn_mask = self._mask_to_attn(seq_valid)  # (B*Rt, 1, A+RpP, A+RpP)

        out = self.blocks[layer_idx](
            x=seq,
            cond=cond_seq,
            t=None,
            num_external_regs=A + Rp,
            attn_mask=attn_mask,
        )  # (B*Rt, A+RpP, D)

        # Strip lang tokens before reshaping back to grid
        updated = out[:, A:].reshape(B, Rt, RpP, D)  # (B, Rt, RpP, D)
        return torch.cat([updated, Z[:, Rt:, :, :]], dim=1)  # (B, RS, RpP, D)

    # ==================== forward ====================

    def forward(
        self,
        x: torch.Tensor,              # (B, T, P, input_dim) noisy motion frames
        x_mask: torch.Tensor,         # (B, T, P) bool
        x_time: torch.Tensor,         # (B, T) diffusion timestamps [0,1]
        x_level: torch.Tensor,        # (B, T, P) per-token noise level [0,1000]
        feat: torch.Tensor,           # (B, P, feat_dim) per-entity identity features
        ctx: torch.Tensor,            # (B, CTX, P, input_dim) clean context frames
        ctx_mask: torch.Tensor,       # (B, CTX, P) bool
        ctx_level: torch.Tensor,      # (B, CTX, P) noise level (0 for clean)
        add_tokens: Optional[torch.Tensor] = None,      # (B, A, D) language tokens
        add_tokens_mask: Optional[torch.Tensor] = None,  # unused (compat with agent)
        image_tokens: Optional[torch.Tensor] = None,      # (B, A_img, D) for s/q layers only
        image_tokens_mask: Optional[torch.Tensor] = None,  # (B, A_img) bool
        group_ids: Optional[torch.Tensor] = None,         # unused (compat with agent)
        x_r_level: Optional[torch.Tensor] = None,         # unused (MeanFlow compat)
        ctx_r_level: Optional[torch.Tensor] = None,        # unused (MeanFlow compat)
    ) -> torch.Tensor:
        """Returns (B, S, P, input_dim) where S = CTX + T."""
        # alias to internal names
        noisy_x, noisy_mask, noisy_time = x, x_mask, x_time
        noise_level, identity_feat, ctx_x, lang_tokens = x_level, feat, ctx, add_tokens

        B, T, P, _ = noisy_x.shape
        device = noisy_x.device
        Rt, Rp, D = self.Rt, self.Rp, self.D
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
        ctx_emb = self.ctx_t_emb(
            1 + torch.arange(N_CTX, device=device)
        )[None].expand(B, -1, -1)  # (B, CTX, D)
        all_time_pos = torch.cat([ctx_emb, time_pos], 1)  # (B, S, D)

        # ---- 2. validity masks (needed for C_raw and M) ----
        all_mask = torch.cat([ctx_mask, noisy_mask], dim=1)  # (B, S, P)
        valid_entities = all_mask.any(dim=1)  # (B, P)
        valid_times = all_mask.any(dim=2)  # (B, S)

        # ---- 3. C_raw conditioning grid (B, Rt+S, Rp+P, cond_dim) ----
        level_pe = (
            self.level_pe(all_level.reshape(B * S * P))
            .reshape(B, S, P, D)
            .to(all_x)
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
            cond_parts.append(
                self.hard_ln_pe(traj_pe[:, None].expand(-1, S, -1, -1))
            )
        C_raw = self._build_C_raw(
            torch.cat(cond_parts, -1), valid_entities, valid_times
        )  # (B, Rt+S, Rp+P, cond_dim)

        # ---- 3. Z latent grid ----
        x_emb = self.x_proj(all_x)  # (B, S, P, D)
        Z = AugGridLatent.build(
            ctx_feat=x_emb[:, :N_CTX],
            x_feat=x_emb[:, N_CTX:],
            reg_t_feat=self.reg_t_init.expand(B, -1, P, -1),
            reg_p_feat=self.reg_p_init.expand(B, S, -1, -1),
            reg_a_feat=self.reg_a_init.expand(B, -1, -1, -1),
            Rt=Rt,
            Rp=Rp,
        )  # (B, Rt+S, Rp+P, D)
        # Z0 highway is only used inside _pre_attention; skip the clone
        # when fusion is disabled to avoid wasted memory (~23-47MB/forward).
        Z0 = Z.clone() if self.enable_pre_attention_fusion else None

        # ---- 4. R time-RoPE grid ----
        R = self._build_R(Z, time_scaled)  # (B, Rt+S, Rp+P)

        # ---- 5. M validity grid (reuses precomputed masks) ----
        M = self._build_M(Z, ctx_mask, noisy_mask)  # (B, Rt+S, Rp+P)
        # Note: _build_M recomputes all_mask/valid_entities/valid_times internally
        # which is redundant but keeps _build_M self-contained.

        # ---- 6. language tokens ----
        if lang_tokens is None:
            lang_tokens = torch.zeros(B, 0, D, device=device)

        # Image tokens concatenated with lang for s/q layers (not p).
        # torch.compile audit: A_img is STATIC per config (V * P_dino is fixed
        # by num_views and image_size). torch.cat on two static-shape tensors
        # is safe — no recompilation storms. No closures, no dynamic masks,
        # no data-dependent branches in the new code path.
        if image_tokens is not None:
            sq_tokens = torch.cat([lang_tokens, image_tokens], dim=1)  # (B, A_lang+A_img, D)
        else:
            sq_tokens = lang_tokens

        # ---- 7. skip list (initial X always prepended) ----
        skip_list = [Z.X.reshape(B * S, P, D).clone()]

        # ---- 8. main loop ----
        for i, lt in enumerate(self.layer_type_string):
            # fresh C from C_raw, then (optionally) enriched by pre_attention
            C = self.cond_binders[i](C_raw)              # (B, Rt+S, Rp+P, D)
            if self.enable_pre_attention_fusion:
                Z, C = self._pre_attention(Z, Z0, C, M, i)   # fusion + Z0 highway

            if lt == "t":
                # T-layer is the only step that does NOT see lang tokens
                Z = self._step_t(Z, C, R, M, i, N_CTX)
            elif lt == "p":
                Z = self._step_p(Z, C, M, i, lang_tokens)  # lang-only
            elif lt == "s":
                Z = self._step_s(Z, C, M, i, sq_tokens)  # lang+image
            elif lt == "q":
                Z = self._step_q(Z, C, M, i, sq_tokens)  # lang+image

            # Collect skip from resolved layer set (any layer type allowed).
            # Order is deterministic ascending because layers execute in order
            # and _skip_layer_idx_list was sorted at init.
            if i in self._skip_layer_idx_set:
                skip_list.append(
                    Z[:, Rt:, Rp:, :].reshape(B * S, P, D).clone()
                )

            # register-specific adaLN+FFN (post-attention, before wrap) -- toggleable
            if self.use_reg_post_ffn:
                Z = self._apply_reg_ffn(Z, i, B, S, P)
            Z = self._wrap_grid(Z, Rt, Rp, N_CTX)

        # ---- 9. output ----
        # skip_list length = len(_skip_layer_idx_list) + 1 (initial X + per-layer skips)
        x_out = torch.cat(skip_list, dim=-1)  # (B*S, P, D * len(skip_list))
        out = self.output_proj(
            x_out.reshape(-1, x_out.shape[-1])
        ).reshape(B, S, P, self.input_dim)
        # strip AugGridLatent subclass if it leaked (deepcopy safety for PL metrics)
        if isinstance(out, AugGridLatent):
            out = out.as_subclass(torch.Tensor)
        return out


# ==================== self-test ====================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    COMMON = dict(
        latent_dim=512,
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
    # Matrix: (layer_type_string, t_layer_dim, skip_layer_idx_list label)
    configs = {
        "tptptpt full": dict(layer_type_string="tptptpt", t_layer_dim=-1, skip_layer_idx_list=None),
        "tqtqtqt full": dict(layer_type_string="tqtqtqt", t_layer_dim=-1, skip_layer_idx_list=None),
        "tptptpt narrow128": dict(layer_type_string="tptptpt", t_layer_dim=128, skip_layer_idx_list=None),
        "tqtqtqt narrow128": dict(layer_type_string="tqtqtqt", t_layer_dim=128, skip_layer_idx_list=None),
        "tqtqtqt narrow128 skip all": dict(
            layer_type_string="tqtqtqt", t_layer_dim=128,
            skip_layer_idx_list=[0, 1, 2, 3, 4, 5, 6],
        ),
        "tqtqtqt narrow128 skip custom": dict(
            layer_type_string="tqtqtqt", t_layer_dim=128,
            skip_layer_idx_list=[1, 3, 5],  # non-T layers only
        ),
    }

    B, T, C, P, A = 2, 5, 4, 8, 1
    # mark 2 of 8 entities as padded (invalid) to test masking
    noisy_mask = torch.ones(B, T, P, dtype=torch.bool, device=device)
    noisy_mask[:, :, -2:] = False  # last 2 entities invalid
    ctx_mask = torch.ones(B, C, P, dtype=torch.bool, device=device)
    ctx_mask[:, :, -2:] = False

    def run_forward(net):
        with torch.no_grad():
            return net(
                x=torch.randn(B, T, P, 9, device=device),
                x_mask=noisy_mask,
                x_time=torch.rand(B, T, device=device),
                x_level=torch.rand(B, T, P, device=device) * 1000,
                feat=torch.randn(B, P, 512, device=device),
                ctx=torch.randn(B, C, P, 9, device=device),
                ctx_mask=ctx_mask,
                ctx_level=torch.zeros(B, C, P, device=device),
                add_tokens=torch.randn(B, A, 512, device=device),
            )

    for name, cfg in configs.items():
        tag = name
        print(f"\n{'='*70}\n{tag}\n{'='*70}")
        net = MotionMachineV11Ultra(**COMMON, **cfg).to(device).eval()
        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(
            f"Params: {n_params:.2f}M  narrow_active={net._narrow_active}  "
            f"skip_idx={net._skip_layer_idx_list}"
        )
        out = run_forward(net)
        assert out.shape == (B, C + T, P, 9), f"FAIL {tag}: {out.shape}"
        print(f"Output: {list(out.shape)} PASS")
    print(f"\n{'='*70}\nAll passed.\n{'='*70}")
