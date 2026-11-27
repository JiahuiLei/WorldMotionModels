"""MotionMachine V11 DenseV2 — Unified register padding with Rt/Rp scheme.

Architecture: All layers at unified width C_D. String-based t/p layer config.
Replaces the separate reg-c/reg-x/reg-a/reg-p registers from DenseV1 with a
unified 2D padding scheme:

The working tensor has shape [S, P] (sequence x entities). DenseV2 pads this
to [S+Rt, P+Rp] where Rt and Rp are configurable register counts:

           P cols          Rp cols
         +-------------+----------+
  S rows |  main [S,P] | Rp [S,Rp]|
         +-------------+----------+
  Rt rows| Rt [Rt,P]   |RtxRp     |
         |             |[Rt,Rp]   |
         +-------------+----------+

Register evolution: NOT re-initialized per layer. Layer 0 uses learnable init
embeddings; layer N>0 uses previous output + additive learnable per-layer delta.

4 FFNs per layer:
  1. Primal FFN (main [S,P] tokens) -- handled inside DiT block
  2. FFN_Rt (trajectory-pooled register row)
  3. FFN_Rp (timeframe-pooled register column)
  4. FFN_corner (global register corner block)

t-layer: Self-attention across S+Rt per entity column (P+Rp columns in parallel).
p-layer: Cross-attention across P+Rp per timestep row (S+Rt rows in parallel).

Example::

    net = MotionMachineV11DenseV2(
        C_D=512, layer_type_string="tptptptp", Rt=4, Rp=4,
    )
    out = net(x, x_mask, x_time, x_level, feat,
              ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
              add_tokens=lang, add_tokens_mask=lang_mask)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    build_attention_mask,
)
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly
from umm.models.umm_v11.world_models.rope_utils import DiTBlockSplitRoPE
from umm.models.umm_v11.world_models.so3_utils import compute_entity_mean_9d

logger = logging.getLogger(__name__)


class _RegisterAdaLNFFN(nn.Module):
    """Combined adaLN-Zero + FFN for register tokens.

    Merges the FFN as a submodule (not passed as argument) so the entire
    module is compile-safe. Follows adaLN-Zero pattern: norm -> modulate ->
    FFN -> gate -> residual.

    Args:
        dim: Token dimension (C_D).
        hidden_dim: Hidden layer dimension for FFN.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Projects conditioning to 3 * dim (scale, shift, gate)
        self.ada_proj = nn.Linear(dim, 3 * dim)
        # Initialize gate to zero (adaLN-Zero)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
        # FFN as submodule (compile-safe)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaLN-Zero modulated FFN.

        Compile safety: Disabled due to tensor view tracking issues in Dynamo
        when the register state (permuted+reshaped) flows into this method.
        The 27 graph breaks per forward (3 types * 9 layers) are acceptable
        since registers are a small fraction of total compute.

        Args:
            x: (..., dim) input tokens.
            cond: (..., dim) conditioning embeddings (learnable per-layer).

        Returns:
            (..., dim) output with residual connection.
        """
        ada_out = self.ada_proj(cond)  # (..., 3*dim)
        scale, shift, gate = ada_out.chunk(3, dim=-1)

        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.ffn(h)
        return x + gate * h


class MotionMachineV11DenseV2(nn.Module):
    """Dense LMM V2 with unified Rt/Rp register padding.

    Args:
        C_D: Unified channel width for all layers (default 512).
        feat_dim: Input feature dimension from identity layer.
        n_head: Number of attention heads.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        layer_type_string: String of layer types (e.g., "tptptptp").
            Valid chars: t (T-attn), p (P-attn), a (T-attn regs only),
            b (P-attn regs only). Must start with 't'.
        Rt: Number of trajectory register tokens (time-dim padding).
        Rp: Number of entity register tokens (entity-dim padding).
        n_ctx: Maximum context length for position embeddings.
        fc_factor: MLP width factor for all layers.
        fc_n_layers: Number of hidden layers in each DiT block MLP.
        fc_out_n_layers: Number of layers in output projection.
        num_blk_internal_registers: Internal register count for DiT blocks.
        time_multiplier: Scale factor for timestep values.
        ada_scale_bias_bound: Bound for adaLN scale/bias.
        gate_bound: Bound for gate values.
        ada_n_layers: Number of layers in adaLN projection.
        add_time_to_feat: Use continuous time PE in conditioning.
        time_rope_flag: Use RoPE for time in t-layers.
        spatial_rope_flag: Use split-head spatial RoPE (requires time_rope_flag).
        n_traj_tokens: If > 0, add trajectory PE as conditioning component.
        qk_norm: Use QK normalization in attention.
        x_pe_dim: If > 0, apply NeRF-style positional encoding to input.
    """

    def __init__(
        self,
        C_D: int = 512,
        feat_dim: int = 512,
        n_head: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        layer_type_string: str = "tptptptp",
        Rt: int = 4,
        Rp: int = 4,
        n_ctx: int = 128,
        fc_factor: int = 4,
        fc_n_layers: int = 2,
        fc_out_n_layers: int = 12,
        num_blk_internal_registers: int = 4,
        time_multiplier: float = 1.0,
        ada_scale_bias_bound: float = 1.0,
        gate_bound: float = 2.0,
        ada_n_layers: int = 1,
        add_time_to_feat: bool = True,
        time_rope_flag: bool = False,
        spatial_rope_flag: bool = False,
        n_traj_tokens: int = 0,
        qk_norm: bool = True,
        x_pe_dim: int = 0,
    ):
        super().__init__()

        self.C_D = C_D
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.layer_type_string = layer_type_string
        self.n_layers = len(layer_type_string)
        self.n_head = n_head
        self.time_multiplier = time_multiplier
        self.time_rope_flag = time_rope_flag
        self.spatial_rope_flag = spatial_rope_flag
        self.add_time_to_feat = add_time_to_feat
        self.n_traj_tokens = n_traj_tokens

        # Validate
        assert len(layer_type_string) > 0, "layer_type_string must not be empty"
        assert layer_type_string[0] == "t", (
            f"layer_type_string must start with 't', got '{layer_type_string[0]}'"
        )
        for ch in layer_type_string:
            assert ch in "tpab", f"Invalid layer type '{ch}', must be t/p/a/b"
        if spatial_rope_flag:
            assert time_rope_flag, "spatial_rope_flag requires time_rope_flag=True"

        # Register counts
        self.Rt = Rt
        self.Rp = Rp

        # Count layer types
        self.n_t_layers = sum(1 for c in layer_type_string if c == "t")
        self.n_p_layers = sum(1 for c in layer_type_string if c == "p")
        self.n_a_layers = sum(1 for c in layer_type_string if c == "a")
        self.n_b_layers = sum(1 for c in layer_type_string if c == "b")

        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        self.hard_ln_cd = HardLayerNorm(self.C_D)

        # =============== Input Projection ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            from umm.models.umm_v9.world_models.wm_utils import NerfStylePE

            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )
            x_proj_in = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.input_dim
        self.x_proj = nn.Linear(x_proj_in, self.C_D)

        # =============== Time/Position Embeddings ===============
        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.C_D)

        self.int_pe = Timesteps(
            self.C_D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )

        if self.add_time_to_feat:
            self.t_pe = Timesteps(
                self.C_D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            self.t_proj = nn.Linear(self.C_D + 1, self.C_D)

        # =============== Trajectory PE ===============
        if self.n_traj_tokens > 0:
            self.traj_pe = Timesteps(
                self.C_D, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )

        # =============== Time Binding (no-RoPE path) ===============
        if not self.time_rope_flag and self.add_time_to_feat:
            self.time_bind_proj = nn.Linear(2 * self.C_D, self.C_D)

        # =============== Conditioning ===============
        self.cond_dim = self.feat_dim + 2 * self.C_D
        if self.n_traj_tokens > 0:
            self.cond_dim += self.C_D

        # Per-layer conditioning binders
        self.cond_binders = nn.ModuleList(
            [nn.Linear(self.cond_dim, self.C_D) for _ in range(self.n_layers)]
        )

        # =============== Unified Register Embeddings ===============
        # Rt registers: trajectory-dimension padding. Shape: (n_layers, Rt, C_D)
        # Each layer gets init (layer 0) or additive delta (layer N>0).
        if self.Rt > 0:
            self.rt_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rt, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.rt_embeds:
                nn.init.normal_(emb, std=0.02)

        # Rp registers: entity-dimension padding. Shape: (n_layers, Rp, C_D)
        if self.Rp > 0:
            self.rp_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.rp_embeds:
                nn.init.normal_(emb, std=0.02)

        # Corner registers: Rt x Rp intersection. Shape: (n_layers, Rt*Rp, C_D)
        if self.Rt > 0 and self.Rp > 0:
            self.corner_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rt * self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.corner_embeds:
                nn.init.normal_(emb, std=0.02)

        # =============== Per-Layer adaLN for Registers ===============
        # Learnable conditioning embeddings for register FFNs
        if self.Rt > 0:
            self.ada_rt_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.Rt, self.C_D))
                for _ in range(self.n_layers)
            ])
        if self.Rp > 0:
            self.ada_rp_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])
        if self.Rt > 0 and self.Rp > 0:
            self.ada_corner_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.Rt * self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])

        # =============== Register adaLN+FFN (per layer, merged for compile safety) ===============
        ffn_hidden = self.C_D * fc_factor
        if self.Rt > 0:
            self.reg_rt_blocks = nn.ModuleList([
                _RegisterAdaLNFFN(self.C_D, ffn_hidden)
                for _ in range(self.n_layers)
            ])
        if self.Rp > 0:
            self.reg_rp_blocks = nn.ModuleList([
                _RegisterAdaLNFFN(self.C_D, ffn_hidden)
                for _ in range(self.n_layers)
            ])
        if self.Rt > 0 and self.Rp > 0:
            self.reg_corner_blocks = nn.ModuleList([
                _RegisterAdaLNFFN(self.C_D, ffn_hidden)
                for _ in range(self.n_layers)
            ])

        # =============== Layer-0 Mixing (gradient highway) ===============
        if self.n_t_layers > 1:
            self.layer0_mix = nn.ModuleList([
                nn.Linear(2 * self.C_D, self.C_D)
                for _ in range(self.n_t_layers - 1)
            ])

        # =============== DiT Blocks ===============
        _dit_kwargs = dict(
            C=self.C_D,
            num_heads=n_head,
            num_registers=num_blk_internal_registers,
            mlp_hidden_dims=[self.C_D * fc_factor] * fc_n_layers,
            ada_scale_bias_bound=ada_scale_bias_bound,
            gate_bound=gate_bound,
            ada_n_layers=ada_n_layers,
            C_feat=self.C_D,
            qk_norm=qk_norm,
        )

        if self.spatial_rope_flag and self.time_rope_flag:
            # Split RoPE: half temporal, half spatial
            _dit_t_kwargs = dict(
                n_temporal_heads=n_head // 2,
                n_spatial_dims=9,
                **_dit_kwargs,
            )
            # t-layer positions: SplitRoPE. p-layer positions: placeholder
            # (p-layers use separate p_blocks_rope with dual spatial RoPE)
            self.blocks = nn.ModuleList([
                DiTBlockSplitRoPE(**_dit_t_kwargs)
                if c == "t" else nn.Identity()
                for c in layer_type_string
            ])
            # P-layer blocks: SplitRoPE with dual spatial
            self._p_blocks_split_rope = True
            self.p_blocks_rope = nn.ModuleList([
                DiTBlockSplitRoPE(**_dit_t_kwargs)
                for c in layer_type_string if c == "p"
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_kwargs) if c in "tp" else nn.Identity()
                for c in layer_type_string
            ])
            self._p_blocks_split_rope = False

        # a/b layer blocks: DiTBlockTimeOnly (no RoPE needed for register-only attention)
        if self.n_a_layers > 0:
            self.a_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_kwargs)
                for _ in range(self.n_a_layers)
            ])
        if self.n_b_layers > 0:
            self.b_blocks = nn.ModuleList([
                DiTBlockTimeOnly(**_dit_kwargs)
                for _ in range(self.n_b_layers)
            ])

        # =============== Output Projection ===============
        # Collect skips from t-layers only + initial embedding
        n_skips = self.n_t_layers + 1
        out_proj_in = self.C_D * n_skips
        _seq = []
        _c_in = out_proj_in
        for _c_out in np.linspace(out_proj_in, self.input_dim, fc_out_n_layers)[:-1]:
            _c_out = int(_c_out)
            _seq.append(nn.Linear(_c_in, _c_out))
            _seq.append(HardLayerNorm(_c_out))
            _seq.append(nn.LeakyReLU())
            _c_in = _c_out
        _seq.append(nn.Linear(_c_in, self.input_dim))
        self.output_proj = nn.Sequential(*_seq)

        self._log_summary()

    def _log_summary(self):
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info("LMM V11 DenseV2 Summary:")
        logger.info(f"  C_D (unified):       {self.C_D}")
        logger.info(f"  layer_type_string:   {self.layer_type_string}")
        logger.info(f"  n_layers:            {self.n_layers}")
        logger.info(f"  t-layers:            {self.n_t_layers}")
        logger.info(f"  p-layers:            {self.n_p_layers}")
        logger.info(f"  a-layers:            {self.n_a_layers}")
        logger.info(f"  b-layers:            {self.n_b_layers}")
        logger.info(f"  Rt (time registers): {self.Rt}")
        logger.info(f"  Rp (entity registers): {self.Rp}")
        logger.info(f"  corner tokens:       {self.Rt * self.Rp}")
        logger.info(f"  n_head:              {self.n_head}")
        logger.info(f"  input_dim:           {self.input_dim}")
        logger.info(f"  time_rope_flag:      {self.time_rope_flag}")
        logger.info(f"  spatial_rope_flag:   {self.spatial_rope_flag}")
        logger.info(f"  n_traj_tokens:       {self.n_traj_tokens}")
        logger.info(f"  params:              {n_params:.2f}M")
        logger.info(
            f"  output skips:        {self.n_t_layers + 1} x {self.C_D}"
            f" = {(self.n_t_layers + 1) * self.C_D}"
        )
        logger.info(f"  register FFNs/layer: Rt={self.Rt > 0} Rp={self.Rp > 0}"
                     f" corner={self.Rt > 0 and self.Rp > 0}")

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

    @torch.compiler.disable
    def _sample_traj_ids(
        self, B: int, P: int, device: torch.device
    ) -> torch.Tensor:
        """Sample trajectory IDs for trajectory PE.

        Training: random permutation per batch item.
        Eval: identity ordering (0, 1, ..., P-1).

        Args:
            B: Batch size.
            P: Number of entities.
            device: torch device.

        Returns:
            (B, P) int tensor of trajectory IDs in [0, n_traj_tokens).
        """
        if self.training:
            ids = torch.stack([
                torch.randperm(self.n_traj_tokens, device=device)[:P]
                for _ in range(B)
            ])  # (B, P)
        else:
            ids = torch.arange(P, device=device)[None].expand(B, -1)  # (B, P)
        return ids

    @torch.compiler.disable
    def _build_register_masks(
        self,
        x_mask: torch.Tensor,
        add_tokens_mask: torch.Tensor,
        S: int,
        P: int,
        T: int,
        N_CTX: int,
        A: int,
        R_blk: int,
        device: torch.device,
    ) -> dict:
        """Build attention masks for t-layer and p-layer with register isolation.

        Registers are always valid for non-padded entities/timesteps. For
        fully-padded entities (column all False) or fully-padded timesteps
        (row all False), registers are ISOLATED: they self-attend only.

        Compile safety: Uses data-dependent mask construction on tensor values.

        Args:
            x_mask: (B, S, P) bool mask. True = valid token.
            S: Sequence length (T + N_CTX).
            P: Number of entities.
            T: Number of noisy timesteps.
            N_CTX: Number of context timesteps.
            A: Number of add_tokens.
            R_blk: Internal register count from DiT block.
            device: torch device.

        Returns:
            dict with keys:
                'attn_mask_t': (B*P_ext, n_head, L_t, L_t) for t-layers
                'attn_mask_p': (B*S_ext, n_head, L_p, L_p) for p-layers
            where P_ext = P + Rp, S_ext = S + Rt.
        """
        B = x_mask.shape[0]
        Rt, Rp = self.Rt, self.Rp
        P_ext = P + Rp  # total entity columns
        S_ext = S + Rt   # total timestep rows

        # ============ t-layer mask ============
        # t-layer: per-column attention across S+Rt tokens
        # Sequence layout per column j (after DiT internal reg prepend):
        #   [internal_regs(R_blk), Rt_registers(Rt), ctx(N_CTX), x(T)]
        # Total L_t = R_blk + Rt + S

        L_t = R_blk + Rt + S

        # Per-column validity from x_mask: (B, P) -- entity valid if ANY timestep valid
        entity_ever_valid = x_mask.any(dim=1)  # (B, P)

        # For main entity columns (j < P): sequence validity is x_mask[:, :, j]
        #   shape (B, P, S) -> (B*P, S) after permute
        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S)  # (B*P, S)

        # For Rp register columns: always valid sequence (they see everything)
        rp_col_valid = torch.ones(B * Rp, S, device=device, dtype=torch.bool)

        # Combine: (B*P_ext, S)
        all_col_seq_valid = torch.cat([traj_valid, rp_col_valid], dim=0)

        # Prepend internal_regs + Rt validity
        ir_valid = torch.ones(B * P_ext, R_blk, device=device, dtype=torch.bool)
        rt_valid = torch.ones(B * P_ext, Rt, device=device, dtype=torch.bool)
        full_valid_t = torch.cat(
            [ir_valid, rt_valid, all_col_seq_valid], dim=1
        )  # (B*P_ext, L_t)

        # Bidirectional validity mask
        val_mask_t = (
            full_valid_t[:, None, :] & full_valid_t[:, :, None]
        )  # (B*P_ext, L_t, L_t)

        # Isolation for fully-padded entity columns:
        # If entity j has entity_ever_valid[b, j] == False, then Rt registers
        # for that column should only self-attend (not cross-attend to main tokens)
        # -- effectively, set Rt positions to only attend to other Rt positions
        # and the diagonal.
        #
        # For Rp register columns: always non-isolated (they are extra columns,
        # not tied to any specific entity validity).

        # Build isolation flag: (B*P_ext,) bool -- True = isolated
        entity_isolated = ~entity_ever_valid  # (B, P) True = isolated
        entity_isolated_flat = entity_isolated.reshape(B * P)  # (B*P,)
        rp_not_isolated = torch.zeros(
            B * Rp, device=device, dtype=torch.bool
        )
        all_isolated = torch.cat(
            [entity_isolated_flat, rp_not_isolated], dim=0
        )  # (B*P_ext,)

        # For isolated columns: zero out cross-attention between Rt and main tokens
        if all_isolated.any():
            iso_idx = all_isolated.nonzero(as_tuple=True)[0]
            rt_start = R_blk
            rt_end = R_blk + Rt
            main_start = R_blk + Rt
            # Rt cannot attend to main tokens
            val_mask_t[iso_idx, rt_start:rt_end, main_start:] = False
            # Main tokens cannot attend to Rt
            val_mask_t[iso_idx, main_start:, rt_start:rt_end] = False

        # Force diagonal True
        diag_t = torch.arange(L_t, device=device)
        val_mask_t[:, diag_t, diag_t] = True

        # Expand for heads: (B*P_ext, n_head, L_t, L_t)
        attn_mask_t = val_mask_t[:, None].expand(-1, self.n_head, -1, -1)

        # ============ p-layer mask ============
        # p-layer: per-row attention across P+Rp tokens
        # Sequence layout per row i (after DiT internal reg prepend):
        #   [internal_regs(R_blk), add_tokens(A), Rp_registers(Rp), entities(P)]
        # Total L_p = R_blk + A + Rp + P

        L_p = R_blk + A + Rp + P

        # Per-row entity validity: entity_ever_valid (B, P) broadcast across all rows
        # For main timestep rows (i < S): use entity_ever_valid
        pert_valid_main = entity_ever_valid[:, None, :].expand(
            -1, S, -1
        ).reshape(B * S, P)  # (B*S, P)

        # For Rt register rows: always valid (registers see all entities)
        pert_valid_rt = entity_ever_valid[:, None, :].expand(
            -1, Rt, -1
        ).reshape(B * Rt, P)  # (B*Rt, P)

        # Combine row validity: (B*S_ext, P)
        all_row_ent_valid = torch.cat(
            [pert_valid_main, pert_valid_rt], dim=0
        )  # (B*S_ext, P)

        # Rp validity: always valid
        rp_valid = torch.ones(
            B * S_ext, Rp, device=device, dtype=torch.bool
        )  # (B*S_ext, Rp)

        # Add token validity: use actual add_tokens_mask, tiled over all rows
        if A > 0:
            # add_tokens_mask: (B, A) -> tile to (B*S_ext, A)
            add_valid_main = (
                add_tokens_mask[:, None, :]
                .expand(-1, S, -1)
                .reshape(B * S, A)
            )  # (B*S, A)
            add_valid_rt = (
                add_tokens_mask[:, None, :]
                .expand(-1, Rt, -1)
                .reshape(B * Rt, A)
            )  # (B*Rt, A)
            add_valid = torch.cat([add_valid_main, add_valid_rt], dim=0)
        else:
            add_valid = torch.zeros(
                B * S_ext, 0, device=device, dtype=torch.bool
            )

        # Full row sequence validity: [add, Rp, entities]
        p_seq_valid = torch.cat(
            [add_valid, rp_valid, all_row_ent_valid], dim=1
        )  # (B*S_ext, A + Rp + P)

        attn_mask_p = build_attention_mask(
            p_seq_valid,
            mode="full",
            left_pad=R_blk,
        )  # (B*S_ext, L_p, L_p)

        # Isolation for fully-padded timestep rows:
        # A timestep row is padded if x_mask[b, s, :].all() == False
        row_ever_valid = x_mask.any(dim=2)  # (B, S) True = at least one entity valid

        # For Rt rows: always non-isolated
        row_isolated_main = ~row_ever_valid  # (B, S) True = isolated
        row_isolated_flat = row_isolated_main.reshape(B * S)
        rt_not_isolated = torch.zeros(
            B * Rt, device=device, dtype=torch.bool
        )
        all_row_isolated = torch.cat(
            [row_isolated_flat, rt_not_isolated], dim=0
        )  # (B*S_ext,)

        if all_row_isolated.any():
            iso_r_idx = all_row_isolated.nonzero(as_tuple=True)[0]
            # Rp registers for isolated rows: only self-attend
            rp_start = R_blk + A
            rp_end = R_blk + A + Rp
            ent_start = R_blk + A + Rp
            # Rp cannot attend to entities
            attn_mask_p[iso_r_idx, rp_start:rp_end, ent_start:] = False
            # Entities cannot attend to Rp
            attn_mask_p[iso_r_idx, ent_start:, rp_start:rp_end] = False

        # Force diagonal True
        diag_p = torch.arange(L_p, device=device)
        attn_mask_p[:, diag_p, diag_p] = True

        # Expand for heads: (B*S_ext, n_head, L_p, L_p)
        attn_mask_p = attn_mask_p[:, None].expand(-1, self.n_head, -1, -1)

        return {
            "attn_mask_t": attn_mask_t,
            "attn_mask_p": attn_mask_p,
        }

    @torch.compiler.disable
    def _compute_spatial_positions(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        N_CTX: int,
        B: int,
        S: int,
        P: int,
    ) -> dict:
        """Compute spatial positions for split RoPE.

        Compile safety: Uses compute_entity_mean_9d which contains SVD.

        Args:
            x: (B, S, P, 9) concatenated input (ctx + noisy).
            x_mask: (B, S, P) bool mask.
            N_CTX: Number of context timesteps.
            B, S, P: Dimensions.

        Returns:
            dict with keys:
                'x_9d_spatial': (B*P, S, 9) per-token spatial positions.
                'mean_ctx_9d': (B, P, 9) mean ctx 9D for dual spatial RoPE.
                'mean_x_9d': (B, P, 9) mean noisy 9D for dual spatial RoPE.
        """
        # t-layer: per-trajectory spatial positions (B*P, S, 9)
        x_9d_spatial = x.detach().permute(0, 2, 1, 3).reshape(B * P, S, 9)

        # p-layer: mean 9D for dual spatial RoPE
        mean_ctx_9d = (
            compute_entity_mean_9d(x[:, :N_CTX], x_mask[:, :N_CTX])
            if N_CTX > 0 else None
        )
        mean_x_9d = compute_entity_mean_9d(x[:, N_CTX:], x_mask[:, N_CTX:])
        if mean_ctx_9d is None:
            mean_ctx_9d = mean_x_9d

        return {
            "x_9d_spatial": x_9d_spatial,
            "mean_ctx_9d": mean_ctx_9d,
            "mean_x_9d": mean_x_9d,
        }

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_time: torch.Tensor,
        x_level: torch.Tensor,
        feat: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        ctx_level: Optional[torch.Tensor] = None,
        add_tokens: Optional[torch.Tensor] = None,
        add_tokens_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        x_r_level: Optional[torch.Tensor] = None,
        ctx_r_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with unified Rt/Rp register padding.

        Args:
            x: (B, T, P, input_dim) noisy input.
            x_mask: (B, T, P) bool mask.
            x_time: (B, T) timestamps.
            x_level: (B, T, P) diffusion levels.
            feat: (B, P, feat_dim) per-entity identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: (B, A, C_D) language tokens.
            add_tokens_mask: (B, A) mask.
            group_ids: (B, P) group IDs (unused, for interface compat).
            x_r_level: (B, T, P) MeanFlow r-level (optional).
            ctx_r_level: (B, C, P) MeanFlow r-level for context (optional).

        Returns:
            (B, S, P, input_dim) predicted output. S = T + N_CTX.
        """
        B, T, P, _ = x.shape
        device = x.device
        Rt, Rp = self.Rt, self.Rp
        x_time = x_time.float().clone() * self.time_multiplier

        # ========== Input Processing ==========
        if self.add_time_to_feat:
            ctx_pos_emb = torch.cat(
                [
                    x_time[..., None],
                    self.t_pe(x_time.reshape(-1)).reshape(*x_time.shape[:2], -1),
                ],
                dim=-1,
            )
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # (B, T, C_D)
        else:
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert N_CTX <= self.n_ctx
            x = torch.cat([ctx, x], dim=1)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)
            x_level = torch.cat([ctx_level, x_level], dim=1)
            if x_r_level is not None and ctx_r_level is not None:
                x_r_level = torch.cat([ctx_r_level, x_r_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # ========== Build Conditioning: (B, S, P, cond_dim) ==========
        level_emb = (
            self.int_pe(x_level.reshape(B * S * P))
            .reshape(B, S, P, self.C_D)
            .to(x)
        )  # (B, S, P, C_D)
        if x_r_level is not None:
            r_emb = (
                self.int_pe(x_r_level.reshape(B * S * P))
                .reshape(B, S, P, self.C_D)
                .to(x)
            )
            level_emb = level_emb + r_emb

        feat_expanded = feat[:, None].expand(-1, S, -1, -1)  # (B, S, P, feat_dim)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(
            -1, -1, P, -1
        )  # (B, S, P, C_D)

        cond_parts = [
            self.hard_ln_feat(feat_expanded),
            self.hard_ln_cd(level_emb),
            self.hard_ln_cd(ctx_pos_emb_expanded),
        ]

        # Trajectory PE as 4th conditioning component
        if self.n_traj_tokens > 0:
            traj_ids = self._sample_traj_ids(B, P, device)  # (B, P)
            traj_pe_feat = self.traj_pe(
                traj_ids.reshape(-1).float()
            ).reshape(B, P, self.C_D)  # (B, P, C_D)
            traj_pe_expanded = traj_pe_feat[:, None, :, :].expand(
                -1, S, -1, -1
            )  # (B, S, P, C_D)
            cond_parts.append(self.hard_ln_cd(traj_pe_expanded))

        cond_bsp = torch.cat(cond_parts, dim=-1)  # (B, S, P, cond_dim)

        # ========== Embed Input ==========
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, P, C_D)

        # Time binding (no-RoPE path)
        if not self.time_rope_flag and self.add_time_to_feat:
            time_emb_expanded = ctx_pos_emb[:, :, None, :].expand(
                -1, -1, P, -1
            )  # (B, S, P, C_D)
            x_emb = self.time_bind_proj(
                torch.cat([x_emb, time_emb_expanded], dim=-1)
            )

        # ========== Prepare Add Tokens ==========
        if add_tokens is not None:
            A = add_tokens.shape[1]
            assert add_tokens_mask is not None
        else:
            A = 0
            add_tokens = torch.zeros(B, 0, self.C_D, device=device)
            add_tokens_mask = torch.zeros(B, 0, device=device).bool()

        # ========== Prepare RoPE ==========
        if self.time_rope_flag:
            # t_rope: (B*P, T) -- only noisy frame timestamps
            t_rope = (
                x_time[:, :, None]
                .expand(-1, -1, P)
                .permute(0, 2, 1)
                .reshape(B * P, T)
            )  # (B*P, T)
        else:
            t_rope = None

        # ========== Spatial Positions for Split RoPE ==========
        if self.spatial_rope_flag:
            sp = self._compute_spatial_positions(x, x_mask, N_CTX, B, S, P)
            x_9d_spatial = sp["x_9d_spatial"]  # (B*P, S, 9)
            mean_ctx_9d = sp["mean_ctx_9d"]     # (B, P, 9)
            mean_x_9d = sp["mean_x_9d"]         # (B, P, 9)
        else:
            x_9d_spatial = None
            mean_ctx_9d = None
            mean_x_9d = None

        # ========== Build Attention Masks ==========
        R_blk = self.blocks[0].R  # internal registers from DiT block

        masks = self._build_register_masks(
            x_mask, add_tokens_mask, S, P, T, N_CTX, A, R_blk, device
        )
        attn_mask_t = masks["attn_mask_t"]  # (B*(P+Rp), n_head, L_t, L_t)
        attn_mask_p = masks["attn_mask_p"]  # (B*(S+Rt), n_head, L_p, L_p)

        # ========== State Initialization ==========
        x_state = x_emb.clone()  # (B, S, P, C_D)
        skip_list = [x_emb.reshape(B * S, P, self.C_D).clone()]

        # Register states -- initialized to None (layer 0 uses raw embeddings)
        rt_state = None   # (B, Rt, P, C_D) -- Rt rows, P entity columns
        rp_state = None   # (B, S, Rp, C_D) -- S time rows, Rp register columns
        corner_state = None  # (B, Rt, Rp, C_D) -- corner block

        # Layer-0 mixing: save initial embedding
        x_state_0 = x_emb.clone()  # (B, S, P, C_D)

        t_layer_idx = 0
        p_layer_idx = 0
        a_layer_idx = 0
        b_layer_idx = 0

        # ========== Main Layer Loop ==========
        for layer_idx, lt in enumerate(self.layer_type_string):
            block = self.blocks[layer_idx]
            cond_binder = self.cond_binders[layer_idx]

            # --- Build register tokens for this layer ---
            # Rt: (B, Rt, C_D) broadcast to all P+Rp columns
            if Rt > 0:
                rt_emb_layer = self.rt_embeds[layer_idx].expand(
                    B, -1, -1
                )  # (B, Rt, C_D)
                if rt_state is None:
                    # Layer 0: init from learnable embedding
                    # rt_state_col: (B, Rt, P, C_D) -- one per entity column
                    rt_cur = rt_emb_layer[:, :, None, :].expand(
                        -1, -1, P, -1
                    )  # (B, Rt, P, C_D)
                else:
                    # Layer N>0: previous output + additive delta
                    rt_cur = rt_state + rt_emb_layer[:, :, None, :].expand(
                        -1, -1, P, -1
                    )

            if Rp > 0:
                rp_emb_layer = self.rp_embeds[layer_idx].expand(
                    B, -1, -1
                )  # (B, Rp, C_D)
                if rp_state is None:
                    rp_cur = rp_emb_layer[:, None, :, :].expand(
                        -1, S, -1, -1
                    )  # (B, S, Rp, C_D)
                else:
                    rp_cur = rp_state + rp_emb_layer[:, None, :, :].expand(
                        -1, S, -1, -1
                    )

            if Rt > 0 and Rp > 0:
                corner_emb_layer = self.corner_embeds[layer_idx].reshape(
                    1, Rt, Rp, self.C_D
                ).expand(B, -1, -1, -1)  # (B, Rt, Rp, C_D)
                if corner_state is None:
                    corner_cur = corner_emb_layer
                else:
                    corner_cur = corner_state + corner_emb_layer

            if lt == "t":
                # ===== t-layer: self-attention across S+Rt per entity column =====

                # Layer-0 mixing (gradient highway)
                if self.n_t_layers > 1 and t_layer_idx > 0:
                    x_mix_input = torch.cat([x_state_0, x_state], dim=-1)
                    x_state = self.layer0_mix[t_layer_idx - 1](x_mix_input)

                # Reshape main tokens to per-column: (B*(P+Rp), S+Rt, C_D)
                # Main entity columns: (B*P, S, C_D)
                x_traj = x_state.permute(0, 2, 1, 3).reshape(
                    B * P, S, self.C_D
                )

                # For each main entity column j: sequence = [Rt_regs, main_tokens]
                # Rt registers for main columns: (B, Rt, P, C_D) -> (B*P, Rt, C_D)
                if Rt > 0:
                    rt_for_main = rt_cur.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)
                    main_col_seq = torch.cat(
                        [rt_for_main, x_traj], dim=1
                    )  # (B*P, Rt+S, C_D)
                else:
                    main_col_seq = x_traj  # (B*P, S, C_D)

                # Rp register columns: sequence = [corner, Rp_main_time_tokens]
                # rp_cur: (B, S, Rp, C_D) -> reshape to Rp columns
                if Rp > 0:
                    rp_col_seq_main = rp_cur.permute(
                        0, 2, 1, 3
                    ).reshape(B * Rp, S, self.C_D)  # (B*Rp, S, C_D)

                    if Rt > 0:
                        # Corner tokens for Rp columns: (B, Rt, Rp, C_D) -> (B*Rp, Rt, C_D)
                        corner_for_rp = corner_cur.permute(
                            0, 2, 1, 3
                        ).reshape(B * Rp, Rt, self.C_D)
                        rp_col_seq = torch.cat(
                            [corner_for_rp, rp_col_seq_main], dim=1
                        )  # (B*Rp, Rt+S, C_D)
                    else:
                        rp_col_seq = rp_col_seq_main

                    # Combine all columns: (B*(P+Rp), Rt+S, C_D)
                    traj_seq = torch.cat(
                        [main_col_seq, rp_col_seq], dim=0
                    )  # (B*(P+Rp), Rt+S, C_D)
                else:
                    traj_seq = main_col_seq  # (B*P, S+Rt, C_D)

                # --- Conditioning ---
                cond_t = cond_binder(cond_bsp)  # (B, S, P, C_D)
                cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                    B * P, S, self.C_D
                )  # (B*P, S, C_D)

                # Rt conditioning: mean over all time tokens
                if Rt > 0:
                    cond_rt_mean = cond_t_per_traj.mean(
                        dim=1, keepdim=True
                    ).expand(-1, Rt, -1)  # (B*P, Rt, C_D)
                    main_cond = torch.cat(
                        [cond_rt_mean, cond_t_per_traj], dim=1
                    )  # (B*P, Rt+S, C_D)
                else:
                    main_cond = cond_t_per_traj

                if Rp > 0:
                    # Rp columns: mean conditioning (zero or global mean)
                    cond_rp_global = cond_t_per_traj.reshape(
                        B, P, S, self.C_D
                    ).mean(dim=1)  # (B, S, C_D) -- mean over entities
                    cond_rp_tiled = cond_rp_global[:, None].expand(
                        -1, Rp, -1, -1
                    ).reshape(B * Rp, S, self.C_D)  # (B*Rp, S, C_D)

                    if Rt > 0:
                        cond_corner = cond_rp_tiled.mean(
                            dim=1, keepdim=True
                        ).expand(-1, Rt, -1)  # (B*Rp, Rt, C_D)
                        rp_cond = torch.cat(
                            [cond_corner, cond_rp_tiled], dim=1
                        )  # (B*Rp, Rt+S, C_D)
                    else:
                        rp_cond = cond_rp_tiled

                    cond_traj = torch.cat(
                        [main_cond, rp_cond], dim=0
                    )  # (B*(P+Rp), Rt+S, C_D)
                else:
                    cond_traj = main_cond

                # --- Set spatial RoPE params ---
                if self.spatial_rope_flag:
                    # Spatial positions: (B*P, S, 9) for main entity columns.
                    # Rp columns need dummy spatial positions (zeros) since they
                    # don't represent real entities, but the batch dimension is
                    # B*(P+Rp) so we must provide matching shape.
                    if Rp > 0:
                        rp_dummy_spatial = torch.zeros(
                            B * Rp, S, 9, device=device, dtype=x_9d_spatial.dtype
                        )
                        x_9d_spatial_ext = torch.cat(
                            [x_9d_spatial, rp_dummy_spatial], dim=0
                        )  # (B*(P+Rp), S, 9)
                    else:
                        x_9d_spatial_ext = x_9d_spatial
                    block.set_spatial_rope_params(
                        spatial_positions=x_9d_spatial_ext,
                        num_external_regs_spatial=Rt,
                    )

                # --- RoPE time: extend for Rp columns ---
                if t_rope is not None and Rp > 0:
                    # Rp columns don't get temporal RoPE (excluded via num_external_regs)
                    # But we need to provide dummy t for the full batch dim
                    # Since block processes B*(P+Rp) but t_rope is (B*P, T),
                    # we need to extend with dummy zeros for Rp columns
                    t_rope_rp = torch.zeros(
                        B * Rp, T, device=device, dtype=t_rope.dtype
                    )
                    t_rope_ext = torch.cat([t_rope, t_rope_rp], dim=0)
                else:
                    t_rope_ext = t_rope

                # --- Run t-block ---
                traj_out = block(
                    x=traj_seq,
                    cond=cond_traj,
                    t=t_rope_ext,
                    num_external_regs=Rt + N_CTX,
                    attn_mask=attn_mask_t,
                )  # (B*(P+Rp), Rt+S, C_D) -- internal regs stripped

                # --- Extract outputs ---
                # Main entity columns: first B*P entries
                main_out = traj_out[:B * P]  # (B*P, Rt+S, C_D)

                if Rt > 0:
                    rt_out_main = main_out[:, :Rt]  # (B*P, Rt, C_D)
                    x_out = main_out[:, Rt:]         # (B*P, S, C_D)
                else:
                    x_out = main_out

                x_state = x_out.reshape(B, P, S, self.C_D).permute(
                    0, 2, 1, 3
                )  # (B, S, P, C_D)

                # Update Rt state from main columns
                if Rt > 0:
                    rt_state = rt_out_main.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, Rt, P, C_D)

                # Rp column outputs
                if Rp > 0:
                    rp_out = traj_out[B * P:]  # (B*Rp, Rt+S, C_D)
                    if Rt > 0:
                        corner_out_from_rp = rp_out[:, :Rt]  # (B*Rp, Rt, C_D)
                        rp_main_out = rp_out[:, Rt:]          # (B*Rp, S, C_D)
                    else:
                        rp_main_out = rp_out
                        corner_out_from_rp = None

                    rp_state = rp_main_out.reshape(
                        B, Rp, S, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, S, Rp, C_D)

                    if Rt > 0 and corner_out_from_rp is not None:
                        corner_state = corner_out_from_rp.reshape(
                            B, Rp, Rt, self.C_D
                        ).permute(0, 2, 1, 3)  # (B, Rt, Rp, C_D)

                # --- Apply register FFNs ---
                if Rt > 0:
                    # rt_state: (B, Rt, P, C_D) -> (B*P, Rt, C_D) for FFN
                    rt_flat = rt_state.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )
                    ada_rt_cond = self.ada_rt_embeds[layer_idx].expand(
                        B * P, -1, -1
                    )  # (B*P, Rt, C_D)
                    rt_flat = self.reg_rt_blocks[layer_idx](
                        rt_flat, ada_rt_cond
                    )
                    rt_state = rt_flat.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, Rt, P, C_D)

                if Rp > 0:
                    # rp_state: (B, S, Rp, C_D) -> (B*S, Rp, C_D) for FFN
                    rp_flat = rp_state.reshape(B * S, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S, -1, -1
                    )  # (B*S, Rp, C_D)
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(
                        B, S, Rp, self.C_D
                    )  # (B, S, Rp, C_D)

                if Rt > 0 and Rp > 0:
                    # corner_state: (B, Rt, Rp, C_D) -> (B, Rt*Rp, C_D)
                    corner_flat = corner_state.reshape(B, Rt * Rp, self.C_D)
                    ada_corner_cond = self.ada_corner_embeds[layer_idx].expand(
                        B, -1, -1
                    )  # (B, Rt*Rp, C_D)
                    corner_flat = self.reg_corner_blocks[layer_idx](
                        corner_flat, ada_corner_cond
                    )
                    corner_state = corner_flat.reshape(
                        B, Rt, Rp, self.C_D
                    )  # (B, Rt, Rp, C_D)

                # Collect skip from t-layer
                skip_list.append(
                    x_state.reshape(B * S, P, self.C_D).clone()
                )
                t_layer_idx += 1

            elif lt == "p":
                # ===== p-layer: cross-attention across P+Rp per timestep row =====

                # Reshape main tokens to per-row: (B*S, P, C_D)
                x_flat = x_state.reshape(B * S, P, self.C_D)

                # Rp register tokens for main rows: (B, S, Rp, C_D) -> (B*S, Rp, C_D)
                if Rp > 0:
                    rp_for_main = rp_cur.reshape(
                        B * S, Rp, self.C_D
                    )  # (B*S, Rp, C_D)

                # Rt register rows: each sees [entities + corner]
                # rt_cur: (B, Rt, P, C_D) -> (B*Rt, P, C_D)
                if Rt > 0:
                    rt_row_ent = rt_cur.reshape(
                        B * Rt, P, self.C_D
                    )  # (B*Rt, P, C_D)

                    if Rp > 0:
                        # Corner for Rt rows: (B, Rt, Rp, C_D) -> (B*Rt, Rp, C_D)
                        corner_for_rt = corner_cur.reshape(
                            B * Rt, Rp, self.C_D
                        )

                # Language tokens tiled over S+Rt: (B*(S+Rt), A, C_D)
                S_ext = S + Rt
                add_tiled_main = (
                    add_tokens.unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(B * S, A, self.C_D)
                )
                if Rt > 0:
                    add_tiled_rt = (
                        add_tokens.unsqueeze(1)
                        .expand(-1, Rt, -1, -1)
                        .reshape(B * Rt, A, self.C_D)
                    )

                # Build per-row sequences:
                # Main rows: [add_tokens, Rp_registers, entities]
                main_parts = [add_tiled_main]
                if Rp > 0:
                    main_parts.append(rp_for_main)
                main_parts.append(x_flat)
                main_row_seq = torch.cat(main_parts, dim=1)  # (B*S, A+Rp+P, C_D)

                # Rt register rows: [add_tokens, corner, entities]
                if Rt > 0:
                    rt_parts = [add_tiled_rt]
                    if Rp > 0:
                        rt_parts.append(corner_for_rt)
                    rt_parts.append(rt_row_ent)
                    rt_row_seq = torch.cat(
                        rt_parts, dim=1
                    )  # (B*Rt, A+Rp+P, C_D)

                    # Combine all rows: (B*(S+Rt), A+Rp+P, C_D)
                    p_seq = torch.cat(
                        [main_row_seq, rt_row_seq], dim=0
                    )
                else:
                    p_seq = main_row_seq  # (B*S, A+Rp+P, C_D)

                # --- Conditioning ---
                cond_p = cond_binder(cond_bsp)  # (B, S, P, C_D)
                cond_flat = cond_p.reshape(B * S, P, self.C_D)

                # Zero conditioning for add_tokens + Rp registers
                cond_prefix_main = torch.zeros(
                    B * S, A + Rp, self.C_D,
                    device=device, dtype=cond_flat.dtype,
                )
                main_cond_full = torch.cat(
                    [cond_prefix_main, cond_flat], dim=1
                )  # (B*S, A+Rp+P, C_D)

                if Rt > 0:
                    # Rt rows: entity conditioning from mean over time
                    cond_rt_ent = cond_bsp.mean(dim=1)  # (B, P, cond_dim)
                    cond_rt_ent_proj = cond_binder(
                        cond_rt_ent[:, None].expand(-1, Rt, -1, -1)
                            .reshape(B, Rt, P, -1)
                    ).reshape(B * Rt, P, self.C_D)  # (B*Rt, P, C_D)

                    cond_prefix_rt = torch.zeros(
                        B * Rt, A + Rp, self.C_D,
                        device=device, dtype=cond_flat.dtype,
                    )
                    rt_cond_full = torch.cat(
                        [cond_prefix_rt, cond_rt_ent_proj], dim=1
                    )  # (B*Rt, A+Rp+P, C_D)

                    cond_full = torch.cat(
                        [main_cond_full, rt_cond_full], dim=0
                    )  # (B*(S+Rt), A+Rp+P, C_D)
                else:
                    cond_full = main_cond_full

                # --- Set dual spatial RoPE for p-layer ---
                if self.spatial_rope_flag and self._p_blocks_split_rope:
                    # Dual spatial positions: (B, P, 9) -> tile to (B*S_ext, P, 9)
                    # where S_ext = S + Rt (total rows being processed)
                    mean_ctx_tiled = mean_ctx_9d[:, None].expand(
                        -1, S_ext, -1, -1
                    ).reshape(B * S_ext, P, 9)  # (B*(S+Rt), P, 9)
                    mean_x_tiled = mean_x_9d[:, None].expand(
                        -1, S_ext, -1, -1
                    ).reshape(B * S_ext, P, 9)  # (B*(S+Rt), P, 9)
                    self.p_blocks_rope[p_layer_idx].set_dual_spatial_rope_params(
                        pos_a=mean_ctx_tiled,
                        pos_b=mean_x_tiled,
                        num_external_regs=A + Rp,
                    )
                    p_block = self.p_blocks_rope[p_layer_idx]
                else:
                    p_block = block

                # --- Run p-block ---
                p_out = p_block(
                    x=p_seq,
                    cond=cond_full,
                    t=None,
                    num_external_regs=A + Rp,
                    attn_mask=attn_mask_p,
                )  # (B*(S+Rt), A+Rp+P, C_D) -- internal regs stripped

                # --- Extract outputs ---
                # Main rows
                main_row_out = p_out[:B * S]  # (B*S, A+Rp+P, C_D)
                x_flat_out = main_row_out[:, A + Rp:]  # (B*S, P, C_D)
                x_state = x_flat_out.reshape(B, S, P, self.C_D)

                if Rp > 0:
                    rp_out_main = main_row_out[:, A:A + Rp]  # (B*S, Rp, C_D)
                    rp_state = rp_out_main.reshape(
                        B, S, Rp, self.C_D
                    )  # (B, S, Rp, C_D)

                if Rt > 0:
                    rt_row_out = p_out[B * S:]  # (B*Rt, A+Rp+P, C_D)
                    rt_ent_out = rt_row_out[:, A + Rp:]  # (B*Rt, P, C_D)
                    rt_state = rt_ent_out.reshape(
                        B, Rt, P, self.C_D
                    )  # (B, Rt, P, C_D)

                    if Rp > 0:
                        corner_out = rt_row_out[:, A:A + Rp]  # (B*Rt, Rp, C_D)
                        corner_state = corner_out.reshape(
                            B, Rt, Rp, self.C_D
                        )  # (B, Rt, Rp, C_D)

                # --- Apply register FFNs ---
                if Rt > 0:
                    rt_flat = rt_state.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )
                    ada_rt_cond = self.ada_rt_embeds[layer_idx].expand(
                        B * P, -1, -1
                    )
                    rt_flat = self.reg_rt_blocks[layer_idx](
                        rt_flat, ada_rt_cond
                    )
                    rt_state = rt_flat.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)

                if Rp > 0:
                    rp_flat = rp_state.reshape(B * S, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S, Rp, self.C_D)

                if Rt > 0 and Rp > 0:
                    corner_flat = corner_state.reshape(B, Rt * Rp, self.C_D)
                    ada_corner_cond = self.ada_corner_embeds[layer_idx].expand(
                        B, -1, -1
                    )
                    corner_flat = self.reg_corner_blocks[layer_idx](
                        corner_flat, ada_corner_cond
                    )
                    corner_state = corner_flat.reshape(B, Rt, Rp, self.C_D)

                p_layer_idx += 1

            elif lt == "a":
                # ===== a-layer: T-attention on registers ONLY [Rt, P+Rp] =====
                # Main tokens pass through unchanged. Only register rows evolve.

                # Per-column sequences: Rt tokens per column
                # Main entity columns: rt_cur (B, Rt, P, C_D) -> (B*P, Rt, C_D)
                if Rt > 0:
                    rt_main_cols = rt_cur.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)

                    if Rp > 0:
                        # Rp register columns: corner (B, Rt, Rp, C_D) -> (B*Rp, Rt, C_D)
                        corner_cols = corner_cur.permute(
                            0, 2, 1, 3
                        ).reshape(B * Rp, Rt, self.C_D)  # (B*Rp, Rt, C_D)
                        a_seq = torch.cat(
                            [rt_main_cols, corner_cols], dim=0
                        )  # (B*(P+Rp), Rt, C_D)
                    else:
                        a_seq = rt_main_cols  # (B*P, Rt, C_D)

                    # Conditioning: mean over time for each entity column
                    cond_a = cond_binder(cond_bsp)  # (B, S, P, C_D)
                    cond_a_mean = cond_a.mean(dim=1)  # (B, P, C_D)
                    cond_a_per_col = cond_a_mean[:, :, None, :].expand(
                        -1, -1, Rt, -1
                    ).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)

                    if Rp > 0:
                        cond_rp_global = cond_a.mean(dim=(1, 2))  # (B, C_D)
                        cond_rp_cols = cond_rp_global[:, None, None, :].expand(
                            -1, Rp, Rt, -1
                        ).reshape(B * Rp, Rt, self.C_D)  # (B*Rp, Rt, C_D)
                        cond_a_full = torch.cat(
                            [cond_a_per_col, cond_rp_cols], dim=0
                        )  # (B*(P+Rp), Rt, C_D)
                    else:
                        cond_a_full = cond_a_per_col

                    # Run a-block (no RoPE, no attn_mask -- all registers attend freely)
                    a_out = self.a_blocks[a_layer_idx](
                        x=a_seq,
                        cond=cond_a_full,
                        t=None,
                        num_external_regs=0,
                        attn_mask=None,
                    )

                    # Extract outputs: update Rt and corner states
                    rt_main_out = a_out[:B * P]  # (B*P, Rt, C_D)
                    rt_state = rt_main_out.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, Rt, P, C_D)

                    if Rp > 0:
                        corner_out_a = a_out[B * P:]  # (B*Rp, Rt, C_D)
                        corner_state = corner_out_a.reshape(
                            B, Rp, Rt, self.C_D
                        ).permute(0, 2, 1, 3)  # (B, Rt, Rp, C_D)

                # Rp was NOT in attention: apply per-layer delta via rp_cur
                if Rp > 0:
                    rp_state = rp_cur

                # Apply register FFNs (all 3 types)
                if Rt > 0:
                    rt_flat = rt_state.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )
                    ada_rt_cond = self.ada_rt_embeds[layer_idx].expand(
                        B * P, -1, -1
                    )
                    rt_flat = self.reg_rt_blocks[layer_idx](
                        rt_flat, ada_rt_cond
                    )
                    rt_state = rt_flat.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)

                if Rp > 0:
                    rp_flat = rp_state.reshape(B * S, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S, Rp, self.C_D)

                if Rt > 0 and Rp > 0:
                    corner_flat = corner_state.reshape(B, Rt * Rp, self.C_D)
                    ada_corner_cond = self.ada_corner_embeds[layer_idx].expand(
                        B, -1, -1
                    )
                    corner_flat = self.reg_corner_blocks[layer_idx](
                        corner_flat, ada_corner_cond
                    )
                    corner_state = corner_flat.reshape(B, Rt, Rp, self.C_D)

                a_layer_idx += 1

            elif lt == "b":
                # ===== b-layer: P-attention on registers ONLY [S+Rt, Rp] =====
                # Main tokens pass through unchanged. Only register columns evolve.

                # Per-row sequences: Rp tokens per row
                if Rp > 0:
                    # Main time rows: rp_cur (B, S, Rp, C_D) -> (B*S, Rp, C_D)
                    rp_main_rows = rp_cur.reshape(
                        B * S, Rp, self.C_D
                    )  # (B*S, Rp, C_D)

                    if Rt > 0:
                        # Rt register rows: corner (B, Rt, Rp, C_D) -> (B*Rt, Rp, C_D)
                        corner_rows = corner_cur.reshape(
                            B * Rt, Rp, self.C_D
                        )  # (B*Rt, Rp, C_D)
                        b_seq = torch.cat(
                            [rp_main_rows, corner_rows], dim=0
                        )  # (B*(S+Rt), Rp, C_D)
                    else:
                        b_seq = rp_main_rows  # (B*S, Rp, C_D)

                    # Conditioning: mean over entities for each time row
                    cond_b = cond_binder(cond_bsp)  # (B, S, P, C_D)
                    cond_b_mean = cond_b.mean(dim=2)  # (B, S, C_D)
                    cond_b_per_row = cond_b_mean[:, :, None, :].expand(
                        -1, -1, Rp, -1
                    ).reshape(B * S, Rp, self.C_D)  # (B*S, Rp, C_D)

                    if Rt > 0:
                        cond_rt_global = cond_b.mean(dim=(1, 2))  # (B, C_D)
                        cond_rt_rows = cond_rt_global[:, None, None, :].expand(
                            -1, Rt, Rp, -1
                        ).reshape(B * Rt, Rp, self.C_D)  # (B*Rt, Rp, C_D)
                        cond_b_full = torch.cat(
                            [cond_b_per_row, cond_rt_rows], dim=0
                        )  # (B*(S+Rt), Rp, C_D)
                    else:
                        cond_b_full = cond_b_per_row

                    # Run b-block (no RoPE, no attn_mask -- all registers attend freely)
                    b_out = self.b_blocks[b_layer_idx](
                        x=b_seq,
                        cond=cond_b_full,
                        t=None,
                        num_external_regs=0,
                        attn_mask=None,
                    )

                    # Extract outputs: update Rp and corner states
                    rp_main_out = b_out[:B * S]  # (B*S, Rp, C_D)
                    rp_state = rp_main_out.reshape(
                        B, S, Rp, self.C_D
                    )  # (B, S, Rp, C_D)

                    if Rt > 0:
                        corner_out_b = b_out[B * S:]  # (B*Rt, Rp, C_D)
                        corner_state = corner_out_b.reshape(
                            B, Rt, Rp, self.C_D
                        )  # (B, Rt, Rp, C_D)

                # Rt was NOT in attention: apply per-layer delta via rt_cur
                if Rt > 0:
                    rt_state = rt_cur

                # Apply register FFNs (all 3 types)
                if Rt > 0:
                    rt_flat = rt_state.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )
                    ada_rt_cond = self.ada_rt_embeds[layer_idx].expand(
                        B * P, -1, -1
                    )
                    rt_flat = self.reg_rt_blocks[layer_idx](
                        rt_flat, ada_rt_cond
                    )
                    rt_state = rt_flat.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)

                if Rp > 0:
                    rp_flat = rp_state.reshape(B * S, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S, Rp, self.C_D)

                if Rt > 0 and Rp > 0:
                    corner_flat = corner_state.reshape(B, Rt * Rp, self.C_D)
                    ada_corner_cond = self.ada_corner_embeds[layer_idx].expand(
                        B, -1, -1
                    )
                    corner_flat = self.reg_corner_blocks[layer_idx](
                        corner_flat, ada_corner_cond
                    )
                    corner_state = corner_flat.reshape(B, Rt, Rp, self.C_D)

                b_layer_idx += 1

        # ========== Output Projection ==========
        # Use t-layer skips only (like compact/dense v1)
        x_out = torch.cat(skip_list, dim=-1)  # (B*S, P, (n_t+1)*C_D)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, P, self.input_dim)
        return ret  # (B, S, P, input_dim)
