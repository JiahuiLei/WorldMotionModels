"""Mixture-of-Tokens DiT block for V11 Compact.

DiTBlockMoT provides token-type-specific processing with shared attention:
- 3 separate adaLN-Zero projections (reg, ctx, x)
- 3 separate LayerNorm pairs (norm1, norm2 per type)
- 3 separate FFNs (reg, ctx, x)
- Shared QKV attention (with optional split RoPE)

Inherits from DiTBlockSplitRoPE for split-head temporal+spatial RoPE support.

torch.compile audit (class-level):
    All index slicing uses R (init constant), n_external_reg (init constant),
    and n_ctx (forward arg, constant per training config). No dynamic control
    flow on tensor values. The 3 ada projections, 3 norm pairs, and 3 FFNs
    are standard nn.Linear/nn.Sequential with GELU. All operations are
    static-shape tensor ops. No closures, no create_block_mask, no
    data-dependent branches. Compile-safe.

    Methods in forward call graph:
    - _prepare_inputs: cat, expand, zeros -- compile-safe
    - _apply_ada: elementwise mul+add with optional tanh -- compile-safe
    - _add_gate: elementwise mul+add with optional tanh -- compile-safe
    - _attn_explicit_time (from DiTBlockSplitRoPE): qkv projection, split
      RoPE, SDPA -- compile-safe (spatial RoPE state is set before compiled
      region via set_spatial_rope_params)
    - norm1_{reg,ctx,x}, norm2_{reg,ctx,x}: nn.LayerNorm -- compile-safe
    - ada_{reg,ctx,x}: nn.Linear or nn.Sequential(Linear) -- compile-safe
    - mlp_{reg,ctx,x}: nn.Sequential(Linear, GELU, ..., Linear) -- compile-safe
    - torch.cat, tensor slicing: static shapes -- compile-safe
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from umm.models.umm_v11.world_models.rope_utils import DiTBlockSplitRoPE


class DiTBlockMoT(DiTBlockSplitRoPE):
    """DiT block with Mixture-of-Tokens routing.

    Three token types (registers, context, noisy/trace) get type-specific
    adaLN, LayerNorm, and FFN processing, but share a single QKV attention.

    Sequence layout after _prepare_inputs:
        [internal_regs(R), external_regs(n_external_reg), ctx(N_CTX), x(T)]

    MoT routing groups:
        reg = positions [:R + n_external_reg]  (internal + external regs)
        ctx = positions [R+n_ext : R+n_ext+N_CTX]
        x   = positions [R+n_ext+N_CTX :]

    Args:
        n_external_reg: Number of external register tokens
            (n_reg_c + n_reg_x + n_reg_a from compact model).
        n_temporal_heads: Heads for temporal RoPE (from DiTBlockSplitRoPE).
        n_spatial_dims: Spatial RoPE dimensions (default 9).
        **kwargs: DiTBlockTimeOnly arguments (C, num_heads, mlp_hidden_dims,
            num_registers, ada_scale_bias_bound, gate_bound, ada_n_layers,
            C_feat, qk_norm).
    """

    def __init__(
        self,
        n_external_reg: int,
        n_temporal_heads: int,
        n_spatial_dims: int = 9,
        **kwargs,
    ):
        # Extract config values needed to build 3x variants BEFORE super()
        # creates (and we delete) the parent's single-stream versions.
        ada_n_layers = kwargs.get("ada_n_layers", 1)
        mlp_hidden_dims = kwargs.get("mlp_hidden_dims", None)
        C = kwargs.get("C", 128)
        C_feat = kwargs.get("C_feat", None)
        if C_feat is None:
            C_feat = C

        super().__init__(
            n_temporal_heads=n_temporal_heads,
            n_spatial_dims=n_spatial_dims,
            **kwargs,
        )

        self.n_external_reg = n_external_reg

        # --- Delete parent's single-stream components ---
        # (replaced with 3x type-specific versions below)
        del self.ada
        del self.norm1
        del self.norm2
        del self.mlp

        # --- 3x adaLN projections (reg / ctx / x) ---
        self.ada_reg = self._build_ada_module(C, C_feat, ada_n_layers)
        self.ada_ctx = self._build_ada_module(C, C_feat, ada_n_layers)
        self.ada_x = self._build_ada_module(C, C_feat, ada_n_layers)

        # --- 3x LayerNorm pairs (elementwise_affine=False, no learnable params) ---
        self.norm1_reg = nn.LayerNorm(C, elementwise_affine=False)
        self.norm1_ctx = nn.LayerNorm(C, elementwise_affine=False)
        self.norm1_x = nn.LayerNorm(C, elementwise_affine=False)
        self.norm2_reg = nn.LayerNorm(C, elementwise_affine=False)
        self.norm2_ctx = nn.LayerNorm(C, elementwise_affine=False)
        self.norm2_x = nn.LayerNorm(C, elementwise_affine=False)

        # --- 3x FFNs (reg / ctx / x) ---
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [int(C * 4.0)]
        self.mlp_reg = self._build_mlp_module(C, mlp_hidden_dims)
        self.mlp_ctx = self._build_mlp_module(C, mlp_hidden_dims)
        self.mlp_x = self._build_mlp_module(C, mlp_hidden_dims)

    @staticmethod
    def _build_ada_module(C: int, C_feat: int, ada_n_layers: int) -> nn.Module:
        """Build an adaLN-Zero projection (replicates DiTBlockTimeOnly logic).

        Args:
            C: Channel dimension.
            C_feat: Conditioning feature dimension.
            ada_n_layers: Number of projection layers.

        Returns:
            nn.Linear or nn.Sequential producing 6*C outputs.
        """
        if ada_n_layers == 1:
            ada = nn.Linear(C_feat, 6 * C)
            nn.init.zeros_(ada.bias)
            return ada
        else:
            _outdims = np.linspace(C_feat, 6 * C, ada_n_layers).astype(int).tolist()
            _outdims = _outdims[1:]
            layers = []
            _c_in = C_feat
            for _c_out in _outdims:
                layers.append(nn.Linear(_c_in, _c_out))
                layers.append(nn.LayerNorm(_c_out))
                layers.append(nn.LeakyReLU(0.2))
                _c_in = _c_out
            # Remove last LayerNorm + LeakyReLU (must not clamp/normalize final ada output)
            layers.pop()  # remove LeakyReLU
            layers.pop()  # remove LayerNorm
            return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp_module(C: int, mlp_hidden_dims: List[int]) -> nn.Sequential:
        """Build an MLP (replicates DiTBlockTimeOnly logic).

        Args:
            C: Input/output channel dimension.
            mlp_hidden_dims: List of hidden layer dimensions.

        Returns:
            nn.Sequential MLP with GELU activations.
        """
        layers: list = []
        in_dim = C
        for h_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate="tanh"))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, C))
        return nn.Sequential(*layers)

    def forward_attn_only(self, x, cond, t=None, num_external_regs=0, attn_mask=None):
        """Not supported -- MoT integrates ada routing into the full forward."""
        raise NotImplementedError(
            "DiTBlockMoT does not support split attn/MLP phases. "
            "Use forward() instead."
        )

    def forward_mlp_only(self, x_in, s2, sc2, g2):
        """Not supported -- MoT integrates ada routing into the full forward."""
        raise NotImplementedError(
            "DiTBlockMoT does not support split attn/MLP phases. "
            "Use forward() instead."
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_external_regs: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        n_ctx: int = 0,
    ) -> torch.Tensor:
        """MoT forward: type-specific ada/norm/FFN with shared attention.

        9-step flow:
          1. Prepend internal registers + route by type
          2. Type-specific adaLN projection (3 separate projections)
          3. Type-specific pre-attention modulation (norm1 + scale/shift)
          4. Concatenate for shared attention
          5. Shared QKV attention (with optional split RoPE)
          6. Route attention output by type
          7. Type-specific gate + residual (attention)
          8. Type-specific MLP (norm2 + scale/shift + FFN + gate + residual)
          9. Concatenate + strip internal registers

        Args:
            x: (B, L_input, C) where L_input = n_external_reg + N_CTX + T.
            cond: (B, L_input, C) or broadcastable conditioning.
            t: (B, T_time) or (T_time,) time values for temporal RoPE.
            num_external_regs: Total non-time tokens for temporal RoPE offset
                (= n_external_reg + N_CTX, set by caller).
            attn_mask: Optional boolean attention mask (B, H, L_total, L_total).
            n_ctx: Number of context tokens in the sequence. Required for
                MoT routing -- determines the ctx/x split point.

        Returns:
            (B, L_input, C) output with internal registers stripped.
        """
        assert x.shape[-1] == self.C

        # === Step 1: Prepend internal registers ===
        x_in, cond_in = self._prepare_inputs(x, cond)
        # x_in: (B, R + L_in, C)

        R = self.R
        n_ext = self.n_external_reg

        # Routing boundaries (after internal-register prepend)
        reg_end = R + n_ext       # end of all registers (internal + external)
        ctx_end = reg_end + n_ctx  # end of context tokens

        # Route tokens by type
        x_reg = x_in[:, :reg_end]          # (B, R+n_ext, C)
        x_ctx = x_in[:, reg_end:ctx_end]   # (B, N_CTX, C)
        x_x = x_in[:, ctx_end:]            # (B, T, C)

        cond_reg = cond_in[:, :reg_end]
        cond_ctx = cond_in[:, reg_end:ctx_end]
        cond_x = cond_in[:, ctx_end:]

        # === Step 2: Type-specific adaLN projection ===
        ada_reg_out = self.ada_reg(cond_reg)   # (B, R+n_ext, 6C)
        ada_ctx_out = self.ada_ctx(cond_ctx)   # (B, N_CTX, 6C)
        ada_x_out = self.ada_x(cond_x)        # (B, T, 6C)

        s1r, sc1r, g1r, s2r, sc2r, g2r = ada_reg_out.chunk(6, dim=-1)
        s1c, sc1c, g1c, s2c, sc2c, g2c = ada_ctx_out.chunk(6, dim=-1)
        s1x, sc1x, g1x, s2x, sc2x, g2x = ada_x_out.chunk(6, dim=-1)

        # === Step 3: Type-specific pre-attention modulation ===
        h_reg = self._apply_ada(self.norm1_reg(x_reg), s1r, sc1r)
        h_ctx = self._apply_ada(self.norm1_ctx(x_ctx), s1c, sc1c)
        h_x = self._apply_ada(self.norm1_x(x_x), s1x, sc1x)

        # === Step 4: Concatenate for shared attention ===
        h_all = torch.cat([h_reg, h_ctx, h_x], dim=1)  # (B, R+L_in, C)

        # === Step 5: Shared QKV attention (with split RoPE from parent) ===
        h_all = self._attn_explicit_time(
            h_all, t, num_external_regs, attn_mask_bool=attn_mask
        )

        # === Step 6: Route attention output by type ===
        attn_reg = h_all[:, :reg_end]
        attn_ctx = h_all[:, reg_end:ctx_end]
        attn_x = h_all[:, ctx_end:]

        # === Step 7: Type-specific gate + residual (attention) ===
        x_reg = self._add_gate(x_reg, g1r, attn_reg)
        x_ctx = self._add_gate(x_ctx, g1c, attn_ctx)
        x_x = self._add_gate(x_x, g1x, attn_x)

        # === Step 8: Type-specific MLP ===
        # reg
        h_reg = self._apply_ada(self.norm2_reg(x_reg), s2r, sc2r)
        h_reg = self.mlp_reg(h_reg)
        x_reg = self._add_gate(x_reg, g2r, h_reg)

        # ctx
        h_ctx = self._apply_ada(self.norm2_ctx(x_ctx), s2c, sc2c)
        h_ctx = self.mlp_ctx(h_ctx)
        x_ctx = self._add_gate(x_ctx, g2c, h_ctx)

        # x (noisy/trace)
        h_x = self._apply_ada(self.norm2_x(x_x), s2x, sc2x)
        h_x = self.mlp_x(h_x)
        x_x = self._add_gate(x_x, g2x, h_x)

        # === Step 9: Concatenate + strip internal registers ===
        x_out = torch.cat([x_reg, x_ctx, x_x], dim=1)  # (B, R+L_in, C)

        if R > 0:
            x_out = x_out[:, R:]  # strip internal registers -> (B, L_in, C)

        return x_out
