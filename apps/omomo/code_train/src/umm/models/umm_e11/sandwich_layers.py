"""Equivariant sandwich boundary layers for E11.

Implements the equivariant-to-invariant encoder (et+ep) and
invariant-to-equivariant decoder (dp+dt) that form the boundary
of the E11 sandwich architecture.

Equivariance proof:
    Under rotation R in SO(3):
    - VN features rotate: x -> x @ R^T
    - Registers rotate: r -> r @ R^T
    - Inner products are invariant: <x@R^T, r@R^T> = <x, r>
    - Free transformer operates on invariants: same output
    - Decoder: scalar_weights (invariant) × r@R^T (equivariant) = output@R^T
    - [Round 4] VN skip: scalar_weights × h_lifted@R^T = (scalar × h_lifted)@R^T
    Therefore: f(x@R^T) = f(x)@R^T. QED.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from umm.models.umm_e11.vn_layers import (
    VNLinear,
    VNLeakyReLU,
    VNLayerNorm,
    VNLinearLeakyReLU,
)

_EPS = 1e-6


class EquivariantEncoder(nn.Module):
    """Two-level equivariant encoder: et (trajectory) + ep (part).

    Produces invariant scalar features from equivariant VN input via
    learned registers and inner products.

    Architecture:
        1. VN layers lift 3 input channels to d_vn channels
        2. Per-part registers: pool lifted features across time
        3. Global register: pool per-part registers across parts, refine with VN
        4. Inner products extract invariant features:
           - Intra-trajectory: lifted_features · per_part_register  [d_vn per token]
           - Norms of lifted features                                [d_vn per token]
           - Inter-trajectory: per_part_register · global_register   [d_vn per part, broadcast]
           - Norms of per-part registers                             [d_vn per part, broadcast]
        5. [Round 4] Per-token lifted features returned for VN skip connection to decoder

    Args:
        d_vn: Number of VN channels for registers.
        input_channels: Number of input VN channels (3 for 9D PoseIO).
    """

    def __init__(self, d_vn: int = 32, input_channels: int = 3, cross_product_invariants: bool = False):
        super().__init__()
        self.d_vn = d_vn
        self.input_channels = input_channels
        self.cross_product_invariants = cross_product_invariants

        # et: Lift input to d_vn VN channels
        self.lift = nn.Sequential(
            VNLinearLeakyReLU(input_channels, d_vn),
            VNLinearLeakyReLU(d_vn, d_vn),
        )

        # ep: Refine global register from pooled per-part registers
        self.global_refine = nn.Sequential(
            VNLinearLeakyReLU(d_vn, d_vn),
        )

        # Invariant feature dimension:
        # Base: 4 * d_vn (norms_h, dots_h_rp, norms_rp, dots_rp_rg)
        # With cross products: +2 * d_vn (cross_h_rp_norm, cross_rp_rg_norm)
        # Cross products give |sin(θ)| — complementary to dot product's cos(θ)
        n_features = 6 if cross_product_invariants else 4
        self.invariant_dim = n_features * d_vn

        logging.info(
            f"EquivariantEncoder: d_vn={d_vn}, invariant_dim={self.invariant_dim}, "
            f"cross_product={cross_product_invariants}"
        )

    @torch.compiler.disable
    def forward(
        self,
        x_vn: torch.Tensor,
        T: int,
        P: int,
        mask: torch.Tensor,
    ):
        """Encode equivariant VN features to invariant scalars.

        Args:
            x_vn: [B, S, C_in, 3] equivariant VN features (S = T*P)
            T: number of timesteps (C+T_trace)
            P: number of parts
            mask: [B, S] bool, True = valid

        Returns:
            invariant: [B, S, invariant_dim] invariant scalar features
            r_part: [B, P, d_vn, 3] per-part equivariant registers (for decoder)
            r_global: [B, d_vn, 3] global equivariant register (for decoder)
            h_lifted: [B, S, d_vn, 3] per-token lifted VN features (for decoder skip)
        """
        B, S, C_in, D3 = x_vn.shape
        d_vn = self.d_vn

        # === et: Lift to d_vn channels ===
        h = self.lift(x_vn)  # [B, S, d_vn, 3]

        # === et: Per-part registers (pool across time) ===
        h_tp = h.reshape(B, T, P, d_vn, D3)  # [B, T, P, d_vn, 3]
        mask_tp = mask.reshape(B, T, P)  # [B, T, P]

        # Masked mean pool over T
        mask_float = mask_tp.float().unsqueeze(-1).unsqueeze(-1)  # [B, T, P, 1, 1]
        count_p = mask_float.sum(dim=1).clamp(min=1)  # [B, P, 1, 1]
        r_part = (h_tp * mask_float).sum(dim=1) / count_p  # [B, P, d_vn, 3]

        # === ep: Global register (pool per-part registers, then refine) ===
        # Only pool over parts that have at least one valid token
        part_valid = (count_p[:, :, 0, 0] > 0).float()  # [B, P] — 1 if part has valid tokens
        part_valid_exp = part_valid.unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]
        n_valid_parts = part_valid.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        r_global = (r_part * part_valid_exp).sum(dim=1) / n_valid_parts.unsqueeze(-1)  # [B, d_vn, 3]
        r_global = self.global_refine(r_global)  # [B, d_vn, 3]

        # === Extract invariant features via inner products ===

        # 1. Norms of lifted features: ||h[c]|| -> [B, S, d_vn]
        norms_h = (h.pow(2).sum(dim=-1) + _EPS).sqrt()  # [B, S, d_vn]

        # 2. Dot products: lifted features · per-part register
        # h: [B, T, P, d_vn, 3], r_part: [B, P, d_vn, 3]
        # Per-channel dot product: sum over spatial dim (3)
        r_part_expanded = r_part.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, T, P, d_vn, 3]
        dots_h_rp = (h_tp * r_part_expanded).sum(dim=-1)  # [B, T, P, d_vn]
        dots_h_rp = dots_h_rp.reshape(B, S, d_vn)  # [B, S, d_vn]

        # 3. Norms of per-part registers: ||r_part[c]|| -> [B, P, d_vn]
        norms_rp = (r_part.pow(2).sum(dim=-1) + _EPS).sqrt()  # [B, P, d_vn]
        # Broadcast over T
        norms_rp = norms_rp.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, S, d_vn)

        # 4. Dot products: per-part register · global register
        # r_part: [B, P, d_vn, 3], r_global: [B, d_vn, 3]
        r_global_expanded = r_global.unsqueeze(1).expand(-1, P, -1, -1)  # [B, P, d_vn, 3]
        dots_rp_rg = (r_part * r_global_expanded).sum(dim=-1)  # [B, P, d_vn]
        # Broadcast over T
        dots_rp_rg = dots_rp_rg.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, S, d_vn)

        # Concatenate all invariant features
        features = [norms_h, dots_h_rp, norms_rp, dots_rp_rg]

        if self.cross_product_invariants:
            # Cross-product norms: ||a × b|| = ||a|| ||b|| |sin(θ)|
            # Complements dot product (cos(θ)) → full angle information
            # Equivariance: ||R(a×b)|| = ||a×b|| (norms are rotation-invariant)

            # 5. ||h × r_part|| — per-token cross with per-part register [B,S,d_vn]
            cross_h_rp = torch.cross(h_tp, r_part_expanded, dim=-1)  # [B,T,P,d_vn,3]
            cross_h_rp_norm = (cross_h_rp.pow(2).sum(dim=-1) + _EPS).sqrt()  # [B,T,P,d_vn]
            cross_h_rp_norm = cross_h_rp_norm.reshape(B, S, d_vn)
            features.append(cross_h_rp_norm)

            # 6. ||r_part × r_global|| — per-part cross with global [B,P,d_vn]
            cross_rp_rg = torch.cross(r_part, r_global_expanded, dim=-1)  # [B,P,d_vn,3]
            cross_rp_rg_norm = (cross_rp_rg.pow(2).sum(dim=-1) + _EPS).sqrt()  # [B,P,d_vn]
            cross_rp_rg_norm = cross_rp_rg_norm.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, S, d_vn)
            features.append(cross_rp_rg_norm)

        invariant = torch.cat(features, dim=-1)
        # [B, S, (4 or 6) * d_vn]

        # [Round 4] Return per-token lifted features for VN skip connection
        return invariant, r_part, r_global, h


class EquivariantDecoder(nn.Module):
    """Equivariant decoder: dp (part) + dt (trajectory) + dskip (per-token).

    Reconstructs equivariant VN output from invariant scalar features
    using saved equivariant registers AND per-token VN features as basis vectors.

    Output = sum of (scalar_weight × equivariant_basis) for each output channel.
    scalar × equivariant = equivariant. Sum of equivariants = equivariant.

    [Round 4] Added VN skip connection: per-token lifted features from encoder
    provide per-token basis vectors, complementing the pooled registers.
    This preserves per-token directional information lost in inner-product encoding.

    Equivariance proof for skip connection:
        h_lifted rotates: h(Rx) = h(x) @ R^T (equivariant)
        w_token is scalar (invariant, from free transformer)
        w_token × h_lifted = equivariant
        Sum with register terms: equivariant + equivariant = equivariant. QED.

    Args:
        d_model: Dimension of free transformer output (scalar features).
        d_vn: Number of VN channels in registers.
        output_channels: Number of output VN channels (3 for 9D PoseIO).
    """

    def __init__(self, d_model: int, d_vn: int = 32, output_channels: int = 3,
                 skip_gate_by_noise: bool = False, max_noise_timestep: int = 100):
        super().__init__()
        self.d_model = d_model
        self.d_vn = d_vn
        self.output_channels = output_channels
        self.skip_gate_by_noise = skip_gate_by_noise
        self.max_noise_timestep = max_noise_timestep

        # dp: Weights for per-part register basis
        self.part_weight_proj = nn.Sequential(
            nn.Linear(d_model, d_vn * 2),
            nn.GELU(),
            nn.Linear(d_vn * 2, output_channels * d_vn),
        )

        # dt: Weights for global register basis
        self.global_weight_proj = nn.Sequential(
            nn.Linear(d_model, d_vn * 2),
            nn.GELU(),
            nn.Linear(d_vn * 2, output_channels * d_vn),
        )

        # [Round 4] dskip: Weights for per-token VN skip connection basis
        self.token_weight_proj = nn.Sequential(
            nn.Linear(d_model, d_vn * 2),
            nn.GELU(),
            nn.Linear(d_vn * 2, output_channels * d_vn),
        )

        # Initialize near zero for stable training (residual-like behavior)
        nn.init.zeros_(self.part_weight_proj[-1].weight)
        nn.init.zeros_(self.part_weight_proj[-1].bias)
        nn.init.zeros_(self.global_weight_proj[-1].weight)
        nn.init.zeros_(self.global_weight_proj[-1].bias)
        nn.init.zeros_(self.token_weight_proj[-1].weight)
        nn.init.zeros_(self.token_weight_proj[-1].bias)

        logging.info(
            f"EquivariantDecoder: d_model={d_model}, d_vn={d_vn}, "
            f"output_channels={output_channels}, skip=True, "
            f"skip_gate_by_noise={skip_gate_by_noise}"
        )

    def forward(
        self,
        x_scalar: torch.Tensor,
        r_part: torch.Tensor,
        r_global: torch.Tensor,
        h_lifted: torch.Tensor,
        T: int,
        P: int,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode invariant scalars to equivariant VN output.

        Args:
            x_scalar: [B, S, d_model] free transformer output (invariant)
            r_part: [B, P, d_vn, 3] per-part equivariant registers
            r_global: [B, d_vn, 3] global equivariant register
            h_lifted: [B, S, d_vn, 3] per-token lifted VN features (skip connection)
            T: number of timesteps
            P: number of parts
            noise_level: [B, S] optional diffusion timestep per token (for skip gating)

        Returns:
            [B, S, output_channels, 3] equivariant VN output
        """
        B, S, _ = x_scalar.shape

        # === dp: Per-part register reconstruction ===
        # Predict scalar weights: [B, S, output_channels * d_vn]
        w_part = self.part_weight_proj(x_scalar)
        w_part = w_part.reshape(B, T, P, self.output_channels, self.d_vn)
        # [B, T, P, C_out, d_vn]

        # r_part: [B, P, d_vn, 3] -> expand over T
        r_part_exp = r_part.unsqueeze(1).expand(-1, T, -1, -1, -1)
        # [B, T, P, d_vn, 3]

        # Weighted sum: [B, T, P, C_out, 3]
        out_part = torch.einsum("btpcd, btpdx -> btpcx", w_part, r_part_exp)

        # === dt: Global register reconstruction ===
        w_global = self.global_weight_proj(x_scalar)
        w_global = w_global.reshape(B, S, self.output_channels, self.d_vn)
        # [B, S, C_out, d_vn]

        # r_global: [B, d_vn, 3] -> expand over S
        r_global_exp = r_global.unsqueeze(1).expand(-1, S, -1, -1)
        # [B, S, d_vn, 3]

        # Weighted sum: [B, S, C_out, 3]
        out_global = torch.einsum("bscd, bsdx -> bscx", w_global, r_global_exp)

        # === [Round 4] dskip: Per-token VN skip connection ===
        w_token = self.token_weight_proj(x_scalar)
        w_token = w_token.reshape(B, S, self.output_channels, self.d_vn)
        # [B, S, C_out, d_vn]

        # h_lifted: [B, S, d_vn, 3] — per-token equivariant basis
        # Weighted sum: [B, S, C_out, 3]
        out_token = torch.einsum("bscd, bsdx -> bscx", w_token, h_lifted)

        # [H1] Noise-gated skip: attenuate skip at high noise levels
        # At t=0 (clean): gate=1.0 (full skip). At t=max (noise): gate=0.0 (no skip).
        # Rationale: h_lifted encodes noisy VN features during early DDIM steps.
        # This gate prevents input noise from contaminating the decoder output.
        # Equivariance preserved: scalar gate × equivariant = equivariant.
        if self.skip_gate_by_noise and noise_level is not None:
            gate = 1.0 - noise_level.float() / self.max_noise_timestep  # [B, S]
            gate = gate.clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
            out_token = out_token * gate

        # Combine: all three terms are equivariant, sum is equivariant
        out_part = out_part.reshape(B, S, self.output_channels, 3)
        output = out_part + out_global + out_token  # [B, S, C_out, 3]

        return output
