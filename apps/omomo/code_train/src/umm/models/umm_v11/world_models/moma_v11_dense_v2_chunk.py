"""MotionMachine V11 DenseV2 Chunk — Chunk-tokenized variant of DenseV2.

Architecture: Same I/O contract as DenseV2 but packs G timesteps into 1 token.
Working tensor goes from [S, P] to [S_c, P] where S_c = ceil(S/G). Each chunk
token encodes G frames.

           P cols          Rp cols
         +-------------+----------+
  S_c    |  main       | Rp       |
  rows   |  [S_c,P]    | [S_c,Rp] |
         +-------------+----------+
  Rt rows| Rt [Rt,P]   |RtxRp     |
         |             |[Rt,Rp]   |
         +-------------+----------+

The chunk layer operates identically to DenseV2 but at the chunk level.
Input/output are still (B, S, P, 9) — chunking and unchunking happen inside.

Reuses _RegisterAdaLNFFN from moma_v11_dense_v2.

Example::

    net = MotionMachineV11DenseV2Chunk(
        C_D=512, layer_type_string="tptptptp", Rt=4, Rp=4,
        chunk_size=4, noise_pe_dim=128,
    )
    out = net(x, x_mask, x_time, x_level, feat,
              ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
              add_tokens=lang, add_tokens_mask=lang_mask)
"""

import logging
import math
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
from umm.models.umm_v11.world_models.moma_v11_dense_v2 import (
    _RegisterAdaLNFFN,
)

logger = logging.getLogger(__name__)


class MotionMachineV11DenseV2Chunk(nn.Module):
    """Dense LMM V2 with chunk tokenization.

    Packs G consecutive frames into a single token. The layer loop operates
    on (B, S_c, P, C_D) where S_c = ceil(S_pad / G). Per-frame noise level
    is encoded as a 128-dim PE concatenated G times into the token, giving
    the network per-frame noise awareness within each chunk.

    Args:
        C_D: Unified channel width for all layers (default 512).
        feat_dim: Input feature dimension from identity layer.
        n_head: Number of attention heads.
        input_dim: Per-element input dimension (e.g., 9 for SE3 param).
        output_dim: Per-element output dimension. Defaults to input_dim.
        layer_type_string: String of layer types (e.g., "tptptptp").
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
        chunk_size: G — number of frames per chunk token.
        noise_pe_dim: Per-frame noise PE dimension (concatenated G times).
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
        chunk_size: int = 4,
        noise_pe_dim: int = 128,
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

        # =============== Chunk Parameters ===============
        self.G = chunk_size
        self.noise_pe_dim = noise_pe_dim

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

        # =============== Input Processing (per-frame, before chunking) ===============
        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            from umm.models.umm_v9.world_models.wm_utils import NerfStylePE

            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )
            x_proj_in_raw = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in_raw = self.input_dim

        # =============== Chunk Input Projection ===============
        # Per-frame noise PE encoder: level -> noise_pe_dim sinusoidal
        self.noise_pe_encoder = Timesteps(
            noise_pe_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        # Project [sinusoidal(noise_pe_dim) + raw_level(1)] -> noise_pe_dim
        self.noise_pe_proj = nn.Linear(noise_pe_dim + 1, noise_pe_dim)

        # Chunk projection: G * (x_proj_in_raw + noise_pe_dim) -> C_D
        chunk_input_dim = self.G * (x_proj_in_raw + noise_pe_dim)
        self.chunk_x_proj = nn.Linear(chunk_input_dim, self.C_D)

        # =============== Time/Position Embeddings ===============
        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.C_D)

        # Full C_D Timesteps for conditioning (level_emb, chunk-level)
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
        if self.Rt > 0:
            self.rt_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rt, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.rt_embeds:
                nn.init.normal_(emb, std=0.02)

        if self.Rp > 0:
            self.rp_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.rp_embeds:
                nn.init.normal_(emb, std=0.02)

        if self.Rt > 0 and self.Rp > 0:
            self.corner_embeds = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.Rt * self.Rp, self.C_D))
                for _ in range(self.n_layers)
            ])
            for emb in self.corner_embeds:
                nn.init.normal_(emb, std=0.02)

        # =============== Per-Layer adaLN for Registers ===============
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

        # =============== Register adaLN+FFN (per layer) ===============
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
            _dit_t_kwargs = dict(
                n_temporal_heads=n_head // 2,
                n_spatial_dims=9,
                **_dit_kwargs,
            )
            self.blocks = nn.ModuleList([
                DiTBlockSplitRoPE(**_dit_t_kwargs)
                if c == "t" else nn.Identity()
                for c in layer_type_string
            ])
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
        # Collect skips from t-layers + initial embedding -> output G * input_dim
        n_skips = self.n_t_layers + 1
        out_proj_in = self.C_D * n_skips
        _seq = []
        _c_in = out_proj_in
        # Intermediate layers taper from out_proj_in toward G * input_dim
        for _c_out in np.linspace(
            out_proj_in, self.G * self.input_dim, fc_out_n_layers
        )[:-1]:
            _c_out = int(_c_out)
            _seq.append(nn.Linear(_c_in, _c_out))
            _seq.append(HardLayerNorm(_c_out))
            _seq.append(nn.LeakyReLU())
            _c_in = _c_out
        _seq.append(nn.Linear(_c_in, self.G * self.input_dim))
        self.output_proj = nn.Sequential(*_seq)

        self._log_summary()

    def _log_summary(self) -> None:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info("LMM V11 DenseV2 Chunk Summary:")
        logger.info(f"  C_D (unified):       {self.C_D}")
        logger.info(f"  chunk_size (G):      {self.G}")
        logger.info(f"  noise_pe_dim:        {self.noise_pe_dim}")
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
        logger.info(
            f"  output_proj out:     {self.G} * {self.input_dim}"
            f" = {self.G * self.input_dim}"
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
    def _pad_to_chunks(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_level: torch.Tensor,
        x_time: torch.Tensor,
        x_r_level: Optional[torch.Tensor],
        S: int,
    ) -> dict:
        """Pad sequence length S to a multiple of G.

        Compile safety: Data-dependent padding size.

        Args:
            x: (B, S, P, input_dim) concatenated input.
            x_mask: (B, S, P) bool mask.
            x_level: (B, S, P) diffusion levels.
            x_time: (B, S) timestamps (already scaled).
            x_r_level: (B, S, P) optional MeanFlow r-level.
            S: Current sequence length.

        Returns:
            dict with padded tensors and metadata:
                'x': (B, S_pad, P, input_dim)
                'x_mask': (B, S_pad, P)
                'x_level': (B, S_pad, P)
                'x_time': (B, S_pad)
                'x_r_level': (B, S_pad, P) or None
                'S_pad': padded sequence length
                'n_pad': number of padding frames added
                'S_c': number of chunks = S_pad // G
        """
        G = self.G
        S_pad = math.ceil(S / G) * G
        n_pad = S_pad - S
        S_c = S_pad // G

        if n_pad > 0:
            B, _, P, D = x.shape
            # Pad x with zeros
            x = torch.cat([
                x,
                torch.zeros(B, n_pad, P, D, device=x.device, dtype=x.dtype),
            ], dim=1)  # (B, S_pad, P, D)

            # Pad x_mask with False (invalid)
            x_mask = torch.cat([
                x_mask,
                torch.zeros(B, n_pad, P, device=x_mask.device, dtype=torch.bool),
            ], dim=1)  # (B, S_pad, P)

            # Pad x_level with 1.0 (fully noised, doesn't matter since masked)
            x_level = torch.cat([
                x_level,
                torch.ones(B, n_pad, P, device=x_level.device, dtype=x_level.dtype),
            ], dim=1)  # (B, S_pad, P)

            # Pad x_time with continuation (last time + 1, 2, ...)
            last_t = x_time[:, -1:]  # (B, 1)
            pad_times = last_t + torch.arange(
                1, n_pad + 1, device=x_time.device, dtype=x_time.dtype
            )[None, :]  # (B, n_pad)
            x_time = torch.cat([x_time, pad_times], dim=1)  # (B, S_pad)

            # Pad x_r_level with 1.0 if present
            if x_r_level is not None:
                x_r_level = torch.cat([
                    x_r_level,
                    torch.ones(
                        B, n_pad, P,
                        device=x_r_level.device, dtype=x_r_level.dtype,
                    ),
                ], dim=1)  # (B, S_pad, P)

        return {
            "x": x,
            "x_mask": x_mask,
            "x_level": x_level,
            "x_time": x_time,
            "x_r_level": x_r_level,
            "S_pad": S_pad,
            "n_pad": n_pad,
            "S_c": S_c,
        }

    @torch.compiler.disable
    def _chunk_inputs(
        self,
        x: torch.Tensor,
        x_level: torch.Tensor,
        B: int,
        S_pad: int,
        P: int,
        S_c: int,
    ) -> torch.Tensor:
        """Build chunk-level input embeddings.

        Per-frame: apply x_pe, encode noise level as noise_pe_dim PE.
        Then reshape G frames into one chunk token and project to C_D.

        Compile safety: Reshape with computed dims.

        Args:
            x: (B, S_pad, P, input_dim) padded input.
            x_level: (B, S_pad, P) diffusion levels.
            B, S_pad, P, S_c: Dimensions.

        Returns:
            (B, S_c, P, C_D) chunk-level embeddings.
        """
        G = self.G

        # Step 1: Per-frame NeRF PE expansion (or identity)
        x_raw = self.x_pe(x)  # (B, S_pad, P, D_raw)
        D_raw = x_raw.shape[-1]

        # Step 2: Per-frame noise PE
        # x_level: (B, S_pad, P) -> flatten for Timesteps
        level_flat = x_level.reshape(-1)  # (B * S_pad * P,)
        noise_pe_sin = self.noise_pe_encoder(level_flat)  # (B*S_pad*P, noise_pe_dim)
        noise_pe_sin = noise_pe_sin.reshape(
            B, S_pad, P, self.noise_pe_dim
        )  # (B, S_pad, P, noise_pe_dim)

        # Concatenate raw level value with sinusoidal encoding
        level_raw = x_level.unsqueeze(-1)  # (B, S_pad, P, 1)
        noise_pe_input = torch.cat(
            [noise_pe_sin, level_raw], dim=-1
        )  # (B, S_pad, P, noise_pe_dim + 1)
        noise_pe = self.noise_pe_proj(noise_pe_input)  # (B, S_pad, P, noise_pe_dim)

        # Step 3: Reshape to chunks
        # x_raw: (B, S_pad, P, D_raw) -> (B, S_c, G, P, D_raw) -> (B, S_c, P, G*D_raw)
        x_chunked = (
            x_raw
            .reshape(B, S_c, G, P, D_raw)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, S_c, P, G * D_raw)
        )  # (B, S_c, P, G * D_raw)

        # noise_pe: (B, S_pad, P, npe) -> (B, S_c, G, P, npe) -> (B, S_c, P, G*npe)
        npe = self.noise_pe_dim
        noise_pe_chunked = (
            noise_pe
            .reshape(B, S_c, G, P, npe)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, S_c, P, G * npe)
        )  # (B, S_c, P, G * noise_pe_dim)

        # Step 4: Concatenate and project to C_D
        chunk_input = torch.cat(
            [x_chunked, noise_pe_chunked], dim=-1
        )  # (B, S_c, P, G * (D_raw + noise_pe_dim))
        x_emb = self.chunk_x_proj(chunk_input)  # (B, S_c, P, C_D)

        return x_emb

    @torch.compiler.disable
    def _unchunk_outputs(
        self,
        ret: torch.Tensor,
        B: int,
        S_c: int,
        P: int,
        S: int,
    ) -> torch.Tensor:
        """Unchunk output from (B*S_c, P, G*input_dim) to (B, S, P, input_dim).

        Compile safety: Reshape + slice with data-dependent dims.

        Args:
            ret: (B*S_c, P, G * input_dim) raw output from output_proj.
            B, S_c, P: Chunk-level dimensions.
            S: Original (unpadded) sequence length.

        Returns:
            (B, S, P, input_dim) output with padding stripped.
        """
        G = self.G
        # (B*S_c, P, G * input_dim) -> (B, S_c, P, G, input_dim)
        ret = ret.reshape(B, S_c, P, G, self.input_dim)
        # (B, S_c, P, G, input_dim) -> (B, S_c, G, P, input_dim) -> (B, S_pad, P, input_dim)
        ret = ret.permute(0, 1, 3, 2, 4).reshape(
            B, S_c * G, P, self.input_dim
        )  # (B, S_pad, P, input_dim)
        # Strip padding
        ret = ret[:, :S, :, :]  # (B, S, P, input_dim)
        return ret

    @torch.compiler.disable
    def _build_register_masks(
        self,
        x_mask: torch.Tensor,
        add_tokens_mask: torch.Tensor,
        S_c: int,
        P: int,
        T: int,
        N_CTX: int,
        A: int,
        R_blk: int,
        device: torch.device,
    ) -> dict:
        """Build attention masks for t-layer and p-layer with register isolation.

        Identical to DenseV2._build_register_masks but operates at chunk level.
        x_mask here is already chunk-level: (B, S_c, P).

        Compile safety: Uses data-dependent mask construction on tensor values.

        Args:
            x_mask: (B, S_c, P) bool mask (chunk-level, any-frame-valid).
            add_tokens_mask: (B, A) mask for add tokens.
            S_c: Chunk sequence length.
            P: Number of entities.
            T: Unused (kept for interface compat).
            N_CTX: Unused (kept for interface compat).
            A: Number of add_tokens.
            R_blk: Internal register count from DiT block.
            device: torch device.

        Returns:
            dict with 'attn_mask_t' and 'attn_mask_p'.
        """
        B = x_mask.shape[0]
        Rt, Rp = self.Rt, self.Rp
        P_ext = P + Rp
        S_ext = S_c + Rt

        # ============ t-layer mask ============
        L_t = R_blk + Rt + S_c

        entity_ever_valid = x_mask.any(dim=1)  # (B, P)

        traj_valid = x_mask.permute(0, 2, 1).reshape(B * P, S_c)  # (B*P, S_c)
        rp_col_valid = torch.ones(B * Rp, S_c, device=device, dtype=torch.bool)
        all_col_seq_valid = torch.cat(
            [traj_valid, rp_col_valid], dim=0
        )  # (B*P_ext, S_c)

        ir_valid = torch.ones(B * P_ext, R_blk, device=device, dtype=torch.bool)
        rt_valid = torch.ones(B * P_ext, Rt, device=device, dtype=torch.bool)
        full_valid_t = torch.cat(
            [ir_valid, rt_valid, all_col_seq_valid], dim=1
        )  # (B*P_ext, L_t)

        val_mask_t = (
            full_valid_t[:, None, :] & full_valid_t[:, :, None]
        )  # (B*P_ext, L_t, L_t)

        entity_isolated = ~entity_ever_valid  # (B, P)
        entity_isolated_flat = entity_isolated.reshape(B * P)
        rp_not_isolated = torch.zeros(B * Rp, device=device, dtype=torch.bool)
        all_isolated = torch.cat(
            [entity_isolated_flat, rp_not_isolated], dim=0
        )  # (B*P_ext,)

        if all_isolated.any():
            iso_idx = all_isolated.nonzero(as_tuple=True)[0]
            rt_start = R_blk
            rt_end = R_blk + Rt
            main_start = R_blk + Rt
            val_mask_t[iso_idx, rt_start:rt_end, main_start:] = False
            val_mask_t[iso_idx, main_start:, rt_start:rt_end] = False

        diag_t = torch.arange(L_t, device=device)
        val_mask_t[:, diag_t, diag_t] = True

        attn_mask_t = val_mask_t[:, None].expand(
            -1, self.n_head, -1, -1
        )  # (B*P_ext, n_head, L_t, L_t)

        # ============ p-layer mask ============
        L_p = R_blk + A + Rp + P

        pert_valid_main = entity_ever_valid[:, None, :].expand(
            -1, S_c, -1
        ).reshape(B * S_c, P)  # (B*S_c, P)

        pert_valid_rt = entity_ever_valid[:, None, :].expand(
            -1, Rt, -1
        ).reshape(B * Rt, P)  # (B*Rt, P)

        all_row_ent_valid = torch.cat(
            [pert_valid_main, pert_valid_rt], dim=0
        )  # (B*S_ext, P)

        rp_valid = torch.ones(
            B * S_ext, Rp, device=device, dtype=torch.bool
        )

        if A > 0:
            add_valid_main = (
                add_tokens_mask[:, None, :]
                .expand(-1, S_c, -1)
                .reshape(B * S_c, A)
            )
            add_valid_rt = (
                add_tokens_mask[:, None, :]
                .expand(-1, Rt, -1)
                .reshape(B * Rt, A)
            )
            add_valid = torch.cat([add_valid_main, add_valid_rt], dim=0)
        else:
            add_valid = torch.zeros(
                B * S_ext, 0, device=device, dtype=torch.bool
            )

        p_seq_valid = torch.cat(
            [add_valid, rp_valid, all_row_ent_valid], dim=1
        )  # (B*S_ext, A + Rp + P)

        attn_mask_p = build_attention_mask(
            p_seq_valid,
            mode="full",
            left_pad=R_blk,
        )  # (B*S_ext, L_p, L_p)

        row_ever_valid = x_mask.any(dim=2)  # (B, S_c)
        row_isolated_main = ~row_ever_valid
        row_isolated_flat = row_isolated_main.reshape(B * S_c)
        rt_not_isolated = torch.zeros(B * Rt, device=device, dtype=torch.bool)
        all_row_isolated = torch.cat(
            [row_isolated_flat, rt_not_isolated], dim=0
        )  # (B*S_ext,)

        if all_row_isolated.any():
            iso_r_idx = all_row_isolated.nonzero(as_tuple=True)[0]
            rp_start = R_blk + A
            rp_end = R_blk + A + Rp
            ent_start = R_blk + A + Rp
            attn_mask_p[iso_r_idx, rp_start:rp_end, ent_start:] = False
            attn_mask_p[iso_r_idx, ent_start:, rp_start:rp_end] = False

        diag_p = torch.arange(L_p, device=device)
        attn_mask_p[:, diag_p, diag_p] = True

        attn_mask_p = attn_mask_p[:, None].expand(
            -1, self.n_head, -1, -1
        )  # (B*S_ext, n_head, L_p, L_p)

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
        S_pad: int,
        P: int,
        S_c: int,
    ) -> dict:
        """Compute chunk-level spatial positions for split RoPE.

        Per-frame spatial positions are mean-pooled within each chunk.

        Compile safety: Uses compute_entity_mean_9d which contains SVD.

        Args:
            x: (B, S_pad, P, 9) padded concatenated input (ctx + noisy).
            x_mask: (B, S_pad, P) bool mask (frame-level, before chunking).
            N_CTX: Number of context timesteps (frame-level).
            B, S_pad, P, S_c: Dimensions.

        Returns:
            dict with:
                'x_9d_spatial': (B*P, S_c, 9) chunk-level spatial positions.
                'mean_ctx_9d': (B, P, 9) mean ctx 9D for dual spatial RoPE.
                'mean_x_9d': (B, P, 9) mean noisy 9D for dual spatial RoPE.
        """
        G = self.G
        # Per-frame spatial positions: (B*P, S_pad, 9)
        x_9d_frame = x.detach().permute(0, 2, 1, 3).reshape(B * P, S_pad, 9)
        # Mean-pool within chunks: (B*P, S_c, G, 9) -> (B*P, S_c, 9)
        x_9d_spatial = (
            x_9d_frame
            .reshape(B * P, S_c, G, 9)
            .mean(dim=2)
        )  # (B*P, S_c, 9)

        # p-layer: mean 9D for dual spatial RoPE (frame-level, not chunk-level)
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
        """Forward pass with chunk tokenization.

        Same I/O contract as DenseV2. Internally chunks the sequence and
        operates at chunk level, then unchunks the output.

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

        # ========== Step 1: Context Concatenation (SAME as DenseV2) ==========
        if self.add_time_to_feat:
            ctx_pos_emb_frame = torch.cat(
                [
                    x_time[..., None],
                    self.t_pe(x_time.reshape(-1)).reshape(
                        *x_time.shape[:2], -1
                    ),
                ],
                dim=-1,
            )
            ctx_pos_emb_frame = self.t_proj(
                ctx_pos_emb_frame
            )  # (B, T, C_D)
        else:
            ctx_pos_emb_frame = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device)
                 * self.n_ctx).long()
            )

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert N_CTX <= self.n_ctx
            x = torch.cat([ctx, x], dim=1)  # (B, S, P, input_dim)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)  # (B, S, P)
            x_level = torch.cat([ctx_level, x_level], dim=1)  # (B, S, P)
            if x_r_level is not None and ctx_r_level is not None:
                x_r_level = torch.cat(
                    [ctx_r_level, x_r_level], dim=1
                )  # (B, S, P)
            _ctx_pos_emb = self.ctx_t_emb(
                torch.arange(N_CTX, device=device)
            )[None].expand(B, -1, -1)  # (B, N_CTX, C_D)
            ctx_pos_emb_frame = torch.cat(
                [_ctx_pos_emb, ctx_pos_emb_frame], dim=1
            )  # (B, S, C_D)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # Extend x_time to (B, S) to match post-concat sequence length.
        # Context frames get negative indices (not used for RoPE in chunk mode,
        # but needed for consistent padding in _pad_to_chunks).
        if N_CTX > 0:
            ctx_times = -torch.arange(
                N_CTX, 0, -1, device=device, dtype=x_time.dtype
            )[None].expand(B, -1)  # (B, N_CTX)
            x_time = torch.cat([ctx_times, x_time], dim=1)  # (B, S)

        # ========== Step 2: Pad S to multiple of G ==========
        pad_result = self._pad_to_chunks(
            x, x_mask, x_level, x_time, x_r_level, S
        )
        x = pad_result["x"]                    # (B, S_pad, P, input_dim)
        x_mask_frame = pad_result["x_mask"]     # (B, S_pad, P) -- frame-level
        x_level = pad_result["x_level"]         # (B, S_pad, P)
        x_time = pad_result["x_time"]           # (B, S_pad)
        x_r_level = pad_result["x_r_level"]     # (B, S_pad, P) or None
        S_pad = pad_result["S_pad"]
        S_c = pad_result["S_c"]

        # Also pad ctx_pos_emb_frame to S_pad
        if pad_result["n_pad"] > 0:
            n_pad = pad_result["n_pad"]
            ctx_pos_emb_frame = torch.cat([
                ctx_pos_emb_frame,
                torch.zeros(
                    B, n_pad, self.C_D,
                    device=device, dtype=ctx_pos_emb_frame.dtype,
                ),
            ], dim=1)  # (B, S_pad, C_D)

        # ========== Step 3: Chunk-level x_mask ==========
        G = self.G
        # x_mask_chunk: (B, S_c, P) -- True if ANY frame in chunk is valid
        x_mask_chunk = (
            x_mask_frame
            .reshape(B, S_c, G, P)
            .any(dim=2)
        )  # (B, S_c, P)

        # ========== Step 4: Chunk Input Embedding ==========
        x_emb = self._chunk_inputs(
            x, x_level, B, S_pad, P, S_c
        )  # (B, S_c, P, C_D)

        # ========== Step 5: Build Chunk-Level Conditioning ==========
        # level_emb: per-frame C_D PE, then mean-pool within chunks
        level_emb_frame = (
            self.int_pe(x_level.reshape(B * S_pad * P))
            .reshape(B, S_pad, P, self.C_D)
            .to(x)
        )  # (B, S_pad, P, C_D)
        level_emb = (
            level_emb_frame
            .reshape(B, S_c, G, P, self.C_D)
            .mean(dim=2)
        )  # (B, S_c, P, C_D)

        if x_r_level is not None:
            r_emb_frame = (
                self.int_pe(x_r_level.reshape(B * S_pad * P))
                .reshape(B, S_pad, P, self.C_D)
                .to(x)
            )  # (B, S_pad, P, C_D)
            r_emb = (
                r_emb_frame
                .reshape(B, S_c, G, P, self.C_D)
                .mean(dim=2)
            )  # (B, S_c, P, C_D)
            level_emb = level_emb + r_emb

        # feat: (B, P, feat_dim) -> expand to chunk level
        feat_expanded = feat[:, None].expand(
            -1, S_c, -1, -1
        )  # (B, S_c, P, feat_dim)

        # ctx_pos_emb: per-frame -> mean-pool within chunks
        ctx_pos_emb_chunk = (
            ctx_pos_emb_frame
            .reshape(B, S_c, G, self.C_D)
            .mean(dim=2)
        )  # (B, S_c, C_D)
        ctx_pos_emb_expanded = ctx_pos_emb_chunk[:, :, None].expand(
            -1, -1, P, -1
        )  # (B, S_c, P, C_D)

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
                -1, S_c, -1, -1
            )  # (B, S_c, P, C_D)
            cond_parts.append(self.hard_ln_cd(traj_pe_expanded))

        cond_bsp = torch.cat(cond_parts, dim=-1)  # (B, S_c, P, cond_dim)

        # ========== Time binding (no-RoPE path) ==========
        if not self.time_rope_flag and self.add_time_to_feat:
            time_emb_expanded = ctx_pos_emb_chunk[:, :, None, :].expand(
                -1, -1, P, -1
            )  # (B, S_c, P, C_D)
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
            # Chunk-level RoPE: sequential chunk indices for all S_c chunks
            chunk_times = torch.arange(
                S_c, device=device, dtype=torch.float32
            )  # (S_c,)
            t_rope = chunk_times[None, :].expand(
                B * P, -1
            )  # (B*P, S_c)
        else:
            t_rope = None

        # ========== Spatial Positions for Split RoPE ==========
        if self.spatial_rope_flag:
            sp = self._compute_spatial_positions(
                x, x_mask_frame, N_CTX, B, S_pad, P, S_c
            )
            x_9d_spatial = sp["x_9d_spatial"]   # (B*P, S_c, 9)
            mean_ctx_9d = sp["mean_ctx_9d"]     # (B, P, 9)
            mean_x_9d = sp["mean_x_9d"]         # (B, P, 9)
        else:
            x_9d_spatial = None
            mean_ctx_9d = None
            mean_x_9d = None

        # ========== Build Attention Masks ==========
        R_blk = self.blocks[0].R  # internal registers from DiT block

        masks = self._build_register_masks(
            x_mask_chunk, add_tokens_mask,
            S_c, P, T, N_CTX, A, R_blk, device
        )
        attn_mask_t = masks["attn_mask_t"]  # (B*(P+Rp), n_head, L_t, L_t)
        attn_mask_p = masks["attn_mask_p"]  # (B*(S_c+Rt), n_head, L_p, L_p)

        # ========== State Initialization ==========
        x_state = x_emb.clone()  # (B, S_c, P, C_D)
        skip_list = [x_emb.reshape(B * S_c, P, self.C_D).clone()]

        # Register states
        rt_state = None    # (B, Rt, P, C_D)
        rp_state = None    # (B, S_c, Rp, C_D)
        corner_state = None  # (B, Rt, Rp, C_D)

        # Layer-0 mixing
        x_state_0 = x_emb.clone()  # (B, S_c, P, C_D)

        t_layer_idx = 0
        p_layer_idx = 0
        a_layer_idx = 0
        b_layer_idx = 0

        # ========== Main Layer Loop ==========
        for layer_idx, lt in enumerate(self.layer_type_string):
            block = self.blocks[layer_idx]
            cond_binder = self.cond_binders[layer_idx]

            # --- Build register tokens for this layer ---
            if Rt > 0:
                rt_emb_layer = self.rt_embeds[layer_idx].expand(
                    B, -1, -1
                )  # (B, Rt, C_D)
                if rt_state is None:
                    rt_cur = rt_emb_layer[:, :, None, :].expand(
                        -1, -1, P, -1
                    )  # (B, Rt, P, C_D)
                else:
                    rt_cur = rt_state + rt_emb_layer[:, :, None, :].expand(
                        -1, -1, P, -1
                    )

            if Rp > 0:
                rp_emb_layer = self.rp_embeds[layer_idx].expand(
                    B, -1, -1
                )  # (B, Rp, C_D)
                if rp_state is None:
                    rp_cur = rp_emb_layer[:, None, :, :].expand(
                        -1, S_c, -1, -1
                    )  # (B, S_c, Rp, C_D)
                else:
                    rp_cur = rp_state + rp_emb_layer[:, None, :, :].expand(
                        -1, S_c, -1, -1
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
                # ===== t-layer: self-attention across S_c+Rt per entity column =====

                # Layer-0 mixing (gradient highway)
                if self.n_t_layers > 1 and t_layer_idx > 0:
                    x_mix_input = torch.cat(
                        [x_state_0, x_state], dim=-1
                    )
                    x_state = self.layer0_mix[t_layer_idx - 1](x_mix_input)

                # Main entity columns: (B*P, S_c, C_D)
                x_traj = x_state.permute(0, 2, 1, 3).reshape(
                    B * P, S_c, self.C_D
                )

                if Rt > 0:
                    rt_for_main = rt_cur.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)
                    main_col_seq = torch.cat(
                        [rt_for_main, x_traj], dim=1
                    )  # (B*P, Rt+S_c, C_D)
                else:
                    main_col_seq = x_traj  # (B*P, S_c, C_D)

                if Rp > 0:
                    rp_col_seq_main = rp_cur.permute(
                        0, 2, 1, 3
                    ).reshape(B * Rp, S_c, self.C_D)  # (B*Rp, S_c, C_D)

                    if Rt > 0:
                        corner_for_rp = corner_cur.permute(
                            0, 2, 1, 3
                        ).reshape(B * Rp, Rt, self.C_D)
                        rp_col_seq = torch.cat(
                            [corner_for_rp, rp_col_seq_main], dim=1
                        )  # (B*Rp, Rt+S_c, C_D)
                    else:
                        rp_col_seq = rp_col_seq_main

                    traj_seq = torch.cat(
                        [main_col_seq, rp_col_seq], dim=0
                    )  # (B*(P+Rp), Rt+S_c, C_D)
                else:
                    traj_seq = main_col_seq

                # --- Conditioning ---
                cond_t = cond_binder(cond_bsp)  # (B, S_c, P, C_D)
                cond_t_per_traj = cond_t.permute(0, 2, 1, 3).reshape(
                    B * P, S_c, self.C_D
                )  # (B*P, S_c, C_D)

                if Rt > 0:
                    cond_rt_mean = cond_t_per_traj.mean(
                        dim=1, keepdim=True
                    ).expand(-1, Rt, -1)  # (B*P, Rt, C_D)
                    main_cond = torch.cat(
                        [cond_rt_mean, cond_t_per_traj], dim=1
                    )  # (B*P, Rt+S_c, C_D)
                else:
                    main_cond = cond_t_per_traj

                if Rp > 0:
                    cond_rp_global = cond_t_per_traj.reshape(
                        B, P, S_c, self.C_D
                    ).mean(dim=1)  # (B, S_c, C_D)
                    cond_rp_tiled = cond_rp_global[:, None].expand(
                        -1, Rp, -1, -1
                    ).reshape(B * Rp, S_c, self.C_D)  # (B*Rp, S_c, C_D)

                    if Rt > 0:
                        cond_corner = cond_rp_tiled.mean(
                            dim=1, keepdim=True
                        ).expand(-1, Rt, -1)  # (B*Rp, Rt, C_D)
                        rp_cond = torch.cat(
                            [cond_corner, cond_rp_tiled], dim=1
                        )  # (B*Rp, Rt+S_c, C_D)
                    else:
                        rp_cond = cond_rp_tiled

                    cond_traj = torch.cat(
                        [main_cond, rp_cond], dim=0
                    )  # (B*(P+Rp), Rt+S_c, C_D)
                else:
                    cond_traj = main_cond

                # --- Set spatial RoPE params ---
                if self.spatial_rope_flag:
                    if Rp > 0:
                        rp_dummy_spatial = torch.zeros(
                            B * Rp, S_c, 9,
                            device=device, dtype=x_9d_spatial.dtype,
                        )
                        x_9d_spatial_ext = torch.cat(
                            [x_9d_spatial, rp_dummy_spatial], dim=0
                        )  # (B*(P+Rp), S_c, 9)
                    else:
                        x_9d_spatial_ext = x_9d_spatial
                    block.set_spatial_rope_params(
                        spatial_positions=x_9d_spatial_ext,
                        num_external_regs_spatial=Rt,
                    )

                # --- RoPE time: extend for Rp columns ---
                if t_rope is not None and Rp > 0:
                    t_rope_rp = torch.zeros(
                        B * Rp, S_c, device=device, dtype=t_rope.dtype
                    )
                    t_rope_ext = torch.cat(
                        [t_rope, t_rope_rp], dim=0
                    )
                else:
                    t_rope_ext = t_rope

                # --- Run t-block ---
                # num_external_regs: Rt only (all chunks get RoPE including ctx chunks)
                traj_out = block(
                    x=traj_seq,
                    cond=cond_traj,
                    t=t_rope_ext,
                    num_external_regs=Rt,
                    attn_mask=attn_mask_t,
                )  # (B*(P+Rp), Rt+S_c, C_D)

                # --- Extract outputs ---
                main_out = traj_out[:B * P]  # (B*P, Rt+S_c, C_D)

                if Rt > 0:
                    rt_out_main = main_out[:, :Rt]  # (B*P, Rt, C_D)
                    x_out = main_out[:, Rt:]         # (B*P, S_c, C_D)
                else:
                    x_out = main_out

                x_state = x_out.reshape(B, P, S_c, self.C_D).permute(
                    0, 2, 1, 3
                )  # (B, S_c, P, C_D)

                if Rt > 0:
                    rt_state = rt_out_main.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, Rt, P, C_D)

                if Rp > 0:
                    rp_out = traj_out[B * P:]  # (B*Rp, Rt+S_c, C_D)
                    if Rt > 0:
                        corner_out_from_rp = rp_out[:, :Rt]  # (B*Rp, Rt, C_D)
                        rp_main_out = rp_out[:, Rt:]          # (B*Rp, S_c, C_D)
                    else:
                        rp_main_out = rp_out
                        corner_out_from_rp = None

                    rp_state = rp_main_out.reshape(
                        B, Rp, S_c, self.C_D
                    ).permute(0, 2, 1, 3)  # (B, S_c, Rp, C_D)

                    if Rt > 0 and corner_out_from_rp is not None:
                        corner_state = corner_out_from_rp.reshape(
                            B, Rp, Rt, self.C_D
                        ).permute(0, 2, 1, 3)  # (B, Rt, Rp, C_D)

                # --- Apply register FFNs ---
                if Rt > 0:
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
                    rp_flat = rp_state.reshape(B * S_c, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S_c, -1, -1
                    )  # (B*S_c, Rp, C_D)
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(
                        B, S_c, Rp, self.C_D
                    )  # (B, S_c, Rp, C_D)

                if Rt > 0 and Rp > 0:
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
                    x_state.reshape(B * S_c, P, self.C_D).clone()
                )
                t_layer_idx += 1

            elif lt == "p":
                # ===== p-layer: cross-attention across P+Rp per chunk row =====

                x_flat = x_state.reshape(B * S_c, P, self.C_D)

                if Rp > 0:
                    rp_for_main = rp_cur.reshape(
                        B * S_c, Rp, self.C_D
                    )  # (B*S_c, Rp, C_D)

                if Rt > 0:
                    rt_row_ent = rt_cur.reshape(
                        B * Rt, P, self.C_D
                    )  # (B*Rt, P, C_D)

                    if Rp > 0:
                        corner_for_rt = corner_cur.reshape(
                            B * Rt, Rp, self.C_D
                        )

                S_ext = S_c + Rt
                add_tiled_main = (
                    add_tokens.unsqueeze(1)
                    .expand(-1, S_c, -1, -1)
                    .reshape(B * S_c, A, self.C_D)
                )
                if Rt > 0:
                    add_tiled_rt = (
                        add_tokens.unsqueeze(1)
                        .expand(-1, Rt, -1, -1)
                        .reshape(B * Rt, A, self.C_D)
                    )

                main_parts = [add_tiled_main]
                if Rp > 0:
                    main_parts.append(rp_for_main)
                main_parts.append(x_flat)
                main_row_seq = torch.cat(
                    main_parts, dim=1
                )  # (B*S_c, A+Rp+P, C_D)

                if Rt > 0:
                    rt_parts = [add_tiled_rt]
                    if Rp > 0:
                        rt_parts.append(corner_for_rt)
                    rt_parts.append(rt_row_ent)
                    rt_row_seq = torch.cat(
                        rt_parts, dim=1
                    )  # (B*Rt, A+Rp+P, C_D)

                    p_seq = torch.cat(
                        [main_row_seq, rt_row_seq], dim=0
                    )
                else:
                    p_seq = main_row_seq

                # --- Conditioning ---
                cond_p = cond_binder(cond_bsp)  # (B, S_c, P, C_D)
                cond_flat = cond_p.reshape(B * S_c, P, self.C_D)

                cond_prefix_main = torch.zeros(
                    B * S_c, A + Rp, self.C_D,
                    device=device, dtype=cond_flat.dtype,
                )
                main_cond_full = torch.cat(
                    [cond_prefix_main, cond_flat], dim=1
                )  # (B*S_c, A+Rp+P, C_D)

                if Rt > 0:
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
                    )  # (B*(S_c+Rt), A+Rp+P, C_D)
                else:
                    cond_full = main_cond_full

                # --- Set dual spatial RoPE for p-layer ---
                if self.spatial_rope_flag and self._p_blocks_split_rope:
                    mean_ctx_tiled = mean_ctx_9d[:, None].expand(
                        -1, S_ext, -1, -1
                    ).reshape(B * S_ext, P, 9)  # (B*(S_c+Rt), P, 9)
                    mean_x_tiled = mean_x_9d[:, None].expand(
                        -1, S_ext, -1, -1
                    ).reshape(B * S_ext, P, 9)  # (B*(S_c+Rt), P, 9)
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
                )  # (B*(S_c+Rt), A+Rp+P, C_D)

                # --- Extract outputs ---
                main_row_out = p_out[:B * S_c]  # (B*S_c, A+Rp+P, C_D)
                x_flat_out = main_row_out[:, A + Rp:]  # (B*S_c, P, C_D)
                x_state = x_flat_out.reshape(B, S_c, P, self.C_D)

                if Rp > 0:
                    rp_out_main = main_row_out[
                        :, A:A + Rp
                    ]  # (B*S_c, Rp, C_D)
                    rp_state = rp_out_main.reshape(
                        B, S_c, Rp, self.C_D
                    )  # (B, S_c, Rp, C_D)

                if Rt > 0:
                    rt_row_out = p_out[B * S_c:]  # (B*Rt, A+Rp+P, C_D)
                    rt_ent_out = rt_row_out[:, A + Rp:]  # (B*Rt, P, C_D)
                    rt_state = rt_ent_out.reshape(
                        B, Rt, P, self.C_D
                    )  # (B, Rt, P, C_D)

                    if Rp > 0:
                        corner_out = rt_row_out[
                            :, A:A + Rp
                        ]  # (B*Rt, Rp, C_D)
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
                    rp_flat = rp_state.reshape(B * S_c, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S_c, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S_c, Rp, self.C_D)

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
                if Rt > 0:
                    rt_main_cols = rt_cur.permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)

                    if Rp > 0:
                        corner_cols = corner_cur.permute(
                            0, 2, 1, 3
                        ).reshape(B * Rp, Rt, self.C_D)
                        a_seq = torch.cat(
                            [rt_main_cols, corner_cols], dim=0
                        )  # (B*(P+Rp), Rt, C_D)
                    else:
                        a_seq = rt_main_cols

                    cond_a = cond_binder(cond_bsp)  # (B, S_c, P, C_D)
                    cond_a_mean = cond_a.mean(dim=1)  # (B, P, C_D)
                    cond_a_per_col = cond_a_mean[:, :, None, :].expand(
                        -1, -1, Rt, -1
                    ).permute(0, 2, 1, 3).reshape(
                        B * P, Rt, self.C_D
                    )  # (B*P, Rt, C_D)

                    if Rp > 0:
                        cond_rp_global = cond_a.mean(dim=(1, 2))  # (B, C_D)
                        cond_rp_cols = cond_rp_global[:, None, None, :].expand(
                            -1, Rp, Rt, -1
                        ).reshape(B * Rp, Rt, self.C_D)
                        cond_a_full = torch.cat(
                            [cond_a_per_col, cond_rp_cols], dim=0
                        )
                    else:
                        cond_a_full = cond_a_per_col

                    a_out = self.a_blocks[a_layer_idx](
                        x=a_seq,
                        cond=cond_a_full,
                        t=None,
                        num_external_regs=0,
                        attn_mask=None,
                    )

                    rt_main_out = a_out[:B * P]
                    rt_state = rt_main_out.reshape(
                        B, P, Rt, self.C_D
                    ).permute(0, 2, 1, 3)

                    if Rp > 0:
                        corner_out_a = a_out[B * P:]
                        corner_state = corner_out_a.reshape(
                            B, Rp, Rt, self.C_D
                        ).permute(0, 2, 1, 3)

                if Rp > 0:
                    rp_state = rp_cur

                # Apply register FFNs
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
                    rp_flat = rp_state.reshape(B * S_c, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S_c, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S_c, Rp, self.C_D)

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
                # ===== b-layer: P-attention on registers ONLY [S_c+Rt, Rp] =====
                if Rp > 0:
                    rp_main_rows = rp_cur.reshape(
                        B * S_c, Rp, self.C_D
                    )  # (B*S_c, Rp, C_D)

                    if Rt > 0:
                        corner_rows = corner_cur.reshape(
                            B * Rt, Rp, self.C_D
                        )
                        b_seq = torch.cat(
                            [rp_main_rows, corner_rows], dim=0
                        )  # (B*(S_c+Rt), Rp, C_D)
                    else:
                        b_seq = rp_main_rows

                    cond_b = cond_binder(cond_bsp)  # (B, S_c, P, C_D)
                    cond_b_mean = cond_b.mean(dim=2)  # (B, S_c, C_D)
                    cond_b_per_row = cond_b_mean[:, :, None, :].expand(
                        -1, -1, Rp, -1
                    ).reshape(B * S_c, Rp, self.C_D)

                    if Rt > 0:
                        cond_rt_global = cond_b.mean(dim=(1, 2))  # (B, C_D)
                        cond_rt_rows = cond_rt_global[:, None, None, :].expand(
                            -1, Rt, Rp, -1
                        ).reshape(B * Rt, Rp, self.C_D)
                        cond_b_full = torch.cat(
                            [cond_b_per_row, cond_rt_rows], dim=0
                        )
                    else:
                        cond_b_full = cond_b_per_row

                    b_out = self.b_blocks[b_layer_idx](
                        x=b_seq,
                        cond=cond_b_full,
                        t=None,
                        num_external_regs=0,
                        attn_mask=None,
                    )

                    rp_main_out = b_out[:B * S_c]
                    rp_state = rp_main_out.reshape(
                        B, S_c, Rp, self.C_D
                    )

                    if Rt > 0:
                        corner_out_b = b_out[B * S_c:]
                        corner_state = corner_out_b.reshape(
                            B, Rt, Rp, self.C_D
                        )

                if Rt > 0:
                    rt_state = rt_cur

                # Apply register FFNs
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
                    rp_flat = rp_state.reshape(B * S_c, Rp, self.C_D)
                    ada_rp_cond = self.ada_rp_embeds[layer_idx].expand(
                        B * S_c, -1, -1
                    )
                    rp_flat = self.reg_rp_blocks[layer_idx](
                        rp_flat, ada_rp_cond
                    )
                    rp_state = rp_flat.reshape(B, S_c, Rp, self.C_D)

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

        # ========== Output Projection & Unchunking ==========
        x_out = torch.cat(skip_list, dim=-1)  # (B*S_c, P, (n_t+1)*C_D)
        ret = self.output_proj(
            x_out.reshape(-1, x_out.shape[-1])
        )  # (B*S_c*P, G * input_dim)
        ret = ret.reshape(B * S_c, P, self.G * self.input_dim)

        # Unchunk: (B*S_c, P, G*input_dim) -> (B, S, P, input_dim)
        ret = self._unchunk_outputs(ret, B, S_c, P, S)

        return ret  # (B, S, P, input_dim)
