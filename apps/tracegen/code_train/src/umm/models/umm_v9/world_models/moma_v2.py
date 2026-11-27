"""MotionMachine V5 -- Perceiver IO latent pooling architecture.

Contains:
    - **IdentifyLayerV2**: Encodes per-joint identity (semantic ID, group ID,
      point cloud geometry, and language) into a conditioning feature vector.
    - **MotionMachineV5**: Transformer-based denoiser with Perceiver IO pooling.
      Compresses the part dimension P→Q via 'q' layers, processes in compact
      latent space via 't'/'p'/'a' layers, and decodes Q→P via 'd' layers.
      Supports async (per-timestep) and synced (fused) attention in q layers.

Layer pattern example: "qtpadqtpadqd"
    q = Perceiver encode (P→Q), d = Perceiver decode (Q→P),
    t = temporal attn, p = part attn, a = global attn.

Example:
    >>> io = IdentifyLayerV2(latent_dim=512, feat_dim=2048)
    >>> net = MotionMachineV5(latent_dim=512, feat_dim=2048,
    ...     layer_type_string="qtpadqtpadqd", n_latent_queries=16)
    >>> feat = io(sem_id, group_id, pcl, pcl_mask)
    >>> out = net(x, x_mask, x_time, x_level, feat)
"""

import torch
from torch import nn
import logging
from diffusers.models.embeddings import Timesteps
from torch.nn import functional as F
import numpy as np

from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    SimpleFC,
    build_attention_mask,
    NerfStylePE,
    disable_model_grad,
)
from umm.models.umm_v9.world_models.pointnet import ResnetPointnet, ResnetPointnetV2
from umm.models.umm_v9.world_models.dit_atten import DiTBlockTimeOnly
from umm.models.umm_v9.world_models.language import T5LanguageEncoder
from umm.models.umm_v9.world_models.perceiver_layers import (
    PerceiverEncodeLayer,
    PerceiverDecodeLayer,
)


# * What is moving
class IdentifyLayerV2(nn.Module):
    """IdentifyLayerV2 encodes semantic ID, group ID, point cloud, and language."""

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 2048,
        num_types: int = 512,
        num_groups: int = 512,
        group_emb_same_init_flag=True,
        point_version: str = "v1",
        point_encoder_latent: int = -1,
        point_input_dim: int = 6,
        point_xyz_scale_factor: float = 1.0,
        langauge_fixed_length: int = 16,
        skip_group_emb: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.skip_group_emb = skip_group_emb
        logging.warning(f"latent_dim: {self.latent_dim}, feat_dim: {self.feat_dim}")
        if self.skip_group_emb:
            logging.warning("skip_group_emb=True: group_id NOT encoded as feature")

        self.n_types = num_types
        self.n_groups = num_groups

        self.type_emb = nn.Embedding(self.n_types, self.latent_dim)
        self.group_emb = nn.Embedding(self.n_groups, self.latent_dim)
        if group_emb_same_init_flag:
            with torch.no_grad():
                logging.warning(f"Group EMB all same init!")
                self.group_emb.weight.copy_(
                    self.group_emb.weight[0].expand_as(self.group_emb.weight)
                )

        self.point_flag = point_encoder_latent > 0
        if self.point_flag:
            if point_version == "v1":
                self.pntnet = ResnetPointnet(
                    c_dim=self.latent_dim,
                    dim=point_input_dim,
                    hidden_dim=point_encoder_latent,
                )
            elif point_version == "v2":
                self.pntnet = ResnetPointnetV2(
                    c_dim=self.latent_dim,
                    dim=point_input_dim,
                    hidden_dim=point_encoder_latent,
                )
        self.point_xyz_scale_factor = point_xyz_scale_factor
        logging.warning(f"Point XYZ scale factor: {self.point_xyz_scale_factor}")
        self.empty_point_emb = nn.Embedding(1, self.latent_dim)

        self.ln_type_emb = HardLayerNorm(self.latent_dim)
        self.ln_group_emb = HardLayerNorm(self.latent_dim)
        self.ln_geo_emb = HardLayerNorm(self.latent_dim)

        # * language
        self.lang_fixed_len = langauge_fixed_length
        self.language_encoder = disable_model_grad(
            T5LanguageEncoder(fixed_length=self.lang_fixed_len)
        )
        self.lang_fc = nn.Linear(768, self.latent_dim)

        bind_in = self.latent_dim * 3
        self.bind_fc = nn.Linear(bind_in, self.feat_dim)
        return

    def forward(self, sem_id, group_id, pcl=None, pcl_valid_mask=None):
        """Forward: [B, N] IDs + optional [B, N, M, 3] PCL → [B, N, feat_dim]."""
        B, N = sem_id.shape
        type_emb = self.type_emb(sem_id)
        if self.skip_group_emb:
            group_emb = torch.zeros(
                B, N, self.latent_dim, device=sem_id.device, dtype=type_emb.dtype
            )
        else:
            group_emb = self.group_emb(group_id)

        geo_emb = self.empty_point_emb(torch.tensor([0], device=sem_id.device))[None]
        geo_emb = geo_emb.repeat(B, N, 1).contiguous()
        if self.point_flag and pcl is not None:
            if pcl_valid_mask is None:
                pcl_valid_mask = torch.ones(B, N, device=sem_id.device).bool()
            valid_pcl = pcl[pcl_valid_mask].clone()
            valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
            valid_pcl_feat = self.pntnet(valid_pcl)
            geo_emb[pcl_valid_mask] = (
                geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
            )

        type_emb = self.ln_type_emb(type_emb)
        group_emb = self.ln_group_emb(group_emb)
        geo_emb = self.ln_geo_emb(geo_emb)

        feat = torch.cat([type_emb, group_emb, geo_emb], dim=-1)
        feat = self.bind_fc(feat)
        return feat  # B,N,D

    @torch.inference_mode()
    def encode_language(self, text_in):
        """Encode text → [B, lang_fixed_len, latent_dim] (no grad)."""
        language_tokens = self.language_encoder.encode(text_in)
        language_tokens = self.lang_fc(language_tokens)
        return language_tokens.detach()

    def get_optim_groups(self, weight_decay=0.0, base_lr=None):
        """Optimizer groups with weight-decay separation (excludes frozen language encoder)."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        param_dict = {
            pn: p
            for pn, p in param_dict.items()
            if not pn.startswith("language_encoder")
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


# * How it moves
class MotionMachineV5(nn.Module):
    """Perceiver IO pooling MotionMachine.

    Maintains dual state:
        - x_p: (B, S, P, D) dense part representation
        - x_q: (B, S, Q, D) compressed latent representation

    Layer types:
        q: Perceiver encode (Q queries attend to P KV) — compresses P→Q
        d: Perceiver decode (P queries attend to Q KV) — expands Q→P, skip collected
        t: Temporal self-attention on Q dim
        p: Part/spatial self-attention on Q dim
        a: Global self-attention on S*Q tokens

    Output: concatenate all d-layer skip outputs + initial x_p → wide MLP → prediction.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 2048,
        n_head: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        layer_type_string: str = "qtpadqtpadqd",
        cond_binder_dim: int = 4096,
        #
        fc_out_n_layers: int = 4,
        fc_factor: int = 2,
        fc_n_layers: int = 2,
        num_register: int = 2,
        num_blk_internal_registers: int = 4,
        n_ctx: int = 128,
        #
        x_pe_dim: int = 0,
        time_multiplier: float = 1.0,
        #
        use_reg_binder: bool = True,
        #
        ada_scale_bias_bound=-1.0,
        gate_bound=-1.0,
        ada_n_layers: int = 1,
        #
        add_first_n_add_token_to_feat: int = -1,
        # time cond control
        time_rope_flag=True,
        add_time_to_feat: bool = False,
        # random ID pe
        num_random_trace_id_pe: int = -1,
        # image cross-attention control
        image_cross_attn_layers: str = "tpa",
        # V5 Perceiver IO params
        n_latent_queries: int = 16,
        q_pool_mode: str = "async",
        # legacy compat (ignored)
        skip_layer_idx: list = None,
    ):
        super().__init__()

        self.layer_type_string = layer_type_string
        self.image_cross_attn_layers = image_cross_attn_layers
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.cond_binder_dim = cond_binder_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.n_blocks = len(self.layer_type_string)
        self.n_head = n_head
        self.num_register = num_register
        self.hard_ln_lat = HardLayerNorm(self.latent_dim)
        self.hard_ln_feat = HardLayerNorm(self.feat_dim)

        # V5 Perceiver IO
        self.n_latent_queries = n_latent_queries
        self.q_pool_mode = q_pool_mode
        self.n_d_layers = sum(1 for c in layer_type_string if c == "d")
        assert self.n_d_layers > 0, "Need at least one d layer for output"
        assert layer_type_string[0] == "q", "First layer must be q (encode P→Q)"
        logging.warning(
            f"V5 Perceiver IO: Q={n_latent_queries}, mode={q_pool_mode}, "
            f"pattern={layer_type_string}, n_d={self.n_d_layers}"
        )

        # Learned latent queries (Perceiver IO style)
        self.latent_queries = nn.Parameter(torch.zeros(1, n_latent_queries, latent_dim))
        nn.init.normal_(self.latent_queries, std=0.02)

        self.time_rope_flag = time_rope_flag
        if self.time_rope_flag:
            logging.info("use time rope")
        self.add_time_to_feat = add_time_to_feat
        if self.add_time_to_feat:
            logging.info("add time to feat by sharing the VT buffer")
            self.t_pe = Timesteps(
                self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
            )
            self.t_proj = nn.Linear(self.latent_dim + 1, self.latent_dim)

        # PE and embeddings
        self.int_pe = Timesteps(
            self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )
        self.time_multiplier = time_multiplier

        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )
            x_proj_in = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.input_dim
        self.x_proj = nn.Linear(x_proj_in, self.latent_dim)

        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.latent_dim)

        # Conditioning binder
        cond_binder_in = self.feat_dim + self.latent_dim * 2
        self.add_first_n_add_token_to_feat = add_first_n_add_token_to_feat
        if self.add_first_n_add_token_to_feat > 0:
            self.add_to_feat_fc1 = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            self.add_to_feat_fc2 = nn.Linear(
                64 * self.add_first_n_add_token_to_feat, self.latent_dim
            )
            cond_binder_in += self.latent_dim
        self.num_random_trace_id_pe = num_random_trace_id_pe
        if self.num_random_trace_id_pe > 0:
            logging.warning(f"use random trace id pe with {self.num_random_trace_id_pe}")
            self.random_trace_id_pe = nn.Embedding(
                self.num_random_trace_id_pe, self.latent_dim
            )
            with torch.no_grad():
                self.random_trace_id_pe.weight.copy_(
                    self.random_trace_id_pe.weight[0].expand_as(
                        self.random_trace_id_pe.weight
                    )
                )
            cond_binder_in += self.latent_dim

        # One cond_binder per layer
        self.cond_binder = nn.ModuleList(
            [nn.Linear(cond_binder_in, self.cond_binder_dim) for _ in range(self.n_blocks)]
        )

        # Build layer blocks based on layer_type_string
        Q = n_latent_queries
        self.num_blk_internal_registers = num_blk_internal_registers
        self.blks = nn.ModuleList()
        for i, lt in enumerate(self.layer_type_string):
            if lt == "q":
                self.blks.append(
                    PerceiverEncodeLayer(
                        C=self.latent_dim,
                        num_heads=n_head,
                        C_feat=self.cond_binder_dim,
                        q_pool_mode=q_pool_mode,
                        mlp_hidden_dims=[self.latent_dim * fc_factor] * fc_n_layers,
                        ada_n_layers=ada_n_layers,
                        ada_scale_bias_bound=ada_scale_bias_bound,
                        gate_bound=gate_bound,
                    )
                )
            elif lt == "d":
                self.blks.append(
                    PerceiverDecodeLayer(
                        C=self.latent_dim,
                        num_heads=n_head,
                        C_feat=self.cond_binder_dim,
                        mlp_hidden_dims=[self.latent_dim * fc_factor] * fc_n_layers,
                        ada_n_layers=ada_n_layers,
                        ada_scale_bias_bound=ada_scale_bias_bound,
                        gate_bound=gate_bound,
                    )
                )
            elif lt in ("t", "p", "a"):
                self.blks.append(
                    DiTBlockTimeOnly(
                        C=self.latent_dim,
                        num_heads=n_head,
                        num_registers=self.num_blk_internal_registers,
                        mlp_hidden_dims=[self.latent_dim * fc_factor] * fc_n_layers,
                        ada_scale_bias_bound=ada_scale_bias_bound,
                        gate_bound=gate_bound,
                        ada_n_layers=ada_n_layers,
                        C_feat=self.cond_binder_dim,
                    )
                )
            else:
                raise ValueError(f"Unknown layer type: {lt}")

        # Register tokens for t/p/a layers (operate on Q dim)
        self.a_reg = nn.Embedding(num_register, self.latent_dim)
        self.t_reg = nn.Embedding(num_register, self.latent_dim)
        self.p_reg = nn.Embedding(num_register, self.latent_dim)

        self.use_reg_binder = use_reg_binder
        # Only create reg_binder for t/p/a layers
        n_tpa = sum(1 for c in layer_type_string if c in "tpa")
        if self.use_reg_binder and n_tpa > 0:
            self.reg_binder = nn.ModuleList(
                [
                    nn.Linear(self.latent_dim * num_register, self.latent_dim)
                    for _ in range(n_tpa)
                ]
            )

        # Output projection: concat initial x_p + all d-layer skips → wide MLP
        # Each skip is (B, S, P, latent_dim), total = (1 + n_d) * latent_dim
        out_proj_c_in = self.latent_dim * (1 + self.n_d_layers)
        _seq = []
        for out_proj_c_out in np.linspace(
            out_proj_c_in, self.input_dim, fc_out_n_layers
        )[:-1]:
            out_proj_c_out = int(out_proj_c_out)
            _seq.append(nn.Linear(out_proj_c_in, out_proj_c_out))
            _seq.append(HardLayerNorm(out_proj_c_out))
            _seq.append(nn.LeakyReLU())
            out_proj_c_in = out_proj_c_out
        _seq.append(nn.Linear(out_proj_c_in, self.input_dim))
        self.output_proj = nn.Sequential(*_seq)

        self.__summary__()

    def __summary__(self):
        logging.info(f"LMM V5 Summary:")
        logging.info(f"  latent_dim: {self.latent_dim}")
        logging.info(f"  x_dim: {self.input_dim}")
        logging.info(f"  n_blocks: {self.n_blocks}")
        logging.info(f"  layer_type_string: {self.layer_type_string}")
        logging.info(f"  n_latent_queries (Q): {self.n_latent_queries}")
        logging.info(f"  q_pool_mode: {self.q_pool_mode}")
        logging.info(f"  n_d_layers (skips): {self.n_d_layers}")

    def get_optim_groups(self, weight_decay=0.0, base_lr=None):
        """Optimizer groups with weight-decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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

    def forward(
        self,
        x,
        x_mask,
        x_time,
        x_level,
        feat,
        ctx=None,
        ctx_mask=None,
        ctx_level=None,
        add_tokens=None,
        add_tokens_mask=None,
        image_tokens=None,
        image_tokens_mask=None,
    ):
        """Forward pass with Perceiver IO pooling.

        Args:
            x: [B, T, N, input_dim] noisy input. N = P (total parts).
            x_mask: [B, T, N] bool mask.
            x_time: [B, T] timestamps.
            x_level: [B, T, N] diffusion levels.
            feat: [B, N, feat_dim] per-part identity features.
            ctx, ctx_mask, ctx_level: optional context frames.
            add_tokens: [B, A, latent_dim] language tokens.
            add_tokens_mask: [B, A] mask.
            image_tokens: [B, I, latent_dim] DINO image tokens.
            image_tokens_mask: [B, I] mask.

        Returns:
            [B, S, N, input_dim] predicted clean output. S = T + N_CTX.
        """
        B, T, N, _ = x.shape
        Q = self.n_latent_queries
        device = x.device
        x_time = x_time.float().clone() * self.time_multiplier

        # ========== Input Processing (same as V2) ==========
        if self.add_time_to_feat:
            ctx_pos_emb = torch.cat(
                [
                    x_time[..., None],
                    self.t_pe(x_time.reshape(-1)).reshape(*x_time.shape[:2], -1),
                ],
                dim=-1,
            )
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # B,T,lat
        else:
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert N_CTX <= self.n_ctx
            assert ctx_mask is not None and ctx_level is not None
            x = torch.cat([ctx, x], dim=1)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)
            x_level = torch.cat([ctx_level, x_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]  # S = T + N_CTX

        # Build P-dim conditioning: (B, S, N, cond_dim)
        level_emb = (
            self.int_pe(x_level.reshape(B * S * N))
            .reshape(B, S, N, self.latent_dim)
            .to(x)
        )
        feat_expanded = feat[:, None].expand(-1, S, -1, -1)  # (B, S, N, feat_dim)
        ctx_pos_emb_expanded = ctx_pos_emb[:, :, None].expand(-1, -1, N, -1)  # (B, S, N, lat)

        cond_p = torch.cat(
            [
                self.hard_ln_feat(feat_expanded),
                self.hard_ln_lat(level_emb),
                self.hard_ln_lat(ctx_pos_emb_expanded),
            ],
            dim=-1,
        )  # (B, S, N, feat+2*lat)

        if self.add_first_n_add_token_to_feat > 0 and add_tokens is not None:
            assert add_tokens.shape[1] >= self.add_first_n_add_token_to_feat
            _add_feat = add_tokens[:, : self.add_first_n_add_token_to_feat].clone().contiguous()
            _add_feat = self.add_to_feat_fc1(_add_feat)
            _add_feat = self.add_to_feat_fc2(_add_feat.reshape(B, -1))
            _add_feat = _add_feat[:, None, None].expand(-1, S, N, -1)
            cond_p = torch.cat([cond_p, self.hard_ln_lat(_add_feat)], dim=-1)

        if self.num_random_trace_id_pe > 0:
            if self.training:
                _id = torch.argsort(
                    torch.rand(B, self.num_random_trace_id_pe, device=device), dim=-1
                )[:, :N]
            else:
                _id = torch.arange(N)[None].repeat(B, 1).to(device).long()
            _id_feat = self.random_trace_id_pe(_id)[:, None].repeat(1, S, 1, 1)
            cond_p = torch.cat([cond_p, self.hard_ln_lat(_id_feat)], dim=-1)

        cond_dim = cond_p.shape[-1]  # feat + 2*lat [+ extras]

        # Build Q-dim conditioning by mean-pooling P-dim cond over parts
        # (B, S, N, cond_dim) → mean over N → (B, S, 1, cond_dim) → tile to Q
        cond_q = cond_p.mean(dim=2, keepdim=True).expand(-1, -1, Q, -1)  # (B, S, Q, cond_dim)

        # Embed input
        x_emb = self.x_proj(self.x_pe(x))  # (B, S, N, latent_dim)

        # ========== Initialize Dual State ==========
        # x_p: dense representation (B*S, N, D)
        x_p = x_emb.reshape(B * S, N, self.latent_dim)
        # x_q: latent queries (B*S, Q, D) — from learned parameters
        x_q = self.latent_queries.expand(B * S, -1, -1)  # (B*S, Q, D)

        # Flatten cond for per-layer projection
        cond_p_flat = cond_p.reshape(B * S, N, cond_dim)   # for q/d layers
        cond_q_flat = cond_q.reshape(B * S, Q, cond_dim)   # for q/t/p/a layers

        # P-dim mask (B*S, N) for q-layer KV masking
        p_mask = x_mask.reshape(B * S, N)

        # Prepare x_time for t layers: need per-Q-track time
        # x_time: (B, T) → for t layers on Q, need (B*Q, T)
        # Note: only T (not N_CTX) timesteps have explicit time; ctx uses position emb
        x_time_for_t = x_time  # (B, T) — will be reshaped per t-layer call

        # ========== Prepare add_tokens for t/p/a ==========
        if add_tokens is not None:
            A = add_tokens.shape[1]
            assert add_tokens_mask is not None
        else:
            A = 0
            add_tokens = torch.zeros(B, 0, self.latent_dim, device=device)
            add_tokens_mask = torch.zeros(B, 0, device=device).bool()

        if image_tokens is not None:
            I = image_tokens.shape[1]
            assert image_tokens_mask is not None
        else:
            I = 0

        def _build_layer_add(layer_type):
            if I > 0 and layer_type in self.image_cross_attn_layers:
                tk = torch.cat([add_tokens, image_tokens], dim=1)
                mk = torch.cat([add_tokens_mask, image_tokens_mask], dim=1)
            else:
                tk = add_tokens
                mk = add_tokens_mask
            return tk, mk

        add_tk_a, add_mk_a = _build_layer_add("a")
        add_tk_t, add_mk_t = _build_layer_add("t")
        add_tk_p, add_mk_p = _build_layer_add("p")
        A_a = add_tk_a.shape[1]
        A_t = add_tk_t.shape[1]
        A_p = add_tk_p.shape[1]

        # ========== Build Attention Masks for Q-dim t/p/a ==========
        # Q tokens are always valid
        q_valid = torch.ones(B, S, Q, dtype=torch.bool, device=device)

        # a-layer mask: full attention over S*Q + add_tokens
        a_reg = self.a_reg(torch.arange(self.num_register, device=device))[None].repeat(B, 1, 1)
        attn_mask_a = build_attention_mask(
            torch.cat([add_mk_a, q_valid.reshape(B, S * Q)], dim=1),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_a = add_tk_a.clone()

        # t-layer mask: temporal attention per Q-track
        t_reg = self.t_reg(torch.arange(self.num_register, device=device))[None].repeat(B * Q, 1, 1)
        attn_mask_t = build_attention_mask(
            torch.cat(
                [
                    add_mk_t[:, None].expand(-1, Q, -1).reshape(B * Q, A_t),
                    q_valid.permute(0, 2, 1).reshape(B * Q, S),  # (B, Q, S) → (B*Q, S)
                ],
                dim=1,
            ),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_t = add_tk_t[:, None].expand(-1, Q, -1, -1).reshape(B * Q, A_t, self.latent_dim)

        # p-layer mask: part attention over Q per timestep
        p_reg = self.p_reg(torch.arange(self.num_register, device=device))[None].repeat(B * S, 1, 1)
        attn_mask_p = build_attention_mask(
            torch.cat(
                [
                    add_mk_p[:, None].expand(-1, S, -1).reshape(B * S, A_p),
                    q_valid.reshape(B * S, Q),
                ],
                dim=1,
            ),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_p = add_tk_p[:, None].expand(-1, S, -1, -1).reshape(B * S, A_p, self.latent_dim)

        # ========== Main Layer Loop ==========
        d_skip_list = [x_emb.reshape(B * S, N, self.latent_dim).clone()]  # initial P-dim state
        tpa_layer_counter = 0  # counter for reg_binder indexing

        for i, blk in enumerate(self.blks):
            lt = self.layer_type_string[i]

            if lt == "q":
                # Encode: Q queries attend to P key-values
                _cond_q = self.cond_binder[i](cond_q_flat)  # (B*S, Q, cond_binder_dim)
                x_q = blk(x_q, x_p, _cond_q, S=S, p_mask=p_mask)  # (B*S, Q, D)

            elif lt == "d":
                # Decode: P queries attend to Q key-values, collect skip
                _cond_p = self.cond_binder[i](cond_p_flat)  # (B*S, N, cond_binder_dim)
                x_p = blk(x_p, x_q, _cond_p)  # (B*S, N, D)
                d_skip_list.append(x_p.clone())

            elif lt == "a":
                # Global attention over all S*Q tokens
                x_q_flat = x_q.reshape(B, S * Q, self.latent_dim)
                _cond_q_4d = self.cond_binder[i](cond_q_flat)  # (B*S, Q, cbd)
                _cond = _cond_q_4d.reshape(B, S * Q, self.cond_binder_dim)

                _x = torch.cat([a_reg, add_a, x_q_flat], dim=1)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B, a_reg.shape[1] + add_a.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                x_q_flat = blk(x=_x, cond=_cond, t=None, attn_mask=attn_mask_a)
                a_reg = x_q_flat[:, : self.num_register]
                x_q_flat = x_q_flat[:, self.num_register + A_a :]
                if self.use_reg_binder:
                    x_q_flat = (
                        x_q_flat
                        + self.reg_binder[tpa_layer_counter](
                            a_reg.reshape(B, self.num_register * self.latent_dim)
                        )[:, None]
                    )
                x_q = x_q_flat.reshape(B * S, Q, self.latent_dim)
                tpa_layer_counter += 1

            elif lt == "t":
                # Temporal attention: reshape (B*S, Q, D) → (B, Q, S, D) → (B*Q, S, D)
                _track = x_q.reshape(B, S, Q, self.latent_dim).permute(0, 2, 1, 3)  # (B, Q, S, D)
                _track = _track.reshape(B * Q, S, self.latent_dim)
                _track = torch.cat([t_reg, add_t, _track], dim=1)

                _cond_q_4d = self.cond_binder[i](cond_q_flat)  # (B*S, Q, cbd)
                _cond = _cond_q_4d.reshape(B, S, Q, self.cond_binder_dim).permute(0, 2, 1, 3)
                _cond = _cond.reshape(B * Q, S, self.cond_binder_dim)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B * Q, t_reg.shape[1] + add_t.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                _track = blk(
                    _track,
                    cond=_cond,
                    t=(
                        x_time_for_t[:, :, None].expand(-1, -1, Q).reshape(B, T, Q)
                        .permute(0, 2, 1).reshape(B * Q, T)
                        if self.time_rope_flag
                        else None
                    ),
                    num_external_regs=self.num_register + A_t + N_CTX,
                    attn_mask=attn_mask_t,
                )
                t_reg = _track[:, : self.num_register]
                _track = _track[:, self.num_register + A_t :]
                if self.use_reg_binder:
                    _track = (
                        _track
                        + self.reg_binder[tpa_layer_counter](
                            t_reg.reshape(B * Q, self.num_register * self.latent_dim)
                        )[:, None]
                    )
                # (B*Q, S, D) → (B, Q, S, D) → (B, S, Q, D) → (B*S, Q, D)
                _track = _track.reshape(B, Q, S, self.latent_dim).permute(0, 2, 1, 3)
                x_q = _track.reshape(B * S, Q, self.latent_dim)
                tpa_layer_counter += 1

            elif lt == "p":
                # Part/spatial attention over Q dim: x_q is already (B*S, Q, D)
                _track = torch.cat([p_reg, add_p, x_q], dim=1)

                _cond = self.cond_binder[i](cond_q_flat)  # (B*S, Q, cbd)
                _cond = torch.cat(
                    [
                        torch.zeros(
                            B * S, p_reg.shape[1] + add_p.shape[1], self.cond_binder_dim,
                            device=device, dtype=_cond.dtype,
                        ),
                        _cond,
                    ],
                    dim=1,
                )
                _track = blk(
                    _track,
                    cond=_cond,
                    t=None,
                    num_external_regs=None,
                    attn_mask=attn_mask_p,
                )
                p_reg = _track[:, : self.num_register]
                _track = _track[:, self.num_register + A_p :]
                if self.use_reg_binder:
                    _track = (
                        _track
                        + self.reg_binder[tpa_layer_counter](
                            p_reg.reshape(B * S, self.num_register * self.latent_dim)
                        )[:, None]
                    )
                x_q = _track
                tpa_layer_counter += 1

        # ========== Output: concat d-layer skips + wide MLP ==========
        # d_skip_list: list of (B*S, N, D) tensors, len = 1 + n_d_layers
        x_out = torch.cat(d_skip_list, dim=-1)  # (B*S, N, (1+n_d)*D)
        ret = self.output_proj(x_out.reshape(-1, x_out.shape[-1]))
        ret = ret.reshape(B, S, N, self.input_dim)
        return ret  # (B, S, N, input_dim)


# Keep V2 name as alias for backward compat during transition
MotionMachineV2 = MotionMachineV5


def test_main():
    """Smoke test for MotionMachineV5 + IdentifyLayerV2."""
    io = IdentifyLayerV2(point_encoder_latent=128)
    io = io.to("cuda")
    io.train()

    net = MotionMachineV5(
        add_first_n_add_token_to_feat=-1,
        layer_type_string="qtpadqtpadqd",
        n_latent_queries=16,
        q_pool_mode="async",
    )
    net = net.to("cuda")
    net.train()

    B, T, N, C = 2, 8, 64, 9
    A = 16
    CTX = 3

    x = torch.randn(B, T, N, C).to("cuda")
    x_mask = torch.ones(B, T, N).to("cuda").bool()
    x_type = torch.randint(0, 7, (B, N)).to("cuda")
    x_group = torch.randint(0, 8, (B, N)).to("cuda")
    t = torch.randint(0, 256, (B, T)).to("cuda") / 256.0
    x_level = torch.zeros(B, T, N).to("cuda")
    pcl = torch.randn(B, N, 10, 6).to("cuda")
    pcl_valid_mask = torch.ones(B, N).to("cuda").bool()

    ctx = torch.randn(B, CTX, N, C).to("cuda")
    ctx_mask = torch.ones(B, CTX, N).to("cuda").bool()
    ctx_level = torch.randint(0, 7, (B, CTX, N)).to("cuda")

    feat = io(x_type, x_group, pcl, pcl_valid_mask)

    a_tokens = io.encode_language(["hello"] * B)
    a_mask = torch.ones(B, A).to("cuda").bool()

    ret = net(
        x=x,
        x_mask=x_mask,
        x_time=t,
        x_level=x_level,
        feat=feat,
        ctx=ctx,
        ctx_mask=ctx_mask,
        ctx_level=ctx_level,
        add_tokens=a_tokens.detach(),
        add_tokens_mask=a_mask.detach(),
    )

    print(f"Output shape: {ret.shape}")
    assert ret.shape == (B, T + CTX, N, C), f"Expected {(B, T + CTX, N, C)}, got {ret.shape}"

    loss = F.mse_loss(ret, ret.detach() - 1)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print("V5 async test passed!")

    # Test synced mode (fresh io/feat to avoid graph reuse)
    io2 = IdentifyLayerV2(point_encoder_latent=128).to("cuda")
    io2.train()
    feat2 = io2(x_type, x_group, pcl, pcl_valid_mask)
    a_tokens2 = io2.encode_language(["hello"] * B)

    net_sync = MotionMachineV5(
        add_first_n_add_token_to_feat=-1,
        layer_type_string="qtpadqd",
        n_latent_queries=16,
        q_pool_mode="synced",
    )
    net_sync = net_sync.to("cuda")
    net_sync.train()

    ret_sync = net_sync(
        x=x, x_mask=x_mask, x_time=t, x_level=x_level, feat=feat2,
        ctx=ctx, ctx_mask=ctx_mask, ctx_level=ctx_level,
        add_tokens=a_tokens2.detach(), add_tokens_mask=a_mask.detach(),
    )
    print(f"Synced output shape: {ret_sync.shape}")
    loss_sync = F.mse_loss(ret_sync, ret_sync.detach() - 1)
    loss_sync.backward()
    print(f"Synced loss: {loss_sync.item():.4f}")
    print("V5 synced test passed!")
    print("All V5 tests passed!")


if __name__ == "__main__":
    test_main()
