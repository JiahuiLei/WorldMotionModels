"""MotionMachine V4 -- all-attention motion prediction model.

Contains:
    - **IdentifyLayerV4**: Encodes per-joint identity (semantic ID, group ID,
      point cloud geometry, and language) into a conditioning feature vector.
      Functionally identical to IdentifyLayerV2.
    - **OutputHead**: Multi-layer output projection with optional LayerNorm
      and LeakyReLU.
    - **MotionMachineV4**: Transformer-based denoiser that uses only global
      ("a") attention layers (no factored temporal/spatial layers), conditioned
      via AdaLN-Zero. Compared to V2, this variant flattens all tokens into a
      single sequence and processes them with unified attention blocks.

Example:
    >>> io = IdentifyLayerV4(latent_dim=512, feat_dim=2048)
    >>> net = MotionMachineV4(latent_dim=512, feat_dim=2048)
    >>> feat = io(sem_id, group_id, pcl, pcl_mask)
    >>> out = net(x, x_mask, x_time, x_level, feat)
"""

#

import torch
from torch import nn
import logging
from diffusers.models.embeddings import Timesteps
from torch.nn import functional as F
import numpy as np
import sys, os, os.path as osp

from umm.models.umm_v1.world_models.wm_utils import (
    HardLayerNorm,
    SimpleFC,
    build_attention_mask,
    NerfStylePE,
    disable_model_grad,
)
from umm.models.umm_v1.world_models.pointnet import ResnetPointnet, ResnetPointnetV2
from umm.models.umm_v1.world_models.dit_atten import DiTBlockTimeOnly
from umm.models.umm_v1.world_models.language import T5LanguageEncoder


# * What is moving
class IdentifyLayerV4(nn.Module):
    """
    IdentifyLayerV4 encodes semantic ID, group ID, point cloud, and language into a single feature vector.

    Args:
        latent_dim (int): Dimension of the latent embedding.
        feat_dim (int): Dimension of the output feature vector.
        num_types (int): Number of semantic types.
        num_groups (int): Number of group IDs.
        group_emb_same_init_flag (bool): Whether to initialize group embeddings with the same value.
        point_version (str): Version of the point cloud encoder ("v1" or "v2").
        point_encoder_latent (int): Latent dimension for point encoder.
        point_input_dim (int): Input dimension for point cloud.
        point_xyz_scale_factor (float): Scale factor for point cloud coordinates.
        langauge_fixed_length (int): Fixed length for language encoding.
    """

    # this is useful for human, hand, manipualtor stuff
    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 2048,
        # *
        num_types: int = 512,
        num_groups: int = 512,
        group_emb_same_init_flag=True,
        # *
        point_version: str = "v1",
        point_encoder_latent: int = -1,
        point_input_dim: int = 6,
        point_xyz_scale_factor: float = 1.0,
        # *
        langauge_fixed_length: int = 16,
    ):
        """Initialize embeddings, point encoder, and language encoder.

        Example:
            >>> layer = IdentifyLayerV4(latent_dim=512, feat_dim=2048,
            ...                         point_encoder_latent=128)
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        logging.warning(f"latent_dim: {self.latent_dim}, feat_dim: {self.feat_dim}")

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
        """
        Forward pass for IdentifyLayerV4.

        Args:
            sem_id (torch.Tensor): Semantic IDs [B, N].
            group_id (torch.Tensor): Group IDs [B, N].
            pcl (torch.Tensor, optional): Point cloud [B, N, PCL, 3].
            pcl_valid_mask (torch.Tensor, optional): Mask for valid points [B, N].

        Returns:
            torch.Tensor: Combined feature embedding [B, N, D].
        """

        B, N = sem_id.shape
        type_emb = self.type_emb(sem_id)
        group_emb = self.group_emb(group_id)

        # encode pcl
        geo_emb = self.empty_point_emb(torch.tensor([0], device=sem_id.device))[None]
        geo_emb = geo_emb.repeat(B, N, 1).contiguous()  # B,N,D
        if self.point_flag and pcl is not None:
            if pcl_valid_mask is None:  # by default all valid
                pcl_valid_mask = torch.ones(B, N, device=sem_id.device).bool()
            valid_pcl = pcl[pcl_valid_mask].clone()  # B',M,3
            valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
            valid_pcl_feat = self.pntnet(valid_pcl)  # B',D
            geo_emb[pcl_valid_mask] = (
                geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
            )

        type_emb = self.ln_type_emb(type_emb)  # B,N,D
        group_emb = self.ln_group_emb(group_emb)  # B,N,D
        geo_emb = self.ln_geo_emb(geo_emb)

        feat = torch.cat([type_emb, group_emb, geo_emb], dim=-1)

        feat = self.bind_fc(feat)

        return feat  # B,N,D

    @torch.inference_mode()
    def encode_language(self, text_in):
        """Encode text strings into latent language tokens (no gradient).

        Runs the frozen T5 encoder and projects the output to ``latent_dim``.

        Args:
            text_in (list[str]): Batch of text strings.

        Returns:
            torch.Tensor: Language token embeddings of shape
                ``[B, lang_fixed_len, latent_dim]``, detached from the graph.

        Example:
            >>> layer = IdentifyLayerV4()
            >>> tokens = layer.encode_language(["pick the cube"])
        """
        language_tokens = self.language_encoder.encode(text_in)
        language_tokens = self.lang_fc(language_tokens)
        return language_tokens.detach()

    def get_optim_groups(self, weight_decay=0.0, base_lr=None):
        """Build optimizer parameter groups with weight-decay separation.

        Splits parameters into two groups: 2-D+ tensors (weight decay) and
        < 2-D tensors (no weight decay). The frozen language encoder parameters
        are excluded.

        Args:
            weight_decay (float): Weight decay for decayable parameters.
            base_lr (float | None): If provided, sets ``lr`` on each group.

        Returns:
            list[dict]: Parameter groups suitable for ``torch.optim`` optimizers.

        Example:
            >>> layer = IdentifyLayerV4()
            >>> groups = layer.get_optim_groups(weight_decay=0.01, base_lr=1e-4)
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # remove language_encoder
        param_dict = {
            pn: p
            for pn, p in param_dict.items()
            if not pn.startswith("language_encoder")
        }
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Set base learning rate if provided
        if base_lr is not None:
            for group in optim_groups:
                group["lr"] = base_lr
        return optim_groups


class OutputHead(nn.Module):
    """Multi-layer output projection head with optional LayerNorm.

    Linearly interpolates hidden dimensions from ``c_in`` down to ``c_out``
    across ``n_layers`` steps, with LeakyReLU activations and optional
    HardLayerNorm between layers.

    Args:
        c_out (int): Output dimension.
        c_in (int): Input dimension.
        n_layers (int): Number of intermediate projection layers.
        ln_flag (bool): Whether to insert HardLayerNorm after each linear.

    Example:
        >>> head = OutputHead(c_out=9, c_in=1536, n_layers=4)
        >>> out = head(torch.randn(8, 1536))  # [8, 9]
    """

    def __init__(self, c_out, c_in, n_layers, ln_flag=True):
        """Initialize the output head layers.

        Example:
            >>> head = OutputHead(c_out=9, c_in=1536, n_layers=4)
        """
        super().__init__()
        _seq = []
        out_proj_c_in = c_in
        for out_proj_c_out in np.linspace(c_in, c_out, n_layers)[:-1]:
            out_proj_c_out = int(out_proj_c_out)
            _seq.append(nn.Linear(out_proj_c_in, out_proj_c_out))
            logging.info(
                f"[OutputHead]: Act(Linear({out_proj_c_in} -> {out_proj_c_out}))"
            )
            if ln_flag:
                _seq.append(HardLayerNorm(out_proj_c_out))
            _seq.append(nn.LeakyReLU())
            out_proj_c_in = out_proj_c_out
        _seq.append(nn.Linear(out_proj_c_in, c_out))
        self.head = nn.Sequential(*_seq)
        # self.res = nn.Linear(c_in, c_out)
        return

    def forward(self, x):
        """Project input through the multi-layer head.

        Args:
            x (torch.Tensor): Input of shape ``[..., c_in]``.

        Returns:
            torch.Tensor: Output of shape ``[..., c_out]``.

        Example:
            >>> head = OutputHead(c_out=9, c_in=1536, n_layers=4)
            >>> out = head(torch.randn(8, 1536))  # [8, 9]
        """
        return self.head(x)  # + self.res(x)


# * How it moves
class MotionMachineV4(nn.Module):
    """
    MotionMachineV4 predicts motion with enhanced features compared to V2.

    Args:
        latent_dim (int): Latent dimension of the model.
        feat_dim (int): Dimension of the conditioning feature.
        n_head (int): Number of attention heads.
        input_dim (int): Dimension of the input state.
        output_dim (int): Dimension of the output prediction.
        layer_type_string (str): String defining the sequence of layer types.
        skip_layer_idx (list): Indices of layers to skip/concatenate.
        cond_binder_dim (int): Dimension of the condition binder.
        fc_out_n_layers (int): Number of layers in the output projection.
        fc_out_ln_flag (bool): Whether to use LayerNorm in output projection.
        fc_factor (int): Expansion factor for MLP layers.
        fc_n_layers (int): Number of layers in the MLP.
        num_register (int): Number of register tokens.
        num_blk_internal_registers (int): Number of internal registers in the block.
        n_ctx (int): Context size.
        x_pe_dim (int): Dimension of NerfStylePE for input X.
        time_multiplier (float): Multiplier for time input.
        use_reg_binder (bool): Whether to use register binder.
        ada_scale_bias_bound (float): Bound for AdaLN scale/bias.
        gate_bound (float): Bound for gate values.
        ada_n_layers (int): Number of AdaLN layers.
        add_first_n_add_token_to_feat (int): Number of additional tokens to add to feature.
        time_rope_flag (bool): Whether to use RoPE for time.
        add_time_to_feat (bool): Whether to add time to feature.
        num_random_trace_id_pe (int): Number of random trace ID positional encodings.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 2048,
        n_head: int = 16,
        input_dim: int = 9,
        output_dim: int = -1,
        layer_type_string: str = "aaaaaa",
        skip_layer_idx: list = [2, 4],
        cond_binder_dim: int = 4096,
        #
        fc_out_n_layers: int = 4,
        fc_out_ln_flag: bool = True,
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
        #
    ):
        """Initialize all-attention DiT blocks, registers, and output head.

        Example:
            >>> net = MotionMachineV4(latent_dim=512, feat_dim=2048)
        """
        super().__init__()

        self.layer_type_string = layer_type_string
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.cond_binder_dim = cond_binder_dim

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else self.input_dim
        self.n_blocks = len(self.layer_type_string)
        for _t in layer_type_string:
            assert _t in ["a"], f"Invalid layer type: {_t}"
        if self.n_blocks - 1 not in skip_layer_idx:
            skip_layer_idx.append(self.n_blocks - 1)
        self.skip_layer_idx = skip_layer_idx
        self.n_head = n_head
        self.num_register = num_register
        self.hard_ln_lat = HardLayerNorm(self.latent_dim)
        self.hard_ln_feat = HardLayerNorm(self.feat_dim)
        logging.warning(f"num_register: {self.num_register}")

        self.time_rope_flag = time_rope_flag
        if self.time_rope_flag:
            logging.info("use time rope")
        self.add_time_to_feat = add_time_to_feat
        if self.add_time_to_feat:
            logging.info("add time to feat by sharing the VT buffer")
            self.t_proj = nn.Linear(self.latent_dim + 1, self.latent_dim)

        # * PE and EMBs
        # diffusion step
        self.int_pe = Timesteps(
            self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
        )  # GPT style PE, works for 0,1,2,3,4,5 ...
        self.time_multiplier = time_multiplier

        self.x_pe_dim = x_pe_dim
        if x_pe_dim > 0:
            logging.warning("use nerfstyle pe for x input")
            self.x_pe = NerfStylePE(
                in_channels=1, num_frequencies=x_pe_dim, include_input=True
            )  # Nerf style PE, for make details in smaller range like std=1.0 index scale
            x_proj_in = self.x_pe.out_channels * self.input_dim
        else:
            self.x_pe = nn.Identity()
            x_proj_in = self.input_dim
        self.x_proj = nn.Linear(x_proj_in, self.latent_dim)

        self.n_ctx = n_ctx
        self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.latent_dim)

        # cond binder
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
            logging.warning(
                f"use random trace id pe with {self.num_random_trace_id_pe}"
            )
            self.random_trace_id_pe = nn.Embedding(
                self.num_random_trace_id_pe, self.latent_dim
            )
            with torch.no_grad():
                logging.warning(f"Random Trace ID PE all same init!")
                self.random_trace_id_pe.weight.copy_(
                    self.random_trace_id_pe.weight[0].expand_as(
                        self.random_trace_id_pe.weight
                    )
                )
            cond_binder_in += self.latent_dim

        self.cond_binder = nn.ModuleList(
            [
                nn.Linear(cond_binder_in, self.cond_binder_dim)
                for _ in range(self.n_blocks)
            ]
        )
        self.empty_cond_emb = nn.Embedding(self.n_blocks, self.cond_binder_dim)

        # output layer
        self.output_head = OutputHead(
            c_out=self.input_dim,
            c_in=self.latent_dim * (1 + len(self.skip_layer_idx)),
            n_layers=fc_out_n_layers,
            ln_flag=fc_out_ln_flag,
        )

        self.num_blk_internal_registers = num_blk_internal_registers
        self.blks = nn.ModuleList(
            [
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
                for i in range(self.n_blocks)
            ]
        )

        self.a_reg = nn.Embedding(num_register, self.latent_dim)

        self.use_reg_binder = use_reg_binder
        if self.use_reg_binder:
            self.reg_binder = nn.ModuleList(
                [
                    nn.Linear(
                        self.latent_dim * num_register,
                        self.latent_dim,
                    )
                    for i in range(self.n_blocks)
                ]
            )

        self.__summary__()
        return

    def __summary__(self):
        """Log a human-readable summary of the model architecture.

        Example:
            >>> net = MotionMachineV4()
            >>> net.__summary__()
        """
        logging.info(f"LMM Summary:")
        logging.info(f"  latent_dim: {self.latent_dim}")
        logging.info(f"  x_dim: {self.input_dim}")
        logging.info(f"  n_blocks: {self.n_blocks}")
        logging.info(f"  layer_type_string: {self.layer_type_string}")
        logging.info(f"  skip_layer_idx: {self.skip_layer_idx}")
        return

    def get_optim_groups(self, weight_decay=0.0, base_lr=None):
        """Build optimizer parameter groups with weight-decay separation.

        Splits parameters into two groups: 2-D+ tensors (weight decay) and
        < 2-D tensors (no weight decay).

        Args:
            weight_decay (float): Weight decay for decayable parameters.
            base_lr (float | None): If provided, sets ``lr`` on each group.

        Returns:
            list[dict]: Parameter groups suitable for ``torch.optim`` optimizers.

        Example:
            >>> net = MotionMachineV4()
            >>> groups = net.get_optim_groups(weight_decay=0.01, base_lr=1e-4)
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Set base learning rate if provided
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
    ):
        """
        Forward pass for MotionMachineV4.

        Args:
            x (torch.Tensor): Input latent sequence [B, T, N, D].
            x_mask (torch.Tensor): Mask for input sequence [B, T, N].
            x_time (torch.Tensor): Timestamp for input sequence [B, T].
            x_level (torch.Tensor): Diffusion step/level [B, T, N].
            feat (torch.Tensor): Conditioning feature [B, N, D].
            ctx (torch.Tensor, optional): Context latent sequence [B, C, N, D].
            ctx_mask (torch.Tensor, optional): Context mask [B, C, N].
            ctx_level (torch.Tensor, optional): Context level [B, C, N].
            add_tokens (torch.Tensor, optional): Additional tokens [B, A, D].
            add_tokens_mask (torch.Tensor, optional): Mask for additional tokens [B, A].

        Returns:
            torch.Tensor: Predicted output [B, T * N, output_dim] (projected).
        """
        # cond: feat + virtual/real_t + level
        # the time pe is for x_time
        # ! The time pe is only applied in t layer for now.

        B, T, N, _ = x.shape
        assert x_mask.shape == (B, T, N)
        assert x_time.shape == (B, T)
        assert x_level.shape == (B, T, N)
        assert feat.shape == (B, N, self.feat_dim)
        device = x.device
        x_time = x_time.float().clone() * self.time_multiplier

        if self.add_time_to_feat:
            ctx_pos_emb = torch.cat(
                [
                    x_time[..., None],
                    self.int_pe(x_time.reshape(-1)).reshape(*x_time.shape[:2], -1),
                ],
                dim=-1,
            )
            ctx_pos_emb = self.t_proj(ctx_pos_emb)  # B,T,lat
        else:
            # ! fix critical bug here since lt3.0+
            ctx_pos_emb = self.ctx_t_emb(
                (torch.ones(B, T, dtype=torch.long, device=device) * self.n_ctx).long()
            )  # ! fetch the last n_ctx position as not to be ctx!!!!!

        if ctx is not None:
            N_CTX = ctx.shape[1]
            assert (
                N_CTX <= self.n_ctx
            ), f"CTX: {N_CTX}, should be < {self.n_ctx} virtual time"
            assert ctx_mask is not None and ctx_level is not None
            assert ctx.shape == (B, N_CTX, N, self.input_dim)
            assert ctx_mask.shape == (B, N_CTX, N)
            assert ctx_level.shape == (B, N_CTX, N)
            x = torch.cat([ctx, x], dim=1)
            x_mask = torch.cat([ctx_mask, x_mask], dim=1)
            x_level = torch.cat([ctx_level, x_level], dim=1)
            _ctx_pos_emb = self.ctx_t_emb(torch.arange(N_CTX, device=device))[
                None
            ].expand(B, -1, -1)
            ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
        else:
            N_CTX = 0

        S = x_level.shape[1]
        level_emb = (
            self.int_pe(x_level.reshape(B * S * N))
            .reshape(B, S, N, self.latent_dim)
            .to(x)
        )
        feat = feat[:, None].expand(-1, S, -1, -1)
        ctx_pos_emb = ctx_pos_emb[:, :, None].expand(-1, -1, N, -1)
        cond = torch.cat(
            [
                self.hard_ln_feat(feat),
                self.hard_ln_lat(level_emb),
                self.hard_ln_lat(ctx_pos_emb),
            ],
            dim=-1,
        )  # B,T,N,feat+2*lat
        if self.add_first_n_add_token_to_feat > 0 and add_tokens is not None:
            assert add_tokens.shape[1] >= self.add_first_n_add_token_to_feat
            _add_feat = (
                add_tokens[:, : self.add_first_n_add_token_to_feat].clone().contiguous()
            )
            _add_feat = self.add_to_feat_fc1(_add_feat)
            _add_feat = self.add_to_feat_fc2(_add_feat.reshape(B, -1))
            _add_feat = _add_feat[:, None, None].expand(-1, S, N, -1)
            cond = torch.cat([cond, self.hard_ln_lat(_add_feat)], dim=-1)
        if self.num_random_trace_id_pe > 0:
            assert (
                N <= self.num_random_trace_id_pe
            ), f"N: {N}, num_random_trace_id_pe: {self.num_random_trace_id_pe}"
            if self.training:
                _id = torch.argsort(
                    torch.rand(B, self.num_random_trace_id_pe, device=device), dim=-1
                )[
                    :, :N
                ]  # B,N
            else:
                _id = torch.arange(N)[None].repeat(B, 1).to(device).long()  # B,N
            _id_feat = self.random_trace_id_pe(_id)  # B,N,Lat
            _id_feat = _id_feat[:, None].repeat(1, S, 1, 1)
            cond = torch.cat([cond, self.hard_ln_lat(_id_feat)], dim=-1)

        x_emb = self.x_proj(self.x_pe(x))  # B,T,N,lat

        ##########################################

        x = x_emb.reshape(B, S * N, self.latent_dim)
        cond = cond.reshape(B, S * N, cond.shape[-1])
        x_time = x_time[:, :, None].expand(-1, -1, N).reshape(B, T * N)

        ##########################################

        if add_tokens is not None:
            A = add_tokens.shape[1]
            assert add_tokens_mask is not None
            assert add_tokens.shape == (B, A, self.latent_dim)
            assert add_tokens_mask.shape == (B, A)
        else:
            A = 0
            add_tokens = torch.zeros(B, A, self.latent_dim, device=device)
            add_tokens_mask = torch.zeros(B, A, device=device).bool()

        # * all layers share the same register
        a_reg = self.a_reg(torch.arange(self.num_register).to(device))  # R,Lat
        a_reg = a_reg[None].repeat(B, 1, 1)  # B,R,D
        attn_mask_a = build_attention_mask(
            torch.cat([add_tokens_mask, x_mask.reshape(B, S * N)], dim=1),
            mode="full",
            left_pad=self.num_register + self.num_blk_internal_registers,
        )[:, None].expand(-1, self.n_head, -1, -1)
        add_a = add_tokens.clone()  # B,A,D

        skip_list = [x.clone()]
        for bid, blk in enumerate(self.blks):
            _cond = self.cond_binder[bid](cond)
            if self.layer_type_string[bid] == "a":
                _x = torch.cat([a_reg, add_a, x], dim=1)
                _empty_cond = self.empty_cond_emb(
                    torch.tensor([bid]).long().to(_cond.device)
                )
                _empty_cond = _empty_cond[None].expand(B, self.num_register + A, -1)
                _cond = torch.cat([_empty_cond, _cond], dim=1)  # dup the prefix
                x = blk(x=_x, cond=_cond, t=None, attn_mask=attn_mask_a)
                a_reg = x[:, : self.num_register]
                # ! also update add_a here!
                add_a = x[:, self.num_register : self.num_register + A]
                x = x[:, self.num_register + A :]
                if self.use_reg_binder:
                    x = (
                        x
                        + self.reg_binder[bid](
                            a_reg.reshape(B, self.num_register * self.latent_dim)
                        )[:, None]
                    )
            else:
                raise ValueError(f"Invalid layer type: {self.layer_type_string[bid]}")

            # x with more fc
            if bid in self.skip_layer_idx:
                skip_list.append(x.clone())

        x = torch.cat(skip_list, dim=-1)
        ret = self.output_head(x.reshape(-1, x.shape[-1]))
        ret = ret.reshape(B, S, N, self.input_dim)
        return ret  # B,S,N,C


def test_main():
    """Smoke test for MotionMachineV4 + IdentifyLayerV4 with random tensors."""

    io = IdentifyLayerV4(point_encoder_latent=128)
    io = io.to("cuda")
    io.train()

    net = MotionMachineV4(add_first_n_add_token_to_feat=-1)
    net = net.to("cuda")
    net.train()

    B, T, N, C = 4, 50, 35, 9
    A = 16
    CTX = 5

    x = torch.randn(B, T, N, C).to("cuda")
    x_mask = torch.ones(B, T, N).to("cuda").bool()
    x_type = torch.randint(0, 7, (B, N)).to("cuda")
    x_group = torch.randint(0, 8, (B, N)).to("cuda")
    t = torch.randint(0, 256, (B, T)).to("cuda") / 256.0
    vt_mask = torch.zeros(B, T).to("cuda").bool()
    x_level = torch.zeros(B, T, N).to("cuda")
    pcl = torch.randn(B, N, 10, 6).to("cuda")
    pcl_valid_mask = torch.ones(B, N).to("cuda").bool()

    ctx = torch.randn(B, CTX, N, C).to("cuda")
    ctx_mask = torch.ones(B, CTX, N).to("cuda").bool()
    ctx_level = torch.randint(0, 7, (B, CTX, N)).to("cuda")

    feat = io(x_type, x_group)

    feat = io(x_type, x_group, pcl, pcl_valid_mask)

    # a_tokens = torch.randn(B, A, feat.shape[-1]).to("cuda")
    # a_mask = torch.ones(B, A).to("cuda").bool()

    a_tokens = io.encode_language(["hello"] * B)
    a_mask = torch.ones(B, A).to("cuda").bool()

    ret = net(
        x=x,
        x_mask=x_mask,
        x_time=t,
        x_level=x_level,
        feat=feat,
        # context
        ctx=ctx,
        ctx_mask=ctx_mask,
        ctx_level=ctx_level,
        #
        add_tokens=a_tokens.detach(),
        add_tokens_mask=a_mask.detach(),
    )

    loss = F.mse_loss(ret, ret - 1)

    loss.backward()

    print(ret.shape)


if __name__ == "__main__":
    test_main()
