"""MotionMachine V1 -- archived/deprecated.

This module contains the original MotionMachine V1 implementation
(IdentifyLayerV1 and MotionMachineV1). The entire file is commented out
and preserved only for historical reference. Use ``moma_v2`` or ``moma_v4``
for active development.
"""

# #

# import torch
# from torch import nn
# import logging
# from diffusers.models.embeddings import Timesteps
# from torch.nn import functional as F
# import numpy as np
# import sys, os, os.path as osp

# # from umm.models.umm_v1.world_models.gpt2 import Block as SingleAttnBlock
# from umm.models.umm_v1.world_models.wm_utils import (
#     HardLayerNorm,
#     SimpleFC,
#     build_attention_mask,
#     NerfStylePE,
#     disable_model_grad,
# )
# from umm.models.umm_v1.world_models.pointnet import ResnetPointnet, ResnetPointnetV2
# from umm.models.umm_v1.world_models.dit_atten import DiTBlockTimeOnly
# from umm.models.umm_v1.world_models.language import T5LanguageEncoder


# # * What is moving
# class IdentifyLayerV1(nn.Module):
#     """
#     IdentifyLayerV1 encodes semantic ID, group ID, point cloud, and language into a single feature vector.

#     Args:
#         latent_dim (int): Dimension of the latent embedding.
#         num_types (int): Number of semantic types.
#         num_groups (int): Number of group IDs.
#         group_emb_same_init_flag (bool): Whether to initialize group embeddings with the same value.
#         point_version (str): Version of the point cloud encoder ("v1" or "v2").
#         point_encoder_latent (int): Latent dimension for point encoder.
#         point_input_dim (int): Input dimension for point cloud.
#         point_xyz_scale_factor (float): Scale factor for point cloud coordinates.
#         langauge_fixed_length (int): Fixed length for language encoding.
#     """

#     # this is useful for human, hand, manipualtor stuff
#     # todo: next version include image features
#     def __init__(
#         self,
#         latent_dim: int = 512,
#         # *
#         num_types: int = 512,
#         num_groups: int = 512,
#         group_emb_same_init_flag=True,
#         # *
#         point_version: str = "v1",
#         point_encoder_latent: int = -1,
#         point_input_dim: int = 6,
#         point_xyz_scale_factor: float = 1.0,
#         # *
#         langauge_fixed_length: int = 16,
#     ):

#         super().__init__()
#         self.latent_dim = latent_dim

#         self.n_types = num_types
#         self.n_groups = num_groups

#         self.type_emb = nn.Embedding(self.n_types, self.latent_dim)
#         self.group_emb = nn.Embedding(self.n_groups, self.latent_dim)
#         if group_emb_same_init_flag:
#             with torch.no_grad():
#                 logging.warning(f"Group EMB all same init!")
#                 self.group_emb.weight.copy_(
#                     self.group_emb.weight[0].expand_as(self.group_emb.weight)
#                 )

#         self.point_flag = point_encoder_latent > 0
#         if self.point_flag:
#             if point_version == "v1":
#                 self.pntnet = ResnetPointnet(
#                     c_dim=self.latent_dim,
#                     dim=point_input_dim,
#                     hidden_dim=point_encoder_latent,
#                 )
#             elif point_version == "v2":
#                 self.pntnet = ResnetPointnetV2(
#                     c_dim=self.latent_dim,
#                     dim=point_input_dim,
#                     hidden_dim=point_encoder_latent,
#                 )
#         self.point_xyz_scale_factor = point_xyz_scale_factor
#         logging.warning(f"Point XYZ scale factor: {self.point_xyz_scale_factor}")
#         self.empty_point_emb = nn.Embedding(1, self.latent_dim)

#         self.ln_type_emb = HardLayerNorm(self.latent_dim)
#         self.ln_group_emb = HardLayerNorm(self.latent_dim)
#         self.ln_geo_emb = HardLayerNorm(self.latent_dim)

#         self.bind_fc = nn.Linear(self.latent_dim * 3, self.latent_dim)

#         # * language
#         self.lang_fixed_len = langauge_fixed_length
#         self.language_encoder = disable_model_grad(
#             T5LanguageEncoder(fixed_length=self.lang_fixed_len)
#         )
#         self.lang_fc = nn.Linear(768, self.latent_dim)

#         return

#     def forward(self, sem_id, group_id, pcl=None, pcl_valid_mask=None):
#         # sem_id: B,N; group_id: B,N; pcl: B,N,PCL,3; pcl_valid_mask: B,N

#         B, N = sem_id.shape
#         type_emb = self.type_emb(sem_id)
#         group_emb = self.group_emb(group_id)

#         # encode pcl
#         geo_emb = self.empty_point_emb(torch.tensor([0], device=sem_id.device))[None]
#         geo_emb = geo_emb.repeat(B, N, 1).contiguous()  # B,N,D
#         if self.point_flag and pcl is not None:
#             if pcl_valid_mask is None:  # by default all valid
#                 pcl_valid_mask = torch.ones(B, N, device=sem_id.device).bool()
#             valid_pcl = pcl[pcl_valid_mask].clone()  # B',M,3
#             valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
#             valid_pcl_feat = self.pntnet(valid_pcl)  # B',D
#             geo_emb[pcl_valid_mask] = (
#                 geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
#             )

#         type_emb = self.ln_type_emb(type_emb)  # B,N,D
#         group_emb = self.ln_group_emb(group_emb)  # B,N,D
#         geo_emb = self.ln_geo_emb(geo_emb)
#         feat = self.bind_fc(torch.cat([type_emb, group_emb, geo_emb], dim=-1))

#         return feat  # B,N,D

#     @torch.inference_mode()
#     def encode_language(self, text_in):
#         language_tokens = self.language_encoder.encode(text_in)
#         language_tokens = self.lang_fc(language_tokens)
#         return language_tokens.detach()

#     def get_optim_groups(self, weight_decay=0.0, base_lr=None):
#         # start with all of the candidate parameters
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         # filter out those that do not require grad
#         param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#         # remove language_encoder
#         param_dict = {
#             pn: p
#             for pn, p in param_dict.items()
#             if not pn.startswith("language_encoder")
#         }
#         # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#         # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#         optim_groups = [
#             {"params": decay_params, "weight_decay": weight_decay},
#             {"params": nodecay_params, "weight_decay": 0.0},
#         ]
#         # Set base learning rate if provided
#         if base_lr is not None:
#             for group in optim_groups:
#                 group["lr"] = base_lr
#         return optim_groups


# # * How it moves
# class MotionMachineV1(nn.Module):
#     """
#     MotionMachineV1 predicts motion using a Transformer-based architecture.

#     Args:
#         latent_dim (int): Latent dimension of the model.
#         n_head (int): Number of attention heads.
#         input_dim (int): Dimension of the input state.
#         output_dim (int): Dimension of the output prediction.
#         layer_type_string (str): String defining the sequence of layer types.
#         skip_layer_idx (list): Indices of layers to skip/concatenate.
#         fc_out_n_layers (int): Number of layers in the output projection.
#         fc_factor (int): Expansion factor for MLP layers.
#         fc_n_layers (int): Number of layers in the MLP.
#         num_register (int): Number of register tokens.
#         num_blk_internal_registers (int): Number of internal registers in the block.
#         n_ctx (int): Context size.
#         x_pe_dim (int): Dimension of NerfStylePE for input X.
#         time_multiplier (float): Multiplier for time input.
#         use_reg_binder (bool): Whether to use register binder.
#         ada_scale_bias_bound (float): Bound for AdaLN scale/bias.
#         gate_bound (float): Bound for gate values.
#         ada_n_layers (int): Number of AdaLN layers.
#     """

#     # ! group can be replaced by mask operation.
#     # todo: remove the g layer unused layers!
#     def __init__(
#         self,
#         latent_dim: int = 512,
#         n_head: int = 16,
#         input_dim: int = 9,
#         output_dim: int = -1,
#         layer_type_string: str = "tpatpa",
#         skip_layer_idx: list = [2, 4],
#         #
#         fc_out_n_layers: int = 4,
#         fc_factor: int = 2,
#         fc_n_layers: int = 2,
#         num_register: int = 2,
#         num_blk_internal_registers: int = 4,
#         n_ctx: int = 128,
#         #
#         x_pe_dim: int = 0,
#         time_multiplier: float = 1.0,
#         #
#         use_reg_binder: bool = True,
#         #
#         ada_scale_bias_bound=-1.0,
#         gate_bound=-1.0,
#         ada_n_layers: int = 1,
#     ):
#         super().__init__()

#         self.layer_type_string = layer_type_string
#         self.latent_dim = latent_dim
#         self.input_dim = input_dim
#         self.output_dim = output_dim if output_dim > 0 else self.input_dim
#         self.n_blocks = len(self.layer_type_string)
#         if self.n_blocks - 1 not in skip_layer_idx:
#             skip_layer_idx.append(self.n_blocks - 1)
#         self.skip_layer_idx = skip_layer_idx
#         self.n_head = n_head
#         self.num_register = num_register
#         self.hard_ln = HardLayerNorm(self.latent_dim)
#         logging.warning(f"num_register: {self.num_register}")

#         # * PE and EMBs
#         # diffusion step
#         self.int_pe = Timesteps(
#             self.latent_dim, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1
#         )  # GPT style PE, works for 0,1,2,3,4,5 ...
#         self.time_multiplier = time_multiplier

#         self.x_pe_dim = x_pe_dim
#         if x_pe_dim > 0:
#             logging.warning("use nerfstyle pe for x input")
#             self.x_pe = NerfStylePE(
#                 in_channels=1, num_frequencies=x_pe_dim, include_input=True
#             )  # Nerf style PE, for make details in smaller range like std=1.0 index scale
#             x_proj_in = self.x_pe.out_channels * self.input_dim
#         else:
#             self.x_pe = nn.Identity()
#             x_proj_in = self.input_dim
#         self.x_proj = nn.Linear(x_proj_in, self.latent_dim)

#         self.n_ctx = n_ctx
#         self.ctx_t_emb = nn.Embedding(self.n_ctx + 1, self.latent_dim)

#         self.cond_binder = nn.ModuleList(
#             [
#                 nn.Linear(self.latent_dim * 3, self.latent_dim)
#                 for _ in range(self.n_blocks)
#             ]
#         )

#         # output layer
#         out_proj_c_in = self.latent_dim * (1 + len(self.skip_layer_idx))
#         _seq = []
#         for out_proj_c_out in np.linspace(
#             out_proj_c_in, self.input_dim, fc_out_n_layers
#         )[:-1]:
#             out_proj_c_out = int(out_proj_c_out)
#             _seq.append(nn.Linear(out_proj_c_in, out_proj_c_out))
#             _seq.append(HardLayerNorm(out_proj_c_out))
#             _seq.append(nn.LeakyReLU())
#             out_proj_c_in = out_proj_c_out
#         _seq.append(nn.Linear(out_proj_c_in, self.input_dim))
#         self.output_proj = nn.Sequential(*_seq)

#         self.num_blk_internal_registers = num_blk_internal_registers
#         self.blks = nn.ModuleList(
#             [
#                 DiTBlockTimeOnly(
#                     C=self.latent_dim,
#                     num_heads=n_head,
#                     num_registers=self.num_blk_internal_registers,
#                     mlp_hidden_dims=[self.latent_dim * fc_factor] * fc_n_layers,
#                     ada_scale_bias_bound=ada_scale_bias_bound,
#                     gate_bound=gate_bound,
#                     ada_n_layers=ada_n_layers,
#                 )
#                 for i in range(self.n_blocks)
#             ]
#         )

#         self.a_reg = nn.Embedding(num_register, self.latent_dim)
#         self.t_reg = nn.Embedding(num_register, self.latent_dim)
#         self.p_reg = nn.Embedding(num_register, self.latent_dim)

#         self.use_reg_binder = use_reg_binder
#         if self.use_reg_binder:
#             self.reg_binder = nn.ModuleList(
#                 [
#                     nn.Linear(
#                         self.latent_dim * num_register,
#                         self.latent_dim,
#                     )
#                     for i in range(self.n_blocks)
#                 ]
#             )

#         self.__summary__()
#         return

#     def __summary__(self):
#         logging.info(f"LMM Summary:")
#         logging.info(f"  latent_dim: {self.latent_dim}")
#         logging.info(f"  x_dim: {self.input_dim}")
#         logging.info(f"  n_blocks: {self.n_blocks}")
#         logging.info(f"  layer_type_string: {self.layer_type_string}")
#         logging.info(f"  skip_layer_idx: {self.skip_layer_idx}")
#         return

#     def get_optim_groups(self, weight_decay=0.0, base_lr=None):
#         # start with all of the candidate parameters
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         # filter out those that do not require grad
#         param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#         # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#         # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#         optim_groups = [
#             {"params": decay_params, "weight_decay": weight_decay},
#             {"params": nodecay_params, "weight_decay": 0.0},
#         ]
#         # Set base learning rate if provided
#         if base_lr is not None:
#             for group in optim_groups:
#                 group["lr"] = base_lr
#         return optim_groups

#     def forward(
#         self,
#         x,  # B,T,N,D;
#         x_mask,  # B,T,N
#         x_time,  # B,T
#         x_level,  # B,T,N
#         feat,  # B,N,D
#         # context
#         # todo: make this optional
#         ctx=None,  # B,C,N,D
#         ctx_mask=None,  # B,C,N
#         ctx_level=None,  # B,C,N
#         # others
#         add_tokens=None,  # B,A,D
#         add_tokens_mask=None,  # B,A
#         # todo: group attention layer
#     ):
#         # cond: feat + virtual/real_t + level
#         # the time pe is for x_time
#         # ! The time pe is only applied in t layer for now.

#         B, T, N, _ = x.shape
#         assert x_mask.shape == (B, T, N)
#         assert x_time.shape == (B, T)
#         assert x_level.shape == (B, T, N)
#         assert feat.shape == (B, N, self.latent_dim)
#         device = x.device
#         x_time = x_time.float().clone() * self.time_multiplier

#         ctx_pos_emb = self.ctx_t_emb(torch.zeros(B, T, dtype=torch.long, device=device))

#         if ctx is not None:
#             CTX = ctx.shape[1]
#             assert (
#                 CTX < self.n_ctx
#             ), f"CTX: {CTX}, should be < {self.n_ctx} virtual time"
#             assert ctx_mask is not None and ctx_level is not None
#             assert (
#                 CTX <= self.n_ctx
#             ), f"CTX: {CTX}, should be <= {self.n_ctx} virtual time"
#             assert ctx.shape == (B, CTX, N, self.input_dim)
#             assert ctx_mask.shape == (B, CTX, N)
#             assert ctx_level.shape == (B, CTX, N)
#             x = torch.cat([ctx, x], dim=1)
#             x_mask = torch.cat([ctx_mask, x_mask], dim=1)
#             x_level = torch.cat([ctx_level, x_level], dim=1)
#             _ctx_pos_emb = self.ctx_t_emb(torch.arange(CTX, device=device))[
#                 None
#             ].expand(B, -1, -1)
#             ctx_pos_emb = torch.cat([_ctx_pos_emb, ctx_pos_emb], dim=1)
#         else:
#             CTX = 0

#         S = x_level.shape[1]
#         level_emb = (
#             self.int_pe(x_level.reshape(B * S * N))
#             .reshape(B, S, N, self.latent_dim)
#             .to(x)
#         )
#         feat = feat[:, None].expand(-1, S, -1, -1)
#         ctx_pos_emb = ctx_pos_emb[:, :, None].expand(-1, -1, N, -1)
#         cond = torch.cat(
#             [self.hard_ln(feat), self.hard_ln(level_emb), self.hard_ln(ctx_pos_emb)],
#             dim=-1,
#         )  # B,T,N,3lat

#         x_emb = self.x_proj(self.x_pe(x))  # B,T,N,lat

#         ##########################################

#         x = x_emb.reshape(B, S * N, self.latent_dim)
#         cond = cond.reshape(B, S * N, 3 * self.latent_dim)
#         x_time = x_time[:, :, None].expand(-1, -1, N).reshape(B, T * N)

#         ##########################################

#         if add_tokens is not None:
#             A = add_tokens.shape[1]
#             assert add_tokens_mask is not None
#             assert add_tokens.shape == (B, A, self.latent_dim)
#             assert add_tokens_mask.shape == (B, A)
#         else:
#             A = 0
#             add_tokens = torch.zeros(B, A, self.latent_dim, device=device)
#             add_tokens_mask = torch.zeros(B, A, device=device).bool()

#         # * all layers share the same register
#         a_reg = self.a_reg(torch.arange(self.num_register).to(device))  # R,Lat
#         a_reg = a_reg[None].repeat(B, 1, 1)  # B,R,D
#         attn_mask_a = build_attention_mask(
#             torch.cat([add_tokens_mask, x_mask.reshape(B, S * N)], dim=1),
#             mode="full",
#             left_pad=self.num_register + self.num_blk_internal_registers,
#         )[:, None].expand(-1, self.n_head, -1, -1)
#         add_a = add_tokens.clone()  # B,A,D

#         # * t layers
#         t_reg = self.t_reg(torch.arange(self.num_register).to(device))
#         t_reg = t_reg[None].repeat(B * N, 1, 1)
#         attn_mask_t = build_attention_mask(
#             torch.cat(
#                 [
#                     add_tokens_mask[:, None].expand(-1, N, -1).reshape(B * N, A),
#                     x_mask.permute(0, 2, 1).reshape(B * N, S),
#                 ],
#                 dim=1,
#             ),
#             mode="full",
#             left_pad=self.num_register + self.num_blk_internal_registers,
#         )[:, None].expand(-1, self.n_head, -1, -1)
#         add_t = (
#             add_tokens[:, None].expand(-1, N, -1, -1).reshape(B * N, A, self.latent_dim)
#         )

#         # * p layers
#         p_reg = self.p_reg(torch.arange(self.num_register).to(device))
#         p_reg = p_reg[None].repeat(B * S, 1, 1)
#         attn_mask_p = build_attention_mask(
#             torch.cat(
#                 [
#                     add_tokens_mask[:, None].expand(-1, S, -1).reshape(B * S, A),
#                     x_mask.reshape(B * S, N),
#                 ],
#                 dim=1,
#             ),
#             mode="full",
#             left_pad=self.num_register + self.num_blk_internal_registers,
#         )[:, None].expand(-1, self.n_head, -1, -1)
#         add_p = (
#             add_tokens[:, None].expand(-1, S, -1, -1).reshape(B * S, A, self.latent_dim)
#         )

#         skip_list = [x.clone()]
#         for i, blk in enumerate(self.blks):
#             _cond = self.cond_binder[i](cond)

#             if self.layer_type_string[i] == "a":
#                 _x = torch.cat([a_reg, add_a, x], dim=1)
#                 _cond = torch.cat([a_reg, add_a, _cond], dim=1)  # dup the prefix
#                 x = blk(x=_x, cond=_cond, t=None, attn_mask=attn_mask_a)
#                 a_reg = x[:, : self.num_register]
#                 x = x[:, self.num_register + A :]
#                 if self.use_reg_binder:
#                     x = (
#                         x
#                         + self.reg_binder[i](
#                             a_reg.reshape(B, self.num_register * self.latent_dim)
#                         )[:, None]
#                     )
#             elif self.layer_type_string[i] == "t":
#                 _track = x.reshape(B, S, N, self.latent_dim).permute(
#                     0, 2, 1, 3
#                 )  # B,N,T,C
#                 _track = _track.reshape(B * N, S, self.latent_dim)
#                 _track = torch.cat([t_reg, add_t, _track], dim=1)
#                 _cond = _cond.reshape(B, S, N, self.latent_dim).permute(0, 2, 1, 3)
#                 _cond = _cond.reshape(B * N, S, self.latent_dim)
#                 _cond = torch.cat([t_reg, add_t, _cond], dim=1)  # dup the prefix
#                 _t = x_time.reshape(B, T, N).permute(0, 2, 1).reshape(B * N, T)
#                 _track = blk(
#                     _track,
#                     cond=_cond,
#                     t=_t,
#                     num_external_regs=self.num_register + A + CTX,
#                     attn_mask=attn_mask_t,
#                 )
#                 t_reg = _track[:, : self.num_register]
#                 _track = _track[:, self.num_register + A :]
#                 if self.use_reg_binder:
#                     _track = (
#                         _track
#                         + self.reg_binder[i](
#                             t_reg.reshape(B * N, self.num_register * self.latent_dim)
#                         )[:, None]
#                     )
#                 _track = _track.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
#                 x = _track.reshape(B, S * N, self.latent_dim)
#             elif self.layer_type_string[i] == "p":
#                 _track = x.reshape(B, S, N, self.latent_dim)
#                 _track = _track.reshape(B * S, N, self.latent_dim)
#                 _track = torch.cat([p_reg, add_p, _track], dim=1)

#                 _cond = _cond.reshape(B, S, N, self.latent_dim)
#                 _cond = _cond.reshape(B * S, N, self.latent_dim)
#                 _cond = torch.cat([p_reg, add_p, _cond], dim=1)

#                 _track = blk(
#                     _track,
#                     cond=_cond,
#                     t=None,
#                     num_external_regs=None,
#                     attn_mask=attn_mask_p,
#                 )
#                 p_reg = _track[:, : self.num_register]
#                 _track = _track[:, self.num_register + A :]
#                 if self.use_reg_binder:
#                     _track = (
#                         _track
#                         + self.reg_binder[i](
#                             p_reg.reshape(B * S, self.num_register * self.latent_dim)
#                         )[:, None]
#                     )
#                 _track = _track.reshape(B, S, N, self.latent_dim)
#                 x = _track.reshape(B, S * N, self.latent_dim)
#             else:
#                 raise ValueError(f"Invalid layer type: {self.layer_type_string[i]}")

#             # x with more fc!
#             if i in self.skip_layer_idx:
#                 skip_list.append(x.clone())

#         x = torch.cat(skip_list, dim=-1)
#         ret = self.output_proj(x.reshape(-1, x.shape[-1]))
#         ret = ret.reshape(B, S, N, self.input_dim)
#         return ret  # B,S,N,C


# def test_main():
#     io = IdentifyLayerV1(point_encoder_latent=128)
#     io = io.to("cuda")
#     io.train()

#     net = MotionMachineV1()
#     net = net.to("cuda")
#     net.train()

#     B, T, N, C = 4, 50, 35, 9
#     A = 7
#     CTX = 5

#     x = torch.randn(B, T, N, C).to("cuda")
#     x_mask = torch.ones(B, T, N).to("cuda").bool()
#     x_type = torch.randint(0, 7, (B, N)).to("cuda")
#     x_group = torch.randint(0, 8, (B, N)).to("cuda")
#     t = torch.randint(0, 256, (B, T)).to("cuda") / 256.0
#     vt_mask = torch.zeros(B, T).to("cuda").bool()
#     x_level = torch.zeros(B, T, N).to("cuda")
#     pcl = torch.randn(B, N, 10, 3).to("cuda")
#     pcl_valid_mask = torch.ones(B, N).to("cuda").bool()

#     ctx = torch.randn(B, CTX, N, C).to("cuda")
#     ctx_mask = torch.ones(B, CTX, N).to("cuda").bool()
#     ctx_level = torch.randint(0, 7, (B, CTX, N)).to("cuda")

#     feat = io(x_type, x_group)

#     feat = io(x_type, x_group, pcl, pcl_valid_mask)

#     a_tokens = torch.randn(B, A, feat.shape[-1]).to("cuda")
#     a_mask = torch.ones(B, A).to("cuda").bool()

#     ret = net(
#         x=x,
#         x_mask=x_mask,
#         x_time=t,
#         x_level=x_level,
#         feat=feat,
#         # context
#         ctx=ctx,
#         ctx_mask=ctx_mask,
#         ctx_level=ctx_level,
#         #
#         # add_tokens=a_tokens,
#         # add_tokens_mask=a_mask,
#     )

#     loss = F.mse_loss(ret, ret - 1)

#     loss.backward()

#     print(ret.shape)


# if __name__ == "__main__":
#     test_main()
