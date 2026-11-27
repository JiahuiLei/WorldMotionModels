"""Core Agent module for the Universal Motion Machine diffusion model.

The :class:`Agent` wraps the world model (MotionMachine), input encoder
(IdentifyLayer), and pose I/O (PoseIO_V3) into a single ``nn.Module``
that handles diffusion training forward passes, DDIM inference, and
specialized human--humanoid retargeting routines.

Example::

    from umm.models.umm_v1.agent import Agent
    agent = Agent(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import sys, os, os.path as osp
import time
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm
from collections import deque
from copy import deepcopy

# from umm.models.umm_v1.world_models.moma_v1 import MotionMachineV1, IdentifyLayerV1
from umm.models.umm_v1.world_models.moma_v2 import MotionMachineV2, IdentifyLayerV2

# from umm.models.umm_v1.world_models.moma_v21 import MotionMachineV21, IdentifyLayerV21
# from umm.models.umm_v1.world_models.moma_v3 import MotionMachineV3, IdentifyLayerV3
from umm.models.umm_v1.world_models.moma_v4 import MotionMachineV4, IdentifyLayerV4
from umm.models.umm_v1.poseio import PoseIO_V3
from umm.models.umm_v1.act_helper import extract_langtable_action
from umm.models.umm_v1.data_utils import prepare_attributes, pad_time
from umm.models.umm_v1.loss_utils import type_mean
from umm.models.umm_v1.df_helpers import compute_x0, compute_snr

# from umm_dataset_config import *
from umm.datasets.umm_v1.dataset_config import (
    HUMANOID_LINK_GROUP_DICT,
    SEM_HUMAN,
    SEM_LEFT_MANO,
    SEM_RIGHT_MANO,
    SEM_HUMANOID,
    UID_SMPLX55_GROUP,
    UID_HUMANOID_GROUP_DICT,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
    SEM_LANGTABLE_ACT,
    SEM_OBJECT,
)


class Agent(nn.Module):
    """
    Handles the core logic for motion generation, diffusion, and interaction with World Models.
    """

    def __init__(self, cfg):
        """
        Initialize the Agent.

        Args:
            cfg (dict): Configuration dictionary containing model, optim, and data settings.
        """
        super().__init__()

        self.cfg, self.model_cfg = cfg, cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        # * model flags
        self.DEBUG = self.model_cfg.get("DEBUG", False)

        # * dim and input output
        self.latent_dim = self.model_cfg.get("latent_dim", 768)

        # * aug the T
        self.time_mask_aug_prob = self.optim_cfg.get("time_mask_aug_prob", 0.0)
        if self.time_mask_aug_prob > 0.0:
            logging.warning(f"time_mask_aug_prob={self.time_mask_aug_prob}")
        self.ctx_mask_aug_prob = self.optim_cfg.get("ctx_mask_aug_prob", 0.0)
        if self.ctx_mask_aug_prob > 0.0:
            logging.warning(f"ctx_mask_aug_prob={self.ctx_mask_aug_prob}")

        # * text cls prob
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)
        if self.text_masked_prob > 0.0:
            logging.warning(f"text_masked_prob={self.text_masked_prob}")

        # * loss weights
        self.diffusion_w_mode = self.optim_cfg.get("diffusion_w_mode", "default")
        assert self.diffusion_w_mode in ["default", "min(snr,gamma)", "1-alphabar"]
        self.diffusion_w_snr_mode_gamma = self.optim_cfg.get(
            "diffusion_w_snr_mode_gamma", 5.0
        )
        logging.warning(f"diffusion_w_mode={self.diffusion_w_mode}")

        self.w_langtable_act = self.optim_cfg.get("w_langtable_act", -1.0)
        if self.w_langtable_act > 0.0:
            logging.warning(f"w_langtable_act={self.w_langtable_act}")

        self.prob_virtual_allknown = self.optim_cfg.get("prob_virtual_allknown", 0.0)
        if self.prob_virtual_allknown > 0.0:
            logging.warning(f"prob_virtual_allknown={self.prob_virtual_allknown}")
        self.prob_bias_to_known = self.optim_cfg.get("prob_bias_to_known", 0.0)
        if self.prob_bias_to_known > 0.0:
            logging.warning(f"prob_bias_to_known={self.prob_bias_to_known}")

        self.uid_synced_noise_level = self.optim_cfg.get(
            "uid_synced_noise_level", False
        )
        if self.uid_synced_noise_level:
            logging.warning(f"uid_synced_noise_level=True")
        self.group_synced_noise_level = self.optim_cfg.get(
            "group_synced_noise_level", False
        )
        if self.group_synced_noise_level:
            logging.warning(f"group_synced_noise_level=True")

        self.w_t_se3 = self.optim_cfg.get("w_t_se3", 1.0)
        self.w_t_xyz = self.optim_cfg.get("w_t_xyz", 1.0)
        self.w_q = self.optim_cfg.get("w_q", 1.0)
        self.w_R = self.optim_cfg.get("w_R", 0.0)

        self.w_virtual = self.optim_cfg.get("w_virtual", 0.1)
        self.w_real = self.optim_cfg.get("w_real", 1.0)

        self.movement_reweight_min_ratio = self.optim_cfg.get(
            "movement_reweight_min_ratio", -1.0
        )
        if self.movement_reweight_min_ratio > 0.0:
            logging.warning(
                f"movement_reweight_min_ratio={self.movement_reweight_min_ratio}"
            )

        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }
        if len(self.gid_w_modifier) > 0:
            logging.warning(f"gid_w_modifier={self.gid_w_modifier}")

        self.loss_type = self.optim_cfg.get("loss_type", "norm")
        self.loss_square = self.optim_cfg.get("loss_square", True)
        self.loss_stable_beta = self.optim_cfg.get("loss_stable_beta", 0.1)

        self.w_act = self.optim_cfg.get("w_act", 0.0)
        self.w_emp = self.optim_cfg.get("w_emp", 0.0)
        self.emp_types = self.optim_cfg.get("emp_types", [])
        self.act_loss_type = self.optim_cfg.get("act_loss_type", "l2")
        self.act_warmup_iters = self.optim_cfg.get("act_warmup_iters", -1)

        # * param
        self.pose_io_debug_flag = self.model_cfg.get("pose_io_debug_flag", False)
        self.pose_io_type = self.model_cfg.get("pose_io_type", "v3")
        if self.pose_io_type == "v3":
            self.pose_io: PoseIO_V3 = PoseIO_V3(
                sem_group_ids=self.model_cfg.get("param_sem_group_ids", [[-1]]),
                rot_len=self.model_cfg.get("rot_len", 1.0),
                param_mean=self.model_cfg.get("param_mean", []),
                param_std=self.model_cfg.get("param_std", []),
                std_factor=self.model_cfg.get("std_factor", []),
                state_mean_std_flag=True,
                state_only_pos_flag=self.model_cfg.get("state_only_pos_flag", True),
            )
        else:
            raise ValueError(f"Invalid pose_io_type: {self.pose_io_type}")
        self.hand_attached_to_wrist_flag = self.model_cfg.get(
            "hand_attached_to_wrist_flag", False
        )
        if self.hand_attached_to_wrist_flag:
            logging.warning("hand_attached_to_wrist_flag set to True")

        lmm_cfg = self.model_cfg.get("lmm", {})
        input_cfg = self.model_cfg.get("input_layer", {})

        # * init diffusion
        self.__init_diffusion__(self.model_cfg.get("diffusion", {}))

        # * input layers, and the lmm is only the machine for latents
        self.input_variant = input_cfg.get("variant", "v1")
        if self.input_variant == "v1":
            self.input_layer: IdentifyLayerV1 = IdentifyLayerV1(
                latent_dim=self.latent_dim,
                num_types=input_cfg.get("num_types", 512),
                num_groups=input_cfg.get("num_groups", 512),
                group_emb_same_init_flag=input_cfg.get(
                    "group_emb_same_init_flag", False
                ),
                # ! default no pcl
                point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
                point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
                point_version=input_cfg.get("point_version", "v1"),
            )
        elif self.input_variant == "v2":
            self.input_layer: IdentifyLayerV2 = IdentifyLayerV2(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                num_types=input_cfg.get("num_types", 512),
                num_groups=input_cfg.get("num_groups", 512),
                group_emb_same_init_flag=input_cfg.get(
                    "group_emb_same_init_flag", False
                ),
                # ! default no pcl
                point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
                point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
                point_version=input_cfg.get("point_version", "v1"),
            )
        elif self.input_variant == "v21":
            self.input_layer: IdentifyLayerV21 = IdentifyLayerV21(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                num_types=input_cfg.get("num_types", 512),
                num_groups=input_cfg.get("num_groups", 512),
                group_emb_same_init_flag=input_cfg.get(
                    "group_emb_same_init_flag", True
                ),
                point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
                point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
                point_version=input_cfg.get("point_version", "v2"),
            )
        elif self.input_variant == "v3":
            self.input_layer: IdentifyLayerV3 = IdentifyLayerV3(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                num_types=input_cfg.get("num_types", 512),
                num_groups=input_cfg.get("num_groups", 512),
                group_emb_same_init_flag=input_cfg.get(
                    "group_emb_same_init_flag", False
                ),
                # ! default no pcl
                point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
                point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
                point_version=input_cfg.get("point_version", "v2"),
            )
        elif self.input_variant == "v4":
            self.input_layer: IdentifyLayerV4 = IdentifyLayerV4(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                num_types=input_cfg.get("num_types", 512),
                num_groups=input_cfg.get("num_groups", 512),
                group_emb_same_init_flag=input_cfg.get(
                    "group_emb_same_init_flag", True
                ),
                point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
                point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 50.0),
                point_version=input_cfg.get("point_version", "v2"),
                langauge_fixed_length=input_cfg.get("langauge_fixed_length", 16),
            )
        else:
            raise ValueError(f"Invalid input_variant: {self.input_variant}")

        self.lmm_variant = lmm_cfg.get("variant", "v1")
        if self.lmm_variant == "v1":
            self.lmm: MotionMachineV1 = MotionMachineV1(
                latent_dim=self.latent_dim,
                n_head=lmm_cfg.get("n_head", 8),
                input_dim=9,
                output_dim=9,
                layer_type_string=lmm_cfg.get("layer_type_string", "tptptptpt"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", []),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 2),
                num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", False),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
            )
        elif self.lmm_variant == "v2":
            self.lmm: MotionMachineV2 = MotionMachineV2(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                n_head=lmm_cfg.get("n_head", 8),
                input_dim=9,
                output_dim=9,
                cond_binder_dim=lmm_cfg.get("cond_binder_dim", 4096),
                layer_type_string=lmm_cfg.get("layer_type_string", "tptptptpt"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", []),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 2),
                num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", False),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
                add_first_n_add_token_to_feat=lmm_cfg.get(
                    "add_first_n_add_token_to_feat", -1
                ),
                time_rope_flag=lmm_cfg.get("time_rope_flag", True),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", False),
                #
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
            )
        elif self.lmm_variant == "v21":
            self.lmm: MotionMachineV21 = MotionMachineV21(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                n_head=lmm_cfg.get("n_head", 8),
                input_dim=9,
                output_dim=9,
                cond_binder_dim=lmm_cfg.get("cond_binder_dim", 4096),
                layer_type_string=lmm_cfg.get("layer_type_string", "tptptptpt"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", []),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 2),
                num_blk_internal_registers=getattr(
                    lmm_cfg, "num_blk_internal_registers", 4
                ),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", False),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
                add_first_n_add_token_to_feat=getattr(
                    lmm_cfg, "add_first_n_add_token_to_feat", -1
                ),
                time_rope_flag=lmm_cfg.get("time_rope_flag", True),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", False),
                #
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
            )
        elif self.lmm_variant == "v3":
            self.lmm: MotionMachineV3 = MotionMachineV3(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                n_head=lmm_cfg.get("n_head", 8),
                input_dim=9,
                output_dim=9,
                cond_binder_dim=lmm_cfg.get("cond_binder_dim", 4096),
                layer_type_string=lmm_cfg.get("layer_type_string", "tptptptpt"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", []),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 2),
                num_blk_internal_registers=getattr(
                    lmm_cfg, "num_blk_internal_registers", 4
                ),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", False),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
                add_first_n_add_token_to_feat=getattr(
                    lmm_cfg, "add_first_n_add_token_to_feat", -1
                ),
                time_rope_flag=lmm_cfg.get("time_rope_flag", True),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", False),
                #
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
            )
        elif self.lmm_variant == "v4":
            self.lmm: MotionMachineV4 = MotionMachineV4(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 2048),
                n_head=lmm_cfg.get("n_head", 8),
                input_dim=9,
                output_dim=9,
                cond_binder_dim=lmm_cfg.get("cond_binder_dim", 4096),
                layer_type_string=lmm_cfg.get("layer_type_string", "aaaaaa"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", []),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 2),
                num_blk_internal_registers=getattr(
                    lmm_cfg, "num_blk_internal_registers", 4
                ),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", False),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
                add_first_n_add_token_to_feat=getattr(
                    lmm_cfg, "add_first_n_add_token_to_feat", -1
                ),
                time_rope_flag=lmm_cfg.get("time_rope_flag", True),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", False),
                #
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
            )
        else:
            raise ValueError(f"Invalid lmm_version: {self.lmm_variant}")

        # buffer record for easier inference!
        ctx_record_cfg = self.model_cfg.get("ctx_record", {})
        self.ctx_record_flag = len(ctx_record_cfg) > 0
        if self.ctx_record_flag:
            self.ctx_record_buffer_size = ctx_record_cfg.get("buffer_size", 16384)
            self.ctx_record_buffer = {}  # the key is the list of the uid

        self.hh_boosted = False
        return

    def loss_kernel(self, pred, gt):
        """Compute element-wise loss between prediction and ground truth.

        Supports ``"norm"`` (L2 norm, optionally squared) and ``"stable"``
        (Huber-like smooth L1/L2 transition controlled by ``loss_stable_beta``).

        Args:
            pred: Predicted tensor of shape ``[..., D]``.
            gt: Ground-truth tensor of shape ``[..., D]``.

        Returns:
            Per-element scalar loss of shape ``[...]``.

        Example::

            loss = agent.loss_kernel(pred_params, gt_params)
        """
        # ! dim is always the last
        if self.loss_type == "norm":
            logging.debug(f"WARNING, THIS IS DANGEROUS! AND DEPRECATED!")
            diff = (pred - gt).norm(dim=-1)
            if self.loss_square:
                diff = diff**2
            return diff
        elif self.loss_type == "stable":
            residual_norm = (pred - gt).norm(dim=-1)
            mse = (pred - gt).pow(2).sum(dim=-1)
            beta = self.loss_stable_beta  # transition point, tune this
            loss = torch.where(
                residual_norm > beta,
                residual_norm - 0.5 * beta,  # norm-like for large errors
                0.5 * mse / beta,  # MSE-like for small errors
            )
            return loss
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

    @torch.no_grad()
    def summarize_ctx_record(self):
        """Print a summary of the context record buffer contents.

        Logs the first three characters of each key and the number of
        stored context frames. No-op if ``ctx_record_flag`` is False.

        Example::

            agent.summarize_ctx_record()
        """
        if not self.ctx_record_flag:
            return
        for k, v in self.ctx_record_buffer.items():
            print(f"ctx_record_buffer: {k[:3]}...: {len(v)}")
        return

    @torch.no_grad()
    def record_ctx(self, ctx, ctx_mask, uid):
        """Store context frames into the per-UID record buffer.

        For each batch element, compresses the valid context by the
        active mask, sorts by UID, and appends to a bounded deque keyed
        by the sorted UID tuple. No-op if ``ctx_record_flag`` is False.

        Args:
            ctx: Context SE3 transforms of shape ``[B, C, P, 4, 4]``.
            ctx_mask: Validity mask of shape ``[B, C, P]``.
            uid: Unique part IDs of shape ``[B, P]``.

        Example::

            agent.record_ctx(context, context_mask, batch["trace_uid"])
        """
        if not self.ctx_record_flag:
            return
        start_t = time.time()
        ctx = ctx.clone()
        ctx_mask = ctx_mask.clone()
        uid = uid.clone()
        # ctx: B,C,P,4,4
        # ctx_mask: B,C,P
        # uid: B,P
        B = len(ctx)
        for bid in range(B):
            c = ctx[bid]
            m = ctx_mask[bid]
            u = uid[bid]  # P
            m_p = m.any(dim=0)  # P
            m_c = m.any(dim=1)  # C
            if m_p.sum() == 0 or m_c.sum() == 0:
                continue
            c = c[m_c][:, m_p]
            u = u[m_p]
            re_index = u.argsort()
            c = c[:, re_index].cpu()
            u = u[re_index].cpu().numpy().tolist()
            u = tuple(u)
            if u not in self.ctx_record_buffer:
                self.ctx_record_buffer[u] = deque(maxlen=self.ctx_record_buffer_size)
            self.ctx_record_buffer[u].extend(c.split(1, dim=0))
        return

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return the state dict, including ``ctx_record_buffer`` if enabled.

        Augments the standard ``nn.Module.state_dict`` with the context
        record buffer stored as a nested dict of tensors (pickle-safe).

        Args:
            destination: Optional pre-allocated ordered dict.
            prefix: Key prefix for all state dict entries.
            keep_vars: If True, return Variables instead of Tensors.

        Returns:
            State dictionary including ``ctx_record_buffer`` when applicable.

        Example::

            sd = agent.state_dict()
        """
        destination = super().state_dict(destination, prefix, keep_vars)
        # Save CPU dict as a nested dict of tensors (pickle-safe in torch.save)
        if self.ctx_record_flag:
            logging.info(f"Assign ctx to statedict: {prefix}ctx_record_buffer")
            destination[prefix + "ctx_record_buffer"] = self.ctx_record_buffer
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load state dict, extracting ``ctx_record_buffer`` before standard loading.

        Pops the ``ctx_record_buffer`` key from the state dict to avoid
        treating it as a parameter, then delegates to the parent loader.

        Args:
            state_dict: State dictionary from checkpoint.
            prefix: Key prefix for this module.
            *args: Additional positional arguments for parent.
            **kwargs: Additional keyword arguments for parent.

        Example::

            agent._load_from_state_dict(sd, prefix="")
        """
        key = prefix + "ctx_record_buffer"
        if self.ctx_record_flag and key in state_dict:
            self.ctx_record_buffer = state_dict.pop(key)
        elif key in state_dict:
            # Pop the key even if ctx_record_flag is False to avoid loading it as a parameter
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return

    def __init_diffusion__(self, cfg):
        """Initialize DDPM training and DDIM generation schedulers.

        Sets up the noise scheduler for training and the generation
        scheduler for inference with configurable timesteps, beta
        schedule, and prediction type.

        Args:
            cfg: Diffusion config dict with keys like ``train_diff_timesteps``,
                ``beta_schedule``, ``prediction_type``, ``gen_timesteps``.

        Example::

            agent.__init_diffusion__(cfg["model"]["diffusion"])
        """
        self.pred_type = cfg.get("prediction_type", "epsilon")
        logging.info(f"Prediction type: {self.pred_type}")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.get("train_diff_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "linear"),
            prediction_type=self.pred_type,
            # rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
            timestep_spacing=cfg.get("timestep_spacing", "trailing"),
            clip_sample=False,
        )
        self.gen_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.get("train_diff_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "linear"),
            prediction_type=self.pred_type,
            # rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
            timestep_spacing=cfg.get("timestep_spacing", "trailing"),
            clip_sample=False,
        )
        self.gen_scheduler.set_timesteps(cfg.get("gen_timesteps", 10))
        logging.info(
            f"Setup Diffusion Schedulers:{self.noise_scheduler}, {self.gen_scheduler}"
        )
        self.explicit_mode_conversion_flag = cfg.get(
            "explicit_mode_conversion_flag", False
        )
        if self.explicit_mode_conversion_flag:
            logging.warning(
                "Explicit mode conversion flag is enabled, assume the network output is always x0"
            )
        return

    @staticmethod
    def filter_ckpt_prefix(checkpoint_dict, prefix):
        """Filter and strip a key prefix from a checkpoint state dict.

        Args:
            checkpoint_dict: Full state dict from a checkpoint.
            prefix: Prefix to match and strip (e.g. ``"model."``).

        Returns:
            New dict containing only keys that started with ``prefix``,
            with the prefix stripped.

        Example::

            lmm_dict = Agent.filter_ckpt_prefix(ckpt, "model.lmm.")
        """
        return {
            k[len(prefix) :]: v
            for k, v in checkpoint_dict.items()
            if k.startswith(prefix)
        }

    def load_from_checkpoint(
        self, checkpoint_dict, lmm_flag=True, input_flag=True, strict=True
    ):
        """Load LMM and/or input layer weights from a filtered checkpoint dict.

        Args:
            checkpoint_dict: State dict already filtered to Agent-level keys
                (i.e. without the ``"model."`` prefix).
            lmm_flag: Whether to load the world model (LMM) weights.
            input_flag: Whether to load the input layer weights.
            strict: Whether to enforce strict key matching.

        Example::

            agent.load_from_checkpoint(ckpt_dict, lmm_flag=True, strict=False)
        """
        if lmm_flag:
            lmm_dict = self.filter_ckpt_prefix(checkpoint_dict, "lmm.")
            self.lmm.load_state_dict(lmm_dict, strict=strict)
        if input_flag:
            input_dict = self.filter_ckpt_prefix(checkpoint_dict, "input_layer.")
            self.input_layer.load_state_dict(input_dict, strict=strict)
        return

    @staticmethod
    @torch.no_grad()
    def get_type_mask(query_type, mark_type):
        """Create a boolean mask indicating which elements match any of the given types.

        Args:
            query_type: Tensor of type IDs to check, arbitrary shape.
            mark_type: List or tensor of type IDs to match against.

        Returns:
            Boolean mask with the same shape as ``query_type``.

        Example::

            mask = Agent.get_type_mask(sem_type, [SEM_HUMAN, SEM_HUMANOID])
        """
        query_shape = query_type.shape
        query_type = query_type.reshape(-1)
        mark_type = torch.as_tensor(mark_type).to(query_type)  # M
        ret = (query_type[..., None] == mark_type[None]).any(dim=-1).bool()
        return ret.reshape(query_shape)

    def x0_to_mode(self, x0, noisy_x, df_timesteps):
        """Convert x0 prediction back to the model's native prediction mode.

        This is the inverse of compute_x0 in df_helpers.py. It handles
        asynchronized diffusion forcing timesteps (varying per element) by
        vectorizing the computation over all elements in parallel.

        Args:
            x0: Predicted original samples, shape [..., channels]
            noisy_x: Noisy samples x_t, shape [..., channels]
            df_timesteps: Diffusion forcing timesteps, shape [...]
        """
        if not self.explicit_mode_conversion_flag:
            return x0
        elif self.pred_type == "epsilon":
            # epsilon = (x_t - sqrt(alpha_t) * x0) / sqrt(1 - alpha_t)
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x0.device)
            alpha_t = alphas_cumprod[df_timesteps].clamp(min=1e-8, max=1.0 - 1e-8)[
                ..., None
            ]
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
            return (noisy_x - sqrt_alpha * x0) / sqrt_one_minus_alpha
        elif self.pred_type == "sample":
            return x0
        elif self.pred_type == "v_prediction":
            # v = (sqrt(alpha_t) * x_t - x0) / sqrt(1 - alpha_t)
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x0.device)
            alpha_t = alphas_cumprod[df_timesteps].clamp(min=1e-8, max=1.0 - 1e-8)[
                ..., None
            ]
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
            return (sqrt_alpha * noisy_x - x0) / sqrt_one_minus_alpha
        else:
            raise ValueError(f"Unsupported prediction type: {self.pred_type}")

    @torch.no_grad()
    def __prepare_df_step__(self, B, C, T, P, device, uids, gids, enforce_known_mask):
        """Sample per-element diffusion timesteps for training.

        Supports three noise-level synchronization modes: per-UID, per-group,
        or fully independent. Optionally biases some timesteps to 0
        (known/clean) based on ``prob_virtual_allknown`` and
        ``prob_bias_to_known`` settings.

        Args:
            B: Batch size.
            C: Number of context frames.
            T: Number of trace (prediction) frames.
            P: Number of parts.
            device: Torch device.
            uids: Unique part IDs of shape ``[B, P]``.
            gids: Group IDs of shape ``[B, P]``.
            enforce_known_mask: Boolean mask of shape ``[B, P]`` forcing
                certain parts to timestep 0, or None.

        Returns:
            Diffusion timesteps of shape ``[B, C+T, P]``.

        Example::

            df_t = agent.__prepare_df_step__(B, C, T, P, device, uids, gids, None)
        """
        if self.uid_synced_noise_level:
            # ! assume the group UIDs will span 0-U-1
            U = int(uids.max() + 1)  # ！ here just take the max
            df_step_buffer = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B, C + T, U),
                device=device,
            ).long()  # B,C+T,U
            # gather the step
            index = uids[:, None].expand(-1, C + T, -1)
            df_timesteps = df_step_buffer.gather(dim=-1, index=index.long())
        elif self.group_synced_noise_level:
            G = int(gids.max() + 1)  # ！ here just take the max
            df_step_buffer = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B, C + T, G),
                device=device,
            ).long()  # B,C+T,G
            # gather the step
            index = gids[:, None].expand(-1, C + T, -1)
            df_timesteps = df_step_buffer.gather(dim=-1, index=index.long())
        else:
            df_timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B, C + T, P),
                device=device,
            ).long()  # B,C+T,P
        if self.prob_virtual_allknown > 0.0:
            if torch.rand(1) < self.prob_virtual_allknown:
                df_timesteps[:, :C] = 0  # all virtual time steps are 0
        if self.prob_bias_to_known > 0.0:
            if torch.rand(1) < self.prob_bias_to_known:
                th = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (1,)
                ).item()
                bias_mask = df_timesteps <= th
                df_timesteps[bias_mask] = 0
        if enforce_known_mask is not None:
            df_timesteps = df_timesteps.permute(0, 2, 1)  # B,N,C+T
            expanded_mask = enforce_known_mask.unsqueeze(-1).expand(-1, -1, T)  # B,N,T
            df_timesteps[:, :, C:][expanded_mask] = 0
            assert (df_timesteps[:, :, C:][expanded_mask] == 0).all()
            df_timesteps = df_timesteps.permute(0, 2, 1)  # B,C+T,N
        return df_timesteps

    def handle_play_policy_data(self, batch):
        """Handle exploration/policy data by enforcing known masks.

        When ``is_explore`` is set in the batch extras, randomly enforces
        either all action parts or all object parts as known (timestep 0)
        and clears the text prompt for exploration samples.

        Args:
            batch: Batch data dictionary containing ``extra``, ``text``,
                ``trace_sem_type``, and ``trace_mask``.

        Returns:
            Tuple of ``(batch, enforce_known_mask)`` where the mask is
            of shape ``[B, N]``.

        Example::

            batch, enforce_mask = agent.handle_play_policy_data(batch)
        """
        if "is_explore" not in batch["extra"]:
            enforce_known_mask = torch.zeros_like(
                batch["trace_sem_type"], dtype=torch.bool
            )  # B,N
            return batch, enforce_known_mask
        # * enforce either all actions or all objects to be known, if is_explore is set to true, do special enforcement here
        # must set explore to true not is policy to true because there maybe other datasets in the same batch in the future.
        explore_mask = torch.as_tensor(batch["extra"]["is_explore"]).to(
            batch["trace_mask"].device
        )  # B
        for i in range(len(explore_mask)):
            if explore_mask[i]:
                batch["text"][i] = ""
        enforce_action_mask = (
            batch["trace_sem_type"] == SEM_LANGTABLE_ACT
        )  # B,N # ! all act known
        B, N = enforce_action_mask.shape
        enforce_object_mask = (
            batch["trace_sem_type"] == SEM_OBJECT
        )  # B,N # ! all objects known
        candidate = torch.stack(
            [enforce_action_mask, enforce_object_mask], dim=-1
        )  # B,N,2
        choice = torch.randint(0, 2, (B,), device=batch["trace_mask"].device)  # B
        enforce_known_mask = torch.gather(
            candidate, dim=-1, index=choice[:, None, None].expand(-1, N, -1)
        )
        enforce_known_mask = enforce_known_mask.squeeze(-1)  # B,N
        enforce_known_mask = (
            enforce_known_mask & explore_mask[:, None]
        )  # ! enforce known only when is not policy data
        return batch, enforce_known_mask

    def forward(self, batch, phase, global_step=0):
        """
        Forward pass for training and validation.

        Args:
            batch (dict): Batch data containing trace, context, masks, and metadata.
            phase (str): 'train' or 'val'.
            global_step (int): Current global step.

        Returns:
            dict: Dictionary containing loss, log_dict, and predicted values.
        """

        # ! todo: the text is not used for now
        ret = {}

        # * preprocess the batch
        batch, enforce_known_mask = self.handle_play_policy_data(batch)

        # * prepare input
        B, T, P = batch["trace_mask"].shape
        C = batch["context_mask"].shape[1]
        device = batch["trace_mask"].device

        timestep = batch["time"]
        trace = batch["trace"].clone()  # B,T,P,4,4
        context = batch["context"].clone()  # B,C,P,4,4
        trace_mask = batch["trace_mask"].clone()  # B,T,P
        context_mask = batch["context_mask"].clone()  # B,C,P

        sem_type = batch["trace_sem_type"].clone()  # B,P
        group_id = batch["trace_group_id"].clone()  # B,P
        rot_mask = batch["trace_rot_mask"].clone()  # B,P

        local_pcl = batch["local_pcl"].clone()  # B,P,PCL,3
        # ! for now do not
        local_pcl_mask = batch["local_pcl_mask"].clone()  # B,P

        if self.training and self.ctx_record_flag:
            self.record_ctx(context, context_mask, batch["trace_uid"])
            # np.savez("./debug.npz", context=context.cpu().numpy(), context_mask=context_mask.cpu().numpy(), uid=batch["trace_uid"].cpu().numpy())

        # * randomly aug the trace_mask and ctx_mask
        if self.time_mask_aug_prob > 0.0:
            if torch.rand(1) < self.time_mask_aug_prob:
                known_t = torch.randint(0, T, (1,)).item()
                trace_mask[:, known_t:, :] = False
        if self.ctx_mask_aug_prob > 0.0:
            if torch.rand(1) < self.ctx_mask_aug_prob:
                known_c = torch.randint(0, C, (1,)).item()
                context_mask[:, known_c:, :] = False

        trace_param = self.pose_io.encode(
            trace, sem_ids=sem_type[:, None].expand(-1, T, -1), stat_mask=trace_mask
        )  # B,T,P,9
        context_param = self.pose_io.encode(
            context, sem_ids=sem_type[:, None].expand(-1, C, -1), stat_mask=context_mask
        )  # B,C,P,9

        diffusion_x = torch.cat([context_param, trace_param], dim=1).clone()  # B,CT,P,7
        noise = torch.randn_like(diffusion_x, device=device)  #

        df_timesteps = self.__prepare_df_step__(
            B,
            C,
            T,
            P,
            device,
            batch["trace_uid"],
            batch["trace_group_id"],
            enforce_known_mask,
        )
        noisy_x = self.noise_scheduler.add_noise(
            original_samples=diffusion_x.reshape(B * (C + T) * P, -1),
            noise=noise.reshape(B * (C + T) * P, -1),
            timesteps=df_timesteps.reshape(B * (C + T) * P),
        ).reshape(B, C + T, P, -1)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            diffusion_tgt = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            diffusion_tgt = self.noise_scheduler.get_velocity(
                diffusion_x.reshape(B * (C + T) * P, -1),
                noise.reshape(B * (C + T) * P, -1),
                df_timesteps.reshape(B * (C + T) * P),
            ).reshape(B, C + T, P, -1)
        elif self.noise_scheduler.config.prediction_type == "sample":
            diffusion_tgt = diffusion_x  # cursor writes this, need to check
        else:
            raise ValueError(
                f"Prediction type {self.noise_scheduler.config.prediction_type} not supported"
            )

        lmm_input = noisy_x  # B,C+T,P,9
        lmm_mask = torch.cat([context_mask, trace_mask], dim=1)  # B,C+T,P

        # mask the rot components
        rot9_mask = self.pose_io.rot_mask(rot_mask)  # B,P,9
        lmm_input = lmm_input * rot9_mask[:, None].float()

        # * cls free mask on text
        text_masked = torch.rand(B) < self.text_masked_prob
        text = [batch["text"][i] if not text_masked[i] else "" for i in range(B)]
        lang_tk = self.input_layer.encode_language(text)
        feat = self.input_layer(sem_type, group_id, local_pcl, local_pcl_mask)

        # logging.debug(
        #     f"x_range: {lmm_input[:, C:].min()}, {lmm_input[:, C:].max()}, {lmm_input[:, C:].mean()}, {lmm_input[:, C:].std()}"
        # )
        # logging.debug(
        #     f"ctx_range: {lmm_input[:, :C].min()}, {lmm_input[:, :C].max()}, {lmm_input[:, :C].mean()}, {lmm_input[:, :C].std()}"
        # )
        # logging.debug(
        #     f"diffusion_tgt: {diffusion_tgt.min()}, {diffusion_tgt.max()}, {diffusion_tgt.mean()}, {diffusion_tgt.std()}"
        # )
        # logging.debug(
        #     f"trace_param: {trace_param.min()}, {trace_param.max()}, {trace_param.mean()}, {trace_param.std()}"
        # )
        # logging.debug(
        #     f"context_param: {context_param.min()}, {context_param.max()}, {context_param.mean()}, {context_param.std()}"
        # )

        out = self.lmm(
            x=lmm_input[:, C:],
            x_mask=lmm_mask[:, C:],
            x_time=timestep,
            x_level=df_timesteps[:, C:],
            feat=feat,
            #
            ctx=lmm_input[:, :C],
            ctx_mask=lmm_mask[:, :C],
            ctx_level=df_timesteps[:, :C],
            #
            add_tokens=lang_tk,
            add_tokens_mask=torch.ones(*lang_tk.shape[:2]).to(device).bool(),
        )  # B,T,P,C
        out = self.x0_to_mode(out, lmm_input, df_timesteps)

        # * supervise the output
        log_dict = {}

        # * supervise the config
        loss = (out * 0.0).mean()

        # * supervise the primary output
        # Efficiently compute x0 from the current prediction using batched operations
        x0_pred = compute_x0(
            scheduler=self.noise_scheduler,
            pred=out,  # B,T,P,7
            time_steps=df_timesteps,  # B,T,P
            noisy_x=noisy_x,  # B,T,P,7
        )
        T_wl_pred = self.pose_io.decode(
            x0_pred, sem_ids=sem_type[:, None].expand(-1, C + T, -1)
        ).detach()

        if self.w_langtable_act > 0.0:
            act_pred, act_pred_mask = extract_langtable_action(
                T_wl_pred[:, C:].float(), batch["trace_uid"], batch["trace_mask"]
            )
            act_gt, act_gt_mask = extract_langtable_action(
                batch["trace"].float(), batch["trace_uid"], batch["trace_mask"]
            )
            act_diff = (act_pred - act_gt).norm(dim=-1)
            act_loss = act_diff[act_pred_mask & act_gt_mask].mean()
            loss = loss + self.w_langtable_act * act_loss
            log_dict[f"{phase}_LOSS/langtable_act"] = act_loss.detach()

        tgt = diffusion_tgt.clone().detach()

        tgt_t = tgt[..., :3]  # B,T,P,3
        tgt_q = tgt[..., 3:]  # B,T,P,4
        pred_t = out[..., :3]
        pred_q = out[..., 3:]

        t_sup_mask = lmm_mask.bool()  # B,T,P
        q_sup_mask = lmm_mask.bool() * rot9_mask[:, None, :, -1].bool()
        v_mask = torch.zeros(B, C + T, P).to(trace_mask).bool()
        v_mask[:, :C] = True  # if true, is a virtual time

        t_r_sup_mask_se3 = t_sup_mask & ~v_mask & rot9_mask[:, None, :, -1].bool()
        t_r_sup_mask_xyz = t_sup_mask & ~v_mask & ~rot9_mask[:, None, :, -1].bool()
        t_v_sup_mask_se3 = t_sup_mask & v_mask & rot9_mask[:, None, :, -1].bool()
        t_v_sup_mask_xyz = t_sup_mask & v_mask & ~rot9_mask[:, None, :, -1].bool()

        q_r_sup_mask = q_sup_mask & ~v_mask
        q_v_sup_mask = q_sup_mask & v_mask

        # the posed t are more important than others
        if self.diffusion_w_mode == "default":
            loss_ele_weight = torch.ones_like(df_timesteps).to(pred_t)
        elif self.diffusion_w_mode == "min(snr,gamma)":
            snr = compute_snr(df_timesteps, self.noise_scheduler)
            loss_ele_weight = torch.minimum(
                snr, torch.tensor(self.diffusion_w_snr_mode_gamma).to(pred_t)
            )
        elif self.diffusion_w_mode == "1-alphabar":
            loss_ele_weight = 1.0 - self.noise_scheduler.alphas_cumprod[df_timesteps]
        else:
            raise ValueError(f"diffusion_w_mode {self.diffusion_w_mode} not supported")

        if self.movement_reweight_min_ratio > 0:
            movement = batch["trace"][..., :3, 3]
            movement = (movement - movement[:, :1]).norm(dim=-1) * batch["trace_mask"]
            movement = movement.max(dim=1).values  # B,N
            max_movement = movement.max(dim=1).values  # B
            movement = (
                movement + max_movement[:, None] * self.movement_reweight_min_ratio
            )
            movement_weight = P * movement / movement.sum(dim=1, keepdim=True)
            loss_ele_weight = loss_ele_weight * movement_weight[:, None]

        loss_t = self.loss_kernel(pred_t, tgt_t)
        loss_q = self.loss_kernel(pred_q, tgt_q)

        loss_t = loss_t * loss_ele_weight
        loss_q = loss_q * loss_ele_weight

        T_wl_gt = torch.cat([context, trace], dim=1)
        R_wl_pred = T_wl_pred[..., :3, :3]
        R_wl_gt = T_wl_gt[..., :3, :3]
        R_loss = torch.einsum("...ji,...jk->...ik", R_wl_pred, R_wl_gt)
        R_loss = 1.0 - (R_loss[..., 0, 0] + R_loss[..., 1, 1] + R_loss[..., 2, 2]) / 3.0

        # pad_sem_type = sem_type[:, None].expand_as(t_r_sup_mask)
        # * real
        pad_uids = batch["trace_uid"][:, None].expand_as(t_r_sup_mask_se3).long()
        if t_r_sup_mask_se3.sum() > 0:
            loss_t_r_se3 = type_mean(
                loss_t[t_r_sup_mask_se3],
                pad_uids[t_r_sup_mask_se3],
                self.gid_w_modifier,
            )
            loss = loss + self.w_t_se3 * self.w_real * loss_t_r_se3
            log_dict[f"{phase}_LOSS/t_real_se3"] = loss_t_r_se3.mean().detach()
        if t_r_sup_mask_xyz.sum() > 0:
            loss_t_r_xyz = type_mean(
                loss_t[t_r_sup_mask_xyz],
                pad_uids[t_r_sup_mask_xyz],
                self.gid_w_modifier,
            )
            loss = loss + self.w_t_xyz * self.w_real * loss_t_r_xyz
            log_dict[f"{phase}_LOSS/t_real_xyz"] = loss_t_r_xyz.mean().detach()
        if q_r_sup_mask.sum() > 0:
            loss_q_r = type_mean(
                loss_q[q_r_sup_mask], pad_uids[q_r_sup_mask], self.gid_w_modifier
            )
            loss = loss + self.w_q * self.w_real * loss_q_r
            log_dict[f"{phase}_LOSS/q_real"] = loss_q_r.mean().detach()
            loss_R_r = type_mean(
                R_loss[q_r_sup_mask], pad_uids[q_r_sup_mask], self.gid_w_modifier
            )
            loss = loss + self.w_R * self.w_real * loss_R_r
            log_dict[f"{phase}_LOSS/R_real"] = loss_R_r.mean().detach()

        # * virtual
        pad_uids = batch["trace_uid"][:, None].expand_as(t_v_sup_mask_se3).long()
        if t_v_sup_mask_se3.sum() > 0:
            loss_t_v_se3 = type_mean(
                loss_t[t_v_sup_mask_se3],
                pad_uids[t_v_sup_mask_se3],
                self.gid_w_modifier,
            )
            loss = loss + self.w_t_se3 * self.w_virtual * loss_t_v_se3
            log_dict[f"{phase}_LOSS/t_virtual_se3"] = loss_t_v_se3.mean().detach()
        if t_v_sup_mask_xyz.sum() > 0:
            loss_t_v_xyz = type_mean(
                loss_t[t_v_sup_mask_xyz],
                pad_uids[t_v_sup_mask_xyz],
                self.gid_w_modifier,
            )
            loss = loss + self.w_t_xyz * self.w_virtual * loss_t_v_xyz
            log_dict[f"{phase}_LOSS/t_virtual_xyz"] = loss_t_v_xyz.mean().detach()
        if q_v_sup_mask.sum() > 0:
            loss_q_v = type_mean(
                loss_q[q_v_sup_mask], pad_uids[q_v_sup_mask], self.gid_w_modifier
            )
            loss = loss + self.w_q * self.w_virtual * loss_q_v
            log_dict[f"{phase}_LOSS/q_virtual"] = loss_q_v.mean().detach()
            loss_R_v = type_mean(
                R_loss[q_v_sup_mask], pad_uids[q_v_sup_mask], self.gid_w_modifier
            )
            loss = loss + self.w_R * self.w_virtual * loss_R_v
            log_dict[f"{phase}_LOSS/R_virtual"] = loss_R_v.mean().detach()
        log_dict[f"{phase}_LOSS/total"] = loss.detach()

        log_dict[f"total_loss/{phase}"] = loss.detach()
        ret["loss"] = loss
        ret["log_dict"] = log_dict

        # for viz
        ret["x0_pred_context"] = T_wl_pred[:, :C].detach()
        ret["x0_pred_trace"] = T_wl_pred[:, C:].detach()
        if self.pose_io_debug_flag:
            self.pose_io.print_stats()

        # add the poseio state to the dict
        log_dict[f"{phase}_STATS/pose_io_ave_mean"] = (
            self.pose_io.running_mean.detach().mean()
        )
        log_dict[f"{phase}_STATS/pose_io_ave_std"] = (
            self.pose_io.running_var.detach().sqrt().mean()
        )

        return ret

    def convert_smplx_hand_param(self, Q, to_local=True):
        """Convert SMPL-X hand joint transforms between local and global space.

        Transforms the 15 left-hand and 15 right-hand joints relative to
        (or back from) their respective wrist frames. Assumes the standard
        SMPL-X 55-joint ordering.

        Args:
            Q: Joint transforms of shape ``[B, 55, 4, 4]``.
            to_local: If True, convert global hand joints to wrist-local
                space. If False, convert from local back to global.

        Returns:
            Modified transforms of shape ``[B, 55, 4, 4]``.

        Example::

            Q_local = agent.convert_smplx_hand_param(Q, to_local=True)
        """
        # Q: B,55,4,4; uids: 55
        # ! assume ordered
        assert Q.shape[1] == 55
        # assert uids.shape[0] == 55
        # sort_idx = torch.argsort(uids)
        Q = Q.clone()
        # Q_sorted = Q[:, sort_idx, :, :]

        with torch.autocast(device_type=Q.device.type, enabled=False):
            # ! force float32 for matrix inversion stability
            Q = Q.float()

            l_wrist = Q[:, 20].clone()
            r_wrist = Q[:, 21].clone()
            if to_local:
                l_T = torch.inverse(l_wrist)
                r_T = torch.inverse(r_wrist)
            else:
                l_T = l_wrist
                r_T = r_wrist
            l_hand = Q[:, 25:40].clone()
            r_hand = Q[:, 40:55].clone()
            l_hand_local = torch.einsum("bij,bnjk->bnik", l_T, l_hand)
            r_hand_local = torch.einsum("bij,bnjk->bnik", r_T, r_hand)

            Q[:, 25:40] = l_hand_local.clone()
            Q[:, 40:55] = r_hand_local.clone()

        # re_sort_idx = torch.argsort(sort_idx)
        # Q = Q[:, re_sort_idx, :, :]
        return Q

    @torch.inference_mode()
    def inference(
        self,
        # X
        trace,  # B,T,P,4,4
        context,  # B,C,P,4,4
        # WHAT TO Generate
        trace_known_mask,  # B,T,P
        context_known_mask,  # B,C,P
        trace_mask,  # B,T,P
        context_mask,  # B,C,P
        # ATTR
        real_timestep,  # B,T
        group_id,  # B,P
        sem_type,  # B,P # todo: later this can be from pixels
        rot_mask,  # B,P
        # DIFFUSION
        condition_timestep: int = 0,
        condition_ctx_timestep: int = -1,
        condition_trace_timestep: int = -1,
        # cfg=-1,
        # TODO: text, rgb, AR inference, cfg version
        seed=42,
        verbose=True,
        #
        local_pcl=None,  # B,P,PCL,3
        local_pcl_mask=None,  # B,P
        text=None,
        #
        add_cfg: float = -1.0,
        #
        left2right_AR_sweep_flag=False,
    ):
        """Run DDIM inference without gradient tracking (convenience wrapper).

        Delegates to :meth:`inference_with_grad` inside a ``torch.no_grad``
        context. See that method for full argument documentation.

        Returns:
            Tuple of ``(ret_context, ret_trace)`` SE3 transforms,
            each of shape ``[B, C/T, P, 4, 4]``.

        Example::

            ctx_pred, trace_pred = agent.inference(
                trace=trace, context=ctx, trace_known_mask=t_known,
                context_known_mask=c_known, trace_mask=t_mask,
                context_mask=c_mask, real_timestep=timestep,
                group_id=gids, sem_type=sem, rot_mask=rot,
            )
        """
        with torch.no_grad():
            return self.inference_with_grad(
                trace=trace,
                context=context,
                trace_known_mask=trace_known_mask,
                context_known_mask=context_known_mask,
                trace_mask=trace_mask,
                context_mask=context_mask,
                real_timestep=real_timestep,
                group_id=group_id,
                sem_type=sem_type,
                rot_mask=rot_mask,
                condition_timestep=condition_timestep,
                condition_ctx_timestep=condition_ctx_timestep,
                condition_trace_timestep=condition_trace_timestep,
                seed=seed,
                verbose=verbose,
                local_pcl=local_pcl,
                local_pcl_mask=local_pcl_mask,
                text=text,
                add_cfg=add_cfg,
                left2right_AR_sweep_flag=left2right_AR_sweep_flag,
            )

    def inference_with_grad(
        self,
        # X
        trace,  # B,T,P,4,4
        context,  # B,C,P,4,4
        # WHAT TO Generate
        trace_known_mask,  # B,T,P
        context_known_mask,  # B,C,P
        trace_mask,  # B,T,P
        context_mask,  # B,C,P
        # ATTR
        real_timestep,  # B,T
        group_id,  # B,P
        sem_type,  # B,P # todo: later this can be from pixels
        rot_mask,  # B,P
        # DIFFUSION
        condition_timestep: int = 0,
        condition_ctx_timestep: int = -1,
        condition_trace_timestep: int = -1,
        # cfg=-1,
        # TODO: text, rgb, AR inference, cfg version
        seed=42,
        verbose=True,
        #
        local_pcl=None,  # B,P,PCL,3
        local_pcl_mask=None,  # B,P
        text=None,
        #
        add_cfg: float = -1.0,
        #
        left2right_AR_sweep_flag=False,
    ):
        """
        Run inference to generate motion trajectories.

        Args:
            trace (torch.Tensor): Initial trace [B, T, P, 4, 4].
            context (torch.Tensor): Context trajectory [B, C, P, 4, 4].
            trace_known_mask (torch.Tensor): Mask of known trace parts [B, T, P].
            context_known_mask (torch.Tensor): Mask of known context parts [B, C, P].
            trace_mask (torch.Tensor): Validity mask for trace [B, T, P].
            context_mask (torch.Tensor): Validity mask for context [B, C, P].
            real_timestep (torch.Tensor): Real timesteps for trace [B, T].
            group_id (torch.Tensor): Group IDs [B, P].
            sem_type (torch.Tensor): Semantic types [B, P].
            rot_mask (torch.Tensor): Rotation mask [B, P].
            condition_timestep (int): Timestep to start generation/conditioning.
            seed (int): Random seed.
            verbose (bool): Whether to log progress.
            local_pcl: Local point clouds.
            local_pcl_mask: Mask for local point clouds.
            text: Text condition.
            add_cfg (float): Classifier-free guidance scale.
            left2right_AR_sweep_flag (bool): Enable AR sweep generation.

        Returns:
            dict: Dictionary containing generated results (x0_pred_trace, etc.).
        """
        if condition_timestep > 0:
            # ！ TODO
            # raise NotImplementedError("TODO MUST  SAPERATE THE CTX AND REAL")
            logging.warning(
                f"Should Update the code to support CTX/REAL diferent modes"
            )
        is_training = self.training
        if is_training:
            logging.warning("Change to evaluation mode")
            self.eval()

        # ! pad and prepare outside the inference function
        #########################################################
        B, T, P = trace.shape[:3]
        C = context.shape[1]
        C_T = C + T
        assert trace.shape == (B, T, P, 4, 4)
        assert context.shape == (B, C, P, 4, 4)
        assert trace_known_mask.shape == (B, T, P)
        assert context_known_mask.shape == (B, C, P)
        assert trace_mask.shape == (B, T, P)
        assert context_mask.shape == (B, C, P)
        assert real_timestep.shape == (B, T)
        assert group_id.shape == (B, P)
        assert sem_type.shape == (B, P)
        #########################################################

        scheduler = self.gen_scheduler
        device = trace.device

        x_input = torch.cat([context, trace], dim=1)  # B,C+T,P,4,4
        known_mask = torch.cat(
            [context_known_mask, trace_known_mask], dim=1
        ).bool()  # B,C+T,P
        known_mask_ctxonly = known_mask.clone()
        known_mask_ctxonly[:, C:] = False
        known_mask_traconly = known_mask.clone()
        known_mask_traconly[:, :C] = False

        x_mask = torch.cat([context_mask, trace_mask], dim=1)  # B,C+T,P
        x_param = self.pose_io.encode(
            x_input, sem_ids=sem_type[:, None].expand(-1, C + T, -1), stat_mask=None
        )  # B,C+T,P,9
        rot_mask_multiplier = self.pose_io.rot_mask(rot_mask)[
            :, None
        ].float()  # B,1,P,9
        g = torch.Generator(device=device)  # or "cpu"
        g.manual_seed(seed)
        noise = torch.randn(x_param.shape, device=device, generator=g)  # B,C+T,P,9
        x = x_param.clone() * known_mask.float()[..., None] + noise * (
            1.0 - known_mask.float()[..., None]
        )

        # virtual_timestep = torch.arange(C, device=device)[None].expand(B, -1)
        # timestep = torch.cat([virtual_timestep, real_timestep], dim=1)  # B,C+T
        # virtual_timestep_mask = torch.zeros(B, C + T).to(trace_mask)
        # virtual_timestep_mask[:, :C] = True  # if true, is a virtual time

        cond = x_param.clone()
        # ! do separately the ctx and trace
        if condition_ctx_timestep < 0:
            condition_ctx_timestep = condition_timestep
        if condition_trace_timestep < 0:
            condition_trace_timestep = condition_timestep
        if condition_ctx_timestep > 0:
            cond[:, :C] = self.noise_scheduler.add_noise(
                original_samples=cond[:, :C].reshape(B * C * P, -1),
                noise=noise[:, :C].reshape(B * C * P, -1),
                timesteps=torch.ones(B * C * P).to(device).int()
                * int(condition_ctx_timestep),
            ).reshape(B, C, P, -1)
        if condition_trace_timestep > 0:
            cond[:, C:] = self.noise_scheduler.add_noise(
                original_samples=cond[:, C:].reshape(B * T * P, -1),
                noise=noise[:, C:].reshape(B * T * P, -1),
                timesteps=torch.ones(B * T * P).to(device).int()
                * int(condition_trace_timestep),
            ).reshape(B, T, P, -1)

        if text is None:
            # ! here use empty str to be empty tokens
            text = [""] * B
        add_tk = self.input_layer.encode_language(text)
        add_tk_mask = torch.ones(*add_tk.shape[:2], device=device).bool()

        if verbose:
            _iter = tqdm(scheduler.timesteps)
        else:
            _iter = scheduler.timesteps

        if left2right_AR_sweep_flag:
            logging.info("Using left2right AR sweep flag")
            empty_add_tk = self.input_layer.encode_language([""] * B)
            feat = self.input_layer(
                sem_type,
                group_id,
                local_pcl,
                local_pcl_mask,
            )
            original_known_mask = known_mask.clone()
            for sweep_t in tqdm(range(C_T)):
                if known_mask[:, : sweep_t + 1].all():
                    continue
                _blocked_mask = (~known_mask).clone()  # if true, need to be blocked
                _blocked_mask[:, : sweep_t + 1] = False

                for t in scheduler.timesteps:  # [999, ... 1, 0]
                    df_t = torch.ones(B, C_T, P, device=device).long() * int(
                        t
                    )  # for now synced
                    df_t[known_mask_ctxonly] = condition_ctx_timestep
                    df_t[known_mask_traconly] = condition_trace_timestep
                    df_t[_blocked_mask] = scheduler.config.num_train_timesteps - 1
                    x[known_mask] = x[known_mask] * 0.0 + cond[known_mask].clone()
                    x[_blocked_mask] = (
                        x[_blocked_mask] * 0.0 + noise[_blocked_mask].clone()
                    )
                    x = x * rot_mask_multiplier.clone()

                    pred = self.lmm(
                        x=x[:, C:],
                        x_mask=x_mask[:, C:],
                        x_time=real_timestep,
                        x_level=df_t[:, C:],
                        feat=feat,
                        #
                        ctx=x[:, :C],
                        ctx_mask=x_mask[:, :C],
                        ctx_level=df_t[:, :C],
                        #
                        add_tokens=add_tk.clone(),
                        add_tokens_mask=add_tk_mask.clone(),
                    )
                    pred = self.x0_to_mode(pred, x, df_t)
                    if add_cfg > 0:
                        pred_uncond = self.lmm(
                            x=x[:, C:],
                            x_mask=x_mask[:, C:],
                            x_time=real_timestep,
                            x_level=df_t[:, C:],
                            feat=feat,
                            #
                            ctx=x[:, :C],
                            ctx_mask=x_mask[:, :C],
                            ctx_level=df_t[:, :C],
                            #
                            add_tokens=empty_add_tk.clone(),
                            add_tokens_mask=add_tk_mask.clone(),
                        )
                        pred_uncond = self.x0_to_mode(pred_uncond, x, df_t)
                        pred = pred_uncond + (pred - pred_uncond) * float(add_cfg)

                    _prev_x = scheduler.step(pred, int(t), x).prev_sample
                    up_data_mask = (~known_mask) & (~_blocked_mask)
                    x[up_data_mask] = x[up_data_mask] * 0.0 + _prev_x[
                        up_data_mask
                    ].clone().to(x)

                known_mask[:, : sweep_t + 1] = True
                cond[:, : sweep_t + 1] = (
                    cond[:, : sweep_t + 1] * 0.0 + x[:, : sweep_t + 1].clone()
                )
            known_mask = original_known_mask.clone()
        else:
            empty_add_tk = self.input_layer.encode_language([""] * B)
            feat = self.input_layer(
                sem_type,
                group_id,
                local_pcl,
                local_pcl_mask,
            )
            for t in _iter:  # [999, ... 1, 0]
                df_t = torch.ones(B, C_T, P, device=device).long() * int(
                    t
                )  # for now synced
                df_t[known_mask_ctxonly] = condition_ctx_timestep
                df_t[known_mask_traconly] = condition_trace_timestep
                x[known_mask] = x[known_mask] * 0.0 + cond[known_mask].clone()
                x = x * rot_mask_multiplier.clone()
                pred = self.lmm(
                    x=x[:, C:],
                    x_mask=x_mask[:, C:],
                    x_time=real_timestep,
                    x_level=df_t[:, C:],
                    feat=feat,
                    #
                    ctx=x[:, :C],
                    ctx_mask=x_mask[:, :C],
                    ctx_level=df_t[:, :C],
                    #
                    add_tokens=add_tk.clone(),
                    add_tokens_mask=add_tk_mask.clone(),
                )
                pred = self.x0_to_mode(pred, x, df_t)
                if add_cfg > 0:
                    pred_uncond = self.lmm(
                        x=x[:, C:],
                        x_mask=x_mask[:, C:],
                        x_time=real_timestep,
                        x_level=df_t[:, C:],
                        feat=feat,
                        #
                        ctx=x[:, :C],
                        ctx_mask=x_mask[:, :C],
                        ctx_level=df_t[:, :C],
                        #
                        add_tokens=empty_add_tk.clone(),
                        add_tokens_mask=add_tk_mask.clone(),
                    )
                    pred_uncond = self.x0_to_mode(pred_uncond, x, df_t)
                    pred = pred_uncond + (pred - pred_uncond) * float(add_cfg)

                _prev_x = scheduler.step(pred, int(t), x).prev_sample
                x[~known_mask] = x[~known_mask] * 0.0 + _prev_x[~known_mask].clone().to(
                    x
                )

        x[known_mask] = x[known_mask] * 0.0 + x_param[known_mask].clone().to(x)
        x = self.pose_io.nonsigular_empty(x, rot_mask_multiplier)

        ret = self.pose_io.decode(
            x, sem_ids=sem_type[:, None].expand(-1, C + T, -1)
        )  # B,C+T,P,4,4
        ret_context = ret[:, :C]
        ret_trace = ret[:, C:]

        if is_training:
            logging.warning("Reverting to training mode")
            self.train()

        return ret_context, ret_trace

    def get_optim_groups(self, weight_decay, lmm_lr, input_layer_lr, image_encoder_lr):
        """Get optimizer parameter groups with per-component learning rates.

        Combines the LMM and input layer parameter groups, each with
        their own learning rate and weight decay settings.

        Args:
            weight_decay: Weight decay for all groups.
            lmm_lr: Learning rate for the world model (LMM).
            input_layer_lr: Learning rate for the input encoder.
            image_encoder_lr: Learning rate for image encoder (reserved).

        Returns:
            List of parameter group dicts for ``torch.optim``.

        Example::

            groups = agent.get_optim_groups(0.1, 6e-4, 6e-4, 6e-4)
        """
        return self.lmm.get_optim_groups(
            weight_decay=weight_decay, base_lr=lmm_lr
        ) + self.input_layer.get_optim_groups(
            weight_decay=weight_decay, base_lr=input_layer_lr
        )

    ####################### Retargeting Helpers ##################################
    prepare_attributes = staticmethod(prepare_attributes)
    pad_time = staticmethod(pad_time)

    @torch.inference_mode()
    def hh_boost(self, C, T, X=9, human_type="smplx", humanoid_type="g1"):
        """JIT-trace the LMM for faster human--humanoid inference.

        Traces the world model with dummy inputs at the specified
        dimensions, caches the attribute tensors and context, and replaces
        ``self.lmm`` with the traced module.

        Args:
            C: Number of context frames.
            T: Number of trace frames.
            X: Feature dimension per part (default 9).
            human_type: Human model type (e.g. ``"smplx"``).
            humanoid_type: Humanoid model type (e.g. ``"g1"``).

        Example::

            agent.hh_boost(C=8, T=30, humanoid_type="g1")
        """
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        self.lmm.to(dtype)

        self.hh_boosted_types = (human_type, humanoid_type)
        self.hh_boosted_C = C
        sem, gids, uids, Ph, Pr = self.prepare_attributes(
            humanoid_type, human_type, device
        )
        ctx = self.fetch_ctx_record(uids, C, mode="latest")
        ctx_known = torch.ones(C, Ph + Pr).to(device).bool()
        # ! debug to viz the ctx

        self.hh_boost_cache = (sem, gids, uids, Ph, Pr, ctx, ctx_known)
        P = Ph + Pr
        add_tk = self.input_layer.encode_language([""])

        with torch.autocast(device_type="cuda", dtype=dtype):
            func = torch.jit.trace(
                self.lmm,
                (
                    torch.zeros(1, T, P, X, dtype=dtype).to(device),  # x
                    torch.ones(1, T, P).bool().to(device),  # x_mask
                    torch.zeros(1, T, dtype=dtype).to(device),  # x_time
                    torch.zeros(1, T, P).long().to(device),  # x_level
                    torch.zeros(1, P, self.input_layer.feat_dim).to(device),  # feat
                    torch.zeros(1, C, P, X, dtype=dtype).to(device),  # ctx
                    torch.ones(1, C, P).bool().to(device),  # ctx_mask
                    torch.zeros(1, C, P).long().to(device),  # ctx_level
                    add_tk,
                    torch.ones(*add_tk.shape[:2], device=device).bool(),
                ),
            )
        self.lmm = func

        # ! run again to check again
        # todo: here no text or add tokens
        with torch.autocast(device_type="cuda", dtype=dtype):
            self.lmm(
                torch.zeros(1, T, P, X, dtype=dtype).to(device),  # x
                torch.ones(1, T, P).bool().to(device),  # x_mask
                torch.zeros(1, T, dtype=dtype).to(device),  # x_time
                torch.zeros(1, T, P).long().to(device),  # x_level
                torch.zeros(1, P, self.input_layer.feat_dim).to(device),  # feat
                torch.zeros(1, C, P, X, dtype=dtype).to(device),  # ctx
                torch.ones(1, C, P).bool().to(device),  # ctx_mask
                torch.zeros(1, C, P).long().to(device),  # ctx_level
                add_tk,
                torch.ones(*add_tk.shape[:2], device=device).bool(),
            )

        print(f"Boosted HH Inference with X={X}, T={T}, C={C}, P={P}")

        self.hh_boosted = True
        return

    @torch.inference_mode()
    def hh_joint_infer(
        self,
        ctx_len,
        # compact input
        # ! if want to gen future, can have invisible future
        human_Q=None,  # None or T,P,4,4
        human_time=None,  # None or T
        human_visible=None,  # None or T,P
        robot_Q=None,  # None or T,P,4,4
        robot_time=None,  # None or T
        robot_visible=None,  # None or T,P
        #
        windows_size=30,
        window_overlap=10,
        human_type="smplx",
        humanoid_type="g1",
        model_fps=30.0,
        device=torch.device("cuda:0"),
        #
        gen_steps=10,
        condition_timestep=0,
        seed=42,
        verbose=True,
    ):
        """Run sliding-window joint human--humanoid motion inference.

        Handles timeline alignment, centering, optional hand-to-wrist
        local conversion, and iterative windowed inference with overlap
        stitching. Uses the context record buffer for conditioning.

        Args:
            ctx_len: Number of context frames to use from the buffer.
            human_Q: Human joint transforms ``[T_h, P_h, 4, 4]`` or None.
            human_time: Human timestamps ``[T_h]`` or None.
            human_visible: Human visibility mask ``[T_h, P_h]`` or None.
            robot_Q: Robot joint transforms ``[T_r, P_r, 4, 4]`` or None.
            robot_time: Robot timestamps ``[T_r]`` or None.
            robot_visible: Robot visibility mask ``[T_r, P_r]`` or None.
            windows_size: Number of frames per inference window.
            window_overlap: Frame overlap between consecutive windows.
            human_type: Human model type (e.g. ``"smplx"``).
            humanoid_type: Humanoid model type (e.g. ``"g1"``).
            model_fps: Model frame rate for timeline quantization.
            device: Torch device.
            gen_steps: Number of DDIM generation steps.
            condition_timestep: Noise level for conditioning known data.
            seed: Random seed.
            verbose: Whether to show progress bars.

        Returns:
            Tuple of ``(ret_human, ret_robot, padded_t)`` where
            ``ret_human`` is ``[T, P_h, 4, 4]``, ``ret_robot`` is
            ``[T, P_r, 4, 4]``, and ``padded_t`` is ``[T]``.

        Example::

            human, robot, times = agent.hh_joint_infer(
                ctx_len=8, human_Q=h_poses, human_time=h_t,
                robot_Q=r_poses, robot_time=r_t,
            )
        """
        ####################################################################
        # ! warning: only support smplx for now, hardcoded
        # human_Q: T,P,4,4; real_time: T; visible: T,P
        # should handle the height outside
        # handle all, including the sliding window.
        ####################################################################

        self.gen_scheduler.set_timesteps(gen_steps)
        if (
            self.hh_boosted
            and human_type == self.hh_boosted_types[0]
            and humanoid_type == self.hh_boosted_types[1]
            and ctx_len == self.hh_boosted_C
        ):
            sem, gids, uids, Ph, Pr, ctx, ctx_known = self.hh_boost_cache
            # Clone to avoid mutating the cached tensors
            ctx = ctx.clone()
            ctx_known = ctx_known.clone()
        else:
            sem, gids, uids, Ph, Pr = self.prepare_attributes(
                humanoid_type, human_type, device
            )
            ctx = self.fetch_ctx_record(uids, ctx_len, mode="latest")
            ctx_known = torch.ones(ctx_len, Ph + Pr).to(device).bool()

        # # ! debug
        # print(f"uids: {uids}")
        # project_root = osp.expanduser("~/projects/UMM")
        # d = np.load(osp.join(project_root, "debug.npz"))
        # print(f"ctx: {ctx}")
        # ctx = torch.randn_like(torch.as_tensor(d["context"]).to(device)[0])
        # print(f"ctx: {ctx}")
        # ctx_known = torch.as_tensor(d["context_mask"]).to(device).bool()[0]
        # _ctx_uids = torch.as_tensor(d["uid"]).to(device)[0]
        # assert (_ctx_uids == uids).all(), f"_ctx_uids: {_ctx_uids}, uids: {uids}"

        # convert to uids order
        # P = Ph + Pr  # ! the order is human first then robot

        # handle the time padding; assume start from the first known
        padded_t, human_t_mask, robot_t_mask = self.pad_time(
            human_time, robot_time, model_fps, device
        )
        T = len(padded_t)

        # prepare the trace
        trace_human = torch.eye(4)[None, None].repeat(T, Ph, 1, 1).to(device)
        trace_human_known = torch.zeros(T, Ph).to(device).bool()
        if human_Q is not None:
            assert human_Q.shape[1] == Ph, f"human_Q.shape: {human_Q.shape}, Ph: {Ph}"
            trace_human[human_t_mask] = human_Q.to(trace_human)
            if human_visible is None:
                human_visible = torch.ones_like(human_Q[..., 0, 0]).to(device).bool()
            trace_human_known[human_t_mask] = human_visible
        trace_robot = torch.eye(4)[None, None].repeat(T, Pr, 1, 1).to(device)
        trace_robot_known = torch.zeros(T, Pr).to(device).bool()
        if robot_Q is not None:
            assert robot_Q.shape[1] == Pr, f"robot_Q.shape: {robot_Q.shape}, Pr: {Pr}"
            trace_robot[robot_t_mask] = robot_Q.to(trace_robot)
            if robot_visible is None:
                robot_visible = torch.ones_like(robot_Q[..., 0, 0]).to(device).bool()
            trace_robot_known[robot_t_mask] = robot_visible

        trace = torch.cat([trace_human, trace_robot], 1)
        trace_known = torch.cat([trace_human_known, trace_robot_known], 1)

        # ref cur t at middle
        cur = 0
        while cur < T:
            # ! slice, modify the time and center
            # ref at the first known joint 0 step.
            _trace = trace[cur : cur + windows_size].clone()
            _trace_known = trace_known[cur : cur + windows_size].clone()
            _timestep = padded_t[cur : cur + windows_size].clone()
            zero_at_ith = _trace_known.any(dim=1).int().argmax()
            if not _trace_known[zero_at_ith].any():
                logging.warning(
                    f"The window {cur}-{cur+windows_size} has nothing known, the center maybe wrong"
                )
            _zero_human_base_T = _trace[zero_at_ith][0]
            _zero_robot_base_T = _trace[zero_at_ith][Ph]
            if _trace_known[zero_at_ith][0]:  # human
                _center = _trace[zero_at_ith][0][:2, 3].clone()  # 2
            elif _trace_known[zero_at_ith][Ph]:  # robot
                _center = trace[zero_at_ith][Ph][:2, 3].clone()
            else:
                logging.warning(
                    f"None of the robot or human root joint is valid at zero ref, use the first root, may be wrong"
                )
                _center = _trace[zero_at_ith][0][:2, 3].clone()  # 2
            # ! wait, is it safe to do the center for the robto??? check this too!!!!
            _trace[..., :2, 3] = _trace[..., :2, 3].clone() - _center

            if self.hand_attached_to_wrist_flag:
                # Do NOT convert ctx — it is already in local coords from training
                _trace[:, :Ph] = self.convert_smplx_hand_param(
                    _trace[:, :Ph].clone(), to_local=True
                )

            _timestep = _timestep - _timestep[zero_at_ith]
            # print(_timestep)

            _gen_ctx, _gen_trace = self.inference(
                # X
                trace=_trace[None],
                context=ctx[None],
                # WHAT TO Generate
                trace_known_mask=_trace_known[None],
                context_known_mask=ctx_known[None],
                trace_mask=torch.ones_like(_trace_known[None]).bool(),
                context_mask=torch.ones_like(ctx_known[None]).bool(),
                # ATTR
                real_timestep=_timestep[None],
                group_id=gids[None, :],  # B,P
                sem_type=sem[None, :],  # B,P # todo: later this can be from pixels
                rot_mask=torch.ones_like(gids[None, :]).bool(),  # B,P
                # DIFFUSION
                condition_trace_timestep=condition_timestep,
                seed=seed,
                verbose=verbose,
            )

            if self.hand_attached_to_wrist_flag:
                _gen_trace[0, :, :Ph] = self.convert_smplx_hand_param(
                    _gen_trace[0, :, :Ph].clone(), to_local=False
                )

            _gen_trace[..., :2, 3] = _gen_trace[..., :2, 3] + _center

            # move back
            trace[cur : cur + windows_size] = _gen_trace[0].clone()
            trace_known[cur : cur + windows_size] = True
            if cur + windows_size >= T:
                break
            cur = cur + windows_size - window_overlap
        ret_human = trace[:T, :Ph]
        ret_robot = trace[:T, Ph:]
        return ret_human, ret_robot, padded_t[:T]

    def fetch_ctx_record(self, uids, C, mode="latest"):
        """Retrieve context frames from the record buffer for given UIDs.

        Looks up the stored context by the sorted UID tuple key, extracts
        the most recent ``C`` frames, and reorders columns to match the
        input UID ordering.

        Args:
            uids: Unique part IDs of shape ``[P]``.
            C: Number of context frames to fetch.
            mode: Retrieval mode; currently only ``"latest"`` is supported.

        Returns:
            Context tensor of shape ``[C, P, 4, 4]`` on the same device
            as ``uids``.

        Raises:
            KeyError: If the UID combination is not found in the buffer.

        Example::

            ctx = agent.fetch_ctx_record(uids, C=8, mode="latest")
        """
        uids = uids.clone()
        _idx = uids.argsort()
        key = tuple(uids[_idx].cpu().numpy().tolist())
        if key not in self.ctx_record_buffer:
            logging.error(
                f"Key {key} not found in ctx_record_buffer: {self.ctx_record_buffer.keys()}"
            )
        buffer = self.ctx_record_buffer[key]
        if mode == "latest":
            ctx = [buffer[i] for i in range(-C, 0)]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        ctx = torch.cat(ctx, 0).to(uids.device)  # C,P,4,4
        # reorder back to the input uid
        inv_idx = _idx.argsort()
        ctx = ctx[:, inv_idx]
        return ctx.clone()
