"""AgentV6NonDF: Non-Diffusion-Forcing agent with masked conditioned DDPM.

Replaces per-element async noise (diffusion forcing) with:
- ONE global noise level ``t`` per sample (standard DDPM)
- Explicit condition tensor ``cond_X`` and mask ``mask``
- Dual conditioning: input concat ``[X; cond_X; mask]`` + AdaLN per-token
- Standard DDIM inference with mask-based conditioning

Based on AgentV6All architecture (two-scale pooled temporal attention).

Example::

    from umm.models.umm_v6_nondf.agent import AgentV6NonDF
    agent = AgentV6NonDF(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import logging
import time
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from umm.models.umm_v6_nondf.world_models.moma_v6 import MotionMachineV6
from umm.models.umm_v6_nondf.poseio import PoseIO_V3
from umm.models.umm_v6_nondf.loss_utils import type_mean, type_mean_balanced
from umm.models.umm_v6_nondf.df_helpers import compute_x0, compute_snr
from umm.models.umm_v6_nondf.act_helper import extract_langtable_action
from umm.datasets.umm_v6_all.dataset_config import (
    SEM_LANGTABLE_ACT,
    SEM_OBJECT,
    SEM_BACKGROUND,
)
from umm.models.umm_v6_nondf.dino_encoder import DINOImageEncoder, IMAGENET_MEAN, IMAGENET_STD
from umm.models.umm_v6_nondf.identify_layer import IdentifyLayerV2Dual


class AgentV6NonDF(nn.Module):
    """Non-Diffusion-Forcing agent with masked conditioned DDPM.

    Replaces diffusion forcing (per-element async noise) with:
    - ONE global noise level per sample (standard DDPM)
    - Three tensors: X (all noised), cond_X (known=real, unknown=learnable), mask
    - Dual conditioning: concat [X; cond_X; mask] as LMM input + AdaLN per-token
    - Loss on ALL valid positions (no per-element SNR weighting)

    Args:
        cfg: Full configuration dict.
    """

    # View name to index mapping for view_ids
    VIEW_NAME_TO_IDX = {
        "head": 0,
        "left_wrist": 1,
        "right_wrist": 2,
        "left_hand": 3,
        "right_hand": 4,
        "third_person": 5,
        "front": 6,
        "side": 7,
    }

    def __init__(self, cfg):
        super().__init__()

        self.cfg, self.model_cfg = cfg, cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        # * dim
        self.latent_dim = self.model_cfg.get("latent_dim", 768)

        # * augmentation probs
        self.time_mask_aug_prob = self.optim_cfg.get("time_mask_aug_prob", 0.0)
        self.ctx_mask_aug_prob = self.optim_cfg.get("ctx_mask_aug_prob", 0.0)

        # * text cls prob
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)
        if self.text_masked_prob > 0.0:
            logging.warning(f"text_masked_prob={self.text_masked_prob}")

        # * loss weights
        self.diffusion_w_mode = self.optim_cfg.get("diffusion_w_mode", "default")
        self.diffusion_w_snr_mode_gamma = self.optim_cfg.get(
            "diffusion_w_snr_mode_gamma", 5.0
        )
        logging.warning(f"diffusion_w_mode={self.diffusion_w_mode}")

        self.w_langtable_act = self.optim_cfg.get("w_langtable_act", -1.0)

        # Non-DF mask scheduling params (replicate v6.8 context scheduling as masks)
        self.prob_virtual_allknown = self.optim_cfg.get("prob_virtual_allknown", 0.0)
        self.prob_bias_to_known = self.optim_cfg.get("prob_bias_to_known", 0.0)
        if self.prob_bias_to_known > 0.0:
            logging.warning(f"prob_bias_to_known={self.prob_bias_to_known}")
        self.ctx_always_known = self.optim_cfg.get("ctx_always_known", False)
        if self.ctx_always_known:
            logging.warning("ctx_always_known=True: context frames will always be known")

        # Mask schedule version: "v1.0" = original, "v1.1" = inference-aware
        self.mask_schedule_version = self.optim_cfg.get("mask_schedule_version", "v1.0")
        # SEM types identifying human parts for h2r/r2h patterns in v1.1
        self.human_sem_types = frozenset(
            self.optim_cfg.get("human_sem_types", [5])
        )
        if self.mask_schedule_version != "v1.0":
            logging.warning(
                f"mask_schedule_version={self.mask_schedule_version}, "
                f"human_sem_types={sorted(self.human_sem_types)}"
            )
        # Debug counters for v1.1 pattern distribution (only used with DEBUG level)
        self._v11_pattern_counts: dict = {}

        self.w_t_se3 = self.optim_cfg.get("w_t_se3", 1.0)
        self.w_t_xyz = self.optim_cfg.get("w_t_xyz", 1.0)
        self.w_q = self.optim_cfg.get("w_q", 1.0)
        self.w_R = self.optim_cfg.get("w_R", 0.0)
        self.w_virtual = self.optim_cfg.get("w_virtual", 0.1)
        self.w_real = self.optim_cfg.get("w_real", 1.0)

        self.movement_reweight_min_ratio = self.optim_cfg.get(
            "movement_reweight_min_ratio", -1.0
        )
        self.movement_reweight_max_ratio = self.optim_cfg.get(
            "movement_reweight_max_ratio", 5.0
        )  # cap max/min weight ratio to prevent gradient instability
        self.movement_scale = self.optim_cfg.get("movement_scale", 1.0)
        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }

        # Scene vs robot+action loss balancing (1:1 weighting)
        self.loss_balance_robot_scene = self.optim_cfg.get(
            "loss_balance_robot_scene", False
        )
        # SEM types considered "scene" for loss balancing
        self._scene_sem_types = frozenset(
            self.optim_cfg.get("scene_sem_types", [SEM_OBJECT, SEM_BACKGROUND])
        )
        if self.loss_balance_robot_scene:
            logging.warning(
                f"loss_balance_robot_scene=True, scene_sem_types={sorted(self._scene_sem_types)}"
            )

        self.loss_type = self.optim_cfg.get("loss_type", "norm")
        self.loss_square = self.optim_cfg.get("loss_square", True)
        self.loss_stable_beta = self.optim_cfg.get("loss_stable_beta", 0.1)

        self.w_act = self.optim_cfg.get("w_act", 0.0)
        self.w_emp = self.optim_cfg.get("w_emp", 0.0)

        # * PoseIO
        self.pose_io_debug_flag = self.model_cfg.get("pose_io_debug_flag", False)
        self.pose_io = PoseIO_V3(
            sem_group_ids=self.model_cfg.get("param_sem_group_ids", [[-1]]),
            rot_len=self.model_cfg.get("rot_len", 1.0),
            param_mean=self.model_cfg.get("param_mean", []),
            param_std=self.model_cfg.get("param_std", []),
            std_factor=self.model_cfg.get("std_factor", []),
            state_mean_std_flag=True,
            state_only_pos_flag=self.model_cfg.get("state_only_pos_flag", True),
        )

        # * Diffusion schedulers
        self.__init_diffusion__(self.model_cfg.get("diffusion", {}))

        # * Input layer (dual-path: PointNet + DINO pixel query)
        input_cfg = self.model_cfg.get("input_layer", {})
        self.input_layer = IdentifyLayerV2Dual(
            latent_dim=self.latent_dim,
            feat_dim=self.model_cfg.get("feat_dim", 2048),
            num_types=input_cfg.get("num_types", 512),
            num_groups=input_cfg.get("num_groups", 512),
            group_emb_same_init_flag=input_cfg.get("group_emb_same_init_flag", False),
            point_encoder_latent=input_cfg.get("point_encoder_latent", 256),
            point_input_dim=input_cfg.get("point_input_dim", 6),
            point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
            point_version=input_cfg.get("point_version", "v2"),
        )

        # * LMM — selectable variant: "v6" (default) or "tpa" (HH2.0 dense style)
        lmm_cfg = self.model_cfg.get("lmm", {})
        self.lmm_variant = lmm_cfg.get("variant", "v6")
        if self.lmm_variant == "tpa":
            from umm.models.umm_v6_nondf.world_models.moma_tpa import MotionMachineTPA
            # TPA operates at a single latent_dim; C_P = latent_dim for token projectors
            self.C_P = self.latent_dim
            self.lmm = MotionMachineTPA(
                latent_dim=self.latent_dim,
                feat_dim=self.model_cfg.get("feat_dim", 512),
                n_head=lmm_cfg.get("n_head", 16),
                input_dim=9,
                output_dim=9,
                layer_type_string=lmm_cfg.get("layer_type_string", "tpatpapt"),
                skip_layer_idx=lmm_cfg.get("skip_layer_idx", [2, 3, 5]),
                cond_binder_dim=lmm_cfg.get("cond_binder_dim", 512),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 12),
                fc_factor=lmm_cfg.get("fc_factor", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
                num_register=lmm_cfg.get("num_register", 4),
                num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
                n_ctx=lmm_cfg.get("n_ctx", 128),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                time_multiplier=lmm_cfg.get("time_multiplier", 10.0),
                use_reg_binder=lmm_cfg.get("use_reg_binder", True),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", 2.0),
                gate_bound=lmm_cfg.get("gate_bound", 2.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 2),
                add_first_n_add_token_to_feat=lmm_cfg.get("add_first_n_add_token_to_feat", -1),
                time_rope_flag=lmm_cfg.get("time_rope_flag", False),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", True),
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
            )
            logging.warning(f"LMM variant: TPA (latent_dim={self.latent_dim}, C_P={self.C_P})")
        else:
            self.C_P = lmm_cfg.get("C_P", 512)  # Wide dim for p-layers
            self.lmm = MotionMachineV6(
                C_T=lmm_cfg.get("C_T", 128),
                C_P=self.C_P,
                feat_dim=self.model_cfg.get("feat_dim", 512),
                n_head_t=lmm_cfg.get("n_head_t", 8),
                n_head_p=lmm_cfg.get("n_head_p", 16),
                input_dim=9,
                output_dim=9,
                n_t_layers=lmm_cfg.get("n_t_layers", 4),
                n_Q=lmm_cfg.get("n_Q", 4),
                n_ctx=lmm_cfg.get("n_ctx", 16),
                fc_factor_t=lmm_cfg.get("fc_factor_t", 2),
                fc_factor_p=lmm_cfg.get("fc_factor_p", 2),
                fc_n_layers=lmm_cfg.get("fc_n_layers", 1),
                fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 4),
                num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
                time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
                ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", -1.0),
                gate_bound=lmm_cfg.get("gate_bound", -1.0),
                ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
                add_time_to_feat=lmm_cfg.get("add_time_to_feat", False),
                time_rope_flag=lmm_cfg.get("time_rope_flag", False),
                num_random_trace_id_pe=lmm_cfg.get("num_random_trace_id_pe", -1),
                dual_t_layer=lmm_cfg.get("dual_t_layer", False),
                traj_pos_emb=lmm_cfg.get("traj_pos_emb", False),
                ctx_p_layer=lmm_cfg.get("ctx_p_layer", False),
                dense_p_block_size=lmm_cfg.get("dense_p_block_size", 0),
                n_dense_p_layers=lmm_cfg.get("n_dense_p_layers", -1),
                x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
                qk_norm=lmm_cfg.get("qk_norm", False),
            )

        # * Context record buffer
        ctx_record_cfg = self.model_cfg.get("ctx_record", {})
        self.ctx_record_flag = len(ctx_record_cfg) > 0
        if self.ctx_record_flag:
            self.ctx_record_buffer_size = ctx_record_cfg.get("buffer_size", 16384)
            self.ctx_record_buffer = {}

        # * Hand-attached-to-wrist flag (for SMPLX hand local-frame conversion)
        self.hand_attached_to_wrist_flag = self.model_cfg.get(
            "hand_attached_to_wrist_flag", False
        )
        if self.hand_attached_to_wrist_flag:
            logging.warning("hand_attached_to_wrist_flag set to True")

        # * DINO image encoder (optional — skip when config is absent/null)
        dino_cfg = self.model_cfg.get("dino_encoder", None)
        self.use_dino = dino_cfg is not None and len(dino_cfg) > 0
        if self.use_dino:
            self.dino_encoder = DINOImageEncoder(
                model_name=dino_cfg.get("model_name", "dinov2_vits14"),
                latent_dim=self.latent_dim,
                num_views=dino_cfg.get("num_views", 8),
                trainable=dino_cfg.get("trainable", False),
            )
            self.image_token_proj = nn.Linear(self.latent_dim, self.C_P)
            # ImageNet normalization buffers: (1, 1, 3, 1, 1) for (B, V, C, H, W)
            self.register_buffer(
                "img_mean",
                torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1),
            )
            self.register_buffer(
                "img_std",
                torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1),
            )
            logging.info("DINO image encoder ENABLED")
        else:
            self.dino_encoder = None
            self.image_token_proj = None
            logging.info("DINO image encoder DISABLED (no dino_encoder config)")

        # * Per-token-type projectors: project from latent_dim to C_P
        # V6: add tokens participate in p-layers at C_P width
        self.lang_token_proj = nn.Linear(self.latent_dim, self.C_P)

        # * Non-DF: learnable embedding for unknown positions
        # Single shared vector (9,) used at every unknown position in cond_X
        self.unknown_cond_emb = nn.Parameter(torch.zeros(9))
        nn.init.normal_(self.unknown_cond_emb, std=0.02)

        logging.info("AgentV6NonDF initialized (non-DF masked conditioned DDPM)")

    # ─────────────────────────────────────────────────────────────────
    # Helper methods (inlined from shared Agent base)
    # ─────────────────────────────────────────────────────────────────

    def loss_kernel(self, pred, gt):
        """Compute element-wise loss between prediction and ground truth."""
        # Use safe_norm (eps in sqrt) to avoid NaN gradient when pred==gt.
        # d/dx ||x|| = x/||x|| is 0/0=NaN at x=0; adding eps prevents this.
        # Critical for DAVIS where ~70% of parts are identity-padded and masked
        # in loss — their zero upstream gradient still NaNs via 0*NaN=NaN.
        _EPS = 1e-8
        if self.loss_type == "norm":
            sq_norm = (pred - gt).pow(2).sum(dim=-1)
            diff = (sq_norm + _EPS).sqrt()
            if self.loss_square:
                diff = diff**2
            return diff
        elif self.loss_type == "stable":
            diff = pred - gt
            sq_norm = diff.pow(2).sum(dim=-1)
            residual_norm = (sq_norm + _EPS).sqrt()
            beta = self.loss_stable_beta
            loss = torch.where(
                residual_norm > beta,
                residual_norm - 0.5 * beta,
                0.5 * sq_norm / beta,
            )
            return loss
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

    def __init_diffusion__(self, cfg):
        """Initialize DDPM training and DDIM generation schedulers."""
        self.pred_type = cfg.get("prediction_type", "epsilon")
        logging.info(f"Prediction type: {self.pred_type}")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.get("train_diff_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "linear"),
            prediction_type=self.pred_type,
            timestep_spacing=cfg.get("timestep_spacing", "trailing"),
            clip_sample=False,
        )
        self.gen_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.get("train_diff_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "linear"),
            prediction_type=self.pred_type,
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

    def x0_to_mode(self, x0, noisy_x, df_timesteps):
        """Convert x0 prediction back to the model's native prediction mode."""
        if not self.explicit_mode_conversion_flag:
            return x0
        elif self.pred_type == "epsilon":
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
    @torch.compiler.disable
    def __prepare_conditioning_mask__(self, B, C, T, P, device, enforce_known_mask,
                                      sem_type=None):
        """Route to the appropriate mask scheduling version.

        Args:
            B: Batch size.
            C: Number of context frames.
            T: Number of trace (prediction) frames.
            P: Number of parts.
            device: Torch device.
            enforce_known_mask: (B, P) bool, force certain parts to known, or None.
            sem_type: (B, P) int tensor of semantic types (needed for v1.1 h2r/r2h).

        Returns:
            cond_known_mask: (B, C+T, P) bool — True=known, False=unknown.
        """
        if self.mask_schedule_version == "v1.1":
            return self._prepare_mask_v11(B, C, T, P, device, enforce_known_mask, sem_type)
        else:
            return self._prepare_mask_v10(B, C, T, P, device, enforce_known_mask)

    @torch.no_grad()
    @torch.compiler.disable
    def _prepare_mask_v10(self, B, C, T, P, device, enforce_known_mask):
        """Original mask scheduling (v1.0): random + bias-to-known.

        Generates a binary mask of shape (B, C+T, P) where 1=known, 0=unknown.
        Replicates the context scheduling patterns from v6.8's __prepare_df_step__
        but expressed as mask patterns instead of noise levels.

        Known positions get real data in cond_X; unknown get learnable embedding.

        Args:
            B: Batch size.
            C: Number of context frames.
            T: Number of trace (prediction) frames.
            P: Number of parts.
            device: Torch device.
            enforce_known_mask: (B, P) bool, force certain parts to known, or None.

        Returns:
            cond_known_mask: (B, C+T, P) bool — True=known, False=unknown.
        """
        # Start with all positions unknown
        cond_known_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)

        # prob_virtual_allknown: probability that all context frames are known
        if self.prob_virtual_allknown > 0.0:
            if torch.rand(1) < self.prob_virtual_allknown:
                cond_known_mask[:, :C] = True

        # prob_bias_to_known: probability that random positions become known
        if self.prob_bias_to_known > 0.0:
            if torch.rand(1) < self.prob_bias_to_known:
                # Randomly mark some positions as known (same logic as v6.8)
                rand_mask = torch.rand(B, C + T, P, device=device)
                th = torch.rand(1).item()  # random threshold
                cond_known_mask = cond_known_mask | (rand_mask < th)

        # enforce_known_mask: force certain parts to always be known
        if enforce_known_mask is not None:
            # enforce_known_mask: (B, P) — expand to trace frames only
            expanded_mask = enforce_known_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, P)
            cond_known_mask[:, C:] = cond_known_mask[:, C:] | expanded_mask

        # ctx_always_known: context frames always known
        if self.ctx_always_known:
            cond_known_mask[:, :C] = True

        return cond_known_mask

    @torch.no_grad()
    @torch.compiler.disable
    def _prepare_mask_v11(self, B, C, T, P, device, enforce_known_mask, sem_type):
        """Inference-aware mask scheduling (v1.1).

        Distribution:
        - 25% random (same as v1.0's prob_bias_to_known behavior)
        - 25% all unknown (context + trace all unknown)
        - 50% inference-style, evenly split 4 ways (12.5% each):
          - H2R: context known, human parts known all trace frames, robot unknown
          - R2H: context known, robot parts known all trace frames, human unknown
          - Future pred: context known, trace [0..t_cut) all known, [t_cut..T) unknown
          - One-frame: context known, one random trace frame known all parts, rest unknown

        Args:
            B: Batch size.
            C: Number of context frames.
            T: Number of trace (prediction) frames.
            P: Number of parts.
            device: Torch device.
            enforce_known_mask: (B, P) bool, force certain parts to known, or None.
            sem_type: (B, P) int tensor of semantic types.

        Returns:
            cond_known_mask: (B, C+T, P) bool — True=known, False=unknown.
        """
        cond_known_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)

        # Draw pattern selector: [0, 1) uniform
        roll = torch.rand(1).item()

        if roll < 0.25:
            # ── Random (25%) ── same as v1.0's prob_bias_to_known ──
            pattern_name = "random"
            rand_mask = torch.rand(B, C + T, P, device=device)
            th = torch.rand(1).item()
            cond_known_mask = rand_mask < th

        elif roll < 0.50:
            # ── All unknown (25%) ── context + trace all unknown ──
            pattern_name = "all_unknown"
            # cond_known_mask stays all-False

        elif roll < 0.625:
            # ── H2R (12.5%) ── context known, human parts known, robot unknown ──
            pattern_name = "h2r"
            cond_known_mask[:, :C] = True  # context known
            # Build human part mask: (B, P) bool
            human_parts = torch.zeros(B, P, device=device, dtype=torch.bool)
            for st in self.human_sem_types:
                human_parts = human_parts | (sem_type == st)
            # Human parts known for all trace frames
            cond_known_mask[:, C:] = human_parts.unsqueeze(1).expand(-1, T, -1)

        elif roll < 0.75:
            # ── R2H (12.5%) ── context known, robot parts known, human unknown ──
            pattern_name = "r2h"
            cond_known_mask[:, :C] = True  # context known
            # Robot parts = NOT human parts
            human_parts = torch.zeros(B, P, device=device, dtype=torch.bool)
            for st in self.human_sem_types:
                human_parts = human_parts | (sem_type == st)
            robot_parts = ~human_parts
            # Robot parts known for all trace frames
            cond_known_mask[:, C:] = robot_parts.unsqueeze(1).expand(-1, T, -1)

        elif roll < 0.875:
            # ── Future prediction (12.5%) ── context known, trace [0..t_cut) known ──
            pattern_name = "future_pred"
            cond_known_mask[:, :C] = True  # context known
            # Random cut: at least 1 frame known, at most T-1
            if T > 1:
                t_cut = torch.randint(1, T, (1,)).item()
            else:
                t_cut = 1  # degenerate: single-frame trace
            cond_known_mask[:, C:C + t_cut] = True  # first t_cut trace frames known

        else:
            # ── One-frame (12.5%) ── context known, one random trace frame known ──
            pattern_name = "one_frame"
            cond_known_mask[:, :C] = True  # context known
            frame_idx = torch.randint(0, T, (1,)).item()
            cond_known_mask[:, C + frame_idx] = True  # one trace frame known

        # enforce_known_mask: force certain parts to always be known (play policy)
        if enforce_known_mask is not None:
            expanded_mask = enforce_known_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, P)
            cond_known_mask[:, C:] = cond_known_mask[:, C:] | expanded_mask

        # ctx_always_known: context frames always known (applied on top)
        if self.ctx_always_known:
            cond_known_mask[:, :C] = True

        # Debug logging: track pattern distribution (guarded to avoid GPU→CPU sync)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self._v11_pattern_counts[pattern_name] = (
                self._v11_pattern_counts.get(pattern_name, 0) + 1
            )
            total = sum(self._v11_pattern_counts.values())
            if total % 50 == 0:
                dist_str = ", ".join(
                    f"{k}: {v}/{total} ({100*v/total:.1f}%)"
                    for k, v in sorted(self._v11_pattern_counts.items())
                )
                logging.debug(f"v1.1 mask pattern distribution ({total} samples): {dist_str}")

        return cond_known_mask

    @torch.compiler.disable
    def handle_play_policy_data(self, batch):
        """Handle exploration/policy data by enforcing known masks."""
        if "is_explore" not in batch["extra"]:
            enforce_known_mask = torch.zeros_like(
                batch["trace_sem_type"], dtype=torch.bool
            )  # (B, N)
            return batch, enforce_known_mask
        explore_mask = torch.as_tensor(batch["extra"]["is_explore"]).to(
            batch["trace_mask"].device
        )  # (B,)
        for i in range(len(explore_mask)):
            if explore_mask[i]:
                batch["text"][i] = ""
        enforce_action_mask = batch["trace_sem_type"] == SEM_LANGTABLE_ACT  # (B, N)
        B, N = enforce_action_mask.shape
        enforce_object_mask = batch["trace_sem_type"] == SEM_OBJECT  # (B, N)
        candidate = torch.stack(
            [enforce_action_mask, enforce_object_mask], dim=-1
        )  # (B, N, 2)
        choice = torch.randint(0, 2, (B,), device=batch["trace_mask"].device)  # (B,)
        enforce_known_mask = torch.gather(
            candidate, dim=-1, index=choice[:, None, None].expand(-1, N, -1)
        ).squeeze(
            -1
        )  # (B, N)
        enforce_known_mask = enforce_known_mask & explore_mask[:, None]
        return batch, enforce_known_mask

    @torch.compiler.disable
    @torch.no_grad()
    def record_ctx(self, ctx, ctx_mask, uid):
        """Store context frames into the per-UID record buffer."""
        if not self.ctx_record_flag:
            return
        ctx, ctx_mask, uid = ctx.clone(), ctx_mask.clone(), uid.clone()
        B = len(ctx)
        for bid in range(B):
            c, m, u = ctx[bid], ctx_mask[bid], uid[bid]
            m_p, m_c = m.any(dim=0), m.any(dim=1)
            if m_p.sum() == 0 or m_c.sum() == 0:
                continue
            c = c[m_c][:, m_p]
            u = u[m_p]
            re_index = u.argsort()
            c = c[:, re_index].cpu()
            u = tuple(u[re_index].cpu().numpy().tolist())
            if u not in self.ctx_record_buffer:
                self.ctx_record_buffer[u] = deque(maxlen=self.ctx_record_buffer_size)
            self.ctx_record_buffer[u].extend(c.split(1, dim=0))

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return state dict, including ctx_record_buffer if enabled."""
        destination = super().state_dict(destination, prefix, keep_vars)
        if self.ctx_record_flag:
            destination[prefix + "ctx_record_buffer"] = self.ctx_record_buffer
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load state dict, extracting ctx_record_buffer."""
        key = prefix + "ctx_record_buffer"
        if self.ctx_record_flag and key in state_dict:
            self.ctx_record_buffer = state_dict.pop(key)
        elif key in state_dict:
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def filter_ckpt_prefix(checkpoint_dict, prefix):
        """Filter and strip a key prefix from a checkpoint state dict."""
        return {
            k[len(prefix) :]: v
            for k, v in checkpoint_dict.items()
            if k.startswith(prefix)
        }

    def load_from_checkpoint(
        self, checkpoint_dict, lmm_flag=True, input_flag=True, strict=True
    ):
        """Load LMM and/or input layer weights from a filtered checkpoint dict."""
        if lmm_flag:
            lmm_dict = self.filter_ckpt_prefix(checkpoint_dict, "lmm.")
            self.lmm.load_state_dict(lmm_dict, strict=strict)
        if input_flag:
            input_dict = self.filter_ckpt_prefix(checkpoint_dict, "input_layer.")
            self.input_layer.load_state_dict(input_dict, strict=strict)

    # ─────────────────────────────────────────────────────────────────
    # V2-specific methods
    # ─────────────────────────────────────────────────────────────────

    def _encode_dino_spatial(self, batch):
        """Encode images from batch and return spatial feature maps + flat tokens.

        Args:
            batch: Batch dict. If ``"images"`` key is present, encodes
                them with DINO. Otherwise returns None.

        Returns:
            Tuple of (dino_spatial, dino_flat_tokens, dino_flat_mask)
            or (None, None, None) if no images or DINO is disabled.
            - dino_spatial: (B, V, grid_h, grid_w, latent_dim)
            - dino_flat_tokens: (B, V*grid_h*grid_w, latent_dim) for add_tokens
            - dino_flat_mask: (B, V*grid_h*grid_w) bool
        """
        if not self.use_dino:
            return None, None, None
        if "images" not in batch:
            return None, None, None

        images = batch["images"]  # (B, V, 3, H, W) float [0, 1]

        # Apply ImageNet normalization
        images = (images - self.img_mean) / self.img_std  # (B, V, 3, H, W)

        view_ids = batch.get(
            "view_ids",
            torch.zeros(
                images.shape[0],
                images.shape[1],
                dtype=torch.long,
                device=images.device,
            ),
        )  # (B, V)

        dino_spatial, dino_flat_tokens, dino_flat_mask = (
            self.dino_encoder.encode_spatial(images, view_ids)
        )
        return dino_spatial, dino_flat_tokens, dino_flat_mask

    def _query_dino_at_pixels(self, dino_spatial, batch):
        """Query DINO spatial features at pixel UV locations for object parts.

        Args:
            dino_spatial: (B, V, grid_h, grid_w, latent_dim) from _encode_dino_spatial.
            batch: Batch dict containing pixel_uv_normalized, pixel_view_idx,
                pixel_query_mask.

        Returns:
            dino_geo_emb: (B, P, latent_dim) DINO features at pixel locations,
                or None if no spatial features or no pixel query data.
        """
        if dino_spatial is None:
            return None
        if "pixel_uv_normalized" not in batch:
            return None

        dino_geo_emb = self.dino_encoder.query_at_pixels(
            dino_spatial,
            batch["pixel_uv_normalized"],  # (B, P, 2)
            batch["pixel_view_idx"],  # (B, P)
            batch["pixel_query_mask"],  # (B, P)
        )  # (B, P, latent_dim)
        return dino_geo_emb

    def forward(self, batch, phase, global_step=0):
        """Forward pass for training and validation (Non-DF masked conditioned DDPM).

        Key differences from v6_all:
        - ONE global noise level per sample (standard DDPM, not per-element)
        - Conditioning mask determines known/unknown positions
        - LMM input is [X; cond_X; mask] (19 channels)
        - AdaLN gets global_t_emb + per-token cond_proj([cond_X, mask])
        - Loss on ALL valid positions (no per-element SNR weighting)

        Args:
            batch: Batch data dict.
            phase: ``"train"`` or ``"val"``.
            global_step: Current global training step.

        Returns:
            Dict with ``"loss"``, ``"log_dict"``, and predicted values.
        """
        ret = {}

        # Preprocess the batch
        batch, enforce_known_mask = self.handle_play_policy_data(batch)

        # Prepare input
        B, T, P = batch["trace_mask"].shape
        C = batch["context_mask"].shape[1]
        device = batch["trace_mask"].device

        timestep = batch["time"]
        trace = batch["trace"].clone()  # (B, T, P, 4, 4)
        context = batch["context"].clone()  # (B, C, P, 4, 4)
        trace_mask = batch["trace_mask"].clone()  # (B, T, P)
        context_mask = batch["context_mask"].clone()  # (B, C, P)

        sem_type = batch["trace_sem_type"].clone()  # (B, P)
        group_id = batch["trace_group_id"].clone()  # (B, P)
        rot_mask = batch["trace_rot_mask"].clone()  # (B, P)

        local_pcl = batch["local_pcl"].clone()  # (B, P, PCL, 3)
        local_pcl_mask = batch["local_pcl_mask"].clone()  # (B, P)

        if self.training and self.ctx_record_flag:
            self.record_ctx(context, context_mask, batch["trace_uid"])

        # Randomly aug the trace_mask and ctx_mask
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
        )  # (B, T, P, 9)
        context_param = self.pose_io.encode(
            context, sem_ids=sem_type[:, None].expand(-1, C, -1), stat_mask=context_mask
        )  # (B, C, P, 9)

        diffusion_x = torch.cat(
            [context_param, trace_param], dim=1
        ).clone()  # (B, C+T, P, 9)
        noise = torch.randn_like(diffusion_x, device=device)

        # Per-frame validity mask (for supervision)
        lmm_mask = torch.cat([context_mask, trace_mask], dim=1)  # (B, C+T, P)
        # Trajectory-level padding mask
        traj_padding_mask = lmm_mask.any(dim=1)  # (B, P)

        # === NON-DF: Sample ONE global noise level per sample ===
        global_t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()  # (B,)
        # Broadcast to (B, C+T, P) — all positions share the same t
        df_timesteps = global_t[:, None, None].expand(B, C + T, P).clone()

        # Invalid frames → full noise
        max_ts = self.noise_scheduler.config.num_train_timesteps - 1
        invalid_frames = ~lmm_mask  # (B, C+T, P)
        df_timesteps[invalid_frames] = max_ts
        diffusion_x[invalid_frames] = noise[invalid_frames]

        # === NON-DF: Add noise with the GLOBAL timestep ===
        # All positions get the same noise level
        noisy_x = self.noise_scheduler.add_noise(
            original_samples=diffusion_x.reshape(B * (C + T) * P, -1),
            noise=noise.reshape(B * (C + T) * P, -1),
            timesteps=df_timesteps.reshape(B * (C + T) * P),
        ).reshape(B, C + T, P, -1)  # (B, C+T, P, 9)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            diffusion_tgt = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            diffusion_tgt = self.noise_scheduler.get_velocity(
                diffusion_x.reshape(B * (C + T) * P, -1),
                noise.reshape(B * (C + T) * P, -1),
                df_timesteps.reshape(B * (C + T) * P),
            ).reshape(B, C + T, P, -1)
        elif self.noise_scheduler.config.prediction_type == "sample":
            diffusion_tgt = diffusion_x
        else:
            raise ValueError(
                f"Prediction type {self.noise_scheduler.config.prediction_type} not supported"
            )

        # === NON-DF: Sample conditioning mask ===
        cond_known_mask = self.__prepare_conditioning_mask__(
            B, C, T, P, device, enforce_known_mask, sem_type=sem_type,
        )  # (B, C+T, P) bool — True=known

        # === NON-DF: Build cond_X tensor ===
        # Known positions get real data, unknown get learnable embedding
        # CRITICAL: use same dtype as diffusion_x to avoid AMP dtype mismatch
        unknown_emb = self.unknown_cond_emb.to(dtype=diffusion_x.dtype)  # (9,)
        cond_X = torch.where(
            cond_known_mask[..., None],  # (B, C+T, P, 1)
            diffusion_x,  # real clean data for known positions
            unknown_emb[None, None, None, :].expand_as(diffusion_x),  # learnable for unknown
        )  # (B, C+T, P, 9)

        # Build mask tensor (B, C+T, P, 1) float for concatenation
        mask_tensor = cond_known_mask.unsqueeze(-1).to(dtype=diffusion_x.dtype)  # (B, C+T, P, 1)

        # Mask the rot components on noisy_x AND cond_X (zero out rotation
        # channels for position-only parts). In v6_all, the entire lmm_input
        # is masked; here we apply separately to both 9-channel components.
        rot9_mask = self.pose_io.rot_mask(rot_mask)  # (B, P, 9)
        rot9_mask_expanded = rot9_mask[:, None].float()  # (B, 1, P, 9)
        noisy_x = noisy_x * rot9_mask_expanded
        cond_X = cond_X * rot9_mask_expanded

        # === NON-DF: Concatenate [X; cond_X; mask] for LMM input ===
        lmm_input = torch.cat([noisy_x, cond_X, mask_tensor], dim=-1)  # (B, C+T, P, 19)

        # Classifier-free mask on text
        text_masked = torch.rand(B) < self.text_masked_prob
        text = [batch["text"][i] if not text_masked[i] else "" for i in range(B)]
        lang_tk = self.input_layer.encode_language(text)  # (B, L, latent_dim)

        # --- Promote images & pixel-query arrays from extra dict to batch tensors ---
        if "images" not in batch and "extra" in batch:
            extra = batch["extra"]
            if "images" in extra and extra["images"][0] is not None:
                batch["images"] = torch.as_tensor(
                    np.stack(extra["images"]), dtype=torch.float32
                ).to(device)  # (B, V, 3, H, W)
            if "view_ids" in extra and extra["view_ids"][0] is not None:
                batch["view_ids"] = torch.as_tensor(
                    np.stack(extra["view_ids"])
                ).to(device)
            if "pixel_uv_normalized" in extra and extra["pixel_uv_normalized"][0] is not None:
                padded_uv, padded_view, padded_qmask = [], [], []
                for bid in range(B):
                    uv_i = extra["pixel_uv_normalized"][bid]
                    vi_i = extra["pixel_view_idx"][bid]
                    qm_i = extra["pixel_query_mask"][bid]
                    P_raw = len(uv_i)
                    pad = P - P_raw
                    if pad > 0:
                        uv_i = np.concatenate([uv_i, np.zeros((pad, 2), dtype=np.float32)], 0)
                        vi_i = np.concatenate([vi_i, np.zeros(pad, dtype=np.int64)], 0)
                        qm_i = np.concatenate([qm_i, np.zeros(pad, dtype=bool)], 0)
                    elif pad < 0:
                        uv_i, vi_i, qm_i = uv_i[:P], vi_i[:P], qm_i[:P]
                    padded_uv.append(uv_i)
                    padded_view.append(vi_i)
                    padded_qmask.append(qm_i)
                batch["pixel_uv_normalized"] = torch.as_tensor(
                    np.stack(padded_uv), dtype=torch.float32).to(device)
                batch["pixel_view_idx"] = torch.as_tensor(
                    np.stack(padded_view), dtype=torch.long).to(device)
                batch["pixel_query_mask"] = torch.as_tensor(
                    np.stack(padded_qmask), dtype=torch.bool).to(device)

        # Encode DINO spatial + pixel query
        dino_spatial, dino_flat_tokens, dino_flat_mask = self._encode_dino_spatial(batch)
        dino_geo_emb = self._query_dino_at_pixels(dino_spatial, batch)

        feat = self.input_layer(
            sem_type,
            group_id,
            local_pcl,
            local_pcl_mask,
            external_geo_emb=dino_geo_emb,
        )  # (B, P, feat_dim)

        # Project lang and image tokens
        lang_proj = self.lang_token_proj(lang_tk.clone())  # (B, L, C_P)
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)
        if self.use_dino and dino_flat_tokens is not None:
            img_proj = self.image_token_proj(dino_flat_tokens.clone())
        else:
            img_proj = None
            dino_flat_mask = None

        # Attention masks
        attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
        attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

        out = self.lmm(
            x=lmm_input[:, C:],
            x_mask=attn_mask_t,
            x_time=timestep,
            x_level=df_timesteps[:, C:],
            feat=feat,
            ctx=lmm_input[:, :C],
            ctx_mask=attn_mask_c,
            ctx_level=df_timesteps[:, :C],
            add_tokens=lang_proj,
            add_tokens_mask=lang_mask,
            image_tokens=img_proj,
            image_tokens_mask=dino_flat_mask,
            # Non-DF conditioning for AdaLN
            cond_x=cond_X,
            cond_mask=mask_tensor,
        )  # (B, C+T, P, 9)
        out = self.x0_to_mode(out, noisy_x, df_timesteps)

        # NaN guard
        if not out.isfinite().all():
            n_bad = (~out.isfinite()).sum().item()
            raise RuntimeError(
                f"NaN/Inf in LMM output: {n_bad}/{out.numel()} elements "
                f"({100*n_bad/out.numel():.1f}%). Fix the source of instability."
            )

        # Supervise the output
        log_dict = {}
        loss = (out * 0.0).mean()

        # Compute x0 prediction for viz
        x0_pred = compute_x0(
            scheduler=self.noise_scheduler,
            pred=out,
            time_steps=df_timesteps,
            noisy_x=noisy_x,
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

        tgt_t = tgt[..., :3]  # (B, C+T, P, 3)
        tgt_q = tgt[..., 3:]  # (B, C+T, P, 6)
        pred_t = out[..., :3]
        pred_q = out[..., 3:]

        t_sup_mask = lmm_mask.bool()  # (B, C+T, P)
        q_sup_mask = lmm_mask.bool() * rot9_mask[:, None, :, -1].bool()
        v_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        v_mask[:, :C] = True

        t_r_sup_mask_se3 = t_sup_mask & ~v_mask & rot9_mask[:, None, :, -1].bool()
        t_r_sup_mask_xyz = t_sup_mask & ~v_mask & ~rot9_mask[:, None, :, -1].bool()
        t_v_sup_mask_se3 = t_sup_mask & v_mask & rot9_mask[:, None, :, -1].bool()
        t_v_sup_mask_xyz = t_sup_mask & v_mask & ~rot9_mask[:, None, :, -1].bool()

        q_r_sup_mask = q_sup_mask & ~v_mask
        q_v_sup_mask = q_sup_mask & v_mask

        # Non-DF: uniform loss weight (one global t → one SNR for whole sample)
        # No per-element SNR weighting needed
        loss_ele_weight = torch.ones(B, C + T, P, device=device, dtype=pred_t.dtype)

        if self.movement_reweight_min_ratio > 0:
            with torch.no_grad():
                mv_pos = batch["trace"][..., :3, 3]
                t_mask = batch["trace_mask"]
                first_valid_t = t_mask.float().argmax(dim=1)
                first_valid_pos = torch.gather(
                    mv_pos, 1,
                    first_valid_t.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 3),
                ).squeeze(1)
                ref = first_valid_pos.unsqueeze(1)
                mv_pos_safe = torch.where(t_mask.unsqueeze(-1), mv_pos, ref)
                mv_disp = (mv_pos_safe - ref).pow(2).sum(dim=-1).clamp(min=1e-12).sqrt()
                mv_per_track = mv_disp.max(dim=1).values
                mv_per_track = mv_per_track * self.movement_scale
                mv_max = mv_per_track.max(dim=1, keepdim=True).values
                mv_per_track = mv_per_track + mv_max * self.movement_reweight_min_ratio
                mv_sum = mv_per_track.sum(dim=1, keepdim=True).clamp(min=1e-8)
                movement_weight = P * mv_per_track / mv_sum
                movement_weight = movement_weight.clamp(max=self.movement_reweight_max_ratio)
                mw_sum = movement_weight.sum(dim=1, keepdim=True)
                if torch.isinf(mw_sum).any() or torch.isnan(mw_sum).any():
                    movement_weight = torch.ones(B, P, device=device, dtype=mv_pos.dtype)
                else:
                    movement_weight = P * movement_weight / mw_sum.clamp(min=1e-8)
            mw_ctx = torch.ones(B, C, P, device=device, dtype=loss_ele_weight.dtype)
            mw_trace = movement_weight.unsqueeze(1).expand(-1, T, -1)
            mw_full = torch.cat([mw_ctx, mw_trace], dim=1)
            loss_ele_weight = loss_ele_weight * mw_full

        loss_t = self.loss_kernel(pred_t, tgt_t)
        loss_q = self.loss_kernel(pred_q, tgt_q)

        loss_t = loss_t * loss_ele_weight
        loss_q = loss_q * loss_ele_weight

        T_wl_gt = torch.cat([context, trace], dim=1)
        R_wl_pred = T_wl_pred[..., :3, :3]
        R_wl_gt = T_wl_gt[..., :3, :3]
        R_loss = torch.einsum("...ji,...jk->...ik", R_wl_pred, R_wl_gt)
        R_loss = 1.0 - (R_loss[..., 0, 0] + R_loss[..., 1, 1] + R_loss[..., 2, 2]) / 3.0

        if self.loss_balance_robot_scene:
            _is_scene_per_part = torch.zeros(B, P, device=device, dtype=torch.bool)
            for _st in self._scene_sem_types:
                _is_scene_per_part = _is_scene_per_part | (sem_type == _st)
            is_scene_expanded = _is_scene_per_part[:, None, :].expand(B, C + T, P)

        def _add_loss_term(loss_vals, sup_mask, weight, tag):
            nonlocal loss, log_dict
            if sup_mask.sum() == 0:
                return
            _pad_uids = batch["trace_uid"][:, None].expand_as(sup_mask).long()
            vals_flat = loss_vals[sup_mask]
            uids_flat = _pad_uids[sup_mask]
            if not self.loss_balance_robot_scene:
                term = type_mean(vals_flat, uids_flat, self.gid_w_modifier)
                loss = loss + weight * term
                log_dict[f"{phase}_LOSS/{tag}"] = term.detach()
            else:
                scene_flat = is_scene_expanded[sup_mask]
                robot_term, scene_term = type_mean_balanced(
                    vals_flat, uids_flat, scene_flat, self.gid_w_modifier
                )
                if robot_term is not None and scene_term is not None:
                    term = 0.5 * robot_term + 0.5 * scene_term
                elif robot_term is not None:
                    term = robot_term
                else:
                    term = scene_term
                loss = loss + weight * term
                log_dict[f"{phase}_LOSS/{tag}"] = term.detach()
                if robot_term is not None:
                    log_dict[f"{phase}_LOSS/{tag}_robot"] = robot_term.detach()
                if scene_term is not None:
                    log_dict[f"{phase}_LOSS/{tag}_scene"] = scene_term.detach()

        _add_loss_term(loss_t, t_r_sup_mask_se3, self.w_t_se3 * self.w_real, "t_real_se3")
        _add_loss_term(loss_t, t_r_sup_mask_xyz, self.w_t_xyz * self.w_real, "t_real_xyz")
        _add_loss_term(loss_q, q_r_sup_mask, self.w_q * self.w_real, "q_real")
        _add_loss_term(R_loss, q_r_sup_mask, self.w_R * self.w_real, "R_real")

        _add_loss_term(loss_t, t_v_sup_mask_se3, self.w_t_se3 * self.w_virtual, "t_virtual_se3")
        _add_loss_term(loss_t, t_v_sup_mask_xyz, self.w_t_xyz * self.w_virtual, "t_virtual_xyz")
        _add_loss_term(loss_q, q_v_sup_mask, self.w_q * self.w_virtual, "q_virtual")
        _add_loss_term(R_loss, q_v_sup_mask, self.w_R * self.w_virtual, "R_virtual")

        if self.loss_balance_robot_scene:
            robot_total = sum(
                v for k, v in log_dict.items()
                if k.endswith("_robot") and k.startswith(f"{phase}_LOSS/")
            )
            scene_total = sum(
                v for k, v in log_dict.items()
                if k.endswith("_scene") and k.startswith(f"{phase}_LOSS/")
            )
            if isinstance(robot_total, torch.Tensor):
                log_dict[f"{phase}_LOSS/balanced_robot_total"] = robot_total.detach()
            if isinstance(scene_total, torch.Tensor):
                log_dict[f"{phase}_LOSS/balanced_scene_total"] = scene_total.detach()

        log_dict[f"{phase}_LOSS/total"] = loss.detach()
        log_dict[f"total_loss/{phase}"] = loss.detach()
        ret["loss"] = loss
        ret["log_dict"] = log_dict

        ret["x0_pred_context"] = T_wl_pred[:, :C].detach()
        ret["x0_pred_trace"] = T_wl_pred[:, C:].detach()
        if self.pose_io_debug_flag:
            self.pose_io.print_stats()

        log_dict[f"{phase}_STATS/pose_io_ave_mean"] = (
            self.pose_io.running_mean.detach().mean()
        )
        log_dict[f"{phase}_STATS/pose_io_ave_std"] = (
            self.pose_io.running_var.detach().sqrt().mean()
        )

        return ret

    @torch.inference_mode()
    def inference(
        self,
        # X
        trace,  # (B, T, P, 4, 4)
        context,  # (B, C, P, 4, 4)
        # WHAT TO Generate
        trace_known_mask,  # (B, T, P)
        context_known_mask,  # (B, C, P)
        trace_mask,  # (B, T, P)
        context_mask,  # (B, C, P)
        # ATTR
        real_timestep,  # (B, T)
        group_id,  # (B, P)
        sem_type,  # (B, P)
        rot_mask,  # (B, P)
        # DIFFUSION
        condition_timestep: int = 0,
        condition_ctx_timestep: int = -1,
        condition_trace_timestep: int = -1,
        seed=42,
        verbose=True,
        #
        local_pcl=None,  # (B, P, PCL, 3)
        local_pcl_mask=None,  # (B, P)
        text=None,
        #
        add_cfg: float = -1.0,
        #
        left2right_AR_sweep_flag=False,
        # V2 additions
        images=None,  # (B, V, 3, H, W)
        view_ids=None,  # (B, V)
        pixel_uv_normalized=None,  # (B, P, 2)
        pixel_view_idx=None,  # (B, P)
        pixel_query_mask=None,  # (B, P)
    ):
        """Run DDIM inference without gradient tracking.

        Thin wrapper around :meth:`inference_with_grad` decorated with
        ``@torch.inference_mode()``.  See that method for full argument docs.

        Returns:
            Tuple of ``(ret_context, ret_trace)`` SE3 transforms.
        """
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
            images=images,
            view_ids=view_ids,
            pixel_uv_normalized=pixel_uv_normalized,
            pixel_view_idx=pixel_view_idx,
            pixel_query_mask=pixel_query_mask,
        )

    def inference_with_grad(
        self,
        # X
        trace,  # (B, T, P, 4, 4)
        context,  # (B, C, P, 4, 4)
        # WHAT TO Generate
        trace_known_mask,  # (B, T, P)
        context_known_mask,  # (B, C, P)
        trace_mask,  # (B, T, P)
        context_mask,  # (B, C, P)
        # ATTR
        real_timestep,  # (B, T)
        group_id,  # (B, P)
        sem_type,  # (B, P)
        rot_mask,  # (B, P)
        # DIFFUSION
        condition_timestep: int = 0,
        condition_ctx_timestep: int = -1,
        condition_trace_timestep: int = -1,
        seed=42,
        verbose=True,
        #
        local_pcl=None,  # (B, P, PCL, 3)
        local_pcl_mask=None,  # (B, P)
        text=None,
        #
        add_cfg: float = -1.0,
        #
        left2right_AR_sweep_flag=False,
        # V2 additions
        images=None,  # (B, V, 3, H, W)
        view_ids=None,  # (B, V)
        pixel_uv_normalized=None,  # (B, P, 2)
        pixel_view_idx=None,  # (B, P)
        pixel_query_mask=None,  # (B, P)
    ):
        """Run standard DDIM inference with mask-based conditioning (Non-DF).

        Each DDIM step uses ONE global noise level. Known positions get real
        data in cond_X; unknown positions get the learnable embedding.
        The model predicts x0 for ALL positions but only unknown positions
        are updated via the DDIM step.

        Returns:
            Tuple of (ret_context, ret_trace) SE3 transforms.
        """
        is_training = self.training
        if is_training:
            logging.warning("Change to evaluation mode")
            self.eval()

        B, T, P = trace.shape[:3]
        C = context.shape[1]
        C_T = C + T
        assert trace.shape == (B, T, P, 4, 4)
        assert context.shape == (B, C, P, 4, 4)

        scheduler = self.gen_scheduler
        device = trace.device

        x_input = torch.cat([context, trace], dim=1)  # (B, C+T, P, 4, 4)
        known_mask = torch.cat(
            [context_known_mask, trace_known_mask], dim=1
        ).bool()  # (B, C+T, P)

        x_mask = torch.cat([context_mask, trace_mask], dim=1)  # (B, C+T, P)
        traj_padding_mask = x_mask.any(dim=1)  # (B, P)
        attn_mask_inf = traj_padding_mask[:, None].expand(-1, C_T, -1)  # (B, C+T, P)

        x_param = self.pose_io.encode(
            x_input, sem_ids=sem_type[:, None].expand(-1, C + T, -1), stat_mask=None
        )  # (B, C+T, P, 9)
        rot_mask_multiplier = self.pose_io.rot_mask(rot_mask)[
            :, None
        ].float()  # (B, 1, P, 9)

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise = torch.randn(x_param.shape, device=device, generator=g)  # (B, C+T, P, 9)

        # Initialize x: known positions get clean data, unknown get noise
        x = x_param.clone() * known_mask.float()[..., None] + noise * (
            1.0 - known_mask.float()[..., None]
        )

        if text is None:
            text = [""] * B
        lang_tk = self.input_layer.encode_language(text)

        # Encode DINO if available
        if self.use_dino and images is not None:
            inf_batch = {"images": images}
            if view_ids is not None:
                inf_batch["view_ids"] = view_ids
            if pixel_uv_normalized is not None:
                inf_batch["pixel_uv_normalized"] = pixel_uv_normalized
                inf_batch["pixel_view_idx"] = pixel_view_idx
                inf_batch["pixel_query_mask"] = pixel_query_mask
            dino_spatial, dino_flat_tokens, dino_flat_mask = self._encode_dino_spatial(inf_batch)
            dino_geo_emb = self._query_dino_at_pixels(dino_spatial, inf_batch)
        else:
            dino_flat_tokens, dino_flat_mask = None, None
            dino_geo_emb = None

        lang_proj = self.lang_token_proj(lang_tk.clone())
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)
        if self.use_dino and dino_flat_tokens is not None:
            img_proj = self.image_token_proj(dino_flat_tokens.clone())
        else:
            img_proj = None
            dino_flat_mask = None

        empty_lang_proj = self.lang_token_proj(
            self.input_layer.encode_language([""] * B).clone()
        )
        empty_lang_mask = torch.ones(B, empty_lang_proj.shape[1], device=device, dtype=torch.bool)

        feat = self.input_layer(
            sem_type, group_id, local_pcl, local_pcl_mask,
            external_geo_emb=dino_geo_emb,
        )

        # === NON-DF: Build cond_X and mask for inference ===
        # Known positions: real data. Unknown: learnable embedding.
        unknown_emb = self.unknown_cond_emb.to(dtype=x_param.dtype)
        cond_X = torch.where(
            known_mask[..., None],
            x_param,
            unknown_emb[None, None, None, :].expand_as(x_param),
        )  # (B, C+T, P, 9)
        # Apply rot_mask to cond_X (zero rotation channels for position-only parts)
        cond_X = cond_X * rot_mask_multiplier
        mask_tensor = known_mask.unsqueeze(-1).to(dtype=x_param.dtype)  # (B, C+T, P, 1)

        if verbose:
            _iter = tqdm(scheduler.timesteps)
        else:
            _iter = scheduler.timesteps

        for t in _iter:  # [999, ... 1, 0]
            # Global t broadcast to all positions
            df_t = torch.ones(B, C_T, P, device=device, dtype=torch.long) * int(t)
            # Known positions: set to clean (t=0)
            df_t[known_mask] = 0

            # Overwrite known positions with clean data
            x[known_mask] = x[known_mask] * 0.0 + x_param[known_mask].clone()
            x = x * rot_mask_multiplier.clone()

            # Build 19-channel input: [x; cond_X; mask]
            lmm_input = torch.cat([x, cond_X, mask_tensor], dim=-1)  # (B, C+T, P, 19)

            pred = self.lmm(
                x=lmm_input[:, C:],
                x_mask=attn_mask_inf[:, C:],
                x_time=real_timestep,
                x_level=df_t[:, C:],
                feat=feat,
                ctx=lmm_input[:, :C],
                ctx_mask=attn_mask_inf[:, :C],
                ctx_level=df_t[:, :C],
                add_tokens=lang_proj.clone(),
                add_tokens_mask=lang_mask.clone(),
                image_tokens=img_proj,
                image_tokens_mask=dino_flat_mask,
                cond_x=cond_X,
                cond_mask=mask_tensor,
            )
            pred = self.x0_to_mode(pred, x, df_t)

            if add_cfg > 0:
                pred_uncond = self.lmm(
                    x=lmm_input[:, C:],
                    x_mask=attn_mask_inf[:, C:],
                    x_time=real_timestep,
                    x_level=df_t[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=attn_mask_inf[:, :C],
                    ctx_level=df_t[:, :C],
                    add_tokens=empty_lang_proj.clone(),
                    add_tokens_mask=empty_lang_mask.clone(),
                    image_tokens=img_proj,
                    image_tokens_mask=dino_flat_mask,
                    cond_x=cond_X,
                    cond_mask=mask_tensor,
                )
                pred_uncond = self.x0_to_mode(pred_uncond, x, df_t)
                pred = pred_uncond + (pred - pred_uncond) * float(add_cfg)

            # Standard DDIM step — only update unknown positions
            _prev_x = scheduler.step(pred, int(t), x).prev_sample
            x[~known_mask] = x[~known_mask] * 0.0 + _prev_x[~known_mask].clone().to(x)

        # Final: known positions get clean data
        x[known_mask] = x[known_mask] * 0.0 + x_param[known_mask].clone().to(x)
        x = self.pose_io.nonsigular_empty(x, rot_mask_multiplier)

        ret = self.pose_io.decode(
            x, sem_ids=sem_type[:, None].expand(-1, C + T, -1)
        )  # (B, C+T, P, 4, 4)
        ret_context = ret[:, :C]
        ret_trace = ret[:, C:]

        if is_training:
            logging.warning("Reverting to training mode")
            self.train()

        return ret_context, ret_trace

    def get_optim_groups(self, weight_decay, lmm_lr, input_layer_lr, image_encoder_lr):
        """Get optimizer parameter groups with per-component learning rates.

        Handles optional DINO encoder — only includes its params when enabled.

        Args:
            weight_decay: Weight decay for all groups.
            lmm_lr: Learning rate for the world model (LMM).
            input_layer_lr: Learning rate for the input encoder.
            image_encoder_lr: Learning rate for DINO encoder trainable params.

        Returns:
            List of parameter group dicts for ``torch.optim``.
        """
        # Token projectors go with input_layer_lr
        token_proj_params = list(self.lang_token_proj.parameters())
        if self.use_dino:
            token_proj_params += list(self.image_token_proj.parameters())
        token_proj_group = [
            {
                "params": token_proj_params,
                "weight_decay": weight_decay,
                "lr": input_layer_lr,
            }
        ]
        # Non-DF: unknown_cond_emb is a direct Agent parameter, not inside any sub-module.
        # It must be explicitly included or it won't be optimized.
        nondf_params_group = [
            {
                "params": [self.unknown_cond_emb],
                "weight_decay": 0.0,  # 1-D param, no weight decay
                "lr": lmm_lr,  # same LR as LMM since it feeds into LMM conditioning
            }
        ]
        groups = (
            self.lmm.get_optim_groups(weight_decay=weight_decay, base_lr=lmm_lr)
            + self.input_layer.get_optim_groups(
                weight_decay=weight_decay, base_lr=input_layer_lr
            )
            + token_proj_group
            + nondf_params_group
        )
        if self.use_dino:
            groups += self.dino_encoder.get_optim_groups(
                weight_decay=weight_decay, base_lr=image_encoder_lr
            )
        return groups

    @staticmethod
    def get_type_mask(uids: torch.Tensor, target_uids) -> torch.Tensor:
        """Return a boolean mask where uids match any of target_uids.

        Args:
            uids: (N,) int tensor of UIDs.
            target_uids: int, or list/tuple of ints to match against.

        Returns:
            (N,) bool tensor.
        """
        if isinstance(target_uids, (int, float)):
            target_uids = [int(target_uids)]
        mask = torch.zeros_like(uids, dtype=torch.bool)
        for uid in target_uids:
            mask |= uids == uid
        return mask

    def convert_smplx_hand_param(self, Q, to_local=True):
        """Convert SMPL-X hand joint transforms between local and global space.

        Transforms the 15 left-hand and 15 right-hand joints relative to
        (or back from) their respective wrist frames. Assumes the standard
        SMPL-X 55-joint ordering.

        Args:
            Q: Joint transforms of shape ``(B, 55, 4, 4)``.
            to_local: If True, convert global hand joints to wrist-local
                space. If False, convert from local back to global.

        Returns:
            Modified transforms of shape ``(B, 55, 4, 4)``.
        """
        assert Q.shape[1] == 55
        Q = Q.clone()

        with torch.autocast(device_type=Q.device.type, enabled=False):
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

        return Q
