"""AgentV14: SIM(3) canonicalization + 6D rotation + unified noise schedulers.

V11 changes from V10:
- x0net + converter: model ALWAYS outputs x0, converter produces target-space output
- 50/50 per-sample noise branching (clean-known vs async)
- Known tokens get x_level=0 during inference (THE critical v10 bug fix)
- Linear loss scaling (s_linear not s_squared)
- No 3-cat input — simple projection of noisy_x only
- Supports DDPM/FM/MeanFlow in one model

Example::

    from umm.models.umm_v14.agent import AgentV14
    agent = AgentV14(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from umm.models.umm_v14.rotation_utils import se3_to_9d, ninep_to_se3, sixd_to_rotmat
from umm.models.umm_v14.canonicalization import (
    Canonicalizer,
    CanonParams,
    compute_canon_params,
    canonicalize,
)
from umm.models.umm_v14.text_codebook_head import (
    TextCodebookHead as _TextCodebookHead,
    tokenize_with_eos_fill as _tokenize_with_eos_fill,
)
from umm.models.umm_v14.diffusion import (
    FlowMatchingScheduler,
    DDPMWrapper,
    MeanFlowScheduler,
    build_diffusion_scheduler,
    compute_x0,
)

# Shared infrastructure — truly generic, cross-version utilities from umm_v9
from umm.models.umm_v9.world_models.language import T5LanguageEncoder
from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    disable_model_grad,
    safe_sqrt,
)
from umm.models.umm_v9.world_models.pointnet import (
    ResnetPointnetV2,
    ResnetPointnetV3,
)
from umm.models.umm_v9.loss_utils import type_mean_compiled as type_mean

logger = logging.getLogger(__name__)



class InputLayerV14(nn.Module):
    """Input layer with ResnetPointnetV2 (BN), separate DINO proj, configurable group_emb.

    Differences from InputLayerV11:
    - PointNet: ResnetPointnetV2 (BatchNorm + Sigmoid) instead of ResnetPointnet (no BN, ReLU).
    - Separate projection layers: geo_proj for PointNet, dino_proj for DINO features.
    - Configurable skip_group_emb (zeros out group embedding when True).
    - Configurable point_rgb_scale_factor (default 1.0) — applied to rgb channels independently.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 512,
        num_types: int = 1024,
        num_groups: int = 1024,
        lang_fixed_length: int = 1,
        point_encoder_latent: int = -1,
        pcl_input_mode: str = "xyz",
        point_xyz_scale_factor: float = 1.0,
        point_rgb_scale_factor: float = 1.0,
        skip_group_emb: bool = False,
        point_version: str = "v2",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.skip_group_emb = skip_group_emb
        self.point_version = point_version

        self.type_emb = nn.Embedding(num_types, latent_dim)
        self.group_emb = nn.Embedding(num_groups, latent_dim)
        with torch.no_grad():
            self.group_emb.weight.copy_(
                self.group_emb.weight[0].expand_as(self.group_emb.weight)
            )

        self.ln_type_emb = HardLayerNorm(latent_dim)
        self.ln_group_emb = HardLayerNorm(latent_dim)

        # --- PointNet V2 geo embedding (BN + Sigmoid) ---
        self.point_flag = point_encoder_latent > 0
        self.point_xyz_scale_factor = point_xyz_scale_factor
        self.point_rgb_scale_factor = point_rgb_scale_factor
        if pcl_input_mode not in ("xyz", "xyz_rgb"):
            raise ValueError(f"Unknown pcl_input_mode: {pcl_input_mode!r}")
        self.point_input_dim = 3 if pcl_input_mode == "xyz" else 6
        self.empty_point_emb = nn.Embedding(1, latent_dim)
        if self.point_flag:
            if point_version == "v2":
                self.pntnet = ResnetPointnetV2(
                    c_dim=latent_dim, dim=self.point_input_dim,
                    hidden_dim=point_encoder_latent,
                )
            elif point_version == "v3":
                self.pntnet = ResnetPointnetV3(
                    c_dim=latent_dim, dim=self.point_input_dim,
                    hidden_dim=point_encoder_latent,
                )
            else:
                raise ValueError(f"Unknown point_version: {point_version!r}")
            self.geo_proj = nn.Linear(latent_dim, latent_dim)

        # --- Separate DINO projection ---
        self.dino_proj = nn.Linear(latent_dim, latent_dim)

        self.ln_local_feat = HardLayerNorm(latent_dim)

        # type + group + geo -> feat
        self.bind_fc = nn.Linear(3 * latent_dim, feat_dim)

        # Language encoder (frozen T5)
        self.lang_fixed_len = lang_fixed_length
        self.language_encoder = disable_model_grad(
            T5LanguageEncoder(fixed_length=self.lang_fixed_len)
        )
        self.lang_fc = nn.Linear(768, latent_dim)

        if self.skip_group_emb:
            logger.warning("InputLayerV14: skip_group_emb=True")
        logger.info(
            f"InputLayerV14: pcl={pcl_input_mode}, pntnet={point_version}, "
            f"xyz_scale={point_xyz_scale_factor}, rgb_scale={point_rgb_scale_factor}, "
            f"skip_group={skip_group_emb}, point_version={point_version}"
        )

    @torch.compiler.disable
    def encode_language(self, text_in):
        """Encode text -> (B, lang_fixed_len, latent_dim).

        V13 fix: the previous umm_v11 version returned `tokens.detach()`, which
        blocked gradients into `self.lang_fc` and left that Linear(768, latent)
        frozen at random init for the entire training run. This single-line
        change restores `lang_fc` training.

        Safety:
        - T5 itself is still fully frozen. `language_encoder.encode()` wraps
          its T5 forward in `torch.no_grad()` (umm_v9/world_models/language.py),
          and all T5 params have `requires_grad=False` via `disable_model_grad`
          applied in `__init__`. The returned T5 hidden states have
          `requires_grad=False`; only `lang_fc`'s own weights contribute a
          trainable path to the output.
        - Compile safety is unchanged: this method is still wrapped in
          `@torch.compiler.disable`, so Dynamo graph-breaks at the call site
          and runs this block eagerly. The returned tensor re-enters the
          compiled region as an opaque input, and autograd handles the
          eager/compiled boundary during backward.
        - Optimizer collection is unchanged: `lang_fc` lives under
          `self.input_layer.lang_fc` and is already picked up by
          `InputLayerV14.get_optim_groups` (the "language_encoder"
          prefix filter excludes only `self.language_encoder.*`, not
          `self.lang_fc.*`).
        """
        tokens = self.language_encoder.encode(text_in)
        tokens = self.lang_fc(tokens)
        return tokens

    @torch.compiler.disable
    def forward(
        self,
        sem_id: torch.Tensor,
        group_id: torch.Tensor,
        pcl: Optional[torch.Tensor] = None,
        pcl_valid_mask: Optional[torch.Tensor] = None,
        external_geo_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-entity feature from type, group, and geo embeddings.

        Args:
            sem_id: (B, P) semantic type IDs.
            group_id: (B, P) group IDs.
            pcl: (B, P, M, D_pcl) local point clouds, or None.
            pcl_valid_mask: (B, P) bool mask for valid PCL entities, or None.
            external_geo_emb: (B, P, latent_dim) DINO features, or None.

        Returns:
            feat: (B, P, feat_dim).
        """
        B, N = sem_id.shape
        type_emb = self.type_emb(sem_id)  # (B, P, D)

        if self.skip_group_emb:
            group_emb = torch.zeros(
                B, N, self.latent_dim, device=sem_id.device, dtype=type_emb.dtype
            )
        else:
            group_emb = self.group_emb(group_id)  # (B, P, D)

        # Geo embedding: PointNetV2 for valid PCL, empty embedding otherwise
        geo_emb = self.empty_point_emb(
            torch.tensor([0], device=sem_id.device)
        )[None].repeat(B, N, 1).contiguous()  # (B, P, D)

        if self.point_flag and pcl is not None and pcl_valid_mask is not None:
            if pcl_valid_mask.any():
                valid_pcl = pcl[pcl_valid_mask][..., :self.point_input_dim].clone()  # (K, M, dim)
                if self.point_version == "v2":
                    valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
                    if self.point_input_dim == 6:
                        valid_pcl[..., 3:6] = valid_pcl[..., 3:6] * self.point_rgb_scale_factor
                # V3: no external scaling; the network normalizes xyz by per-entity
                # median radial length internally and color is left untouched.
                valid_pcl_feat = self.geo_proj(self.pntnet(valid_pcl))  # (K, D)
                geo_emb[pcl_valid_mask] = (
                    geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
                )

        # DINO path: separate dino_proj for parts WITHOUT valid PCL
        if external_geo_emb is not None:
            if pcl_valid_mask is not None:
                dino_mask = ~pcl_valid_mask  # (B, P)
                if dino_mask.any():
                    geo_emb[dino_mask] = (
                        geo_emb[dino_mask].clone() * 0.0
                        + self.dino_proj(external_geo_emb[dino_mask])
                    )
            else:
                geo_emb = geo_emb + self.dino_proj(external_geo_emb)

        type_emb = self.ln_type_emb(type_emb)  # (B, P, D)
        group_emb = self.ln_group_emb(group_emb)  # (B, P, D)
        geo_emb = self.ln_local_feat(geo_emb)  # (B, P, D)

        feat = self.bind_fc(torch.cat([type_emb, group_emb, geo_emb], dim=-1))
        return feat  # (B, P, feat_dim)

    def get_optim_groups(self, weight_decay: float = 0.0, base_lr: Optional[float] = None):
        """Optimizer param groups (excludes frozen language encoder)."""
        param_dict = {
            pn: p for pn, p in self.named_parameters()
            if p.requires_grad and not pn.startswith("language_encoder")
        }
        decay = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        if base_lr is not None:
            for g in groups:
                g["lr"] = base_lr
        return groups


class AgentV14(nn.Module):
    """V11 Agent: SIM(3) canon + 6D rotation + unified schedulers.

    Args:
        cfg: Full configuration dictionary.
    """

    @staticmethod
    def _resolve_n_ctx(cfg: dict, lmm_cfg: dict) -> int:
        """Derive n_ctx from dataset configs."""
        explicit = lmm_cfg.get("n_ctx", -1)
        dataset_configs = cfg.get("dataset", {}).get("configs", [])
        max_C = max((dc.get("max_C", 0) for dc in dataset_configs), default=0)
        if explicit > 0:
            if max_C > explicit:
                logger.warning(f"n_ctx={explicit} < dataset max_C={max_C}, expanding")
            return max(explicit, max_C)
        return max(max_C, 128) if max_C > 0 else 128

    @staticmethod
    def _resolve_canon_std_scalar(model_cfg: dict) -> float:
        """Resolve target std for codebook normalization.

        Priority chain (per critic R6 + qa Q6 + LT-path extension):
          1. model.text_gen.canon_std (explicit override)
          2. model.text_gen.std_factor (alias)
          3. model.pose_io.std_factor[0] -- assert all entries equal (R6, Q6)
          4. model.canonicalization.target_pos_std (LT path; LT has no pose_io)
          5. Hard fallback 0.5 with WARNING.
        """
        text_gen_cfg = model_cfg.get("text_gen", {}) or {}
        pose_io_cfg = model_cfg.get("pose_io", {}) or {}
        canon_cfg = model_cfg.get("canonicalization", {}) or {}

        explicit = text_gen_cfg.get("canon_std", None)
        source = "text_gen.canon_std"
        if explicit is None:
            explicit = text_gen_cfg.get("std_factor", None)
            source = "text_gen.std_factor"
        if explicit is None:
            explicit = pose_io_cfg.get("std_factor", None)
            source = "pose_io.std_factor"
        if explicit is None:
            explicit = canon_cfg.get("target_pos_std", None)
            source = "canonicalization.target_pos_std"
        if explicit is None:
            logger.warning(
                "[v14 codebook] no canon_std source found "
                "(checked text_gen.canon_std/std_factor, pose_io.std_factor, "
                "canonicalization.target_pos_std); falling back to 0.5. "
                "Set text_gen.canon_std explicitly to silence this warning."
            )
            return 0.5
        if isinstance(explicit, (list, tuple)):
            first = float(explicit[0])
            for v in explicit:
                if abs(float(v) - first) > 1e-9:
                    raise ValueError(
                        f"canon_std source '{source}' must be uniform when passed "
                        f"as a list; got {explicit}"
                    )
            logger.info(f"[v14 codebook] canon_std_scalar={first:.4f} from {source}[:]")
            return first
        val = float(explicit)
        logger.info(f"[v14 codebook] canon_std_scalar={val:.4f} from {source}")
        return val

    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})

        # Wrist-local hand coordinates: V9 viz checks this to de-localize
        self.hand_attached_to_wrist_flag = self.model_cfg.get(
            "hand_attached_to_wrist_flag", False
        )
        self.optim_cfg = self.model_cfg.get("optim", {})

        # Dimensions
        self.latent_dim = self.model_cfg.get("latent_dim", 512)
        self.C_P = self.model_cfg.get("lmm", {}).get("C_P", 512)
        self.input_proj_dim = self.model_cfg.get("input_proj_dim", 64)

        # Canonicalization
        canon_cfg = self.model_cfg.get("canonicalization", {})
        self.target_pos_std = canon_cfg.get("target_pos_std", 0.5)
        self.canon_rotation = canon_cfg.get("canon_rotation", True)
        self.canonicalizer = Canonicalizer(target_pos_std=self.target_pos_std)

        # C1: Rotation parameterization — "6d_unit" (default) or "6d_wing"
        self.rot_param = self.model_cfg.get("rot_parameterization", "6d_unit")
        self.wing_len = self.model_cfg.get("rot_wing_len", 0.2)

        # Diffusion
        diffusion_cfg = self.model_cfg.get("diffusion", {})
        self.diffusion = build_diffusion_scheduler(diffusion_cfg)
        self.is_flow_matching = isinstance(self.diffusion, FlowMatchingScheduler)
        self.is_meanflow = isinstance(self.diffusion, MeanFlowScheduler)
        # MeanFlow shares FM code paths (continuous time, x0 recovery formula).
        # Differences: (1) output is avg velocity not x0, (2) r_level conditioning,
        # (3) simplified finite-difference target (JVP incompatible with compile).
        if self.is_meanflow:
            self.is_flow_matching = True
        # Scale for diffusion time → embedding space
        self.diffusion_time_scale = diffusion_cfg.get("time_scale", 1000.0)
        # Single source of truth: gen_timesteps (from diffusion scheduler config).
        # n_inference_steps is an optional override, NOT a separate default.
        self.n_inference_steps = diffusion_cfg.get(
            "n_inference_steps",
            diffusion_cfg.get("gen_timesteps", 50),  # fall back to scheduler's gen_timesteps
        )

        # Loss config
        self.weight_by = self.optim_cfg.get("weight_by", "uid")
        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }

        # Context always known (never noised during training)
        self.ctx_always_known = self.optim_cfg.get("ctx_always_known", False)

        # V13.3.4 two-stage training mask + inference
        self.twostage_training_mask = bool(self.model_cfg.get("twostage_training_mask", False))
        self.twostage_inference = bool(self.model_cfg.get("twostage_inference", False))

        # Loss reweighting (min-SNR, epsilon experiments)
        self.loss_weighting = diffusion_cfg.get("loss_weighting", None)
        self.min_snr_gamma = diffusion_cfg.get("min_snr_gamma", 5.0)

        # V11: LMM ALWAYS outputs x0. Converter transforms x0 to eps/vel/FM-vel.
        # Default: None (no conversion, x0 mode). All configs MUST set this explicitly.
        self.network_output_space = diffusion_cfg.get("network_output_space", None)

        # Loss space: "native" (default) = compute loss in prediction space
        # (eps for eps, vel for vel, x0 for x0).
        # "x0" = convert model output to x0 and compute loss there (Exp 4).
        self.loss_space = diffusion_cfg.get("loss_space", "native")

        # Stable loss: Smooth L1 (Huber) with configurable beta.
        # None = MSE (default for backward compat).
        # float (e.g. 0.1) = smooth_l1_loss(beta=loss_beta).
        self.loss_beta = self.optim_cfg.get("loss_beta", None)

        # R_geodesic loss weight (default 0 = disabled, must set in config)
        self.w_R = self.optim_cfg.get("w_R", 0.0)

        # H1: Virtual/real loss weighting (context frames down-weighted)
        self.w_real = self.optim_cfg.get("w_real", 1.0)
        self.w_virtual = self.optim_cfg.get("w_virtual", 1.0)

        # M5: Movement reweighting — up-weight moving entities
        self.movement_reweight_min_ratio = self.optim_cfg.get(
            "movement_reweight_min_ratio", 0.0
        )
        self.movement_reweight_max_ratio = self.optim_cfg.get(
            "movement_reweight_max_ratio", 10.0
        )
        self.movement_scale = self.optim_cfg.get("movement_scale", 1.0)

        # Warn: loss_space=x0 + min_snr = double 1/SNR weighting
        if self.loss_space == "x0" and self.loss_weighting == "min_snr":
            logger.warning(
                "loss_space='x0' + loss_weighting='min_snr' causes double 1/SNR: "
                "x0 loss has implicit 1/SNR from compute_x0, and min-SNR adds "
                "another SNR-dependent weight. This is probably unintended."
            )

        # --- Input construction ---
        # V11: no agent-level projection. LMM has its own x_proj.

        # --- Input layer (type + group + language) — V13: Super-only ---
        input_cfg = self.model_cfg.get("input_layer", {})
        self.input_layer = InputLayerV14(
            latent_dim=self.latent_dim,
            feat_dim=self.model_cfg.get("feat_dim", 512),
            num_types=input_cfg.get("num_types", 1024),
            num_groups=input_cfg.get("num_groups", 1024),
            lang_fixed_length=input_cfg.get("language_fixed_length", 1),
            point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
            pcl_input_mode=input_cfg.get("pcl_input_mode", "xyz"),
            point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
            point_rgb_scale_factor=input_cfg.get("point_rgb_scale_factor", 1.0),
            skip_group_emb=input_cfg.get("skip_group_emb", False),
            point_version=input_cfg.get("point_version", "v2"),
        )

        # --- LMM (V13: Super or MoE) ---
        lmm_cfg = self.model_cfg.get("lmm", {})
        layer_type_string = lmm_cfg.get("layer_type_string", None)
        lmm_mode = lmm_cfg.get("mode", None)
        assert lmm_mode == "super", (
            f"umm_v14 only supports lmm.mode='super', got {lmm_mode!r}"
        )
        lmm_architecture = lmm_cfg.get("architecture", "super")
        if lmm_architecture == "moe":
            from umm.models.umm_v14.world_models.moma_v11_moe import (
                MotionMachineV11Moe,
            )
            LmmClass = MotionMachineV11Moe
        else:
            from umm.models.umm_v14.world_models.moma_v11_final_super import (
                MotionMachineV14Super,
            )
            LmmClass = MotionMachineV14Super
        self.lmm = LmmClass(
            D1=lmm_cfg.get("D1", 512),
            D2=lmm_cfg.get("D2", 768),
            n_heads=lmm_cfg.get("n_heads", 16),
            fc_factor=lmm_cfg.get("fc_factor", 4),
            fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
            layer_type_string=layer_type_string,
            Rt=lmm_cfg.get("Rt", 4),
            Rp=lmm_cfg.get("Rp", 4),
            feat_dim=self.model_cfg.get("feat_dim", 512),
            input_dim=9,
            n_ctx=self._resolve_n_ctx(cfg, lmm_cfg),
            fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 12),
            n_traj_tokens=lmm_cfg.get("n_traj_tokens", 0),
            time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
            ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", 1.0),
            gate_bound=lmm_cfg.get("gate_bound", 2.0),
            ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
            skip_layer_idx_list=lmm_cfg.get("skip_layer_idx_list", None),
            cond_dim_override=lmm_cfg.get("cond_dim", -1),
            # V13: 2D RoPE knobs (defaults preserve current 1D behavior).
            rope_mode=lmm_cfg.get("rope_mode", "1d_time"),
            entity_rope_multiplier=lmm_cfg.get("entity_rope_multiplier", 0.0),
            # V13.2 LT explore flags (defaults = OLD behavior).
            disable_trajpe=lmm_cfg.get("disable_trajpe", False),
            traj_id_sampler=lmm_cfg.get("traj_id_sampler", "permN"),
            traj_pe_init=lmm_cfg.get("traj_pe_init", "same"),
            # V13.2 V5: per-sample dynamic ctx/reg positions (back-compat False).
            ctx_reg_unique_positions=lmm_cfg.get("ctx_reg_unique_positions", False),
            # V13.2 explore 11: force all-zero traj_ids (dual ablation).
            constant_traj_id=lmm_cfg.get("constant_traj_id", False),
            # V13.2 explore 13/16: per-layer feat fusion MLPs.
            feat_fusion_mlp=lmm_cfg.get("feat_fusion_mlp", False),
            # V13.3 unidirectional ctx attention mask.
            ctx_unidirectional_mask=lmm_cfg.get("ctx_unidirectional_mask", False),
            # V13.7 extend add_token projector+injection to t/p layers.
            extra_add_token_layers=lmm_cfg.get("extra_add_token_layers", ""),
        )

        # Language token projector → LMM's add_token width (Super uses D1).
        _lang_proj_dim = lmm_cfg.get("D1", 512)
        self.lang_token_proj = nn.Linear(self.latent_dim, _lang_proj_dim)

        # --- DINO image encoder (optional — gated on `dino_encoder:` config block) ---
        dino_cfg = self.model_cfg.get("dino_encoder", None)
        self.use_dino = dino_cfg is not None and len(dino_cfg) > 0
        if self.use_dino:
            # Lazy import: defensive against historical DINOv3 dynamo contamination.
            # The dino_encoder module touches torch.hub / sys.path; importing it
            # only when actually needed keeps non-DINO configs clean.
            from umm.models.umm_v14.dino_encoder import (
                DINOImageEncoder,
                IMAGENET_MEAN,
                IMAGENET_STD,
            )
            self.dino_encoder = DINOImageEncoder(
                model_name=dino_cfg.get("model_name", "dinov3_vits16"),
                latent_dim=self.latent_dim,
                num_views=dino_cfg.get("num_views", 8),
                trainable=dino_cfg.get("trainable", False),
            )
            # Project DINO patch tokens into add_tokens dimension (same as lang).
            self.image_token_proj = nn.Linear(self.latent_dim, _lang_proj_dim)
            # Type embedding: 0=text, 1=image. Added to projected tokens to
            # distinguish modality in cross-attention.
            # NOTE: breaks v12 checkpoint compatibility — v12.1 trains from scratch.
            self.add_token_type_emb = nn.Embedding(2, _lang_proj_dim)
            # Project DINO pixel-query features before adding into InputLayer feat.
            # bias=False so masked-out positions (qmask=False → zeroed by
            # dino_encoder.query_at_pixels) project to zero, NOT a learnable
            # bias. Without this, every padded entity would receive a
            # learnable bias contamination via the geo path.
            self.dino_geo_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            # ImageNet normalization buffers: (1, 1, 3, 1, 1) for (B, V, 3, H, W)
            self.register_buffer(
                "img_mean",
                torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1),
            )
            self.register_buffer(
                "img_std",
                torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1),
            )
            logger.info("DINO image encoder ENABLED")
        else:
            self.dino_encoder = None
            self.image_token_proj = None
            self.dino_geo_proj = None
            self.add_token_type_emb = None
            logger.info("DINO image encoder DISABLED (no dino_encoder config)")

        # --- V13.2 V6: optional input cross-attention layer ---
        # Block-A self-attn over [lang || img? || entity], Block-B lang+register
        # global latent, global latent channel-wise concat to every per-entity
        # feat. Default disabled; when disabled the module is NOT instantiated
        # so param count stays identical to base config.
        input_xattn_cfg = self.model_cfg.get("input_xattn", {}) or {}
        self.input_xattn_enable = bool(input_xattn_cfg.get("enable", False))
        if self.input_xattn_enable:
            from umm.models.umm_v14.input_xattn import InputLayerCrossAttn
            self.input_xattn = InputLayerCrossAttn(
                feat_dim=self.model_cfg.get("feat_dim", 512),
                D1=lmm_cfg.get("D1", 512),
                n_layers=int(input_xattn_cfg.get("n_layers", 2)),
                n_register=int(input_xattn_cfg.get("n_register", 2)),
                n_heads=int(input_xattn_cfg.get("n_heads", 8)),
            )
            logger.info(
                f"V13.2 input_xattn ENABLED: n_layers="
                f"{input_xattn_cfg.get('n_layers', 2)} "
                f"n_register={input_xattn_cfg.get('n_register', 2)} "
                f"n_heads={input_xattn_cfg.get('n_heads', 8)}"
            )
        else:
            self.input_xattn = None

        # --- V13.2 Explore 10: optional input_xattn_v2 ---
        input_xattn_v2_cfg = self.model_cfg.get("input_xattn_v2", {}) or {}
        self.input_xattn_v2_enable = bool(input_xattn_v2_cfg.get("enable", False))
        if self.input_xattn_v2_enable:
            assert not self.input_xattn_enable, (
                "input_xattn (V6) and input_xattn_v2 (Explore 10) are mutually exclusive"
            )
            from umm.models.umm_v14.input_xattn_v2 import InputLayerCrossAttnV2
            self.input_xattn_v2 = InputLayerCrossAttnV2(
                feat_dim=self.model_cfg.get("feat_dim", 512),
                D1=lmm_cfg.get("D1", 512),
                n_layers=int(input_xattn_v2_cfg.get("n_layers", 2)),
                n_register=int(input_xattn_v2_cfg.get("n_register", 2)),
                n_heads=int(input_xattn_v2_cfg.get("n_heads", 8)),
                learned_traj_id=bool(input_xattn_v2_cfg.get("learned_traj_id", False)),
                traj_id_bound_K=float(input_xattn_v2_cfg.get("traj_id_bound_K", 1.0)),
                traj_id_K_warmup_start=int(input_xattn_v2_cfg.get("traj_id_K_warmup_start", 0)),
                traj_id_K_warmup_duration=int(input_xattn_v2_cfg.get("traj_id_K_warmup_duration", 0)),
            )
            logger.info(
                f"V13.2 input_xattn_v2 ENABLED: n_layers={input_xattn_v2_cfg.get('n_layers', 2)} "
                f"n_heads={input_xattn_v2_cfg.get('n_heads', 8)} "
                f"learned_traj_id={input_xattn_v2_cfg.get('learned_traj_id', False)} "
                f"K={input_xattn_v2_cfg.get('traj_id_bound_K', 1.0)} "
                f"K_warmup_start={input_xattn_v2_cfg.get('traj_id_K_warmup_start', 0)} "
                f"K_warmup_duration={input_xattn_v2_cfg.get('traj_id_K_warmup_duration', 0)}"
            )
        else:
            self.input_xattn_v2 = None

        # Text masking prob for classifier-free guidance
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)

        # ====================================================================
        # V14 Joint text-track plumbing (codebook + head).
        # The codebook is the FROZEN T5 input-embedding table, normalized once
        # at __init__ to zero per-dim mean and std = canon_std_scalar.
        # ====================================================================
        text_gen_cfg = self.model_cfg.get("text_gen", {})
        self.text_gen_enable = bool(text_gen_cfg.get("enable", False))
        self.text_gen_L_text = int(text_gen_cfg.get("L_text", 32))
        self.text_gen_lambda_text = float(text_gen_cfg.get("lambda_text", 0.0))
        self.text_gen_lambda_round = float(text_gen_cfg.get("lambda_round", 0.0))
        self.text_gen_tau = float(text_gen_cfg.get("tau", 1.0))
        self.text_gen_extra_layers = str(text_gen_cfg.get("extra_text_layers", ""))
        # only wires t/p text injection branches as a no-op (the dispatch
        # explicitly defers t/p to extra_add_token_layers-style follow-up).
        # Refuse non-empty extra_text_layers loudly so a config doesn't silently
        # build dead per-layer projections that consume optimizer state.
        if self.text_gen_extra_layers:
            raise NotImplementedError(
                f"text_gen.extra_text_layers={self.text_gen_extra_layers!r} is "
                "reserved but not wired (t/p branches do not read "
                "text_token_projs). Either keep extra_text_layers='' or wire "
                "the t/p text injection in moma_v11_final_super._step_regular."
            )

        # Resolve canon_std_scalar — used as the target std for the normalized codebook.
        # Priority chain implemented in _resolve_canon_std_scalar:
        #   text_gen.canon_std > text_gen.std_factor > pose_io.std_factor[0]
        #   > canonicalization.target_pos_std (LT path) > hard-fallback 0.5 with WARNING.
        canon_std_scalar = self._resolve_canon_std_scalar(self.model_cfg)

        # Build normalized codebook from T5 shared.weight.
        # Static, computed once, stored as a non-trainable buffer.
        # NO per-batch running stats (prompt §1).
        with torch.no_grad():
            C_raw = (
                self.input_layer.language_encoder.model.shared.weight.detach().clone().float()
            )                                          # (V, 768)
            mu = C_raw.mean(dim=0, keepdim=True)        # (1, 768)
            sigma = C_raw.std(dim=0)                    # (768,) per-dim, qa Q3
            # Guard against degenerate dims (sigma == 0 would explode the norm).
            sigma_safe = torch.clamp(sigma, min=1e-8)
            C_norm = (C_raw - mu) / sigma_safe * canon_std_scalar
        # Buffer: persistent=True so it round-trips checkpoints (cheap, V * 768 floats).
        self.register_buffer("codebook", C_norm)        # (V, 768), requires_grad=False
        self.canon_std_scalar = canon_std_scalar
        # NOTE: prompt §1 reserves an `unfreeze_codebook_after_steps` flag, but
        # the unfreeze hook is NOT wired in . If/when needed, a future SG
        # must (a) read this config key, (b) call `self.codebook.requires_grad_(True)`
        # at the right global step, and (c) add the codebook to optim_groups.
        # We do NOT read the key here to avoid creating a silent dead config.
        if text_gen_cfg.get("unfreeze_codebook_after_steps", -1) != -1:
            raise NotImplementedError(
                "text_gen.unfreeze_codebook_after_steps is reserved but not wired "
                "; remove the key or implement the unfreeze hook."
            )

        # Sanity check + log.
        post_mean_abs_mean = self.codebook.mean(dim=0).abs().mean().item()
        post_std_mean = self.codebook.std(dim=0).mean().item()
        logger.info(
            f"[v14 codebook] V={C_raw.shape[0]} D=768 "
            f"raw mu.abs.mean={mu.abs().mean().item():.3e} sigma.mean={sigma.mean().item():.3e} "
            f"sigma.min={sigma.min().item():.3e} sigma.max={sigma.max().item():.3e} | "
            f"canon_std_scalar={canon_std_scalar:.4f} | "
            f"norm mu.abs.mean={post_mean_abs_mean:.3e} norm sigma.mean={post_std_mean:.4f}"
        )
        # Hard rigor bar (criterion 3): must hold by construction unless dead dims existed.
        assert post_mean_abs_mean < 1e-5, f"codebook per-dim |mean| = {post_mean_abs_mean}"
        lo = 0.95 * canon_std_scalar
        hi = 1.05 * canon_std_scalar
        assert lo <= post_std_mean <= hi, (
            f"codebook per-dim sigma.mean={post_std_mean} not in [{lo}, {hi}]"
        )

        # TextCodebookHead is constructed only when text_gen.enable=True.
        # When disabled, AgentV14 stays bit-identical to v13.2 (criterion 2a).
        if self.text_gen_enable:
            TextCodebookHead = _TextCodebookHead
            cond_d1_eff = self.lmm.cond_d1_eff
            cond_d2_eff = self.lmm.cond_d2_eff
            # CRITICAL: defaults MUST match MotionMachineV14Super's defaults
            # (D1=512, D2=768) so a config that omits lmm.D1/D2 doesn't produce
            # mismatched per-layer projection widths. Since the LMM was already
            # constructed above using the same defaults, just read from self.lmm
            # to remove the trap entirely.
            self.text_codebook_head = TextCodebookHead(
                D1=self.lmm.D1,
                D2=self.lmm.D2,
                cond_d1_eff=cond_d1_eff,
                cond_d2_eff=cond_d2_eff,
                L_text=self.text_gen_L_text,
                layer_type_string=layer_type_string,
                extra_text_layers=self.text_gen_extra_layers,
            )
            logger.info(
                f"[v14 text] TextCodebookHead built: L_text={self.text_gen_L_text} "
                f"extra_layers='{self.text_gen_extra_layers}' "
                f"text_token_projs.keys={list(self.text_codebook_head.text_token_projs.keys())}"
            )
        else:
            self.text_codebook_head = None

        logger.info(
            f"AgentV14: flow_matching={self.is_flow_matching}, "
            f"sync_noise_prob={getattr(self.diffusion, 'sync_noise_prob', 0.0)}, "
            f"target_pos_std={self.target_pos_std}, "
            f"input_proj=simple(v11)"
        )

    # ─────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _compute_effective_K(self, step: int, training: bool) -> float:
        """Compute K-warmup effective bound as a Python float.

        Training schedule (when warmup_duration > 0):
          [0, warmup_start)                     -> 0.0
          [warmup_start, warmup_start+duration) -> K * (step-start)/duration
          [warmup_start+duration, inf)          -> K (full)

        Inference OR warmup_duration == 0 -> K (full, no warmup).

        Compile-safety: decorated with @torch.compiler.disable and returns a
        plain Python float so the compiled region never sees a changing int.
        """
        if self.input_xattn_v2 is None:
            return 0.0
        K = float(self.input_xattn_v2.traj_id_bound_K)
        warmup_start = int(self.input_xattn_v2.traj_id_K_warmup_start)
        warmup_duration = int(self.input_xattn_v2.traj_id_K_warmup_duration)
        if not training or warmup_duration <= 0:
            return K
        if step < warmup_start:
            return 0.0
        if step < warmup_start + warmup_duration:
            return K * float(step - warmup_start) / float(warmup_duration)
        return K

    @staticmethod
    @torch.compiler.disable
    def _check_finite(tensor: torch.Tensor) -> bool:
        """Check for NaN/Inf. Returns True if finite, False if any NaN/Inf.

        Compile-unsafe (CPU-GPU sync). Previously raised RuntimeError; now
        returns a bool so the forward path can skip the offending sample via
        the DAVIS NaN-skip guard pattern (see MEMORY.md session lessons).
        """
        if not tensor.isfinite().all():
            n_bad = (~tensor.isfinite()).sum().item()
            logging.warning(
                f"NaN/Inf in LMM output: {n_bad}/{tensor.numel()} elements "
                f"({100 * n_bad / tensor.numel():.1f}%) — skipping this step "
                f"(zero-grad loss, no metric log)"
            )
            return False
        return True

    @torch.compiler.disable
    def _safe_encode_text(self, text_list, B):
        """Safely encode text into language tokens. Single shared entry point.

        Defends against:
          - text_list is None (no "text" key in batch / dataset has no language)
          - text_list contains None or non-string elements (use "" instead)
          - length mismatch with batch size
        Applies text_masked_prob during training (classifier-free random drop
        of the entire string to "").

        Called from BOTH training forward AND inference, so any defensive
        layer added here covers both code paths.
        """
        if text_list is None:
            text_list = [""] * B
        else:
            # Normalize: None or non-string → ""
            text_list = [t if isinstance(t, str) else "" for t in text_list]
        if len(text_list) != B:
            raise ValueError(
                f"text list length {len(text_list)} != batch size {B}"
            )

        # Classifier-free random drop during training
        if self.text_masked_prob > 0.0 and self.training:
            text_masked = torch.rand(B) < self.text_masked_prob
            text_list = [
                text_list[i] if not text_masked[i] else "" for i in range(B)
            ]

        return self.input_layer.encode_language(text_list)

    def _build_v11_input(self, noisy_x: torch.Tensor) -> torch.Tensor:
        """V11: simple projection of noisy_x. No 3-cat input.

        Args:
            noisy_x: (B, N, 9) noised canonicalized poses.

        Returns:
            lmm_input: (B, N, 9) ready for LMM.
        """
        return noisy_x  # (B, N, 9)

    def _canon_9d_to_wing(self, x_9d: torch.Tensor) -> torch.Tensor:
        """Convert [pos, rot6d] to [pos, pos+R0*wl, pos+R1*wl] in canon space.

        Args:
            x_9d: (*, 9) canonicalized 9D poses [pos(3), rot6d(6)].

        Returns:
            (*, 9) wing representation [pos(3), wing0(3), wing1(3)].
        """
        from umm.models.umm_v14.rotation_utils import sixd_to_rotmat
        pos = x_9d[..., :3]
        rot6d = x_9d[..., 3:]
        R = sixd_to_rotmat(rot6d)  # (*, 3, 3)
        wing0 = pos + R[..., :, 0] * self.wing_len
        wing1 = pos + R[..., :, 1] * self.wing_len
        return torch.cat([pos, wing0, wing1], dim=-1)

    def _wing_to_canon_9d(self, x_wing: torch.Tensor) -> torch.Tensor:
        """Convert [pos, wing0, wing1] back to [pos, rot6d].

        Args:
            x_wing: (*, 9) wing representation [pos(3), wing0(3), wing1(3)].

        Returns:
            (*, 9) standard 9D [pos(3), rot6d(6)].
        """
        pos = x_wing[..., :3]
        wing0 = x_wing[..., 3:6]
        wing1 = x_wing[..., 6:9]
        col0 = (wing0 - pos) / self.wing_len
        col1 = (wing1 - pos) / self.wing_len
        rot6d = torch.cat([col0, col1], dim=-1)
        return torch.cat([pos, rot6d], dim=-1)

    @staticmethod
    @torch.compiler.disable
    def _check_movement_weight_valid(mw_sum, B, P, device, dtype):
        """Check movement weight validity. Compile-unsafe (.any() sync)."""
        if torch.isinf(mw_sum).any() or torch.isnan(mw_sum).any():
            logging.warning(
                "movement_weight sum is inf/NaN -- falling back to uniform weights"
            )
            return torch.ones(B, P, device=device, dtype=dtype), True
        return mw_sum, False

    @torch.compiler.disable
    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        loss_type_ids: torch.Tensor,
        phase: str,
        token_weights: Optional[torch.Tensor] = None,
        pos_scale: Optional[torch.Tensor] = None,
        virtual_mask: Optional[torch.Tensor] = None,
        movement_weight: Optional[torch.Tensor] = None,
        r_geodesic_loss: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute loss on valid tokens with type-mean weighting.

        Loss kernel: L2 norm of error vector, then Huber on the norm (v9x-style).
        This couples channels (gradient direction = unit error vector) and
        normalizes gradients (Huber caps gradient magnitude at 1 for large errors).

        Args:
            pred: (B, C+T, P, 9) model output in native prediction space.
            target: (B, C+T, P, 9) target in native prediction space.
            valid_mask: (B, C+T, P) bool.
            rot_mask: (B, P) bool -- entities with rotation.
            loss_type_ids: (B, P) int -- IDs for type_mean grouping.
            phase: "train" or "val".
            token_weights: Optional (B, C+T, P) per-token weights (e.g. min-SNR).
            pos_scale: Optional (B, 1, 1) GT-canon linear scale multiplier for position loss.
            virtual_mask: Optional (B, C+T, P) bool -- True for context (virtual) frames.
            movement_weight: Optional (B, C+T, P) per-token movement reweighting.
            r_geodesic_loss: Optional (B, C+T, P) pre-computed R_geodesic values.

        Returns:
            Tuple of (loss, log_dict).
        """
        log_dict = {}

        if self.loss_beta is not None:
            # V9x-style: L2 norm then Huber (couples channels, normalizes gradients)
            _EPS = 1e-8
            pos_diff = pred[..., :3] - target[..., :3]
            pos_sq_norm = pos_diff.pow(2).sum(dim=-1)  # (B, C+T, P)
            pos_norm = (pos_sq_norm + _EPS).sqrt()
            # Huber on the norm: linear for ||e|| > beta, quadratic for ||e|| < beta
            pos_loss = torch.where(
                pos_norm > self.loss_beta,
                pos_norm - 0.5 * self.loss_beta,
                0.5 * pos_sq_norm / self.loss_beta,
            )

            rot_diff = pred[..., 3:] - target[..., 3:]
            rot_sq_norm = rot_diff.pow(2).sum(dim=-1)  # (B, C+T, P)
            rot_norm = (rot_sq_norm + _EPS).sqrt()
            rot_loss = torch.where(
                rot_norm > self.loss_beta,
                rot_norm - 0.5 * self.loss_beta,
                0.5 * rot_sq_norm / self.loss_beta,
            )
        else:
            # MSE fallback
            pos_loss = (pred[..., :3] - target[..., :3]).pow(2).sum(dim=-1)
            rot_loss = (pred[..., 3:] - target[..., 3:]).pow(2).sum(dim=-1)

        # Apply GT-canon scale factor to position loss only.
        # GT-canon is a SIM(3) transform: T(x) = s * R * (x - mean).
        # Rotation loss is NOT affected by GT-canon.
        if pos_scale is not None:
            pos_loss = pos_loss * pos_scale  # (B, C+T, P) * (B, 1, 1)

        # Apply per-token weights (e.g. min-SNR reweighting)
        if token_weights is not None:
            pos_loss = pos_loss * token_weights
            rot_loss = rot_loss * token_weights

        # M5: Movement reweighting -- up-weight moving entities
        if movement_weight is not None:
            pos_loss = pos_loss * movement_weight
            rot_loss = rot_loss * movement_weight

        # Mask rotation loss for position-only entities
        rot_valid = valid_mask & rot_mask[:, None, :].expand_as(valid_mask)

        # Type IDs expanded to (B, C+T, P)
        _ids = loss_type_ids[:, None, :].expand_as(valid_mask).long()

        # Compute type-mean losses -- separate real/virtual (v9x-style)
        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Separate real/virtual masks
        real_mask = valid_mask & ~virtual_mask if virtual_mask is not None else valid_mask
        virt_mask_full = valid_mask & virtual_mask if virtual_mask is not None else None

        # Position loss -- real
        if real_mask.any():
            pos_real_vals = pos_loss[real_mask]
            pos_real_ids = _ids[real_mask]
            pos_real_term = type_mean(pos_real_vals, pos_real_ids, self.gid_w_modifier) * self.w_real
            loss = loss + pos_real_term
            log_dict[f"{phase}_LOSS/pos_real"] = pos_real_term.detach()

        # Position loss -- virtual
        if virt_mask_full is not None and virt_mask_full.any():
            pos_virt_vals = pos_loss[virt_mask_full]
            pos_virt_ids = _ids[virt_mask_full]
            pos_virt_term = type_mean(pos_virt_vals, pos_virt_ids, self.gid_w_modifier) * self.w_virtual
            loss = loss + pos_virt_term
            log_dict[f"{phase}_LOSS/pos_virt"] = pos_virt_term.detach()

        # Rotation loss -- real
        rot_real_mask = rot_valid & ~virtual_mask if virtual_mask is not None else rot_valid
        rot_virt_mask = rot_valid & virtual_mask if virtual_mask is not None else None

        if rot_real_mask.any():
            rot_real_vals = rot_loss[rot_real_mask]
            rot_real_ids = _ids[rot_real_mask]
            rot_real_term = type_mean(rot_real_vals, rot_real_ids, self.gid_w_modifier) * self.w_real
            loss = loss + rot_real_term
            log_dict[f"{phase}_LOSS/rot_real"] = rot_real_term.detach()

        # Rotation loss -- virtual
        if rot_virt_mask is not None and rot_virt_mask.any():
            rot_virt_vals = rot_loss[rot_virt_mask]
            rot_virt_ids = _ids[rot_virt_mask]
            rot_virt_term = type_mean(rot_virt_vals, rot_virt_ids, self.gid_w_modifier) * self.w_virtual
            loss = loss + rot_virt_term
            log_dict[f"{phase}_LOSS/rot_virt"] = rot_virt_term.detach()

        # R_geodesic loss term
        if r_geodesic_loss is not None and self.w_R > 0:
            r_geo_valid = rot_valid  # same mask as rotation
            if r_geo_valid.any():
                r_geo_vals = r_geodesic_loss[r_geo_valid]
                r_geo_ids = _ids[r_geo_valid]
                r_geo_term = type_mean(r_geo_vals, r_geo_ids, self.gid_w_modifier) * self.w_R
                loss = loss + r_geo_term
                log_dict[f"{phase}_LOSS/R_geo"] = r_geo_term.detach()

        log_dict[f"{phase}_LOSS/total"] = loss.detach()
        log_dict[f"total_loss/{phase}"] = loss.detach()

        return loss, log_dict

    # ─────────────────────────────────────────────────────────────────
    # DINO helpers (optional — no-op when use_dino=False)
    # ─────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _encode_dino_spatial(self, batch: dict):
        """Encode batch images with DINO into spatial feature maps + flat tokens.

        Reads ``batch["extra"]["images"]`` which is a list (length B) of np
        arrays (or tensors) of shape ``(V, 3, H, W)`` or ``(3, H, W)``.
        Stacks into ``(B, V, 3, H, W)`` (single-view samples are unsqueezed
        to ``V=1``).

        Returns:
            Tuple of ``(dino_spatial, dino_flat_tokens, dino_flat_mask)``
            shaped ``(B, V, GH, GW, D)``, ``(B, V*GH*GW, D)``,
            ``(B, V*GH*GW)`` — or three ``None``s when DINO is disabled,
            the key is missing, or any per-sample entry is missing.
        """
        if not self.use_dino:
            return None, None, None
        extra = batch.get("extra", None)
        if not isinstance(extra, dict) or "images" not in extra:
            return None, None, None

        raw_list = extra["images"]
        if raw_list is None:
            return None, None, None

        # Filter out None entries while preserving batch alignment.
        if any(item is None for item in raw_list):
            return None, None, None

        device = next(self.parameters()).device
        img_tensors = []
        for item in raw_list:
            t = torch.as_tensor(item, dtype=torch.float32)
            if t.ndim == 3:
                t = t.unsqueeze(0)  # (1, 3, H, W)
            elif t.ndim != 4:
                raise ValueError(
                    f"DINO images: expected per-sample shape (3,H,W) or (V,3,H,W), got {tuple(t.shape)}"
                )
            img_tensors.append(t)
        images = torch.stack(img_tensors, dim=0).to(device)  # (B, V, 3, H, W)

        # ImageNet normalization
        images = (images - self.img_mean) / self.img_std

        B, V = images.shape[0], images.shape[1]
        raw_view_ids = extra.get("view_ids", None)
        if raw_view_ids is None or any(item is None for item in raw_view_ids):
            view_ids = torch.zeros(B, V, dtype=torch.long, device=device)
        else:
            view_ids = torch.stack(
                [torch.as_tensor(item, dtype=torch.long) for item in raw_view_ids],
                dim=0,
            ).to(device)

        return self.dino_encoder.encode_spatial(images, view_ids)

    @torch.compiler.disable
    def _query_dino_at_pixels(
        self,
        dino_spatial,
        batch: dict,
        target_P: Optional[int] = None,
    ):
        """Query DINO spatial features at per-part pixel UV locations.

        Reads ``batch["extra"]`` keys ``pixel_uv_normalized``,
        ``pixel_view_idx``, ``pixel_query_mask`` (each a list of length B of
        np arrays). Pads to ``target_P`` (the LMM entity count); per-sample
        lists may be shorter.

        Returns:
            ``(B, target_P, latent_dim)`` projected DINO geo features, or
            ``None`` if data is missing or DINO is disabled.
        """
        if dino_spatial is None or not self.use_dino:
            return None
        extra = batch.get("extra", None)
        if not isinstance(extra, dict) or "pixel_uv_normalized" not in extra:
            return None

        uv_list = extra["pixel_uv_normalized"]
        view_list = extra.get("pixel_view_idx")
        qmask_list = extra.get("pixel_query_mask")
        if uv_list is None or view_list is None or qmask_list is None:
            return None
        if any(item is None for item in uv_list):
            return None
        if any(item is None for item in view_list):
            return None
        if any(item is None for item in qmask_list):
            return None

        if target_P is None:
            raise ValueError(
                "_query_dino_at_pixels requires target_P (the LMM entity count). "
                "Caller must pass the post-collate max_P from batch['trace_sem_type']."
            )
        B = len(uv_list)
        device = dino_spatial.device

        uv_padded = torch.zeros(B, target_P, 2, dtype=torch.float32, device=device)
        view_padded = torch.zeros(B, target_P, dtype=torch.long, device=device)
        qmask_padded = torch.zeros(B, target_P, dtype=torch.bool, device=device)

        # Per-sample pixel arrays from datasets are shape (P_per_sample, 2)
        # in the SAME ordering as the per-sample CanonicalDatasetOutput
        # entities. After collate, the trace tensor is padded to
        # max_P_in_batch but the `extra` dict-of-lists is NOT padded
        # (`dataset_template.py:312-317`). So per-sample UV length is the
        # original P_per_sample ≤ target_P; we left-align and zero-pad
        # the rest. Padded slots get qmask=False so query_at_pixels
        # zeros their features (and the routing in InputLayerV14 only
        # uses non-PCL entities anyway). v7 used this same pattern at
        # umm_v7/agent.py:728-750.
        for bi in range(B):
            uv_arr = torch.as_tensor(uv_list[bi], dtype=torch.float32)
            view_arr = torch.as_tensor(view_list[bi], dtype=torch.long)
            qmask_arr = torch.as_tensor(qmask_list[bi], dtype=torch.bool)
            n = min(int(uv_arr.shape[0]), target_P)
            if n > 0:
                uv_padded[bi, :n] = uv_arr[:n].to(device)
                view_padded[bi, :n] = view_arr[:n].to(device)
                qmask_padded[bi, :n] = qmask_arr[:n].to(device)

        dino_geo = self.dino_encoder.query_at_pixels(
            dino_spatial, uv_padded, view_padded, qmask_padded
        )  # (B, target_P, latent_dim)
        return self.dino_geo_proj(dino_geo)

    # ─────────────────────────────────────────────────────────────────
    # real-time text prediction printer
    # ─────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _v14_text_realtime_print(
        self,
        text_x0_pred: Optional[torch.Tensor],
        text_ids: Optional[torch.Tensor],
        global_step: int,
        phase: str,
        m: Optional[torch.Tensor] = None,
    ) -> None:
        """Print gold vs predicted text token strings for the first m=0 sample.

        Q13: pick the first row with m==0 (noised-text row -- the row whose
        loss is being trained). If `m` is None (back-compat), fall back to k=0.
        If `m` is provided but contains no m=0 entry, silently return.

        Decode-heavy ops (argmax, tokenizer.batch_decode) are all Python-side,
        hence the @torch.compiler.disable guard.

        Args:
            text_x0_pred: (B, L_text, 768) predicted codebook embeddings, or None.
            text_ids: (B, L_text) gold token ids, or None.
            global_step: Current training step (logged for readability).
            phase: "train" or "val" (logged for readability).
            m: (B,) long mask. 1 = clean (text-as-known), 0 = noised. Optional.
        """
        if text_x0_pred is None or text_ids is None:
            return
        try:
            with torch.no_grad():
                codebook = self.codebook.to(dtype=text_x0_pred.dtype)
                scores = text_x0_pred @ codebook.T           # (B, L_text, V)
                pred_ids = scores.argmax(dim=-1)              # (B, L_text)
            if m is None:
                k = 0  # back-compat: first sample
                m_val_str = "?"
            else:
                m0_idx = torch.where(m == 0)[0]
                if m0_idx.numel() == 0:
                    # All-m=1 batch: silently skip per Q13
                    return
                k = int(m0_idx[0].item())
                m_val_str = str(int(m[k].item()))
            tokenizer = self.input_layer.language_encoder.tokenizer
            gold_str = tokenizer.batch_decode(
                text_ids[k:k + 1].cpu(), skip_special_tokens=True
            )[0]
            pred_str = tokenizer.batch_decode(
                pred_ids[k:k + 1].cpu(), skip_special_tokens=True
            )[0]
            acc = (pred_ids[k] == text_ids[k]).float().mean().item()
            print(
                f"[v14 text] step={global_step} phase={phase} k={k} m={m_val_str} acc={acc:.3f} "
                f"gold={gold_str!r} pred={pred_str!r}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"[v14 text print] failed: {e!r}")

    @torch.compiler.disable
    def _v14_print_m_split(
        self,
        m: torch.Tensor,
        t_text: torch.Tensor,
        global_step: int,
        phase: str,
    ) -> None:
        """Print per-sample m and t_text for the first 2 batches per phase.

        Cap controlled by attribute counters `_v14_m_split_print_count_{phase}`.
        """
        try:
            attr = f"_v14_m_split_print_count_{phase}"
            cnt = int(getattr(self, attr, 0))
            if cnt >= 2:
                return
            setattr(self, attr, cnt + 1)
            m_list = m.detach().cpu().tolist()
            t_list = t_text.detach().cpu().tolist()
            print(
                f"[v14 m-mask] step={global_step} phase={phase} "
                f"m={m_list} t_text={t_list}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"[v14 m-mask print] failed: {e!r}")

    @torch.compiler.disable
    def _v14_print_loss_components(
        self,
        loss_motion: torch.Tensor,
        loss_total: torch.Tensor,
        L_text_mse: Optional[torch.Tensor],
        L_round: Optional[torch.Tensor],
        L_text: Optional[torch.Tensor],
        global_step: int,
        phase: str,
    ) -> None:
        """Print L_motion, L_text_mse, L_round, L_text_total, loss for the first 4 batches per phase.

        Cap controlled by attribute counters `_v14_loss_print_count_{phase}`.
        ALL `.item()` GPU-CPU syncs live behind the counter gate so the steady
        state (>=4 prints fired) is sync-free.
        """
        try:
            attr = f"_v14_loss_print_count_{phase}"
            cnt = int(getattr(self, attr, 0))
            if cnt >= 4:
                return
            setattr(self, attr, cnt + 1)
            motion_str = f"{loss_motion.detach().item():.6f}"
            mse_str = f"{L_text_mse.detach().item():.6f}" if L_text_mse is not None else "n/a"
            round_str = f"{L_round.detach().item():.6f}" if L_round is not None else "n/a"
            text_str = f"{L_text.detach().item():.6f}" if L_text is not None else "n/a"
            print(
                f"[v14 loss] step={global_step} phase={phase} "
                f"L_motion={motion_str} L_text_mse={mse_str} L_round={round_str} "
                f"L_text_total={text_str} loss={loss_total.detach().item():.6f}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"[v14 loss print] failed: {e!r}")

    def _lmm_motion_only(self, **kwargs):
        """Call self.lmm and always return the motion-output tensor.

        Two responsibilities:
        1. v14 LMM returns (motion, x0_text_pred) when text_tokens is provided,
           else just motion. Tuple-unwrap so all inference call sites get a
           plain tensor regardless of 's noised-text inference path.
        2. R1 enforcement on the inference side: when text_gen.enable=True,
           the add_tokens lang branch is DISABLED (mirrors training at
           agent.forward). Forcing add_tokens=None here covers all 10
           inference call sites in one place -- train/inference distribution
           mismatch is structurally impossible.
        """
        if self.text_gen_enable:
            kwargs["add_tokens"] = None
            kwargs["add_tokens_mask"] = None
        out = self.lmm(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    # ─────────────────────────────────────────────────────────────────
    # Forward (training)
    # ─────────────────────────────────────────────────────────────────

    def forward(self, batch: dict, phase: str, global_step: int = 0) -> dict:
        """Forward pass for training and validation.

        Flow:
        1. Extract batch → concat ctx+trace
        2. SE3 → 9D (pos + 6D rot)
        3. Canonicalize using known+valid tokens
        4. Add noise (flow matching or DDPM)
        5. Build input [noised_X, known_indicator, known_X]
        6. LMM → velocity/noise prediction
        7. Loss in canonicalized 9D space

        Args:
            batch: Batch dict from V10 dataset collate.
            phase: "train" or "val".
            global_step: Current training step.

        Returns:
            Dict with "loss", "log_dict", and predicted SE3 transforms.
        """
        ret = {}

        B, T, P = batch["trace_mask"].shape
        C = batch["context_mask"].shape[1]
        device = batch["trace_mask"].device

        trace = batch["trace"].clone()           # (B, T, P, 4, 4)
        context = batch["context"].clone()       # (B, C, P, 4, 4)
        trace_mask = batch["trace_mask"].clone()   # (B, T, P)
        context_mask = batch["context_mask"].clone()  # (B, C, P)
        known_mask = batch["known_mask"].clone()   # (B, T, P)
        ctx_known_mask = batch["context_known_mask"].clone()  # (B, C, P)

        sem_type = batch["trace_sem_type"].clone()   # (B, P) int
        group_id = batch["trace_group_id"].clone()   # (B, P) int
        rot_mask = batch["trace_rot_mask"].clone()   # (B, P) bool
        timestep = batch["time"]                     # (B, T) float

        # Batch-level finite guard: skip the whole step if ANY input tensor is
        # non-finite. Protects against upstream dataset bugs producing NaN/Inf
        # that would poison the LMM. Same skip pattern as the NaN guard on LMM
        # output — returns a zero-grad dummy loss so optimizer.step() is safe.
        if not (
            torch.isfinite(trace).all()
            and torch.isfinite(context).all()
            and torch.isfinite(timestep).all()
        ):
            n_bad_trace = int((~torch.isfinite(trace)).sum().item())
            n_bad_ctx = int((~torch.isfinite(context)).sum().item())
            logging.warning(
                f"agent.forward: batch contains NaN/Inf "
                f"(trace={n_bad_trace}, context={n_bad_ctx}) — skipping step, "
                f"check dataset pipeline."
            )
            _any_param = next(self.lmm.parameters())
            zero_loss = (_any_param * 0.0).sum()
            ret["loss"] = zero_loss
            ret["log_dict"] = {
                f"{phase}_LOSS/total": torch.tensor(0.0, device=device),
                f"total_loss/{phase}": torch.tensor(0.0, device=device),
                f"{phase}_LOSS/bad_batch_skip": torch.tensor(1.0, device=device),
            }
            return ret

        # 1. Concatenate context + trace
        all_se3 = torch.cat([context, trace], dim=1)       # (B, C+T, P, 4, 4)
        all_mask = torch.cat([context_mask, trace_mask], dim=1)  # (B, C+T, P) bool
        all_known = torch.cat([ctx_known_mask, known_mask], dim=1)  # (B, C+T, P) bool

        # Context always known option
        if self.ctx_always_known:
            all_known[:, :C] = context_mask[:, :C]

        # 2. SE3 → 9D
        all_9d = se3_to_9d(all_se3)  # (B, C+T, P, 9)

        # Zero out rotation for position-only entities
        rot_9d_mask = torch.ones(B, 1, P, 9, device=device, dtype=all_9d.dtype)
        rot_9d_mask[..., 3:] = rot_mask[:, None, :, None].expand(-1, 1, -1, 6).float()
        all_9d = all_9d * rot_9d_mask

        # 3. Canonicalize using known+valid tokens
        # Flatten (C+T, P) → N for canonicalization
        N = (C + T) * P
        all_9d_flat = all_9d.reshape(B, N, 9)                    # (B, N, 9)
        all_mask_flat = all_mask.reshape(B, N)                    # (B, N)
        all_known_flat = all_known.reshape(B, N)                  # (B, N)
        known_valid_flat = all_known_flat & all_mask_flat          # (B, N) — for canon

        canon_9d_flat, canon_params = self.canonicalizer.forward(
            all_9d_flat, known_valid_flat
        )  # (B, N, 9), CanonParams

        # Override R_canon to identity when rotation canonicalization is disabled.
        # Position centering + scaling still apply. Only rotation alignment is skipped.
        if not self.canon_rotation:
            canon_params = CanonParams(
                mean_pos=canon_params.mean_pos,
                scale=canon_params.scale,
                R_canon=torch.eye(3, device=device, dtype=canon_params.R_canon.dtype)
                    .unsqueeze(0).expand(B, -1, -1),
            )
            # Re-canonicalize with identity rotation (keeps center + scale, drops rotation)
            canon_9d_flat = canonicalize(all_9d_flat, canon_params)

        # C1: Convert canonicalized 9D to wing space if configured.
        # Wing space puts all 9 channels in position-magnitude space,
        # avoiding the ~2:1 scale imbalance between unit-vector rotation
        # and canonicalized position.
        if self.rot_param == "6d_wing":
            canon_9d_flat = self._canon_9d_to_wing(canon_9d_flat)

        # 4. Add noise (with 50/50 per-sample branching)
        # CRITICAL: pass known_mask so schedulers can do clean-known vs async branching.
        r_diffusion_levels = None  # Only set for MeanFlow
        if self.is_meanflow:
            noisy_flat, t_levels, r_levels, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat, known_mask=all_known_flat,
            )
            # t_levels for LMM timestep embedding (same as FM)
            diffusion_levels = t_levels * self.diffusion_time_scale  # (B, N)
            # r_levels for MeanFlow interval conditioning
            r_diffusion_levels = r_levels * self.diffusion_time_scale  # (B, N)
        elif self.is_flow_matching:
            noisy_flat, t_levels, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat, known_mask=all_known_flat,
            )
            # Scale t_levels for LMM timestep embedding
            diffusion_levels = t_levels * self.diffusion_time_scale  # (B, N)
        else:
            noisy_flat, timesteps_int, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat, known_mask=all_known_flat,
            )
            diffusion_levels = timesteps_int.float()  # (B, N)

        # V13.3.4 Conditional training mask (25% of samples)
        unknown_x_mask_for_lmm = None
        if self.twostage_training_mask:
            clean_known_branch = getattr(self.diffusion, '_last_clean_known_branch', None)
            if clean_known_branch is not None:
                # Second 50% coin flip -> 25% total
                second_flip = torch.rand(B, 1, device=device) < 0.5
                use_restricted = clean_known_branch & second_flip  # (B, 1)
                if use_restricted.any():
                    # V13.3.4 Unknown X = valid trace positions that are NOT known
                    unknown_x_mask_for_lmm = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
                    # Context frames are never "unknown X"
                    # Trace: valid but not known
                    trace_unknown = trace_mask & ~known_mask  # (B, T, P)
                    unknown_x_mask_for_lmm[:, C:] = trace_unknown
                    # Only apply for restricted samples
                    unknown_x_mask_for_lmm = unknown_x_mask_for_lmm & use_restricted[:, :, None]
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        n_restricted = int(use_restricted.sum().item())
                        logging.debug(f"conditional mask: {n_restricted}/{B} samples restricted")

        # Invalid tokens → fully noised (already handled by diffusion: t=1 or max_ts)
        # Additionally overwrite their encoded values with pure noise
        invalid_flat = ~all_mask_flat  # (B, N)
        if noise_flat is not None:
            noisy_flat = torch.where(
                invalid_flat.unsqueeze(-1).expand_as(noisy_flat),
                noise_flat,
                noisy_flat,
            )

        # C3 FIX: Re-apply rotation mask to noisy input.
        # Position-only entities must have zero rotation in LMM input.
        # The initial rot_9d_mask was applied to clean data, but add_noise
        # introduces random noise in the rotation channels.
        rot_9d_mask_flat = rot_9d_mask.reshape(B, 1, P, 9).expand(-1, C + T, -1, -1).reshape(B, N, 9)
        noisy_flat = noisy_flat * rot_9d_mask_flat

        # V13.3.5: Cur_only special noise assignment (training only).
        # When mask scheduler chose a cur_only pattern, override noise levels:
        # - Known trace tokens: already clean (t=0)
        # - All other trace X: fully noised (t=max)
        # - Context: ONE random t shared across ALL context entities
        _mask_pnames = batch.get("extra", {}).get("mask_pattern_name", None)
        if _mask_pnames is not None:
            for b_idx in range(B):
                _pn = _mask_pnames[b_idx] if _mask_pnames[b_idx] is not None else ""
                if "cur_only" in _pn:
                    # Determine max noise level
                    if self.is_flow_matching or self.is_meanflow:
                        _max_t = 1.0 * self.diffusion_time_scale
                    else:
                        _max_t = float(self.diffusion.num_train_timesteps - 1)

                    # Work with (C+T, P) grid view of flat diffusion_levels
                    _dl = diffusion_levels[b_idx].reshape(C + T, P)

                    # Trace not-known tokens -> fully noised
                    _tnk = all_mask[b_idx, C:] & ~all_known[b_idx, C:]  # (T, P)
                    _dl[C:][_tnk] = _max_t
                    # Also replace their noisy values with pure noise
                    _tnk_flat = torch.zeros(N, dtype=torch.bool, device=device)
                    _tnk_flat[C * P:] = _tnk.reshape(-1)
                    if noise_flat is not None:
                        noisy_flat[b_idx][_tnk_flat] = noise_flat[b_idx][_tnk_flat]

                    # Context: single random t shared across ALL valid ctx entities
                    _cv = all_mask[b_idx, :C]  # (C, P)
                    _ctx_t = None
                    if _cv.any():
                        _rng = torch.rand(1).item()  # CPU-only, no CUDA sync
                        if self.is_flow_matching or self.is_meanflow:
                            _ctx_t = _rng * self.diffusion_time_scale
                        else:
                            _ctx_t = float(int(_rng * (self.diffusion.num_train_timesteps - 1)))
                        _dl[:C][_cv] = _ctx_t

                        # Re-noise ctx to the synced t level
                        _cf = torch.zeros(N, dtype=torch.bool, device=device)
                        _cf[:C * P] = _cv.reshape(-1)
                        if noise_flat is not None:
                            if self.is_flow_matching or self.is_meanflow:
                                noisy_flat[b_idx][_cf] = (
                                    (1.0 - _rng) * canon_9d_flat[b_idx][_cf]
                                    + _rng * noise_flat[b_idx][_cf]
                                )
                            else:
                                _acp = self.diffusion.noise_scheduler.alphas_cumprod.to(device)
                                _a = _acp[int(_ctx_t)]
                                noisy_flat[b_idx][_cf] = (
                                    _a.sqrt() * canon_9d_flat[b_idx][_cf]
                                    + (1.0 - _a).sqrt() * noise_flat[b_idx][_cf]
                                )

                    # Re-apply rotation mask after modifying noisy values
                    noisy_flat[b_idx] = noisy_flat[b_idx] * rot_9d_mask_flat[b_idx]

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        _ct_str = f"{_ctx_t:.1f}" if _ctx_t is not None else "N/A"
                        logging.debug(
                            f"V13.3.5 cur_only[{b_idx}]: trace_nk_t={_max_t}, ctx_t={_ct_str}"
                        )

        # 5. Build V11 input (simple projection, no 3-cat)
        lmm_input_flat = self._build_v11_input(noisy_flat)  # (B, N, 9)

        # Reshape back to (B, C+T, P, 9)
        lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)
        diff_levels = diffusion_levels.reshape(B, C + T, P)
        r_diff_levels = (
            r_diffusion_levels.reshape(B, C + T, P)
            if r_diffusion_levels is not None else None
        )

        # Stash tensors for mask/noise viz (detach — no grad needed).
        # Normalize diff_levels to [0, 1] for the renderer (clips to [0, 1]).
        #   - DDPM: integer in [0, num_train_timesteps - 1] -> divide by (N-1)
        #   - FM / MeanFlow: float in [0, time_scale] -> divide by time_scale
        _viz_levels = diff_levels.detach().float()
        if self.is_flow_matching or self.is_meanflow:
            _viz_levels = _viz_levels / max(self.diffusion_time_scale, 1.0)
        else:
            _n_train = int(getattr(self.diffusion, "num_train_timesteps", 1))
            _viz_levels = _viz_levels / max(_n_train - 1, 1)
        ret["viz_diff_levels"] = _viz_levels        # (B, C+T, P) in [0, 1]
        ret["viz_all_known"] = all_known.detach()   # (B, C+T, P) bool
        ret["viz_all_mask"] = all_mask.detach()     # (B, C+T, P) bool
        ret["viz_C"] = C

        # Trajectory-level padding mask for attention
        traj_padding_mask = all_mask.any(dim=1)  # (B, P)
        attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)  # (B, T, P)
        attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)  # (B, C, P)

        # 6. Language tokens (single shared safe encoder, defends None/missing)
        lang_tk = self._safe_encode_text(batch.get("text"), B)
        lang_proj = self.lang_token_proj(lang_tk.clone())  # (B, L, C_P)
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)

        # Entity features
        local_pcl = batch.get("local_pcl", None)  # (B, P, PCL, D_pcl) or None
        local_pcl_mask = batch.get("local_pcl_mask", None)  # (B, P) bool or None
        # Scale pcl xyz by canonicalization scale to match canon coordinate system.
        if local_pcl is not None:
            pcl_scale = canon_params.scale[:, :, None, None]  # (B, 1, 1, 1)
            local_pcl = local_pcl.clone()
            local_pcl[..., :3] = local_pcl[..., :3] * pcl_scale

        # --- DINO path (no-op when use_dino=False) ---
        # Compute DINO features BEFORE input_layer so we can pass external_geo_emb
        # in a single input_layer call (avoids running the PointNet twice).
        # Reads batch["extra"]["images"] (list-of-arrays) inside the helper.
        dino_geo_emb = None
        dino_flat_tokens = None
        dino_flat_mask = None
        if self.use_dino:
            dino_spatial, dino_flat_tokens, dino_flat_mask = self._encode_dino_spatial(batch)
            dino_geo_emb = self._query_dino_at_pixels(dino_spatial, batch, target_P=P)

        feat = self.input_layer(
            sem_type, group_id,
            pcl=local_pcl, pcl_valid_mask=local_pcl_mask,
            external_geo_emb=dino_geo_emb,
        )  # (B, P, feat_dim)

        img_proj = None
        img_mask = None
        if self.use_dino and dino_flat_tokens is not None:
            img_proj = self.image_token_proj(dino_flat_tokens.clone())  # (B, V*P_dino, D)
            img_mask = dino_flat_mask  # (B, V*P_dino)
            # Add type embeddings: 0=text, 1=image
            lang_proj = lang_proj + self.add_token_type_emb(
                torch.zeros(lang_proj.shape[1], device=lang_proj.device, dtype=torch.long)
            )
            img_proj = img_proj + self.add_token_type_emb(
                torch.ones(img_proj.shape[1], device=img_proj.device, dtype=torch.long)
            )

        # V13.2 V6: optional cross-attn input layer. Runs AFTER the standard
        # input_layer + image_token_proj + lang_token_proj, BEFORE the LMM.
        # Per-entity validity is the trajectory padding mask (True = valid).
        if self.input_xattn_enable:
            valid_entity_mask = traj_padding_mask  # (B, P) bool
            lang_proj, img_proj, feat = self.input_xattn(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask,
            )

        # V13.2 Explore 10: optional input_xattn_v2.
        # Explore 12 K-warmup: compute effective_K as a Python float OUTSIDE
        # the compile boundary. Passing global_step (a changing int) into the
        # compiled region would trigger Dynamo recompilation every step.
        # `training=self.training` -- val/test sees the post-warmup full K
        # (non-training branch of _compute_effective_K returns bound_K).
        learned_traj_ids = None
        if self.input_xattn_v2_enable:
            valid_entity_mask = traj_padding_mask  # (B, P) bool
            step_i = int(global_step)
            effective_K = self._compute_effective_K(step_i, training=self.training)
            # Log K progression (rate-limited). fast_dev_run prints every step.
            if step_i % 100 == 0 or step_i < 10:
                logger.info(
                    f"[input_xattn_v2] step={step_i} training={self.training} "
                    f"effective_K={effective_K:.4f} "
                    f"bound_K={self.input_xattn_v2.traj_id_bound_K} "
                    f"warmup_start={self.input_xattn_v2.traj_id_K_warmup_start} "
                    f"warmup_duration={self.input_xattn_v2.traj_id_K_warmup_duration}"
                )
            feat, learned_traj_ids = self.input_xattn_v2(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask,
                effective_K=effective_K,
            )

        # + Build text track tokens (if text_gen.enable=True).
        text_tokens_d1 = None
        text_tokens_mask_lmm = None
        text_ids_for_print = None  # type: Optional[torch.Tensor]
        # captures (set ONLY when text_gen enabled):
        m = None                              # (B,) long: 1=clean, 0=noised
        t_text = None                         # (B,) long
        x0_text_norm_for_loss = None          # (B, L_text, 768) clean target (detached)
        text_ids_for_loss = None              # (B, L_text) gold token ids
        if self.text_gen_enable and self.text_codebook_head is not None:
            raw_texts = batch.get("text", None)
            if raw_texts is None:
                raw_texts = [""] * B
            # Fail-loud type/length validation: silent broadcast bugs at codebook
            # lookup are the worst kind of training divergence.
            if isinstance(raw_texts, str):
                raise TypeError(
                    f"batch['text'] must be a list of B={B} strings, got bare str "
                    f"({raw_texts!r}). Wrap in a list."
                )
            if not isinstance(raw_texts, (list, tuple)) or len(raw_texts) != B:
                raise ValueError(
                    f"batch['text'] must be a list/tuple of length B={B}, got "
                    f"type={type(raw_texts).__name__} len={len(raw_texts) if hasattr(raw_texts, '__len__') else '?'}"
                )
            tokenizer = self.input_layer.language_encoder.tokenizer
            text_ids, text_mask = _tokenize_with_eos_fill(raw_texts, tokenizer, self.text_gen_L_text)
            text_ids = text_ids.to(device)   # (B, L_text)
            text_mask = text_mask.to(device)  # (B, L_text)
            # Look up codebook in autocast-correct dtype.
            codebook = self.codebook
            if torch.is_autocast_enabled():
                codebook = codebook.to(dtype=torch.get_autocast_gpu_dtype())
            x0_text_norm = codebook[text_ids]                          # (B, L_text, 768) clean

            # per-sample Bernoulli(0.5) m-mask for diffusion forcing.
            # m=1 (clean): t_text=0 -> add_noise returns x0 exactly (text-as-known branch)
            # m=0 (noised): sample t_text from same DDPM scheduler as motion
            m = (torch.rand(B, device=device) < 0.5).long()             # (B,)
            if self.is_flow_matching or self.is_meanflow:
                # FM/MeanFlow path: linear interpolation in [0, 1].
                t_sched = torch.rand(B, device=device, dtype=x0_text_norm.dtype)
                t_text_f = torch.where(m.bool(), torch.zeros_like(t_sched), t_sched)
                noise_text = torch.randn_like(x0_text_norm)
                t_expanded = t_text_f.view(B, 1, 1)
                noisy_text = (1.0 - t_expanded) * x0_text_norm + t_expanded * noise_text
                # For visibility/print: log discretized timestep (×100 for parity with DDPM scale).
                t_text = (t_text_f * 100.0).long()
                # m=1 rows: t_text_f=0 -> noisy_text == x0 already (FM is exact identity at t=0),
                # but be explicit to mirror the DDPM clamp and survive future scheduler edits.
                m_clean = m.bool()
                if m_clean.any():
                    noisy_text = torch.where(
                        m_clean.view(B, 1, 1), x0_text_norm, noisy_text,
                    )
            else:
                # DDPM path: integer timesteps in [0, num_train_timesteps).
                t_sched = torch.randint(
                    0, self.diffusion.num_train_timesteps, (B,),
                    device=device, dtype=torch.long,
                )
                t_text = torch.where(m.bool(), torch.zeros_like(t_sched), t_sched)
                noise_text = torch.randn_like(x0_text_norm)
                L_text_dim = x0_text_norm.shape[1]
                # noise_scheduler.add_noise expects (N, D); flatten across L.
                noisy_text = self.diffusion.noise_scheduler.add_noise(
                    original_samples=x0_text_norm.reshape(B * L_text_dim, 768),
                    noise=noise_text.reshape(B * L_text_dim, 768),
                    timesteps=t_text.repeat_interleave(L_text_dim),
                ).reshape(B, L_text_dim, 768)
                # critic: DDPMScheduler.add_noise(x0, noise, t=0) is NOT
                # identity (sqrt(alpha_0)*x0 + sqrt(1-alpha_0)*noise leaves a
                # ~1.6% noise residual). Force m=1 ("clean / known") rows back
                # to exact x0 so the train-time text-as-known branch matches
                # the inference contract.
                m_clean = m.bool()
                if m_clean.any():
                    noisy_text = torch.where(
                        m_clean.view(B, 1, 1), x0_text_norm, noisy_text,
                    )

            # Project the (possibly noised) text into D1 -- this replaces the
            # clean projection from . m=1 rows pass x0 through untouched.
            text_tokens_d1 = self.text_codebook_head.forward_input(noisy_text)  # (B, L_text, D1)
            text_tokens_mask_lmm = text_mask
            text_ids_for_print = text_ids
            # captures for joint-loss block below.
            x0_text_norm_for_loss = x0_text_norm.detach()
            text_ids_for_loss = text_ids
            # Visibility: per-sample m for the first 2 batches in each phase.
            self._v14_print_m_split(m, t_text, global_step, phase)

        # 7. LMM forward
        lmm_kwargs = dict(
            x=lmm_input[:, C:],
            x_mask=attn_mask_t,
            x_time=timestep,
            x_level=diff_levels[:, C:],
            feat=feat,
            ctx=lmm_input[:, :C],
            ctx_mask=attn_mask_c,
            ctx_level=diff_levels[:, :C],
            # R1: when text_gen enabled, old lang branch is disabled so
            # the model sees text ONLY via the new text track (prevents duplicate
            # language signal and preserves zero-init criterion 2b).
            add_tokens=None if self.text_gen_enable else lang_proj,
            add_tokens_mask=None if self.text_gen_enable else lang_mask,
            image_tokens=img_proj,
            image_tokens_mask=img_mask,
            group_ids=group_id,
            # text-track kwargs (None when text_gen disabled -> back-compat).
            text_tokens=text_tokens_d1,
            text_tokens_mask=text_tokens_mask_lmm,
            text_codebook_head=self.text_codebook_head,
        )
        # MeanFlow: pass r_levels as additional interval conditioning
        if r_diff_levels is not None:
            lmm_kwargs["x_r_level"] = r_diff_levels[:, C:]
            lmm_kwargs["ctx_r_level"] = r_diff_levels[:, :C]
        # V13.2 Explore 10: pass learned traj_ids to LMM
        if learned_traj_ids is not None:
            lmm_kwargs["learned_traj_ids"] = learned_traj_ids
        # V13.3.4 pass dynamic unknown_x_mask to LMM
        if unknown_x_mask_for_lmm is not None:
            lmm_kwargs["unknown_x_mask"] = unknown_x_mask_for_lmm
        lmm_raw_out = self.lmm(**lmm_kwargs)
        # unpack tuple when text track active.
        if isinstance(lmm_raw_out, tuple):
            out, text_x0_pred = lmm_raw_out  # out: (B, S, P, 9), text_x0_pred: (B, L, 768)
        else:
            out, text_x0_pred = lmm_raw_out, None  # (B, C+T, P, 9)

        # + real-time text print (both train and val, before NaN
        # guard to also catch what happens when motion goes NaN -- text pred
        # may still be OK). picks the first m=0 row (noised target).
        self._v14_text_realtime_print(
            text_x0_pred, text_ids_for_print, global_step, phase, m=m
        )

        # NaN/Inf skip guard (DAVIS lesson): if LMM output has any NaN/Inf,
        # return a zero-grad dummy loss and skip this optimizer step. This
        # prevents a single bad step (e.g. fp16 overflow at high step counts)
        # from killing a long-running training run.
        # critic WARN: also guard text_x0_pred. If text head overflows to
        # NaN while motion is finite, text loss would silently propagate NaN
        # into total loss -- skip the step and surface via nan_skip metric.
        out_finite = self._check_finite(out)
        text_finite = (
            text_x0_pred is None or self._check_finite(text_x0_pred)
        )
        if not (out_finite and text_finite):
            # Build a zero-grad loss tensor that preserves the computation
            # graph connection to parameters (so optimizer.step() is safe).
            # `out` itself has NaN, so we cannot use it — use zero times a
            # single LMM param so autograd has something to traverse.
            _any_param = next(self.lmm.parameters())
            zero_loss = (_any_param * 0.0).sum()
            ret["loss"] = zero_loss
            ret["log_dict"] = {
                f"{phase}_LOSS/total": torch.tensor(0.0, device=device),
                f"total_loss/{phase}": torch.tensor(0.0, device=device),
                f"{phase}_LOSS/nan_skip": torch.tensor(1.0, device=device),
                f"{phase}_LOSS/nan_skip_text": torch.tensor(
                    0.0 if text_finite else 1.0, device=device,
                ),
            }
            return ret

        # 7b. Network output space conversion (Exp 3: x0_to_eps, Exp: x0_to_vel)
        # V11: LMM outputs x0 for DDPM/FM. Save it BEFORE any conversion.
        # MeanFlow is the sole exception: output is avg velocity, NOT x0.
        if self.is_meanflow:
            lmm_x0_output = None  # MeanFlow: output is avg velocity
        else:
            lmm_x0_output = out  # DDPM/FM: LMM output IS x0
        if self.network_output_space == "x0_to_eps" and not self.is_flow_matching:
            _EPS = 1e-6
            noisy_for_conv = noisy_flat.reshape(B, C + T, P, 9)
            with torch.autocast(device_type=device.type, enabled=False):
                out_f = out.float()
                noisy_f = noisy_for_conv.float()
                ts_flat = diff_levels.long().reshape(-1)  # (B*(C+T)*P,)
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_t = alphas_cp[ts_flat]  # (B*(C+T)*P,)
                # safe_sqrt: cannot produce NaN from negative/zero/NaN input.
                sqrt_alpha = safe_sqrt(alpha_t).reshape(B, C + T, P, 1)
                sqrt_1m_alpha = safe_sqrt(1.0 - alpha_t).reshape(B, C + T, P, 1)
                # Extra guard: clamp to _EPS for the division below.
                sqrt_1m_alpha = sqrt_1m_alpha.clamp(min=_EPS)
                out_eps = (noisy_f - sqrt_alpha * out_f) / sqrt_1m_alpha
            out = out_eps.to(out.dtype)
        elif self.network_output_space == "x0_to_vel" and not self.is_flow_matching:
            _EPS = 1e-6
            noisy_for_conv = noisy_flat.reshape(B, C + T, P, 9)
            with torch.autocast(device_type=device.type, enabled=False):
                out_f = out.float()
                noisy_f = noisy_for_conv.float()
                ts_flat = diff_levels.long().reshape(-1)  # (B*(C+T)*P,)
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_t = alphas_cp[ts_flat]  # (B*(C+T)*P,)
                # safe_sqrt: cannot produce NaN from negative/zero/NaN input.
                sqrt_alpha = safe_sqrt(alpha_t).reshape(B, C + T, P, 1)
                sqrt_1m_alpha = safe_sqrt(1.0 - alpha_t).reshape(B, C + T, P, 1)
                # Extra guard: clamp to _EPS for the division below.
                sqrt_1m_alpha = sqrt_1m_alpha.clamp(min=_EPS)
                # v = (sqrt(alpha_t) * x_t - x0) / sqrt(1 - alpha_t)
                out_vel = (sqrt_alpha * noisy_f - out_f) / sqrt_1m_alpha
            out = out_vel.to(out.dtype)
        elif self.network_output_space == "x0_to_vel" and self.is_flow_matching:
            # FM: v = eps - x0. noise_flat IS eps, out IS x0_pred
            out = noise_flat.reshape(B, C + T, P, 9) - out

        # 8. Decode x0 from model output (for viz, metrics, and x0-loss-space)
        # NOT used for loss in native mode — loss uses `out` and `target_flat` directly.
        # FM: x_t = x_0 + t*v -> x_0 = x_t - t*v
        # DDPM: depends on prediction_type (see compute_x0)
        # x0_to_eps/x0_to_vel: use saved LMM x0 directly (NOT compute_x0 which round-trips)
        noisy_reshaped = noisy_flat.reshape(B, C + T, P, 9)
        # Always compute x0 with gradient to avoid torch.set_grad_enabled
        # inside compiled forward() (causes graph break → fallback OOM).
        # For native loss mode, x0_pred_canon gradient graph is unused and
        # freed after forward — no memory cost.
        if lmm_x0_output is not None:
            # x0_to_eps or x0_to_vel: LMM output IS x0, use directly
            x0_pred_canon = lmm_x0_output
        elif self.is_flow_matching or self.is_meanflow:
            t_exp = diff_levels.unsqueeze(-1) / self.diffusion_time_scale  # (B, C+T, P, 1)
            x0_pred_canon = noisy_reshaped - t_exp * out
        else:
            x0_pred_canon = compute_x0(
                self.diffusion.noise_scheduler,
                out, diff_levels.long(), noisy_reshaped,
            )

        # x0 ground truth = clean canonicalized poses
        x0_gt_canon = canon_9d_flat.reshape(B, C + T, P, 9)

        # C1: Convert x0 from wing to standard 9D for GT-canon metrics,
        # decanonicalization, and SE3 conversion. Keep wing-space copies
        # for loss computation (loss operates in wing space when active).
        if self.rot_param == "6d_wing":
            x0_pred_canon_std = self._wing_to_canon_9d(x0_pred_canon)
            x0_gt_canon_std = self._wing_to_canon_9d(x0_gt_canon)
        else:
            x0_pred_canon_std = x0_pred_canon
            x0_gt_canon_std = x0_gt_canon

        # 9. GT-canon: extract scale factor for loss + compute GT-canon
        #    x0 for per-timestep metrics.
        #
        # GT-canon is a SIM(3) transform: T(x) = s * R * (x - mean).
        # Mathematically: MSE(T(pred_pos), T(target_pos)) = s^2 * MSE(pred, target).
        #
        # V11 uses LINEAR s (not s^2) — design decision to avoid over-correction
        # for samples with large spatial scale. s^2 was too aggressive, causing
        # gradient explosion on high-scale samples.
        #
        # Rotation loss is NOT affected by GT-canon (R orthogonal → norm-preserving).
        # NOTE: uses all_mask_flat (all valid tokens), NOT known_valid_flat.
        # GT-canon uses ALL valid GT tokens for a more stable reference frame.
        # canon_9d_flat is clean data (before noise), so unknown tokens still
        # have correct positions — they just weren't shown to the model.
        gt_canon_params = compute_canon_params(
            canon_9d_flat, all_mask_flat, self.target_pos_std
        )
        # s per sample, shape (B, 1) → (B, 1, 1) for broadcast over (B, C+T, P)
        # (LINEAR s, not s^2 — V11 design decision)
        s_linear = gt_canon_params.scale.unsqueeze(-1)  # (B, 1, 1) LINEAR (not s^2)

        # GT-canon x0 for per-timestep accuracy metrics (detached, not in loss)
        with torch.no_grad():
            x0_pred_gt = canonicalize(
                x0_pred_canon_std.detach().reshape(B, N, 9), gt_canon_params
            ).reshape(B, C + T, P, 9)
            x0_gt_gt = canonicalize(
                x0_gt_canon_std.reshape(B, N, 9), gt_canon_params
            ).reshape(B, C + T, P, 9)

        # Select loss weighting IDs
        if self.weight_by == "group_id":
            loss_type_ids = group_id
        elif self.weight_by == "sem_id":
            loss_type_ids = sem_type
        else:
            loss_type_ids = batch["trace_uid"]

        # Compute per-token loss weights (min-SNR reweighting, DDPM only)
        token_weights = None
        if self.loss_weighting == "min_snr" and not self.is_flow_matching:
            snr_w = self.diffusion.compute_snr_weights(
                diff_levels.long(), gamma=self.min_snr_gamma
            )  # (B, C+T, P)
            token_weights = snr_w

        # 10. Compute loss in the appropriate prediction space
        if self.loss_space == "x0":
            # Exp 4: x0 loss on eps/vel output.
            # Convert model output → x0 (done in step 8), compute MSE in x0 space.
            # This is genuinely different from native eps loss (MSE on noise).
            loss_pred = x0_pred_canon
            loss_target = x0_gt_canon
        elif (
            self.network_output_space == "x0_to_vel"
            and self.is_flow_matching
        ):
            # FM x0_to_vel: out = noise - x0_pred (velocity prediction).
            # Target must also be in velocity space: noise - x0_gt.
            # target_flat from FM scheduler is x0, which is WRONG as vel target.
            loss_pred = out  # noise - x0_pred
            loss_target = noise_flat.reshape(B, C + T, P, 9) - x0_gt_canon  # noise - x0_gt
        elif (
            self.network_output_space == "x0_to_eps"
            and not self.is_flow_matching
        ):
            # DDPM eps mode: out is eps_pred (converted from x0), target is noise.
            # BUG FIX: target_flat is x_0 (from x0net add_noise), NOT noise.
            # The correct eps target is the random noise used in add_noise.
            loss_pred = out  # eps_pred
            loss_target = noise_flat.reshape(B, C + T, P, 9)  # noise
        elif (
            self.network_output_space == "x0_to_vel"
            and not self.is_flow_matching
        ):
            # DDPM vel mode: out is vel_pred (converted from x0), target is velocity.
            # BUG FIX: target_flat is x_0, NOT velocity.
            # Velocity target: v = sqrt(alpha_t)*noise - sqrt(1-alpha_t)*x_0
            _VEL_EPS = 1e-6
            with torch.autocast(device_type=device.type, enabled=False):
                _ts_flat = diff_levels.long().reshape(-1)
                _alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                _alpha_t = _alphas_cp[_ts_flat]
                _sqrt_a = safe_sqrt(_alpha_t).reshape(B, C + T, P, 1)
                _sqrt_1ma = safe_sqrt(1.0 - _alpha_t).reshape(B, C + T, P, 1).clamp(min=_VEL_EPS)
            _noise_reshaped = noise_flat.reshape(B, C + T, P, 9).float()
            vel_target = (_sqrt_a * _noise_reshaped - _sqrt_1ma * x0_gt_canon.float())
            loss_pred = out
            loss_target = vel_target.to(out.dtype)
        else:
            # x0 native (no conversion) — target is x_0.
            # MeanFlow: target is u_target (avg velocity from scheduler).
            loss_pred = out
            loss_target = target_flat.reshape(B, C + T, P, 9)

        # H1: Virtual/real mask — context frames are "virtual"
        virtual_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        virtual_mask[:, :C] = True

        # M5: Movement reweighting — up-weight moving entities
        movement_weight = None
        if self.movement_reweight_min_ratio > 0:
            with torch.no_grad():
                mv_pos = batch["trace"][..., :3, 3]  # (B, T, P, 3)
                t_mask = batch["trace_mask"]  # (B, T, P)

                # First valid position per track (not t=0 which may be identity-padded)
                first_valid_t = t_mask.float().argmax(dim=1)  # (B, P)
                first_valid_pos = torch.gather(
                    mv_pos, 1,
                    first_valid_t.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 3),
                ).squeeze(1)  # (B, P, 3)

                # Displacement from first valid frame per track
                ref = first_valid_pos.unsqueeze(1)  # (B, 1, P, 3)
                mv_pos_safe = torch.where(t_mask.unsqueeze(-1), mv_pos, ref)
                mv_disp = (mv_pos_safe - ref).pow(2).sum(dim=-1).clamp(min=1e-12).sqrt()
                mv_per_track = mv_disp.max(dim=1).values  # (B, P)
                mv_per_track = mv_per_track * self.movement_scale
                mv_max = mv_per_track.max(dim=1, keepdim=True).values  # (B, 1)
                # Floor: static tracks get min_ratio * max_movement
                mv_per_track = mv_per_track + mv_max * self.movement_reweight_min_ratio
                mv_sum = mv_per_track.sum(dim=1, keepdim=True).clamp(min=1e-8)
                mw = P * mv_per_track / mv_sum  # sums to ~P
                mw = mw.clamp(max=self.movement_reweight_max_ratio)
                # Re-normalize after clamping
                mw_sum = mw.sum(dim=1, keepdim=True)
                mw_sum, fell_back = self._check_movement_weight_valid(
                    mw_sum, B, P, device, mv_pos.dtype
                )
                if fell_back:
                    mw = mw_sum  # uniform weights
                else:
                    mw = P * mw / mw_sum.clamp(min=1e-8)
                # Apply to trace frames only, context gets weight 1.0
                mw_ctx = torch.ones(B, C, P, device=device, dtype=mw.dtype)
                mw_trace = mw.unsqueeze(1).expand(-1, T, -1)
                movement_weight = torch.cat([mw_ctx, mw_trace], dim=1)  # (B, C+T, P)

        # R_geodesic loss: 1 - tr(R_pred^T @ R_gt) / 3
        # Rotation canon is a global rotation, so R_pred^T @ R_gt is the same
        # in canon or original space (R_canon cancels out). Compute in loss space.
        if self.w_R > 0:
            pred_rot6d = loss_pred[..., 3:]  # (B, C+T, P, 6)
            gt_rot6d = loss_target[..., 3:]
            R_pred = sixd_to_rotmat(pred_rot6d)  # (B, C+T, P, 3, 3)
            R_gt = sixd_to_rotmat(gt_rot6d)
            # tr(R_pred^T @ R_gt) = sum of diagonal of R_pred^T @ R_gt
            RtR = torch.einsum("...ji,...jk->...ik", R_pred, R_gt)  # (B, C+T, P, 3, 3)
            r_geodesic = 1.0 - (RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]) / 3.0  # (B, C+T, P)
        else:
            r_geodesic = None

        loss, log_dict = self._compute_loss(
            loss_pred, loss_target, all_mask, rot_mask, loss_type_ids, phase,
            token_weights=token_weights,
            pos_scale=s_linear,
            virtual_mask=virtual_mask,
            movement_weight=movement_weight,
            r_geodesic_loss=r_geodesic,
        )

        # joint text loss (m=0 only -> diffusion-forcing variant).
        # Capture L_motion as a *tensor* for the breakdown print BEFORE adding
        # L_text. Detached so the autograd graph for the live loss is unaffected,
        # and NOT .item()-synced here -- the print helper does the sync only
        # behind its per-phase counter gate (steady state is sync-free).
        L_motion_tensor = loss.detach()
        L_text_mse = None  # type: Optional[torch.Tensor]
        L_round = None      # type: Optional[torch.Tensor]
        L_text = None       # type: Optional[torch.Tensor]
        if (
            self.text_gen_enable
            and text_x0_pred is not None
            and x0_text_norm_for_loss is not None
            and text_ids_for_loss is not None
            and m is not None
        ):
            m_bool = (m == 0)
            cb = self.codebook.to(dtype=text_x0_pred.dtype)
            V_cb = cb.shape[0]
            if m_bool.any():
                sel_pred = text_x0_pred[m_bool]                 # (M, L_text, 768)
                sel_target = x0_text_norm_for_loss[m_bool]      # (M, L_text, 768)
                sel_ids = text_ids_for_loss[m_bool]             # (M, L_text)
                L_text_mse = (sel_pred - sel_target).pow(2).mean()
                # CE rounding: F.cross_entropy does log_softmax internally --
                # NEVER softmax before CE.
                logits = (sel_pred @ cb.T) / self.text_gen_tau   # (M, L_text, V)
                L_round = F.cross_entropy(
                    logits.reshape(-1, V_cb),
                    sel_ids.reshape(-1),
                )
                L_text = L_text_mse + self.text_gen_lambda_round * L_round
            else:
                # All-m=1 batch -> zero-grad guard via text_codebook_head param
                # (mirrors the DAVIS NaN-guard idiom from v13.2).
                _any = next(self.text_codebook_head.parameters())
                L_text_mse = (_any * 0.0).sum()
                L_round = (_any * 0.0).sum()
                L_text = (_any * 0.0).sum()
            loss = loss + self.text_gen_lambda_text * L_text
            log_dict[f"{phase}_LOSS/text_mse"] = L_text_mse.detach()
            log_dict[f"{phase}_LOSS/text_round"] = L_round.detach()
            log_dict[f"{phase}_LOSS/text_total"] = L_text.detach()
            log_dict[f"{phase}_LOSS/total"] = loss.detach()
            log_dict[f"total_loss/{phase}"] = loss.detach()
            # metrics (Item 4): per-batch token accuracy split by m.
            with torch.no_grad():
                pred_ids_full = (text_x0_pred @ cb.T).argmax(dim=-1)   # (B, L_text)
                correct = (pred_ids_full == text_ids_for_loss)         # (B, L_text)
                if m_bool.any():
                    log_dict[f"{phase}_text/x0_acc_m0"] = (
                        correct[m_bool].float().mean()
                    )
                m1_bool = (m == 1)
                if m1_bool.any():
                    log_dict[f"{phase}_text/x0_acc_m1"] = (
                        correct[m1_bool].float().mean()
                    )
                log_dict[f"{phase}_text/m_frac_zero"] = (
                    (m == 0).float().mean()
                )
            # Print L_motion / L_text components for fast_dev_run audit
            # (gated by per-phase counter < 4).
            self._v14_print_loss_components(
                L_motion_tensor, loss, L_text_mse, L_round, L_text, global_step, phase,
            )

        ret["loss"] = loss
        ret["log_dict"] = log_dict

        # 11. Per-timestep-bucket X0 accuracy metrics
        with torch.no_grad():
            # Normalize timesteps to [0, 1]
            if self.is_flow_matching:
                t_norm = diff_levels / self.diffusion_time_scale  # already [0, 1]
            else:
                t_norm = diff_levels / float(self.diffusion.num_train_timesteps)

            # Per-token MSE between x0_pred and x0_gt (in GT-canon space)
            x0_mse = (x0_pred_gt - x0_gt_gt).pow(2).sum(dim=-1)  # (B, C+T, P)
            x0_mse_valid = x0_mse[all_mask]  # (K,) only valid tokens
            t_valid = t_norm[all_mask]  # (K,)

            # 4 buckets: [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0]
            bucket_edges = [0.0, 0.25, 0.5, 0.75, 1.01]
            for i in range(4):
                bucket_mask = (t_valid >= bucket_edges[i]) & (t_valid < bucket_edges[i + 1])
                if bucket_mask.any():
                    log_dict[f"{phase}_X0_MSE/t_{bucket_edges[i]:.2f}_{bucket_edges[i+1]:.2f}"] = (
                        x0_mse_valid[bucket_mask].mean()
                    )

        # Decanonicalize x0 prediction → original space for viz (detached)
        with torch.no_grad():
            x0_pred_flat = x0_pred_canon_std.detach().reshape(B, N, 9)
            x0_pred_orig = self.canonicalizer.inverse(x0_pred_flat, canon_params)
            x0_pred_orig = x0_pred_orig.reshape(B, C + T, P, 9)
            x0_se3 = ninep_to_se3(x0_pred_orig)  # (B, C+T, P, 4, 4)

            ret["x0_pred_context"] = x0_se3[:, :C].detach()
            ret["x0_pred_trace"] = x0_se3[:, C:].detach()

        return ret

    # ─────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────

    def _prepare_inference(
        self,
        trace: torch.Tensor,
        context: torch.Tensor,
        trace_known_mask: torch.Tensor,
        context_known_mask: torch.Tensor,
        trace_mask: torch.Tensor,
        context_mask: torch.Tensor,
        real_timestep: torch.Tensor,
        group_id: torch.Tensor,
        sem_type: torch.Tensor,
        rot_mask: torch.Tensor,
        text: Optional[list] = None,
        local_pcl: Optional[torch.Tensor] = None,
        local_pcl_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        pixel_uv_normalized: Optional[torch.Tensor] = None,
        pixel_view_idx: Optional[torch.Tensor] = None,
        pixel_query_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> dict:
        """V13.3.4 Extract inference preparation into reusable helper.

        Returns dict with all precomputed tensors needed for denoising loops.
        """
        B, T, P = trace.shape[:3]
        C = context.shape[1]
        device = trace.device

        if text is None:
            text = [""] * B
        if n_steps is None:
            n_steps = self.n_inference_steps

        # 1. Concatenate and encode
        all_se3 = torch.cat([context, trace], dim=1)  # (B, C+T, P, 4, 4)
        all_mask = torch.cat([context_mask, trace_mask], dim=1)
        all_known = torch.cat([context_known_mask, trace_known_mask], dim=1)

        # Context always known option (parity with forward())
        if self.ctx_always_known:
            all_known[:, :C] = context_mask[:, :C]

        all_9d = se3_to_9d(all_se3)  # (B, C+T, P, 9)
        # Zero rot for position-only entities
        rot_9d_mask = torch.ones(B, 1, P, 9, device=device, dtype=all_9d.dtype)
        rot_9d_mask[..., 3:] = rot_mask[:, None, :, None].expand(-1, 1, -1, 6).float()
        all_9d = all_9d * rot_9d_mask

        # 2. Flatten and canonicalize
        N = (C + T) * P
        all_9d_flat = all_9d.reshape(B, N, 9)
        all_mask_flat = all_mask.reshape(B, N)
        all_known_flat = all_known.reshape(B, N)
        known_valid_flat = all_known_flat & all_mask_flat

        canon_9d_flat, canon_params = self.canonicalizer.forward(
            all_9d_flat, known_valid_flat
        )

        # Override R_canon to identity when rotation canonicalization disabled (parity with forward)
        if not self.canon_rotation:
            canon_params = CanonParams(
                mean_pos=canon_params.mean_pos,
                scale=canon_params.scale,
                R_canon=torch.eye(3, device=device, dtype=canon_params.R_canon.dtype)
                    .unsqueeze(0).expand(B, -1, -1),
            )
            canon_9d_flat = canonicalize(all_9d_flat, canon_params)

        # C1: Convert to wing space after canonicalization (parity with forward)
        if self.rot_param == "6d_wing":
            canon_9d_flat = self._canon_9d_to_wing(canon_9d_flat)

        # 3. Prepare language + features (single shared safe encoder)
        lang_tk = self._safe_encode_text(text, B)
        lang_proj = self.lang_token_proj(lang_tk.clone())
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)

        # --- DINO path (no-op when use_dino=False) ---
        dino_geo_emb = None
        dino_flat_tokens = None
        dino_flat_mask = None
        if self.use_dino:
            inf_extra = {}
            if images is not None:
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                inf_extra["images"] = [images[i] for i in range(images.shape[0])]
                if view_ids is not None:
                    inf_extra["view_ids"] = [view_ids[i] for i in range(view_ids.shape[0])]
            if (
                pixel_uv_normalized is not None
                and pixel_view_idx is not None
                and pixel_query_mask is not None
            ):
                inf_extra["pixel_uv_normalized"] = [
                    pixel_uv_normalized[i] for i in range(pixel_uv_normalized.shape[0])
                ]
                inf_extra["pixel_view_idx"] = [
                    pixel_view_idx[i] for i in range(pixel_view_idx.shape[0])
                ]
                inf_extra["pixel_query_mask"] = [
                    pixel_query_mask[i] for i in range(pixel_query_mask.shape[0])
                ]
            inf_batch = {"extra": inf_extra}
            dino_spatial, dino_flat_tokens, dino_flat_mask = self._encode_dino_spatial(inf_batch)
            dino_geo_emb = self._query_dino_at_pixels(dino_spatial, inf_batch, target_P=P)

        # Scale pcl xyz by canonicalization scale (parity with forward)
        if local_pcl is not None:
            pcl_scale = canon_params.scale[:, :, None, None]
            local_pcl = local_pcl.clone()
            local_pcl[..., :3] = local_pcl[..., :3] * pcl_scale

        feat = self.input_layer(
            sem_type, group_id,
            pcl=local_pcl, pcl_valid_mask=local_pcl_mask,
            external_geo_emb=dino_geo_emb,
        )

        img_proj = None
        img_mask = None
        if self.use_dino and dino_flat_tokens is not None:
            img_proj = self.image_token_proj(dino_flat_tokens.clone())
            img_mask = dino_flat_mask
            lang_proj = lang_proj + self.add_token_type_emb(
                torch.zeros(lang_proj.shape[1], device=lang_proj.device, dtype=torch.long)
            )
            img_proj = img_proj + self.add_token_type_emb(
                torch.ones(img_proj.shape[1], device=img_proj.device, dtype=torch.long)
            )

        # Attention mask
        traj_padding_mask = all_mask.any(dim=1)  # (B, P)

        # V13.2 V6: optional cross-attn input layer (inference path).
        if self.input_xattn_enable:
            valid_entity_mask_inf = traj_padding_mask
            lang_proj, img_proj, feat = self.input_xattn(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask_inf,
            )

        # V13.2 Explore 10: input_xattn_v2 (inference path, runs once).
        learned_traj_ids_inf = None
        if self.input_xattn_v2_enable:
            valid_entity_mask_inf = traj_padding_mask
            feat, learned_traj_ids_inf = self.input_xattn_v2(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask_inf,
                effective_K=self.input_xattn_v2.traj_id_bound_K,
            )

        # C3: Prepare rot_9d_mask_flat for re-zeroing rotation in denoising loop
        rot_9d_mask_flat = rot_9d_mask.reshape(B, 1, P, 9).expand(
            -1, C + T, -1, -1
        ).reshape(B, N, 9)

        return dict(
            B=B, T=T, P=P, C=C, N=N, device=device, n_steps=n_steps,
            canon_9d_flat=canon_9d_flat, canon_params=canon_params,
            all_mask=all_mask, all_mask_flat=all_mask_flat,
            all_known=all_known, all_known_flat=all_known_flat,
            known_valid_flat=known_valid_flat,
            rot_9d_mask_flat=rot_9d_mask_flat, rot_9d_mask=rot_9d_mask,
            lang_proj=lang_proj, lang_mask=lang_mask,
            img_proj=img_proj, img_mask=img_mask,
            feat=feat,
            traj_padding_mask=traj_padding_mask,
            learned_traj_ids_inf=learned_traj_ids_inf,
            real_timestep=real_timestep,
            group_id=group_id,
        )

    @torch.inference_mode()
    def inference(
        self,
        trace: torch.Tensor,           # (B, T, P, 4, 4)
        context: torch.Tensor,          # (B, C, P, 4, 4)
        trace_known_mask: torch.Tensor,  # (B, T, P)
        context_known_mask: torch.Tensor,  # (B, C, P)
        trace_mask: torch.Tensor,       # (B, T, P)
        context_mask: torch.Tensor,     # (B, C, P)
        real_timestep: torch.Tensor,    # (B, T)
        group_id: torch.Tensor,         # (B, P)
        sem_type: torch.Tensor,         # (B, P)
        rot_mask: torch.Tensor,         # (B, P)
        text: Optional[list] = None,
        local_pcl: Optional[torch.Tensor] = None,
        local_pcl_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        pixel_uv_normalized: Optional[torch.Tensor] = None,
        pixel_view_idx: Optional[torch.Tensor] = None,
        pixel_query_mask: Optional[torch.Tensor] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        return_intermediates: bool = False,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        """Run inference (Euler for flow matching, DDIM for DDPM).

        Same input signature as forward() for compatibility with viz code.

        Args:
            trace: (B, T, P, 4, 4) SE3 matrices.
            context: (B, C, P, 4, 4) context SE3.
            trace_known_mask: (B, T, P) which trace tokens are known.
            context_known_mask: (B, C, P) which context tokens are known.
            trace_mask: (B, T, P) validity mask.
            context_mask: (B, C, P) validity mask.
            real_timestep: (B, T) real time stamps.
            group_id: (B, P) group IDs.
            sem_type: (B, P) semantic types.
            rot_mask: (B, P) entities with rotation.
            text: List of text strings, or None.
            images: Optional (B, V, 3, H, W) or (B, 3, H, W) float images in
                [0,1] for the DINO encoder. Ignored if DINO is disabled.
            view_ids: Optional (B, V) int view indices for DINO.
            pixel_uv_normalized: Optional (B, P, 2) normalized UV for
                per-part DINO pixel queries.
            pixel_view_idx: Optional (B, P) int view index per part.
            pixel_query_mask: Optional (B, P) bool mask for pixel queries.
            seed: Random seed.
            n_steps: Override inference steps.
            verbose: Print progress bar.
            return_intermediates: If True, capture X0 predictions at
                25%/50%/75%/100% of denoising and return as dict.
            cfg_scale: Classifier-free guidance scale. 1.0 = no guidance.
                When != 1.0, runs LMM twice (conditional + unconditional)
                and blends: pred = uncond + cfg_scale * (cond - uncond).
            **kwargs: Absorbs unknown kwargs from viz code.

        Returns:
            If return_intermediates=False (default): tuple (ret_context, ret_trace).
            If return_intermediates=True: dict with keys "context", "trace",
            and "intermediates" (list of dicts with step/pct/context/trace).
        """
        is_training = self.training
        if is_training:
            self.eval()

        # V13.3.4 use shared preparation helper
        prep = self._prepare_inference(
            trace=trace, context=context,
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=trace_mask, context_mask=context_mask,
            real_timestep=real_timestep,
            group_id=group_id, sem_type=sem_type, rot_mask=rot_mask,
            text=text, local_pcl=local_pcl, local_pcl_mask=local_pcl_mask,
            images=images, view_ids=view_ids,
            pixel_uv_normalized=pixel_uv_normalized,
            pixel_view_idx=pixel_view_idx,
            pixel_query_mask=pixel_query_mask,
            n_steps=n_steps,
        )
        B = prep["B"]; T = prep["T"]; P = prep["P"]; C = prep["C"]
        N = prep["N"]; device = prep["device"]; n_steps = prep["n_steps"]
        canon_9d_flat = prep["canon_9d_flat"]; canon_params = prep["canon_params"]
        all_mask_flat = prep["all_mask_flat"]
        known_valid_flat = prep["known_valid_flat"]
        rot_9d_mask_flat = prep["rot_9d_mask_flat"]
        lang_proj = prep["lang_proj"]; lang_mask = prep["lang_mask"]
        img_proj = prep["img_proj"]; img_mask = prep["img_mask"]
        feat = prep["feat"]
        traj_padding_mask = prep["traj_padding_mask"]
        learned_traj_ids_inf = prep["learned_traj_ids_inf"]
        real_timestep = prep["real_timestep"]
        group_id = prep["group_id"]

        # 4. Initialize from noise
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise = torch.randn(B, N, 9, device=device, generator=g)

        # Known tokens start at their clean canonicalized values
        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(noise),
            canon_9d_flat,
            noise,
        )
        # C3 FIX: zero rotation for position-only entities in initial x_t
        x_t = x_t * rot_9d_mask_flat

        # V13.3.5: Prepare unconditional text for CFG
        lang_proj_uncond = torch.zeros_like(lang_proj) if cfg_scale != 1.0 else None

        # 5. Denoising loop
        intermediates = []

        # Helper: decode x0, decanonicalize → SE3 for intermediate snapshots.
        def _snapshot_x0(x0_canon_flat, step_idx, total_steps, pct):
            """Capture intermediate x0 prediction and store as SE3."""
            _x0 = x0_canon_flat.reshape(B, N, 9)
            # C1: convert wing → standard 9D before decanonicalization
            if self.rot_param == "6d_wing":
                _x0 = self._wing_to_canon_9d(_x0)
            x0_orig = self.canonicalizer.inverse(_x0, canon_params)
            x0_orig = x0_orig.reshape(B, C + T, P, 9)
            x0_se3 = ninep_to_se3(x0_orig)
            intermediates.append({
                "step": step_idx,
                "total": total_steps,
                "pct": pct,
                "context": x0_se3[:, :C],
                "trace": x0_se3[:, C:],
            })

        if self.is_meanflow:
            # MeanFlow inference: Euler with r=0 conditioning.
            # Model outputs avg velocity u(z_t, r, t). At inference, r=0
            # gives the mean velocity from 0 to t.
            # Step: z_{t-dt} = z_t - dt * u(z_t, r=0, t)
            dt = 1.0 / n_steps
            snapshot_steps = {
                int(n_steps * f): int(f * 100)
                for f in [0.25, 0.5, 0.75]
            }
            iterator = range(n_steps)
            if verbose:
                iterator = tqdm(iterator, desc="MeanFlow inference")

            for step_idx in iterator:
                t = 1.0 - step_idx * dt

                # Clamp known tokens
                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    canon_9d_flat,
                    x_t,
                )

                # C3 FIX: zero rotation for position-only entities
                x_t = x_t * rot_9d_mask_flat

                # Build input with t-level and r-level (r=0 at inference)
                t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)
                t_levels[known_valid_flat] = 0.0  # known tokens → clean
                r_levels_inf = torch.zeros_like(t_levels)  # r=0 at inference
                diff_levels = (t_levels * self.diffusion_time_scale).reshape(B, C + T, P)
                r_diff_levels = (r_levels_inf * self.diffusion_time_scale).reshape(B, C + T, P)

                lmm_input_flat = self._build_v11_input(x_t)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

                _mf_lmm_kwargs = dict(
                    x=lmm_input[:, C:],
                    x_mask=attn_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=attn_mask_c,
                    ctx_level=diff_levels[:, :C],
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    image_tokens=img_proj.clone() if img_proj is not None else None,
                    image_tokens_mask=img_mask.clone() if img_mask is not None else None,
                    group_ids=group_id,
                    x_r_level=r_diff_levels[:, C:],
                    ctx_r_level=r_diff_levels[:, :C],
                )
                if learned_traj_ids_inf is not None:
                    _mf_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                model_out = self._lmm_motion_only(**_mf_lmm_kwargs)  # (B, C+T, P, 9) — avg velocity prediction

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_mf_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    out_uncond = self._lmm_motion_only(**_uncond_kwargs)
                    model_out = out_uncond + cfg_scale * (model_out - out_uncond)

                # MeanFlow Euler step: z_{t-dt} = z_t - dt * u_pred
                u_pred = model_out.reshape(B, N, 9)
                x_t = x_t - dt * u_pred

                # Snapshot x0: x0 = z_t - t * u_pred (same recovery as FM)
                if return_intermediates and step_idx in snapshot_steps:
                    x0_est = x_t  # after step, x_t is closer to x_0
                    _snapshot_x0(x0_est, step_idx, n_steps,
                                 snapshot_steps[step_idx])

        elif self.is_flow_matching:
            dt = 1.0 / n_steps
            # Snapshot at 25%, 50%, 75%, 100%
            snapshot_steps = {
                int(n_steps * f): int(f * 100)
                for f in [0.25, 0.5, 0.75]
            }
            iterator = range(n_steps)
            if verbose:
                iterator = tqdm(iterator, desc="FM inference")

            for step_idx in iterator:
                t = 1.0 - step_idx * dt

                # Clamp known tokens
                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    canon_9d_flat,
                    x_t,
                )
                # C3 FIX: zero rotation for position-only entities
                x_t = x_t * rot_9d_mask_flat

                # Build input
                t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)
                # V11 FIX: known tokens get x_level=0 (THE v10 bug fix)
                t_levels[known_valid_flat] = 0.0
                diff_levels = (t_levels * self.diffusion_time_scale).reshape(B, C + T, P)
                lmm_input_flat = self._build_v11_input(x_t)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

                _fm_lmm_kwargs = dict(
                    x=lmm_input[:, C:],
                    x_mask=attn_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=attn_mask_c,
                    ctx_level=diff_levels[:, :C],
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    image_tokens=img_proj.clone() if img_proj is not None else None,
                    image_tokens_mask=img_mask.clone() if img_mask is not None else None,
                    group_ids=group_id,
                )
                if learned_traj_ids_inf is not None:
                    _fm_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                model_out = self._lmm_motion_only(**_fm_lmm_kwargs)  # (B, C+T, P, 9)

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_fm_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    out_uncond = self._lmm_motion_only(**_uncond_kwargs)
                    model_out = out_uncond + cfg_scale * (model_out - out_uncond)

                # FM inference step — V11: LMM ALWAYS outputs x0.
                # Direct x0 scheduling (avoids velocity conversion instability).
                #
                # From FM: x_t = (1-t)*x0 + t*eps
                # Estimate eps: eps = (x_t - (1-t)*x0_pred) / t
                # Next step: x_{t-dt} = (1-(t-dt))*x0_pred + (t-dt)*eps
                model_out_flat = model_out.reshape(B, N, 9)
                t_next = t - dt
                x0_pred = model_out_flat
                eps_est = (x_t - (1.0 - t) * x0_pred) / max(t, 1e-6)
                x_t = (1.0 - t_next) * x0_pred + t_next * eps_est

                # Capture intermediate x0 at snapshot steps
                if return_intermediates and step_idx in snapshot_steps:
                    _snapshot_x0(model_out, step_idx, n_steps,
                                 snapshot_steps[step_idx])

        else:
            # DDPM/DDIM inference
            scheduler = self.diffusion.gen_scheduler
            if n_steps is not None:
                scheduler.set_timesteps(n_steps)
            ddim_steps = list(scheduler.timesteps)
            total_ddim = len(ddim_steps)
            snapshot_steps = {
                int(total_ddim * f): int(f * 100)
                for f in [0.25, 0.5, 0.75]
            }
            iterator = enumerate(ddim_steps)
            if verbose:
                iterator = tqdm(iterator, total=total_ddim, desc="DDIM inference")

            for ddim_idx, t_int in iterator:
                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    canon_9d_flat,
                    x_t,
                )
                # C3 FIX: zero rotation for position-only entities
                x_t = x_t * rot_9d_mask_flat

                t_tensor = torch.full((B, N), int(t_int), device=device, dtype=torch.long)
                # V11 FIX: known tokens get x_level=0 (THE v10 bug fix)
                t_tensor[known_valid_flat] = 0
                diff_levels = t_tensor.float().reshape(B, C + T, P)
                lmm_input_flat = self._build_v11_input(x_t)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

                _ddpm_lmm_kwargs = dict(
                    x=lmm_input[:, C:],
                    x_mask=attn_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=attn_mask_c,
                    ctx_level=diff_levels[:, :C],
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    image_tokens=img_proj.clone() if img_proj is not None else None,
                    image_tokens_mask=img_mask.clone() if img_mask is not None else None,
                    group_ids=group_id,
                )
                if learned_traj_ids_inf is not None:
                    _ddpm_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                pred = self._lmm_motion_only(**_ddpm_lmm_kwargs)

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_ddpm_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    pred_uncond = self._lmm_motion_only(**_uncond_kwargs)
                    pred = pred_uncond + cfg_scale * (pred - pred_uncond)

                # V11: LMM ALWAYS outputs x0. Convert to scheduler's prediction_type.
                # Snapshot x0 BEFORE converting.
                if return_intermediates and ddim_idx in snapshot_steps:
                    _snapshot_x0(pred, ddim_idx, total_ddim,
                                 snapshot_steps[ddim_idx])

                _pred_type = self.diffusion.prediction_type
                if _pred_type == "sample":
                    # Scheduler expects x0 directly — no conversion needed.
                    pass
                else:
                    _EPS = 1e-6
                    alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                    alpha_val = alphas_cp[int(t_int)]
                    sqrt_alpha = safe_sqrt(alpha_val)
                    sqrt_1m_alpha = safe_sqrt(1.0 - alpha_val).clamp(min=_EPS)
                    x_t_reshaped = x_t.reshape(B, C + T, P, 9).float()
                    if _pred_type == "epsilon":
                        # x0 → eps: eps = (x_t - sqrt(a)*x0) / sqrt(1-a)
                        pred = (x_t_reshaped - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                    elif _pred_type == "v_prediction":
                        # x0 → vel: v = (sqrt(a)*x_t - x0) / sqrt(1-a)
                        pred = (sqrt_alpha * x_t_reshaped - pred.float()) / sqrt_1m_alpha
                    else:
                        raise ValueError(f"Unsupported prediction_type: {_pred_type}")
                    pred = pred.to(x_t.dtype)

                pred_flat = pred.reshape(B * N, 9)
                prev_x = scheduler.step(
                    pred_flat, int(t_int), x_t.reshape(B * N, 9)
                ).prev_sample.reshape(B, N, 9)

                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    x_t,
                    prev_x,
                )

        # Final clamping
        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(x_t),
            canon_9d_flat,
            x_t,
        )

        # C1: Convert back from wing to standard 9D before decanonicalization
        if self.rot_param == "6d_wing":
            x_t = self._wing_to_canon_9d(x_t)

        # 6. Decanonicalize and convert back to SE3
        x_orig = self.canonicalizer.inverse(x_t, canon_params)
        x_orig = x_orig.reshape(B, C + T, P, 9)
        result_se3 = ninep_to_se3(x_orig)  # (B, C+T, P, 4, 4)

        ret_context = result_se3[:, :C]
        ret_trace = result_se3[:, C:]

        # 100% snapshot = final result
        if return_intermediates:
            intermediates.append({
                "step": n_steps if self.is_flow_matching else total_ddim,
                "total": n_steps if self.is_flow_matching else total_ddim,
                "pct": 100,
                "context": ret_context,
                "trace": ret_trace,
            })

        if is_training:
            self.train()

        if return_intermediates:
            return {
                "context": ret_context,
                "trace": ret_trace,
                "intermediates": intermediates,
            }
        return ret_context, ret_trace

    @torch.inference_mode()
    def inference_twostage(
        self,
        trace: torch.Tensor,
        context: torch.Tensor,
        trace_known_mask: torch.Tensor,
        context_known_mask: torch.Tensor,
        trace_mask: torch.Tensor,
        context_mask: torch.Tensor,
        real_timestep: torch.Tensor,
        group_id: torch.Tensor,
        sem_type: torch.Tensor,
        rot_mask: torch.Tensor,
        text: Optional[list] = None,
        local_pcl: Optional[torch.Tensor] = None,
        local_pcl_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        pixel_uv_normalized: Optional[torch.Tensor] = None,
        pixel_view_idx: Optional[torch.Tensor] = None,
        pixel_query_mask: Optional[torch.Tensor] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        """V13.3.5: Two-stage inference redesign.

        Stage 1: Only current frame known (no history). Single denoising pass.
                 Canonicalization uses ONLY the latest one timestep.
                 Generates context.
        Stage 2: History + generated ctx as known. Full denoising.
                 Canonicalization uses all known + generated ctx.
                 CRITICAL: passes all-False unknown_x_mask for symmetric attention.

        Returns:
            tuple (ret_context, ret_trace) -- same as inference().
        """
        if self.is_flow_matching or self.is_meanflow:
            raise NotImplementedError(
                "inference_twostage only supports DDPM/DDIM scheduler. "
                "FM and MeanFlow not implemented."
            )

        is_training = self.training
        if is_training:
            self.eval()

        B, T, P = trace.shape[:3]
        C = context.shape[1]
        device = trace.device

        # Find current frame index (last known frame in original mask)
        orig_trace_known = trace_known_mask.clone()
        cur_indices = []
        for b in range(B):
            known_times = orig_trace_known[b].any(dim=-1).nonzero(as_tuple=False)  # (K, 1)
            if len(known_times) > 0:
                cur_indices.append(int(known_times[-1, 0]))
            else:
                cur_indices.append(0)

        # ===================== Stage 1: Current-frame-only =====================
        # Modify masks: only current frame known, no context known
        s1_trace_known = torch.zeros_like(trace_known_mask)
        for b in range(B):
            s1_trace_known[b, cur_indices[b]] = trace_known_mask[b, cur_indices[b]]
        s1_ctx_known = torch.zeros_like(context_known_mask)  # ctx unknown in stage 1

        # Prepare with restricted known mask (cano uses only current frame)
        prep_s1 = self._prepare_inference(
            trace=trace, context=context,
            trace_known_mask=s1_trace_known,
            context_known_mask=s1_ctx_known,
            trace_mask=trace_mask, context_mask=context_mask,
            real_timestep=real_timestep,
            group_id=group_id, sem_type=sem_type, rot_mask=rot_mask,
            text=text, local_pcl=local_pcl, local_pcl_mask=local_pcl_mask,
            images=images, view_ids=view_ids,
            pixel_uv_normalized=pixel_uv_normalized,
            pixel_view_idx=pixel_view_idx,
            pixel_query_mask=pixel_query_mask,
            n_steps=n_steps,
        )
        N = prep_s1["N"]; n_steps_val = prep_s1["n_steps"]
        canon_9d_s1 = prep_s1["canon_9d_flat"]
        canon_params_s1 = prep_s1["canon_params"]
        known_valid_s1 = prep_s1["known_valid_flat"]
        rot_mask_flat_s1 = prep_s1["rot_9d_mask_flat"]
        all_mask_s1 = prep_s1["all_mask"]
        all_known_s1 = prep_s1["all_known"]
        lang_proj_s1 = prep_s1["lang_proj"]; lang_mask_s1 = prep_s1["lang_mask"]
        img_proj_s1 = prep_s1["img_proj"]; img_mask_s1 = prep_s1["img_mask"]
        feat_s1 = prep_s1["feat"]
        traj_padding_mask_s1 = prep_s1["traj_padding_mask"]
        learned_traj_ids_s1 = prep_s1["learned_traj_ids_inf"]
        real_timestep_s1 = prep_s1["real_timestep"]
        group_id_s1 = prep_s1["group_id"]

        # Initialize from noise
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise_s1 = torch.randn(B, N, 9, device=device, generator=g)
        x_t = torch.where(
            known_valid_s1.unsqueeze(-1).expand_as(noise_s1),
            canon_9d_s1, noise_s1,
        )
        x_t = x_t * rot_mask_flat_s1

        # CFG prep for stage 1
        lang_proj_uncond_s1 = torch.zeros_like(lang_proj_s1) if cfg_scale != 1.0 else None

        # Build unknown_x_mask for stage 1 (ctx cannot see unknown trace X)
        unknown_x_mask_s1 = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        unknown_x_mask_s1[:, C:] = all_mask_s1[:, C:] & ~all_known_s1[:, C:]

        # Stage 1 DDPM loop
        scheduler = self.diffusion.gen_scheduler
        if n_steps_val is not None:
            scheduler.set_timesteps(n_steps_val)
        ddim_steps = list(scheduler.timesteps)
        total_ddim = len(ddim_steps)

        iterator_s1 = enumerate(ddim_steps)
        if verbose:
            iterator_s1 = tqdm(iterator_s1, total=total_ddim, desc="TwoStage S1")

        for ddim_idx, t_int in iterator_s1:
            x_t = torch.where(
                known_valid_s1.unsqueeze(-1).expand_as(x_t),
                canon_9d_s1, x_t,
            )
            x_t = x_t * rot_mask_flat_s1

            t_tensor = torch.full((B, N), int(t_int), device=device, dtype=torch.long)
            t_tensor[known_valid_s1] = 0
            diff_levels = t_tensor.float().reshape(B, C + T, P)
            lmm_input_flat = self._build_v11_input(x_t)
            lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

            attn_mask_t = traj_padding_mask_s1[:, None].expand(-1, T, -1)
            attn_mask_c = traj_padding_mask_s1[:, None].expand(-1, C, -1)

            _s1_lmm_kwargs = dict(
                x=lmm_input[:, C:],
                x_mask=attn_mask_t,
                x_time=real_timestep_s1,
                x_level=diff_levels[:, C:],
                feat=feat_s1,
                ctx=lmm_input[:, :C],
                ctx_mask=attn_mask_c,
                ctx_level=diff_levels[:, :C],
                add_tokens=lang_proj_s1.clone(),
                add_tokens_mask=lang_mask_s1.clone(),
                image_tokens=img_proj_s1.clone() if img_proj_s1 is not None else None,
                image_tokens_mask=img_mask_s1.clone() if img_mask_s1 is not None else None,
                group_ids=group_id_s1,
                unknown_x_mask=unknown_x_mask_s1,
            )
            if learned_traj_ids_s1 is not None:
                _s1_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_s1
            pred = self._lmm_motion_only(**_s1_lmm_kwargs)

            # V13.3.5: Classifier-free guidance
            if cfg_scale != 1.0:
                _uncond_kwargs = dict(_s1_lmm_kwargs)
                _uncond_kwargs["add_tokens"] = lang_proj_uncond_s1.clone()
                pred_uncond = self._lmm_motion_only(**_uncond_kwargs)
                pred = pred_uncond + cfg_scale * (pred - pred_uncond)

            # DDPM x0->pred_type conversion
            _pred_type = self.diffusion.prediction_type
            if _pred_type == "sample":
                pass
            else:
                _EPS = 1e-6
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_val = alphas_cp[int(t_int)]
                sqrt_alpha = safe_sqrt(alpha_val)
                sqrt_1m_alpha = safe_sqrt(1.0 - alpha_val).clamp(min=_EPS)
                x_t_reshaped = x_t.reshape(B, C + T, P, 9).float()
                if _pred_type == "epsilon":
                    pred = (x_t_reshaped - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                elif _pred_type == "v_prediction":
                    pred = (sqrt_alpha * x_t_reshaped - pred.float()) / sqrt_1m_alpha
                else:
                    raise ValueError(f"Unsupported prediction_type: {_pred_type}")
                pred = pred.to(x_t.dtype)

            pred_flat = pred.reshape(B * N, 9)
            prev_x = scheduler.step(
                pred_flat, int(t_int), x_t.reshape(B * N, 9)
            ).prev_sample.reshape(B, N, 9)

            x_t = torch.where(
                known_valid_s1.unsqueeze(-1).expand_as(x_t),
                x_t, prev_x,
            )

        # Final clamp known for stage 1
        x_t = torch.where(
            known_valid_s1.unsqueeze(-1).expand_as(x_t),
            canon_9d_s1, x_t,
        )

        # Decanonicalize stage 1 output to get SE3 for stage 2 input
        _s1_x = x_t
        if self.rot_param == "6d_wing":
            _s1_x = self._wing_to_canon_9d(_s1_x)
        s1_orig = self.canonicalizer.inverse(_s1_x, canon_params_s1)
        s1_orig = s1_orig.reshape(B, C + T, P, 9)
        s1_se3 = ninep_to_se3(s1_orig)
        s1_ctx_se3 = s1_se3[:, :C]  # Generated ctx in original space

        if verbose:
            logger.info("TwoStage S1 done: ctx generated")

        # ===================== Stage 2: Full attention =====================
        # History + generated ctx are all known
        s2_ctx_known = context_mask.clone()  # all valid ctx is "known" (from S1)
        s2_trace_known = orig_trace_known.clone()  # restore original known trace

        # Use generated ctx from stage 1
        s2_context = s1_ctx_se3.clone()

        # Prepare with full known mask (cano uses all known + generated ctx)
        prep_s2 = self._prepare_inference(
            trace=trace, context=s2_context,
            trace_known_mask=s2_trace_known,
            context_known_mask=s2_ctx_known,
            trace_mask=trace_mask, context_mask=context_mask,
            real_timestep=real_timestep,
            group_id=group_id, sem_type=sem_type, rot_mask=rot_mask,
            text=text, local_pcl=local_pcl, local_pcl_mask=local_pcl_mask,
            images=images, view_ids=view_ids,
            pixel_uv_normalized=pixel_uv_normalized,
            pixel_view_idx=pixel_view_idx,
            pixel_query_mask=pixel_query_mask,
            n_steps=n_steps,
        )
        canon_9d_s2 = prep_s2["canon_9d_flat"]
        canon_params_s2 = prep_s2["canon_params"]
        known_valid_s2 = prep_s2["known_valid_flat"]
        rot_mask_flat_s2 = prep_s2["rot_9d_mask_flat"]
        lang_proj_s2 = prep_s2["lang_proj"]; lang_mask_s2 = prep_s2["lang_mask"]
        img_proj_s2 = prep_s2["img_proj"]; img_mask_s2 = prep_s2["img_mask"]
        feat_s2 = prep_s2["feat"]
        traj_padding_mask_s2 = prep_s2["traj_padding_mask"]
        learned_traj_ids_s2 = prep_s2["learned_traj_ids_inf"]
        real_timestep_s2 = prep_s2["real_timestep"]
        group_id_s2 = prep_s2["group_id"]

        # Initialize stage 2 from noise, keeping known tokens clean
        g2 = torch.Generator(device=device)
        g2.manual_seed(seed + 1)
        noise_s2 = torch.randn(B, N, 9, device=device, generator=g2)
        x_t = torch.where(
            known_valid_s2.unsqueeze(-1).expand_as(noise_s2),
            canon_9d_s2, noise_s2,
        )
        x_t = x_t * rot_mask_flat_s2

        # CFG prep for stage 2
        lang_proj_uncond_s2 = torch.zeros_like(lang_proj_s2) if cfg_scale != 1.0 else None

        # CRITICAL: Stage 2 uses all-False unknown_x_mask for symmetric attention
        # This overrides ctx_unidirectional_mask=True from config
        symmetric_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)

        # Stage 2 DDPM loop — reset scheduler (DDIMScheduler is stateful)
        scheduler.set_timesteps(n_steps_val if n_steps_val is not None else self.n_inference_steps)
        ddim_steps_s2 = list(scheduler.timesteps)

        iterator_s2 = enumerate(ddim_steps_s2)
        if verbose:
            iterator_s2 = tqdm(iterator_s2, total=len(ddim_steps_s2), desc="TwoStage S2")

        for ddim_idx, t_int in iterator_s2:
            x_t = torch.where(
                known_valid_s2.unsqueeze(-1).expand_as(x_t),
                canon_9d_s2, x_t,
            )
            x_t = x_t * rot_mask_flat_s2

            t_tensor = torch.full((B, N), int(t_int), device=device, dtype=torch.long)
            t_tensor[known_valid_s2] = 0
            diff_levels = t_tensor.float().reshape(B, C + T, P)
            lmm_input_flat = self._build_v11_input(x_t)
            lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

            attn_mask_t = traj_padding_mask_s2[:, None].expand(-1, T, -1)
            attn_mask_c = traj_padding_mask_s2[:, None].expand(-1, C, -1)

            _s2_lmm_kwargs = dict(
                x=lmm_input[:, C:],
                x_mask=attn_mask_t,
                x_time=real_timestep_s2,
                x_level=diff_levels[:, C:],
                feat=feat_s2,
                ctx=lmm_input[:, :C],
                ctx_mask=attn_mask_c,
                ctx_level=diff_levels[:, :C],
                add_tokens=lang_proj_s2.clone(),
                add_tokens_mask=lang_mask_s2.clone(),
                image_tokens=img_proj_s2.clone() if img_proj_s2 is not None else None,
                image_tokens_mask=img_mask_s2.clone() if img_mask_s2 is not None else None,
                group_ids=group_id_s2,
                unknown_x_mask=symmetric_mask,
            )
            if learned_traj_ids_s2 is not None:
                _s2_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_s2
            pred = self._lmm_motion_only(**_s2_lmm_kwargs)

            # V13.3.5: Classifier-free guidance
            if cfg_scale != 1.0:
                _uncond_kwargs = dict(_s2_lmm_kwargs)
                _uncond_kwargs["add_tokens"] = lang_proj_uncond_s2.clone()
                pred_uncond = self._lmm_motion_only(**_uncond_kwargs)
                pred = pred_uncond + cfg_scale * (pred - pred_uncond)

            # DDPM x0->pred_type conversion
            _pred_type = self.diffusion.prediction_type
            if _pred_type == "sample":
                pass
            else:
                _EPS = 1e-6
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_val = alphas_cp[int(t_int)]
                sqrt_alpha = safe_sqrt(alpha_val)
                sqrt_1m_alpha = safe_sqrt(1.0 - alpha_val).clamp(min=_EPS)
                x_t_reshaped = x_t.reshape(B, C + T, P, 9).float()
                if _pred_type == "epsilon":
                    pred = (x_t_reshaped - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                elif _pred_type == "v_prediction":
                    pred = (sqrt_alpha * x_t_reshaped - pred.float()) / sqrt_1m_alpha
                else:
                    raise ValueError(f"Unsupported prediction_type: {_pred_type}")
                pred = pred.to(x_t.dtype)

            pred_flat = pred.reshape(B * N, 9)
            prev_x = scheduler.step(
                pred_flat, int(t_int), x_t.reshape(B * N, 9)
            ).prev_sample.reshape(B, N, 9)

            x_t = torch.where(
                known_valid_s2.unsqueeze(-1).expand_as(x_t),
                x_t, prev_x,
            )

        # Final clamp
        x_t = torch.where(
            known_valid_s2.unsqueeze(-1).expand_as(x_t),
            canon_9d_s2, x_t,
        )

        # C1: Convert back from wing to standard 9D before decanonicalization
        if self.rot_param == "6d_wing":
            x_t = self._wing_to_canon_9d(x_t)

        # Decanonicalize and convert back to SE3
        x_orig = self.canonicalizer.inverse(x_t, canon_params_s2)
        x_orig = x_orig.reshape(B, C + T, P, 9)
        result_se3 = ninep_to_se3(x_orig)

        ret_context = result_se3[:, :C]
        ret_trace = result_se3[:, C:]

        if is_training:
            self.train()

        return ret_context, ret_trace

    # ─────────────────────────────────────────────────────────────────
    # Joint denoise sampler with 3 modes
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_text(
        self,
        batch: dict,
        mode: str,                    # "from_ctx" | "from_x" | "from_ctx_x"
        n_steps: int = 50,
        g_text: float = 4.0,
        g_motion: float = 2.0,
        seed: int = 42,
    ) -> dict:
        """Joint denoise sampler for v14 text track.

        Mode contract:
          from_ctx:    ctx rows known   -> denoise (x rows + text)  joint
          from_x:      x rows known     -> denoise (ctx rows + text) joint
          from_ctx_x:  ctx + x known    -> denoise (text only)       text-only

        IMPORTANT: when ``self.ctx_always_known=True`` (the
        UMM default for v14 LT configs), the canonicalizer + scheduler will
        clamp ALL ctx tokens to "known" regardless of the mask passed here.
        Effect: ``from_x`` mode silently behaves as ``from_ctx_x`` for the
        motion side. The text side is unaffected. This is documented, not a
        bug -- to genuinely denoise ctx rows we would need to disable
        ctx_always_known in the config (which is not the v14 default).

        The motion side reuses ``self.inference()`` (DDPM/DDIM, FM Euler, or
        MeanFlow Euler depending on configured scheduler); the text side runs
        a SINGLE forward through ``self.lmm`` (1-step DDIM-like approximation).
        Per-track CFG follows the m=0 noised-text training distribution:
          - Text "uncond" pass replaces ``batch['text']`` with all-EOS-id
            sequences (neutral prompt, no learned null-embedding -- matches
            qa Q-spec rejecting extra params). Blend on text x0_pred.
          - Motion CFG goes through ``self.inference(cfg_scale=g_motion)``,
            which the existing inference() path already supports.

        Documented simplifications (acceptable per dispatch §5):
          1. Text denoising is 1-step. Multi-step text DDIM with shared
             timetable is left to a future SG.
          2. ``g_motion`` is consumed solely by ``self.inference``; the text
             pass does NOT also do motion-CFG (would require running the
             motion sampler 4x).
          3. The text noise input is sampled once at t=t_start (high noise)
             rather than progressively annealed -- matches the m=0 training
             distribution at the highest sampled t.

        Args:
            batch: Standard v14 batch dict. Must contain ``trace``, ``context``,
                ``trace_mask``, ``context_mask``, ``trace_sem_type``,
                ``trace_group_id``, ``trace_rot_mask``, ``time``, and ``text``
                (list[str] of length B). Optional: ``known_mask``,
                ``context_known_mask``, ``extra`` (DINO/pixel data).
            mode: One of {"from_ctx", "from_x", "from_ctx_x"}.
            n_steps: Number of motion denoising steps.
            g_text: Text CFG scale. 1.0 = no CFG.
            g_motion: Motion CFG scale (passed to ``self.inference``).
            seed: RNG seed for noise.

        Returns:
            dict with keys:
              pred_ids:               (B, L_text) torch.long argmax tokens.
              pred_text_clean:        list[str] decoded with skip_special=True.
              pred_text_with_special: list[str] decoded with skip_special=False.
              motion_context:         (B, C, P, 4, 4) generated/clean ctx.
              motion_trace:           (B, T, P, 4, 4) generated/clean trace.
              mode:                   the mode str.

        Raises:
            ValueError: If ``mode`` is not one of the 3 supported strings.
            RuntimeError: If text generation is not enabled on this Agent.
        """
        # ---- Input validation ----
        if mode not in {"from_ctx", "from_x", "from_ctx_x"}:
            raise ValueError(
                f"sample_text: mode must be one of "
                f"{{'from_ctx','from_x','from_ctx_x'}}, got {mode!r}"
            )
        if not self.text_gen_enable or self.text_codebook_head is None:
            raise RuntimeError(
                "sample_text requires text_gen.enable=True and a built "
                "TextCodebookHead. Got "
                f"text_gen_enable={self.text_gen_enable}, "
                f"text_codebook_head is None: {self.text_codebook_head is None}"
            )

        is_training = self.training
        if is_training:
            self.eval()

        # ---- Pull required batch tensors ----
        trace = batch["trace"]              # (B, T, P, 4, 4)
        context = batch["context"]          # (B, C, P, 4, 4)
        trace_mask = batch["trace_mask"]    # (B, T, P) bool
        context_mask = batch["context_mask"]  # (B, C, P) bool
        sem_type = batch["trace_sem_type"]
        group_id = batch["trace_group_id"]
        rot_mask = batch["trace_rot_mask"]
        real_timestep = batch["time"]
        device = trace.device
        B, T, P = trace_mask.shape
        C = context_mask.shape[1]

        # ---- Build mode-specific known masks ----
        zero_trace_known = torch.zeros_like(trace_mask, dtype=torch.bool)
        zero_ctx_known = torch.zeros_like(context_mask, dtype=torch.bool)
        full_trace_known = trace_mask.bool()
        full_ctx_known = context_mask.bool()

        if mode == "from_ctx":
            # Ctx known; x + text denoised
            ctx_known_mask = full_ctx_known
            trace_known_mask = zero_trace_known
        elif mode == "from_x":
            # X known; ctx + text denoised
            ctx_known_mask = zero_ctx_known
            trace_known_mask = full_trace_known
        else:  # from_ctx_x
            # Ctx + x known; text only denoised
            ctx_known_mask = full_ctx_known
            trace_known_mask = full_trace_known

        # ---- Optional inputs (DINO + pixel queries) ----
        # Pull from batch["extra"] if present (matches forward() convention).
        extra = batch.get("extra", {}) or {}
        # When extra holds list-of-arrays per sample, hand them to inference()
        # via dedicated kwargs only if all B samples are present and stackable.
        # Fail-soft: if any required key is missing, skip the optional pathway.
        local_pcl = batch.get("local_pcl", None)
        local_pcl_mask = batch.get("local_pcl_mask", None)
        # Note: images/pixel_* are passed as stacked tensors only if the caller
        # has done that; raw extras-as-list is consumed inside _prepare_inference
        # via inf_batch wiring -- we don't synthesize that here.
        text_list = batch.get("text", None)

        # ---- 1. Motion side: defer to existing inference ----
        # Use cfg_scale=g_motion so existing per-track motion CFG is honoured.
        motion_context, motion_trace = self.inference(
            trace=trace,
            context=context,
            trace_known_mask=trace_known_mask,
            context_known_mask=ctx_known_mask,
            trace_mask=trace_mask,
            context_mask=context_mask,
            real_timestep=real_timestep,
            group_id=group_id,
            sem_type=sem_type,
            rot_mask=rot_mask,
            text=text_list,
            local_pcl=local_pcl,
            local_pcl_mask=local_pcl_mask,
            seed=seed,
            n_steps=n_steps,
            verbose=False,
            cfg_scale=g_motion,
        )

        # ---- 2. Text side: 1-step text x0 prediction via direct LMM call ----
        # We need motion-side conditioning identical to the inference() loop's
        # final step so the text head sees the same context distribution. Use
        # _prepare_inference once and feed the motion stream a noised x_t at
        # the start-of-denoise timestep.
        prep = self._prepare_inference(
            trace=trace, context=context,
            trace_known_mask=trace_known_mask,
            context_known_mask=ctx_known_mask,
            trace_mask=trace_mask, context_mask=context_mask,
            real_timestep=real_timestep,
            group_id=group_id, sem_type=sem_type, rot_mask=rot_mask,
            text=text_list, local_pcl=local_pcl, local_pcl_mask=local_pcl_mask,
            n_steps=n_steps,
        )
        N = prep["N"]
        canon_9d_flat = prep["canon_9d_flat"]
        known_valid_flat = prep["known_valid_flat"]
        rot_9d_mask_flat = prep["rot_9d_mask_flat"]
        lang_proj = prep["lang_proj"]; lang_mask = prep["lang_mask"]
        img_proj = prep["img_proj"]; img_mask = prep["img_mask"]
        feat = prep["feat"]
        traj_padding_mask = prep["traj_padding_mask"]
        learned_traj_ids_inf = prep["learned_traj_ids_inf"]
        real_timestep_p = prep["real_timestep"]
        group_id_p = prep["group_id"]

        # Noised motion x_t at start-of-denoise (highest noise). Known tokens
        # clamped to clean canonical values to mirror the inference() loop's
        # first-step contract.
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise_motion = torch.randn(B, N, 9, device=device, generator=g)
        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(noise_motion),
            canon_9d_flat,
            noise_motion,
        )
        x_t = x_t * rot_9d_mask_flat

        # Diffusion-level scalar at start-of-denoise.
        if self.is_flow_matching or self.is_meanflow:
            t_start = 1.0
            t_levels = torch.full(
                (B, N), t_start, device=device, dtype=x_t.dtype,
            )
            t_levels[known_valid_flat] = 0.0
            diff_levels = (t_levels * self.diffusion_time_scale).reshape(
                B, C + T, P,
            )
            r_diff_levels = (
                torch.zeros_like(t_levels) * self.diffusion_time_scale
            ).reshape(B, C + T, P) if self.is_meanflow else None
        else:
            t_int = int(self.diffusion.num_train_timesteps - 1)
            t_tensor = torch.full(
                (B, N), t_int, device=device, dtype=torch.long,
            )
            t_tensor[known_valid_flat] = 0
            diff_levels = t_tensor.float().reshape(B, C + T, P)
            r_diff_levels = None

        lmm_input_flat = self._build_v11_input(x_t)
        lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)
        attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
        attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

        # ---- Build TEXT INPUT (cond pass): noised text matching m=0 ----
        tokenizer = self.input_layer.language_encoder.tokenizer
        if text_list is None:
            text_list = [""] * B
        text_ids_cond, _text_mask = _tokenize_with_eos_fill(
            text_list, tokenizer, self.text_gen_L_text,
        )
        text_ids_cond = text_ids_cond.to(device)
        text_mask = torch.ones(
            B, self.text_gen_L_text, device=device, dtype=torch.bool,
        )

        # Codebook lookup for the conditional target (only needed for the
        # noised input under FM/MeanFlow at t=1 -- pure noise -- we still
        # synthesize a noisy_text from the codebook to share dtype/shape).
        codebook = self.codebook
        if torch.is_autocast_enabled():
            codebook = codebook.to(dtype=torch.get_autocast_gpu_dtype())
        x0_text_cond = codebook[text_ids_cond]      # (B, L_text, 768)

        # Noised text at start-of-denoise. Per dispatch: the m=0 path during
        # training learns to recover x0 from any timestep, so a single forward
        # at the highest noise level produces a usable x0_pred.
        gtxt = torch.Generator(device=device)
        gtxt.manual_seed(seed + 17)
        noise_text = torch.randn(
            B, self.text_gen_L_text, 768,
            device=device, dtype=x0_text_cond.dtype, generator=gtxt,
        )
        # reviewer LATENT-RISK: pre-declare t_text so the CFG block
        # below cannot NameError if any future scheduler path is added.
        # FM/MeanFlow set it to None; DDPM overrides with the real timestep.
        t_text: Optional[torch.Tensor] = None
        if self.is_flow_matching or self.is_meanflow:
            # FM: x_t = (1-t)*x0 + t*eps; at t=1 this is pure noise.
            noisy_text = noise_text
        else:
            L_text_dim = self.text_gen_L_text
            t_text = torch.full(
                (B,),
                int(self.diffusion.num_train_timesteps - 1),
                device=device, dtype=torch.long,
            )
            noisy_text = self.diffusion.noise_scheduler.add_noise(
                original_samples=x0_text_cond.reshape(B * L_text_dim, 768),
                noise=noise_text.reshape(B * L_text_dim, 768),
                timesteps=t_text.repeat_interleave(L_text_dim),
            ).reshape(B, L_text_dim, 768)

        text_tokens_d1 = self.text_codebook_head.forward_input(noisy_text)

        # ---- LMM forward (cond pass) ----
        # NOTE: _lmm_motion_only strips the tuple, so call self.lmm directly
        # to recover x0_text_pred from the tuple contract. text_gen=True
        # already disables add_tokens at training; mirror that here by passing
        # add_tokens=None so train/inference distributions match.
        _lmm_kwargs = dict(
            x=lmm_input[:, C:],
            x_mask=attn_mask_t,
            x_time=real_timestep_p,
            x_level=diff_levels[:, C:],
            feat=feat,
            ctx=lmm_input[:, :C],
            ctx_mask=attn_mask_c,
            ctx_level=diff_levels[:, :C],
            add_tokens=None,
            add_tokens_mask=None,
            image_tokens=img_proj.clone() if img_proj is not None else None,
            image_tokens_mask=img_mask.clone() if img_mask is not None else None,
            group_ids=group_id_p,
            text_tokens=text_tokens_d1,
            text_tokens_mask=text_mask,
            text_codebook_head=self.text_codebook_head,
        )
        if r_diff_levels is not None:
            _lmm_kwargs["x_r_level"] = r_diff_levels[:, C:]
            _lmm_kwargs["ctx_r_level"] = r_diff_levels[:, :C]
        if learned_traj_ids_inf is not None:
            _lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf

        out_cond = self.lmm(**_lmm_kwargs)
        if not isinstance(out_cond, tuple):
            raise RuntimeError(
                "sample_text: expected (motion, text_x0_pred) tuple from LMM "
                "with text_tokens set, got single tensor. Check the wiring."
            )
        _motion_cond, x0_text_pred_cond = out_cond  # (B, L_text, 768)

        # ---- CFG: optional uncond pass with EOS-filled neutral text ----
        if g_text != 1.0:
            eos_id = int(tokenizer.eos_token_id)
            text_ids_uncond = torch.full(
                (B, self.text_gen_L_text), eos_id, device=device,
                dtype=torch.long,
            )
            x0_text_uncond_target = codebook[text_ids_uncond]
            # Use the SAME noise tensor for fair CFG blending.
            if self.is_flow_matching or self.is_meanflow:
                noisy_text_uncond = noise_text
            else:
                if t_text is None:
                    raise RuntimeError(
                        "sample_text CFG: t_text is None on a non-FM/MeanFlow "
                        "path -- new scheduler added without wiring t_text. "
                        "Ensure the scheduler branch above sets t_text."
                    )
                noisy_text_uncond = self.diffusion.noise_scheduler.add_noise(
                    original_samples=x0_text_uncond_target.reshape(
                        B * self.text_gen_L_text, 768,
                    ),
                    noise=noise_text.reshape(B * self.text_gen_L_text, 768),
                    timesteps=t_text.repeat_interleave(self.text_gen_L_text),
                ).reshape(B, self.text_gen_L_text, 768)
            text_tokens_d1_uncond = self.text_codebook_head.forward_input(
                noisy_text_uncond,
            )
            _uncond_kwargs = dict(_lmm_kwargs)
            _uncond_kwargs["text_tokens"] = text_tokens_d1_uncond
            out_uncond = self.lmm(**_uncond_kwargs)
            if not isinstance(out_uncond, tuple):
                raise RuntimeError(
                    "sample_text: uncond LMM pass returned non-tuple. "
                    "Check the wiring."
                )
            _motion_uncond, x0_text_pred_uncond = out_uncond
            x0_text_pred = (
                x0_text_pred_uncond
                + g_text * (x0_text_pred_cond - x0_text_pred_uncond)
            )
        else:
            x0_text_pred = x0_text_pred_cond

        # ---- Final NN-round to token ids ----
        cb = self.codebook.to(dtype=x0_text_pred.dtype)
        # (B, L_text, V): score by dot-product against codebook rows.
        scores = x0_text_pred @ cb.T
        pred_ids = scores.argmax(dim=-1)              # (B, L_text)

        # ---- Decode strings (CPU, Python loop -- inference only) ----
        pred_ids_cpu = pred_ids.detach().cpu()
        pred_text_clean = tokenizer.batch_decode(
            pred_ids_cpu, skip_special_tokens=True,
        )
        pred_text_with_special = tokenizer.batch_decode(
            pred_ids_cpu, skip_special_tokens=False,
        )

        if is_training:
            self.train()

        # critic W1: self.inference() runs under @torch.inference_mode(),
        # which marks returned tensors as inference_mode_tensor. Calling code
        # outside no_grad cannot save those for backward / autograd-track them.
        # Clone+detach to plain tensors so callers (viz code, downstream tests)
        # can use them freely.
        return {
            "pred_ids": pred_ids.detach().clone(),
            "pred_text_clean": pred_text_clean,
            "pred_text_with_special": pred_text_with_special,
            "motion_context": motion_context.detach().clone(),
            "motion_trace": motion_trace.detach().clone(),
            "mode": mode,
        }

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def get_optim_groups(
        self,
        weight_decay: float,
        lmm_lr: float,
        input_layer_lr: float,
        image_encoder_lr: Optional[float] = None,
        **kwargs,
    ):
        """Get optimizer parameter groups with per-component learning rates.

        When ``use_dino`` is True, also includes the DINO encoder's
        trainable params (view_emb / proj / ln). ``image_encoder_lr``
        controls their base LR; if ``None``, ``lmm_lr`` is used as a sane
        fallback so the DINO aux modules never get a missing LR.
        """
        # Agent-level params (V11: lang_token_proj, plus DINO projectors
        # when enabled). Projectors share ``input_layer_lr``.
        agent_params = list(self.lang_token_proj.parameters())
        if self.use_dino:
            agent_params += list(self.image_token_proj.parameters())
            agent_params += list(self.add_token_type_emb.parameters())
            agent_params += list(self.dino_geo_proj.parameters())
        # V13.2 V6: input cross-attn params share the agent/input_layer LR.
        # Only present when input_xattn.enable is True; otherwise stays None
        # and its params never exist.
        if self.input_xattn_enable and self.input_xattn is not None:
            agent_params += list(self.input_xattn.parameters())
        # V13.2 Explore 10: input_xattn_v2 params share input_layer LR.
        if self.input_xattn_v2_enable and self.input_xattn_v2 is not None:
            agent_params += list(self.input_xattn_v2.parameters())
        # reviewer FIX-REQUIRED: TextCodebookHead trainable params
        # (text_input_proj, text_head, text_token_projs, text_pos_emb,
        #  role_emb_d1, role_emb_d2) MUST be in the optimizer or the joint
        # text loss flows gradients into a frozen head -- silent broken
        # training. Share input_layer_lr (matches the lang/image projectors).
        if self.text_gen_enable and self.text_codebook_head is not None:
            agent_params += list(self.text_codebook_head.parameters())
        agent_group = [
            {"params": agent_params, "weight_decay": weight_decay, "lr": input_layer_lr}
        ]

        groups = (
            self.lmm.get_optim_groups(weight_decay=weight_decay, base_lr=lmm_lr)
            + self.input_layer.get_optim_groups(weight_decay=weight_decay, base_lr=input_layer_lr)
            + agent_group
        )
        if self.use_dino:
            _img_lr = image_encoder_lr if image_encoder_lr is not None else lmm_lr
            groups += self.dino_encoder.get_optim_groups(
                weight_decay=weight_decay, base_lr=_img_lr
            )
        return groups

    @staticmethod
    def filter_ckpt_prefix(checkpoint_dict, prefix):
        """Filter and strip key prefix from checkpoint dict."""
        return {
            k[len(prefix):]: v
            for k, v in checkpoint_dict.items()
            if k.startswith(prefix)
        }

    def load_from_checkpoint(
        self, checkpoint_dict, lmm_flag=True, input_flag=True, strict=True
    ):
        """Load LMM and/or input layer weights from checkpoint."""
        if lmm_flag:
            lmm_dict = self.filter_ckpt_prefix(checkpoint_dict, "lmm.")
            self.lmm.load_state_dict(lmm_dict, strict=strict)
        if input_flag:
            input_dict = self.filter_ckpt_prefix(checkpoint_dict, "input_layer.")
            self.input_layer.load_state_dict(input_dict, strict=strict)

    def convert_smplx_hand_param(self, Q: torch.Tensor, to_local: bool = True) -> torch.Tensor:
        """Convert SMPL-X hand joints between wrist-local and global space.

        Transforms the 15 left-hand and 15 right-hand finger joints
        relative to (or back from) their respective wrist frames.
        Assumes the standard SMPL-X 55-joint ordering.

        Args:
            Q: Joint transforms ``(B, 55, 4, 4)``.
            to_local: True → global to local; False → local to global.

        Returns:
            Modified transforms ``(B, 55, 4, 4)``.
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

    @staticmethod
    def get_type_mask(uids: torch.Tensor, target_uids) -> torch.Tensor:
        """Return boolean mask where uids match any target."""
        if isinstance(target_uids, (int, float)):
            target_uids = [int(target_uids)]
        mask = torch.zeros_like(uids, dtype=torch.bool)
        for uid in target_uids:
            mask |= uids == uid
        return mask
