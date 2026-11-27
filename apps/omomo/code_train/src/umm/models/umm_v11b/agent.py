"""AgentV11B: FM-only, synced noise, MLP-based P-layer LMM.

V11B changes from V11:
- FM-only (no DDPM, no MeanFlow)
- Synced noise only (one t per sample, no async, no 50/50 branching)
- Loss on ALL valid slots (known + unknown), not just unknown
- Uses MotionMachineV11B world model (MLP input/output + P-layers)
- Agent passes noise_level as (B,) scalar tensor to the LMM
- No network_output_space conversion (always x0)
- No loss_space config (always native x0)
- No min-SNR reweighting

Example::

    from umm.models.umm_v11b.agent import AgentV11B
    agent = AgentV11B(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from umm.models.umm_v11b.rotation_utils import se3_to_9d, ninep_to_se3, sixd_to_rotmat
from umm.models.umm_v11b.canonicalization import (
    Canonicalizer,
    CanonParams,
    compute_canon_params,
    canonicalize,
)
from umm.models.umm_v11b.diffusion import (
    FlowMatchingScheduler,
    build_diffusion_scheduler,
)

# Shared infrastructure -- truly generic, not V9-specific
from umm.models.umm_v9.world_models.language import T5LanguageEncoder
from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    disable_model_grad,
)
from umm.models.umm_v9.world_models.pointnet import ResnetPointnet
from umm.models.umm_v9.loss_utils import type_mean_compiled as type_mean

logger = logging.getLogger(__name__)


class InputLayerV11B(nn.Module):
    """Input layer for V11B: type + group + geo (PointNet) embeddings + language.

    Args:
        latent_dim: Embedding dimension.
        feat_dim: Output feature dimension.
        num_types: Number of semantic types.
        num_groups: Number of group IDs.
        lang_fixed_length: Fixed token length for language encoder.
        point_encoder_latent: Hidden dim for PointNet encoder. -1 disables.
        pcl_input_mode: "xyz" (3D) or "xyz_rgb" (6D) point cloud input.
        point_xyz_scale_factor: Scale factor applied to xyz coords before PointNet.
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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

        self.type_emb = nn.Embedding(num_types, latent_dim)
        self.group_emb = nn.Embedding(num_groups, latent_dim)
        # Init all group embeddings to same value for training robustness
        with torch.no_grad():
            self.group_emb.weight.copy_(
                self.group_emb.weight[0].expand_as(self.group_emb.weight)
            )

        self.ln_type = HardLayerNorm(latent_dim)
        self.ln_group = HardLayerNorm(latent_dim)

        # --- PointNet geo embedding ---
        self.point_flag = point_encoder_latent > 0
        self.point_xyz_scale_factor = point_xyz_scale_factor
        if pcl_input_mode not in ("xyz", "xyz_rgb"):
            raise ValueError(f"Unknown pcl_input_mode: {pcl_input_mode!r}. Expected 'xyz' or 'xyz_rgb'.")
        self.point_input_dim = 3 if pcl_input_mode == "xyz" else 6
        self.empty_point_emb = nn.Embedding(1, latent_dim)
        if self.point_flag:
            point_input_dim = self.point_input_dim
            self.pntnet = ResnetPointnet(
                c_dim=latent_dim, dim=point_input_dim,
                hidden_dim=point_encoder_latent,
            )
            self.geo_proj = nn.Linear(latent_dim, latent_dim)
        self.ln_geo = HardLayerNorm(latent_dim)

        # type + group + geo -> feat
        self.bind_fc = nn.Linear(3 * latent_dim, feat_dim)

        # Language encoder (frozen T5)
        self.lang_fixed_len = lang_fixed_length
        self.language_encoder = disable_model_grad(
            T5LanguageEncoder(fixed_length=self.lang_fixed_len)
        )
        self.lang_fc = nn.Linear(768, latent_dim)

    @torch.compiler.disable
    def encode_language(self, text_in):
        """Encode text -> (B, lang_fixed_len, latent_dim). No grad."""
        tokens = self.language_encoder.encode(text_in)
        tokens = self.lang_fc(tokens)
        return tokens.detach()

    @torch.compiler.disable
    def forward(
        self,
        sem_id: torch.Tensor,
        group_id: torch.Tensor,
        pcl: Optional[torch.Tensor] = None,
        pcl_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-entity feature from type, group, and geo embeddings.

        Args:
            sem_id: (B, P) semantic type IDs.
            group_id: (B, P) group IDs.
            pcl: (B, P, M, D_pcl) local point clouds, or None.
            pcl_valid_mask: (B, P) bool mask for valid PCL entities, or None.

        Returns:
            feat: (B, P, feat_dim).
        """
        type_emb = self.ln_type(self.type_emb(sem_id))    # (B, P, D)
        group_emb = self.ln_group(self.group_emb(group_id))  # (B, P, D)

        # Geo embedding (PointNet for valid PCL, empty embedding otherwise)
        B, N = sem_id.shape
        geo_emb = self.empty_point_emb(
            torch.tensor([0], device=sem_id.device)
        )[None].repeat(B, N, 1).contiguous()  # (B, P, D)
        if self.point_flag and pcl is not None and pcl_valid_mask is not None:
            if pcl_valid_mask.any():
                valid_pcl = pcl[pcl_valid_mask][..., :self.point_input_dim].clone()  # (K, M, point_input_dim)
                valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
                valid_pcl_feat = self.geo_proj(self.pntnet(valid_pcl))  # (K, D)
                geo_emb[pcl_valid_mask] = geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
        geo_emb = self.ln_geo(geo_emb)

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


class AgentV11B(nn.Module):
    """V11B Agent: FM-only, synced noise, MLP-based P-layer LMM.

    Args:
        cfg: Full configuration dictionary.
    """

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

        # Canonicalization
        canon_cfg = self.model_cfg.get("canonicalization", {})
        self.target_pos_std = canon_cfg.get("target_pos_std", 0.5)
        self.canonicalizer = Canonicalizer(target_pos_std=self.target_pos_std)

        # C1: Rotation parameterization -- "6d_unit" (default) or "6d_wing"
        self.rot_param = self.model_cfg.get("rot_parameterization", "6d_unit")
        self.wing_len = self.model_cfg.get("rot_wing_len", 0.2)

        # Diffusion scheduler (FM or DDPM)
        diffusion_cfg = self.model_cfg.get("diffusion", {})
        self.diffusion = build_diffusion_scheduler(diffusion_cfg)
        self.diffusion_time_scale = diffusion_cfg.get("time_scale", 1000.0)
        self.n_inference_steps = diffusion_cfg.get(
            "n_inference_steps",
            diffusion_cfg.get("gen_timesteps", 50),
        )

        # Loss config
        self.weight_by = self.optim_cfg.get("weight_by", "uid")
        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }

        # Context always known (never noised during training)
        self.ctx_always_known = self.optim_cfg.get("ctx_always_known", False)

        # Stable loss: Smooth L1 (Huber) with configurable beta.
        # None = MSE (default for backward compat).
        self.loss_beta = self.optim_cfg.get("loss_beta", None)

        # R_geodesic loss weight (default 0 = disabled, must set in config)
        self.w_R = self.optim_cfg.get("w_R", 0.0)

        # H1: Virtual/real loss weighting (context frames down-weighted)
        self.w_real = self.optim_cfg.get("w_real", 1.0)
        self.w_virtual = self.optim_cfg.get("w_virtual", 1.0)

        # M5: Movement reweighting -- up-weight moving entities
        self.movement_reweight_min_ratio = self.optim_cfg.get(
            "movement_reweight_min_ratio", 0.0
        )
        self.movement_reweight_max_ratio = self.optim_cfg.get(
            "movement_reweight_max_ratio", 10.0
        )
        self.movement_scale = self.optim_cfg.get("movement_scale", 1.0)

        # --- Input layer (type + group + language) ---
        input_cfg = self.model_cfg.get("input_layer", {})
        self.input_layer = InputLayerV11B(
            latent_dim=self.latent_dim,
            feat_dim=self.model_cfg.get("feat_dim", 512),
            num_types=input_cfg.get("num_types", 1024),
            num_groups=input_cfg.get("num_groups", 1024),
            lang_fixed_length=input_cfg.get("language_fixed_length", 1),
            point_encoder_latent=input_cfg.get("point_encoder_latent", -1),
            pcl_input_mode=input_cfg.get("pcl_input_mode", "xyz"),
            point_xyz_scale_factor=input_cfg.get("point_xyz_scale_factor", 1.0),
        )

        # --- LMM: MotionMachineV11B (MLP input/output + P-layers) ---
        lmm_cfg = self.model_cfg.get("lmm", {})
        from umm.models.umm_v11b.world_models.moma_v11b import MotionMachineV11B
        self.lmm = MotionMachineV11B(
            C_P=self.C_P,
            feat_dim=self.model_cfg.get("feat_dim", 512),
            n_p_layers=lmm_cfg.get("n_p_layers", 3),
            n_head_p=lmm_cfg.get("n_head_p", 16),
            len_ctx=lmm_cfg.get("len_ctx", 1),
            max_T=max(
                (dc.get("max_T", 30) for dc in cfg.get("dataset", {}).get("configs", [{}])),
                default=30,
            ),
            num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
            time_multiplier=lmm_cfg.get("time_multiplier", 30.0),
            ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", 1.0),
            gate_bound=lmm_cfg.get("gate_bound", 2.0),
            ada_n_layers=lmm_cfg.get("ada_n_layers", 2),
            fc_factor_p=lmm_cfg.get("fc_factor_p", 2),
            fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
            add_time_to_feat=lmm_cfg.get("add_time_to_feat", True),
            qk_norm=lmm_cfg.get("qk_norm", True),
        )

        # Language token projector -> LMM's add_token width (C_P)
        self.lang_token_proj = nn.Linear(self.latent_dim, self.C_P)

        # Text masking prob for classifier-free guidance
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)

        logger.info(
            f"AgentV11B: fm_only=True, synced_noise=True, "
            f"target_pos_std={self.target_pos_std}"
        )

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------

    @staticmethod
    @torch.compiler.disable
    def _check_finite(tensor: torch.Tensor):
        """Check for NaN/Inf. Compile-unsafe (CPU-GPU sync)."""
        if not tensor.isfinite().all():
            n_bad = (~tensor.isfinite()).sum().item()
            raise RuntimeError(
                f"NaN/Inf in LMM output: {n_bad}/{tensor.numel()} elements "
                f"({100*n_bad/tensor.numel():.1f}%)"
            )

    @torch.compiler.disable
    def _encode_text_with_masking(self, text_list, B, masked_prob):
        """Encode text with classifier-free masking. Compile-unsafe."""
        text_masked = torch.rand(B) < masked_prob
        text = [text_list[i] if not text_masked[i] else "" for i in range(B)]
        return self.input_layer.encode_language(text)

    def _build_v11_input(self, noisy_x: torch.Tensor) -> torch.Tensor:
        """V11B: simple pass-through of noisy_x. No 3-cat input.

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
            pred: (B, C+T, P, 9) model output (x0 prediction).
            target: (B, C+T, P, 9) target (clean x0).
            valid_mask: (B, C+T, P) bool -- ALL valid tokens (known + unknown).
            rot_mask: (B, P) bool -- entities with rotation.
            loss_type_ids: (B, P) int -- IDs for type_mean grouping.
            phase: "train" or "val".
            token_weights: Optional (B, C+T, P) per-token weights.
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
        if pos_scale is not None:
            pos_loss = pos_loss * pos_scale  # (B, C+T, P) * (B, 1, 1)

        # Apply per-token weights
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

    # -----------------------------------------------------------------
    # Forward (training)
    # -----------------------------------------------------------------

    def forward(self, batch: dict, phase: str, global_step: int = 0) -> dict:
        """Forward pass for training and validation.

        V11B flow:
        1. Extract batch -> concat ctx+trace
        2. SE3 -> 9D (pos + 6D rot)
        3. Canonicalize using known+valid tokens
        4. Add synced FM noise (known=clean, unknown=noised, invalid=pure noise)
        5. Build input (noisy_x pass-through)
        6. LMM -> x0 prediction (with noise_level scalar)
        7. Loss on ALL valid tokens in canonicalized 9D space

        Args:
            batch: Batch dict from dataset collate.
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

        # 1. Concatenate context + trace
        all_se3 = torch.cat([context, trace], dim=1)       # (B, C+T, P, 4, 4)
        all_mask = torch.cat([context_mask, trace_mask], dim=1)  # (B, C+T, P) bool
        all_known = torch.cat([ctx_known_mask, known_mask], dim=1)  # (B, C+T, P) bool

        # Context always known option
        if self.ctx_always_known:
            all_known[:, :C] = context_mask[:, :C]

        # 2. SE3 -> 9D
        all_9d = se3_to_9d(all_se3)  # (B, C+T, P, 9)

        # Zero out rotation for position-only entities
        rot_9d_mask = torch.ones(B, 1, P, 9, device=device, dtype=all_9d.dtype)
        rot_9d_mask[..., 3:] = rot_mask[:, None, :, None].expand(-1, 1, -1, 6).float()
        all_9d = all_9d * rot_9d_mask

        # 3. Canonicalize using known+valid tokens
        # Flatten (C+T, P) -> N for canonicalization
        N = (C + T) * P
        all_9d_flat = all_9d.reshape(B, N, 9)                    # (B, N, 9)
        all_mask_flat = all_mask.reshape(B, N)                    # (B, N)
        all_known_flat = all_known.reshape(B, N)                  # (B, N)
        known_valid_flat = all_known_flat & all_mask_flat          # (B, N) -- for canon

        canon_9d_flat, canon_params = self.canonicalizer.forward(
            all_9d_flat, known_valid_flat
        )  # (B, N, 9), CanonParams

        # C1: Convert canonicalized 9D to wing space if configured.
        if self.rot_param == "6d_wing":
            canon_9d_flat = self._canon_9d_to_wing(canon_9d_flat)

        # 4. Add synced FM noise
        # V11B: synced noise, clean-known mode always.
        # known+valid -> t=0, unknown+valid -> global t, invalid -> t=1.0
        noisy_flat, t_levels, noise_flat, target_flat = self.diffusion.add_noise(
            canon_9d_flat, valid_mask=all_mask_flat, known_mask=all_known_flat,
        )

        # Extract the global noise level per sample for noise_level conditioning.
        # Shape: (B,) scalar noise level per sample.
        # unknown+valid tokens carry the actual noise level; known have 0, invalid have max.
        unknown_valid_flat = ~all_known_flat & all_mask_flat  # (B, N)
        if self.diffusion.noise_type == "ddpm":
            # DDPM: t_levels already has integer timestep for unknown+valid tokens
            # Known tokens have 0, invalid have num_train_timesteps-1
            # Take the timestep from any unknown+valid token
            t_masked = torch.where(unknown_valid_flat, t_levels, torch.full_like(t_levels, -1.0))
            noise_level = t_masked.max(dim=1).values  # (B,)
            noise_level = noise_level.clamp(min=0.0)
        else:
            # FM: recover global t from unknown_valid tokens
            t_per_sample = t_levels.reshape(B, N)
            t_masked = torch.where(unknown_valid_flat, t_per_sample, torch.full_like(t_per_sample, -1.0))
            noise_level = t_masked.max(dim=1).values  # (B,)
            # If no unknown+valid tokens exist in a sample, noise_level = -1 -> clamp to 0
            noise_level = noise_level.clamp(min=0.0)

        # Scale t_levels for LMM timestep embedding
        diffusion_levels = t_levels * self.diffusion_time_scale  # (B, N)

        # Invalid tokens -> fully noised (already handled by diffusion: t=1)
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
        rot_9d_mask_flat = rot_9d_mask.reshape(B, 1, P, 9).expand(-1, C + T, -1, -1).reshape(B, N, 9)
        noisy_flat = noisy_flat * rot_9d_mask_flat

        # 5. Build V11B input (pass-through)
        lmm_input_flat = self._build_v11_input(noisy_flat)  # (B, N, 9)

        # Reshape back to (B, C+T, P, 9)
        lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)
        diff_levels = diffusion_levels.reshape(B, C + T, P)

        # V11B: pass per-timestep validity masks to LMM.
        # The LMM uses these for: (1) invalid_emb substitution per-timestep,
        # (2) mask feature in MLP input. Entity-level padding for attention
        # is derived inside the LMM from mask_pe.sum(dim=-1) > 0.
        attn_mask_t = trace_mask  # (B, T, P) per-timestep validity
        attn_mask_c = context_mask  # (B, C, P) per-timestep validity

        # 6. Language tokens
        if self.text_masked_prob > 0.0 and self.training:
            lang_tk = self._encode_text_with_masking(
                batch["text"], B, self.text_masked_prob
            )
        else:
            lang_tk = self.input_layer.encode_language(batch["text"])

        lang_proj = self.lang_token_proj(lang_tk.clone())  # (B, L, C_P)
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)

        # Entity features
        local_pcl = batch.get("local_pcl", None)  # (B, P, PCL, D_pcl) or None
        local_pcl_mask = batch.get("local_pcl_mask", None)  # (B, P) bool or None
        feat = self.input_layer(sem_type, group_id, pcl=local_pcl, pcl_valid_mask=local_pcl_mask)  # (B, P, feat_dim)

        # 7. LMM forward -- V11B passes noise_level as (B,) scalar
        out = self.lmm(
            x=lmm_input[:, C:],        # (B, T, P, 9)
            x_mask=attn_mask_t,         # (B, T, P)
            x_time=timestep,            # (B, T)
            x_level=diff_levels[:, C:], # (B, T, P) -- per-token levels for compat
            feat=feat,                  # (B, P, feat_dim)
            ctx=lmm_input[:, :C],       # (B, C, P, 9)
            ctx_mask=attn_mask_c,       # (B, C, P)
            ctx_level=diff_levels[:, :C],  # (B, C, P)
            noise_level=noise_level,    # (B,) GLOBAL noise level
            add_tokens=lang_proj,
            add_tokens_mask=lang_mask,
            group_ids=group_id,
        )  # (B, C+T, P, 9)

        self._check_finite(out)

        # V11B: LMM always outputs x0 -- no conversion needed
        x0_pred_canon = out

        # x0 ground truth = clean canonicalized poses
        x0_gt_canon = canon_9d_flat.reshape(B, C + T, P, 9)

        # C1: Convert x0 from wing to standard 9D for GT-canon metrics,
        # decanonicalization, and SE3 conversion.
        if self.rot_param == "6d_wing":
            x0_pred_canon_std = self._wing_to_canon_9d(x0_pred_canon)
            x0_gt_canon_std = self._wing_to_canon_9d(x0_gt_canon)
        else:
            x0_pred_canon_std = x0_pred_canon
            x0_gt_canon_std = x0_gt_canon

        # 8. GT-canon: extract scale factor for loss + compute GT-canon x0.
        # V11B uses LINEAR s (not s^2).
        gt_canon_params = compute_canon_params(
            canon_9d_flat, all_mask_flat, self.target_pos_std
        )
        s_linear = gt_canon_params.scale.unsqueeze(-1)  # (B, 1, 1) LINEAR

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

        # 9. Compute loss: x0 prediction vs clean x0, on ALL valid tokens
        loss_pred = x0_pred_canon
        loss_target = x0_gt_canon

        # H1: Virtual/real mask -- context frames are "virtual"
        virtual_mask = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        virtual_mask[:, :C] = True

        # M5: Movement reweighting -- up-weight moving entities
        movement_weight = None
        if self.movement_reweight_min_ratio > 0:
            with torch.no_grad():
                mv_pos = batch["trace"][..., :3, 3]  # (B, T, P, 3)
                t_mask = batch["trace_mask"]  # (B, T, P)

                # First valid position per track
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
        if self.w_R > 0:
            pred_rot6d = loss_pred[..., 3:]  # (B, C+T, P, 6)
            gt_rot6d = loss_target[..., 3:]
            R_pred = sixd_to_rotmat(pred_rot6d)  # (B, C+T, P, 3, 3)
            R_gt = sixd_to_rotmat(gt_rot6d)
            RtR = torch.einsum("...ji,...jk->...ik", R_pred, R_gt)  # (B, C+T, P, 3, 3)
            r_geodesic = 1.0 - (RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]) / 3.0  # (B, C+T, P)
        else:
            r_geodesic = None

        loss, log_dict = self._compute_loss(
            loss_pred, loss_target, all_mask, rot_mask, loss_type_ids, phase,
            token_weights=None,
            pos_scale=s_linear,
            virtual_mask=virtual_mask,
            movement_weight=movement_weight,
            r_geodesic_loss=r_geodesic,
        )

        ret["loss"] = loss
        ret["log_dict"] = log_dict

        # 10. Per-timestep-bucket X0 accuracy metrics
        with torch.no_grad():
            if self.diffusion.noise_type == "ddpm":
                # DDPM: t_levels are integer timesteps [0, num_train_timesteps-1]
                # Normalize to [0, 1] for bucket metrics
                t_norm = diff_levels / (self.diffusion.num_train_timesteps - 1)
            else:
                t_norm = diff_levels / self.diffusion_time_scale  # FM: already [0, 1]

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

        # Decanonicalize x0 prediction -> original space for viz (detached)
        with torch.no_grad():
            x0_pred_flat = x0_pred_canon_std.detach().reshape(B, N, 9)
            x0_pred_orig = self.canonicalizer.inverse(x0_pred_flat, canon_params)
            x0_pred_orig = x0_pred_orig.reshape(B, C + T, P, 9)
            x0_se3 = ninep_to_se3(x0_pred_orig)  # (B, C+T, P, 4, 4)

            ret["x0_pred_context"] = x0_se3[:, :C].detach()
            ret["x0_pred_trace"] = x0_se3[:, C:].detach()

        return ret

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

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
        seed: int = 42,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        return_intermediates: bool = False,
    ):
        """Run FM Euler inference.

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
            local_pcl: (B, P, PCL, D_pcl) local point clouds, or None.
            local_pcl_mask: (B, P) bool mask for valid PCL entities, or None.
            seed: Random seed.
            n_steps: Override inference steps.
            verbose: Print progress bar.
            return_intermediates: If True, capture X0 predictions at
                25%/50%/75%/100% of denoising and return as dict.

        Returns:
            If return_intermediates=False (default): tuple (ret_context, ret_trace).
            If return_intermediates=True: dict with keys "context", "trace",
            and "intermediates" (list of dicts with step/pct/context/trace).
        """
        is_training = self.training
        if is_training:
            self.eval()

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

        # C1: Convert to wing space after canonicalization (parity with forward)
        if self.rot_param == "6d_wing":
            canon_9d_flat = self._canon_9d_to_wing(canon_9d_flat)

        # 3. Prepare language + features
        lang_tk = self.input_layer.encode_language(text)
        lang_proj = self.lang_token_proj(lang_tk.clone())
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)
        feat = self.input_layer(sem_type, group_id, pcl=local_pcl, pcl_valid_mask=local_pcl_mask)

        # Per-timestep validity masks for LMM (same as forward)
        inf_mask_t = trace_mask  # (B, T, P)
        inf_mask_c = context_mask  # (B, C, P)

        # C3: Prepare rot_9d_mask_flat for re-zeroing rotation in denoising loop
        rot_9d_mask_flat = rot_9d_mask.reshape(B, 1, P, 9).expand(
            -1, C + T, -1, -1
        ).reshape(B, N, 9)

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

        # 5. Denoising loop (FM Euler or DDPM reverse diffusion)
        intermediates = []

        # Helper: decode x0, decanonicalize -> SE3 for intermediate snapshots.
        def _snapshot_x0(x0_canon_flat, step_idx, total_steps, pct):
            """Capture intermediate x0 prediction and store as SE3."""
            _x0 = x0_canon_flat.reshape(B, N, 9)
            # C1: convert wing -> standard 9D before decanonicalization
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

        if self.diffusion.noise_type == "ddpm":
            # DDPM reverse diffusion using diffusers DDPMScheduler
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as DiffusersDDPM
            ddpm_sched = DiffusersDDPM(
                num_train_timesteps=self.diffusion.num_train_timesteps,
                beta_schedule=self.diffusion.beta_schedule,
                prediction_type="sample",  # model outputs x0
                clip_sample=False,  # pose data exceeds [-1, 1]
            )
            ddpm_sched.set_timesteps(n_steps)

            total_ddpm_steps = len(ddpm_sched.timesteps)
            snapshot_steps = {}
            for pct in [25, 50, 75]:
                step_idx = int(total_ddpm_steps * pct / 100)
                if step_idx < total_ddpm_steps:
                    snapshot_steps[step_idx] = pct

            iterator = enumerate(ddpm_sched.timesteps)
            if verbose:
                iterator = tqdm(enumerate(ddpm_sched.timesteps), total=total_ddpm_steps, desc="DDPM inference (V11B)")

            for step_idx, t_val in iterator:
                # Clamp known tokens to clean values
                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    canon_9d_flat,
                    x_t,
                )
                # C3 FIX: zero rotation for position-only entities
                x_t = x_t * rot_9d_mask_flat

                # Build t_levels: integer timestep for unknown tokens, 0 for known
                t_int = t_val.item()
                t_levels_inf = torch.full((B, N), float(t_int), device=device, dtype=x_t.dtype)
                t_levels_inf[known_valid_flat] = 0.0
                diff_levels = (t_levels_inf * self.diffusion_time_scale).reshape(B, C + T, P)
                lmm_input_flat = self._build_v11_input(x_t)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                # noise_level = integer timestep as (B,) tensor
                noise_level_inf = torch.full((B,), float(t_int), device=device, dtype=x_t.dtype)

                model_out = self.lmm(
                    x=lmm_input[:, C:],
                    x_mask=inf_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=inf_mask_c,
                    ctx_level=diff_levels[:, :C],
                    noise_level=noise_level_inf,
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    group_ids=group_id,
                )  # (B, C+T, P, 9) -- x0 prediction

                # DDPM step: model outputs x0 -> scheduler computes x_{t-1}
                model_out_flat = model_out.reshape(B, N, 9)
                # DDPMScheduler.step expects: model_output, timestep, sample
                # With prediction_type="sample", model_output IS x0
                scheduler_out = ddpm_sched.step(model_out_flat, t_val, x_t)
                x_t = scheduler_out.prev_sample

                # Capture intermediate x0 at snapshot steps
                if return_intermediates and step_idx in snapshot_steps:
                    _snapshot_x0(model_out, step_idx, total_ddpm_steps,
                                 snapshot_steps[step_idx])
        else:
            # FM Euler denoising loop
            dt = 1.0 / n_steps
            # Snapshot at 25%, 50%, 75%
            snapshot_steps = {
                int(n_steps * f): int(f * 100)
                for f in [0.25, 0.5, 0.75]
            }
            iterator = range(n_steps)
            if verbose:
                iterator = tqdm(iterator, desc="FM inference (V11B)")

            for step_idx in iterator:
                t = 1.0 - step_idx * dt

                # Clamp known tokens to clean values
                x_t = torch.where(
                    known_valid_flat.unsqueeze(-1).expand_as(x_t),
                    canon_9d_flat,
                    x_t,
                )
                # C3 FIX: zero rotation for position-only entities
                x_t = x_t * rot_9d_mask_flat

                # Build input with t-level
                t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)
                # Known tokens get x_level=0
                t_levels[known_valid_flat] = 0.0
                diff_levels = (t_levels * self.diffusion_time_scale).reshape(B, C + T, P)
                lmm_input_flat = self._build_v11_input(x_t)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                # noise_level = global t for this denoising step, as (B,) tensor
                noise_level_inf = torch.full((B,), t, device=device, dtype=x_t.dtype)

                model_out = self.lmm(
                    x=lmm_input[:, C:],
                    x_mask=inf_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels[:, C:],
                    feat=feat,
                    ctx=lmm_input[:, :C],
                    ctx_mask=inf_mask_c,
                    ctx_level=diff_levels[:, :C],
                    noise_level=noise_level_inf,
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    group_ids=group_id,
                )  # (B, C+T, P, 9) -- x0 prediction

                # FM inference step: direct x0 scheduling.
                # From FM: x_t = (1-t)*x0 + t*eps
                # Estimate eps: eps = (x_t - (1-t)*x0_pred) / t
                # Next step: x_{t-dt} = (1-(t-dt))*x0_pred + (t-dt)*eps
                model_out_flat = model_out.reshape(B, N, 9)
                t_next = t - dt
                x0_pred = model_out_flat
                t_clamped = torch.tensor(t, device=device).clamp(min=1e-6)
                eps_est = (x_t - (1.0 - t) * x0_pred) / t_clamped
                x_t = (1.0 - t_next) * x0_pred + t_next * eps_est

                # Capture intermediate x0 at snapshot steps
                if return_intermediates and step_idx in snapshot_steps:
                    _snapshot_x0(model_out, step_idx, n_steps,
                                 snapshot_steps[step_idx])

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
                "step": n_steps,
                "total": n_steps,
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

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def get_optim_groups(
        self,
        weight_decay: float,
        lmm_lr: float,
        input_layer_lr: float,
        **kwargs,
    ):
        """Get optimizer parameter groups with per-component learning rates."""
        # Agent-level params (lang_token_proj)
        agent_params = list(self.lang_token_proj.parameters())
        agent_group = [
            {"params": agent_params, "weight_decay": weight_decay, "lr": input_layer_lr}
        ]

        return (
            self.lmm.get_optim_groups(weight_decay=weight_decay, base_lr=lmm_lr)
            + self.input_layer.get_optim_groups(weight_decay=weight_decay, base_lr=input_layer_lr)
            + agent_group
        )

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
            to_local: True -> global to local; False -> local to global.

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
