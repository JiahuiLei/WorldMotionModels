"""AgentV10: SIM(3) canonicalization + 6D rotation + flow matching agent.

V10 design differences from V9:
- No PoseIO — replaced by per-sample SIM(3) canonicalization + 6D rotation
- No DINO/images — pose-only
- Flow matching (default) or DDPM noise
- Async noise per token (default) or sync
- New input format: [noised_X, known_indicator, known_X] → project → LMM
- known_mask from dataset (not enforce_known_mask heuristic)
- Loss on ALL valid tokens equally (no known/unknown weighting distinction)

Example::

    from umm.models.umm_v10.agent import AgentV10
    agent = AgentV10(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from umm.models.umm_v10.rotation_utils import se3_to_9d, ninep_to_se3
from umm.models.umm_v10.canonicalization import (
    Canonicalizer,
    CanonParams,
    compute_canon_params,
    canonicalize,
)
from umm.models.umm_v10.diffusion import (
    FlowMatchingScheduler,
    DDPMWrapper,
    MeanFlowScheduler,
    build_diffusion_scheduler,
    compute_x0,
)

# Shared infrastructure — truly generic, not V9-specific
from umm.models.umm_v9.world_models.moma_v6 import MotionMachineV7
from umm.models.umm_v9.world_models.language import T5LanguageEncoder
from umm.models.umm_v9.world_models.wm_utils import (
    HardLayerNorm,
    disable_model_grad,
)
from umm.models.umm_v9.loss_utils import type_mean_compiled as type_mean

logger = logging.getLogger(__name__)


class InputLayerV10(nn.Module):
    """Simplified input layer for V10: type + group embeddings + language.

    No PointNet, no DINO. Pure token identity embeddings.

    Args:
        latent_dim: Embedding dimension.
        feat_dim: Output feature dimension.
        num_types: Number of semantic types.
        num_groups: Number of group IDs.
        lang_fixed_length: Fixed token length for language encoder.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        feat_dim: int = 512,
        num_types: int = 1024,
        num_groups: int = 1024,
        lang_fixed_length: int = 1,
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

        # type + group → feat
        self.bind_fc = nn.Linear(2 * latent_dim, feat_dim)

        # Language encoder (frozen T5)
        self.lang_fixed_len = lang_fixed_length
        self.language_encoder = disable_model_grad(
            T5LanguageEncoder(fixed_length=self.lang_fixed_len)
        )
        self.lang_fc = nn.Linear(768, latent_dim)

    @torch.compiler.disable
    def encode_language(self, text_in):
        """Encode text → (B, lang_fixed_len, latent_dim). No grad."""
        tokens = self.language_encoder.encode(text_in)
        tokens = self.lang_fc(tokens)
        return tokens.detach()

    @torch.compiler.disable
    def forward(self, sem_id: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        """Compute per-entity feature from type and group embeddings.

        Args:
            sem_id: (B, P) semantic type IDs.
            group_id: (B, P) group IDs.

        Returns:
            feat: (B, P, feat_dim).
        """
        type_emb = self.ln_type(self.type_emb(sem_id))    # (B, P, D)
        group_emb = self.ln_group(self.group_emb(group_id))  # (B, P, D)
        feat = self.bind_fc(torch.cat([type_emb, group_emb], dim=-1))
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


class AgentV10(nn.Module):
    """V10 Agent: SIM(3) canon + 6D rotation + flow matching.

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
        self.canonicalizer = Canonicalizer(target_pos_std=self.target_pos_std)

        # Diffusion
        diffusion_cfg = self.model_cfg.get("diffusion", {})
        self.diffusion = build_diffusion_scheduler(diffusion_cfg)
        self.is_flow_matching = isinstance(self.diffusion, FlowMatchingScheduler)
        self.is_meanflow = isinstance(self.diffusion, MeanFlowScheduler)
        # MeanFlow uses the same FM noise process — treat as FM for all code paths
        if self.is_meanflow:
            self.is_flow_matching = True
        # Scale for diffusion time → embedding space
        self.diffusion_time_scale = diffusion_cfg.get("time_scale", 1000.0)
        self.n_inference_steps = diffusion_cfg.get("n_inference_steps", 50)

        # Loss config
        self.weight_by = self.optim_cfg.get("weight_by", "uid")
        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }

        # Context always known (never noised during training)
        self.ctx_always_known = self.optim_cfg.get("ctx_always_known", False)

        # Loss reweighting (min-SNR, epsilon experiments)
        self.loss_weighting = diffusion_cfg.get("loss_weighting", None)
        self.min_snr_gamma = diffusion_cfg.get("min_snr_gamma", 5.0)

        # Network output space conversion (exp 3: x0_to_eps)
        self.network_output_space = diffusion_cfg.get("network_output_space", None)

        # Loss space: "native" (default) = compute loss in prediction space
        # (eps for eps, vel for vel, x0 for x0).
        # "x0" = convert model output to x0 and compute loss there (Exp 4).
        self.loss_space = diffusion_cfg.get("loss_space", "native")

        # Stable loss: Smooth L1 (Huber) with configurable beta.
        # None = MSE (default for backward compat).
        # float (e.g. 0.1) = smooth_l1_loss(beta=loss_beta).
        self.loss_beta = self.optim_cfg.get("loss_beta", None)

        # Warn: loss_space=x0 + min_snr = double 1/SNR weighting
        if self.loss_space == "x0" and self.loss_weighting == "min_snr":
            logger.warning(
                "loss_space='x0' + loss_weighting='min_snr' causes double 1/SNR: "
                "x0 loss has implicit 1/SNR from compute_x0, and min-SNR adds "
                "another SNR-dependent weight. This is probably unintended."
            )

        # --- Input construction ---
        # input_mode: "v10_3cat" (default) — 3-part input [noised_X, indicator, known_X]
        #             "noisy_only" (v10.3) — just projected noisy_x, no known/unknown signal
        self.input_mode = self.model_cfg.get("input_mode", "v10_3cat")

        # All params always created for checkpoint compatibility
        self.x_proj = nn.Linear(9, self.input_proj_dim)
        self.known_x_proj = nn.Linear(9, self.input_proj_dim)
        self.known_emb = nn.Embedding(2, self.input_proj_dim)  # 0=unknown, 1=known
        self.unknown_token = nn.Parameter(torch.randn(9) * 0.02)
        if self.input_mode == "v10_3cat":
            self.input_merge = nn.Linear(3 * self.input_proj_dim, 9)  # → LMM input_dim
        else:
            # noisy_only: x_proj(9→input_proj_dim) → merge(input_proj_dim→9)
            self.input_merge = nn.Linear(self.input_proj_dim, 9)

        # --- Input layer (type + group + language) ---
        input_cfg = self.model_cfg.get("input_layer", {})
        self.input_layer = InputLayerV10(
            latent_dim=self.latent_dim,
            feat_dim=self.model_cfg.get("feat_dim", 512),
            num_types=input_cfg.get("num_types", 1024),
            num_groups=input_cfg.get("num_groups", 1024),
            lang_fixed_length=input_cfg.get("langauge_fixed_length", 1),
        )

        # --- LMM (reuse proven V7 architecture) ---
        lmm_cfg = self.model_cfg.get("lmm", {})
        self.lmm = MotionMachineV7(
            C_T=lmm_cfg.get("C_T", 128),
            C_P=self.C_P,
            feat_dim=self.model_cfg.get("feat_dim", 512),
            n_head_t=lmm_cfg.get("n_head_t", 8),
            n_head_p=lmm_cfg.get("n_head_p", 16),
            input_dim=9,
            output_dim=9,
            n_t_layers=lmm_cfg.get("n_t_layers", 4),
            n_reg_c=lmm_cfg.get("n_reg_c", 2),
            n_reg_x=lmm_cfg.get("n_reg_x", 2),
            n_reg_a=lmm_cfg.get("n_reg_a", 2),
            n_ctx=self._resolve_n_ctx(cfg, lmm_cfg),
            fc_factor_t=lmm_cfg.get("fc_factor_t", 4),
            fc_factor_p=lmm_cfg.get("fc_factor_p", 2),
            fc_n_layers=lmm_cfg.get("fc_n_layers", 2),
            fc_out_n_layers=lmm_cfg.get("fc_out_n_layers", 12),
            num_blk_internal_registers=lmm_cfg.get("num_blk_internal_registers", 4),
            time_multiplier=lmm_cfg.get("time_multiplier", 1.0),
            ada_scale_bias_bound=lmm_cfg.get("ada_scale_bias_bound", 1.0),
            gate_bound=lmm_cfg.get("gate_bound", 2.0),
            ada_n_layers=lmm_cfg.get("ada_n_layers", 1),
            add_time_to_feat=lmm_cfg.get("add_time_to_feat", True),
            time_rope_flag=lmm_cfg.get("time_rope_flag", False),
            isolated_ctx_x=lmm_cfg.get("isolated_ctx_x", False),
            unified_p_layer=lmm_cfg.get("unified_p_layer", False),
            full_attention_t=lmm_cfg.get("full_attention_t", False),
            parallel_p_layer=lmm_cfg.get("parallel_p_layer", False),
            group_p_layer=lmm_cfg.get("group_p_layer", False),
            qk_norm=lmm_cfg.get("qk_norm", True),
            x_pe_dim=lmm_cfg.get("x_pe_dim", 0),
        )

        # Language token projector → C_P for p-layer cross-attention
        self.lang_token_proj = nn.Linear(self.latent_dim, self.C_P)

        # Text masking prob for classifier-free guidance
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)

        logger.info(
            f"AgentV10: flow_matching={self.is_flow_matching}, "
            f"async={getattr(self.diffusion, 'async_noise', True)}, "
            f"target_pos_std={self.target_pos_std}, "
            f"input_proj_dim={self.input_proj_dim}, "
            f"input_mode={self.input_mode}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────

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

    @torch.compiler.disable
    def _build_v10_input(
        self,
        noisy_x: torch.Tensor,
        canon_x: torch.Tensor,
        known_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Build LMM input. Compile-unsafe (boolean indexing).

        input_mode="v10_3cat": three-part input [noised_X, indicator, known_X].
        input_mode="noisy_only": just projected noisy_x (v10.3 clean input).

        Args:
            noisy_x: (B, N, 9) noised canonicalized poses.
            canon_x: (B, N, 9) clean canonicalized poses.
            known_mask_flat: (B, N) bool — which tokens are known.

        Returns:
            lmm_input: (B, N, 9) ready for LMM.
        """
        if self.input_mode == "noisy_only":
            return self.input_merge(self.x_proj(noisy_x))  # (B, N, 9)
        # (a) Project noised X
        a = self.x_proj(noisy_x)  # (B, N, input_proj_dim)

        # (b) Known/unknown indicator
        indicator = self.known_emb(known_mask_flat.long())  # (B, N, input_proj_dim)

        # (c) Known X with unknown filled by learnable token
        unknown_fill = self.unknown_token.unsqueeze(0).unsqueeze(0).expand_as(canon_x)
        known_x_filled = torch.where(
            known_mask_flat.unsqueeze(-1).expand_as(canon_x),
            canon_x,
            unknown_fill,
        )  # (B, N, 9)
        c = self.known_x_proj(known_x_filled)  # (B, N, input_proj_dim)

        # Merge
        merged = torch.cat([a, indicator, c], dim=-1)  # (B, N, 3*input_proj_dim)
        return self.input_merge(merged)  # (B, N, 9)

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
    ) -> tuple:
        """Compute loss on valid tokens with type-mean weighting.

        Uses Smooth L1 (Huber) loss when self.loss_beta is set, otherwise MSE.
        Smooth L1 clips gradients for per-element errors > beta, preventing
        explosion in x0 loss space or at high noise timesteps.

        Args:
            pred: (B, C+T, P, 9) model output in native prediction space.
            target: (B, C+T, P, 9) target in native prediction space.
            valid_mask: (B, C+T, P) bool.
            rot_mask: (B, P) bool — entities with rotation.
            loss_type_ids: (B, P) int — IDs for type_mean grouping.
            phase: "train" or "val".
            token_weights: Optional (B, C+T, P) per-token weights (e.g. min-SNR).
            pos_scale: Optional (B, 1, 1) GT-canon s^2 multiplier for position loss.

        Returns:
            Tuple of (loss, log_dict).
        """
        log_dict = {}

        if self.loss_beta is not None:
            # Smooth L1 (Huber) loss: linear for |e| > beta, quadratic for |e| < beta
            pos_loss = F.smooth_l1_loss(
                pred[..., :3], target[..., :3],
                beta=self.loss_beta, reduction="none",
            ).sum(dim=-1)  # (B, C+T, P)
            rot_loss = F.smooth_l1_loss(
                pred[..., 3:], target[..., 3:],
                beta=self.loss_beta, reduction="none",
            ).sum(dim=-1)  # (B, C+T, P)
        else:
            # MSE loss
            pos_loss = (pred[..., :3] - target[..., :3]).pow(2).sum(dim=-1)  # (B, C+T, P)
            rot_loss = (pred[..., 3:] - target[..., 3:]).pow(2).sum(dim=-1)  # (B, C+T, P)

        # Apply GT-canon scale factor s^2 to position loss only.
        # GT-canon is a SIM(3) transform: T(x) = s * R * (x - mean).
        # For MSE: MSE(T(pred), T(target)) = s^2 * MSE(pred, target) (exact).
        # For Smooth L1: s^2 scaling is a heuristic that normalizes for
        # spatial scale across samples. Not an exact SIM(3) equivalence
        # (HuberL1(s*e) != s^2 * HuberL1(e) in the linear regime).
        # Rotation loss is NOT affected by GT-canon.
        if pos_scale is not None:
            pos_loss = pos_loss * pos_scale  # (B, C+T, P) * (B, 1, 1)

        # Apply per-token weights (e.g. min-SNR reweighting)
        if token_weights is not None:
            pos_loss = pos_loss * token_weights
            rot_loss = rot_loss * token_weights

        # Mask rotation loss for position-only entities
        rot_valid = valid_mask & rot_mask[:, None, :].expand_as(valid_mask)

        # Compute type-mean losses
        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Position loss on all valid tokens
        if valid_mask.any():
            _ids = loss_type_ids[:, None, :].expand_as(valid_mask).long()
            pos_vals = pos_loss[valid_mask]
            pos_ids = _ids[valid_mask]
            pos_term = type_mean(pos_vals, pos_ids, self.gid_w_modifier)
            loss = loss + pos_term
            log_dict[f"{phase}_LOSS/pos"] = pos_term.detach()

        # Rotation loss on valid+has-rotation tokens
        if rot_valid.any():
            _ids = loss_type_ids[:, None, :].expand_as(valid_mask).long()
            rot_vals = rot_loss[rot_valid]
            rot_ids = _ids[rot_valid]
            rot_term = type_mean(rot_vals, rot_ids, self.gid_w_modifier)
            loss = loss + rot_term
            log_dict[f"{phase}_LOSS/rot"] = rot_term.detach()

        log_dict[f"{phase}_LOSS/total"] = loss.detach()
        log_dict[f"total_loss/{phase}"] = loss.detach()

        return loss, log_dict

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

        # 4. Add noise
        if self.is_meanflow:
            noisy_flat, t_levels, r_levels, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat,
            )
            # Use t_levels for LMM timestep embedding (same as FM)
            diffusion_levels = t_levels * self.diffusion_time_scale  # (B, N)
        elif self.is_flow_matching:
            noisy_flat, t_levels, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat,
            )
            # Scale t_levels for LMM timestep embedding
            diffusion_levels = t_levels * self.diffusion_time_scale  # (B, N)
        else:
            noisy_flat, timesteps_int, noise_flat, target_flat = self.diffusion.add_noise(
                canon_9d_flat, valid_mask=all_mask_flat,
            )
            diffusion_levels = timesteps_int.float()  # (B, N)

        # Invalid tokens → fully noised (already handled by diffusion: t=1 or max_ts)
        # Additionally overwrite their encoded values with pure noise
        invalid_flat = ~all_mask_flat  # (B, N)
        if noise_flat is not None:
            noisy_flat = torch.where(
                invalid_flat.unsqueeze(-1).expand_as(noisy_flat),
                noise_flat,
                noisy_flat,
            )

        # 5. Build V10 input
        lmm_input_flat = self._build_v10_input(
            noisy_flat, canon_9d_flat, all_known_flat & all_mask_flat,
        )  # (B, N, 9)

        # Reshape back to (B, C+T, P, 9)
        lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)
        diff_levels = diffusion_levels.reshape(B, C + T, P)

        # Trajectory-level padding mask for attention
        traj_padding_mask = all_mask.any(dim=1)  # (B, P)
        attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)  # (B, T, P)
        attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)  # (B, C, P)

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
        feat = self.input_layer(sem_type, group_id)  # (B, P, feat_dim)

        # 7. LMM forward
        out = self.lmm(
            x=lmm_input[:, C:],
            x_mask=attn_mask_t,
            x_time=timestep,
            x_level=diff_levels[:, C:],
            feat=feat,
            ctx=lmm_input[:, :C],
            ctx_mask=attn_mask_c,
            ctx_level=diff_levels[:, :C],
            add_tokens=lang_proj,
            add_tokens_mask=lang_mask,
            group_ids=group_id,
        )  # (B, C+T, P, 9)

        self._check_finite(out)

        # 7b. Network output space conversion (Exp 3: x0_to_eps, Exp: x0_to_vel)
        # If network_output_space == "x0_to_eps": LMM outputs x0, convert to eps.
        # If network_output_space == "x0_to_vel": LMM outputs x0, convert to v_prediction.
        # CRITICAL: Save LMM x0 output BEFORE conversion for GT-canon loss.
        # compute_x0 would round-trip back to x0 (identity), making the
        # experiment a no-op. Use the saved x0 directly for loss.
        lmm_x0_output = None  # set for x0_to_eps or x0_to_vel mode
        if self.network_output_space == "x0_to_eps" and not self.is_flow_matching:
            lmm_x0_output = out  # save LMM's raw x0 prediction
            _EPS = 1e-6
            noisy_for_conv = noisy_flat.reshape(B, C + T, P, 9)
            with torch.autocast(device_type=device.type, enabled=False):
                out_f = out.float()
                noisy_f = noisy_for_conv.float()
                ts_flat = diff_levels.long().reshape(-1)  # (B*(C+T)*P,)
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_t = alphas_cp[ts_flat]  # (B*(C+T)*P,)
                sqrt_alpha = torch.sqrt(alpha_t).reshape(B, C + T, P, 1)
                sqrt_1m_alpha = torch.sqrt(1.0 - alpha_t).reshape(B, C + T, P, 1)
                # Guard: sqrt(1-alpha_t) -> 0 at low timesteps
                sqrt_1m_alpha = sqrt_1m_alpha.clamp(min=_EPS)
                out_eps = (noisy_f - sqrt_alpha * out_f) / sqrt_1m_alpha
            out = out_eps.to(out.dtype)
        elif self.network_output_space == "x0_to_vel" and not self.is_flow_matching:
            lmm_x0_output = out  # save LMM's raw x0 prediction
            _EPS = 1e-6
            noisy_for_conv = noisy_flat.reshape(B, C + T, P, 9)
            with torch.autocast(device_type=device.type, enabled=False):
                out_f = out.float()
                noisy_f = noisy_for_conv.float()
                ts_flat = diff_levels.long().reshape(-1)  # (B*(C+T)*P,)
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_t = alphas_cp[ts_flat]  # (B*(C+T)*P,)
                sqrt_alpha = torch.sqrt(alpha_t).reshape(B, C + T, P, 1)
                sqrt_1m_alpha = torch.sqrt(1.0 - alpha_t).reshape(B, C + T, P, 1)
                # Guard: sqrt(1-alpha_t) -> 0 at low timesteps (alpha_t -> 1)
                sqrt_1m_alpha = sqrt_1m_alpha.clamp(min=_EPS)
                # v = (sqrt(alpha_t) * x_t - x0) / sqrt(1 - alpha_t)
                out_vel = (sqrt_alpha * noisy_f - out_f) / sqrt_1m_alpha
            out = out_vel.to(out.dtype)
        elif self.network_output_space == "x0_to_vel" and self.is_flow_matching:
            lmm_x0_output = out  # save LMM's raw x0 prediction
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

        # 9. GT-canon: extract scale factor s^2 for loss + compute GT-canon
        #    x0 for per-timestep metrics.
        #
        # GT-canon is a SIM(3) transform: T(x) = s * R * (x - mean).
        # Applied to both sides of MSE:
        #   MSE(T(pred_pos), T(target_pos)) = s^2 * MSE(pred_pos, target_pos)
        #     (translation cancels, R orthogonal → norm-preserving)
        #   MSE(T(pred_rot), T(target_rot)) = MSE(pred_rot, target_rot)
        #     (R_canon orthogonal → norm-preserving on 6D columns)
        #
        # Therefore GT-canon contributes ONLY s^2 on position loss.
        # We compute loss in the NATIVE prediction space (eps/vel/x0),
        # NOT in x0 space. No compute_x0 in the loss path for eps/vel.
        # NOTE: uses all_mask_flat (all valid tokens), NOT known_valid_flat.
        # GT-canon uses ALL valid GT tokens for a more stable reference frame.
        # canon_9d_flat is clean data (before noise), so unknown tokens still
        # have correct positions — they just weren't shown to the model.
        gt_canon_params = compute_canon_params(
            canon_9d_flat, all_mask_flat, self.target_pos_std
        )
        # s^2 per sample, shape (B, 1) → (B, 1, 1) for broadcast over (B, C+T, P)
        s_squared = gt_canon_params.scale.pow(2).unsqueeze(-1)  # (B, 1, 1)

        # GT-canon x0 for per-timestep accuracy metrics (detached, not in loss)
        with torch.no_grad():
            x0_pred_gt = canonicalize(
                x0_pred_canon.detach().reshape(B, N, 9), gt_canon_params
            ).reshape(B, C + T, P, 9)
            x0_gt_gt = canonicalize(
                x0_gt_canon.reshape(B, N, 9), gt_canon_params
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
        else:
            # Native prediction space loss (default):
            #   eps mode  → MSE(eps_pred, noise)
            #   vel mode  → MSE(vel_pred, vel_target)
            #   x0 mode   → MSE(x0_pred, x0_gt)
            # target_flat was computed by add_noise in step 4 and already contains
            # the correct native target for each prediction_type.
            loss_pred = out  # (B, C+T, P, 9) model output in native space
            loss_target = target_flat.reshape(B, C + T, P, 9)

        loss, log_dict = self._compute_loss(
            loss_pred, loss_target, all_mask, rot_mask, loss_type_ids, phase,
            token_weights=token_weights,
            pos_scale=s_squared,
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
            x0_pred_flat = x0_pred_canon.detach().reshape(B, N, 9)
            x0_pred_orig = self.canonicalizer.inverse(x0_pred_flat, canon_params)
            x0_pred_orig = x0_pred_orig.reshape(B, C + T, P, 9)
            x0_se3 = ninep_to_se3(x0_pred_orig)  # (B, C+T, P, 4, 4)

            ret["x0_pred_context"] = x0_se3[:, :C].detach()
            ret["x0_pred_trace"] = x0_se3[:, C:].detach()

        return ret

    # ─────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────

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
        seed: int = 42,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        return_intermediates: bool = False,
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

        # 3. Prepare language + features
        lang_tk = self.input_layer.encode_language(text)
        lang_proj = self.lang_token_proj(lang_tk.clone())
        lang_mask = torch.ones(B, lang_proj.shape[1], device=device, dtype=torch.bool)
        feat = self.input_layer(sem_type, group_id)

        # Attention mask
        traj_padding_mask = all_mask.any(dim=1)  # (B, P)

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

        # 5. Denoising loop
        intermediates = []

        # Helper: decode x0 from current state + model output, decanonicalize → SE3
        _is_x0_output = (self.network_output_space in ("x0_to_vel", "x0_to_eps"))

        def _snapshot_x0(x_t_cur, model_out, step_idx, total_steps, pct, t_val=None):
            """Capture intermediate x0 prediction and store as SE3."""
            if self.is_flow_matching:
                if _is_x0_output:
                    # Network outputs x0 directly
                    x0_canon = model_out.reshape(B, N, 9)
                else:
                    # x0 = x_t - t * v_pred
                    x0_canon = x_t_cur - t_val * model_out.reshape(B, N, 9)
            else:
                # x0 from DDPM compute_x0
                t_tensor_snap = torch.full(
                    (B, C + T, P), int(t_val), device=device, dtype=torch.long
                )
                x0_canon = compute_x0(
                    self.diffusion.noise_scheduler,
                    model_out, t_tensor_snap, x_t_cur.reshape(B, C + T, P, 9),
                ).reshape(B, N, 9)
            x0_orig = self.canonicalizer.inverse(x0_canon, canon_params)
            x0_orig = x0_orig.reshape(B, C + T, P, 9)
            x0_se3 = ninep_to_se3(x0_orig)
            intermediates.append({
                "step": step_idx,
                "total": total_steps,
                "pct": pct,
                "context": x0_se3[:, :C],
                "trace": x0_se3[:, C:],
            })

        if self.is_flow_matching:
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

                # Build input
                t_levels = torch.full((B, N), t, device=device, dtype=x_t.dtype)
                diff_levels = (t_levels * self.diffusion_time_scale).reshape(B, C + T, P)
                lmm_input_flat = self._build_v10_input(x_t, canon_9d_flat, known_valid_flat)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

                model_out = self.lmm(
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
                    group_ids=group_id,
                )  # (B, C+T, P, 9)

                # FM inference step
                model_out_flat = model_out.reshape(B, N, 9)

                if _is_x0_output:
                    # Model outputs x0 directly.
                    # Use direct x0 scheduling instead of velocity conversion
                    # to avoid numerical instability from dividing by small t.
                    #
                    # From FM: x_t = (1-t)*x0 + t*eps
                    # Estimate eps: eps = (x_t - (1-t)*x0_pred) / t
                    # Next step: x_{t-dt} = (1-(t-dt))*x0_pred + (t-dt)*eps
                    t_next = t - dt
                    x0_pred = model_out_flat
                    eps_est = (x_t - (1.0 - t) * x0_pred) / max(t, 1e-6)
                    x_t = (1.0 - t_next) * x0_pred + t_next * eps_est
                else:
                    # Standard FM: model outputs velocity, Euler step
                    v_pred_flat = model_out_flat
                    x_t = x_t - dt * v_pred_flat

                # Capture intermediate x0 at snapshot steps
                if return_intermediates and step_idx in snapshot_steps:
                    _snapshot_x0(x_t, model_out, step_idx, n_steps,
                                 snapshot_steps[step_idx], t_val=t)

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

                t_tensor = torch.full((B, N), int(t_int), device=device, dtype=torch.long)
                diff_levels = t_tensor.float().reshape(B, C + T, P)
                lmm_input_flat = self._build_v10_input(x_t, canon_9d_flat, known_valid_flat)
                lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

                attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)

                pred = self.lmm(
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
                    group_ids=group_id,
                )

                # Network output space conversion (Exp 3: x0_to_eps, Exp: x0_to_vel)
                if self.network_output_space == "x0_to_eps":
                    _EPS = 1e-6
                    alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                    alpha_val = alphas_cp[int(t_int)]
                    sqrt_alpha = torch.sqrt(alpha_val)
                    sqrt_1m_alpha = torch.sqrt(1.0 - alpha_val).clamp(min=_EPS)
                    # x0_pred (pred) → eps: eps = (x_t - sqrt(a)*x0) / sqrt(1-a)
                    pred = (x_t.reshape(B, C + T, P, 9).float() - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                    pred = pred.to(x_t.dtype)
                elif self.network_output_space == "x0_to_vel":
                    _EPS = 1e-6
                    alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                    alpha_val = alphas_cp[int(t_int)]
                    sqrt_alpha = torch.sqrt(alpha_val)
                    sqrt_1m_alpha = torch.sqrt(1.0 - alpha_val).clamp(min=_EPS)
                    # x0_pred (pred) → vel: vel = (sqrt(a)*x_t - x0) / sqrt(1-a)
                    pred = (sqrt_alpha * x_t.reshape(B, C + T, P, 9).float() - pred.float()) / sqrt_1m_alpha
                    pred = pred.to(x_t.dtype)

                # Capture intermediate x0 at snapshot steps
                if return_intermediates and ddim_idx in snapshot_steps:
                    _snapshot_x0(x_t, pred, ddim_idx, total_ddim,
                                 snapshot_steps[ddim_idx], t_val=int(t_int))

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

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def get_optim_groups(
        self,
        weight_decay: float,
        lmm_lr: float,
        input_layer_lr: float,
        **kwargs,
    ):
        """Get optimizer parameter groups with per-component learning rates."""
        # Agent-level params (input construction layers + projector)
        agent_params = (
            list(self.x_proj.parameters())
            + list(self.known_x_proj.parameters())
            + list(self.known_emb.parameters())
            + [self.unknown_token]
            + list(self.input_merge.parameters())
            + list(self.lang_token_proj.parameters())
        )
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
