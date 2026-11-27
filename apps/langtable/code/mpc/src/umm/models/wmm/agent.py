"""AgentV11: SIM(3) canonicalization + 6D rotation + unified noise schedulers.

V11 changes from V10:
- x0net + converter: model ALWAYS outputs x0, converter produces target-space output
- 50/50 per-sample noise branching (clean-known vs async)
- Known tokens get x_level=0 during inference (THE critical v10 bug fix)
- Linear loss scaling (s_linear not s_squared)
- No 3-cat input — simple projection of noisy_x only
- Supports DDPM/FM/MeanFlow in one model

Example::

    from umm.models.wmm.agent import AgentV11
    agent = AgentV11(cfg)
    ret = agent(batch, phase="train", global_step=0)
"""

import logging
import os.path as osp
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from umm.models.wmm.rotation_utils import se3_to_9d, ninep_to_se3, sixd_to_rotmat
# NOTE: SDF / SMPL-X / pytorch3d helpers are lazy-imported INSIDE the
# hand-object loss methods. Importing pytorch3d.transforms or smplx at
# module load time changes CUDA kernel/RNG init order and breaks the
# byte-equal back-compat guarantee for w1.6.0.yaml (verified empirically:
# rot_real diff of 2 ULP appeared when moved to top-level). Lazy-import
# is the FIRST-CALL cost only; subsequent calls hit sys.modules cache.
from umm.models.wmm.canonicalization import (
    Canonicalizer,
    CanonParams,
    compute_canon_params,
    canonicalize,
)
from umm.models.wmm.diffusion import (
    FlowMatchingScheduler,
    DDPMWrapper,
    MeanFlowScheduler,
    build_diffusion_scheduler,
    compute_x0,
)
# w1.6.0: SDF + contact loss helpers (Phase B1).
# Imports gated to avoid pulling smplx/trimesh into non-OMOMO configs.
# We import lazily inside the helper methods so that w1.5.x configs without
# the loss enabled never touch these dependencies.
from umm.datasets.umm_v6_mixed.dataset_config import SEM_OBJECT

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



class InputLayerV11Super(nn.Module):
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
        t5_vocab_only: bool = False,
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

        # Language encoder (frozen T5). ``t5_vocab_only=True`` bypasses the
        # T5 transformer stack and routes ``encode_language`` through the
        # vocab embedding lookup only (used by Job 4 of v13.8 LT rescue).
        # NOTE: voc-only models cannot strict-load a full-T5 checkpoint —
        # the checkpoint carries ~97 ``encoder.block.*`` and
        # ``encoder.final_layer_norm.*`` keys that have been deleted here.
        # Fresh-train only; any transfer/resume from a full-T5 checkpoint
        # must pre-filter those keys.
        self.lang_fixed_len = lang_fixed_length
        self.t5_vocab_only = t5_vocab_only
        self.language_encoder = disable_model_grad(
            T5LanguageEncoder(
                fixed_length=self.lang_fixed_len,
                skip_encoder_transformer=t5_vocab_only,
            )
        )
        self.lang_fc = nn.Linear(768, latent_dim)

        if self.skip_group_emb:
            logger.warning("InputLayerV11Super: skip_group_emb=True")
        logger.info(
            f"InputLayerV11Super: pcl={pcl_input_mode}, pntnet={point_version}, "
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
          `InputLayerV11Super.get_optim_groups` (the "language_encoder"
          prefix filter excludes only `self.language_encoder.*`, not
          `self.lang_fc.*`).
        """
        if self.t5_vocab_only:
            tokens = self.language_encoder.encode_vocab(text_in)
        else:
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


class AgentV11(nn.Module):
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
            n_ctx = max(explicit, max_C)
        else:
            n_ctx = max(max_C, 128) if max_C > 0 else 128
        # Invariant: n_ctx must be at least max_C across all
        # dataset configs (explicit assert for visibility).
        assert n_ctx >= max_C, (
            f"n_ctx={n_ctx} < dataset max_C={max_C}; bump model.lmm.n_ctx in YAML."
        )
        return n_ctx

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
        # Bounded variant (back-compat: missing keys → original behavior)
        self.canon_bounded = canon_cfg.get("bounded", False)
        self.canon_bounded_dep_thr = canon_cfg.get("bounded_dep_thr", 1.5)
        self.canon_bounded_topk_frac = canon_cfg.get("bounded_topk_frac", 0.1)
        self.canon_bounded_atanh_eps = canon_cfg.get("bounded_atanh_eps", 1.0e-4)
        self.canon_per_axis_std = canon_cfg.get("per_axis_std", False)
        # Canon source selection: which tokens contribute to canon_params.
        #   "known_valid" (default) — current behavior: known & valid tokens (ctx + T).
        #   "ctx_valid"             — all valid ctx tokens only (ignore known mask,
        #                              exclude every T-slice token). Used by w1.5.1
        #                              so canon is constant across the T sequence.
        # The selected mask is applied LOCALLY at the canonicalizer call site only;
        # all_known / all_known_flat / known_valid_flat are NOT mutated.
        self.canon_source = canon_cfg.get("canon_source", "known_valid")
        if self.canon_source not in ("known_valid", "ctx_valid"):
            raise ValueError(
                f"Invalid model.canonicalization.canon_source={self.canon_source!r}; "
                f"expected 'known_valid' or 'ctx_valid'."
            )
        self.canonicalizer = Canonicalizer(
            target_pos_std=self.target_pos_std,
            bounded=self.canon_bounded,
            bounded_dep_thr=self.canon_bounded_dep_thr,
            bounded_topk_frac=self.canon_bounded_topk_frac,
            bounded_atanh_eps=self.canon_bounded_atanh_eps,
            canon_rotation=self.canon_rotation,
            per_axis_std=self.canon_per_axis_std,
        )

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

        # V13.3.4: two-stage training mask + inference
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
        # w_q: multiplier on rotation (rot6d-channel) loss terms. Multiplies
        # both rot_real and rot_virt terms (after w_real / w_virtual). Default
        # 1.0 = no change. Set <1 to weaken rotation supervision (e.g. when
        # rotation is a constant identity placeholder).
        self.w_q = self.optim_cfg.get("w_q", 1.0)

        # H1: Virtual/real loss weighting (context frames down-weighted)
        self.w_real = self.optim_cfg.get("w_real", 1.0)
        self.w_virtual = self.optim_cfg.get("w_virtual", 1.0)

        # w_mae: per-channel L1 (sum-across-xyz) auxiliary loss on POSITION only.
        # Default 0.0 = disabled (current behavior unchanged). When > 0, adds
        # type_mean(|pred[:3]-target[:3]|.sum(-1)) * w_real * w_mae for real
        # and * w_virtual * w_mae for virtual. Shares pos_scale, w_ctx,
        # token_weights, movement_weight modifiers with the primary pos loss.
        # Use to optimize the eval MAE metric directly.
        self.w_mae = self.optim_cfg.get("w_mae", 0.0)

        # w_ctx: scale loss on the first C (context) positions. Default 1.0 = no-op.
        # OMOMO w1.1 sets 0.1: ctx is identity T-pose and only enters loss when
        # base_random randomly hides it (~12% of samples via mask_scheduler's
        # base_random path with ctx_known ~ Bern(0.5)); down-weighting keeps
        # these rare ctx-unknown gradients from dominating. NOTE: stacks
        # multiplicatively with w_virtual for virtual-token ctx rows — intended
        # when w_virtual=1.0 (current OMOMO); combining w_ctx!=1.0 AND
        # w_virtual!=1.0 yields effective scale w_ctx*w_virtual.
        self.w_ctx = self.optim_cfg.get("w_ctx", 1.0)

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
        self.input_layer = InputLayerV11Super(
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
            t5_vocab_only=input_cfg.get("t5_vocab_only", False),
        )

        # --- LMM (V13: Super or MoE) ---
        lmm_cfg = self.model_cfg.get("lmm", {})
        layer_type_string = lmm_cfg.get("layer_type_string", None)
        lmm_mode = lmm_cfg.get("mode", None)
        assert lmm_mode == "super", (
            f"wmm only supports lmm.mode='super', got {lmm_mode!r}"
        )

        # w1.0.2: pre-LMM config validation — raise ValueError (NOT
        # AssertionError) BEFORE constructing the LMM so the failure mode is
        # "config error" not "internal invariant". Mirrors the mutual-
        # exclusion assert inside moma_v11_final_super.__init__ but fires
        # earlier with a cleaner message.
        _xattn_v3_enable_early = bool(
            (self.model_cfg.get("input_xattn_v3", {}) or {}).get("enable", False)
        )
        if _xattn_v3_enable_early:
            _add_first_n_early = int(lmm_cfg.get("add_first_n_add_token_to_feat", -1))
            if _add_first_n_early > 0:
                raise ValueError(
                    f"input_xattn_v3.enable=True requires "
                    f"model.lmm.add_first_n_add_token_to_feat <= 0 "
                    f"(got {_add_first_n_early}). Both occupy the same "
                    "cond_parts slot; pick one path."
                )
        lmm_architecture = lmm_cfg.get("architecture", "super")
        if lmm_architecture == "moe":
            from umm.models.wmm.world_models.moma_v11_moe import (
                MotionMachineV11Moe,
            )
            LmmClass = MotionMachineV11Moe
        else:
            from umm.models.wmm.world_models.moma_v11_final_super import (
                MotionMachineV11Super,
            )
            LmmClass = MotionMachineV11Super
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
            # V13.3: unidirectional ctx attention mask.
            ctx_unidirectional_mask=lmm_cfg.get("ctx_unidirectional_mask", False),
            # V13.7: extend add_token projector+injection to t/p layers.
            extra_add_token_layers=lmm_cfg.get("extra_add_token_layers", ""),
            # V13.8 Feature A: project first N lang add_tokens into feat.
            add_first_n_add_token_to_feat=lmm_cfg.get(
                "add_first_n_add_token_to_feat", -1
            ),
            # w1.0.2: pooled-lang (B, D1) occupies the cond_parts slot.
            # Enabled automatically when model.input_xattn_v3.enable=True.
            pooled_lang_cond=bool(
                (self.model_cfg.get("input_xattn_v3", {}) or {}).get("enable", False)
            ),
            # w1.3.x: 3-way MoT block (per-stream FFN+ada for reg/ctx/x).
            # Default False = bit-identical to w1.0/w1.1/w1.2 (uses DiTBlockDualFFN).
            # Only routed by Super architecture; Moe ignores via **kwargs filter.
            mot_3way=lmm_cfg.get("mot_3way", False),
            # w1.4.0: bias model toward current (anchor) frame. Default
            # False = bit-identical to w1.3.0 and earlier.
            bias_to_cur_frame=lmm_cfg.get("bias_to_cur_frame", False),
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
            from umm.models.wmm.dino_encoder import (
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
            from umm.models.wmm.input_xattn import InputLayerCrossAttn
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
            from umm.models.wmm.input_xattn_v2 import InputLayerCrossAttnV2
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

        # --- w1.0.2: optional input_xattn_v3 ---
        # V3 forks the v2 pipeline and exposes a projected pooled-lang vector
        # (B, D1) for insertion into the LMM's cond_parts (replaces the
        # hard_ln_lat / add_first_n_add_token_to_feat slot). Mutually
        # exclusive with v1 (input_xattn), v2 (input_xattn_v2), AND
        # add_first_n_add_token_to_feat > 0.
        input_xattn_v3_cfg = self.model_cfg.get("input_xattn_v3", {}) or {}
        self.input_xattn_v3_enable = bool(input_xattn_v3_cfg.get("enable", False))
        if self.input_xattn_v3_enable:
            if self.input_xattn_enable:
                raise ValueError(
                    "input_xattn (V6) and input_xattn_v3 (w1.0.2) are mutually exclusive"
                )
            if self.input_xattn_v2_enable:
                raise ValueError(
                    "input_xattn_v2 and input_xattn_v3 are mutually exclusive"
                )
            _add_first_n = int(lmm_cfg.get("add_first_n_add_token_to_feat", -1))
            if _add_first_n > 0:
                raise ValueError(
                    f"input_xattn_v3.enable=True requires "
                    f"model.lmm.add_first_n_add_token_to_feat <= 0 (got {_add_first_n}). "
                    "Both occupy the same cond_parts slot; pick one path."
                )
            from umm.models.wmm.input_xattn_v3 import InputLayerCrossAttnV3
            self.input_xattn_v3 = InputLayerCrossAttnV3(
                feat_dim=self.model_cfg.get("feat_dim", 512),
                D1=lmm_cfg.get("D1", 512),
                n_layers=int(input_xattn_v3_cfg.get("n_layers", 2)),
                n_register=int(input_xattn_v3_cfg.get("n_register", 2)),
                n_heads=int(input_xattn_v3_cfg.get("n_heads", 8)),
                learned_traj_id=bool(input_xattn_v3_cfg.get("learned_traj_id", False)),
                traj_id_bound_K=float(input_xattn_v3_cfg.get("traj_id_bound_K", 1.0)),
                traj_id_K_warmup_start=int(input_xattn_v3_cfg.get("traj_id_K_warmup_start", 0)),
                traj_id_K_warmup_duration=int(input_xattn_v3_cfg.get("traj_id_K_warmup_duration", 0)),
            )
            logger.info(
                f"w1.0.2 input_xattn_v3 ENABLED: n_layers={input_xattn_v3_cfg.get('n_layers', 2)} "
                f"n_register={input_xattn_v3_cfg.get('n_register', 2)} "
                f"n_heads={input_xattn_v3_cfg.get('n_heads', 8)} "
                f"learned_traj_id={input_xattn_v3_cfg.get('learned_traj_id', False)} "
                f"K={input_xattn_v3_cfg.get('traj_id_bound_K', 1.0)}"
            )
        else:
            self.input_xattn_v3 = None

        # Text masking prob for classifier-free guidance
        self.text_masked_prob = self.optim_cfg.get("text_masked_prob", 0.0)

        # w1.6.0 — SDF penetration + contact loss (Phase B1).
        # All gates default to 0 / OFF so w1.5.x configs are bit-identical.
        # See doc: w1.6.0_omomo_handobj plan task B1.
        self.loss_sdf_penetration_weight = float(
            self.model_cfg.get("loss_sdf_penetration_weight", 0.0)
        )
        self.loss_contact_weight = float(
            self.model_cfg.get("loss_contact_weight", 0.0)
        )
        self.contact_threshold_m = float(
            self.model_cfg.get("contact_threshold_m", 0.02)
        )
        self.contact_label_eps_m = float(
            self.model_cfg.get("contact_label_eps_m", 0.05)
        )
        self.penetration_margin_m = float(
            self.model_cfg.get("penetration_margin_m", 0.005)
        )
        # w1.6.2 — full-body SMPL-X mesh penetration loss (Phase C).
        # Default 0 → OFF so w1.6.0/w1.6.1 yamls are byte-identical pre/post.
        # When > 0, queries SDF at all 10,475 SMPL-X mesh verts in GT object's
        # rest frame and penalizes verts more than ``full_body_pen_allowance_m``
        # inside the surface (matches eval P_hand metric semantics).
        self.loss_full_body_penetration_weight = float(
            self.model_cfg.get("loss_full_body_penetration_weight", 0.0)
        )
        self.full_body_pen_allowance_m = float(
            self.model_cfg.get("full_body_pen_allowance_m", 0.005)
        )
        # w1.7.x.2 — linearly weight aux losses by per-token diffusion timestep.
        # Default False → behavior identical to pre-w1.7.x.2 configs.
        # When True, multiplies aux loss numerators by w(t) = 1 - t / (T_diff - 1)
        # per-token (palm losses use wrist slot t; full-body uses per-frame mean
        # of body slot t). Rationale: at high noise levels the predicted x0
        # is unreliable so penetration/contact gradients are noisy; downweight.
        self.aux_loss_t_weighted = bool(
            self.model_cfg.get("aux_loss_t_weighted", False)
        )
        self.hand_palm_joint_idx = tuple(
            self.model_cfg.get("hand_palm_joint_idx", (25, 40))
        )
        self.hand_wrist_joint_idx = tuple(
            self.model_cfg.get("hand_wrist_joint_idx", (20, 21))
        )
        self.sdf_v2_dir = str(
            self.model_cfg.get(
                "sdf_v2_dir", "data_wmm/omomo_v2/rest_object_sdf_v2"
            )
        )
        # NOTE: smplx_joints_forward expects the PARENT dir (containing 'smplx/'),
        # not the 'smplx/' dir itself. Default to OMOMO v2 parent.
        self.smplx_dir_handobj = str(
            self.model_cfg.get("smplx_dir", "data_wmm/omomo_v2")
        )
        self.rest_geo_dir = str(
            self.model_cfg.get("rest_geo_dir", "data_wmm/omomo_v2/rest_object_geo")
        )
        # Model-side caches (lazy populated on first hot-loop hit). Buffers
        # NOT registered (we don't want them in state_dict; they're rebuilt on
        # process restart from disk).
        self._sdf_cache: Dict[str, tuple] = {}
        self._rest_pcl_cache: Dict[str, torch.Tensor] = {}
        self._palm_offset_cache: Dict[tuple, torch.Tensor] = {}

        logger.info(
            f"AgentV11: flow_matching={self.is_flow_matching}, "
            f"target_pos_std={self.target_pos_std}, "
            f"input_proj=simple(v11)"
        )
        if (
            self.loss_sdf_penetration_weight > 0.0
            or self.loss_contact_weight > 0.0
            or self.loss_full_body_penetration_weight > 0.0
        ):
            logger.info(
                f"AgentV11 hand-object loss ENABLED: "
                f"sdf_w={self.loss_sdf_penetration_weight} "
                f"contact_w={self.loss_contact_weight} "
                f"full_body_pen_w={self.loss_full_body_penetration_weight} "
                f"contact_thr={self.contact_threshold_m}m "
                f"label_eps={self.contact_label_eps_m}m "
                f"pen_margin={self.penetration_margin_m}m "
                f"full_body_pen_allowance={self.full_body_pen_allowance_m}m "
                f"aux_loss_t_weighted={self.aux_loss_t_weighted} "
                f"sdf_v2_dir={self.sdf_v2_dir}"
            )

    # ─────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _compute_effective_K(self, step: int, training: bool) -> float:
        """Compute K-warmup effective bound as a Python float.

        Reads warmup config from whichever xattn module is active
        (input_xattn_v2 or input_xattn_v3 — v3 shares the same K/warmup
        schema as v2). Mutually exclusive: at most one is non-None.

        Training schedule (when warmup_duration > 0):
          [0, warmup_start)                     -> 0.0
          [warmup_start, warmup_start+duration) -> K * (step-start)/duration
          [warmup_start+duration, inf)          -> K (full)

        Inference OR warmup_duration == 0 -> K (full, no warmup).

        Compile-safety: decorated with @torch.compiler.disable and returns a
        plain Python float so the compiled region never sees a changing int.
        """
        mod = self.input_xattn_v2 if self.input_xattn_v2 is not None else self.input_xattn_v3
        if mod is None:
            return 0.0
        K = float(mod.traj_id_bound_K)
        warmup_start = int(mod.traj_id_K_warmup_start)
        warmup_duration = int(mod.traj_id_K_warmup_duration)
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
        the DAVIS NaN-skip guard pattern.
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
        from umm.models.wmm.rotation_utils import sixd_to_rotmat
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

    # ─────────────────────────────────────────────────────────────────
    # Hand-object loss (w1.6.0 Phase B1).
    # All methods are @torch.compiler.disable: mixed Python control flow,
    # data-dependent branching on cfg flags, .item() syncs, and lazy disk I/O.
    # Wrapping the entire feature in a single compile boundary avoids
    # graph-break recompiles in the LMM's hot loop.
    # ─────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _get_palm_rel_to_wrist(self, batch: dict) -> torch.Tensor:
        """Cached palm-relative-to-wrist offset, (B, 2, 3) in metres.

        Always uses neutral SMPL-X with zero body pose: produces the rest-pose
        palm offsets for each sample's betas. Per-gender SMPL-X bias is ~1cm,
        well below contact_threshold_m (2cm) — acceptable.

        Cache key: ("neutral", betas.tobytes()) per sample. Reused betas across
        the batch hit the cache.

        Args:
            batch: training batch dict. Reads batch["extra"]["betas"]: list[np.ndarray (16,)].

        Returns:
            (B, 2, 3) float32 tensor on self device, palm offset in zero-pose
            wrist-aligned frame. order = (L, R) following self.hand_palm_joint_idx.
        """
        # Lazy import — avoids pulling smplx into non-OMOMO configs.
        from umm.models.wmm.smplx_joints import smplx_joints_forward

        device = next(self.parameters()).device
        extras = batch["extra"]
        betas_list = extras["betas"]  # list[np.ndarray (16,)]
        B = len(betas_list)
        betas_np = np.stack(betas_list, axis=0).astype(np.float32)  # (B, 16)

        out = []
        for b in range(B):
            key = ("neutral", betas_np[b].tobytes())
            if key not in self._palm_offset_cache:
                betas_t = torch.from_numpy(betas_np[b:b + 1]).to(device)  # (1, 16)
                # T=1 single rest-pose frame, all body pose = 0, root pos/orient = 0.
                joints = smplx_joints_forward(
                    betas=betas_t,
                    pose_body_aa=torch.zeros(1, 1, 21, 3, device=device),
                    root_orient_aa=torch.zeros(1, 1, 3, device=device),
                    root_trans=torch.zeros(1, 1, 3, device=device),
                    smplx_dir=self.smplx_dir_handobj,
                    gender="neutral",
                )[0, 0]  # (J, 3)
                palm = joints[list(self.hand_palm_joint_idx)]   # (2, 3)
                wrist = joints[list(self.hand_wrist_joint_idx)]  # (2, 3)
                self._palm_offset_cache[key] = (palm - wrist).detach()
            out.append(self._palm_offset_cache[key])
        return torch.stack(out, dim=0)  # (B, 2, 3)

    @torch.compiler.disable
    def _gather_per_sample_assets(
        self,
        obj_categories: list,
        device: torch.device,
    ) -> tuple:
        """Build per-sample SDF + rest PCL tensors from the per-category cache.

        Lazily loads each new category from disk on first encounter:
          - SDF grid (256, 256, 256) + AABB centroid (3,) + extents (3,)
          - Rest-frame surface PCL (256, 3) sampled via trimesh

        Args:
            obj_categories: list[str] of length B, each a category stem (e.g. "clothesstand").
            device: target device for cached tensors.

        Returns:
            sdf_grids:    (B, 256, 256, 256) float32
            sdf_centroids:(B, 3) float32
            sdf_extents:  (B, 3) float32
            rest_pcls:    (B, 256, 3) float32, in object rest frame
        """
        # Lazy imports — only OMOMO needs these.
        import trimesh
        from umm.models.wmm.sdf_utils import load_sdf_asset

        sdf_grids = []
        centroids = []
        extents = []
        rest_pcls = []
        for cat in obj_categories:
            if cat not in self._sdf_cache:
                stem = osp.join(self.sdf_v2_dir, cat)
                sdf_np, c_np, e_np = load_sdf_asset(stem)  # numpy arrays
                self._sdf_cache[cat] = (
                    torch.from_numpy(sdf_np).to(device),
                    torch.from_numpy(c_np).to(device),
                    torch.from_numpy(e_np).to(device),
                )
                logger.info(
                    f"[handobj] loaded SDF for cat={cat} "
                    f"shape={sdf_np.shape} extents={e_np.tolist()}"
                )
            if cat not in self._rest_pcl_cache:
                ply_pcl_path = osp.join(self.rest_geo_dir, f"{cat}.ply")
                mesh = trimesh.load(ply_pcl_path, force="mesh", process=False)
                pts, _ = trimesh.sample.sample_surface(mesh, 256)
                self._rest_pcl_cache[cat] = (
                    torch.from_numpy(np.asarray(pts, dtype=np.float32)).to(device)
                )
                logger.info(
                    f"[handobj] loaded rest PCL for cat={cat} "
                    f"shape={tuple(self._rest_pcl_cache[cat].shape)}"
                )
            sdf_t, c_t, e_t = self._sdf_cache[cat]
            sdf_grids.append(sdf_t)
            centroids.append(c_t)
            extents.append(e_t)
            rest_pcls.append(self._rest_pcl_cache[cat])
        return (
            torch.stack(sdf_grids, dim=0),
            torch.stack(centroids, dim=0),
            torch.stack(extents, dim=0),
            torch.stack(rest_pcls, dim=0),
        )

    @torch.compiler.disable
    def _compute_gt_contact_label(
        self,
        batch: dict,
        rest_pcls: torch.Tensor,
        T: int,
        C: int,
    ) -> torch.Tensor:
        """Compute GT contact label (B, T, 2) from GT wrist positions and obj PCL.

        Wrist-vs-palm offset is ~7cm; with contact_label_eps_m=5cm a few labels
        may flip. Acceptable for v1.6 (refined in v1.6.2 by including palm offset).

        Reads:
          batch["extra"]["lhand_wrist_jpos"]: list[np.ndarray (T_full, 3)]
          batch["extra"]["rhand_wrist_jpos"]: list[np.ndarray (T_full, 3)]
          batch["trace"][:, obj_slot]:       GT object SE3 (B, T_full, 4, 4)

        Returns:
            (B, T, 2) float in {0, 1}; 1 == contact (GT min-dist < eps).
        """
        device = rest_pcls.device
        extras = batch["extra"]
        lwrist_list = extras.get("lhand_wrist_jpos")
        rwrist_list = extras.get("rhand_wrist_jpos")
        if lwrist_list is None or rwrist_list is None:
            # Dataset did not emit wrist positions; default = no contact (zero loss).
            B = rest_pcls.shape[0]
            return torch.zeros(B, T, 2, device=device)
        B = len(lwrist_list)
        # Each entry is (T_full, 3) — slice the last T frames (training-trace window).
        # NOTE: omomo dataset stores hand wrist arrays of length self.max_T already
        # (line 558-559), and trace covers the whole T window after C ctx prepend.
        # We slice [..., :T, :] to match the training trace span.
        lwrist = torch.from_numpy(
            np.stack([np.asarray(x, dtype=np.float32)[:T] for x in lwrist_list], axis=0)
        ).to(device)  # (B, T, 3)
        rwrist = torch.from_numpy(
            np.stack([np.asarray(x, dtype=np.float32)[:T] for x in rwrist_list], axis=0)
        ).to(device)
        gt_palms = torch.stack([lwrist, rwrist], dim=2)  # (B, T, 2, 3)

        # Find object slot. trace_sem_type may be (B, P) or (P,).
        sem_type = batch["trace_sem_type"]
        if sem_type.dim() == 2:
            obj_slot = (sem_type[0] == SEM_OBJECT).nonzero(as_tuple=True)[0].item()
        else:
            obj_slot = (sem_type == SEM_OBJECT).nonzero(as_tuple=True)[0].item()

        # GT object pose. trace shape: (B, P, T, 4, 4). The trace span is T frames.
        # batch["trace"] indexing convention check below.
        trace = batch["trace"]
        if trace.shape[1] == sem_type.shape[-1]:
            # trace shape = (B, P, T, 4, 4)
            gt_obj_se3 = trace[:, obj_slot, :T]  # (B, T, 4, 4)
        else:
            # trace shape = (B, T, P, 4, 4)
            gt_obj_se3 = trace[:, :T, obj_slot]  # (B, T, 4, 4)
        gt_obj_R = gt_obj_se3[..., :3, :3]
        gt_obj_t = gt_obj_se3[..., :3, 3]
        gt_obj_pcl_world = (
            torch.einsum("btij,bpj->btpi", gt_obj_R, rest_pcls)
            + gt_obj_t[:, :, None, :]
        )  # (B, T, 256, 3)

        BT = B * T
        d = torch.cdist(
            gt_palms.reshape(BT, 2, 3),
            gt_obj_pcl_world.reshape(BT, 256, 3),
        )  # (B*T, 2, 256)
        min_dist = d.min(dim=-1).values.reshape(B, T, 2)
        return (min_dist < self.contact_label_eps_m).float()

    def _normalize_diff_levels(self, diff_levels: torch.Tensor) -> torch.Tensor:
        """Map per-token diffusion timesteps to [0, 1].

        DDPM: integer in [0, num_train_timesteps-1] → divide by (N-1).
        FM / MeanFlow: float in [0, time_scale] → divide by time_scale.

        Same DENOMINATOR convention as the viz normalization in
        ``forward()`` (lines that compute ``viz_diff_levels``), but ALSO
        clamps to [0, 1]. The clamp is defensive — the timestep should
        already be in range, but a stray off-by-one in a custom diffusion
        scheduler must not produce a negative weight downstream.
        Args:
            diff_levels: (B, C+T, P) per-token diffusion timestep tensor
                as stored in `forward()`.
        Returns:
            Tensor of same shape, values in [0, 1] (clamped).
        """
        if self.is_flow_matching or self.is_meanflow:
            denom = max(self.diffusion_time_scale, 1.0)
            return (diff_levels.float() / denom).clamp(min=0.0, max=1.0)
        n_train = int(getattr(self.diffusion, "num_train_timesteps", 1))
        denom = max(n_train - 1, 1)
        return (diff_levels.float() / denom).clamp(min=0.0, max=1.0)

    @torch.compiler.disable
    def _compute_hand_object_loss(
        self,
        x0_pred_canon_std: torch.Tensor,
        canon_params,
        batch: dict,
        B: int,
        C: int,
        T: int,
        P: int,
        phase: str,
        log_dict: dict,
        diff_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDF penetration + contact loss on x0_pred. NO-OP when both weights == 0.

        All tensor ops within @torch.compiler.disable boundary to avoid compile
        recompiles on shape-varying SDF queries.

        Args:
            x0_pred_canon_std: (B, C+T, P, 9) standard 9D canon predicted x0.
                Gradient flows through this tensor — DO NOT detach.
            canon_params: CanonParams used by self.canonicalizer for inverse.
            batch: training batch dict.
            B, C, T, P: standard agent shape integers.
            phase: "train"|"val" log key prefix.
            log_dict: in-out — populated with this term's logs.
            diff_levels: (B, C+T, P) per-token diffusion timestep (DDPM int
                or FM/MeanFlow scaled). Required when
                ``self.aux_loss_t_weighted=True``; else ignored. Pre-w1.7.x.2
                callers may pass ``None``.

        Returns:
            scalar loss tensor (already weighted). Zero when feature disabled.
        """
        device = x0_pred_canon_std.device
        if (
            self.loss_sdf_penetration_weight == 0.0
            and self.loss_contact_weight == 0.0
            and self.loss_full_body_penetration_weight == 0.0  # w1.6.2 gate
        ):
            return torch.zeros((), device=device)

        # 1. De-canonicalize x0_pred WITH gradient flow (mirror of viz block but
        #    NOT detached). canonicalizer.inverse expects (B, N, 9) flat.
        N = (C + T) * P
        x0_pred_flat = x0_pred_canon_std.reshape(B, N, 9)
        x0_pred_orig = self.canonicalizer.inverse(x0_pred_flat, canon_params)
        x0_pred_orig = x0_pred_orig.reshape(B, C + T, P, 9)
        x0_pred_se3 = ninep_to_se3(x0_pred_orig)  # (B, C+T, P, 4, 4)

        # 2. Object slot from semantics.
        sem_type = batch["trace_sem_type"]
        if sem_type.dim() == 2:
            obj_slot = (sem_type[0] == SEM_OBJECT).nonzero(as_tuple=True)[0].item()
        else:
            obj_slot = (sem_type == SEM_OBJECT).nonzero(as_tuple=True)[0].item()

        # 3. Predicted wrist SE3, then palm = wrist_t + R_wrist @ palm_rel.
        pred_wrist_se3 = x0_pred_se3[:, :, list(self.hand_wrist_joint_idx)]  # (B, C+T, 2, 4, 4)
        pred_wrist_R = pred_wrist_se3[..., :3, :3]
        pred_wrist_t = pred_wrist_se3[..., :3, 3]                            # (B, C+T, 2, 3)

        palm_rel = self._get_palm_rel_to_wrist(batch)                        # (B, 2, 3)
        # pred_wrist_R: (B, C+T, i_hand=2, j_row=3, k_col=3). palm_rel: (B, i_hand=2, k_col=3).
        # palm_offset_world[b, t, i, j] = sum_k pred_wrist_R[b, t, i, j, k] * palm_rel[b, i, k]
        palm_offset_world = torch.einsum(
            "btijk,bik->btij", pred_wrist_R, palm_rel
        )
        pred_palm_world = pred_wrist_t + palm_offset_world  # (B, C+T, 2, 3)

        # 4. Object SE3 from prediction.
        pred_obj_se3 = x0_pred_se3[:, :, obj_slot]  # (B, C+T, 4, 4)
        pred_obj_R = pred_obj_se3[..., :3, :3]
        pred_obj_t = pred_obj_se3[..., :3, 3]

        # Slice T-window only (skip C ctx).
        pred_palm_world_t = pred_palm_world[:, C:]  # (B, T, 2, 3)
        pred_obj_R_t = pred_obj_R[:, C:]            # (B, T, 3, 3)
        pred_obj_t_t = pred_obj_t[:, C:]            # (B, T, 3)

        # 5. Per-sample SDF + rest PCL via category-keyed cache.
        obj_categories = batch["extra"]["obj_category"]  # list[str] length B
        sdf_grids, sdf_centroids, sdf_extents, rest_pcls = (
            self._gather_per_sample_assets(obj_categories, device=device)
        )

        # 6. SDF query at predicted palm.
        # Lazy import — see top-of-file note re: byte-equal back-compat.
        from umm.models.wmm.sdf_utils import query_sdf_at_world_points
        sdf_at_palm = query_sdf_at_world_points(
            pred_palm_world_t, pred_obj_R_t, pred_obj_t_t,
            sdf_grids, sdf_centroids, sdf_extents,
        )  # (B, T, 2) signed metres

        total = torch.zeros((), device=device)

        # 6b. Per-palm time-weight w(t) = 1 - t/(T_diff-1), shape (B, T, 2).
        # Wrist-slot t (slots 20=L, 21=R) is the kinematic parent of each palm.
        # Only computed when the new flag is enabled; else palm_t_weight = 1.0.
        # If the flag is True but diff_levels was not piped through (caller
        # bug), warn loudly so the silent fallback doesn't hide a regression.
        palm_t_weight = None
        if self.aux_loss_t_weighted:
            if diff_levels is None:
                logger.warning(
                    "aux_loss_t_weighted=True but diff_levels=None passed "
                    "to _compute_hand_object_loss; falling back to unweighted "
                    "aux losses. Caller did not thread diff_levels through."
                )
            else:
                t_norm = self._normalize_diff_levels(diff_levels)  # (B, C+T, P)
                # Slice T-window, then per-palm wrist slots.
                t_palm = t_norm[:, C:][:, :, list(self.hand_wrist_joint_idx)]  # (B, T, 2)
                palm_t_weight = (1.0 - t_palm).clamp(min=0.0, max=1.0)
                # Log once per call (active for BOTH SDF and contact paths).
                log_dict[f"{phase}_palm_t_weight_mean"] = (
                    palm_t_weight.mean().detach()
                )

        # 7. Penetration loss: relu(-sdf + margin) (5mm soft hinge by default).
        if self.loss_sdf_penetration_weight > 0.0:
            pen_per_palm = torch.relu(
                -sdf_at_palm + self.penetration_margin_m
            )  # (B, T, 2)
            if palm_t_weight is not None:
                pen_per_palm = pen_per_palm * palm_t_weight
            loss_sdf = pen_per_palm.mean()
            log_dict[f"{phase}_loss_sdf_penetration"] = loss_sdf.detach()
            log_dict[f"{phase}_sdf_at_palm_min"] = sdf_at_palm.min().detach()
            log_dict[f"{phase}_sdf_at_palm_mean"] = sdf_at_palm.mean().detach()
            total = total + self.loss_sdf_penetration_weight * loss_sdf

        # 8. Contact loss: cdist palm -> world-frame PCL, masked by GT label.
        if self.loss_contact_weight > 0.0:
            # World-frame obj PCL via predicted obj pose.
            obj_pcl_world_t = (
                torch.einsum("btij,bpj->btpi", pred_obj_R_t, rest_pcls)
                + pred_obj_t_t[:, :, None, :]
            )  # (B, T, 256, 3)

            BT = B * T
            d = torch.cdist(
                pred_palm_world_t.reshape(BT, 2, 3),
                obj_pcl_world_t.reshape(BT, 256, 3),
            )  # (B*T, 2, 256)
            min_dist = d.min(dim=-1).values.reshape(B, T, 2)  # (B, T, 2)

            gt_contact_label = self._compute_gt_contact_label(
                batch, rest_pcls, T, C
            )  # (B, T, 2) float
            n_contact = gt_contact_label.sum().clamp_min(1.0)
            contact_per_palm = (
                torch.relu(min_dist - self.contact_threshold_m)
                * gt_contact_label
            )  # (B, T, 2)
            if palm_t_weight is not None:
                contact_per_palm = contact_per_palm * palm_t_weight
            # n_contact stays unweighted in denominator: keeps loss in metres.
            loss_contact = contact_per_palm.sum() / n_contact
            log_dict[f"{phase}_loss_contact"] = loss_contact.detach()
            log_dict[f"{phase}_contact_label_frac"] = (
                n_contact / float(B * T * 2)
            ).detach()
            total = total + self.loss_contact_weight * loss_contact

        # 9. w1.6.2: full-body SMPL-X mesh penetration loss.
        #    Queries SDF at all 10,475 SMPL-X verts in GT object's rest frame.
        #    Matches the eval P_hand metric (collision.py) semantics.
        #    Default weight 0 → branch INERT for w1.6.0/w1.6.1.
        if self.loss_full_body_penetration_weight > 0.0:
            loss_fb = self._compute_full_body_penetration_loss(
                x0_pred_se3=x0_pred_se3, batch=batch,
                B=B, C=C, T=T, obj_slot=obj_slot,
                sdf_grids=sdf_grids,
                sdf_centroids=sdf_centroids,
                sdf_extents=sdf_extents,
                log_dict=log_dict, phase=phase,
                diff_levels=diff_levels,
            )
            total = total + self.loss_full_body_penetration_weight * loss_fb

        return total

    @torch.compiler.disable
    def _compute_full_body_penetration_loss(
        self,
        x0_pred_se3: torch.Tensor,    # (B, C+T, P, 4, 4)
        batch: dict,
        B: int,
        C: int,
        T: int,
        obj_slot: int,
        sdf_grids: torch.Tensor,      # (B, 256, 256, 256)
        sdf_centroids: torch.Tensor,  # (B, 3)
        sdf_extents: torch.Tensor,    # (B, 3)
        log_dict: dict,
        phase: str,
        diff_levels: Optional[torch.Tensor] = None,  # (B, C+T, P)
    ) -> torch.Tensor:
        """Full-body SMPL-X mesh penetration loss vs object SDF (w1.6.2).

        Targets the P_hand eval metric (`src/umm/eval/omomo/collision.py`).
        Queries the per-sample object SDF at all 10,475 SMPL-X mesh verts
        driven by the predicted body Q SE(3) (slots 0..21 of the OMOMO
        trace), in the GT object's rest frame. Penalizes verts more than
        ``full_body_pen_allowance_m`` (default 5 mm) inside the surface.

        **GT object pose** (not predicted): matches metric semantics and
        decouples body-pen loss from obj-pose prediction error. This is a
        deliberate design choice.

        **Per-gender batching**: betas + gender vary per sample; SMPL-X FK
        is called once per gender group then results gathered back. On
        per-gender FK failure, the affected samples' contribution is zero
        and a warning is printed (per project error-handling rule).

        Args:
            x0_pred_se3: predicted per-joint world SE(3) (B, C+T, P, 4, 4).
                Body joints occupy slots [0..21] in OMOMO trace (P=23).
            batch: training batch dict.
            B, C, T: standard agent shape integers.
            obj_slot: index of the object entity in trace_sem_type (already
                resolved by caller).
            sdf_grids/centroids/extents: per-sample SDF assets (already
                gathered by ``_gather_per_sample_assets``).
            log_dict: in-out dict — populated with per-batch metrics.
            phase: ``"train"`` or ``"val"`` log key prefix.

        Returns:
            Scalar loss tensor (mean penetration depth over violating
            verts, in metres). Zero when no verts violate the allowance
            or all per-gender FK calls fail.
        """
        # Lazy imports — see top-of-file note re: byte-equal back-compat.
        # Loading pytorch3d / smplx at module top changes CUDA kernel init
        # order and perturbs fp32 results by ~2 ULP on existing palm path.
        from umm.models.wmm.se3_to_aa import se3_to_aa_torch
        from umm.models.wmm.smplx_joints import (
            smplx_pelvis_rest_offset_torch,
            smplx_verts_forward,
        )
        from umm.models.wmm.sdf_utils import query_sdf_at_world_points

        device = x0_pred_se3.device

        # 1. Slice predicted body Q (22 joints) for the T-window only.
        pred_body_Q = x0_pred_se3[:, C:C + T, 0:22]  # (B, T, 22, 4, 4)

        # 2. GT object SE(3) for the T-window. Same indexing pattern as
        #    _compute_gt_contact_label (handles (B,P,T) and (B,T,P) layouts).
        trace = batch["trace"]
        sem_type = batch["trace_sem_type"]
        if trace.shape[1] == sem_type.shape[-1]:
            gt_obj_se3 = trace[:, obj_slot, :T]   # (B, T, 4, 4)
        else:
            gt_obj_se3 = trace[:, :T, obj_slot]   # (B, T, 4, 4)
        gt_obj_R = gt_obj_se3[..., :3, :3]
        gt_obj_t = gt_obj_se3[..., :3, 3]

        # 3. Per-gender SMPL-X FK. Group batch indices by gender, call FK
        #    per group, gather verts back into (B, T, 10475, 3).
        extras = batch["extra"]
        betas_list = extras["betas"]   # list[np.ndarray (16,)] of length B
        genders = extras["gender"]     # list[str] of length B

        verts_per_sample: List[Optional[torch.Tensor]] = [None] * B
        for gender in set(genders):
            idx_b = [b for b in range(B) if genders[b] == gender]
            if not idx_b:
                continue
            try:
                betas_g = torch.stack(
                    [
                        torch.from_numpy(
                            np.asarray(betas_list[b], dtype=np.float32)
                        )
                        for b in idx_b
                    ],
                    dim=0,
                ).to(device)  # (Bg, 16)
                Q_g = pred_body_Q[idx_b]  # (Bg, T, 22, 4, 4)
                pelvis_g = smplx_pelvis_rest_offset_torch(
                    betas=betas_g,
                    gender=gender,
                    smplx_dir=self.smplx_dir_handobj,
                )
                root_trans_g, root_aa_g, body_aa_g = se3_to_aa_torch(
                    Q_g, pelvis_g
                )
                verts_g = smplx_verts_forward(
                    betas=betas_g,
                    pose_body_aa=body_aa_g,
                    root_orient_aa=root_aa_g,
                    root_trans=root_trans_g,
                    smplx_dir=self.smplx_dir_handobj,
                    gender=gender,
                )  # (Bg, T, 10475, 3)
                for i, b in enumerate(idx_b):
                    verts_per_sample[b] = verts_g[i]
            except Exception as e:
                # Per project error handling: log + continue (zero
                # contribution from this group). Never silently swallow.
                # flush=True so warnings are visible in redirected
                # background-job logs (otherwise buffered until exit).
                print(
                    f"WARN: full-body penetration FK failed for "
                    f"gender={gender!r} samples={idx_b}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        valid_idx = [b for b in range(B) if verts_per_sample[b] is not None]
        if not valid_idx:
            # Emit ALL the same log keys as the success path (filled with
            # safe defaults) so downstream W&B charts don't have gaps when
            # every per-gender FK fails for a given step.
            zero = torch.zeros((), device=device)
            log_dict[f"{phase}_loss_full_body_pen"] = zero
            log_dict[f"{phase}_full_body_pen_max_cm"] = zero
            log_dict[f"{phase}_full_body_pen_n_viol"] = zero
            log_dict[f"{phase}_full_body_pen_n_valid"] = 0.0
            return zero

        verts_world = torch.stack(
            [verts_per_sample[b] for b in valid_idx], dim=0
        )  # (Bv, T, 10475, 3)

        sdf_grids_v = sdf_grids[valid_idx]
        sdf_centroids_v = sdf_centroids[valid_idx]
        sdf_extents_v = sdf_extents[valid_idx]
        gt_obj_R_v = gt_obj_R[valid_idx]
        gt_obj_t_v = gt_obj_t[valid_idx]

        # 4. SDF query at all 10,475 verts in GT object rest frame.
        sdf_at_verts = query_sdf_at_world_points(
            verts_world, gt_obj_R_v, gt_obj_t_v,
            sdf_grids_v, sdf_centroids_v, sdf_extents_v,
        )  # (Bv, T, 10475) signed metres

        # 4b. Mask out padded frames (frames >= actual_len): both body and
        #     object are identity SE(3) in pad frames, which collide at the
        #     world origin. Without masking, padded frames inject ~10475×
        #     spurious penetration into the loss every batch — biasing the
        #     gradient and dominating the real signal. Use the dataset's
        #     trace_mask (True = valid, False = pad). Body joints share the
        #     same per-frame mask so picking joint 0 is canonical.
        trace_mask = batch["trace_mask"]  # (B, T, P) or (B, P, T)
        if trace_mask.shape[1] == sem_type.shape[-1]:
            # (B, P, T) layout
            valid_frame_mask = trace_mask[:, 0, :T]      # (B, T) bool — body slot 0
        else:
            # (B, T, P) layout (canonical for OMOMO at runtime)
            valid_frame_mask = trace_mask[:, :T, 0]      # (B, T) bool — body slot 0
        valid_frame_mask_v = valid_frame_mask[valid_idx].float()  # (Bv, T)

        # 5. Penetration loss: penalize verts beyond allowance INSIDE the
        #    surface, ONLY on valid frames. Mean over violators only (matches
        #    metric averaging pattern; stable signal even when most verts
        #    are non-violating).
        pen_depth_phys = torch.relu(
            -sdf_at_verts - self.full_body_pen_allowance_m
        )  # (Bv, T, 10475) — physical penetration depth in metres
        # Apply per-frame validity mask (broadcast (Bv, T, 1) → (Bv, T, V)).
        pen_depth_phys = pen_depth_phys * valid_frame_mask_v[:, :, None]
        # 5b. Per-frame time-weight w(t) over body slots, shape (Bv, T).
        # Only when aux_loss_t_weighted=True; else weight is 1 everywhere.
        # Body slots 0..21 (full SMPL-X body excluding object). Per-frame mean
        # over body joints' t — any noisy joint slightly drops the weight.
        # IMPORTANT: keep ``pen_depth_phys`` (UN-weighted) for monitoring
        # metrics so logged values stay interpretable as physical depth.
        # Apply weights only to the loss numerator ``pen_depth_loss``.
        if (
            self.aux_loss_t_weighted
            and diff_levels is not None
        ):
            t_norm = self._normalize_diff_levels(diff_levels)  # (B, C+T, P)
            t_body_mean = t_norm[:, C:][:, :, 0:22].mean(dim=-1)  # (B, T)
            t_body_mean_v = t_body_mean[valid_idx]  # (Bv, T)
            body_t_weight = (1.0 - t_body_mean_v).clamp(min=0.0, max=1.0)  # (Bv, T)
            pen_depth_loss = pen_depth_phys * body_t_weight[:, :, None]
            log_dict[f"{phase}_body_t_weight_mean"] = body_t_weight.mean().detach()
        else:
            pen_depth_loss = pen_depth_phys
        # n_viol counted on the PHYSICAL tensor — keeps the denominator
        # interpretable (number of vert-frames with real geometric overlap),
        # independent of the time-weight gating. Loss numerator uses the
        # weighted tensor so contributions decay smoothly with t.
        n_viol = (pen_depth_phys > 0).float().sum().clamp_min(1.0)
        loss_pen = pen_depth_loss.sum() / n_viol

        log_dict[f"{phase}_loss_full_body_pen"] = loss_pen.detach()
        # Monitoring stats use the UN-weighted physical tensor so values stay
        # interpretable as actual penetration depths (cm) and counts.
        log_dict[f"{phase}_full_body_pen_max_cm"] = (
            pen_depth_phys.max() * 100.0
        ).detach()
        log_dict[f"{phase}_full_body_pen_n_viol"] = n_viol.detach()
        log_dict[f"{phase}_full_body_pen_n_valid"] = float(len(valid_idx))
        log_dict[f"{phase}_full_body_pen_n_valid_frames"] = (
            valid_frame_mask_v.sum().detach()
        )

        return loss_pen

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
        C: int = 0,
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

        # Auxiliary per-channel L1 on position only (mirrors eval MAE metric).
        # Computed once; gated by self.w_mae > 0 below. Shares pos_scale,
        # w_ctx, token_weights, movement_weight modifiers with pos_loss.
        pos_mae_loss = None
        if self.w_mae > 0:
            pos_mae_loss = (pred[..., :3] - target[..., :3]).abs().sum(dim=-1)  # (B, C+T, P)

        # Apply GT-canon scale factor to position loss only.
        # GT-canon is a SIM(3) transform: T(x) = s * R * (x - mean).
        # Rotation loss is NOT affected by GT-canon.
        if pos_scale is not None:
            pos_loss = pos_loss * pos_scale  # (B, C+T, P) * (B, 1, 1)
            if pos_mae_loss is not None:
                pos_mae_loss = pos_mae_loss * pos_scale

        # w_ctx: scale context-frame loss by optional weight (default 1.0 = no-op back-compat).
        # Applied BEFORE mask selection so Huber/MSE values in ctx slots are scaled.
        if self.w_ctx != 1.0 and C > 0:
            _ctx_scale = torch.ones_like(pos_loss)
            _ctx_scale[:, :C] = self.w_ctx
            pos_loss = pos_loss * _ctx_scale
            rot_loss = rot_loss * _ctx_scale
            if pos_mae_loss is not None:
                pos_mae_loss = pos_mae_loss * _ctx_scale

        # Apply per-token weights (e.g. min-SNR reweighting)
        if token_weights is not None:
            pos_loss = pos_loss * token_weights
            rot_loss = rot_loss * token_weights
            if pos_mae_loss is not None:
                pos_mae_loss = pos_mae_loss * token_weights

        # M5: Movement reweighting -- up-weight moving entities
        if movement_weight is not None:
            pos_loss = pos_loss * movement_weight
            rot_loss = rot_loss * movement_weight
            if pos_mae_loss is not None:
                pos_mae_loss = pos_mae_loss * movement_weight

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

        # Auxiliary MAE (per-channel L1 on position) -- real / virtual.
        # Gated by w_mae > 0 (default 0.0 = disabled, no compute beyond the
        # cheap absdiff above). Mirrors the eval `trajectory_mae` metric.
        if pos_mae_loss is not None:
            if real_mask.any():
                mae_real_vals = pos_mae_loss[real_mask]
                mae_real_ids = _ids[real_mask]
                mae_real_term = (
                    type_mean(mae_real_vals, mae_real_ids, self.gid_w_modifier)
                    * self.w_real * self.w_mae
                )
                loss = loss + mae_real_term
                log_dict[f"{phase}_LOSS/pos_mae_real"] = mae_real_term.detach()

            if virt_mask_full is not None and virt_mask_full.any():
                mae_virt_vals = pos_mae_loss[virt_mask_full]
                mae_virt_ids = _ids[virt_mask_full]
                mae_virt_term = (
                    type_mean(mae_virt_vals, mae_virt_ids, self.gid_w_modifier)
                    * self.w_virtual * self.w_mae
                )
                loss = loss + mae_virt_term
                log_dict[f"{phase}_LOSS/pos_mae_virt"] = mae_virt_term.detach()

        # Rotation loss -- real
        rot_real_mask = rot_valid & ~virtual_mask if virtual_mask is not None else rot_valid
        rot_virt_mask = rot_valid & virtual_mask if virtual_mask is not None else None

        if rot_real_mask.any():
            rot_real_vals = rot_loss[rot_real_mask]
            rot_real_ids = _ids[rot_real_mask]
            rot_real_term = type_mean(rot_real_vals, rot_real_ids, self.gid_w_modifier) * self.w_real * self.w_q
            loss = loss + rot_real_term
            log_dict[f"{phase}_LOSS/rot_real"] = rot_real_term.detach()

        # Rotation loss -- virtual
        if rot_virt_mask is not None and rot_virt_mask.any():
            rot_virt_vals = rot_loss[rot_virt_mask]
            rot_virt_ids = _ids[rot_virt_mask]
            rot_virt_term = type_mean(rot_virt_vals, rot_virt_ids, self.gid_w_modifier) * self.w_virtual * self.w_q
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
        # zeros their features (and the routing in InputLayerV11Super only
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

        # w1.4.0: optional per-(b, p) cur-frame index (T-space, [0, T)) from
        # the dataset adapter. Stacked from extra (which is a list per key).
        # Falls back to None for datasets that don't emit it; LMM forward
        # asserts non-None only when bias_to_cur_frame=True.
        cur_frame_idx = None
        _extra = batch.get("extra", None)
        if _extra is not None:
            _cf_list = _extra.get("cur_frame_idx", None)
            if (
                _cf_list is not None
                and len(_cf_list) > 0
                and all(t is not None for t in _cf_list)
            ):
                cur_frame_idx = torch.stack(
                    [t.to(device=device, dtype=torch.long) for t in _cf_list], dim=0
                )  # (B, P) long

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

        # Canon source selection. canon_mask is LOCAL — do NOT mutate
        # all_known / all_known_flat / known_valid_flat. Downstream code
        # (diffusion noise scheduling, LMM attention masks, loss weighting,
        # async-noise pathways) must continue to see the unmodified all_known.
        if self.canon_source == "ctx_valid":
            # Use ALL valid ctx tokens (ignore known mask, exclude all T-slice tokens).
            # NOTE: assumes C >= 1. When C == 0 (no ctx slot), every sample lands
            # at zero valid tokens and the canonicalizer's existing identity
            # fallback fires per-sample (loss is masked, so safe). OMOMO uses C=1.
            canon_mask_2d = torch.zeros_like(all_mask)  # (B, C+T, P) bool
            canon_mask_2d[:, :C] = context_mask[:, :C]   # only valid ctx; T side stays False
            canon_mask = canon_mask_2d.reshape(B, N)     # (B, N)
        else:
            canon_mask = known_valid_flat

        canon_9d_flat, canon_params = self.canonicalizer.forward(
            all_9d_flat, canon_mask
        )  # (B, N, 9), CanonParams

        # Override R_canon to identity when rotation canonicalization is disabled.
        # Position centering + scaling still apply. Only rotation alignment is skipped.
        if not self.canon_rotation:
            canon_params = CanonParams(
                mean_pos=canon_params.mean_pos,
                scale=canon_params.scale,
                R_canon=torch.eye(3, device=device, dtype=canon_params.R_canon.dtype)
                    .unsqueeze(0).expand(B, -1, -1),
                bounded=canon_params.bounded,
                atanh_eps=canon_params.atanh_eps,
                ratio_inside_thr=canon_params.ratio_inside_thr,
                std_axis=canon_params.std_axis,
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

        # V13.3.4: Conditional training mask (25% of samples)
        unknown_x_mask_for_lmm = None
        if self.twostage_training_mask:
            clean_known_branch = getattr(self.diffusion, '_last_clean_known_branch', None)
            if clean_known_branch is not None:
                # Second 50% coin flip -> 25% total
                second_flip = torch.rand(B, 1, device=device) < 0.5
                use_restricted = clean_known_branch & second_flip  # (B, 1)
                if use_restricted.any():
                    # V13.3.4: Unknown X = valid trace positions that are NOT known
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
        # canon_params.scale is (B, 3) per-axis; broadcast over (B, P, PCL, 3) xyz last-dim.
        # NOTE: bounded variant applies tanh on the LMM 9D pos but NOT on pcl xyz.
        # Both variants here apply only the linear per-axis scale to pcl xyz, which is
        # the correct geometric operation for the linear-scaled space the LMM input
        # representation lives in. (For the original variant all 3 axes are equal,
        # so this reduces to the legacy uniform-scale broadcast.)
        if local_pcl is not None:
            pcl_scale = canon_params.scale[:, None, None, :]  # (B, 1, 1, 3)
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
            # one-time DINO sanity log at
            # global_step==0. Proves DINO image features are non-zero before
            # they enter the LMM. Gated to fire EXACTLY ONCE per training run
            # to avoid spam. agent.forward is NOT inside the torch.compile
            # boundary (only self.lmm is compiled in pl_model.py), so the
            # one-shot branch costs no recompile. The .item() guards follow
            # the project's "no GPU sync in logging" rule
            # (Debug Logging Performance);
            # fires only at step 0 if INFO logging is enabled.
            if int(global_step) == 0 and logger.isEnabledFor(logging.INFO):
                _img_proj_d = img_proj.detach().float()
                logger.info(
                    "[DINO step-0 sanity] img_proj.shape=%s std=%.6e norm=%.6e "
                    "img_mask.sum=%d",
                    tuple(img_proj.shape),
                    float(_img_proj_d.std().item()),
                    float(_img_proj_d.norm().item()),
                    int(img_mask.sum().item()),
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

        # w1.0.2: optional input_xattn_v3.
        # Same K-warmup mechanism as v2; additionally surfaces a projected
        # pooled-lang vector `pooled_lang_proj` (B, D1) routed to the LMM's
        # cond_parts (replaces the `add_first_n_add_token_to_feat` slot).
        pooled_lang_proj = None
        if self.input_xattn_v3_enable:
            valid_entity_mask = traj_padding_mask  # (B, P) bool
            step_i = int(global_step)
            effective_K = self._compute_effective_K(step_i, training=self.training)
            if step_i % 100 == 0 or step_i < 10:
                logger.info(
                    f"[input_xattn_v3] step={step_i} training={self.training} "
                    f"effective_K={effective_K:.4f} "
                    f"bound_K={self.input_xattn_v3.traj_id_bound_K} "
                    f"warmup_start={self.input_xattn_v3.traj_id_K_warmup_start} "
                    f"warmup_duration={self.input_xattn_v3.traj_id_K_warmup_duration}"
                )
            v3_out = self.input_xattn_v3(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask,
                effective_K=effective_K,
            )
            feat = v3_out.feat_out
            learned_traj_ids = v3_out.traj_ids
            pooled_lang_proj = v3_out.pooled_lang_proj
            # Shape assert at the wiring point: (B, D1). D1 is read from the
            # LMM (same value passed to v3 at construction).
            _B = feat.shape[0]
            _D1 = int(self.lmm.D1)
            assert pooled_lang_proj.shape == (_B, _D1), (
                f"expected pooled_lang_proj shape (B={_B}, D1={_D1}), "
                f"got {tuple(pooled_lang_proj.shape)}"
            )

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
            add_tokens=lang_proj,
            add_tokens_mask=lang_mask,
            image_tokens=img_proj,
            image_tokens_mask=img_mask,
            group_ids=group_id,
            cur_frame_idx=cur_frame_idx,
        )
        # MeanFlow: pass r_levels as additional interval conditioning
        if r_diff_levels is not None:
            lmm_kwargs["x_r_level"] = r_diff_levels[:, C:]
            lmm_kwargs["ctx_r_level"] = r_diff_levels[:, :C]
        # V13.2 Explore 10: pass learned traj_ids to LMM
        if learned_traj_ids is not None:
            lmm_kwargs["learned_traj_ids"] = learned_traj_ids
        # V13.3.4: pass dynamic unknown_x_mask to LMM
        if unknown_x_mask_for_lmm is not None:
            lmm_kwargs["unknown_x_mask"] = unknown_x_mask_for_lmm
        # w1.0.2: pooled-lang (B, D1) into cond_parts.
        if pooled_lang_proj is not None:
            lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj
        out = self.lmm(**lmm_kwargs)  # (B, C+T, P, 9)

        # NaN/Inf skip guard (DAVIS lesson): if LMM output has any NaN/Inf,
        # return a zero-grad dummy loss and skip this optimizer step. This
        # prevents a single bad step (e.g. fp16 overflow at high step counts)
        # from killing a long-running training run.
        if not self._check_finite(out):
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
        # Bounded variant: skip GT-canon entirely.
        #
        # WHY: GT-canon is a SIM(3) reweighting that only makes sense when
        # canon_9d_flat lives in linear (un-saturated) position space. Under
        # bounded=True, canon_9d_flat positions have been passed through tanh
        # and lie in (-1, 1) by construction with std≈target_pos_std on the
        # compute set. Two failure modes if we reused the legacy GT-canon
        # path here:
        #   (1) compute_canon_params would compute a SIM(3) on tanh-compressed
        #       data, yielding a scalar scale ≈ 1 (effectively identity) —
        #       garbage in, garbage out for s_linear and x0_*_gt metrics.
        #   (2) canonicalize() then re-applies a linear transform to data
        #       that is already in (-1, 1) — semantically meaningless and
        #       misleading for per-timestep metrics.
        # Bounded data is already at unit-ish scale by tanh construction, so
        # neither reweighting nor a "GT-canon transform" is needed: identity
        # reweight for the loss + identity passthrough for x0_*_gt metrics.
        if self.canon_bounded:
            s_linear = torch.ones(
                B, 1, 1, device=device, dtype=canon_9d_flat.dtype,
            )
            with torch.no_grad():
                x0_pred_gt = x0_pred_canon_std.detach().reshape(B, C + T, P, 9)
                x0_gt_gt = x0_gt_canon_std.reshape(B, C + T, P, 9)
        else:
            gt_canon_params = compute_canon_params(
                canon_9d_flat,
                all_mask_flat,
                target_pos_std=self.target_pos_std,
                canon_rotation=self.canon_rotation,
                per_axis_std=self.canon_per_axis_std,
            )
            # s per sample, shape (B, 3) → (B, 1, 1) for broadcast over (B, C+T, P).
            # When per_axis_std=False (legacy): all 3 axes are identical (scalar
            # broadcast in the bounded=False path), so mean(...) reduces to that
            # scalar. When per_axis_std=True: 3 distinct per-axis scales → mean
            # gives a single sample-wise reweighting; consistent with how the
            # main canon was computed.
            # (LINEAR s, not s^2 — V11 design decision)
            s_linear = gt_canon_params.scale.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

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
            C=C,
        )

        # Hand-object loss (SDF penetration + contact). NO-OP when both weights == 0.
        # Gated entirely inside _compute_hand_object_loss for back-compat with w1.5.x.
        # diff_levels shape (B, C+T, P) — per-token diffusion timestep, used
        # only when aux_loss_t_weighted=True (else passed but ignored).
        hand_obj_loss = self._compute_hand_object_loss(
            x0_pred_canon_std, canon_params, batch, B, C, T, P, phase, log_dict,
            diff_levels=diff_levels,
        )
        loss = loss + hand_obj_loss
        log_dict[f"{phase}_LOSS/hand_obj"] = hand_obj_loss.detach()

        ret["loss"] = loss

        # Canon diagnostic stats — log per-axis mean/std/scale and ratio of
        # known points inside the depth threshold (bounded variant only).
        # All values reduced over the batch dim with .mean() so wandb sees a
        # scalar per axis.
        with torch.no_grad():
            cm = canon_params.mean_pos.detach()  # (B, 3)
            cs = canon_params.scale.detach()     # (B, 3)
            for i, ax in enumerate("xyz"):
                log_dict[f"{phase}_canon/mean_{ax}"] = cm[:, i].mean()
                log_dict[f"{phase}_canon/scale_{ax}"] = cs[:, i].mean()
            if canon_params.ratio_inside_thr is not None:
                log_dict[f"{phase}_canon/ratio_inside_thr"] = (
                    canon_params.ratio_inside_thr.detach().mean()
                )
            if canon_params.std_axis is not None:
                csa = canon_params.std_axis.detach()  # (B, 3)
                for i, ax in enumerate("xyz"):
                    log_dict[f"{phase}_canon/std_{ax}"] = csa[:, i].mean()

        ret["log_dict"] = log_dict

        # 11. Per-timestep-bucket X0 accuracy metrics
        with torch.no_grad():
            # Normalize timesteps to [0, 1]
            if self.is_flow_matching:
                t_norm = diff_levels / self.diffusion_time_scale  # already [0, 1]
            else:
                t_norm = diff_levels / float(self.diffusion.num_train_timesteps)

            # Per-token squared error between x0_pred and x0_gt (in GT-canon space).
            # Split pos vs rot — under rot_parameterization=6d_wing, the wing→canon
            # inverse divides residuals by wl (=0.2 default), inflating wing-space
            # error by 1/wl per dim and ~2/wl² per rot6d-dim variance, so the
            # combined metric was rot-dominated and ~300× amplified vs wing loss.
            pos_se = (x0_pred_gt[..., :3] - x0_gt_gt[..., :3]).pow(2).sum(dim=-1)   # 3 dims
            rot_se = (x0_pred_gt[..., 3:] - x0_gt_gt[..., 3:]).pow(2).sum(dim=-1)   # 6 dims
            x0_mse = pos_se + rot_se                                                # 9 dims (legacy)
            pos_mse_valid = pos_se[all_mask]
            rot_mse_valid = rot_se[all_mask]
            x0_mse_valid = x0_mse[all_mask]  # (K,) only valid tokens
            t_valid = t_norm[all_mask]  # (K,)

            # 4 buckets: [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0]
            bucket_edges = [0.0, 0.25, 0.5, 0.75, 1.01]
            for i in range(4):
                bucket_mask = (t_valid >= bucket_edges[i]) & (t_valid < bucket_edges[i + 1])
                if bucket_mask.any():
                    edge_name = f"t_{bucket_edges[i]:.2f}_{bucket_edges[i+1]:.2f}"
                    log_dict[f"{phase}_X0_MSE/{edge_name}"] = x0_mse_valid[bucket_mask].mean()
                    log_dict[f"{phase}_X0_MSE_pos/{edge_name}"] = pos_mse_valid[bucket_mask].mean()
                    log_dict[f"{phase}_X0_MSE_rot/{edge_name}"] = rot_mse_valid[bucket_mask].mean()

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
        """V13.3.4: Extract inference preparation into reusable helper.

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

        # Canon source selection (parity with forward). canon_mask is LOCAL —
        # do NOT mutate all_known / all_known_flat / known_valid_flat.
        if self.canon_source == "ctx_valid":
            canon_mask_2d = torch.zeros_like(all_mask)  # (B, C+T, P) bool
            canon_mask_2d[:, :C] = context_mask[:, :C]
            canon_mask = canon_mask_2d.reshape(B, N)
        else:
            canon_mask = known_valid_flat

        canon_9d_flat, canon_params = self.canonicalizer.forward(
            all_9d_flat, canon_mask
        )

        # Override R_canon to identity when rotation canonicalization disabled (parity with forward)
        if not self.canon_rotation:
            canon_params = CanonParams(
                mean_pos=canon_params.mean_pos,
                scale=canon_params.scale,
                R_canon=torch.eye(3, device=device, dtype=canon_params.R_canon.dtype)
                    .unsqueeze(0).expand(B, -1, -1),
                bounded=canon_params.bounded,
                atanh_eps=canon_params.atanh_eps,
                ratio_inside_thr=canon_params.ratio_inside_thr,
                std_axis=canon_params.std_axis,
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

        # Scale pcl xyz by canonicalization scale (parity with forward).
        # (B, 3) per-axis scale broadcast over (B, P, PCL, 3) xyz last-dim.
        if local_pcl is not None:
            pcl_scale = canon_params.scale[:, None, None, :]  # (B, 1, 1, 3)
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
        pooled_lang_proj_inf = None
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

        # w1.0.2: input_xattn_v3 (inference path, runs once).
        # Surfaces pooled_lang_proj that every LMM call below must pass.
        if self.input_xattn_v3_enable:
            valid_entity_mask_inf = traj_padding_mask
            v3_out = self.input_xattn_v3(
                lang_proj=lang_proj,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=valid_entity_mask_inf,
                effective_K=self.input_xattn_v3.traj_id_bound_K,
            )
            feat = v3_out.feat_out
            learned_traj_ids_inf = v3_out.traj_ids
            pooled_lang_proj_inf = v3_out.pooled_lang_proj
            _B_inf = feat.shape[0]
            _D1_inf = int(self.lmm.D1)
            assert pooled_lang_proj_inf.shape == (_B_inf, _D1_inf), (
                f"expected pooled_lang_proj shape (B={_B_inf}, D1={_D1_inf}), "
                f"got {tuple(pooled_lang_proj_inf.shape)}"
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
            pooled_lang_proj_inf=pooled_lang_proj_inf,
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
        cur_frame_idx: Optional[torch.Tensor] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        return_intermediates: bool = False,
        cfg_scale: float = 1.0,
        scheduler_mode: str = "ddim_gen",
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

        # V13.3.4: use shared preparation helper
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
        pooled_lang_proj_inf = prep["pooled_lang_proj_inf"]
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

        # w1.0.2: for CFG uncond, recompute v3 on zero-lang_proj so
        # ``pooled_lang_proj_uncond`` receives the SAME literal-zero
        # substitution already applied to ``add_tokens`` via
        # ``lang_proj_uncond`` above. This mirrors the existing CFG uncond
        # pattern (zero substitution at the conditioning channel) and
        # ensures classifier-free guidance drops the language signal from
        # BOTH channels (add_tokens AND pooled_lang_proj).
        #
        # NOTE: training does not literally zero lang_proj at
        # text_masked_prob; ``lang_fc(T5(""))`` is non-zero. The literal-
        # zero substitution here is an approximation (same approximation
        # already used for ``add_tokens``) rather than training-parity.
        # ``feat`` passed in is post-v3 (prep dict ran v3 once with real
        # lang_proj), but Transformer 1 (the pooler) only reads lang_proj +
        # register tokens — feat is consumed by Transformer 2 which
        # produces feat_out that we discard. Only ``.pooled_lang_proj`` is
        # extracted. One recompute per inference call; shared across all
        # denoising steps.
        pooled_lang_proj_uncond = None
        if self.input_xattn_v3_enable and lang_proj_uncond is not None:
            v3_uncond_out = self.input_xattn_v3(
                lang_proj=lang_proj_uncond,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=traj_padding_mask,
                effective_K=self.input_xattn_v3.traj_id_bound_K,
            )
            pooled_lang_proj_uncond = v3_uncond_out.pooled_lang_proj

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
                    cur_frame_idx=cur_frame_idx,
                )
                if learned_traj_ids_inf is not None:
                    _mf_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                if pooled_lang_proj_inf is not None:
                    _mf_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_inf
                model_out = self.lmm(**_mf_lmm_kwargs)  # (B, C+T, P, 9) — avg velocity prediction

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_mf_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    if pooled_lang_proj_uncond is not None:
                        _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond
                    out_uncond = self.lmm(**_uncond_kwargs)
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
                    cur_frame_idx=cur_frame_idx,
                )
                if learned_traj_ids_inf is not None:
                    _fm_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                if pooled_lang_proj_inf is not None:
                    _fm_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_inf
                model_out = self.lmm(**_fm_lmm_kwargs)  # (B, C+T, P, 9)

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_fm_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    if pooled_lang_proj_uncond is not None:
                        _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond
                    out_uncond = self.lmm(**_uncond_kwargs)
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
            # scheduler_mode opt-in for full
            # DDPM=1000 generation. Default "ddim_gen" keeps the existing line
            # above byte-identical for all pre-existing callers.
            if scheduler_mode != "ddim_gen":
                scheduler = self.diffusion.get_gen_scheduler(scheduler_mode)
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
                    cur_frame_idx=cur_frame_idx,
                )
                if learned_traj_ids_inf is not None:
                    _ddpm_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_inf
                if pooled_lang_proj_inf is not None:
                    _ddpm_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_inf
                pred = self.lmm(**_ddpm_lmm_kwargs)

                # V13.3.5: Classifier-free guidance
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_ddpm_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    if pooled_lang_proj_uncond is not None:
                        _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond
                    pred_uncond = self.lmm(**_uncond_kwargs)
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

    def inference_with_grad(
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
        cur_frame_idx: Optional[torch.Tensor] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
        n_grad_steps: int = -1,
        verbose: bool = False,
        **kwargs,
    ):
        """DDIM/DDPM inference with autograd enabled — for MPC.

        Mirrors ``inference()`` for the DDPM/DDIM branch only (FM /
        MeanFlow not supported here; the WMM w1.0.x checkpoints used for
        LangTable MPC are DDPM with prediction_type=sample). Differences:

        - No ``@torch.inference_mode()`` decorator: gradients flow back
          to the input ``trace`` (and any other leaves with
          ``requires_grad=True``) through the canonicalizer + DDIM loop.
        - ``n_grad_steps`` (int, default -1):
            * -1     → every DDIM step is grad-enabled (most expensive).
            * 0      → no DDIM step keeps grad; output is detached.
            * N > 0  → only the LAST N DDIM steps are grad-enabled; the
              first ``len(timesteps) - N`` steps run under
              ``torch.no_grad()`` and the resulting ``x_t`` is detached
              before the grad-enabled tail begins. This is the standard
              "diffusion-policy MPC" cost-saving trick.
        - CFG, CHOIS, smoothing, and intermediate snapshots are all
          omitted: not needed for the LangTable MPC harness, and pruning
          them keeps this method auditable. Existing ``inference()``
          remains byte-identical.

        Returns:
            Tuple ``(ret_context, ret_trace)`` of SE3 tensors with
            gradients attached to whichever input leaves had
            ``requires_grad=True``.
        """
        if self.is_flow_matching or self.is_meanflow:
            raise NotImplementedError(
                "inference_with_grad: only DDPM/DDIM supported. Got "
                f"is_flow_matching={self.is_flow_matching}, "
                f"is_meanflow={self.is_meanflow}. Port DDIM grad path to "
                "FM/MF if needed."
            )

        is_training = self.training
        if is_training:
            self.eval()

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
        known_valid_flat = prep["known_valid_flat"]
        rot_9d_mask_flat = prep["rot_9d_mask_flat"]
        lang_proj = prep["lang_proj"]; lang_mask = prep["lang_mask"]
        img_proj = prep["img_proj"]; img_mask = prep["img_mask"]
        feat = prep["feat"]
        traj_padding_mask = prep["traj_padding_mask"]
        learned_traj_ids_inf = prep["learned_traj_ids_inf"]
        pooled_lang_proj_inf = prep["pooled_lang_proj_inf"]
        real_timestep = prep["real_timestep"]
        group_id = prep["group_id"]

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise = torch.randn(B, N, 9, device=device, generator=g)
        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(noise),
            canon_9d_flat,
            noise,
        )
        x_t = x_t * rot_9d_mask_flat

        scheduler = self.diffusion.gen_scheduler
        if n_steps is not None:
            scheduler.set_timesteps(n_steps)
        ddim_steps = list(scheduler.timesteps)
        total_ddim = len(ddim_steps)
        if n_grad_steps < 0:
            grad_start_idx = 0
        else:
            grad_start_idx = max(0, total_ddim - int(n_grad_steps))

        iterator = enumerate(ddim_steps)
        if verbose:
            iterator = tqdm(iterator, total=total_ddim, desc="DDIM grad")

        def _ddim_step(x_t_in, t_int):
            x_t_local = torch.where(
                known_valid_flat.unsqueeze(-1).expand_as(x_t_in),
                canon_9d_flat,
                x_t_in,
            )
            x_t_local = x_t_local * rot_9d_mask_flat

            t_tensor = torch.full((B, N), int(t_int), device=device, dtype=torch.long)
            t_tensor[known_valid_flat] = 0
            diff_levels = t_tensor.float().reshape(B, C + T, P)
            lmm_input_flat = self._build_v11_input(x_t_local)
            lmm_input = lmm_input_flat.reshape(B, C + T, P, 9)

            attn_mask_t = traj_padding_mask[:, None].expand(-1, T, -1)
            attn_mask_c = traj_padding_mask[:, None].expand(-1, C, -1)
            _kwargs = dict(
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
                cur_frame_idx=cur_frame_idx,
            )
            if learned_traj_ids_inf is not None:
                _kwargs["learned_traj_ids"] = learned_traj_ids_inf
            if pooled_lang_proj_inf is not None:
                _kwargs["pooled_lang_proj"] = pooled_lang_proj_inf
            pred = self.lmm(**_kwargs)

            _pred_type = self.diffusion.prediction_type
            if _pred_type == "sample":
                pass
            else:
                _EPS = 1e-6
                alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                alpha_val = alphas_cp[int(t_int)]
                sqrt_alpha = safe_sqrt(alpha_val)
                sqrt_1m_alpha = safe_sqrt(1.0 - alpha_val).clamp(min=_EPS)
                x_t_reshaped = x_t_local.reshape(B, C + T, P, 9).float()
                if _pred_type == "epsilon":
                    pred = (x_t_reshaped - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                elif _pred_type == "v_prediction":
                    pred = (sqrt_alpha * x_t_reshaped - pred.float()) / sqrt_1m_alpha
                else:
                    raise ValueError(f"Unsupported prediction_type: {_pred_type}")
                pred = pred.to(x_t_local.dtype)

            pred_flat = pred.reshape(B * N, 9)
            prev_x = scheduler.step(
                pred_flat, int(t_int), x_t_local.reshape(B * N, 9)
            ).prev_sample.reshape(B, N, 9)

            x_t_out = torch.where(
                known_valid_flat.unsqueeze(-1).expand_as(x_t_local),
                x_t_local,
                prev_x,
            )
            return x_t_out

        for ddim_idx, t_int in iterator:
            if ddim_idx < grad_start_idx:
                with torch.no_grad():
                    x_t = _ddim_step(x_t, t_int)
                x_t = x_t.detach()
            else:
                x_t = _ddim_step(x_t, t_int)

        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(x_t),
            canon_9d_flat,
            x_t,
        )
        if self.rot_param == "6d_wing":
            x_t = self._wing_to_canon_9d(x_t)

        x_orig = self.canonicalizer.inverse(x_t, canon_params)
        x_orig = x_orig.reshape(B, C + T, P, 9)
        result_se3 = ninep_to_se3(x_orig)

        ret_context = result_se3[:, :C]
        ret_trace = result_se3[:, C:]

        if is_training:
            self.train()
        return ret_context, ret_trace

    @torch.inference_mode()
    def inference_sliding_window(
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
        cfg_scale: float = 1.0,
        scheduler_mode: str = "ddim_gen",
        K_train: int = 128,
        anchor_count: Optional[int] = None,
        cur_frame_idx: Optional[torch.Tensor] = None,
        _debug_record: Optional[list] = None,
        **kwargs,
    ):
        """K_train -> P sliding-window diffusion-forcing DDPM inference.

         Implements the two-level
        diffusion-forcing pool described as follows:

        - The eval batch has P trajectories. The model was trained on
          windows of K_train trajectories. We denoise the full P-pool by
          repeatedly running the model on K_train-sized sub-batches drawn
          from the pool.
        - Two-level invariant: at every sub-step within an outer iter, the
          K_train tokens are split into (a) anchors at level ``t-1`` and
          (b) fresh tokens at level ``t``. Known/ctx tokens are level 0.
          Three levels NEVER coexist within the pool.
        - One outer iter advances every trajectory in the pool from level
          ``t`` to level ``t-1``. ``S_t`` tracks which trajectories have
          already advanced this outer iter; the iter ends only when
          ``|S_t| = P``.

        Args mirror :meth:`inference`. Additional kwargs:
            K_train: Trained window width. Required.
            anchor_count: Number of anchor tokens per non-first sub-step.
                Defaults to ``K_train // 2``.
            _debug_record: Optional list. When provided, the method
                appends a per-sub-step dict for unit-test inspection
                (see contract block below).

        Returns:
            Tuple ``(ret_context, ret_trace)`` matching :meth:`inference`.

        Design notes:

        * **Conditioning runs once over full P**, not per sub-batch. The
          upstream ``_prepare_inference`` computes ``feat``,
          ``pooled_lang_proj``, ``learned_traj_ids``, ``img_proj``,
          ``lang_proj`` over all P=400 entities; the per-sub-batch path
          then ``gather``s the per-entity rows. Training saw windows of
          K_train entities co-attending in v3, so eval-time full-P v3
          attention is a train/eval distribution shift on the v3 feature
          path. This is intentional ("Reuse
          ``self._prepare_inference(...)``"): re-running v3 per
          sub-batch would (a) make the per-entity conditioning depend
          on the random sub-batch composition (breaking determinism
          across sub-steps) and (b) recompute the language pool every
          sub-step at non-trivial cost. The trade-off is accepted —
          callers needing strict K_train-window v3 semantics must
          replicate the sub-batch pattern at the v3 layer too, which is
          out of scope here.
        * **Final outer iter: anchor diff_levels collapse to 0.** When
          the scheduler steps to ``t_minus_1 == 0`` (the last outer
          iter), anchor positions are labeled ``diff_levels=0`` —
          numerically indistinguishable from known/ctx clean
          conditioning, even though anchor positions hold partially-
          denoised trajectory content. Strategy (i) ``overwrite-back``
          (see below) salvages the OUTPUT (anchors keep their
          unchanged input value regardless of LMM prediction), but the
          fresh tokens in the same sub-batch consume cross-attention
          from these mislabeled anchors. The two-level invariant
          ``{0, t, t-1}`` collapses to ``{0, t}`` for the final iter.
          The dispatch spec accepts this collapse (last-iter target IS
          level 0 by definition); flagged as latent risk for downstream
          analysis.
        """
        is_training = self.training
        if is_training:
            self.eval()

        if K_train is None or int(K_train) <= 0:
            raise ValueError(
                f"inference_sliding_window: K_train must be a positive int, got {K_train!r}"
            )
        K_train = int(K_train)
        if anchor_count is None:
            anchor_count = K_train // 2
        anchor_count = int(anchor_count)
        if anchor_count < 0 or anchor_count >= K_train:
            raise ValueError(
                f"inference_sliding_window: anchor_count must be in [0, K_train), "
                f"got anchor_count={anchor_count}, K_train={K_train}"
            )

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
        known_valid_flat = prep["known_valid_flat"]
        rot_9d_mask_flat = prep["rot_9d_mask_flat"]
        lang_proj = prep["lang_proj"]; lang_mask = prep["lang_mask"]
        img_proj = prep["img_proj"]; img_mask = prep["img_mask"]
        feat = prep["feat"]
        traj_padding_mask = prep["traj_padding_mask"]
        learned_traj_ids_inf = prep["learned_traj_ids_inf"]
        pooled_lang_proj_inf = prep["pooled_lang_proj_inf"]
        real_timestep = prep["real_timestep"]
        group_id = prep["group_id"]

        if K_train > P:
            raise ValueError(
                f"inference_sliding_window: K_train ({K_train}) must be <= P ({P})"
            )

        CT = C + T  # tokens per trajectory

        # Initialize from noise (same as inference())
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        noise = torch.randn(B, N, 9, device=device, generator=g)
        x_t = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(noise),
            canon_9d_flat,
            noise,
        )
        x_t = x_t * rot_9d_mask_flat

        # Reshape per-P views once. We will use per-P shape (B, CT, P, ...)
        # for trajectory indexing and (B, K*CT, ...) for LMM/scheduler calls.
        x_t = x_t.reshape(B, CT, P, 9)
        canon_9d_per = canon_9d_flat.reshape(B, CT, P, 9)
        known_valid_per = known_valid_flat.reshape(B, CT, P)
        rot_9d_mask_per = rot_9d_mask_flat.reshape(B, CT, P, 9)

        # CFG uncond (mirror inference()).
        lang_proj_uncond = torch.zeros_like(lang_proj) if cfg_scale != 1.0 else None
        pooled_lang_proj_uncond = None
        if self.input_xattn_v3_enable and lang_proj_uncond is not None:
            v3_uncond_out = self.input_xattn_v3(
                lang_proj=lang_proj_uncond,
                lang_mask=lang_mask,
                img_proj=img_proj,
                img_mask=img_mask,
                feat=feat,
                valid_entity_mask=traj_padding_mask,
                effective_K=self.input_xattn_v3.traj_id_bound_K,
            )
            pooled_lang_proj_uncond = v3_uncond_out.pooled_lang_proj

        # DDPM scheduler.
        scheduler = self.diffusion.gen_scheduler
        if scheduler_mode != "ddim_gen":
            scheduler = self.diffusion.get_gen_scheduler(scheduler_mode)
        if n_steps is not None:
            scheduler.set_timesteps(n_steps)
        ddim_steps = list(scheduler.timesteps)
        total_ddim = len(ddim_steps)
        outer_iter = enumerate(ddim_steps)
        if verbose:
            outer_iter = tqdm(outer_iter, total=total_ddim,
                              desc="SlidingWindow DDIM inference")

        # iter 02: stable_ids_per disabled — testing real_timestep fix alone.
        # Iter 01 (with stable_ids_per active) still showed track collapse at
        # step 2000, so the SW stable-id mechanism is not sufficient. Reverted
        # to the legacy per-sub-step _sample_traj_ids path.
        stable_ids_per = None

        # Sub-batch indexing helper. ``sub_indices``: (B, K_train) long.
        def _gather_per_traj(per_traj: torch.Tensor, sub_idx: torch.Tensor) -> torch.Tensor:
            """Gather per-trajectory tensor along P axis using sub_idx (B,K)."""
            # per_traj shape: (B, ..., P) or (B, ..., P, D)
            shape = per_traj.shape
            assert shape[0] == B
            # Find the P axis -- it must be position-keyed. We require the
            # caller to pass tensors whose third axis is P (matches the
            # standard (B, CT, P, ...) layout used in this method).
            # We index along axis=2.
            P_axis = 2
            assert shape[P_axis] == P, (
                f"_gather_per_traj: expected P at axis {P_axis}, got {shape}"
            )
            # Build broadcastable index of shape (B, *ones, K, *ones)
            view = [1] * len(shape)
            view[0] = B
            view[P_axis] = K_train
            idx = sub_idx.view(B, 1, K_train, *([1] * (len(shape) - 3)))
            # Expand non-(B, P_axis) dims to match per_traj.
            expand_shape = list(shape)
            expand_shape[P_axis] = K_train
            idx = idx.expand(expand_shape)
            return per_traj.gather(P_axis, idx)

        for ddim_idx, t_int in outer_iter:
            t_int = int(t_int)
            # t-1 in DDPM scheduler units.
            if ddim_idx + 1 < total_ddim:
                t_minus_1 = int(ddim_steps[ddim_idx + 1])
            else:
                # Final outer iter: target is 0.
                t_minus_1 = 0

            # S_t tracks per-batch which trajectories already at t-1.
            # Per-batch boolean mask of shape (B, P).
            S_t_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

            sub_step = 0
            while True:
                fresh_mask = ~S_t_mask  # (B, P)
                fresh_remaining = fresh_mask.sum(dim=1)  # (B,)
                # All B elements must finish in lock-step (same number of
                # outer-iter sub-steps) so the sub-batches stay aligned in
                # shape. fresh_remaining is identical across B in practice
                # because the per-step (A, K_fresh) split is identical
                # across B; we still take the min to be safe.
                min_remaining = int(fresh_remaining.min().item())
                if min_remaining == 0:
                    break

                if sub_step == 0:
                    A = 0
                    K_fresh = K_train
                else:
                    if min_remaining < anchor_count:
                        # Last sub-step: shrink fresh, grow anchors.
                        K_fresh = min_remaining
                        A = K_train - K_fresh
                    else:
                        A = anchor_count
                        K_fresh = K_train - A

                # Per-batch random selection. Use a generator seeded by
                # (seed, ddim_idx, sub_step) for determinism.
                step_gen = torch.Generator(device=device)
                step_gen.manual_seed(
                    int(seed) + 1009 * int(ddim_idx) + 31 * int(sub_step)
                )

                fresh_indices_list = []
                anchor_indices_list = []
                for b in range(B):
                    fresh_pool = torch.nonzero(fresh_mask[b], as_tuple=False).flatten()
                    anchor_pool = torch.nonzero(S_t_mask[b], as_tuple=False).flatten()
                    # Fresh: sample K_fresh WITHOUT replacement from fresh_pool.
                    n_fresh_pool = int(fresh_pool.numel())
                    if n_fresh_pool < K_fresh:
                        raise RuntimeError(
                            f"inference_sliding_window: fresh pool too small "
                            f"({n_fresh_pool}) for K_fresh={K_fresh} at sub_step={sub_step}"
                        )
                    perm = torch.randperm(n_fresh_pool, generator=step_gen, device=device)
                    fresh_b = fresh_pool[perm[:K_fresh]]
                    fresh_indices_list.append(fresh_b)
                    # Anchors: sample A WITH replacement from anchor_pool.
                    if A > 0:
                        n_anchor_pool = int(anchor_pool.numel())
                        if n_anchor_pool == 0:
                            raise RuntimeError(
                                f"inference_sliding_window: anchor pool empty at "
                                f"sub_step={sub_step}, but A={A}"
                            )
                        anchor_pick = torch.randint(
                            low=0, high=n_anchor_pool,
                            size=(A,), generator=step_gen, device=device,
                        )
                        anchor_b = anchor_pool[anchor_pick]
                    else:
                        anchor_b = torch.empty(0, dtype=torch.long, device=device)
                    anchor_indices_list.append(anchor_b)

                fresh_indices = torch.stack(fresh_indices_list, dim=0)   # (B, K_fresh)
                if A > 0:
                    anchor_indices = torch.stack(anchor_indices_list, dim=0)  # (B, A)
                    # Concat: fresh first, anchors after.
                    sub_indices = torch.cat([fresh_indices, anchor_indices], dim=1)  # (B, K_train)
                else:
                    anchor_indices = torch.empty(B, 0, dtype=torch.long, device=device)
                    sub_indices = fresh_indices

                # w1.4.0: gather cur_frame_idx for the K_train sub-batch.
                # cur_frame_idx is (B, P) in T-space; gather along P axis.
                cur_frame_idx_sub = None
                if cur_frame_idx is not None:
                    cur_frame_idx_sub = cur_frame_idx.gather(1, sub_indices)  # (B, K_train)

                # Gather sub-batch tensors along P axis.
                # x_t: (B, CT, P, 9) -> (B, CT, K_train, 9)
                x_t_sub = _gather_per_traj(x_t, sub_indices)
                canon_sub = _gather_per_traj(canon_9d_per, sub_indices)
                known_valid_sub = _gather_per_traj(known_valid_per, sub_indices)
                rot_9d_mask_sub = _gather_per_traj(rot_9d_mask_per, sub_indices)
                # feat: (B, P, F) -> (B, K_train, F).
                F_dim = feat.shape[-1]
                feat_sub = feat.gather(
                    1, sub_indices.unsqueeze(-1).expand(B, K_train, F_dim),
                )
                # group_id: (B, P) -> (B, K_train).
                group_id_sub = group_id.gather(1, sub_indices)
                # traj_padding_mask: (B, P) -> (B, K_train).
                traj_padding_mask_sub = traj_padding_mask.gather(1, sub_indices)
                # learned_traj_ids: (B, P) float optional -> (B, K_train).
                learned_traj_ids_sub = None
                if learned_traj_ids_inf is not None:
                    if learned_traj_ids_inf.dim() == 2:
                        learned_traj_ids_sub = learned_traj_ids_inf.gather(1, sub_indices)
                    elif learned_traj_ids_inf.dim() == 3:
                        D_id = learned_traj_ids_inf.shape[-1]
                        learned_traj_ids_sub = learned_traj_ids_inf.gather(
                            1, sub_indices.unsqueeze(-1).expand(B, K_train, D_id),
                        )
                    else:
                        raise ValueError(
                            f"inference_sliding_window: unexpected learned_traj_ids "
                            f"shape {tuple(learned_traj_ids_inf.shape)}"
                        )
                elif stable_ids_per is not None:
                    # w1.5.0 k128 fix: gather the pre-sampled stable ids for
                    # this sub-batch's global indices. Same global trajectory
                    # always sees the same id across all sub-steps/outer iters.
                    learned_traj_ids_sub = stable_ids_per.gather(1, sub_indices)

                # Re-clamp known tokens in the sub-batch to canonical values
                # (mirrors inference()). known_valid_sub shape: (B, CT, K_train).
                x_t_sub = torch.where(
                    known_valid_sub.unsqueeze(-1).expand_as(x_t_sub),
                    canon_sub,
                    x_t_sub,
                )
                # Zero rotation for position-only entities in sub-batch.
                x_t_sub = x_t_sub * rot_9d_mask_sub
                # Snapshot input value for anchor overwrite-back later.
                x_t_sub_input = x_t_sub.clone()

                # Build per-token diff_levels: (B, CT, K_train).
                #   - known tokens -> 0
                #   - anchor positions (last A trajectories) -> t-1
                #   - fresh positions (first K_fresh trajectories) -> t
                t_levels = torch.full(
                    (B, CT, K_train), float(t_int), device=device, dtype=torch.float32,
                )
                if A > 0:
                    t_levels[:, :, K_fresh:] = float(t_minus_1)
                # Known tokens -> 0 wherever they are in the sub-batch.
                t_levels = torch.where(
                    known_valid_sub, torch.zeros_like(t_levels), t_levels,
                )
                diff_levels_sub = t_levels  # already (B, CT, K_train)

                # Build LMM input — flat (B, K_train*CT, 9), then reshape to
                # (B, CT, K_train, 9) via _build_v11_input convention.
                # _build_v11_input expects flat (B, N', 9) where N' = K*CT.
                # We pack along (CT, K_train) such that .reshape(B, CT, K_train, 9)
                # roundtrips. The convention used in inference() is:
                #   flat = per.reshape(B, CT*P, 9)
                # so we mirror it with K_train in place of P.
                x_t_sub_flat = x_t_sub.reshape(B, CT * K_train, 9)
                lmm_input_flat = self._build_v11_input(x_t_sub_flat)
                lmm_input = lmm_input_flat.reshape(B, CT, K_train, 9)

                attn_mask_t = traj_padding_mask_sub[:, None].expand(-1, T, -1)
                attn_mask_c = traj_padding_mask_sub[:, None].expand(-1, C, -1)

                _ddpm_lmm_kwargs = dict(
                    x=lmm_input[:, C:],
                    x_mask=attn_mask_t,
                    x_time=real_timestep,
                    x_level=diff_levels_sub[:, C:],
                    feat=feat_sub,
                    ctx=lmm_input[:, :C],
                    ctx_mask=attn_mask_c,
                    ctx_level=diff_levels_sub[:, :C],
                    add_tokens=lang_proj.clone(),
                    add_tokens_mask=lang_mask.clone(),
                    image_tokens=img_proj.clone() if img_proj is not None else None,
                    image_tokens_mask=img_mask.clone() if img_mask is not None else None,
                    group_ids=group_id_sub,
                    cur_frame_idx=cur_frame_idx_sub,
                )
                if learned_traj_ids_sub is not None:
                    _ddpm_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_sub
                if pooled_lang_proj_inf is not None:
                    _ddpm_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_inf
                pred = self.lmm(**_ddpm_lmm_kwargs)  # (B, CT, K_train, 9)

                # CFG.
                if cfg_scale != 1.0:
                    _uncond_kwargs = dict(_ddpm_lmm_kwargs)
                    _uncond_kwargs["add_tokens"] = lang_proj_uncond.clone()
                    if pooled_lang_proj_uncond is not None:
                        _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond
                    pred_uncond = self.lmm(**_uncond_kwargs)
                    pred = pred_uncond + cfg_scale * (pred - pred_uncond)

                # V11 LMM outputs x0. Convert to scheduler's prediction_type.
                _pred_type = self.diffusion.prediction_type
                if _pred_type == "sample":
                    pass
                else:
                    _EPS = 1e-6
                    alphas_cp = self.diffusion.noise_scheduler.alphas_cumprod.to(device).float()
                    alpha_val = alphas_cp[int(t_int)]
                    sqrt_alpha = safe_sqrt(alpha_val)
                    sqrt_1m_alpha = safe_sqrt(1.0 - alpha_val).clamp(min=_EPS)
                    x_t_sub_reshaped = x_t_sub.float()
                    if _pred_type == "epsilon":
                        pred = (x_t_sub_reshaped - sqrt_alpha * pred.float()) / sqrt_1m_alpha
                    elif _pred_type == "v_prediction":
                        pred = (sqrt_alpha * x_t_sub_reshaped - pred.float()) / sqrt_1m_alpha
                    else:
                        raise ValueError(f"Unsupported prediction_type: {_pred_type}")
                    pred = pred.to(x_t_sub.dtype)

                # scheduler.step on the full sub-batch flat tensor.
                N_sub = CT * K_train
                pred_flat = pred.reshape(B * N_sub, 9)
                prev_sub_flat = scheduler.step(
                    pred_flat, int(t_int), x_t_sub.reshape(B * N_sub, 9)
                ).prev_sample
                prev_sub = prev_sub_flat.reshape(B, CT, K_train, 9)

                # Overwrite anchor positions with their unchanged input value.
                # Anchors are the last A entries in K_train axis (axis 2).
                if A > 0:
                    prev_sub[:, :, K_fresh:, :] = x_t_sub_input[:, :, K_fresh:, :]

                # Re-clamp known tokens in prev_sub to canonical values.
                prev_sub = torch.where(
                    known_valid_sub.unsqueeze(-1).expand_as(prev_sub),
                    canon_sub,
                    prev_sub,
                )

                # Scatter-back ONLY the K_fresh fresh slice into x_t along P.
                # fresh_indices: (B, K_fresh). prev_sub fresh slice:
                # (B, CT, K_fresh, 9).
                fresh_slice = prev_sub[:, :, :K_fresh, :].contiguous()
                # scatter: x_t (B, CT, P, 9) at axis=2 with fresh_indices.
                idx_for_scatter = fresh_indices.view(B, 1, K_fresh, 1).expand(
                    B, CT, K_fresh, 9,
                )
                x_t.scatter_(2, idx_for_scatter, fresh_slice)

                # Update S_t: mark fresh trajectories as "advanced this iter".
                # fresh_indices: (B, K_fresh).
                S_t_mask.scatter_(
                    1, fresh_indices, torch.ones_like(fresh_indices, dtype=torch.bool),
                )

                # Two-level invariant assertion (cheap per-token check).
                # diff_levels_sub values are in {0, t, t-1} only.
                with torch.no_grad():
                    levels_unique = torch.unique(diff_levels_sub)
                    allowed = {0.0, float(t_int), float(t_minus_1)}
                    bad = [
                        float(v.item()) for v in levels_unique
                        if float(v.item()) not in allowed
                    ]
                    if bad:
                        raise AssertionError(
                            f"inference_sliding_window: two-level invariant violated "
                            f"at outer_iter={ddim_idx} sub_step={sub_step}: found "
                            f"unexpected level values {bad}, allowed={sorted(allowed)}"
                        )

                if _debug_record is not None:
                    _debug_record.append({
                        "outer_iter": ddim_idx,
                        "sub_step": sub_step,
                        "t": int(t_int),
                        "t_minus_1": int(t_minus_1),
                        "fresh_indices": fresh_indices.detach().cpu(),
                        "anchor_indices": anchor_indices.detach().cpu(),
                        "diff_levels": diff_levels_sub.detach().cpu(),
                        "anchor_input_values": (
                            x_t_sub_input[:, :, K_fresh:, :].detach().cpu()
                            if A > 0 else None
                        ),
                        "anchor_output_values": (
                            prev_sub[:, :, K_fresh:, :].detach().cpu()
                            if A > 0 else None
                        ),
                        "S_t_size_after": int(S_t_mask.sum(dim=1).min().item()),
                        "P": int(P),
                    })

                sub_step += 1

            # End of outer iter — every trajectory must have advanced.
            assert int(S_t_mask.all(dim=1).all().item()) == 1, (
                f"inference_sliding_window: outer_iter={ddim_idx} ended with "
                f"|S_t| < P (per-batch advanced count: {S_t_mask.sum(dim=1).tolist()})"
            )

        # Final clamp + decanonicalize, mirroring inference().
        x_t_flat = x_t.reshape(B, N, 9)
        x_t_flat = torch.where(
            known_valid_flat.unsqueeze(-1).expand_as(x_t_flat),
            canon_9d_flat,
            x_t_flat,
        )
        if self.rot_param == "6d_wing":
            x_t_flat = self._wing_to_canon_9d(x_t_flat)
        x_orig = self.canonicalizer.inverse(x_t_flat, canon_params)
        x_orig = x_orig.reshape(B, C + T, P, 9)
        result_se3 = ninep_to_se3(x_orig)  # (B, C+T, P, 4, 4)
        ret_context = result_se3[:, :C]
        ret_trace = result_se3[:, C:]

        if is_training:
            self.train()

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
        scheduler_mode: str = "ddim_gen",
        cur_frame_idx: Optional[torch.Tensor] = None,    # w1.4.0: (B, P) long, T-space [0, T)
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
        pooled_lang_proj_s1 = prep_s1["pooled_lang_proj_inf"]
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

        # w1.0.2: recompute v3 pooled_lang for stage-1 CFG uncond.
        # Same rationale as ``inference()``: literal-zero substitution at the
        # pooled_lang channel mirroring the existing ``add_tokens`` uncond
        # pattern. ``feat_s1`` is post-v3 from prep_s1; only
        # ``.pooled_lang_proj`` is extracted (Transformer 1 is feat-
        # independent).
        pooled_lang_proj_uncond_s1 = None
        if self.input_xattn_v3_enable and lang_proj_uncond_s1 is not None:
            v3_uncond_out_s1 = self.input_xattn_v3(
                lang_proj=lang_proj_uncond_s1,
                lang_mask=lang_mask_s1,
                img_proj=img_proj_s1,
                img_mask=img_mask_s1,
                feat=feat_s1,
                valid_entity_mask=traj_padding_mask_s1,
                effective_K=self.input_xattn_v3.traj_id_bound_K,
            )
            pooled_lang_proj_uncond_s1 = v3_uncond_out_s1.pooled_lang_proj

        # Build unknown_x_mask for stage 1 (ctx cannot see unknown trace X)
        unknown_x_mask_s1 = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        unknown_x_mask_s1[:, C:] = all_mask_s1[:, C:] & ~all_known_s1[:, C:]

        # Stage 1 DDPM loop
        scheduler = self.diffusion.gen_scheduler
        # scheduler_mode opt-in for full
        # DDPM=1000 generation. Default "ddim_gen" is byte-identical.
        if scheduler_mode != "ddim_gen":
            scheduler = self.diffusion.get_gen_scheduler(scheduler_mode)
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
                cur_frame_idx=cur_frame_idx,
            )
            if learned_traj_ids_s1 is not None:
                _s1_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_s1
            if pooled_lang_proj_s1 is not None:
                _s1_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_s1
            pred = self.lmm(**_s1_lmm_kwargs)

            # V13.3.5: Classifier-free guidance
            if cfg_scale != 1.0:
                _uncond_kwargs = dict(_s1_lmm_kwargs)
                _uncond_kwargs["add_tokens"] = lang_proj_uncond_s1.clone()
                if pooled_lang_proj_uncond_s1 is not None:
                    _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond_s1
                pred_uncond = self.lmm(**_uncond_kwargs)
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
        pooled_lang_proj_s2 = prep_s2["pooled_lang_proj_inf"]
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

        # w1.0.2: recompute v3 pooled_lang for stage-2 CFG uncond.
        # ``feat_s2`` is post-v3 from prep_s2 (which re-ran v3 on the s2
        # context); only ``.pooled_lang_proj`` is extracted. Same literal-
        # zero-substitution rationale as ``inference()``.
        pooled_lang_proj_uncond_s2 = None
        if self.input_xattn_v3_enable and lang_proj_uncond_s2 is not None:
            v3_uncond_out_s2 = self.input_xattn_v3(
                lang_proj=lang_proj_uncond_s2,
                lang_mask=lang_mask_s2,
                img_proj=img_proj_s2,
                img_mask=img_mask_s2,
                feat=feat_s2,
                valid_entity_mask=traj_padding_mask_s2,
                effective_K=self.input_xattn_v3.traj_id_bound_K,
            )
            pooled_lang_proj_uncond_s2 = v3_uncond_out_s2.pooled_lang_proj

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
                cur_frame_idx=cur_frame_idx,
            )
            if learned_traj_ids_s2 is not None:
                _s2_lmm_kwargs["learned_traj_ids"] = learned_traj_ids_s2
            if pooled_lang_proj_s2 is not None:
                _s2_lmm_kwargs["pooled_lang_proj"] = pooled_lang_proj_s2
            pred = self.lmm(**_s2_lmm_kwargs)

            # V13.3.5: Classifier-free guidance
            if cfg_scale != 1.0:
                _uncond_kwargs = dict(_s2_lmm_kwargs)
                _uncond_kwargs["add_tokens"] = lang_proj_uncond_s2.clone()
                if pooled_lang_proj_uncond_s2 is not None:
                    _uncond_kwargs["pooled_lang_proj"] = pooled_lang_proj_uncond_s2
                pred_uncond = self.lmm(**_uncond_kwargs)
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
        # w1.0.2: input_xattn_v3 params share input_layer LR.
        if self.input_xattn_v3_enable and self.input_xattn_v3 is not None:
            agent_params += list(self.input_xattn_v3.parameters())
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
