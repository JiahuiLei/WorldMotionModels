"""AgentE11: Equivariant Sandwich Architecture for Trajectory Diffusion Forcing.

Architecture: [et, ep, t, p, t, p, t, dp, dt]
- et/ep: EquivariantEncoder (VN lift + learned registers + inner products -> invariants)
- t, p: Free (unconstrained) ScalarAxialBlocks with full AdaLN shift+scale
- dp/dt: EquivariantDecoder (scalar weights x equivariant registers -> equivariant output)

SIM(3) equivariance through:
1. SEPARATION: split 9D [p0,p1,p2] into position p0 and rotation offsets delta_i = p_i - p0
2. CANONICALIZATION: center + scale-normalize p0 only (deltas are SO(3)-covariant)
3. SANDWICH: equivariant boundary (VN) + free middle (standard transformer)
4. DE-CANONICALIZATION: restore to world frame

Based on Kaba et al. (ICML 2023): equivariant boundary + unconstrained middle =
universal approximator of equivariant functions.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from umm.models.umm_e11.sandwich_layers import EquivariantEncoder, EquivariantDecoder
from umm.models.umm_e11.free_transformer import FreeTransformer
from umm.models.umm_e11.poseio import PoseIO_V3
from umm.models.umm_e11.loss_utils import type_mean_compiled as type_mean
from umm.models.umm_e11.df_helpers import compute_x0, compute_snr


class AgentE11(nn.Module):
    """Equivariant Sandwich trajectory diffusion forcing agent.

    Key design: canonicalize -> EquivariantEncoder -> FreeTransformer -> EquivariantDecoder -> de-canonicalize.

    Args:
        cfg: Full configuration dict.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        # * Loss config
        self.w_t_se3 = self.optim_cfg.get("w_t_se3", 1.0)
        self.w_t_xyz = self.optim_cfg.get("w_t_xyz", 1.0)
        self.w_q = self.optim_cfg.get("w_q", 1.0)
        self.w_R = self.optim_cfg.get("w_R", 0.0)
        self.w_virtual = self.optim_cfg.get("w_virtual", 0.1)
        self.w_real = self.optim_cfg.get("w_real", 1.0)
        self.loss_type = self.optim_cfg.get("loss_type", "norm")
        self.loss_square = self.optim_cfg.get("loss_square", True)
        self.loss_stable_beta = self.optim_cfg.get("loss_stable_beta", 0.1)
        self.prob_virtual_allknown = self.optim_cfg.get("prob_virtual_allknown", 0.0)
        self.prob_bias_to_known = self.optim_cfg.get("prob_bias_to_known", 0.0)
        self.ctx_always_known = self.optim_cfg.get("ctx_always_known", False)
        self.gid_w_modifier = {
            int(k): v for k, v in self.optim_cfg.get("gid_w_modifier", {}).items()
        }

        # * Viz-required attributes
        self.hand_attached_to_wrist_flag = self.model_cfg.get(
            "hand_attached_to_wrist_flag", False
        )

        # * PoseIO — MUST use zero mean, isotropic std for equivariance
        self.rot_len = self.model_cfg.get("rot_len", 0.5)
        self.pose_io_debug_flag = self.model_cfg.get("pose_io_debug_flag", False)
        param_mean = self.model_cfg.get("param_mean", [])
        param_std = self.model_cfg.get("param_std", [])
        self.pose_io = PoseIO_V3(
            sem_group_ids=self.model_cfg.get("param_sem_group_ids", [[-1]]),
            rot_len=self.rot_len,
            param_mean=param_mean,
            param_std=param_std,
            std_factor=self.model_cfg.get("std_factor", []),
            state_mean_std_flag=True,
            state_only_pos_flag=self.model_cfg.get("state_only_pos_flag", True),
        )

        # Warn if PoseIO params break equivariance
        for grp_mean in param_mean:
            if any(m != 0.0 for m in grp_mean):
                logging.warning(
                    "E11: Non-zero PoseIO param_mean %s will break SIM(3) equivariance", grp_mean
                )
        for grp_std in param_std:
            if len(set(grp_std)) > 1:
                logging.warning(
                    "E11: Non-isotropic PoseIO param_std %s will break SIM(3) equivariance", grp_std
                )

        # * Diffusion schedulers
        self.__init_diffusion__(self.model_cfg.get("diffusion", {}))

        # * Sandwich architecture
        sandwich_cfg = self.model_cfg.get("sandwich", {})
        d_vn = sandwich_cfg.get("d_vn", 32)
        d_model = sandwich_cfg.get("d_model", 384)
        n_heads = sandwich_cfg.get("n_heads", 8)
        n_layers = sandwich_cfg.get("n_layers", 6)
        d_cond = sandwich_cfg.get("d_cond", 256)
        ffn_factor = sandwich_cfg.get("ffn_factor", 4)

        # [H2] Cross-product invariant features
        cross_product_invariants = sandwich_cfg.get("cross_product_invariants", False)
        # [H1] Noise-gated skip connection
        self.skip_gate_by_noise = sandwich_cfg.get("skip_gate_by_noise", False)

        # Equivariant encoder boundary (et + ep)
        self.encoder = EquivariantEncoder(
            d_vn=d_vn,
            input_channels=3,  # 3 anchor points from PoseIO 9D
            cross_product_invariants=cross_product_invariants,
        )

        # Free (unconstrained) scalar transformer middle
        diff_cfg = self.model_cfg.get("diffusion", {})
        max_noise_ts = diff_cfg.get("train_diff_timesteps", 100)
        self.free_transformer = FreeTransformer(
            invariant_dim=self.encoder.invariant_dim,  # 4 or 6 * d_vn
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_cond=d_cond,
            ffn_factor=ffn_factor,
            qk_norm=sandwich_cfg.get("qk_norm", True),
            num_types=sandwich_cfg.get("num_types", 1024),
            num_groups=sandwich_cfg.get("num_groups", 1024),
            time_multiplier=sandwich_cfg.get("time_multiplier", 10.0),
            multi_layer_skip=sandwich_cfg.get("multi_layer_skip", False),
        )

        # Equivariant decoder boundary (dp + dt)
        self.decoder = EquivariantDecoder(
            d_model=d_model,
            d_vn=d_vn,
            output_channels=3,  # 3 anchor points for PoseIO 9D
            skip_gate_by_noise=self.skip_gate_by_noise,
            max_noise_timestep=max_noise_ts,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(
            f"AgentE11 initialized (sandwich architecture): "
            f"d_vn={d_vn}, d_model={d_model}, n_layers={n_layers}, "
            f"n_heads={n_heads}, d_cond={d_cond}, "
            f"invariant_dim={self.encoder.invariant_dim}, "
            f"{n_params:,} params"
        )

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def get_type_mask(uids: torch.Tensor, target_uids) -> torch.Tensor:
        """Return a boolean mask where uids match any of target_uids."""
        if isinstance(target_uids, (int, float)):
            target_uids = [int(target_uids)]
        mask = torch.zeros_like(uids, dtype=torch.bool)
        for uid in target_uids:
            mask |= uids == uid
        return mask

    def convert_smplx_hand_param(self, Q, to_local=True):
        """Convert SMPL-X hand joints between local and global wrist space."""
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
            Q[:, 25:40] = torch.einsum("bij,bnjk->bnik", l_T, Q[:, 25:40].clone())
            Q[:, 40:55] = torch.einsum("bij,bnjk->bnik", r_T, Q[:, 40:55].clone())
        return Q

    @staticmethod
    def _fill_zero_rotation_9d(
        param_9d: torch.Tensor,
        rot9_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fill position-only parts with zero-delta rotation (p1=p2=p0).

        For parts where rot_mask channels 3:9 are False, the zeroed rotation
        channels would create delta = 0 - p0 = -p0 in canonicalize_9d, which
        depends on absolute position and breaks equivariance.

        Fix: set p1 = p2 = p0, so deltas become [0, 0, 0]. Zero vectors are
        equivariant under all rotations: R @ 0 = 0. VN layers handle zero
        input safely (VNLayerNorm adds eps, no NaN).

        Args:
            param_9d: [..., 9] PoseIO params
            rot9_mask: [..., 9] bool mask (True=keep, False=position-only)
        """
        needs_fill = ~rot9_mask[..., 3]  # [...] parts with no rotation
        if not needs_fill.any():
            return param_9d

        param_9d = param_9d.clone()
        p0 = param_9d[..., :3]  # [..., 3]

        # Set p1 = p0 and p2 = p0 so delta = p_i - p0 = 0 (equivariant)
        param_9d[..., 3:6] = torch.where(
            needs_fill.unsqueeze(-1), p0, param_9d[..., 3:6]
        )
        param_9d[..., 6:9] = torch.where(
            needs_fill.unsqueeze(-1), p0, param_9d[..., 6:9]
        )
        return param_9d

    # -----------------------------------------------------------------
    # Canonicalization (identical to E10 — proven correct)
    # -----------------------------------------------------------------

    @staticmethod
    @torch.compiler.disable
    def canonicalize_9d(param_9d: torch.Tensor, mask: torch.Tensor):
        """Canonicalize 9D: separate position from rotation offsets, normalize position only.

        9D = [p0, p1, p2] where p0 = position, p_i = p0 + R[:,i]*rl.
        Under SIM(3) g = (Rg, tg, sg):
          p0    -> sg*Rg@p0 + tg    (full SIM(3))
          delta_i = p_i - p0 -> Rg@delta_i  (SO(3) only, NO scaling)

        After canonicalization, all 3 channels are pure SO(3)-covariant.

        Args:
            param_9d: [B, S, 9] PoseIO params
            mask: [B, S] bool, True = valid

        Returns:
            param_canon: [B, S, 9] = [p0_canon, delta1, delta2]
            mu: [B, 3] centroid of p0
            scale: [B] RMS scale of centered p0
        """
        B, S, _ = param_9d.shape
        mask_float = mask.float().unsqueeze(-1)  # [B, S, 1]
        count = mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]

        # Separate position from rotation offsets
        p0 = param_9d[..., :3]  # [B, S, 3] — position
        delta1 = param_9d[..., 3:6] - p0  # [B, S, 3] — R[:,0]*rl (SO(3)-covariant)
        delta2 = param_9d[..., 6:9] - p0  # [B, S, 3] — R[:,1]*rl (SO(3)-covariant)

        # Centroid from p0 only
        mu = (p0 * mask_float).sum(dim=1, keepdim=True) / count  # [B, 1, 3]

        # Center p0
        p0c = p0 - mu  # [B, S, 3]

        # Scale from p0 RMS
        sq_norms = (p0c * mask_float).pow(2).sum(dim=-1)  # [B, S]
        rms = (sq_norms.sum(dim=1) / count[:, 0, 0]).sqrt()  # [B]
        scale = rms.clamp(min=1e-6)  # [B]
        s = scale[:, None, None]  # [B, 1, 1]

        # Position gets centered + scaled; rotation offsets pass through
        param_canon = torch.cat([p0c / s, delta1, delta2], dim=-1)
        mu = mu.squeeze(1)  # [B, 3]

        return param_canon, mu, scale

    @staticmethod
    def de_canonicalize_9d(param_canon: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor):
        """Inverse of canonicalize_9d.

        param_canon = [p0_canon, delta1, delta2].
        Reconstruct: p0 = p0_canon*scale + mu, p_i = p0 + delta_i.
        """
        s = scale[:, None, None]
        m = mu[:, None, :]  # [B, 1, 3]
        p0 = param_canon[..., :3] * s + m
        p1 = p0 + param_canon[..., 3:6]  # delta1 added to world p0
        p2 = p0 + param_canon[..., 6:9]  # delta2 added to world p0
        return torch.cat([p0, p1, p2], dim=-1)

    # -----------------------------------------------------------------
    # Diffusion setup
    # -----------------------------------------------------------------

    def __init_diffusion__(self, cfg):
        self.pred_type = cfg.get("prediction_type", "epsilon")
        assert self.pred_type == "sample", (
            f"E11 equivariant agent only supports prediction_type='sample' "
            f"(de-canonicalization adds scale+offset, which corrupts predicted noise/velocity). "
            f"Got: {self.pred_type}"
        )
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

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def loss_kernel(self, pred, gt):
        _EPS = 1e-8
        if self.loss_type == "norm":
            sq_norm = (pred - gt).pow(2).sum(dim=-1)
            diff = (sq_norm + _EPS).sqrt()
            if self.loss_square:
                diff = diff ** 2
            return diff
        elif self.loss_type == "stable":
            diff = pred - gt
            sq_norm = diff.pow(2).sum(dim=-1)
            residual_norm = (sq_norm + _EPS).sqrt()
            beta = self.loss_stable_beta
            return torch.where(
                residual_norm > beta,
                residual_norm - 0.5 * beta,
                0.5 * sq_norm / beta,
            )
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

    # -----------------------------------------------------------------
    # Core equivariant forward pass (THE E-TEST TARGET)
    # -----------------------------------------------------------------

    @torch.compiler.disable
    def _prepare_df_step(self, B, C, T, P, device):
        df_timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B, C + T, P), device=device,
        ).long()
        if self.prob_virtual_allknown > 0.0 and torch.rand(1) < self.prob_virtual_allknown:
            df_timesteps[:, :C] = 0
        if self.prob_bias_to_known > 0.0 and torch.rand(1) < self.prob_bias_to_known:
            th = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,)).item()
            df_timesteps[df_timesteps <= th] = 0
        if self.ctx_always_known:
            df_timesteps[:, :C] = 0
        return df_timesteps

    def equivariant_forward(
        self,
        noisy_param: torch.Tensor,
        df_timesteps: torch.Tensor,
        sem_type: torch.Tensor,
        group_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Core equivariant sandwich forward: canonicalize -> encode -> free -> decode -> de-canonicalize.

        This is the function tested by the E-test.

        Args:
            noisy_param: [B, C+T, P, 9] noisy PoseIO params
            df_timesteps: [B, C+T, P] diffusion timesteps
            sem_type: [B, P] semantic type IDs
            group_id: [B, P] group IDs
            mask: [B, C+T, P] bool, True = valid

        Returns:
            output: [B, C+T, P, 9] predicted x0 in world space
        """
        B, CT, P, D = noisy_param.shape
        S = CT * P

        # 1. Flatten and canonicalize
        noisy_flat = noisy_param.reshape(B, S, 9)
        mask_flat = mask.reshape(B, S)
        noisy_canon, mu, scale = self.canonicalize_9d(noisy_flat, mask_flat)

        # 2. Reshape to VN format: [B, S, 3, 3]
        x_vn = noisy_canon.reshape(B, S, 3, 3)

        # 3. Equivariant encoder: VN -> invariant scalars + registers + lifted features
        invariant, r_part, r_global, h_lifted = self.encoder(x_vn, T=CT, P=P, mask=mask_flat)
        # invariant: [B, S, 4*d_vn], r_part: [B, P, d_vn, 3], r_global: [B, d_vn, 3]
        # h_lifted: [B, S, d_vn, 3] per-token VN features for skip connection

        # 4. Free transformer on invariant features
        noise_level = df_timesteps.reshape(B, S)
        h = self.free_transformer(
            invariant, noise_level, sem_type, group_id, T=CT, P=P, mask=mask_flat
        )  # [B, S, d_model]

        # 5. Equivariant decoder: scalar features + registers + skip -> VN output
        out_vn = self.decoder(
            h, r_part, r_global, h_lifted, T=CT, P=P,
            noise_level=noise_level if self.skip_gate_by_noise else None,
        )  # [B, S, 3, 3]

        # 6. De-canonicalize
        out_canon = out_vn.reshape(B, S, 9)
        out_world = self.de_canonicalize_9d(out_canon, mu, scale)

        return out_world.reshape(B, CT, P, 9)

    @staticmethod
    @torch.compiler.disable
    def _check_finite(tensor):
        if not tensor.isfinite().all():
            n_bad = (~tensor.isfinite()).sum().item()
            raise RuntimeError(
                f"NaN/Inf in model output: {n_bad}/{tensor.numel()} ({100*n_bad/tensor.numel():.1f}%)"
            )

    @torch.compiler.disable
    def _compute_loss_terms(
        self, loss, log_dict, phase, loss_t, loss_q,
        t_r_mask, q_r_mask, t_v_mask, q_v_mask, trace_uid,
    ):
        def _add(vals, m, w, tag):
            nonlocal loss, log_dict
            if m.sum() == 0:
                return
            _uids = trace_uid[:, None].expand_as(m).long()
            term = type_mean(vals[m], _uids[m], self.gid_w_modifier)
            loss = loss + w * term
            log_dict[f"{phase}_LOSS/{tag}"] = term.detach()

        _add(loss_t, t_r_mask, self.w_t_se3 * self.w_real, "t_real")
        _add(loss_q, q_r_mask, self.w_q * self.w_real, "q_real")
        _add(loss_t, t_v_mask, self.w_t_se3 * self.w_virtual, "t_virtual")
        _add(loss_q, q_v_mask, self.w_q * self.w_virtual, "q_virtual")
        return loss, log_dict

    # -----------------------------------------------------------------
    # Training forward
    # -----------------------------------------------------------------

    def forward(self, batch, phase, global_step=0):
        """Full training forward pass."""
        ret = {}
        B, T, P = batch["trace_mask"].shape
        C = batch["context_mask"].shape[1]
        device = batch["trace_mask"].device

        trace = batch["trace"].clone()
        context = batch["context"].clone()
        trace_mask = batch["trace_mask"].clone()
        context_mask = batch["context_mask"].clone()
        sem_type = batch["trace_sem_type"].clone()
        group_id = batch["trace_group_id"].clone()
        rot_mask = batch["trace_rot_mask"].clone()

        # PoseIO encode
        trace_param = self.pose_io.encode(
            trace, sem_ids=sem_type[:, None].expand(-1, T, -1), stat_mask=trace_mask
        )
        context_param = self.pose_io.encode(
            context, sem_ids=sem_type[:, None].expand(-1, C, -1), stat_mask=context_mask
        )

        diffusion_x = torch.cat([context_param, trace_param], dim=1)
        noise = torch.randn_like(diffusion_x)
        lmm_mask = torch.cat([context_mask, trace_mask], dim=1)

        df_timesteps = self._prepare_df_step(B, C, T, P, device)

        max_ts = self.noise_scheduler.config.num_train_timesteps - 1
        invalid = ~lmm_mask
        df_timesteps[invalid] = max_ts
        diffusion_x[invalid] = noise[invalid]

        noisy_x = self.noise_scheduler.add_noise(
            diffusion_x.reshape(B * (C + T) * P, -1),
            noise.reshape(B * (C + T) * P, -1),
            df_timesteps.reshape(B * (C + T) * P),
        ).reshape(B, C + T, P, -1)

        if self.pred_type == "epsilon":
            tgt = noise
        elif self.pred_type == "sample":
            tgt = diffusion_x
        elif self.pred_type == "v_prediction":
            tgt = self.noise_scheduler.get_velocity(
                diffusion_x.reshape(-1, 9), noise.reshape(-1, 9),
                df_timesteps.reshape(-1),
            ).reshape(B, C + T, P, -1)
        else:
            raise ValueError(f"Unknown: {self.pred_type}")

        rot9_mask = self.pose_io.rot_mask(rot_mask)
        noisy_x = noisy_x * rot9_mask[:, None].float()
        # Fill position-only parts with zero deltas (p1=p2=p0) to preserve equivariance
        # (zeroed rotation channels create delta=-p0 which depends on absolute position)
        noisy_x = self._fill_zero_rotation_9d(noisy_x, rot9_mask[:, None].expand_as(noisy_x))

        out = self.equivariant_forward(noisy_x, df_timesteps, sem_type, group_id, lmm_mask)
        self._check_finite(out)

        log_dict = {}
        loss = (out * 0.0).mean()

        tgt = tgt.detach()
        loss_t = self.loss_kernel(out[..., :3], tgt[..., :3])
        loss_q = self.loss_kernel(out[..., 3:], tgt[..., 3:])

        t_sup = lmm_mask.bool()
        q_sup = t_sup & rot9_mask[:, None, :, -1].bool()
        v = torch.zeros(B, C + T, P, device=device, dtype=torch.bool)
        v[:, :C] = True

        loss, log_dict = self._compute_loss_terms(
            loss, log_dict, phase, loss_t, loss_q,
            t_sup & ~v, q_sup & ~v, t_sup & v, q_sup & v, batch["trace_uid"],
        )

        log_dict[f"{phase}_LOSS/total"] = loss.detach()
        log_dict[f"total_loss/{phase}"] = loss.detach()
        ret["loss"] = loss
        ret["log_dict"] = log_dict

        x0_pred = compute_x0(self.noise_scheduler, out, df_timesteps, noisy_x)
        T_wl = self.pose_io.decode(
            x0_pred, sem_ids=sem_type[:, None].expand(-1, C + T, -1)
        ).detach()
        ret["x0_pred_context"] = T_wl[:, :C]
        ret["x0_pred_trace"] = T_wl[:, C:]

        if self.pose_io_debug_flag:
            self.pose_io.print_stats()
        log_dict[f"{phase}_STATS/pose_io_ave_mean"] = self.pose_io.running_mean.mean()
        log_dict[f"{phase}_STATS/pose_io_ave_std"] = self.pose_io.running_var.sqrt().mean()

        return ret

    # -----------------------------------------------------------------
    # Inference (mirrors forward exactly)
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def inference(
        self, trace, context,
        trace_known_mask, context_known_mask,
        trace_mask, context_mask,
        real_timestep, group_id, sem_type, rot_mask,
        condition_timestep=0, gen_n_steps=-1, text=None, extra=None,
        **kwargs,
    ):
        """DDIM inference — same signature as E10 and v9."""
        B, T, P = trace_mask.shape
        C = context_mask.shape[1]
        device = trace.device

        trace_param = self.pose_io.encode(
            trace, sem_ids=sem_type[:, None].expand(-1, T, -1), stat_mask=None
        )
        context_param = self.pose_io.encode(
            context, sem_ids=sem_type[:, None].expand(-1, C, -1), stat_mask=None
        )
        x_all = torch.cat([context_param, trace_param], dim=1)
        known = torch.cat([context_known_mask, trace_known_mask], dim=1)
        full = torch.cat([context_mask, trace_mask], dim=1)
        rot9 = self.pose_io.rot_mask(rot_mask)

        max_ts = self.noise_scheduler.config.num_train_timesteps - 1
        df_t = torch.zeros(B, C + T, P, device=device, dtype=torch.long)
        df_t[~known] = max_ts
        df_t[~full] = max_ts

        noise = torch.randn_like(x_all)
        x_all[~known] = noise[~known]
        x_all = x_all * rot9[:, None].float()
        # Fill position-only parts with zero deltas (p1=p2=p0) to preserve equivariance
        x_all = self._fill_zero_rotation_9d(x_all, rot9[:, None].expand_as(x_all))

        for t_step in self.gen_scheduler.timesteps:
            df_now = df_t.clone()
            df_now[~known] = int(t_step)

            pred = self.equivariant_forward(x_all, df_now, sem_type, group_id, full)
            x0 = compute_x0(self.noise_scheduler, pred, df_now, x_all)

            if int(t_step) > 0:
                alpha_t = self.noise_scheduler.alphas_cumprod[int(t_step)]
                eps = (x_all - alpha_t.sqrt() * x0) / (1 - alpha_t).sqrt().clamp(min=1e-8)
                prev_idx = max(
                    int(t_step) - self.noise_scheduler.config.num_train_timesteps // len(self.gen_scheduler.timesteps),
                    0,
                )
                alpha_prev = self.noise_scheduler.alphas_cumprod[prev_idx]
                x_prev = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * eps
                x_all[~known] = x_prev[~known]
            else:
                x_all[~known] = x0[~known]
            x_all = x_all * rot9[:, None].float()
            # Fill position-only parts with zero deltas (p1=p2=p0) to preserve equivariance
            x_all = self._fill_zero_rotation_9d(x_all, rot9[:, None].expand_as(x_all))

        T_pred = self.pose_io.decode(x_all, sem_ids=sem_type[:, None].expand(-1, C + T, -1))
        return T_pred[:, :C], T_pred[:, C:]  # (context, trace) — matching v9 API
