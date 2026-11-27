"""PyTorch Lightning training module for UMM V11.

Wraps AgentV11 with EMA, optimizer, gradient monitoring, and viz dispatch.

Example::

    from umm.models.wmm.model import PLModel
    model = PLModel(cfg)
    trainer.fit(model, datamodule)
"""

import json
import logging
import time
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.utilities import rank_zero_only

from umm.models.wmm.agent import AgentV11
from umm.models.umm_v9.world_models.wm_utils import disable_model_grad


def sync_model_buffers(source_model, target_model):
    """Copy all buffers from source to target (EMA only tracks parameters)."""
    with torch.no_grad():
        for buf_src, buf_tgt in zip(source_model.buffers(), target_model.buffers()):
            buf_tgt.copy_(buf_src)


def _build_noise_mask_figure(
    diff_levels: torch.Tensor,
    all_known: torch.Tensor,
    all_mask: torch.Tensor,
    C: int,
    phase: str,
):
    """Build the mask/noise matplotlib figure for up to 4 samples.

    Two rows per sample:
    - Row 1 (noise level): valid slots use **viridis**, invalid slots use
      **grayscale** with the SAME noise-level intensity. This lets you
      visually confirm invalid slots are actually noised to ``t=1``
      (they'll show as the brightest gray, not blank).
    - Row 2 (known mask): valid slots use **RdYlGn**, invalid slots use
      a uniform dark gray (``known`` is meaningless for invalid tokens).

    Red vertical dashed line marks the ``ctx|trace`` boundary.

    Args:
        diff_levels: ``(B, C+T, P)`` float — diffusion timestep per token.
        all_known: ``(B, C+T, P)`` bool — known indicator mask.
        all_mask: ``(B, C+T, P)`` bool — validity mask.
        C: number of context tokens (x-axis boundary).
        phase: ``"train"`` / ``"val"`` (for suptitle).

    Returns:
        ``matplotlib.figure.Figure`` with a ``(2, n_samples)`` grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    B = diff_levels.shape[0]
    n_samples = min(4, B)
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
    if n_samples == 1:
        axes = axes[:, None]  # (2, 1)

    viridis = plt.get_cmap("viridis")
    gray_cmap = plt.get_cmap("gray")
    rdylgn = plt.get_cmap("RdYlGn")

    for i in range(n_samples):
        levels = diff_levels[i].detach().cpu().float().numpy()  # (C+T, P)
        mask = all_mask[i].detach().cpu().numpy().astype(bool)  # (C+T, P)
        known = all_known[i].detach().cpu().float().numpy()     # (C+T, P)

        levels_clamped = np.clip(levels, 0.0, 1.0)
        known_clamped = np.clip(known, 0.0, 1.0)
        mask_3d = mask[..., None]  # (C+T, P, 1)

        # Row 1: noise level — viridis (valid) OR grayscale (invalid).
        # Both share the noise-level intensity, so invalid slots visibly
        # prove they're being noised to t=1 (brightest gray).
        rgb_noise_valid = viridis(levels_clamped)[..., :3]   # (C+T, P, 3)
        rgb_noise_invalid = gray_cmap(levels_clamped)[..., :3]
        rgb_noise = np.where(mask_3d, rgb_noise_valid, rgb_noise_invalid)

        ax = axes[0, i]
        ax.imshow(
            rgb_noise.transpose(1, 0, 2),  # (C+T, P, 3) -> (P, C+T, 3)
            aspect="auto", interpolation="nearest",
        )
        ax.axvline(x=C - 0.5, color="red", linewidth=1, linestyle="--")
        ax.set_title(f"Sample {i}: noise level (invalid=gray)", fontsize=8)
        ax.set_ylabel("Entity P" if i == 0 else "")
        ax.set_xlabel("Time (C|T)")
        sm_noise = ScalarMappable(
            cmap=viridis, norm=Normalize(vmin=0.0, vmax=1.0)
        )
        sm_noise.set_array([])
        plt.colorbar(sm_noise, ax=ax, fraction=0.046)

        # Row 2: known mask — RdYlGn (valid) OR uniform dark gray (invalid).
        # known is binary/meaningless for invalid tokens, so use a flat
        # neutral color to distinguish from valid green/red cells.
        rgb_known_valid = rdylgn(known_clamped)[..., :3]
        rgb_known_invalid = np.full_like(rgb_known_valid, 0.3)  # dark gray
        rgb_known = np.where(mask_3d, rgb_known_valid, rgb_known_invalid)

        ax = axes[1, i]
        ax.imshow(
            rgb_known.transpose(1, 0, 2),
            aspect="auto", interpolation="nearest",
        )
        ax.axvline(x=C - 0.5, color="red", linewidth=1, linestyle="--")
        ax.set_title(f"Sample {i}: known mask (invalid=gray)", fontsize=8)
        ax.set_ylabel("Entity P" if i == 0 else "")
        ax.set_xlabel("Time (C|T)")
        sm_known = ScalarMappable(
            cmap=rdylgn, norm=Normalize(vmin=0.0, vmax=1.0)
        )
        sm_known.set_array([])
        plt.colorbar(sm_known, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{phase} -- red line = ctx|trace boundary (gray = invalid slot)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


class PLModel(pl.LightningModule):
    """PyTorch Lightning module for UMM V11.

    Args:
        cfg: Full configuration dict. Must include ``model.name = "umm_v11"``.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()

        if cfg is None:
            cfg = {}

        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.optim_cfg = self.model_cfg.get("optim", {})

        self.model: AgentV11 = self._get_agent(
            cfg,
            init_fn=self.model_cfg.get("init_fn", None),
            strict_flag=self.model_cfg.get("init_strict_flag", True),
            lmm_flag=self.model_cfg.get("init_lmm_flag", True),
            input_flag=self.model_cfg.get("init_input_flag", True),
        )

        if self.model_cfg.get("compile_flag", False):
            logging.warning("@" * 40)
            logging.warning("Compiling LMM submodule only...")
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 256
            compile_dynamic = self.model_cfg.get("compile_dynamic", True)
            compile_mode = self.model_cfg.get("compile_mode", None)
            logging.warning(f"compile dynamic={compile_dynamic}, mode={compile_mode}")
            self.model.lmm = torch.compile(
                self.model.lmm, dynamic=compile_dynamic, mode=compile_mode,
            )
            logging.warning("LMM compiled.")

        self.use_ema = self.model_cfg.get("use_ema", True)
        if self.use_ema:
            from diffusers.training_utils import EMAModel

            self.ema = EMAModel(
                parameters=self.model.parameters(),
                power=0.75,
                decay=self.model_cfg.get("ema_decay", 0.9999),
            )
            self.ema_model = disable_model_grad(AgentV11(cfg))
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

        self.color_plate = cm.get_cmap("Set3").colors[:10]

        # Initialize domain visualizers
        self._init_visualizers()

    def _init_visualizers(self):
        """Initialize domain-specific visualizers for V11.

        Each visualizer is gated by the presence of its `model.<domain>:`
        config block. Visualizers are lazy-imported so domains without their
        deps installed don't break the others.
        """
        # --- Human + Humanoid (HH) ---
        human_cfg = self.model_cfg.get("human", {})
        humanoid_cfg = self.model_cfg.get("humanoid", {})
        if human_cfg or humanoid_cfg:
            from umm.models.wmm.viz_helpers.human_viz import HumanVisualizerV11
            self.human_visualizer = HumanVisualizerV11(
                self, human_cfg, humanoid_cfg
            )
        else:
            self.human_visualizer = None

        # --- TACO ---
        taco_cfg = self.model_cfg.get("taco", {})
        if taco_cfg:
            from umm.models.wmm.viz_helpers.taco_viz import TacoVisualizer
            self.taco_visualizer = TacoVisualizer(self, taco_cfg)
        else:
            self.taco_visualizer = None

        # --- LangTable ---
        langtable_cfg = self.model_cfg.get("langtable", {})
        if langtable_cfg:
            from umm.models.wmm.viz_helpers.langtable_viz import LangTableVisualizer
            self.langtable_visualizer = LangTableVisualizer(self, langtable_cfg)
        else:
            self.langtable_visualizer = None

        # --- Manipulator (motion planning) ---
        manipulator_cfg = self.model_cfg.get("manipulator", {})
        if manipulator_cfg:
            from umm.models.wmm.viz_helpers.manipulator_viz import ManipulatorVisualizer
            self.manipulator_visualizer = ManipulatorVisualizer(self, manipulator_cfg)
        else:
            self.manipulator_visualizer = None

        # --- ManiSkill ---
        maniskill_cfg = self.model_cfg.get("maniskill", {})
        if maniskill_cfg:
            from umm.models.wmm.viz_helpers.maniskill_viz import ManiskillVisualizer
            self.maniskill_visualizer = ManiskillVisualizer(self, maniskill_cfg)
        else:
            self.maniskill_visualizer = None

        # --- MS3 state (schema-v3 single-task) ---
        ms3_state_cfg = self.model_cfg.get("ms3_state", {})
        if ms3_state_cfg:
            # task name resolved from dataset config — single-task per-config
            ms3_task = None
            ds_cfgs = self.cfg.get("dataset", {}).get("configs", [])
            for dcfg in ds_cfgs:
                if dcfg.get("type") == "ms3_state":
                    ms3_task = dcfg.get("task")
                    break
            if ms3_task is None:
                logging.warning(
                    "model.ms3_state is set but no dataset.configs[*].type=ms3_state "
                    "entry with 'task' field; Ms3StateVisualizer disabled."
                )
                self.ms3_state_visualizer = None
            else:
                from umm.models.wmm.viz_helpers.ms3_state_viz import Ms3StateVisualizer
                self.ms3_state_visualizer = Ms3StateVisualizer(
                    self, ms3_state_cfg, task=ms3_task,
                )
        else:
            self.ms3_state_visualizer = None

        # --- Behavior (BEHAVIOR-1K) — top-level visualize.behavior block ---
        behavior_cfg = self.cfg.get("visualize", {}).get("behavior", {})
        if behavior_cfg:
            from umm.models.wmm.viz_helpers.behavior_viz_v2 import (
                BehaviorVisualizer,
            )
            self.behavior_visualizer = BehaviorVisualizer(self, behavior_cfg)
            logging.info(
                f"BehaviorVisualizer wired: flag={self.behavior_visualizer.flag}"
            )
        else:
            self.behavior_visualizer = None

        # --- ManipTrans (Dexterous) ---
        maniptrans_cfg = self.cfg.get("visualize", {}).get("maniptrans", {})
        if maniptrans_cfg:
            from umm.models.wmm.viz_helpers.maniptrans_viz import ManipTransVisualizer
            self.maniptrans_visualizer = ManipTransVisualizer(self, maniptrans_cfg)
        else:
            self.maniptrans_visualizer = None

        # --- Real (DAVIS / Bridge / SynthVerse via shared visualize.real block) ---
        real_cfg = self.cfg.get("visualize", {}).get("real", {})
        if real_cfg:
            from umm.models.wmm.viz_helpers.real_viz import RealVisualizer
            self.real_visualizer = RealVisualizer(self, real_cfg)
            logging.info(
                f"RealVisualizer wired: flag={self.real_visualizer.flag}"
            )
        else:
            self.real_visualizer = None

        # --- HUMOTO ---
        humoto_cfg = self.model_cfg.get("humoto", {})
        if humoto_cfg:
            from umm.models.wmm.viz_helpers.humoto_viz import HumotoVisualizer
            self.humoto_visualizer = HumotoVisualizer(self, humoto_cfg)
        else:
            self.humoto_visualizer = None

        # --- OMOMO ---
        # Reads cfg.visualize.omomo (paths + camera). OmomoVisualizer GT branch
        # renders SMPL-X body + object mesh + PCL overlay and logs via
        # pl_module.logger.log_video; inference-mode branches are warning-stubs
        # (to be implemented when agent.inference OMOMO path is production-ready).
        omomo_cfg = self.cfg.get("visualize", {}).get("omomo", {})
        if omomo_cfg:
            from umm.models.wmm.viz_helpers.omomo_viz import OmomoVisualizer
            self.omomo_visualizer = OmomoVisualizer(self, omomo_cfg)
        else:
            self.omomo_visualizer = None

        # --- GRAB ---
        # Reads cfg.visualize.grab. GrabVisualizer renders two MANO hands +
        # rigid object + 256-pt PCL overlay; logs GT + ctx via the
        # pl.logger.log_video / log_image callbacks. Inference-mode branches
        # deferred until wmm Agent GRAB inference signature stabilizes.
        grab_cfg = self.cfg.get("visualize", {}).get("grab", {})
        if grab_cfg:
            from umm.models.wmm.viz_helpers.grab_viz import GrabVisualizer
            self.grab_visualizer = GrabVisualizer(self, grab_cfg)
        else:
            self.grab_visualizer = None

        # --- TG (Tracegen —) ---
        # Reads top-level `visualize.tg` block. Inert if block is missing
        # or empty. GT-only viz — pred overlay is deferred to a follow-up step.
        tg_viz_cfg = self.cfg.get("visualize", {}).get("tg", {})
        if tg_viz_cfg:
            from umm.models.wmm.viz_helpers.tg_viz import TgVisualizer
            self.tg_visualizer = TgVisualizer(self, tg_viz_cfg)
        else:
            self.tg_visualizer = None

    @torch.inference_mode()
    def viz(self, batch_idx, batch, step_output, phase, **kwargs):
        """Dispatch visualization to registered domain-specific visualizers."""
        torch.cuda.empty_cache()
        is_training = self.training
        self.eval()
        logging.warning(f"Start viz batchidx={batch_idx} in phase={phase} ...")
        start_time = time.time()

        # --- mask/noise matplotlib viz (always run if tensors are present) ---
        # No try/except here: VizCallback already respects viz_try. Swallowing
        # errors locally would hide viz bugs during development (viz_try=False).
        if step_output is not None and "viz_diff_levels" in step_output:
            self._viz_noise_and_masks(
                step_output["viz_diff_levels"],
                step_output["viz_all_known"],
                step_output["viz_all_mask"],
                int(step_output["viz_C"]),
                phase,
                batch_idx,
            )

        if self.use_ema:
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            eval_model = self.ema_model
        else:
            eval_model = self.model
        eval_model.eval()

        if self.human_visualizer is not None and (
            getattr(self.human_visualizer, "human_flag", False)
            or getattr(self.human_visualizer, "humanoid_flag", False)
        ):
            self.human_visualizer.human_humanoid_viz_routine(
                step_output, batch, phase, batch_idx, eval_model
            )

        # --- TACO viz dispatch ---
        # Note: TACO routine signature is (batch, phase, batch_idx, eval_model)
        # — no step_output param.
        if self.taco_visualizer is not None and getattr(
            self.taco_visualizer, "flag", False
        ):
            self.taco_visualizer.taco_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- LangTable viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model, step_output)
        if self.langtable_visualizer is not None and getattr(
            self.langtable_visualizer, "flag", False
        ):
            self.langtable_visualizer.langtable_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        # --- Manipulator (MP) viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model)
        if self.manipulator_visualizer is not None and getattr(
            self.manipulator_visualizer, "flag", False
        ):
            self.manipulator_visualizer.manipulator_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- ManiSkill viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model, step_output)
        if self.maniskill_visualizer is not None and getattr(
            self.maniskill_visualizer, "flag", False
        ):
            self.maniskill_visualizer.maniskill_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        # --- MS3 state viz dispatch ---
        if self.ms3_state_visualizer is not None and getattr(
            self.ms3_state_visualizer, "flag", False
        ):
            self.ms3_state_visualizer.ms3_state_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        # --- ManipTrans (Dexterous) viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model)
        if self.maniptrans_visualizer is not None:
            self.maniptrans_visualizer.maniptrans_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- Behavior (BEHAVIOR-1K) viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model, step_output)
        if self.behavior_visualizer is not None and getattr(
            self.behavior_visualizer, "flag", False
        ):
            logging.info("Dispatching behavior_viz_routine ...")
            self.behavior_visualizer.behavior_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        # --- Real (DAVIS/Bridge/SynthVerse) viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model, step_output)
        if self.real_visualizer is not None and getattr(
            self.real_visualizer, "flag", False
        ):
            logging.info("Dispatching real_viz_routine ...")
            self.real_visualizer.real_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        # --- HUMOTO viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model)
        if self.humoto_visualizer is not None and getattr(
            self.humoto_visualizer, "flag", False
        ):
            self.humoto_visualizer.humoto_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- OMOMO viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model). Mirrors humoto.
        if self.omomo_visualizer is not None and getattr(
            self.omomo_visualizer, "flag", False
        ):
            self.omomo_visualizer.omomo_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- GRAB viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model). Mirrors omomo.
        if self.grab_visualizer is not None and getattr(
            self.grab_visualizer, "flag", False
        ):
            self.grab_visualizer.grab_viz_routine(
                batch, phase, batch_idx, eval_model
            )

        # --- TG viz dispatch ---
        # Routine signature: (batch, phase, batch_idx, eval_model, step_output)
        if self.tg_visualizer is not None and getattr(
            self.tg_visualizer, "flag", False
        ):
            self.tg_visualizer.tg_viz_routine(
                batch, phase, batch_idx, eval_model, step_output
            )

        elapsed = time.time() - start_time
        logging.warning(f"Viz done in {elapsed:.1f}s")

        if is_training:
            self.train()

    @rank_zero_only
    def _viz_noise_and_masks(
        self,
        diff_levels: torch.Tensor,    # (B, C+T, P) float
        all_known: torch.Tensor,      # (B, C+T, P) bool
        all_mask: torch.Tensor,       # (B, C+T, P) bool
        C: int,
        phase: str,
        batch_idx: int,
    ) -> None:
        """Log mask/noise heatmap to the experiment logger.

        Delegates figure construction to :func:`_build_noise_mask_figure`.
        Invalid slots render in grayscale (with noise-level intensity) so
        they visibly prove the invalid → ``t=1`` pipeline step.
        """
        import io

        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        fig = _build_noise_mask_figure(
            diff_levels, all_known, all_mask, C, phase
        )

        # fig -> numpy (H, W, 3) uint8 -> torch (3, H, W) float in [0, 1]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_np = np.array(PILImage.open(buf))[..., :3]  # (H, W, 3) uint8
        img_chw = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        if hasattr(self, "logger") and self.logger is not None and hasattr(
            self.logger, "log_image"
        ):
            self.logger.log_image(
                phase,
                "noise_and_masks",
                img_chw,
                self.global_step,
                batch_idx,
            )

    @staticmethod
    def _get_agent(
        cfg,
        init_fn=None,
        ckpt_prefix="model.",
        strict_flag=True,
        lmm_flag=True,
        input_flag=False,
    ):
        """Create and optionally initialize AgentV11 from checkpoint."""
        model = AgentV11(cfg)

        model_cfg = cfg.get("model", {})
        if init_fn is None:
            init_fn = model_cfg.get("init_fn", None)
        if init_fn is not None:
            checkpoint_dict = AgentV11.filter_ckpt_prefix(
                torch.load(init_fn)["state_dict"], ckpt_prefix
            )
            new_dict = {}
            for k, v in checkpoint_dict.items():
                if k.startswith("_orig_mod."):
                    new_dict[k[len("_orig_mod."):]] = v
                else:
                    new_dict[k] = v
            checkpoint_dict = new_dict

            model.load_from_checkpoint(
                checkpoint_dict=checkpoint_dict,
                lmm_flag=lmm_flag,
                input_flag=input_flag,
                strict=strict_flag,
            )
            logging.info(
                f"Loaded V11 checkpoint from {init_fn} "
                f"Strict={strict_flag} LMM={lmm_flag} Input={input_flag}"
            )
        return model

    @rank_zero_only
    def on_train_start(self):
        """Log full training config to wandb."""
        from umm.training_utils.log_helpers import CustomLogger
        if isinstance(self.logger, CustomLogger) and hasattr(self.logger, "wandb_logger"):
            cfg_plain = json.loads(json.dumps(self.cfg, default=str))
            self.logger.wandb_logger.experiment.config.update(
                cfg_plain, allow_val_change=True
            )
            logging.info("Logged full config to wandb")

    def training_step(self, batch, batch_idx):
        """Single training step."""
        self.train()
        ret = self.model(batch, "train", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k, v,
                on_step=True, on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Single validation step."""
        self.eval()
        ret = self.model(batch, "val", global_step=self.global_step)
        for k, v in ret["log_dict"].items():
            self.log(
                k, v,
                on_step=True, on_epoch=True,
                batch_size=len(batch["trace"]),
                prog_bar=False,
            )
        return ret

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        grad_clip_val = self.trainer.gradient_clip_val
        grad_clip_algorithm = self.trainer.gradient_clip_algorithm

        if grad_clip_val is not None and grad_clip_val > 0:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if len(parameters) > 0:
                device = parameters[0].device
                if grad_clip_algorithm == "norm":
                    total_norm = torch.norm(
                        torch.stack([
                            torch.norm(p.grad.detach(), 2.0).to(device)
                            for p in parameters
                        ]), 2.0,
                    )
                else:
                    total_norm = max(
                        [p.grad.detach().abs().max().item() for p in parameters]
                    )

                self.log(
                    "monitoring/global_grad_norm",
                    total_norm,
                    on_step=True, on_epoch=True,
                    batch_size=1, prog_bar=False,
                )

    def optimizer_step(self, *args, **kwargs):
        """Perform optimizer step and update EMA."""
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            model_device = next(self.model.parameters()).device
            if len(self.ema.shadow_params) > 0:
                if self.ema.shadow_params[0].device != model_device:
                    self.ema.to(model_device)
            ema_model_device = next(self.ema_model.parameters()).device
            if ema_model_device != model_device:
                self.ema_model.to(model_device)
            self.ema.step(self.model.parameters())
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)

    def configure_optimizers(self):
        """Configure optimizer with warmup + cosine decay + flat min_lr schedule."""
        weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        lr = self.optim_cfg.get("lr", 1e-4)
        lmm_lr = self.optim_cfg.get("lmm_lr", lr)
        input_layer_lr = self.optim_cfg.get("input_layer_lr", lr)
        betas = self.optim_cfg.get("betas", [0.9, 0.999])

        param_groups = self.model.get_optim_groups(
            weight_decay=weight_decay,
            lmm_lr=lmm_lr,
            input_layer_lr=input_layer_lr,
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=tuple(betas),
        )

        # Warmup + cosine decay + flat min_lr (matches v9x LambdaLR pattern)
        warmup_iters = self.optim_cfg.get("lr_warmup_iters", 100)
        decay_iters = self.optim_cfg.get("lr_decay_iters", 300000)
        min_lr = self.optim_cfg.get("min_lr", 1e-5)

        def make_lr_func(base_lr):
            """Create a cosine-warmup LR lambda for a specific base learning rate."""
            min_lr_ratio = min_lr / lr  # ratio relative to global base lr
            actual_min_lr = base_lr * min_lr_ratio

            def lr_func(it):
                if it < warmup_iters:
                    return (it + 1) / (warmup_iters + 1.0)
                elif it > decay_iters:
                    return actual_min_lr / base_lr
                else:
                    decay_ratio = (it - warmup_iters) / (
                        decay_iters - warmup_iters
                    )
                    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
                    ret = actual_min_lr + coeff * (base_lr - actual_min_lr)
                    return ret / base_lr

            return lr_func

        lr_funcs = []
        for group in param_groups:
            group_base_lr = group.get("lr", lr)
            lr_funcs.append(make_lr_func(group_base_lr))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_funcs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state to checkpoint."""
        if self.use_ema:
            # Clone to CPU to avoid GPU peak-memory spike during ckpt save.
            # GPU clone (default) doubles model memory transiently and OOMs
            # for capacity-scaled variants (e.g., w1.0.1 ada_n_layers=5 at
            # BS=2 with T=200 still hits 24 GiB ceiling on RTX A5000).
            checkpoint["ema_state_dict"] = {
                "shadow_params": [p.detach().cpu().clone() for p in self.ema.shadow_params],
                "decay": self.ema.decay,
                "optimization_step": self.ema.optimization_step,
            }

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA state from checkpoint."""
        if self.use_ema and "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            for sp, loaded_sp in zip(
                self.ema.shadow_params, ema_state["shadow_params"]
            ):
                sp.data.copy_(loaded_sp.data)
            self.ema.decay = ema_state.get("decay", self.ema.decay)
            self.ema.optimization_step = ema_state.get(
                "optimization_step", 0
            )
            self.ema.copy_to(self.ema_model.parameters())
            sync_model_buffers(self.model, self.ema_model)
            logging.info("Restored EMA state from checkpoint")
        elif self.use_ema:
            logging.warning(
                "No EMA state in checkpoint — EMA will restart from "
                "current weights"
            )
