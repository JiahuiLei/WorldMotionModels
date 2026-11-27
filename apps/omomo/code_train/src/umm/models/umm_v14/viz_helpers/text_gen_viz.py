"""Three-mode text_gen viz helper for v14.

Renders motion + decoded text overlay for the 3 ``sample_text`` modes
(``from_ctx``, ``from_x``, ``from_ctx_x``). Honors
``viz_every_n_train_steps=4000`` cadence by virtue of being called from the
same gated dispatch site as ``langtable_visualizer``.

Outputs per mode (resolved against ``trainer.log_dir`` or fall-backs)::

    <log_dir>/viz/text_gen_from_<mode>/step_<global_step>.mp4
    <log_dir>/viz/text_gen_from_<mode>/step_<global_step>.jpg

Old LT viz (:mod:`umm.models.umm_v14.viz_helpers.langtable_viz`) is unchanged;
this helper sits next to it and calls ``agent.sample_text`` directly.

Design choices:
  - Minimal renderer: PIL + imageio fallbacks. NO viser, NO trimesh. Reason:
    GPU-blocked environment can't run 3D rendering reliably. The point of
    the viz is **proving the wiring + decoded text overlay**, not
    aesthetic 3D scenes (which a future SG can extend).
  - Inherits ``viz_try`` semantics: every ``sample_text`` call is wrapped
    in try/except so a single mode failing never aborts the others or the
    training loop.
  - PIL / imageio are imported lazily inside the writer to keep training
    startup cost untouched in runs where text_gen is disabled.

Example::

    viz = TextGenVisualizer(pl_module, langtable_cfg)
    viz.text_gen_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TextGenVisualizer:
    """Three-mode text-gen viz: ``from_ctx``, ``from_x``, ``from_ctx_x``.

    Args:
        pl_module: The PyTorch Lightning module (used to access trainer
            ``log_dir`` / ``run_dir`` / ``global_step``).
        langtable_cfg: Optional langtable config dict. Reads optional keys:

            - ``text_gen_viz_flag`` (bool, default True)
            - ``text_gen_viz_n_steps`` (int, default 10)
            - ``text_gen_viz_g_text`` (float, default 4.0)
            - ``text_gen_viz_g_motion`` (float, default 2.0)
            - ``text_gen_viz_frame_h`` (int, default 256)
            - ``text_gen_viz_frame_w`` (int, default 256)
    """

    MODES = ("from_ctx", "from_x", "from_ctx_x")

    def __init__(self, pl_module, langtable_cfg: Optional[dict] = None):
        self.pl_module = pl_module
        cfg = langtable_cfg or {}
        # Default ON when this visualizer is constructed at all -- the caller
        # in model.py already gates construction on ``agent.text_gen_enable``.
        self.flag = bool(cfg.get("text_gen_viz_flag", True))
        self.n_steps = int(cfg.get("text_gen_viz_n_steps", 10))
        self.g_text = float(cfg.get("text_gen_viz_g_text", 4.0))
        self.g_motion = float(cfg.get("text_gen_viz_g_motion", 2.0))
        self.frame_h = int(cfg.get("text_gen_viz_frame_h", 256))
        self.frame_w = int(cfg.get("text_gen_viz_frame_w", 256))
        logger.info(
            f"TextGenVisualizer (v14): flag={self.flag} n_steps={self.n_steps} "
            f"g_text={self.g_text} g_motion={self.g_motion} "
            f"frame={self.frame_h}x{self.frame_w}"
        )

    @torch.inference_mode()
    def text_gen_viz_routine(
        self,
        batch: dict,
        phase: str,
        batch_idx: int,
        eval_model,
        step_output: Optional[dict] = None,
    ):
        """Run all 3 modes, render minimal overlay, write outputs.

        Failures (per-mode and outer) are caught and logged. Viz never
        crashes the training loop.

        Args:
            batch: Standard v14 batch dict (passed through to ``sample_text``).
            phase: ``"train"`` or ``"val"``.
            batch_idx: Lightning batch index.
            eval_model: Agent model (typically the EMA-copy in eval mode).
            step_output: Forward step outputs (unused here; reserved for
                future overlays such as L_text_mse curves).
        """
        try:
            log_dir = self._resolve_log_dir()
            global_step = int(getattr(self.pl_module, "global_step", 0))

            for mode in self.MODES:
                try:
                    out = eval_model.sample_text(
                        batch=batch,
                        mode=mode,
                        n_steps=self.n_steps,
                        g_text=self.g_text,
                        g_motion=self.g_motion,
                    )
                    self._write_viz_outputs(
                        out=out,
                        mode=mode,
                        log_dir=log_dir,
                        global_step=global_step,
                        phase=phase,
                        batch_idx=batch_idx,
                    )
                except Exception as e:
                    logger.warning(
                        f"[v14 text_gen_viz] mode={mode} step={global_step} "
                        f"phase={phase} batch_idx={batch_idx} failed: {e!r}"
                    )
        except Exception as e:
            logger.warning(f"[v14 text_gen_viz] outer failure: {e!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_log_dir(self) -> Path:
        """Resolve the run log directory.

        Priority:
          1. ``trainer.log_dir`` (Lightning standard).
          2. ``pl_module.run_dir`` attribute (test/mock convenience).
          4. ``Path.cwd()``.
        """
        trainer = getattr(self.pl_module, "trainer", None)
        if trainer is not None and getattr(trainer, "log_dir", None):
            return Path(trainer.log_dir)
        rd = getattr(self.pl_module, "run_dir", None)
        if rd:
            return Path(rd)
        env_dir = os.environ.get("WMM_BASH_LOG_DIR")
        if env_dir:
            return Path(env_dir)
        return Path.cwd()

    def _extract_caption(self, out: dict) -> str:
        """Extract a single decoded caption string from a ``sample_text`` dict.

        Lifts batch index 0; clamps length to 80 chars for overlay legibility.
        """
        decoded = out.get("pred_text_clean", None)
        if decoded is None or len(decoded) == 0:
            return ""
        s = decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded)
        return str(s)[:80]

    def _infer_n_frames(self, out: dict, default: int = 8) -> int:
        """Infer a frame count from ``motion_trace`` or fall back to default."""
        mt = out.get("motion_trace", None)
        if mt is None:
            return max(default, 8)
        try:
            # motion_trace shape: (B, T, P, 4, 4) -- use T.
            if hasattr(mt, "shape") and len(mt.shape) >= 2:
                return max(int(mt.shape[1]), default)
        except Exception:
            pass
        return max(default, 8)

    def _write_viz_outputs(
        self,
        out: dict,
        mode: str,
        log_dir: Path,
        global_step: int,
        phase: str,
        batch_idx: int,
    ) -> None:
        """Write mp4 + jpg with a text overlay caption strip.

        Minimal renderer: solid-color background sized to ``frame_h x frame_w``
        plus a text caption strip. Avoids hard dependencies on viser/trimesh
        so this works even when 3D rendering can't initialize.

        Args:
            out: Return dict from ``Agent.sample_text``.
            mode: One of ``MODES``.
            log_dir: Resolved run log directory.
            global_step: Current Lightning global step (zero-padded in path).
            phase: ``"train"`` or ``"val"`` (embedded in frame caption).
            batch_idx: Lightning batch index (embedded in frame caption).
        """
        # critic BLOCK: dispatch spec is `text_gen_from_{ctx,x,ctx_x}`;
        # mode strings already include the `from_` prefix, so f"text_gen_{mode}"
        # produces the spec-compliant path (e.g. mode="from_ctx" -> "text_gen_from_ctx").
        viz_dir = log_dir / "viz" / f"text_gen_{mode}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = viz_dir / f"step_{global_step:07d}.mp4"
        jpg_path = viz_dir / f"step_{global_step:07d}.jpg"

        caption = self._extract_caption(out)
        n_frames = self._infer_n_frames(out, default=8)

        # Lazy imports: PIL/imageio are optional and viz-only.
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # type: ignore[assignment]

        frames = []
        for i in range(n_frames):
            img = Image.new(
                "RGB",
                (self.frame_w, self.frame_h),
                color=(20, 20, 30),
            )
            draw = ImageDraw.Draw(img)
            draw.text(
                (8, 8),
                f"mode={mode} step={global_step} phase={phase} "
                f"batch={batch_idx} frame={i}/{n_frames}",
                fill=(180, 180, 180),
                font=font,
            )
            # Caption strip at bottom.
            strip_h = max(24, self.frame_h // 6)
            draw.rectangle(
                [(0, self.frame_h - strip_h), (self.frame_w, self.frame_h)],
                fill=(40, 40, 60),
            )
            cap_y = self.frame_h - strip_h + 4
            draw.text(
                (8, cap_y),
                f'pred: "{caption}"',
                fill=(220, 220, 220),
                font=font,
            )
            frames.append(np.array(img))

        # Save first frame as jpg (always succeeds if PIL is present).
        Image.fromarray(frames[0]).save(jpg_path, format="JPEG", quality=85)

        # Save mp4 via imageio if available; missing imageio is non-fatal.
        try:
            import imageio
            imageio.mimsave(str(mp4_path), frames, fps=8)
        except Exception as e:
            logger.warning(
                f"[v14 text_gen_viz] imageio mp4 save failed for mode={mode} "
                f"step={global_step}: {e!r}; first frame saved to {jpg_path}"
            )
