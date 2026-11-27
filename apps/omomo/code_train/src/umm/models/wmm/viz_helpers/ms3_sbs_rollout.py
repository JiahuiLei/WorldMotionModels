"""Side-by-side MS3 rollout viz (SG-R).

Builds a per-episode mp4 showing:
- LEFT:  hallucinated context rendered via Ms3StateVisualizer meshes.
- RIGHT: sapien ``env.render()`` rgb_array frame (captured by MS3Evaluator).
- BOTTOM: caption ``success=<bool>  task=<task>  step=<t>/<T>``.

Port of the langtable deploy rollout viz pattern
(``src/umm/models/wmm/viz_helpers/langtable_viz.py`` lines 751-950)
adapted to the MS3 evaluator integration.

Example::

    renderer = Ms3SbsRolloutRenderer(pl_module, ms3_state_cfg, task)
    frames = []
    for step, rgb_rollout, ctx_poses in records:
        left = renderer.render_left_frame(ctx_poses, pcl, uids, viz_T_cam)
        sbs = renderer.stitch_sbs(left, rgb_rollout)
        frames.append(sbs)
    renderer.write_mp4(frames, "/path/to/out_sbs.mp4",
                       success=True, task="PushCube-v1",
                       total_steps=len(frames))
"""

from __future__ import annotations

import logging
import os
import os.path as osp
from typing import List

import imageio
import numpy as np
import torch

from umm.models.wmm.viz_helpers.ms3_state_viz import Ms3StateVisualizer
from umm.models.wmm.viz_helpers.utils import (
    make_viz_T_cam,
    put_caption_at_bottom,
)

logger = logging.getLogger(__name__)


class Ms3SbsRolloutRenderer:
    """Builds side-by-side mp4 of hallucinated goal vs env rollout.

    The LEFT panel renders the hallucinated context (one pose per part) via
    :class:`Ms3StateVisualizer`. The RIGHT panel is passed in from the caller
    (sapien ``env.render()`` frames captured during rollout).

    Args:
        pl_module: UMM wmm PLModel (needed for :class:`Ms3StateVisualizer`
            construction).
        ms3_state_cfg: dict with ``{model_path, object_path, num_pts}`` —
            same layout as ``cfg['model']['ms3_state']`` consumed by the
            training visualizer.
        task: MS3 task id (e.g. ``"PushCube-v1"``).
        video_fps: frames per second of the output mp4 (default 20).
        frame_h: LEFT-panel render height in pixels (default 512).
        frame_w: LEFT-panel render width in pixels (default 512).
    """

    def __init__(
        self,
        pl_module,
        ms3_state_cfg: dict,
        task: str,
        video_fps: int = 20,
        frame_h: int = 512,
        frame_w: int = 512,
    ):
        self.task = task
        self.video_fps = int(video_fps)
        self.frame_h = int(frame_h)
        self.frame_w = int(frame_w)

        # Use Ms3StateVisualizer directly — it owns the mesh-rendering
        # pipeline (pyrender path, palm TCP-offset correction, color coding).
        # We only need its ``render_viz_frames`` method.
        self._viz = Ms3StateVisualizer(pl_module, ms3_state_cfg, task=task)
        if not self._viz.flag:
            raise RuntimeError(
                f"Ms3SbsRolloutRenderer: Ms3StateVisualizer failed to "
                f"initialise for task={task!r}. Check ms3_state_cfg: "
                f"{list(ms3_state_cfg.keys()) if ms3_state_cfg else None}."
            )

        # Pre-compute a default camera pose once (no batch-cam available at
        # eval time). Uses the same make_viz_T_cam helper as training viz.
        viz_cfg = getattr(self._viz, "ms3_state_viz_cfg", {}) or {}
        self._viz_T_cam = make_viz_T_cam(viz_cfg, batch=None)

    # ------------------------------------------------------------------
    def render_left_frame(
        self,
        ctx_poses: torch.Tensor,
        local_pcl: torch.Tensor,
        prompt_types: torch.Tensor,
    ) -> np.ndarray:
        """Render a single LEFT frame from a ``(P, 4, 4)`` hallucinated ctx.

        Args:
            ctx_poses: ``(P, 4, 4)`` float32 part poses.
            local_pcl: ``(P, num_pts, 6)`` float32 local PCLs (used for mesh
                vertex-color averaging in mesh_mode).
            prompt_types: ``(P,)`` long tensor of UIDs.

        Returns:
            ``(H, W, 3)`` uint8 RGB frame.
        """
        if ctx_poses.ndim != 3 or ctx_poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"ctx_poses must be (P, 4, 4), got {tuple(ctx_poses.shape)}"
            )
        P = ctx_poses.shape[0]
        T_wl = ctx_poses.unsqueeze(0).to(torch.float32)  # (T=1, P, 4, 4)

        mesh_mode = bool(self._viz.ms3_state_viz_cfg.get("mesh_mode", True))
        local_pcl_mask = torch.ones(P, dtype=torch.bool)

        frames = self._viz.render_viz_frames(
            task=self.task,
            prompt_type=prompt_types,
            T_wl=T_wl,
            viz_T_cam=self._viz_T_cam,
            local_pcl=local_pcl,
            local_pcl_mask=local_pcl_mask,
            H=self.frame_h,
            W=self.frame_w,
            pcl_mode=not mesh_mode,
        )
        if not frames:
            raise RuntimeError("render_viz_frames returned empty list")
        # frames[0] is RGBA uint8; strip alpha.
        return np.asarray(frames[0])[..., :3].astype(np.uint8)

    # ------------------------------------------------------------------
    def stitch_sbs(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Horizontally concat LEFT + RIGHT panels.

        If heights differ, the shorter one is resized to the taller's height
        preserving its width via cv2. Output is ``(H, W_l + W_r, 3)`` uint8.
        """
        import cv2

        left = np.asarray(left)[..., :3]
        right = np.asarray(right)[..., :3]
        if left.dtype != np.uint8:
            left = left.astype(np.uint8)
        if right.dtype != np.uint8:
            right = right.astype(np.uint8)

        # Harmonise heights.
        Hl, Wl = left.shape[:2]
        Hr, Wr = right.shape[:2]
        if Hl != Hr:
            target_H = max(Hl, Hr)
            if Hl != target_H:
                new_Wl = max(1, int(round(Wl * target_H / Hl)))
                left = cv2.resize(
                    left, (new_Wl, target_H), interpolation=cv2.INTER_LINEAR,
                )
            if Hr != target_H:
                new_Wr = max(1, int(round(Wr * target_H / Hr)))
                right = cv2.resize(
                    right, (new_Wr, target_H), interpolation=cv2.INTER_LINEAR,
                )
        return np.concatenate([left, right], axis=1)

    # ------------------------------------------------------------------
    def build_mp4(
        self,
        sbs_frames: List[np.ndarray],
        out_path: str,
        success: bool,
        task: str,
        total_steps: int,
    ) -> str:
        """Write captioned mp4 to ``out_path`` and return the path.

        Uses :pyfunc:`put_caption_at_bottom` for the caption bar so the
        caption rendering is consistent with the rest of the viz stack.
        Per-frame step index is ``1..len(frames)``; the caption carries
        ``success=<bool> task=<task> step=<t>/<T>`` for every frame.
        """
        if not sbs_frames:
            raise ValueError("build_mp4: sbs_frames is empty")

        # Generate per-frame caption. put_caption_at_bottom renders one
        # caption across all frames; we want per-frame step info, so call
        # it frame-by-frame (inefficient but correct; mp4 lengths are small).
        captioned: List[np.ndarray] = []
        for t, frame in enumerate(sbs_frames, start=1):
            caption = (
                f"success={success}  task={task}  step={t}/{total_steps}"
            )
            out_list = put_caption_at_bottom([frame], caption)
            captioned.append(out_list[0])

        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        imageio.mimsave(
            out_path,
            [np.asarray(f) for f in captioned],
            fps=self.video_fps,
        )
        logger.info(
            "Ms3SbsRolloutRenderer: wrote %d-frame sbs mp4 to %s",
            len(captioned), out_path,
        )
        return out_path
