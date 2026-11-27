"""Visualization helpers for Language Table tabletop manipulation domain (V6_All).

Renders robot end-effector and block object meshes/point-clouds for GT and
inference visualization during training. Based on the V1 langtable_viz but
adapted for the V6_All architecture with optional DINO image conditioning.

NOTE: Environment rollout visualization is disabled in this version to avoid
the language_table dependency. Only GT + x0-pred + inference modes are active.

Example::

    viz = LangTableVisualizer(pl_module, langtable_cfg)
    viz.langtable_viz_routine(batch, "val", 0, eval_model, step_output)
"""

import logging

import numpy as np
import torch
import trimesh
from trimesh.creation import icosphere as TriSphere
from copy import deepcopy
import matplotlib.colors as mcolors

from umm.datasets.umm_v6_all.dataset_config import (
    UID_LANGTABLE_OBJECT_GROUP,
    UID_LANGTABLE_EE,
    UID_LANGTABLE_ACT,
    SEM_LANGTABLE_ACT,
    SEM_LANGTABLE_EE,
    SEM_OBJECT,
)
from umm.models.umm_v6_nondf.viz_helpers.utils import (
    make_square_cat_image,
    make_viz_T_cam,
    sample_list,
    put_caption_at_bottom,
)
from umm.models.umm_v6_nondf.viz_helpers.mesh_visualizer import visualize_a_mesh

# Constants
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class LangTableVisualizer:
    """Visualizer for Language Table environment trajectories (V6_All).

    Handles rendering of robot end-effector, action target, and object meshes
    as trimesh objects. Supports GT, x0-pred, and inference visualization modes
    during training.

    Args:
        pl_module: Parent PyTorch Lightning module (PLModel).
        langtable_cfg: Config dict from ``model.langtable`` with mesh paths,
            ``num_pts``, and rollout parameters.
    """

    def __init__(self, pl_module, langtable_cfg):
        self.pl = pl_module
        if not langtable_cfg or len(langtable_cfg) == 0:
            self.flag = False
            self.robot_data = None
            return
        self.flag = True

        self.robot_data = np.load(langtable_cfg.get("model_path", ""))
        self.object_data = np.load(langtable_cfg.get("object_path", ""))

        # Build robot and object meshes keyed by UID
        self.mesh = {}
        self.ee_uid = langtable_cfg.get("ee_uid", 16777219)
        _v = self.robot_data[f"{self.ee_uid}_v"]
        _f = self.robot_data[f"{self.ee_uid}_f"]
        self.mesh[UID_LANGTABLE_EE] = trimesh.Trimesh(vertices=_v, faces=_f)
        self.mesh[UID_LANGTABLE_ACT] = trimesh.Trimesh(vertices=_v, faces=_f)
        object_ids = [k[:-2] for k in self.object_data.keys() if k.endswith("_v")]
        for k in object_ids:
            _v = self.object_data[k + "_v"]
            _f = self.object_data[k + "_f"]
            self.mesh[UID_LANGTABLE_OBJECT_GROUP[int(k)]] = trimesh.Trimesh(
                vertices=_v, faces=_f
            )

        self.num_pts = langtable_cfg.get("num_pts", -1)
        assert self.num_pts > 0, "must set num_pts in langtable config"

        self.window = langtable_cfg.get("window", 20)
        self.hist = langtable_cfg.get("hist", 1)
        self.act_horizon = langtable_cfg.get("act_horizon", 10)

        if not isinstance(self.act_horizon, list):
            self.act_horizon = [self.act_horizon]
        if not isinstance(self.hist, list):
            self.hist = [self.hist]
        if not isinstance(self.window, list):
            self.window = [self.window]

        # Rollout disabled by default (requires language_table env dependency)
        self.rollout_flag = langtable_cfg.get("rollout_flag", False)
        logging.info(f"LangTableVisualizer (v6_all): rollout_flag={self.rollout_flag}")

        self.langtable_viz_cfg = self.pl.cfg.get("visualize", {}).get("langtable", {})
        self.add_cfg = self.langtable_viz_cfg.get("add_cfg", [1.0])

    def __make_dummy_frames__(self, T, H, W):
        """Create blank placeholder frames."""
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(T)]

    def render_viz_frames(
        self,
        prompt_type,
        T_wl,        # (T, N, 4, 4)
        viz_T_cam,
        local_pcl,   # (N, PCL, 3+)
        local_pcl_mask,  # (N,)
        H=512,
        W=512,
        local_pcl_max_N=100,
        local_pcl_r=0.005,
        pcl_alpha=1.0,
        marking_mask=None,
        marking_alpha=0.75,
        viz_mask=None,
        pcl_mode=False,
    ):
        """Render Language Table scene frames.

        Args:
            prompt_type: UID for each part (N,).
            T_wl: World-to-link transforms (T, N, 4, 4).
            viz_T_cam: Camera-to-world transform (4, 4).
            local_pcl: Local point clouds (N, PCL, 6).
            local_pcl_mask: Mask for valid point clouds (N,).
            H: Image height.
            W: Image width.
            marking_mask: Per-frame per-part mask for marking predictions (T, N).

        Returns:
            list[np.ndarray]: Rendered RGBA frames.
        """
        if self.robot_data is None:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)

        T, N = T_wl.shape[:2]

        if marking_mask is not None:
            assert marking_mask.shape == (T, N)
        else:
            marking_mask = torch.zeros((T, N), dtype=torch.bool)

        if viz_mask is not None:
            assert viz_mask.shape == (T, N)
        else:
            viz_mask = torch.ones((T, N), dtype=torch.bool)

        frames = []
        for t in range(T):
            _mm = []
            for i in range(N):
                if not local_pcl_mask[i]:
                    continue
                if not viz_mask[t, i]:
                    continue

                _pcl = local_pcl[i]  # (PCL, 6)
                if len(_pcl) > local_pcl_max_N:
                    _pcl = _pcl[:local_pcl_max_N]
                _rgb = _pcl[:, 3:].clone().cpu()
                if prompt_type[i] == UID_LANGTABLE_EE:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 1] = 1.0  # green for EE
                elif prompt_type[i] == UID_LANGTABLE_ACT:
                    _rgb = _rgb * 0.0 + 0.2
                    _rgb[:, 0] = 1.0  # red for action target

                # Color marking for predicted vs known
                if marking_mask[t, i]:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = 1.0
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)
                else:
                    _hsv = mcolors.rgb_to_hsv(_rgb)
                    _hsv[..., 1] = _hsv[..., 1] * 0.7
                    _hsv[..., 2] = 1.0
                    _rgb = mcolors.hsv_to_rgb(_hsv)

                _pcl_alpha = marking_alpha if marking_mask[t, i] else pcl_alpha
                if pcl_mode:
                    _pcl = (T_wl[t, i][:3, :3] @ _pcl[:, :3].T).T + T_wl[t, i][
                        None, :3, 3
                    ]
                    _m_pcl = []
                    for _idx in range(len(_pcl)):
                        sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                        sph.apply_translation(_pcl[_idx].cpu().numpy())
                        sph.visual.vertex_colors = _rgb[_idx].tolist() + [_pcl_alpha]
                        _m_pcl.append(sph)
                    _m_pcl = trimesh.util.concatenate(_m_pcl)
                else:
                    _mesh = self.mesh[int(prompt_type[i])]
                    _mesh = _mesh.copy()
                    _mesh.apply_transform(T_wl[t, i].cpu().numpy())
                    _mesh.visual.vertex_colors = _rgb.mean(0).tolist() + [_pcl_alpha]
                    _m_pcl = _mesh
                _mm.append(_m_pcl)

            viz_m = trimesh.util.concatenate(_mm)
            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam,
                mesh=viz_m,
                background_color=BACKGROUND_COLOR,
                ground_color=GROUND_COLOR,
                lighting=True,
                ground_z=0.0,
                ground_grid_size=0.1,
                width=W,
                height=H,
            )
            frames.append(_fr)
        return frames

    def make_viz_T_cam(self, batch):
        """Build visualization camera from config.

        Args:
            batch: Training batch (unused for fixed-camera viz).

        Returns:
            torch.Tensor: 4x4 camera-to-world transform.
        """
        return make_viz_T_cam(self.langtable_viz_cfg, batch)

    @staticmethod
    def get_type_mask(uid_tensor, uid_list):
        """Create a boolean mask for parts matching any UID in uid_list.

        Args:
            uid_tensor: (P,) tensor of UIDs.
            uid_list: list of UIDs to match.

        Returns:
            torch.Tensor: (P,) boolean mask.
        """
        uid_cpu = uid_tensor.cpu()
        mask = torch.zeros(len(uid_cpu), dtype=torch.bool)
        for uid in uid_list:
            mask = mask | (uid_cpu == uid)
        return mask

    @torch.inference_mode()
    def langtable_viz_routine(self, batch, phase, batch_idx, eval_model, step_output):
        """Top-level visualization routine for Language Table.

        Finds the first batch item with LangTable UIDs, logs GT, x0-pred, and
        inference visualizations. Supports image-conditioned inference when DINO
        is enabled.

        Args:
            batch: Training batch dict.
            phase: "train" or "val".
            batch_idx: Batch index for logging.
            eval_model: Agent model for inference.
            step_output: Training step outputs with x0_pred_trace/context.
        """
        if not self.flag:
            return

        viz_T_cam = self.make_viz_T_cam(batch)

        # Find a batch item corresponding to LangTable
        bid = None
        for i in range(len(batch["trace"])):
            mask = self.get_type_mask(
                batch["trace_uid"][i],
                UID_LANGTABLE_OBJECT_GROUP + [UID_LANGTABLE_EE, UID_LANGTABLE_ACT],
            )
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no langtable data, skip viz"
            )
            return

        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(f"LT VIZ phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}")

        # --- Subroutine: render and log a single trace visualization ---
        def _subroutine(T_wl, ref_T_wl, name, known_n_real, known_n_ctx, seed):
            trace_marker_mask = torch.ones(*T_wl.shape[:2]).bool()
            if isinstance(known_n_real, list):
                for idx in known_n_real:
                    trace_marker_mask[idx] = False
            else:
                trace_marker_mask[:known_n_real] = False

            viz = self.render_viz_frames(
                prompt_type=batch["trace_uid"][bid],
                T_wl=T_wl,
                viz_T_cam=viz_T_cam,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                marking_mask=trace_marker_mask,
                H=360,
                W=640,
            )
            viz = put_caption_at_bottom(viz, batch["text"][bid])
            self.pl.logger.log_video(
                phase,
                f"{name}_seed{seed}",
                torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
                caption=batch["text"][bid],
            )
            if ref_T_wl is not None:
                ctx_marker_mask = torch.ones(*ref_T_wl.shape[:2]).bool()
                if isinstance(known_n_ctx, list):
                    for idx in known_n_ctx:
                        ctx_marker_mask[idx] = False
                else:
                    ctx_marker_mask[:known_n_ctx] = False
                viz_ctx = self.render_viz_frames(
                    prompt_type=batch["trace_uid"][bid],
                    T_wl=ref_T_wl,
                    viz_T_cam=viz_T_cam,
                    local_pcl=batch["local_pcl"][bid],
                    local_pcl_mask=batch["local_pcl_mask"][bid],
                    marking_mask=ctx_marker_mask,
                    H=360,
                    W=640,
                )
                viz_img = make_square_cat_image(viz_ctx)
                viz_img = viz_img[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"ctx_{name}_seed{seed}",
                    viz_img[::1, ::1].transpose(2, 0, 1).astype(np.float32) / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

        # --- GT trace ---
        _subroutine(
            batch["trace"][bid].float().cpu(),
            batch["context"][bid].float().cpu(),
            "GT",
            n_steps,
            100000,
            "gt",
        )

        # --- x0-pred ---
        _subroutine(
            step_output["x0_pred_trace"][bid].float().cpu(),
            step_output["x0_pred_context"][bid].float().cpu(),
            "X0_PRED",
            -1,
            -1,
            -1,
        )

        # --- Prepare image data for inference (if DINO is enabled) ---
        images_for_inf = None
        view_ids_for_inf = None
        if eval_model.use_dino and "extra" in batch:
            extra = batch["extra"]
            if "images" in extra and extra["images"][bid] is not None:
                images_for_inf = torch.as_tensor(
                    np.stack([extra["images"][bid]]),
                    dtype=torch.float32,
                ).to(batch["trace"].device)  # (1, V, 3, H, W)
            if "view_ids" in extra and extra["view_ids"][bid] is not None:
                view_ids_for_inf = torch.as_tensor(
                    np.stack([extra["view_ids"][bid]])
                ).to(batch["trace"].device)

        # --- Inference: future prediction with goal ---
        dataset_trace_mask = batch["trace_mask"][bid]  # (T, P)
        dataset_ctx_mask = batch["context_mask"][bid]  # (C, P)

        pad_mask = batch["trace_mask"][bid].any(0) | batch["context_mask"][bid].any(0)
        pad_mask_trace = (
            pad_mask[None].bool().clone().repeat(batch["trace_mask"].shape[1], 1)
        )[None]
        pad_mask_ctx = pad_mask[None, None].bool().clone()

        for seed in [42, 1234]:
            trace_known_mask = dataset_trace_mask[None].bool().clone()
            context_known_mask = dataset_ctx_mask[None].bool().clone()
            n_steps_int = int((batch["time"][bid] <= 0).sum())
            trace_known_mask[0, n_steps_int:] = False

            gen_context, gen_trace = eval_model.inference(
                trace=batch["trace"][bid:bid + 1],
                context=batch["context"][bid:bid + 1],
                trace_known_mask=trace_known_mask,
                context_known_mask=context_known_mask,
                trace_mask=pad_mask_trace,
                context_mask=pad_mask_ctx,
                real_timestep=batch["time"][bid:bid + 1],
                group_id=batch["trace_group_id"][bid:bid + 1],
                sem_type=batch["trace_sem_type"][bid:bid + 1],
                rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                local_pcl=batch["local_pcl"][bid:bid + 1],
                local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                text=[batch["text"][bid]],
                seed=seed,
                left2right_AR_sweep_flag=False,
                images=images_for_inf,
                view_ids=view_ids_for_inf,
            )
            _subroutine(
                gen_trace[0].float().cpu(),
                None,
                "Future(w. goal)",
                n_steps_int,
                -1,
                seed,
            )

            # --- Inference: future prediction without goal (text-conditioned) ---
            context_known_mask[...] = False
            for cfg in self.add_cfg:
                gen_context, gen_trace = eval_model.inference(
                    trace=batch["trace"][bid:bid + 1],
                    context=batch["context"][bid:bid + 1],
                    trace_known_mask=trace_known_mask,
                    context_known_mask=context_known_mask,
                    trace_mask=pad_mask_trace,
                    context_mask=pad_mask_ctx,
                    real_timestep=batch["time"][bid:bid + 1],
                    group_id=batch["trace_group_id"][bid:bid + 1],
                    sem_type=batch["trace_sem_type"][bid:bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid:bid + 1],
                    local_pcl=batch["local_pcl"][bid:bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid:bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                    add_cfg=cfg,
                    left2right_AR_sweep_flag=False,
                    images=images_for_inf,
                    view_ids=view_ids_for_inf,
                )
                _subroutine(
                    gen_trace[0].float().cpu(),
                    gen_context[0].float().cpu(),
                    f"Future(no goal)_{cfg}",
                    n_steps_int,
                    0,
                    seed,
                )
