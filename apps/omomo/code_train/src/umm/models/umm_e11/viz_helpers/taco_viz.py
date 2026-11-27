"""Visualization helpers for TACO (hand-object interaction) domain.

Renders bimanual MANO hand meshes alongside tool/target object meshes
or point-cloud proxies. Supports GT replay, future prediction,
object-to-hand (O2H), and hand-to-object (H2O) visualization modes.

Ported from umm_v1 taco_viz to work with the v6_all model architecture.

Example::

    viz = TacoVisualizer(pl_module, taco_cfg)
    viz.taco_viz_routine(batch, "val", 0, eval_model)
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import trimesh
from trimesh.creation import icosphere as TriSphere
from trimesh import Trimesh
from copy import deepcopy
import os.path as osp

from .utils import weighted_procrustes, make_square_cat_image, make_viz_T_cam
from umm.models.umm_e11.agent import AgentE11 as AgentV7  # compat alias
from umm.models.umm_e11.viz_helpers.mesh_visualizer import visualize_a_mesh
from lib_human import HumanModel
from umm.datasets.umm_e10.dataset_config import (
    UID_MANO_LEFT_HAND,
    UID_MANO_RIGHT_HAND,
    UID_TACO_TOOL,
    UID_TACO_TARGET,
)
from umm.models.umm_e11.world_models.wm_utils import disable_model_grad

# Constants for visualization colors
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)


class TacoVisualizer:
    """Visualizer for TACO bimanual hand-object interaction sequences.

    Loads left/right MANO hand models and pre-computed tool/target object
    meshes, then renders per-frame 3D scenes for training-time logging.

    Example::

        viz = TacoVisualizer(pl_module, taco_cfg)
        if viz.flag:
            frames = viz.viz_taco(prompt_type, T_wl, cam, n, l_beta, r_beta)
    """

    def __init__(self, pl_module, taco_cfg):
        """Initialize the TACO visualizer.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            taco_cfg (dict): Configuration dict with keys ``model_path``,
                ``mesh_fn``, and optional color overrides.
        """
        self.pl = pl_module
        if taco_cfg == {} or len(taco_cfg) == 0 or taco_cfg is None:
            self.flag = False
            return
        self.flag = True

        left_hand_model = HumanModel(
            model_path=taco_cfg.get("model_path", ""),
            model_type="mano",
            is_rhand=False,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        right_hand_model = HumanModel(
            model_path=taco_cfg.get("model_path", ""),
            model_type="mano",
            is_rhand=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        self.left_hand_model = disable_model_grad(left_hand_model)
        self.right_hand_model = disable_model_grad(right_hand_model)

        self.left_hand_uids = UID_MANO_LEFT_HAND
        self.right_hand_uids = UID_MANO_RIGHT_HAND
        self.tool_uid = UID_TACO_TOOL
        self.target_uid = UID_TACO_TARGET

        self.left_hand_color = taco_cfg.get("left_hand_color", (1.0, 0.0, 0.0, 1.0))
        self.right_hand_color = taco_cfg.get(
            "right_hand_color", (0.0, 1.0, 0.0, 1.0)
        )
        self.tool_color = taco_cfg.get("tool_color", (0.0, 0.0, 1.0, 1.0))
        self.target_color = taco_cfg.get("target_color", (1.0, 1.0, 0.0, 1.0))

        mesh_data = np.load(taco_cfg.get("mesh_fn", ""))
        self.taco_objects = {
            k: torch.from_numpy(mesh_data[k]) for k in mesh_data.files
        }

        self.taco_viz_cfg = self.pl.cfg.get("visualize", {}).get("taco", {})

    def viz_taco(
        self,
        prompt_type,
        T_wl,
        viz_T_cam,
        _n,
        l_beta,
        r_beta,
        tool_id=None,
        target_id=None,
        ref_T_wl=None,
    ):
        """Render a TACO hand-object interaction sequence.

        Args:
            prompt_type (torch.Tensor): Per-point UID tensor (P,).
            T_wl (torch.Tensor): World-to-link transforms (T, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera-to-world pose (4, 4).
            _n (int or list): Seen-step boundary for coloring.
            l_beta (torch.Tensor): Left-hand MANO shape parameters.
            r_beta (torch.Tensor): Right-hand MANO shape parameters.
            tool_id (str, optional): Key into ``self.taco_objects`` for tool mesh.
            target_id (str, optional): Key into ``self.taco_objects`` for target.
            ref_T_wl (torch.Tensor, optional): Reference poses for Procrustes
                alignment (C, P, 4, 4).

        Returns:
            list[np.ndarray]: Rendered RGBA frames, one per timestep.
        """
        left_hand_mask = AgentV7.get_type_mask(prompt_type, self.left_hand_uids)
        right_hand_mask = AgentV7.get_type_mask(prompt_type, self.right_hand_uids)
        lh_Q = T_wl[:, left_hand_mask]
        rh_Q = T_wl[:, right_hand_mask]

        lh_v, lh_f = self.left_hand_model.animate_with_Q(
            shape_params=l_beta[None], Q=lh_Q
        )
        rh_v, rh_f = self.right_hand_model.animate_with_Q(
            shape_params=r_beta[None], Q=rh_Q
        )

        # object
        tool_mask = AgentV7.get_type_mask(prompt_type, [self.tool_uid])
        target_mask = AgentV7.get_type_mask(prompt_type, [self.target_uid])
        if tool_id is not None and target_id is not None:
            obj_flag = True
            tool_pcl = T_wl[:, tool_mask, :3, 3]  # T,N,3
            target_pcl = T_wl[:, target_mask, :3, 3]  # T,N,3
            ref_tool_pcl = ref_T_wl[:1, tool_mask, :3, 3]  # 1,N,3
            ref_target_pcl = ref_T_wl[:1, target_mask, :3, 3]  # 1,N,3
            # solve procrustes transform ref to live
            tool_Q = weighted_procrustes(
                ref_tool_pcl.repeat(len(T_wl), 1, 1), tool_pcl
            )
            target_Q = weighted_procrustes(
                ref_target_pcl.repeat(len(T_wl), 1, 1), target_pcl
            )

            tool_v = self.taco_objects[f"{tool_id}_v"]
            tool_f = self.taco_objects[f"{tool_id}_f"]
            target_v = self.taco_objects[f"{target_id}_v"]
            target_f = self.taco_objects[f"{target_id}_f"]
            tool_v = (
                torch.einsum("tij,pj->tpi", tool_Q[:, :3, :3], tool_v.to(tool_Q))
                + tool_Q[:, :3, 3][:, None]
            )
            target_v = (
                torch.einsum(
                    "tij,pj->tpi",
                    target_Q[:, :3, :3],
                    target_v.to(target_Q),
                )
                + target_Q[:, :3, 3][:, None]
            )
        else:
            obj_flag = False
            tool_pcl = T_wl[:, tool_mask, :3, 3].cpu().numpy()  # T,N,3
            target_pcl = T_wl[:, target_mask, :3, 3].cpu().numpy()  # T,N,3

            sphere = TriSphere(radius=0.01, subdivisions=1)
            tool_mesh_list = []
            target_mesh_list = []
            for _t in range(tool_pcl.shape[0]):
                _tool_list, _target_list = [], []
                for _idx in range(tool_pcl.shape[1]):
                    sph = deepcopy(sphere)
                    sph.apply_translation(tool_pcl[_t, _idx])
                    sph.visual.vertex_colors = self.tool_color
                    _tool_list.append(sph)
                    sph = deepcopy(sphere)
                    sph.apply_translation(target_pcl[_t, _idx])
                    sph.visual.vertex_colors = self.target_color
                    _target_list.append(sph)
                tool_mesh_list.append(trimesh.util.concatenate(_tool_list))
                target_mesh_list.append(trimesh.util.concatenate(_target_list))

        frames = []
        for t in tqdm(range(len(T_wl)), desc="taco_viz"):
            m_l = Trimesh(
                vertices=lh_v[t].cpu().numpy(),
                faces=lh_f.cpu().numpy(),
            )
            m_r = Trimesh(
                vertices=rh_v[t].cpu().numpy(),
                faces=rh_f.cpu().numpy(),
            )
            m_l.visual.vertex_colors = self.left_hand_color
            m_r.visual.vertex_colors = self.right_hand_color

            if obj_flag:
                m_t = Trimesh(
                    vertices=tool_v[t].cpu().numpy(),
                    faces=tool_f.long().cpu().numpy(),
                )
                m_t.visual.vertex_colors = self.tool_color
                m_o = Trimesh(
                    vertices=target_v[t].cpu().numpy(),
                    faces=target_f.long().cpu().numpy(),
                )
                m_o.visual.vertex_colors = self.target_color
            else:
                m_t = tool_mesh_list[t]
                m_o = target_mesh_list[t]

            m = trimesh.util.concatenate([m_l, m_r, m_t, m_o])
            if isinstance(_n, list):
                background_color = deepcopy(PRED_COLOR) if t not in _n else SEEN_COLOR
            else:
                background_color = deepcopy(PRED_COLOR) if t >= _n else SEEN_COLOR

            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam.cpu().numpy(),
                mesh=m,
                background_color=background_color,
                lighting=True,
                ground_z=-0.2,
            )
            _fr[:10, ..., :3] = np.asarray(background_color)[:3] * 255.0
            frames.append(_fr)

        return frames

    def _viz_taco_subroutine(
        self,
        gen_trace,
        gen_context,
        viz_T_cam,
        batch,
        phase,
        bid,
        batch_idx,
        viz_seen_step,
        hoi_viz_name="",
        context_viz_name="",
        viz_context_flag=True,
        viz_mesh_flag=True,
    ):
        """Internal routine that renders and logs TACO visualizations.

        Args:
            gen_trace (torch.Tensor or None): Generated trace (T, P, 4, 4).
            gen_context (torch.Tensor or None): Generated context (C, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera pose.
            batch (dict): Full training batch.
            phase (str): ``"train"`` or ``"val"``.
            bid (int): Batch item index.
            batch_idx (int): Global batch index for logging.
            viz_seen_step (int or list): Boundary between seen/predicted frames.
            hoi_viz_name (str): Logger tag for the HOI video.
            context_viz_name (str): Logger tag for the context image.
            viz_context_flag (bool): Whether to visualize/log context.
            viz_mesh_flag (bool): Whether to render full object meshes.
        """
        if gen_trace is not None:
            if gen_context is not None and viz_mesh_flag:
                taco_viz = self.viz_taco(
                    batch["trace_uid"][bid],
                    gen_trace,
                    viz_T_cam,
                    viz_seen_step,
                    l_beta=batch["extra"]["beta_l"][bid],
                    r_beta=batch["extra"]["beta_r"][bid],
                    tool_id=batch["extra"]["tool_id"][bid],
                    target_id=batch["extra"]["target_id"][bid],
                    ref_T_wl=gen_context,
                )
                self.pl.logger.log_video(
                    phase,
                    f"OBJ_MESH_{hoi_viz_name}",
                    torch.from_numpy(np.stack(taco_viz, 0))[..., :3]
                    .float()
                    .permute(0, 3, 1, 2)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

            taco_viz = self.viz_taco(
                batch["trace_uid"][bid],
                gen_trace,
                viz_T_cam,
                viz_seen_step,
                l_beta=batch["extra"]["beta_l"][bid],
                r_beta=batch["extra"]["beta_r"][bid],
            )
            self.pl.logger.log_video(
                phase,
                hoi_viz_name,
                torch.from_numpy(np.stack(taco_viz, 0))[..., :3]
                .float()
                .permute(0, 3, 1, 2)
                / 255.0,
                self.pl.global_step,
                batch_idx,
            )
        if gen_context is not None and viz_context_flag:
            context_viz = self.viz_taco(
                batch["trace_uid"][bid],
                gen_context,
                viz_T_cam,
                viz_seen_step,
                l_beta=batch["extra"]["beta_l"][bid],
                r_beta=batch["extra"]["beta_r"][bid],
            )
            context_viz = make_square_cat_image(context_viz)
            self.pl.logger.log_image(
                phase,
                context_viz_name,
                context_viz[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0,
                self.pl.global_step,
                batch_idx,
            )

        return

    def taco_viz_routine(self, batch, phase, batch_idx, eval_model):
        """Top-level visualization routine for TACO.

        Finds a suitable batch item containing both hands and tool/target,
        then renders GT, half-future prediction, single-step prediction,
        O2H, and H2O (with/without context) modes.

        Args:
            batch (dict): Training batch.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference.
        """
        if not self.flag:
            return

        bid = None
        for i in range(len(batch["trace"])):
            l_mask = AgentV7.get_type_mask(
                batch["trace_uid"][i], self.left_hand_uids
            )
            r_mask = AgentV7.get_type_mask(
                batch["trace_uid"][i], self.right_hand_uids
            )
            if (
                l_mask.any()
                and r_mask.any()
                and "tool_id" in batch["extra"]
                and "target_id" in batch["extra"]
            ):
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no TACO data in batch, skip viz"
            )
            return

        viz_T_cam = make_viz_T_cam(self.taco_viz_cfg, batch)

        # * GT
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_taco_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            gen_context=batch["context"][bid],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            hoi_viz_name=f"GT_hoi_{phase}",
            context_viz_name=f"GT_context_{phase}",
        )

        # * Half future prediction
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        trace_known_mask[0, n_steps // 2 :] = False
        gen_context, gen_trace = eval_model.inference(
            trace=batch["trace"][bid : bid + 1],
            context=batch["context"][bid : bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid : bid + 1],
            context_mask=batch["context_mask"][bid : bid + 1],
            real_timestep=batch["time"][bid : bid + 1],
            group_id=batch["trace_group_id"][bid : bid + 1],
            sem_type=batch["trace_sem_type"][bid : bid + 1],
            rot_mask=batch["trace_rot_mask"][bid : bid + 1],
            local_pcl=batch["local_pcl"][bid : bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
            text=[batch["text"][bid]],
        )
        self._viz_taco_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            hoi_viz_name=f"FuturePred_hoi_seen_half_{phase}",
            context_viz_name=f"FuturePred_context_seen_half_{phase}",
            viz_context_flag=False,
        )

        # * 1 step future prediction
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask[:, 1:] = False
        gen_context, gen_trace = eval_model.inference(
            trace=batch["trace"][bid : bid + 1],
            context=batch["context"][bid : bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid : bid + 1],
            context_mask=batch["context_mask"][bid : bid + 1],
            real_timestep=batch["time"][bid : bid + 1],
            group_id=batch["trace_group_id"][bid : bid + 1],
            sem_type=batch["trace_sem_type"][bid : bid + 1],
            rot_mask=batch["trace_rot_mask"][bid : bid + 1],
            local_pcl=batch["local_pcl"][bid : bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
            text=[batch["text"][bid]],
        )
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_taco_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=1,
            hoi_viz_name=f"FuturePred_hoi_seen_1_{phase}",
            context_viz_name=f"FuturePred_context_seen_1_{phase}",
            viz_context_flag=False,
        )

        # * o2h
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        tool_mask = AgentV7.get_type_mask(
            batch["trace_uid"][bid], [self.tool_uid]
        )  # P
        target_mask = AgentV7.get_type_mask(
            batch["trace_uid"][bid], [self.target_uid]
        )  # P
        obj_mask = tool_mask | target_mask
        trace_known_mask[0, :, ~obj_mask] = False  # set robot to unknown
        gen_context, gen_trace = eval_model.inference(
            trace=batch["trace"][bid : bid + 1],
            context=batch["context"][bid : bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid : bid + 1],
            context_mask=batch["context_mask"][bid : bid + 1],
            real_timestep=batch["time"][bid : bid + 1],
            group_id=batch["trace_group_id"][bid : bid + 1],
            sem_type=batch["trace_sem_type"][bid : bid + 1],
            rot_mask=batch["trace_rot_mask"][bid : bid + 1],
            local_pcl=batch["local_pcl"][bid : bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
            text=[batch["text"][bid]],
        )
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_taco_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=0,
            hoi_viz_name=f"O2H_hoi_{phase}",
            context_viz_name=f"O2H_context_{phase}",
            viz_context_flag=False,
        )

        # * h2o (known context)
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        tool_mask = AgentV7.get_type_mask(
            batch["trace_uid"][bid], [self.tool_uid]
        )  # P
        target_mask = AgentV7.get_type_mask(
            batch["trace_uid"][bid], [self.target_uid]
        )  # P
        obj_mask = tool_mask | target_mask
        trace_known_mask[0, :, obj_mask] = False  # set object to unknown
        gen_context, gen_trace = eval_model.inference(
            trace=batch["trace"][bid : bid + 1],
            context=batch["context"][bid : bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid : bid + 1],
            context_mask=batch["context_mask"][bid : bid + 1],
            real_timestep=batch["time"][bid : bid + 1],
            group_id=batch["trace_group_id"][bid : bid + 1],
            sem_type=batch["trace_sem_type"][bid : bid + 1],
            rot_mask=batch["trace_rot_mask"][bid : bid + 1],
            local_pcl=batch["local_pcl"][bid : bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
            text=[batch["text"][bid]],
        )
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_taco_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=0,
            hoi_viz_name=f"H2O(known_context)_hoi_{phase}",
            context_viz_name=f"H2O(known_context)_context_{phase}",
            viz_context_flag=False,
        )

        # * h2o (unknown context)
        context_known_mask = torch.zeros_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        gen_context, gen_trace = eval_model.inference(
            trace=batch["trace"][bid : bid + 1],
            context=batch["context"][bid : bid + 1],
            trace_known_mask=trace_known_mask,
            context_known_mask=context_known_mask,
            trace_mask=batch["trace_mask"][bid : bid + 1],
            context_mask=batch["context_mask"][bid : bid + 1],
            real_timestep=batch["time"][bid : bid + 1],
            group_id=batch["trace_group_id"][bid : bid + 1],
            sem_type=batch["trace_sem_type"][bid : bid + 1],
            rot_mask=batch["trace_rot_mask"][bid : bid + 1],
            local_pcl=batch["local_pcl"][bid : bid + 1],
            local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
            text=[batch["text"][bid]],
        )
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_taco_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=0,
            hoi_viz_name=f"H2O(unknown_context)_hoi_{phase}",
            context_viz_name=f"H2O(unknown_context)_context_{phase}",
            viz_context_flag=True,
            viz_mesh_flag=False,
        )
        return
