"""Visualization helpers for ManipTrans (dexterous manipulation) domain.

Renders Inspire dexterous hand link meshes alongside manipulated objects,
with optional Procrustes-based mesh alignment. Supports GT replay,
O2H, H2O, and future-prediction visualization modes.

Example::

    viz = ManipTransVisualizer(pl_module, maniptrans_cfg)
    viz.maniptrans_viz_routine(batch, "val", 0, eval_model)
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import trimesh
from trimesh.creation import icosphere as TriSphere
from copy import deepcopy
import sys
import os.path as osp

from .utils import weighted_procrustes, make_square_cat_image, make_viz_T_cam
from umm.models.umm_v1.agent import Agent
from umm.models.umm_v1.viz_helpers.mesh_visualizer import visualize_a_mesh
from umm.datasets.umm_v1.dataset_config import *

# Constants from model.py
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class ManipTransVisualizer:
    """
    Visualizer for ManipTrans (Dexterous Manipulation) tasks.
    Handles visualization of dexterous hands (Inspire) and manipulated objects.
    """

    def __init__(self, pl_module, maniptrans_cfg):
        """Initialize the ManipTrans visualizer.

        Loads dexterous-hand and object meshes from an ``.npz`` archive
        specified by ``mesh_fn`` in ``maniptrans_cfg``.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            maniptrans_cfg (dict): Config dict with ``mesh_fn`` key.

        Example::

            viz = ManipTransVisualizer(pl_module, cfg["visualize"]["maniptrans"])
        """
        self.pl = pl_module
        if maniptrans_cfg == {} or len(maniptrans_cfg) == 0 or maniptrans_cfg is None:
            self.flag = False
            return
        self.flag = True

        mesh_fn = maniptrans_cfg.get("mesh_fn", "")
        self.dex_objects = {}
        mesh_data = np.load(mesh_fn)
        # Parse keys: mesh_{id}_vertices
        mesh_ids = set()
        for k in mesh_data.files:
            if k.endswith("_vertices"):
                # format: mesh_{id}_vertices
                # id start at 5, end at -9
                mid = k[5:-9]
                mesh_ids.add(mid)

        for mid in tqdm(mesh_ids):
            v_key = f"mesh_{mid}_vertices"
            f_key = f"mesh_{mid}_faces"
            if v_key in mesh_data:
                v = mesh_data[v_key]
                f = mesh_data[f_key]
                self.dex_objects[str(mid)] = trimesh.Trimesh(
                    vertices=v, faces=f, process=False
                )
        logging.info(f"Loaded {len(self.dex_objects)} meshes for Dex Viz.")

        self.dex_viz_cfg = self.pl.cfg.get("visualize", {}).get("maniptrans", {})
        self.maniptrans_left_hand_uids = UID_MANIPTRANS_INSPIRE_LH
        self.maniptrans_right_hand_uids = UID_MANIPTRANS_INSPIRE_RH

        self.color_plate = cm.get_cmap("Set3").colors

    def viz_maniptrans(
        self,
        prompt_type,
        rot_mask,
        T_wl,  # T,P,4,4
        ref_T_wl,  # C,P,4,4
        viz_T_cam,
        _n,
        mesh_pointers=None,
        viz_procrustes=False,
    ):
        """
        Visualize the dexterous manipulation scene.

        Args:
            prompt_type (torch.Tensor): Tensor of prompt types.
            rot_mask (torch.Tensor): Rotation mask.
            T_wl (torch.Tensor): World-to-link transforms [T, P, 4, 4].
            ref_T_wl (torch.Tensor): Reference world-to-link transforms [C, P, 4, 4].
            viz_T_cam (torch.Tensor): Camera pose for visualization.
            _n (int or list): Current step index or list of indices for seen/unseen coloring.
            mesh_pointers (dict, optional): Pointers to mesh IDs for hands and objects.
            viz_procrustes (bool): Whether to align meshes using Procrustes analysis.

        Returns:
            list: List of rendered frames (numpy arrays).
        """
        # ! FOR NOW THIS VIZ ONE OBJECT!!!!! ONLY WORK FOR FAVOR

        # Viz Dex Hand (Inspire) + Objects
        left_hand_mask = Agent.get_type_mask(
            prompt_type, self.maniptrans_left_hand_uids
        )
        right_hand_mask = Agent.get_type_mask(
            prompt_type, self.maniptrans_right_hand_uids
        )
        # Collect PCLs (Fallback or Primary)
        lh_Q = T_wl[:, left_hand_mask]  # T, N_l, 4, 4
        rh_Q = T_wl[:, right_hand_mask]  # T, N_r, 4, 4
        sphere = TriSphere(radius=0.015, subdivisions=1)

        # Helper for mesh viz
        def get_mesh_for_id(mid):
            """Return a deep copy of the mesh for the given mesh ID.

            Args:
                mid: Mesh identifier (int or str).

            Returns:
                trimesh.Trimesh or None: Deep-copied mesh, or None if not found.

            Example::

                m = get_mesh_for_id(42)
            """
            smid = str(mid)
            if self.dex_objects and smid in self.dex_objects:
                # Return deepcopy to allow transforms
                return deepcopy(self.dex_objects[smid])
            return None

        for t in tqdm(range(len(T_wl)), desc="Viz Dexterous"):
            meshes = []

            # Left Hand
            ci = 0
            if left_hand_mask.any():
                # Try mesh viz if pointers available
                if mesh_pointers and "hand_lh" in mesh_pointers:
                    mids = mesh_pointers["hand_lh"]
                    # Ensure N matches
                    if len(mids) == lh_Q.shape[1]:
                        for i, mid in enumerate(mids):
                            m = get_mesh_for_id(mid)
                            transform = lh_Q[t, i].cpu().numpy()
                            m.apply_transform(transform)
                            m.visual.vertex_colors = self.color_plate[
                                ci % len(self.color_plate)
                            ]
                            meshes.append(m)
                            ci += 1

            # Right Hand
            if right_hand_mask.any():
                if mesh_pointers and "hand_rh" in mesh_pointers:
                    mids = mesh_pointers["hand_rh"]
                    if len(mids) == rh_Q.shape[1]:
                        for i, mid in enumerate(mids):
                            m = get_mesh_for_id(mid)
                            transform = rh_Q[t, i].cpu().numpy()
                            m.apply_transform(transform)
                            m.visual.vertex_colors = self.color_plate[
                                ci % len(self.color_plate)
                            ]
                            meshes.append(m)
                            ci += 1

            # Objects and scene
            for pointer_name, color in zip(
                ["objects", "scene"],
                [
                    self.dex_viz_cfg.get("object_color", [254, 255, 196]),
                    self.dex_viz_cfg.get("scene_color", [121, 158, 255]),
                ],
            ):
                for uid, mid in mesh_pointers[pointer_name].items():
                    obj_mask = Agent.get_type_mask(prompt_type, [uid])
                    if not obj_mask.any():
                        continue
                    obj_Q = T_wl[:, obj_mask]  # T, N_o, 4, 4
                    obj_pcl = obj_Q[:, :, :3, 3]  # T, N_o, 3
                    if viz_procrustes and ref_T_wl is not None:
                        # Procrustes Viz
                        ref_obj_pcl = ref_T_wl[:1, obj_mask, :3, 3].clone()  # 1, N_o, 3
                        T_proc = weighted_procrustes(
                            ref_obj_pcl.repeat(len(obj_pcl), 1, 1), obj_pcl
                        )
                        m = get_mesh_for_id(mid)
                        transform = T_proc[t].cpu().numpy()
                        m.apply_transform(transform)
                        m.visual.vertex_colors = color
                        meshes.append(m)
                    else:
                        _pcl = obj_pcl[t].cpu().numpy()
                        _m_list = []
                        for _p in _pcl:
                            s = deepcopy(sphere)
                            s.apply_translation(_p)
                            s.visual.vertex_colors = color
                            _m_list.append(s)
                        if _m_list:
                            meshes.append(trimesh.util.concatenate(_m_list))
            if not meshes:
                frames.append(np.zeros((512, 512, 3), dtype=np.uint8))
                continue

            final_mesh = trimesh.util.concatenate(meshes)

            if isinstance(_n, list):
                background_color = deepcopy(PRED_COLOR) if t not in _n else SEEN_COLOR
            else:
                background_color = deepcopy(PRED_COLOR) if t >= _n else SEEN_COLOR

            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam.cpu().numpy(),
                mesh=final_mesh,
                # background_color=GROUND_COLOR,
                lighting=True,
                ground_z=0.0,
                # ground_color=GROUND_COLOR,
                ground_size=5.0,
                ground_grid_size=20,
                width=768,
                height=512,
            )

            _fr[-15:, ..., :3] = np.asarray(background_color)[:3] * 255.0
            frames.append(_fr)

        return frames

    def _viz_maniptrans_subroutine(
        self,
        gen_trace,
        gen_context,
        viz_T_cam,
        batch,
        phase,
        bid,
        batch_idx,
        viz_seen_step,
        viz_name="",
        context_viz_name="",
        viz_context_flag=True,
        viz_procrustes=True,  # Default True as per request to have mesh viz
    ):
        """Internal routine that renders and logs ManipTrans visualizations.

        Renders the trace as a video and optionally the context as a grid
        image, then logs both via ``self.pl.logger``.

        Args:
            gen_trace (torch.Tensor): Generated trace (T, P, 4, 4).
            gen_context (torch.Tensor): Generated context (C, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera pose.
            batch (dict): Full training batch.
            phase (str): ``"train"`` or ``"val"``.
            bid (int): Batch item index.
            batch_idx (int): Global batch index for logging.
            viz_seen_step (int or list): Boundary between seen/predicted frames.
            viz_name (str): Logger tag for the video.
            context_viz_name (str): Logger tag for the context image.
            viz_context_flag (bool): Whether to log context.
            viz_procrustes (bool): Whether to use Procrustes alignment for
                object meshes.

        Example::

            self._viz_maniptrans_subroutine(
                gen_trace, gen_context, cam, batch, "val", 0, 0, 10,
                viz_name="GT_DEX_val",
            )
        """

        # Check mesh info
        mesh_pointers = None
        if "extra" in batch and "mesh_pointers" in batch["extra"]:
            mesh_pointers = batch["extra"]["mesh_pointers"][bid]

        dex_viz = self.viz_maniptrans(
            batch["trace_uid"][bid],
            batch["trace_rot_mask"][bid],
            gen_trace,
            gen_context,  # ref_T_wl
            viz_T_cam,
            viz_seen_step,
            mesh_pointers=mesh_pointers,
            viz_procrustes=viz_procrustes,
        )

        self.pl.logger.log_video(
            phase,
            viz_name,
            torch.from_numpy(np.stack(dex_viz, 0))[..., :3].float().permute(0, 3, 1, 2)
            / 255.0,
            self.pl.global_step,
            batch_idx,
        )

        if viz_context_flag:
            ctx_viz = self.viz_maniptrans(
                batch["trace_uid"][bid],
                batch["trace_rot_mask"][bid],
                gen_context,
                None,
                viz_T_cam,
                viz_seen_step,  # irrelevant for ctx
                mesh_pointers=mesh_pointers,
            )
            ctx_viz = make_square_cat_image(ctx_viz)
            self.pl.logger.log_image(
                phase,
                context_viz_name,
                ctx_viz[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0,
                self.pl.global_step,
                batch_idx,
            )
        return

    def maniptrans_viz_routine(self, batch, phase, batch_idx, eval_model):
        """
        Main visualization routine for ManipTrans.
        Visualizes O2H, H2O, and future predictions.

        Args:
            batch (dict): Batch data.
            phase (str): 'train' or 'val'.
            batch_idx (int): Batch index.
            eval_model (pl.LightningModule): The evaluation model.
        """
        if not self.flag:
            return

        bid = None
        # Logic to find a relevant batch item: contains dex hand?
        # Check for Inspire Hand UIDs
        all_dex_uids = self.all_dex_uids

        for i in range(len(batch["trace"])):
            mask = Agent.get_type_mask(batch["trace_uid"][i], all_dex_uids)
            if mask.any():
                bid = i
                break

        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no Dexterous Hand found, skip viz"
            )
            return

        viz_T_cam = make_viz_T_cam(self.dex_viz_cfg, batch)

        # * O2H
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        hand_mask = Agent.get_type_mask(
            batch["trace_uid"][bid],
            UID_MANIPTRANS_INSPIRE_LH + UID_MANIPTRANS_INSPIRE_RH,
        )
        trace_known_mask[:, :, hand_mask] = False
        n_steps = batch["trace_mask"][bid][:, 0].sum()
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
        )
        self._viz_maniptrans_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_name=f"O2H_{phase}",
            viz_context_flag=False,
        )

        # * H2O
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        object_mask = Agent.get_type_mask(
            batch["trace_uid"][bid], UID_MANIPTRANS_OBJECT_GROUP
        )
        trace_known_mask[:, :, object_mask] = False
        n_steps = batch["trace_mask"][bid][:, 0].sum()
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
        )
        self._viz_maniptrans_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_name=f"H2O_{phase}",
            viz_context_flag=False,
        )

        # * Half future predict
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
        )
        self._viz_maniptrans_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            viz_name=f"FuturePred_DEX_seen_half_{phase}",
            viz_context_flag=False,
        )

        # * 1 future predict
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        trace_known_mask[0, 1:] = False
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
        )
        self._viz_maniptrans_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=1,
            viz_name=f"FuturePred_DEX_seen_1_{phase}",
            viz_context_flag=False,
        )

        # * Viz GTmaniptrans
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_maniptrans_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            gen_context=batch["context"][bid],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_name=f"GT_DEX_{phase}",
            context_viz_name=f"GT_DEX_ctx_{phase}",
        )

        return
