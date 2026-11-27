"""Visualization helpers for manipulator robot motion-planning domain.

Renders articulated robot link meshes, obstacle point clouds, and
optional per-link local point clouds. Supports GT, motion-planning with
IK, start/end connection, obstacle hallucination, and future-prediction
visualization modes.

Example::

    viz = ManipulatorVisualizer(pl_module, manipulator_cfg)
    viz.manipulator_viz_routine(batch, "val", 0, eval_model)
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


class ManipulatorVisualizer:
    """
    Visualizer for manipulator robots.
    Handles loading robot meshes, visualizing joint configurations, and rendering point clouds.
    """

    def __init__(self, pl_module, manipulator_cfg):
        """Initialize the manipulator visualizer.

        Loads per-robot-type link meshes from ``.npz`` files in the
        configured ``model_dir_root``.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            manipulator_cfg (dict): Config dict with ``model_dir_root`` key.

        Example::

            viz = ManipulatorVisualizer(pl_module, cfg["visualize"]["manipulator"])
        """
        self.pl = pl_module
        if (
            manipulator_cfg == {}
            or len(manipulator_cfg) == 0
            or manipulator_cfg is None
        ):
            self.flag = False
            return
        self.flag = True

        model_root = manipulator_cfg.get("model_dir_root", "")
        # assert len(model_root) > 0, "model_dir_root must be specified"

        robot_npzs = [f for f in os.listdir(model_root) if f.endswith(".npz")]
        self.manipulator_meshes = {}
        for fn in robot_npzs:
            mname = fn.split(".npz")[0]
            assert mname in UID_MANIPULATOR_DICT.keys()
            data = np.load(os.path.join(model_root, fn), allow_pickle=True)
            m_list = []
            vlist, flist = data["mesh_vs"], data["mesh_fs"]
            for v, f in zip(vlist, flist):
                if len(v) == 0 or len(f) == 0:
                    logging.warning(f"Empty mesh found in {fn}")
                    m_list.append(None)
                    continue
                m = trimesh.Trimesh(vertices=v, faces=f)
                m_list.append(m)
            self.manipulator_meshes[mname] = m_list
        # color
        self.obstacle_color = np.array([0.5, 0.5, 1.0, 1.0])
        self.sphere = TriSphere(radius=0.02, subdivisions=1)

        self.manipulator_viz_cfg = self.pl.cfg.get("visualize", {}).get(
            "manipulator", {}
        )

    def viz_manipulator(
        self,
        prompt_type,
        rot_mask,
        T_wl,  # T,P,4,4
        ref_T_wl,  # C,P,4,4
        viz_T_cam,
        _n,
        robot_type,
        scale_factor,
        special_color_ith_link=None,
        viz_link_mesh=True,
        viz_link_pcl=False,
        local_pcl=None,  # P,PCL,3
        local_pcl_mask=None,  # P
        robot_Q_from_pcl_ifpossible=True,
        H=512,
        W=512,
        local_pcl_max_N=32,
        local_pcl_r=0.01,
        pcl_r=0.01,
    ):
        """
        Visualize the manipulator robot.

        Args:
            prompt_type (torch.Tensor): Tensor of prompt types.
            rot_mask (torch.Tensor): Rotation mask.
            T_wl (torch.Tensor): World-to-link transforms [T, P, 4, 4].
            ref_T_wl (torch.Tensor): Reference world-to-link transforms [C, P, 4, 4].
            viz_T_cam (torch.Tensor): Camera pose for visualization.
            _n (int or list): Current step index or list of indices for seen/unseen coloring.
            robot_type (str): Type of the robot (e.g., 'panda').
            scale_factor (float): Scale factor for the robot mesh.
            special_color_ith_link (int, optional): Index of a link to highlight.
            viz_link_mesh (bool): Whether to visualize link meshes.
            viz_link_pcl (bool): Whether to visualize link point clouds.
            local_pcl (torch.Tensor, optional): Local point clouds [P, PCL, 3].
            local_pcl_mask (torch.Tensor, optional): Mask for local point clouds [P].
            robot_Q_from_pcl_ifpossible (bool): Recompute Q from point clouds if possible.
            H (int): Image height.
            W (int): Image width.
            local_pcl_max_N (int): Max points for local point cloud.
            local_pcl_r (float): Radius for local point cloud spheres.
            pcl_r (float): Radius for link point cloud spheres.

        Returns:
            list: List of rendered frames (numpy arrays).
        """
        # ! assume the Q is in order!!!
        robot_uid = UID_MANIPULATOR_DICT[robot_type]
        robot_mask = Agent.get_type_mask(prompt_type, robot_uid)
        obs_mask = Agent.get_type_mask(prompt_type, [UID_MOTIONPLANNING_BOX_OBSTACLE])

        robot_Q = T_wl[:, robot_mask & rot_mask].clone()
        obs_Q = T_wl[:, obs_mask]
        obs_pcl = obs_Q[:, :, :3, 3]

        # append local pcl if any
        if (obs_mask & rot_mask).any() and local_pcl is not None:
            for _pcl, _valid, _T in zip(
                local_pcl[obs_mask & rot_mask],
                local_pcl_mask[obs_mask & rot_mask],
                T_wl[:, obs_mask & rot_mask].permute(1, 0, 2, 3),
            ):
                if not _valid:
                    continue
                _trace = (
                    torch.einsum("tij,nj->tni", _T[:, :3, :3], _pcl)
                    + _T[:, None, :3, 3]
                )
                obs_pcl = torch.cat([obs_pcl, _trace], dim=1)

        rm_source = self.manipulator_meshes[robot_type]
        assert robot_Q.shape[1] == len(rm_source)
        frames = []
        for t in tqdm(range(len(T_wl))):
            # make the robot mesh
            _mm = []
            for i in range(len(rm_source)):
                # ! identify the pcl
                _link_pcl_mask = (prompt_type == robot_uid[i]) & ~rot_mask
                if special_color_ith_link is not None and i == special_color_ith_link:
                    _color = [0.0, 1.0, 0.0]
                else:
                    _color = self.pl.color_plate[i % len(self.pl.color_plate)]
                if rm_source[i] is None:
                    continue
                if _link_pcl_mask.any() and viz_link_pcl:
                    _link_pcl = T_wl[t, _link_pcl_mask, :3, 3]
                    if robot_Q_from_pcl_ifpossible:
                        # there is a pcl for this link, try to re-compute the robot_Q
                        assert ref_T_wl is not None, "ref_T_wl is None"
                        # ! re-compute the robot_Q from the links
                        _ref_pcl = ref_T_wl[:1, _link_pcl_mask, :3, 3]
                        _Q = weighted_procrustes(_ref_pcl, _link_pcl[None])[0]
                        # ! here use the reference absolute pose to compose, assume the ctx is GT
                        _base_Q = ref_T_wl[0, robot_mask & rot_mask][i].clone()
                        robot_Q[t, i] = _Q @ _base_Q
                    _m_pcl = []
                    for _idx in range(len(_link_pcl)):
                        sph = TriSphere(radius=pcl_r, subdivisions=1)
                        sph.apply_translation(_link_pcl[_idx].cpu().numpy())
                        sph.visual.vertex_colors = [it * 0.8 for it in _color[:3]]
                        _m_pcl.append(sph)
                    _m_pcl = trimesh.util.concatenate(_m_pcl)
                    _mm.append(_m_pcl)

                if viz_link_mesh:
                    _m = deepcopy(rm_source[i])
                    _m.apply_scale(scale_factor)
                    _m.apply_transform(robot_Q[t, i].cpu().numpy())
                    _c = np.ones((len(_m.vertices), 4), dtype=np.float32)
                    _c[:, :3] = _c[:, :3] * np.asarray(_color)[None, :3]
                    if viz_link_pcl:
                        _c[:, 3] = 0.75
                    _m.visual.vertex_colors = _c
                    _mm.append(_m)

                if local_pcl is not None:
                    assert local_pcl_mask is not None
                    if local_pcl_mask[robot_mask][i]:
                        _pcl = local_pcl[robot_mask][i]  # N,3
                        if len(_pcl) > local_pcl_max_N:
                            _pcl = _pcl[:local_pcl_max_N]
                        _pcl = (robot_Q[t, i][:3, :3] @ _pcl.T).T + robot_Q[t, i][
                            None, :3, 3
                        ]
                        _m_pcl = []
                        for _idx in range(len(_pcl)):
                            sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                            sph.apply_translation(_pcl[_idx].cpu().numpy())
                            sph.visual.vertex_colors = [it * 0.8 for it in _color[:3]]
                            _m_pcl.append(sph)
                        _m_pcl = trimesh.util.concatenate(_m_pcl)
                        _mm.append(_m_pcl)

            robot_m = trimesh.util.concatenate(_mm)

            _obs_list = []
            for _idx in range(obs_pcl.shape[1]):
                sph = deepcopy(self.sphere)
                sph.apply_translation(obs_pcl[t, _idx].cpu().numpy())
                sph.visual.vertex_colors = self.obstacle_color
                _obs_list.append(sph)
            obs_m = trimesh.util.concatenate(_obs_list)

            viz_m = trimesh.util.concatenate([robot_m, obs_m])
            if isinstance(_n, list):
                background_color = deepcopy(PRED_COLOR) if t not in _n else SEEN_COLOR
            else:
                background_color = deepcopy(PRED_COLOR) if t >= _n else SEEN_COLOR

            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam.cpu().numpy(),
                mesh=viz_m,
                # background_color=background_color,
                background_color=BACKGROUND_COLOR,
                lighting=True,
                ground_z=-0.2,  # for better viz the object ctx ref
                # ground_color=GROUND_COLOR,
                ground_size=10.0,
                ground_grid_size=30,
                width=W,
                height=H,
            )
            # imageio.imsave("./debug/dbg.jpg", _fr[...,:3])
            _fr[-15:, ..., :3] = np.asarray(background_color)[:3] * 255.0
            frames.append(_fr)

        return frames

    def _viz_manipulator_subroutine(
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
        special_color_ith_link=None,
        viz_link_mesh=True,
        viz_link_pcl=False,
        robot_Q_from_pcl_ifpossible=True,
    ):
        """Internal routine that renders and logs manipulator visualizations.

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
            hoi_viz_name (str): Logger tag for the HOI video.
            context_viz_name (str): Logger tag for the context image.
            viz_context_flag (bool): Whether to visualize/log context.
            special_color_ith_link (int, optional): Link index to highlight green.
            viz_link_mesh (bool): Whether to render link meshes.
            viz_link_pcl (bool): Whether to render link point clouds.
            robot_Q_from_pcl_ifpossible (bool): Re-solve Q from PCL via Procrustes.

        Example::

            self._viz_manipulator_subroutine(
                gen_trace, gen_context, cam, batch, "val", 0, 0, 10,
                hoi_viz_name="GT_manipulator_val",
            )
        """

        H = self.manipulator_viz_cfg.get("H", 480)
        W = self.manipulator_viz_cfg.get("W", 960)
        assert gen_trace is not None and gen_context is not None
        viz = self.viz_manipulator(
            batch["trace_uid"][bid],
            batch["trace_rot_mask"][bid],
            gen_trace,
            gen_context,
            viz_T_cam,
            viz_seen_step,
            scale_factor=batch["extra"]["scale_factor"][bid],
            robot_type=batch["extra"]["manipulator_type"][bid],
            special_color_ith_link=special_color_ith_link,
            viz_link_mesh=viz_link_mesh,
            viz_link_pcl=viz_link_pcl,
            robot_Q_from_pcl_ifpossible=robot_Q_from_pcl_ifpossible,
            local_pcl=batch["local_pcl"][bid],
            local_pcl_mask=batch["local_pcl_mask"][bid],
            H=H,
            W=W,
        )
        self.pl.logger.log_video(
            phase,
            hoi_viz_name,
            torch.from_numpy(np.stack(viz, 0))[..., :3].float().permute(0, 3, 1, 2)
            / 255.0,
            self.pl.global_step,
            batch_idx,
        )
        if viz_context_flag:
            ctx_viz = self.viz_manipulator(
                batch["trace_uid"][bid],
                batch["trace_rot_mask"][bid],
                gen_context,
                None,
                viz_T_cam,
                viz_seen_step,
                scale_factor=batch["extra"]["scale_factor"][bid],
                robot_type=batch["extra"]["manipulator_type"][bid],
                special_color_ith_link=special_color_ith_link,
                viz_link_mesh=viz_link_mesh,
                viz_link_pcl=viz_link_pcl,
                robot_Q_from_pcl_ifpossible=False,
                local_pcl=batch["local_pcl"][bid],
                local_pcl_mask=batch["local_pcl_mask"][bid],
                H=H,
                W=W,
            )
            # ctx_viz = np.concatenate(ctx_viz, 1)
            ctx_viz = make_square_cat_image(ctx_viz)
            self.pl.logger.log_image(
                phase,
                context_viz_name,
                ctx_viz[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0,
                self.pl.global_step,
                batch_idx,
            )

        return

    def manipulator_viz_routine(self, batch, phase, batch_idx, eval_model):
        """
        Main visualization routine for manipulators.
        Dispatches visualization tasks based on batch content.

        Args:
            batch (dict): Batch data.
            phase (str): 'train' or 'val'.
            batch_idx (int): Batch index.
            eval_model (pl.LightningModule): The evaluation model.
        """
        if not self.flag:
            return

        viz_T_cam = make_viz_T_cam(self.manipulator_viz_cfg, batch)
        for i in range(len(batch["trace"])):
            robot_type = batch["extra"]["manipulator_type"][i]
            if robot_type is None:
                continue
            if robot_type in self.manipulator_meshes.keys():
                cur_mesh = self.manipulator_meshes[robot_type]
            uids = UID_MANIPULATOR_DICT[robot_type]
            mask = Agent.get_type_mask(batch["trace_uid"][i], uids)
            if mask.any():
                bid = i
                break
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no human or humanoid, skip viz"
            )
            return

        viz_T_cam = make_viz_T_cam(self.manipulator_viz_cfg, batch)
        viz_pcl_flag = self.manipulator_viz_cfg.get("viz_pcl", False)
        # self.get_viz_T_wc(viz_stand_at, viz_look_at)

        robot_type = batch["extra"]["manipulator_type"][bid]
        robot_uids = UID_MANIPULATOR_DICT[robot_type]
        robot_mask = Agent.get_type_mask(batch["trace_uid"][bid], robot_uids)
        obs_mask = ~robot_mask

        # * GT
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        logging.info(
            f"VIZ GT, phase: {phase}, batch_idx: {batch_idx}, n_steps: {n_steps}"
        )
        self._viz_manipulator_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            gen_context=batch["context"][bid],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            hoi_viz_name=f"GT_manipulator_{phase}",
            context_viz_name=f"GT_context_{phase}",
        )
        if viz_pcl_flag:
            self._viz_manipulator_subroutine(
                gen_trace=batch["trace"][bid, :n_steps],
                gen_context=batch["context"][bid],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=n_steps,
                hoi_viz_name=f"GT_PCL_manipulator_{phase}",
                context_viz_name=f"GT_PCL_context_{phase}",
                viz_link_pcl=True,
                viz_link_mesh=True,
            )

        # * MP with IK
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        trace_known_mask = trace_known_mask * obs_mask[None, None]
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        ee_mask = batch["trace_uid"][bid] == robot_uids[-1]
        trace_known_mask[:, 0, ee_mask] = True
        trace_known_mask[:, n_steps - 1, ee_mask] = True
        # ！make sure the object is known
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
        )
        self._viz_manipulator_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=[0, n_steps - 1],
            hoi_viz_name=f"MPIK_connect_hoi_{phase}",
            context_viz_name=f"MPIK_connect_context_{phase}",
            viz_context_flag=False,
            special_color_ith_link=len(robot_uids) - 1,
        )
        if viz_pcl_flag:
            self._viz_manipulator_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=gen_context[0],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=[0, n_steps - 1],
                hoi_viz_name=f"MPIK_PCL_connect_hoi_{phase}",
                context_viz_name=f"MPIK_PCL_connect_context_{phase}",
                viz_link_pcl=True,
                viz_link_mesh=True,
                viz_context_flag=False,
                special_color_ith_link=len(robot_uids) - 1,
            )

        # * connect staring and end
        n_steps = batch["trace_mask"][bid][:, 0].sum()

        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        trace_known_mask = trace_known_mask * obs_mask[None, None]
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask[:, 0] = True
        trace_known_mask[:, n_steps - 1] = True
        # ！make sure the object is known
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
        )
        self._viz_manipulator_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=[0, n_steps - 1],
            hoi_viz_name=f"MP_connect_hoi_{phase}",
            context_viz_name=f"MP_connect_context_{phase}",
            viz_context_flag=False,
        )
        if viz_pcl_flag:
            self._viz_manipulator_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=gen_context[0],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=[0, n_steps - 1],
                hoi_viz_name=f"MP_connect_PCL_hoi_{phase}",
                context_viz_name=f"MP_connect_PCL_context_{phase}",
                viz_context_flag=False,
                viz_link_pcl=True,
                viz_link_mesh=True,
            )

        # * Obstacle halluciantion
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask = trace_known_mask * robot_mask[None, None]
        # ！make sure the object is known
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
        )
        self._viz_manipulator_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=-1,
            hoi_viz_name=f"OBST_HALLU_connect_hoi_{phase}",
            context_viz_name=f"OBST_HALLU_connect_context_{phase}",
            viz_context_flag=False,
            robot_Q_from_pcl_ifpossible=False,
        )

        # * future prediciton
        trace_known_mask = torch.ones_like(batch["trace_mask"][bid : bid + 1]).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask[:, n_steps // 2 :] = False
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
        )
        self._viz_manipulator_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=gen_context[0],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            hoi_viz_name=f"FUTURE_pred_{phase}",
            context_viz_name=f"FUTURE_context_{phase}",
            viz_context_flag=False,
        )
        if viz_pcl_flag:
            self._viz_manipulator_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=gen_context[0],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=n_steps // 2,
                hoi_viz_name=f"FUTURE_PCL_pred_{phase}",
                context_viz_name=f"FUTURE_PCL_context_{phase}",
                viz_context_flag=False,
                viz_link_pcl=True,
                viz_link_mesh=True,
            )

        return
