"""Visualization helpers for manipulator robot motion-planning domain (v6_all).

Ported from umm_v1 manipulator_viz.py to use the v6_all agent's inference
method and v6_all dataset_config UIDs/SEMs. Renders articulated robot link
meshes, obstacle point clouds, and optional per-link local point clouds.

Example::

    viz = ManipulatorVisualizer(pl_module, manipulator_cfg)
    viz.manipulator_viz_routine(batch, "val", 0, eval_model)
"""

import os
import logging
from copy import deepcopy

import numpy as np
import torch
import trimesh
from trimesh.creation import icosphere as TriSphere
from tqdm import tqdm

from umm.models.umm_v9.viz_helpers.utils import (
    weighted_procrustes,
    make_square_cat_image,
    make_viz_T_cam,
)
from umm.models.umm_v9.viz_helpers.mesh_visualizer import visualize_a_mesh
from umm.models.umm_v9.agent import AgentV7
from umm.datasets.umm_v9.dataset_config import (
    UID_MANIPULATOR_DICT,
    UID_MOTIONPLANNING_BOX_OBSTACLE,
)

# Constants
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class ManipulatorVisualizer:
    """Visualizer for manipulator robots in the v6_all architecture.

    Handles loading robot meshes, visualizing joint configurations,
    and rendering point clouds with obstacle avoidance scenarios.

    Args:
        pl_module: Parent PyTorch Lightning module (``PLModel``).
        manipulator_cfg: Config dict with ``model_dir_root`` key pointing
            to the directory containing per-robot ``.npz`` mesh files.
    """

    def __init__(self, pl_module, manipulator_cfg):
        self.pl = pl_module
        if not manipulator_cfg or len(manipulator_cfg) == 0:
            self.flag = False
            return
        self.flag = True

        model_root = manipulator_cfg.get("model_dir_root", "")

        robot_npzs = [f for f in os.listdir(model_root) if f.endswith(".npz")]
        self.manipulator_meshes = {}
        for fn in robot_npzs:
            mname = fn.split(".npz")[0]
            assert mname in UID_MANIPULATOR_DICT.keys(), (
                f"Unknown manipulator type '{mname}' in {model_root}. "
                f"Expected one of {list(UID_MANIPULATOR_DICT.keys())}"
            )
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

        self.obstacle_color = np.array([0.5, 0.5, 1.0, 1.0])
        self.sphere = TriSphere(radius=0.02, subdivisions=1)

        self.manipulator_viz_cfg = self.pl.cfg.get("visualize", {}).get(
            "manipulator", {}
        )

    def viz_manipulator(
        self,
        prompt_type,
        rot_mask,
        T_wl,       # (T, P, 4, 4)
        ref_T_wl,   # (C, P, 4, 4) or None
        viz_T_cam,
        _n,
        robot_type,
        scale_factor,
        special_color_ith_link=None,
        viz_link_mesh=True,
        viz_link_pcl=False,
        local_pcl=None,      # (P, PCL, 3)
        local_pcl_mask=None,  # (P,)
        robot_Q_from_pcl_ifpossible=True,
        H=512,
        W=512,
        local_pcl_max_N=32,
        local_pcl_r=0.01,
        pcl_r=0.01,
    ):
        """Render a manipulator robot trajectory with obstacles.

        Args:
            prompt_type: UID tensor per entity (P,).
            rot_mask: Rotation mask per entity (P,).
            T_wl: World-to-link transforms (T, P, 4, 4).
            ref_T_wl: Reference context transforms (C, P, 4, 4) or None.
            viz_T_cam: Camera pose for visualization (4, 4).
            _n: Seen step boundary (int or list of ints).
            robot_type: Robot name string (e.g. 'panda').
            scale_factor: Mesh scale factor.
            special_color_ith_link: Link index to highlight green.
            viz_link_mesh: Whether to render link meshes.
            viz_link_pcl: Whether to render link point clouds.
            local_pcl: Local point clouds (P, PCL, 3).
            local_pcl_mask: PCL validity (P,).
            robot_Q_from_pcl_ifpossible: Recompute Q from PCL via Procrustes.
            H: Image height.
            W: Image width.
            local_pcl_max_N: Max points for local PCL display.
            local_pcl_r: Radius for local PCL spheres.
            pcl_r: Radius for link PCL spheres.

        Returns:
            List of rendered frames as numpy arrays (H, W, 4).
        """
        robot_uid = UID_MANIPULATOR_DICT[robot_type]
        robot_mask = AgentV7.get_type_mask(prompt_type, robot_uid)
        obs_mask = AgentV7.get_type_mask(
            prompt_type, [UID_MOTIONPLANNING_BOX_OBSTACLE]
        )

        robot_Q = T_wl[:, robot_mask & rot_mask].clone()
        obs_Q = T_wl[:, obs_mask]
        obs_pcl = obs_Q[:, :, :3, 3]

        # Append local PCL for obstacle entities if available
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
        assert robot_Q.shape[1] == len(rm_source), (
            f"Robot Q shape {robot_Q.shape[1]} != mesh count {len(rm_source)} "
            f"for robot_type={robot_type}"
        )

        frames = []
        for t in tqdm(range(len(T_wl)), desc="Rendering manipulator"):
            _mm = []
            for i in range(len(rm_source)):
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
                        assert ref_T_wl is not None, "ref_T_wl is None"
                        _ref_pcl = ref_T_wl[:1, _link_pcl_mask, :3, 3]
                        _Q = weighted_procrustes(_ref_pcl, _link_pcl[None])[0]
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
                    _m.visual.vertex_colors = (_c * 255).astype(np.uint8)
                    _mm.append(_m)

                if local_pcl is not None:
                    assert local_pcl_mask is not None
                    if local_pcl_mask[robot_mask][i]:
                        _pcl = local_pcl[robot_mask][i]  # (N, 3)
                        if len(_pcl) > local_pcl_max_N:
                            _pcl = _pcl[:local_pcl_max_N]
                        _pcl = (
                            robot_Q[t, i][:3, :3] @ _pcl.T
                        ).T + robot_Q[t, i][None, :3, 3]
                        _m_pcl = []
                        for _idx in range(len(_pcl)):
                            sph = TriSphere(radius=local_pcl_r, subdivisions=1)
                            sph.apply_translation(_pcl[_idx].cpu().numpy())
                            sph.visual.vertex_colors = [
                                it * 0.8 for it in _color[:3]
                            ]
                            _m_pcl.append(sph)
                        _m_pcl = trimesh.util.concatenate(_m_pcl)
                        _mm.append(_m_pcl)

            robot_m = trimesh.util.concatenate(_mm)

            _obs_list = []
            for _idx in range(obs_pcl.shape[1]):
                sph = deepcopy(self.sphere)
                sph.apply_translation(obs_pcl[t, _idx].cpu().numpy())
                sph.visual.vertex_colors = (self.obstacle_color * 255).astype(
                    np.uint8
                )
                _obs_list.append(sph)
            obs_m = trimesh.util.concatenate(_obs_list)

            viz_m = trimesh.util.concatenate([robot_m, obs_m])
            if isinstance(_n, list):
                background_color = (
                    deepcopy(PRED_COLOR) if t not in _n else SEEN_COLOR
                )
            else:
                background_color = deepcopy(PRED_COLOR) if t >= _n else SEEN_COLOR

            _fr = visualize_a_mesh(
                cam_T_wc=viz_T_cam.cpu().numpy(),
                mesh=viz_m,
                background_color=BACKGROUND_COLOR,
                lighting=True,
                ground_z=-0.2,
                ground_size=10.0,
                ground_grid_size=30,
                width=W,
                height=H,
            )
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
            torch.from_numpy(np.stack(viz, 0))[..., :3]
            .float()
            .permute(0, 3, 1, 2)
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
            ctx_viz = make_square_cat_image(ctx_viz)
            self.pl.logger.log_image(
                phase,
                context_viz_name,
                ctx_viz[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0,
                self.pl.global_step,
                batch_idx,
            )

    def manipulator_viz_routine(self, batch, phase, batch_idx, eval_model):
        """Main visualization routine for manipulators.

        Dispatches GT, motion-planning-with-IK, start-end-connect,
        obstacle-hallucination, and future-prediction visualizations.

        Args:
            batch: Batch data dict.
            phase: 'train' or 'val'.
            batch_idx: Batch index.
            eval_model: The evaluation model (AgentV7).
        """
        if not self.flag:
            return

        viz_T_cam = make_viz_T_cam(self.manipulator_viz_cfg, batch)

        # Find the first batch item that has a valid manipulator
        bid = None
        for i in range(len(batch["trace"])):
            robot_type = batch["extra"]["manipulator_type"][i]
            if robot_type is None:
                continue
            if robot_type in self.manipulator_meshes:
                uids = UID_MANIPULATOR_DICT[robot_type]
                mask = AgentV7.get_type_mask(batch["trace_uid"][i], uids)
                if mask.any():
                    bid = i
                    break

        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, "
                f"no manipulator found, skip viz"
            )
            return

        viz_T_cam = make_viz_T_cam(self.manipulator_viz_cfg, batch)
        viz_pcl_flag = self.manipulator_viz_cfg.get("viz_pcl", False)

        robot_type = batch["extra"]["manipulator_type"][bid]
        robot_uids = UID_MANIPULATOR_DICT[robot_type]
        robot_mask = AgentV7.get_type_mask(batch["trace_uid"][bid], robot_uids)
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

        # * MP with IK (connect start/end EE only)
        trace_known_mask = torch.ones_like(
            batch["trace_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask = trace_known_mask * obs_mask[None, None]
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        ee_mask = batch["trace_uid"][bid] == robot_uids[-1]
        trace_known_mask[:, 0, ee_mask] = True
        trace_known_mask[:, n_steps - 1, ee_mask] = True

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

        # * Connect start and end (all links known at t=0 and t=end)
        trace_known_mask = torch.ones_like(
            batch["trace_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask = trace_known_mask * obs_mask[None, None]
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask[:, 0] = True
        trace_known_mask[:, n_steps - 1] = True

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

        # * Obstacle hallucination (only robot known)
        trace_known_mask = torch.ones_like(
            batch["trace_mask"][bid : bid + 1]
        ).bool()
        context_known_mask = torch.ones_like(
            batch["context_mask"][bid : bid + 1]
        ).bool()
        trace_known_mask = trace_known_mask * robot_mask[None, None]

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

        # * Future prediction (first half known)
        trace_known_mask = torch.ones_like(
            batch["trace_mask"][bid : bid + 1]
        ).bool()
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
