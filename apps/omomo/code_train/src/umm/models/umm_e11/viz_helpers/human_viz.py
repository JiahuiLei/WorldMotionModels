"""Visualization helpers for human and humanoid (robot) domains.

Renders SMPL/SMPLX human body meshes and articulated humanoid robot
link meshes, with support for multi-person/multi-robot scenes. Supports
GT replay, human-to-robot (H2R), robot-to-human (R2H), unconditional
generation, in-filling, and future-prediction visualization modes.

Example::

    viz = HumanVisualizer(pl_module, human_cfg, humanoid_cfg)
    viz.human_humanoid_viz_routine(step_output, batch, "val", 0, eval_model)
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
from copy import deepcopy
import sys, os, os.path as osp
from .utils import make_square_cat_image, make_viz_T_cam
from umm.models.umm_e11.viz_helpers.mesh_visualizer import visualize_a_mesh
from lib_human import HumanModel
from umm.models.umm_e11.world_models.wm_utils import disable_model_grad

# Constants from model.py
SEEN_COLOR = (177 / 255.0, 194 / 255.0, 158 / 255.0, 1.0)
PRED_COLOR = (250 / 255.0, 218 / 255.0, 122 / 255.0, 1.0)
GROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)


class HumanVisualizer:
    """Visualizer for human body and humanoid robot motion sequences.

    Handles SMPL/SMPLX body model animation and per-link humanoid robot
    mesh rendering. Supports multi-agent scenes and blended overlay
    visualization of human and robot simultaneously.

    Example::

        viz = HumanVisualizer(pl_module, human_cfg, humanoid_cfg)
        viz.human_humanoid_viz_routine(step_output, batch, "val", 0, eval_model)
    """

    def __init__(self, pl_module, human_cfg, humanoid_cfg):
        """Initialize the human/humanoid visualizer.

        Loads SMPL/SMPLX body model and/or humanoid link meshes from
        ``.npz`` files based on provided config dicts.

        Args:
            pl_module: Parent PyTorch Lightning module (``PLModel``).
            human_cfg (dict): Config for SMPL/SMPLX model (``model_path``,
                ``model_type``, ``gender``).
            humanoid_cfg (dict): Config for humanoid link meshes
                (``mesh_fn`` mapping robot type to ``.npz`` path).

        Example::

            viz = HumanVisualizer(pl_module, human_cfg, humanoid_cfg)
        """
        self.pl = pl_module

        # Human init
        if human_cfg == {} or len(human_cfg) == 0 or human_cfg is None:
            self.human_flag = False
        else:
            self.human_flag = True
            human_model = HumanModel(
                model_path=osp.abspath(human_cfg.get("model_path", "")),
                model_type=human_cfg.get("model_type", "smpl"),
                gender=human_cfg.get("gender", "neutral"),
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            self.human_model = disable_model_grad(human_model)
            logging.info(
                f"Human model initialized, num_joints: {self.human_model.num_joints}"
            )

        # Humanoid init
        if humanoid_cfg == {} or len(humanoid_cfg) == 0 or humanoid_cfg is None:
            self.humanoid_flag = False
        else:
            self.humanoid_flag = True
            humanoid_mesh_fns = humanoid_cfg.get("mesh_fn", {"g1": ""})
            self.humanoid_link_meshes = {}

            for key, mesh_fn in humanoid_mesh_fns.items():
                humanoid_mesh_data = np.load(osp.abspath(mesh_fn))
                names = set([k[:-2] for k in humanoid_mesh_data.files])
                buffer = {}
                for name in names:
                    buffer[name] = trimesh.Trimesh(
                        vertices=humanoid_mesh_data[f"{name}_v"],
                        faces=humanoid_mesh_data[f"{name}_f"],
                        vertex_colors=humanoid_mesh_data[f"{name}_c"].astype(np.float32)
                        / 255.0,
                    )
                logging.info(f"Loaded humanoid mesh {mesh_fn} with names {names}")
                self.humanoid_link_meshes[key] = buffer

        self.humanhumanoid_viz_cfg = self.pl.cfg.get("visualize", {}).get(
            "humanhumanoid", {}
        )

    def __make_dummy_frames__(self, T, H=512, W=512, color=BACKGROUND_COLOR):
        """Create a list of blank placeholder frames.

        Args:
            T (int): Number of frames to generate.
            H (int): Frame height.
            W (int): Frame width.
            color (tuple): RGBA background color (values in [0, 1]).

        Returns:
            list[np.ndarray]: List of T blank frames, each (H, W, 3).

        Example::

            blanks = self.__make_dummy_frames__(10, H=480, W=640)
        """
        dummy = (
            np.ones((H, W, 3), dtype=np.float32) * np.array(color[:3])[None, None, :]
        ) * 255.0
        return [dummy for t in range(T)]

    def viz_human(
        self,
        T_wl,
        viz_T_cam,
        _n,
        uid,
        gid,
        human_uid,
        human_gid,
        beta=None,
        viz_mask=None,
        H=512,
        W=512,
    ):
        """Render human body mesh frames for one or more humans.

        Animates SMPL/SMPLX body meshes via ``animate_with_Q``, handling
        local/global factored poses and multi-person coloring.

        Args:
            T_wl (torch.Tensor): World-to-link transforms (T, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera-to-world pose (4, 4).
            _n (int or list): Seen-step boundary for coloring.
            uid (torch.Tensor): Per-point UID tensor.
            gid (torch.Tensor): Per-point group-ID tensor.
            human_uid (list): Per-human list of UID lists.
            human_gid (list): Per-human list of group IDs.
            beta (list, optional): Per-human SMPL shape parameters.
            viz_mask (torch.Tensor, optional): Visibility mask (T, P).
            H (int): Output frame height.
            W (int): Output frame width.

        Returns:
            list[np.ndarray]: Rendered frames, one per timestep.

        Example::

            frames = viz.viz_human(T_wl, cam, 10, uid, gid, h_uid, h_gid)
        """
        if viz_mask is None:
            viz_mask = torch.ones(T_wl.shape[:2], dtype=torch.bool).to(T_wl.device)
        if human_uid is None:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)

        # make vertex lists
        VTX = []
        viz_human_mask = torch.zeros_like(viz_mask[0])
        for _idx, _gid in enumerate(human_gid):
            if human_uid[_idx] is None:
                continue
            human_mask = self.pl.model.get_type_mask(uid, human_uid[_idx])
            group_mask = self.pl.model.get_type_mask(gid, [_gid])
            human_mask = human_mask & group_mask
            if human_mask.sum() == 0:
                continue
            viz_human_mask = viz_human_mask | human_mask
            human_Q = T_wl[:, human_mask]
            if self.humanhumanoid_viz_cfg.get("local_global_factor_flag", False):
                _ref_Q = human_Q[:, :1].float()
                _local_Q = human_Q[:, 1:].float()
                _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q, _local_Q)
                human_Q = torch.cat([_ref_Q, _local_Q], dim=1)
            elif self.humanhumanoid_viz_cfg.get(
                "local_global_trans_factor_flag", False
            ):
                _ref_Q = human_Q[:, :1].float()
                _ref_Q_t = _ref_Q[..., :3, 3]
                _local_Q = human_Q[:, 1:].float()
                _local_Q[..., :3, 3] = _local_Q[..., :3, 3] + _ref_Q_t
                human_Q = torch.cat([_ref_Q, _local_Q], dim=1)
            elif self.pl.model.hand_attached_to_wrist_flag:
                # _human_uid = uid[human_mask]
                human_Q = self.pl.model.convert_smplx_hand_param(
                    human_Q, to_local=False
                )
            if beta[_idx] is None:
                beta[_idx] = torch.zeros(self.human_model.num_betas).to(human_Q)
            else:
                if beta[_idx].ndim == 2:
                    beta[_idx] = beta[_idx][0]
            human_v_list = []
            for _t in range(human_Q.shape[0]):
                human_v, _ = self.human_model.animate_with_Q(
                    shape_params=beta[_idx][None, : self.human_model.num_betas],
                    Q=human_Q[None, _t],
                )
                human_v_list.append(human_v)
            human_v_list = torch.cat(human_v_list, dim=0).cpu()  # T,V,3
            VTX.append(human_v_list)
        if len(VTX) == 0:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)
        logging.info(f"Viz {len(VTX)} humans")

        viz_mask = viz_mask[:, viz_human_mask]
        viz_mask_t = viz_mask.any(dim=-1)

        human_f = self.human_model.faces.to(human_v_list)

        human_viz = []
        for _t in tqdm(range(human_Q.shape[0])):
            if isinstance(_n, list):
                indicator_c = PRED_COLOR[:3] if _t not in _n else SEEN_COLOR[:3]
            else:
                indicator_c = PRED_COLOR[:3] if _t >= _n else SEEN_COLOR[:3]
            if viz_mask_t[_t]:
                # make mesh
                cur_mesh = []
                for vi, v in enumerate(VTX):
                    _v = v[_t].cpu().numpy()
                    if self.humanhumanoid_viz_cfg.get("loc_stride", -1.0) > 0:
                        # manually set a location
                        stride = self.humanhumanoid_viz_cfg.get("loc_stride", -1.0)
                        x_offset = (vi - (len(VTX) - 1) / 2.0) * stride
                        _v[:, 0] = _v[:, 0] + x_offset
                    _f = human_f.cpu().numpy()
                    _c = self.pl.color_plate[vi % len(self.pl.color_plate)]
                    cur_mesh.append(Trimesh(vertices=_v, faces=_f, vertex_colors=_c))
                cur_mesh = trimesh.util.concatenate(cur_mesh)
                h_rgb = visualize_a_mesh(
                    cam_T_wc=viz_T_cam.cpu().numpy(),
                    mesh=cur_mesh,
                    background_color=BACKGROUND_COLOR,
                    lighting=True,
                    ground_color=GROUND_COLOR,
                    width=W,
                    height=H,
                    ground_size=self.humanhumanoid_viz_cfg.get("ground_size", 20),
                    ground_grid_size=self.humanhumanoid_viz_cfg.get(
                        "ground_grid_size", 40
                    ),
                    ground_z=self.humanhumanoid_viz_cfg.get("ground_z", 0.0),
                )
                h_rgb[:10, ..., :3] = np.asarray(indicator_c)[:3] * 255.0
                human_viz.append(h_rgb)
            else:
                human_viz.append(
                    (
                        np.ones((H, W, 3), dtype=np.float32)
                        * np.array(GROUND_COLOR[:3])[None, None, :]
                    )
                    * 255.0
                )
        return human_viz

    def viz_humanoid(
        self,
        T_wl,
        viz_T_cam,
        _n,
        uid,
        gid,
        # grouped
        robot_uid,
        robot_gid,
        robot_link_names,
        robot_type,
        viz_mask=None,
        # keep_color_ratio=0.5,
        robot_alpha=1.0,
        H=512,
        W=512,
    ):
        """Render humanoid robot mesh frames for one or more robots.

        Assembles per-link meshes from ``self.humanoid_link_meshes``,
        applies the predicted transforms, and handles local/global
        factored poses and multi-robot scenes.

        Args:
            T_wl (torch.Tensor): World-to-link transforms (T, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera-to-world pose (4, 4).
            _n (int or list): Seen-step boundary for coloring.
            uid (torch.Tensor): Per-point UID tensor.
            gid (torch.Tensor): Per-point group-ID tensor.
            robot_uid (list): Per-robot list of UID lists.
            robot_gid (list): Per-robot list of group IDs.
            robot_link_names (list): Per-robot list of link name lists.
            robot_type (list): Per-robot type strings (e.g. ``"g1"``).
            viz_mask (torch.Tensor, optional): Visibility mask (T, P).
            robot_alpha (float): Mesh alpha for robot links.
            H (int): Output frame height.
            W (int): Output frame width.

        Returns:
            list[np.ndarray]: Rendered frames, one per timestep.

        Example::

            frames = viz.viz_humanoid(T_wl, cam, 10, uid, gid,
                r_uid, r_gid, r_names, r_type)
        """
        if viz_mask is None:
            viz_mask = torch.ones(T_wl.shape[:2], dtype=torch.bool).to(T_wl.device)
        frames = []
        # return self.__make_dummy_frames__(T_wl.shape[0])

        Q_LIST, TYPE_LIST, LINK_NAME_LIST, VIZ_MASK_LIST = [], [], [], []
        GID_LIST = []
        for _idx, _gid in enumerate(robot_gid):
            if robot_uid[_idx] is None:
                continue
            robot_mask = self.pl.model.get_type_mask(uid, robot_uid[_idx])
            group_mask = self.pl.model.get_type_mask(gid, [_gid])
            robot_mask = robot_mask & group_mask
            if robot_mask.sum() == 0:
                continue
            _robot_Q = T_wl[:, robot_mask]
            if self.humanhumanoid_viz_cfg.get("local_global_factor_flag", False):
                _ref_Q = _robot_Q[:, :1].float()
                _local_Q = _robot_Q[:, 1:].float()
                _local_Q = torch.einsum("...ij,...jk->...ik", _ref_Q, _local_Q)
                _robot_Q = torch.cat([_ref_Q, _local_Q], dim=1)
            elif self.humanhumanoid_viz_cfg.get(
                "local_global_trans_factor_flag", False
            ):
                _ref_Q = _robot_Q[:, :1].float()
                _ref_Q_t = _ref_Q[..., :3, 3]
                _local_Q = _robot_Q[:, 1:].float()
                _local_Q[..., :3, 3] = _local_Q[..., :3, 3] + _ref_Q_t
                _robot_Q = torch.cat([_ref_Q, _local_Q], dim=1)
            Q_LIST.append(_robot_Q)
            assert int(robot_mask.sum()) == len(robot_link_names[_idx])
            TYPE_LIST.append(robot_type[_idx])
            LINK_NAME_LIST.append(robot_link_names[_idx])
            VIZ_MASK_LIST.append(viz_mask[:, robot_mask])
            GID_LIST.append(_gid)
        if len(Q_LIST) == 0:
            return self.__make_dummy_frames__(T_wl.shape[0], H=H, W=W)

        for t in tqdm(range(len(Q_LIST[0]))):
            cur_mesh = []
            # for Q, robot_type, link_names, viz_mask in zip(Q_LIST, TYPE_LIST, LINK_NAME_LIST, VIZ_MASK_LIST):
            for ri in range(len(Q_LIST)):
                Q = Q_LIST[ri]
                robot_type = TYPE_LIST[ri].lower()
                link_names = LINK_NAME_LIST[ri]
                viz_mask = VIZ_MASK_LIST[ri]
                # gid = GID_LIST[ri]
                # c = [it * 255 for it in HH_COLORPLATE[gid % len(HH_COLORPLATE)]]
                for i in range(len(link_names)):
                    if not viz_mask[t, i]:
                        continue
                    if link_names[i] not in self.humanoid_link_meshes[robot_type]:
                        continue
                    humanoid_mesh = deepcopy(
                        self.humanoid_link_meshes[robot_type][link_names[i]]
                    )
                    # try:
                    #     vc = humanoid_mesh.visual.vertex_colors
                    #     _vc = vc[..., :3] * keep_color_ratio + np.array(
                    #         c, dtype=np.uint8
                    #     )[None] * (1 - keep_color_ratio)
                    #     vc[..., :3] = _vc
                    #     vc[..., 3] = int(robot_alpha * 255)
                    #     humanoid_mesh.visual.vertex_colors = vc.astype(np.uint8)
                    # except:
                    #     pass
                    transform = Q[t, i]
                    if self.humanhumanoid_viz_cfg.get("loc_stride", -1.0) > 0:
                        # manually set a location
                        stride = self.humanhumanoid_viz_cfg.get("loc_stride", -1.0)
                        x_offset = (ri - (len(Q_LIST) - 1) / 2.0) * stride
                        transform = transform.clone()
                        transform[0, 3] = transform[0, 3] + x_offset
                    humanoid_mesh.apply_transform(transform.cpu().numpy())
                    cur_mesh.append(humanoid_mesh)
            if len(cur_mesh) > 0:
                cur_mesh = trimesh.util.concatenate(cur_mesh)
                if isinstance(_n, list):
                    indicator_c = PRED_COLOR[:3] if t not in _n else SEEN_COLOR[:3]
                else:
                    indicator_c = PRED_COLOR[:3] if t >= _n else SEEN_COLOR[:3]
                _fr = visualize_a_mesh(
                    cam_T_wc=viz_T_cam.cpu().numpy(),
                    mesh=cur_mesh,
                    background_color=BACKGROUND_COLOR,
                    lighting=True,
                    ground_color=GROUND_COLOR,
                    width=W,
                    height=H,
                    ground_size=self.humanhumanoid_viz_cfg.get("ground_size", 20),
                    ground_grid_size=self.humanhumanoid_viz_cfg.get(
                        "ground_grid_size", 40
                    ),
                    ground_z=self.humanhumanoid_viz_cfg.get("ground_z", 0.0),
                )
                _fr[:10, ..., :3] = np.asarray(indicator_c)[:3] * 255.0
            else:
                _fr = (
                    np.ones((H, W, 3), dtype=np.float32)
                    * np.array(GROUND_COLOR[:3])[None, None, :]
                ) * 255.0
            frames.append(_fr)
        return frames

    def _viz_hh_subroutine(
        self,
        gen_trace,
        gen_context,
        viz_T_cam,
        batch,
        phase,
        bid,
        batch_idx,
        viz_seen_step,
        viz_trace_mask=None,
        viz_ctx_mask=None,
        human_viz_name="",
        humanoid_viz_name="",
        compose_viz_name="",
        context_viz_name="",
        log_human_flag=True,
        log_humanoid_flag=True,
        log_compose_flag=True,
    ):
        """Internal routine that renders and logs human/humanoid visualizations.

        Renders human body, humanoid robot, and blended overlay videos as
        well as context grid images, then logs all via ``self.pl.logger``.

        Args:
            gen_trace (torch.Tensor or None): Generated trace (T, P, 4, 4).
            gen_context (torch.Tensor or None): Generated context (C, P, 4, 4).
            viz_T_cam (torch.Tensor): Camera pose.
            batch (dict): Full training batch.
            phase (str): ``"train"`` or ``"val"``.
            bid (int): Batch item index.
            batch_idx (int): Global batch index for logging.
            viz_seen_step (int or list): Boundary between seen/predicted frames.
            viz_trace_mask (torch.Tensor, optional): Visibility mask for trace.
            viz_ctx_mask (torch.Tensor, optional): Visibility mask for context.
            human_viz_name (str): Logger tag for human video.
            humanoid_viz_name (str): Logger tag for humanoid video.
            compose_viz_name (str): Logger tag for blended video.
            context_viz_name (str): Logger tag prefix for context images.
            log_human_flag (bool): Whether to log human video.
            log_humanoid_flag (bool): Whether to log humanoid video.
            log_compose_flag (bool): Whether to log blended overlay video.

        Example::

            self._viz_hh_subroutine(
                gen_trace, gen_context, cam, batch, "val", 0, 0, 10,
                human_viz_name="GT_human_val",
            )
        """
        H = self.humanhumanoid_viz_cfg.get("H", 480)
        W = self.humanhumanoid_viz_cfg.get("W", 960)
        prompt_uids = batch["trace_uid"][bid]
        prompt_gid = batch["trace_group_id"][bid]
        caption = batch["text"][bid] if batch["text"] is not None else None

        human_viz = None
        humanoid_viz = None
        context_human_viz = None
        context_humanoid_viz = None

        extra_dict = {k: v[bid] for k, v in batch["extra"].items()}

        if self.human_flag:  # * extract human
            if gen_trace is not None:
                human_viz = self.viz_human(
                    gen_trace,
                    viz_T_cam,
                    viz_seen_step,
                    uid=prompt_uids,
                    gid=prompt_gid,
                    # these are per-human grouped
                    human_uid=extra_dict["human_uids"],
                    human_gid=extra_dict["gid"],
                    beta=extra_dict["beta"] if "beta" in extra_dict else None,
                    viz_mask=viz_trace_mask,
                    H=H,
                    W=W,
                )
                if log_human_flag:
                    self.pl.logger.log_video(
                        phase,
                        human_viz_name,
                        torch.from_numpy(np.stack(human_viz, 0))[..., :3]
                        .float()
                        .permute(0, 3, 1, 2)
                        / 255.0,
                        self.pl.global_step,
                        batch_idx,
                        caption=caption,
                    )
            if gen_context is not None:
                context_human_viz = self.viz_human(
                    gen_context,
                    viz_T_cam,
                    len(gen_context),
                    uid=prompt_uids,
                    gid=prompt_gid,
                    human_uid=extra_dict["human_uids"],
                    human_gid=extra_dict["gid"],
                    beta=extra_dict["beta"] if "beta" in extra_dict else None,
                    viz_mask=viz_ctx_mask,
                    H=H,
                    W=W,
                )
                # context_human_viz = np.concatenate(context_human_viz, axis=1)
                context_human_viz = make_square_cat_image(context_human_viz)
                context_human_viz = context_human_viz[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"{context_viz_name}_human",
                    context_human_viz[::1, ::1].transpose(2, 0, 1).astype(np.float32)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

        if (
            self.humanoid_flag
            and "humanoid_uids" in batch["extra"].keys()
            and extra_dict["humanoid_uids"] is not None
        ):
            if gen_trace is not None:
                humanoid_viz = self.viz_humanoid(
                    gen_trace,
                    viz_T_cam,
                    viz_seen_step,
                    uid=prompt_uids,
                    gid=prompt_gid,
                    # grouped
                    robot_uid=extra_dict["humanoid_uids"],
                    robot_gid=extra_dict["gid"],
                    robot_link_names=extra_dict["humanoid_link_names"],
                    robot_type=extra_dict["humanoid_type"],
                    viz_mask=viz_trace_mask,
                    H=H,
                    W=W,
                )
                if log_humanoid_flag:
                    self.pl.logger.log_video(
                        phase,
                        humanoid_viz_name,
                        torch.from_numpy(np.stack(humanoid_viz, 0))[..., :3]
                        .float()
                        .permute(0, 3, 1, 2)
                        / 255.0,
                        self.pl.global_step,
                        batch_idx,
                        caption=(
                            batch["text"][bid] if batch["text"] is not None else None
                        ),
                    )
            if gen_context is not None:
                context_humanoid_viz = self.viz_humanoid(
                    gen_context,
                    viz_T_cam,
                    len(gen_context),
                    uid=prompt_uids,
                    gid=prompt_gid,
                    # grouped
                    robot_uid=extra_dict["humanoid_uids"],
                    robot_gid=extra_dict["gid"],
                    robot_link_names=extra_dict["humanoid_link_names"],
                    robot_type=extra_dict["humanoid_type"],
                    viz_mask=viz_ctx_mask,
                    H=H,
                    W=W,
                )
                # context_humanoid_viz = np.concatenate(context_humanoid_viz, axis=1)
                context_humanoid_viz = make_square_cat_image(context_humanoid_viz)
                context_humanoid_viz = context_humanoid_viz[..., :3]
                self.pl.logger.log_image(
                    phase,
                    f"{context_viz_name}_humanoid",
                    context_humanoid_viz[::1, ::1].transpose(2, 0, 1).astype(np.float32)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                )

        if human_viz is not None and humanoid_viz is not None:
            blended_human_robot_viz = [
                it1[..., :3] * 0.5 + it2[..., :3] * 0.5
                for it1, it2 in zip(human_viz, humanoid_viz)
            ]
            if log_compose_flag:
                self.pl.logger.log_video(
                    phase,
                    compose_viz_name,
                    torch.from_numpy(np.stack(blended_human_robot_viz, 0))[..., :3]
                    .float()
                    .permute(0, 3, 1, 2)
                    / 255.0,
                    self.pl.global_step,
                    batch_idx,
                    caption=caption,
                )
        if context_human_viz is not None and context_humanoid_viz is not None:
            blended_cotext_humanoid_viz = (
                0.5 * context_human_viz + 0.5 * context_humanoid_viz
            )
            self.pl.logger.log_image(
                phase,
                f"{context_viz_name}_blended",
                blended_cotext_humanoid_viz[::1, ::1]
                .transpose(2, 0, 1)
                .astype(np.float32)
                / 255.0,
                self.pl.global_step,
                batch_idx,
            )
        return

    def human_humanoid_viz_routine(
        self, step_output, batch, phase, batch_idx, eval_model, lite=True
    ):
        """Top-level visualization routine for human and humanoid domains.

        Identifies the first suitable batch item, then renders GT, x0-pred,
        H2R, R2H, unconditional, in-filling, multi-seed context generation,
        and future-prediction modes (subset controlled by ``lite``).

        Args:
            step_output (dict): Training step outputs containing
                ``x0_pred_trace`` and ``x0_pred_context``.
            batch (dict): Training batch.
            phase (str): ``"train"`` or ``"val"``.
            batch_idx (int): Batch index for logging.
            eval_model: Agent model used for inference.
            lite (bool): If True, skip expensive unconditional and
                in-filling visualizations.

        Example::

            viz.human_humanoid_viz_routine(
                step_output, batch, "val", 0, eval_model, lite=True
            )
        """
        if not (self.human_flag or self.humanoid_flag):
            return

        # identify best bid: prefer items with BOTH human and humanoid
        # so that H2R and R2H viz are both meaningful
        B = len(batch["trace"])
        bid_both = None
        bid_any = None
        for i in range(B):
            has_human = (
                "human_model_type" in batch["extra"]
                and batch["extra"]["human_model_type"][i] is not None
                and "smplx" in batch["extra"]["human_model_type"][i]
            )
            has_humanoid = False
            if (
                "humanoid_type" in batch["extra"]
                and batch["extra"]["humanoid_type"][i] is not None
            ):
                for ht in batch["extra"]["humanoid_type"][i]:
                    if ht is not None:
                        has_humanoid = True
                        break
            if has_human and has_humanoid:
                bid_both = i
                break
            if bid_any is None and (has_human or has_humanoid):
                bid_any = i
        bid = bid_both if bid_both is not None else bid_any
        if bid is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no human or humanoid in any batch item, skip viz"
            )
            return
        if bid_both is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, no batch item has both human+humanoid, using bid={bid} (single-embodiment)"
            )

        viz_T_cam = make_viz_T_cam(self.humanhumanoid_viz_cfg, batch)
        dataset_trace_mask = batch["trace_mask"][bid]  # T,P
        dataset_ctx_mask = batch["context_mask"][bid]  # C,P
        extra_dict = {k: it[bid] for k, it in batch["extra"].items()}

        # * GT
        n_steps = batch["trace_mask"][bid][:, 0].sum()
        self._viz_hh_subroutine(
            gen_trace=batch["trace"][bid, :n_steps],
            gen_context=batch["context"][bid],
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps,
            viz_trace_mask=dataset_trace_mask,  # use here for gt viz
            viz_ctx_mask=dataset_ctx_mask,
            human_viz_name=f"GT_human_{phase}",
            humanoid_viz_name=f"GT_humanoid_{phase}",
            compose_viz_name=f"GT_blended_{phase}",
            context_viz_name=f"GT_context_{phase}",
        )
        # # ! debug
        # return

        if not lite:
            self._viz_hh_subroutine(
                gen_trace=step_output["x0_pred_trace"][bid, :n_steps].float(),
                gen_context=step_output["x0_pred_context"][bid].float(),
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=n_steps,
                viz_trace_mask=dataset_trace_mask,  # use here for gt viz
                viz_ctx_mask=dataset_ctx_mask,
                human_viz_name=f"PREDX0_human_{phase}",
                humanoid_viz_name=f"PREDX0_humanoid_{phase}",
                compose_viz_name=f"PREDX0_blended_{phase}",
                context_viz_name=f"PREDX0_context_{phase}",
            )

        # * One H2R
        _h2r_ok = (
            "human_model_type" in extra_dict
            and extra_dict["human_model_type"] is not None
            and "smplx" in extra_dict["human_model_type"]
        )
        if not _h2r_ok:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, bid={bid}: H2R viz skipped — no smplx human in this batch item"
            )
        if _h2r_ok:
            # find first idx
            _idx = extra_dict["human_model_type"].index("smplx")
            human_mask = eval_model.get_type_mask(
                batch["trace_uid"][bid], extra_dict["human_uids"][_idx]
            )
            human_mask = human_mask & eval_model.get_type_mask(
                batch["trace_group_id"][bid], extra_dict["gid"][_idx]
            )
            trace_known_mask = dataset_trace_mask[None].bool().clone()
            context_known_mask = dataset_ctx_mask[None].bool().clone()
            trace_known_mask[0, :, ~human_mask] = (
                False  # set robot and other human to unknown
            )
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
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=None,
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                human_viz_name=f"H2R_human_gt_{phase}",
                humanoid_viz_name=f"H2R_humanoid_retargeted_{phase}",
                compose_viz_name=f"H2R_blended{phase}",
            )

        # * one R2H
        trace_known_mask = dataset_trace_mask[None].bool().clone()
        context_known_mask = dataset_ctx_mask[None].bool().clone()
        humanoid_mask = None
        if "humanoid_type" in extra_dict and extra_dict["humanoid_type"] is not None:
            for i in range(len(extra_dict["humanoid_type"])):
                if extra_dict["humanoid_type"][i] is not None:
                    humanoid_mask = eval_model.get_type_mask(
                        batch["trace_uid"][bid], extra_dict["humanoid_uids"][i]
                    )  # P
                    humanoid_mask = humanoid_mask & eval_model.get_type_mask(
                        batch["trace_group_id"][bid], extra_dict["gid"][i]
                    )
                    break
        if humanoid_mask is None:
            logging.warning(
                f"phase: {phase}, batch_idx: {batch_idx}, bid={bid}: R2H viz skipped — no humanoid in this batch item"
            )
        if humanoid_mask is not None:
            trace_known_mask[0, :, ~humanoid_mask] = False  # set human to unknown
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
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=None,
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                human_viz_name=f"R2H_human_retargeted_{phase}",
                humanoid_viz_name=f"R2H_humanoid_gt_{phase}",
                compose_viz_name=f"R2H_blended_{phase}",
            )

        # * uncond
        if not lite:
            gen_context, gen_trace = eval_model.inference(
                trace=batch["trace"][bid : bid + 1],
                context=batch["context"][bid : bid + 1],
                trace_known_mask=torch.zeros_like(
                    batch["trace_mask"][bid : bid + 1]
                ).bool(),
                context_known_mask=torch.zeros_like(
                    batch["context_mask"][bid : bid + 1]
                ).bool(),
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
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=gen_context[0],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=0,
                human_viz_name=f"Uncond_human_{phase}",
                humanoid_viz_name=f"Uncond_humanoid_{phase}",
                compose_viz_name=f"Uncond_blended_{phase}",
                context_viz_name=f"Uncond_context_{phase}",
            )

        # * no context filling
        if not lite:
            trace_known_mask = dataset_trace_mask[None].clone().bool()
            n_steps = batch["trace_mask"][bid][:, 0].sum()
            trace_known_mask[0, 1 : n_steps - 1] = False
            _dataset_x_mask = dataset_trace_mask.clone()
            seen_start_end = trace_known_mask[0, 0] | trace_known_mask[0, n_steps - 1]
            _dataset_x_mask[1 : n_steps - 1] = seen_start_end[None]
            gen_context, gen_trace = eval_model.inference(
                trace=batch["trace"][bid : bid + 1],
                context=batch["context"][bid : bid + 1],
                trace_known_mask=trace_known_mask,
                context_known_mask=torch.zeros_like(
                    batch["context_mask"][bid : bid + 1]
                ).bool(),
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
            self._viz_hh_subroutine(
                gen_trace=gen_trace[0, :n_steps],
                gen_context=gen_context[0],
                viz_T_cam=viz_T_cam,
                batch=batch,
                phase=phase,
                bid=bid,
                batch_idx=batch_idx,
                viz_seen_step=[0, n_steps - 1],
                human_viz_name=f"FillingNoCtx_human_{phase}",
                humanoid_viz_name=f"FillingNoCtx_humanoid_{phase}",
                compose_viz_name=f"FillingNoCtx_blended_{phase}",
                context_viz_name=f"FillingNoCtx_context_{phase}",
            )
            # * Gen context different seeds!!
            for seed in [42, 24]:
                gen_context, _ = eval_model.inference(
                    trace=batch["trace"][bid : bid + 1],
                    context=batch["context"][bid : bid + 1],
                    trace_known_mask=dataset_trace_mask[None].bool(),
                    context_known_mask=torch.zeros_like(
                        batch["context_mask"][bid : bid + 1]
                    ).bool(),
                    trace_mask=batch["trace_mask"][bid : bid + 1],
                    context_mask=batch["context_mask"][bid : bid + 1],
                    real_timestep=batch["time"][bid : bid + 1],
                    group_id=batch["trace_group_id"][bid : bid + 1],
                    sem_type=batch["trace_sem_type"][bid : bid + 1],
                    rot_mask=batch["trace_rot_mask"][bid : bid + 1],
                    local_pcl=batch["local_pcl"][bid : bid + 1],
                    local_pcl_mask=batch["local_pcl_mask"][bid : bid + 1],
                    text=[batch["text"][bid]],
                    seed=seed,
                )
                n_steps = batch["trace_mask"][bid][:, 0].sum()
                self._viz_hh_subroutine(
                    gen_trace=None,
                    gen_context=gen_context[0],
                    viz_T_cam=viz_T_cam,
                    batch=batch,
                    phase=phase,
                    bid=bid,
                    batch_idx=batch_idx,
                    viz_seen_step=0,
                    viz_trace_mask=dataset_trace_mask,
                    viz_ctx_mask=torch.ones_like(dataset_ctx_mask),
                    context_viz_name=f"GenContext_{phase}_seed{seed}",
                )

        # * Half future prediction
        trace_known_mask = dataset_trace_mask[None].bool().clone()
        context_known_mask = dataset_ctx_mask[None].bool().clone()
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
        self._viz_hh_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=None,
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=n_steps // 2,
            human_viz_name=f"FuturePred_huaman_seen_half_{phase}",
            humanoid_viz_name=f"FuturePred_humanoid_seen_half_{phase}",
            compose_viz_name=f"FuturePred_blended_seen_half_{phase}",
        )

        # * 1 step future prediction
        trace_known_mask = dataset_trace_mask[None].bool().clone()
        context_known_mask = dataset_ctx_mask[None].bool().clone()
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
        self._viz_hh_subroutine(
            gen_trace=gen_trace[0, :n_steps],
            gen_context=None,
            viz_T_cam=viz_T_cam,
            batch=batch,
            phase=phase,
            bid=bid,
            batch_idx=batch_idx,
            viz_seen_step=1,
            human_viz_name=f"FuturePred_human_seen_1_{phase}",
            humanoid_viz_name=f"FuturePred_humanoid_seen_1_{phase}",
            compose_viz_name=f"FuturePred_blended_seen_1_{phase}",
        )
        return
