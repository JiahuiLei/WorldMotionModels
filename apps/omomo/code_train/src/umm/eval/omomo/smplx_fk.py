"""SMPL-X forward kinematics using the ``smplx`` pip package.

Replaces ``human_body_prior.body_model.BodyModel`` from the OMOMO baseline.

Baseline reference: ``ext/baselines/omomo_release/trainer_full_body_manip_diffusion.py:42-117``
(``run_smplx_model``). We replicate that function's contract exactly:

* Inputs: ``root_trans (BS,T,3)``, ``aa_rot_rep (BS,T,22,3)`` where
  ``[:,:,0]`` is root orientation and ``[:,:,1:22]`` is body; ``betas (BS,16)``;
  ``gender_list`` of length BS with entries ``'male'`` or ``'female'``.
* Pads 30 zero joints on the hand axis to reach 52 total (21 body + 30 hand
  + root) before reshaping for the body model, matching baseline lines 49-50.
* Groups frames by gender, runs each gender through the corresponding model,
  then reorders back to the original batch order via ``cat_idx_map``.
* ``return_joints24=True`` emits baseline joints ``[:22] + [28] + [43]``
  (left/right middle-finger tips, per trainer lines 98-102).

SMPL-X parity knobs
-------------------
``smplx.create`` is called with ``use_pca=False`` and ``flat_hand_mean=True``
so the hand pose is fully specified by the 30 zero axis-angle joints
(15 per hand) and matches the default behaviour of ``BodyModel`` for the
SMPL-X skeleton. ``num_betas=16`` matches the baseline (confirmed at
``ext/baselines/omomo_release/manip/data/hand_foot_dataset.py:240-244``).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import smplx
import torch


class SmplxFK:
    """SMPL-X forward kinematics wrapper replicating OMOMO ``run_smplx_model``.

    Args:
        model_dir: Directory containing ``SMPLX_MALE.npz`` and
            ``SMPLX_FEMALE.npz`` (at minimum). Can also hold
            ``SMPLX_NEUTRAL.npz``.
        device: Torch device string. SMPL-X parameters are moved here.
        num_betas: Number of shape coefficients. Must be 16 for OMOMO parity.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        num_betas: int = 16,
    ) -> None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"SmplxFK: model_dir does not exist: {model_dir}"
            )
        male_npz = os.path.join(model_dir, "SMPLX_MALE.npz")
        female_npz = os.path.join(model_dir, "SMPLX_FEMALE.npz")
        if not os.path.isfile(male_npz):
            raise FileNotFoundError(f"SmplxFK: missing {male_npz}")
        if not os.path.isfile(female_npz):
            raise FileNotFoundError(f"SmplxFK: missing {female_npz}")

        self.device = torch.device(device)
        self.num_betas = num_betas

        # ``smplx.create`` accepts a direct path to a model file — we pass
        # the SMPLX_{MALE,FEMALE}.npz directly to avoid needing the
        # <dir>/smplx/<files> folder structure the "folder mode" wants.
        self.models: Dict[str, torch.nn.Module] = {}
        gender_to_path = {"male": male_npz, "female": female_npz}
        for gender, npz_path in gender_to_path.items():
            m = smplx.create(
                model_path=npz_path,
                model_type="smplx",
                gender=gender,
                num_betas=num_betas,
                use_pca=False,
                flat_hand_mean=True,
                ext="npz",
            ).to(self.device)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.models[gender] = m

        # Faces are identical across male/female topologies for SMPL-X.
        self.faces: np.ndarray = self.models["male"].faces.astype(np.int64)

    @torch.no_grad()
    def run(
        self,
        root_trans: torch.Tensor,
        aa_rot_rep: torch.Tensor,
        betas: torch.Tensor,
        gender_list: List[str],
        return_joints24: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Run SMPL-X FK, replicating OMOMO's ``run_smplx_model``.

        Args:
            root_trans: ``(BS, T, 3)`` translation in world frame.
            aa_rot_rep: ``(BS, T, J, 3)`` axis-angle. ``J`` must be 22 (root
                + 21 body) or 52 (root + 21 body + 30 hand). If 22, 30 zero
                hand joints are appended to match the OMOMO trainer.
            betas: ``(BS, 16)`` shape coefficients.
            gender_list: length-BS list of ``'male'`` / ``'female'``.
            return_joints24: If True, return joints as
                ``[:22] + [28] + [43]`` (body + L/R middle-finger tips),
                shape ``(BS, T, 24, 3)``. Else, ``(BS, T, 22, 3)``.

        Returns:
            ``(joints, verts, faces)`` where:

            * ``joints`` is ``(BS, T, 22 or 24, 3)`` torch tensor.
            * ``verts`` is ``(BS, T, 10475, 3)`` torch tensor.
            * ``faces`` is ``(F, 3)`` int64 numpy array.
        """
        if aa_rot_rep.ndim != 4 or aa_rot_rep.shape[-1] != 3:
            raise ValueError(
                f"aa_rot_rep must be (BS,T,J,3); got {tuple(aa_rot_rep.shape)}"
            )
        bs, num_steps, num_joints, _ = aa_rot_rep.shape
        if num_joints not in (22, 52):
            raise ValueError(
                f"aa_rot_rep expected J in (22, 52); got J={num_joints}"
            )
        if betas.shape != (bs, self.num_betas):
            raise ValueError(
                f"betas must be (BS,{self.num_betas}); got {tuple(betas.shape)}"
            )
        if len(gender_list) != bs:
            raise ValueError(
                f"gender_list must have length BS={bs}; got {len(gender_list)}"
            )

        root_trans = root_trans.to(self.device)
        aa_rot_rep = aa_rot_rep.to(self.device)
        betas = betas.to(self.device)

        # Pad hand pose with 30 zero joints (15 per hand) if needed.
        if num_joints != 52:
            padding_zeros_hand = torch.zeros(
                bs, num_steps, 30, 3, device=self.device, dtype=aa_rot_rep.dtype
            )
            aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2)

        # Flatten time dim.
        aa_flat = aa_rot_rep.reshape(bs * num_steps, -1, 3)  # (B, 52, 3)
        betas_rep = (
            betas[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)
        )  # (B, 16)

        gender_arr = np.asarray(gender_list)[:, np.newaxis].repeat(num_steps, axis=1)
        gender_arr = gender_arr.reshape(-1).tolist()  # length B

        smpl_trans = root_trans.reshape(-1, 3)  # (B, 3)
        smpl_root_orient = aa_flat[:, 0, :]  # (B, 3)
        smpl_pose_body = aa_flat[:, 1:22, :].reshape(-1, 63)  # (B, 63)
        smpl_pose_hand = aa_flat[:, 22:, :].reshape(-1, 90)  # (B, 90)

        B = smpl_trans.shape[0]
        gender_names = ["male", "female"]

        # Split hand pose into (L, R), 45 each.
        left_hand_pose_all = smpl_pose_hand[:, :45]
        right_hand_pose_all = smpl_pose_hand[:, 45:]

        pred_joints_chunks: List[torch.Tensor] = []
        pred_verts_chunks: List[torch.Tensor] = []
        prev_nbidx = 0
        cat_idx_map = np.ones((B,), dtype=np.int64) * -1
        gender_np = np.asarray(gender_arr)

        for gender_name in gender_names:
            mask = gender_np == gender_name
            nbidx = int(np.sum(mask))
            cat_idx_map[mask] = np.arange(
                prev_nbidx, prev_nbidx + nbidx, dtype=np.int64
            )
            prev_nbidx += nbidx
            if nbidx == 0:
                continue

            mask_t = torch.from_numpy(mask).to(self.device)
            g_trans = smpl_trans[mask_t]
            g_root = smpl_root_orient[mask_t]
            g_betas = betas_rep[mask_t]
            g_body = smpl_pose_body[mask_t]
            g_lh = left_hand_pose_all[mask_t]
            g_rh = right_hand_pose_all[mask_t]

            bm = self.models[gender_name]
            zero3 = torch.zeros(nbidx, 3, device=self.device, dtype=g_root.dtype)
            zero_expr = torch.zeros(
                nbidx, bm.num_expression_coeffs, device=self.device, dtype=g_root.dtype
            )
            out = bm(
                betas=g_betas,
                body_pose=g_body,
                global_orient=g_root,
                transl=g_trans,
                left_hand_pose=g_lh,
                right_hand_pose=g_rh,
                jaw_pose=zero3,
                leye_pose=zero3,
                reye_pose=zero3,
                expression=zero_expr,
                return_verts=True,
                return_full_pose=False,
            )
            # ``out.joints`` is (N, 127, 3) for SMPL-X by default (includes
            # landmark points). The first 55 correspond to SMPL-X skeleton
            # joints in the same order as ``BodyModel.Jtr``.
            pred_joints_chunks.append(out.joints[:, :55, :].contiguous())
            pred_verts_chunks.append(out.vertices.contiguous())

        if not pred_joints_chunks:
            raise ValueError(
                "SmplxFK.run: gender_list contained no 'male' or 'female' entries"
            )

        x_joints_all = torch.cat(pred_joints_chunks, dim=0)  # (B', 55, 3)
        x_verts = torch.cat(pred_verts_chunks, dim=0)  # (B', 10475, 3)

        if return_joints24:
            lmiddle_index = 28
            rmiddle_index = 43
            x_joints = torch.cat(
                (
                    x_joints_all[:, :22, :],
                    x_joints_all[:, lmiddle_index : lmiddle_index + 1, :],
                    x_joints_all[:, rmiddle_index : rmiddle_index + 1, :],
                ),
                dim=1,
            )
        else:
            x_joints = x_joints_all[:, :22, :]

        # Reorder back to original batch ordering.
        inv_idx = torch.from_numpy(cat_idx_map).to(self.device)
        x_joints = x_joints[inv_idx]
        x_verts = x_verts[inv_idx]

        j_out = x_joints.reshape(bs, num_steps, -1, 3)
        v_out = x_verts.reshape(bs, num_steps, -1, 3)
        return j_out, v_out, self.faces
