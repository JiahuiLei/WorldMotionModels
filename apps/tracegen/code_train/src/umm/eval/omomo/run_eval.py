"""CLI: run OMOMO-style metrics on paired ``.npz`` predictions.

Usage
-----
Paired directories (Q12 schema — of
``<seq_name>_w<window_idx>.npz`` files::

    python -m umm.eval.omomo.run_eval \
        --gt_dir <dir> --pred_dir <dir> \
        --out metrics.json \
        [--smplx_model_dir data/omomo/raw/smpl_all_models/smplx] \
        [--sdf_folder data/omomo/raw/data/rest_object_sdf_256_npy_files] \
        [--skip_pbody_objects woodchair whitechair largebox largetable plasticbox trashcan] \
        [--device cuda] \
        [--compute_collision]

Schema (per ``.npz``)
---------------------
* ``root_trans``: ``float32 (T,3)``
* ``root_orient_aa``: ``float32 (T,3)``
* ``pose_body_aa``: ``float32 (T,21,3)``
* ``betas``: ``float32 (16,)``
* ``gender``: ``str``
* ``seq_name``: ``str``
* ``window_idx``: ``int32``
* ``actual_len``: ``int32``
* ``T``: ``int32``

Collision (Pbody) is opt-in via ``--compute_collision``. When enabled, the
**GT** ``.npz`` must additionally contain:

* ``obj_trans``: ``float32 (T,3)``
* ``obj_rot_mat``: ``float32 (T,3,3)``
* ``obj_scale``: ``float32 (T,)``
* ``obj_name``: ``str``
* ``obj_verts``: ``float32 (T, No, 3)`` world-frame
* ``obj_faces``: ``int64 (Nf, 3)`` (optional; kept for signature parity)

Output ``metrics.json`` schema
------------------------------
::

    {
      "aggregate": {"cf1": ..., "c_pct": ..., "mpjpe_mm": ..., "pbody_skip_noisy": ...,
                    "pbody_all": ..., "hand_jpe_mm": ..., "mpvpe_mm": ..., "troot_mm": ...,
                    "oroot": ..., "foot_sliding": ...},
      "per_window": [{"stem": ..., "cf1": ..., "c_pct": ..., "mpjpe_mm": ..., ...}, ...],
      "config": {"skip_objects": [...], "version": "..."}
    }
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pytorch3d.transforms import axis_angle_to_matrix

from umm.eval.omomo.collision import compute_collision
from umm.eval.omomo.metrics import (
    compute_foot_floor_height,
    compute_metrics,
    compute_object_trans_rot_metrics,
    determine_floor_height_and_contacts,
)
from umm.eval.omomo.smplx_fk import SmplxFK

PORT_VERSION = "sg04.2026-04-20"

DEFAULT_SKIP_PBODY_OBJECTS: List[str] = [
    "woodchair",
    "whitechair",
    "largebox",
    "largetable",
    "plasticbox",
    "trashcan",
]

logger = logging.getLogger("umm.eval.omomo.run_eval")


def _load_npz(path: str) -> Dict[str, Any]:
    """Load ``.npz`` window file into plain dict, converting scalar arrays."""
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind in ("U", "S", "O") and arr.ndim == 0:
            out[key] = arr.item()
        elif arr.ndim == 0:
            out[key] = arr.item()
        else:
            out[key] = arr
    return out


def _pair_dirs(gt_dir: str, pred_dir: str) -> List[Tuple[str, str, str]]:
    """Return triples ``(stem, gt_path, pred_path)``. Error on mismatch."""
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    gt_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_paths}
    pred_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in pred_paths}
    missing_in_pred = sorted(set(gt_stems) - set(pred_stems))
    missing_in_gt = sorted(set(pred_stems) - set(gt_stems))
    if missing_in_pred or missing_in_gt:
        raise ValueError(
            "run_eval: mismatched .npz stems between gt and pred.\n"
            f"  in gt not in pred: {missing_in_pred[:5]}... ({len(missing_in_pred)} total)\n"
            f"  in pred not in gt: {missing_in_gt[:5]}... ({len(missing_in_gt)} total)"
        )
    return [(s, gt_stems[s], pred_stems[s]) for s in sorted(gt_stems)]


def _aa_to_rot_mat_batch(root_orient_aa: np.ndarray) -> np.ndarray:
    """``(T,3)`` axis-angle -> ``(T,22,3,3)`` with root at index 0, rest = I.

    The baseline ``compute_metrics`` only reads ``[:, 0]``, so the body
    rotations 1..21 are irrelevant for ``rot_dist``. We still construct the
    full tensor to match the function's declared shape.
    """
    T = root_orient_aa.shape[0]
    root_rot = (
        axis_angle_to_matrix(torch.from_numpy(root_orient_aa).float())
        .cpu()
        .numpy()
    )  # (T,3,3)
    out = np.tile(np.eye(3, dtype=np.float32), (T, 22, 1, 1))
    out[:, 0] = root_rot.astype(np.float32)
    return out


def _run_fk(
    fk: SmplxFK,
    npz: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SMPL-X FK on one window. Returns ``(joints24, verts)`` on device."""
    root_trans = torch.from_numpy(npz["root_trans"]).float().unsqueeze(0).to(device)  # (1,T,3)
    root_aa = torch.from_numpy(npz["root_orient_aa"]).float()  # (T,3)
    body_aa = torch.from_numpy(npz["pose_body_aa"]).float()  # (T,21,3)
    aa = torch.cat([root_aa.unsqueeze(1), body_aa], dim=1).unsqueeze(0).to(device)  # (1,T,22,3)
    betas = torch.from_numpy(npz["betas"]).float().unsqueeze(0).to(device)  # (1,16)

    gender = str(npz["gender"])
    if gender not in ("male", "female"):
        # Baseline only accepts male/female; fall back to 'male' for
        # neutral, with an explicit log so the user knows.
        logger.warning(
            "gender=%r not in {male,female}; mapping to 'male' for FK.",
            gender,
        )
        gender = "male"

    joints, verts, _ = fk.run(
        root_trans, aa, betas, [gender], return_joints24=True
    )
    # (1,T,24,3), (1,T,10475,3) -> drop batch dim
    return joints.squeeze(0), verts.squeeze(0)


def evaluate_pair(
    fk: SmplxFK,
    gt_npz: Dict[str, Any],
    pred_npz: Dict[str, Any],
    device: torch.device,
    compute_collision_flag: bool,
    sdf_folder: Optional[str],
) -> Dict[str, float]:
    """Evaluate one (GT, pred) pair -> per-window metrics dict."""
    actual_len = int(gt_npz["actual_len"])
    T = int(gt_npz.get("T", gt_npz["root_trans"].shape[0]))

    gt_joints, gt_verts = _run_fk(fk, gt_npz, device)
    pred_joints, pred_verts = _run_fk(fk, pred_npz, device)

    gt_trans = torch.from_numpy(gt_npz["root_trans"]).float().to(device)
    pred_trans = torch.from_numpy(pred_npz["root_trans"]).float().to(device)
    gt_rot_mat = torch.from_numpy(
        _aa_to_rot_mat_batch(gt_npz["root_orient_aa"])
    ).to(device)
    pred_rot_mat = torch.from_numpy(
        _aa_to_rot_mat_batch(pred_npz["root_orient_aa"])
    ).to(device)

    # Q12 schema compat: derive obj verts on demand from obj_mesh_path + obj_rot/obj_trans
    # if precomputed obj_verts missing. Normalize obj_rot → obj_rot_mat alias.
    if "obj_rot" in gt_npz and "obj_rot_mat" not in gt_npz:
        gt_npz = dict(gt_npz)
        gt_npz["obj_rot_mat"] = gt_npz["obj_rot"]
    # Cache mesh.vertices once — pred_obj_verts derivation reuses them.
    rest_verts_t: Optional[torch.Tensor] = None
    obj_scale_t: Optional[torch.Tensor] = None
    has_obj_verts = "obj_verts" in gt_npz
    if has_obj_verts:
        obj_verts = torch.from_numpy(gt_npz["obj_verts"]).float().to(device)
        obj_faces = gt_npz["obj_faces"] if "obj_faces" in gt_npz else np.zeros(
            (0, 3), dtype=np.int64
        )
    elif "obj_mesh_path" in gt_npz and "obj_rot_mat" in gt_npz and "obj_trans" in gt_npz:
        import trimesh
        from pathlib import Path as _P
        mesh_rel = str(gt_npz["obj_mesh_path"])
        # Object name without extension and without the "_cleaned_simplified.obj" suffix.
        obj_basename = mesh_rel.rsplit("/", 1)[-1].split(".")[0]
        if obj_basename.endswith("_cleaned_simplified"):
            obj_basename = obj_basename[: -len("_cleaned_simplified")]
        elif obj_basename.endswith("_cleaned"):
            obj_basename = obj_basename[: -len("_cleaned")]

        # Prefer CHOIS's pre-baked rest mesh (rest_object_geo/<obj>.ply) when
        # present — these are first-frame transformed + COM-centered, and
        # the eval formula is `obj_rot @ rest + obj_com_pos` (no scale).
        # See `cano_traj_dataset.py:388-444 convert_rest_pose_obj_geometry`
        # and `:446-450 load_object_geometry_w_rest_geo`. This is what
        # CHOIS's native eval consumes — necessary for CF1 parity.
        # Fallback: OMOMO captured-objects raw mesh + the
        # `scale × R × rest + trans` formula.
        # Resolution order:
        # 1. In-repo vendored data (eval/omomo/data/rest_object_geo/) — preferred
        #    for fresh clones.
        # 2. cwd-relative `./data/omomo/raw/...` — legacy behavior for -like
        #    worktrees where the OMOMO raw dump is chdir'd.
        # 3. OMOMO raw captured_objects/<mesh_path> — fallback for full raw dumps.
        # Environment override: OMOMO_EVAL_DATA_ROOT=/abs/path → looks for
        # {root}/rest_object_geo/<obj>.ply first. Set by callers who vendor
        # data outside the default eval/omomo/data/ location.
        _env_root = os.environ.get("OMOMO_EVAL_DATA_ROOT", "")
        _repo_data = _P(__file__).resolve().parent.parent.parent.parent.parent / "data/omomo"
        chois_rest_candidates = []
        if _env_root:
            chois_rest_candidates.append(f"{_env_root}/rest_object_geo/{obj_basename}.ply")
        chois_rest_candidates.extend([
            str(_repo_data / "rest_object_geo" / f"{obj_basename}.ply"),
            f"./data/omomo/raw/data/rest_object_geo/{obj_basename}.ply",
            f"data/omomo/raw/data/rest_object_geo/{obj_basename}.ply",
        ])
        omomo_raw_candidates = [
            mesh_rel,
            f"./data/omomo/raw/data/captured_objects/{mesh_rel}",
            f"data/omomo/raw/data/captured_objects/{mesh_rel}",
        ]
        mesh = None
        use_chois_rest = False
        for cand in chois_rest_candidates:
            if _P(cand).exists():
                mesh = trimesh.load(cand, process=False)
                use_chois_rest = True
                break
        if mesh is None:
            for cand in omomo_raw_candidates:
                if _P(cand).exists():
                    mesh = trimesh.load(cand, process=False)
                    break

        if mesh is None or not hasattr(mesh, "vertices"):
            # LOUD — silently falling back here turns ALL contact/collision
            # metrics (cprec/crec/cf1/c_pct/pbody) to NaN. Surface it so
            # callers notice. Common cause: eval harness running from a cwd
            # where `./data/omomo/raw/data/rest_object_geo/<obj>.ply` does
            # not resolve; fix by chdir-ing into the worktree root that
            # holds the OMOMO raw dump before calling run_eval.
            import warnings
            warnings.warn(
                f"[umm.eval.omomo.run_eval] obj_mesh fallback: no rest_object_geo "
                f"or captured_objects mesh found for '{obj_basename}'. Tried "
                f"{chois_rest_candidates + omomo_raw_candidates}. Contact "
                f"metrics will be NaN for this window. cwd={os.getcwd()}.",
                RuntimeWarning,
                stacklevel=2,
            )
            has_obj_verts = False
            obj_verts = torch.zeros(T, 1, 3, device=device, dtype=torch.float32)
            obj_faces = np.zeros((0, 3), dtype=np.int64)
        else:
            rest_verts = torch.from_numpy(
                np.asarray(mesh.vertices, dtype=np.float32)
            ).to(device)  # (Nv, 3)
            obj_rot_t = torch.from_numpy(
                np.asarray(gt_npz["obj_rot_mat"], dtype=np.float32)
            ).to(device)  # (T, 3, 3)
            obj_trans_t = torch.from_numpy(
                np.asarray(gt_npz["obj_trans"], dtype=np.float32)
            ).to(device)  # (T, 3)
            if "obj_scale" in gt_npz:
                obj_scale_raw = np.asarray(gt_npz["obj_scale"], dtype=np.float32)
                if obj_scale_raw.ndim == 0:
                    obj_scale_t = torch.full(
                        (T,), float(obj_scale_raw), device=device, dtype=torch.float32
                    )
                else:
                    obj_scale_t = torch.from_numpy(obj_scale_raw).to(device)
                    if obj_scale_t.shape[0] < T:
                        pad = torch.full(
                            (T - obj_scale_t.shape[0],),
                            float(obj_scale_t[-1]) if obj_scale_t.numel() else 1.0,
                            device=device,
                            dtype=torch.float32,
                        )
                        obj_scale_t = torch.cat([obj_scale_t, pad])
                    obj_scale_t = obj_scale_t[:T]
            else:
                obj_scale_t = torch.ones(T, device=device, dtype=torch.float32)

            rotated = torch.einsum("tij,nj->tni", obj_rot_t, rest_verts)  # (T, Nv, 3)
            if use_chois_rest:
                # CHOIS formula: rest mesh is pre-transformed + centered.
                # `obj_verts[t] = obj_rot[t] @ rest + obj_com_pos[t]`
                obj_verts = rotated + obj_trans_t[:, None]
            else:
                # OMOMO formula (hand_foot_dataset.py:291).
                obj_verts = obj_scale_t[:, None, None] * rotated + obj_trans_t[:, None]
            obj_faces = (
                np.asarray(mesh.faces, dtype=np.int64)
                if hasattr(mesh, "faces") else np.zeros((0, 3), dtype=np.int64)
            )
            has_obj_verts = True
            rest_verts_t = rest_verts  # cache for pred_obj derivation below
            # Stash flag so the pred-obj branch below knows which formula to use.
            obj_geometry_protocol_chois = use_chois_rest
    else:
        has_obj_verts = False
        obj_verts = torch.zeros(T, 1, 3, device=device, dtype=torch.float32)
        obj_faces = np.zeros((0, 3), dtype=np.int64)

    # CHOIS-eval parity: when the predicted .npz carries the predicted object
    # pose (Q12 v2: pred_obj_trans + pred_obj_rot_mat), derive pred_obj_verts
    # using whichever object-geometry formula was selected for the GT side
    # (CHOIS rest-pose: no scale; OMOMO raw: scale × R × rest + trans). This
    # triggers CHOIS-style 'pred contact vs pred object' semantics in
    # compute_metrics.
    pred_obj_verts: Optional[torch.Tensor] = None
    if (
        has_obj_verts
        and rest_verts_t is not None
        and obj_scale_t is not None
        and "pred_obj_trans" in pred_npz
        and "pred_obj_rot_mat" in pred_npz
    ):
        pred_obj_trans_t = torch.from_numpy(
            np.asarray(pred_npz["pred_obj_trans"], dtype=np.float32)
        ).to(device)  # (T, 3)
        pred_obj_rot_t = torch.from_numpy(
            np.asarray(pred_npz["pred_obj_rot_mat"], dtype=np.float32)
        ).to(device)  # (T, 3, 3)
        pred_rotated = torch.einsum(
            "tij,nj->tni", pred_obj_rot_t, rest_verts_t
        )  # (T, Nv, 3)
        if locals().get("obj_geometry_protocol_chois", False):
            pred_obj_verts = pred_rotated + pred_obj_trans_t[:, None]
        else:
            pred_obj_verts = (
                obj_scale_t[:, None, None] * pred_rotated + pred_obj_trans_t[:, None]
            )

    (
        lhand_jpe,
        rhand_jpe,
        hand_jpe,
        mpvpe,
        mpjpe,
        rot_dist,
        trans_err,
        gt_contact_dist,
        contact_dist,
        gt_foot_sliding_jnts,
        foot_sliding_jnts,
        contact_precision,
        contact_recall,
        contact_acc,
        contact_f1_score,
    ) = compute_metrics(
        gt_verts,
        pred_verts,
        gt_joints,
        pred_joints,
        fk.faces,
        gt_trans,
        pred_trans,
        gt_rot_mat,
        pred_rot_mat,
        obj_verts,
        obj_faces,
        actual_len=actual_len,
        use_joints24=True,
        pred_obj_verts=pred_obj_verts,
    )

    out: Dict[str, float] = {
        "hand_jpe_mm": hand_jpe,
        "lhand_jpe_mm": lhand_jpe,
        "rhand_jpe_mm": rhand_jpe,
        "mpvpe_mm": mpvpe,
        "mpjpe_mm": mpjpe,
        "troot_mm": trans_err,
        "oroot": rot_dist,
        "foot_sliding": foot_sliding_jnts,
        "gt_foot_sliding": gt_foot_sliding_jnts,
        "contact_dist": contact_dist,
        "gt_contact_dist": gt_contact_dist,
        "cprec": contact_precision,
        "crec": contact_recall,
        "cacc": contact_acc,
        "cf1": contact_f1_score,
        # c_pct: classification accuracy of per-frame contact/no-contact,
        # matching baseline `contact_acc` = (TP+TN)/(TP+FP+TN+FN). If InfBaGel
        # Table 5 defines C% differently (e.g. recall on GT-positive frames),
        # bit-parity against baseline eval will surface the drift and
        # this alias must be reconsidered.
        "c_pct": contact_acc,
    }

    # Overwrite contact fields with NaN when obj_verts was missing — the
    # compute_metrics output against zero-object would be a degenerate
    # "no contact" result that silently looks valid.
    if not has_obj_verts:
        for k in ("cf1", "c_pct", "cprec", "crec", "cacc",
                  "contact_dist", "gt_contact_dist"):
            out[k] = float("nan")

    # CHOIS paper Table 1 cols 11-16 — T_s / T_e / T_xy / T_obj / O_obj (object
    # trajectory + rotation) + H_feet (foot-floor gap). All require the GT obj
    # pose + (for object cols) the predicted obj pose from the Q12 v2 fields.
    if (
        "obj_trans" in gt_npz
        and ("obj_rot_mat" in gt_npz or "obj_rot" in gt_npz)
        and "pred_obj_trans" in pred_npz
        and "pred_obj_rot_mat" in pred_npz
    ):
        gt_obj_rot_np = np.asarray(
            gt_npz.get("obj_rot_mat", gt_npz.get("obj_rot")), dtype=np.float32
        )
        gt_obj_trans_np = np.asarray(gt_npz["obj_trans"], dtype=np.float32)
        pr_obj_trans_np = np.asarray(
            pred_npz["pred_obj_trans"], dtype=np.float32
        )
        pr_obj_rot_np = np.asarray(
            pred_npz["pred_obj_rot_mat"], dtype=np.float32
        )
        obj_extra = compute_object_trans_rot_metrics(
            gt_obj_trans_np,
            pr_obj_trans_np,
            gt_obj_rot_np,
            pr_obj_rot_np,
            actual_len=actual_len,
        )
        out.update(obj_extra)  # t_s_cm, t_e_cm, t_xy_cm, t_obj_cm, o_obj
    else:
        for k in ("t_s_cm", "t_e_cm", "t_xy_cm", "t_obj_cm", "o_obj"):
            out[k] = float("nan")

    # H_feet: mean |min-foot-z − floor_height| in cm across valid frames.
    pred_jpos_np = pred_joints.detach().cpu().numpy()
    floor_h = determine_floor_height_and_contacts(pred_jpos_np, fps=30)
    out["h_feet_cm"] = compute_foot_floor_height(
        pred_jpos_np, floor_h, actual_len=actual_len
    )

    # Re-derive per-window gt/pred contact-frame counts (no metrics.py change,
    # keeps bit-parity intact). Uses the same 5cm threshold + joints24 hand
    # indices (22, 23) as compute_metrics for joints24=True. When the pred
    # .npz carries pred_obj_verts (CHOIS Q12 v2), the pred side uses pred-obj
    # for parity with the new compute_metrics signature.
    # These drive the aggregates cf1_gt_contact_only + c_pct_pred.
    if has_obj_verts:
        contact_threh = 0.05  # joints24 threshold; matches metrics.py:331
        ov_gt = obj_verts[:actual_len]  # (L, No, 3)
        ov_pr = (
            pred_obj_verts[:actual_len]
            if pred_obj_verts is not None
            else ov_gt
        )
        gj = gt_joints[:actual_len]
        pj = pred_joints[:actual_len]
        lhand_gt = gj[:, 22, :]
        rhand_gt = gj[:, 23, :]
        lhand_pr = pj[:, 22, :]
        rhand_pr = pj[:, 23, :]

        def _min_dist(hand_t: torch.Tensor, ov: torch.Tensor) -> torch.Tensor:
            d = torch.sqrt(
                ((hand_t[:, None, :] - ov) ** 2).sum(dim=-1)
            )  # (L, No)
            return d.min(dim=1)[0]

        gt_any_contact = (
            (_min_dist(lhand_gt, ov_gt) < contact_threh)
            | (_min_dist(rhand_gt, ov_gt) < contact_threh)
        )
        pr_any_contact = (
            (_min_dist(lhand_pr, ov_pr) < contact_threh)
            | (_min_dist(rhand_pr, ov_pr) < contact_threh)
        )
        out["gt_contact_frames"] = int(gt_any_contact.sum().item())
        out["pred_contact_frames"] = int(pr_any_contact.sum().item())
        out["n_frames_valid"] = int(actual_len)
    else:
        out["gt_contact_frames"] = -1  # sentinel: unknown (no obj data)
        out["pred_contact_frames"] = -1
        out["n_frames_valid"] = int(actual_len)

    # Pbody (optional) - Q12 compat: derive obj_name from seq_name if missing
    derived_obj_name = ""
    if "obj_name" in gt_npz:
        derived_obj_name = str(gt_npz["obj_name"])
    elif "seq_name" in gt_npz:
        # seq_name format: subXX_<object>_<take>
        parts = str(gt_npz["seq_name"]).split("_")
        derived_obj_name = parts[1] if len(parts) >= 2 else ""

    pbody_val: Optional[float] = None
    if compute_collision_flag:
        has_trans = "obj_trans" in gt_npz
        has_rot = "obj_rot_mat" in gt_npz or "obj_rot" in gt_npz
        if not (has_trans and has_rot and derived_obj_name):
            logger.warning(
                "collision: gt window missing obj fields; skipping Pbody (obj_name=%s)",
                derived_obj_name,
            )
        else:
            try:
                obj_trans_t = torch.from_numpy(
                    np.asarray(gt_npz["obj_trans"], dtype=np.float32)
                ).float().to(device)
                rot_key = "obj_rot_mat" if "obj_rot_mat" in gt_npz else "obj_rot"
                obj_rot_t = torch.from_numpy(
                    np.asarray(gt_npz[rot_key], dtype=np.float32)
                ).float().to(device)
                if "obj_scale" in gt_npz:
                    obj_scale_t = torch.from_numpy(
                        np.asarray(gt_npz["obj_scale"], dtype=np.float32)
                    ).float().to(device)
                else:
                    obj_scale_t = torch.ones(T, device=device, dtype=torch.float32)
                pen_percent, pen_loss = compute_collision(
                    pred_verts,
                    fk.faces,
                    obj_verts,
                    obj_faces,
                    derived_obj_name,
                    obj_scale_t,
                    obj_rot_t,
                    obj_trans_t,
                    actual_len=actual_len,
                    sdf_folder=sdf_folder or "./data/omomo/raw/data/rest_object_sdf_256_npy_files",
                )
                pbody_val = pen_percent
                # pen_loss is the mean penetration DEPTH in meters on frames
                # where penetration exceeded the 4cm threshold; 0.0 when no
                # such frames. Surface both meters and millimeters so
                # can compare against paper's Pbody regardless of its unit
                # (paper reports 0.59, unclear cm vs % vs fraction).
                out["pbody_loss_m"] = pen_loss
                out["pbody_loss_mm"] = pen_loss * 1000.0
            except FileNotFoundError as exc:
                logger.warning("collision: SDF file not found: %s", exc)
            except OSError as exc:
                logger.warning("collision: IO error: %s", exc)
    out["pbody"] = pbody_val if pbody_val is not None else float("nan")
    out["obj_name"] = derived_obj_name
    return out


def _aggregate(
    per_window: List[Dict[str, Any]],
    skip_objects: List[str],
) -> Dict[str, Any]:
    """Aggregate per-window metrics.

    For every aggregated field we also emit ``<field>_n_used`` so the
    downstream reader can see how many windows contributed (NaN-valued
    entries are dropped rather than averaged in). Pbody reports both
    ``pbody_all`` and ``pbody_skip_noisy``.
    """
    def _mean_and_n(key: str) -> Tuple[float, int]:
        vals: List[float] = []
        for w in per_window:
            v = w.get(key)
            if not isinstance(v, (int, float)):
                continue
            vf = float(v)
            if np.isnan(vf):
                continue
            vals.append(vf)
        if vals:
            return float(np.mean(vals)), len(vals)
        return float("nan"), 0

    keys = [
        # body / contact
        "cf1", "c_pct", "cprec", "crec", "cacc",
        "mpjpe_mm", "hand_jpe_mm", "mpvpe_mm", "troot_mm", "oroot",
        "foot_sliding",
        # CHOIS paper Table 1 cols 11-16 (object + feet)
        "t_s_cm", "t_e_cm", "t_xy_cm", "t_obj_cm", "o_obj", "h_feet_cm",
    ]
    agg: Dict[str, Any] = {}
    for k in keys:
        m, n = _mean_and_n(k)
        agg[k] = m
        agg[f"{k}_n_used"] = n

    pbody_all_vals: List[float] = []
    pbody_skip_vals: List[float] = []
    skip_set = set(skip_objects)
    for w in per_window:
        v = w.get("pbody", float("nan"))
        if isinstance(v, float) and np.isnan(v):
            continue
        pbody_all_vals.append(float(v))
        if w.get("obj_name", "") not in skip_set:
            pbody_skip_vals.append(float(v))
    agg["pbody_all"] = (
        float(np.mean(pbody_all_vals)) if pbody_all_vals else float("nan")
    )
    agg["pbody_all_n_used"] = len(pbody_all_vals)
    agg["pbody_skip_noisy"] = (
        float(np.mean(pbody_skip_vals)) if pbody_skip_vals else float("nan")
    )
    agg["pbody_skip_noisy_n_used"] = len(pbody_skip_vals)

    # Mean penetration DEPTH (mm) aggregate — paper's Pbody column may be in
    # centimeters of depth rather than a fraction. Mirror the all-vs-skip-noisy
    # split so has a direct comparison.
    pbody_mm_all: List[float] = []
    pbody_mm_skip: List[float] = []
    for w in per_window:
        v = w.get("pbody_loss_mm")
        if not isinstance(v, (int, float)):
            continue
        vf = float(v)
        if np.isnan(vf):
            continue
        pbody_mm_all.append(vf)
        if w.get("obj_name", "") not in skip_set:
            pbody_mm_skip.append(vf)
    agg["pbody_loss_mm_all"] = (
        float(np.mean(pbody_mm_all)) if pbody_mm_all else float("nan")
    )
    agg["pbody_loss_mm_all_n_used"] = len(pbody_mm_all)
    agg["pbody_loss_mm_skip_noisy"] = (
        float(np.mean(pbody_mm_skip)) if pbody_mm_skip else float("nan")
    )
    agg["pbody_loss_mm_skip_noisy_n_used"] = len(pbody_mm_skip)

    # cf1_gt_contact_only: mean cf1 over windows whose GT has >= 1 contact
    # frame (by the compute_metrics 5cm threshold). Drops the zero-denom
    # branch penalty from "no-contact GT" windows. This aligns with the
    # common HOI-eval aggregator (CHOIS/InfBaGel reproductions on issue #10).
    cf1_gt_only_vals: List[float] = []
    for w in per_window:
        cf1 = w.get("cf1")
        gt_contact_frames = w.get("gt_contact_frames", -1)
        if not isinstance(cf1, (int, float)) or np.isnan(float(cf1)):
            continue
        if gt_contact_frames is None or gt_contact_frames <= 0:
            continue
        cf1_gt_only_vals.append(float(cf1))
    agg["cf1_gt_contact_only"] = (
        float(np.mean(cf1_gt_only_vals)) if cf1_gt_only_vals else float("nan")
    )
    agg["cf1_gt_contact_only_n_used"] = len(cf1_gt_only_vals)

    # c_pct_pred: mean per-window fraction of valid frames where pred
    # reports any-hand contact. Matches the simpler "pred-positive
    # fraction" reading that paper C% may use.
    c_pct_pred_vals: List[float] = []
    for w in per_window:
        pred_cf = w.get("pred_contact_frames", -1)
        n_valid = w.get("n_frames_valid", 0)
        if pred_cf is None or pred_cf < 0 or not n_valid:
            continue
        c_pct_pred_vals.append(float(pred_cf) / float(n_valid))
    agg["c_pct_pred"] = (
        float(np.mean(c_pct_pred_vals)) if c_pct_pred_vals else float("nan")
    )
    agg["c_pct_pred_n_used"] = len(c_pct_pred_vals)

    agg["n_windows"] = len(per_window)
    return agg


def run_eval(
    gt_dir: str,
    pred_dir: str,
    out: str,
    smplx_model_dir: str,
    sdf_folder: Optional[str] = None,
    skip_pbody_objects: Optional[List[str]] = None,
    device: str = "cuda",
    compute_collision_flag: bool = False,
) -> Dict[str, Any]:
    """Programmatic entry point. Returns the metrics dict and writes to ``out``."""
    if skip_pbody_objects is None:
        skip_pbody_objects = DEFAULT_SKIP_PBODY_OBJECTS

    torch_device = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")

    fk = SmplxFK(smplx_model_dir, device=str(torch_device))

    pairs = _pair_dirs(gt_dir, pred_dir)
    per_window: List[Dict[str, Any]] = []
    for stem, gt_p, pred_p in pairs:
        gt_npz = _load_npz(gt_p)
        pred_npz = _load_npz(pred_p)
        metrics = evaluate_pair(
            fk,
            gt_npz,
            pred_npz,
            torch_device,
            compute_collision_flag,
            sdf_folder,
        )
        metrics["stem"] = stem
        per_window.append(metrics)

    aggregate = _aggregate(per_window, skip_pbody_objects)

    out_dict: Dict[str, Any] = {
        "aggregate": aggregate,
        "per_window": per_window,
        "config": {
            "skip_objects": list(skip_pbody_objects),
            "version": PORT_VERSION,
            "compute_collision": bool(compute_collision_flag),
            "smplx_model_dir": smplx_model_dir,
            "sdf_folder": sdf_folder,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(out_dict, fh, indent=2, default=_json_default)
    return out_dict


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OMOMO-style metrics on paired .npz predictions.",
    )
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--smplx_model_dir",
        default="data/omomo/raw/smpl_all_models/smplx",
        help="Dir with SMPLX_{MALE,FEMALE}.npz. Default matches the OMOMO data layout.",
    )
    p.add_argument(
        "--sdf_folder",
        default="data/omomo/raw/data/rest_object_sdf_256_npy_files",
    )
    p.add_argument(
        "--skip_pbody_objects",
        nargs="*",
        default=DEFAULT_SKIP_PBODY_OBJECTS,
        help="Object names excluded from pbody_skip_noisy aggregate.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--compute_collision",
        action="store_true",
        help="Compute Pbody. Requires obj_* fields in GT .npz and SDF files.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_eval(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        out=args.out,
        smplx_model_dir=args.smplx_model_dir,
        sdf_folder=args.sdf_folder,
        skip_pbody_objects=list(args.skip_pbody_objects),
        device=args.device,
        compute_collision_flag=args.compute_collision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
