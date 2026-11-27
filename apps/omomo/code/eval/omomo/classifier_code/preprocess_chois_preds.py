"""Preprocess CHOIS K=1 prediction NPZs + OMOMO GT test windows into joint positions.

Uses OMOMO's OWN FK pipeline (see `chois_release/manip/data/cano_traj_dataset.py`
`process_window_data`) rather than SMPL-H BodyModel LBS. This is critical
for paper-comparable FID / R_prec: GT's `motion[:, :72]` was produced via

    curr_seq_local_jpos[:, 0, :] = window_root_trans - trans2joint
    curr_seq_local_jpos[:, 1:, :] = rest_human_offsets[1:, :]
    jpos = quat_fk_torch(axis_angle_to_quat(pose), curr_seq_local_jpos)

PLUS a `move_to_zero_trans` step that zeros the first-frame pelvis XY. Using
SMPL-H BodyModel LBS produces subject-betas-scaled bones that differ from
GT by ≈2-3 cm at hand joints — a systematic bias injected into every
pred-vs-GT embedding.

The fix here: for each pred, look up the matching GT window's
`rest_human_offsets` (24,3) and `trans2joint` (3,), replicate OMOMO's FK
exactly on pred's (root_trans, root_orient_aa, pose_body_aa).

Runs in `omomo_env` (need pytorch3d.transforms which CHOIS uses).
If pytorch3d unavailable, falls back to a numpy quat_fk implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from os.path import join as pjoin
from typing import Any

import joblib
import numpy as np
import torch


# --- Local FK implementation (mirrors `chois_release/manip/data/cano_traj_dataset.py`). ---

def _axis_angle_to_quat_np(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle → (..., 4) quaternion [w, x, y, z]. Pure numpy."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.where(small, np.array([1.0, 0.0, 0.0]), aa / np.where(small, 1.0, angle))
    half = 0.5 * angle
    w = np.cos(half)
    xyz = np.sin(half) * axis
    q = np.concatenate([w, xyz], axis=-1)
    return q


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product on [..., 4] arrays. Quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)
    return out


def _quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (..., 3) by unit quaternion q (..., 4) [w, x, y, z]."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * np.cross(xyz, v, axis=-1)
    out = v + w * t + np.cross(xyz, t, axis=-1)
    return out


def _get_smpl_parents_24(smplh_kintree: np.ndarray) -> np.ndarray:
    """Replicates CHOIS `get_smpl_parents(use_joints24=True)`.

    kintree[0, :23] + [kintree[0][37]] → 24 parents, with index 0 set to -1.
    """
    parents = smplh_kintree[0, :23].astype(np.int64).copy()
    parents[0] = -1
    parents = np.concatenate([parents, [int(smplh_kintree[0, 37])]])
    return parents  # shape (24,)


def _quat_fk_np(lquat: np.ndarray, lpos: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Forward-kinematics in quaternion form.

    Args:
      lquat: (..., J, 4) local rotations (per-joint, relative to parent)
      lpos:  (..., J, 3) local positions — position offsets from parent
             (root's slot contains world-space root trans)
      parents: (J,) parent indices; root has -1.

    Returns:
      gpos: (..., J, 3) world-space joint positions.
    """
    J = parents.shape[0]
    lrot_quat = [lquat[..., 0:1, :]]
    gpos = [lpos[..., 0:1, :]]  # root is already world
    for j in range(1, J):
        p = parents[j]
        # Rotate local offset into world using accumulated parent rotation.
        rotated = _quat_apply_np(lrot_quat[p], lpos[..., j : j + 1, :])
        gpos.append(rotated + gpos[p])
        # Accumulate rotation.
        if j < lquat.shape[-2]:
            lrot_quat.append(_quat_mul_np(lrot_quat[p], lquat[..., j : j + 1, :]))
    return np.concatenate(gpos, axis=-2)


_KINTREE_CACHE: dict[str, np.ndarray] = {}


def _load_smplh_kintree(smplh_dir: str) -> np.ndarray:
    """Load `kintree_table` from any SMPL-H gender NPZ (same across male/female)."""
    if "kt" in _KINTREE_CACHE:
        return _KINTREE_CACHE["kt"]
    for g in ("male", "female", "neutral"):
        p = pjoin(smplh_dir, g, "model.npz")
        if os.path.isfile(p):
            data = np.load(p)
            kt = data["kintree_table"].astype(np.int64)
            _KINTREE_CACHE["kt"] = kt
            return kt
    raise FileNotFoundError(f"No SMPL-H model.npz under {smplh_dir}")


def _fk_pred_to_omomo_jpos(
    pred: dict, gt_entry: dict, parents: np.ndarray,
) -> np.ndarray:
    """Replicate OMOMO's `process_window_data` FK on a pred.

    Uses:
      - pred.root_trans (T,3), pred.root_orient_aa (T,3), pred.pose_body_aa (T,21,3)
      - gt_entry.rest_human_offsets (24,3), gt_entry.trans2joint (3,)
    Applies:
      - curr_seq_local_jpos[:, 0, :] = root_trans - trans2joint
      - curr_seq_local_jpos[:, 1:, :] = rest_human_offsets[1:, :]
      - quat_fk(axis_angle_to_quat(root + body), local_jpos) → jpos
      - move_to_zero_trans: subtract first-frame jpos[0, 0, [x, y]] (keep z).
    """
    T = int(pred["root_trans"].shape[0])
    J = 24
    # Axis-angle per joint: root orient (T,3) + pose_body (T,21,3) + zeros for hand roots.
    aa = np.zeros((T, J, 3), dtype=np.float32)
    aa[:, 0, :] = pred["root_orient_aa"].astype(np.float32)
    aa[:, 1:22, :] = pred["pose_body_aa"].astype(np.float32)
    # Hand joints 22, 23 (left/right middle-finger MCP) get zero rotation
    # (they're not predicted by CHOIS and are treated as rigid in OMOMO).

    lquat = _axis_angle_to_quat_np(aa)  # (T, 24, 4)

    rho = np.asarray(gt_entry["rest_human_offsets"], dtype=np.float32)  # (24, 3)
    t2j = np.asarray(gt_entry["trans2joint"], dtype=np.float32)  # (3,)

    # OMOMO's move_to_zero_trans acts on root_trans BEFORE FK:
    #   move_to_zero = root_trans[0].copy(); move_to_zero[2] = 0
    #   root_trans -= move_to_zero
    # This leaves pelvis_at_frame_0 = root_trans[0] - trans2joint = -trans2joint[:2]
    # plus z = root_trans[0, 2] - trans2joint[2].
    root_trans = pred["root_trans"].astype(np.float32).copy()  # (T, 3)
    move_to_zero = root_trans[0].copy()
    move_to_zero[2] = 0.0
    root_trans = root_trans - move_to_zero[None]

    local_jpos = np.broadcast_to(rho[None], (T, J, 3)).copy()  # (T, 24, 3)
    local_jpos[:, 0, :] = root_trans - t2j[None]

    gpos = _quat_fk_np(lquat, local_jpos, parents)  # (T, 24, 3)
    return gpos.astype(np.float32)


def _build_seq_window_index(test_pickle: dict[Any, Any]) -> dict[int, int]:
    """Pass-through: CHOIS pred `window_idx` IS the integer key in the GT
    test pickle (verified: sub16_clothesstand_000_w0 → key 0; _w5 → key 5 with
    seq_name matching). Seq-name consistency is enforced at lookup time — a
    RuntimeError is raised on any mismatch, so a wrong mapping cannot cause
    silent drops."""
    return {int(k): int(k) for k in test_pickle.keys()}


def _fk_sanity_report(jpos: np.ndarray) -> dict:
    """Produce bone-length statistics for a single (T, 24, 3) sample.

    CHOIS 24-joint layout (from OMOMO process_window_data with rest_offsets):
    body [0..21] + L-middle-finger MCP (joint 22) + R-middle-finger MCP (joint 23).
    Expected bone lengths with rest_offsets (subject-averaged):
      pelvis→spine1  ~0.12 m
      pelvis→head    ~0.60 m
      L-wrist→L-mid  ~0.11 m
      R-wrist→R-mid  ~0.11 m
    """
    def _d(j1, j2):
        return float(np.linalg.norm(jpos[:, j1] - jpos[:, j2], axis=-1).mean())
    return {
        "pelvis_to_spine1 (~0.12m)": _d(0, 3),
        "pelvis_to_head (~0.6m)": _d(0, 15),
        "spine1_to_spine2 (~0.1m)": _d(3, 6),
        "L_wrist_to_L_mid_MCP (~0.11m)": _d(20, 22),
        "R_wrist_to_R_mid_MCP (~0.11m)": _d(21, 23),
        "pelvis_z_mean (~0.9m)": float(jpos[:, 0, 2].mean()),
        "first_frame_pelvis_xy": list(jpos[0, 0, :2].tolist()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--smplh_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"Loading GT pickle {args.test_pickle} …")
    test_p = joblib.load(args.test_pickle)
    print(f"  {len(test_p)} windows")

    print(f"Loading text annos from {args.text_dir} …")
    text_map: dict[str, str] = {}
    for fn in sorted(os.listdir(args.text_dir)):
        if not fn.endswith(".json"):
            continue
        with open(pjoin(args.text_dir, fn)) as f:
            d = json.load(f)
        for k, v in d.items():
            text_map[k] = v
    print(f"  {len(text_map)} unique seq_name → text entries")

    idx_map = _build_seq_window_index(test_p)

    kt = _load_smplh_kintree(args.smplh_dir)
    parents = _get_smpl_parents_24(kt)  # (24,)
    print(f"parents: {parents.tolist()}")

    pred_files = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".npz"))
    print(f"Processing {len(pred_files)} prediction NPZs …")

    pred_jpos_list: list[np.ndarray] = []
    gt_jpos_list: list[np.ndarray] = []
    seq_names: list[str] = []
    window_idxs: list[int] = []
    start_t_idxs: list[int] = []
    texts: list[str] = []
    actual_lens: list[int] = []
    missing_gt = 0
    missing_text = 0
    for i, fn in enumerate(pred_files):
        d = np.load(pjoin(args.pred_dir, fn), allow_pickle=True)
        seq_name = str(d["seq_name"])
        window_idx = int(d["window_idx"])
        actual_len = int(d["actual_len"])
        key = idx_map.get(window_idx)
        if key is None or key not in test_p:
            missing_gt += 1
            if missing_gt < 5:
                print(f"  MISSING GT for {fn} (seq={seq_name} w={window_idx})")
            continue
        gt_entry = test_p[key]
        if gt_entry["seq_name"] != seq_name:
            raise RuntimeError(
                f"Seq-name mismatch for {fn}: pred_seq={seq_name} "
                f"gt_seq={gt_entry['seq_name']} at pickle key={key}."
            )
        gt_motion = gt_entry["motion"][:, :72].reshape(-1, 24, 3).astype(np.float32)
        assert gt_motion.shape[0] == 120, f"Unexpected GT T={gt_motion.shape[0]}"
        if seq_name not in text_map:
            missing_text += 1
            continue
        txt = text_map[seq_name]

        pred_dict = {k: d[k] for k in d.files}
        jpos = _fk_pred_to_omomo_jpos(pred_dict, gt_entry, parents)  # (120, 24, 3)

        pred_jpos_list.append(jpos)
        gt_jpos_list.append(gt_motion)
        seq_names.append(seq_name)
        window_idxs.append(window_idx)
        start_t_idxs.append(int(gt_entry["start_t_idx"]))
        texts.append(txt)
        actual_lens.append(actual_len)
        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(pred_files)} processed")

    print(
        f"Matched {len(pred_jpos_list)} pairs. "
        f"missing_gt={missing_gt} missing_text={missing_text}"
    )

    if pred_jpos_list:
        print("FK sanity (first pred sample):")
        for k, v in _fk_sanity_report(pred_jpos_list[0]).items():
            print(f"  {k}: {v}")
        print("FK sanity (first GT sample):")
        for k, v in _fk_sanity_report(gt_jpos_list[0]).items():
            print(f"  {k}: {v}")
        # Quantify bone-length delta pred-vs-GT for middle-finger bones.
        pj, gj = pred_jpos_list[0], gt_jpos_list[0]
        dL_p = np.linalg.norm(pj[:, 20] - pj[:, 22], axis=-1).mean()
        dL_g = np.linalg.norm(gj[:, 20] - gj[:, 22], axis=-1).mean()
        print(f"L-wrist→L-mid pred={dL_p:.3f} gt={dL_g:.3f} |Δ|={abs(dL_p - dL_g)*1000:.1f} mm")

    out = {
        "pred_jpos": np.stack(pred_jpos_list, axis=0),
        "gt_jpos": np.stack(gt_jpos_list, axis=0),
        "seq_names": np.array(seq_names),
        "window_idx": np.array(window_idxs, dtype=np.int32),
        "start_t_idx": np.array(start_t_idxs, dtype=np.int32),
        "texts": np.array(texts),
        "actual_lens": np.array(actual_lens, dtype=np.int32),
    }
    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    np.savez(args.output, **out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
