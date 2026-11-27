"""OMOMO -> WMM per-window ``.npz`` converter (CHOIS Table 1 protocol).

Builds the files the OMOMO experiment reads::

    <out_dir>/test_npz/<seq_name>_w<k>.npz     482 test windows
    <out_dir>/train_npz/<seq_name>_w<k>.npz    13,425 training windows
    <out_dir>/splits.json                      window lists of both splits

Usage (run from ``apps/omomo/code_train`` with the ``umm`` environment)::

    python -m prepare_data.omomo.convert --split test  --omomo_dir ... --chois_test_cache ... \
        --text_dir ... --smplh_dir ... --smplx_dir ... --rest_object_geo ... --out_dir ...
    python -m prepare_data.omomo.convert --split train ...          # same arguments
    python -m prepare_data.omomo.convert --write_splits --out_dir ...

``apps/omomo/prepare_data/build_all.sh`` runs these steps with the right paths.

Inputs
------
``--omomo_dir``         the ``data/`` folder of the OMOMO dataset download. Uses
                        ``train_diffusion_manip_window_120_cano_joints24.p``, ``object_bps_npy_files_joints24/``
                        (train) and ``object_bps_npy_files_for_eval_joints24/`` (test).
``--chois_test_cache``  ``cano_test_diffusion_manip_window_120_joints24.p``: the test windows as canonicalised by
                        the CHOIS data code (``apps/omomo/prepare_data/build_chois_cache.py``).
``--text_dir``          text annotations split into ``train/`` and ``test/`` (``build_text_anno.py``).
``--smplh_dir``         SMPL-H (AMASS version) folder with ``male/model.npz``; only its kinematic tree is read.
``--smplx_dir``         SMPL-X folder with ``SMPLX_{MALE,FEMALE,NEUTRAL}.npz``.
``--rest_object_geo``   per-object rest pose files ``<object>.json`` written by the CHOIS data code.

Protocol (CHOIS Table 1, ``cano_traj_dataset.py``)
--------------------------------------------------
Stage 1 drops vacuum and mop; stage 2 keeps sequences with a text annotation; stage 3 (test only) keeps the first
window of each sequence that is at least 120 frames long.

| Split | Source windows | After 1 | After 2 | After 3 |
|-------|----------------|---------|---------|---------|
| test  | 3250 (CHOIS test cache) | 3250 | 3224 | 482 |
| train | 15976 (OMOMO train windows) | 13647 | 13425 | - |

Fields of one window file (T = 120 frames, ``L = actual_len`` valid frames, frames ``L..T-1`` padded)
--------------------------------------------------------------------------------------------------
``root_trans (T, 3)``, ``root_orient_aa (T, 3)``, ``pose_body_aa (T, 21, 3)``, ``lhand_wrist_jpos (T, 3)``,
``rhand_wrist_jpos (T, 3)``, ``obj_trans (T, 3)`` (object centre of mass), ``obj_rot (T, 3, 3)`` (in the frame of
``rest_object_geo/<object>.ply``), ``obj_scale (T,)``, ``Q (T, 22, 4, 4)`` (per-joint world pose from SMPL-X forward
kinematics), all float32; ``betas (16,)`` float32, ``gender`` str, ``obj_bps (1024, 3)`` float32 (first-frame BPS
of the object if OMOMO ships it for this window key, else zeros; not read by the model), ``obj_mesh_path`` str,
``text`` str, ``waypoint_indices (5,)`` int32 (frames 0, 29, 59, 89, L-1; -1 for a waypoint beyond L),
``first_root_trans (3,)``, ``first_root_orient_aa (3,)``, ``first_pose_body_aa (21, 3)``, ``window_idx``,
``actual_len``, ``T`` (int32 scalars), ``seq_name`` str.

Padding uses identity transforms for rotations (``obj_rot``, ``Q``), 1.0 for ``obj_scale`` and zeros otherwise.
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import pytorch3d.transforms as p3d_transforms

from prepare_data.omomo.build_q import (
    build_q_from_pickle_fields,
    smplx_world_transforms,
)

T = 120
CHOIS_EXCLUDED_OBJECTS = ("vacuum", "mop")
FILTERS = {
    "train": ["exclude_vacuum_mop", "filter_out_seq_wo_text"],
    "test": ["exclude_vacuum_mop", "filter_out_seq_wo_text",
             "filter_out_short_sequences (motion.shape[0]>=120 AND start_t_idx==0)"],
}

_SMPL_PARENTS_22: Optional[np.ndarray] = None
_REST_POSE_ROT_CACHE: Dict[str, np.ndarray] = {}
_TEXT_CACHE: Dict[str, str] = {}
_TEXT_MISSING: set = set()


def _get_smpl_parents_22(smplh_dir: str) -> np.ndarray:
    """SMPL-H parent array of the first 22 joints (root parent = -1), shape (22,) int32."""
    global _SMPL_PARENTS_22
    if _SMPL_PARENTS_22 is None:
        npz_data = np.load(osp.join(smplh_dir, "male", "model.npz"))
        parents = npz_data["kintree_table"][0, :22].copy().astype(np.int32)
        parents[0] = -1
        _SMPL_PARENTS_22 = parents
    return _SMPL_PARENTS_22


def _get_rest_pose_rot(object_name: str, rest_object_geo: str) -> np.ndarray:
    """``rest_pose_ori_obj_rot`` (3, 3) float32 of one object, from ``<rest_object_geo>/<object>.json``."""
    cached = _REST_POSE_ROT_CACHE.get(object_name)
    if cached is not None:
        return cached
    rest_rot_path = Path(rest_object_geo) / f"{object_name}.json"
    if not rest_rot_path.exists():
        raise FileNotFoundError(f"missing rest pose file for object {object_name!r}: {rest_rot_path}")
    with open(rest_rot_path) as f:
        data = json.load(f)
    rest_rot = np.asarray(data["rest_pose_ori_obj_rot"], dtype=np.float32)
    if rest_rot.shape != (3, 3):
        raise ValueError(f"rest_pose_ori_obj_rot for {object_name!r} must be (3, 3); got {rest_rot.shape}")
    _REST_POSE_ROT_CACHE[object_name] = rest_rot
    return rest_rot


def quat_ik_torch(grot_mat: torch.Tensor, parents: np.ndarray) -> torch.Tensor:
    """Global (T, J, 3, 3) joint rotations -> local rotations, as in OMOMO's ``quat_ik_torch``."""
    grot = p3d_transforms.matrix_to_quaternion(grot_mat)  # (T, J, 4)
    local_quat = torch.cat(
        [
            grot[..., :1, :],
            p3d_transforms.quaternion_multiply(
                p3d_transforms.quaternion_invert(grot[..., parents[1:], :]),
                grot[..., 1:, :],
            ),
        ],
        dim=-2,
    )  # (T, J, 4)
    return p3d_transforms.quaternion_to_matrix(local_quat)  # (T, J, 3, 3)


def _load_text(seq_name: str, text_base_dir: str) -> str:
    """Caption of one sequence from ``<text_base_dir>/<seq_name>.json`` (``{seq_name: caption}``), or ''."""
    if seq_name in _TEXT_CACHE:
        return _TEXT_CACHE[seq_name]
    if seq_name in _TEXT_MISSING:
        return ""
    json_path = osp.join(text_base_dir, f"{seq_name}.json")
    if not osp.exists(json_path):
        _TEXT_MISSING.add(seq_name)
        return ""
    with open(json_path, "r") as f:
        data = json.load(f)
    text = data.get(seq_name, "")
    _TEXT_CACHE[seq_name] = text
    return text


def _apply_chois_filters(data: Dict, text_base_dir: str, split: str) -> Dict:
    """CHOIS Table 1 filter chain (see module docstring); keeps the original window keys."""
    after_obj = {
        k: w for k, w in data.items()
        if w["seq_name"].split("_")[1] not in CHOIS_EXCLUDED_OBJECTS
    }
    after_text = {
        k: w for k, w in after_obj.items()
        if osp.exists(osp.join(text_base_dir, f"{w['seq_name']}.json"))
    }
    if split == "train":
        print(f"[convert] train filter: {len(data)} -> {len(after_obj)} (no vacuum/mop) "
              f"-> {len(after_text)} (has text)", flush=True)
        return after_text
    after_short = {
        k: w for k, w in after_text.items()
        if w["motion"].shape[0] >= 120 and w["start_t_idx"] == 0
    }
    print(f"[convert] test filter: {len(data)} -> {len(after_obj)} (no vacuum/mop) -> {len(after_text)} "
          f"(has text) -> {len(after_short)} (first window, >= 120 frames)", flush=True)
    return after_short


def convert_split(args: argparse.Namespace) -> None:
    """Convert one split to ``<out_dir>/<split>_npz/`` (optionally one shard of it)."""
    split = args.split
    if split == "train":
        pickle_path = osp.join(args.omomo_dir, "train_diffusion_manip_window_120_cano_joints24.p")
        bps_base = osp.join(args.omomo_dir, "object_bps_npy_files_joints24")
        # OMOMO's train windows store the raw object rotation: bring it into the rest-mesh frame.
        correct_obj_rot = True
    else:
        pickle_path = args.chois_test_cache
        bps_base = osp.join(args.omomo_dir, "object_bps_npy_files_for_eval_joints24")
        # The CHOIS test cache already stores rotations in the rest-mesh frame.
        correct_obj_rot = False
    text_base_dir = osp.join(args.text_dir, split)
    out_dir = osp.join(args.out_dir, f"{split}_npz")
    os.makedirs(out_dir, exist_ok=True)
    parents = _get_smpl_parents_22(args.smplh_dir)

    print(f"[convert] loading {pickle_path}", flush=True)
    data = joblib.load(pickle_path)
    data = _apply_chois_filters(data, text_base_dir, split)
    all_keys = sorted(data.keys())
    if args.limit is not None:
        all_keys = all_keys[:args.limit]
    keys = all_keys[args.shard_idx::args.num_shards]
    print(f"[convert] {split}: {len(all_keys)} windows, this shard {len(keys)} -> {out_dir}", flush=True)

    for idx_k, k in enumerate(keys):
        w = data[k]
        seq_name = str(w["seq_name"])
        stem = f"{seq_name}_w{k}"
        out_path = osp.join(out_dir, f"{stem}.npz")
        if args.skip_existing and osp.exists(out_path):
            continue

        motion = w["motion"].astype(np.float32)  # (L, 276)
        actual_len = motion.shape[0]
        global_jpos = motion[:, :72].reshape(actual_len, 24, 3)  # (L, 24, 3)
        global_rot_6d = motion[:, 144:276].reshape(actual_len, 22, 6)  # (L, 22, 6)

        global_rot_mat = p3d_transforms.rotation_6d_to_matrix(torch.from_numpy(global_rot_6d).float())
        local_rot_mat = quat_ik_torch(global_rot_mat, parents)  # (L, 22, 3, 3)
        aa = p3d_transforms.matrix_to_axis_angle(local_rot_mat).numpy()  # (L, 22, 3)
        root_orient_aa = aa[:, 0].astype(np.float32)  # (L, 3)
        pose_body_aa = aa[:, 1:22].astype(np.float32)  # (L, 21, 3)

        trans2joint = np.asarray(w["trans2joint"]).reshape(3).astype(np.float32)
        root_trans = (global_jpos[:, 0] + trans2joint).astype(np.float32)  # (L, 3)
        lhand_wrist_jpos = global_jpos[:, 22].astype(np.float32)  # (L, 3)
        rhand_wrist_jpos = global_jpos[:, 23].astype(np.float32)  # (L, 3)

        # Object position = centre of mass (the rest meshes are centred on it), not OMOMO's pivot translation.
        obj_trans = np.asarray(w["window_obj_com_pos"]).astype(np.float32)[:actual_len]  # (L, 3)
        obj_rot = np.asarray(w["obj_rot_mat"]).astype(np.float32)[:actual_len]  # (L, 3, 3)
        if "obj_scale" in w:
            obj_scale_raw = np.asarray(w["obj_scale"]).astype(np.float32)
            if obj_scale_raw.ndim == 0:
                obj_scale = np.full((actual_len,), float(obj_scale_raw), dtype=np.float32)
            else:
                obj_scale = obj_scale_raw.reshape(-1)[:actual_len].astype(np.float32)
        else:
            obj_scale = np.ones((actual_len,), dtype=np.float32)  # (L,)

        object_name = seq_name.split("_")[1]
        if correct_obj_rot:
            rest_pose_rot = _get_rest_pose_rot(object_name, args.rest_object_geo)  # (3, 3)
            obj_rot = np.einsum("tij,jk->tik", obj_rot, rest_pose_rot.T).astype(np.float32)  # (L, 3, 3)

        betas = np.asarray(w["betas"]).reshape(-1).astype(np.float32)  # (16,)
        gender = str(w["gender"])

        global_rot_mat_smplx, global_jpos_smplx = smplx_world_transforms(
            root_trans=root_trans,
            root_orient_aa=root_orient_aa,
            pose_body_aa=pose_body_aa,
            betas=betas,
            gender=gender,
            smplx_dir=args.smplx_dir,
        )
        Q = build_q_from_pickle_fields(global_rot_mat_smplx, global_jpos_smplx)  # (L, 22, 4, 4)

        bps_path = osp.join(bps_base, f"{seq_name}_{k}.npy")
        if osp.exists(bps_path):
            obj_bps_full = np.load(bps_path).astype(np.float32)  # (L, 1024, 3)
            if obj_bps_full.shape[0] != actual_len:
                raise AssertionError(f"BPS length mismatch: {obj_bps_full.shape[0]} vs actual_len={actual_len} "
                                     f"for {seq_name}_{k}")
            obj_bps = obj_bps_full[0].astype(np.float32)  # (1024, 3)
        else:
            obj_bps = np.zeros((1024, 3), dtype=np.float32)

        obj_mesh_path = f"{object_name}_cleaned_simplified.obj"
        text = _load_text(seq_name, text_base_dir)

        # CHOIS Table 1 waypoints: frames 29, 59, 89 and the last frame; frame 0 anchors the trajectory.
        intermediate = [i if i < actual_len else -1 for i in (29, 59, 89)]
        waypoint_indices = np.array([0, *intermediate, max(actual_len - 1, 0)], dtype=np.int32)  # (5,)

        first_root_trans = root_trans[0].astype(np.float32)  # (3,)
        first_root_orient_aa = root_orient_aa[0].astype(np.float32)  # (3,)
        first_pose_body_aa = pose_body_aa[0].astype(np.float32)  # (21, 3)

        def _pad(arr: np.ndarray) -> np.ndarray:
            if actual_len >= T:
                return arr[:T]
            pad_shape = list(arr.shape)
            pad_shape[0] = T - actual_len
            return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        obj_rot_padded = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))  # (T, 3, 3)
        obj_rot_padded[:actual_len] = obj_rot
        obj_scale_padded = np.ones((T,), dtype=np.float32)  # (T,)
        obj_scale_padded[:actual_len] = obj_scale
        Q_padded = np.tile(np.eye(4, dtype=np.float32)[None, None], (T, 22, 1, 1))  # (T, 22, 4, 4)
        Q_padded[:actual_len] = Q

        np.savez(
            out_path,
            root_trans=_pad(root_trans),
            root_orient_aa=_pad(root_orient_aa),
            pose_body_aa=_pad(pose_body_aa),
            lhand_wrist_jpos=_pad(lhand_wrist_jpos),
            rhand_wrist_jpos=_pad(rhand_wrist_jpos),
            obj_trans=_pad(obj_trans),
            obj_rot=obj_rot_padded,
            obj_scale=obj_scale_padded,
            Q=Q_padded,
            betas=betas,
            gender=np.array(gender, dtype=object),
            obj_bps=obj_bps,
            obj_mesh_path=np.array(obj_mesh_path, dtype=object),
            text=np.array(text, dtype=object),
            waypoint_indices=waypoint_indices,
            first_root_trans=first_root_trans,
            first_root_orient_aa=first_root_orient_aa,
            first_pose_body_aa=first_pose_body_aa,
            window_idx=np.int32(k),
            actual_len=np.int32(actual_len),
            T=np.int32(T),
            seq_name=np.array(seq_name, dtype=object),
        )
        if (idx_k + 1) % args.progress_every == 0:
            print(f"[convert] {idx_k + 1}/{len(keys)}", flush=True)

    if _TEXT_MISSING:
        print(f"[convert] WARNING: {len(_TEXT_MISSING)} sequences have no text annotation", file=sys.stderr)

    if args.num_shards == 1 and args.limit is None:
        expected = {f"{data[k]['seq_name']}_w{k}" for k in all_keys}
        observed = {osp.basename(p)[:-4] for p in glob.glob(osp.join(out_dir, "*.npz"))}
        if expected != observed:
            sys.exit(f"[convert] ERROR: {out_dir} holds {len(observed)} windows, expected {len(expected)} "
                     f"(missing {len(expected - observed)}, extra {len(observed - expected)})")
        print(f"[convert] {split}: {len(observed)} windows written", flush=True)


def write_splits(out_dir: str) -> None:
    """Write ``<out_dir>/splits.json`` from the window files in ``test_npz/`` and ``train_npz/``."""
    stems = {s: sorted(osp.basename(p)[:-4] for p in glob.glob(osp.join(out_dir, f"{s}_npz", "*.npz")))
             for s in ("test", "train")}
    splits = {
        "test": stems["test"],
        "subject_held_out": {
            "test_subjects": ["sub16", "sub17"],
            "train_subjects": [f"sub{i}" for i in range(1, 16)],
        },
        "total_windows": {"test": len(stems["test"]), "train": len(stems["train"])},
        "protocol": "chois-table-1",
        "excluded_objects": list(CHOIS_EXCLUDED_OBJECTS),
        "filters": FILTERS,
        "train": stems["train"],
    }
    path = osp.join(out_dir, "splits.json")
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[convert] wrote {path} ({len(stems['test'])} test, {len(stems['train'])} train windows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--write_splits", action="store_true", help="only write <out_dir>/splits.json")
    parser.add_argument("--omomo_dir", help="data/ folder of the OMOMO download")
    parser.add_argument("--chois_test_cache", help="cano_test_diffusion_manip_window_120_joints24.p")
    parser.add_argument("--text_dir", help="text annotations with train/ and test/ subfolders")
    parser.add_argument("--smplh_dir", help="SMPL-H (AMASS) folder with male/model.npz")
    parser.add_argument("--smplx_dir", help="SMPL-X folder with SMPLX_{MALE,FEMALE,NEUTRAL}.npz")
    parser.add_argument("--rest_object_geo", help="rest_object_geo/ written by the CHOIS data code")
    parser.add_argument("--out_dir", required=True, help="output folder, e.g. release_data/omomo/data_wmm/omomo_v3")
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows (quick test)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    if args.write_splits:
        write_splits(args.out_dir)
        return
    if args.split is None:
        parser.error("--split is required unless --write_splits is given")
    required = ["omomo_dir", "text_dir", "smplh_dir", "smplx_dir", "rest_object_geo"]
    if args.split == "test":
        required.append("chois_test_cache")
    missing = [n for n in required if getattr(args, n) is None]
    if missing:
        parser.error("missing " + ", ".join("--" + n for n in missing))
    if not 0 <= args.shard_idx < args.num_shards:
        parser.error("need 0 <= --shard_idx < --num_shards")
    convert_split(args)


if __name__ == "__main__":
    main()
