"""Build the CHOIS-processed OMOMO files with the CHOIS data code vendored in ``code/ext/baselines/chois_release``.

    conda activate chois_repro
    python apps/omomo/prepare_data/build_chois_cache.py --chois_root <dir>

``<chois_root>`` must hold (``build_all.sh`` links them in):

    train_diffusion_manip_seq_joints24.p, test_diffusion_manip_seq_joints24.p,
    min_max_mean_std_data_window_120_cano_joints24.p, captured_objects/      (OMOMO download, data/)
    smpl_all_models/smplx/SMPLX_{MALE,FEMALE}.npz                             (SMPL-X)
    smpl_all_models/smplh_amass/male/model.npz                                (SMPL-H, AMASS version)

Written into ``<chois_root>`` by CHOIS's own code (``manip/data/cano_traj_dataset.py``):

| File | Shape / content | Made by |
|---|---|---|
| ``rest_object_geo/<object>.{ply,json,npy}`` | 13 objects: mesh centred on its centre of mass; json with ``rest_pose_ori_obj_rot`` (3, 3), ``rest_pose_ori_com_pos`` (1, 3), ``obj_trans_to_com_pos`` (1, 3); npy BPS (1, 1024, 3) float32 | ``extract_rest_pose_object_geometry_and_rotation`` on the training sequences, as at the start of CHOIS training |
| ``cano_min_max_mean_std_data_window_120_joints24.p`` | joint / object position ranges | OMOMO's ``min_max_mean_std_data_window_120_cano_joints24.p`` plus the object keys under the names CHOIS reads (``obj_com_pos_min`` = ``cond_obj_com_pos_min``, ...) |
| ``cano_test_diffusion_manip_window_120_joints24.p`` | 3,250 test windows (``motion (L, 276)``, ``obj_rot_mat (L, 3, 3)``, ``window_obj_com_pos (L, 3)``, ...) | ``CanoObjectTrajDataset(train=False)`` |

CHOIS also writes a BPS file per test window (``cano_object_bps_npy_files_for_test_joints24_120/``, about 5 GB); WMM
does not read them, so they are deleted at the end unless ``--keep_bps`` is given. Needs one CUDA GPU (SMPL-X runs
on it); about 20 minutes.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import joblib

CHOIS_DIR = Path(__file__).resolve().parents[1] / "code" / "ext" / "baselines" / "chois_release"
STATS = "cano_min_max_mean_std_data_window_120_joints24.p"
RENAMED_STATS = {"cond_obj_com_pos_min": "obj_com_pos_min", "cond_obj_com_pos_max": "obj_com_pos_max",
                 "cond_obj_bps_min": "obj_bps_min", "cond_obj_bps_max": "obj_bps_max"}


def write_chois_stats(root: Path) -> None:
    """OMOMO stores the object statistics as ``cond_*``; CHOIS's loader reads them without the prefix."""
    if (root / STATS).is_file():
        return
    src = joblib.load(root / "min_max_mean_std_data_window_120_cano_joints24.p")
    out = dict(src)
    for old, new in RENAMED_STATS.items():
        if old in src and new not in out:
            out[new] = src[old]
    joblib.dump(out, root / STATS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chois_root", required=True, type=Path)
    parser.add_argument("--keep_bps", action="store_true")
    args = parser.parse_args()
    root = args.chois_root.resolve()
    smplh = root / "smpl_all_models" / "smplh_amass"
    if not (smplh / "male" / "model.npz").is_file():
        raise SystemExit(f"SMPL-H model not found: {smplh}/male/model.npz")
    os.environ["CHOIS_SMPLH_PATH"] = str(smplh)

    # CHOIS loads ./bps.pt and imports its packages relative to its own folder.
    os.chdir(CHOIS_DIR)
    sys.path.insert(0, str(CHOIS_DIR))
    from manip.data.cano_traj_dataset import CanoObjectTrajDataset

    write_chois_stats(root)

    geo = root / "rest_object_geo"
    if not geo.is_dir() or not any(geo.iterdir()):
        # The rest pose of each object comes from its first training sequence, exactly as when CHOIS builds its
        # training set; only this step of the training-set construction is run.
        ds = CanoObjectTrajDataset.__new__(CanoObjectTrajDataset)
        ds.data_root_folder = str(root)
        ds.obj_geo_root_folder = str(root / "captured_objects")
        ds.rest_object_geo_folder = str(geo)
        ds.bps_path = "./bps.pt"
        geo.mkdir(parents=True, exist_ok=True)
        ds.prep_bps_data()
        ds.data_dict = joblib.load(root / "train_diffusion_manip_seq_joints24.p")
        ds.extract_rest_pose_object_geometry_and_rotation()
        print(f"wrote {len(list(geo.iterdir()))} files to {geo}")

    CanoObjectTrajDataset(train=False, data_root_folder=str(root), window=120, use_object_splits=False,
                          input_language_condition=False, use_random_frame_bps=True, use_object_keypoints=True)
    print(f"wrote {root / 'cano_test_diffusion_manip_window_120_joints24.p'}")

    if not args.keep_bps:
        for name in ("cano_object_bps_npy_files_for_test_joints24_120", "cano_object_bps_npy_files_joints24_120"):
            p = root / name
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)


if __name__ == "__main__":
    main()
