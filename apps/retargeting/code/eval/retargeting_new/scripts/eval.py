import os
import sys
import os.path as osp
import numpy as np
import torch
import trimesh
from tqdm import tqdm
import json
from copy import deepcopy
from torch import nn
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import argparse


# Helper to add project root to path
def setup_path():
    # PATCH 2026-05-01: file is now at eval/retargeting_new/scripts/, 3 levels
    # below repo root. Original was in eval/humanoid/ (2 levels).
    current_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.abspath(osp.join(current_dir, "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also ensure current dir is in path for local imports like 'metrics'
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    return project_root


PROJECT_ROOT = setup_path()

# Imports from project
# IMPORT-PATCHED 2026-04-30 to run on current HEAD; the original at
# eval/humanoid/eval.py keeps the historical (now-broken) imports.
#   models.gemm_v1.viz_helpers.mesh_visualizer  → REMOVED (was unused here)
#   datasets.gemm.gemm_dataset_config           → umm.datasets.umm_v1.dataset_config
#   data_processing.lib_human.human_model       → lib_human.human_model
from umm.datasets.umm_v1.dataset_config import G1_LINK_NAMES
from lib_twist.g1_interface import G1Interface
from lib_human.human_model import HumanModel

try:
    from metrics import fail_detect, align_phy_to_kin, keypoint_track_error
except ImportError:
    from eval.retargeting_new.scripts.metrics import (
        fail_detect,
        align_phy_to_kin,
        keypoint_track_error,
    )


class Evaluator:
    def __init__(
        self,
        human_name,
        kinematic_name,
        physical_name,
        test_split_fn,
        device="cpu",
        max_workers=12,
        N_first=None,
    ):
        self.project_root = PROJECT_ROOT
        self.device = device
        self.max_workers = max_workers

        # Paths — overridable via data_root_subdir to support the post-2026-04
        # layout where humanoid eval data sits under logs/eval/humanoid/
        # rather than the historical logs/eval/. Default kept for back-compat.
        data_root_subdir = getattr(self, "_data_root_subdir", "logs/eval/humanoid")
        self.root = osp.join(self.project_root, data_root_subdir)
        self.human_name = human_name
        self.kinematic_name = kinematic_name
        self.physical_name = physical_name

        # Determine strict path for test_split_fn, handling both absolute and relative
        if osp.isabs(test_split_fn):
            self.test_split_fn = test_split_fn
        else:
            self.test_split_fn = osp.join(self.project_root, test_split_fn)

        self.human_dir = osp.join(self.root, self.human_name)
        self.kinematic_dir = osp.join(self.root, self.kinematic_name)
        self.physical_dir = osp.join(self.root, self.physical_name)
        self.viz_save_dir = osp.join(
            self.project_root, f"logs/eval_viz/{self.physical_name}"
        )

        os.makedirs(self.viz_save_dir, exist_ok=True)

        # Load Split
        with open(self.test_split_fn, "r") as f:
            self.test_split = json.load(f)
            if N_first is not None:
                self.test_split = self.test_split[:N_first]

        # Check existence
        self.co_exists = []
        for sid, _ in self.test_split:
            _human_fn = osp.join(self.human_dir, f"{sid}.npz")
            _kinematic_fn = osp.join(self.kinematic_dir, f"{sid}.npz")
            _physical_fn = osp.join(self.physical_dir, f"{sid}.npz")
            if (
                osp.exists(_human_fn)
                and osp.exists(_kinematic_fn)
                and osp.exists(_physical_fn)
            ):
                self.co_exists.append(sid)
            else:
                print("not exists", _human_fn, _kinematic_fn, _physical_fn)
        print(f"Found {len(self.co_exists)}/{len(self.test_split)} samples.")
        assert len(self.co_exists) == len(self.test_split)

        # Initialize Models
        self.init_models()

    def init_models(self):
        urdf_file = "./ext/lib_twist/src/lib_twist/assets/g1/g1_custom_collision_with_fixed_hand.urdf"
        urdf_file = osp.join(self.project_root, urdf_file)
        self.G1I = G1Interface(urdf_file, device=self.device)

        human_file = "./data/smpls"
        human_file = osp.join(self.project_root, human_file)
        self.HM = HumanModel(
            model_path=human_file,
            model_type="smplx",
            gender="neutral",
            device=self.device,
            num_betas=300,
        )

    @torch.inference_mode()
    def eval_single(self, sid):
        # * load
        human_data = np.load(osp.join(self.human_dir, sid + ".npz"))
        kinematics_data = np.load(osp.join(self.kinematic_dir, sid + ".npz"))
        physics_data = np.load(osp.join(self.physical_dir, sid + ".npz"))

        all_link_names = deepcopy(self.G1I.link_names)
        all_link_names.remove("pelvis")  # put pelvis first
        all_link_names = ["pelvis"] + all_link_names

        # * compute keypoints
        human_Q, human_V, human_F, _ = self.HM.forward(
            shape_params=human_data["betas"][0].astype(np.float32),
            pose_params=human_data["pose"][0].astype(np.float32),
            transl=human_data["global_t"][0].astype(np.float32),
        )

        if "root_quat_xyzw" in kinematics_data:
            kin_root_quat_xyzw = kinematics_data["root_quat_xyzw"].astype(np.float32)
        else:
            kin_root_quat_xyzw = kinematics_data["root_rot"].astype(np.float32)
        kin_Q = self.G1I.FK(
            root_pos=kinematics_data["root_pos"].astype(np.float32),
            root_quat_xyzw=kin_root_quat_xyzw,
            dof_pos=kinematics_data["dof_pos"].astype(np.float32),
            names=self.G1I.link_names,
        )[:, 1:]
        # ! here use the first rollout to viz
        phy_Q = self.G1I.FK(
            root_pos=physics_data["root_pos"][0].astype(np.float32),
            root_quat_xyzw=physics_data["root_rot"][0].astype(np.float32),
            dof_pos=physics_data["dof_pos"][0].astype(np.float32),
            names=self.G1I.link_names,
        )[:, 1:]

        phy_Q = align_phy_to_kin(phy_Q, kin_Q)

        ret = {"link_names": all_link_names}
        # compute each metric
        ret = fail_detect(human_Q[:, 0], kin_Q[:, 0], phy_Q[:, 0], ret)

        # compute tracking error
        ret = keypoint_track_error(kin_Q, phy_Q, ret)

        if "run_time" in kinematics_data:
            n_frames = len(kin_Q)
            ret["total_frames"] = n_frames
            ret["run_time"] = float(kinematics_data["run_time"])
            ret["processing_fps"] = n_frames / float(ret["run_time"])

        return ret

    def run(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            RES = list(
                tqdm(
                    executor.map(self.eval_single, self.co_exists),
                    total=len(self.co_exists),
                )
            )

        # * summarize
        except_names = [
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "waist_yaw_link",
        ]
        EVAL_KEYPOINTS = [f for f in G1_LINK_NAMES if f not in except_names]

        results = []
        for r in tqdm(RES):
            buf = {
                "SR-seq": float(r["standing"].all()),
                "SR-ratio": float(r["standing"].mean()),
                "SR-succ_n": float(r["standing"].sum()),
                "SR-total_n": float(len(r["standing"])),
            }

            if "processing_fps" in r:
                buf["retgt_fps"] = float(r["processing_fps"])
                buf["retgt_runtime"] = float(r["run_time"])
                buf["retgt_nframes"] = float(r["total_frames"])

            # * global
            r["standing"] = r["standing"] > 0
            # Get indices for EVAL_KEYPOINTS
            index = [r["link_names"].index(key) for key in EVAL_KEYPOINTS]
            index = torch.tensor(index)

            gerr = r["global_tracking_err"][r["standing"]]
            buf[f"gerr"] = float(gerr.mean())
            lerr = r["local_tracking_err"][r["standing"]]
            buf[f"lerr"] = float(lerr.mean())

            gerr = r["global_tracking_err"]
            buf[f"gerr_all"] = float(gerr.mean())
            lerr = r["local_tracking_err"]
            buf[f"lerr_all"] = float(lerr.mean())

            # * details
            for key in EVAL_KEYPOINTS:
                idx = r["link_names"].index(key)
                lerr = r["local_tracking_err"][:, idx]
                buf[f"lerr__{key}"] = float(lerr[r["standing"]].mean())
            for key in EVAL_KEYPOINTS:
                idx = r["link_names"].index(key)
                gerr = r["global_tracking_err"][:, idx]
                buf[f"gerr__{key}"] = float(gerr[r["standing"]].mean())
            results.append(buf)

        df = pd.DataFrame(results)
        df_stats = pd.DataFrame([df.mean(), df.std()], index=["mean", "std"])
        df = pd.concat([df_stats, df])
        xls_fn = self.physical_dir + f"_report_{len(results)}.xlsx"
        df.to_excel(xls_fn)
        print(f"Saved report to {xls_fn}")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_name", type=str, default="amass_human_test")
    parser.add_argument("--kinematic_name", type=str, default="g7.17.2_v1_160k")
    parser.add_argument(
        "--physical_name", type=str, default="g7.17.2_v1_160k_mjc_rollout"
    )
    parser.add_argument(
        "--test_split_fn",
        type=str,
        default="./data_splits/amass_humanoid_v2/full_test.json",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    # Optional: limit number of samples for testing
    parser.add_argument("--N_first", type=int, default=None)

    args = parser.parse_args()

    evaluator = Evaluator(
        human_name=args.human_name,
        kinematic_name=args.kinematic_name,
        physical_name=args.physical_name,
        test_split_fn=args.test_split_fn,
        max_workers=args.workers,
        device=args.device,
        N_first=args.N_first,
    )
    evaluator.run()
