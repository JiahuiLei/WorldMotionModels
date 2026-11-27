import numpy as np
import torch
import os, sys, os.path as osp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# PATCH 2026-05-01: file-relative root (script lives at eval/retargeting_new/scripts/,
# so go up 3 levels to repo root). Original used os.getcwd()-based which only worked
# when run from eval/humanoid/.
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib_twist.g1_interface import G1Interface
from lib_twist.mujoco_lowlevel_controller import MuJoCoLowLevelController

# IMPORT-PATCHED 2026-04-30
# Original at eval/humanoid/2mujoco_rollout.py keeps the historical broken import:
#   datasets.gemm.gemm_dataset_config → umm.datasets.umm_v1.dataset_config
from umm.datasets.umm_v1.dataset_config import G1_LINK_NAMES

URDF_FN = (
    "./ext/lib_twist/src/lib_twist/assets/g1/g1_custom_collision_with_fixed_hand.urdf"
)
XML_FN = "./ext/lib_twist/src/lib_twist/assets/g1/g1_sim2sim_with_wrist_roll.xml"
POLICY_FN = "./ext/lib_twist/src/lib_twist/assets/twist_general_motion_tracker.pt"
URDF_FN = osp.join(PROJECT_ROOT, URDF_FN)
XML_FN = osp.join(PROJECT_ROOT, XML_FN)
POLICY_FN = osp.join(PROJECT_ROOT, POLICY_FN)

G1I = G1Interface(URDF_FN, device="cpu")


def rollout_thread(src_fn, save_fn, num_rollouts=1):
    # ! now no domain randomization, so just 1 rollout
    controller_env = MuJoCoLowLevelController(
        xml_file=XML_FN,
        policy_path=POLICY_FN,
        device="cpu",  # or 'cpu'
        control_dt=0.02,  # 50 Hz control rate
        sim_dt=0.001,  # 1000 Hz simulation rate
    )

    data = np.load(src_fn)
    root_pos = data["root_pos"].astype("float32")
    if "root_quat_xyzw" in data:
        root_quat_xyzw = data["root_quat_xyzw"].astype("float32")
    else:  # for some buggy version
        root_quat_xyzw = data["root_rot"].astype("float32")
    dof_pos = data["dof_pos"].astype("float32")
    N30 = len(root_pos)
    cmd50 = G1I.get_mimic_obs(
        root_pos, root_quat_xyzw, dof_pos, fps=30, control_dt=0.02
    )
    # N = len(cmd50)

    results = []
    for ri in range(num_rollouts):
        run_stats_50fps = controller_env.run(
            target_commands=cmd50,
            verbose=False,
        )
        run_stats_fps_aligned = G1I.convert_fps(run_stats_50fps, 30.0)
        for k in run_stats_fps_aligned.keys():
            if isinstance(run_stats_fps_aligned[k], np.ndarray):
                if len(run_stats_fps_aligned[k]) > N30:
                    run_stats_fps_aligned[k] = run_stats_fps_aligned[k][-N30:]
                if len(run_stats_fps_aligned[k]) < N30:
                    # ! manually dup at the end
                    n_missing = N30 - len(run_stats_fps_aligned[k])
                    print(f"WARNING! DUP PAD {n_missing} at the end")
                    run_stats_fps_aligned[k] = np.concatenate(
                        [
                            run_stats_fps_aligned[k],
                            run_stats_fps_aligned[k][-1:].repeat(n_missing, axis=0),
                        ],
                        axis=0,
                    )
                assert (
                    len(run_stats_fps_aligned[k]) == N30
                ), f" {k} {len(run_stats_fps_aligned[k])} {N30}"
        results.append(run_stats_fps_aligned)
    ret = {}
    for k in results[0].keys():
        ret[k] = np.stack([r[k] for r in results], axis=0)
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    np.savez(save_fn, **ret)
    return


def main(src, dst, num_rollouts=1, thread=1):
    src_fns = [fn for fn in os.listdir(src) if fn.endswith(".npz")]
    src_fns.sort()
    if thread == 1:
        for fn in tqdm(src_fns):
            rollout_thread(os.path.join(src, fn), os.path.join(dst, fn), num_rollouts)
    else:
        with ProcessPoolExecutor(max_workers=thread) as executor:
            futures = [
                executor.submit(
                    rollout_thread,
                    os.path.join(src, fn),
                    os.path.join(dst, fn),
                    num_rollouts,
                )
                for fn in src_fns
            ]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, default="")
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--thread", type=int, default=12)
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    if len(dst) == 0:
        dst = src + "_mjc_rollout"
    num_rollouts = args.num_rollouts
    thread = args.thread
    main(src, dst, num_rollouts=num_rollouts, thread=thread)
