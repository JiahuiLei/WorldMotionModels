import os

# Set environment variables BEFORE importing torch/numpy to avoid contention
# User requested 8 threads per worker.
# We usually set OMP/MKL to 1 for multiprocessing to avoid oversubscription,
# letting the worker threads handle it.
# However, if GMR relies on numpy/scipy parallel ops (like BLAS), we might want
# to control this. But usually explicit multiprocessing with affinity is better controlled
# by restricting thread counts.
# Let's set defaults, but allow override if needed via worker setup.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import os.path as osp
import argparse
import multiprocessing
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

# Add project root to sys.path
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from datasets.gemm.gemm_dataset_config import G1_LINK_NAMES


class GMR_Wrapper:
    def __init__(self, robot="unitree_g1"):
        self.fps = 30.0
        self.robot = robot

    @staticmethod
    def load_smplx_ours(
        pose_body63, root_orient, trans, betas, smplx_body_model_path, gender="neutral"
    ):
        # pose_body: N, 63
        # root_orient: N, 3
        # betas: N, 16
        body_model = smplx.create(
            smplx_body_model_path,
            "smplx",
            gender=gender,
            use_pca=False,
        )

        num_frames = pose_body63.shape[0]
        smplx_output = body_model(
            betas=torch.tensor(betas).float().view(1, -1),  # (16,)
            global_orient=torch.tensor(root_orient).float(),  # (N, 3)
            body_pose=torch.tensor(pose_body63).float(),  # (N, 63)
            transl=torch.tensor(trans).float(),  # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        if len(betas.shape) == 1:
            human_height = 1.66 + 0.1 * betas[0]
        else:
            human_height = 1.66 + 0.1 * betas[0, 0]

        return body_model, smplx_output, human_height

    @staticmethod
    def get_smplx_data_offline_fast_ours(num_frames, body_model, smplx_output):
        """
        Must return a dictionary with the following structure:
        {
            "Hips": (position, orientation),
            "Spine": (position, orientation),
            ...
        }
        """
        global_orient = smplx_output.global_orient.squeeze()
        full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
        joints = smplx_output.joints.detach().numpy().squeeze()
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents

        smplx_data_frames = []
        for curr_frame in range(len(global_orient)):
            result = {}
            single_global_orient = global_orient[curr_frame]
            single_full_body_pose = full_body_pose[curr_frame]
            single_joints = joints[curr_frame]
            joint_orientations = []
            for i, joint_name in enumerate(joint_names):
                if i == 0:
                    rot = R.from_rotvec(single_global_orient)
                else:
                    rot = joint_orientations[parents[i]] * R.from_rotvec(
                        single_full_body_pose[i].squeeze()
                    )
                joint_orientations.append(rot)
                result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

            smplx_data_frames.append(result)

        return smplx_data_frames

    @staticmethod
    def load_human_data(human_fn):
        fps = 30  # by default all the dataset should align to this fps
        human_data = np.load(human_fn)
        beta = human_data["betas"][0][0]  # 300
        # human_Q = human_data["Q"][0]  # N,55,4,4
        human_trans = human_data["global_t"][0]  # N,3
        # human_Q[:, :, :3, 3] = human_Q[:, :, :3, 3] + human_trans[:, None, :]
        global_t = human_data["global_t"][0]  # T,3
        pose = human_data["pose"][0]  # T,165
        return pose, global_t, beta

    def retarget_phuma(self, human_fn, save_fn, skip=1):
        try:
            smplx_pose, smplx_transl, smplx_beta = self.load_human_data(human_fn)

            # ! interpolate, ensure two ends are known
            if skip > 1:
                index = np.arange(0, len(smplx_pose), skip).tolist()
                if len(smplx_pose) - 1 not in index:
                    index = index + [len(smplx_pose) - 1]
                index = np.array(index)
                interp_mask = np.zeros(len(smplx_pose), dtype=bool)
                interp_mask[index] = True  # later use this to do linear interp
                smplx_pose = smplx_pose[index]
                smplx_transl = smplx_transl[index]
                # smplx_beta = smplx_beta[index]

            body_model, smplx_output, human_height = self.load_smplx_ours(
                pose_body63=smplx_pose[:, 3:66],
                root_orient=smplx_pose[:, :3],
                trans=smplx_transl,
                betas=smplx_beta[:16],  # follow GMR use 16
                smplx_body_model_path=osp.join(PROJECT_ROOT, "data", "smpls"),
                gender="neutral",
            )
            smplx_data_frames = self.get_smplx_data_offline_fast_ours(
                len(smplx_pose), body_model, smplx_output
            )
            retarget = GMR(
                actual_human_height=human_height,
                src_human="smplx",
                tgt_robot=self.robot,
            )
            # print(f"Processing {human_fn}...")
            start_t = time.time()
            qpos_list = []

            # GMR might not be parallel safe if it uses shared global state, but usually ok per process.
            # We don't use tqdm inside worker for cleanliness usually, but user might want progress.
            # For bulk processing, maybe skip tqdm per file or keep as is.
            # I'll iterate directly.
            for i in range(len(smplx_data_frames)):
                qpos = retarget.retarget(smplx_data_frames[i])
                qpos_list.append(qpos)

            runtime = time.time() - start_t
            # print(f"Runtime: {runtime} for {len(smplx_data_frames)} frames")

            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            # save from wxyz to xyzw
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])

            # ! convert to twist 23 dim from 29 dim remove the two wrist https://github.com/YanjieZe/TWIST/issues/15: [19, 20, 21, 26, 27, 28]
            remove_dim = [19, 20, 21, 26, 27, 28]
            dof_pos = np.delete(dof_pos, remove_dim, axis=1)
            assert dof_pos.shape[1] == 23, "TWIST REMOVE TWO WRIST FROM 29 -> 23"

            # ! de-interpolate
            if skip > 1:
                full_idx = np.arange(len(interp_mask))
                root_pos = interp1d(index, root_pos, axis=0)(full_idx)
                dof_pos = interp1d(index, dof_pos, axis=0)(full_idx)
                slerp = Slerp(index, R.from_quat(root_rot))
                root_rot = slerp(full_idx).as_quat()
                assert (
                    len(root_pos) == len(root_rot) == len(dof_pos) == len(interp_mask)
                )

            local_body_pos = None
            body_names = None
            # motion_data construction omitted as it was just dict, we save directly below

            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            np.savez_compressed(
                save_fn,
                root_pos=root_pos,
                root_quat_xyzw=root_rot,
                dof_pos=dof_pos,
                run_time=runtime,
                fps=self.fps,
            )
        except Exception as e:
            print(f"Error processing {human_fn}: {e}")
            import traceback

            traceback.print_exc()


def worker_process(worker_id, cpu_ids, task_queue, args_dict):
    """
    Worker process for parallel retargeting.
    """
    # 1. CPU Pinning
    if cpu_ids:
        try:
            os.sched_setaffinity(0, cpu_ids)
            # print(f"Worker {worker_id} initialized with affinity: {os.sched_getaffinity(0)}")
        except Exception as e:
            print(f"Worker {worker_id} failed to set affinity: {e}")

    # 2. Thread Configuration
    # User requested 8 threads per worker.
    # If explicit cpu_ids provided, they might constrain us, but we set torch threads anyway.
    # Note: If we really want 8 threads, we should ensure we have 8 cores assigned.
    # The caller is responsible for assigning 8 cores if cpus_per_worker is set to 8.

    # However, user explicitly said "every worker need to use 8 thread instead of 4".
    # In phuma script we did: num_threads = len(cpu_ids) if cpu_ids else 1.
    # Here we will respect the passed expectation, but really we should just use len(cpu_ids)
    # assuming main provides 8 cores.
    # Or explicitly set to 8.
    num_threads = 8  # Explicitly requested
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    # 3. Initialize Model
    # device = args_dict["device"] # GMR seems CPU only mostly
    # Initialize GMR wrapper
    gmr_wrapper = GMR_Wrapper()

    # 4. Process Loop
    while True:
        task = task_queue.get()
        if task is None:
            break

        sid, s_len, human_fn, save_fn = task
        # No num_iter arg for GMR used in reference, it does per frame.
        skip = args_dict.get("skip", 1)
        gmr_wrapper.retarget_phuma(human_fn, save_fn, skip=skip)


def main():
    parser = argparse.ArgumentParser(description="Parallel GMR Inference")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes (Default 1 due to high thread count per worker)",
    )
    parser.add_argument(
        "--cpus_per_worker",
        type=int,
        default=8,
        help="Number of CPUs per worker (Default 8)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Skip frames for faster inference (interp back)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/eval/amass_gmr_inferred",
        help="Output directory",
    )
    parser.add_argument(
        "--human_data_root",
        type=str,
        default="./data/amass_phuma_v2/g1/human_states/",
        help="Input data root",
    )
    parser.add_argument(
        "--test_split_fn",
        type=str,
        default="./data_splits/amass_humanoid_v2/full_test.json",
        help="Test split JSON",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda:X)"
    )

    args = parser.parse_args()

    # Expand paths
    args.save_dir = osp.join(PROJECT_ROOT, args.save_dir)
    args.human_data_root = osp.join(PROJECT_ROOT, args.human_data_root)
    args.test_split_fn = osp.join(PROJECT_ROOT, args.test_split_fn)

    # Check CPU availability
    try:
        available_cpus = sorted(list(os.sched_getaffinity(0)))
    except AttributeError:
        try:
            available_cpus = list(range(len(os.sched_getaffinity(0))))
        except:
            available_cpus = list(range(multiprocessing.cpu_count()))

    total_cpus = len(available_cpus)
    required_cpus = args.num_workers * args.cpus_per_worker

    if required_cpus > total_cpus:
        print(
            f"WARNING: Requested {required_cpus} CPUs but only {total_cpus} available/allowed in mask."
        )

    print(f"Starting {args.num_workers} workers with {args.cpus_per_worker} CPUs each.")
    print(f"Allowed CPUs: {available_cpus}")

    # Load Split
    with open(args.test_split_fn, "r") as f:
        test_split = json.load(f)

    # Queue setup
    task_queue = multiprocessing.Queue()

    # Fill Queue
    print(f"Queueing {len(test_split)} tasks...")
    for i in range(len(test_split)):
        sid, s_len = test_split[i]
        human_fn = os.path.join(args.human_data_root, sid + ".npz")
        save_fn = os.path.join(args.save_dir, sid + ".npz")
        task_queue.put((sid, s_len, human_fn, save_fn))

    # Add stop sentinels
    for _ in range(args.num_workers):
        task_queue.put(None)

    # Start Workers
    processes = []
    for i in range(args.num_workers):
        # Calculate CPU set for this worker based on available CPUs
        start_idx = i * args.cpus_per_worker
        end_idx = start_idx + args.cpus_per_worker

        if end_idx <= len(available_cpus):
            cpu_ids = set(available_cpus[start_idx:end_idx])
        else:
            # Fallback
            cpu_ids = set()
            print(
                f"Warning: Worker {i} oversubscribing available CPUs/No dedicated CPUs."
            )

        # Pass args as dict
        args_dict = {"device": args.device, "skip": args.skip}

        p = multiprocessing.Process(
            target=worker_process, args=(i, cpu_ids, task_queue, args_dict)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    print("All tasks completed.")


if __name__ == "__main__":
    main()
