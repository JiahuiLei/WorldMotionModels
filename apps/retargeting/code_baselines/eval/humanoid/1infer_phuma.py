import os

# Set environment variables BEFORE importing torch/numpy to avoid contention
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
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R

# Add project root to sys.path
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.lib_twist.g1_interface import G1Interface
from data_processing.lib_human.human_model import HumanModel
from data_processing.lib_phuma.phuma_wrapper import SMPLXPreprocessor, PHUMA_Processor


class RetargetPhuma:
    def __init__(self, fps, skip=1, device="cpu"):
        self.fps = fps / skip
        self.skip = skip
        self.processor = SMPLXPreprocessor(fps=self.fps)
        self.retargetor = PHUMA_Processor(device=device, fps=self.fps)

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

    def retarget_phuma(self, human_fn, save_fn, num_iter=3001):
        try:
            smplx_pose, smplx_transl, smplx_beta = self.load_human_data(human_fn)

            # ! interpolate, ensure two ends are known
            if self.skip > 1:
                index = np.arange(0, len(smplx_pose), self.skip).tolist()
                if len(smplx_pose) - 1 not in index:
                    index = index + [len(smplx_pose) - 1]
                index = np.array(index)
                interp_mask = np.zeros(len(smplx_pose), dtype=bool)
                interp_mask[index] = True  # later use this to do linear interp
                smplx_pose = smplx_pose[index]
                smplx_transl = smplx_transl[index]

            smplx_pose = smplx_pose.astype("float32")
            smplx_transl = smplx_transl.astype("float32")
            smplx_beta = smplx_beta.astype("float32")
            global_orient = smplx_pose[:, :3]
            body_pose = smplx_pose[:, 3:]
            global_orient, transl = self.processor.amass_to_phuma_space(
                global_orient, smplx_transl, betas=smplx_beta
            )
            betas = smplx_beta
            cur_results, failure_results = self.processor.curate(
                body_pose=body_pose.copy(),
                transl=transl.copy(),
                global_orient=global_orient.copy(),
                betas=betas.copy(),
                chunk_size_seconds=100000000.0,
                chunk_overlap_seconds=1.0,
                visualize_flag=False,
                return_failure_results=True,
            )
            cur_passed = len(cur_results) > 0
            results = cur_results + failure_results  # ! consider all!
            if len(results) != 1:
                print(f"Warning: Expected 1 result, got {len(results)} for {human_fn}")
                return

            # print(f"Retargeting {human_fn}...")
            start_t = time.time()
            ret_result = self.retargetor.retarget(
                curated_item=results[0],
                robot_name="g1",
                num_iter_dof=num_iter,
                print_every=max(1, num_iter // 10),
                visualize=False,
            )
            runtime = time.time() - start_t
            # interpolate back
            root_pos = ret_result["root_pos"]
            root_rot = ret_result["root_ori"]
            dof_pos = ret_result["dof_pos"]

            # ! de-interpolate
            if self.skip > 1:
                # use linear interp to undo the above slicing
                full_idx = np.arange(len(interp_mask))
                root_pos = interp1d(index, root_pos, axis=0)(full_idx)
                dof_pos = interp1d(index, dof_pos, axis=0)(full_idx)

                slerp = Slerp(index, R.from_quat(root_rot))
                root_rot = slerp(full_idx).as_quat()

                assert (
                    len(root_pos) == len(root_rot) == len(dof_pos) == len(interp_mask)
                )

            # print(f"Runtime for {human_fn}: {runtime}s")

            # save
            if ret_result is not None:
                os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                np.savez(
                    save_fn,
                    root_pos=root_pos,
                    root_quat_xyzw=root_rot,
                    dof_pos=dof_pos,
                    run_time=runtime,
                    fps=self.fps * self.skip,
                    cur_passed=cur_passed,
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
            print(
                f"Worker {worker_id} initialized with affinity: {os.sched_getaffinity(0)}"
            )
        except Exception as e:
            print(f"Worker {worker_id} failed to set affinity: {e}")

    # 2. Thread Configuration
    # Allow the worker to use all assigned CPUs for intra-op parallelism
    num_threads = len(cpu_ids) if cpu_ids else 1
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    # print(f"Worker {worker_id}: Affinity={os.sched_getaffinity(0)}, PyTorch Threads={torch.get_num_threads()}")

    # 3. Initialize Model
    # Reconstruct args from dict if needed or just use values
    device = args_dict["device"]
    fps = 30.0  # fixed in notebook
    skip = args_dict.get("skip", 1)

    phuma = RetargetPhuma(fps=fps, skip=skip, device=device)

    # 4. Process Loop
    while True:
        task = task_queue.get()
        if task is None:
            break

        sid, s_len, human_fn, save_fn = task
        phuma.retarget_phuma(human_fn, save_fn, num_iter=args_dict["num_iter"])

    # print(f"Worker {worker_id} finished.")


def main():
    parser = argparse.ArgumentParser(description="Parallel PHUMA Inference")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--cpus_per_worker", type=int, default=1, help="Number of CPUs per worker"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/eval/amass_phuma3001",
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
        "--num_iter", type=int, default=3001, help="Number of iterations"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda:X)"
    )
    parser.add_argument("--skip", type=int, default=1, help="Frame skipping factor")

    args = parser.parse_args()

    # Expand paths
    args.save_dir = osp.join(PROJECT_ROOT, args.save_dir)
    args.human_data_root = osp.join(PROJECT_ROOT, args.human_data_root)
    args.test_split_fn = osp.join(PROJECT_ROOT, args.test_split_fn)

    # Check CPU availability
    try:
        available_cpus = sorted(list(os.sched_getaffinity(0)))
    except AttributeError:
        # Fallback for systems without sched_getaffinity
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
            # Fallback if oversubscribed: just explicitly default or wrap around?
            # Wrapping around might cause collision but better than crash.
            # User wants "non collided cpus".
            # If oversubscribed, we just give what we can or warn.
            print(f"Warning: Worker {i} oversubscribing available CPUs.")
            cpu_ids = set()  # Or maybe fallback to none to let OS decide?
            # Let's try to wrap around to keep it running
            # cpu_ids = {available_cpus[x % len(available_cpus)] for x in range(start_idx, end_idx)}

        # Pass args as dict
        args_dict = {
            "device": args.device,
            "num_iter": args.num_iter,
            "skip": args.skip,
        }

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
