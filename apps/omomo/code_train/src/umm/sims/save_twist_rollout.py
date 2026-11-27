"""Batch physics rollout and forward-kinematics export for G1 humanoid motions.

Reads motion ``.pkl`` files produced by the TWIST tracker, replays each
through a MuJoCo low-level controller, and writes:

1. A physics-simulated ``.pkl`` with root/DOF states at 30 fps.
2. A compressed ``.npz`` with aligned SE(3) link poses obtained via FK.

The script supports multi-process execution via ``multiprocessing.Pool``
(spawn context) with per-worker CPU pinning and single-thread enforcement
to avoid oversubscription.

Usage (standalone)::

    WORKERS=8 python -m umm.sims.save_twist_rollout
"""

import os as _early_os

_early_os.environ.setdefault("OMP_NUM_THREADS", "1")
_early_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_early_os.environ.setdefault("MKL_NUM_THREADS", "1")
_early_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_early_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_early_os.environ.setdefault("BLIS_NUM_THREADS", "1")

from lib_twist.mujoco_lowlevel_controller import (
    MuJoCoLowLevelController,
    create_constant_command,
    save_motion_pkl,
    convert_fps,
)
import numpy as np
import os
from lib_twist.g1_interface_utils import DEFAULT_MIMIC_OBS
from lib_twist.g1_interface import G1Interface
import multiprocessing as mp
from math import ceil
import torch


class Processor:
    """Physics-rollout processor for G1 humanoid motions.

    Loads a G1 URDF and MuJoCo XML once, then exposes a ``process`` method
    that takes a raw TWIST motion ``.pkl``, simulates it through the
    low-level controller, and exports aligned FK poses.

    Example:
        >>> proc = Processor()
        >>> proc.process("motion.pkl", "out.pkl", "out.npz")
    """

    def __init__(self):
        """Initialise the G1 URDF interface and MuJoCo low-level controller."""

        urdf_file = "./ext/lib_twist/src/lib_twist/assets/g1/g1_custom_collision_with_fixed_hand.urdf"
        self.G1I = G1Interface(urdf_file, device="cpu")

        XML_FILE = (
            "./ext/lib_twist/src/lib_twist/assets/g1/g1_sim2sim_with_wrist_roll.xml"
        )
        POLICY_PATH = (
            "./ext/lib_twist/src/lib_twist/assets/twist_general_motion_tracker.pt"
        )

        # Create controller
        self.controller_env = MuJoCoLowLevelController(
            xml_file=XML_FILE,
            policy_path=POLICY_PATH,
            device="cpu",  # or 'cpu'
            control_dt=0.02,  # 50 Hz control rate
            sim_dt=0.001,  # 1000 Hz simulation rate
        )

    def process(self, motion_pkl_path: str, save_pkl_path: str, save_Q_path: str):
        """Run a single motion through the physics controller and export results.

        Steps:
            1. Parse the TWIST ``.pkl`` motion file.
            2. Compute mimic observations and run the MuJoCo controller at
               50 Hz, then convert to 30 fps.
            3. Align simulated trajectory lengths with the original motion.
            4. Save the physics-simulated states to ``save_pkl_path``.
            5. Compute forward kinematics, align to the reference first frame,
               and save the link SE(3) poses to ``save_Q_path`` (``.npz``).

        Args:
            motion_pkl_path: Path to the input TWIST motion ``.pkl`` file.
            save_pkl_path: Output path for the physics-simulated state
                ``.pkl`` file.
            save_Q_path: Output path for the aligned FK poses ``.npz`` file.
        """
        os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_Q_path), exist_ok=True)

        motion_data = G1Interface.parse_twist_pkl_motion(motion_pkl_path)
        commands = self.G1I.get_mimic_obs(
            motion_data["root_pos"],
            motion_data["root_rot"],
            motion_data["dof_pos"],
            motion_data["fps"],
            1 / 50.0,
        )
        stats_50fps = self.controller_env.run(
            target_commands=commands,
        )
        stats = convert_fps(stats_50fps, 30)
        # assert (
        #     len(motion_data["root_pos"]) == len(stats["root_pos"]) - 1
        # ), f"len(motion_data['root_pos']) != len(stats['root_pos'])-1: {len(motion_data['root_pos'])} != {len(stats['root_pos'])-1}"

        assert len(motion_data["root_pos"]) <= len(
            stats["root_pos"]
        ), f"len(motion_data['root_pos']) != len(stats['root_pos'])-1: {len(motion_data['root_pos'])} != {len(stats['root_pos'])-1}"
        if len(motion_data["root_pos"]) < len(stats["root_pos"]) - 1:
            _n = len(motion_data["root_pos"]) + 1
            stats["root_pos"] = stats["root_pos"][:_n]
            stats["root_rot"] = stats["root_rot"][:_n]
            stats["dof_pos"] = stats["dof_pos"][:_n]
            stats["root_vel"] = stats["root_vel"][:_n]
            stats["root_ang_vel"] = stats["root_ang_vel"][:_n]
            stats["dof_vel"] = stats["dof_vel"][:_n]
        elif len(motion_data["root_pos"]) == len(stats["root_pos"]):
            # pad wiht the last one state!
            stats["root_pos"] = np.concatenate(
                [stats["root_pos"], stats["root_pos"][-1:]], axis=0
            )
            stats["root_rot"] = np.concatenate(
                [stats["root_rot"], stats["root_rot"][-1:]], axis=0
            )
            stats["dof_pos"] = np.concatenate(
                [stats["dof_pos"], stats["dof_pos"][-1:]], axis=0
            )
            stats["root_vel"] = np.concatenate(
                [stats["root_vel"], stats["root_vel"][-1:]], axis=0
            )
            stats["root_ang_vel"] = np.concatenate(
                [stats["root_ang_vel"], stats["root_ang_vel"][-1:]], axis=0
            )
            stats["dof_vel"] = np.concatenate(
                [stats["dof_vel"], stats["dof_vel"][-1:]], axis=0
            )
        assert (
            len(motion_data["root_pos"]) == len(stats["root_pos"]) - 1
        ), f"len(motion_data['root_pos']) != len(stats['root_pos'])-1: {len(motion_data['root_pos'])} != {len(stats['root_pos'])-1}"

        save_motion_pkl(stats, save_pkl_path, start_from=1)
        Q = self.G1I.FK(
            stats["root_pos"],
            stats["root_rot"],
            stats["dof_pos"],
            names=self.G1I.link_names,
        )

        motion_data = G1Interface.parse_twist_pkl_motion(motion_pkl_path)
        Q_ref = self.G1I.FK(
            motion_data["root_pos"][:1],
            motion_data["root_rot"][:1],
            motion_data["dof_pos"][:1],
            names=self.G1I.link_names,
        )
        Q_ref = Q_ref[0, 0]  # T_
        # Q is T_wl, so Q' = T_wl @ Tll'; so right operation
        Q_cur = Q[0, 0]  # T_w_cur
        # Q_w_cur @ Q_cur_w @ Q_w_ref = Q_w_ref
        # right_T  = torch.linalg.inv(Q_cur) @ Q_ref
        left_T = Q_ref @ Q_cur.inverse()
        Q_aligned = torch.einsum("ij,tnjk->tnik", left_T, Q)

        save_dict = {
            "Q": Q_aligned.cpu().numpy().astype(np.float16),
            "fps": int(motion_data["fps"]),
            "names": self.G1I.link_names,
        }
        np.savez_compressed(save_Q_path, **save_dict)
        return


# Global holder for a single Processor per worker
_WORKER_PROCESSOR = None


def _limit_threads_to_one() -> None:
    """Force all major numeric backends to use a single CPU thread.

    Sets environment variables for OpenMP, OpenBLAS, MKL, NumExpr,
    vecLib, and BLIS, and configures PyTorch intra-/inter-op thread
    counts to 1.  Called both at module level (via early ``setdefault``)
    and inside each pool worker initialiser.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    # PyTorch controls
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _set_cpu_affinity_for_worker(worker_index: int) -> None:
    """Pin the current process to a single CPU core (Linux only, best-effort).

    Uses ``os.sched_setaffinity`` to bind the process to the core at
    position ``worker_index % len(available_cores)``.  Silently ignored
    on non-Linux platforms or on failure.

    Args:
        worker_index: Zero-based index of the pool worker.
    """
    try:
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            available = sorted(list(os.sched_getaffinity(0)))
            if not available:
                return
            cpu_id = available[worker_index % len(available)]
            os.sched_setaffinity(0, {cpu_id})
    except Exception:
        # Affinity is best-effort; ignore failures
        pass


def _init_worker(total_workers: int) -> None:
    """Pool initialiser: enforce single-thread limits, pin CPU, and build a ``Processor``.

    Called once per ``multiprocessing.Pool`` worker at spawn time via
    ``initializer=_init_worker``.  The heavy ``Processor`` instance is
    stored in the module-level ``_WORKER_PROCESSOR`` global so it is
    reused across all tasks dispatched to the same worker.

    Args:
        total_workers: Total number of pool workers (unused directly but
            available for future scheduling logic).
    """
    _limit_threads_to_one()
    try:
        # In Pool workers, identity is (1..N)
        ident = getattr(mp.current_process(), "_identity", (1,))
        worker_idx = (ident[0] - 1) if ident else 0
    except Exception:
        worker_idx = 0
    _set_cpu_affinity_for_worker(worker_idx)
    global _WORKER_PROCESSOR
    _WORKER_PROCESSOR = Processor()


def _process_chunk(arg_chunk):
    """Process a chunk of motion files inside a pool worker.

    Each element of *arg_chunk* is a 3-tuple
    ``(motion_pkl_path, save_pkl_path, save_Q_path)`` forwarded to
    ``Processor.process``.  Errors on individual items are caught and
    printed so that one bad file does not abort the entire chunk.

    Args:
        arg_chunk: Sequence of ``(motion_pkl_path, save_pkl_path,
            save_Q_path)`` tuples.

    Returns:
        Number of successfully processed items in the chunk.
    """
    global _WORKER_PROCESSOR
    if _WORKER_PROCESSOR is None:
        _WORKER_PROCESSOR = Processor()
    completed = 0
    for arg in arg_chunk:
        try:
            _WORKER_PROCESSOR.process(*arg)
            completed += 1
        except Exception as e:
            print(f"[worker] error processing {arg[0]}: {e}", flush=True)
    return completed


if __name__ == "__main__":
    from tqdm import tqdm
    import os.path as osp

    # motion_path = "/media/ray/1tbssd/twist_track_dataset/twist_motion_dataset"
    # phy_pkl_path = "../data/twist_rollout2/g1_phy_pkls"
    # npz_path = "../data/twist_rollout2/g1_phy_states"

    motion_path = (
        "data/twist_track_dataset/twist_motion_dataset"
    )
    phy_pkl_path = "data/twist_rollout2/g1_phy_pkls"
    npz_path = "data/twist_rollout2/g1_phy_states"

    # Find all pkl files recursively under motion_path
    pkl_files = []
    for root, dirs, files in os.walk(motion_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    print(f"Found {len(pkl_files)} pkl files")
    pkl_files.sort()
    args = []
    for pkl_file in pkl_files:
        name = f"{osp.basename(osp.dirname(pkl_file))}__{osp.basename(pkl_file)}"
        args.append(
            (
                pkl_file,
                os.path.join(phy_pkl_path, name),
                os.path.join(npz_path, name.replace(".pkl", ".npz")),
            )
        )

    # print()
    # processor = Processor()
    # for arg in tqdm(args):
    #     processor.process(*arg)
    # raise Exception("Stop here")

    # Determine workers and split the workload into chunks per worker
    available_cpus = (
        len(sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    num_workers = int(os.environ.get("WORKERS", str(available_cpus)))
    num_workers = max(
        1, min(num_workers, available_cpus - 2 if available_cpus else num_workers)
    )

    if num_workers == 1:
        # Single-process fallback (still respect single-thread limits)
        _limit_threads_to_one()
        processor = Processor()
        for arg in tqdm(args):
            processor.process(*arg)
    else:
        chunk_size = ceil(len(args) / num_workers) if len(args) > 0 else 1
        chunks = [args[i : i + chunk_size] for i in range(0, len(args), chunk_size)]
        ctx = mp.get_context("spawn")
        total = len(args)
        with ctx.Pool(
            processes=num_workers, initializer=_init_worker, initargs=(num_workers,)
        ) as pool:
            completed = 0
            for n in tqdm(
                pool.imap_unordered(_process_chunk, chunks), total=len(chunks)
            ):
                completed += n
        print(f"Completed {completed}/{total} items with {num_workers} workers.")
