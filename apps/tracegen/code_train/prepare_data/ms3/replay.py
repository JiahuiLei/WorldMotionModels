"""CLI entry-point for ms3 replay conversion.

Usage:
    python -m prepare_data.ms3.replay \
        --env_id PushCube-v1 \
        --traj_path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
        --output_dir data/ms3/umm_format/PushCube-v1/100 \
        --count 100
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

from .ms3_replay_handler import MS3ReplayHandler, save_task_meshes


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env_id", required=True,
                   help="ManiSkill env id (e.g. PushCube-v1).")
    p.add_argument("--traj_path", required=True,
                   help="Path to trajectory.state.<control_mode>.physx_cpu.h5.")
    p.add_argument("--output_dir", required=True,
                   help="Destination dir for per-episode npz + meta.json.")
    p.add_argument("--count", type=int, default=None,
                   help="Number of demos to convert (default: all).")
    p.add_argument("--control-mode", type=str, default=None,
                   help="Override the control mode (default: auto-detect "
                        "from the JSON sidecar). Must be one of "
                        "{pd_ee_delta_pos, pd_ee_delta_pose}.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    traj_path = os.path.expanduser(args.traj_path)
    output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Save per-task meshes at the task root (one level up from output_dir).
    # Shared across demo-count subdirs (100/, 1000/, …) — see schema.py.
    task_root = osp.dirname(osp.abspath(output_dir))
    save_task_meshes(args.env_id, task_root)

    handler = MS3ReplayHandler(args.env_id)
    summary = handler.replay_trajectory(
        traj_h5_path=traj_path,
        output_dir=output_dir,
        count=args.count,
        control_mode=args.control_mode,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_ep_seeds"}, indent=2))
    print(f"Wrote {summary['converted_count']} episodes to {output_dir}")
    print(f"Task meshes at: {task_root}/robot_embodiment.npz + object_embodiment.npz")


if __name__ == "__main__":
    main()
