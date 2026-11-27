"""Round-trip bit-exact correctness test for ms3 umm_format.

Compares each converted ``traj_<N>.npz`` field against the source
``trajectory.state.<control_mode>.physx_cpu.h5`` used to produce it. Purpose:
catch ANY quantization / sign / axis-swap / ordering bug introduced by the
convert pipeline, before a downstream UMM model learns on the npz and
produces rollouts that diverge from the env dynamics.

Why this matters: ``verify_replay.py`` is a *behavioral* check (does the saved
action sequence reach the success flag?). This diff is a *structural* check
(is the npz byte-identical to its h5 source for every field?). Both are
required.

Tolerances
----------
* ``action``, ``success``, ``terminated``, ``truncated`` — **exact equality.**
  No tolerance permitted. These are the load-bearing fields for BC training
  and evaluator dispatch.
* ``qpos``, ``qvel``, ``tcp_pose``, ``obj_poses/*``, ``goal/*`` — compared
  against slices of the h5 ``obs`` array (per-task layout from
  the design notes). The npz
  values were produced by calling ``env.unwrapped._get_obs_state_dict`` on
  the same env_state that produced the h5 ``obs``; they should be bit-equal.
  We use **exact equality first**; if that fails, a secondary allclose with
  ``atol=1e-6`` is reported as diagnostic (but still flagged as FAIL).
* ``scene_params/*`` (schema v3) — compared against a fresh env reset at the
  stored ``meta["seed"]``. The stored values were extracted from h5.obs so
  env-reset verification is a fully independent second source. Exact
  float32 equality required.

Episode matching
----------------
Episodes are matched by the ``meta["episode_id"]`` field in the npz against
the ``traj_<id>`` group name in the h5. The vendor's retargeter may drop
some episodes (e.g. PegInsertionSide 946/1000 survived pos→pose retarget),
so file-index matching is unsafe.

CLI
---
    python -m prepare_data.ms3.round_trip_diff \\
        --env_id PushCube-v1 \\
        --h5 ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \\
        --input_dir data/ms3/umm_format/PushCube-v1/100

    # PegIns with scene_params verification (defaults ON; env-reset per ep):
    python -m prepare_data.ms3.round_trip_diff \\
        --env_id PegInsertionSide-v1 \\
        --h5 ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \\
        --input_dir data/ms3/umm_format/PegInsertionSide-v1/100

    # Skip scene_params (speed up when not needed):
    python -m prepare_data.ms3.round_trip_diff ... --no-scene-params
"""
from __future__ import annotations

import argparse
import glob
import json
import os.path as osp
import sys
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from prepare_data.ms3.schema import (
    load_episode,
    OBS_SLICES,
    TASK_CONTROL_MODE,
    TASK_SCENE_PARAM_FIELDS,
)


def _npz_field(ep: dict, key: str) -> np.ndarray:
    """Resolve a dot-like key ('qpos', 'goal/goal_pos', 'obj_poses/cubeA_pose')."""
    if "/" in key:
        group, leaf = key.split("/", 1)
        return np.asarray(ep[group][leaf])
    return np.asarray(ep[key])


def _diff(npz_arr: np.ndarray, h5_arr: np.ndarray, label: str, eid: int) -> dict:
    """Return per-field diff summary. Empty dict == exact match."""
    if npz_arr.shape != h5_arr.shape:
        return {
            "label": label, "eid": eid,
            "reason": f"shape mismatch npz={npz_arr.shape} h5={h5_arr.shape}",
        }
    if np.array_equal(npz_arr, h5_arr):
        return {}
    # Not exact — diagnostic stats.
    diff = np.abs(npz_arr.astype(np.float64) - h5_arr.astype(np.float64))
    n_bad = int((diff > 0).sum())
    max_abs = float(diff.max())
    # First offending index for human readability.
    bad_idx = tuple(int(x) for x in np.argwhere(diff > 0)[0])
    return {
        "label": label, "eid": eid,
        "reason": f"not bit-equal: n_bad={n_bad}, max_abs={max_abs:.3e}, "
                  f"first_bad_idx={bad_idx}, "
                  f"npz[idx]={float(npz_arr[bad_idx]):.9e}, "
                  f"h5[idx]={float(h5_arr[bad_idx]):.9e}",
        "max_abs": max_abs,
        "n_bad": n_bad,
    }


def _extract_env_scene_params(env_id: str, seed: int) -> Dict[str, np.ndarray]:
    """Reset a fresh env at ``seed`` and extract its scene_params.

    Ground-truth source: reads directly from ``env.unwrapped.peg_half_sizes``
    / ``peg_head_offsets`` (the objects instantiated at scene-build time).
    Independent of ``h5.obs`` — that's the whole point of the round-trip
    check.
    """
    spec = TASK_SCENE_PARAM_FIELDS.get(env_id, {})
    if not spec:
        return {}

    # Import here so the import cost only hits when scene_params checking
    # is actually enabled.
    import gymnasium as gym
    import mani_skill  # noqa: F401  # registers env ids
    import torch

    env = gym.make(env_id, obs_mode="state")
    try:
        env.reset(seed=int(seed))
        u = env.unwrapped
        # Defensive: this helper indexes [0] on per-env tensors (phs, pho)
        # and assumes a single-env build. If gym.make ever returns a
        # num_envs>1 env by default, the wrong peg would be silently read.
        num_envs = int(getattr(u, "num_envs", 1))
        if num_envs != 1:
            raise RuntimeError(
                f"_extract_env_scene_params expects num_envs=1, got "
                f"{num_envs}. Indexing [0] would silently pick one env "
                f"out of many; fix by iterating per-env or forcing "
                f"num_envs=1 at gym.make."
            )
        out: Dict[str, np.ndarray] = {}
        if env_id == "PegInsertionSide-v1":
            phs = u.peg_half_sizes
            pho = u.peg_head_offsets.p  # (num_envs, 3)
            bho = u.box_hole_offsets.p  # (num_envs, 3)
            if torch.is_tensor(phs):
                phs = phs.cpu().numpy()
            if torch.is_tensor(pho):
                pho = pho.cpu().numpy()
            if torch.is_tensor(bho):
                bho = bho.cpu().numpy()
            # (num_envs=1, 3) → (3,)
            out["peg_half_sizes"]   = np.asarray(phs[0], dtype=np.float32)
            out["peg_head_offsets"] = np.asarray(pho[0], dtype=np.float32)
            out["box_hole_offsets"] = np.asarray(bho[0], dtype=np.float32)
        else:
            raise RuntimeError(
                f"TASK_SCENE_PARAM_FIELDS has a non-empty spec for "
                f"{env_id!r} but _extract_env_scene_params has no handler."
            )
        # Shape gate.
        for k, expected_shape in spec.items():
            assert out[k].shape == expected_shape, (
                f"env scene_params[{k!r}] shape {out[k].shape} != "
                f"{expected_shape}"
            )
        return out
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env_id", required=True, choices=sorted(OBS_SLICES))
    p.add_argument("--h5", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--json_out", default=None,
                   help="Optional: write per-field count summary to JSON.")
    p.add_argument(
        "--no-scene-params", action="store_true",
        help="Skip scene_params env-reset verification (speeds up the "
             "check when env construction is expensive).",
    )
    args = p.parse_args()

    h5_path = osp.expanduser(args.h5)
    input_dir = osp.expanduser(args.input_dir)

    # Sanity: control_mode from h5 sidecar should match TASK_CONTROL_MODE.
    json_sidecar = h5_path[:-3] + ".json"
    with open(json_sidecar) as f:
        cm = json.load(f)["env_info"]["env_kwargs"]["control_mode"]
    expected_cm = TASK_CONTROL_MODE[args.env_id]
    if cm != expected_cm:
        print(f"FAIL: h5 control_mode {cm!r} != TASK_CONTROL_MODE[{args.env_id!r}] = {expected_cm!r}",
              file=sys.stderr)
        return 2

    traj_paths = sorted(glob.glob(osp.join(input_dir, "traj_*.npz")))
    if not traj_paths:
        print(f"FAIL: no traj_*.npz in {input_dir}", file=sys.stderr)
        return 2

    slice_map = OBS_SLICES[args.env_id]
    scene_spec = TASK_SCENE_PARAM_FIELDS.get(args.env_id, {})
    scene_check = (not args.no_scene_params) and bool(scene_spec)
    # Counts per field (match / mismatch).
    match_count: Dict[str, int] = {
        "action": 0, "success": 0, "terminated": 0, "truncated": 0,
        **{k: 0 for k in slice_map},
    }
    if scene_check:
        for k in scene_spec:
            match_count[f"scene_params/{k}"] = 0
    mismatches: List[dict] = []

    with h5py.File(h5_path, "r") as hf:
        h5_keys = set(hf.keys())
        for np_path in traj_paths:
            ep = load_episode(np_path)
            eid = int(ep["meta"]["episode_id"])
            traj_key = f"traj_{eid}"
            if traj_key not in h5_keys:
                mismatches.append({
                    "label": "traj_key", "eid": eid,
                    "reason": f"h5 missing group {traj_key!r}"
                })
                continue
            g = hf[traj_key]

            # Per-transition fields (length T): action, success, terminated, truncated.
            for field in ("action", "success", "terminated", "truncated"):
                npz_v = np.asarray(ep[field])
                # h5 names 'actions' plural for the action slot.
                h5_key = "actions" if field == "action" else field
                h5_v = np.asarray(g[h5_key][:])
                # match dtype expectation: h5 bools are already bool.
                d = _diff(npz_v, h5_v, field, eid)
                if not d:
                    match_count[field] += 1
                else:
                    mismatches.append(d)

            # Per-state fields: slice h5 obs and compare to npz.
            obs = np.asarray(g["obs"][:])  # (T+1, D)
            for key, (s, e) in slice_map.items():
                h5_slice = obs[:, s:e]
                npz_v = _npz_field(ep, key)
                d = _diff(npz_v, h5_slice, key, eid)
                if not d:
                    match_count[key] += 1
                else:
                    mismatches.append(d)

            # scene_params (v3): re-reset env at stored seed, compare
            # against npz. Independent source — the npz was extracted from
            # h5.obs; the env is the original ground truth.
            if scene_check:
                seed = int(ep["meta"]["seed"])
                env_sp = _extract_env_scene_params(args.env_id, seed)
                npz_sp = ep.get("scene_params", {})
                for k in scene_spec:
                    label = f"scene_params/{k}"
                    if k not in npz_sp:
                        mismatches.append({
                            "label": label, "eid": eid,
                            "reason": f"npz missing scene_params[{k!r}]",
                        })
                        continue
                    d = _diff(np.asarray(npz_sp[k]), env_sp[k], label, eid)
                    if not d:
                        match_count[label] += 1
                    else:
                        mismatches.append(d)

    total = len(traj_paths)
    print(f"\n=== round-trip diff: {args.env_id} — {total} episodes ===")
    print(f"{'field':30s}  match/total")
    for k, v in match_count.items():
        flag = "OK" if v == total else "FAIL"
        print(f"  {k:28s}  {v}/{total}  [{flag}]")
    if mismatches:
        print(f"\nFIRST 10 MISMATCHES (of {len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"  eid={m['eid']}  field={m['label']}  {m['reason']}")

    passed = all(c == total for c in match_count.values()) and not mismatches

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "env_id": args.env_id,
                "total_episodes": total,
                "match_count": match_count,
                "mismatches_count": len(mismatches),
                "first_mismatches": mismatches[:50],
                "passed": bool(passed),
            }, f, indent=2)
        print(f"wrote {args.json_out}")

    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
