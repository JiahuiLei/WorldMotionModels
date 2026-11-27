"""Disk-only reference metric recompute for WMM TraceGen eval outputs.

Reads ``<dst>/samples/*.npz`` produced by ``eval_and_save_wmm.py``,
applies the authors' filter (``kept_by_filter``), runs the byte-identical
``_compute_trajectory_metrics_abs`` body, and writes
``<dst>/metrics.json`` + ``<dst>/metrics.txt``.

Called automatically by ``eval_and_save_wmm.py`` at the end of a run.
Can also be invoked manually:

    python eval/tracegen/eval_compute_metrics_wmm.py <dst>
"""

import json
import logging
import os
import os.path as osp
import sys
import time
from glob import glob
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Verbatim copy of _compute_trajectory_metrics_abs from tg_eval_callback.py
# DO NOT refactor into a shared module — the rule is byte-identity with the
# callback body (lei-align constraint: no callback refactor).
# ────────────────────────────────────────────────────────────────────────────


def _compute_trajectory_metrics_abs(
    pred_abs: torch.Tensor,        # (N, T, 3)  absolute predicted traj
    gt_abs: torch.Tensor,          # (N, T, 3)  absolute ground-truth traj
    all_masks: torch.Tensor,       # (N, T, 3)  per-coord bool/0-1 mask
) -> Dict[str, float]:
    """Authors' ``_compute_trajectory_metrics`` inlined, ``absolute_action=True``.

    Attribution: adapted from
    ``data_wmm/tracegen/repo/trainer/trainer.py:790-846`` — the
    ``_compute_trajectory_metrics`` method body.

    Deviations from the source (all intentional and documented):

    1. ``cumsum`` step SKIPPED. Authors' pred / gt are relative deltas;
       they accumulate via ``torch.cumsum(_, dim=1)`` for the
       ``absolute_action=False`` branch. Our pred + gt are already
       absolute (Q9), so the ``absolute_action=True`` branch is
       used, which passes them through unchanged.
    2. ``torch.cat((targets[:, :1, :], preds), dim=1)`` step SKIPPED.
       Authors' ``predicted_trajectory`` has 32 steps (future only);
       they prepend GT step-0 to match the target's 33 steps. Our
       ``AgentV11.inference`` receives ``trace`` with
       ``trace_known_mask[:, 0] = True``, so ``ret_trace[:, 0]``
       is the input's known GT at t=0. The concat is therefore a
       no-op on our pred and is removed for clarity.
    3. ``TRACEGEN_STRIP_Z`` env gating OMITTED. Droid / Epic reference
       numbers in ``data_wmm/tracegen/eval.md`` were reproduced without
       stripping z. For 3D envs the metric is defined over (x, y, z);
       no gating is needed.
    4. ``all_masks is None`` fallback OMITTED. The authors' fallback
       calls ``torch.ones(gt_abs, ...)`` which is itself buggy (passes
       a tensor as the ``size`` argument); we always pass a concrete
       mask, so the branch is unreachable and removed.

    Inputs are expected already flattened to ``(N, T, 3)`` where ``N``
    covers ``B * 400`` keypoints across the batch; the caller flattens.

    Returns:
        Dict with keys ``trajectory_mse``, ``trajectory_mae``, ``endpoint_mse``.
        Returns an empty dict if ``mask.sum() <= 0``.
    """
    # ---- Authors' body (absolute_action=True branch) ----
    # See the docstring above for the explicit list of every deviation
    # from the source. The arithmetic below is byte-identical to
    # trainer.py:816-844.
    all_masks = all_masks.to(pred_abs.dtype)
    total_valid = all_masks.sum()
    if total_valid.item() <= 0:
        log.warning("_compute_trajectory_metrics_abs: no valid masked entries for trajectory metrics")
        return {}

    diff = pred_abs - gt_abs                                # (N, T, 3)
    trajectory_mse = (diff.pow(2) * all_masks).sum() / total_valid
    trajectory_mae = (diff.abs() * all_masks).sum() / total_valid

    valid_mask_time = (all_masks.sum(dim=-1) > 0).to(pred_abs.dtype)  # (N, T)
    valid_counts = valid_mask_time.sum(dim=1)
    has_valid = valid_counts > 0
    if has_valid.any():
        last_idx = (valid_counts.clamp(min=1) - 1).long()
        gather_index = last_idx.view(-1, 1, 1).expand(-1, 1, pred_abs.size(-1))
        pred_endpoints = (pred_abs * all_masks).gather(1, gather_index).squeeze(1)
        target_endpoints = (gt_abs * all_masks).gather(1, gather_index).squeeze(1)
        endpoint_mse = F.mse_loss(
            pred_endpoints[has_valid], target_endpoints[has_valid]
        ).item()
    else:
        endpoint_mse = 0.0

    return {
        "trajectory_mse": trajectory_mse.item(),
        "trajectory_mae": trajectory_mae.item(),
        "endpoint_mse": endpoint_mse,
    }


# ────────────────────────────────────────────────────────────────────────────
# Required keys / shapes (same as eval_and_save_wmm.py integrity gate)
# ────────────────────────────────────────────────────────────────────────────

_REQUIRED_NPZ_KEYS = {
    "pred_trace_abs",
    "gt_trace_abs",
    "pred_trace_se3",
    "gt_trace_se3",
    "trace_valid_mask_per_coord",
    "valid_steps",
    "movement_bool",
    "gt_mask",
    "first_keypoint",
    "image_uint8",
    "depth_raw",
    "kept_by_filter",
}

_REQUIRED_JSON_KEYS = {
    "episode_id",
    "frame_id",
    "text",
    "env_name",
    "seed_used",
    "kept_by_filter",
    "valid_steps_int",
    "movement_count",
    "gt_mask_count",
    "subset_global_idx",
}

_NPZ_EXPECTED_SHAPES = {
    "pred_trace_abs": (33, 400, 3),
    "gt_trace_abs": (33, 400, 3),
    "pred_trace_se3": (33, 400, 4, 4),
    "gt_trace_se3": (33, 400, 4, 4),
    "trace_valid_mask_per_coord": (33, 400, 3),
    "movement_bool": (400,),
    "gt_mask": (400,),
    "first_keypoint": (400, 2),
    "image_uint8": (3, 384, 384),
}


def _run_integrity_gate(dst: str, n_expected: int) -> None:
    """Verify all output files exist with correct shapes.

    Args:
        dst: Eval output root.
        n_expected: Expected number of subset samples.

    Raises:
        RuntimeError: If any file is missing or has wrong shape.
    """
    npz_files = sorted(glob(osp.join(dst, "samples", "*.npz")))
    json_files = sorted(glob(osp.join(dst, "samples", "*.json")))

    if len(npz_files) != n_expected:
        raise RuntimeError(
            "Integrity gate: expected {} npz files, got {}.".format(
                n_expected, len(npz_files)
            )
        )
    if len(json_files) != n_expected:
        raise RuntimeError(
            "Integrity gate: expected {} json files, got {}.".format(
                n_expected, len(json_files)
            )
        )

    for jpath in json_files:
        with open(jpath) as f:
            jdata = json.load(f)
        missing = _REQUIRED_JSON_KEYS - set(jdata.keys())
        if missing:
            raise RuntimeError(
                "Integrity gate: json {} missing keys: {}".format(jpath, missing)
            )

    for npath in npz_files:
        with np.load(npath) as npz:
            present = set(npz.files)
            missing = _REQUIRED_NPZ_KEYS - present
            if missing:
                raise RuntimeError(
                    "Integrity gate: npz {} missing keys: {}".format(npath, missing)
                )
            for key, expected_shape in _NPZ_EXPECTED_SHAPES.items():
                actual = npz[key].shape
                if actual != expected_shape:
                    raise RuntimeError(
                        "Integrity gate: npz {} key '{}' shape {} != expected {}.".format(
                            npath, key, actual, expected_shape
                        )
                    )

    log.info("Integrity gate PASSED: %d npz + %d json verified.", n_expected, n_expected)


def main() -> None:
    """Disk-only metric recompute from saved eval outputs."""
    if len(sys.argv) < 2:
        print("Usage: python eval_compute_metrics_wmm.py <dst>")
        sys.exit(1)

    dst = sys.argv[1]
    t0 = time.time()

    # ── 1. Read rollout_config.json for sanity ───────────────────────────────
    rollout_cfg_path = osp.join(dst, "rollout_config.json")
    if not osp.isfile(rollout_cfg_path):
        raise RuntimeError(
            "rollout_config.json not found at {}. "
            "Is this a valid eval output dir?".format(rollout_cfg_path)
        )
    with open(rollout_cfg_path) as f:
        rollout_cfg = json.load(f)

    env_name = rollout_cfg.get("env", "unknown")
    n_expected = int(rollout_cfg.get("num_samples_subset", -1))
    if n_expected <= 0:
        raise RuntimeError(
            "rollout_config.json has invalid num_samples_subset: {}".format(n_expected)
        )
    log.info(
        "eval_compute_metrics_wmm: dst=%s  env=%s  expected=%d",
        dst, env_name, n_expected,
    )

    # ── 2. Integrity gate ────────────────────────────────────────────────────
    _run_integrity_gate(dst, n_expected)

    # ── 3. Load + accumulate kept samples ────────────────────────────────────
    npz_files = sorted(glob(osp.join(dst, "samples", "*.npz")))
    json_files = sorted(glob(osp.join(dst, "samples", "*.json")))

    pred_list = []
    gt_list = []
    mask_list = []
    n_samples_valid = 0

    for npath, jpath in zip(npz_files, json_files):
        with open(jpath) as f:
            jdata = json.load(f)
        kept = bool(jdata["kept_by_filter"])

        if not kept:
            continue

        with np.load(npath) as npz:
            pred_abs = npz["pred_trace_abs"].astype(np.float32)  # (33, 400, 3)
            gt_abs = npz["gt_trace_abs"].astype(np.float32)      # (33, 400, 3)
            mask_pc = npz["trace_valid_mask_per_coord"].astype(bool)  # (33, 400, 3)

        pred_list.append(torch.from_numpy(pred_abs))
        gt_list.append(torch.from_numpy(gt_abs))
        mask_list.append(torch.from_numpy(mask_pc))
        n_samples_valid += 1

    log.info(
        "eval_compute_metrics_wmm: loaded %d / %d kept samples",
        n_samples_valid, n_expected,
    )

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    if n_samples_valid == 0:
        log.warning("eval_compute_metrics_wmm: no kept samples; metrics will be NaN.")
        metrics: Dict[str, Any] = {
            "trajectory_mse_x100": float("nan"),
            "trajectory_mae_x100": float("nan"),
            "endpoint_mse_x100": float("nan"),
            "n_samples_evaluated": n_expected,
            "n_samples_valid": 0,
            "wall_clock_s": time.time() - t0,
        }
    else:
        pred_cat = torch.stack(pred_list, dim=0)   # (n_valid, 33, 400, 3)
        gt_cat = torch.stack(gt_list, dim=0)       # (n_valid, 33, 400, 3)
        mask_cat = torch.stack(mask_list, dim=0)   # (n_valid, 33, 400, 3)

        B_eff, T, P, C = pred_cat.shape
        assert T == 33 and P == 400 and C == 3, (
            "Unexpected shape: {}".format(pred_cat.shape)
        )
        pred_flat = pred_cat.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)
        gt_flat = gt_cat.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)
        mask_flat = mask_cat.permute(0, 2, 1, 3).reshape(B_eff * P, T, C)

        raw = _compute_trajectory_metrics_abs(pred_flat, gt_flat, mask_flat)
        wall_clock_s = time.time() - t0

        if not raw:
            metrics = {
                "trajectory_mse_x100": float("nan"),
                "trajectory_mae_x100": float("nan"),
                "endpoint_mse_x100": float("nan"),
                "n_samples_evaluated": n_expected,
                "n_samples_valid": n_samples_valid,
                "wall_clock_s": wall_clock_s,
            }
        else:
            metrics = {
                "trajectory_mse_x100": float(raw["trajectory_mse"] * 100.0),
                "trajectory_mae_x100": float(raw["trajectory_mae"] * 100.0),
                "endpoint_mse_x100": float(raw["endpoint_mse"] * 100.0),
                "n_samples_evaluated": n_expected,
                "n_samples_valid": n_samples_valid,
                "wall_clock_s": wall_clock_s,
            }

    # ── 5. Write metrics.json ────────────────────────────────────────────────
    # RFC 8259-compliant: NaN -> null. Strict consumers (jq, JS) reject bare
    # `NaN`. Python's json.loads can read both, so the round-trip used by
    # the bit-identity assertion in eval_and_save still works after we read
    # `null` back as None and translate to float('nan') there.
    def _nan_to_null(v: Any) -> Any:
        if isinstance(v, float) and (v != v):  # NaN check (NaN != NaN)
            return None
        return v
    metrics_serializable = {k: _nan_to_null(v) for k, v in metrics.items()}
    metrics_json_path = osp.join(dst, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2, allow_nan=False)
    log.info("Wrote %s", metrics_json_path)

    # ── 6. Write metrics.txt ─────────────────────────────────────────────────
    metrics_txt_path = osp.join(dst, "metrics.txt")
    txt_lines = [
        "=== TraceGen disk-recompute metrics ===",
        "env: {}".format(env_name),
        "n_samples_evaluated: {}".format(metrics["n_samples_evaluated"]),
        "n_samples_valid: {}".format(metrics["n_samples_valid"]),
        "trajectory_mse_x100:  {:.4f}".format(metrics["trajectory_mse_x100"]),
        "trajectory_mae_x100:  {:.4f}".format(metrics["trajectory_mae_x100"]),
        "endpoint_mse_x100:    {:.4f}".format(metrics["endpoint_mse_x100"]),
        "wall_clock_s: {:.1f}".format(metrics["wall_clock_s"]),
    ]
    txt_content = "\n".join(txt_lines) + "\n"
    with open(metrics_txt_path, "w") as f:
        f.write(txt_content)
    log.info("Wrote %s", metrics_txt_path)

    # ── 7. Print to stdout ───────────────────────────────────────────────────
    print(txt_content)


if __name__ == "__main__":
    main()
