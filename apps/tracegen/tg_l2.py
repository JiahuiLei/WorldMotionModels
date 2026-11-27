"""Recompute the TraceGen table from fresh Layer-2 and Layer-3 evaluation outputs.

Layer-2 runs: ``L2_outputs/<run>_<timestamp>/samples/sample_NNNNNN.{npz,json}`` written by
``code/eval/tracegen/eval_and_save_wmm.py`` (same layout as the original evaluation); Layer 3 (retrained
checkpoints, ``L3_eval.sh``) writes the same layout to ``L3_eval_outputs/<run>_<env>_<timestamp>/``.
Baseline runs: ``L2_outputs/tracegen_authors_<env>_<timestamp>/eval.log`` written by the
TraceGen authors' ``test_benchmark.py``, which prints only dataset-level numbers.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import tg_l1

# (table row, env) -> L2 run name used by L2_eval.sh
L2_RUNS = {
    ("per-dataset (P=128)", "droid"): "droid_k128_50k_droid",
    ("per-dataset (P=128)", "epic"): "epic_k128_30k_epic",
    ("per-dataset (P=400)", "droid"): "droid_k400_80k_droid",
    ("per-dataset (P=400)", "epic"): "epic_k400_60k_epic",
    ("Joint (P=128)", "droid"): "droidepic_k128_45k_droid",
    ("Joint (P=128)", "epic"): "droidepic_k128_45k_epic",
    ("Joint (P=400)", "droid"): "droidepic_k400_68k_droid",
    ("Joint (P=400)", "epic"): "droidepic_k400_68k_epic",
}
# (table row, env) -> L3 run name used by L3_eval.sh (retrained checkpoint, evaluated like Layer 2)
L3_RUNS = {
    ("per-dataset (P=128)", "droid"): "droid_k128_droid",
    ("per-dataset (P=128)", "epic"): "epic_k128_epic",
    ("Joint (P=128)", "droid"): "droidepic_k128_droid",
    ("Joint (P=128)", "epic"): "droidepic_k128_epic",
    ("per-dataset (P=400)", "droid"): "droid_k400_droid",
    ("per-dataset (P=400)", "epic"): "epic_k400_epic",
    ("Joint (P=400)", "droid"): "droidepic_k400_droid",
    ("Joint (P=400)", "epic"): "droidepic_k400_epic",
}
BASELINE_ENVS = {"droid": "droid", "epic": "epickitchen"}
CHUNK = 128  # samples per batch; per_sample_errors is much slower on larger batches (123 s for 512 samples, 0.16 s for 128)


def l2_root() -> Path:
    """Return release_data/tracegen/L2_outputs (same $WMM_DATA override as Layer 1)."""
    return tg_l1.l1_root().parent / "L2_outputs"


def l3_root() -> Path:
    """Return release_data/tracegen/L3_eval_outputs (same $WMM_DATA override as Layer 1)."""
    return tg_l1.l1_root().parent / "L3_eval_outputs"


def finished_run(prefix: str, root: Path = None):
    """Newest finished output directory ``<prefix>_<timestamp>`` under root (default L2_outputs), or None."""
    done = sorted(p for p in (root or l2_root()).glob(prefix + "_2*") if (p / "metrics.json").is_file())
    return done[-1] if done else None


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, float_format="%.17g")
    tmp.replace(path)


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError("%s is missing: run `python apps/tracegen/L2_cache.py` first" % path)
    return pd.read_csv(path)


def l1_cache_path(row: str, env: str) -> Path:
    return l2_root() / "L1_cache" / (tg_l1.RUNS[(row, env)] + ".csv")


def l2_cache_path(run_dir: Path) -> Path:
    return run_dir / "per_sample_errors.csv"


def l1_per_sample(row: str, env: str) -> pd.DataFrame:
    """Layer-1 per-sample error sums of the original run, read from L2_outputs/L1_cache/<run>.csv."""
    return _read_cache(l1_cache_path(row, env))


def run_per_sample(run_dir: Path) -> pd.DataFrame:
    """Per-sample error sums of one finished L2 run, read from <run_dir>/per_sample_errors.csv."""
    return _read_cache(l2_cache_path(run_dir))


def build_l1_cache(row: str, env: str) -> Path:
    """Write L2_outputs/L1_cache/<run>.csv from the Layer-1 chunks (kept if it exists)."""
    path = l1_cache_path(row, env)
    if not path.is_file():
        _write_csv(tg_l1.run_per_sample(tg_l1.RUNS[(row, env)], env), path)
    return path


def build_l2_cache(run_dir: Path) -> Path:
    """Write <run_dir>/per_sample_errors.csv (all samples; 'kept' = authors' filter) from the run's sample files."""
    path = l2_cache_path(run_dir)
    if path.is_file():
        return path
    with open(run_dir / "rollout_config.json") as f:
        n = int(json.load(f)["num_samples_subset"])
    parts = []
    for c0 in range(0, n, CHUNK):
        idx = range(c0, min(c0 + CHUNK, n))
        pred, gt, mask, vs, mv, gm = [], [], [], [], [], []
        for i in idx:
            with np.load(run_dir / "samples" / ("sample_%06d.npz" % i)) as z:
                pred.append(z["pred_trace_abs"])
                gt.append(z["gt_trace_abs"])
                mask.append(z["trace_valid_mask_per_coord"])
                vs.append(z["valid_steps"])
                mv.append(z["movement_bool"])
                gm.append(z["gt_mask"])
        kept = tg_l1.authors_filter(np.array(vs), np.stack(mv), np.stack(gm))
        err = tg_l1.per_sample_errors(np.stack(pred), np.stack(gt), np.stack(mask))
        parts.append(pd.DataFrame(dict(sample=np.array(idx), kept=kept, **err)))
    _write_csv(pd.concat(parts, ignore_index=True), path)
    return path


def baseline_metrics(env: str):
    """Dataset-level numbers (x100) printed by the authors' code in the newest finished baseline run, or None."""
    runs = sorted(l2_root().glob("tracegen_authors_%s_2*" % BASELINE_ENVS[env]))
    for run in reversed(runs):
        text = (run / "eval.log").read_text(errors="replace") if (run / "eval.log").is_file() else ""
        if not re.search(r"END eval: \w+ rc=0 ", text):
            continue
        vals = {}
        for key, name in (("Trajectory MSE", "MSE"), ("Trajectory MAE", "MAE"), ("Endpoint MSE", "Endp.")):
            found = re.findall(key + r":\s*([0-9.eE+-]+)", text)
            if not found:
                raise RuntimeError("%s: '%s' missing in finished eval.log" % (run, key))
            vals[name] = 100 * float(found[-1])
        vals["run"] = run.name
        return vals
    return None
