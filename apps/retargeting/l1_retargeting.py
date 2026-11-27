"""Helpers for apps/retargeting/L1_tables.ipynb: locate data, load per-sequence results, recompute the table."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_FIRST = 2000

# (method, setting) -> stem of the scorer output <stem>_mjc_rollout_report_2000.xlsx
RUNS = {
    ("GMR", "clean"): "amass_gmr_inferred",
    ("PHUMA", "clean"): "amass_phuma3001",
    ("WMM", "clean"): "w1.0.1_v4_38k",
    ("GMR", "infill"): "amass_gmr_inferred_3fps",
    ("PHUMA", "infill"): "amass_phuma3001_3fps",
    ("WMM", "infill"): "w1.0.1_v4_38k_3fps",
    ("GMR", "noisy"): "amass_gmr_inferred_q0.1_t0.1",
    ("PHUMA", "noisy"): "amass_phuma3001_q0.1_t0.1",
    ("WMM", "noisy"): "w1.0.1_v4_38k_q0.1_t0.1_ct10",
}

SHA256 = {
    "reports/amass_gmr_inferred_mjc_rollout_report_2000.xlsx": "ca561673d9da2800b857bec242a8d604fa704fd081123f891189344c342bce13",
    "reports/amass_phuma3001_mjc_rollout_report_2000.xlsx": "2ba9684b5c8ecefd27df01a841b6279fb8bfaefe756b7f027ffcae417a6d5f44",
    "reports/w1.0.1_v4_38k_mjc_rollout_report_2000.xlsx": "75b41481ec993ca170c67806ee26f937efb5c5b8c3719290b9e587b480cfbf14",
    "reports/amass_gmr_inferred_3fps_mjc_rollout_report_2000.xlsx": "1cd2137f181d61f480fdf2042deed3f77466da3e5c7e4fb29bf78a46f059db91",
    "reports/amass_phuma3001_3fps_mjc_rollout_report_2000.xlsx": "2537b13581c3c0b87b3faca8f8297760da392c8569c8cef86ba0cf84d6e92b37",
    "reports/w1.0.1_v4_38k_3fps_mjc_rollout_report_2000.xlsx": "e56b145af2f4ca06a8e395e81ab036f8636337cf62f28eaacbbd7a35a78102c9",
    "reports/amass_gmr_inferred_q0.1_t0.1_mjc_rollout_report_2000.xlsx": "202765ee2d93ba09fcef3c7e9a143d69df80942e4b2bff325eef75f817ed62b9",
    "reports/amass_phuma3001_q0.1_t0.1_mjc_rollout_report_2000.xlsx": "0016c2f00a14fc5d52be3cb73e818b813d2082704575c89a016c1e83848969ec",
    "reports/w1.0.1_v4_38k_q0.1_t0.1_ct10_mjc_rollout_report_2000.xlsx": "c299557bff4c6c3d8212dae31f43c713b9657489504310f8c5a81bf7032584dc",
    "splits/amass_humanoid_v2_full_test.json": "8f2820bec822303fa183be615a7d1d1cdfec79363c303aa3023be404873f85b3",
}

# Camera-ready tables/exp_retgt.tex, transcribed verbatim (strings keep the printed precision).
PAPER = {
    ("GMR", "clean"): {"FPS": "35.03", "SR-seq": "0.7945", "gerr": "0.6911", "lerr": "0.0651"},
    ("PHUMA", "clean"): {"FPS": "2.27", "SR-seq": "0.8255", "gerr": "0.5509", "lerr": "0.0530"},
    ("WMM", "clean"): {"FPS": "86.97", "SR-seq": "0.8690", "gerr": "0.5938", "lerr": "0.0420"},
    ("GMR", "infill"): {"SR-seq": "0.8095", "gerr": "0.6542", "lerr": "0.0592"},
    ("PHUMA", "infill"): {"SR-seq": "0.6535", "gerr": "1.3744", "lerr": "0.0745"},
    ("WMM", "infill"): {"SR-seq": "0.8525", "gerr": "0.6371", "lerr": "0.0518"},
    ("GMR", "noisy"): {"SR-seq": "0.7160", "gerr": "1.0196", "lerr": "0.1158"},
    ("PHUMA", "noisy"): {"SR-seq": "0.8060", "gerr": "0.7269", "lerr": "0.0703"},
    ("WMM", "noisy"): {"SR-seq": "0.8530", "gerr": "0.7785", "lerr": "0.0691"},
}

METHODS = ["GMR", "PHUMA", "WMM"]
SETTINGS = ["clean", "infill", "noisy"]
SETTING_TITLES = {"clean": "Clean (30 FPS)", "infill": "In-fill (3 FPS keyframes)", "noisy": "Noisy (sigma=0.1)"}
METRICS = ["FPS", "SR-seq", "gerr", "lerr"]


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "apps" / "retargeting").is_dir():
            return p
    raise FileNotFoundError(f"no directory containing apps/retargeting/ above {start}")


def app_data_dir(start: Path) -> Path:
    data_root = Path(os.environ["WMM_DATA"]) if "WMM_DATA" in os.environ else find_repo_root(start) / "release_data"
    return data_root / "retargeting"


def l1_dir(start: Path) -> Path:
    path = app_data_dir(start) / "L1_results"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found; download the L1 results (apps/retargeting/README.md) or set WMM_DATA")
    return path


def l2_reports_dir(start: Path) -> Path:
    path = app_data_dir(start) / "L2_outputs" / "reports"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found; run apps/retargeting/L2_eval.sh first (README, Layer 2)")
    return path


def verify_checksums(l1: Path) -> None:
    for rel, expected in SHA256.items():
        digest = hashlib.sha256((l1 / rel).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"sha256 mismatch for {rel}: {digest} != {expected}")


def test_sids(l1: Path) -> list:
    split = json.loads((l1 / "splits" / "amass_humanoid_v2_full_test.json").read_text())
    return [entry[0] for entry in split[:N_FIRST]]


def load_per_sequence(reports: Path, stem: str, sids: list) -> pd.DataFrame:
    """One row per test sequence; the scorer's own mean/std rows are dropped."""
    df = pd.read_excel(reports / f"{stem}_mjc_rollout_report_2000.xlsx", index_col=0)
    per_seq = df.drop(index=["mean", "std"])
    if list(per_seq.index) != list(range(N_FIRST)):
        raise ValueError(f"{stem}: expected rows 0..{N_FIRST - 1}")
    per_seq.index = pd.Index(sids, name="sid")
    return per_seq.astype(float)


def recompute(per_seq: pd.DataFrame) -> dict:
    """Dataset-level numbers from per-sequence rows (frame counts and logged wall-clock times)."""
    sr_seq = (per_seq["SR-succ_n"] == per_seq["SR-total_n"]).astype(float)
    if not np.array_equal(sr_seq.values, per_seq["SR-seq"].values):
        raise ValueError("SR-seq column disagrees with standing-frame counts")
    fps_seq = per_seq["retgt_nframes"] / per_seq["retgt_runtime"]
    if not np.allclose(fps_seq.values, per_seq["retgt_fps"].values, rtol=1e-12, atol=0):
        raise ValueError("retgt_fps column disagrees with nframes / runtime")
    if per_seq[["gerr", "lerr"]].isna().any().any():
        raise ValueError("NaN gerr/lerr (a sequence with zero standing frames)")
    return {
        "FPS": float(fps_seq.mean()),
        "SR-seq": float(sr_seq.mean()),
        "gerr": float(per_seq["gerr"].mean()),
        "lerr": float(per_seq["lerr"].mean()),
    }


def compare_with_paper(results: dict) -> pd.DataFrame:
    rows = []
    for key, paper_cells in PAPER.items():
        for metric, paper_str in paper_cells.items():
            decimals = len(paper_str.split(".")[1])
            value = results[key][metric]
            ours_str = f"{value:.{decimals}f}"
            rows.append(
                {
                    "method": key[0],
                    "setting": key[1],
                    "metric": metric,
                    "paper": paper_str,
                    "recomputed": ours_str,
                    "full_precision": value,
                    "match": ours_str == paper_str,
                }
            )
    return pd.DataFrame(rows)


def paper_layout(results: dict) -> pd.DataFrame:
    columns = [("Clean (30 FPS)", m) for m in METRICS]
    columns += [(SETTING_TITLES[s], m) for s in ["infill", "noisy"] for m in METRICS[1:]]
    table = pd.DataFrame(index=METHODS, columns=pd.MultiIndex.from_tuples(columns), dtype=object)
    for method in METHODS:
        for setting in SETTINGS:
            for metric in METRICS:
                if setting != "clean" and metric == "FPS":
                    continue
                decimals = 2 if metric == "FPS" else 4
                table.loc[method, (SETTING_TITLES[setting], metric)] = f"{results[(method, setting)][metric]:.{decimals}f}"
    return table
