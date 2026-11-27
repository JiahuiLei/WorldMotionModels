"""Generate stratified train/val/test splits for oracle_gen_block8 data.

Produces JSONs in the `[[seq_id, length_str], ...]` format expected by
``LangTableOracleDataset`` via ``base_dataset.load_indices``.

Example usage::

    python prepare_data/langtable/official/make_splits.py \
        --root data/languagetable_official/oracle_gen_block8_5k \
        --split_dir data_splits/langtable_official \
        --prefix b8_5k \
        --val_frac 0.05 --test_frac 0.05

Writes per-task JSONs:
    data_splits/langtable_official/b8_5k_block2block_{train,val,test}.json
    ... (one per task subdir)
"""
import argparse
import glob
import json
import os
import os.path as osp
import random
import sys

import numpy as np

PROJECT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.insert(0, osp.join(PROJECT, "ext", "lib_io"))
from lib_io import UnifiedIO  # noqa: E402


def episode_length(state_npz_path):
    """Length of an episode = number of frames.

    For UMM v13 loader, length = len(states[ee_uid]) (which is T frames).
    """
    io = UnifiedIO()
    d = io.load_npzs(state_npz_path)
    states = d["states"].item()
    sem = d["sem_type_uid"].item()
    ee_uid = sem["endeffector"][0]
    T = len(states[ee_uid])
    return T


def build_task_split(task_dir, val_frac, test_frac, seed):
    """Return {split: [[seq_id, length_str], ...]} for a single task dir.

    Per-task splits (written to ``<prefix>_<task>_{train,val,test}.json``) emit
    FLAT seq_ids so that callers can point ``data_root`` directly at the task
    subdir (legacy convention preserved for single-task training).

    The caller is responsible for prefixing seq_ids with ``"<task>/"`` when
    merging across tasks for the unified ``all_*`` split.
    """
    rgbs_zips = sorted(glob.glob(osp.join(task_dir, "images", "*.zip")))
    if not rgbs_zips:
        return {}
    entries = []
    for z in rgbs_zips:
        sid = osp.splitext(osp.basename(z))[0]
        state_npz = osp.join(task_dir, "states", f"{sid}.npz")
        try:
            T = episode_length(state_npz)
        except Exception as e:
            print(f"  WARN: skipping {sid}: {e}", flush=True)
            continue
        entries.append([sid, str(T)])

    rng = random.Random(seed)
    rng.shuffle(entries)
    N = len(entries)
    n_val = max(1, int(round(N * val_frac)))
    n_test = max(1, int(round(N * test_frac)))
    n_train = N - n_val - n_test
    splits = {
        "train": entries[:n_train],
        "val": entries[n_train:n_train + n_val],
        "test": entries[n_train + n_val:],
    }
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root dir containing per-task subdirs")
    parser.add_argument("--split_dir", required=True, help="Output dir for split JSONs")
    parser.add_argument("--prefix", required=True, help="Filename prefix, e.g. 'b8_5k'")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260419)
    args = parser.parse_args()

    os.makedirs(args.split_dir, exist_ok=True)

    task_dirs = sorted([d for d in glob.glob(osp.join(args.root, "*")) if osp.isdir(d) and osp.exists(osp.join(d, "images"))])
    print(f"Task dirs: {[osp.basename(d) for d in task_dirs]}")

    all_splits = {"train": [], "val": [], "test": []}
    summary = []
    for td in task_dirs:
        task = osp.basename(td)
        print(f"\n=== {task} ===")
        splits = build_task_split(td, args.val_frac, args.test_frac, args.seed)
        if not splits:
            print("  no episodes, skip")
            continue
        for split, entries in splits.items():
            # Per-task split file — FLAT seq_ids, meant for data_root=<task_dir>.
            out_path = osp.join(args.split_dir, f"{args.prefix}_{task}_{split}.json")
            with open(out_path, "w") as f:
                json.dump(entries, f, indent=2)
            print(f"  {split}: {len(entries)} -> {out_path}")
            # When merging for the unified all_* split, prefix each entry with
            # the task dir so episodes stay addressable relative to the
            # shared data_root parent (seq_ids collide across tasks — 4455
            # collisions observed in oracle_gen_block8_181k).
            all_splits[split].extend([[f"{task}/{sid}", T] for sid, T in entries])
        summary.append({"task": task, **{k: len(v) for k, v in splits.items()}})

    # Unified "all" split — concat across tasks
    for split in ("train", "val", "test"):
        out_path = osp.join(args.split_dir, f"{args.prefix}_all_{split}.json")
        with open(out_path, "w") as f:
            json.dump(all_splits[split], f, indent=2)
        print(f"\n[all] {split}: {len(all_splits[split])} -> {out_path}")

    with open(osp.join(args.split_dir, f"{args.prefix}_summary.json"), "w") as f:
        json.dump({"per_task": summary, "total": {k: len(v) for k, v in all_splits.items()}, "val_frac": args.val_frac, "test_frac": args.test_frac, "seed": args.seed}, f, indent=2)


if __name__ == "__main__":
    main()
