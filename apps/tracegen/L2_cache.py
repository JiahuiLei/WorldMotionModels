"""Precompute the small per-sample files that L2_compare.ipynb and L3_compare.ipynb read.

    python apps/tracegen/L2_cache.py

For every finished Layer-2 or Layer-3 run (``L2_outputs/<run>_<time>/metrics.json`` or
``L3_eval_outputs/<run>_<env>_<time>/metrics.json`` exists) it reads the run's sample files once and writes
``per_sample_errors.csv`` (one row per test sample, ~0.6-0.9 MB) next to them; for every table cell it also writes the
Layer-1 per-sample sums to ``L2_outputs/L1_cache/<run>.csv``. Files that already exist are kept.
Each run takes about a minute (it reads 25-34 GB); runs are processed in parallel. On a machine under memory pressure
every fresh allocation can be very slow (up to an hour per run); letting Python reuse its memory fixes that:
`MALLOC_MMAP_THRESHOLD_=4294967296 MALLOC_TRIM_THRESHOLD_=68719476736 python apps/tracegen/L2_cache.py` (47 s for two runs).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import tg_l2


def _l1(key):
    return "L1 %s" % tg_l2.build_l1_cache(*key)


def _run(layer, name):
    run = tg_l2.finished_run(name, tg_l2.l3_root() if layer == "L3" else tg_l2.l2_root())
    if run is None:
        return "%s %s: not finished" % (layer, name)
    return "%s %s" % (layer, tg_l2.build_l2_cache(run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4, help="parallel processes (each needs about 3 GB of memory)")
    args = parser.parse_args()
    with ProcessPoolExecutor(args.jobs) as ex:
        futures = [ex.submit(_l1, key) for key in tg_l2.L2_RUNS]
        futures += [ex.submit(_run, "L2", n) for n in tg_l2.L2_RUNS.values()]
        futures += [ex.submit(_run, "L3", n) for n in tg_l2.L3_RUNS.values()]
        for f in futures:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
