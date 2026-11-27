# 2019.May Jiahui old codebase from GenRe
# 2020.Sep Jiahui code basefrom occ-flow
# 2023.Jun Jiahui New codebase from PL
# 2024.Aug Jiahui updated once in deepmind
# 2026.Feb Jiahui updated once with agents

import os
import sys
import multiprocessing as _mp


def _cfg_has_ms3_eval() -> bool:
    """cheap pre-scan of the ``--config`` YAML for an ``ms3_eval``
    top-level key. Runs at module-import time, BEFORE any torch / umm /
    ManiSkill import, so the spawn switch can be opt-in per-config rather
    than flipping the repo-wide default. Returns False on any error
    (unreadable file, no --config, YAML parse failure) so non-ms3
    configs keep the legacy fork behaviour untouched.
    """
    import yaml
    cfg_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            cfg_path = sys.argv[i + 1]
            break
        if a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
            break
    if cfg_path is None or not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            y = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(y, dict) and y.get("ms3_eval") is not None


# opt-in spawn switch for ms3_eval configs ONLY. Fork remains the
# default for every other training config — their Datasets carry ctypes
# (e.g. TurboJPEG() inside ext/lib_io/lib_io/unified_io.py via
# ``self.io = UnifiedIO()`` on BaseDataset and its subclasses) which are
# not picklable under spawn.
#
# Why spawn is needed for ms3_eval:
# ``MS3EvalCallback`` constructs ``AsyncVectorEnv(context="forkserver",
# num_envs=10)`` mid-training (see eval/ms3/evaluator.py:830-838). When
# the enclosing Python process started with ``fork`` AND has already
# initialized CUDA in the main thread, the forkserver subprocess pool
# inherits dirty CUDA handles and hangs around the 9th env. Setting the
# global start method to ``spawn`` before torch import gives forkserver
# a clean bootstrap. Empirically verified by with 2 × 10-env ×
# 100-episode smokes (PushCube-v1 + PegInsertionSide-v1) running to
# completion without hang.
_SG07_FORCE_SPAWN = _cfg_has_ms3_eval()
if _SG07_FORCE_SPAWN:
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Guard against the rare case where a parent process already
        # consumed a context before us; force=True normally precludes
        # this, so log and fall through to the assert below.
        pass
    # Fail loud if the platform refused spawn (e.g., non-POSIX or malformed
    # method name) — running under fork with MS3EvalCallback forkserver
    # hangs silently after minutes, so fail now.
    assert _mp.get_start_method(allow_none=False) in ("spawn", "forkserver"), (
        f"Requested 'spawn' but multiprocessing start method is "
        f"{_mp.get_start_method()!r}. MS3EvalCallback forkserver "
        "AsyncVectorEnv will hang under 'fork'."
    )

from umm.training_utils import get_config, init_setup, get_logger, get_trainer
from umm.datasets import get_data_module
from umm.models import get_model
import logging


def _coerce_workers_for_spawn(cfg) -> None:
    """if the ms3_eval spawn switch fired above, DataLoader workers
    must use a context that pickles the dataset. Every UMM dataset
    subclass stores ``self.io = UnifiedIO()`` whose ``TurboJPEG()`` C
    handle is non-picklable, so workers>0 pickling fails. Coerce to 0
    with a visible warning instead of letting the launch die with an
    opaque ``ValueError: ctypes objects containing pointers cannot be
    pickled`` traceback inside the first dataloader iter.
    """
    if not _SG07_FORCE_SPAWN:
        return
    w = int(cfg.get("workers", 0))
    if w > 0:
        logging.warning(
            "Spawn is in force for ms3_eval cfg; coercing "
            "cfg.workers %d -> 0 because UnifiedIO.TurboJPEG ctypes "
            "cannot be pickled across spawn. Pass --workers 0 "
            "explicitly to silence this warning, or refactor the "
            "dataset to lazily create io in worker_init_fn.",
            w,
        )
        cfg.workers = 0


# Multiprocessing protection to prevent subprocess re-execution
if __name__ == "__main__":
    # * Get cfg
    cfg = get_config()
    _coerce_workers_for_spawn(cfg)
    logger = get_logger(cfg)
    init_setup(cfg)
    # * Get dataset
    dm = get_data_module(cfg)
    # * Get PL Model
    pl_model = get_model(cfg)
    # * Config PL Trainer
    trainer = get_trainer(cfg, logger, use_distributed_sampler=True)
    # * Run
    trainer.fit(model=pl_model, datamodule=dm, ckpt_path=cfg.__init_ckpt_path__)
    logging.info("finished!")
