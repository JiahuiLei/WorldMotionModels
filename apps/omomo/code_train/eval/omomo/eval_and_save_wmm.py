"""Standalone OMOMO eval-and-save: K=20 best-of-MPJPE inference + scorer.

Mirrors :mod:`eval.langtable_official.eval_and_save_wmm` for the OMOMO
domain. For a single ckpt + config + output dir:

1. Backup ckpt + ``src/`` + ``configs/`` + ``eval/`` + ``train.py`` +
   git state into ``<dst>/`` (self-contained snapshot).
2. Run K=20 diffusion inference on each test seq, with explicit per-k
   seed (k=0..K-1).
3. Best-of-K selection: per-seq tensor MPJPE on body-joint world
   positions (root-zeroed, entities 0..21). Winner saved to
   ``<dst>/preds_all/<stem>.npz`` (CHOIS schema, scorer-compatible).
4. Full record: all K samples saved to
   ``<dst>/preds_all_K20/<stem>_k<k>.npz``.
5. Per-seq metadata appended to ``<dst>/preds_meta.jsonl``.
6. Subprocess-invoke ``eval/omomo/eval_chois_table1.py`` to score.
7. Print 16-col CHOIS Table 1 (paper vs ours vs Δ vs verdict).

CLI::

    python eval/omomo/eval_and_save_wmm.py \\
        --checkpoint <abs-path-to.ckpt> \\
        --config configs/wmm/om/w1.5.0.yaml \\
        --output ./logs/eval/<label> \\
        --mode {debug,full} \\
        [--n_steps N] [--batch_size B]

Modes:

* ``debug``: 8 seqs (first 8 by stem-sort), K=20, n_steps=10,
  scheduler_mode="ddim_gen" — exercises the full best-of-K pipeline
  without paying full DDPM cost.
* ``full``: ALL test seqs (sorted glob), K=20, scheduler_mode="ddpm_full".
  By default ``n_steps=None`` so the scheduler uses all
  ``num_train_timesteps`` steps. ``--n_steps N`` is honored if explicitly
  passed.

GPU: pin externally via ``CUDA_VISIBLE_DEVICES=N``; the script always
uses ``cuda:0`` internally. K is fixed at 20 in BOTH modes per CHOIS
protocol.

Refuses to run if ``<dst>_<timestamp>`` exists and is non-empty.
"""
import argparse
import json
import logging
import os
import os.path as osp
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
# Repo root for `prepare_data.*` (top-level package, not under src/) and
# for `eval.omomo.eval_chois_table1` import (used by _print_table1).
sys.path.insert(0, PROJECT)
sys.path.insert(0, osp.join(PROJECT, "src"))

# Reused helpers from the live training callback. We import — never copy —
# so a single source of truth defines the GT npz schema, the batch builder,
# and the CHOIS-schema pred npz writer. Constants pinned here mirror what
# the callback uses internally.
from umm.callbacks.wmm_omomo_eval import (  # noqa: E402
    _build_batch_from_gt_files,
    _save_pred_npz,
    _NUM_BODY_JOINTS,
    _T_FIXED,
    _K_PRODUCTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# K=20 CHOIS protocol. Hardcoded in BOTH modes (debug intentionally exercises
# the full best-of-K plumbing — without K=20, debug would not test selection,
# the K20 archive dir, or per-k seeds).
_K = _K_PRODUCTION  # 20 — sourced from the callback to stay in sync


# ---------------------------------------------------------------------------
# Loading / setup
# ---------------------------------------------------------------------------


def load_pl_module(
    checkpoint_path: str, config_path: str, device: torch.device
) -> Tuple[Any, Any]:
    """Load PLModel from ``checkpoint_path`` with ``config_path``.

    Returns ``(pl_module, cfg)``. Mirrors ``eval/langtable_official/
    eval_and_save_wmm.py::load_wmm_agent`` but without the LangTable
    visualizer (OMOMO eval does not use a sim env).

    Args:
        checkpoint_path: Path to the .ckpt.
        config_path: Path to the YAML config used at training time.
        device: Target torch device.

    Returns:
        (pl_module, cfg) where pl_module is in eval mode on ``device``.
    """
    from umm.training_utils import get_config
    from umm.models.wmm.model import PLModel

    argv_bak = sys.argv[:]
    sys.argv = ["eval", "--config", config_path]
    try:
        cfg = get_config()
    finally:
        sys.argv = argv_bak

    pl_module = PLModel.load_from_checkpoint(
        checkpoint_path, cfg=cfg, map_location=device, strict=False
    )
    pl_module.to(device).eval()
    return pl_module, cfg


def get_git_sha(project_root: str) -> str:
    """Return short HEAD sha or ``"unknown"`` on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception as exc:
        log.warning("get_git_sha: git rev-parse failed: %r", exc)
        return "unknown"


def assert_dst_empty(dst: str) -> None:
    """Refuse to run if ``dst`` exists and is non-empty (exit 2)."""
    if osp.isdir(dst) and len(os.listdir(dst)) > 0:
        log.error(
            "Output directory %s already exists and is non-empty. "
            "Refusing to overwrite. Pick a fresh path or remove it manually.",
            dst,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Source + git snapshot
# ---------------------------------------------------------------------------


def snapshot_src(dst: str, project_root: str) -> float:
    """Mirror training's ``src_bck`` convention into ``<dst>/src_bck/``.

    Copies ``src/``, ``configs/``, ``eval/`` (FULL — every domain), and
    ``train.py``. Excludes large data artifacts and Python bytecode.
    Returns wall seconds spent in the snapshot.
    """
    t0 = time.time()
    src_bck = osp.join(dst, "src_bck")
    os.makedirs(src_bck, exist_ok=True)
    excludes = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        # serialized data
        "--exclude=*.npz", "--exclude=*.npy", "--exclude=*.pkl",
        "--exclude=*.p",
        "--exclude=*.pt", "--exclude=*.ckpt", "--exclude=*.bin",
        "--exclude=*.zip", "--exclude=*.tar", "--exclude=*.tar.gz",
        "--exclude=*.gz", "--exclude=*.h5", "--exclude=*.hdf5",
        # docs / media
        "--exclude=*.pdf", "--exclude=*.mp4",
        "--exclude=*.png", "--exclude=*.jpg", "--exclude=*.jpeg",
        # 3D mesh / sim assets
        "--exclude=*.obj", "--exclude=*.ply", "--exclude=*.stl",
        "--exclude=*.dae", "--exclude=*.glb", "--exclude=*.gltf",
        # any nested data/ dir under code trees (e.g. eval/omomo/data/)
        "--exclude=data/",
    ]
    for sub in ("src", "configs", "eval"):
        src_path = osp.join(project_root, sub)
        if not osp.isdir(src_path):
            log.warning("snapshot_src: %s not found, skipping", src_path)
            continue
        dst_path = osp.join(src_bck, sub) + "/"
        subprocess.run(
            ["rsync", "-a"] + excludes + [src_path + "/", dst_path],
            check=True,
        )
    train_py = osp.join(project_root, "train.py")
    if osp.isfile(train_py):
        shutil.copyfile(train_py, osp.join(src_bck, "train.py"))
    return time.time() - t0


def _git_cmd_prefix() -> List[str]:
    """Return the git command used for the provenance record."""
    return ["git"]


def write_git_info(dst: str, project_root: str) -> None:
    """Write ``<dst>/git_info.txt`` with branch/HEAD/status/diff/log -1.

    Captures every signal needed to identify the exact code state behind
    the eval — even if the branch is later force-pushed or the local
    workspace is wiped. Read-only git operations only.
    """
    info_path = osp.join(dst, "git_info.txt")
    git = _git_cmd_prefix()
    queries = [
        ("branch", git + ["rev-parse", "--abbrev-ref", "HEAD"]),
        ("HEAD", git + ["rev-parse", "HEAD"]),
        ("HEAD short", git + ["rev-parse", "--short", "HEAD"]),
        ("upstream", git + ["rev-parse", "--abbrev-ref", "@{u}"]),
        ("upstream HEAD", git + ["rev-parse", "@{u}"]),
        ("log -1",
         git + ["log", "-1",
                "--pretty=format:%h %an %ad %s", "--date=iso"]),
        ("log -5 (compact)",
         git + ["log", "-5", "--oneline"]),
        ("status (porcelain)", git + ["status", "--porcelain"]),
        ("diff --stat (unstaged)", git + ["diff", "--stat"]),
        ("diff --stat (staged)", git + ["diff", "--cached", "--stat"]),
        ("remote -v", git + ["remote", "-v"]),
    ]
    parts = [
        f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"project_root: {project_root}",
        f"hostname: {os.uname().nodename}",
    ]
    for label, cmd in queries:
        try:
            out = subprocess.check_output(
                cmd, cwd=project_root, stderr=subprocess.DEVNULL
            ).decode().rstrip()
        except Exception as exc:
            out = f"<error: {type(exc).__name__}: {exc}>"
        parts.append(f"\n--- {label} ---\n{out}")
    with open(info_path, "w") as fh:
        fh.write("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Provenance + script backup
# ---------------------------------------------------------------------------


def write_rollout_config(
    dst: str,
    args: argparse.Namespace,
    K: int,
    n_steps_resolved: Any,
    batch_size: int,
    scheduler_mode: str,
    num_seqs: int,
    device: torch.device,
    project_root: str,
    timestamp: str,
    dst_ckpt_path: Optional[str] = None,
) -> None:
    """Write ``<dst>/rollout_config.json`` with full run provenance.

    Args:
        dst_ckpt_path: Absolute path to the ckpt copy on disk.
            If ``None``, derived from ``src_ckpt`` basename (legacy
            single-GPU naming). Pass an explicit path when the ckpt was
            normalized to a different name (e.g. ``last.ckpt`` in
            ``--setup_only`` mode) so the provenance record matches
            what is actually on disk.
    """
    src_ckpt = osp.abspath(args.checkpoint)
    if dst_ckpt_path is None:
        dst_ckpt_path = osp.join(dst, osp.basename(src_ckpt))
    rollout_cfg = {
        "checkpoint": args.checkpoint,
        "checkpoint_src": src_ckpt,
        "checkpoint_loaded_from": dst_ckpt_path,
        "config": args.config,
        "output": args.output,
        "output_resolved": dst,
        "mode": args.mode,
        "K": int(K),
        # n_steps_resolved may be ``None`` (let scheduler decide for ddpm_full)
        # or an int. Persist verbatim — JSON null preserves the distinction.
        "n_steps_resolved": n_steps_resolved,
        "batch_size": int(batch_size),
        "scheduler_mode": scheduler_mode,
        "num_seqs": int(num_seqs),
        "device": str(device),
        "dtype": "bfloat16-autocast",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "git_sha": get_git_sha(project_root),
        "hostname": socket.gethostname(),
        "timestamp": timestamp,
        # reference lock: full mode follows fixed CHOIS protocol (K=20). Debug
        # mode shares K=20 but uses a fast scheduler; not a "production" run.
        "protocol_locked": (args.mode == "full"),
    }
    with open(osp.join(dst, "rollout_config.json"), "w") as fh:
        json.dump(rollout_cfg, fh, indent=2)


# ---------------------------------------------------------------------------
# Inference + best-of-K
# ---------------------------------------------------------------------------


def _get_eval_agent(pl_module: Any) -> Any:
    """Return the EMA agent if available, else ``pl_module.model``.

    Mirrors the wmm callback's choice: training quality reports use the
    EMA copy when ``use_ema`` is set, which is the same artifact the
    on-the-fly callback runs CHOIS Table 1 against.

    When falling back to the non-EMA path, log a LOUD warning: the
    non-EMA `model` is the COMPILED module (PLModel sets up
    ``torch.compile`` on the LMM submodule). First-call recompile on
    eval shapes can take 30s+ and the resulting eval is non-comparable
    with the on-the-fly callback's EMA-based metrics.
    """
    if getattr(pl_module, "use_ema", False) and getattr(pl_module, "ema_model", None) is not None:
        return pl_module.ema_model
    # Fallback: the wmm callback also accepts ``.agent`` if defined; keep
    # parity with that resolution path so a custom PLModel works too.
    fallback = getattr(pl_module, "model", None) or getattr(pl_module, "agent", None)
    if fallback is None:
        return None
    log.warning(
        "Falling back to non-EMA agent (use_ema=False or ema_model missing). "
        "This is the COMPILED module — first-call recompile may take 30s+, "
        "and metrics will NOT be comparable with the on-the-fly callback's "
        "EMA-based CHOIS Table 1.",
    )
    return fallback


def _run_inference_once(
    agent: Any,
    batch: Dict[str, Any],
    seed: int,
    n_steps: Optional[int],
    scheduler_mode: str,
    chois_mode: str = "replace",
    chois_guide_scale: float = 1.0,
    smooth_lambda: float = 0.05,
) -> torch.Tensor:
    """Single ``agent.inference()`` pass; returns ``ret_trace`` (B, T, P, 4, 4).

    Inlined from ``WmmOmomoEvalCallback._run_inference_batch`` to avoid
    coupling to a private method, but the call signature and arg names are
    identical so the behavior tracks the callback exactly.

    bf16 autocast is the default eval dtype (matches tracegen eval); recorded
    in rollout_config.json as ``"dtype": "bfloat16-autocast"``.

    CHOIS-faithful default: M1 (``chois_mode="replace"``) + M3
    (``smooth_lambda=0.05``). Caller is expected to have built ``batch``
    with ``partial_chois="strict"`` so ``chois_pos_target`` /
    ``chois_pos_mask`` are present; pass-through is None otherwise.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = agent.inference(
            trace=batch["trace"],
            context=batch["context"],
            trace_known_mask=batch["known_mask"],
            context_known_mask=batch["context_known_mask"],
            trace_mask=batch["trace_mask"],
            context_mask=batch["context_mask"],
            real_timestep=batch["time"],
            group_id=batch["trace_group_id"],
            sem_type=batch["trace_sem_type"],
            rot_mask=batch["trace_rot_mask"],
            text=batch["text"],
            local_pcl=batch["local_pcl"],
            local_pcl_mask=batch["local_pcl_mask"],
            seed=seed,
            n_steps=n_steps,
            verbose=False,
            scheduler_mode=scheduler_mode,
            # CHOIS sampling-time hooks (None when batch built with off).
            chois_pos_target=batch.get("chois_pos_target"),
            chois_pos_mask=batch.get("chois_pos_mask"),
            chois_mode=chois_mode,
            chois_guide_scale=chois_guide_scale,
            smooth_lambda=smooth_lambda,
        )
    # result is (ret_context, ret_trace); ret_trace shape: (B, T, P=23, 4, 4)
    _, ret_trace = result
    return ret_trace


def _best_of_k_per_seq(
    ret_trace_per_k_b: List[torch.Tensor],   # K tensors, each (T, P=23, 4, 4)
    gt_Q: np.ndarray,                         # (T, 22, 4, 4) GT body Q
    actual_len: int,
) -> Tuple[int, List[float]]:
    """Pick the k whose body-joint MPJPE (root-zeroed) vs GT is minimal.

    Selection metric — root-zeroed L2 mean over (T, J) on body-joint
    world positions extracted from ``Q[:, :22, :3, 3]``. This is the
    same selection metric used for the CHOIS baseline (best of K by
    root-zeroed joint error) with two
    documented differences:

    1. **Joint count: 22 (ours) vs 24 (CHOIS).** Our model emits 22
       body-Q joints (pelvis + 21 SMPL-X body), the CHOIS adapter uses
       24 (pelvis + 21 body + 2 hand-adjacent joints). Selection
       compares samples against THE SAME GT, so missing joints subtract
       a small constant from every k uniformly and cannot change the
       argmin ordering on smooth body trajectories.
    2. **Frame: world (ours) vs canonicalized SMPL-X (CHOIS).** We
       root-zero by subtracting the per-frame pelvis position, so any
       global rigid translation is cancelled. Rotation invariance is
       not strictly preserved (CHOIS canonicalization also rotates),
       but selection arg-mins are dominated by per-joint reconstruction
       error, not the global frame choice. Confirmed acceptable in the
       lei-align spec (Q6) — the selected k is byte-identically saved
       in CHOIS schema downstream and the scorer (``run_eval``) computes
       the official MPJPE in canonicalized SMPL-X space, which is what
       the paper table reports.

    Only frames ``[:actual_len]`` are scored — the dataset converter
    identity-pads ``t >= actual_len`` in the GT Q field, and the model
    likewise has no signal there.

    Args:
        ret_trace_per_k_b: K per-sample tensors of shape (T, P, 4, 4) on CPU.
        gt_Q: GT body Q in world frame, shape (T, 22, 4, 4).
        actual_len: True sequence length (un-padded frame count).

    Returns:
        (best_k, all_mpjpe[K]) where ``all_mpjpe[k]`` is the float MPJPE
        of sample k (in metres, mean over frames * joints).
    """
    # Defensive shape assertions (cheap, defense-in-depth — upstream
    # ``_build_batch_from_gt_files`` already raises ValueError if the
    # GT npz lacks the expected (T, 22, 4, 4) Q field, but a corrupt
    # in-memory dict could still slip through).
    if gt_Q.ndim != 4 or gt_Q.shape[1] != _NUM_BODY_JOINTS or gt_Q.shape[2:] != (4, 4):
        raise ValueError(
            f"_best_of_k_per_seq: gt_Q shape {gt_Q.shape} is not "
            f"(T, {_NUM_BODY_JOINTS}, 4, 4) — corrupt GT?"
        )
    if actual_len <= 0 or actual_len > gt_Q.shape[0]:
        raise ValueError(
            f"_best_of_k_per_seq: actual_len={actual_len} out of range "
            f"for gt_Q.shape[0]={gt_Q.shape[0]}"
        )
    for k_idx, t in enumerate(ret_trace_per_k_b):
        if t.ndim != 4 or t.shape[1] < _NUM_BODY_JOINTS or t.shape[2:] != (4, 4):
            raise ValueError(
                f"_best_of_k_per_seq: ret_trace[{k_idx}] shape {t.shape} "
                f"is not (T, P>={_NUM_BODY_JOINTS}, 4, 4)"
            )
        if t.shape[0] < actual_len:
            raise ValueError(
                f"_best_of_k_per_seq: ret_trace[{k_idx}].shape[0]={t.shape[0]} "
                f"< actual_len={actual_len}"
            )

    # Stack K samples and slice to body joints + xyz columns of the
    # translation block. body_pos: (K, T_eff, 22, 3) in world frame
    # (only frames [:actual_len] are scored — padded frames at
    # t >= actual_len are identity-padded by the converter and would
    # contaminate the MPJPE if included).
    body_pos_per_k = torch.stack(
        [t[:actual_len, :_NUM_BODY_JOINTS, :3, 3] for t in ret_trace_per_k_b],
        dim=0,
    )                                                     # (K, T_eff, 22, 3)
    pelvis_per_k = body_pos_per_k[:, :, 0:1, :]           # (K, T_eff, 1, 3)
    pred_root0 = body_pos_per_k - pelvis_per_k            # (K, T_eff, 22, 3)

    gt_body_pos = torch.as_tensor(
        gt_Q[:actual_len, :_NUM_BODY_JOINTS, :3, 3], dtype=torch.float32,
    )                                                     # (T_eff, 22, 3)
    gt_root0 = gt_body_pos - gt_body_pos[:, 0:1, :]       # (T_eff, 22, 3)

    per_joint_err = (pred_root0 - gt_root0[None]).norm(dim=-1)  # (K, T_eff, 22)
    mpjpe_per_k = per_joint_err.mean(dim=(1, 2))                # (K,)
    best_k = int(torch.argmin(mpjpe_per_k).item())
    all_mpjpe = [float(x) for x in mpjpe_per_k.tolist()]
    return best_k, all_mpjpe


# ---------------------------------------------------------------------------
# Scorer subprocess + final report
# ---------------------------------------------------------------------------


def _ensure_scorer_data_symlinks(
    project_root: str,
    gt_dir: str,
    smplx_dir: str,
    sdf_folder: str,
    obj_geo_dir: str,
) -> None:
    """Create idempotent symlinks at ``<project_root>/data/omomo/{...}``.

    The scorer ``eval_chois_table1.py`` expects its data under
    ``_REPO_DATA_ROOT = <project_root>/data/omomo`` (a fixed convention).
    In a fresh git worktree, ``data/`` is gitignored and absent, so the
    scorer falls back to stale per-host worktree paths that do not exist.

    We pre-create symlinks pointing at the cfg-resolved data dirs (Stage A)
    PLUS the in-repo vendored Stage B deps (LFS-fetched at
    ``<repo>/eval/omomo/data/`` and a couple under
    ``<repo>/data_wmm/omomo_v2/artifacts/``). Any subdir target that does
    not exist on disk is silently skipped — the scorer will then fall back
    to its own defaults for that one dep. All symlinks are local to the
    worktree (``data/`` is gitignored) and idempotent.
    """
    repo_data = osp.join(project_root, "data", "omomo")
    os.makedirs(repo_data, exist_ok=True)
    # Stage B deps live under eval/omomo/data/ (vendored LFS, scorer-private).
    # glove_vocab is under data_wmm/omomo_v2/artifacts/.
    eval_data = osp.join(project_root, "eval", "omomo", "data")
    artifacts = osp.join(project_root, "data_wmm", "omomo_v2", "artifacts")
    # subdir name in <repo>/data/omomo  -> cfg-resolved abs target
    targets = {
        # Stage A (cfg-driven)
        "test_npz":            gt_dir,
        "smplx":               smplx_dir,
        "rest_object_sdf":     sdf_folder,
        "rest_object_geo":     obj_geo_dir,
        # Stage B (vendored LFS, fixed locations under the repo)
        "chois_test_cache":    osp.join(eval_data, "chois_test_cache"),
        "stats":               osp.join(eval_data, "stats"),
        "omomo_text_anno":     osp.join(eval_data, "omomo_text_anno"),
        "smplh_amass":         osp.join(eval_data, "smplh_amass"),
        # glove_vocab nested under data/omomo/artifacts/
        "artifacts/glove_vocab": osp.join(artifacts, "glove_vocab"),
    }
    for sub, tgt in targets.items():
        if not osp.exists(tgt):
            log.info(
                "_ensure_scorer_data_symlinks: skip %s -> %s (target missing)",
                sub, tgt,
            )
            continue
        link_path = osp.join(repo_data, sub)
        # Make any intermediate dirs for nested keys (e.g. artifacts/glove_vocab)
        os.makedirs(osp.dirname(link_path), exist_ok=True)
        tgt_abs = str(Path(tgt).resolve())
        if osp.islink(link_path):
            current = os.readlink(link_path)
            if current == tgt_abs or osp.realpath(link_path) == tgt_abs:
                continue
            # Stale link to a different target — relink.
            os.unlink(link_path)
        elif osp.exists(link_path):
            # Real file/dir at this path — leave untouched and warn.
            log.warning(
                "_ensure_scorer_data_symlinks: %s already exists as a "
                "non-symlink, skipping. Scorer may use it instead of %s.",
                link_path, tgt_abs,
            )
            continue
        os.symlink(tgt_abs, link_path)
        log.info("symlink %s -> %s", link_path, tgt_abs)


def _run_scorer_subprocess(
    project_root: str, dst: str, label: str, gpu: int = 0,
    skip_fid: bool = False,
    gt_dir: Optional[str] = None,
    smplx_dir: Optional[str] = None,
    sdf_folder: Optional[str] = None,
    obj_geo_root: Optional[str] = None,
    classifier_ckpt: Optional[str] = None,
) -> None:
    """Invoke ``eval/omomo/eval_chois_table1.py`` on ``<dst>/preds_all/``.

    The scorer module is the reference for CHOIS Table 1 metrics — we call it
    as a subprocess (not in-process) so its CUDA env-var setup
    (``CUDA_VISIBLE_DEVICES``, ``OMOMO_EVAL_DATA_ROOT``) cannot interfere
    with our process state. Fail loud on non-zero return code.

    Path overrides (gt_dir, smplx_dir, sdf_folder, obj_geo_root,
    classifier_ckpt) are forwarded to the scorer's CLI. The scorer's
    built-in defaults point to a stale per-host worktree path
 that does not exist on
    every machine; passing the cfg-resolved paths explicitly avoids
    that brittle fallback.

    Args:
        project_root: Repo root (the script lives at <root>/eval/omomo/...).
        dst: Resolved output dir; preds live at ``<dst>/preds_all/``.
        label: Short tag for the scorer's classifier intermediates.
        gpu: GPU index for the scorer subprocess. Caller already pinned
            ``CUDA_VISIBLE_DEVICES`` so this is the in-mask index (0).
        skip_fid: If True, pass ``--skip_fid`` so the scorer skips Stage B
            (FID + R_prec via the external classifier subprocess chain).
            Required for smoke / debug runs where the classifier ckpt or
            its deps may be absent — Stage A still produces the 14
            non-classifier columns; R_prec/FID return NaN.
        gt_dir: GT npz directory (forwarded as ``--gt_dir``). Resolve
            relative paths to absolute before calling.
        smplx_dir: SMPL-X model dir (forwarded as ``--smplx_dir``).
        sdf_folder: Per-object SDF dir (forwarded as ``--sdf_folder``).
        obj_geo_root: Parent of ``rest_object_geo/<obj>.ply`` (forwarded as
            ``--obj_geo_root``).
        classifier_ckpt: Stage-B classifier ckpt (only used when
            ``skip_fid=False``; forwarded as ``--classifier_ckpt``).
    """
    preds_dir = osp.join(dst, "preds_all")
    out_dir = osp.join(dst, "results")
    cmd = [
        sys.executable,
        osp.join(project_root, "eval/omomo/eval_chois_table1.py"),
        "--preds_dir", str(Path(preds_dir).resolve()),
        "--out", str(Path(out_dir).resolve()),
        "--label", label,
        "--gpu", str(gpu),
    ]
    if gt_dir is not None:
        cmd += ["--gt_dir", str(Path(gt_dir).resolve())]
    if smplx_dir is not None:
        cmd += ["--smplx_dir", str(Path(smplx_dir).resolve())]
    if sdf_folder is not None:
        cmd += ["--sdf_folder", str(Path(sdf_folder).resolve())]
    if obj_geo_root is not None:
        cmd += ["--obj_geo_root", str(Path(obj_geo_root).resolve())]
    if classifier_ckpt is not None and not skip_fid:
        cmd += ["--classifier_ckpt", str(Path(classifier_ckpt).resolve())]
    if skip_fid:
        cmd.append("--skip_fid")
    log.info("Scorer subprocess: %s", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=project_root, check=False)
    if ret.returncode != 0:
        raise RuntimeError(
            f"eval_chois_table1.py failed with exit code {ret.returncode}"
        )


def _print_table1(dst: str) -> None:
    """Read ``<dst>/results/metrics.json`` and print 16-col CHOIS table.

    Reuses ``COLUMN_ORDER``, ``PAPER_CHOIS_OURS``, and ``_verdict`` from
    the scorer module so the printed table matches the scorer's own
    ``CHOIS_TABLE1.md`` exactly.
    """
    # Local import: scorer module may need its own sys.path setup, kept
    # late so missing optional deps don't break the rest of the run.
    from eval.omomo.eval_chois_table1 import (
        COLUMN_ORDER, PAPER_CHOIS_OURS, _verdict,
    )

    metrics_path = osp.join(dst, "results", "metrics.json")
    with open(metrics_path) as fh:
        m = json.load(fh)

    rows = []
    for col in COLUMN_ORDER:
        ours = m.get(col, None)
        paper = PAPER_CHOIS_OURS[col]
        if ours is None:
            ours_str = "  NaN"
            delta_str = "  N/A"
            verdict = "NaN"
        else:
            ours_str = f"{float(ours):6.2f}"
            delta_str = f"{float(ours) - paper:+6.2f}"
            verdict = _verdict(col, float(ours), paper)
        rows.append((col, f"{paper:6.2f}", ours_str, delta_str, verdict))

    # Print: terminal-friendly column-aligned table.
    print()
    print("=" * 64)
    print("CHOIS Table 1 — 16-col comparison (paper vs ours)")
    print("=" * 64)
    print(f"  {'metric':<10} {'paper':>7}  {'ours':>7}  {'Δ':>7}   verdict")
    print("  " + "-" * 56)
    for col, paper_s, ours_s, delta_s, verdict in rows:
        print(f"  {col:<10} {paper_s:>7}  {ours_s:>7}  {delta_s:>7}   {verdict}")
    print("=" * 64)
    print(f"Full bundle: {dst}")
    print(f"Markdown:    {osp.join(dst, 'results', 'CHOIS_TABLE1.md')}")
    print()


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def _resolve_mode_params(
    args: argparse.Namespace, cfg: Any,
) -> Tuple[List[str], int, Optional[int], int, str]:
    """Resolve ``(gt_paths, K, n_steps, batch_size, scheduler_mode)``.

    Args:
        args: Parsed CLI namespace.
        cfg: Loaded training config (Mapping-like).

    Returns:
        gt_paths: Sorted absolute paths of GT npz to evaluate.
        K: 20 in both modes.
        n_steps: For ``debug``, defaults to 10. For ``full``, defaults
            to ``None`` (= scheduler uses all ``num_train_timesteps`` for
            ``ddpm_full``). ``--n_steps`` overrides either default.
        batch_size: Defaults to ``cfg.callbacks.omomo_eval.max_bs_start``.
        scheduler_mode: ``"ddim_gen"`` for debug, ``"ddpm_full"`` for full.
    """
    cb_cfg = cfg.get("callbacks", {}).get("omomo_eval", {})
    gt_dir = cb_cfg.get("gt_dir")
    if gt_dir is None:
        raise RuntimeError("cfg.callbacks.omomo_eval.gt_dir not set")
    # Resolve relative paths against project root (mirrors training behavior).
    if not osp.isabs(gt_dir):
        gt_dir = osp.abspath(osp.join(PROJECT, gt_dir))
    if not osp.isdir(gt_dir):
        raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")

    all_paths = sorted(str(p) for p in Path(gt_dir).glob("*.npz"))
    if not all_paths:
        raise FileNotFoundError(f"No .npz files in gt_dir={gt_dir}")

    # Sharding: contiguous slice across N shards. shard_idx in [0, num_shards).
    # Applied AFTER mode-based filtering below so debug + sharding still works
    # for plumbing tests (8 seqs split N-ways; in practice debug usually uses
    # num_shards=1).
    if args.mode == "debug":
        gt_paths = all_paths[:8]
        scheduler_mode = "ddim_gen"
        # Debug default: 10 ddim steps. CLI override honored.
        n_steps: Optional[int] = int(args.n_steps) if args.n_steps is not None else 10
    elif args.mode == "full":
        gt_paths = all_paths
        scheduler_mode = "ddpm_full"
        # Full default: n_steps=None so the scheduler uses all
        # num_train_timesteps. CLI override honored — but warn loudly if
        # it exceeds num_train_timesteps because set_timesteps() will fail.
        if args.n_steps is None:
            n_steps = None
        else:
            n_train = int(cfg.get("model", {}).get("diffusion", {}).get(
                "num_train_timesteps", 0,
            ))
            if n_train > 0 and int(args.n_steps) > n_train:
                raise RuntimeError(
                    f"--n_steps={args.n_steps} > num_train_timesteps={n_train}; "
                    f"DDPM scheduler.set_timesteps() will fail. Drop --n_steps "
                    f"to use all {n_train} train timesteps."
                )
            n_steps = int(args.n_steps)
    else:
        raise ValueError(f"unknown --mode: {args.mode!r}")

    K = _K  # always 20

    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    else:
        bs_default = cb_cfg.get("max_bs_start", 8)
        batch_size = int(bs_default)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # Apply contiguous shard slice. Each shard covers gt_paths[start:end] where
    # start = shard_idx * ceil(N/M), end = start + ceil(N/M). Last shard takes
    # the remainder (may be smaller). Mirrors CHOIS adapter
    # contiguous-window protocol (as for the CHOIS baseline) so all shards together
    # produce exactly the full dataset, no overlap, no gaps.
    if args.num_shards > 1:
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError(
                f"--shard_idx={args.shard_idx} out of range "
                f"[0, {args.num_shards})"
            )
        n_total = len(gt_paths)
        per_shard = (n_total + args.num_shards - 1) // args.num_shards
        start = args.shard_idx * per_shard
        end = min(start + per_shard, n_total)
        gt_paths = gt_paths[start:end]
        log.info(
            "Sharding: shard %d/%d covers gt_paths[%d:%d] (%d / %d total)",
            args.shard_idx, args.num_shards, start, end, len(gt_paths), n_total,
        )

    return gt_paths, K, n_steps, batch_size, scheduler_mode


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone OMOMO eval-and-save: K=20 best-of-MPJPE diffusion "
            "inference + scorer subprocess. Mirrors LangTable eval_and_save_wmm.py."
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Absolute path to UMM WMM .ckpt",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to training YAML config (e.g. configs/wmm/om/w1.5.0.yaml)",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output directory base (auto-suffixed with _YYYYMMDD-HHMMSS). "
            "Refuses to run if the resolved dir is non-empty. "
            "Required UNLESS --shared_dst is given. Cannot be combined "
            "with --shared_dst."
        ),
    )
    parser.add_argument(
        "--mode", required=True, choices=("debug", "full"),
        help=(
            "debug: 8 seqs, K=20, n_steps=10, ddim_gen. "
            "full: ALL seqs, K=20, ddpm_full (n_steps=None by default)."
        ),
    )
    parser.add_argument(
        "--n_steps", type=int, default=None,
        help=(
            "Override denoising step count. For ddim_gen (debug), default=10. "
            "For ddpm_full (full), default=None (use all num_train_timesteps). "
            "Setting n_steps > num_train_timesteps in full mode is a hard error."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=(
            "Inference batch size. Default = cfg.callbacks.omomo_eval.max_bs_start "
            "(8 for w1.5.0). 24 GB GPU is plenty; OOM at the default is a real bug."
        ),
    )
    parser.add_argument(
        "--skip_fid", action="store_true",
        help=(
            "Pass --skip_fid to the scorer subprocess (skip Stage B "
            "FID + R_prec via the external classifier). Auto-set in "
            "--mode debug. Use this for full mode if classifier deps "
            "are unavailable."
        ),
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help=(
            "Total number of parallel shards across multiple GPUs. "
            "Each shard processes a contiguous slice of the GT seq list "
            "(gt_paths[shard_idx * per_shard : (shard_idx+1) * per_shard]). "
            "When >1, the script SKIPS the scorer subprocess (each shard "
            "writes only its own preds_all/) — call eval_chois_table1.py "
            "manually on a merged preds_all dir, or use the launcher "
            "script to run all shards + merge + score in one go."
        ),
    )
    parser.add_argument(
        "--shard_idx", type=int, default=0,
        help=(
            "This shard's index in [0, num_shards). Required when "
            "--num_shards > 1. With --output (legacy path), each shard "
            "auto-suffixes its dst with _shard<idx>of<N> so concurrent "
            "shards do not collide. With --shared_dst (new path), shards "
            "write into the SAME dst (filenames are stem-disjoint)."
        ),
    )
    # ------------------------------------------------------------------
    # Single-dir orchestration flags (used by launch_sharded.sh).
    #
    # --setup_only:  do one-time setup (ckpt copy, src_bck, git_info,
    #                _script_backup, rollout_config) on --output, then
    #                exit 0 WITHOUT loading the model or running inference.
    #                Used by Phase 1 of launch_sharded.sh so the 5 GB
    #                ckpt and the src_bck snapshot are produced ONCE per
    #                eval, not N times.
    #
    # --shared_dst:  inference-only mode. Use the given path as dst
    #                directly (no auto-suffix, no empty check). Skips
    #                ckpt copy / src_bck / git_info / _script_backup /
    #                rollout_config (those came from setup_only).
    #                Loads the ckpt from <shared_dst>/last.ckpt.
    #                Writes per-shard meta to preds_meta_shard<i>.jsonl
    #                (no clobber across shards) and a status sentinel
    #                at _shard_status_<i>.json. Used by Phase 2 of
    #                launch_sharded.sh.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--setup_only", action="store_true",
        help=(
            "Phase-1 mode for launch_sharded.sh: do ckpt copy + src_bck "
            "+ git_info + _script_backup + rollout_config on --output, "
            "then exit 0. No model load, no inference. Mutex with "
            "--shared_dst."
        ),
    )
    parser.add_argument(
        "--shared_dst", default=None,
        help=(
            "Phase-2 mode for launch_sharded.sh: use this path as dst "
            "directly (no auto-suffix, no empty check). Skips one-time "
            "setup (already done by --setup_only). Loads ckpt from "
            "<shared_dst>/last.ckpt. Writes preds_meta_shard<i>.jsonl "
            "and _shard_status_<i>.json. Mutex with --output and "
            "--setup_only."
        ),
    )
    # CHOIS-faithful sampling defaults (chosen 2026-04-30 visual sweep).
    # M1 (replace) + M3 (smoothing 0.05) on strict partial waypoints.
    # ANY non-default setting logs a loud warning so a regression to the
    # leaky old protocol cannot happen silently.
    parser.add_argument(
        "--partial_chois", choices=("strict", "off"), default="strict",
        help=(
            "CHOIS partial-conditioning mode for the eval batch builder. "
            "'strict' (default) enforces the paper's xy-only/xyz-only "
            "channel mask at object waypoints 29/59/89/end. 'off' restores "
            "the legacy full-SE(3) leak (NOT paper-fair — warns loudly)."
        ),
    )
    parser.add_argument(
        "--chois_mode", choices=("replace", "guidance", "off"),
        default="replace",
        help=(
            "Sampling-time hook applied with partial_chois='strict'. "
            "'replace' (default = M1) hard-replaces x_t known channels "
            "every DDIM step (RePaint). 'guidance' (M2) does autograd-"
            "based classifier guidance through the LMM (~3x compute). "
            "'off' disables the hook (model denoises freely on partial "
            "frames — warns loudly)."
        ),
    )
    parser.add_argument(
        "--chois_guide_scale", type=float, default=1.0,
        help="Step size for chois_mode='guidance' (M2). Ignored otherwise.",
    )
    parser.add_argument(
        "--smooth_lambda", type=float, default=0.05,
        help=(
            "Acceleration-smoothing strength (M3). 0.0 disables. Default "
            "0.05 matches the chosen visual-sweep optimum."
        ),
    )
    args = parser.parse_args()

    # ─── CHOIS default-drift warnings ──────────────────────────────────
    # The chosen fair-eval default (2026-04-30) is partial_chois='strict'
    # + chois_mode='replace' + smooth_lambda=0.05. Any deviation logs a
    # loud warning so K=20 production runs cannot silently regress.
    if args.partial_chois != "strict":
        log.warning(
            "eval_and_save_wmm: --partial_chois=%r — eval is leaking full "
            "SE(3) (z + rot) at object waypoints. Table-1 numbers will "
            "NOT be CHOIS-paper-fair. Default is 'strict'.",
            args.partial_chois,
        )
    if args.chois_mode != "replace":
        log.warning(
            "eval_and_save_wmm: --chois_mode=%r — partial-channel "
            "constraint is NOT enforced via M1 (RePaint). Default is "
            "'replace' (the chosen fair-eval mode).",
            args.chois_mode,
        )
    if abs(args.smooth_lambda - 0.05) > 1e-9:
        log.warning(
            "eval_and_save_wmm: --smooth_lambda=%g — M3 acceleration "
            "smoothing is %s the canonical default (0.05).",
            args.smooth_lambda,
            "DISABLED" if args.smooth_lambda == 0.0 else "off",
        )

    # ------------------------------------------------------------------
    # Argument validation: enforce mutual exclusion and required combos.
    # FAIL LOUD here so the user does not waste GPU on a misconfigured
    # invocation.
    # ------------------------------------------------------------------
    if args.shared_dst is not None:
        if args.output is not None:
            parser.error(
                "--shared_dst and --output are mutually exclusive. "
                "Pick ONE: --output for own-dst single-GPU/legacy-shard "
                "path, --shared_dst for shared-dst (Phase 2) path."
            )
        if args.setup_only:
            parser.error("--shared_dst and --setup_only are mutually exclusive.")
        if args.num_shards < 1:
            parser.error("--num_shards must be >= 1 with --shared_dst.")
        if not (0 <= args.shard_idx < args.num_shards):
            parser.error(
                f"--shard_idx={args.shard_idx} out of range [0, {args.num_shards}) "
                f"with --num_shards={args.num_shards}."
            )
    else:
        if args.output is None:
            parser.error("Either --output or --shared_dst is required.")
        if args.setup_only and args.num_shards < 1:
            parser.error("--num_shards must be >= 1 with --setup_only.")

    # ------------------------------------------------------------------
    # Resolve dst path. Three orchestration modes:
    #
    #   --shared_dst PATH:  use PATH directly (no suffix, no empty check).
    #                       One-time setup files MUST already exist (created
    #                       by an earlier --setup_only call). Verified below.
    #
    #   --setup_only:       resolve dst as <output>_<TS>/ (no _shardIofN).
    #                       Create dst, do setup, exit. Refuse to overwrite.
    #
    #   default (legacy):   resolve dst as <output>_<TS>[_shard<i>of<N>]/.
    #                       Original behavior preserved for single-GPU and
    #                       legacy own-dst sharded runs.
    # ------------------------------------------------------------------
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    timestamp_compact = time.strftime("%Y%m%d-%H%M%S")
    if args.shared_dst is not None:
        args.output_user_input = args.shared_dst
        dst = osp.abspath(args.shared_dst)
        # CRITICAL: the shared dst must already exist with one-time setup
        # done. Validate every required file. Fail loud — never proceed
        # with a half-set-up dst.
        if not osp.isdir(dst):
            raise FileNotFoundError(
                f"--shared_dst does not exist: {dst!r}. Phase 1 (--setup_only) "
                f"must run first to populate this dir."
            )
        for required in ("last.ckpt", "src_bck", "git_info.txt",
                         "rollout_config.json", "_script_backup.py"):
            req_path = osp.join(dst, required)
            if not osp.exists(req_path):
                raise FileNotFoundError(
                    f"--shared_dst is missing one-time setup artifact: "
                    f"{req_path!r}. Phase 1 (--setup_only) did not complete "
                    f"successfully — re-run setup before launching shards."
                )
    else:
        args.output_user_input = args.output
        if args.setup_only:
            # Caller (launcher) controls the exact dst path. NO auto-TS
            # suffix here — the launcher already has TS in FINAL_DST and
            # double-suffixing would force a rename + a stale path in
            # rollout_config.json. The launcher must pass the final
            # path it wants.
            args.output = osp.abspath(args.output)
        else:
            # Legacy own-dst path: auto-append _<TS>[_shard<i>of<N>].
            args.output = f"{args.output}_{timestamp_compact}"
            if args.num_shards > 1:
                args.output = f"{args.output}_shard{args.shard_idx}of{args.num_shards}"
            args.output = osp.abspath(args.output)
        dst = args.output  # local alias for readability — same string
        assert_dst_empty(dst)
        os.makedirs(dst, exist_ok=True)

    log.info("=== OMOMO eval_and_save_wmm ===")
    log.info("mode:                 %s",
             "setup_only" if args.setup_only
             else ("shared_dst" if args.shared_dst is not None else "own_dst"))
    log.info("dst:                  %s", dst)
    log.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    log.info("hostname:             %s", socket.gethostname())
    log.info("git sha:              %s", get_git_sha(PROJECT))

    # ------------------------------------------------------------------
    # One-time setup: ckpt copy + src_bck + git_info + _script_backup.
    # SKIPPED in --shared_dst mode (already done by Phase 1).
    # ------------------------------------------------------------------
    src_ckpt = osp.abspath(args.checkpoint)
    if not osp.isfile(src_ckpt):
        raise FileNotFoundError(f"--checkpoint not found: {src_ckpt}")

    if args.shared_dst is None:
        # Naming policy:
        #   - --setup_only: copy to <dst>/last.ckpt (canonical name; matches
        #     eval_wmm.md §1 "<dst>/last.ckpt: Local copy of the ckpt used")
        #   - legacy own-dst path: keep original basename for backward compat
        if args.setup_only:
            dst_ckpt = osp.join(dst, "last.ckpt")
        else:
            dst_ckpt = osp.join(dst, osp.basename(src_ckpt))
        log.info(
            "Copying ckpt: %s -> %s (size=%.1f MB)",
            src_ckpt, dst_ckpt, osp.getsize(src_ckpt) / (1 << 20),
        )
        t_cp = time.time()
        shutil.copyfile(src_ckpt, dst_ckpt)
        log.info("Ckpt copy done in %.1fs", time.time() - t_cp)

        log.info("Snapshotting src/configs/eval into %s/src_bck/ ...", dst)
        snap_sec = snapshot_src(dst, PROJECT)
        log.info("src snapshot done in %.1fs", snap_sec)
        write_git_info(dst, PROJECT)
        log.info("Wrote git_info.txt")

        # _script_backup.py — verbatim copy of THIS script in dst.
        shutil.copyfile(__file__, osp.join(dst, "_script_backup.py"))
        log.info("Wrote _script_backup.py")
    else:
        # Shared-dst path: ckpt was copied as <dst>/last.ckpt by Phase 1.
        dst_ckpt = osp.join(dst, "last.ckpt")
        log.info("Reusing one-time setup at %s (skipping ckpt copy + src_bck)", dst)

    # ------------------------------------------------------------------
    # Load CFG ONLY (cheap — no model weights). cfg drives mode params,
    # rollout_config, and PREFLIGHT path validation. The MODEL is loaded
    # later (skipped in --setup_only mode).
    # ------------------------------------------------------------------
    device = torch.device("cuda:0")
    from umm.training_utils import get_config  # noqa: E402
    argv_bak = sys.argv[:]
    sys.argv = ["eval", "--config", args.config]
    try:
        cfg = get_config()
    finally:
        sys.argv = argv_bak

    # ------------------------------------------------------------------
    # Resolve mode-specific params (gt_paths, K, n_steps, batch_size,
    # scheduler_mode). Pure args+cfg — no model needed. Includes the
    # contiguous shard slice when --num_shards > 1.
    #
    # Provenance subtlety: in --setup_only mode we are writing
    # rollout_config FOR THE WHOLE EVAL, not for this shard. We must
    # record the FULL gt_paths count (e.g. 482), not the per-shard
    # slice count (e.g. 61). Force-resolve with num_shards=1 just for
    # the rollout_config write, then restore.
    # ------------------------------------------------------------------
    gt_paths, K, n_steps, batch_size, scheduler_mode = _resolve_mode_params(args, cfg)
    log.info(
        "mode=%s K=%d n_steps=%s batch_size=%d scheduler_mode=%s num_seqs=%d",
        args.mode, K, n_steps, batch_size, scheduler_mode, len(gt_paths),
    )

    # ------------------------------------------------------------------
    # Write rollout_config.json. In --setup_only it is the final
    # provenance file FOR THE WHOLE EVAL. In --shared_dst it would
    # clobber Phase 1's copy — skip there. In legacy own-dst it is
    # written here as before.
    # ------------------------------------------------------------------
    if args.shared_dst is None:
        if args.setup_only and args.num_shards > 1:
            # Re-resolve with num_shards=1 just for accurate provenance
            # (full dataset count, not per-shard slice count).
            saved_num_shards = args.num_shards
            saved_shard_idx = args.shard_idx
            args.num_shards = 1
            args.shard_idx = 0
            try:
                full_paths, _, _, _, _ = _resolve_mode_params(args, cfg)
                num_seqs_for_provenance = len(full_paths)
            finally:
                args.num_shards = saved_num_shards
                args.shard_idx = saved_shard_idx
        else:
            num_seqs_for_provenance = len(gt_paths)
        write_rollout_config(
            dst=dst, args=args, K=K, n_steps_resolved=n_steps,
            batch_size=batch_size, scheduler_mode=scheduler_mode,
            num_seqs=num_seqs_for_provenance, device=device,
            project_root=PROJECT, timestamp=timestamp_iso,
            dst_ckpt_path=dst_ckpt,
        )
        log.info("Wrote rollout_config.json (num_seqs=%d)", num_seqs_for_provenance)

    # ------------------------------------------------------------------
    # --setup_only EXIT: verify every one-time setup artifact is on
    # disk, then exit 0 WITHOUT loading the model. Hard error if any
    # missing — Phase 2 will refuse to start without these.
    # ------------------------------------------------------------------
    if args.setup_only:
        required_setup = (
            "last.ckpt",        # ckpt copy
            "src_bck",          # source snapshot dir
            "git_info.txt",     # git provenance
            "_script_backup.py",  # this script verbatim
            "rollout_config.json",  # protocol params (K, n_steps, BS, scheduler)
        )
        missing = []
        for name in required_setup:
            p = osp.join(dst, name)
            if not osp.exists(p):
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                "SETUP_ONLY FAILED: required artifact(s) missing after "
                "setup phase: " + ", ".join(repr(m) for m in missing) +
                ". Phase 2 (--shared_dst) will refuse to launch."
            )
        log.info("SETUP_ONLY OK: all 5 one-time artifacts present at %s", dst)
        log.info("=== eval_and_save_wmm setup_only complete ===")
        return

    # ------------------------------------------------------------------
    # Load MODEL from the ckpt copy. Heavy step (multi-GB ckpt + GPU
    # placement). Skipped in --setup_only.
    # ------------------------------------------------------------------
    pl_module, _ = load_pl_module(dst_ckpt, args.config, device)
    agent = _get_eval_agent(pl_module)
    if agent is None:
        raise RuntimeError(
            "Could not resolve eval agent: pl_module has neither .ema_model "
            "nor .model nor .agent."
        )
    agent.eval()

    # ------------------------------------------------------------------
    # Inference loop: chunked over seqs, K per-seed passes per chunk.
    # ------------------------------------------------------------------
    preds_all_dir = osp.join(dst, "preds_all")
    preds_K_dir = osp.join(dst, "preds_all_K20")
    os.makedirs(preds_all_dir, exist_ok=True)
    os.makedirs(preds_K_dir, exist_ok=True)
    # Per-shard meta in --shared_dst mode (no clobber across shards;
    # launcher concatenates Phase 3). Single file in legacy own-dst /
    # single-GPU mode (existing behavior).
    if args.shared_dst is not None:
        meta_jsonl = osp.join(dst, f"preds_meta_shard{args.shard_idx}.jsonl")
    else:
        meta_jsonl = osp.join(dst, "preds_meta.jsonl")

    # OMOMO callback's smplx_dir / obj_geo_dir / sdf_folder live under
    # cfg.callbacks.omomo_eval. smplx_dir + obj_geo_dir are needed by
    # _build_batch_from_gt_files (DA-pose ctx) and _save_pred_npz
    # (Q->AA inversion). sdf_folder is needed downstream by the scorer
    # subprocess (Stage A collision metrics).
    cb_cfg = cfg.get("callbacks", {}).get("omomo_eval", {})
    gt_dir = cb_cfg.get("gt_dir")
    smplx_dir = cb_cfg.get("smplx_dir")
    obj_geo_dir = cb_cfg.get("obj_geo_dir")
    sdf_folder = cb_cfg.get("sdf_folder")
    classifier_ckpt = cb_cfg.get("classifier_ckpt")  # may be None
    if gt_dir is None:
        raise RuntimeError(
            "cfg.callbacks.omomo_eval.gt_dir is not set — required for "
            "scorer subprocess. Add it to the YAML config."
        )
    if not osp.isabs(gt_dir):
        gt_dir = osp.abspath(osp.join(PROJECT, gt_dir))
    if smplx_dir is None:
        raise RuntimeError(
            "cfg.callbacks.omomo_eval.smplx_dir is not set — required for "
            "Q->AA inversion in _save_pred_npz. Add it to the YAML config."
        )
    if obj_geo_dir is None:
        raise RuntimeError(
            "cfg.callbacks.omomo_eval.obj_geo_dir is not set — required for "
            "rest-mesh PCL lookup in _build_batch_from_gt_files. Add it "
            "to the YAML config."
        )
    if sdf_folder is None:
        raise RuntimeError(
            "cfg.callbacks.omomo_eval.sdf_folder is not set — required for "
            "Stage A collision metrics in the scorer. Add it to the YAML config."
        )
    if not osp.isabs(smplx_dir):
        smplx_dir = osp.abspath(osp.join(PROJECT, smplx_dir))
    if not osp.isabs(obj_geo_dir):
        obj_geo_dir = osp.abspath(osp.join(PROJECT, obj_geo_dir))
    if not osp.isabs(sdf_folder):
        sdf_folder = osp.abspath(osp.join(PROJECT, sdf_folder))
    if classifier_ckpt is not None and not osp.isabs(classifier_ckpt):
        classifier_ckpt = osp.abspath(osp.join(PROJECT, classifier_ckpt))

    # PREFLIGHT: validate scorer dependencies BEFORE the multi-hour
    # inference. Without this, a missing path is only discovered at the
    # very end, wasting all GPU time. Each check raises with a precise
    # actionable message naming the cfg key + offending path.
    skip_fid_preflight = bool(args.skip_fid) or (args.mode == "debug")
    _preflight_required = [
        ("gt_dir",       gt_dir,       "directory"),
        ("smplx_dir",    smplx_dir,    "directory"),
        ("obj_geo_dir",  obj_geo_dir,  "directory"),
        ("sdf_folder",   sdf_folder,   "directory"),
    ]
    for key, path_, kind in _preflight_required:
        if not osp.isdir(path_):
            raise FileNotFoundError(
                f"PREFLIGHT FAILED: cfg.callbacks.omomo_eval.{key} = {path_!r} "
                f"is not an existing {kind}. Fix the cfg path before re-running "
                f"so a 2-hour inference is not wasted on a missing dep."
            )
    if not skip_fid_preflight:
        if classifier_ckpt is None:
            raise RuntimeError(
                "PREFLIGHT FAILED: cfg.callbacks.omomo_eval.classifier_ckpt is "
                "not set, but --skip_fid was not passed. Either set "
                "classifier_ckpt in the cfg (Stage B FID + R_prec) or pass "
                "--skip_fid to skip Stage B (R_prec, FID will be NaN)."
            )
        if not osp.isfile(classifier_ckpt):
            raise FileNotFoundError(
                f"PREFLIGHT FAILED: cfg.callbacks.omomo_eval.classifier_ckpt = "
                f"{classifier_ckpt!r} is not a file. Either fix the cfg path or "
                f"pass --skip_fid to skip Stage B."
            )
    log.info("PREFLIGHT OK: gt/smplx/obj_geo/sdf%s",
             "" if skip_fid_preflight else " + classifier_ckpt")

    # ------------------------------------------------------------------
    # Status sentinel writer for --shared_dst mode. Each shard MUST land
    # a status file regardless of success/failure so the launcher (Phase
    # 3) can verify integrity. NEVER skip this on exception path.
    # ------------------------------------------------------------------
    shard_status_path: Optional[str] = None
    if args.shared_dst is not None:
        shard_status_path = osp.join(
            dst, f"_shard_status_{args.shard_idx}.json"
        )

    def _write_shard_status(
        exit_status: str,
        n_stems_processed: int,
        n_stems_expected: int,
        preds_all_count: int,
        preds_K_count: int,
        missing_stems: List[str],
        error: Optional[str] = None,
        ts_start: Optional[str] = None,
        ts_end: Optional[str] = None,
    ) -> None:
        """Write shard status sentinel atomically (write tmp + rename)."""
        if shard_status_path is None:
            return
        payload: Dict[str, Any] = {
            "shard_idx": int(args.shard_idx),
            "num_shards": int(args.num_shards),
            "exit_status": exit_status,  # "success" | "failed"
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ts_start": ts_start,
            "ts_end": ts_end or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_stems_processed": int(n_stems_processed),
            "n_stems_expected": int(n_stems_expected),
            "preds_all_count": int(preds_all_count),
            "preds_K_count": int(preds_K_count),
            "K": int(K),
            "missing_stems": list(missing_stems),
            "error": error,
        }
        tmp = shard_status_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, shard_status_path)

    t_infer_start = time.time()
    ts_infer_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    n_done = 0
    # Hold meta_jsonl open for the duration of the loop — avoids 482
    # open/close cycles on NFS. Flush after every line so a crash mid-run
    # still leaves a usable jsonl on disk for inspection.
    meta_fh = open(meta_jsonl, "w")
    # try/except/finally: try wraps the inference loop, except lands a
    # "failed" sentinel BEFORE re-raising (so launcher can distinguish
    # crashed shards from still-running ones), finally closes meta_fh.
    try:
        with torch.no_grad():
            for chunk_start in range(0, len(gt_paths), batch_size):
                chunk_paths = gt_paths[chunk_start: chunk_start + batch_size]
                log.info(
                    "chunk %d-%d / %d (size=%d)",
                    chunk_start, chunk_start + len(chunk_paths), len(gt_paths),
                    len(chunk_paths),
                )
                chunk_batch, chunk_gt = _build_batch_from_gt_files(
                    chunk_paths, device, obj_geo_dir, smplx_dir=smplx_dir,
                    partial_chois=args.partial_chois,
                )

                # K inference passes, each with explicit per-pass seed.
                ret_per_k: List[torch.Tensor] = []  # K tensors of (B, T, P, 4, 4) on CPU
                for k in range(K):
                    t_k0 = time.time()
                    ret_trace = _run_inference_once(
                        agent=agent, batch=chunk_batch,
                        seed=k, n_steps=n_steps, scheduler_mode=scheduler_mode,
                        chois_mode=args.chois_mode,
                        chois_guide_scale=args.chois_guide_scale,
                        smooth_lambda=args.smooth_lambda,
                    )
                    # Move to CPU immediately to keep peak GPU memory tight; we
                    # need K tensors simultaneously for best-of-K below.
                    ret_per_k.append(ret_trace.detach().cpu())
                    log.info(
                        "  k=%d/%d done in %.1fs", k + 1, K, time.time() - t_k0,
                    )

                # Per-seq best-of-K and save.
                for b, gt_path in enumerate(chunk_paths):
                    stem = Path(gt_path).stem
                    gt = chunk_gt[b]
                    actual_len = int(min(gt["actual_len"], _T_FIXED))
                    # ret_trace_b for sample b across all K — list of (T, P, 4, 4) CPU tensors.
                    ret_b_per_k = [ret_per_k[k][b] for k in range(K)]

                    best_k, all_mpjpe = _best_of_k_per_seq(
                        ret_b_per_k, gt["Q"], actual_len,
                    )

                    # Winner -> preds_all/<stem>.npz (CHOIS schema, scorer-compatible).
                    _save_pred_npz(
                        preds_all_dir, stem, ret_b_per_k[best_k], gt,
                        smplx_dir=smplx_dir,
                    )
                    # All K -> preds_all_K20/<stem>_k<k>.npz (full record).
                    for k in range(K):
                        _save_pred_npz(
                            preds_K_dir, f"{stem}_k{k}", ret_b_per_k[k], gt,
                            smplx_dir=smplx_dir,
                        )

                    # Per-seq meta line — flush immediately so a crash mid-run
                    # still leaves a usable jsonl for inspection.
                    meta_line = {
                        "stem": stem,
                        "best_k": int(best_k),
                        # Seed is just k (one-to-one) — recorded explicitly so
                        # downstream code does not need to know the convention.
                        "best_seed": int(best_k),
                        "all_seeds": list(range(K)),
                        "all_mpjpe": all_mpjpe,
                        "actual_len": actual_len,
                    }
                    meta_fh.write(json.dumps(meta_line) + "\n")
                    meta_fh.flush()

                    n_done += 1
                    if n_done % 10 == 0 or n_done == len(gt_paths):
                        log.info(
                            "  saved %d/%d (latest: %s, best_k=%d, best_mpjpe=%.4f)",
                            n_done, len(gt_paths), stem, best_k, all_mpjpe[best_k],
                        )

                # Free GPU + CPU memory before the next chunk.
                del chunk_batch, ret_per_k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except BaseException as exc:
        # Land a "failed" sentinel BEFORE re-raising so the launcher
        # can detect the failure (vs a still-running shard) without
        # ambiguity. Re-raise so the process exits non-zero.
        if shard_status_path is not None:
            try:
                # Count THIS shard's contribution only (race-safe).
                # Listing the shared dir would leak counts from peer
                # shards. Walk this shard's stems and check existence.
                my_preds_all_now = 0
                my_preds_K_now = 0
                if osp.isdir(preds_all_dir) and osp.isdir(preds_K_dir):
                    for p in gt_paths:
                        s = Path(p).stem
                        if osp.isfile(osp.join(preds_all_dir, f"{s}.npz")):
                            my_preds_all_now += 1
                        for k in range(K):
                            if osp.isfile(osp.join(preds_K_dir, f"{s}_k{k}.npz")):
                                my_preds_K_now += 1
                _write_shard_status(
                    exit_status="failed",
                    n_stems_processed=n_done,
                    n_stems_expected=len(gt_paths),
                    preds_all_count=my_preds_all_now,
                    preds_K_count=my_preds_K_now,
                    missing_stems=[],
                    error=f"{type(exc).__name__}: {exc}",
                    ts_start=ts_infer_start,
                )
            except Exception as sentinel_exc:
                log.error("Failed to write failure sentinel: %r", sentinel_exc)
        raise
    finally:
        meta_fh.close()

    t_infer = time.time() - t_infer_start
    log.info(
        "Inference + save done: %d seqs in %.1fs (%.2fs/seq)",
        n_done, t_infer, t_infer / max(n_done, 1),
    )

    # ------------------------------------------------------------------
    # Per-shard INTEGRITY CHECK. Verify every expected stem produced its
    # winner npz AND its K=20 archive npz. Hard error on ANY missing.
    # NEVER let a shard silently report success on partial output —
    # downstream metrics computed against an incomplete preds_all/ would
    # silently misreport (fewer denominators).
    # ------------------------------------------------------------------
    expected_stems = [Path(p).stem for p in gt_paths]
    missing_winners = [
        s for s in expected_stems
        if not osp.isfile(osp.join(preds_all_dir, f"{s}.npz"))
    ]
    missing_K_per_stem: List[str] = []
    for s in expected_stems:
        for k in range(K):
            if not osp.isfile(osp.join(preds_K_dir, f"{s}_k{k}.npz")):
                missing_K_per_stem.append(f"{s}_k{k}")
                break  # report the stem once

    if missing_winners or missing_K_per_stem:
        msg_parts = []
        if missing_winners:
            msg_parts.append(
                f"{len(missing_winners)} winner npz missing in preds_all/ "
                f"(first few: {missing_winners[:5]})"
            )
        if missing_K_per_stem:
            msg_parts.append(
                f"{len(missing_K_per_stem)} stems missing at least one of "
                f"K={K} files in preds_all_K20/ (first few: "
                f"{missing_K_per_stem[:5]})"
            )
        # Land failure sentinel for the launcher BEFORE raising.
        # Counts reflect THIS shard only (race-safe — global dir
        # listing leaks peer counts).
        if shard_status_path is not None:
            my_preds_all_now = sum(
                1 for s in expected_stems
                if osp.isfile(osp.join(preds_all_dir, f"{s}.npz"))
            )
            my_preds_K_now = sum(
                1 for s in expected_stems for k in range(K)
                if osp.isfile(osp.join(preds_K_dir, f"{s}_k{k}.npz"))
            )
            _write_shard_status(
                exit_status="failed",
                n_stems_processed=n_done,
                n_stems_expected=len(gt_paths),
                preds_all_count=my_preds_all_now,
                preds_K_count=my_preds_K_now,
                missing_stems=missing_winners or missing_K_per_stem,
                error="INTEGRITY_FAIL: " + " | ".join(msg_parts),
                ts_start=ts_infer_start,
            )
        raise RuntimeError(
            "INTEGRITY CHECK FAILED for shard "
            f"{args.shard_idx}/{args.num_shards}: " + " | ".join(msg_parts) +
            ". Refusing to report success on partial output."
        )

    log.info(
        "INTEGRITY OK: %d/%d winner npz + %d/%d K-archive npz",
        len(expected_stems), len(expected_stems),
        len(expected_stems) * K, len(expected_stems) * K,
    )

    # ------------------------------------------------------------------
    # Land success sentinel for --shared_dst Phase 3 to consume.
    #
    # IMPORTANT: counts MUST reflect THIS shard's contribution only,
    # not a global dir listing. Concurrent shards write into the same
    # preds_all/ — listing it here would race-leak counts from peer
    # shards (shard 0 finishing last would see ALL files; shard 7
    # finishing first would see only its own + whatever happened to
    # land before it). Count only files for this shard's expected
    # stems (verified to exist by INTEGRITY CHECK above).
    # ------------------------------------------------------------------
    if shard_status_path is not None:
        # By the time we reach here, INTEGRITY CHECK has confirmed all
        # expected stems have their winner + K archive on disk. So:
        my_preds_all_count = len(expected_stems)
        my_preds_K_count = len(expected_stems) * K
        _write_shard_status(
            exit_status="success",
            n_stems_processed=n_done,
            n_stems_expected=len(gt_paths),
            preds_all_count=my_preds_all_count,
            preds_K_count=my_preds_K_count,
            missing_stems=[],
            error=None,
            ts_start=ts_infer_start,
        )
        log.info("Wrote shard status sentinel: %s", shard_status_path)

    # ------------------------------------------------------------------
    # SHARD EARLY-RETURN: a single shard owns only a slice of GT, so it
    # cannot run the scorer. The launcher (Phase 4) does that on the
    # full union.
    # ------------------------------------------------------------------
    if args.shared_dst is not None or args.num_shards > 1:
        log.info(
            "Shard %d/%d done. Skipping scorer (launcher Phase 4 will "
            "run eval_chois_table1.py on the full preds_all/).",
            args.shard_idx, args.num_shards,
        )
        log.info("=== eval_and_save_wmm shard complete ===")
        return

    # ------------------------------------------------------------------
    # Subprocess scoring + final 16-col print.
    # ------------------------------------------------------------------
    label = osp.basename(dst)
    # Debug mode auto-skips FID (classifier ckpt + deps may be absent on
    # smoke-test hosts; Stage A still produces the 14 non-classifier
    # columns). User can also pass --skip_fid in full mode.
    skip_fid = bool(args.skip_fid) or (args.mode == "debug")
    # Pre-create idempotent symlinks at <repo>/data/omomo/{...} so the
    # scorer's _REPO_DATA_ROOT-based fallbacks resolve. The scorer's
    # ``OMOMO_EVAL_DATA_ROOT`` env-var overwrite (eval_chois_table1.py:612)
    # then points at our worktree copy of these dirs and run_eval.py:236
    # finds rest_object_geo without reaching the broken fallback.
    _ensure_scorer_data_symlinks(
        project_root=PROJECT, gt_dir=gt_dir, smplx_dir=smplx_dir,
        sdf_folder=sdf_folder, obj_geo_dir=obj_geo_dir,
    )
    # The scorer's run_eval._pair_dirs requires EXACT stem parity between
    # gt and pred. In debug mode we evaluated only 8 of 482 stems, so
    # build a subset gt dir of symlinks to just the evaluated stems.
    # Mirrors the on-the-fly callback's subset pattern
    # (wmm_omomo_eval.py:802-807).
    eval_stems = {Path(p).stem for p in gt_paths}
    subset_gt_dir = osp.join(dst, "_scorer_gt_subset")
    os.makedirs(subset_gt_dir, exist_ok=True)
    for stem in eval_stems:
        link = osp.join(subset_gt_dir, f"{stem}.npz")
        target = osp.join(gt_dir, f"{stem}.npz")
        if not osp.exists(target):
            raise FileNotFoundError(
                f"Expected GT npz missing for evaluated stem {stem}: {target}"
            )
        if osp.islink(link) and osp.realpath(link) == osp.realpath(target):
            continue
        if osp.islink(link) or osp.exists(link):
            os.unlink(link)
        os.symlink(target, link)
    log.info(
        "Built scorer GT subset (%d symlinks) at %s",
        len(eval_stems), subset_gt_dir,
    )
    # GPU pin for scorer subprocess. eval_chois_table1.py:609 OVERWRITES
    # the inherited CUDA_VISIBLE_DEVICES with str(args.gpu) BEFORE torch
    # import — so passing gpu=0 would silently send Stage A to PHYSICAL
    # GPU 0 regardless of which GPU the wrapper was pinned to. Forward
    # the wrapper's first CUDA_VISIBLE_DEVICES entry (the actual physical
    # GPU index) so the scorer's overwrite resolves to the SAME GPU the
    # wrapper used. If unset (CPU-only or auto), fall back to 0.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    scorer_gpu = int(cvd.split(",")[0]) if cvd and cvd.split(",")[0].isdigit() else 0
    log.info(
        "Running scorer subprocess (label=%s skip_fid=%s gpu=%d) ...",
        label, skip_fid, scorer_gpu,
    )
    t_score = time.time()
    _run_scorer_subprocess(
        PROJECT, dst, label=label, gpu=scorer_gpu, skip_fid=skip_fid,
        gt_dir=subset_gt_dir, smplx_dir=smplx_dir, sdf_folder=sdf_folder,
        obj_geo_root=PROJECT,
        classifier_ckpt=classifier_ckpt,
    )
    log.info("Scorer subprocess done in %.1fs", time.time() - t_score)

    _print_table1(dst)

    log.info("=== eval_and_save_wmm complete ===")


if __name__ == "__main__":
    main()
