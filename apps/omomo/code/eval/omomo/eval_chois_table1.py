"""Canonical harness reproducing CHOIS Paper Table 1 under our eval pipeline.

Table 1 source (ground truth paper table — every column and value):
    eval/omomo/baselines/papers/tex_sources/2312.03913/sec_eccv/4_experiment.tex
    - Column headers: sec_eccv/4_experiment.tex:43
    - CHOIS ("ours") row: sec_eccv/4_experiment.tex:54
      T_s=1.71, T_e=6.31, T_xy=2.87, H_feet=4.20, FS=0.35, R_prec=0.64,
      FID=0.69, C_prec=0.80, C_rec=0.64, C_F1=0.67, C_%=0.54, P_hand=0.59,
      MPJPE=15.30, T_root=24.43, T_obj=12.53, O_obj=0.99

Pipeline:
    (a) umm.eval.omomo.run_eval over (preds_dir, gt_dir) -> metrics.json
        (compute_collision=True; needed for P_hand + FS + H_feet parity).
    (b) Optional FID + R_prec via the text-motion classifier in
        ``eval/omomo/classifier_code`` (preprocess + eval_fid_rprec.py +
        eval_rprec_dedup.py, run as subprocesses).
    (c) Canonical results bundle:
          <out>/metrics.json      flat dict: 16 CHOIS columns + provenance
          <out>/CHOIS_TABLE1.md   side-by-side ours vs paper + verdicts
          <out>/run_meta.json     git sha, timestamps, wall time, gpu, host

Example::
    python eval/omomo/eval_chois_table1.py \
        --preds_dir <folder of predicted windows> \
        --out /tmp/test_chois_table1_out \
        --label test-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `umm` importable when invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger("eval.omomo.eval_chois_table1")

# ---------------------------------------------------------------------------
# CHOIS Paper Table 1 — "CHOIS (ours)" row (sec_eccv/4_experiment.tex:54)
# ---------------------------------------------------------------------------
PAPER_CHOIS_OURS: Dict[str, float] = {
    "T_s": 1.71,
    "T_e": 6.31,
    "T_xy": 2.87,
    "H_feet": 4.20,
    "FS": 0.35,
    "R_prec": 0.64,
    "FID": 0.69,
    "C_prec": 0.80,
    "C_rec": 0.64,
    "C_F1": 0.67,
    "C_%": 0.54,
    "P_hand": 0.59,
    "MPJPE": 15.30,
    "T_root": 24.43,
    "T_obj": 12.53,
    "O_obj": 0.99,
}

# Columns where higher is BETTER (arrow ↑ in the paper table).
# R_prec / C_prec / C_rec / C_F1 have ↑; C_% has no arrow but is a
# desirable-contact-frequency metric (higher = better contact plausibility).
HIGHER_IS_BETTER: set = {"R_prec", "C_prec", "C_rec", "C_F1", "C_%"}

# Column display order — same as the paper's left-to-right order.
COLUMN_ORDER: List[str] = [
    "T_s", "T_e", "T_xy", "H_feet", "FS",
    "R_prec", "FID",
    "C_prec", "C_rec", "C_F1", "C_%", "P_hand",
    "MPJPE", "T_root", "T_obj", "O_obj",
]

# Data lives under <code root>/data/omomo (L2_eval.sh and L3_train.sh link it there).
_REPO_DATA_ROOT = _REPO_ROOT / "data/omomo"
DEFAULT_GT_DIR = str(_REPO_DATA_ROOT / "test_npz")
DEFAULT_SMPLX_DIR = str(_REPO_DATA_ROOT / "smplx")
DEFAULT_SDF_DIR = str(_REPO_DATA_ROOT / "rest_object_sdf")
# Root-dir for run_eval's cwd-relative rest-mesh lookup (`<root>/data/omomo/rest_object_geo/<obj>.ply`).
DEFAULT_OBJ_GEO_ROOT = str(_REPO_ROOT)
# Text-motion classifier code (FID, R_prec) and its weights.
CLASSIFIER_REPO = _REPO_ROOT / "eval/omomo/classifier_code"
DEFAULT_CLASSIFIER_CKPT = str(_REPO_DATA_ROOT / "artifacts/classifier/text_motion_features/model/finest.tar")
# GloVe vectors of the ~113 words used by the OMOMO text annotations.
DEFAULT_GLOVE_DIR = str(_REPO_DATA_ROOT / "artifacts/glove_vocab")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git rev-parse failed: %s", exc)
        return "unknown"


def _require_dir(path: str, label: str) -> None:
    """Fail-loud if a required directory does not exist."""
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label}: directory does not exist: {path}")


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label}: file does not exist: {path}")


def _verdict(label: str, ours: float, paper: float) -> str:
    """Classify (ours vs paper) per the spec's ±10 / 25% thresholds.

    Returns one of BEATS / PASS / CLOSE / OFF / NaN.
    """
    if ours is None or (isinstance(ours, float) and math.isnan(ours)):
        return "NaN"
    if paper == 0:
        # Guard: paper cannot be zero for any of the 16 cols; if it were, fall
        # back to a tight tolerance around absolute difference.
        diff = abs(ours - paper)
        if diff < 0.05:
            return "PASS"
        return "OFF"

    # "higher is better": ours > paper is good; ours < paper is bad.
    # "lower  is better": ours < paper is good; ours > paper is bad.
    higher_better = label in HIGHER_IS_BETTER
    if higher_better:
        rel = (ours - paper) / abs(paper)  # > 0 = better
    else:
        rel = (paper - ours) / abs(paper)  # > 0 = better

    if rel > 0.10:
        return "BEATS"
    if rel >= -0.10:
        return "PASS"
    if rel >= -0.25:
        return "CLOSE"
    return "OFF"


# ---------------------------------------------------------------------------
# Stage A: run umm.eval.omomo.run_eval programmatically
# ---------------------------------------------------------------------------


def run_stage_a(
    preds_dir: str,
    gt_dir: str,
    smplx_dir: str,
    sdf_folder: str,
    out_json: str,
    device: str = "cuda",
    cwd_for_obj_geo: Optional[str] = None,
) -> Dict[str, Any]:
    """Call run_eval as a Python function. Writes metrics.json to out_json.

    ``run_eval`` looks up per-object rest meshes via relative paths like
    ``./data/omomo/raw/data/rest_object_geo/<obj>.ply`` (see
    ``src/umm/eval/omomo/run_eval.py:224``). When our preds sit outside that
    data root, the lookup silently falls back to zero-object and all contact
    metrics go NaN. ``cwd_for_obj_geo`` temporarily chdirs so the relative
    paths resolve, then restores.
    """
    from umm.eval.omomo.run_eval import run_eval  # noqa: WPS433 (local import OK)

    logger.info("Stage A: run_eval on %d preds vs %s", _count_npz(preds_dir), gt_dir)
    original_cwd = os.getcwd()
    try:
        if cwd_for_obj_geo is not None:
            _require_dir(cwd_for_obj_geo, "cwd_for_obj_geo")
            logger.debug("chdir to %s for rest-object-geo relative lookup",
                         cwd_for_obj_geo)
            os.chdir(cwd_for_obj_geo)
        metrics = run_eval(
            gt_dir=gt_dir,
            pred_dir=preds_dir,
            out=out_json,
            smplx_model_dir=smplx_dir,
            sdf_folder=sdf_folder,
            device=device,
            compute_collision_flag=True,
        )
    finally:
        os.chdir(original_cwd)
    return metrics


def _count_npz(d: str) -> int:
    return sum(1 for _ in Path(d).glob("*.npz"))


# ---------------------------------------------------------------------------
# Stage B: FID + R_prec via external CHOIS classifier repo (subprocess)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ClassifierOutputs:
    fid: float
    r_prec_chois_top3: float
    r_prec_chois_top1: float
    r_prec_chois_top2: float
    r_prec_dedup_top3: float
    eval_json: str
    dedup_json: str


def run_stage_b(
    preds_dir: str,
    classifier_ckpt: str,
    out_dir: str,
    gpu: int,
    tag: str,
) -> ClassifierOutputs:
    """Run preprocess + eval_fid_rprec + eval_rprec_dedup as subprocesses.

    files under ``<out_dir>/classifier/`` so nothing pollutes the classifier
    repo for concurrent runs.
    """
    _require_dir(str(CLASSIFIER_REPO), "classifier_repo")
    _require_file(classifier_ckpt, "classifier_ckpt")

    work = Path(out_dir) / "classifier"
    work.mkdir(parents=True, exist_ok=True)
    preds_npz = work / f"chois_preds_processed_{tag}.npz"
    emb_npz = work / f"embeddings_{tag}.npz"
    eval_json = work / f"eval_{tag}.json"
    dedup_json = work / f"eval_{tag}_dedup.json"
    preprocess_log = work / f"preprocess_{tag}.log"
    eval_log = work / f"eval_{tag}.log"
    dedup_log = work / f"eval_{tag}_dedup.log"

    # Classifier inputs under <code root>/data/omomo (checked below).
    test_pickle = _REPO_DATA_ROOT / "chois_test_cache" / "cano_test_diffusion_manip_window_120_joints24.p"
    text_dir = _REPO_DATA_ROOT / "omomo_text_anno" / "test"
    smplh_dir = _REPO_DATA_ROOT / "smplh_amass"
    stats_pkl = _REPO_DATA_ROOT / "stats" / "cano_min_max_mean_std_data_window_120_joints24.p"
    glove_dir = Path(DEFAULT_GLOVE_DIR)

    for p, lbl in [
        (test_pickle, "test_pickle"),
        (text_dir, "text_dir"),
        (smplh_dir, "smplh_dir"),
        (stats_pkl, "stats_pkl"),
        (glove_dir, "glove_dir"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"classifier dep missing: {lbl}: {p}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # (1) preprocess: FK + canonicalize preds -> processed.npz
    logger.info("Stage B.1: preprocess_chois_preds.py -> %s", preds_npz.name)
    cmd_pp = [
        "python3", "preprocess_chois_preds.py",
        "--pred_dir", preds_dir,
        "--test_pickle", str(test_pickle),
        "--text_dir", str(text_dir),
        "--smplh_dir", str(smplh_dir),
        "--output", str(preds_npz),
    ]
    with preprocess_log.open("w") as fh:
        ret = subprocess.run(
            cmd_pp, cwd=str(CLASSIFIER_REPO), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    if ret.returncode != 0:
        raise RuntimeError(
            f"preprocess_chois_preds.py failed (rc={ret.returncode}); see {preprocess_log}"
        )

    # (2) eval FID + R_prec
    logger.info("Stage B.2: eval_fid_rprec.py on GPU %d", gpu)
    # classifier_ckpt is a file (finest.tar). eval_fid_rprec.py takes
    # --ckpt_dir pointing to the PARENT of model/finest.tar.
    ckpt_dir = Path(classifier_ckpt).parent.parent
    cmd_eval = [
        "python3", "t2m_eval_umm/eval_fid_rprec.py",
        "--preds_npz", str(preds_npz),
        "--stats_pkl", str(stats_pkl),
        "--glove_dir", str(glove_dir),
        "--ckpt_dir", str(ckpt_dir),
        "--batch_size", "32",
        "--gpu_id", "0",  # inside the subprocess, CUDA_VISIBLE_DEVICES already masks
        "--out_json", str(eval_json),
        "--embeddings_npz", str(emb_npz),
    ]
    with eval_log.open("w") as fh:
        ret = subprocess.run(
            cmd_eval, cwd=str(CLASSIFIER_REPO), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    if ret.returncode != 0:
        raise RuntimeError(
            f"eval_fid_rprec.py failed (rc={ret.returncode}); see {eval_log}"
        )

    # (3) dedup R_prec (89 unique texts)
    logger.info("Stage B.3: eval_rprec_dedup.py")
    cmd_dedup = [
        "python3", "t2m_eval_umm/eval_rprec_dedup.py",
        "--embeddings_npz", str(emb_npz),
        "--out_json", str(dedup_json),
    ]
    with dedup_log.open("w") as fh:
        ret = subprocess.run(
            cmd_dedup, cwd=str(CLASSIFIER_REPO), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    if ret.returncode != 0:
        raise RuntimeError(
            f"eval_rprec_dedup.py failed (rc={ret.returncode}); see {dedup_log}"
        )

    with eval_json.open() as fh:
        ej = json.load(fh)
    with dedup_json.open() as fh:
        dj = json.load(fh)

    return ClassifierOutputs(
        fid=float(ej["fid_gt_vs_chois"]),
        r_prec_chois_top1=float(ej["r_prec_chois"]["1"]),
        r_prec_chois_top2=float(ej["r_prec_chois"]["2"]),
        r_prec_chois_top3=float(ej["r_prec_chois"]["3"]),
        r_prec_dedup_top3=float(dj["r_prec_chois_dedup_text"]["3"]),
        eval_json=str(eval_json),
        dedup_json=str(dedup_json),
    )


# ---------------------------------------------------------------------------
# Column mapping from our aggregate -> CHOIS paper columns
# ---------------------------------------------------------------------------


def build_ours_row(
    agg: Dict[str, Any],
    classifier: Optional[ClassifierOutputs],
) -> Dict[str, float]:
    """Assemble the 16 CHOIS columns from our aggregate dict + classifier out.

    Unit-conversion rules (spec §3 + verified against metrics.py):
      * mpjpe_mm / hand_jpe_mm / mpvpe_mm / troot_mm / pbody_loss_mm_skip_noisy
        are in millimetres -> divide by 10 to get cm.
      * foot_sliding is also mm (metrics.py:150) -> divide by 10 to get cm,
        so it matches the paper's FS unit (~0.35 range).
      * t_s_cm / t_e_cm / t_xy_cm / t_obj_cm / h_feet_cm are already in cm.
      * o_obj is an unitless Frobenius-like rotation distance (as-is).
      * cf1_gt_contact_only / c_pct_pred / cprec / crec are fractions (as-is).
    """
    out: Dict[str, float] = {}
    out["T_s"] = float(agg["t_s_cm"])
    out["T_e"] = float(agg["t_e_cm"])
    out["T_xy"] = float(agg["t_xy_cm"])
    out["H_feet"] = float(agg["h_feet_cm"])
    out["FS"] = float(agg["foot_sliding"]) / 10.0  # mm -> cm
    out["C_prec"] = float(agg["cprec"])
    out["C_rec"] = float(agg["crec"])
    out["C_F1"] = float(agg["cf1_gt_contact_only"])
    out["C_%"] = float(agg["c_pct_pred"])
    out["P_hand"] = float(agg["pbody_loss_mm_skip_noisy"]) / 10.0  # mm -> cm
    out["MPJPE"] = float(agg["mpjpe_mm"]) / 10.0
    out["T_root"] = float(agg["troot_mm"]) / 10.0
    out["T_obj"] = float(agg["t_obj_cm"])
    out["O_obj"] = float(agg["o_obj"])

    if classifier is not None:
        out["R_prec"] = classifier.r_prec_chois_top3
        out["FID"] = classifier.fid
    else:
        out["R_prec"] = float("nan")
        out["FID"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# CHOIS_TABLE1.md renderer
# ---------------------------------------------------------------------------


def render_table_md(
    ours: Dict[str, float],
    paper: Dict[str, float],
    verdicts: Dict[str, str],
    label: str,
    provenance: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# CHOIS Paper Table 1 — ours vs paper\n")
    lines.append(f"**Label:** `{label}`\n")
    lines.append(
        "**Paper source:** "
        "`eval/omomo/baselines/papers/tex_sources/2312.03913/sec_eccv/"
        "4_experiment.tex:54` (CHOIS \"(ours)\" row).\n"
    )
    lines.append("**Verdicts:** BEATS = ours >10% better · PASS = within ±10% · "
                 "CLOSE = 10–25% worse · OFF = >25% worse. Higher-is-better: "
                 f"{sorted(HIGHER_IS_BETTER)}.\n")
    lines.append("")

    header = "| Column | Better | Paper (CHOIS) | Ours | Δ (ours-paper) | Verdict |"
    sep = "|---|---|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)
    for col in COLUMN_ORDER:
        direction = "↑" if col in HIGHER_IS_BETTER else "↓"
        p = paper[col]
        o = ours.get(col, float("nan"))
        if isinstance(o, float) and math.isnan(o):
            ours_str = "NaN"
            delta_str = "—"
        else:
            ours_str = f"{o:.3f}"
            delta_str = f"{o - p:+.3f}"
        lines.append(
            f"| {col} | {direction} | {p:.3f} | {ours_str} | {delta_str} | "
            f"{verdicts[col]} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    for k in ("preds_dir", "gt_dir", "classifier_ckpt", "git_sha", "date",
              "gpu", "host", "wall_time_s"):
        if k in provenance:
            lines.append(f"- **{k}**: `{provenance[k]}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--preds_dir",
        required=True,
    )
    p.add_argument("--gt_dir", default=DEFAULT_GT_DIR)
    p.add_argument(
        "--obj_geo_root",
        default=DEFAULT_OBJ_GEO_ROOT,
        help="Directory whose `data/omomo/rest_object_geo/<obj>.ply` is read by run_eval (default: the code root).",
    )
    p.add_argument("--classifier_ckpt", default=DEFAULT_CLASSIFIER_CKPT)
    p.add_argument("--smplx_dir", default=DEFAULT_SMPLX_DIR)
    p.add_argument("--sdf_folder", default=DEFAULT_SDF_DIR)
    p.add_argument("--out", default="eval/omomo/results/latest/")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default="ours-K20-DDPM1000")
    p.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID + R_prec classifier pass (useful if classifier repo absent).",
    )
    p.add_argument(
        "--reuse_stage_a",
        help="If set, skip stage A and load this metrics.json (for dev).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Fail-loud on required inputs. Resolve to absolute because stage A may
    # chdir mid-run (for rest-object-geo relative lookups).
    args.preds_dir = str(Path(args.preds_dir).resolve())
    args.gt_dir = str(Path(args.gt_dir).resolve())
    args.smplx_dir = str(Path(args.smplx_dir).resolve())
    args.sdf_folder = str(Path(args.sdf_folder).resolve())
    _require_dir(args.preds_dir, "preds_dir")
    _require_dir(args.gt_dir, "gt_dir")
    _require_dir(args.smplx_dir, "smplx_dir")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # ---- Stage A ---------------------------------------------------------
    stage_a_json = str((out_dir / "metrics_raw.json").resolve())
    if args.reuse_stage_a:
        _require_file(args.reuse_stage_a, "reuse_stage_a")
        with open(args.reuse_stage_a) as fh:
            raw = json.load(fh)
        logger.info("Stage A: reusing %s (%d windows)", args.reuse_stage_a,
                    raw.get("aggregate", {}).get("n_windows", -1))
        shutil.copyfile(args.reuse_stage_a, stage_a_json)
    else:
        # Pin the GPU before importing torch.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # Tell run_eval.py where to find rest_object_geo/ via env var
        # (new resolution path added to run_eval.py).
        os.environ["OMOMO_EVAL_DATA_ROOT"] = str(_REPO_DATA_ROOT)
        raw = run_stage_a(
            preds_dir=args.preds_dir,
            gt_dir=args.gt_dir,
            smplx_dir=args.smplx_dir,
            sdf_folder=args.sdf_folder,
            out_json=stage_a_json,
            device="cuda",
            cwd_for_obj_geo=args.obj_geo_root,
        )
    agg = raw["aggregate"]

    # ---- Stage B ---------------------------------------------------------
    classifier: Optional[ClassifierOutputs] = None
    if not args.skip_fid:
        try:
            classifier = run_stage_b(
                preds_dir=args.preds_dir,
                classifier_ckpt=args.classifier_ckpt,
                out_dir=str(out_dir),
                gpu=args.gpu,
                tag=args.label.replace("/", "_").replace(" ", "_"),
            )
        except FileNotFoundError as exc:
            logger.warning(
                "FID/R_prec deps missing: %s — continuing with --skip_fid style NaN rows.",
                exc,
            )
            classifier = None

    # ---- Assemble canonical row -----------------------------------------
    ours = build_ours_row(agg, classifier)
    verdicts = {col: _verdict(col, ours[col], PAPER_CHOIS_OURS[col]) for col in COLUMN_ORDER}

    t_end = time.time()
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    wall = t_end - t_start

    provenance = {
        "label": args.label,
        "preds_dir": args.preds_dir,
        "gt_dir": args.gt_dir,
        "classifier_ckpt": args.classifier_ckpt if classifier is not None else None,
        "git_sha": _git_sha(_REPO_ROOT),
        "date": start_iso,
        "gpu": args.gpu,
        "host": socket.gethostname(),
        "wall_time_s": round(wall, 2),
        "n_windows": int(agg.get("n_windows", -1)),
        "stage_a_version": raw.get("config", {}).get("version", "unknown"),
        "skip_fid": bool(args.skip_fid or classifier is None),
    }

    # ---- metrics.json (flat, one key per CHOIS col + provenance) ---------
    def _nan_safe(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    metrics_flat: Dict[str, Any] = {col: _nan_safe(ours[col]) for col in COLUMN_ORDER}
    metrics_flat["paper_chois_ours"] = dict(PAPER_CHOIS_OURS)
    metrics_flat["verdicts"] = verdicts
    metrics_flat["provenance"] = provenance
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(metrics_flat, fh, indent=2, allow_nan=False)

    # ---- CHOIS_TABLE1.md -------------------------------------------------
    md = render_table_md(ours, PAPER_CHOIS_OURS, verdicts, args.label, provenance)
    (out_dir / "CHOIS_TABLE1.md").write_text(md)

    # ---- run_meta.json ---------------------------------------------------
    meta = {
        "git_sha": provenance["git_sha"],
        "start_iso": start_iso,
        "end_iso": end_iso,
        "wall_time_s": round(wall, 2),
        "gpu": args.gpu,
        "host": provenance["host"],
        "python": sys.version,
        "argv": sys.argv,
    }
    with (out_dir / "run_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- Console summary -------------------------------------------------
    logger.info("=" * 70)
    logger.info("CHOIS Table 1 — label=%s", args.label)
    logger.info("%-8s %10s %10s %8s  %s", "col", "paper", "ours", "Δ", "verdict")
    for col in COLUMN_ORDER:
        p = PAPER_CHOIS_OURS[col]
        o = ours[col]
        if isinstance(o, float) and math.isnan(o):
            logger.info("%-8s %10.3f %10s %8s  %s", col, p, "NaN", "-", verdicts[col])
        else:
            logger.info("%-8s %10.3f %10.3f %+8.3f  %s",
                        col, p, o, o - p, verdicts[col])
    logger.info("=" * 70)
    logger.info("Wrote: %s/{metrics.json,CHOIS_TABLE1.md,run_meta.json}", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
