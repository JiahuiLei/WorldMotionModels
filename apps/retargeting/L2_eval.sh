#!/usr/bin/env bash
# Layer 2: compute one row of the retargeting table from scratch: retarget -> MuJoCo rollout -> score.
#
# Usage:  apps/retargeting/L2_eval.sh <row> [stage]
#   row   : wmm_clean | wmm_infill | wmm_noisy
#           gmr_clean | gmr_infill | gmr_noisy
#           phuma_clean | phuma_infill | phuma_noisy
#   stage : all (default) | infer | rollout | score
# Environment:
#   CUDA_VISIBLE_DEVICES  GPU for WMM inference (one GPU per process)
#   N_FIRST               score only the first N test clips (default 2000, as in the paper); a small value
#                         (e.g. 20) gives a quick check of the whole pipeline
#   SPLIT_RANGE           start:end: run `infer` only on test-split entries [start, end) to spread one
#                         row over several GPUs or CPU sets; run `rollout` and `score` once afterwards
#   WMM_CKPT              WMM checkpoint (default release_data/retargeting/checkpoints/w1.0.1_v4_38k_ctx.ckpt,
#                         the released paper checkpoint with its context buffer); e.g. a Layer-3 checkpoint
#   WMM_TAG               suffix for WMM run names, so outputs of another checkpoint do not overwrite
#                         the paper checkpoint's (e.g. WMM_TAG=_l3 -> w1.0.1_v4_38k_l3...)
#   WMM_DATA              data root (default <repo>/release_data)
#   PYTHON                interpreter of the "wmm" env (WMM, rollout, scoring)
#   PYTHON_GMR            interpreter of the "gmr" env (GMR retargeting)
#   PYTHON_PHUMA          interpreter of the "phuma" env (PHUMA retargeting)
#   MJC_THREADS           MuJoCo rollout worker processes (default 16)
#   GMR_WORKERS           GMR worker processes, 8 pinned CPUs each (default 8)
#   PHUMA_WORKERS         PHUMA worker processes, 4 pinned CPUs each (default 20)
#   SCORE_WORKERS         scorer threads (default 32)
# Baseline workers pin themselves to consecutive CPUs of the allowed set; restrict
# that set with `taskset -c <cpus>` when other jobs share the machine.
set -euo pipefail

APP=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$APP/../.." && pwd)
DATA=$(readlink -f "${WMM_DATA:-$REPO/release_data}/retargeting")
PY=${PYTHON:-python}
ROW=${1:?row required}
STAGE=${2:-all}
N_FIRST=${N_FIRST:-2000}
MJC_THREADS=${MJC_THREADS:-16}
SCORE_WORKERS=${SCORE_WORKERS:-32}

CLEAN=amass_human_test
NOISY=amass_human_test_q0.1_t0.1
case "$ROW" in
  wmm_clean)    RUN=w1.0.1_v4_38k;                HUMAN=$CLEAN; ARGS="--cond_step 0 --skip 1" ;;
  wmm_infill)   RUN=w1.0.1_v4_38k_3fps;           HUMAN=$CLEAN; ARGS="--cond_step 0 --skip 10" ;;
  wmm_noisy)    RUN=w1.0.1_v4_38k_q0.1_t0.1_ct10; HUMAN=$NOISY; ARGS="--cond_step 10 --skip 1" ;;
  gmr_clean)    RUN=amass_gmr_inferred;           HUMAN=$CLEAN; ARGS="--skip 1" ;;
  gmr_infill)   RUN=amass_gmr_inferred_3fps;      HUMAN=$CLEAN; ARGS="--skip 10" ;;
  gmr_noisy)    RUN=amass_gmr_inferred_q0.1_t0.1; HUMAN=$NOISY; ARGS="--skip 1" ;;
  phuma_clean)  RUN=amass_phuma3001;              HUMAN=$CLEAN; ARGS="--skip 1" ;;
  phuma_infill) RUN=amass_phuma3001_3fps;         HUMAN=$CLEAN; ARGS="--skip 10" ;;
  phuma_noisy)  RUN=amass_phuma3001_q0.1_t0.1;    HUMAN=$NOISY; ARGS="--skip 1" ;;
  *) echo "unknown row: $ROW" >&2; exit 2 ;;
esac
case "$ROW" in wmm_*) RUN=$RUN${WMM_TAG:-} ;; esac
WMM_CKPT=${WMM_CKPT:-$DATA/checkpoints/w1.0.1_v4_38k_ctx.ckpt}
for need in "$DATA/L2_eval/$HUMAN" "$DATA/L2_eval/smpls/smplx" "$DATA/code_ext"; do
  [ -e "$need" ] || { echo "missing $need (see apps/retargeting/README.md, Layer 2)" >&2; exit 1; }
done

# The evaluation scripts resolve every path relative to their project root (three levels above the
# script). Build that root from symbolic links so the scripts run unmodified and all outputs land in
# $DATA/L2_outputs.
WR=$DATA/L2_work
OUT=$DATA/L2_outputs
H=$WR/logs/eval/humanoid
mkdir -p "$WR/ext" "$WR/data/amass_phuma_v2" "$WR/data_splits/amass_humanoid_v2" "$H" \
         "$OUT/$RUN" "$OUT/${RUN}_mjc_rollout" "$OUT/reports"
ln -sfn "$APP/code/src"                         "$WR/src"
ln -sfn "$APP/code/prepare_data"                "$WR/prepare_data"
ln -sfn "$APP/code/configs"                     "$WR/configs"
[ -L "$WR/eval" ] && unlink "$WR/eval"
mkdir -p "$WR/eval/retargeting_new/scripts"
ln -sfn "$APP/code/eval/ms3"                    "$WR/eval/ms3"
cp -f "$APP"/code/eval/retargeting_new/scripts/*.py "$WR/eval/retargeting_new/scripts/"
ln -sfn "$APP/code/ext/lib_human"               "$WR/ext/lib_human"
ln -sfn "$DATA/code_ext/ext/lib_twist"          "$WR/ext/lib_twist"
ln -sfn "$DATA/code_ext/ext/pytorch_kinematics" "$WR/ext/pytorch_kinematics"
ln -sfn "$DATA/L2_eval/smpls"                   "$WR/data/smpls"
ln -sfn "$DATA/L2_eval/amass_phuma"             "$WR/data/amass_phuma"
ln -sfn "$DATA/L2_eval/amass_phuma_v2/g1"       "$WR/data/amass_phuma_v2/g1"
ln -sfn "$DATA/L1_results/splits/amass_humanoid_v2_full_test.json" \
        "$WR/data_splits/amass_humanoid_v2/full_test.json"
ln -sfn "$DATA/L2_eval/$CLEAN"                  "$H/$CLEAN"
ln -sfn "$DATA/L2_eval/$NOISY"                  "$H/$NOISY"
ln -sfn "$OUT/$RUN"                             "$H/$RUN"
ln -sfn "$OUT/${RUN}_mjc_rollout"               "$H/${RUN}_mjc_rollout"

# lib_io and lib_mani are imported by the umm package (their functions are not used by this config).
export PYTHONPATH=$WR/src:$WR/ext/lib_twist/src:$WR/ext/lib_human/src:$WR/ext/pytorch_kinematics/src:$APP/code/train/ext/lib_io:$APP/code/train/ext/lib_mani${PYTHONPATH:+:$PYTHONPATH}
S=$WR/eval/retargeting_new/scripts
cd "$WR"
# SAPIEN (imported by the umm package) segfaults without a visible GPU unless it renders on the CPU.
export SAPIEN_RENDER_DEVICE=cpu SAPIEN_DISABLE_RAY_TRACING=1
echo "[L2] row=$ROW run=$RUN stage=$STAGE n_first=$N_FIRST CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} data=$DATA"

# GMR and PHUMA run with the baseline code laid out as in the repository the baseline runs came from:
# the scripts resolve paths two levels above themselves.
baseline_root() {
  WRB=$DATA/L2_work_baselines
  local X=$DATA/code_ext/baselines
  mkdir -p "$WRB/eval/humanoid" "$WRB/envs" "$WRB/data_processing" "$WRB/data" \
           "$WRB/data_splits/amass_humanoid_v2" "$WRB/logs/eval"
  cp -f "$APP"/code_baselines/eval/humanoid/*.py "$WRB/eval/humanoid/"
  ln -sfn "$APP/code_baselines/datasets"        "$WRB/datasets"
  ln -sfn "$X/envs/lib_twist"                   "$WRB/envs/lib_twist"
  ln -sfn "$X/data_processing/lib_human"        "$WRB/data_processing/lib_human"
  ln -sfn "$X/data_processing/lib_phuma"        "$WRB/data_processing/lib_phuma"
  ln -sfn "$X/data_processing/io_utils"         "$WRB/data_processing/io_utils"
  ln -sfn "$X/data_processing/GMR"              "$WRB/data_processing/GMR"
  ln -sfn "$DATA/L2_eval/smpls"                 "$WRB/data/smpls"
  # PHUMA reads the SMPL-X model from PHUMA/asset/human_model/smplx
  ln -sfn "$DATA/L2_eval/smpls/smplx"           "$X/data_processing/lib_phuma/PHUMA/asset/human_model/smplx"
  ln -sfn "$DATA/L1_results/splits/amass_humanoid_v2_full_test.json" \
          "$WRB/data_splits/amass_humanoid_v2/full_test.json"
  ln -sfn "$DATA/L2_eval/$HUMAN"                "$WRB/logs/eval/$HUMAN"
  ln -sfn "$OUT/$RUN"                           "$WRB/logs/eval/$RUN"
  cd "$WRB"
}

# Sets $split to the test split (first N_FIRST entries), or to its SPLIT_RANGE slice, relative to the
# current directory.
select_split() {
  local range=${SPLIT_RANGE:-0:$N_FIRST}
  split=data_splits/amass_humanoid_v2/full_test_${range/:/_}.json
  "$PY" -c 'import json, sys; a, b = map(int, sys.argv[1].split(":")); json.dump(json.load(open(sys.argv[2]))[a:b], open(sys.argv[3], "w"))' \
    "$range" data_splits/amass_humanoid_v2/full_test.json "$split"
}

infer() {
  case "$ROW" in
    wmm_*)
      [ -f "$WMM_CKPT" ] || { echo "missing $WMM_CKPT (download the checkpoints archive)" >&2; exit 1; }
      select_split
      # runpy keeps scripts/ off sys.path, so `import eval` finds the eval/ package, not scripts/eval.py
      "$PY" -c 'import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' \
        "$S/1infer_ours.py" \
        --config configs/wmm/hh/w1.0.1.yaml \
        --ckpt "$WMM_CKPT" \
        --save_dir "logs/eval/humanoid/$RUN" \
        --human_data_root "logs/eval/humanoid/$HUMAN/" \
        --test_split_fn "$split" \
        $ARGS --warmup --device cuda:0 ;;
    gmr_*)
      baseline_root
      select_split
      PYTHONPATH=$WRB/data_processing/GMR:$WRB/envs/lib_twist/pytorch_kinematics/src "${PYTHON_GMR:?set PYTHON_GMR}" "$WRB/eval/humanoid/1infer_gmr.py" \
        --human_data_root "./logs/eval/$HUMAN" --save_dir "./logs/eval/$RUN" --test_split_fn "$split" \
        --num_workers "${GMR_WORKERS:-8}" --cpus_per_worker 8 $ARGS ;;
    phuma_*)
      baseline_root
      select_split
      PYTHONPATH=$WRB/envs/lib_twist/pytorch_kinematics/src "${PYTHON_PHUMA:?set PYTHON_PHUMA}" "$WRB/eval/humanoid/1infer_phuma.py" \
        --human_data_root "./logs/eval/$HUMAN" --save_dir "./logs/eval/$RUN" --test_split_fn "$split" \
        --num_workers "${PHUMA_WORKERS:-20}" --cpus_per_worker 4 $ARGS ;;
  esac
}

rollout() {
  "$PY" "$S/2mujoco_rollout.py" --src "$H/$RUN" --thread "$MJC_THREADS"
}

score() {
  "$PY" "$S/eval.py" --human_name "$HUMAN" --kinematic_name "$RUN" \
    --physical_name "${RUN}_mjc_rollout" --N_first "$N_FIRST" --workers "$SCORE_WORKERS"
  mv "$H/${RUN}_mjc_rollout_report_${N_FIRST}.xlsx" "$OUT/reports/"
  echo "[L2] report: $OUT/reports/${RUN}_mjc_rollout_report_${N_FIRST}.xlsx"
}

case "$STAGE" in
  all)     infer; rollout; score ;;
  infer)   infer ;;
  rollout) rollout ;;
  score)   score ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
