#!/usr/bin/env bash
# Layer 3: retrain the humanoid-retargeting WMM checkpoint (paper config w1.0.1, checkpoint at step 38,000).
#
#   apps/retargeting/L3_train.sh [gpu] [extra train.py args]      e.g. apps/retargeting/L3_train.sh 3
#
# The paper checkpoint was trained on ONE GPU with batch 8. The trainer uses every listed GPU
# (data-parallel, 8 samples per GPU), so listing more GPUs changes the global batch and the recipe.
#
# Environment:
#   WMM_DATA        release data root (default: <repo>/release_data)
#   PYTHON          python of the `wmm` conda env (default: python on PATH)
#   MAX_STEPS       training steps (default: 38000, the step of the paper checkpoint)
#   RUN_NAME        experiment name; logs and checkpoints go to L3_outputs/<RUN_NAME>/ (default: w1.0.1_hh_l3)
#
# Score L3_outputs/<RUN_NAME>/version_0/ckpt_keep/for_inference.ckpt (written at the end, see below). An
# interrupted run can resume from its newest checkpoint (saved every 2,000 steps; give the full path), but a
# resumed run restarts the data stream, so it is not equivalent to one pass:
#   apps/retargeting/L3_train.sh <gpu> --resume <release_data>/retargeting/L3_outputs/<RUN_NAME>/version_0/ckpt_keep/last.ckpt
set -euo pipefail

GPUS=${1:-${CUDA_VISIBLE_DEVICES:-0}}
shift || true
APP=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$APP/../.." && pwd)
DATA=$(readlink -f "${WMM_DATA:-$ROOT/release_data}/retargeting")
PY=${PYTHON:-python}
CODE=$APP/code/train
OUT=$DATA/L3_outputs

# lib_twist (robot assets, 105 MB) and pytorch_kinematics live in the data release. The in-training
# retargeting check runs eval/retargeting_new/scripts/2mujoco_rollout.py, which looks for them in <code>/ext.
for lib in lib_twist pytorch_kinematics; do
  ln -sfn "$DATA/code_ext/ext/$lib" "$CODE/ext/$lib"
done

export CUDA_VISIBLE_DEVICES=$GPUS
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export WANDB_MODE=offline
RUN=${RUN_NAME:-w1.0.1_hh_l3}
# A private torch.compile cache, compiled in-process, so concurrent jobs of the same user cannot remove
# kernel files this run is using.
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor_cache_$RUN TRITON_CACHE_DIR=$OUT/.triton_cache_$RUN
export TORCHINDUCTOR_COMPILE_THREADS=1
PYTHONPATH=$CODE:$CODE/src
for lib in "$CODE"/ext/*; do
  if [ -d "$lib/src" ]; then PYTHONPATH=$PYTHONPATH:$lib/src; else PYTHONPATH=$PYTHONPATH:$lib; fi
done
export PYTHONPATH

# The config names ./data, ./data_splits, ./ext and ./logs/eval relative to the working directory.
cd "$DATA/L3_train"
mkdir -p "$OUT"
# The trainer deletes checkpoints/last.ckpt when training ends. last.ckpt is rewritten in place, so a hard link
# to it always holds the newest checkpoint and survives that deletion. It sits in ckpt_keep/, a sibling of
# checkpoints/, because `--resume <path>` reads the version number from the folder two levels above the
# checkpoint.
(
  while sleep 10; do
    for f in "$OUT/$RUN"/version_*/checkpoints/last.ckpt; do
      [ -f "$f" ] || continue
      k=$(dirname "$(dirname "$f")")/ckpt_keep/last.ckpt
      mkdir -p "$(dirname "$k")"
      [ "$f" -ef "$k" ] || ln -f "$f" "$k"
    done
  done
) &
KEEPER=$!
trap 'kill $KEEPER' EXIT
"$PY" "$CODE/train.py" \
  --config "$CODE/configs/wmm/hh/w1.0.1.yaml" \
  --workers 12 --viz_try \
  --expname "$RUN" \
  --log_dir "$OUT" \
  --max_steps "${MAX_STEPS:-38000}" \
  "$@"
# Inference reads the EMA model's context buffer, which only the in-training check (callbacks.hh_eval) fills;
# L3_share_ctx_buffer.py fills it from the recorded buffer when the check was disabled (README, Layer 3).
V=$(ls -d "$OUT/$RUN"/version_* | sort -V | tail -1)
"$PY" "$APP/L3_share_ctx_buffer.py" "$V/ckpt_keep/last.ckpt" "$V/ckpt_keep/for_inference.ckpt"
