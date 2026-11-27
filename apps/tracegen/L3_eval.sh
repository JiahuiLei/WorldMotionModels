#!/usr/bin/env bash
# Layer 3: evaluate a retrained checkpoint on a full TraceGen test split, exactly like Layer 2.
#
#   bash apps/tracegen/L3_eval.sh <gpu> <run> <droid|epic>     e.g. bash apps/tracegen/L3_eval.sh 0 droid_k128 droid
#   bash apps/tracegen/L3_eval.sh <gpu> <run> <droid|epic> <shard> <shards> <stamp>
#
# The second form splits the test split into <shards> parts, one per GPU: start shard 0 first (it prepares the output
# folder and, at the end, waits for the others and computes the metrics), then shards 1 … <shards>-1 within 15 minutes,
# all with the same <stamp> (e.g. 20261123-0700), which names the output folder.
#
# Checkpoint: L3_outputs/w1.6.0_<run>_L3/ckpt_keep/last_link.ckpt (the final training step; the trainer deletes
# checkpoints/last.ckpt at the end of fit). The script refuses a checkpoint whose step is not the paper step.
# Output: $WMM_DATA/tracegen/L3_eval_outputs/<run>_<env>_<time or stamp>/ (same layout as Layer 2).
# For a run trained with RUN_NAME=<name> (L3_train.sh), set the same RUN_NAME: it reads L3_outputs/<name>/ and writes
# <name without w1.6.0_>_<env>_….
set -euo pipefail

GPU="${1:?usage: L3_eval.sh <gpu> <run> <droid|epic> [<shard> <shards> <stamp>]}"
RUN="${2:?usage: L3_eval.sh <gpu> <run> <droid|epic> [<shard> <shards> <stamp>]}"
ENV="${3:?usage: L3_eval.sh <gpu> <run> <droid|epic> [<shard> <shards> <stamp>]}"
SHARD="${4:-0}"
SHARDS="${5:-1}"
case "$RUN" in
  droid_k128) STEPS=50000 ;;
  epic_k128) STEPS=30000 ;;
  droid_k400) STEPS=80000 ;;
  epic_k400) STEPS=60000 ;;
  droidepic_k128) STEPS=46000 ;;
  droidepic_k400) STEPS=68000 ;;
  *) echo "unknown run: $RUN" >&2; exit 2 ;;
esac
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${WMM_DATA:-$(cd "$APP/../.." && pwd)/release_data}/tracegen"
PY="${WMM_PYTHON:-python}"
CODE="$APP/code"
RUNDIR="$DATA/L3_outputs/${RUN_NAME:-w1.6.0_${RUN}_L3}"
CKPT="$RUNDIR/ckpt_keep/last_link.ckpt"
NAME="${RUN}_${ENV}"
[ -n "${RUN_NAME:-}" ] && NAME="${RUN_NAME#w1.6.0_}_${ENV}"

step=$("$PY" -c "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)['global_step']))" "$CKPT")
if [ "$step" != "$STEPS" ]; then
  echo "$CKPT is at step $step, expected $STEPS (training not finished?)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$DATA/checkpoints/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WMM_DINOV3_REPO="$CODE/ext/dinov3" WMM_DINOV3_CKPT_DIR="$DATA/checkpoints/dinov3"
# Put the released code first on the Python path so that no other installed copy of `umm` is imported.
# umm.datasets.wmm (loaded by the run) imports lib_mani, which imports pytorch_kinematics.
export PYTHONPATH="$CODE/src:$CODE:$CODE/ext/lib_io:$CODE/ext/lib_mani:$CODE/ext/pytorch_kinematics/src${PYTHONPATH:+:$PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  checkpoint=$CKPT  step=$step  shard $SHARD of $SHARDS"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$GPU"

OUTPUT="$DATA/L3_eval_outputs/$NAME"
if [ "$SHARDS" -gt 1 ]; then
  OUTPUT="${OUTPUT}_${6:?a stamp shared by all shards is required when <shards> > 1}"
fi
# The config reads test data from ./data_wmm/tracegen/data, relative to the working directory.
cd "$DATA/L2_eval"
"$PY" "$CODE/eval/tracegen/eval_and_save_wmm.py" \
  --checkpoint "$CKPT" \
  --config "$CODE/configs/wmm/tg/w1.6.0_${RUN}_tpsqspt.yaml" \
  --output "$OUTPUT" \
  --env "$ENV" --eval_effort 1.0 --seed 42 --gen_timesteps 50 \
  --batch_size 8 --num_workers 8 \
  --shard_id "$SHARD" --num_shards "$SHARDS"
