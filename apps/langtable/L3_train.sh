#!/usr/bin/env bash
# Layer 3: retrain the Language Table WMM checkpoint (paper config w1.0.3_b8all) from the released data.
#
#   apps/langtable/L3_train.sh [gpus]      e.g. apps/langtable/L3_train.sh 0
#
# The paper checkpoint was trained with batch 64 per GPU and taken at step 180,000. The trainer uses every
# listed GPU (data-parallel, 64 samples per GPU), so the global batch grows with the number of GPUs.
#
# Environment:
#   WMM_DATA        release data root (default: <repo>/release_data)
#   WMM_LT_PYTHON   python of the `wmm-lt` conda env (default: python on PATH)
#   MAX_STEPS       training steps (default: 180000, the step of the paper checkpoint)
#   RUN_NAME        experiment name; logs and checkpoints go to L3_outputs/<RUN_NAME>/ (default: w1.0.3_b8all_l3)
# Extra arguments after the GPU list are passed to train.py (e.g. --init_seed 43 --viz_every_n_train_steps -1).
# Do not resume a stopped run with --resume: the checkpoint holds no random-generator state and train.py reseeds
# at every launch, so the resumed run replays the training-sample sequence from its beginning. Start a new run.
set -euo pipefail

GPUS=${1:-${CUDA_VISIBLE_DEVICES:-0}}
shift || true
APP=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$APP/../.." && pwd)
DATA=${WMM_DATA:-$ROOT/release_data}/langtable
PY=${WMM_LT_PYTHON:-python}
CODE=$APP/code/train
OUT=$DATA/L3_outputs

export CUDA_VISIBLE_DEVICES=$GPUS
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# T5 text encoder from the Layer 2 copy, offline; Weights & Biases logs stay local.
export HF_HOME=$DATA/L2_eval/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
# The model is wrapped in torch.compile. Give this run its own kernel caches: the default /tmp/torchinductor_$USER
# is shared by every job of the user, and another job cleaning it can crash this one. Compile in-process
# (no compile worker subprocesses).
CACHE=/tmp/langtable_${RUN_NAME:-w1.0.3_b8all_l3}_$$
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
PYTHONPATH=$CODE/src
for lib in "$CODE"/ext/*; do
  if [ -d "$lib/src" ]; then PYTHONPATH=$PYTHONPATH:$lib/src; else PYTHONPATH=$PYTHONPATH:$lib; fi
done
export PYTHONPATH

# The config names data_wmm/... and data_splits/... relative to the working directory. `--workers 12` is the
# paper run's command-line value (its wandb-metadata.json args); the config file alone gives 10.
cd "$DATA/L3_train"
mkdir -p "$OUT"
"$PY" "$CODE/train.py" \
  --config "$CODE/configs/wmm/lt/w1.0.3_b8all.yaml" \
  --expname "${RUN_NAME:-w1.0.3_b8all_l3}" \
  --log_dir "$OUT" \
  --workers 12 \
  --max_steps "${MAX_STEPS:-180000}" \
  "$@"
