#!/usr/bin/env bash
# Layer 3: retrain the WMM OMOMO model (config w1.8.1.1) from scratch.
#
#   apps/omomo/L3_train.sh <gpu> [extra train.py options]
#
# One GPU, batch 16, no gradient accumulation: the setting of the paper checkpoint (1 RTX A6000, 839 steps per
# epoch, about 46 h to step 164,000). The paper checkpoint is the run's last.ckpt at step 164,000.
# The config trains to 1M steps; this script stops at 164,000 (override with --max_steps N).
# Resume a stopped run with --resume <absolute path to last.ckpt>. A resumed run re-seeds (seed 42) and its data sampler
# draws from the global random generator, so it replays the data order of a fresh start.
# Checkpoints: $WMM_DATA/omomo/L3_outputs/w1.8.1.1_omomo/version_*/checkpoints/last.ckpt while training. The trainer
# deletes that file when it stops; this script keeps it as $WMM_DATA/omomo/L3_outputs/final/last.ckpt.
# Every 10k steps the run evaluates 2% of the OMOMO test windows and logs omomo_eval/MPJPE (best checkpoint).
# Weights & Biases logging is on in the config; run `wandb offline` first if you have no account.
set -euo pipefail

GPU=${1:?usage: L3_train.sh <gpu> [extra train.py options]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DATA=$(cd "${WMM_DATA:-$REPO/release_data}/omomo" && pwd -P)
CODE=$HERE/code_train
PY=$(command -v "${WMM_PYTHON:-python}")
export PATH=$(dirname "$PY"):$PATH

IFS=',' read -ra GPUS <<< "$GPU"
[ ${#GPUS[@]} -eq 1 ] || { echo "use exactly one GPU: the paper checkpoint was trained on one GPU with batch 16 (and DDP + torch.compile fails in torch 2.3.1)" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i "$GPU"

# Paths in the config (./data_wmm/omomo_v3/...) are relative to the code root.
ln -sfn "$DATA/L3_train/data_wmm" "$CODE/data_wmm"
# Inputs of the FID/R_prec/contact scorer used by the evaluation every 10k steps (shared with Layer 2).
mkdir -p "$CODE/data/omomo/artifacts"
for n in chois_test_cache stats smplh_amass omomo_text_anno; do
  ln -sfn "$DATA/L2_eval/classifier_data/$n" "$CODE/data/omomo/$n"
done
for n in rest_object_geo rest_object_sdf smplx test_npz; do
  ln -sfn "$DATA/L2_eval/data_wmm/omomo_v3/$n" "$CODE/data/omomo/$n"
done
ln -sfn "$DATA/L2_eval/data_wmm/omomo_v3/artifacts/glove_vocab" "$CODE/data/omomo/artifacts/glove_vocab"
export PYTHONPATH=$CODE/src:$CODE:$CODE/ext/lib_mani:$CODE/ext/lib_io:$CODE/ext/pytorch_kinematics/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=$DATA/L2_eval/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The model is compiled with torch.compile. Its kernel cache goes to a fresh node-local folder per run and kernels are
# compiled in one process: a shared cache (/tmp/torchinductor_$USER) and parallel compile workers both can make the
# first step fail with FileNotFoundError on a *.tmp.pid_* cache file.
CACHE=${WMM_COMPILE_CACHE:-$(mktemp -d /tmp/omomo_l3_compile.XXXXXX)}
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_COMPILE_THREADS=1
echo "compile cache: $CACHE"

cd "$CODE"
"$PY" train.py --config configs/wmm/om/w1.8.1.1.yaml --workers 12 --viz_try \
  --log_dir "$DATA/L3_outputs" --max_steps 164000 "$@" &
TRAIN=$!
trap 'kill -INT $TRAIN 2>/dev/null' INT TERM
# The trainer rewrites checkpoints/last.ckpt in place at every save and deletes it a few seconds after the last step.
# A hard link follows every rewrite and keeps the final file.
KEEP=$DATA/L3_outputs/final
mkdir -p "$KEEP"
while kill -0 $TRAIN 2>/dev/null; do
  last=$(ls -t "$DATA"/L3_outputs/w1.8.1.1_omomo/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
  [ -z "$last" ] || ln -f "$last" "$KEEP/last.ckpt" 2>/dev/null || true
  sleep 2
done
wait $TRAIN
