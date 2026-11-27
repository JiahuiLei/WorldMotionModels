#!/usr/bin/env bash
# Layer 3: retrain one TraceGen WMM checkpoint of the paper from the released training data.
#
#   bash apps/tracegen/L3_train.sh <run> [gpus]      e.g. bash apps/tracegen/L3_train.sh droid_k128 0
#
# <run>            table row                      step of the paper checkpoint (MAX_STEPS default)
# droid_k128       per-dataset P=128, Droid        50000
# epic_k128        per-dataset P=128, EpicKitchen  30000
# droid_k400       per-dataset P=400, Droid        80000
# epic_k400        per-dataset P=400, EpicKitchen  60000
# droidepic_k128   Joint P=128 (both columns)      46000 (the file is named 45k.ckpt; its global_step is 46000)
# droidepic_k400   Joint P=400 (both columns)      68000
#
# Training stops at MAX_STEPS. The trainer writes `last.ckpt` every 2,000 steps, so at the end `last.ckpt` holds exactly
# step MAX_STEPS: that is the counterpart of the paper checkpoint for every row. The `epoch=…_step=…ckpt` file next to it
# is the in-training-MAE best; for the two joint rows the paper did not use the best (see README, Layer 3).
#
# The paper runs trained on one 48 GB GPU each with the config's per-GPU batch (12 for P=128, 4 for P=400).
# Listing several GPUs trains data-parallel and multiplies the global batch.
#
# Environment:
#   WMM_DATA      release data root (default: <repo>/release_data)
#   WMM_PYTHON    python of the `umm` conda env (default: python on PATH)
#   MAX_STEPS     training steps (default: the paper checkpoint's step, table above)
#   RUN_NAME      experiment name; logs and checkpoints go to L3_outputs/<RUN_NAME>/ (default: w1.6.0_<run>_L3)
# Extra arguments after the GPU list are passed to train.py. To resume a stopped run, pass the absolute path of its
# checkpoint, e.g. `--resume $WMM_DATA/tracegen/L3_outputs/w1.6.0_droid_k128_L3/version_0/checkpoints/last.ckpt`.
# Do not pass a bare version number: train.py then looks under ./logs, not under L3_outputs, and silently starts over.
set -euo pipefail

RUN="${1:?usage: L3_train.sh <run> [gpus]}"
GPUS="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
shift $(( $# >= 2 ? 2 : 1 ))
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
CODE="$APP/code_train"
OUT="$DATA/L3_outputs"

export CUDA_VISIBLE_DEVICES="$GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# T5 text encoder and DINOv3 image encoder from checkpoints/ (prepare_data/download_weights.py), offline;
# Weights & Biases logs stay local.
export HF_HOME="$DATA/checkpoints/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export WMM_DINOV3_REPO="$CODE/ext/dinov3" WMM_DINOV3_CKPT_DIR="$DATA/checkpoints/dinov3"
# torch.compile is on. A private kernel cache per run keeps other jobs on the machine from deleting its kernels, and
# compiling in-process (no compile worker pool) avoids compile-worker crashes on busy machines.
CACHE="${TMPDIR:-/tmp}/tracegen_${RUN}_$$"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# Put code_train first on the Python path so that no other installed copy of `umm` is imported.
# umm.datasets.wmm imports lib_mani, which imports pytorch_kinematics.
export PYTHONPATH="$CODE/src:$CODE:$CODE/ext/lib_io:$CODE/ext/lib_mani:$CODE/ext/pytorch_kinematics/src${PYTHONPATH:+:$PYTHONPATH}"

# The trainer rewrites checkpoints/last.ckpt in place and deletes it when fit ends; after a resume it writes
# last-v1.ckpt instead. While training runs, keep a hard link ckpt_keep/last_link.ckpt to the newest last*.ckpt every
# 3 s, so the final-step checkpoint survives (L3_eval.sh reads it).
EXP="${RUN_NAME:-w1.6.0_${RUN}_L3}"
KEEP="$OUT/$EXP/ckpt_keep"
mkdir -p "$KEEP"
(
  set +e +o pipefail  # before the first save `ls` finds nothing; that must not end the loop
  while true; do
    ck=$(ls -t "$OUT/$EXP"/version_*/checkpoints/last*.ckpt 2>/dev/null | head -1)
    if [ -n "$ck" ] && { [ ! -e "$KEEP/last_link.ckpt" ] || [ "$ck" -nt "$KEEP/last_link.ckpt" ]; }; then
      ln -f "$ck" "$KEEP/last_link.ckpt"
    fi
    sleep 3
  done
) &
LINKER=$!

# The config names ./data_wmm/... relative to the working directory.
cd "$DATA/L3_train"
rc=0
"$PY" "$CODE/train.py" \
  --config "$CODE/configs/wmm/tg/w1.6.0_${RUN}_tpsqspt.yaml" \
  --expname "$EXP" \
  --log_dir "$OUT" \
  --workers 12 \
  --max_steps "${MAX_STEPS:-$STEPS}" \
  "$@" || rc=$?
kill "$LINKER" 2>/dev/null || true
exit "$rc"
