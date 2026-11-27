#!/usr/bin/env bash
# Layer 2 baseline: evaluate the TraceGen authors' released checkpoint with the authors' own
# evaluation code (baseline/TraceGen, upstream commit 728621eb) on the full test split.
#
#   bash apps/tracegen/L2_eval_baseline.sh <GPU> <droid|epickitchen>
#
# Evaluation settings for the authors' released checkpoints: decoder with 6 layers, 12 heads, width 768; batch 8.
# Output: $WMM_DATA/tracegen/L2_outputs/tracegen_authors_<env>_<timestamp>/eval.log
set -euo pipefail

GPU="${1:?usage: L2_eval_baseline.sh <GPU> <droid|epickitchen>}"
ENV_LOWER="${2:?usage: L2_eval_baseline.sh <GPU> <droid|epickitchen>}"
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${WMM_DATA:-$(cd "$APP/../.." && pwd)/release_data}/tracegen"
PY="${TRACEGEN_PYTHON:-python}"
REPO="$APP/baseline/TraceGen"

case "$ENV_LOWER" in
  droid) ENV_UPPER=Droid; TEST="$DATA/L2_eval/data_wmm/tracegen/data/Droid/droid_test" ;;
  epickitchen) ENV_UPPER=EpicKitchen; TEST="$DATA/L2_eval/data_wmm/tracegen/data/Epickitchen/epic_test" ;;
  *) echo "unknown env: $ENV_LOWER" >&2; exit 2 ;;
esac
CKPT="$DATA/checkpoints/tracegen_authors/$ENV_UPPER/tracegen_${ENV_LOWER}.pth"
RUN="$DATA/L2_outputs/tracegen_authors_${ENV_LOWER}_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN/cache" "$RUN/results"

cp "$REPO/cfg/train.yaml" "$RUN/train_${ENV_LOWER}.yaml"
cat > "$RUN/train_${ENV_LOWER}.local.yaml" <<EOF
data:
  dataset_dirs:
    - $TEST
  cache_dir: $RUN/cache
  num_workers: 4
logging:
  use_wandb: false
  checkpoint_dir: $RUN/results/
test_path: $TEST
EOF

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$DATA/checkpoints/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  run=$RUN"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$GPU"

LOG="$RUN/eval.log"
# Record why the evaluation ends: exit code, or the signal that stopped it.
trap 'echo "=== $(date -u +%FT%TZ) launcher received SIGTERM ===" >> "$LOG"' TERM
trap 'echo "=== $(date -u +%FT%TZ) launcher received SIGHUP ===" >> "$LOG"' HUP
cd "$REPO"
echo "=== $(date -u +%FT%TZ) BEGIN eval: $ENV_LOWER  checkpoint: $CKPT ===" >> "$LOG"
rc=0
PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 "$PY" test_benchmark.py \
  --config "$RUN/train_${ENV_LOWER}.yaml" \
  --override \
    train.batch_size=8 \
    train.lr_decoder=1.5e-4 \
    model.decoder.num_layers=6 \
    model.decoder.num_attention_heads=12 \
    model.decoder.latent_dim=768 \
    data.num_workers=4 \
    hardware.mixed_precision=true \
    logging.use_wandb=false \
    logging.log_every=2000 \
  --resume "$CKPT" >> "$LOG" 2>&1 || rc=$?
sig=""
if [ "$rc" -gt 128 ]; then sig=" signal=SIG$(kill -l $((rc - 128)))"; fi
echo "=== $(date -u +%FT%TZ) END eval: $ENV_LOWER rc=$rc$sig ===" | tee -a "$LOG"
exit "$rc"
