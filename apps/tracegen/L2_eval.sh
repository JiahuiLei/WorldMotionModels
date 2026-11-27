#!/usr/bin/env bash
# Layer 2: run WMM inference again on the full TraceGen test splits.
#
#   bash apps/tracegen/L2_eval.sh <GPU>            # worker: takes the next unclaimed run, repeats until none left
#   bash apps/tracegen/L2_eval.sh <GPU> <run>      # one run only
#
# Start one worker per free GPU; workers share the queue through claim
# directories, so adding a GPU later just means starting another worker.
# Outputs: $WMM_DATA/tracegen/L2_outputs/<run>_<timestamp>/ (same layout as the original evaluation).
# EVAL_EFFORT=<fraction in (0, 1]> evaluates only that share of the test split (quick check; default 1.0 = full split).
set -euo pipefail

GPU="${1:?usage: L2_eval.sh <GPU> [run]}"
ONLY="${2:-}"
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${WMM_DATA:-$(cd "$APP/../.." && pwd)/release_data}/tracegen"
PY="${WMM_PYTHON:-python}"
CODE="$APP/code"
OUT="$DATA/L2_outputs"
CLAIMS="$OUT/.claims"

# run name | checkpoint (under $DATA/checkpoints) | config | env
RUNS=(
  "droid_k128_50k_droid|w1.6.0_droid_k128_v9_50k/50k.ckpt|w1.6.0_droid_k128_tpsqspt.yaml|droid"
  "epic_k128_30k_epic|w1.6.0_epic_k128_v10_30k/30k.ckpt|w1.6.0_epic_k128_tpsqspt.yaml|epic"
  "droid_k400_80k_droid|w1.6.0_droid_k400_v4_80k/80k.ckpt|w1.6.0_droid_k400_tpsqspt.yaml|droid"
  "epic_k400_60k_epic|w1.6.0_epic_k400_v9_60k/60k.ckpt|w1.6.0_epic_k400_tpsqspt.yaml|epic"
  "droidepic_k128_45k_droid|w1.6.0_droidepic_k128_v9_45k/45k.ckpt|w1.6.0_droidepic_k128_tpsqspt.yaml|droid"
  "droidepic_k128_45k_epic|w1.6.0_droidepic_k128_v9_45k/45k.ckpt|w1.6.0_droidepic_k128_tpsqspt.yaml|epic"
  "droidepic_k400_68k_droid|w1.6.0_droidepic_k400_v10_68k/68k.ckpt|w1.6.0_droidepic_k400_tpsqspt.yaml|droid"
  "droidepic_k400_68k_epic|w1.6.0_droidepic_k400_v10_68k/68k.ckpt|w1.6.0_droidepic_k400_tpsqspt.yaml|epic"
)

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$DATA/checkpoints/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WMM_DINOV3_REPO="$CODE/ext/dinov3" WMM_DINOV3_CKPT_DIR="$DATA/checkpoints/dinov3"
# Put the released code first on the Python path so that no other installed copy of `umm` is imported.
# umm.datasets.wmm (loaded by the run) imports lib_mani, which imports pytorch_kinematics.
export PYTHONPATH="$CODE/src:$CODE:$CODE/ext/lib_io:$CODE/ext/lib_mani:$CODE/ext/pytorch_kinematics/src${PYTHONPATH:+:$PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  data=$DATA"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$GPU"
mkdir -p "$CLAIMS"

run_one() {
  local name="$1" ckpt="$2" cfg="$3" env="$4"
  echo "[$(date -u +%FT%TZ)] START $name on GPU $GPU"
  # The config reads test data from ./data_wmm/tracegen/data, relative to the working directory.
  (cd "$DATA/L2_eval" && "$PY" "$CODE/eval/tracegen/eval_and_save_wmm.py" \
    --checkpoint "$DATA/checkpoints/$ckpt" \
    --config "$CODE/configs/wmm/tg/$cfg" \
    --output "$OUT/$name" \
    --env "$env" --eval_effort "${EVAL_EFFORT:-1.0}" --seed 42 --gen_timesteps 50 \
    --batch_size 8 --num_workers 8)
  echo "[$(date -u +%FT%TZ)] DONE $name"
}

for entry in "${RUNS[@]}"; do
  IFS='|' read -r name ckpt cfg env <<<"$entry"
  if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then continue; fi
  mkdir "$CLAIMS/$name" 2>/dev/null || { [[ -n "$ONLY" ]] && { echo "$name already claimed" >&2; exit 1; }; continue; }
  echo "GPU $GPU" > "$CLAIMS/$name/owner"
  run_one "$name" "$ckpt" "$cfg" "$env"
done
