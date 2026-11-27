#!/usr/bin/env bash
# Layer 2: run one Language Table evaluation from the released checkpoint.
#
#   apps/langtable/L2_eval.sh plain [gpu]        WMM as a plain policy (Tab. 1 WMM row, Tab. 2 Plain)   1 GPU
#   apps/langtable/L2_eval.sh mpc   [gpu,gpu,..] WMM inside MPC (Tab. 2 MPC), one shard per listed GPU
#   apps/langtable/L2_eval.sh orig               public LangTable BC+ResNet checkpoint with the original
#                                                language_table/eval/main.py (baseline row), CPU only
#
# Environment:
#   WMM_DATA        release data root (default: <repo>/release_data)
#   WMM_LT_PYTHON   python of the `wmm-lt` conda env, used by plain and mpc (default: python on PATH)
#   WMM_LT_ORIG_PYTHON python of the `lt-orig` conda env, used by orig (default: python on PATH)
#   WMM_LT_CKPT     checkpoint to evaluate (default: the released EMA-only step-180000 checkpoint)
#   RUN_NAME        output folder name under L2_outputs/ (default: the row name)
#   WMM_LT_FAMILIES comma-separated task families to run (default: all five)
#   WMM_LT_SMOKE_EPISODES  run only the first N episodes per family (quick setup check, not a table result)
# Outputs: $WMM_DATA/langtable/L2_outputs/$RUN_NAME/ (must not exist yet).
set -euo pipefail

ROW=${1:?usage: L2_eval.sh plain|mpc|orig [gpus]}
GPUS=${2:-${CUDA_VISIBLE_DEVICES:-0}}
APP=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$APP/../.." && pwd)
DATA=${WMM_DATA:-$ROOT/release_data}/langtable
PY=${WMM_LT_PYTHON:-python}
CKPT=${WMM_LT_CKPT:-$DATA/checkpoints/w1.0.3_b8all_step180000_ema.ckpt}
OUT=$DATA/L2_outputs/${RUN_NAME:-$ROW}

export HF_HOME=$DATA/L2_eval/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The training config names the robot and block meshes as ./data_wmm/..., relative to this folder.
cd "$DATA/L2_eval"
mkdir -p "$DATA/L2_outputs"

case $ROW in
  plain)
    export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    CMD=("$APP/code/plain/eval/langtable_official/eval_and_save_wmm.py")
    if [ -n "${WMM_LT_FAMILIES:-}${WMM_LT_SMOKE_EPISODES:-}" ]; then
      CMD=("$APP/eval_plain_family.py" --families "${WMM_LT_FAMILIES:-blocktoblock,blocktoabsolutelocation,blocktoblockrelativelocation,blocktorelativelocation,separate}")
      [ -z "${WMM_LT_SMOKE_EPISODES:-}" ] || CMD+=(--smoke_episodes "$WMM_LT_SMOKE_EPISODES")
    fi
    "$PY" "${CMD[@]}" \
      --checkpoint "$CKPT" \
      --config "$APP/code/plain/configs/wmm/lt/w1.0.3_b8all.yaml" \
      --output "$OUT"
    ;;
  mpc)
    NSHARDS=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
    unset CUDA_VISIBLE_DEVICES
    echo "MPC shards: $NSHARDS on physical GPUs $GPUS"
    "$PY" "$APP/code/mpc/eval/langtable_official/eval_protocol_a_wmm_mpc.py" \
      --checkpoint "$CKPT" \
      --config "$APP/code/mpc/configs/wmm/lt/w1.0.3_b8all.yaml" \
      --output "$OUT" \
      --policy_mode mpc --num_rollouts "${WMM_LT_SMOKE_EPISODES:-50}" --env_seed 0 \
      --reward_filter "${WMM_LT_FAMILIES:-}" \
      --hist 5 --window 20 --act_horizon 10 --max_steps 200 \
      --K 4 --noise_std 0.02 --lr_schedule 1.0x4,0.5x4,0.1x4 \
      --n_grad_steps 10 --n_steps 10 --save_intermediates 1 \
      --num_shards "$NSHARDS" --gpus "$GPUS"
    ;;
  orig)
    export CUDA_VISIBLE_DEVICES=""
    [ ! -e "$OUT" ] || { echo "$OUT exists; pick another RUN_NAME" >&2; exit 2; }
    mkdir -p "$OUT"
    cd "$APP/code/langtable_orig"
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      "${WMM_LT_ORIG_PYTHON:-python}" language_table/eval/main.py \
      --config=language_table/train/configs/language_table_resnet_sim_local.py \
      --checkpoint_path="$DATA/checkpoints/bc_resnet_sim_checkpoint_955000" \
      --workdir="$OUT" 2>&1 | tee "$OUT/eval.log"
    echo "done: $OUT"
    exit 0
    ;;
  *)
    echo "unknown row: $ROW (expected plain, mpc or orig)" >&2
    exit 2
    ;;
esac

# Both evaluation scripts first copy the checkpoint into the output folder; drop that duplicate.
rm -f "$OUT/$(basename "$CKPT")"
echo "done: $OUT"
