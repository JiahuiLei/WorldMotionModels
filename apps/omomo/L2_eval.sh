#!/usr/bin/env bash
# Layer 2: run the paper checkpoint on the OMOMO test set and score it with the table's scorer.
#
#   apps/omomo/L2_eval.sh ours <gpu>         # all 482 test windows, M7-strict, best of K=20 (about 5.5 h on 1 GPU)
#   apps/omomo/L2_eval.sh ours_smoke <gpu>   # the first 9 test windows only (a few minutes), same protocol
#
# Outputs go to $WMM_DATA/omomo/L2_outputs/<row>/ (default WMM_DATA = <repo>/release_data).
# WMM_PYTHON selects the interpreter of the `umm` conda environment (default: `python` on PATH).
# WMM_CKPT evaluates another checkpoint (e.g. one from L3_train.sh); L2_OUT_NAME names its output folder.
set -euo pipefail

ROW=${1:?usage: L2_eval.sh ours|ours_smoke <gpu>}
GPU=${2:-${CUDA_VISIBLE_DEVICES:-0}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DATA=$(cd "${WMM_DATA:-$REPO/release_data}/omomo" && pwd -P)
CODE=$HERE/code
OUT=$DATA/L2_outputs/${L2_OUT_NAME:-$ROW}
WMM_PY=$(command -v "${WMM_PYTHON:-python}")
# The scorer starts its classifier helpers as `python3`; they must come from the same environment.
export PATH=$(dirname "$WMM_PY"):$PATH

export CUDA_VISIBLE_DEVICES=$GPU
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i "$GPU"

# The evaluation code reads its inputs through <code>/data_wmm and <code>/data/omomo (paths relative to the code root).
ln -sfn "$DATA/L2_eval/data_wmm" "$CODE/data_wmm"
mkdir -p "$CODE/data/omomo/artifacts"
for n in chois_test_cache stats smplh_amass omomo_text_anno; do
  ln -sfn "$DATA/L2_eval/classifier_data/$n" "$CODE/data/omomo/$n"
done
for n in rest_object_geo rest_object_sdf smplx test_npz; do
  ln -sfn "$DATA/L2_eval/data_wmm/omomo_v3/$n" "$CODE/data/omomo/$n"
done
ln -sfn "$DATA/L2_eval/data_wmm/omomo_v3/artifacts/glove_vocab" "$CODE/data/omomo/artifacts/glove_vocab"
export HF_HOME=$DATA/L2_eval/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$CODE/src:$CODE:$CODE/ext/lib_mani:$CODE/ext/lib_io:$CODE/ext/pytorch_kinematics/src${PYTHONPATH:+:$PYTHONPATH}

CKPT=${WMM_CKPT:-$DATA/checkpoints/w1.8.1.1_omomo_165k.ckpt}
cd "$CODE"
case $ROW in
  ours)
    "$WMM_PY" eval/omomo/eval_and_save_wmm.py --checkpoint "$CKPT" --config configs/wmm/om/w1.8.1.1.yaml \
      --output "$OUT" --mode full --batch_size 12 --protocol m7 --m7_no_fuzzy
    ;;
  ours_smoke)
    # One shard of 54 = the first 9 windows; a sharded run writes predictions only, so the scorer is called here.
    "$WMM_PY" eval/omomo/eval_and_save_wmm.py --checkpoint "$CKPT" --config configs/wmm/om/w1.8.1.1.yaml \
      --output "$OUT" --mode full --batch_size 12 --protocol m7 --m7_no_fuzzy --num_shards 54 --shard_idx 0
    ;;
  *)
    echo "unknown row: $ROW" >&2
    exit 2
    ;;
esac
# The evaluation script appends _<timestamp> (and _shard<i>of<N>) to --output; point L2_outputs/<row> at the newest run.
RUN=$(ls -d "${OUT}"_* | sort | tail -1)
ln -sfn "$(basename "$RUN")" "$OUT"
if [ "$ROW" = ours_smoke ]; then
  mkdir -p "$RUN/gt_subset"
  for p in "$RUN"/preds_all/*.npz; do ln -sfn "$DATA/L2_eval/data_wmm/omomo_v3/test_npz/$(basename "$p")" "$RUN/gt_subset/"; done
  # FID and R_prec need many windows; the smoke row scores the 13 table columns only.
  "$WMM_PY" eval/omomo/eval_chois_table1.py --preds_dir "$RUN/preds_all" --out "$RUN/results" --label ours_smoke \
    --gpu 0 --gt_dir "$RUN/gt_subset" --smplx_dir data_wmm/omomo_v3/smplx \
    --sdf_folder data_wmm/omomo_v3/rest_object_sdf --obj_geo_root . --skip_fid
fi
