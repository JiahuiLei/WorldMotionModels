#!/usr/bin/env bash
# Build every OMOMO input of Layers 2 and 3 from the official downloads (see the README, "Data you build").
#
#   apps/omomo/prepare_data/build_all.sh <gpu>
#
# Needs, under $WMM_DATA/omomo (default WMM_DATA = <repo>/release_data):
#   downloads/omomo_raw.tar.gz, downloads/omomo_text_anno.zip      (prepare_data/download.sh omomo text)
#   body_models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz              (SMPL-X, you download it)
#   body_models/smplh_amass/{male,female,neutral}/model.npz        (SMPL-H as used by AMASS, you download it)
# WMM_PYTHON / CHOIS_PYTHON: interpreters of the `umm` and `chois_repro` environments.
# About 1.5 h (the training windows take most of it) and 30 GB of temporary space under $WMM_DATA/omomo/build.
set -euo pipefail

GPU=${1:?usage: build_all.sh <gpu>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP=$(cd "$HERE/.." && pwd)
REPO=$(cd "$APP/../.." && pwd)
mkdir -p "${WMM_DATA:-$REPO/release_data}/omomo"
DATA=$(cd "${WMM_DATA:-$REPO/release_data}/omomo" && pwd -P)
WMM_PY=$(command -v "${WMM_PYTHON:-python}")
CHOIS_PY=$(command -v "${CHOIS_PYTHON:?set CHOIS_PYTHON to the python of the chois_repro environment}")
BUILD=$DATA/build
L2=$DATA/L2_eval
V3=$L2/data_wmm/omomo_v3
BM=$DATA/body_models
export CUDA_VISIBLE_DEVICES=$GPU
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for f in downloads/omomo_raw.tar.gz downloads/omomo_text_anno.zip body_models/smplx/SMPLX_MALE.npz \
         body_models/smplx/SMPLX_FEMALE.npz body_models/smplx/SMPLX_NEUTRAL.npz \
         body_models/smplh_amass/male/model.npz body_models/smplh_amass/female/model.npz \
         body_models/smplh_amass/neutral/model.npz; do
  [ -f "$DATA/$f" ] || { echo "missing $DATA/$f (see the README)" >&2; exit 1; }
done

echo "== 1/7 unpack the OMOMO archive"
mkdir -p "$BUILD/omomo"
[ -f "$BUILD/omomo/data/train_diffusion_manip_window_120_cano_joints24.p" ] || tar xzf "$DATA/downloads/omomo_raw.tar.gz" -C "$BUILD/omomo"
OM=$BUILD/omomo/data

echo "== 2/7 text annotations"
"$WMM_PY" "$HERE/build_text_anno.py" --zip "$DATA/downloads/omomo_text_anno.zip" --out "$V3/omomo_text_anno"

echo "== 3/7 CHOIS data code: object rest poses, canonical test windows, statistics"
CH=$BUILD/chois
mkdir -p "$CH/smpl_all_models"
for f in train_diffusion_manip_seq_joints24.p test_diffusion_manip_seq_joints24.p \
         min_max_mean_std_data_window_120_cano_joints24.p captured_objects; do
  ln -sfn "$OM/$f" "$CH/$f"
done
ln -sfn "$BM/smplx" "$CH/smpl_all_models/smplx"
ln -sfn "$BM/smplh_amass" "$CH/smpl_all_models/smplh_amass"
"$CHOIS_PY" "$HERE/build_chois_cache.py" --chois_root "$CH"
mkdir -p "$V3/rest_object_geo" "$L2/classifier_data/chois_test_cache" "$L2/classifier_data/stats"
cp "$CH"/rest_object_geo/* "$V3/rest_object_geo/"
cp "$CH/cano_test_diffusion_manip_window_120_joints24.p" "$L2/classifier_data/chois_test_cache/"
cp "$CH/cano_min_max_mean_std_data_window_120_joints24.p" "$L2/classifier_data/stats/"

echo "== 4/7 body models and object SDFs"
ln -sfn "$BM/smplx" "$V3/smplx"
ln -sfn "$BM/smplh_amass" "$L2/classifier_data/smplh_amass"
ln -sfn ../data_wmm/omomo_v3/omomo_text_anno "$L2/classifier_data/omomo_text_anno"
ln -sfn omomo_v3 "$L2/data_wmm/omomo_v2"
"$WMM_PY" "$HERE/build_rest_object_sdf.py" --omomo_dir "$OM" --out "$V3/rest_object_sdf"
(cd "$L2" && "$WMM_PY" "$HERE/bake_omomo_sdf_v2.py" --ply_dir data_wmm/omomo_v2/rest_object_geo \
  --out_dir data_wmm/omomo_v3/rest_object_sdf_v2 --num_workers 13)

echo "== 5/7 test and training windows"
ARGS=(--omomo_dir "$OM" --text_dir "$V3/omomo_text_anno" --smplh_dir "$BM/smplh_amass" --smplx_dir "$BM/smplx"
      --rest_object_geo "$V3/rest_object_geo" --out_dir "$V3")
cd "$APP/code_train"
"$WMM_PY" -m prepare_data.omomo.convert --split test --chois_test_cache "$L2/classifier_data/chois_test_cache/cano_test_diffusion_manip_window_120_joints24.p" "${ARGS[@]}"
NS=${CONVERT_SHARDS:-8}
for ((s = 0; s < NS; s++)); do
  "$WMM_PY" -m prepare_data.omomo.convert --split train --shard_idx $s --num_shards $NS --progress_every 1000 "${ARGS[@]}" &
done
wait
"$WMM_PY" -m prepare_data.omomo.convert --split train --skip_existing "${ARGS[@]}"
"$WMM_PY" -m prepare_data.omomo.convert --write_splits --out_dir "$V3"

echo "== 6/7 Layer-3 layout"
T3=$DATA/L3_train/data_wmm/omomo_v3
mkdir -p "$T3"
rm -rf "$T3/train_npz"
mv "$V3/train_npz" "$T3/train_npz"
for n in artifacts omomo_text_anno rest_object_geo rest_object_sdf rest_object_sdf_v2 smplx splits.json test_npz; do
  ln -sfn "../../../L2_eval/data_wmm/omomo_v3/$n" "$T3/$n"
done
ln -sfn omomo_v3 "$DATA/L3_train/data_wmm/omomo_v2"

echo "== 7/7 check"
"$WMM_PY" -m prepare_data.omomo.validate_schema --npz_dir "$V3/test_npz"
"$WMM_PY" -m prepare_data.omomo.validate_schema --npz_dir "$T3/train_npz"
echo "done; the temporary folder $BUILD can be deleted"
