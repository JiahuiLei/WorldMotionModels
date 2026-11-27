#!/usr/bin/env bash
# Download the public third-party files the Language Table evaluation needs and check each one against the
# sha256 of the copy the paper evaluation used (prepare_data/SHA256SUMS).
#
#   apps/langtable/prepare_data/download_third_party.sh
#
# Downloads (into $WMM_DATA/langtable/):
#   checkpoints/bc_resnet_sim_checkpoint_955000  Language Table BC+ResNet checkpoint (Lynch et al. 2022), from the
#                                                public bucket gs://gresearch/robotics/language_table_checkpoints/
#   L2_eval/hf_home/hub/models--t5-base/         T5-base tokenizer and encoder (Hugging Face `t5-base`, revision
#                                                a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1), in Hugging Face cache layout
#
# Environment:
#   WMM_DATA        release data root (default: <repo>/release_data)
#   WMM_LT_PYTHON   python with huggingface_hub, e.g. the `wmm-lt` conda env (default: python on PATH)
set -euo pipefail

APP=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$APP/../.." && pwd)
DATA=${WMM_DATA:-$ROOT/release_data}/langtable
PY=${WMM_LT_PYTHON:-python}
BC_URL=https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000
T5_REVISION=a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1

mkdir -p "$DATA/checkpoints"
BC=$DATA/checkpoints/bc_resnet_sim_checkpoint_955000
if [ ! -f "$BC" ]; then
  curl -fL --retry 10 --retry-all-errors -C - -o "$BC.part" "$BC_URL"
  mv "$BC.part" "$BC"
fi

HF_HOME=$DATA/L2_eval/hf_home HF_HUB_DISABLE_XET=1 "$PY" - "$T5_REVISION" <<'EOF'
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

revision = sys.argv[1]
cache = Path(os.environ['HF_HOME']) / 'hub'
for name in ['config.json', 'model.safetensors', 'spiece.model', 'tokenizer.json']:
    hf_hub_download('t5-base', name, revision=revision, cache_dir=cache)
# Offline loading by name reads the revision from refs/main.
refs = cache / 'models--t5-base' / 'refs'
refs.mkdir(parents=True, exist_ok=True)
(refs / 'main').write_text(revision)
EOF

cd "$DATA"
sha256sum -c "$APP/prepare_data/SHA256SUMS"
echo "done: $DATA"
