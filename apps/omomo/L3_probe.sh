#!/usr/bin/env bash
# Layer 3 probe: evaluate an intermediate checkpoint of your retraining run on all 482 test windows,
# exactly like `L2_eval.sh ours` (M7-strict, best of K=20), and print the 13 table columns next to the paper.
#
#   apps/omomo/L3_probe.sh <gpu> <file.ckpt> <name>   # e.g. release_data/omomo/L3_outputs/final/last.ckpt 164000
#
# About 5.5 h on one RTX A6000. Output: $WMM_DATA/omomo/L2_outputs/probe_step<name>/.
set -euo pipefail

GPU=${1:?usage: L3_probe.sh <gpu> <file.ckpt> <name>}
SRC=$(realpath "${2:?usage: L3_probe.sh <gpu> <file.ckpt> <name>}")
STEP=${3:?give a name for the checkpoint, e.g. its training step}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DATA=$(cd "${WMM_DATA:-$REPO/release_data}/omomo" && pwd -P)
PY=$(command -v "${WMM_PYTHON:-python}")

[ -f "$SRC" ] || { echo "no checkpoint $SRC" >&2; exit 1; }
PROBE=$DATA/L3_outputs/probes/step$STEP
mkdir -p "$PROBE"
# A hard link (or copy) pins this checkpoint even if the trainer later rewrites the original file.
ln -f "$SRC" "$PROBE/model.ckpt" 2>/dev/null || cp "$SRC" "$PROBE/model.ckpt"
echo "probing step $STEP ($SRC) on GPU $GPU"

WMM_CKPT=$PROBE/model.ckpt L2_OUT_NAME=probe_step$STEP WMM_PYTHON=$PY "$HERE/L2_eval.sh" ours "$GPU"

"$PY" - "$HERE" "$DATA/L2_outputs/probe_step$STEP/results" "$STEP" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import omomo_l1 as L
probe = L.table_row(L.load_per_window(Path(sys.argv[2])))
paper = L.PAPER_TABLE['WMM (ours)']
print(f'step {sys.argv[3]}: column, paper (step 164000), this checkpoint')
for c, p in zip(L.COLUMNS, paper):
    print(f'{c:7s} {p:8.2f} {probe[c]:8.3f}')
PYEOF
