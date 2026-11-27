#!/usr/bin/env bash
# Download the publicly available third-party files of the OMOMO experiment and check their sha256.
#
#   apps/omomo/prepare_data/download.sh [what...]      # what: text t5 omomo (default: text t5)
#
#   text   OMOMO text annotations, omomo_text_anno.zip (1.4 MB) from the OMOMO repository at a fixed commit
#          -> $WMM_DATA/omomo/downloads/omomo_text_anno.zip
#   t5     the frozen T5-base text encoder (0.9 GB) from Hugging Face at a fixed revision
#          -> $WMM_DATA/omomo/L2_eval/hf_cache (a Hugging Face cache folder, used as HF_HOME)
#   omomo  the OMOMO dataset archive (22,177,103,436 B) from the authors' Google Drive link (needs `pip install
#          gdown`) -> $WMM_DATA/omomo/downloads/omomo_raw.tar.gz. Read the dataset's terms first: it contains SMPL-H
#          body parameters, so you must also hold an SMPL-H/SMPL-X licence (see the README).
#          Broken connections are resumed (up to OMOMO_TRIES attempts, default 30, waiting 30 s .. 10 min between
#          them); running the script again also resumes. Exit code 3: Google Drive's download quota for the file is
#          used up; try again later or download it in a browser.
#
# Run in the `umm` environment (it provides huggingface_hub). Default WMM_DATA = <repo>/release_data.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
DATA=${WMM_DATA:-$REPO/release_data}/omomo
DL=$DATA/downloads
mkdir -p "$DL"

OMOMO_REPO_COMMIT=e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7
TEXT_SHA256=2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c
T5_REVISION=a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
OMOMO_DRIVE_ID=1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm
OMOMO_SHA256=c2de0244240d0650de69b5d82f9e2159aa8ee09b36555d9754fe9f51c5af5061
OMOMO_SIZE=22177103436

check() {  # file sha256
  echo "$2  $1" | sha256sum -c - || { echo "sha256 mismatch: $1" >&2; exit 1; }
}

fetch_text() {
  curl -fL --retry 3 -o "$DL/omomo_text_anno.zip" \
    "https://raw.githubusercontent.com/lijiaman/omomo_release/$OMOMO_REPO_COMMIT/omomo_text_anno.zip"
  check "$DL/omomo_text_anno.zip" "$TEXT_SHA256"
}

fetch_t5() {
  local cache=$DATA/L2_eval/hf_cache/hub PY
  PY=$(command -v "${WMM_PYTHON:-python}") || { echo "python not found (activate the umm environment)" >&2; exit 1; }
  HF_HUB_OFFLINE=0 "$PY" - "$cache" "$T5_REVISION" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
cache, rev = sys.argv[1], sys.argv[2]
for name in ("config.json", "tokenizer.json", "spiece.model", "model.safetensors"):
    print(hf_hub_download("t5-base", name, revision=rev, cache_dir=cache))
# The model code loads "t5-base" offline; refs/main tells the cache which snapshot that is.
ref = Path(cache, "models--t5-base", "refs", "main")
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(rev)
PYEOF
  local snap=$cache/models--t5-base/snapshots/$T5_REVISION
  check "$snap/config.json" 46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033
  check "$snap/tokenizer.json" d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066
  check "$snap/spiece.model" d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86
  check "$snap/model.safetensors" a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4
}

# gdown --continue resumes from the single <output>*.part file and skips the download whenever <output> exists, even
# when a broken connection left it short. A short <output> therefore becomes the resume file again.
omomo_resume_file() {  # out
  local out=$1 have part psize
  have=$(stat -c %s "$out")
  if [ "$have" -gt "$OMOMO_SIZE" ]; then
    echo "$out is larger than the archive ($have > $OMOMO_SIZE bytes); delete it and run again" >&2
    exit 1
  fi
  echo "$(basename "$out") has $have of $OMOMO_SIZE bytes: resuming" >&2
  part=$(find "$(dirname "$out")" -maxdepth 1 -name "$(basename "$out")*.part" | head -1)
  if [ -n "$part" ]; then
    psize=$(stat -c %s "$part")
    if [ "$psize" -ge "$have" ]; then
      rm -f "$out"
      return
    fi
    rm -f "$part"
  fi
  mv "$out" "$out.resume.part"
}

fetch_omomo() {
  command -v gdown >/dev/null || { echo "install gdown first: pip install gdown" >&2; exit 1; }
  local out=$DL/omomo_raw.tar.gz log tries=${OMOMO_TRIES:-30} wait_s=30 n rc
  log=$(mktemp)
  for ((n = 1; n <= tries; n++)); do
    if [ -f "$out" ]; then
      [ "$(stat -c %s "$out")" -eq "$OMOMO_SIZE" ] && break
      omomo_resume_file "$out"
    fi
    rc=0
    gdown --continue -O "$out" "$OMOMO_DRIVE_ID" 2>&1 | tee "$log" || rc=$?
    if grep -q "Too many users have viewed or downloaded this file" "$log"; then
      rm -f "$log"
      cat >&2 <<EOF

Google Drive's download quota for the OMOMO archive is used up for now (this can last up to 24 hours).
Retry later with:  $0 omomo   (it resumes the partial download)
or download it in a browser from https://drive.google.com/uc?id=$OMOMO_DRIVE_ID
and save it as $out
EOF
      exit 3
    fi
    [ -f "$out" ] && [ "$(stat -c %s "$out")" -eq "$OMOMO_SIZE" ] && break
    echo "attempt $n/$tries stopped early (gdown exit $rc); retrying in $wait_s s" >&2
    sleep "$wait_s"
    wait_s=$((wait_s * 2 > 600 ? 600 : wait_s * 2))
  done
  rm -f "$log"
  if [ ! -f "$out" ] || [ "$(stat -c %s "$out")" -ne "$OMOMO_SIZE" ]; then
    echo "OMOMO archive incomplete after $tries attempts; run '$0 omomo' again to resume" >&2
    exit 1
  fi
  echo "$OMOMO_SHA256  $out" | sha256sum -c - || {
    echo "sha256 mismatch: $out has the right size but other content; delete it and $out*.part, then run again" >&2
    exit 1
  }
}

WHAT=("$@")
[ ${#WHAT[@]} -gt 0 ] || WHAT=(text t5)
for w in "${WHAT[@]}"; do
  case $w in
    text) fetch_text ;;
    t5) fetch_t5 ;;
    omomo) fetch_omomo ;;
    *) echo "unknown item: $w (text, t5, omomo)" >&2; exit 2 ;;
  esac
done
