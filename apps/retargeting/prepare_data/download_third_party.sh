#!/usr/bin/env bash
# Download the publicly available third-party files the retargeting app needs and check their sha256.
#
#   apps/retargeting/prepare_data/download_third_party.sh [<release_data>/retargeting]
#
# Files:
#   TWIST general motion tracker (MIT, https://github.com/YanjieZe/TWIST, commit b6af769)
#     -> code_ext/ext/lib_twist/src/lib_twist/assets/twist_general_motion_tracker.pt
set -euo pipefail

APP=$(cd "$(dirname "$0")/.." && pwd)
DATA=${1:-$APP/../../release_data/retargeting}

fetch() {
  local url=$1 dst=$2 sha=$3
  if [ -f "$dst" ] && echo "$sha  $dst" | sha256sum -c --status; then
    echo "ok (present): $dst"; return
  fi
  mkdir -p "$(dirname "$dst")"
  curl -fL --retry 3 -o "$dst.part" "$url"
  echo "$sha  $dst.part" | sha256sum -c --status || { echo "sha256 mismatch: $url" >&2; rm -f "$dst.part"; exit 1; }
  mv "$dst.part" "$dst"
  echo "ok: $dst"
}

fetch https://raw.githubusercontent.com/YanjieZe/TWIST/b6af7695346dfaf9ad5481b07e0442a7de152c8f/assets/twist_general_motion_tracker.pt \
  "$DATA/code_ext/ext/lib_twist/src/lib_twist/assets/twist_general_motion_tracker.pt" \
  0ec24c54c1e1502f5c7bd27efb39ecc0630c06b5c01dc640a6d493d672fc515c
