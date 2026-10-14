#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ConvFinQA
PROJECT=/root/autodl-tmp/project/convfinqa_posttrain
for item in data experiments analysis scripts outputs; do
  SRC="$PROJECT/$item"
  DST="$ROOT/$item"
  if [ ! -d "$SRC" ]; then
    echo "[SKIP] Source missing: $SRC"
    continue
  fi
  if [ -L "$DST" ]; then
    if [ "$(readlink -f "$DST")" = "$(readlink -f "$SRC")" ]; then
      echo "[OK] $DST -> $SRC"
    else
      echo "[ERROR] Existing symlink points elsewhere: $DST" >&2
      exit 1
    fi
  elif [ -e "$DST" ]; then
    echo "[ERROR] Path already exists, refusing to overwrite: $DST" >&2
    exit 1
  else
    ln -s "$SRC" "$DST"
    echo "[LINK] $DST -> $SRC"
  fi
done
