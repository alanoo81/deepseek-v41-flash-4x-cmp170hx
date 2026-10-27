#!/usr/bin/env bash
# deepseek-v41-flash-4x-cmp170hx (MIT): copy the DeepSeek-V4.1-Flash Engram tables into a tmpfs (RAM), to be read in
# ENGRAM_STORAGE=ssd mode. A tmpfs is lost on every reboot of the machine or container, so the patched start.sh
# calls this script before serve.sh. Copies 2 x ~101 GB from MODELS_DIR (about 1.5-2 min from NVMe), and only the
# files that are missing or whose size differs. Needs the right to mount a tmpfs (root or a privileged container).
# Paths: MODELS_DIR and ENGRAM_HOST_DIR come from .env (start.sh exports them); the defaults are the ones we used.
set -euo pipefail
SRC=${MODELS_DIR:-/opt/models}/DeepSeek-V4.1-Flash-1017-engram
DST=${ENGRAM_HOST_DIR:-/opt/engram-ram/DeepSeek-V4.1-Flash-1017-engram}
MNT=$(dirname "$DST")
SIZE=${ENGRAM_TMPFS_SIZE:-200G}
mkdir -p "$MNT"
mountpoint -q "$MNT" || mount -t tmpfs -o "size=$SIZE,mode=0755" tmpfs "$MNT"
mkdir -p "$DST"
for f in config.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  if [ ! -f "$DST/$f" ] || [ "$(stat -c %s "$SRC/$f")" != "$(stat -c %s "$DST/$f")" ]; then
    echo "engram-ram: copying $f"
    cp "$SRC/$f" "$DST/$f.part" && mv "$DST/$f.part" "$DST/$f"
  fi
done
echo "engram-ram: ready ($(du -sh "$DST" | cut -f1) in $DST)"
