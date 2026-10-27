#!/usr/bin/env bash
# Derived from tools/download.sh of GNmac-lab/deepseek-v41-flash-1017-cmp170hx @ 2ad5741ab1 (release 1.2.0),
# Copyright (c) 2026 GNmac-lab, MIT licence (see NOTICE.md). Changes (c) 2026 the deepseek-v41-flash-4x-cmp170hx
# contributors, MIT:
#   - Hugging Face revisions are PINNED (the original downloads from main);
#   - also fetches the 5 shards of the LibertAI 3.5 bpw pack that hold the layer 18-22 experts at 3 bits
#     (model-00020..00024-of-00044); make_hybrid_graft.py grafts them in place of the 2.9 pack's 2-bit layers;
#   - the 2.9 shards of layers 18-22 are kept, so 2 bits and 3 bits can still be compared;
#   - does not build the hybrid (run tools/make_hybrid_graft.py afterwards).
# Resumable: rerun after an interruption (curl -C -, .done markers for the official tensors). HF_TOKEN is optional.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
M=${MODELS_DIR:?set MODELS_DIR in .env or in the environment}
P=${PARALLEL:-4}
LIB_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw;  LIB_REV=0be63c3337efbb2a0061d5b8bb4006fa186d433f
L35_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw;  L35_REV=04a492fc3b4d25ddbab8f2d2f07f235ec77c9010
OFF_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017;               OFF_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
QCFG_URL=https://raw.githubusercontent.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/b8b9e993e015f35c22ab7202b4ef3c05f8b663f0/quantization_config.json
LIB=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
L35=$M/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-L18-22
ENG=$M/DeepSeek-V4.1-Flash-1017-engram
OFF=$M/DeepSeek-V4.1-Flash-1017-official-nonexpert
mkdir -p "$LIB" "$L35" "$ENG" "$OFF"
export AUTH_HDR="${HF_TOKEN:+Authorization: Bearer $HF_TOKEN}"

fetch() {  # fetch REPO REV FILE DEST_DIR: resumable, retried
  local repo=$1 rev=$2 f=$3 dst=$4
  mkdir -p "$dst/$(dirname "$f")"
  for _ in 1 2 3 4 5; do
    if curl -sfL -C - --retry 3 ${AUTH_HDR:+-H "$AUTH_HDR"} -o "$dst/$f" "https://huggingface.co/$repo/resolve/$rev/$f"; then
      echo "ok   $repo/$f"; return 0
    fi
    sleep 5
  done
  echo "FAIL $repo/$f"; return 1
}
export -f fetch

echo "== 1/5 LibertAI EXL3 2.9 bpw pack @${LIB_REV:0:8} -> $LIB"
curl -sfL "https://huggingface.co/api/models/$LIB_REPO/tree/$LIB_REV?recursive=true" |
  python3 -c "import json, sys; [print(e['path']) for e in json.load(sys.stdin) if e['type'] == 'file']" > "$LIB/.files"
{ grep safetensors "$LIB/.files" || true; grep -v safetensors "$LIB/.files" || true; } |
  xargs -P "$P" -I{} bash -c 'fetch "$0" "$1" "$2" "$3"' "$LIB_REPO" "$LIB_REV" {} "$LIB"

echo "== 2/5 LibertAI 3.5 bpw pack @${L35_REV:0:8}, layers 18-22 only -> $L35"
printf '%s\n' model-00020-of-00044.safetensors model-00021-of-00044.safetensors model-00022-of-00044.safetensors \
              model-00023-of-00044.safetensors model-00024-of-00044.safetensors model.safetensors.index.json \
              config.json quantization_config.json exl3_k_map.json |
  xargs -P "$P" -I{} bash -c 'fetch "$0" "$1" "$2" "$3"' "$L35_REPO" "$L35_REV" {} "$L35"

echo "== 3/5 official non-expert tensors @${OFF_REV:0:8} -> $OFF"
OFF_REV=$OFF_REV python3 tools/fetch_official_nonexpert.py "$OFF"

echo "== 4/5 official Engram tables @${OFF_REV:0:8} -> $ENG"
printf '%s\n' model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json |
  xargs -P 2 -I{} bash -c 'fetch "$0" "$1" "$2" "$3"' "$OFF_REPO" "$OFF_REV" {} "$ENG"

echo "== 5/5 Curtis White's quantization_config.json (pinned commit)"
curl -sfL -o "$M/quantization_config.json" "$QCFG_URL"
echo "downloads done; now build the hybrid with tools/make_hybrid_graft.py"
