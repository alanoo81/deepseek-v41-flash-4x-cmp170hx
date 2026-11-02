#!/bin/bash
# Builds kvc-overlay/ in a Morrowmake 1.0.1 checkout: the six engine files the compact KV cache modifies,
# taken from the release image itself and patched, plus the new sm80_kv_pack.py.
# usage: build-overlay.sh /path/to/morrowmake-checkout
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT="$(cd "${1:?usage: build-overlay.sh MORROWMAKE_CHECKOUT}" && pwd)"
IMAGE="$(grep -oP '^RELEASE_IMAGE=\K.*' "$CHECKOUT/start.sh" | tr -d '"')"
[ -n "$IMAGE" ] || { echo "no RELEASE_IMAGE in $CHECKOUT/start.sh" >&2; exit 1; }
SRC=/opt/assembly/plugin/vllm_sm80/models/deepseek_v41
OUT="$CHECKOUT/kvc-overlay"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/vllm_sm80/models/deepseek_v41/ops"
for f in ops/sm80_attn_core.py ops/sm80_attn_pipeline.py ops/sm80_sparse_attn.py ops/sm80_attn_fused.py \
         ops/sm80_index_scores.py sm80_attention.py; do
    docker run --rm --entrypoint cat "$IMAGE" "$SRC/$f" > "$WORK/vllm_sm80/models/deepseek_v41/$f"
done
(cd "$WORK" && patch -p1 --forward --quiet < "$HERE/compact-kv.patch")
cp "$HERE/vllm_sm80/models/deepseek_v41/ops/sm80_kv_pack.py" "$WORK/vllm_sm80/models/deepseek_v41/ops/"
rm -rf "$OUT"
mkdir -p "$OUT"
cp -r "$WORK/vllm_sm80" "$OUT/"
echo "kvc-overlay built from $IMAGE:"
(cd "$OUT" && find . -name '*.py' | sort)
