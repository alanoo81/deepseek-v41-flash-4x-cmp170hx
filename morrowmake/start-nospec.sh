#!/bin/bash
# The configuration of this repository WITHOUT DSpark: about 2.1x the KV pool, single-stream decode ~79 tok/s.
# Pass more vLLM arguments through MORE_ARGS, e.g. MORE_ARGS="--max-model-len 1048576".
set -euo pipefail
cd "$(dirname "$0")"
./start.sh stop >/dev/null 2>&1 || true
CONTAINER_NAME=dsv41-mm EXTRA_ARGS="--speculative-config null ${MORE_ARGS:-}" exec ./start.sh
