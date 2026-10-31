#!/bin/bash
# The configuration of this repository WITH DSpark (the release default): one server on the four cards.
set -euo pipefail
cd "$(dirname "$0")"
./start.sh stop >/dev/null 2>&1 || true
CONTAINER_NAME=dsv41-mm-dspark exec ./start.sh
