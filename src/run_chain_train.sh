#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
while pgrep -f "candgen2.py train" >/dev/null; do sleep 30; done
echo "=== candgen train finished, starting features ==="
$PY $SRC/run_features.py train
