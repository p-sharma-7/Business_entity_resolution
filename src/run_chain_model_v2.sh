#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
$PY $SRC/run_train_gpu.py --rounds 5000
$PY $SRC/run_decide.py
echo "=== V2 MODEL + DECISION READY ==="
