#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
$PY $SRC/run_train_gpu.py --rounds 1500
$PY $SRC/run_decide.py
echo "=== MODEL + DECISION READY ==="
