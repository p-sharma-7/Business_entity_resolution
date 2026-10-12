#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
$PY $SRC/run_candgen2.py train
echo "=== candgen train done -> features ==="
$PY $SRC/run_features.py train
echo "=== features train done ==="
