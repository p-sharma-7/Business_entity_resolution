#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
while pgrep -f run_ann.py >/dev/null; do sleep 20; done
$PY $SRC/run_candgen2.py train
