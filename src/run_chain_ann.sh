#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
# wait for the encoder to release the GPUs
while pgrep -f run_encode_e5.py >/dev/null; do sleep 20; done
for split in train test; do
  for view in name combined; do
    $PY $SRC/run_ann.py $split $view --k 30
  done
done
