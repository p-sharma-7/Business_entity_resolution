#!/usr/bin/env bash
set -euo pipefail
PY=python
for split in train test; do
  for s in 1 2 3; do
    $PY ${ER_ROOT:-.}/src/run_build_views.py $split $s
  done
done
