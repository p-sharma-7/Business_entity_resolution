#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
for split in train test; do
  for s in 1 2 3; do
    $PY $SRC/run_encode_e5.py $split $s name     --bs 640 --maxlen 48 2>&1 | grep -vE "Loading weights|it/s\]"
  done
done
for split in train test; do
  for s in 1 2 3; do
    $PY $SRC/run_encode_e5.py $split $s combined --bs 384 --maxlen 96 2>&1 | grep -vE "Loading weights|it/s\]"
  done
done
