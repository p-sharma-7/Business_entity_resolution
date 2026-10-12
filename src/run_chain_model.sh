#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
W=${ER_ROOT:-.}/work
LOG=${ER_ROOT:-.}/logs/er_main.log
until grep -q "TRAIN FEATURES READY" $LOG; do
  if grep -qE "Traceback|EXIT [1-9]" $LOG; then echo "main chain failed"; exit 1; fi
  sleep 30
done
for p in S2_India S2_US S3_India S3_US; do
  [ -f $W/feat/train_$p.npz ] || { echo "MISSING feat $p"; exit 1; }
done
echo "=== all 4 train partitions present; GPU training on GPUs 0-4 ==="
$PY $SRC/run_train_gpu.py --rounds 1500
$PY $SRC/run_decide.py
echo "=== MODEL + DECISION READY ==="
