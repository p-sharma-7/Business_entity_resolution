#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2
PY=python
SRC=${ER_ROOT:-.}/src
W=${ER_ROOT:-.}/work
M=${ER_ROOT:-.}/models/e5_ft

# name view re-encoded with the fine-tuned encoder (recall@30 0.556 -> 0.860)
for split in train test; do for s in 1 2 3; do
  $PY $SRC/run_encode_e5.py $split $s name --bs 640 --maxlen 48 --model $M 2>&1 | grep -vE "Loading weights|it/s\]"
done; done

rm -f $W/ann/*_name.npz
for split in train test; do
  $PY $SRC/run_ann.py $split name --k 40
done
echo "=== ANN refreshed with fine-tuned name view ==="
