#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
W=${ER_ROOT:-.}/work
M=${ER_ROOT:-.}/models/e5_ho

echo "=== [1/6] encode all views with held-out encoder ==="
for split in train test; do for s in 1 2 3; do for v in combined name; do
  $PY $SRC/run_encode_e5.py $split $s $v --bs 512 --maxlen 64 --model $M 2>&1 | grep -E "^done|Error|Traceback" || true
done; done; done

echo "=== [2/6] ANN ==="
rm -f $W/ann/*.npz
for split in train test; do
  $PY $SRC/run_ann.py $split combined --k 30
  $PY $SRC/run_ann.py $split name --k 30
done

echo "=== [3/6] candidates train (lite) ==="
rm -f $W/cand/*.npz $W/feat/*.npz
$PY $SRC/run_candgen2.py train --channels lite --ann_keep 20 --ann_keep_name 10

echo "=== [4/6] features train ==="
$PY $SRC/run_features.py train
echo "=== TRAIN FEATURES READY ==="

echo "=== [5/6] candidates test (lite) ==="
$PY $SRC/run_candgen2.py test --channels lite --ann_keep 20 --ann_keep_name 10
echo "=== [6/6] features test ==="
$PY $SRC/run_features.py test
echo "=== TEST FEATURES READY ==="
