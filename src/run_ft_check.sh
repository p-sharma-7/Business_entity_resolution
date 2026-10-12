#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
M=${ER_ROOT:-.}/models/e5_ft
for s in 1 2; do
  $PY $SRC/run_encode_e5.py train $s name --bs 640 --maxlen 48 --model $M --tag _ft 2>&1 | grep -vE "Loading weights|it/s\]"
done
$PY $SRC/run_ft_eval.py --tag _ft
