#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
M=${ER_ROOT:-.}/models/e5_ho
$PY $SRC/run_finetune_e5.py --gpu 0 --holdout_fold 4 --out $M
export CUDA_VISIBLE_DEVICES=0,1,2
for s in 1 2; do for v in combined name; do
  $PY $SRC/run_encode_e5.py train $s $v --bs 512 --maxlen 64 --model $M --tag _ho 2>&1 | grep -vE "Loading weights|it/s\]"
done; done
echo "=== HONEST EVAL (queries the encoder never saw) ==="
$PY $SRC/run_ft_eval_honest.py --tags "_ho,_f2" --view combined
$PY $SRC/run_ft_eval_honest.py --tags "_ho" --view name
