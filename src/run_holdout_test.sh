#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2
PY=python
SRC=${ER_ROOT:-.}/src
M=${ER_ROOT:-.}/models/e5_ho
$PY $SRC/run_finetune_e5.py --gpu 0 --holdout_fold 0 --pairs 2000000 --out $M
for s in 1 2; do for v in name combined; do
  $PY $SRC/run_encode_e5.py train $s $v --bs 512 --maxlen 64 --model $M --tag _ho --gpus 1 2>&1 | grep -vE "Loading weights|it/s\]"
done; done
echo "=== HELD-OUT retrieval (entities the encoder never saw) ==="
$PY $SRC/run_ft_eval.py --tag _ho --view combined --holdout_fold 0
$PY $SRC/run_ft_eval.py --tag _ho --view name --holdout_fold 0
