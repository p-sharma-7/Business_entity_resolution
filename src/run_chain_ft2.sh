#!/usr/bin/env bash
set -euo pipefail
PY=python
SRC=${ER_ROOT:-.}/src
W=${ER_ROOT:-.}/work
M=${ER_ROOT:-.}/models/e5_ft2
while pgrep -f run_finetune_e5.py >/dev/null; do sleep 30; done

echo "=== gate: does the mixed fine-tune improve BOTH views? ==="
for s in 1 2; do for v in name combined; do
  $PY $SRC/run_encode_e5.py train $s $v --bs 512 --maxlen 64 --model $M --tag _f2 2>&1 | grep -vE "Loading weights|it/s\]"
done; done
$PY $SRC/run_ft_eval.py --tag _f2 --view name
$PY $SRC/run_ft_eval.py --tag _f2 --view combined
