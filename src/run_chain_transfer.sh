#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
until grep -q "V8 READY" $R/logs/er_lex2.log; do
  if grep -qE "Traceback|EXIT [1-9]" $R/logs/er_lex2.log; then echo "V8 chain failed; continuing anyway"; break; fi
  sleep 30
done
cd $R/src
echo "=== [1] US-only model (GPUs 0-2) ==="
ER_TAG=_us ER_COUNTRIES=US $PY run_train_gpu.py --rounds 5000
ER_TAG=_us $PY run_eval_tag.py
echo "=== [2] transfer to India with real labels ==="
CUDA_VISIBLE_DEVICES=0 $PY run_transfer_eval.py
echo "=== TRANSFER READY ==="
