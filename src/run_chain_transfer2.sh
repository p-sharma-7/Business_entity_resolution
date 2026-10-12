#!/usr/bin/env bash
# US->India self-training test bed on GPUs 4,5 (GPU 0 runs the France self-training).
set -euo pipefail
PY=python
cd ${ER_ROOT:-.}/src
echo "=== [1] US-only model: reuse stage 1, train stage 2 (GPUs 1,2) ==="
ER_GPUS=1,2 ER_REUSE_S1=1 ER_TAG=_us ER_COUNTRIES=US $PY run_train_gpu.py --rounds 5000
ER_TAG=_us $PY run_eval_tag.py
echo "=== [2] transfer to India with real labels ==="
CUDA_VISIBLE_DEVICES=1 $PY run_transfer_eval.py
echo "=== TRANSFER READY ==="
