#!/usr/bin/env bash
# Second reranker (multilingual-e5-base) on GPUs 3-5; production CE untouched.
set -euo pipefail
PY=python
TR=torchrun
cd ${ER_ROOT:-.}/src
export ER_CE_BASE=intfloat/multilingual-e5-base
export ER_CE_OUT=${ER_ROOT:-.}/models/ce_b ER_CE_TAG=_b ER_CE_FREEZE_EMB=1 ER_CE_GC=0
echo "=== [1] train e5-base reranker on folds 0-3 ==="
CUDA_VISIBLE_DEVICES=0,1,2 $TR --standalone --nproc_per_node=3 run_ce.py train --bs 32 --lr 3e-5 --max_pairs 2000000
echo "=== [2] score fold-4 band ==="
$PY run_ce.py score train
echo "=== CEB FOLD4 READY ==="
