#!/usr/bin/env bash
set -euo pipefail
PY=python
TR=torchrun
SRC=${ER_ROOT:-.}/src
LOG=${ER_ROOT:-.}/logs/er_model2.log
cd $SRC
[ -f ${ER_ROOT:-.}/work/pred/ce_train_pairs.npz ] || $PY run_ce.py prep
CUDA_VISIBLE_DEVICES=0,1,2 $TR --standalone --nproc_per_node=5 run_ce.py train --bs 64 --max_pairs 6000000
$PY run_ce.py score train
$PY run_stage3.py
echo "=== STAGE3 READY ==="
