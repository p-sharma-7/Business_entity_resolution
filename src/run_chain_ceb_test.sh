#!/usr/bin/env bash
# Score the India/US test band with the e5-base reranker, then assemble test_final_ce2.
set -euo pipefail
PY=python
cd ${ER_ROOT:-.}/src
export ER_CE_OUT=${ER_ROOT:-.}/models/ce_b ER_CE_TAG=_b
echo "=== [1] score India/US test band (e5-base) ==="
$PY run_ce.py score test
echo "=== [2] assemble ==="
CUDA_VISIBLE_DEVICES=0 $PY run_final_ce2.py assemble
echo "=== CEB TEST READY ==="
