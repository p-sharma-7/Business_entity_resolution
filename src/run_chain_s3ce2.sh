#!/usr/bin/env bash
# Wait for the e5-base reranker's fold-4 scores, then run the stage-3 A/B.
set -uo pipefail
cd ${ER_ROOT:-.}
until grep -qE "CEB FOLD4 READY|Traceback|EXIT [1-9]" logs/er_ceb.log; do sleep 30; done
grep -q "CEB FOLD4 READY" logs/er_ceb.log || { echo "reranker chain failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python src/run_stage3_ce2.py
