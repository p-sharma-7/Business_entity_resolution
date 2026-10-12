#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
until grep -q "TEST FEATURES READY" $R/logs/er_main.log && grep -q "V2 MODEL + DECISION READY" $R/logs/er_model2.log; do
  if grep -qE "Traceback|EXIT [1-9]" $R/logs/er_main.log; then echo "test features failed"; exit 1; fi
  sleep 60
done
echo "=== [A] v2 fallback submission ==="
CUDA_VISIBLE_DEVICES=1 $PY $R/src/run_predict.py
mkdir -p $R/output_v2_fallback && cp $R/output/*.tsv $R/output_v2_fallback/
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== [B] waiting for stage 3 ==="
until grep -q "STAGE3 READY" $R/logs/er_ce.log; do
  if grep -qE "EXIT [1-9]" $R/logs/er_ce.log; then echo "stage3 chain failed"; exit 1; fi
  sleep 60
done
cd $R/src && $PY run_final.py
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== FINAL SUBMISSION READY ==="
