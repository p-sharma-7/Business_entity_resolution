#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
cd $R/src
echo "=== [1] test scoring on GPUs 0-4 ==="
$PY run_predict_par.py
echo "=== [2] validate v2 fallback ==="
$PY $R/utils/validate_submission.py --matching $R/output_v2_fallback/matching_results.tsv \
    --candidate $R/output_v2_fallback/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== [3] cross-encoder + stage 3 on test ==="
CUDA_VISIBLE_DEVICES=0 $PY run_final.py
echo "=== [4] validate final submission ==="
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== FINAL SUBMISSION READY ==="
