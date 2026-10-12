#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
echo "=== [A] v2 fallback submission ==="
CUDA_VISIBLE_DEVICES=0 $PY $R/src/run_predict.py
mkdir -p $R/output_v2_fallback && cp $R/output/*.tsv $R/output_v2_fallback/
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== V2 FALLBACK READY ==="
