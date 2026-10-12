#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
until grep -q "V4 READY" $R/logs/er_v4.log; do
  if grep -qE "Traceback|EXIT [1-9]" $R/logs/er_v4.log; then echo "V4 failed"; exit 1; fi
  sleep 30
done
cd $R/src
echo "=== [1] sibling features on test (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 $PY run_sibling.py test
echo "=== [2] V5 = V4 + sibling stage 3 ==="
for th in 0.85 0.92 0.95; do
  ER_STAGE3=stage3b_sib ER_TH_FRANCE=$th PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py 2>&1 | grep -E "selected|France:|India:|US:|wrote"
  mkdir -p $R/output_variants/V5_sib_france_th$th
  cp $R/output/matching_results.tsv $R/output_variants/V5_sib_france_th$th/
  $PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
      --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test | tail -1
done
echo "=== V5 READY ==="
