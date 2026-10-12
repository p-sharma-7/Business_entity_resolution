#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
cd $R/src
echo "=== [1] pruned final assembly (K=10) ==="
PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py
echo "=== [2] validate ==="
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== [3] package ==="
STG=${ER_WORK:-./work}/pkg/Orchestrator_submission
rm -f $STG/output/*.tsv
cp $R/output/matching_results.tsv $R/output/candidate_pairs.tsv $STG/output/
cp $R/src/run_final.py $R/src/run_prune_sim.py $STG/code/business_entity_resolution/src/
cp $R/Documentation_template.md $STG/
cd $STG && zip -r -q -9 $R/Orchestrator_submission.zip output code Documentation_template.md
ls -la $R/Orchestrator_submission.zip
echo "=== PRUNED SUBMISSION READY ==="
