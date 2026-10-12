#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
until grep -q "LEX TRAINED" $R/logs/er_lex.log; do
  if grep -qE "Traceback|EXIT [1-9]" $R/logs/er_lex.log; then echo "lex training failed"; exit 1; fi
  sleep 30
done
cd $R/src
echo "=== [1] honest fold-4: production stage 2 vs language-agnostic stage 2 ==="
ER_TAG=_lex $PY run_eval_tag.py
$PY run_eval_tag.py
echo "=== [2] score France test with the language-agnostic model (GPUs 0-2) ==="
ER_TAG=_lex ER_DROP=107,110,111,112,113,114,115,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149 ER_ONLY_COUNTRY=France $PY run_predict_par.py
echo "=== [3] V8 = India/US from V6 + France from the language-agnostic model ==="
$PY run_build_mix.py V8_france_lex $R/work/pred/test_scores_lex_France.npz 0.6,0.7,0.8,0.9
for th in 0.6 0.7 0.8 0.9; do
  $PY $R/utils/validate_submission.py --matching $R/output_variants/V8_france_lex_th$th/matching_results.tsv       --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test | tail -1
done
echo "=== V8 READY ==="
