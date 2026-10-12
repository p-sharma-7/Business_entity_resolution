#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
until grep -q "FRANCE FEATURES READY" $R/logs/er_fr1.log; do
  if grep -qE "Traceback|EXIT [1-9]" $R/logs/er_fr1.log; then echo "france chain 1 failed"; exit 1; fi
  sleep 30
done
cd $R/src
echo "=== [4] re-score France on GPUs 0-4 and merge ==="
ER_ONLY_COUNTRY=France $PY run_predict_par.py
echo "=== [5] pruned final assembly ==="
PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py
echo "=== [6] validate ==="
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
echo "=== [7] France uncertainty, before vs after ==="
$PY - <<'PYEOF'
import numpy as np, pyarrow.parquet as pq
W="${ER_ROOT:-.}/work"
c1=np.asarray(pq.read_table(f"{W}/normalized/test_s1.parquet",columns=["country"]).column("country").to_pylist())
for tag,f in (("before",f"{W}/pred/test_final_prefr.npz"),("after",f"{W}/pred/test_final.npz")):
    z=np.load(f); q,cal,keep=z["q"],z["cal"],z["keep"]
    for c in ("France","India","US"):
        m=c1[q]==c; ns=int((c1==c).sum())
        k=np.bincount(q[keep&m],minlength=len(c1))[c1==c]
        print(f"{tag:<7}{c:<7} uncertain(.3-.9)/100 S1={100*((cal[m]>=.3)&(cal[m]<.9)).sum()/ns:6.2f} "
              f"pred/S1={k.mean():.3f} empty={100*(k==0).mean():.2f}%")
PYEOF
echo "=== [8] package ==="
STG=${ER_WORK:-./work}/pkg/Orchestrator_submission
rm -f $STG/output/*.tsv
cp $R/output/matching_results.tsv $R/output/candidate_pairs.tsv $STG/output/
for f in run_build_views.py run_mine_selftrain.py run_predict_par.py run_final.py; do
  sed 's#sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))#sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))#' \
      $R/src/$f > $STG/code/business_entity_resolution/src/$f
done
cp $R/Documentation_template.md $STG/
rm -f $R/Orchestrator_submission.zip
cd $STG && zip -r -q -9 $R/Orchestrator_submission.zip output code Documentation_template.md
ls -la $R/Orchestrator_submission.zip
echo "=== FRANCE-FIXED SUBMISSION READY ==="
