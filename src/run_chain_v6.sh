#!/usr/bin/env bash
set -euo pipefail
PY=python
TR=torchrun
R=${ER_ROOT:-.}
export ER_CE_FR_TAG=2
cd $R/src
echo "=== [1] pseudo-labels from V4 stage 2 ==="
$PY run_ce_fr.py prep
echo "=== [2] retrain French CE from the original reranker (GPUs 0-2) ==="
CUDA_VISIBLE_DEVICES=0,1,2 $TR --standalone --nproc_per_node=3 run_ce_fr.py train --bs 96 --lr 2e-5
echo "=== [3] splice: India/US reused, France scored with CE round 2 ==="
cp $R/work/pred/V4_ce_score_test.npz $R/work/pred/ce_score_test.npz
cp $R/work/pred/V4_ce_scores_test.npy $R/work/pred/ce_scores_test.npy
$PY run_ce_splice.py
echo "=== [4] final (France thresholds) ==="
for th in 0.92 0.95 0.97; do
  ER_TH_FRANCE=$th PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py 2>&1 | grep -E "selected|France:"
  mkdir -p $R/output_variants/V6_france_th$th
  cp $R/output/matching_results.tsv $R/output_variants/V6_france_th$th/
  $PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
      --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test | tail -1
done
cp $R/work/pred/test_final.npz $R/work/pred/test_final_V6.npz
echo "=== [5] did the override disappear? ==="
$PY - <<'PYEOF'
import sys; sys.path.insert(0,'${ER_ROOT:-.}/src')
import numpy as np, pyarrow.parquet as pq
from er import decide as D
W="${ER_ROOT:-.}/work"
c1=np.asarray(pq.read_table(f"{W}/normalized/test_s1.parquet",columns=["country"]).column("country").to_pylist())
for tag,f in (("V4",f"{W}/pred/test_final_V4.npz"),("V6",f"{W}/pred/test_final_V6.npz")):
    z=np.load(f); q,cal=z["q"],z["cal"]; tk=z["t"].astype(np.int64)+z["src"].astype(np.int64)*10_000_000
    u=D.unique_target_filter(tk,cal); m=c1[q]=="France"; ns=int((c1=="France").sum())
    print(f"{tag} France: uncertain(.3-.9)/100S1={100*((cal[m]>=.3)&(cal[m]<.9)).sum()/ns:6.2f}  "
          f"links in [.90,.97)={int((u&m&(cal>=.9)&(cal<.97)).sum()):,}  links>=.95={int((u&m&(cal>=.95)).sum()):,}")
PYEOF
echo "=== V6 READY ==="
