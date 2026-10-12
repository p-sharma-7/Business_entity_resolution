#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
cd $R/src
echo "=== [1] rebuild test views (France type words are content again) ==="
for s in 1 2 3; do $PY run_build_views.py test $s; done
echo "=== [2] France candidates + features ==="
rm -f $R/work/cand/test_S2_France.npz $R/work/cand/test_S3_France.npz
$PY run_candgen2.py test --only S2_France --channels lite --ann_keep 20 --ann_keep_name 10
$PY run_candgen2.py test --only S3_France --channels lite --ann_keep 20 --ann_keep_name 10
rm -f $R/work/feat/test_S2_France.npz $R/work/feat/test_S3_France.npz $R/work/feat/test_S2_France_X.npy $R/work/feat/test_S3_France_X.npy
$PY run_features.py test
echo "=== [3] re-score France (GPUs 0-2) ==="
ER_ONLY_COUNTRY=France $PY run_predict_par.py
echo "=== [4] cross-encoder: reuse India/US, France-adapted CE for France ==="
$PY run_ce_splice.py
echo "=== [5] final assembly, France threshold variants ==="
for th in 0.85 0.92 0.95; do
  ER_TH_FRANCE=$th PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py 2>&1 | grep -E "selected|France:|India:|US:|wrote"
  mkdir -p $R/output_variants/V4_france_th$th
  cp $R/output/matching_results.tsv $R/output_variants/V4_france_th$th/
  $PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
      --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test | tail -1
done
cp $R/output/candidate_pairs.tsv $R/output_variants/V4_candidate_pairs.tsv
echo "=== [6] diagnostics ==="
$PY - <<'PYEOF'
import numpy as np, pyarrow.parquet as pq
W="${ER_ROOT:-.}/work"
c1=np.asarray(pq.read_table(f"{W}/normalized/test_s1.parquet",columns=["country"]).column("country").to_pylist())
for tag,f in (("V3",f"{W}/pred/V3_state/test_final.npz"),("V4",f"{W}/pred/test_final.npz")):
    z=np.load(f); q,cal=z["q"],z["cal"]
    for c in ("France","India","US"):
        m=c1[q]==c; ns=int((c1==c).sum())
        print(f"{tag} {c:<7} uncertain(.3-.9)/100S1={100*((cal[m]>=.3)&(cal[m]<.9)).sum()/ns:6.2f}  links cal>=.85: {int((m&(cal>=.85)).sum()):,}")
PYEOF
echo "=== V4 READY ==="
