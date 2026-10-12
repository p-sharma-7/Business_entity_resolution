#!/usr/bin/env bash
set -euo pipefail
PY=python
TR=torchrun
R=${ER_ROOT:-.}
cd $R/src
echo "=== [1] French pseudo-labels ==="
[ -f $R/work/pred/ce_fr_train.parquet ] || $PY run_ce_fr.py prep
echo "=== [2] adapt cross-encoder (GPUs 0-2) ==="
CUDA_VISIBLE_DEVICES=0,1,2 $TR --standalone --nproc_per_node=3 run_ce_fr.py train --bs 96 --lr 2e-5
echo "=== [3] re-score French uncertain band ==="
$PY run_ce_fr.py score
echo "=== [4] final assembly ==="
PRUNE_K=10 CUDA_VISIBLE_DEVICES=0 $PY run_final.py
$PY $R/utils/validate_submission.py --matching $R/output/matching_results.tsv \
    --candidate $R/output/candidate_pairs.tsv --test-dir $R/dataset/test
mkdir -p $R/output_variants/V3_france_ce && cp $R/output/matching_results.tsv $R/output_variants/V3_france_ce/
echo "=== [5] diagnostics ==="
$PY - <<'PYEOF'
import numpy as np, pyarrow.parquet as pq
W="${ER_ROOT:-.}/work"
zs=np.load(f"{W}/pred/test_scores.npz"); meta=np.load(f"{W}/pred/ce_score_test.npz")
c1=np.asarray(pq.read_table(f"{W}/normalized/test_s1.parquet",columns=["country"]).column("country").to_pylist())
idx=meta["idx"]; q=zs["q"][idx]; p2=zs["p2"][idx]
for tag,f in (("old CE",f"{W}/pred/ce_scores_test_usindia_ce.npy"),("France CE",f"{W}/pred/ce_scores_test.npy")):
    ce=np.load(f)
    for c in ("France","India","US"):
        m=c1[q]==c; x=ce[m]; p=p2[m]
        print(f"{tag:<10} {c:<7} CE unsure(.2-.8)={100*((x>.2)&(x<.8)).mean():5.1f}%  "
              f"p2<.1 but CE>.5={100*(x[p<.1]>.5).mean():5.2f}%  p2>.9 but CE<.5={100*(x[p>.9]<.5).mean():5.1f}%")
for tag,f in (("V1",f"{W}/pred/test_final_V1.npz"),("V3",f"{W}/pred/test_final.npz")):
    z=np.load(f); q,cal,keep=z["q"],z["cal"],z["keep"]
    for c in ("France","India","US"):
        m=c1[q]==c; ns=int((c1==c).sum()); k=np.bincount(q[keep&m],minlength=len(c1))[c1==c]
        print(f"{tag} {c:<7} uncertain(.3-.9)/100S1={100*((cal[m]>=.3)&(cal[m]<.9)).sum()/ns:6.2f} pred/S1={k.mean():.3f} empty={100*(k==0).mean():.2f}%")
PYEOF
echo "=== V3 READY ==="
