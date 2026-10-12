#!/usr/bin/env bash
set -euo pipefail
PY=python
R=${ER_ROOT:-.}
cd $R/src
echo "=== [1] rebuild test views with self-trained France tables ==="
for s in 1 2 3; do $PY run_build_views.py test $s; done
echo "=== [2] France candidates ==="
rm -f $R/work/cand/test_S2_France.npz $R/work/cand/test_S3_France.npz
$PY run_candgen2.py test --only S2_France --channels lite --ann_keep 20 --ann_keep_name 10
$PY run_candgen2.py test --only S3_France --channels lite --ann_keep 20 --ann_keep_name 10
echo "=== [3] France features ==="
rm -f $R/work/feat/test_S2_France.npz $R/work/feat/test_S3_France.npz $R/work/feat/test_S2_France_X.npy $R/work/feat/test_S3_France_X.npy
$PY run_features.py test
echo "=== FRANCE FEATURES READY ==="
