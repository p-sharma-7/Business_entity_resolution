"""Test assembly with the second reranker (India/US); France unchanged.

  prep      write the India/US uncertain-band pair list for the e5-base reranker
  assemble  stage 3 of models/stage3_ce2 on India/US band pairs; France keeps
            the production scores (its French-adapted pipeline); writes
            work/pred/test_final_ce2.npz in the layout of test_final.npz, for
            run_legalrule.py (ER_FINAL=test_final_ce2).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
import joblib
from er.io_utils import WORK
from run_ce import LO, HI
from run_stage3 import make_features, SUBF
from run_stage3_ce2 import extra_features, OUT as S3DIR       # ER_S3_OUT selects the variant
from run_prune_sim import prune_mask

PRUNE_K = 10


def main(mode):
    import xgboost as xgb
    t0 = time.time()
    z = np.load(f"{WORK}/pred/test_scores.npz")
    q, t, src, p1, p2 = z["q"], z["t"], z["src"], z["p1"], z["p2"]
    band = np.flatnonzero((p2 > LO) & (p2 < HI))
    meta = np.load(f"{WORK}/pred/ce_score_test.npz")
    assert np.array_equal(meta["idx"], band), "production reranker scores do not match the band"
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["country"]).column(0).to_pylist())
    if mode == "prep":
        iu = band[c1[q[band]] != "France"]
        np.savez(f"{WORK}/pred/ce_score_test_b.npz", idx=iu, q=q[iu], t=t[iu], src=src[iu])
        print(f"India/US band pairs to score: {len(iu):,} (France band {len(band) - len(iu):,} left alone)")
        return
    ce = np.full(len(q), np.nan, np.float32); ce[band] = np.load(f"{WORK}/pred/ce_scores_test.npy")
    zb = np.load(f"{WORK}/pred/ce_score_test_b.npz")
    ceb = np.full(len(q), np.nan, np.float32); ceb[zb["idx"]] = np.load(f"{WORK}/pred/ce_scores_test_b.npy")
    kept = np.flatnonzero(prune_mask(q, src, p1, PRUNE_K))
    q, t, src, p1, p2, ce, ceb = (a[kept] for a in (q, t, src, p1, p2, ce, ceb))
    prod = np.load(f"{WORK}/pred/test_final.npz")
    assert all(np.array_equal(prod[k], v) for k, v in (("q", q), ("t", t), ("src", src))), \
        "pruned pairs differ from the production assembly"
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    idx = np.flatnonzero(~np.isnan(ce))
    X, names = make_features(q, tkey, src, p1, p2, ce, rows=idx)
    Xb, nb = extra_features(ce, ceb, idx)
    cfg = json.load(open(f"{S3DIR}/config.json"))
    assert names + nb == cfg["features"]
    X = np.hstack([X, Xb])
    acc = np.zeros(len(idx), np.float64)
    ns = cfg.get("seeds", 1)
    for f in range(SUBF):
        for k in range(ns):
            b = xgb.Booster(); b.load_model(f"{S3DIR}/f{f}.json" if ns == 1 else f"{S3DIR}/f{f}_s{k}.json")
            b.set_param({"device": "cuda"})
            acc += b.inplace_predict(X, iteration_range=(0, b.best_iteration + 1))
    p3 = p2.astype(np.float32).copy(); p3[idx] = acc / (SUBF * ns)
    cal = joblib.load(f"{S3DIR}/calibrator.pkl").predict(p3).astype(np.float32)
    fr = c1[q] == "France"
    cal[fr] = prod["cal"][fr]; p3[fr] = prod["p3"][fr]           # France: production pipeline
    iu_band = (~fr[idx]).sum()
    assert np.isfinite(ceb[idx][~fr[idx]]).all(), "India/US band pairs without e5-base scores"
    np.savez(f"{WORK}/pred/test_final_ce2.npz", q=q, t=t, src=src, p3=p3, cal=cal, keep=prod["keep"])
    print(f"India/US band pairs re-scored: {iu_band:,}; thresholds {cfg['per_country']} "
          f"(honest fold-4 {cfg['honest_macro_f05']:.5f}) ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
