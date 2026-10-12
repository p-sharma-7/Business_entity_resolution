"""Stage 3 with a second reranker (A/B on the honest fold-4 harness).

A: production stage-3 features (stage-2 scores, e5-small reranker, groups).
B: A + the multilingual-e5-base reranker (models/ce_b) and the mean of both.
Both are fit and scored identically (fold-4 sub-folds, ALL S1 competing for
each target, per-country thresholds), so the difference is the reranker alone.
B's models go to models/stage3_ce2/; production files are not touched.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_ce import LO, HI
from run_stage3 import make_features, sub_fold, PARAMS, SUBF

OUT = os.environ.get("ER_S3_OUT", f"{ROOT}/models/stage3_ce2")
SEEDS = int(os.environ.get("ER_S3_SEEDS", "1"))      # >1: average seed-bagged boosters


def extra_features(ce, ceb, rows):
    has = ~np.isnan(ceb)
    mean = np.where(has, 0.5 * (np.nan_to_num(ce, nan=0.0) + np.nan_to_num(ceb, nan=0.0)), -1.0)
    return (np.column_stack([np.nan_to_num(ceb[rows], nan=-1.0), mean[rows]]).astype(np.float32),
            ["ce_b", "ce_mean"])


def evaluate(X, names, ctx, save_dir=None):
    import xgboost as xgb, joblib
    q, tkey, y, p2, r4, b4, sf, qq, qc, ent, tc = ctx
    y4 = y[r4]
    p3_4 = p2[r4].astype(np.float32).copy()
    for f in range(SUBF):
        tr = b4 & (sf != f); va = b4 & (sf == f)
        dtr = xgb.QuantileDMatrix(X[tr], label=y4[tr], max_bin=PARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(X[va], label=y4[va], ref=dtr, max_bin=PARAMS["max_bin"])
        acc = np.zeros(int(va.sum()), np.float64)
        for k in range(SEEDS):
            bst = xgb.train(dict(PARAMS, seed=k), dtr, 3000, evals=[(dva, "val")],
                            early_stopping_rounds=80, verbose_eval=False)
            acc += bst.inplace_predict(X[va], iteration_range=(0, bst.best_iteration + 1))
            if save_dir:
                bst.save_model(f"{save_dir}/f{f}.json" if SEEDS == 1 else f"{save_dir}/f{f}_s{k}.json")
        p3_4[va] = acc / SEEDS
    cal3_4 = np.zeros(len(r4), np.float32)
    for f in range(SUBF):
        tr = sf != f
        cal3_4[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3_4[tr], y4[tr]).predict(p3_4[~tr])
    iso3 = IsotonicRegression(out_of_bounds="clip").fit(p3_4, y4)
    full3 = iso3.predict(p2).astype(np.float32); full3[r4] = cal3_4   # other folds: p2 only

    def f05(k):
        return D.macro_f05_fast(qq[k], np.ones(int(k.sum()), bool), y4[k], len(ent), tc)

    uniq3 = D.unique_target_filter(tkey, full3)[r4]
    ths = np.round(np.arange(0.30, 0.91, 0.05), 2)
    sc = {th: f05(uniq3 & (cal3_4 >= th)) for th in ths}
    bth = max(sc, key=sc.get)
    per, keep = {}, uniq3 & (cal3_4 >= bth)
    for c in ("India", "US"):
        m = qc == c; bt, bs = float(bth), f05(keep)
        for th in ths:
            k2 = keep.copy(); k2[m] = uniq3[m] & (cal3_4[m] >= th)
            s_ = f05(k2)
            if s_ > bs:
                bt, bs = float(th), s_
        per[c] = bt; keep[m] = uniq3[m] & (cal3_4[m] >= bt)
    final = f05(keep)
    if save_dir:
        joblib.dump(iso3, f"{save_dir}/calibrator.pkl")
        json.dump({"features": names, "threshold": float(bth), "per_country": per, "seeds": SEEDS,
                   "honest_macro_f05": float(final)}, open(f"{save_dir}/config.json", "w"), indent=2)
    return final, per


def main():
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2, fold = (z["q"], z["t"], z["src"], z["y"], z["p1"], z["p2"], z["fold"])
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    e_ids = tb.column("entity_id").to_pylist(); ecty = np.asarray(tb.column("country").to_pylist())
    cs = np.load(f"{WORK}/pred/ce_score_train.npz")
    ce = np.full(len(y), np.nan, np.float32); ce[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train.npy")
    ceb = np.full(len(y), np.nan, np.float32); ceb[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train_b.npy")
    r4 = np.flatnonzero(fold == 4)
    band4 = (fold == 4) & (p2 > LO) & (p2 < HI)
    X, names = make_features(q, tkey, src, p1, p2, ce, rows=r4)
    Xb, nb = extra_features(ce, ceb, r4)
    ent = np.flatnonzero(np.array([fold_of(e) == 4 for e in e_ids]))
    sf_ent = np.full(len(e_ids), -1, np.int8); sf_ent[ent] = [sub_fold(e_ids[i]) for i in ent]
    remap = np.full(len(e_ids), -1, np.int64); remap[ent] = np.arange(len(ent))
    gt = read_ground_truth("train")
    tc = np.array([len(gt[e_ids[i]]) for i in ent], np.int32)
    ctx = (q, tkey, y, p2, r4, band4[r4], sf_ent[q[r4]], remap[q[r4]], ecty[q[r4]], ent, tc)
    m = band4[r4]
    yb = y[r4][m]
    for nm, sc in (("e5-small CE", ce[r4][m]), ("e5-base CE", ceb[r4][m])):
        acc = ((sc >= 0.5) == (yb == 1)).mean()
        ll = -np.mean(yb * np.log(np.clip(sc, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - sc, 1e-6, 1)))
        print(f"fold-4 band pairs={m.sum():,}  {nm}: accuracy={acc:.4f} logloss={ll:.4f}", flush=True)
    if os.environ.get("ER_S3_ONLY_B") == "1":
        a = 0.99064                                   # A, measured in the first run
    else:
        a, pa = evaluate(X, names, ctx)
        print(f"A  production features            honest fold-4 macro-F0.5 = {a:.5f} {pa} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    b, pb = evaluate(np.hstack([X, Xb]), names + nb, ctx, save_dir=OUT)
    print(f"B  + e5-base reranker, {SEEDS} seed(s)  honest fold-4 macro-F0.5 = {b:.5f} {pb}  (B - A = {b - a:+.5f})", flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
