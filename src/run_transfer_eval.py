"""Unseen-country test bed: US-trained model applied to India, with real labels.

France has no labels, so every French fix has been checked only on the public
leaderboard. Training on US alone and treating India as the unseen country
gives the same situation WITH labels, so transfer strategies can be measured:

  A  US-only model, US-calibrated threshold          (what France gets today)
  B  A, but with the best threshold for India        (value of re-tuning)
  C  self-training: a model fit on India pseudo-labels from A (confident
     positives / confident negatives), India folds 0-3, scored on fold 4
  D  reference: the production model trained WITH India labels

All numbers are macro-F0.5 on India fold-4 entities with the real labels.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
import joblib
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_train_gpu import group_stats

TAG = "_us"
CTY = "India"


def xgb_avg(paths, X):
    import xgboost as xgb
    acc = np.zeros(len(X), np.float64)
    for p in paths:
        b = xgb.Booster(); b.load_model(p); b.set_param({"device": "cuda"})
        acc += b.inplace_predict(X, iteration_range=(0, b.best_iteration + 1))
    return (acc / len(paths)).astype(np.float32)


def main():
    import xgboost as xgb
    t0 = time.time()
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    ids = tb.column("entity_id").to_pylist()
    gt = read_ground_truth("train"); N = np.array([len(gt[e]) for e in ids], np.int32)
    ef = np.array([fold_of(e) for e in ids], np.int8)

    Q, T, S, Y, Xs = [], [], [], [], []
    for ts in (2, 3):
        p = f"{WORK}/feat/train_S{ts}_{CTY}.npz"
        z = np.load(p)
        X = np.load(p.replace(".npz", "_X.npy"), mmap_mode="r")
        Q.append(z["q"].astype(np.int64)); T.append(z["t"].astype(np.int64))
        S.append(np.full(len(z["q"]), ts, np.int8)); Y.append(z["y"].astype(np.int8)); Xs.append((X, ts))
    q = np.concatenate(Q); t = np.concatenate(T); src = np.concatenate(S); y = np.concatenate(Y)
    tkey = t + src.astype(np.int64) * np.int64(10_000_000)
    print(f"{CTY}: pairs={len(q):,} ({time.time()-t0:.0f}s)", flush=True)

    def chunks(extra):
        """(feature chunk, global row offset) pairs over all pairs"""
        off = 0
        for Xc in feats(extra):
            yield Xc, off
            off += len(Xc)

    def feats(extra=None):
        out, off = [], 0
        for X, ts in Xs:
            n = X.shape[0]
            for s0 in range(0, n, 4_000_000):
                e0 = min(s0 + 4_000_000, n)
                cols = [np.asarray(X[s0:e0], np.float32), np.full((e0 - s0, 1), ts, np.float32)]
                if extra is not None:
                    cols.append(extra[off + s0:off + e0])
                yield np.hstack(cols)
            off += n

    m1 = [f"{ROOT}/models/xgb{TAG}_stage1_fold{f}.json" for f in range(5)]
    m2 = [f"{ROOT}/models/xgb{TAG}_stage2_fold{f}.json" for f in range(5)]
    cache = f"{WORK}/pred/transfer_cache{TAG}_{CTY}.npz"          # scoring 86M pairs takes ~50 min
    if os.path.exists(cache):
        zc = np.load(cache); p1, p2 = zc["p1"], zc["p2"]
        tr_, tb_, ts_, tc = group_stats(tkey, p1); qr_, qb, qs_, qc = group_stats(q, p1)
        comp = np.column_stack([tr_, tb_, p1 - tb_, tb_ - ts_, tc, qr_, qb, p1 - qb, qb - qs_, qc, p1]).astype(np.float32)
    else:
        p1 = np.concatenate([xgb_avg(m1, X) for X in feats()])
        tr_, tb_, ts_, tc = group_stats(tkey, p1); qr_, qb, qs_, qc = group_stats(q, p1)
        comp = np.column_stack([tr_, tb_, p1 - tb_, tb_ - ts_, tc, qr_, qb, p1 - qb, qb - qs_, qc, p1]).astype(np.float32)
        p2 = np.concatenate([xgb_avg(m2, X) for X in feats(comp)])
        np.savez(cache, p1=p1, p2=p2)
    zu = np.load(f"{WORK}/pred/oof{TAG}.npz")
    iso_us = IsotonicRegression(out_of_bounds="clip").fit(zu["p2"], zu["y"])
    cal = iso_us.predict(p2).astype(np.float32)
    print(f"US-only model applied to {CTY} ({time.time()-t0:.0f}s)", flush=True)

    cty_all = np.asarray(tb.column("country").to_pylist())
    ent4 = np.flatnonzero((ef == 4) & (cty_all == CTY))       # every India fold-4 S1, singletons included
    rm = np.full(len(ids), -1, np.int64); rm[ent4] = np.arange(len(ent4))
    m4 = rm[q] >= 0

    def f05(score, th):
        u = D.unique_target_filter(tkey, score)
        k = u & (score >= th) & m4
        return D.macro_f05_fast(rm[q[k]], np.ones(int(k.sum()), bool), y[k], len(ent4), N[ent4])

    us_th = json.load(open(f"{ROOT}/eda/eval{TAG}.json"))["fold4 all"][0] if os.path.exists(f"{ROOT}/eda/eval{TAG}.json") else 0.6
    ths = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97)
    A = f05(cal, us_th)
    Bs = {th: f05(cal, th) for th in ths}
    print(f"A  US-only model, US threshold {us_th}:           {A:.5f}", flush=True)
    print("B  US-only model, threshold sweep:  " + "  ".join(f"{th}:{v:.5f}" for th, v in Bs.items()), flush=True)

    # ---- C: self-training on India pseudo-labels (folds 0-3), eval fold 4 ----
    u = D.unique_target_filter(tkey, cal)
    fold = ef[q]
    pos = u & (cal >= 0.99); neg = cal <= 0.01                   # same cutoffs as run_selftrain_fr.py
    lab = np.where(pos, 1, np.where(neg, 0, -1)).astype(np.int8)
    tr = (fold != 4) & (lab >= 0)
    rng = np.random.default_rng(0)
    tr &= (lab == 1) | (rng.random(len(lab)) < 0.12)            # subsample easy negatives (fits an 11 GB GPU)
    extra = np.column_stack([comp, p2[:, None]])                 # same features as France
    Xtr = np.vstack([Xc[tr[o0:o0 + len(Xc)]] for Xc, o0 in chunks(extra)])
    print(f"C  pseudo-labels: pos={int((tr&(lab==1)).sum()):,} neg={int((tr&(lab==0)).sum()):,} "
          f"(pseudo-label precision vs truth: pos {y[tr&(lab==1)].mean():.4f}, "
          f"neg {1-y[tr&(lab==0)].mean():.4f}) ({time.time()-t0:.0f}s)", flush=True)
    params = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_bin=128,
                  learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5.0, eval_metric="logloss")
    import gc; gc.collect()
    d = xgb.QuantileDMatrix(Xtr, label=lab[tr], max_bin=128); del Xtr; gc.collect()
    bst = xgb.train(params, d, 600); del d; gc.collect()
    ps = np.concatenate([bst.inplace_predict(Xc).astype(np.float32) for Xc, _ in chunks(extra)])
    Cs = {th: f05(ps, th) for th in ths}
    print("C  self-trained, threshold sweep:   " + "  ".join(f"{th}:{v:.5f}" for th, v in Cs.items()), flush=True)
    blend = (0.5 * cal + 0.5 * ps).astype(np.float32)
    Bl = {th: f05(blend, th) for th in ths}
    print("C' blend(US-only, self-trained):    " + "  ".join(f"{th}:{v:.5f}" for th, v in Bl.items()), flush=True)
    json.dump({"A": A, "B": Bs, "C": Cs, "Cblend": Bl}, open(f"{ROOT}/eda/transfer_eval.json", "w"), indent=1)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
