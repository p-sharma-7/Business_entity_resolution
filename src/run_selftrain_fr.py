"""France-specific pair model via self-training (run only if the US->India
test bed shows self-training helps an unseen country with real labels).

Pseudo-labels come from the current best France scores (V6: stage 3 with the
French-adapted cross-encoder): confident positives (cal >= POS and winning
their target) and confident negatives (cal <= NEG). A GBDT is cross-fitted
over French S1 entities (5 folds): each pair is scored by a model that never
saw that pair's own pseudo-label, so it cannot simply echo its training label.
The self-trained score is then blended with V6.

Writes work/pred/fr_selftrain.npz (q, t, src, pc) over the pruned French pairs,
consumable by run_build_mix.py.
"""
import sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK
from er import decide as D
from run_train_gpu import group_stats

POS, NEG, W_SELF = 0.99, 0.01, float(os.environ.get("ER_W_SELF", "0.5"))


def key(q, t, s):
    return (q.astype(np.int64) << np.int64(33)) | (t.astype(np.int64) << np.int64(2)) | s.astype(np.int64)


def main():
    import xgboost as xgb
    t0 = time.time()
    zf = np.load(f"{WORK}/pred/test_final_V6.npz")
    q, t, src, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
    s1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["entity_id", "country"])
    ids = s1.column("entity_id").to_pylist(); c1 = np.asarray(s1.column("country").to_pylist())
    fr = np.flatnonzero(c1[q] == "France")
    q, t, src, cal = q[fr], t[fr], src[fr], cal[fr]
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)

    # features for these pairs from the France feature files
    X = np.empty((len(q), 150), np.float32)
    got = np.zeros(len(q), bool)
    pk = key(q, t, src)
    for ts in (2, 3):
        p = f"{WORK}/feat/test_S{ts}_France.npz"
        z = np.load(p); XM = np.load(p.replace(".npz", "_X.npy"), mmap_mode="r")
        fk = key(z["q"], z["t"], np.full(len(z["q"]), ts)); o = np.argsort(fk); fs = fk[o]
        m = np.flatnonzero(src == ts)
        pos = np.clip(np.searchsorted(fs, pk[m]), 0, len(fs) - 1); hit = fs[pos] == pk[m]
        rows = o[pos[hit]]; srt = np.argsort(rows)
        X[m[hit][srt]] = np.asarray(XM[rows[srt]], np.float32)
        got[m[hit]] = True
    assert got.all(), f"{(~got).sum()} France pairs without features"
    zs = np.load(f"{WORK}/pred/test_scores.npz")
    sk = key(zs["q"], zs["t"], zs["src"]); o = np.argsort(sk)
    pos = np.clip(np.searchsorted(sk[o], pk), 0, len(sk) - 1)
    p1 = zs["p1"][o][pos].astype(np.float32); p2 = zs["p2"][o][pos].astype(np.float32)
    tr_, tb, tsec, tc = group_stats(tk, p1); qr_, qb, qsec, qc = group_stats(q, p1)
    F = np.hstack([X, src[:, None].astype(np.float32),
                   np.column_stack([tr_, tb, p1 - tb, tb - tsec, tc, qr_, qb, p1 - qb, qb - qsec, qc, p1, p2])])
    print(f"France pairs={len(q):,} features={F.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    win = D.unique_target_filter(tk, cal)
    lab = np.where(win & (cal >= POS), 1, np.where(cal <= NEG, 0, -1)).astype(np.int8)
    fold = np.array([int(hashlib.md5(ids[e].encode()).hexdigest(), 16) % 5 for e in q], np.int8)
    rng = np.random.default_rng(0)
    params = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_bin=128,
                  learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5.0, eval_metric="logloss")
    ps = np.zeros(len(q), np.float32)
    for f in range(5):
        tr = (fold != f) & (lab >= 0) & ((lab == 1) | (rng.random(len(q)) < 0.25))
        d = xgb.QuantileDMatrix(F[tr], label=lab[tr], max_bin=128)
        b = xgb.train(params, d, 600)
        m = fold == f
        ps[m] = b.inplace_predict(F[m])
        print(f"  fold {f}: trained on {int(tr.sum()):,} pseudo-labels ({time.time()-t0:.0f}s)", flush=True)
    pc = (W_SELF * ps + (1 - W_SELF) * cal).astype(np.float32)
    np.savez(f"{WORK}/pred/fr_selftrain.npz", q=q, t=t, src=src, pc=pc, ps=ps, cal=cal)
    print(f"agreement with V6 at 0.97: self>=0.97 & V6>=0.97: {int(((ps>=0.97)&(cal>=0.97)).sum()):,}; "
          f"V6 only: {int(((ps<0.97)&(cal>=0.97)).sum()):,}; self only: {int(((ps>=0.97)&(cal<0.97)).sum()):,}", flush=True)


if __name__ == "__main__":
    main()
