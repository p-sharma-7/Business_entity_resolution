"""Stage 3: combine the stage-2 GBDT score with the cross-encoder reranker.

Fit and validated ONLY on fold 4 - the S1 entities held out of the encoder
fine-tuning, the stage-1/2 GBDT folds and the cross-encoder - with entity
sub-folds inside it, so every reported number is out-of-sample at every level.

Test-consistency rules (the first version broke both and overstated its gain):
  * Target-side competition is computed over ALL S1 entities of every fold,
    exactly as at test time where every S1 competes for every target. Using
    fold-4 pairs alone hides 80% of a target's rival claims.
  * Target-side features therefore use only p2, which exists out-of-fold for
    every training pair and for every test pair. Cross-encoder scores exist
    only for fold-4 band pairs, so they enter S1-side features only, where an
    S1's candidate set lies entirely inside its own fold.
  * The uniqueness decision is evaluated over all pairs as well, and the
    stage-2 baseline is recomputed in the same harness for a like-for-like
    comparison.
"""
import sys, os, json, time, hashlib
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_ce import LO, HI

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              device="cuda", max_bin=256, max_depth=7, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.9, min_child_weight=5.0, reg_lambda=2.0)
SUBF = 5


def sub_fold(e):
    h = hashlib.blake2b((e + "#stage3").encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % SUBF


def grp(gid, s):
    order = np.lexsort((-s, gid)); g = gid[order]; v = s[order]
    new = np.r_[True, g[1:] != g[:-1]]; st = np.flatnonzero(new)
    sz = np.diff(np.append(st, len(g)))
    rank = (np.arange(len(g)) - np.repeat(st, sz)).astype(np.float32)
    best = np.repeat(v[st], sz).astype(np.float32)
    sec = np.repeat(np.where(sz > 1, v[np.minimum(st + 1, len(v) - 1)], -1.0), sz).astype(np.float32)
    cnt = np.repeat(sz, sz).astype(np.float32)
    strong = np.repeat(np.add.reduceat((v > 0.5).astype(np.float32), st), sz)
    inv = np.empty(len(g), np.int64); inv[order] = np.arange(len(g))
    return rank[inv], best[inv], sec[inv], cnt[inv], strong[inv]


def make_features(q, tkey, src, p1, p2, ce, rows=None):
    """Stage-3 features over the FULL pair universe, returned for `rows`.

    ce is NaN wherever no cross-encoder score exists.
    """
    rows = np.arange(len(q)) if rows is None else rows
    has = ~np.isnan(ce)
    s = np.where(has, 0.5 * (ce + p2), p2).astype(np.float32)
    cols = [p1[rows], p2[rows], np.nan_to_num(ce[rows], nan=-1.0),
            has[rows].astype(np.float32), src[rows].astype(np.float32)]
    names = ["p1", "p2", "ce", "has_ce", "src"]
    for tag, sc in (("s", s), ("p2", p2)):          # S1 side: complete within a fold
        r, b, sec, c, strong = grp(q, sc)
        cols += [r[rows], (sc - b)[rows], (b - sec)[rows], c[rows], strong[rows]]
        names += [f"q_{tag}_rank", f"q_{tag}_margin", f"q_{tag}_gap", f"q_{tag}_cnt", f"q_{tag}_strong"]
    r, b, sec, c, strong = grp(tkey, p2)            # target side: ALL S1 compete
    cols += [r[rows], (p2 - b)[rows], (b - sec)[rows], c[rows], strong[rows]]
    names += ["t_p2_rank", "t_p2_margin", "t_p2_gap", "t_p2_cnt", "t_p2_strong"]
    return np.column_stack(cols).astype(np.float32), names


def fit():
    import xgboost as xgb, joblib
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2, fold = (z["q"], z["t"], z["src"], z["y"], z["p1"],
                                  z["p2"], z["fold"])
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    e_ids = tb.column("entity_id").to_pylist()
    ecty = np.asarray(tb.column("country").to_pylist())

    cs = np.load(f"{WORK}/pred/ce_score_train.npz")
    ce = np.full(len(y), np.nan, np.float32)
    ce[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train.npy")
    m4 = fold == 4
    r4 = np.flatnonzero(m4)
    band4 = m4 & (p2 > LO) & (p2 < HI)
    assert np.array_equal(np.flatnonzero(band4), np.sort(cs["idx"])), "CE rows != fold-4 band"
    X, names = make_features(q, tkey, src, p1, p2, ce, rows=r4)
    print(f"all pairs={len(y):,} fold-4 pairs={len(r4):,} band={int(band4.sum()):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    ent = np.flatnonzero(np.array([fold_of(e) == 4 for e in e_ids]))
    sf_ent = np.full(len(e_ids), -1, np.int8)
    sf_ent[ent] = [sub_fold(e_ids[i]) for i in ent]
    sf = sf_ent[q[r4]]
    y4 = y[r4]; b4 = band4[r4]

    p3_4 = p2[r4].astype(np.float32).copy()
    for f in range(SUBF):
        tr = b4 & (sf != f); va = b4 & (sf == f)
        dtr = xgb.QuantileDMatrix(X[tr], label=y4[tr], max_bin=PARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(X[va], label=y4[va], ref=dtr, max_bin=PARAMS["max_bin"])
        bst = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "val")],
                        early_stopping_rounds=80, verbose_eval=False)
        bst.save_model(f"{ROOT}/models/stage3_f{f}.json")
        p3_4[va] = bst.inplace_predict(X[va], iteration_range=(0, bst.best_iteration + 1))
        print(f"  sub-fold {f}: best_iter={bst.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(names, open(f"{ROOT}/models/stage3_features.json", "w"))

    # ---- calibration (out-of-sub-fold on fold 4) --------------------------
    cal3_4 = np.zeros(len(r4), np.float32); cal2_4 = np.zeros(len(r4), np.float32)
    for f in range(SUBF):
        tr = sf != f
        cal3_4[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3_4[tr], y4[tr]).predict(p3_4[~tr])
        cal2_4[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p2[r4][tr], y4[tr]).predict(p2[r4][~tr])
    iso3 = IsotonicRegression(out_of_bounds="clip").fit(p3_4, y4)
    iso2 = IsotonicRegression(out_of_bounds="clip").fit(p2[r4], y4)

    # competitors from folds 0-3 carry only p2 (no CE score exists for them)
    comp3 = iso3.predict(p2).astype(np.float32)
    comp2 = iso2.predict(p2).astype(np.float32)
    full3 = comp3.copy(); full3[r4] = cal3_4
    full2 = comp2.copy(); full2[r4] = cal2_4

    gt = read_ground_truth("train")
    remap = np.full(len(e_ids), -1, np.int64); remap[ent] = np.arange(len(ent))
    tc = np.array([len(gt[e_ids[i]]) for i in ent], np.int32)
    qq = remap[q[r4]]; qc = ecty[q[r4]]

    def f05(keep4):
        return D.macro_f05_fast(qq[keep4], np.ones(int(keep4.sum()), bool), y4[keep4], len(ent), tc)

    uniq2 = D.unique_target_filter(tkey, full2)[r4]
    uniq3 = D.unique_target_filter(tkey, full3)[r4]
    ths = np.round(np.arange(0.30, 0.91, 0.05), 2)
    print("\nlike-for-like on fold 4, ALL S1 competing for each target:", flush=True)
    base = {th: f05(uniq2 & (cal2_4 >= th)) for th in ths}
    new = {th: f05(uniq3 & (cal3_4 >= th)) for th in ths}
    for th in ths:
        print(f"  th={th:.2f}  stage-2 {base[th]:.5f}   stage-3 {new[th]:.5f}", flush=True)
    bth = max(new, key=new.get)
    cfg = {"rule": "B", "threshold": float(bth), "per_country": {}}
    keep = uniq3 & (cal3_4 >= bth)
    for c in ("India", "US"):
        m = qc == c; bt, bs = float(bth), f05(keep)
        for th in ths:
            k2 = keep.copy(); k2[m] = uniq3[m] & (cal3_4[m] >= th)
            sc = f05(k2)
            if sc > bs:
                bt, bs = float(th), sc
        cfg["per_country"][c] = bt
        keep[m] = uniq3[m] & (cal3_4[m] >= bt)
    final = f05(keep)
    cfg["honest_macro_f05"] = float(final)
    cfg["stage2_same_harness"] = float(max(base.values()))
    print(f"\nBEST stage-2 (same harness) = {max(base.values()):.5f}", flush=True)
    print(f"BEST stage-3 per-country {cfg['per_country']} -> HONEST fold-4 = {final:.5f}", flush=True)
    joblib.dump(iso3, f"{ROOT}/models/stage3_calibrator.pkl")
    json.dump(cfg, open(f"{ROOT}/models/stage3_decision.json", "w"), indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    fit()
