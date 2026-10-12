"""Stage 3, version b: adds collective (sibling) evidence. A/B on fold 4.

Both variants are fit identically on the PRUNED universe (as used at test),
with entity sub-folds inside the honest fold 4:
  base  - the current stage-3 feature set
  sib   - base + sibling features: how closely the candidate resembles the
          S1's other confident matches (encoder cosine, name and combined
          views), and how that compares with the other S1s claiming the same
          target.
The variant with the higher honest macro-F0.5 is saved as the stage-3b model.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_stage3 import make_features, sub_fold, grp, SUBF, PARAMS

SIB_COLS = ["sib_cn_max", "sib_cn_mean", "sib_cc_max", "sib_cc_mean", "nsib"]


def sib_features(sib, tkey_u, q_u, rows):
    """Sibling columns + their S1-side and target-side competition, for rows."""
    cols, names = [], []
    for c in SIB_COLS:
        v = np.nan_to_num(sib[c], nan=-1.0).astype(np.float32)
        cols.append(v[rows]); names.append(c)
    v = np.nan_to_num(sib["sib_cc_max"], nan=-1.0).astype(np.float32)
    for side, gid in (("t", tkey_u), ("q", q_u)):
        r, b, sec, cnt, strong = grp(gid, v)
        cols += [r[rows], (v - b)[rows], (b - sec)[rows]]
        names += [f"{side}_sib_rank", f"{side}_sib_margin", f"{side}_sib_gap"]
    return np.column_stack(cols).astype(np.float32), names


def main():
    import xgboost as xgb, joblib
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2, fold = (z["q"], z["t"], z["src"], z["y"], z["p1"], z["p2"], z["fold"])
    s = np.load(f"{WORK}/pred/sib_train.npz")
    U = s["U"]
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    e_ids = tb.column("entity_id").to_pylist(); ecty = np.asarray(tb.column("country").to_pylist())
    cs = np.load(f"{WORK}/pred/ce_score_train.npz")
    ce = np.full(len(y), np.nan, np.float32); ce[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train.npy")

    qu, tku, su, yu = q[U], tkey[U], src[U], y[U]
    loc4 = np.flatnonzero(fold[U] == 4)
    Xb, nb = make_features(qu, tku, su, p1[U], p2[U], ce[U], rows=loc4)
    Xs, ns = sib_features(s, tku, qu, loc4)
    y4 = yu[loc4]; has = ~np.isnan(ce[U][loc4])
    ent = np.flatnonzero(np.array([fold_of(e) == 4 for e in e_ids]))
    sf_ent = np.full(len(e_ids), -1, np.int8); sf_ent[ent] = [sub_fold(e_ids[i]) for i in ent]
    sf = sf_ent[qu[loc4]]
    gt = read_ground_truth("train")
    remap = np.full(len(e_ids), -1, np.int64); remap[ent] = np.arange(len(ent))
    tc = np.array([len(gt[e_ids[i]]) for i in ent], np.int32)
    qq = remap[qu[loc4]]; qc = ecty[qu[loc4]]
    print(f"pruned universe={len(U):,} fold-4 rows={len(loc4):,} band={int(has.sum()):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for tag, X, names in (("base", Xb, nb), ("sib", np.hstack([Xb, Xs]), nb + ns)):
        p3 = p2[U][loc4].astype(np.float32).copy()
        boosters = []
        for f in range(SUBF):
            tr = has & (sf != f); va = has & (sf == f)
            dtr = xgb.QuantileDMatrix(X[tr], label=y4[tr], max_bin=PARAMS["max_bin"])
            dva = xgb.QuantileDMatrix(X[va], label=y4[va], ref=dtr, max_bin=PARAMS["max_bin"])
            b = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=80,
                          verbose_eval=False)
            p3[va] = b.inplace_predict(X[va], iteration_range=(0, b.best_iteration + 1))
            boosters.append(b)
        cal = np.zeros(len(loc4), np.float32)
        for f in range(SUBF):
            tr = sf != f
            cal[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3[tr], y4[tr]).predict(p3[~tr])
        iso = IsotonicRegression(out_of_bounds="clip").fit(p3, y4)
        full = iso.predict(p2[U]).astype(np.float32); full[loc4] = cal
        uniq = D.unique_target_filter(tku, full)[loc4]
        best = (None, -1)
        for th in np.round(np.arange(0.5, 0.96, 0.05), 2):
            sel = uniq & (cal >= th)
            sc = D.macro_f05_fast(qq[sel], np.ones(int(sel.sum()), bool), y4[sel], len(ent), tc)
            if sc > best[1]:
                best = (float(th), sc)
        results[tag] = best
        print(f"[{tag}] features={len(names)}  best th={best[0]}  HONEST fold-4 macro-F0.5={best[1]:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if tag == "sib":
            imp = sorted(boosters[0].get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:12]
            print("   top gain:", [(names[int(k[1:])], round(v, 1)) for k, v in imp], flush=True)
        out = f"{ROOT}/models/stage3b_{tag}"
        os.makedirs(out, exist_ok=True)
        for f, b in enumerate(boosters):
            b.save_model(f"{out}/f{f}.json")
        joblib.dump(iso, f"{out}/calibrator.pkl")
        json.dump({"features": names, "threshold": best[0], "honest_macro_f05": best[1]},
                  open(f"{out}/config.json", "w"), indent=2)
    json.dump(results, open(f"{ROOT}/eda/stage3b_ab.json", "w"), indent=2)
    print(f"\nA/B: base {results['base'][1]:.5f}  vs  sib {results['sib'][1]:.5f}  "
          f"(delta {results['sib'][1]-results['base'][1]:+.5f})")


if __name__ == "__main__":
    main()
