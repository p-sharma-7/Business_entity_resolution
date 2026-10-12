"""Hypothesis test: is the test S1 file a subsample, leaving orphan targets?

Test vs train, same model: target records per S1 x1.23, targets with a
confident claimant x0.81 in every country, contested targets x~0.6 (~0.81^2).
All three match an S1 file subsampled to ~81% with every S2/S3 record kept.

If so, ~19% of targets lose their true parent. In training the uniqueness
filter hands a doubly-claimed target to its true parent; with the parent
absent a look-alike S1 wins it unopposed - a false merge validation never saw.

Simulation on the honest fold-4 slice: drop a random fraction of S1 entities
from ALL folds (their S2/S3 records stay), rebuild the stage-3 universe and
the uniqueness decision without them, and score the remaining fold-4
entities. drop=0 reproduces the reference.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_stage3 import make_features, sub_fold, SUBF
from run_prune_sim import prune_mask


def main():
    import xgboost as xgb
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2, fold = (z["q"], z["t"], z["src"], z["y"], z["p1"], z["p2"], z["fold"])
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    e_ids = tb.column("entity_id").to_pylist(); ecty = np.asarray(tb.column("country").to_pylist())
    cs = np.load(f"{WORK}/pred/ce_score_train.npz")
    ce = np.full(len(y), np.nan, np.float32); ce[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train.npy")
    B = []
    for f in range(SUBF):
        b = xgb.Booster(); b.load_model(f"{ROOT}/models/stage3_f{f}.json"); b.set_param({"device": "cuda"})
        B.append(b)
    ent_fold = np.array([fold_of(e) for e in e_ids], np.int8)
    sf_ent = np.full(len(e_ids), -1, np.int8)
    f4 = np.flatnonzero(ent_fold == 4)
    sf_ent[f4] = [sub_fold(e_ids[i]) for i in f4]
    gt = read_ground_truth("train")
    ncount = np.array([len(gt[e]) for e in e_ids], np.int32)
    prune = prune_mask(q, src, p1, 10)
    rng = np.random.default_rng(123)
    u01 = rng.random(len(e_ids))
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

    ths = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    res = {}
    for drop in (0.0, 0.10, 0.19, 0.25):
        present = u01 >= drop                         # S1 entities kept
        U = np.flatnonzero(prune & present[q])
        qu, tku, su, yu, p1u, p2u, ceu = q[U], tkey[U], src[U], y[U], p1[U], p2[U], ce[U]
        r4 = np.flatnonzero(fold[U] == 4)
        X, _ = make_features(qu, tku, su, p1u, p2u, ceu, rows=r4)
        y4 = yu[r4]; has = ~np.isnan(ceu[r4]); sf = sf_ent[qu[r4]]
        p3 = p2u[r4].astype(np.float32).copy()
        for f in range(SUBF):
            m = has & (sf == f)
            if m.any():
                p3[m] = B[f].inplace_predict(X[m], iteration_range=(0, B[f].best_iteration + 1))
        cal = np.zeros(len(r4), np.float32)
        for f in range(SUBF):
            tr = sf != f
            cal[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3[tr], y4[tr]).predict(p3[~tr])
        iso = IsotonicRegression(out_of_bounds="clip").fit(p3, y4)
        full = iso.predict(p2u).astype(np.float32); full[r4] = cal
        uniq = D.unique_target_filter(tku, full)[r4]
        # evaluate on the fold-4 S1 entities that are still present; their
        # true-match count is unchanged (their own targets all still exist)
        ev = np.flatnonzero((ent_fold == 4) & present)
        remap = np.full(len(e_ids), -1, np.int64); remap[ev] = np.arange(len(ev))
        qq = remap[qu[r4]]
        line = []
        for th in ths:
            sel = uniq & (cal >= th)
            s = D.macro_f05_fast(qq[sel], np.ones(int(sel.sum()), bool), y4[sel], len(ev), ncount[ev])
            fp = int((sel & (y4 == 0)).sum())
            line.append((th, s, fp))
        res[drop] = line
        best = max(line, key=lambda x: x[1])
        print(f"drop={drop:.2f}  " + "  ".join(f"th{th}:{s:.5f}" for th, s, _ in line)
              + f"  | best th={best[0]} -> {best[1]:.5f}  FP@0.7={dict((a,c) for a,_,c in line)[0.7]:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(f"{ROOT}/eda/dropout_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
