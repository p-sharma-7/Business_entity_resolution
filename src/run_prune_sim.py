"""What does a learned pruner cost? Measured on the honest fold-4 slice.

The submission zip is capped at 512 MB, and candidate_pairs.tsv at ~101
candidates per S1 (175M ids) compresses to ~950 MB. Random 9-10 digit ids are
near-incompressible, so the candidate set itself must shrink.

Pruner: the stage-1 GBDT keeps each S1's top-K candidates per source by p1
(plus any pair with p1 >= P1_FLOOR). The final matcher - stage 2, the
cross-encoder and stage 3 - then only ever sees surviving pairs, so its
set-level statistics (target-side competition in stage 3) are recomputed over
the pruned universe exactly as they would be at test time.

Reports, per K: pairs kept, positive recall of the pruner, and the honest
fold-4 macro-F0.5 of the full pipeline run on the pruned set.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from er.candgen import topk_mask
from run_ce import LO, HI
from run_stage3 import make_features, sub_fold, SUBF

P1_FLOOR = 0.05


def prune_mask(q, src, p1, k):
    g = q.astype(np.int64) * 4 + src.astype(np.int64)
    return topk_mask(g, p1, k) | (p1 >= P1_FLOOR)


def main():
    import xgboost as xgb
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2, fold = (z["q"], z["t"], z["src"], z["y"], z["p1"], z["p2"], z["fold"])
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    e_ids = tb.column("entity_id").to_pylist(); ecty = np.asarray(tb.column("country").to_pylist())
    cs = np.load(f"{WORK}/pred/ce_score_train.npz")
    ce = np.full(len(y), np.nan, np.float32)
    ce[cs["idx"]] = np.load(f"{WORK}/pred/ce_scores_train.npy")
    boosters = []
    for f in range(SUBF):
        b = xgb.Booster(); b.load_model(f"{ROOT}/models/stage3_f{f}.json"); b.set_param({"device": "cuda"})
        boosters.append(b)
    cfg = json.load(open(f"{ROOT}/models/stage3_decision.json"))

    ent = np.flatnonzero(np.array([fold_of(e) == 4 for e in e_ids]))
    sf_ent = np.full(len(e_ids), -1, np.int8); sf_ent[ent] = [sub_fold(e_ids[i]) for i in ent]
    gt = read_ground_truth("train")
    remap = np.full(len(e_ids), -1, np.int64); remap[ent] = np.arange(len(ent))
    tc = np.array([len(gt[e_ids[i]]) for i in ent], np.int32)
    n_s1 = len(e_ids)
    print(f"loaded ({time.time()-t0:.0f}s); reference unpruned honest = {cfg['honest_macro_f05']:.5f}", flush=True)

    res = {}
    for k in (6, 8, 10, 12, 15):
        keep = prune_mask(q, src, p1, k)
        U = np.flatnonzero(keep)                       # pruned universe, all folds
        qu, tku, su, yu, p1u, p2u, ceu = q[U], tkey[U], src[U], y[U], p1[U], p2[U], ce[U]
        r4 = np.flatnonzero(fold[U] == 4)
        X, _ = make_features(qu, tku, su, p1u, p2u, ceu, rows=r4)
        y4 = yu[r4]; has = ~np.isnan(ceu[r4]); sf = sf_ent[qu[r4]]
        p3 = p2u[r4].astype(np.float32).copy()
        for f in range(SUBF):
            m = has & (sf == f)
            if m.any():
                p3[m] = boosters[f].inplace_predict(X[m], iteration_range=(0, boosters[f].best_iteration + 1))
        cal = np.zeros(len(r4), np.float32)
        for f in range(SUBF):
            tr = sf != f
            cal[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3[tr], y4[tr]).predict(p3[~tr])
        iso = IsotonicRegression(out_of_bounds="clip").fit(p3, y4)
        full = iso.predict(p2u).astype(np.float32); full[r4] = cal
        uniq = D.unique_target_filter(tku, full)[r4]
        th = np.full(len(r4), cfg["threshold"], np.float32)
        qc = ecty[qu[r4]]
        for c, v in cfg["per_country"].items():
            th[qc == c] = v
        sel = uniq & (cal >= th)
        qq = remap[qu[r4]]
        f05 = D.macro_f05_fast(qq[sel], np.ones(int(sel.sum()), bool), y4[sel], len(ent), tc)
        rec = float(y[keep].sum() / y.sum())
        res[k] = dict(pairs=int(keep.sum()), per_s1=float(keep.sum() / n_s1),
                      pruner_recall=rec, honest_f05=float(f05))
        print(f"K={k:>2}: pairs={keep.sum():>12,} per_S1={keep.sum()/n_s1:5.1f} "
              f"pruner_recall={rec:.5f}  HONEST fold-4 macro-F0.5={f05:.5f} ({time.time()-t0:.0f}s)", flush=True)

    zt = np.load(f"{WORK}/pred/test_scores.npz")
    n1 = len(pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["entity_id"]))
    for k in res:
        n = int(prune_mask(zt["q"], zt["src"], zt["p1"], k).sum())
        res[k]["test_pairs"] = n
        print(f"TEST K={k:>2}: pairs={n:,} per_S1={n/n1:.1f}  est. zip for candidates "
              f"~{n*5.4/1e6:.0f} MB", flush=True)
    json.dump(res, open(f"{ROOT}/eda/prune_sim.json", "w"), indent=2)


if __name__ == "__main__":
    main()
