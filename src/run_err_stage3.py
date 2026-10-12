"""Where does the final (stage-3) pipeline still lose F0.5 on the honest slice?

Rebuilds the fold-4 stage-3 predictions exactly as validated (pruned K=10
universe, all S1 competing), saves them, and decomposes the lost score by
error type so the next change is chosen from evidence.
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
    ent_fold = np.array([fold_of(e) for e in e_ids], np.int8)
    f4 = np.flatnonzero(ent_fold == 4)
    sf_ent = np.full(len(e_ids), -1, np.int8); sf_ent[f4] = [sub_fold(e_ids[i]) for i in f4]
    gt = read_ground_truth("train"); ncount = np.array([len(gt[e]) for e in e_ids], np.int32)
    cfg = json.load(open(f"{ROOT}/models/stage3_decision.json"))

    U = np.flatnonzero(prune_mask(q, src, p1, 10))
    r4 = U[fold[U] == 4]
    X, _ = make_features(q[U], tkey[U], src[U], p1[U], p2[U], ce[U],
                         rows=np.flatnonzero(fold[U] == 4))
    has = ~np.isnan(ce[r4]); sf = sf_ent[q[r4]]; y4 = y[r4]
    p3 = p2[r4].astype(np.float32).copy()
    for f in range(SUBF):
        b = xgb.Booster(); b.load_model(f"{ROOT}/models/stage3_f{f}.json"); b.set_param({"device": "cuda"})
        m = has & (sf == f)
        p3[m] = b.inplace_predict(X[m], iteration_range=(0, b.best_iteration + 1))
    cal = np.zeros(len(r4), np.float32)
    for f in range(SUBF):
        tr = sf != f
        cal[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p3[tr], y4[tr]).predict(p3[~tr])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p3, y4)
    full = iso.predict(p2[U]).astype(np.float32)
    pos_in_U = np.searchsorted(U, r4); full[pos_in_U] = cal
    uniq = D.unique_target_filter(tkey[U], full)[pos_in_U]
    th = np.where(np.isin(ecty[q[r4]], list(cfg["per_country"])),
                  np.vectorize(lambda c: cfg["per_country"].get(c, cfg["threshold"]))(ecty[q[r4]]),
                  cfg["threshold"]).astype(np.float32)
    sel = uniq & (cal >= th)
    np.savez(f"{WORK}/pred/fold4_stage3.npz", rows=r4, cal=cal, p3=p3, uniq=uniq, sel=sel, th=th)
    print(f"rebuilt fold-4 stage-3 predictions ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-entity decomposition ------------------------------------------
    qe = q[r4]
    k = np.bincount(qe[sel], minlength=len(e_ids)).astype(float)
    tp = np.bincount(qe[sel], weights=y4[sel].astype(float), minlength=len(e_ids))
    N = ncount.astype(float)
    sc = np.where(N == 0, (k == 0).astype(float), np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * N, 1e-9)))
    fp = k - tp
    # true links that never reached the pruned candidate set
    cand_pos = np.bincount(qe[y4 == 1], minlength=len(e_ids)).astype(float)
    miss_cand = N - cand_pos
    fn_model = cand_pos - tp
    E = f4
    loss = 1 - sc[E]; tot = loss.sum()
    print(f"\nfold-4 entities={len(E):,}  macro-F0.5={sc[E].mean():.5f}  total loss={tot:.0f} "
          f"(= {tot/len(E):.5f} of score)", flush=True)
    cats = {
        "singleton given a match":        (N[E] == 0) & (k[E] > 0),
        "has matches, predicted none":    (N[E] > 0) & (k[E] == 0),
        "missed links only (no FP)":      (N[E] > 0) & (k[E] > 0) & (fp[E] == 0) & (tp[E] < N[E]),
        "false links only (no miss)":     (N[E] > 0) & (fp[E] > 0) & (tp[E] == N[E]),
        "both false and missed links":    (N[E] > 0) & (fp[E] > 0) & (tp[E] < N[E]) & (k[E] > 0),
    }
    for name, m in cats.items():
        print(f"  {name:<30} entities={int(m.sum()):>6,}  loss share={loss[m].sum()/tot:6.1%}", flush=True)
    print(f"  missed links: {int((N[E]-tp[E]).sum()):,} total, of which never in candidates: "
          f"{int(miss_cand[E].sum()):,}; rejected by model: {int(fn_model[E].sum()):,}", flush=True)
    print(f"  false links:  {int(fp[E].sum()):,}", flush=True)

    # ---- characterise the model's pair-level mistakes -------------------------
    names = ["t_addr_missing", "nsim", "asim", "e5n_cos", "e5c_cos", "first_num_eq"]
    fn_rows = np.flatnonzero((y4 == 1) & ~sel)
    fp_rows = np.flatnonzero((y4 == 0) & sel)
    print(f"\nrejected true links={len(fn_rows):,}: lost on uniqueness (another S1 won)="
          f"{100*(~uniq[fn_rows]).mean():.1f}%, below threshold={100*(uniq[fn_rows] & (cal[fn_rows] < th[fn_rows])).mean():.1f}%, "
          f"cal median={np.median(cal[fn_rows]):.3f}", flush=True)
    print(f"accepted false links={len(fp_rows):,}: cal median={np.median(cal[fp_rows]):.3f}", flush=True)
    json.dump({"f05": float(sc[E].mean())}, open(f"{ROOT}/eda/err_stage3.json", "w"))


if __name__ == "__main__":
    main()
