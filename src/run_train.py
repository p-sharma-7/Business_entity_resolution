"""Two-stage LightGBM pair classifier with S1-grouped folds.

Stage 1 scores a pair from its own evidence. Stage 2 adds how that score
COMPETES on the target side - rank among all S1 entities claiming this target,
gap to the best claim, how many claims are strong. That is the single most
valuable signal here: every matched S2/S3 record has exactly one true S1
parent (verified on all 7,638,365 training links), so a pair that is only the
third-best claim on its target is almost certainly a false merge no matter how
good it looks alone.

Memory: features live on disk as float16 per partition. Training rows are
subsampled (all positives, a fraction of negatives) so the training matrix
stays modest, while out-of-fold scores are produced for EVERY pair - the
thresholds are tuned on those, and isotonic calibration on the full OOF set
removes the bias the negative subsampling introduces.
"""
import sys, os, time, json, argparse, gc
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import lightgbm as lgb
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT, fold_of

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.08,
              num_leaves=255, max_depth=-1, min_data_in_leaf=150,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=0.5, lambda_l2=2.0, num_threads=38, verbosity=-1,
              max_bin=127)


def group_stats(gid, score):
    """rank / best / second / count of `score` within each group, row-aligned."""
    order = np.lexsort((-score, gid))
    g = gid[order]; s = score[order]
    new = np.empty(len(g), bool); new[0] = True
    np.not_equal(g[1:], g[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, len(g)))
    rank = (np.arange(len(g)) - np.repeat(starts, sizes)).astype(np.float32)
    best = np.repeat(s[starts], sizes).astype(np.float32)
    sec = np.repeat(np.where(sizes > 1, s[np.minimum(starts + 1, len(s) - 1)], -1.0),
                    sizes).astype(np.float32)
    cnt = np.repeat(sizes, sizes).astype(np.float32)
    inv = np.empty(len(g), np.int64); inv[order] = np.arange(len(g))
    return rank[inv], best[inv], sec[inv], cnt[inv]


def ensure_npy(npz_path):
    """Expose the feature matrix as a raw .npy so it can be memory-mapped.

    np.load(..., mmap_mode='r') is silently ignored for .npz archives: every
    z['X'][rows] access re-reads the whole 14-22 GB partition. Extracting once
    turns hundreds of full re-reads per fold into true page-level access.
    """
    xp = npz_path.replace(".npz", "_X.npy")
    if not os.path.exists(xp):
        t = time.time()
        X = np.load(npz_path)["X"]
        np.save(xp + ".tmp.npy", X)
        os.replace(xp + ".tmp.npy", xp)
        print(f"  extracted {xp} {X.shape} ({time.time()-t:.0f}s)", flush=True)
        del X
    return xp


def partitions(split):
    cty1 = sorted(set(pq.read_table(f"{WORK}/normalized/{split}_s1.parquet",
                                    columns=["country"]).column("country").to_pylist()))
    out = []
    for ts in (2, 3):
        for cty in cty1:
            p = f"{WORK}/feat/{split}_S{ts}_{cty}.npz"
            if os.path.exists(p):
                out.append((ts, cty, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--neg_frac", type=float, default=0.03,
                    help="fraction of EASY negatives kept in addition to hard ones")
    ap.add_argument("--hard_k", type=int, default=3,
                    help="most-confusable negatives kept per S1 per source")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    parts = partitions("train")
    print("partitions:", [p[2] for p in parts], flush=True)

    # ---- pass 1: metadata only -------------------------------------------
    metas = []
    for ts, cty, p in parts:
        z = np.load(p)
        metas.append(dict(ts=ts, cty=cty, path=p, xpath=ensure_npy(p), n=len(z["q"]),
                          q=np.asarray(z["q"], np.int64), t=np.asarray(z["t"], np.int64),
                          y=np.asarray(z["y"], np.int8),
                          cheap=np.asarray(z["cheap"], np.float32)))
        print(f"  {p}: n={metas[-1]['n']:,} pos={int(metas[-1]['y'].sum()):,}", flush=True)
    q = np.concatenate([m["q"] for m in metas])
    t = np.concatenate([m["t"] for m in metas])
    y = np.concatenate([m["y"] for m in metas])
    src = np.concatenate([np.full(m["n"], m["ts"], np.int8) for m in metas])
    offs = np.cumsum([0] + [m["n"] for m in metas])
    tkey = t + src.astype(np.int64) * np.int64(10_000_000)
    N = len(y)
    print(f"total pairs={N:,} pos={int(y.sum()):,} ({100*y.mean():.3f}%) ({time.time()-t0:.0f}s)", flush=True)

    # Folds come from a stable hash of the S1 entity_id, the SAME function the
    # encoder fine-tuning used to hold out fold 4. Without that alignment the
    # validation fold would contain entities the encoder had already seen and
    # the estimate would not transfer to the test set.
    ids1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    f_of = np.fromiter((fold_of(e, a.folds) for e in ids1), np.int8, len(ids1))
    fold = f_of[q]
    print(f"fold sizes: {np.bincount(fold).tolist()}", flush=True)

    # rows used for FITTING: every positive plus a sample of negatives
    # Hard-negative sampling. Precision - which F0.5 weights double - is lost
    # on the CONFUSABLE non-matches (near-duplicate distractors at the same
    # address), not on the easy ones, so every S1's top-k highest-scoring
    # negatives are kept and only a small random slice of the rest. The bias
    # this introduces in raw probabilities is monotone, and isotonic
    # calibration on the full out-of-fold set removes it.
    from er.candgen import topk_mask
    rng = np.random.default_rng(a.seed)
    hard = np.zeros(N, bool)
    for mi, m in enumerate(metas):
        lo, hi = offs[mi], offs[mi + 1]
        negscore = np.where(m["y"] == 0, m["cheap"], -1.0).astype(np.float32)
        hard[lo:hi] = (m["y"] == 0) & topk_mask(m["q"], negscore, a.hard_k)
    take = (y == 1) | hard | (rng.random(N) < a.neg_frac)
    print(f"sampling: pos={int(y.sum()):,} hard_neg={int(hard.sum()):,} "
          f"total_fit={int(take.sum()):,}", flush=True)
    print(f"fit rows={int(take.sum()):,} ({100*take.mean():.1f}%)", flush=True)

    names = open(f"{ROOT}/models/feature_names.txt").read().split("\n")

    XM = [np.load(m["xpath"], mmap_mode="r") for m in metas]

    def load_X(mi, rows=None, extra_cols=None):
        X = np.asarray(XM[mi][rows] if rows is not None else XM[mi], np.float32)
        s = np.full(len(X), metas[mi]["ts"], np.float32)
        cols = [X, s[:, None]]
        if extra_cols is not None:
            cols.append(extra_cols)
        return np.hstack(cols)

    names1 = names + ["src_is_s3"]
    oof = np.zeros(N, np.float32)

    def run_stage(stage, feat_names, extra_all):
        oof_s = np.zeros(N, np.float32)
        for f in range(a.folds):
            Xtr, ytr, Xva, yva = [], [], [], []
            for mi, m in enumerate(metas):
                lo, hi = offs[mi], offs[mi + 1]
                sel = take[lo:hi]
                fl = fold[lo:hi]
                rtr = np.flatnonzero(sel & (fl != f))
                rva = np.flatnonzero(sel & (fl == f))
                ex = extra_all[lo:hi] if extra_all is not None else None
                if len(rtr):
                    Xtr.append(load_X(mi, rtr, ex[rtr] if ex is not None else None))
                    ytr.append(m["y"][rtr])
                if len(rva):
                    Xva.append(load_X(mi, rva, ex[rva] if ex is not None else None))
                    yva.append(m["y"][rva])
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            Xva = np.vstack(Xva); yva = np.concatenate(yva)
            print(f"[s{stage} fold {f}] fit={Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)
            ds = lgb.Dataset(Xtr, label=ytr, feature_name=feat_names, free_raw_data=True)
            dv = lgb.Dataset(Xva, label=yva, reference=ds, free_raw_data=True)
            m_ = lgb.train(PARAMS, ds, num_boost_round=a.rounds, valid_sets=[dv],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(100)])
            m_.save_model(f"{ROOT}/models/lgb_stage{stage}_fold{f}.txt")
            del Xtr, ytr, Xva, yva, ds, dv; gc.collect()
            print(f"[s{stage} fold {f}] best_iter={m_.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
            # score EVERY row of this fold, not just the sampled ones
            for mi, m in enumerate(metas):
                lo, hi = offs[mi], offs[mi + 1]
                rows = np.flatnonzero(fold[lo:hi] == f)
                if not len(rows):
                    continue
                ex = extra_all[lo:hi] if extra_all is not None else None
                for s0 in range(0, len(rows), 8_000_000):
                    rr = rows[s0:s0 + 8_000_000]
                    Xb = load_X(mi, rr, ex[rr] if ex is not None else None)
                    oof_s[lo + rr] = m_.predict(Xb, num_iteration=m_.best_iteration)
                    del Xb
                gc.collect()
            print(f"[s{stage} fold {f}] oof scored ({time.time()-t0:.0f}s)", flush=True)
        return oof_s

    oof1 = run_stage(1, names1, None)
    np.save(f"{WORK}/pred/oof1.npy", oof1)

    tr_, tb, tsec, tc = group_stats(tkey, oof1)
    qr_, qb, qsec, qc = group_stats(q, oof1)
    comp = np.column_stack([tr_, tb, oof1 - tb, tb - tsec, tc,
                            qr_, qb, oof1 - qb, qb - qsec, qc, oof1]).astype(np.float32)
    comp_names = ["t_prank", "t_pbest", "t_pmargin", "t_ptop2gap", "t_pcnt",
                  "q_prank", "q_pbest", "q_pmargin", "q_ptop2gap", "q_pcnt", "p1"]
    names2 = names1 + comp_names
    oof2 = run_stage(2, names2, comp)

    os.makedirs(f"{WORK}/pred", exist_ok=True)
    np.savez(f"{WORK}/pred/oof.npz", p1=oof1, p2=oof2, y=y, q=q, t=t, src=src, fold=fold)
    open(f"{ROOT}/models/feature_names_stage2.txt", "w").write("\n".join(names2))
    b = lgb.Booster(model_file=f"{ROOT}/models/lgb_stage2_fold0.txt")
    imp = sorted(zip(names2, b.feature_importance("gain")), key=lambda kv: -kv[1])
    print("\ntop features by gain:")
    for n_, g in imp[:40]:
        print(f"   {n_:<26} {g:,.0f}")
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
