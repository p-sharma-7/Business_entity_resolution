"""Two-stage pair classifier on GPU: XGBoost, one fold per GPU, folds in parallel.

Same design as run_train.py (S1-grouped folds aligned with the encoder
holdout, hard-negative sampling, stage 2 = target-side competition over
stage-1 out-of-fold scores), but each fold trains on its own GPU at the same
time. GPU use is capped by GPUS so the remaining devices stay free for other
users of this machine.
"""
import sys, os, time, json, argparse, subprocess
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT, fold_of

GPUS = [int(g) for g in os.environ.get("ER_GPUS", "0,1,2").split(",")]   # capped at 3 GPUs; the rest stay free for other users
PY = sys.executable
# ER_TAG writes a separate model family (paths suffixed) so experiments never
# overwrite the production models; ER_DROP lists feature columns to exclude,
# e.g. every neural (E5 / ANN) feature for the language-agnostic model.
TAG = os.environ.get("ER_TAG", "")
DROP = {int(x) for x in os.environ.get("ER_DROP", "").split(",") if x.strip()}
META = f"{WORK}/pred/gpu_meta{TAG}.npz"
COMP = f"{WORK}/pred/comp_feats{TAG}.npy"


def keep_cols(n):
    return np.array([i for i in range(n) if i not in DROP], np.int64)

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cuda", max_bin=128,
                  grow_policy="lossguide", max_leaves=255, max_depth=0,
                  learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5.0, reg_lambda=2.0, reg_alpha=0.5)


def group_stats(gid, score):
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
    xp = npz_path.replace(".npz", "_X.npy")
    if not os.path.exists(xp):
        X = np.load(npz_path)["X"]
        np.save(xp + ".tmp.npy", X); os.replace(xp + ".tmp.npy", xp)
    return xp


# --------------------------------------------------------------------- worker
def worker(stage, f, rounds):
    import xgboost as xgb
    m = np.load(META, allow_pickle=True)
    xpaths = list(m["xpaths"]); offs = m["offs"]; srcs = m["srcs"]
    take, fold, y = m["take"], m["fold"], m["y"]
    comp = np.load(COMP, mmap_mode="r") if stage == 2 else None
    XM = [np.load(p, mmap_mode="r") for p in xpaths]
    t0 = time.time()

    def rows_matrix(mask):
        parts = []
        for mi in range(len(XM)):
            lo, hi = offs[mi], offs[mi + 1]
            r = np.flatnonzero(mask[lo:hi])
            if not len(r):
                continue
            X = np.asarray(XM[mi][r], np.float32)[:, keep_cols(XM[mi].shape[1])]
            cols = [X, np.full((len(r), 1), srcs[mi], np.float32)]
            if comp is not None:
                cols.append(np.asarray(comp[lo + r], np.float32))
            parts.append(np.hstack(cols))
        return np.vstack(parts)

    tr = take & (fold != f)
    va = take & (fold == f)
    Xtr = rows_matrix(tr); ytr = y[tr]
    Xva = rows_matrix(va); yva = y[va]
    print(f"[s{stage} f{f}] fit={Xtr.shape} val={Xva.shape} ({time.time()-t0:.0f}s)", flush=True)
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=XGB_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr, max_bin=XGB_PARAMS["max_bin"])
    del Xtr, Xva
    bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=rounds,
                    evals=[(dva, "val")], early_stopping_rounds=50,
                    verbose_eval=100)
    bst.save_model(f"{ROOT}/models/xgb{TAG}_stage{stage}_fold{f}.json")
    print(f"[s{stage} f{f}] best_iter={bst.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    del dtr, dva

    # score EVERY row of this fold, not just sampled ones
    rng = (0, bst.best_iteration + 1)
    out = np.zeros(len(y), np.float32)
    for mi in range(len(XM)):
        lo, hi = offs[mi], offs[mi + 1]
        r = np.flatnonzero(fold[lo:hi] == f)
        for s0 in range(0, len(r), 4_000_000):
            rr = r[s0:s0 + 4_000_000]
            X = np.asarray(XM[mi][rr], np.float32)[:, keep_cols(XM[mi].shape[1])]
            cols = [X, np.full((len(rr), 1), srcs[mi], np.float32)]
            if comp is not None:
                cols.append(np.asarray(comp[lo + rr], np.float32))
            out[lo + rr] = bst.inplace_predict(np.hstack(cols), iteration_range=rng)
    sel = np.flatnonzero(fold == f)
    np.savez(f"{WORK}/pred/oof{TAG}_s{stage}_f{f}.npz", idx=sel, p=out[sel])
    print(f"[s{stage} f{f}] oof scored ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------- main
def _run_waves(cmds_envs_logs):
    """Run worker processes at most len(GPUS) at a time (one per GPU)."""
    import subprocess
    for w in range(0, len(cmds_envs_logs), len(GPUS)):
        procs = []
        for cmd, env, logpath in cmds_envs_logs[w:w + len(GPUS)]:
            log = open(logpath, "w")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for pr, log in procs:
            pr.wait(); log.close()
            if pr.returncode:
                return False
    return True


def run_stage(stage, folds, rounds, n):
    jobs = [([PY, __file__, "--worker", str(stage), str(f), "--rounds", str(rounds)],
             dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[f % len(GPUS)])),
             f"{ROOT}/logs/er_gpu{TAG}_s{stage}_f{f}.log") for f in range(folds)]
    if not _run_waves(jobs):
        raise SystemExit(f"stage {stage} worker failed; see logs/er_gpu_s{stage}_f*.log")
    return collect(stage, folds, n)


def collect(stage, folds, n):
    oof = np.zeros(n, np.float32)
    for f in range(folds):
        z = np.load(f"{WORK}/pred/oof{TAG}_s{stage}_f{f}.npz")
        oof[z["idx"]] = z["p"]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--neg_frac", type=float, default=0.03)
    ap.add_argument("--hard_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], a.rounds); return

    t0 = time.time()
    cty1 = sorted(set(pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                                    columns=["country"]).column("country").to_pylist()))
    only = os.environ.get("ER_COUNTRIES")      # e.g. "US": train on one country only
    parts = [(ts, c, f"{WORK}/feat/train_S{ts}_{c}.npz") for ts in (2, 3) for c in cty1
             if not only or c in only.split(",")]
    parts = [p for p in parts if os.path.exists(p[2])]
    qs, ts_, ys, cheaps, srcs, xpaths, ns = [], [], [], [], [], [], []
    for ts, c, p in parts:
        z = np.load(p)
        qs.append(z["q"].astype(np.int64)); ts_.append(z["t"].astype(np.int64))
        ys.append(z["y"].astype(np.int8)); cheaps.append(z["cheap"].astype(np.float32))
        srcs.append(ts); xpaths.append(ensure_npy(p)); ns.append(len(z["q"]))
        print(f"  {p}: n={ns[-1]:,} pos={int(ys[-1].sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    offs = np.cumsum([0] + ns)
    q = np.concatenate(qs); t = np.concatenate(ts_); y = np.concatenate(ys)
    src = np.concatenate([np.full(n, s, np.int8) for n, s in zip(ns, srcs)])
    N = len(y)

    ids1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    f_of = np.fromiter((fold_of(e, a.folds) for e in ids1), np.int8, len(ids1))
    fold = f_of[q]

    from er.candgen import topk_mask
    rng = np.random.default_rng(a.seed)
    hard = np.zeros(N, bool)
    for mi in range(len(parts)):
        lo, hi = offs[mi], offs[mi + 1]
        neg = np.where(ys[mi] == 0, cheaps[mi], -1.0).astype(np.float32)
        hard[lo:hi] = (ys[mi] == 0) & topk_mask(qs[mi], neg, a.hard_k)
    take = (y == 1) | hard | (rng.random(N) < a.neg_frac)
    print(f"total={N:,} pos={int(y.sum()):,} hard_neg={int(hard.sum()):,} "
          f"fit={int(take.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(META, xpaths=np.array(xpaths, dtype=object), offs=offs,
             srcs=np.array(srcs), take=take, fold=fold, y=y)

    # ER_REUSE_S1=1 resumes from existing stage-1 fold outputs (same META/seed)
    have1 = all(os.path.exists(f"{WORK}/pred/oof{TAG}_s1_f{f}.npz") for f in range(a.folds))
    oof1 = (collect(1, a.folds, N) if os.environ.get("ER_REUSE_S1") == "1" and have1
            else run_stage(1, a.folds, a.rounds, N))
    print(f"stage 1 done ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{WORK}/pred/oof1{TAG}.npy", oof1)

    tkey = t + src.astype(np.int64) * np.int64(10_000_000)
    tr_, tb, tsec, tc = group_stats(tkey, oof1)
    qr_, qb, qsec, qc = group_stats(q, oof1)
    comp = np.column_stack([tr_, tb, oof1 - tb, tb - tsec, tc,
                            qr_, qb, oof1 - qb, qb - qsec, qc, oof1]).astype(np.float32)
    np.save(COMP, comp); del comp

    oof2 = run_stage(2, a.folds, a.rounds, N)
    print(f"stage 2 done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{WORK}/pred/oof{TAG}.npz", p1=oof1, p2=oof2, y=y, q=q, t=t, src=src, fold=fold)
    json.dump({"model": "xgb", "folds": a.folds}, open(f"{ROOT}/models/model_type{TAG}.json", "w"))
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
