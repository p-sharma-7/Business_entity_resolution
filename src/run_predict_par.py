"""Test-set scoring with the stage-1/2 fold models, one fold model per GPU.

Single-GPU scoring was GPU-bound (~14 min per large partition): each pair has
to pass through five fold models. Here fold model f runs on GPU f over every
partition, so the work splits five ways. GPUs 5 and 6 stay free.

Writes work/pred/test_scores.npz (q, t, src, p1, p2, pc) consumed by
run_final.py, plus the stage-2-only fallback submission.
"""
import sys, os, json, time, subprocess
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT
from er import decide as D

SPLIT = "test"
GPUS = [0, 1, 2]          # capped at 3 GPUs; the rest stay free for other users
FOLDS = 5
CH = 4_000_000
COMP = f"{WORK}/pred/{SPLIT}_comp.npy"
ONLY = os.environ.get("ER_ONLY_COUNTRY")   # re-score one country and merge
# Tagged model family (e.g. the language-agnostic _lex models) and the feature
# columns it was trained without; tagged runs write their own score file.
TAG = os.environ.get("ER_TAG", "")
DROP = {int(x) for x in os.environ.get("ER_DROP", "").split(",") if x.strip()}
COMP = f"{WORK}/pred/{SPLIT}_comp{TAG}.npy"


def keep_cols(n):
    return np.array([i for i in range(n) if i not in DROP], np.int64)


def partitions():
    cty1 = sorted(set(pq.read_table(f"{WORK}/normalized/{SPLIT}_s1.parquet",
                                    columns=["country"]).column("country").to_pylist()))
    out = []
    for ts in (2, 3):
        for c in cty1:
            if ONLY and c != ONLY:
                continue
            p = f"{WORK}/feat/{SPLIT}_S{ts}_{c}.npz"
            if os.path.exists(p):
                xp = p.replace(".npz", "_X.npy")
                if not os.path.exists(xp):
                    X = np.load(p)["X"]; np.save(xp + ".tmp.npy", X); os.replace(xp + ".tmp.npy", xp)
                out.append((ts, c, p, xp))
    return out


def worker(stage, f):
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(f"{ROOT}/models/xgb{TAG}_stage{stage}_fold{f}.json")
    b.set_param({"device": "cuda"})
    rng = (0, b.best_iteration + 1)
    comp = np.load(COMP, mmap_mode="r") if stage == 2 else None
    outs, off, t0 = [], 0, time.time()
    for ts, c, p, xp in partitions():
        XM = np.load(xp, mmap_mode="r"); n = XM.shape[0]
        o = np.empty(n, np.float32)
        for s in range(0, n, CH):
            e = min(s + CH, n)
            cols = [np.asarray(XM[s:e], np.float32)[:, keep_cols(XM.shape[1])],
                    np.full((e - s, 1), ts, np.float32)]
            if comp is not None:
                cols.append(np.asarray(comp[off + s:off + e], np.float32))
            o[s:e] = b.inplace_predict(np.hstack(cols), iteration_range=rng)
        outs.append(o); off += n
        print(f"[s{stage} f{f}] {p} ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{WORK}/pred/{SPLIT}{TAG}_p{stage}_f{f}.npy", np.concatenate(outs))


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


def run_stage(stage):
    jobs = [([sys.executable, __file__, "_worker", str(stage), str(f)],
             dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[f % len(GPUS)])),
             f"{ROOT}/logs/er_pred{TAG}_s{stage}_f{f}.log") for f in range(FOLDS)]
    if not _run_waves(jobs):
        raise SystemExit(f"stage {stage} scoring failed; see logs/er_pred_s{stage}_f*.log")
    return np.mean([np.load(f"{WORK}/pred/{SPLIT}{TAG}_p{stage}_f{f}.npy") for f in range(FOLDS)], axis=0)


def group_stats(gid, score):
    order = np.lexsort((-score, gid)); g = gid[order]; s = score[order]
    new = np.r_[True, g[1:] != g[:-1]]; st = np.flatnonzero(new)
    sz = np.diff(np.append(st, len(g)))
    rank = (np.arange(len(g)) - np.repeat(st, sz)).astype(np.float32)
    best = np.repeat(s[st], sz).astype(np.float32)
    sec = np.repeat(np.where(sz > 1, s[np.minimum(st + 1, len(s) - 1)], -1.0), sz).astype(np.float32)
    cnt = np.repeat(sz, sz).astype(np.float32)
    inv = np.empty(len(g), np.int64); inv[order] = np.arange(len(g))
    return rank[inv], best[inv], sec[inv], cnt[inv]


def main():
    import joblib
    t0 = time.time()
    parts = partitions()
    qs, ts_, ss = [], [], []
    for ts, c, p, xp in parts:
        z = np.load(p)
        qs.append(z["q"].astype(np.int64)); ts_.append(z["t"].astype(np.int64))
        ss.append(np.full(len(z["q"]), ts, np.int8))
    q = np.concatenate(qs); t = np.concatenate(ts_); src = np.concatenate(ss)
    tkey = t + src.astype(np.int64) * np.int64(10_000_000)
    print(f"test pairs={len(q):,} partitions={len(parts)} ({time.time()-t0:.0f}s)", flush=True)

    p1 = run_stage(1).astype(np.float32)
    print(f"stage 1 done ({time.time()-t0:.0f}s)", flush=True)
    tr_, tb, tsec, tc = group_stats(tkey, p1)
    qr_, qb, qsec, qc = group_stats(q, p1)
    np.save(COMP, np.column_stack([tr_, tb, p1 - tb, tb - tsec, tc,
                                   qr_, qb, p1 - qb, qb - qsec, qc, p1]).astype(np.float32))
    p2 = run_stage(2).astype(np.float32)
    print(f"stage 2 done ({time.time()-t0:.0f}s)", flush=True)

    if TAG:
        pc = joblib.load(f"{ROOT}/models/calibrator{TAG}.pkl").predict(p2).astype(np.float32)
        out = f"{WORK}/pred/{SPLIT}_scores{TAG}{'_' + ONLY if ONLY else ''}.npz"
        np.savez(out, q=q, t=t, src=src, p1=p1, p2=p2, pc=pc)
        print(f"saved {out}", flush=True)
        return
    cfg = json.load(open(f"{ROOT}/models/decision.json"))
    pc = joblib.load(f"{ROOT}/models/calibrator.pkl").predict(p2).astype(np.float32)
    if ONLY:
        # Competition statistics never cross countries (every candidate pair
        # shares its S1's country), so one country can be re-scored alone and
        # merged with the untouched rows of the others.
        old = dict(np.load(f"{WORK}/pred/{SPLIT}_scores.npz"))
        c1 = np.asarray(pq.read_table(f"{WORK}/normalized/{SPLIT}_s1.parquet",
                                      columns=["country"]).column("country").to_pylist())
        keep_old = c1[old["q"]] != ONLY
        new = dict(q=q, t=t, src=src, p1=p1, p2=p2, pc=pc)
        merged = {k: np.concatenate([old[k][keep_old], new[k]]) for k in new}
        np.savez(f"{WORK}/pred/{SPLIT}_scores.npz", **merged)
        print(f"merged {ONLY}: replaced {int((~keep_old).sum()):,} rows with {len(q):,} "
              f"-> total {len(merged['q']):,}", flush=True)
        return
    np.savez(f"{WORK}/pred/{SPLIT}_scores.npz", q=q, t=t, src=src, p1=p1, p2=p2, pc=pc)
    print(f"saved {SPLIT}_scores.npz", flush=True)

    # ---- stage-2-only fallback submission --------------------------------
    keep = D.unique_target_filter(tkey, pc) & (pc >= cfg["threshold"])
    ids1 = pq.read_table(f"{WORK}/normalized/{SPLIT}_s1.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    idsT = {s: pq.read_table(f"{WORK}/normalized/{SPLIT}_s{s}.parquet",
                             columns=["entity_id"]).column("entity_id").to_pylist() for s in (2, 3)}
    outdir = f"{ROOT}/output_v2_fallback"; os.makedirs(outdir, exist_ok=True)
    for path, mask, col in ((f"{outdir}/matching_results.tsv", keep, "matched_entity_ids"),
                            (f"{outdir}/candidate_pairs.tsv", np.ones(len(q), bool), "candidate_entity_ids")):
        buckets = [[] for _ in range(len(ids1))]
        sel = np.flatnonzero(mask)
        for i in sel[np.argsort(q[sel], kind="stable")]:
            buckets[q[i]].append(idsT[int(src[i])][t[i]])
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for i, e in enumerate(ids1):
                fh.write(e + "\t" + ",".join(dict.fromkeys(buckets[i])) + "\n")
        print(f"wrote {path}", flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
