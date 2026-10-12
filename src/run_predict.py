"""Score test candidates with the trained two-stage model and write the
submission files.

Outputs, per the challenge spec:
  output/matching_results.tsv  one row per test S1 entity, empty when no match
  output/candidate_pairs.tsv   the exact set fed to the model for inference
"""
import sys, os, json, time, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import lightgbm as lgb
import pyarrow.parquet as pq
import joblib
from er.io_utils import WORK, ROOT
from er import decide as D
from run_train import group_stats, ensure_npy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", default=f"{ROOT}/output")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.outdir, exist_ok=True)

    cfg = json.load(open(f"{ROOT}/models/decision.json"))
    cal = joblib.load(f"{ROOT}/models/calibrator.pkl")
    mt = json.load(open(f"{ROOT}/models/model_type.json"))["model"] \
        if os.path.exists(f"{ROOT}/models/model_type.json") else "lgb"
    if mt == "xgb":
        import xgboost as xgb
        def load_xgb(path):
            b = xgb.Booster(); b.load_model(path); b.set_param({"device": "cuda"})
            return b
        m1 = [load_xgb(f"{ROOT}/models/xgb_stage1_fold{f}.json") for f in range(a.folds)]
        m2 = [load_xgb(f"{ROOT}/models/xgb_stage2_fold{f}.json") for f in range(a.folds)]
    else:
        m1 = [lgb.Booster(model_file=f"{ROOT}/models/lgb_stage1_fold{f}.txt") for f in range(a.folds)]
        m2 = [lgb.Booster(model_file=f"{ROOT}/models/lgb_stage2_fold{f}.txt") for f in range(a.folds)]

    ids1 = pq.read_table(f"{WORK}/normalized/{a.split}_s1.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    cty1 = np.asarray(pq.read_table(f"{WORK}/normalized/{a.split}_s1.parquet",
                                    columns=["country"]).column("country").to_pylist())
    idsT = {s: pq.read_table(f"{WORK}/normalized/{a.split}_s{s}.parquet",
                             columns=["entity_id"]).column("entity_id").to_pylist()
            for s in (2, 3)}
    n_ent = len(ids1)

    # Streamed in two passes so only one partition's matrix is ever resident:
    # holding all six test partitions in float32 would need ~90 GB.
    parts = []
    for ts in (2, 3):
        for cty in sorted(set(cty1.tolist())):
            p = f"{WORK}/feat/{a.split}_S{ts}_{cty}.npz"
            if os.path.exists(p):
                parts.append((ts, cty, p, ensure_npy(p)))
    print("partitions:", [x[2] for x in parts], flush=True)

    def predict_avg(models, X):
        acc = np.zeros(len(X), np.float64)
        for m in models:
            if mt == "xgb":
                acc += m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1))
            else:
                acc += m.predict(X, num_iteration=m.best_iteration)
        return (acc / len(models)).astype(np.float32)

    CH = 6_000_000
    Qs, Ts, Ss, P1 = [], [], [], []
    for ts, cty, p, xp in parts:
        z = np.load(p)
        q_ = z["q"].astype(np.int64); t_ = z["t"].astype(np.int64)
        XM = np.load(xp, mmap_mode="r")
        p1 = np.empty(len(q_), np.float32)
        for s0 in range(0, len(q_), CH):
            X = np.asarray(XM[s0:s0 + CH], np.float32)
            X = np.hstack([X, np.full((len(X), 1), ts, np.float32)])
            p1[s0:s0 + CH] = predict_avg(m1, X)
            del X
        Qs.append(q_); Ts.append(t_); Ss.append(np.full(len(q_), ts, np.int8)); P1.append(p1)
        print(f"  stage1 {p} n={len(q_):,} ({time.time()-t0:.0f}s)", flush=True)

    q = np.concatenate(Qs); t = np.concatenate(Ts); src = np.concatenate(Ss)
    p1 = np.concatenate(P1)
    tkey = t + src.astype(np.int64) * np.int64(10_000_000)
    tr_, tb, tsec, tc = group_stats(tkey, p1)
    qr_, qb, qsec, qc = group_stats(q, p1)
    comp = np.column_stack([tr_, tb, p1 - tb, tb - tsec, tc,
                            qr_, qb, p1 - qb, qb - qsec, qc, p1]).astype(np.float32)

    p2 = np.empty(len(q), np.float32)
    off = 0
    for (ts, cty, p, xp), qpart in zip(parts, Qs):
        n = len(qpart)
        XM = np.load(xp, mmap_mode="r")
        for s0 in range(0, n, CH):
            e0 = min(s0 + CH, n)
            X = np.asarray(XM[s0:e0], np.float32)
            X = np.hstack([X, np.full((len(X), 1), ts, np.float32), comp[off + s0:off + e0]])
            p2[off + s0:off + e0] = predict_avg(m2, X)
            del X
        off += n
        print(f"  stage2 {p} ({time.time()-t0:.0f}s)", flush=True)
    print(f"stage2 done ({time.time()-t0:.0f}s)", flush=True)

    pc = cal.predict(p2).astype(np.float32)
    np.savez(f"{WORK}/pred/{a.split}_scores.npz", q=q, t=t, src=src, p1=p1, p2=p2, pc=pc)

    rule = cfg["rule"]
    if rule.startswith("D") or rule.startswith("B"):
        uniq = D.unique_target_filter(tkey, pc)
    else:
        uniq = np.ones(len(pc), bool)
    if rule.startswith("C") or rule.startswith("D"):
        pu = np.where(uniq, pc, 0.0)
        keep = D.select_sets(q, pu, n_ent, m_scale=cfg["m_scale"],
                             p_floor=cfg["p_floor"]) & uniq
    else:
        keep = uniq & (pc >= cfg["threshold"])
    print(f"selected {int(keep.sum()):,} pairs ({time.time()-t0:.0f}s)", flush=True)

    # ---- write submission files -----------------------------------------
    def write(path, mask, col):
        buckets = [[] for _ in range(n_ent)]
        sel = np.flatnonzero(mask)
        order = sel[np.argsort(q[sel], kind="stable")]
        for i in order:
            buckets[q[i]].append(idsT[int(src[i])][t[i]])
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for i, e in enumerate(ids1):
                v = buckets[i]
                fh.write(e + "\t" + (",".join(dict.fromkeys(v)) if v else "") + "\n")
        nz = sum(1 for b in buckets if b)
        print(f"wrote {path}: {n_ent:,} rows, {nz:,} non-empty, "
              f"{sum(len(b) for b in buckets):,} ids", flush=True)

    write(f"{a.outdir}/matching_results.tsv", keep, "matched_entity_ids")
    write(f"{a.outdir}/candidate_pairs.tsv", np.ones(len(q), bool), "candidate_entity_ids")
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
