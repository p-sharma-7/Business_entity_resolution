"""How much does the neural channel actually recover, and in which direction?

S1->target and target->S1 are not symmetric here. S1 is the deduplicated
reference side and is ~2.4x smaller, so a target may rank its true parent far
higher than the parent ranks it - especially for the 39% of S1 records that
share a normalized name.
"""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, pyarrow.parquet as pq, faiss
from er.io_utils import read_ground_truth, WORK

for cty, ts in (("India", 2), ("US", 2)):
    cty1 = np.asarray(pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                                    columns=["country"]).column("country").to_pylist())
    ids1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    ctyT = np.asarray(pq.read_table(f"{WORK}/normalized/train_s{ts}.parquet",
                                    columns=["country"]).column("country").to_pylist())
    idsT = pq.read_table(f"{WORK}/normalized/train_s{ts}.parquet",
                         columns=["entity_id"]).column("entity_id").to_pylist()
    qi = np.flatnonzero(cty1 == cty); ti = np.flatnonzero(ctyT == cty)
    qloc = np.full(len(ids1), -1, np.int64); qloc[qi] = np.arange(len(qi))
    tloc = np.full(len(idsT), -1, np.int64); tloc[ti] = np.arange(len(ti))
    gt = read_ground_truth("train")
    r1 = {e: i for i, e in enumerate(ids1)}; rT = {e: i for i, e in enumerate(idsT)}
    pairs = []
    for s1, tg in gt.items():
        i = qloc[r1[s1]]
        if i < 0: continue
        for e in tg:
            j = rT.get(e)
            if j is not None and tloc[j] >= 0:
                pairs.append((i, tloc[j]))
    P = np.asarray(pairs, np.int64)
    truth = np.sort(P[:, 0] * np.int64(6_000_000) + P[:, 1])
    print(f"\n### {cty} S{ts}: truth={len(truth):,} q={len(qi):,} t={len(ti):,}", flush=True)

    for view in ("name", "combined"):
        z = np.load(f"{WORK}/ann/train_S{ts}_{cty}_{view}.npz")
        aq = qloc[z["q"].astype(np.int64)]
        at = tloc[z["t"].astype(np.int64)]
        for k in (5, 10, 30):
            g = np.sort((np.repeat(aq, k) * np.int64(6_000_000) + at[:, :k].reshape(-1)))
            idx = np.clip(np.searchsorted(g, truth), 0, len(g) - 1)
            print(f"  fwd  {view:<9} k={k:<3} recall={float((g[idx]==truth).mean()):.5f}", flush=True)

    # reverse direction: index S1, query with targets
    for view in ("name",):
        E1 = np.load(f"{WORK}/emb/train_s1_{view}.npy", mmap_mode="r")
        ET = np.load(f"{WORK}/emb/train_s{ts}_{view}.npy", mmap_mode="r")
        t0 = time.time()
        xb = np.ascontiguousarray(E1[qi], dtype=np.float32)
        index = faiss.IndexFlatIP(xb.shape[1])
        co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
        gi = faiss.index_cpu_to_all_gpus(index, co=co); gi.add(xb)
        K = 30
        I = np.empty((len(ti), K), np.int64)
        for s in range(0, len(ti), 65536):
            e = min(s + 65536, len(ti))
            _, I[s:e] = gi.search(np.ascontiguousarray(ET[ti[s:e]], dtype=np.float32), K)
        for k in (5, 10, 30):
            g = np.sort(I[:, :k].reshape(-1) * np.int64(6_000_000) +
                        np.repeat(np.arange(len(ti), dtype=np.int64), k))
            idx = np.clip(np.searchsorted(g, truth), 0, len(g) - 1)
            print(f"  rev  {view:<9} k={k:<3} recall={float((g[idx]==truth).mean()):.5f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(f"{WORK}/ann/train_S{ts}_{cty}_rev_{view}.npz",
                 t=ti.astype(np.int32), q=qi[I].astype(np.int32))
        del gi, xb
