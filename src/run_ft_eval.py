"""Does fine-tuning actually improve RETRIEVAL, not just training loss?

Measures name-view recall@k on the India S2 partition - the slice where the
base encoder was weakest (0.556 @k=30) and where the unrecovered candidates
are concentrated.
"""
import sys, time, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, pyarrow.parquet as pq, faiss
from er.io_utils import read_ground_truth, WORK

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_ft")
ap.add_argument("--cty", default="India"); ap.add_argument("--ts", type=int, default=2)
ap.add_argument("--view", default="name")
ap.add_argument("--holdout_fold", type=int, default=-1)
a = ap.parse_args()
cty, ts = a.cty, a.ts

c1 = np.asarray(pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["country"]).column("country").to_pylist())
i1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
cT = np.asarray(pq.read_table(f"{WORK}/normalized/train_s{ts}.parquet", columns=["country"]).column("country").to_pylist())
iT = pq.read_table(f"{WORK}/normalized/train_s{ts}.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
qi = np.flatnonzero(c1 == cty); ti = np.flatnonzero(cT == cty)
ql = np.full(len(i1), -1, np.int64); ql[qi] = np.arange(len(qi))
tl = np.full(len(iT), -1, np.int64); tl[ti] = np.arange(len(ti))
gt = read_ground_truth("train"); r1 = {e: i for i, e in enumerate(i1)}; rT = {e: i for i, e in enumerate(iT)}
import hashlib


def fold_of(e):
    return int(hashlib.blake2b(e.encode(), digest_size=2).hexdigest(), 16) % 5


P = []
for s1, tg in gt.items():
    i = ql[r1[s1]]
    if i < 0: continue
    if a.holdout_fold >= 0 and fold_of(s1) != a.holdout_fold:
        continue          # score ONLY entities the encoder never trained on
    for e in tg:
        j = rT.get(e)
        if j is not None and tl[j] >= 0: P.append((i, tl[j]))
P = np.asarray(P, np.int64)
truth = np.sort(P[:, 0] * np.int64(6_000_000) + P[:, 1])
print(f"{cty} S{ts}: truth={len(truth):,}", flush=True)

for tag in ("", a.tag):
    try:
        E1 = np.load(f"{WORK}/emb/train_s1_{a.view}{tag}.npy", mmap_mode="r")
        ET = np.load(f"{WORK}/emb/train_s{ts}_{a.view}{tag}.npy", mmap_mode="r")
    except FileNotFoundError:
        print(f"  [{tag or 'base'}] embeddings missing, skipped", flush=True); continue
    t0 = time.time()
    xb = np.ascontiguousarray(ET[ti], dtype=np.float32)
    idx = faiss.IndexFlatIP(xb.shape[1])
    co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
    gi = faiss.index_cpu_to_all_gpus(idx, co=co); gi.add(xb)
    K = 50
    I = np.empty((len(qi), K), np.int64)
    for s in range(0, len(qi), 65536):
        e = min(s + 65536, len(qi))
        _, I[s:e] = gi.search(np.ascontiguousarray(E1[qi[s:e]], dtype=np.float32), K)
    for k in (10, 30, 50):
        g = np.sort(np.repeat(np.arange(len(qi), dtype=np.int64), k) * np.int64(6_000_000)
                    + I[:, :k].reshape(-1))
        ii = np.clip(np.searchsorted(g, truth), 0, len(g) - 1)
        print(f"  [{tag or 'base'}] {a.view} recall@{k:<3} = {float((g[ii]==truth).mean()):.5f} ({time.time()-t0:.0f}s)", flush=True)
    del gi, xb
