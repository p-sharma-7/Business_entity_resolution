"""Neural candidate channel: exact inner-product search on the GPUs.

Embeddings are L2-normalised, so inner product is cosine. With only ~3M
targets per (country, source) partition and 7 GPUs available, an exact
IndexFlatIP sharded across the devices is both faster to build and strictly
better for recall than IVF-SQ8 - there is no reason to accept approximation
error here.

This channel exists for the slice lexical blocking cannot reach: targets whose
name is transliterated into an Indic script (85% of the residual after
character-n-gram blocking) while their address has been truncated to
low-information tokens.
"""
import sys, os, time, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
import faiss
from er.io_utils import WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("view", choices=["name", "combined"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--qbs", type=int, default=65536)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cty1 = pq.read_table(f"{WORK}/normalized/{a.split}_s1.parquet",
                         columns=["country"]).column("country").to_pylist()
    cty1 = np.asarray(cty1)
    E1 = np.load(f"{WORK}/emb/{a.split}_s1_{a.view}{a.tag}.npy", mmap_mode="r")
    os.makedirs(f"{WORK}/ann", exist_ok=True)

    res = faiss.StandardGpuResources if False else None
    for ts in (2, 3):
        ctyT = np.asarray(pq.read_table(f"{WORK}/normalized/{a.split}_s{ts}.parquet",
                                        columns=["country"]).column("country").to_pylist())
        ET = np.load(f"{WORK}/emb/{a.split}_s{ts}_{a.view}{a.tag}.npy", mmap_mode="r")
        for cty in sorted(set(cty1.tolist())):
            qi = np.flatnonzero(cty1 == cty)
            ti = np.flatnonzero(ctyT == cty)
            if len(qi) == 0 or len(ti) == 0:
                continue
            out = f"{WORK}/ann/{a.split}_S{ts}_{cty}_{a.view}{a.tag}.npz"
            if os.path.exists(out):
                print(f"skip {out}", flush=True); continue
            t0 = time.time()
            xb = np.ascontiguousarray(ET[ti], dtype=np.float32)
            index = faiss.IndexFlatIP(xb.shape[1])
            co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
            gindex = faiss.index_cpu_to_all_gpus(index, co=co)
            gindex.add(xb)
            print(f"[{a.split} S{ts} {cty} {a.view}] index {len(ti):,} built {time.time()-t0:.0f}s", flush=True)
            D = np.empty((len(qi), a.k), np.float32)
            I = np.empty((len(qi), a.k), np.int64)
            for s in range(0, len(qi), a.qbs):
                e = min(s + a.qbs, len(qi))
                xq = np.ascontiguousarray(E1[qi[s:e]], dtype=np.float32)
                D[s:e], I[s:e] = gindex.search(xq, a.k)
            np.savez(out, q=qi.astype(np.int32), t=ti[I].astype(np.int32),
                     d=D.astype(np.float16))
            print(f"   searched {len(qi):,} in {time.time()-t0:.0f}s -> {out}", flush=True)
            del gindex, xb


if __name__ == "__main__":
    main()
