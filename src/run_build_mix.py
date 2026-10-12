"""Combine India/US decisions from the production pipeline with France decisions
from another scorer, restricted to the same pruned candidate set.

Usage: run_build_mix.py <name> <france_scores.npz> <th1,th2,...>
The France scores file must hold q, t, src, pc (calibrated) for France pairs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT
from er import decide as D


def key(q, t, s):
    return (q.astype(np.int64) << np.int64(33)) | (t.astype(np.int64) << np.int64(2)) | s.astype(np.int64)


def main(name, fr_path, ths):
    zf = np.load(f"{WORK}/pred/test_final_V6.npz")          # production pruned set + stage-3 cal
    q, t, src, cal6 = zf["q"], zf["t"], zf["src"], zf["cal"]
    s1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["entity_id", "country"])
    ids1 = s1.column("entity_id").to_pylist(); c1 = np.asarray(s1.column("country").to_pylist())
    idsT = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet",
                             columns=["entity_id"]).column("entity_id").to_pylist() for s in (2, 3)}
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    fr = c1[q] == "France"
    keep_iu = D.unique_target_filter(tk, cal6) & (cal6 >= 0.7) & ~fr

    zr = np.load(fr_path)
    rk = key(zr["q"], zr["t"], zr["src"]); o = np.argsort(rk)
    pk = key(q[fr], t[fr], src[fr])
    pos = np.clip(np.searchsorted(rk[o], pk), 0, len(rk) - 1)
    hit = rk[o][pos] == pk
    assert hit.all(), f"{(~hit).sum()} pruned France pairs missing from {fr_path}"
    pc = np.full(len(q), -1.0, np.float32); pc[np.flatnonzero(fr)] = zr["pc"][o][pos]
    uf = D.unique_target_filter(tk[fr], pc[fr])
    for th in ths:
        k = keep_iu.copy()
        kf = np.zeros(len(q), bool); kf[np.flatnonzero(fr)] = uf & (pc[fr] >= th)
        k |= kf
        out = f"{ROOT}/output_variants/{name}_th{th}"; os.makedirs(out, exist_ok=True)
        b = [[] for _ in ids1]; sel = np.flatnonzero(k)
        for i in sel[np.argsort(q[sel], kind="stable")]:
            b[q[i]].append(idsT[int(src[i])][t[i]])
        with open(f"{out}/matching_results.tsv", "w", newline="") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            for i, e in enumerate(ids1):
                fh.write(e + "\t" + ",".join(dict.fromkeys(b[i])) + "\n")
        n = np.bincount(q[kf], minlength=len(ids1))[c1 == "France"]
        print(f"{name} th={th}: France links {int(kf.sum()):,} pred/S1={n.mean():.3f} "
              f"empty={100*(n==0).mean():.2f}%  -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")])
