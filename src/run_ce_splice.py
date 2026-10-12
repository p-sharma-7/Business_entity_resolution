"""Rebuild the test cross-encoder score file after France was re-scored.

The uncertain band is defined on stage-2 p2, so when France's p2 changes the
band's row set changes too. India/US pairs are unchanged: their scores are
reused exactly, matched by (S1, target, source) key. French band pairs are
scored with the France-adapted cross-encoder. The result is written in the
layout run_final.py expects, so it reuses the file instead of re-scoring.
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK
from run_ce import LO, HI, GPUS

COUNTRY = "France"


def key(q, t, s):
    return (q.astype(np.int64) << np.int64(33)) | (t.astype(np.int64) << np.int64(2)) | s.astype(np.int64)


def main():
    z = np.load(f"{WORK}/pred/test_scores.npz")
    q, t, src, p2 = z["q"], z["t"], z["src"], z["p2"]
    band = np.flatnonzero((p2 > LO) & (p2 < HI))
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet",
                                  columns=["country"]).column("country").to_pylist())
    fr = c1[q[band]] == COUNTRY
    old = np.load(f"{WORK}/pred/ce_score_test.npz")
    old_s = np.load(f"{WORK}/pred/ce_scores_test.npy")
    ok = key(old["q"], old["t"], old["src"])
    o = np.argsort(ok); ok_sorted = ok[o]; os_sorted = old_s[o]
    ce = np.full(len(band), np.nan, np.float32)
    nk = key(q[band[~fr]], t[band[~fr]], src[band[~fr]])
    pos = np.clip(np.searchsorted(ok_sorted, nk), 0, len(ok_sorted) - 1)
    hit = ok_sorted[pos] == nk
    assert hit.all(), f"{(~hit).sum()} India/US band pairs missing from the previous CE file"
    ce[~fr] = os_sorted[pos]
    print(f"band={len(band):,}: reused {int((~fr).sum()):,} India/US scores; "
          f"scoring {int(fr.sum()):,} French pairs with the France-adapted CE", flush=True)

    fb = band[fr]
    np.savez(f"{WORK}/pred/ce_fr_score.npz", q=q[fb], t=t[fb], src=src[fb])
    np.lib.format.open_memmap(f"{WORK}/pred/ce_fr_scores.npy", mode="w+", dtype=np.float32,
                              shape=(len(fb),)).flush()
    here = os.path.dirname(os.path.abspath(__file__))
    procs = [subprocess.Popen([sys.executable, f"{here}/run_ce_fr.py", "_worker", str(r), str(len(GPUS))],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
             for r, g in enumerate(GPUS)]
    for p in procs:
        p.wait()
        if p.returncode:
            raise SystemExit("French CE scoring failed")
    ce[fr] = np.load(f"{WORK}/pred/ce_fr_scores.npy")
    assert np.isfinite(ce).all()
    np.savez(f"{WORK}/pred/ce_score_test.npz", idx=band, q=q[band], t=t[band], src=src[band])
    np.save(f"{WORK}/pred/ce_scores_test.npy", ce)
    print("wrote ce_score_test.npz / ce_scores_test.npy aligned to the new band", flush=True)


if __name__ == "__main__":
    main()
