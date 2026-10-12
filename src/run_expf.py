"""Per-entity expected-F0.5 link selection vs the global threshold (fold 4).

With calibrated link probabilities p (target-uniqueness winners only), an
entity's expected F0.5 for keeping its top-k links is approximated by
  k = 0 : P(no true link) = prod(1 - p)
  k >= 1: 1.25 * sum_{i<=k} p_i / (0.25 * E[N] + k),   E[N] = sum(p) + m0
(m0: expected true links outside the candidates). F0.5's break-even therefore
depends on the entity: ~0.5 for a first link, ~0.77 for a fourth. The global
threshold (0.7) is compared on the honest fold-4 harness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, read_ground_truth, fold_of
from er import decide as D


def expf_select(q, p, cand, m0=0.0, floor=0.02):
    """Boolean selection maximising the approximate expected F0.5 per entity."""
    idx = np.flatnonzero(cand & (p > floor))
    o = idx[np.lexsort((-p[idx], q[idx]))]
    qs = q[o]; ps = p[o].astype(np.float64)
    st = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]]); en = np.r_[st[1:], len(o)]
    sel = np.zeros(len(q), bool)
    for a, b in zip(st, en):
        pp = ps[a:b]
        en_ = pp.sum() + m0
        k = np.arange(1, len(pp) + 1)
        ef = 1.25 * np.cumsum(pp) / (0.25 * en_ + k)
        e0 = np.prod(1 - pp) * np.exp(-m0)
        best = int(np.argmax(ef))
        if ef[best] > e0:
            sel[o[a:a + best + 1]] = True
    return sel


def main():
    z = np.load(f"{WORK}/pred/oof.npz"); f4 = np.load(f"{WORK}/pred/fold4_stage3.npz")
    r4, sel, cal, uniq = f4["rows"], f4["sel"], f4["cal"], f4["uniq"]
    q, y = z["q"][r4], z["y"][r4]
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"]).to_pydict()
    ids, cty = tb["entity_id"], np.asarray(tb["country"])
    gt = read_ground_truth("train")
    E = np.array([i for i, (e, c) in enumerate(zip(ids, cty)) if c != "France" and fold_of(e) == 4])
    rm = np.full(len(ids), -1, np.int64); rm[E] = np.arange(len(E)); N = np.array([len(gt[ids[e]]) for e in E])

    def f05(k):
        return D.macro_f05_fast(rm[q[k]], np.ones(int(k.sum()), bool), y[k], len(E), N)

    base = f05(sel)
    print(f"global threshold (production): {base:.5f}  links={int(sel.sum()):,}")
    for m0 in (0.0, 0.02, 0.05):
        s = expf_select(q, cal, uniq, m0)
        print(f"expected-F0.5 per entity m0={m0:.2f}: {f05(s):.5f} ({f05(s)-base:+.5f})  links={int(s.sum()):,}")


if __name__ == "__main__":
    main()
