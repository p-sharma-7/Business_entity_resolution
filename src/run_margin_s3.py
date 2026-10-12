"""Close-contest rule at STAGE 3, in a consistently scored fold-4 sub-world.

The fold-4 stage-3 harness lets fold-4 S1s compete with other folds' S1s that
only have stage-2 scores, so contests there are inconsistent. Here the world is
restricted to fold-4 S1s and the targets whose TRUE parent is a fold-4 S1 (every
fold-4 entity's true set lies inside it, so each entity's F0.5 is exact). All
contests are then between stage-3 scores, as on test. Rival density is 1/5 of
the real world, so the gain is diluted, but the per-contest true share of the
winner - the quantity the rule acts on - is measured at stage 3.

Tuned on stage-3 subfolds {0,1,2} ("A"), checked on {3,4} ("B").
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D
from run_stage3 import sub_fold


def contests(tkey, cal):
    order = np.lexsort((-cal, tkey)); g = tkey[order]
    new = np.empty(len(g), bool); new[0] = True; np.not_equal(g[1:], g[:-1], out=new[1:])
    st = np.flatnonzero(new); sz = np.diff(np.append(st, len(g)))
    win = np.zeros(len(cal), bool); win[order[st]] = True
    sec = np.zeros(len(cal), np.float32)
    sec[order[st]] = np.where(sz > 1, cal[order][np.minimum(st + 1, len(g) - 1)], 0.0)
    return win, sec


def main():
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz"); f4 = np.load(f"{WORK}/pred/fold4_stage3.npz")
    r4, cal = f4["rows"], f4["cal"]
    q, t, src, y = z["q"][r4], z["t"][r4], z["src"][r4], z["y"][r4]
    ids = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    gt = read_ground_truth("train"); N = np.array([len(gt[e]) for e in ids], np.int32)
    ef = np.array([fold_of(e) for e in ids], np.int8)
    parent_f4 = set()
    for e, ts in gt.items():
        if ts and fold_of(e) == 4:
            parent_f4.update(ts)
    tid = {s: pq.read_table(f"{WORK}/normalized/train_s{s}.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
           for s in (2, 3)}
    inw = {s: np.fromiter((x in parent_f4 for x in tid[s]), bool, len(tid[s])) for s in (2, 3)}
    keepw = np.where(src == 2, inw[2][np.minimum(t, len(inw[2]) - 1)], inw[3][np.minimum(t, len(inw[3]) - 1)])
    q, t, src, y, cal = q[keepw], t[keepw], src[keepw], y[keepw], cal[keepw]
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    E = np.flatnonzero(ef == 4)
    sf = np.array([sub_fold(ids[e]) for e in E], np.int8)
    rm = np.full(len(ids), -1, np.int64); rm[E] = np.arange(len(E))
    print(f"sub-world pairs={len(q):,} fold-4 entities={len(E):,} ({time.time()-t0:.0f}s)", flush=True)
    win, sec = contests(tkey, cal)
    parts = {"A": np.isin(sf, (0, 1, 2)), "B": np.isin(sf, (3, 4))}

    def f05(keep, part):
        e = parts[part]; r = np.full(len(E), -1, np.int64); r[e] = np.arange(int(e.sum()))
        qq = r[rm[q]]; m = keep & (qq >= 0)
        return D.macro_f05_fast(qq[m], np.ones(int(m.sum()), bool), y[m], int(e.sum()), N[E][e])

    base = win & (cal >= 0.7)
    b = {p: f05(base, p) for p in parts}
    print(f"baseline stage-3 decision (th 0.7): A={b['A']:.5f} B={b['B']:.5f} all-world-harness ref 0.99062")
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        s = base & (sec >= lo) & (sec < hi)
        print(f"  selected winners, runner-up in [{lo},{hi}): n={int(s.sum()):,} true share={y[s].mean():.4f}")
    res = {}
    for lam in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for th in (0.4, 0.5, 0.6, 0.7):
            k = win & (cal - lam * sec >= th)
            res[(lam, th)] = (f05(k, "A"), f05(k, "B"))
    for k_, v in sorted(res.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  lam={k_[0]} th={k_[1]}: A={v[0]:.5f} B={v[1]:.5f} dA={v[0]-b['A']:+.5f} dB={v[1]-b['B']:+.5f}")
    best = max(res, key=lambda k_: res[k_][0])
    print(f"tuned on A -> lam={best[0]} th={best[1]}; held-out B delta {res[best][1]-b['B']:+.5f}")
    json.dump({f"{k_[0]}_{k_[1]}": v for k_, v in res.items()} | {"base": b}, open(f"{ROOT}/eda/margin_s3.json", "w"))


if __name__ == "__main__":
    main()
