"""Close-contest rule, validated where every S1 is scored the same way.

Most remaining India/US loss is target-uniqueness contests: an address-less
target that two S1 entities with (near-)identical names both claim. A 50/50
link costs more (FP) than it earns (TP) under F0.5, so the winner of a close
contest may be better dropped. The fold-4 stage-3 harness scores the other
folds' competitors with stage 2, so contests there are inconsistent; stage-2
OOF scores cover ALL S1 uniformly, so the rule is tuned here on half A of the
entities and checked on half B.

Rule: keep winner W of target T iff  cal_W - lam * cal_R >= th, where R is the
runner-up claim on T (0 if none); optionally only when T has no address.
"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
import joblib
from er.io_utils import WORK, ROOT, read_ground_truth
from er import decide as D
from run_prune_sim import prune_mask


def main():
    t0 = time.time()
    z = np.load(f"{WORK}/pred/oof.npz")
    q, t, src, y, p1, p2 = z["q"], z["t"], z["src"], z["y"], z["p1"], z["p2"]
    U = np.flatnonzero(prune_mask(q, src, p1, 10))
    q, t, src, y, p2 = q[U], t[U], src[U], y[U], p2[U]
    cal = joblib.load(f"{ROOT}/models/calibrator.pkl").predict(p2).astype(np.float32)
    tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    ids = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    gt = read_ground_truth("train"); N = np.array([len(gt[e]) for e in ids], np.int32)
    half = np.array([int(hashlib.md5(e.encode()).hexdigest(), 16) & 1 for e in ids], np.int8)
    amiss = {s: np.asarray(pq.read_table(f"{WORK}/normalized/train_s{s}.parquet", columns=["addr_missing"])
                           .column("addr_missing").to_pylist(), bool) for s in (2, 3)}
    tmiss = np.where(src == 2, amiss[2][np.minimum(t, len(amiss[2]) - 1)], amiss[3][np.minimum(t, len(amiss[3]) - 1)])
    print(f"pairs={len(q):,} ({time.time()-t0:.0f}s)", flush=True)

    # winner / runner-up per target
    order = np.lexsort((-cal, tkey)); g = tkey[order]
    new = np.empty(len(g), bool); new[0] = True; np.not_equal(g[1:], g[:-1], out=new[1:])
    st = np.flatnonzero(new); sz = np.diff(np.append(st, len(g)))
    win = np.zeros(len(q), bool); win[order[st]] = True
    second = np.zeros(len(q), np.float32)
    r2 = np.where(sz > 1, cal[order][np.minimum(st + 1, len(g) - 1)], 0.0)
    second[order[st]] = r2

    ents = {h: np.flatnonzero(half == h) for h in (0, 1)}
    rm = {}
    for h, e in ents.items():
        r = np.full(len(ids), -1, np.int64); r[e] = np.arange(len(e)); rm[h] = r

    def f05(keep, h):
        m = keep & (rm[h][q] >= 0)
        return D.macro_f05_fast(rm[h][q[m]], np.ones(int(m.sum()), bool), y[m], len(ents[h]), N[ents[h]])

    res = {}
    for scope in (() if os.environ.get("ER_ONLY_FR") else ("all", "noaddr")):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            adj = cal - lam * second if scope == "all" else np.where(tmiss, cal - lam * second, cal)
            for th in (0.5, 0.6, 0.7, 0.8):
                k = win & (adj >= th)
                res[(scope, lam, th)] = (f05(k, 0), f05(k, 1))
        print(f"  scope={scope} done ({time.time()-t0:.0f}s)", flush=True)
    base = res.get(("all", 0.0, 0.7), (0.0, 0.0))
    print(f"baseline (lam=0, th=0.7): A={base[0]:.5f} B={base[1]:.5f}")
    for k_, v in sorted(res.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {k_}: A={v[0]:.5f} B={v[1]:.5f}  dA={v[0]-base[0]:+.5f} dB={v[1]-base[1]:+.5f}")
    bestA = max(res, key=lambda k_: res[k_][0]) if res else None
    if res: print(f"tuned on A -> {bestA}; held-out B: {res[bestA][1]:.5f} vs baseline {base[1]:.5f} "
          f"(delta {res[bestA][1]-base[1]:+.5f})")
    m = win & (cal >= 0.7)
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        s = m & (second >= lo) & (second < hi)
        print(f"  selected winners with runner-up in [{lo},{hi}): n={int(s.sum()):,} true share={y[s].mean():.4f} "
              f"(no-addr share {tmiss[s].mean():.2f})")
    json.dump({str(k_): v for k_, v in res.items()}, open(f"{ROOT}/eda/margin_rule.json", "w"), indent=0)

    # France operating point: winner cal >= 0.97, drop if runner-up >= x
    print("France-like operating point (winner cal >= 0.97):")
    hi = win & (cal >= 0.97)
    b97 = (f05(hi, 0), f05(hi, 1))
    for lo_, hi_ in ((0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        s_ = hi & (second >= lo_) & (second < hi_)
        print(f"  winner>=0.97, runner-up in [{lo_},{hi_}): n={int(s_.sum()):,} true share={y[s_].mean():.4f}")
    for x in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        k = hi & (second < x)
        a_, b_ = f05(k, 0), f05(k, 1)
        print(f"  drop if runner-up >= {x}: A {a_-b97[0]:+.5f}  B {b_-b97[1]:+.5f}  (dropped {int((hi & (second >= x)).sum()):,})")


if __name__ == "__main__":
    main()
