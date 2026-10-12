"""House-number consistency: France's generator barely noises house numbers.

Confident matches whose first house number differs from the S1's: India 23.5%,
US 11.5%, France 2.1%. The model learned number mismatch as weak evidence on
India/US; on France it should be strong. A link is flagged when its house
number differs from the S1's (allowing digit-drop / padding relations) AND
from the majority number of the S1's other confident links (so a noisy S1
number, with siblings agreeing among themselves, does not trigger it).

Reports the flag rate per country and, on India/US fold 4, the true-match
share of flagged links (real labels).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyarrow.parquet as pq
from collections import Counter
from er.io_utils import WORK
from er import decide as D


def related(a, b):
    return a == b or a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)


def flags(q, t, src, keep, S1fn, Tfn):
    """Per-link category for kept links."""
    idx = np.flatnonzero(keep)
    by_s1 = {}
    for i in idx:
        by_s1.setdefault(int(q[i]), []).append(i)
    cat = np.full(len(q), "", object)
    for s, rows in by_s1.items():
        fs = S1fn[s]
        nums = {i: Tfn[int(src[i])][t[i]] for i in rows}
        for i in rows:
            fn = nums[i]
            if not fn or not fs:
                cat[i] = "no number"; continue
            if related(fn, fs):
                cat[i] = "agrees with S1"; continue
            sib = [nums[j] for j in rows if j != i and nums[j]]
            if not sib:
                cat[i] = "differs, no siblings"; continue
            maj, _ = Counter(sib).most_common(1)[0]
            cat[i] = "differs from S1, agrees w/ siblings" if related(fn, maj) else "differs from S1 AND siblings"
    return cat


def main():
    # India/US fold 4, real labels
    z = np.load(f"{WORK}/pred/oof.npz"); f4 = np.load(f"{WORK}/pred/fold4_stage3.npz")
    r4, sel = f4["rows"], f4["sel"]
    q, t, src, y = z["q"][r4], z["t"][r4], z["src"][r4], z["y"][r4]
    S1fn = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["addr_first_num"]).column("addr_first_num").to_pylist()
    Tfn = {s: pq.read_table(f"{WORK}/normalized/train_s{s}.parquet", columns=["addr_first_num"]).column("addr_first_num").to_pylist() for s in (2, 3)}
    cat = flags(q, t, src, sel, S1fn, Tfn)
    print("INDIA/US fold 4 selected links (real labels):")
    for c in sorted(set(cat[sel])):
        m = sel & (cat == c)
        print(f"   {c:<36} n={int(m.sum()):>9,} ({100*m.sum()/sel.sum():5.2f}%)  true share={y[m].mean():.4f}")

    # test, V6 France th 0.97 / India-US 0.7
    zf = np.load(f"{WORK}/pred/test_final_V6.npz"); qt, tt, st, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["country"]).column("country").to_pylist())
    u = D.unique_target_filter(tt.astype(np.int64) + st.astype(np.int64) * np.int64(10_000_000), cal)
    keep = u & np.where(c1[qt] == "France", cal >= 0.97, cal >= 0.7)
    S1t = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["addr_first_num"]).column("addr_first_num").to_pylist()
    Tt = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet", columns=["addr_first_num"]).column("addr_first_num").to_pylist() for s in (2, 3)}
    catt = flags(qt, tt, st, keep, S1t, Tt)
    for c in ("France", "India", "US"):
        m = keep & (c1[qt] == c)
        print(f"TEST {c} ({int(m.sum()):,} links):")
        for k in sorted(set(catt[m])):
            mm = m & (catt == k)
            print(f"   {k:<36} n={int(mm.sum()):>9,} ({100*mm.sum()/m.sum():5.2f}%)")
    np.savez(f"{WORK}/pred/numrule_V6.npz", cat=catt.astype(str), keep=keep)


if __name__ == "__main__":
    main()
