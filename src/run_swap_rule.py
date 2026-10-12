"""Validate a "one-word swap at the same address" veto on labelled data first.

Pattern: the S1 name and the candidate name differ by exactly ONE content word
on each side, neither word is a generic filler the generator swaps into true
matches (the label-mined sinks: center, services, care, ...), and the address
is the same. On the leaderboard, French links of this shape behaved like false
matches. Before applying the veto to France it is measured on India/US fold 4,
where the labels are real: if the flagged selected links there are mostly
wrong, the rule reflects the generator's distractor process, not a French
quirk, and is safe to transfer.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT

COLS = ["name_content", "addr_content", "addr_first_num", "addr_missing"]


def sinks():
    s = set()
    for f in ("name_ALL", "addr_ALL", "name_US", "name_India"):
        p = f"{ROOT}/models/alias_{f}.json"
        if os.path.exists(p):
            s |= set(json.load(open(p, encoding="utf-8")).get("sinks", []))
    return s | {"fils", "services"}          # France sinks that passed the drop-rate guard


def flag(qv, tv, qi, ti, SINK):
    """Boolean flag per pair (qi into S1 views, ti into target views)."""
    out = np.zeros(len(qi), bool)
    for k, (a, b) in enumerate(zip(qi, ti)):
        A = set(qv["name_content"][a].split()); B = set(tv["name_content"][b].split())
        da, db = A - B, B - A
        if len(da) != 1 or len(db) != 1:
            continue
        x, y = next(iter(da)), next(iter(db))
        if x in SINK or y in SINK or len(x) < 3 or len(y) < 3:
            continue
        if tv["addr_missing"][b] or qv["addr_missing"][a]:
            continue
        if qv["addr_first_num"][a] != tv["addr_first_num"][b]:
            continue
        aa = set(qv["addr_content"][a].split()); ab = set(tv["addr_content"][b].split())
        if not aa or not ab or len(aa & ab) / len(aa | ab) < 0.6:
            continue
        # a typo'd word (small edit distance) is noise, not a different word
        from rapidfuzz.distance import Levenshtein
        if Levenshtein.normalized_similarity(x, y) >= 0.6:
            continue
        out[k] = True
    return out


def main():
    SINK = sinks()
    # ---- India/US fold 4, real labels ----------------------------------------
    z = np.load(f"{WORK}/pred/oof.npz"); f4 = np.load(f"{WORK}/pred/fold4_stage3.npz")
    r4, sel = f4["rows"], f4["sel"]
    q, t, src, y = z["q"][r4], z["t"][r4], z["src"][r4], z["y"][r4]
    S1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=COLS).to_pydict()
    T = {s: pq.read_table(f"{WORK}/normalized/train_s{s}.parquet", columns=COLS).to_pydict() for s in (2, 3)}
    fl = np.zeros(len(r4), bool)
    for s in (2, 3):
        m = np.flatnonzero(sel & (src == s))
        fl[m] = flag(S1, T[s], q[m], t[m], SINK)
    n = int(fl.sum())
    prec = float(y[fl].mean()) if n else float("nan")
    print(f"India/US fold 4: selected links={int(sel.sum()):,}; flagged by the swap rule={n:,}; "
          f"share of flagged that are TRUE matches = {prec:.3f}", flush=True)

    # ---- France (V6 selection), no labels -------------------------------------
    zf = np.load(f"{WORK}/pred/test_final_V6.npz")
    qf, tf, sf, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["country"]).column("country").to_pylist())
    from er import decide as D
    tk = tf.astype(np.int64) + sf.astype(np.int64) * np.int64(10_000_000)
    keep = D.unique_target_filter(tk, cal) & np.where(c1[qf] == "France", cal >= 0.95, cal >= 0.7)
    Sq = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=COLS).to_pydict()
    St = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet", columns=COLS).to_pydict() for s in (2, 3)}
    flt = np.zeros(len(qf), bool)
    for s in (2, 3):
        m = np.flatnonzero(keep & (sf == s))
        flt[m] = flag(Sq, St[s], qf[m], tf[m], SINK)
    for c in ("France", "India", "US"):
        mc = keep & (c1[qf] == c)
        print(f"test {c:<7} selected={int(mc.sum()):>9,} flagged={int((flt & mc).sum()):>7,} "
              f"({100*(flt & mc).sum()/mc.sum():.3f}%)", flush=True)
    np.savez(f"{WORK}/pred/swap_flags_V6.npz", flag=flt)
    json.dump({"iu_flagged": n, "iu_true_share": prec}, open(f"{ROOT}/eda/swap_rule.json", "w"))


if __name__ == "__main__":
    main()
