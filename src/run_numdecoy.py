"""Same-form decoys: separate house-number typos from sibling-company records.

Known decoys (legal-form conflict + different house number) are the reference
population; non-conflict links whose number differs are a mix of true matches
with a number typo and same-form decoys. Two signals are compared:
  support : other candidate records of the same S1 at the SAME different
            number on the S1's street (a decoy company has several records)
  edit1   : the two numbers differ by one digit edit (a typo) vs anything else
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from rapidfuzz.distance import Levenshtein
from er.io_utils import WORK
from er.legalform import legal_form, relation

ND = ("differs from S1 AND siblings", "differs, no siblings", "differs from S1, agrees w/ siblings")


def main():
    zf = np.load(f"{WORK}/pred/test_final.npz"); q, t, src = zf["q"], zf["t"], zf["src"]
    nr = np.load(f"{WORK}/pred/numrule_V6.npz"); cat, keep = nr["cat"], nr["keep"]
    C = ["name_pre", "addr_first_num", "addr_content", "name_content"]
    S1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=C + ["country"]).to_pydict()
    c1 = np.asarray(S1["country"])
    T = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet", columns=C).to_pydict() for s in (2, 3)}
    fr = c1[q] == "France"
    o = np.argsort(q, kind="stable"); qs = q[o]
    sel = np.flatnonzero(keep & fr & np.isin(cat, ND))
    rows = []
    for i in sel:
        a = q[i]; s = int(src[i]); b = t[i]
        na, nb = S1["addr_first_num"][a], T[s]["addr_first_num"][b]
        if not na or not nb:
            continue
        rel = relation(legal_form(S1["name_pre"][a]), legal_form(T[s]["name_pre"][b]))
        sa = set(S1["addr_content"][a].split())
        lo, hi = np.searchsorted(qs, [a, a + 1])
        sup = 0
        na_ = set(S1["name_content"][a].split())
        for j in o[lo:hi]:
            if j == i:
                continue
            Tj = T[int(src[j])]
            if Tj["addr_first_num"][t[j]] == nb:
                sb = set(Tj["addr_content"][t[j]].split()); nm = set(Tj["name_content"][t[j]].split())
                if (sa and sb and len(sa & sb) / len(sa | sb) >= 0.5
                        and na_ and nm and len(na_ & nm) / len(na_ | nm) >= 0.5):
                    sup += 1                      # same company name at the new number
        e1 = Levenshtein.distance(na, nb) == 1
        rows.append((i, rel, sup, e1))
    rel = np.array([r[1] for r in rows]); sup = np.array([r[2] for r in rows]); e1 = np.array([r[3] for r in rows])
    names = {0: "both forms missing", 1: "one form missing", 2: "same form", 3: "FORM CONFLICT (known decoys)"}
    print(f"French selected links whose house number differs: {len(rows):,}")
    print(f"{'legal-form relation':<30} {'links':>6} {'support>=1':>11} {'one-digit edit':>15}")
    for r in (3, 2, 1, 0):
        m = rel == r
        print(f"{names[r]:<30} {m.sum():>6,} {100*(sup[m]>=1).mean():>10.1f}% {100*e1[m].mean():>14.1f}%")
    for r in (2, 1, 0):
        m = rel == r
        print(f"  {names[r]}: support>=1 & not one-digit: {int((m&(sup>=1)&~e1).sum()):,} | support>=1 & one-digit: {int((m&(sup>=1)&e1).sum()):,} "
              f"| no support & not one-digit: {int((m&(sup==0)&~e1).sum()):,} | no support & one-digit: {int((m&(sup==0)&e1).sum()):,}")
    np.savez(f"{WORK}/pred/numdecoy.npz", idx=np.array([r[0] for r in rows]), rel=rel, sup=sup, e1=e1)


if __name__ == "__main__":
    main()
