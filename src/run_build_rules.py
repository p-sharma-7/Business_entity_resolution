"""Build a France post-rule variant on top of the V6 production scores.

Starts from V6 (stage-3 cal, target-uniqueness, India/US th 0.7, France th
ER_TH_FR) and removes French links flagged by label-validated rules:

  numrule  : house number differs from the S1's AND from the S1's other links
             (or differs with no siblings). France's generator almost never
             noises house numbers (2.1% of confident links vs 23.5% India,
             11.5% US), so a conflict there is strong evidence of a wrong link.
  contest  : the target is also claimed by ANOTHER S1 with cal >= ER_CONTEST.
             On India/US (all S1, real labels) such winners are right only
             53-55% of the time, below the ~75% precision F0.5 needs to keep a
             link; France has 500x more of them (identical names, no address).

A rule never removes an S1's last remaining link (ER_PROTECT=1, default): if
that link is right the entity drops from ~1.0 to 0, and emptying only pays when
the entity is a true singleton (~6%), so the per-link break-even assumed above
does not hold there.

India/US links are never changed.  Usage:
  ER_TH_FR=0.97 ER_NUMRULE=1 ER_CONTEST=0.7 python run_build_rules.py <name>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT
from er import decide as D

NUMRULE_CATS = ("differs from S1 AND siblings", "differs, no siblings")


def main(name):
    th_fr = float(os.environ.get("ER_TH_FR", "0.97"))
    use_num = os.environ.get("ER_NUMRULE", "1") == "1"
    contest = float(os.environ.get("ER_CONTEST", "2"))      # > 1 disables
    zf = np.load(f"{WORK}/pred/test_final_V6.npz")
    q, t, src, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
    s1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["entity_id", "country"])
    ids1 = s1.column("entity_id").to_pylist(); c1 = np.asarray(s1.column("country").to_pylist())
    idsT = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet",
                             columns=["entity_id"]).column("entity_id").to_pylist() for s in (2, 3)}
    fr = c1[q] == "France"
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    win = D.unique_target_filter(tk, cal)
    keep = win & np.where(fr, cal >= th_fr, cal >= 0.7)
    n0 = int((keep & fr).sum())
    base = keep.copy()
    drop = np.zeros(len(q), bool)

    if use_num:
        nr = np.load(f"{WORK}/pred/numrule_V6.npz")
        assert th_fr == 0.97 and np.array_equal(nr["keep"], keep), "numrule flags were computed for V6 th 0.97"
        flag = np.isin(nr["cat"], NUMRULE_CATS) & fr
        drop |= flag
        print(f"numrule flagged {int(flag.sum()):,} French links", flush=True)

    if contest <= 1:
        order = np.lexsort((-cal, tk)); g = tk[order]
        new = np.empty(len(g), bool); new[0] = True; np.not_equal(g[1:], g[:-1], out=new[1:])
        st = np.flatnonzero(new); sz = np.diff(np.append(st, len(g)))
        second = np.zeros(len(q), np.float32)
        second[order[st]] = np.where(sz > 1, cal[order][np.minimum(st + 1, len(g) - 1)], 0.0)
        flag = base & fr & (second >= contest)
        drop |= flag
        print(f"contest rule (runner-up >= {contest}) flagged {int(flag.sum()):,} French links", flush=True)

    keep = base & ~drop
    if os.environ.get("ER_PROTECT", "1") == "1":
        left = np.bincount(q[keep], minlength=len(ids1))
        restore = base & drop & (left[q] == 0)
        # restore only the entity's single best flagged link
        r = np.flatnonzero(restore); r = r[np.lexsort((-cal[r], q[r]))]
        first = np.r_[True, q[r][1:] != q[r][:-1]] if len(r) else np.zeros(0, bool)
        keep[r[first]] = True
        print(f"protect: kept {int(first.sum()):,} flagged links that were their S1's only link", flush=True)
    print(f"removed {int((base & ~keep & fr).sum()):,} French links in total", flush=True)

    out = f"{ROOT}/output_variants/{name}"; os.makedirs(out, exist_ok=True)
    b = [[] for _ in ids1]; sel = np.flatnonzero(keep)
    for i in sel[np.argsort(q[sel], kind="stable")]:
        b[q[i]].append(idsT[int(src[i])][t[i]])
    with open(f"{out}/matching_results.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for i, e in enumerate(ids1):
            fh.write(e + "\t" + ",".join(dict.fromkeys(b[i])) + "\n")
    nf = np.bincount(q[keep & fr], minlength=len(ids1))[c1 == "France"]
    print(f"{name}: France links {n0:,} -> {int((keep & fr).sum()):,}; pred/S1={nf.mean():.3f} "
          f"empty={100*(nf == 0).mean():.2f}% -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
