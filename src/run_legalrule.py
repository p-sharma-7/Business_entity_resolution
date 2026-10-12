"""French legal-form decoys: remove links to a sibling-company record.

The French generator plants decoy records of an S1 company: same name, a
DIFFERENT legal form (sarl->sa, sas->sasu, sci->snc, ...) and, 97% of the
time, a different house number on the same street. The view builder treats
French legal forms as droppable suffixes, so the model sees these decoys as
perfect matches (score ~1.0). On India/US, where forms survive normalisation,
the model already rejects form conflicts from labels (selected conflict links
are 99.7% true).

True matches never REPLACE a legal form (the generator only drops one or adds
a random one, uniformly over forms); every observed form change carries a
different house number 97% of the time (non-conflict links: 1.3%).

Rule (France only, on the selected links):
  * legal-form conflict (both named, different)                  -> remove
  * S1 has no form, record adds a decoy form (sa, snc) AND the
    house number differs (sa: 873 links vs ~27 for any other form) -> remove
Writes output_variants/<name>/matching_results.tsv.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT
from er import decide as D
from er.legalform import legal_form, legal_form_iu

DECOY_FORMS = ("sa", "snc")
TH_FR, TH_IU = float(os.environ.get("ER_TH_FRANCE", "0.97")), 0.7


def related(a, b):
    return a == b or a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)


def decoy_mask(q, t, src, sel, S1, T):
    out = np.zeros(len(q), bool)
    for i in np.flatnonzero(sel):
        a, b, s = q[i], t[i], int(src[i])
        fa, fb = legal_form(S1["name_pre"][a]), legal_form(T[s]["name_pre"][b])
        if fa and fb:
            out[i] = not set(fa.split("|")) & set(fb.split("|"))   # form conflict
        elif fb in DECOY_FORMS:
            na, nb = S1["addr_first_num"][a], T[s]["addr_first_num"][b]
            out[i] = bool(na and nb and not related(na, nb))       # SA version elsewhere
    return out


def numdecoy_mask(q, t, src, sel, S1, T):
    """Same-form sibling-company decoys: the record's house number differs from
    the S1's and from the S1's other selected links (or it has no others).

    France's generator barely noises house numbers (1.3% of its reliable links
    differ, vs 11.8% of India/US true matches), and these links carry the decoy
    signature as often as the legal-form decoys: another record with the same
    company name sits at the new number for 61-65% of them (known decoys: 57%);
    a typo would have none. A number found anywhere in the other address (e.g.
    an apartment number parsed first) never counts as a difference, and links
    whose number agrees with the S1's other links are kept (the S1's own
    number may be the odd one)."""
    out = np.zeros(len(q), bool)
    idx = np.flatnonzero(sel)
    by = {}
    for i in idx:
        by.setdefault(int(q[i]), []).append(i)
    for a, rows in by.items():
        na = S1["addr_first_num"][a]
        if not na:
            continue
        set_a = set(S1["addr_numset"][a].split(","))
        nums = {i: T[int(src[i])]["addr_first_num"][t[i]] for i in rows}
        for i in rows:
            nb = nums[i]
            if not nb or related(na, nb) or nb in set_a:
                continue
            if na in set(T[int(src[i])]["addr_numset"][t[i]].split(",")):
                continue
            sib = [nums[j] for j in rows if j != i and nums[j]]
            if sib and related(nb, Counter(sib).most_common(1)[0][0]):
                continue
            out[i] = True
    return out


# France legal-form noise, measured on its own reliable links (house number
# agrees with the S1): a named form is kept 70% / dropped 30%; a form-less name
# gains a random form 18% of the time (~3% per form); a form is never replaced.
def form_lik(fs, ft):
    if fs and ft:
        return 0.70 if set(fs.split("|")) & set(ft.split("|")) else 0.001
    if ft:
        return 0.03
    return 0.30 if fs else 0.82


def contested_mask(q, t, src, cal, sel, S1, T, min_ru, post_min=0.9, lf=None):
    """Selected links whose target is also claimed by another S1 at cal >=
    min_ru. The owner is inferred from the legal-form likelihood (prior ~ cal);
    the winner is dropped when its posterior is below post_min, the F0.5
    break-even for adding a link to an entity with other links (~0.73-0.77).
    India/US labels (stage-2 OOF, every S1 scored alike): contest winners are
    right 43% of the time; the form posterior lifts that only above 0.9 (77%),
    so 0.9 is the default. lf(i) returns the form parser for link i's country."""
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    o = np.lexsort((-cal, tk)); tks = tk[o]
    out = np.zeros(len(q), bool)
    for i in np.flatnonzero(sel):
        lo, hi = np.searchsorted(tks, [tk[i], tk[i] + 1])
        riv = [j for j in o[lo:hi] if j != i and q[j] != q[i] and cal[j] >= min_ru]
        if not riv:
            continue
        f = lf(i) if lf else legal_form
        ft = f(T[int(src[i])]["name_pre"][t[i]])
        w = cal[i] * form_lik(f(S1["name_pre"][q[i]]), ft)
        z = w + sum(cal[j] * form_lik(f(S1["name_pre"][q[j]]), ft) for j in riv)
        out[i] = w / z < post_min
    return out


def main(name):
    zf = np.load(f"{WORK}/pred/{os.environ.get('ER_FINAL', 'test_final')}.npz")   # run_final.py output
    q, t, src, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
    C = ["name_pre", "addr_first_num", "addr_numset"]
    s1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=["entity_id", "country"] + C).to_pydict()
    ids1, c1 = s1["entity_id"], np.asarray(s1["country"])
    T = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet", columns=["entity_id"] + C).to_pydict() for s in (2, 3)}
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    fr = c1[q] == "France"
    th = np.where(fr, TH_FR, TH_IU).astype(np.float32)
    for c in ("India", "US"):                     # per-country overrides (stage-3 config)
        if os.environ.get(f"ER_TH_{c.upper()}"):
            th[c1[q] == c] = float(os.environ[f"ER_TH_{c.upper()}"])
    keep = D.unique_target_filter(tk, cal) & (cal >= th)
    dec = decoy_mask(q, t, src, keep & fr, s1, T)
    if os.environ.get("ER_NUMDECOY") == "1":
        nd = numdecoy_mask(q, t, src, keep & fr, s1, T) & ~dec
        print(f"same-form house-number decoys: {int(nd.sum()):,}", flush=True)
        dec |= nd
    ru = float(os.environ.get("ER_CONTEST", "0"))
    if ru > 0:
        # contests over France and (ER_CONTEST_IU=1) India/US; forms parsed per country
        scope = fr | (os.environ.get("ER_CONTEST_IU") == "1")
        lf = lambda i: legal_form if fr[i] else legal_form_iu
        con = contested_mask(q, t, src, cal, keep & scope & ~dec, s1, T, ru,
                             float(os.environ.get("ER_CONTEST_POST", "0.9")), lf)
        n0 = int(con.sum())
        # never drop an S1's last link on a coin flip: for a one-link entity a
        # ~50% link breaks even (F 1->0 if right, 0->1 if it is a singleton)
        left = np.bincount(q[keep & ~dec & ~con], minlength=len(ids1))
        con &= left[q] > 0
        print(f"contested winners (runner-up >= {ru}): France {int((con & fr).sum()):,}, "
              f"India/US {int((con & ~fr).sum()):,}; kept as last link: {n0 - int(con.sum()):,}", flush=True)
        dec |= con
    final = keep & ~dec
    print(f"France links {int((keep & fr).sum()):,} -> {int((final & fr).sum()):,}; "
          f"India/US links {int((keep & ~fr).sum()):,} -> {int((final & ~fr).sum()):,}", flush=True)
    out = f"{ROOT}/output_variants/{name}"; os.makedirs(out, exist_ok=True)
    b = [[] for _ in ids1]; sel = np.flatnonzero(final)
    for i in sel[np.argsort(q[sel], kind="stable")]:
        b[q[i]].append(T[int(src[i])]["entity_id"][t[i]])
    with open(f"{out}/matching_results.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for i, e in enumerate(ids1):
            fh.write(e + "\t" + ",".join(dict.fromkeys(b[i])) + "\n")
    np.save(f"{WORK}/pred/legal_decoy_{name}.npy", dec)
    print(f"wrote {out}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
