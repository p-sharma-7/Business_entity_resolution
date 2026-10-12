"""Pair feature extraction.

Split by cost:
  * set features run over CSR-packed id sets with numba and never materialise a
    string, so they are affordable on the full raw candidate list;
  * string features (edit distance, token-sort/-set, Jaro-Winkler) use
    rapidfuzz's elementwise `process.cpdist(..., workers=-1)`, which is C++ and
    multithreaded, and run only on the pruned list;
  * group features are computed last, over both the S1 side and the TARGET
    side, because target-side competition is the strongest precision signal
    available (every matched target has exactly one true S1 parent).
"""
from __future__ import annotations
import numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from er.setfeat import build_csr, set_feats

SET_SPECS = [
    ("ntok",  "name_core",       0),
    ("ncon",  "name_content",    0),
    ("nch3",  "name_compact_hg", 3),
    ("nch4",  "name_compact_hg", 4),
    ("atok",  "addr_content",    0),
    ("ach3",  "addr_compact",    3),
    ("ach4",  "addr_compact",    4),
    ("anum",  "_numset",         0),
]

EQ_SPECS = ["name_compact", "name_compact_hg", "name_core_sorted",
            "name_content_sorted", "addr_compact", "addr_sorted",
            "addr_words_sorted", "addr_content"]

STR_SPECS = [
    ("nb",  "name_basic",       [("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio),
                                 ("tset", fuzz.token_set_ratio), ("part", fuzz.partial_ratio)]),
    ("nc",  "name_compact_hg",  [("ratio", fuzz.ratio),
                                 ("jw", JaroWinkler.normalized_similarity),
                                 ("lev", Levenshtein.normalized_similarity)]),
    ("ab",  "addr_canon",       [("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio),
                                 ("tset", fuzz.token_set_ratio), ("part", fuzz.partial_ratio)]),
    ("ac",  "addr_compact",     [("ratio", fuzz.ratio),
                                 ("jw", JaroWinkler.normalized_similarity)]),
]


def build_csr_all(cols):
    out = {}
    for tag, field, ng in SET_SPECS:
        src = ([s.replace(",", " ") for s in cols["addr_numset"]]
               if field == "_numset" else cols[field])
        out[tag] = build_csr(src, ngram=ng)
    return out


def hash_cols(cols):
    from er.blocking import hash_keys_fast
    return {f: hash_keys_fast(cols[f]) for f in EQ_SPECS}


def _grp_stats(gid, score, n_groups):
    """Per-group best / second / rank / count for a score, aligned to rows."""
    order = np.lexsort((-score, gid))
    g = gid[order]; s = score[order]
    new = np.empty(len(g), bool); new[0] = True
    np.not_equal(g[1:], g[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, len(g)))
    rank = np.arange(len(g)) - np.repeat(starts, sizes)
    best = np.repeat(s[starts], sizes)
    sec = np.where(sizes > 1, s[np.minimum(starts + 1, len(s) - 1)], -1.0)
    sec = np.repeat(sec, sizes)
    cnt = np.repeat(sizes, sizes)
    inv = np.empty(len(g), np.int64); inv[order] = np.arange(len(g))
    return (rank[inv].astype(np.float32), best[inv].astype(np.float32),
            sec[inv].astype(np.float32), cnt[inv].astype(np.float32))


def extract(q, t, qrow, trow, chan, cheap, qcsr, tcsr, qhash, thash,
            acos_name=None, acos_comb=None, extra=None, str_chunk=4_000_000):
    """Return (feature_names, float32 matrix)."""
    n = len(qrow)
    feats, names = [], []

    def add(name, arr):
        names.append(name); feats.append(np.asarray(arr, dtype=np.float32))

    # ---- set overlap -----------------------------------------------------
    for tag, _, _ in SET_SPECS:
        jac, dice, ca, cb, cos, inter, na, nb = set_feats(qrow, trow, qcsr[tag], tcsr[tag])
        add(f"{tag}_jac", jac); add(f"{tag}_dice", dice)
        add(f"{tag}_ca", ca);   add(f"{tag}_cb", cb)
        add(f"{tag}_cos", cos)
        add(f"{tag}_cmax", np.maximum(ca, cb))
        add(f"{tag}_na", na);   add(f"{tag}_nb", nb)

    # ---- exact view equality --------------------------------------------
    for f in EQ_SPECS:
        eq = (qhash[f][qrow] == thash[f][trow]) & (qhash[f][qrow] != 0)
        add(f"eq_{f}", eq)

    # ---- string similarity (pruned set only) ----------------------------
    for tag, field, scorers in STR_SPECS:
        qs_all, ts_all = q[field], t[field]
        for sname, scorer in scorers:
            out = np.empty(n, np.float32)
            for s in range(0, n, str_chunk):
                e = min(s + str_chunk, n)
                qa = [qs_all[i] for i in qrow[s:e]]
                ta = [ts_all[j] for j in trow[s:e]]
                out[s:e] = process.cpdist(qa, ta, scorer=scorer, workers=-1)
            if out.max(initial=0) > 1.5:
                out /= 100.0
            add(f"{tag}_{sname}", out)

    # ---- scalar / numeric ------------------------------------------------
    qlen = np.asarray([len(s) for s in q["name_compact"]], np.float32)
    tlen = np.asarray([len(s) for s in t["name_compact"]], np.float32)
    add("nlen_ratio", np.minimum(qlen[qrow], tlen[trow]) / np.maximum(np.maximum(qlen[qrow], tlen[trow]), 1))
    qal = np.asarray([len(s) for s in q["addr_compact"]], np.float32)
    tal = np.asarray([len(s) for s in t["addr_compact"]], np.float32)
    add("alen_ratio", np.minimum(qal[qrow], tal[trow]) / np.maximum(np.maximum(qal[qrow], tal[trow]), 1))
    add("q_addr_missing", np.asarray(q["addr_missing"], np.float32)[qrow])
    add("t_addr_missing", np.asarray(t["addr_missing"], np.float32)[trow])
    qnt = np.asarray(q["name_ntok"], np.float32) if "name_ntok" in q else np.zeros(len(qlen), np.float32)
    tnt = np.asarray(t["name_ntok"], np.float32) if "name_ntok" in t else np.zeros(len(tlen), np.float32)
    add("ntok_diff", np.abs(qnt[qrow] - tnt[trow]))
    qna = np.asarray([0 if s.isascii() else 1 for s in q["name_pre"]], np.float32)
    tna = np.asarray([0 if s.isascii() else 1 for s in t["name_pre"]], np.float32)
    add("t_name_nonascii", tna[trow])
    add("script_mismatch", (qna[qrow] != tna[trow]).astype(np.float32))

    qfn = np.asarray(q["addr_first_num"]); tfn = np.asarray(t["addr_first_num"])
    fn_eq = (qfn[qrow] == tfn[trow]) & (qfn[qrow] != "")
    add("first_num_eq", fn_eq)
    add("first_num_both", (qfn[qrow] != "") & (tfn[trow] != ""))

    # ---- provenance ------------------------------------------------------
    from er import candgen as CG
    bits = [("ch_name_compact", CG.CH_NAME_COMPACT), ("ch_name_hg", CG.CH_NAME_HG),
            ("ch_name_core", CG.CH_NAME_CORE), ("ch_name_content", CG.CH_NAME_CONTENT),
            ("ch_addr_compact", CG.CH_ADDR_COMPACT), ("ch_addr_sorted", CG.CH_ADDR_SORTED),
            ("ch_addr_words", CG.CH_ADDR_WORDS), ("ch_addr_content", CG.CH_ADDR_CONTENT),
            ("ch_name_num", CG.CH_NAME_NUM), ("ch_acro_num", CG.CH_ACRO_NUM),
            ("ch_addrw_num", CG.CH_ADDRW_NUM), ("ch_rare_name", CG.CH_RARE_NAME),
            ("ch_rare_addr", CG.CH_RARE_ADDR), ("ch_ann", CG.CH_ANN)]
    for bn, bv in bits:
        add(bn, (chan & bv) != 0)
    add("n_channels", np.asarray([bin(int(c)).count("1") for c in chan], np.float32))
    add("cheap_score", cheap)
    # Absolute E5 cosine is weak (everything sits near 0.91); its value is
    # relative - how this pair compares with the best neural match for the
    # same S1 and for the same target.
    an = acos_name if acos_name is not None else np.zeros(n, np.float32)
    ac = acos_comb if acos_comb is not None else np.zeros(n, np.float32)
    add("ann_cos_name", an); add("ann_cos_comb", ac)
    for nm2, col in (("annn", an), ("annc", ac)):
        _, b_, s_, c_ = _grp_stats(qrow, col, 0)
        add(f"q_{nm2}_margin", col - b_)
        _, b2, s2, c2 = _grp_stats(trow, col, 0)
        add(f"t_{nm2}_margin", col - b2)

    # ---- cross-field interactions ---------------------------------------
    F = {nm: feats[i] for i, nm in enumerate(names)}
    nsim = np.maximum(F["nch3_cos"], F["ntok_jac"])
    asim = np.maximum(F["ach3_cos"], F["atok_jac"])
    add("name_x_addr", nsim * asim)
    add("name_minus_addr", nsim - asim)
    add("high_name_low_addr", ((nsim > 0.8) & (asim < 0.4)).astype(np.float32))
    add("low_name_high_addr", ((nsim < 0.4) & (asim > 0.8)).astype(np.float32))
    add("name_x_numeq", nsim * F["first_num_eq"])
    add("addr_x_numeq", asim * F["first_num_eq"])
    add("nsim", nsim); add("asim", asim)
    combo = (0.5 * asim + 0.5 * nsim).astype(np.float32)

    # ---- group competition (S1 side and TARGET side) ---------------------
    for gname, gid in (("q", qrow), ("t", trow)):
        rank, best, sec, cnt = _grp_stats(gid, combo, 0)
        add(f"{gname}_rank", rank)
        add(f"{gname}_best", best)
        add(f"{gname}_margin", combo - best)
        add(f"{gname}_top2gap", best - sec)
        add(f"{gname}_cnt", cnt)

    if extra:
        for k, v in extra.items():
            add(k, v)
    return names, np.column_stack(feats)
