"""Candidate generation: retrieve broadly, then prune bidirectionally.

Strategy
--------
1. RETRIEVE generously from complementary channels (exact multi-view keys +
   rare-token inverted indexes). Recall at this stage is what caps the whole
   system, so the channels are deliberately loose.
2. SCORE every raw candidate with cheap numba set features (token and
   character n-gram overlap on name and address). This costs ~1e8 pairs/s, so
   even a billion raw pairs is minutes, not hours.
3. PRUNE bidirectionally. A pair survives if it is among the best K for its S1
   *or* among the best K for its target, or if an exact key produced it.

The bidirectional rule matters because the ground truth is one-to-many from S1
but strictly one-to-one from the target side: every matched S2/S3 record has
exactly one S1 parent. A target that its true parent ranks 40th may still rank
that parent 1st, and keeping it is what preserves recall for S1 entities with
many similar-looking neighbours (39% of S1 records share a normalized name).
"""
from __future__ import annotations
import numpy as np
from er.blocking import KeyIndex, hash_keys_fast
from er.rareblock import build_df, RareIndex
from er.setfeat import build_csr, set_feats

# channel bit flags, kept on each surviving pair as provenance features
CH_NAME_COMPACT   = 1 << 0
CH_NAME_HG        = 1 << 1
CH_NAME_CORE      = 1 << 2
CH_NAME_CONTENT   = 1 << 3
CH_ADDR_COMPACT   = 1 << 4
CH_ADDR_SORTED    = 1 << 5
CH_ADDR_WORDS     = 1 << 6
CH_ADDR_CONTENT   = 1 << 7
CH_NAME_NUM       = 1 << 8
CH_ACRO_NUM       = 1 << 9
CH_ADDRW_NUM      = 1 << 10
CH_RARE_NAME      = 1 << 11
CH_RARE_ADDR      = 1 << 12
CH_RARE_ADDRNUM   = 1 << 13

CH_ANN = 1 << 14

# Keys kept unconditionally must be genuinely high precision. name_core and
# addr_words are NOT: 39% of S1 records share a normalized name (one bucket
# holds 253), so retaining every hit of theirs triples candidate volume while
# adding pairs the scorer would have kept anyway if they were any good.
EXACT_MASK = (CH_NAME_COMPACT | CH_NAME_HG | CH_ADDR_COMPACT |
              CH_ADDR_SORTED | CH_NAME_NUM | CH_ADDRW_NUM)

KEY_SPECS = [
    (CH_NAME_COMPACT, lambda d: d["name_compact"]),
    (CH_NAME_HG,      lambda d: d["name_compact_hg"]),
    (CH_NAME_CORE,    lambda d: d["name_core_sorted"]),
    (CH_NAME_CONTENT, lambda d: d["name_content_sorted"]),
    (CH_ADDR_COMPACT, lambda d: d["addr_compact"]),
    (CH_ADDR_SORTED,  lambda d: d["addr_sorted"]),
    (CH_ADDR_WORDS,   lambda d: d["addr_words_sorted"]),
    (CH_ADDR_CONTENT, lambda d: d["addr_content"]),
    (CH_NAME_NUM,     lambda d: [a + "|" + b for a, b in zip(d["name_core_sorted"], d["addr_first_num"])]),
    (CH_ACRO_NUM,     lambda d: [(a + "|" + b) if len(a) >= 2 and b else ""
                                 for a, b in zip(d["name_acronym"], d["addr_first_num"])]),
    (CH_ADDRW_NUM,    lambda d: [a + "|" + b for a, b in zip(d["addr_words_sorted"], d["addr_first_num"])]),
]


def _take(cols, idx):
    return {k: [v[i] for i in idx] for k, v in cols.items()}


def retrieve(q, t, max_bucket=400, rare_cfg=((("name_core", 6, 4, 4000), CH_RARE_NAME),
                                             (("addr_content", 8, 5, 4000), CH_RARE_ADDR))):
    """q, t: dicts of per-record view lists (already restricted to one country).

    Returns (qrow, trow, chan) int arrays over LOCAL row indices.
    """
    qs, ts, cs = [], [], []
    for flag, fn in KEY_SPECS:
        qk = hash_keys_fast(fn(q))
        tk = hash_keys_fast(fn(t))
        a, b = KeyIndex(tk).lookup(qk, max_bucket=max_bucket)
        if len(a):
            qs.append(a); ts.append(b)
            cs.append(np.full(len(a), flag, dtype=np.int32))

    for (fld, per_rec, per_q, mdf), flag in rare_cfg:
        tt = [s.split() for s in t[fld]]
        qq = [s.split() for s in q[fld]]
        df = build_df(tt)
        ri = RareIndex(tt, df, max_df=mdf, per_record=per_rec)
        a, b = ri.probe(qq, df, per_query=per_q, max_bucket=max_bucket, max_df=mdf)
        if len(a):
            qs.append(a); ts.append(b)
            cs.append(np.full(len(a), flag, dtype=np.int32))

    if not qs:
        z = np.empty(0, np.int64)
        return z, z, np.empty(0, np.int32)
    qrow = np.concatenate(qs); trow = np.concatenate(ts); chan = np.concatenate(cs)

    # merge duplicates, OR-ing the provenance flags
    key = (qrow.astype(np.int64) << np.int64(23)) | trow.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]; chan = chan[order]
    uniq, start = np.unique(key, return_index=True)
    merged = np.bitwise_or.reduceat(chan, start) if len(start) else chan
    return (uniq >> np.int64(23)).astype(np.int64), (uniq & np.int64((1 << 23) - 1)).astype(np.int64), merged


def cheap_scores(qrow, trow, csr):
    """IDF-weighted ranking score used only for pruning.

    Weighting is what keeps the top-k boundary meaningful: sharing
    'pratap nagar sector 26' must outrank sharing 'jaipur rajasthan', and
    unweighted containment cannot tell them apart. The full feature set is
    computed later, over the far smaller surviving list."""
    from er.setfeat import wlight_feats
    ncos, ncmax = wlight_feats(qrow, trow, *csr["nch3"][:2], csr["nch3"][2])
    acos_, acmax = wlight_feats(qrow, trow, *csr["ach3"][:2], csr["ach3"][2])
    tcos, tcmax = wlight_feats(qrow, trow, *csr["atok"][:2], csr["atok"][2])
    name = np.maximum(ncos, 0.6 * ncmax)
    addr = np.maximum(0.6 * acos_ + 0.4 * acmax, 0.5 * tcos + 0.5 * tcmax)
    score = (0.45 * addr + 0.35 * name + 0.20 * np.maximum(name, addr)).astype(np.float32)
    return score, None


def build_pair_csr(qcols, tcols):
    """Shared-vocabulary CSR + target-side IDF for the three ranking sets."""
    from er.setfeat import build_csr_dense
    out = {}
    for tag, field, ng in (("nch3", "name_compact_hg", 3),
                           ("atok", "addr_content", 0),
                           ("ach3", "addr_compact", 3)):
        out[tag] = build_csr_dense(qcols[field], tcols[field], ngram=ng)
    return out


def topk_mask(group, score, k):
    """Boolean mask keeping the k highest-scoring rows within each group id."""
    order = np.lexsort((-score, group))
    g = group[order]
    newg = np.empty(len(g), bool); newg[0] = True
    np.not_equal(g[1:], g[:-1], out=newg[1:])
    starts = np.flatnonzero(newg)
    rank = np.arange(len(g)) - np.repeat(starts, np.diff(np.append(starts, len(g))))
    keep = np.zeros(len(g), bool)
    keep[order[rank < k]] = True
    return keep


def prune(qrow, trow, chan, score, k_q=30, k_t=6):
    """Keep pairs that are top-k for their S1, top-k for their target, or exact."""
    keep = (chan & EXACT_MASK) != 0
    keep |= topk_mask(qrow, score, k_q)
    keep |= topk_mask(trow, score, k_t)
    return keep
