"""Target-side index for one (country, source) partition.

All expensive structures are built once per partition and then probed with
query blocks, so their cost amortises over the whole S1 side instead of being
repeated. Document frequencies for the character-n-gram channels are estimated
from a sample of targets - picking which grams are rare does not need exact
counts, and sampling turns the dominant cost of the whole stage into a minor
one - while the index itself still covers every target.
"""
from __future__ import annotations
import numpy as np
import multiprocessing as mp
from collections import Counter
from er.blocking import KeyIndex, hash_keys_fast
from er.rareblock import build_df, RareIndex
from er import candgen as CG

_SHARED = {}


def _grams(s, k):
    return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else ([s] if s else [])


def _df_chunk(args):
    lo, hi, field, k = args
    c = Counter()
    data = _SHARED[field]
    for i in range(lo, hi):
        s = data[i]
        c.update(set(_grams(s, k) if k else s.split()))
    return c


def parallel_df(data, field, k, sample=600_000, nproc=30, seed=0):
    """Estimate document frequency, optionally from a sample of records."""
    n = len(data)
    idx = np.arange(n)
    if sample and n > sample:
        idx = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        sub = [data[i] for i in idx]
        scale = n / sample
    else:
        sub, scale = data, 1.0
    _SHARED[field] = sub
    m = len(sub)
    CH = max(20000, m // (nproc * 4))
    spans = [(i, min(i + CH, m), field, k) for i in range(0, m, CH)]
    total = Counter()
    with mp.Pool(nproc) as pool:
        for c in pool.imap_unordered(_df_chunk, spans, chunksize=1):
            total.update(c)
    _SHARED.pop(field, None)
    if scale != 1.0:
        for key in total:
            total[key] = int(total[key] * scale)
    return total


class TargetIndex:
    """Exact-key, rare-token, rare-char-ngram and number indexes for a partition."""

    # name char-4-grams produced ~292 candidates per query for only +0.5%
    # recall, and the E5 channel now covers fuzzy name matching far better,
    # so it is capped hard rather than dropped.
    NGRAM_CH = [("addr_compact", 5, 10, 8, 3000, CG.CH_RARE_ADDR),
                ("name_compact_hg", 4, 6, 4, 1500, CG.CH_RARE_NAME)]
    TOK_CH = [("name_core", 6, 4, 4000, CG.CH_RARE_NAME),
              ("addr_content", 8, 5, 4000, CG.CH_RARE_ADDR)]

    def __init__(self, t, max_bucket=400, verbose=True, channels="all"):
        """channels: 'all', or 'lite' to drop the two most expensive lexical
        channels (character-n-gram and number x rare-token indexes). They cost
        the bulk of the stage's runtime and exist to reach the slice a strong
        neural channel now covers directly."""
        self.t = t
        self.max_bucket = max_bucket
        self.lite = (channels == "lite")
        self.keys = []
        for flag, fn in CG.KEY_SPECS:
            self.keys.append((flag, fn, KeyIndex(hash_keys_fast(fn(t)))))

        self.tok = []
        for fld, pr, pq_, mdf, flag in self.TOK_CH:
            data = [s.split() for s in t[fld]]
            df = parallel_df(t[fld], f"tok_{fld}", 0)
            self.tok.append((fld, 0, pq_, mdf, flag, df, RareIndex(data, df, max_df=mdf, per_record=pr)))

        self.ng = []
        for fld, k, pr, pq_, mdf, flag in ([] if self.lite else self.NGRAM_CH):
            data = [_grams(s, k) for s in t[fld]]
            df = parallel_df(t[fld], f"ng_{fld}", k)
            self.ng.append((fld, k, pq_, mdf, flag, df, RareIndex(data, df, max_df=mdf, per_record=pr)))
            del data

        nums = [s.replace(",", " ").split() for s in t["addr_numset"]]
        dfn = build_df(nums)
        self.num = (dfn, RareIndex(nums, dfn, max_df=20000, per_record=8))

        self.dft = self.tok[1][5]          # addr_content DF, reused below
        self.numtok = None
        if not self.lite:
            tt = [self._numtok(nums[i], t["addr_content"][i].split(), 4, 3)
                  for i in range(len(nums))]
            dfnt = build_df(tt)
            self.numtok = (dfnt, RareIndex(tt, dfnt, max_df=4000, per_record=12))
            del tt
        del nums

    def _numtok(self, nums, toks, nn, nt):
        rare = [w for _, w in sorted((self.dft.get(w, 1), w) for w in set(toks))[:nt]]
        return [f"{n}|{w}" for n in nums[:nn] for w in rare]

    def retrieve(self, q):
        """q: dict of view lists for one query block. Returns local (qrow,trow,chan)."""
        qs, ts, cs = [], [], []

        def push(a, b, flag):
            if len(a):
                qs.append(a); ts.append(b); cs.append(np.full(len(a), flag, np.int32))

        for flag, fn, ki in self.keys:
            a, b = ki.lookup(hash_keys_fast(fn(q)), max_bucket=self.max_bucket)
            push(a, b, flag)
        for fld, _, pq_, mdf, flag, df, ri in self.tok:
            a, b = ri.probe([s.split() for s in q[fld]], df, per_query=pq_,
                            max_bucket=self.max_bucket, max_df=mdf)
            push(a, b, flag)
        for fld, k, pq_, mdf, flag, df, ri in self.ng:
            mb = self.max_bucket if fld == "addr_compact" else 120
            a, b = ri.probe([_grams(s, k) for s in q[fld]], df, per_query=pq_,
                            max_bucket=mb, max_df=mdf)
            push(a, b, flag)

        qnums = [s.replace(",", " ").split() for s in q["addr_numset"]]
        dfn, rin = self.num
        a, b = rin.probe(qnums, dfn, per_query=8, max_bucket=600, max_df=20000)
        push(a, b, CG.CH_RARE_ADDRNUM)

        if self.numtok is not None:
            dfnt, rint = self.numtok
            qq = [self._numtok(qnums[i], q["addr_content"][i].split(), 5, 4)
                  for i in range(len(qnums))]
            a, b = rint.probe(qq, dfnt, per_query=12, max_bucket=250, max_df=4000)
            push(a, b, CG.CH_RARE_ADDRNUM)

        if not qs:
            z = np.empty(0, np.int64)
            return z, z, np.empty(0, np.int32)
        return np.concatenate(qs), np.concatenate(ts), np.concatenate(cs)


def dedupe(qrow, trow, chan, aux=None, shift=23):
    key = (qrow.astype(np.int64) << np.int64(shift)) | trow.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]; chan = chan[order]
    uniq, start = np.unique(key, return_index=True)
    merged = np.bitwise_or.reduceat(chan, start)
    out = ((uniq >> np.int64(shift)).astype(np.int64),
           (uniq & np.int64((1 << shift) - 1)).astype(np.int64), merged)
    if aux is not None:
        for col in (aux if isinstance(aux, (list, tuple)) else [aux]):
            out = out + (np.maximum.reduceat(col[order], start),)
    return out
