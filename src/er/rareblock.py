"""Rare-token inverted-index blocking.

Full-key equality is brittle here: one typo'd city token ('Pougkheepsie') or
one dropped house number breaks a whole-address key. But a true pair almost
always still shares at least one DISTINCTIVE token ('sheafe'). Indexing targets
on their low-document-frequency tokens and probing with the query's rarest
tokens recovers those pairs at a fraction of TF-IDF's cost.
"""
from __future__ import annotations
import numpy as np
from collections import Counter


def build_df(token_lists) -> Counter:
    df = Counter()
    for toks in token_lists:
        for t in set(toks):
            df[t] += 1
    return df


class RareIndex:
    """Inverted index over tokens whose document frequency is below max_df."""

    def __init__(self, token_lists, df: Counter, max_df: int = 2000,
                 per_record: int = 6):
        rows, keys = [], []
        for i, toks in enumerate(token_lists):
            if not toks:
                continue
            uniq = set(toks)
            cand = sorted(((df.get(t, 1), t) for t in uniq))[:per_record]
            for d, t in cand:
                if d <= max_df:
                    rows.append(i)
                    keys.append(hash(t))
        self.rows = np.asarray(rows, dtype=np.int64)
        kh = np.asarray(keys, dtype=np.int64).view(np.uint64)
        order = np.argsort(kh, kind="stable")
        self.rows = self.rows[order]
        self.keys = kh[order]

    def probe(self, token_lists, df: Counter, per_query: int = 4,
              max_bucket: int = 400, max_df: int = 2000):
        """Return (query_row, target_row) for shared rare tokens."""
        qrows, qkeys = [], []
        for i, toks in enumerate(token_lists):
            if not toks:
                continue
            uniq = set(toks)
            cand = sorted(((df.get(t, 1), t) for t in uniq))[:per_query]
            for d, t in cand:
                if d <= max_df:
                    qrows.append(i)
                    qkeys.append(hash(t))
        if not qrows:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        qrows = np.asarray(qrows, dtype=np.int64)
        qk = np.asarray(qkeys, dtype=np.int64).view(np.uint64)
        lo = np.searchsorted(self.keys, qk, "left")
        hi = np.searchsorted(self.keys, qk, "right")
        cnt = (hi - lo).astype(np.int64)
        cnt[cnt > max_bucket] = 0
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        out_q = np.repeat(qrows, cnt)
        starts = np.repeat(lo, cnt)
        base = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        offs = np.arange(total, dtype=np.int64) - np.repeat(base, cnt)
        out_t = self.rows[starts + offs]
        return out_q, out_t
