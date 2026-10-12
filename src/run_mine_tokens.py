"""Measure, per token, how often it survives a true match (alias-aware).

A token that is freely added or dropped across a genuine pair - legal suffixes
(ltd/llc), generic business words (center/services), address filler
(unit/door/no) - must not appear in an exact blocking key and must be weak
evidence in features.

Crucially this runs AFTER alias canonicalisation. Measured on raw tokens,
'rd' scores 0.887 "unstable" purely because its partner wrote 'Road'; once
rd->road is applied, the remaining drop rate is the real instability.
"""
import sys, json, time, os
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
from numba import njit
from er.io_utils import read_source, read_ground_truth, ROOT
from er.normalize import basic, nfkc_fold, _NULL_TOKENS, _ORD_SUFFIX

ALIAS = {}


def _tok_chunk(args):
    lo, hi, field = args
    idx = 0 if field == "name" else 1
    out = []
    for i in range(lo, hi):
        s = REC[i][idx]
        ts = []
        for t in basic(nfkc_fold(s)).split():
            if t in _NULL_TOKENS:
                continue
            if field == "addr":
                m = _ORD_SUFFIX.match(t)
                if m:
                    t = m.group(1)
                if t.isdigit():
                    t = t.lstrip("0") or "0"
                t = ALIAS.get(t, t)
            else:
                t = ALIAS.get(t, t)
            ts.append(t)
        out.append(ts)
    return lo, out


@njit(cache=True)
def _walk(pi, pj, vals, offs, seen, diff):
    for p in range(pi.shape[0]):
        i = pi[p]; j = pj[p]
        a0 = offs[i]; a1 = offs[i + 1]
        b0 = offs[j]; b1 = offs[j + 1]
        if a1 == a0 or b1 == b0:
            continue
        x = a0; y = b0
        while x < a1 or y < b1:
            if x < a1 and y < b1:
                u = vals[x]; v = vals[y]
                if u == v:
                    seen[u] += 1; x += 1; y += 1
                elif u < v:
                    seen[u] += 1; diff[u] += 1; x += 1
                else:
                    seen[v] += 1; diff[v] += 1; y += 1
            elif x < a1:
                u = vals[x]; seen[u] += 1; diff[u] += 1; x += 1
            else:
                v = vals[y]; seen[v] += 1; diff[v] += 1; y += 1


if __name__ == "__main__":
    t0 = time.time()
    ids, names, addrs = [], [], []
    for s in (1, 2, 3):
        d = read_source("train", s)
        ids += d["id"]; names += d["name"]; addrs += d["addr"]
    REC = list(zip(names, addrs))
    del names, addrs
    row = {e: i for i, e in enumerate(ids)}
    gt = read_ground_truth("train")
    pi, pj = [], []
    for s1, tg in gt.items():
        i = row.get(s1)
        if i is None: continue
        for t in tg:
            j = row.get(t)
            if j is not None:
                pi.append(i); pj.append(j)
    pi = np.asarray(pi, np.int64); pj = np.asarray(pj, np.int64)
    print(f"loaded {len(REC):,} records, {len(pi):,} pairs in {time.time()-t0:.0f}s", flush=True)

    for field in ("addr", "name"):
        ALIAS = {}
        ap = f"{ROOT}/models/alias_{field}_ALL.json"
        if os.path.exists(ap):
            ALIAS = json.load(open(ap, encoding="utf-8"))["canon"]
        print(f"[{field}] alias canon entries={len(ALIAS):,}", flush=True)

        t = time.time()
        n = len(REC)
        CH = 40000
        spans = [(i, min(i + CH, n), field) for i in range(0, n, CH)]
        parts = [None] * len(spans)
        with mp.Pool(34) as pool:
            for lo, out in pool.imap_unordered(_tok_chunk, spans, chunksize=2):
                parts[lo // CH] = out
        toklists = [x for p in parts for x in p]
        del parts
        print(f"  tokenized in {time.time()-t:.0f}s", flush=True)

        t = time.time()
        vocab = {}
        offs = np.zeros(n + 1, np.int64)
        flat = []
        for i, ts in enumerate(toklists):
            u = set()
            for x in ts:
                v = vocab.get(x)
                if v is None:
                    v = len(vocab); vocab[x] = v
                u.add(v)
            a = np.fromiter(sorted(u), np.int32, len(u))
            flat.append(a)
            offs[i + 1] = offs[i] + a.size
        vals = np.concatenate(flat); del flat, toklists
        print(f"  vocab={len(vocab):,} ids in {time.time()-t:.0f}s", flush=True)

        seen = np.zeros(len(vocab), np.int64); diff = np.zeros(len(vocab), np.int64)
        t = time.time()
        _walk(pi, pj, vals, offs, seen, diff)
        print(f"  walked pairs in {time.time()-t:.0f}s", flush=True)

        inv = [None] * len(vocab)
        for k, v in vocab.items(): inv[v] = k
        keep = np.flatnonzero(seen >= 2000)
        rows = sorted(((inv[i], int(seen[i]), float(diff[i] / seen[i])) for i in keep),
                      key=lambda r: -r[1])
        unstable = {tk: round(r, 4) for tk, nn, r in rows if r >= 0.35}
        json.dump({"npairs": int(len(pi)), "unstable": unstable,
                   "top_frequent": [[a, b, round(c, 4)] for a, b, c in rows[:150]]},
                  open(f"{ROOT}/models/tokstats_{field}.json", "w"),
                  indent=1, ensure_ascii=False)
        print(f"=== {field}: tokens>=2000={len(rows):,} unstable={len(unstable):,}", flush=True)
        for a, b, c in rows[:40]:
            print(f"    {a:<24} {b:>9,}  {c:.3f}", flush=True)
