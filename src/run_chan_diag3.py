"""Diagnostic 3: number-anchored channels on top of the existing stack.

The residual after char-n-gram blocking is dominated by Indic-script targets
whose address has been truncated to a few components. Those still carry the
distinctive house/survey number ('a-101', '17/1168', 'sy no. 172'), while the
surviving word tokens are low-information (jaipur, rajasthan). Numbers are
therefore the right anchor for this slice.
"""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from er.io_utils import read_ground_truth, WORK
from er.blocking import KeyIndex, hash_keys_fast
from er.rareblock import build_df, RareIndex
from er import candgen as CG

NEED = ["entity_id", "country", "name_pre", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym", "name_core",
        "name_content", "addr_pre", "addr_compact", "addr_sorted",
        "addr_words_sorted", "addr_content", "addr_first_num", "addr_numset",
        "addr_missing"]
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 40000


def load(src):
    t = pq.read_table(f"{WORK}/normalized/train_s{src}.parquet", columns=NEED)
    d = {c: t.column(c).to_pylist() for c in NEED}
    d["_cty"] = np.asarray(d["country"]); return d


def ngrams(s, k):
    return [s[i:i+k] for i in range(len(s)-k+1)] if len(s) >= k else ([s] if s else [])


S = {s: load(s) for s in (1, 2, 3)}
gt = read_ground_truth("train")
rowof = {}
for s in (2, 3):
    for i, e in enumerate(S[s]["entity_id"]): rowof[e] = (s, i)

rng = np.random.default_rng(0)
for cty in ("India", "US"):
    qall = np.flatnonzero(S[1]["_cty"] == cty)
    qi = np.sort(rng.choice(qall, size=min(NSAMP, len(qall)), replace=False))
    qpos = {int(r): k for k, r in enumerate(qi)}
    for ts in (2, 3):
        ti = np.flatnonzero(S[ts]["_cty"] == cty)
        tpos = np.full(len(S[ts]["entity_id"]), -1, np.int64); tpos[ti] = np.arange(len(ti))
        tru = []
        for r in qi:
            for e in gt[S[1]["entity_id"][r]]:
                st = rowof.get(e)
                if st and st[0] == ts: tru.append(qpos[int(r)]*np.int64(6_000_000)+tpos[st[1]])
        tru = np.unique(np.asarray(tru, np.int64))
        q = {k: [S[1][k][i] for i in qi] for k in NEED if k != "country"}
        t = {k: [S[ts][k][i] for i in ti] for k in NEED if k != "country"}
        print(f"\n===== {cty} S{ts} q={len(qi):,} t={len(ti):,} truth={len(tru):,}", flush=True)
        cum = np.empty(0, np.int64)

        def rep(label, a, b, t0):
            global cum
            p = np.unique(a.astype(np.int64)*6_000_000 + b)
            cum = np.union1d(cum, p)
            i2 = np.clip(np.searchsorted(cum, tru), 0, max(len(cum)-1, 0))
            print(f"  {label:<20} +{len(p):>11,} | cum={len(cum):>12,} cum_recall={float((cum[i2]==tru).mean()):.5f} ({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time(); qs, bs = [], []
        for flag, fn in CG.KEY_SPECS:
            a, b = KeyIndex(hash_keys_fast(fn(t))).lookup(hash_keys_fast(fn(q)), max_bucket=400)
            if len(a): qs.append(a); bs.append(b)
        rep("exact_keys", np.concatenate(qs), np.concatenate(bs), t0)
        for fld, pr, pq_, mdf, lab in (("name_core", 6, 4, 4000, "rare_name_tok"),
                                       ("addr_content", 8, 5, 4000, "rare_addr_tok")):
            t0 = time.time()
            tt = [x.split() for x in t[fld]]; qq = [x.split() for x in q[fld]]
            df = build_df(tt); ri = RareIndex(tt, df, max_df=mdf, per_record=pr)
            rep(lab, *ri.probe(qq, df, per_query=pq_, max_bucket=400, max_df=mdf), t0)
        for fld, k, pr, pq_, mdf, lab in (("addr_compact", 5, 10, 8, 3000, "rare_addr_ch5"),
                                          ("name_compact_hg", 4, 8, 6, 3000, "rare_name_ch4")):
            t0 = time.time()
            tt = [ngrams(x, k) for x in t[fld]]; qq = [ngrams(x, k) for x in q[fld]]
            df = build_df(tt); ri = RareIndex(tt, df, max_df=mdf, per_record=pr)
            rep(lab, *ri.probe(qq, df, per_query=pq_, max_bucket=400, max_df=mdf), t0)

        # ---- NEW: pure address-number index ------------------------------
        t0 = time.time()
        tnum = [x.replace(",", " ").split() for x in t["addr_numset"]]
        qnum = [x.replace(",", " ").split() for x in q["addr_numset"]]
        df = build_df(tnum); ri = RareIndex(tnum, df, max_df=20000, per_record=8)
        rep("rare_addr_num", *ri.probe(qnum, df, per_query=8, max_bucket=600, max_df=20000), t0)

        # ---- NEW: (number x rare address token) composite ----------------
        t0 = time.time()
        dft = build_df([x.split() for x in t["addr_content"]])
        def numtok(nums, toks, nn, nt):
            rare = [w for _, w in sorted((dft.get(w, 1), w) for w in set(toks))[:nt]]
            return [f"{n}|{w}" for n in nums[:nn] for w in rare]
        tt = [numtok(tnum[i], t["addr_content"][i].split(), 4, 3) for i in range(len(tnum))]
        qq = [numtok(qnum[i], q["addr_content"][i].split(), 5, 4) for i in range(len(qnum))]
        df2 = build_df(tt); ri = RareIndex(tt, df2, max_df=4000, per_record=12)
        rep("num_x_raretok", *ri.probe(qq, df2, per_query=20, max_bucket=400, max_df=4000), t0)

        i2 = np.clip(np.searchsorted(cum, tru), 0, max(len(cum)-1, 0))
        miss = tru[cum[i2] != tru]
        print(f"  MISSED {len(miss):,} of {len(tru):,} = {100*len(miss)/len(tru):.3f}%", flush=True)
        mq = (miss//6_000_000).astype(int); mt = (miss % 6_000_000).astype(int)
        d = min(3000, len(mq))
        if d:
            na = sum(1 for j in mt[:d] if t["addr_missing"][j])
            nn = sum(1 for j in mt[:d] if not t["name_pre"][j].isascii())
            print(f"   of first {d}: addr_missing={100*na/d:.1f}%  nonascii_name={100*nn/d:.1f}%", flush=True)
            for a_, b_ in list(zip(mq, mt))[:5]:
                print(f"    S1 {q['name_pre'][a_]!r} | {q['addr_pre'][a_]!r}")
                print(f"    T  {t['name_pre'][b_]!r} | {t['addr_pre'][b_]!r}")
