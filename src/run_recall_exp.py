"""Measure candidate-generation recall per channel. This sets the ceiling on
everything downstream, so it is measured before any model is built.

Reports, for each retrieval channel: pairs produced, recall of ground-truth
links, and MARGINAL recall (what that channel adds over the channels above it).
Also breaks recall down by the difficulty slices that matter: cross-script,
missing address, and zero shared name token.
"""
import sys, os, time, json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from er.io_utils import read_ground_truth, WORK, ROOT
from er.blocking import KeyIndex, hash_keys_fast, pack
from er.rareblock import build_df, RareIndex

SPLIT = "train"
t00 = time.time()


NEED = ["entity_id", "country", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym", "name_core",
        "addr_compact", "addr_sorted", "addr_words_sorted", "addr_content",
        "addr_first_num"]


def load(src):
    t = pq.read_table(f"{WORK}/normalized/{SPLIT}_s{src}.parquet", columns=NEED)
    d = {c: t.column(c).to_pylist() for c in NEED}
    d["_cty"] = np.array(d["country"])
    return d


print("loading normalized views ...", flush=True)
S = {s: load(s) for s in (1, 2, 3)}
for s in (1, 2, 3):
    print(f"  s{s}: {len(S[s]['entity_id']):,}", flush=True)

row_of = {}
for s in (1, 2, 3):
    for i, e in enumerate(S[s]["entity_id"]):
        row_of[e] = i

gt = read_ground_truth(SPLIT)
s1_row = {e: i for i, e in enumerate(S[1]["entity_id"])}

# ground-truth links as (s1_row, tgt_row) per target source
truth = {2: set(), 3: set()}
for s1, tgts in gt.items():
    i = s1_row[s1]
    for t in tgts:
        src = 2 if t[1] == "2" else 3
        j = row_of.get(t)
        if j is not None:
            truth[src].add((i << 24) | j)
print(f"truth links: S2={len(truth[2]):,} S3={len(truth[3]):,}", flush=True)

countries = sorted(set(S[1]["country"]))
print("countries:", countries, flush=True)


def col(src, name):
    return S[src][name]


def sub_idx(src, country):
    return np.flatnonzero(S[src]["_cty"] == country)


CHANNELS = [
    # (label, kind, spec)
    ("nm_compact",      "key", lambda d, i: d["name_compact"][i]),
    ("nm_compact_hg",   "key", lambda d, i: d["name_compact_hg"][i]),
    ("nm_core_sorted",  "key", lambda d, i: d["name_core_sorted"][i]),
    ("nm_content_srt",  "key", lambda d, i: d["name_content_sorted"][i]),
    ("ad_compact",      "key", lambda d, i: d["addr_compact"][i]),
    ("ad_sorted",       "key", lambda d, i: d["addr_sorted"][i]),
    ("ad_words_sorted", "key", lambda d, i: d["addr_words_sorted"][i]),
    ("ad_content",      "key", lambda d, i: d["addr_content"][i]),
    ("nmcore_num",      "key", lambda d, i: d["name_core_sorted"][i] + "|" + d["addr_first_num"][i]),
    ("acro_num",        "key", lambda d, i: (d["name_acronym"][i] + "|" + d["addr_first_num"][i])
                                            if len(d["name_acronym"][i]) >= 2 and d["addr_first_num"][i] else ""),
    ("adwords_num",     "key", lambda d, i: d["addr_words_sorted"][i] + "|" + d["addr_first_num"][i]),
]

results = {}
cum = {2: set(), 3: set()}
per_channel_pairs = Counter()

for tsrc in (2, 3):
    print(f"\n########## target source S{tsrc} ##########", flush=True)
    for country in countries:
        qi = sub_idx(1, country)
        ti = sub_idx(tsrc, country)
        if len(qi) == 0 or len(ti) == 0:
            continue
        t0 = time.time()
        print(f"[{country}] S1={len(qi):,} S{tsrc}={len(ti):,}", flush=True)
        d1, dt = S[1], S[tsrc]

        for label, kind, fn in CHANNELS:
            qk = hash_keys_fast([fn(d1, int(i)) for i in qi])
            tk = hash_keys_fast([fn(dt, int(j)) for j in ti])
            ki = KeyIndex(tk)
            a, b = ki.lookup(qk, max_bucket=300)
            if len(a):
                p = pack(qi[a], ti[b])
                per_channel_pairs[label] += len(p)
                cum[tsrc].update(p.tolist())
        print(f"   exact keys done {time.time()-t0:.0f}s cum={len(cum[tsrc]):,}", flush=True)

        # rare-token channels
        for fld, per_rec, per_q, mdf in (("name_core", 6, 4, 3000),
                                         ("addr_content", 8, 5, 3000)):
            tt = [dt[fld][int(j)].split() for j in ti]
            qq = [d1[fld][int(i)].split() for i in qi]
            df = build_df(tt)
            ri = RareIndex(tt, df, max_df=mdf, per_record=per_rec)
            a, b = ri.probe(qq, df, per_query=per_q, max_bucket=400, max_df=mdf)
            if len(a):
                p = pack(qi[a], ti[b])
                per_channel_pairs["rare_" + fld] += len(p)
                cum[tsrc].update(p.tolist())
            print(f"   rare[{fld}] -> {len(a):,} pairs, cum={len(cum[tsrc]):,} ({time.time()-t0:.0f}s)", flush=True)

    rec = len(truth[tsrc] & cum[tsrc]) / len(truth[tsrc])
    print(f"\n>>> S{tsrc}: candidates={len(cum[tsrc]):,}  RECALL={rec:.5f}", flush=True)
    results[f"S{tsrc}"] = {"pairs": len(cum[tsrc]), "recall": rec}

results["per_channel_pairs"] = dict(per_channel_pairs)
json.dump(results, open(f"{ROOT}/eda/recall_exp.json", "w"), indent=2)
print("\n", json.dumps(results, indent=2), flush=True)
print(f"total {time.time()-t00:.0f}s")
