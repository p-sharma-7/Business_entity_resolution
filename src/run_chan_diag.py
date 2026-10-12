"""Per-channel recall diagnostic on a sample of S1 against the FULL target pool.

Sampling only the query side keeps the measurement honest (targets are never
reduced) while making channel configuration cheap enough to iterate on.
"""
import sys, time, json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import read_ground_truth, WORK, ROOT
from er.blocking import KeyIndex, hash_keys_fast
from er.rareblock import build_df, RareIndex
from er import candgen as CG

NEED = ["entity_id", "country", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym", "name_core",
        "name_content", "addr_compact", "addr_sorted", "addr_words_sorted",
        "addr_content", "addr_first_num", "addr_numset", "addr_missing"]
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 60000


def load(src):
    t = pq.read_table(f"{WORK}/normalized/train_s{src}.parquet", columns=NEED)
    d = {c: t.column(c).to_pylist() for c in NEED}
    d["_cty"] = np.asarray(d["country"])
    return d


S = {s: load(s) for s in (1, 2, 3)}
gt = read_ground_truth("train")
s1row = {e: i for i, e in enumerate(S[1]["entity_id"])}
rowof = {}
for s in (2, 3):
    for i, e in enumerate(S[s]["entity_id"]):
        rowof[e] = (s, i)

rng = np.random.default_rng(0)
report = {}
for cty in ("India", "US"):
    qall = np.flatnonzero(S[1]["_cty"] == cty)
    qi = np.sort(rng.choice(qall, size=min(NSAMP, len(qall)), replace=False))
    qpos = {int(r): k for k, r in enumerate(qi)}
    for ts in (2, 3):
        ti = np.flatnonzero(S[ts]["_cty"] == cty)
        tpos = np.full(len(S[ts]["entity_id"]), -1, np.int64)
        tpos[ti] = np.arange(len(ti))
        # truth restricted to sampled queries
        tru = []
        for r in qi:
            e1 = S[1]["entity_id"][r]
            for e in gt[e1]:
                st = rowof.get(e)
                if st and st[0] == ts:
                    tru.append(qpos[int(r)] * np.int64(6_000_000) + tpos[st[1]])
        tru = np.unique(np.asarray(tru, np.int64))
        q = {k: [S[1][k][i] for i in qi] for k in NEED if k != "country"}
        t = {k: [S[ts][k][i] for i in ti] for k in NEED if k != "country"}
        print(f"\n===== {cty} S{ts}  queries={len(qi):,} targets={len(ti):,} truth={len(tru):,}", flush=True)

        cum = np.empty(0, np.int64)
        rows = []
        def report_chan(label, a, b, t0):
            global cum
            p = np.unique(a.astype(np.int64) * 6_000_000 + b)
            s = np.sort(p)
            idx = np.clip(np.searchsorted(s, tru), 0, max(len(s) - 1, 0))
            own = float((s[idx] == tru).mean()) if len(s) else 0.0
            cum = np.union1d(cum, p)
            sc = cum
            idx2 = np.clip(np.searchsorted(sc, tru), 0, max(len(sc) - 1, 0))
            cr = float((sc[idx2] == tru).mean()) if len(sc) else 0.0
            rows.append((label, len(p), own, len(cum), cr))
            print(f"  {label:<18} pairs={len(p):>12,} recall={own:.4f} | cum_pairs={len(cum):>12,} cum_recall={cr:.5f}  ({time.time()-t0:.0f}s)", flush=True)

        for flag, fn in CG.KEY_SPECS:
            t0 = time.time()
            name = [k for k, v in vars(CG).items() if v == flag and k.startswith("CH_")][0]
            qk = hash_keys_fast(fn(q)); tk = hash_keys_fast(fn(t))
            a, b = KeyIndex(tk).lookup(qk, max_bucket=400)
            report_chan(name[3:].lower(), a, b, t0)

        for fld, per_rec, per_q, mdf in (("name_core", 6, 4, 4000),
                                         ("name_content", 5, 3, 4000),
                                         ("addr_content", 8, 5, 4000)):
            t0 = time.time()
            tt = [s.split() for s in t[fld]]; qq = [s.split() for s in q[fld]]
            df = build_df(tt)
            ri = RareIndex(tt, df, max_df=mdf, per_record=per_rec)
            a, b = ri.probe(qq, df, per_query=per_q, max_bucket=400, max_df=mdf)
            report_chan("rare_" + fld, a, b, t0)
        report[f"{cty}_S{ts}"] = rows
json.dump({k: [list(r) for r in v] for k, v in report.items()},
          open(f"{ROOT}/eda/chan_diag.json", "w"), indent=1)
