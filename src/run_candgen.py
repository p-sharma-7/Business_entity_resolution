"""Run candidate generation for a split and report recall (train) or write
candidates (test). Processes one (country, target-source) partition at a time.
"""
import sys, os, time, json, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from er.io_utils import read_ground_truth, WORK, ROOT
from er import candgen as CG

NEED = ["entity_id", "country", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym", "name_core",
        "name_content", "addr_compact", "addr_sorted", "addr_words_sorted",
        "addr_content", "addr_first_num", "addr_numset", "addr_missing"]


def load(split, src):
    t = pq.read_table(f"{WORK}/normalized/{split}_s{src}.parquet", columns=NEED)
    d = {c: t.column(c).to_pylist() for c in NEED}
    d["_cty"] = np.asarray(d["country"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--kq", type=int, default=30)
    ap.add_argument("--kt", type=int, default=6)
    ap.add_argument("--maxbucket", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    split = a.split
    t00 = time.time()

    S = {s: load(split, s) for s in (1, 2, 3)}
    for s in (1, 2, 3):
        print(f"  s{s}: {len(S[s]['entity_id']):,}", flush=True)

    truth = {}
    if split == "train":
        gt = read_ground_truth("train")
        s1row = {e: i for i, e in enumerate(S[1]["entity_id"])}
        rowof = {}
        for s in (2, 3):
            for i, e in enumerate(S[s]["entity_id"]):
                rowof[e] = (s, i)
        for ts in (2, 3):
            truth[ts] = []
        for s1, tg in gt.items():
            i = s1row[s1]
            for e in tg:
                st = rowof.get(e)
                if st:
                    truth[st[0]].append((i << 23) | st[1] if False else (i, st[1]))
        for ts in (2, 3):
            arr = np.asarray(truth[ts], dtype=np.int64)
            truth[ts] = np.sort(arr[:, 0] * np.int64(6_000_000) + arr[:, 1])
            print(f"  truth S{ts}: {len(truth[ts]):,}", flush=True)

    countries = sorted(set(S[1]["_cty"].tolist()))
    print("countries:", countries, flush=True)

    stats = {}
    os.makedirs(f"{WORK}/cand", exist_ok=True)
    for ts in (2, 3):
        tot_pairs = 0; hit = 0; raw_tot = 0; raw_hit = 0
        for cty in countries:
            qi = np.flatnonzero(S[1]["_cty"] == cty)
            ti = np.flatnonzero(S[ts]["_cty"] == cty)
            if len(qi) == 0 or len(ti) == 0:
                continue
            t0 = time.time()
            q = {k: [S[1][k][i] for i in qi] for k in NEED if k != "country"}
            t = {k: [S[ts][k][i] for i in ti] for k in NEED if k != "country"}

            qrow, trow, chan = CG.retrieve(q, t, max_bucket=a.maxbucket)
            print(f"[{cty} S{ts}] raw={len(qrow):,} ({time.time()-t0:.0f}s)", flush=True)

            qcsr = CG.build_side_csr(q); tcsr = CG.build_side_csr(t)
            score, _ = CG.cheap_scores(qrow, trow, qcsr, tcsr)
            keep = CG.prune(qrow, trow, chan, score, k_q=a.kq, k_t=a.kt)

            gq, gt_, gc, gs = qi[qrow], ti[trow], chan, score
            if split == "train":
                glob_raw = gq.astype(np.int64) * 6_000_000 + gt_
                raw_tot += len(glob_raw)
                raw_hit += int(np.isin(truth[ts], glob_raw, assume_unique=False).sum()) if False else 0
                gk = glob_raw[keep]
                idx = np.searchsorted(np.sort(gk), truth[ts])
                sk = np.sort(gk)
                idx = np.clip(idx, 0, len(sk) - 1)
                hit += int((sk[idx] == truth[ts]).sum())
                # raw recall too
                sr = np.sort(glob_raw)
                ir = np.clip(np.searchsorted(sr, truth[ts]), 0, len(sr) - 1)
                raw_hit += int((sr[ir] == truth[ts]).sum())
            tot_pairs += int(keep.sum())
            print(f"   kept={int(keep.sum()):,} ({100*keep.mean():.1f}%) "
                  f"per_s1={keep.sum()/max(len(qi),1):.1f} ({time.time()-t0:.0f}s)", flush=True)

            if a.out:
                np.savez(f"{WORK}/cand/{split}_S{ts}_{cty}.npz",
                         q=gq[keep].astype(np.int32), t=gt_[keep].astype(np.int32),
                         chan=gc[keep].astype(np.int32), score=gs[keep])
        if split == "train":
            r = hit / len(truth[ts]); rr = raw_hit / len(truth[ts])
            print(f"\n>>> S{ts}: kept={tot_pairs:,}  RAW_RECALL={rr:.5f}  PRUNED_RECALL={r:.5f}\n", flush=True)
            stats[f"S{ts}"] = {"pairs": tot_pairs, "raw_recall": rr, "recall": r}
        else:
            print(f"\n>>> S{ts}: kept={tot_pairs:,}\n", flush=True)
            stats[f"S{ts}"] = {"pairs": tot_pairs}
    json.dump(stats, open(f"{ROOT}/eda/candgen_{split}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
