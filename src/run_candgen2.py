"""Production candidate generation.

Per (country, target-source) partition:
  build target indexes once -> stream S1 in blocks -> union lexical + neural
  channels -> cheap numba scoring -> keep top-K per S1 and top-K per target
  (block-local) -> one global target-side pass -> write.

Both pruning directions are kept because the ground truth is one-to-many from
S1 but one-to-one from the target: a target its true parent ranks 40th may
still rank that parent first, and that is the pair worth keeping.
"""
import sys, os, time, json, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import read_ground_truth, WORK, ROOT
from er import candgen as CG
from er.partindex import TargetIndex, dedupe

NEED = ["entity_id", "country", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym", "name_core",
        "name_content", "addr_compact", "addr_sorted", "addr_words_sorted",
        "addr_content", "addr_first_num", "addr_numset", "addr_missing"]
MUL = np.int64(6_000_000)


def load(split, src):
    t = pq.read_table(f"{WORK}/normalized/{split}_s{src}.parquet", columns=NEED)
    d = {c: t.column(c).to_pylist() for c in NEED}
    d["_cty"] = np.asarray(d["country"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--kq", type=int, default=30)
    ap.add_argument("--kt", type=int, default=3)
    ap.add_argument("--kq_final", type=int, default=28)
    ap.add_argument("--kt_final", type=int, default=4)
    ap.add_argument("--ann_keep", type=int, default=16)
    ap.add_argument("--ann_keep_name", type=int, default=18)
    ap.add_argument("--block", type=int, default=120000)
    ap.add_argument("--ann_k", type=int, default=30)
    ap.add_argument("--no_ann", action="store_true")
    ap.add_argument("--only", default=None, help="limit to one partition, e.g. S2_India")
    ap.add_argument("--channels", default="all", choices=["all", "lite"])
    a = ap.parse_args()
    split = a.split
    t00 = time.time()

    S = {s: load(split, s) for s in (1, 2, 3)}
    truth = {}
    if split == "train":
        gt = read_ground_truth("train")
        s1row = {e: i for i, e in enumerate(S[1]["entity_id"])}
        rowof = {}
        for s in (2, 3):
            for i, e in enumerate(S[s]["entity_id"]):
                rowof[e] = (s, i)
        for ts in (2, 3):
            truth[ts] = {}
        for s1, tg in gt.items():
            i = s1row[s1]
            for e in tg:
                st = rowof.get(e)
                if st:
                    truth[st[0]].setdefault(i, []).append(st[1])

    os.makedirs(f"{WORK}/cand", exist_ok=True)
    stats = {}
    for ts in (2, 3):
        for cty in sorted(set(S[1]["_cty"].tolist())):
            if a.only and a.only != f"S{ts}_{cty}":
                continue
            out = f"{WORK}/cand/{split}_S{ts}_{cty}.npz"
            if os.path.exists(out):
                print(f"skip {out}", flush=True); continue
            qi = np.flatnonzero(S[1]["_cty"] == cty)
            ti = np.flatnonzero(S[ts]["_cty"] == cty)
            if len(qi) == 0 or len(ti) == 0:
                continue
            t0 = time.time()
            q = {k: [S[1][k][i] for i in qi] for k in NEED if k != "country"}
            t = {k: [S[ts][k][i] for i in ti] for k in NEED if k != "country"}
            tloc = np.full(len(S[ts]["entity_id"]), -1, np.int64); tloc[ti] = np.arange(len(ti))
            qloc = np.full(len(S[1]["entity_id"]), -1, np.int64); qloc[qi] = np.arange(len(qi))

            TI = TargetIndex(t, channels=a.channels)
            print(f"[{split} S{ts} {cty}] q={len(qi):,} t={len(ti):,} index built {time.time()-t0:.0f}s", flush=True)
            pcsr = CG.build_pair_csr(q, t)
            print(f"   csr built {time.time()-t0:.0f}s", flush=True)

            ann = {}
            if not a.no_ann:
                for view in ("name", "combined"):
                    p = f"{WORK}/ann/{split}_S{ts}_{cty}_{view}.npz"
                    if os.path.exists(p):
                        z = np.load(p)
                        ann[view] = (qloc[z["q"].astype(np.int64)],
                                     tloc[z["t"].astype(np.int64)],
                                     z["d"].astype(np.float32))

            keepQ, keepT, keepC, keepS, keepA, keepB = [], [], [], [], [], []
            for bs in range(0, len(qi), a.block):
                be = min(bs + a.block, len(qi))
                qb = {k: v[bs:be] for k, v in q.items()}
                qr, tr, ch = TI.retrieve(qb)
                qr = qr + bs
                parts = [(qr, tr, ch)]
                # One cosine column per ANN view. They are NOT folded into the
                # lexical score: E5 cosines sit in a narrow 0.90-0.92 band for
                # almost every pair, so blending them in replaces a
                # discriminative lexical ranking with a near-constant floor.
                # The views earn their keep through their own top-k retention
                # and as features, where the model can use them relatively.
                views = ["name", "combined"]
                acols = {v: [np.zeros(len(qr), np.float32)] for v in views}
                for view in views:
                    if view not in ann:
                        continue
                    aq, at, ad = ann[view]
                    m = (aq >= bs) & (aq < be)
                    if not m.any():
                        continue
                    aqq = np.repeat(aq[m], at.shape[1])
                    att = at[m].reshape(-1)
                    add_ = ad[m].reshape(-1).astype(np.float32)
                    ok = att >= 0
                    parts.append((aqq[ok], att[ok],
                                  np.full(int(ok.sum()), CG.CH_ANN, np.int32)))
                    for v2 in views:
                        acols[v2].append(add_[ok] if v2 == view
                                         else np.zeros(int(ok.sum()), np.float32))
                qr = np.concatenate([p[0] for p in parts])
                tr = np.concatenate([p[1] for p in parts])
                ch = np.concatenate([p[2] for p in parts])
                aname = np.concatenate(acols["name"])
                acomb = np.concatenate(acols["combined"])
                qr, tr, ch, aname, acomb = dedupe(qr, tr, ch, [aname, acomb])
                score, _ = CG.cheap_scores(qr, tr, pcsr)
                keep = (ch & CG.EXACT_MASK) != 0
                keep |= CG.topk_mask(qr, score, a.kq)
                keep |= CG.topk_mask(tr, score, a.kt)
                for acol, kk in ((acomb, a.ann_keep), (aname, a.ann_keep_name)):
                    sel = acol > 0
                    if sel.any():
                        keep |= sel & CG.topk_mask(np.where(sel, qr, -1), acol, kk)
                keepQ.append(qr[keep].astype(np.int32)); keepT.append(tr[keep].astype(np.int32))
                keepC.append(ch[keep]); keepS.append(score[keep])
                keepA.append(aname[keep]); keepB.append(acomb[keep])
                if bs % (a.block * 4) == 0:
                    print(f"   block {bs:,}/{len(qi):,} raw={len(qr):,} kept={int(keep.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

            Q = np.concatenate(keepQ); T = np.concatenate(keepT)
            C = np.concatenate(keepC); SC = np.concatenate(keepS)
            AN = np.concatenate(keepA); AC = np.concatenate(keepB)
            del keepQ, keepT, keepC, keepS, keepA, keepB
            final = (C & CG.EXACT_MASK) != 0
            Q64 = Q.astype(np.int64)
            final |= CG.topk_mask(Q64, SC, a.kq_final)
            final |= CG.topk_mask(T.astype(np.int64), SC, a.kt_final)
            # The neural channel must survive this pass too. Pruning globally
            # on the lexical score alone silently undid the per-block ANN
            # retention, discarding exactly the cross-script recall the
            # channel exists to provide.
            for acol, kk in ((AC, a.ann_keep), (AN, a.ann_keep_name)):
                sel = acol > 0
                if sel.any():
                    final |= sel & CG.topk_mask(np.where(sel, Q64, -1), acol, kk)
            Q, T, C, SC, AN, AC = (Q[final], T[final], C[final], SC[final],
                                   AN[final], AC[final])

            np.savez(out, q=qi[Q].astype(np.int32), t=ti[T].astype(np.int32),
                     chan=C, score=SC, acos_name=AN, acos_comb=AC)
            msg = f">>> [{split} S{ts} {cty}] kept={len(Q):,} per_s1={len(Q)/len(qi):.1f} ({time.time()-t0:.0f}s)"
            if split == "train":
                gk = np.sort(Q.astype(np.int64) * MUL + T)
                tot = hit = 0
                for i_glob, tl in truth[ts].items():
                    il = qloc[i_glob]
                    if il < 0: continue
                    for tg_ in tl:
                        tl2 = tloc[tg_]
                        if tl2 < 0: continue
                        tot += 1
                        k = il * MUL + tl2
                        j = np.searchsorted(gk, k)
                        if j < len(gk) and gk[j] == k: hit += 1
                msg += f"  RECALL={hit/max(tot,1):.5f} ({hit:,}/{tot:,})"
                stats[f"S{ts}_{cty}"] = {"pairs": int(len(Q)), "recall": hit / max(tot, 1),
                                         "truth": tot}
            print(msg, flush=True)
    json.dump(stats, open(f"{ROOT}/eda/candgen2_{split}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"TOTAL {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
