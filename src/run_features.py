"""Extract pair features for every candidate of a split, partition by partition.

Features are computed for ALL candidates rather than a training subsample,
because the target-side competition features ('how does this pair rank among
all S1 entities claiming this target') are only meaningful when the target's
competitor set is complete. Truncating S1 would bias exactly the signal the
decision layer leans on hardest.
"""
import sys, os, time, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import read_ground_truth, WORK, ROOT
from er import features as FE

NEED = ["entity_id", "country", "name_pre", "name_basic", "name_compact",
        "name_compact_hg", "name_core_sorted", "name_content_sorted",
        "name_acronym", "name_core", "name_content", "name_ntok",
        "addr_pre", "addr_canon", "addr_compact", "addr_sorted",
        "addr_words_sorted", "addr_content", "addr_first_num", "addr_numset",
        "addr_missing", "addr_ntok"]


def e5_pair_feats(split, ts, gq, gt, qrow, trow, chunk=2_000_000):
    """Exact fine-tuned-encoder cosine for EVERY candidate pair, both views.

    The retrieval cosine only exists for pairs the ANN happened to return, so
    for every other pair it reads 0 - "missing", not "dissimilar". Error
    analysis showed that gap sitting under both main error families:
    cross-script true matches ('laxmi healthcare' vs its Devanagari rendering)
    scored low, and cross-script distractors ('hyderabad technologies' vs
    'HiTech Consultants' in Telugu) scored high, because only the encoder can
    read the Indic name and its opinion was absent.
    """
    out = {}
    for view in ("name", "combined"):
        E1 = np.load(f"{WORK}/emb/{split}_s1_{view}.npy", mmap_mode="r")
        ET = np.load(f"{WORK}/emb/{split}_s{ts}_{view}.npy", mmap_mode="r")
        cos = np.empty(len(gq), np.float32)
        for s in range(0, len(gq), chunk):
            e = min(s + chunk, len(gq))
            a = np.asarray(E1[gq[s:e]], np.float32)
            b = np.asarray(ET[gt[s:e]], np.float32)
            cos[s:e] = np.einsum("ij,ij->i", a, b)
        out[view] = cos
    names, cols = [], []
    for view, cos in out.items():
        v = "n" if view == "name" else "c"
        rq, bq, sq, _ = FE._grp_stats(qrow, cos, 0)
        rt, bt, st, _ = FE._grp_stats(trow, cos, 0)
        names += [f"e5{v}_cos", f"e5{v}_qrank", f"e5{v}_qmargin", f"e5{v}_qgap",
                  f"e5{v}_trank", f"e5{v}_tmargin", f"e5{v}_tgap"]
        cols += [cos, rq, cos - bq, bq - sq, rt, cos - bt, bt - st]
    names += ["e5_name_x_comb", "e5_name_minus_comb"]
    cols += [out["name"] * out["combined"], out["name"] - out["combined"]]
    return names, np.column_stack(cols).astype(np.float32)


def load(split, src):
    t = pq.read_table(f"{WORK}/normalized/{split}_s{src}.parquet", columns=NEED)
    return {c: t.column(c).to_pylist() for c in NEED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--chunk", type=int, default=6_000_000)
    a = ap.parse_args()
    split = a.split
    os.makedirs(f"{WORK}/feat", exist_ok=True)
    t00 = time.time()

    S = {s: load(split, s) for s in (1, 2, 3)}
    cty1 = np.asarray(S[1]["country"])

    gtmap = None
    if split == "train":
        gt = read_ground_truth("train")
        idx1 = {e: i for i, e in enumerate(S[1]["entity_id"])}
        gtmap = {}
        for s in (2, 3):
            ridx = {e: i for i, e in enumerate(S[s]["entity_id"])}
            gtmap[s] = set()
            for s1, tg in gt.items():
                i = idx1.get(s1)
                if i is None: continue
                for e in tg:
                    j = ridx.get(e)
                    if j is not None:
                        gtmap[s].add((i << 23) | j)

    names_written = False
    for ts in (2, 3):
        for cty in sorted(set(cty1.tolist())):
            src = f"{WORK}/cand/{split}_S{ts}_{cty}.npz"
            if not os.path.exists(src):
                continue
            out = f"{WORK}/feat/{split}_S{ts}_{cty}.npz"
            if os.path.exists(out):
                print(f"skip {out}", flush=True); continue
            t0 = time.time()
            z = np.load(src)
            gq = z["q"].astype(np.int64); gt_ = z["t"].astype(np.int64)
            chan = z["chan"]; cheap = z["score"]
            acos_n = z["acos_name"] if "acos_name" in z else np.zeros(len(chan), np.float32)
            acos_c = z["acos_comb"] if "acos_comb" in z else np.zeros(len(chan), np.float32)
            qi = np.unique(gq); ti = np.unique(gt_)
            qmap = np.full(len(S[1]["entity_id"]), -1, np.int64); qmap[qi] = np.arange(len(qi))
            tmap = np.full(len(S[ts]["entity_id"]), -1, np.int64); tmap[ti] = np.arange(len(ti))
            qrow = qmap[gq]; trow = tmap[gt_]
            q = {k: [S[1][k][i] for i in qi] for k in NEED}
            t = {k: [S[ts][k][i] for i in ti] for k in NEED}
            print(f"[{split} S{ts} {cty}] pairs={len(gq):,} q={len(qi):,} t={len(ti):,}", flush=True)

            qcsr = FE.build_csr_all(q); tcsr = FE.build_csr_all(t)
            qh = FE.hash_cols(q); th = FE.hash_cols(t)
            print(f"   csr+hash {time.time()-t0:.0f}s", flush=True)

            mats, names = [], None
            for s in range(0, len(gq), a.chunk):
                e = min(s + a.chunk, len(gq))
                nm, M = FE.extract(q, t, qrow[s:e], trow[s:e], chan[s:e], cheap[s:e],
                                   qcsr, tcsr, qh, th,
                                   acos_name=acos_n[s:e], acos_comb=acos_c[s:e])
                # float16 halves both the feature store and peak training RAM;
                # with max_bin=127 the tree splits cannot resolve finer anyway.
                names = nm; mats.append(M.astype(np.float16))
                print(f"   feat {e:,}/{len(gq):,} ({time.time()-t0:.0f}s)", flush=True)
            X = np.concatenate(mats, 0) if len(mats) > 1 else mats[0]
            del mats
            en, EX = e5_pair_feats(split, ts, gq, gt_, qrow, trow)
            X = np.hstack([X, EX.astype(np.float16)]); names = names + en
            del EX
            print(f"   e5 pair features added ({time.time()-t0:.0f}s)", flush=True)

            y = None
            if split == "train":
                key = (gq << np.int64(23)) | gt_
                gs = gtmap[ts]
                y = np.fromiter((1 if int(k) in gs else 0 for k in key), np.int8, len(key))
                print(f"   positives={int(y.sum()):,} ({100*y.mean():.3f}%)", flush=True)

            np.savez(out, X=X, q=z["q"], t=z["t"], chan=chan, cheap=cheap,
                     acos_name=acos_n, acos_comb=acos_c,
                     **({"y": y} if y is not None else {}))
            if not names_written:
                open(f"{ROOT}/models/feature_names.txt", "w").write("\n".join(names))
                names_written = True
            print(f"   wrote {out} X={X.shape} {os.path.getsize(out)/1e9:.1f}GB ({time.time()-t0:.0f}s)", flush=True)
            del X
    print(f"TOTAL {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
