"""Collective (sibling) evidence for stage 3.

Every true match of an S1 is a noisy copy of the same underlying business, so
a candidate that closely resembles the S1's OTHER confident matches is likely
a match, while a look-alike from a different business at another address will
not resemble them. The comparison is candidate-vs-candidate (both are
generator-noised S2/S3 records), done with the fine-tuned encoder's
embeddings, so it is language-agnostic and applies to France unchanged.

Siblings of an S1 = its other candidates with stage-2 p2 >= SIB_P (top SIB_K),
a quantity available out-of-fold on train and for every test pair.

Usage: run_sibling.py <split>     (train: pruned universe of all folds)
Writes work/pred/sib_<split>.npz aligned to the pruned universe rows.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from er.io_utils import WORK
from run_prune_sim import prune_mask

SIB_P, SIB_K, MAXM = 0.9, 6, 48


def main(split):
    t0 = time.time()
    if split == "train":
        z = np.load(f"{WORK}/pred/oof.npz")
    else:
        z = np.load(f"{WORK}/pred/{split}_scores.npz")
    q, t, src, p1, p2 = z["q"], z["t"], z["src"], z["p1"], z["p2"]
    U = np.flatnonzero(prune_mask(q, src, p1, 10))
    qu, tu, su, p2u = q[U], t[U], src[U], p2[U].astype(np.float32)
    n = len(U)
    order = np.lexsort((-p2u, qu))
    g = qu[order]
    st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]); en = np.r_[st[1:], n]
    sizes = en - st
    print(f"{split}: universe={n:,} S1 groups={len(st):,} max group={sizes.max()} "
          f"(capped at {MAXM}) ({time.time()-t0:.0f}s)", flush=True)

    dev = "cuda:0"
    feats = {k: np.full(n, np.nan, np.float32) for k in
             ("sib_cn_max", "sib_cn_mean", "sib_cc_max", "sib_cc_mean")}
    nsib = np.zeros(n, np.float32)
    E = {}
    for view in ("name", "combined"):
        E[view] = {s: np.load(f"{WORK}/emb/{split}_s{s}_{view}.npy", mmap_mode="r") for s in (2, 3)}

    def gather(view, rows):
        out = np.empty((len(rows), 384), np.float16)
        for s in (2, 3):
            m = su[rows] == s
            if m.any():
                idx = tu[rows[m]]
                srt = np.argsort(idx)
                tmp = np.asarray(E[view][s][idx[srt]])
                o2 = np.empty_like(tmp); o2[srt] = tmp
                out[m] = o2
        return out

    CH = 60000
    for c0 in range(0, len(st), CH):
        gs = st[c0:c0 + CH]; ge = en[c0:c0 + CH]; G = len(gs)
        msz = np.minimum(ge - gs, MAXM)
        # member rows (in `order` space) padded to MAXM
        ar = np.arange(MAXM)[None, :]
        mem = gs[:, None] + ar
        mvalid = ar < msz[:, None]
        mem_rows = np.where(mvalid, order[np.minimum(mem, n - 1)], -1)
        # siblings: first SIB_K members (sorted by p2 desc) with p2 >= SIB_P
        sp2 = np.where(mvalid[:, :SIB_K], p2u[np.maximum(mem_rows[:, :SIB_K], 0)], -1)
        svalid = mvalid[:, :SIB_K] & (sp2 >= SIB_P)
        sib_rows = np.where(svalid, mem_rows[:, :SIB_K], -1)
        flat_m = mem_rows[mvalid]; flat_s = sib_rows[svalid]
        res = {}
        for view, tag in (("name", "cn"), ("combined", "cc")):
            Em = np.zeros((G, MAXM, 384), np.float16); Es = np.zeros((G, SIB_K, 384), np.float16)
            Em[mvalid] = gather(view, flat_m)
            if len(flat_s):
                Es[svalid] = gather(view, flat_s)
            A = torch.from_numpy(Em).to(dev); B = torch.from_numpy(Es).to(dev)
            cos = torch.bmm(A, B.transpose(1, 2)).float()          # G x MAXM x SIB_K
            # exclude padding and the member itself when it is also a sibling
            ms = torch.from_numpy(svalid).to(dev)[:, None, :].expand(-1, MAXM, -1).clone()
            same = torch.from_numpy(mem_rows[:, :, None] == sib_rows[:, None, :]).to(dev)
            ms &= ~same
            cnt = ms.sum(-1).float()
            mx = torch.where(ms, cos, torch.full_like(cos, -2.0)).max(-1).values
            mean = torch.where(ms, cos, torch.zeros_like(cos)).sum(-1) / cnt.clamp(min=1)
            res[tag] = (mx.cpu().numpy(), mean.cpu().numpy(), cnt.cpu().numpy())
        for tag in ("cn", "cc"):
            mx, mean, cnt = res[tag]
            has = mvalid & (cnt > 0)
            rows = mem_rows[has]
            feats[f"sib_{tag}_max"][rows] = mx[has]
            feats[f"sib_{tag}_mean"][rows] = mean[has]
        nsib[mem_rows[mvalid]] = res["cc"][2][mvalid]
        if c0 % (CH * 10) == 0:
            print(f"  {c0+G:,}/{len(st):,} groups ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{WORK}/pred/sib_{split}.npz", U=U, nsib=nsib, **feats)
    cov = 100 * np.isfinite(feats["sib_cc_max"]).mean()
    print(f"wrote sib_{split}.npz; pairs with >=1 sibling: {cov:.1f}% ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
