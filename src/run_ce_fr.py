"""Adapt the cross-encoder to French text with pseudo-labels.

Diagnosis (test, uncertain-band pairs): the reranker trained on US/India text
is itself unsure on 31.6% of French pairs (3.6-4.2% elsewhere) and calls 8.6%
of pairs "match" that stage 2 rates p2 < 0.1 (0.2-0.3% elsewhere). Stage 3
trusts it - on held-out US/India it is 97.5% accurate - so on France it pushes
clear non-matches into matches. France scores ~0.94 on the leaderboard while
India/US hold ~0.99.

Fix: continue training the reranker on French pairs whose label is nearly
certain from stage 2 (which stays reliable on France: its fold models agree
and its uncertainty is close to training). Positives: calibrated p2 >= 0.97
and the winning claim on the target. Negatives: France candidates the pruner
ranked highly but stage 2 puts at <= 0.02 (hard negatives). Replay of the
original US/India reranker data prevents forgetting.

Modes:  prep | train (torchrun) | score
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from er.io_utils import WORK, ROOT
from run_ce import build_model, texts, MAXLEN, GPUS

COUNTRY = "France"
INIT = f"{ROOT}/models/ce"
# Round 2 retrains from the ORIGINAL US/India reranker (INIT) with labels from
# the V4 stage 2, so the round-1 bias ("type-word swap = match", learned from
# pseudo-labels made while type words were wrongly treated as filler) is not
# inherited. ER_CE_FR_TAG selects the output model and pair file.
_TAG = os.environ.get("ER_CE_FR_TAG", "")
OUT = f"{ROOT}/models/ce_fr{_TAG}"
PAIRS = f"{WORK}/pred/ce_fr_train{_TAG}.parquet"


def prep():
    import joblib
    from er import decide as D
    from run_prune_sim import prune_mask
    rng = np.random.default_rng(0)
    z = np.load(f"{WORK}/pred/test_scores.npz")
    q, t, src, p1, p2 = z["q"], z["t"], z["src"], z["p1"], z["p2"]
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet",
                                  columns=["country"]).column("country").to_pylist())
    pc = joblib.load(f"{ROOT}/models/calibrator.pkl").predict(p2).astype(np.float32)
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    win = D.unique_target_filter(tk, pc)
    fr = c1[q] == COUNTRY
    pruned = prune_mask(q, src, p1, 10)
    pos = np.flatnonzero(fr & win & (pc >= 0.97))
    neg_all = np.flatnonzero(fr & pruned & (pc <= 0.02))
    hard = neg_all[np.argsort(-p1[neg_all])][: 2 * len(pos)]      # hardest first
    T = texts("test")
    ii = np.r_[pos, hard]
    A = [T[1][q[i]] for i in ii]
    B = [T[int(src[i])][t[i]] for i in ii]
    Y = np.r_[np.ones(len(pos)), np.zeros(len(hard))].astype(np.float32)
    print(f"French pseudo-labels: pos={len(pos):,} hard neg={len(hard):,} "
          f"(from {len(neg_all):,} confident rejects)", flush=True)
    r = np.load(f"{WORK}/pred/ce_train_pairs.npz")
    sel = rng.choice(len(r["q"]), size=min(1_000_000, len(r["q"])), replace=False)
    Tt = texts("train")
    A += [Tt[1][i] for i in r["q"][sel]]
    B += [Tt[int(s)][j] for s, j in zip(r["src"][sel], r["t"][sel])]
    Y = np.r_[Y, r["y"][sel].astype(np.float32)]
    perm = rng.permutation(len(Y))
    pq.write_table(pa.table({"a": [A[i] for i in perm], "b": [B[i] for i in perm],
                             "y": Y[perm]}), PAIRS)
    print(f"wrote {PAIRS}: {len(Y):,} pairs (France {len(ii):,}, replay {len(sel):,})", flush=True)


def train(a):
    import torch, torch.distributed as dist
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = f"cuda:{local}"
    tok = AutoTokenizer.from_pretrained(INIT)
    tb = pq.read_table(PAIRS)
    A = tb.column("a").to_pylist(); B = tb.column("b").to_pylist(); Y = np.asarray(tb.column("y"))
    idx = np.arange(rank, len(Y), world)
    model = build_model(INIT, f"{INIT}/head.pt").to(dev)
    model.enc.gradient_checkpointing_enable()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(idx) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda"); model.train(); t0 = time.time()
    for st in range(steps):
        b = idx[st * a.bs:(st + 1) * a.bs]
        enc = tok([A[i] for i in b], [B[i] for i in b], padding=True,
                  truncation="longest_first", max_length=MAXLEN, return_tensors="pt").to(dev)
        lab = torch.from_numpy(Y[b]).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logit = model(enc["input_ids"], enc["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logit.float(), lab)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if rank == 0 and st % 500 == 0:
            print(f"  step {st}/{steps} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        model.module.enc.save_pretrained(OUT); tok.save_pretrained(OUT)
        torch.save(model.module.head.state_dict(), f"{OUT}/head.pt")
        print(f"saved {OUT} ({time.time()-t0:.0f}s)", flush=True)
    dist.barrier(); dist.destroy_process_group()


def _score_worker(rank, world, bs=512):
    import torch
    from transformers import AutoTokenizer
    torch.cuda.set_device(0)
    tok = AutoTokenizer.from_pretrained(OUT)
    model = build_model(OUT, f"{OUT}/head.pt").half().cuda().eval()
    T = texts("test")
    z = np.load(f"{WORK}/pred/ce_fr_score.npz")
    q, t, src = z["q"], z["t"], z["src"]
    n = len(q); lo, hi = n * rank // world, n * (rank + 1) // world
    out = np.lib.format.open_memmap(f"{WORK}/pred/ce_fr_scores.npy", mode="r+")
    with torch.inference_mode():
        for s in range(lo, hi, bs):
            e = min(s + bs, hi)
            enc = tok([T[1][i] for i in q[s:e]], [T[int(x)][j] for x, j in zip(src[s:e], t[s:e])],
                      padding=True, truncation="longest_first", max_length=MAXLEN,
                      return_tensors="pt").to("cuda")
            out[s:e] = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]).float()).cpu().numpy()
    out.flush()


def score():
    """Re-score the French rows of the test uncertain band and splice them in."""
    import subprocess
    meta = np.load(f"{WORK}/pred/ce_score_test.npz")
    c1 = np.asarray(pq.read_table(f"{WORK}/normalized/test_s1.parquet",
                                  columns=["country"]).column("country").to_pylist())
    sel = np.flatnonzero(c1[meta["q"]] == COUNTRY)
    np.savez(f"{WORK}/pred/ce_fr_score.npz", q=meta["q"][sel], t=meta["t"][sel], src=meta["src"][sel])
    np.lib.format.open_memmap(f"{WORK}/pred/ce_fr_scores.npy", mode="w+", dtype=np.float32,
                              shape=(len(sel),)).flush()
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "_worker", str(r), str(len(GPUS))],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
             for r, g in enumerate(GPUS)]
    for p in procs:
        p.wait()
        if p.returncode:
            raise SystemExit("scoring failed")
    base = f"{WORK}/pred/ce_scores_test.npy"
    orig = f"{WORK}/pred/ce_scores_test_usindia_ce.npy"
    if not os.path.exists(orig):
        np.save(orig, np.load(base))
    ce = np.load(orig).copy()
    ce[sel] = np.load(f"{WORK}/pred/ce_fr_scores.npy")
    np.save(base, ce)
    print(f"spliced {len(sel):,} France scores into {base}", flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "prep":
        prep()
    elif m == "train":
        ap = argparse.ArgumentParser(); ap.add_argument("mode")
        ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-5)
        train(ap.parse_args())
    elif m == "score":
        score()
    elif m == "_worker":
        _score_worker(int(sys.argv[2]), int(sys.argv[3]))
