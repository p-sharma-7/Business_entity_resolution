"""Cross-encoder reranker for the pairs the GBDT is unsure about.

Error analysis of the stage-2 GBDT showed its remaining mistakes need the two
records read JOINTLY: a pseudo-word brand name ('nylaflux') on the S1's exact
address is a match; a one-token change ('rz trading' vs 'rzj trading') at the
same address is a distractor; a Devanagari rendering of the S1 name with a
truncated address is a match. Independent per-field similarity features blur
these; a cross-encoder attends across both records at once.

Scope: only pairs whose stage-2 probability is in (LO, HI). That band holds
~4.5% of candidates but nearly every error, so the reranker's cost is small.

Honesty: the reranker trains on S1 folds 0-3 only (the same entity folds the
encoder and GBDT use). Fold 4 and the test set are therefore both unseen,
which is what lets the stage-3 model be validated on fold 4.

Modes:
  prep   build the training / scoring pair lists
  train  torchrun --nproc_per_node=5 run_ce.py train
  score  run_ce.py score <split>
"""
import sys, os, time, math, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from er.io_utils import WORK, ROOT

LO, HI = 0.005, 0.995
# ER_CE_BASE / ER_CE_OUT / ER_CE_TAG train and score an alternative reranker
# without touching the production one (models/ce, ce_scores_<split>.npy)
BASE = os.environ.get("ER_CE_BASE", f"{ROOT}/models/e5_ho")
OUT = os.environ.get("ER_CE_OUT", f"{ROOT}/models/ce")
TAG = os.environ.get("ER_CE_TAG", "")
GPUS = [0, 1, 2]          # capped at 3 GPUs; the rest stay free for other users
MAXLEN = 128


def texts(split):
    """Per-source 'name | address' strings, indexed by row."""
    d = {}
    for s in (1, 2, 3):
        tb = pq.read_table(f"{WORK}/normalized/{split}_s{s}.parquet",
                           columns=["name_pre", "addr_pre"])
        n = tb.column("name_pre").to_pylist(); a = tb.column("addr_pre").to_pylist()
        d[s] = [f"{x} | {y}" if y else f"{x} | -" for x, y in zip(n, a)]
    return d


# ------------------------------------------------------------------- prep
def prep():
    z = np.load(f"{WORK}/pred/oof.npz")
    p, y, q, t, src, fold = z["p2"], z["y"], z["q"], z["t"], z["src"], z["fold"]
    band = (p > LO) & (p < HI)
    tr = band & (fold != 4)
    rng = np.random.default_rng(0)
    # a small slice of confident pairs keeps the reranker calibrated at the edges
    easy = (~band) & (fold != 4) & (rng.random(len(p)) < 0.004)
    sel = np.flatnonzero(tr | easy)
    rng.shuffle(sel)
    np.savez(f"{WORK}/pred/ce_train_pairs.npz", q=q[sel], t=t[sel], src=src[sel],
             y=y[sel].astype(np.float32))
    print(f"train pairs={len(sel):,} pos={int(y[sel].sum()):,} "
          f"(band={int(tr.sum()):,} easy={int(easy.sum()):,})", flush=True)
    sc = np.flatnonzero(band & (fold == 4))
    np.savez(f"{WORK}/pred/ce_score_train.npz", idx=sc, q=q[sc], t=t[sc], src=src[sc])
    print(f"fold-4 scoring pairs={len(sc):,}", flush=True)


# ------------------------------------------------------------------ model
def build_model(path, head_path=None):
    import torch, torch.nn as nn
    from transformers import AutoModel

    class CE(nn.Module):
        def __init__(self):
            super().__init__()
            # the encoder checkpoint was saved in fp16; load fp32 master weights
            # (autocast handles fp16 compute) so the fp32 head and optimizer agree
            # no pooler: it is unused (we mean-pool), and DDP rejects parameters
            # that receive no gradient
            self.enc = AutoModel.from_pretrained(path, dtype=torch.float32,
                                                 add_pooling_layer=False)
            self.head = nn.Linear(self.enc.config.hidden_size, 1)

        def forward(self, ids, mask):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return self.head(pooled).squeeze(-1)

    m = CE()
    if head_path:
        m.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    return m


# ------------------------------------------------------------------ train
def train(a):
    import torch, torch.distributed as dist
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    tok = AutoTokenizer.from_pretrained(BASE)
    T = texts("train")
    z = np.load(f"{WORK}/pred/ce_train_pairs.npz")
    q, t, src, y = z["q"], z["t"], z["src"], z["y"]
    n = min(len(q), a.max_pairs)
    idx = np.arange(rank, n, world)          # disjoint shard per rank
    model = build_model(BASE).to(dev)
    if os.environ.get("ER_CE_FREEZE_EMB") == "1":    # large multilingual vocab: keep it fixed
        for prm in model.enc.embeddings.parameters():
            prm.requires_grad_(False)
    if os.environ.get("ER_CE_GC", "1") == "1":
        model.enc.gradient_checkpointing_enable()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    steps = len(idx) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda")
    model.train(); t0 = time.time()
    for st in range(steps):
        b = idx[st * a.bs:(st + 1) * a.bs]
        A = [T[1][i] for i in q[b]]
        B = [T[int(s)][j] for s, j in zip(src[b], t[b])]
        enc = tok(A, B, padding=True, truncation="longest_first", max_length=MAXLEN,
                  return_tensors="pt").to(dev)
        lab = torch.from_numpy(y[b]).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logit = model(enc["input_ids"], enc["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logit.float(), lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if rank == 0 and st % 500 == 0:
            with torch.no_grad():
                acc = ((logit > 0).float() == lab).float().mean().item()
            print(f"  step {st}/{steps} loss={loss.item():.4f} acc={acc:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        model.module.enc.save_pretrained(OUT); tok.save_pretrained(OUT)
        torch.save(model.module.head.state_dict(), f"{OUT}/head.pt")
        print(f"saved {OUT} ({time.time()-t0:.0f}s)", flush=True)
    dist.barrier(); dist.destroy_process_group()


# ------------------------------------------------------------------ score
def _score_worker(rank, world, split, pairs_path, out_path, bs):
    import torch
    from transformers import AutoTokenizer
    torch.cuda.set_device(0)                      # CUDA_VISIBLE_DEVICES pins the GPU
    tok = AutoTokenizer.from_pretrained(OUT)
    model = build_model(OUT, f"{OUT}/head.pt").half().cuda().eval()
    T = texts(split)
    z = np.load(pairs_path)
    q, t, src = z["q"], z["t"], z["src"]
    n = len(q); lo = n * rank // world; hi = n * (rank + 1) // world
    out = np.lib.format.open_memmap(out_path, mode="r+")
    t0 = time.time()
    with torch.inference_mode():
        for s in range(lo, hi, bs):
            e = min(s + bs, hi)
            A = [T[1][i] for i in q[s:e]]
            B = [T[int(x)][j] for x, j in zip(src[s:e], t[s:e])]
            enc = tok(A, B, padding=True, truncation="longest_first", max_length=MAXLEN,
                      return_tensors="pt").to("cuda")
            out[s:e] = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]).float()).cpu().numpy()
            if rank == 0 and (s - lo) % (bs * 200) == 0:
                print(f"  gpu-rank0 {s-lo:,}/{hi-lo:,} ({time.time()-t0:.0f}s)", flush=True)
    out.flush()


def pairs_file(split):
    """A tagged pair list (e.g. India/US band only) overrides the production one."""
    p = f"{WORK}/pred/ce_score_{split}{TAG}.npz"
    return p if TAG and os.path.exists(p) else f"{WORK}/pred/ce_score_{split}.npz"


def score(split, bs=512):
    import subprocess
    pairs_path = pairs_file(split)
    n = len(np.load(pairs_path)["q"])
    out_path = f"{WORK}/pred/ce_scores_{split}{TAG}.npy"
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n,)).flush()
    procs = []
    for r, g in enumerate(GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, __file__, "_worker", split,
                                       str(r), str(len(GPUS)), str(bs)], env=env))
    for p in procs:
        p.wait()
        if p.returncode:
            raise SystemExit("scoring worker failed")
    print(f"scored {n:,} pairs -> {out_path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        prep()
    elif mode == "train":
        ap = argparse.ArgumentParser()
        ap.add_argument("mode"); ap.add_argument("--bs", type=int, default=64)
        ap.add_argument("--lr", type=float, default=3e-5)
        ap.add_argument("--max_pairs", type=int, default=6_000_000)
        train(ap.parse_args())
    elif mode == "score":
        score(sys.argv[2])
    elif mode == "_worker":
        split, r, w, bs = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        _score_worker(r, w, split, pairs_file(split),
                      f"{WORK}/pred/ce_scores_{split}{TAG}.npy", bs)
