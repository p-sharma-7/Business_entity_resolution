"""Fine-tune multilingual-E5-small on challenge pairs only.

Motivation, from measurement: the combined-view ANN already reaches 0.931
recall because a target's address stays largely Latin even when its name is
transliterated. The NAME view manages only 0.556, and the candidates still
missed after every lexical channel are 83% Indic-script names. Teaching the
encoder that 'baba foods private limited' and its Devanagari rendering are the
same entity attacks exactly that residual.

Training data is drawn solely from train_ground_truth.tsv. Cross-script
positives are oversampled to roughly half the batch, since they are the slice
that matters and only ~13% of links naturally.

Loss is in-batch softmax (MultipleNegativesRanking): every other positive in
the batch acts as a negative, which is well matched to a retrieval objective.
"""
import sys, os, time, random, argparse, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from er.io_utils import read_ground_truth, fold_of, WORK, ROOT

MODEL = "intfloat/multilingual-e5-small"


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2_400_000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cross_frac", type=float, default=0.5)
    ap.add_argument("--mix_combined", type=int, default=1)
    ap.add_argument("--holdout_fold", type=int, default=-1,
                    help="exclude this S1 fold (of 5) from training, for an "
                         "honest held-out retrieval measurement")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=f"{ROOT}/models/e5_ft")
    a = ap.parse_args()
    t0 = time.time()
    dev = f"cuda:{a.gpu}"

    names, addrs = {}, {}
    for s in (1, 2, 3):
        tb = pq.read_table(f"{WORK}/normalized/train_s{s}.parquet",
                           columns=["entity_id", "name_pre", "addr_pre"])
        for e, n, ad in zip(tb.column("entity_id").to_pylist(),
                            tb.column("name_pre").to_pylist(),
                            tb.column("addr_pre").to_pylist()):
            names[e] = n; addrs[e] = ad
    gt = read_ground_truth("train")

    def comb(e):
        return f"[NAME] {names[e]} [ADDRESS] {addrs[e]}"

    # Both views are trained in one encoder: the name view carries the
    # cross-script signal, the combined view carries address evidence, and a
    # single model keeps inference and storage cost unchanged.
    cross, same = [], []
    n_held = 0
    for s1, tg in gt.items():
        a1 = names.get(s1)
        if not a1:
            continue
        if a.holdout_fold >= 0 and fold_of(s1, a.folds) == a.holdout_fold:
            n_held += 1
            continue
        for e in tg:
            b1 = names.get(e)
            if b1 is None or not b1.strip():
                continue
            bucket = cross if not b1.isascii() else same
            bucket.append((a1, b1))
            if a.mix_combined:
                bucket.append((comb(s1), comb(e)))
    print(f"pairs: cross_script={len(cross):,} same_script={len(same):,} "
          f"held_out_entities={n_held:,} ({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(0)
    n_cross = min(len(cross), int(a.pairs * a.cross_frac))
    n_same = min(len(same), a.pairs - n_cross)
    data = rng.sample(cross, n_cross) + rng.sample(same, n_same)
    rng.shuffle(data)
    print(f"training on {len(data):,} pairs ({100*n_cross/len(data):.0f}% cross-script)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(data) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    step = 0
    for ep in range(a.epochs):
        for i in range(0, len(data) - a.bs + 1, a.bs):
            batch = data[i:i + a.bs]
            qs = [f"query: {x[0]}" for x in batch]
            ps = [f"passage: {x[1]}" for x in batch]
            bq = tok(qs, padding=True, truncation=True, max_length=a.maxlen,
                     return_tensors="pt").to(dev)
            bp = tok(ps, padding=True, truncation=True, max_length=a.maxlen,
                     return_tensors="pt").to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                eq = F.normalize(mean_pool(model(**bq).last_hidden_state, bq["attention_mask"]), dim=-1)
                epp = F.normalize(mean_pool(model(**bp).last_hidden_state, bp["attention_mask"]), dim=-1)
                logits = eq @ epp.T * a.scale
                labels = torch.arange(len(batch), device=dev)
                loss = 0.5 * (F.cross_entropy(logits, labels) +
                              F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(1) == labels).float().mean().item()
                print(f"  step {step}/{steps} loss={loss.item():.4f} inbatch_acc={acc:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.half().save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
