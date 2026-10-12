"""Encode records with multilingual-E5-small across all available GPUs.

Two views per record:
  name view      "query: <name>"          / "passage: <name>"
  combined view  "query: [NAME] .. [ADDRESS] .."
S1 records use the `query:` prefix, S2/S3 records use `passage:`, matching how
E5 was trained.

Vectors are written as float16 .npy (384 dims) so 10M records cost 7.7 GB and
can be memory-mapped by the ANN stage.
"""
import sys, os, time, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import torch
import torch.multiprocessing as tmp
import pyarrow.parquet as pq
from er.io_utils import WORK

MODEL = "intfloat/multilingual-e5-small"
DIM = 384


def build_texts(tbl, view, prefix):
    names = tbl.column("name_pre").to_pylist()
    if view == "name":
        return [f"{prefix}: {n}" for n in names]
    addrs = tbl.column("addr_pre").to_pylist()
    return [f"{prefix}: [NAME] {n} [ADDRESS] {a}" for n, a in zip(names, addrs)]


def worker(rank, world, texts_path, out_path, n, maxlen, bs, model_path):
    from transformers import AutoTokenizer, AutoModel
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path, dtype=torch.float16).to(dev).eval()

    texts = np.load(texts_path, allow_pickle=True)
    lo = (n * rank) // world
    hi = (n * (rank + 1)) // world
    out = np.lib.format.open_memmap(out_path, mode="r+", dtype=np.float16,
                                    shape=(n, DIM))
    t0 = time.time()
    with torch.inference_mode():
        for s in range(lo, hi, bs):
            e = min(s + bs, hi)
            batch = tok(list(texts[s:e]), padding=True, truncation=True,
                        max_length=maxlen, return_tensors="pt").to(dev)
            h = mdl(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)     # mean pooling
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out[s:e] = emb.cpu().numpy().astype(np.float16)
            if rank == 0 and (s - lo) % (bs * 200) == 0:
                done = s - lo + 1
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  gpu0 {done:,}/{hi-lo:,} {rate:,.0f}/s "
                      f"eta {(hi-lo-done)/max(rate,1):,.0f}s", flush=True)
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("src", type=int)
    ap.add_argument("view", choices=["name", "combined"])
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tbl = pq.read_table(f"{WORK}/normalized/{a.split}_s{a.src}.parquet",
                        columns=["name_pre", "addr_pre"])
    prefix = "query" if a.src == 1 else "passage"
    texts = build_texts(tbl, a.view, prefix)
    n = len(texts)
    os.makedirs(f"{WORK}/emb", exist_ok=True)
    tag = f"{a.split}_s{a.src}_{a.view}{a.tag}"
    tp = f"{WORK}/tmp/texts_{tag}.npy"
    np.save(tp, np.array(texts, dtype=object), allow_pickle=True)
    del texts

    out_path = f"{WORK}/emb/{tag}.npy"
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, DIM)).flush()
    print(f"{tag}: {n:,} records on {a.gpus} GPUs -> {out_path}", flush=True)

    t0 = time.time()
    tmp.spawn(worker, args=(a.gpus, tp, out_path, n, a.maxlen, a.bs, a.model),
              nprocs=a.gpus, join=True)
    os.remove(tp)
    print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
