"""EDA pass 2: inspect the actual noise model in positive pairs."""
import random, unicodedata, sys, json
from collections import Counter
DATA = "${ER_ROOT:-.}/dataset"

def read_source(path):
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p)==4: yield p

rec = {}
for s in (1,2,3):
    for eid,nm,ad,co in read_source(f"{DATA}/train/train_source{s}.tsv"):
        rec[eid]=(nm,ad,co)
print("loaded", len(rec), flush=True)

gt={}
with open(f"{DATA}/train/train_ground_truth.tsv",encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        p=line.rstrip("\n").split("\t")
        gt[p[0]]=[x for x in (p[1].split(",") if len(p)>1 and p[1] else []) if x]

rnd=random.Random(7)
keys=[k for k,v in gt.items() if v]
sel=rnd.sample(keys, 40)
for k in sel:
    n,a,c = rec[k]
    print("="*100)
    print(f"S1 [{c}] {n!r}")
    print(f"        {a!r}")
    for t in gt[k]:
        if t in rec:
            n2,a2,c2 = rec[t]
            print(f"  {t[:2]} [{c2}] {n2!r}")
            print(f"        {a2!r}")
