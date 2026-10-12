"""Do French postal codes corrupt the house-number features?"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyarrow.parquet as pq
from er.io_utils import WORK
from er import decide as D
COLS = ["addr_pre", "addr_first_num", "addr_numset", "country"]
S1 = pq.read_table(f"{WORK}/normalized/test_s1.parquet", columns=COLS).to_pydict()
T = {s: pq.read_table(f"{WORK}/normalized/test_s{s}.parquet", columns=COLS).to_pydict() for s in (2, 3)}
five = re.compile(r"\b\d{5,6}\b")
for c in ("France", "India", "US"):
    a = [x for x, cc in zip(S1["addr_pre"], S1["country"]) if cc == c]
    fn = [x for x, cc in zip(S1["addr_first_num"], S1["country"]) if cc == c]
    has = np.array([bool(five.search(x)) for x in a])
    fnpost = np.array([len(f) >= 5 for f in fn])
    print(f"S1 {c:<7}: addresses with a 5-6 digit number = {100*has.mean():5.1f}%   "
          f"first_num is itself 5-6 digits = {100*fnpost.mean():5.1f}%")
zf = np.load(f"{WORK}/pred/test_final_V6.npz"); q, t, src, cal = zf["q"], zf["t"], zf["src"], zf["cal"]
c1 = np.asarray(S1["country"])
u = D.unique_target_filter(t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000), cal)
for c in ("France", "India", "US"):
    m = np.flatnonzero(u & (c1[q] == c) & (cal >= 0.99))
    rng = np.random.default_rng(0); m = rng.choice(m, min(200000, len(m)), replace=False)
    diff = post = 0; n = 0
    for i in m:
        a = S1["addr_first_num"][q[i]]; b = T[int(src[i])]["addr_first_num"][t[i]]
        if not a or not b: continue
        n += 1
        if a != b:
            diff += 1
            if len(a) >= 5 or len(b) >= 5: post += 1
    print(f"confident {c:<7} pairs: first_num differs {100*diff/n:5.2f}%  of which a 5-6 digit number is involved: {100*post/max(diff,1):5.1f}%")
