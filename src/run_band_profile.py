"""What were the French links the leaderboard said to remove? And how do the
same patterns behave on India/US, where labels are real?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein
from er.io_utils import WORK
from er import decide as D
COLS = ["name_content", "addr_content", "addr_first_num", "addr_missing", "addr_numset"]

def profile(qv, tv, qi, ti):
    """category per pair: numdiff / swap / addr-missing / other"""
    cat = np.empty(len(qi), object)
    for k, (a, b) in enumerate(zip(qi, ti)):
        if tv["addr_missing"][b]:
            cat[k] = "target has no address"; continue
        fa, fb = qv["addr_first_num"][a], tv["addr_first_num"][b]
        A = set(qv["name_content"][a].split()); B = set(tv["name_content"][b].split())
        if fa and fb and fa != fb and not (fa.endswith(fb) or fb.endswith(fa) or fa.startswith(fb) or fb.startswith(fa)):
            cat[k] = "house number differs" + (" (same name)" if A == B else ""); continue
        da, db = A - B, B - A
        if len(da) == 1 and len(db) == 1:
            x, y = next(iter(da)), next(iter(db))
            cat[k] = "one-word swap" if Levenshtein.normalized_similarity(x, y) < 0.6 else "typo"; continue
        cat[k] = "other"
    return cat

# ---- India/US fold 4 selected links, real labels
z = np.load(f"{WORK}/pred/oof.npz"); f4 = np.load(f"{WORK}/pred/fold4_stage3.npz")
r4, sel, cal4 = f4["rows"], f4["sel"], f4["cal"]
q, t, src, y = z["q"][r4], z["t"][r4], z["src"][r4], z["y"][r4]
S1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=COLS).to_pydict()
T = {s: pq.read_table(f"{WORK}/normalized/train_s{s}.parquet", columns=COLS).to_pydict() for s in (2, 3)}
cat = np.empty(len(r4), object)
for s in (2, 3):
    m = np.flatnonzero(sel & (src == s)); cat[m] = profile(S1, T[s], q[m], t[m])
print("INDIA/US fold 4, selected links (real labels):")
for c in sorted(set(cat[sel])):
    m = sel & (cat == c)
    print(f"   {c:<34} n={int(m.sum()):>9,} ({100*m.sum()/sel.sum():5.2f}%)  true share={y[m].mean():.4f}")
mid = sel & (cal4 < 0.95)
print("  among selected links with cal in [0.7,0.95):")
for c in sorted(set(cat[mid])):
    m = mid & (cat == c)
    print(f"   {c:<34} n={int(m.sum()):>7,}  true share={y[m].mean():.4f}")

# ---- France V3 links the leaderboard said to remove (cal 0.5-0.95)
zf = np.load(f"{WORK}/pred/V3_state/test_final.npz")
qf, tf, sf, calf = zf["q"], zf["t"], zf["src"], zf["cal"]
Sq = pq.read_table(f"{WORK}/pred/V3_state/test_s1.parquet", columns=COLS + ["country"]).to_pydict()
St = {s: pq.read_table(f"{WORK}/pred/V3_state/test_s{s}.parquet", columns=COLS).to_pydict() for s in (2, 3)}
c1 = np.asarray(Sq["country"])
tk = tf.astype(np.int64) + sf.astype(np.int64) * np.int64(10_000_000)
u = D.unique_target_filter(tk, calf)
for name, lo, hi in (("France removed band [0.5,0.95)", 0.5, 0.95), ("France kept >=0.95", 0.95, 1.01)):
    band = u & (c1[qf] == "France") & (calf >= lo) & (calf < hi)
    catf = np.empty(len(qf), object)
    for s in (2, 3):
        m = np.flatnonzero(band & (sf == s)); catf[m] = profile(Sq, St[s], qf[m], tf[m])
    print(f"{name}: n={int(band.sum()):,}")
    for c in sorted(set(catf[band])):
        m = band & (catf == c)
        print(f"   {c:<34} n={int(m.sum()):>8,} ({100*m.sum()/band.sum():5.2f}%)")
