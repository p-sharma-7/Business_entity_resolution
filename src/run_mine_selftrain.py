"""Self-trained lookup tables for a country that has no training labels.

France appears only in the test set, so its alias and token-stability tables
cannot be mined from train_ground_truth.tsv. Two failures follow from that:
  * addresses fell back to the pooled US+India alias map, whose US state
    abbreviations rewrite French function words ('de' -> 'delaware',
    'la' -> 'louisiana'): 'delaware' became the most frequent token in French
    S1 addresses;
  * French legal forms (sarl, sas, sasu, eurl, sa, sci, ...) were unknown, so
    a suffix swap the generator makes freely counted as a name conflict.

Fix: run the SAME miners used on the training labels, over high-confidence
predicted France pairs (pseudo-labels from the current model). No external
data is involved; only the challenge's own test records.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from er.io_utils import WORK, ROOT, read_source
from er.normalize import basic, nfkc_fold, _NULL_TOKENS, _ORD_SUFFIX
from er.mine_alias import collect_substitutions, build_alias_map

COUNTRY = "France"
MIN_CONF = 0.85
SCALE = 0.1          # pseudo-pair volume is ~10% of the training label volume


def toks(s, alias=None, addr=False):
    out = []
    for t in basic(nfkc_fold(s)).split():
        if t in _NULL_TOKENS:
            continue
        if addr:
            m = _ORD_SUFFIX.match(t)
            if m:
                t = m.group(1)
            if t.isdigit():
                t = t.lstrip("0") or "0"
        if alias:
            t = alias.get(t, t)
        out.append(t)
    return out


def drop_rates(pairs, alias, addr, min_seen):
    seen, diff = Counter(), Counter()
    for a, b in pairs:
        sa, sb = set(toks(a, alias, addr)), set(toks(b, alias, addr))
        if not sa or not sb:
            continue
        for w in sa | sb:
            seen[w] += 1
            if (w in sa) != (w in sb):
                diff[w] += 1
    rows = sorted(((w, n, diff[w] / n) for w, n in seen.items() if n >= min_seen), key=lambda r: -r[1])
    return rows


def main():
    t0 = time.time()
    z = np.load(f"{WORK}/pred/test_final.npz")
    q, t, src, cal, keep = z["q"], z["t"], z["src"], z["cal"], z["keep"]
    s1 = read_source("test", 1)
    tg = {s: read_source("test", s) for s in (2, 3)}
    c1 = np.asarray(s1["country"])
    sel = np.flatnonzero(keep & (cal >= MIN_CONF) & (c1[q] == COUNTRY))
    names = [(s1["name"][q[i]], tg[int(src[i])]["name"][t[i]]) for i in sel]
    addrs = [(s1["addr"][q[i]], tg[int(src[i])]["addr"][t[i]]) for i in sel]
    print(f"{COUNTRY} pseudo-positive pairs (keep & cal>={MIN_CONF}): {len(sel):,} ({time.time()-t0:.0f}s)", flush=True)

    mc = max(8, int(40 * SCALE))
    out = {}
    for field, pairs in (("addr", addrs), ("name", names)):
        sub = collect_substitutions(pairs)
        canon, pairs_eq, sinks = build_alias_map(sub, min_count=mc)
        out[field] = canon
        json.dump({"canon": canon, "pairs": pairs_eq, "sinks": sinks},
                  open(f"{ROOT}/models/alias_{field}_{COUNTRY}.json", "w"),
                  indent=0, ensure_ascii=False, sort_keys=True)
        print(f"[{field}] witnesses={len(sub):,} canon={len(canon)} sinks={sinks[:20]}", flush=True)
        print("   canon sample:", dict(list(canon.items())[:30]), flush=True)

    min_seen = max(200, int(2000 * SCALE))
    for field, pairs, alias, addr in (("name", names, None, False),
                                      ("addr", addrs, out["addr"], True)):
        rows = drop_rates(pairs, alias, addr, min_seen)
        unstable = {w: round(r, 4) for w, n, r in rows if r >= 0.35}
        json.dump({"npairs": len(pairs), "unstable": unstable,
                   "top_frequent": [[w, n, round(r, 4)] for w, n, r in rows[:150]]},
                  open(f"{ROOT}/models/tokstats_{field}_{COUNTRY}.json", "w"),
                  indent=1, ensure_ascii=False)
        print(f"\n=== {field}: unstable={len(unstable)}", flush=True)
        for w, n, r in rows[:40]:
            print(f"    {w:<20} {n:>8,}  {r:.3f}", flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
