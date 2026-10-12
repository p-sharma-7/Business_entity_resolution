import sys, json, time, os
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from er.io_utils import read_source, read_ground_truth, ROOT
from er.mine_alias import mine_from_ground_truth, build_alias_map

t0 = time.time()
rec = {}
cty = {}
for s in (1, 2, 3):
    d = read_source("train", s)
    for eid, nm, ad, co in zip(d["id"], d["name"], d["addr"], d["country"]):
        rec[eid] = (nm, ad); cty[eid] = co
print(f"loaded {len(rec):,} records in {time.time()-t0:.0f}s", flush=True)
gt = read_ground_truth("train")

os.makedirs(f"{ROOT}/models", exist_ok=True)
for field in ("addr", "name"):
    # mine per country so US/India conventions do not contaminate each other
    for country in ("US", "India", "ALL"):
        ids = None if country == "ALL" else {k for k in gt if cty.get(k) == country}
        t = time.time()
        sub = mine_from_ground_truth(rec, gt, field, s1_ids=ids)
        canon, pairs = build_alias_map(sub)
        tag = f"{field}_{country}"
        # raw witness counts are kept so the maps can be rebuilt without
        # re-scanning 7.6M pairs
        json.dump({"canon": canon, "pairs": pairs},
                  open(f"{ROOT}/models/alias_{tag}.json", "w"),
                  indent=0, ensure_ascii=False, sort_keys=True)
        json.dump({f"{a}\t{b}": n for (a, b), n in sub.items() if n >= 10},
                  open(f"{ROOT}/models/aliasraw_{tag}.json", "w"),
                  ensure_ascii=False)
        print(f"[{tag}] witnesses={len(sub):,} canon={len(canon):,} pairs={len(pairs):,} ({time.time()-t:.0f}s)", flush=True)
        import itertools
        print("   canon sample:", dict(itertools.islice(canon.items(), 22)), flush=True)
