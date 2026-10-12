"""Rebuild alias tables from cached witness counts (no 7.6M-pair rescan)."""
import sys, json, itertools
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from collections import Counter
from er.io_utils import ROOT
from er.mine_alias import build_alias_map

for field in ("addr", "name"):
    for country in ("US", "India", "ALL"):
        tag = f"{field}_{country}"
        raw = json.load(open(f"{ROOT}/models/aliasraw_{tag}.json", encoding="utf-8"))
        sub = Counter()
        for k, n in raw.items():
            a, b = k.split("\t")
            sub[(a, b)] = n
        canon, pairs, sinks = build_alias_map(sub)
        json.dump({"canon": canon, "pairs": pairs, "sinks": sinks},
                  open(f"{ROOT}/models/alias_{tag}.json", "w"),
                  indent=0, ensure_ascii=False, sort_keys=True)
        print(f"[{tag}] canon={len(canon)} pairs={len(pairs)} sinks={len(sinks)}")
        print("   canon:", dict(itertools.islice(canon.items(), 20)))
        print("   sinks:", sinks[:25])
