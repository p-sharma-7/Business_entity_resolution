"""EDA pass 1: structural facts about the challenge data.

Verifies (or refutes) every claim the proposed architecture relies on.
Writes JSON to eda/eda_01.json and a human-readable report to stdout.
"""
import csv, json, os, sys, unicodedata, random
from collections import Counter, defaultdict

DATA = "${ER_ROOT:-.}/dataset"
OUT  = "${ER_ROOT:-.}/eda"
os.makedirs(OUT, exist_ok=True)
R = {}

def read_source(path):
    """Yield (entity_id, name, address, country). Tab-split, no quoting."""
    with open(path, encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        assert header == ["entity_id","business_name","business_address","country"], header
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            yield p[0], p[1], p[2], p[3]

def script_of(s):
    """Dominant unicode script bucket of a string."""
    if not s: return "empty"
    if all(ord(c) < 128 for c in s): return "ascii"
    for c in s:
        if ord(c) < 128: continue
        try: nm = unicodedata.name(c)
        except ValueError: continue
        for tag in ("DEVANAGARI","BENGALI","TAMIL","TELUGU","KANNADA","MALAYALAM",
                    "GUJARATI","GURMUKHI","ORIYA","ARABIC","CYRILLIC","CJK","HAN"):
            if tag in nm: return tag
        if "LATIN" in nm: return "latin_ext"
    return "other"

# ---------------------------------------------------------------- sources
src_stats = {}
records = {}          # split -> source -> {eid: (name, addr, country)}
for split in ("train","test"):
    records[split] = {}
    for s in (1,2,3):
        path = f"{DATA}/{split}/{split}_source{s}.tsv"
        d = {}
        cty = Counter(); scr_n = Counter(); scr_a = Counter()
        miss_n = miss_a = 0
        for eid, nm, ad, co in read_source(path):
            d[eid] = (nm, ad, co)
            cty[co] += 1
            if not nm.strip(): miss_n += 1
            if not ad.strip(): miss_a += 1
        records[split][s] = d
        n = len(d)
        # script stats on a sample (full pass is slow for unicodedata.name)
        rnd = random.Random(0)
        sample = rnd.sample(list(d.values()), min(200000, n))
        for nm, ad, co in sample:
            scr_n[script_of(nm)] += 1; scr_a[script_of(ad)] += 1
        src_stats[f"{split}_S{s}"] = dict(
            n=n, countries=dict(cty),
            missing_name=miss_n, missing_addr=miss_a,
            missing_addr_pct=round(100*miss_a/n, 3),
            name_script_pct={k: round(100*v/len(sample),3) for k,v in scr_n.most_common()},
            addr_script_pct={k: round(100*v/len(sample),3) for k,v in scr_a.most_common()},
            dup_ids = n - len(set(d)),
        )
        print(f"{split} S{s}: n={n:,} countries={dict(cty)} missing_addr={miss_a:,} ({100*miss_a/n:.2f}%) missing_name={miss_n:,}")
        print(f"    name scripts: {dict(scr_n.most_common(5))}")
        sys.stdout.flush()
R["sources"] = src_stats

# ---------------------------------------------------------------- ground truth
gt = {}
with open(f"{DATA}/train/train_ground_truth.tsv", encoding="utf-8") as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    assert hdr == ["source1_entity_id","matched_entity_ids"], hdr
    for line in fh:
        p = line.rstrip("\n").split("\t")
        s1 = p[0]
        ids = [x for x in (p[1].split(",") if len(p)>1 and p[1] else []) if x]
        gt[s1] = ids

n_gt = len(gt)
sizes = Counter(len(v) for v in gt.values())
tot_links = sum(len(v) for v in gt.values())
singles = sizes.get(0,0)
print(f"\nGT rows={n_gt:,} links={tot_links:,} mean={tot_links/n_gt:.3f} "
      f"max={max(sizes)} singletons={singles:,} ({100*singles/n_gt:.3f}%)")
R["gt"] = dict(rows=n_gt, links=tot_links, mean=round(tot_links/n_gt,4),
               max=max(sizes), singleton_pct=round(100*singles/n_gt,4),
               size_hist={str(k): v for k,v in sorted(sizes.items())})

# every S1 in gt present in source1?
s1_ids = set(records["train"][1])
R["gt_covers_s1"] = dict(gt_rows=n_gt, s1_records=len(s1_ids),
                         gt_not_in_s1=len(set(gt)-s1_ids), s1_not_in_gt=len(s1_ids-set(gt)))
print("GT/S1 coverage:", R["gt_covers_s1"])

# ---------------------------------------------------------------- target uniqueness
tgt_claims = Counter()
for s1, ids in gt.items():
    for t in ids: tgt_claims[t] += 1
multi = Counter(tgt_claims.values())
R["target_claim_hist"] = {str(k): v for k,v in sorted(multi.items())}
n_multi = sum(v for k,v in multi.items() if k > 1)
print(f"\nTargets claimed by >1 S1: {n_multi:,} of {len(tgt_claims):,} "
      f"({100*n_multi/len(tgt_claims):.4f}%)  hist={dict(sorted(multi.items())[:6])}")

# how many S2/S3 records are matched at all
s2_ids = set(records["train"][2]); s3_ids = set(records["train"][3])
matched_s2 = sum(1 for t in tgt_claims if t.startswith("S2-"))
matched_s3 = sum(1 for t in tgt_claims if t.startswith("S3-"))
R["target_coverage"] = dict(
    s2_total=len(s2_ids), s2_matched=matched_s2, s2_matched_pct=round(100*matched_s2/len(s2_ids),3),
    s3_total=len(s3_ids), s3_matched=matched_s3, s3_matched_pct=round(100*matched_s3/len(s3_ids),3),
    gt_targets_missing_from_sources=len(set(tgt_claims)-s2_ids-s3_ids))
print("Target coverage:", R["target_coverage"])

# ---------------------------------------------------------------- country agreement
all_tr = {}
for s in (1,2,3): all_tr.update(records["train"][s])
disagree = 0; checked = 0; pairs_country = Counter()
for s1, ids in gt.items():
    c1 = all_tr.get(s1, (None,None,None))[2]
    for t in ids:
        rec = all_tr.get(t)
        if rec is None: continue
        checked += 1
        pairs_country[(c1, rec[2])] += 1
        if rec[2] != c1: disagree += 1
R["country_agreement"] = dict(checked=checked, disagree=disagree,
                              disagree_pct=round(100*disagree/checked,6),
                              pair_country_hist={f"{a}->{b}": v for (a,b),v in pairs_country.most_common(10)})
print(f"\nCountry: checked={checked:,} disagree={disagree:,} ({100*disagree/checked:.6f}%)")
print("  ", R["country_agreement"]["pair_country_hist"])

# ---------------------------------------------------------------- S1 name duplication
from collections import Counter as C2
def norm_simple(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())
name_counts = C2(norm_simple(v[0]) for v in records["train"][1].values())
dupd = sum(c for c in name_counts.values() if c > 1)
R["s1_name_dup"] = dict(unique_names=len(name_counts),
                        records_sharing_name=dupd,
                        pct=round(100*dupd/len(records["train"][1]),3),
                        max_per_name=max(name_counts.values()),
                        top=[[k,v] for k,v in name_counts.most_common(5)])
print(f"\nS1 normalized-name duplication: {dupd:,} records ({100*dupd/len(records['train'][1]):.2f}%) "
      f"share a name; max bucket={max(name_counts.values())}")

json.dump(R, open(f"{OUT}/eda_01.json","w"), indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}/eda_01.json")
