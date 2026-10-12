"""Materialize all normalization views for one split/source into Parquet.

Usage: run_build_views.py <split> <source>
Writes work/normalized/<split>_s<source>.parquet
"""
import sys, os, json, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import multiprocessing as mp
import pyarrow as pa, pyarrow.parquet as pq
from er.io_utils import read_source, ROOT, WORK
from er.normalize import name_views, addr_views

ALIAS_BY_CTY, NALIAS_BY_CTY, SUFFIX, NGEN, AGEN = {}, {}, None, None, None
TOK_BY_CTY = {}          # country -> (SUFFIX, NGEN, AGEN) for self-trained countries
SELF_TRAINED = ("France",)
# Pseudo-labels under-count token drops (a pair whose suffix was swapped is
# less often high-confidence), so self-trained tables use cut-offs 0.05 lower
# than the 0.35 / 0.55 applied to the training-label tables.
ST_UNSTABLE, ST_SUFFIX = 0.30, 0.50


def load_tables():
    """Alias / unstable-token tables mined from the training pairs.

    Country-specific alias maps are used where they exist so US and Indian
    conventions do not contaminate each other; an unseen country (France)
    falls back to the pooled map, which is harmless because a French address
    simply will not contain the tokens it knows about.
    """
    global ALIAS_BY_CTY, NALIAS_BY_CTY, SUFFIX, NGEN, AGEN
    for field, store in (("addr", ALIAS_BY_CTY), ("name", NALIAS_BY_CTY)):
        for cty in ("US", "India") + SELF_TRAINED:
            p = f"{ROOT}/models/alias_{field}_{cty}.json"
            if os.path.exists(p):
                canon = json.load(open(p, encoding="utf-8"))["canon"]
                if cty in SELF_TRAINED:
                    # self-trained numeric rewrites ('15' -> '0015') are
                    # zero-padding witnesses, already handled by canonical
                    # number stripping. Training-label tables are left exactly
                    # as the models were trained on them.
                    canon = {k: v for k, v in canon.items() if not k.isdigit()}
                store[cty] = canon
        # An unseen country without its own table gets NO borrowed aliases: the
        # pooled map's US state codes rewrote French 'de'/'la' into
        # 'delaware'/'louisiana'.
        store["ALL"] = {}
    ns = json.load(open(f"{ROOT}/models/tokstats_name.json", encoding="utf-8"))
    as_ = json.load(open(f"{ROOT}/models/tokstats_addr.json", encoding="utf-8"))
    sinks = set()
    for f2 in ("name_ALL", "addr_ALL"):
        p = f"{ROOT}/models/alias_{f2}.json"
        if os.path.exists(p):
            sinks |= set(json.load(open(p, encoding="utf-8")).get("sinks", []))
    # a token freely added/dropped across a true match cannot carry identity
    SUFFIX = frozenset(t for t, r in ns["unstable"].items() if r >= 0.55) | sinks
    NGEN = frozenset(ns["unstable"]) | sinks
    AGEN = frozenset(as_["unstable"]) | sinks
    for cty in SELF_TRAINED:
        pn = f"{ROOT}/models/tokstats_name_{cty}.json"
        pa = f"{ROOT}/models/tokstats_addr_{cty}.json"
        if not (os.path.exists(pn) and os.path.exists(pa)):
            continue
        cn = json.load(open(pn, encoding="utf-8"))["unstable"]
        ca = json.load(open(pa, encoding="utf-8"))["unstable"]
        # the miner stores tokens >= 0.35; re-read the full frequency table so
        # the lower self-trained cut-off can apply
        fn = {w: r for w, n, r in json.load(open(pn, encoding="utf-8"))["top_frequent"]}
        fa = {w: r for w, n, r in json.load(open(pa, encoding="utf-8"))["top_frequent"]}
        fn.update(cn); fa.update(ca)
        # A pseudo-labelled "generic sink" is trusted only if the token also
        # measurably drops out across pseudo-matches. Without this guard the
        # sinks mined for France were French type words (comite, ecole, club,
        # amicale...) - but those were learned from a model that was ACCEPTING
        # same-address distractors differing only in that word, i.e.
        # confirmation bias. Dropping them made 'traversee comite' and
        # 'traversee ecole' identical; the leaderboard confirmed those links
        # were false matches.
        csinks = set()
        for f2, rates in (("name", fn), ("addr", fa)):
            p = f"{ROOT}/models/alias_{f2}_{cty}.json"
            if os.path.exists(p):
                raw = set(json.load(open(p, encoding="utf-8")).get("sinks", []))
                kept = {w for w in raw if rates.get(w, 0.0) >= ST_UNSTABLE}
                csinks |= kept
                print(f"  {cty} {f2} sinks kept {sorted(kept)} dropped {sorted(raw - kept)}", flush=True)
        TOK_BY_CTY[cty] = (
            SUFFIX | frozenset(w for w, r in fn.items() if r >= ST_SUFFIX) | csinks,
            NGEN | frozenset(w for w, r in fn.items() if r >= ST_UNSTABLE) | csinks,
            AGEN | frozenset(w for w, r in fa.items() if r >= ST_UNSTABLE) | csinks,
        )
        print(f"  self-trained tables for {cty}: +{len(TOK_BY_CTY[cty][1]) - len(NGEN)} name, "
              f"+{len(TOK_BY_CTY[cty][2]) - len(AGEN)} address unstable tokens, "
              f"{len(ALIAS_BY_CTY.get(cty, {}))} address aliases", flush=True)


def work(args):
    lo, hi = args
    out = []
    for i in range(lo, hi):
        c = CTYS[i]
        suf, ngen, agen = TOK_BY_CTY.get(c, (SUFFIX, NGEN, AGEN))
        nv = name_views(NAMES[i], suf, ngen)
        av = addr_views(ADDRS[i], ALIAS_BY_CTY.get(c, ALIAS_BY_CTY["ALL"]), agen)
        out.append((
            nv["pre"], nv["basic"], nv["compact"], nv["compact_hg"],
            nv["core_sorted"], nv["content_sorted"], nv["acronym"],
            " ".join(nv["core"]), " ".join(nv["content"]), nv["n_tok"],
            av["pre"], av["canon"], av["compact"], av["sorted"],
            av["words_sorted"], " ".join(av["content"]),
            ",".join(av["numset"]), ",".join(av["nums"]), av["first_num"],
            av["missing"], av["n_tok"],
        ))
    return lo, out


COLS = ["name_pre", "name_basic", "name_compact", "name_compact_hg",
        "name_core_sorted", "name_content_sorted", "name_acronym",
        "name_core", "name_content", "name_ntok",
        "addr_pre", "addr_canon", "addr_compact", "addr_sorted",
        "addr_words_sorted", "addr_content",
        "addr_numset", "addr_nums", "addr_first_num",
        "addr_missing", "addr_ntok"]


if __name__ == "__main__":
    split, src = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    load_tables()
    d = read_source(split, src)
    NAMES, ADDRS, CTYS = d["name"], d["addr"], d["country"]
    n = len(NAMES)
    print(f"{split} s{src}: {n:,} records", flush=True)

    CH = 20000
    spans = [(i, min(i + CH, n)) for i in range(0, n, CH)]
    res = [None] * len(spans)
    with mp.Pool(36) as pool:
        for k, (lo, out) in enumerate(pool.imap_unordered(work, spans, chunksize=4)):
            res[lo // CH] = out
            if k % 40 == 0:
                print(f"  {k}/{len(spans)} chunks  {time.time()-t0:.0f}s", flush=True)
    rows = [r for chunk in res for r in chunk]
    assert len(rows) == n, (len(rows), n)

    arrays = {"entity_id": pa.array(d["id"]), "country": pa.array(d["country"]).dictionary_encode()}
    for j, c in enumerate(COLS):
        arrays[c] = pa.array([r[j] for r in rows])
    tbl = pa.table(arrays)
    os.makedirs(f"{WORK}/normalized", exist_ok=True)
    out = f"{WORK}/normalized/{split}_s{src}.parquet"
    pq.write_table(tbl, out, compression="zstd", compression_level=3)
    print(f"wrote {out}  {os.path.getsize(out)/1e9:.2f} GB  total {time.time()-t0:.0f}s", flush=True)
