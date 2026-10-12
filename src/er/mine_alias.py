"""Learn token-equivalence tables from the challenge's own matched pairs.

No external gazetteer is used or needed. For every ground-truth pair we align
the two token multisets; when exactly one token differs on each side, that is a
clean 1-for-1 substitution witnessed by a real match (Rd/Road, IA/Iowa,
dilli/Delhi, KL/Kerala, Ltd/Limited). Aggregating those witnesses over millions
of pairs yields a high-precision alias table that covers whatever scripts and
conventions the data happens to contain.

For a country present only at test time (France), the same routine can be run
over high-confidence *predicted* pairs, so the table extends itself without any
external data. See mine_alias_selftrain().
"""
from __future__ import annotations
import json, sys, os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from er.normalize import basic, nfkc_fold, _NULL_TOKENS

MIN_COUNT = 40          # a substitution must be witnessed this often
MIN_RATIO = 0.55        # and must dominate the source token's alternatives


def _toks(s: str) -> list[str]:
    return [t for t in basic(nfkc_fold(s)).split() if t not in _NULL_TOKENS]


def collect_substitutions(pairs) -> Counter:
    """pairs: iterable of (text_a, text_b). Returns Counter{(a_tok,b_tok):n}."""
    sub = Counter()
    for a, b in pairs:
        ta, tb = _toks(a), _toks(b)
        if not ta or not tb:
            continue
        sa, sb = set(ta), set(tb)
        da, db = sa - sb, sb - sa
        # only unambiguous 1<->1 residuals are trustworthy witnesses
        if len(da) == 1 and len(db) == 1:
            x, y = da.pop(), db.pop()
            if x != y and len(x) <= 40 and len(y) <= 40:
                sub[(x, y)] += 1
                sub[(y, x)] += 1
    return sub


def build_alias_map(sub: Counter, min_count=MIN_COUNT, min_ratio=MIN_RATIO,
                    sink_indegree=6, rev_ratio=0.25):
    """Witnessed substitutions -> (canonical map, equivalence pairs, sinks).

    Two failure modes have to be guarded against, both observed in this data:

    1. AMBIGUOUS ABBREVIATIONS. 'st' witnesses both 'street' and 'saint', 'ct'
       both 'court' and 'connecticut'. A union-find would fuse Connecticut with
       Court, so no transitive closure is taken and an ambiguous source token
       is left unrewritten.

    2. GENERIC SINKS. The generator swaps a real word for a filler one
       ('Rns Hospital' -> 'Rns Center', 'Recall Exim Center' -> '... Services'),
       so hundreds of unrelated tokens all "substitute" for 'center'. Treating
       those as aliases would make 'Ford Motors' equal 'Center Motors'. A
       target that many distinct sources map into is therefore classified as a
       generic sink, not an alias: the mapping is dropped and the sink token is
       reported so it can join the low-information vocabulary instead. Genuine
       abbreviations survive the cull because they are MUTUAL - 'limited' maps
       back to 'ltd' - whereas 'center' never maps back to 'ford'.
    """
    by_src = defaultdict(Counter)
    for (x, y), n in sub.items():
        if n >= min_count:
            by_src[x][y] = n
    tot = {x: sum(c.values()) for x, c in by_src.items()}

    # Orient every witnessed pair ABBREVIATION -> EXPANSION (shorter to longer).
    # Choosing the long form as canonical is what prevents the ambiguous-
    # abbreviation collapse: with 'ct' as a target, both 'court' and
    # 'connecticut' would rewrite to it and become equal. Oriented outward,
    # 'ct' instead becomes a source with two competing expansions, and the
    # dominance test below simply declines to rewrite it.
    oriented = defaultdict(Counter)
    for (x, y), n in sub.items():
        if n < min_count or x == y:
            continue
        if len(x) < len(y):
            src, dst = x, y
        elif len(y) < len(x):
            src, dst = y, x
        else:
            src, dst = (x, y) if tot.get(x, 0) <= tot.get(y, 0) else (y, x)
        oriented[src][dst] = max(oriented[src][dst], n)

    edges = {}
    for x, cand in oriented.items():
        y, n = cand.most_common(1)[0]
        if n / sum(cand.values()) >= min_ratio:      # unambiguous expansion
            edges[x] = (y, n)

    indeg = Counter(y for y, _ in edges.values())
    sinks = {y for y, k in indeg.items() if k > sink_indegree}

    def mutual(x, y):
        c = by_src.get(y)
        if not c:
            return False
        return c.most_common(1)[0][0] == x or c.get(x, 0) / tot[y] >= min_ratio

    canon, keep_edges = {}, []
    for x, (y, n) in edges.items():
        if y in sinks and not mutual(x, y):
            continue                       # generic swap, not an alias
        if n / tot.get(y, n) < rev_ratio:
            continue                       # target's mass sits elsewhere
        keep_edges.append((x, y))
        canon[x] = y
    canon = {k: v for k, v in canon.items() if v not in canon}   # no chains

    alias_pairs = sorted({(min(x, y), max(x, y)) for x, y in keep_edges})
    return canon, alias_pairs, sorted(sinks)


def mine_from_ground_truth(records: dict, gt: dict, field: str,
                           s1_ids=None, max_pairs=None):
    """records: {eid: (name, addr)}. field: 'name' or 'addr'.

    s1_ids restricts the mining to a subset of S1 entities so that validation
    folds can mine without touching their own held-out labels.
    """
    idx = 0 if field == "name" else 1
    def gen():
        n = 0
        for s1, targets in gt.items():
            if s1_ids is not None and s1 not in s1_ids:
                continue
            a = records.get(s1)
            if a is None:
                continue
            for t in targets:
                b = records.get(t)
                if b is None:
                    continue
                yield a[idx], b[idx]
                n += 1
                if max_pairs and n >= max_pairs:
                    return
    return collect_substitutions(gen())
