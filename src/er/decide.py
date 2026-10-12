"""Decision layer: turn calibrated pair probabilities into per-S1 match sets.

For one S1 entity, if we output k candidates and the entity truly has N
matches, the metric collapses to a strikingly simple form:

    F0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP)
         = 1.25*TP / (1.25*TP + 0.25*(N-TP) + (k-TP))
         = 1.25*TP / (k + 0.25*N)

because the TP terms cancel exactly. With independent per-pair probabilities
that gives E[TP] = sum of the chosen p, E[N] = sum of ALL candidate p, so the
best k is found by sorting by p and walking outward - an O(n log n) exact
maximisation of the metric per entity, rather than one global threshold.

The empty prediction is scored separately: an S1 with no true matches earns a
full 1.0 for predicting nothing, and P(N=0) = prod(1-p). Singletons therefore
fall out of the same objective and need no separate detector.

Target-side uniqueness (every matched S2/S3 record has exactly one S1 parent,
verified on all 7,638,365 training links) is applied first as a competition
filter, which is where most false merges are removed.
"""
from __future__ import annotations
import numpy as np


def unique_target_filter(tkey, p, keep_margin=0.0):
    """Zero out every claim on a target except its strongest one."""
    order = np.lexsort((-p, tkey))
    g = tkey[order]
    new = np.empty(len(g), bool); new[0] = True
    np.not_equal(g[1:], g[:-1], out=new[1:])
    best = np.zeros(len(g), bool); best[np.flatnonzero(new)] = True
    out = np.zeros(len(p), bool); out[order] = best
    if keep_margin > 0:
        sizes = np.diff(np.append(np.flatnonzero(new), len(g)))
        second = np.repeat(np.where(sizes > 1,
                                    p[order][np.minimum(np.flatnonzero(new) + 1, len(g) - 1)],
                                    0.0), sizes)
        margin_ok = np.zeros(len(g), bool)
        margin_ok[np.flatnonzero(new)] = (p[order][np.flatnonzero(new)] - second[np.flatnonzero(new)]) >= keep_margin
        ok = np.zeros(len(p), bool); ok[order] = margin_ok
        out &= ok
    return out


def select_sets(q, p, n_entities, m_scale=1.0, p_floor=0.0, max_k=None):
    """Per-entity expected-F0.5 maximisation. Returns boolean keep mask."""
    order = np.lexsort((-p, q))
    gq = q[order]; gp = p[order]
    new = np.empty(len(gq), bool); new[0] = True
    np.not_equal(gq[1:], gq[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, len(gq)))

    csum = np.cumsum(gp)
    base = np.repeat(np.concatenate(([0.0], csum[starts[1:] - 1])), sizes)
    prefix = csum - base                       # running sum within group
    M = np.repeat(prefix[starts + sizes - 1], sizes) * m_scale
    k = (np.arange(len(gq)) - np.repeat(starts, sizes) + 1).astype(np.float64)
    gain = 1.25 * prefix / (k + 0.25 * M)

    # log P(N = 0) for the empty prediction
    lg = np.log(np.maximum(1.0 - gp, 1e-12))
    lcsum = np.cumsum(lg)
    lbase = np.repeat(np.concatenate(([0.0], lcsum[starts[1:] - 1])), sizes)
    empty = np.exp((lcsum - lbase)[starts + sizes - 1])

    keep = np.zeros(len(gq), bool)
    for gi in range(len(starts)):
        s, n = starts[gi], sizes[gi]
        g = gain[s:s + n]
        if max_k:
            n_eff = min(n, max_k); g = g[:n_eff]
        else:
            n_eff = n
        kb = int(np.argmax(g))
        if g[kb] <= empty[gi]:
            continue
        sel = gp[s:s + kb + 1] >= p_floor
        keep[s:s + kb + 1] = sel
    out = np.zeros(len(p), bool); out[order] = keep
    return out


def macro_f05(pred_sets, true_sets, all_entities):
    """Mean per-entity F0.5 over every S1 entity, singletons included."""
    tot = 0.0
    for e in all_entities:
        P = pred_sets.get(e, ())
        T = true_sets.get(e, ())
        if not T:
            tot += 1.0 if not P else 0.0
            continue
        if not P:
            continue
        tp = len(set(P) & set(T))
        if tp == 0:
            continue
        tot += 1.25 * tp / (len(P) + 0.25 * len(T))
    return tot / len(all_entities)


def macro_f05_fast(q, keep, y, n_entities, true_counts):
    """Vectorised macro-F0.5 given a keep mask over candidate pairs.

    true_counts[e] is |T| for entity e, including entities with no candidates.
    """
    sel = np.flatnonzero(keep)
    qs = q[sel]; ys = y[sel]
    k = np.bincount(qs, minlength=n_entities).astype(np.float64)
    tp = np.bincount(qs, weights=ys.astype(np.float64), minlength=n_entities)
    N = true_counts.astype(np.float64)
    score = np.zeros(n_entities)
    has_t = N > 0
    pred_empty = k == 0
    score[~has_t & pred_empty] = 1.0
    m = has_t & ~pred_empty
    score[m] = 1.25 * tp[m] / (k[m] + 0.25 * N[m])
    return score.mean()
