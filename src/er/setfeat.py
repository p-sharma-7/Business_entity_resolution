"""CSR-packed set features with numba kernels.

At ~10^8 candidate pairs, a Python-level similarity call per pair is hopeless
(hours per feature). Instead each record's token set and character n-gram set
is hashed to sorted int32 ids and packed CSR-style; intersection size is then a
linear merge that numba runs in parallel at ~10^8 pairs/second/feature.

Everything derived from an intersection (Jaccard, Dice, both containments,
IDF-weighted cosine) comes from a single pass.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange


def build_csr(seqs, ngram: int = 0):
    """Pack per-record token or char-ngram id sets into (vals, offs).

    ngram=0 treats each sequence as a whitespace token list; ngram=k builds
    character k-grams of the joined string.
    """
    offs = np.zeros(len(seqs) + 1, dtype=np.int64)
    chunks = []
    for i, s in enumerate(seqs):
        if ngram:
            t = s
            items = {t[k:k + ngram] for k in range(max(len(t) - ngram + 1, 0))} if len(t) >= ngram else ({t} if t else set())
        else:
            items = set(s.split()) if isinstance(s, str) else set(s)
        if items:
            a = np.fromiter((hash(x) & 0x7FFFFFFF for x in items), dtype=np.int32, count=len(items))
            a.sort()
            chunks.append(a)
            offs[i + 1] = offs[i] + a.size
        else:
            chunks.append(np.empty(0, np.int32))
            offs[i + 1] = offs[i]
    vals = np.concatenate(chunks) if chunks else np.empty(0, np.int32)
    return vals, offs


@njit(cache=True, parallel=True, fastmath=True)
def _inter_sizes(qi, tj, av, ao, bv, bo, out_inter, out_na, out_nb):
    for p in prange(qi.shape[0]):
        i = qi[p]; j = tj[p]
        a0 = ao[i]; a1 = ao[i + 1]
        b0 = bo[j]; b1 = bo[j + 1]
        na = a1 - a0; nb = b1 - b0
        c = 0
        x = a0; y = b0
        while x < a1 and y < b1:
            u = av[x]; v = bv[y]
            if u == v:
                c += 1; x += 1; y += 1
            elif u < v:
                x += 1
            else:
                y += 1
        out_inter[p] = c
        out_na[p] = na
        out_nb[p] = nb


def set_feats(qi, tj, A, B):
    """Return (jaccard, dice, containment_a_in_b, containment_b_in_a, inter, na, nb)."""
    av, ao = A; bv, bo = B
    n = qi.shape[0]
    inter = np.empty(n, np.int32); na = np.empty(n, np.int32); nb = np.empty(n, np.int32)
    _inter_sizes(qi, tj, av, ao, bv, bo, inter, na, nb)
    f = inter.astype(np.float32)
    fa = na.astype(np.float32); fb = nb.astype(np.float32)
    union = fa + fb - f
    jac = np.where(union > 0, f / np.maximum(union, 1e-9), 0.0).astype(np.float32)
    dice = np.where(fa + fb > 0, 2 * f / np.maximum(fa + fb, 1e-9), 0.0).astype(np.float32)
    ca = np.where(fa > 0, f / np.maximum(fa, 1e-9), 0.0).astype(np.float32)
    cb = np.where(fb > 0, f / np.maximum(fb, 1e-9), 0.0).astype(np.float32)
    cos = np.where((fa > 0) & (fb > 0), f / np.maximum(np.sqrt(fa * fb), 1e-9), 0.0).astype(np.float32)
    return jac, dice, ca, cb, cos, inter, na, nb


@njit(cache=True, parallel=True, fastmath=True)
def _widf(qi, tj, av, ao, bv, bo, keys, wts, out_i, out_a, out_b):
    """IDF-weighted intersection / self-weights. keys must be sorted."""
    nk = keys.shape[0]
    for p in prange(qi.shape[0]):
        i = qi[p]; j = tj[p]
        a0 = ao[i]; a1 = ao[i + 1]
        b0 = bo[j]; b1 = bo[j + 1]
        wi = 0.0; wa = 0.0; wb = 0.0
        x = a0; y = b0
        while x < a1 and y < b1:
            u = av[x]; v = bv[y]
            if u == v:
                lo = 0; hi = nk - 1; w = 0.0
                while lo <= hi:
                    m = (lo + hi) >> 1
                    if keys[m] == u:
                        w = wts[m]; break
                    elif keys[m] < u:
                        lo = m + 1
                    else:
                        hi = m - 1
                wi += w; x += 1; y += 1
            elif u < v:
                x += 1
            else:
                y += 1
        for x in range(a0, a1):
            u = av[x]
            lo = 0; hi = nk - 1; w = 0.0
            while lo <= hi:
                m = (lo + hi) >> 1
                if keys[m] == u:
                    w = wts[m]; break
                elif keys[m] < u:
                    lo = m + 1
                else:
                    hi = m - 1
            wa += w
        for y in range(b0, b1):
            v = bv[y]
            lo = 0; hi = nk - 1; w = 0.0
            while lo <= hi:
                m = (lo + hi) >> 1
                if keys[m] == v:
                    w = wts[m]; break
                elif keys[m] < v:
                    lo = m + 1
                else:
                    hi = m - 1
            wb += w
        out_i[p] = wi; out_a[p] = wa; out_b[p] = wb


def idf_table(vals, offs, n_docs):
    """IDF over the packed id universe."""
    keys, counts = np.unique(vals, return_counts=True)
    idf = np.log(1.0 + n_docs / (1.0 + counts)).astype(np.float32)
    return keys.astype(np.int32), idf


def widf_feats(qi, tj, A, B, keys, wts):
    av, ao = A; bv, bo = B
    n = qi.shape[0]
    wi = np.empty(n, np.float32); wa = np.empty(n, np.float32); wb = np.empty(n, np.float32)
    _widf(qi, tj, av, ao, bv, bo, keys, wts, wi, wa, wb)
    cos = np.where((wa > 0) & (wb > 0), wi / np.maximum(np.sqrt(wa * wb), 1e-9), 0.0).astype(np.float32)
    ca = np.where(wa > 0, wi / np.maximum(wa, 1e-9), 0.0).astype(np.float32)
    cb = np.where(wb > 0, wi / np.maximum(wb, 1e-9), 0.0).astype(np.float32)
    return cos, ca, cb


@njit(cache=True, parallel=True, fastmath=True)
def _light(qi, tj, av, ao, bv, bo, out_jac, out_cmax, out_cos):
    """Jaccard, max-containment and cosine in a single merge pass.

    The full set_feats allocates eight arrays per call; at 10^8 candidate pairs
    that allocation churn, not the merge itself, dominates the pruning stage.
    """
    for p in prange(qi.shape[0]):
        i = qi[p]; j = tj[p]
        a0 = ao[i]; a1 = ao[i + 1]
        b0 = bo[j]; b1 = bo[j + 1]
        na = a1 - a0; nb = b1 - b0
        if na == 0 or nb == 0:
            out_jac[p] = 0.0; out_cmax[p] = 0.0; out_cos[p] = 0.0
            continue
        c = 0
        x = a0; y = b0
        while x < a1 and y < b1:
            u = av[x]; v = bv[y]
            if u == v:
                c += 1; x += 1; y += 1
            elif u < v:
                x += 1
            else:
                y += 1
        f = np.float32(c)
        out_jac[p] = f / np.float32(na + nb - c)
        ca = f / np.float32(na); cb = f / np.float32(nb)
        out_cmax[p] = ca if ca > cb else cb
        out_cos[p] = f / np.sqrt(np.float32(na) * np.float32(nb))


def light_feats(qi, tj, A, B):
    n = qi.shape[0]
    j = np.empty(n, np.float32); c = np.empty(n, np.float32); o = np.empty(n, np.float32)
    _light(qi, tj, A[0], A[1], B[0], B[1], j, c, o)
    return j, c, o


def build_csr_dense(seqs_q, seqs_t, ngram: int = 0):
    """CSR for both sides over a SHARED dense vocabulary, plus target-side IDF.

    Dense ids turn the weight lookup into a direct array index, so IDF-weighted
    overlap costs the same as unweighted. That matters because the unweighted
    version saturates: an address token set that is a subset of the query's
    scores containment 1.0 whether it shared 'jaipur rajasthan' or
    'pratap nagar sector 26', producing huge ties exactly at the top-k
    boundary where the pruner has to choose.
    """
    vocab = {}

    def pack(seqs):
        offs = np.zeros(len(seqs) + 1, dtype=np.int64)
        chunks = []
        for i, s in enumerate(seqs):
            if ngram:
                items = ({s[k:k + ngram] for k in range(len(s) - ngram + 1)}
                         if len(s) >= ngram else ({s} if s else set()))
            else:
                items = set(s.split()) if isinstance(s, str) else set(s)
            if items:
                a = np.empty(len(items), np.int32)
                for j, x in enumerate(items):
                    v = vocab.get(x)
                    if v is None:
                        v = len(vocab); vocab[x] = v
                    a[j] = v
                a.sort()
                chunks.append(a); offs[i + 1] = offs[i] + a.size
            else:
                chunks.append(np.empty(0, np.int32)); offs[i + 1] = offs[i]
        return (np.concatenate(chunks) if chunks else np.empty(0, np.int32)), offs

    tv, to = pack(seqs_t)
    qv, qo = pack(seqs_q)
    V = len(vocab)
    df = np.bincount(tv, minlength=V).astype(np.float32)
    idf = np.log(1.0 + len(seqs_t) / (1.0 + df)).astype(np.float32)
    return (qv, qo), (tv, to), idf


@njit(cache=True, parallel=True, fastmath=True)
def _wlight(qi, tj, av, ao, bv, bo, w, out_cos, out_cmax):
    for p in prange(qi.shape[0]):
        i = qi[p]; j = tj[p]
        a0 = ao[i]; a1 = ao[i + 1]
        b0 = bo[j]; b1 = bo[j + 1]
        if a1 == a0 or b1 == b0:
            out_cos[p] = 0.0; out_cmax[p] = 0.0
            continue
        wi = np.float32(0.0); wa = np.float32(0.0); wb = np.float32(0.0)
        x = a0; y = b0
        while x < a1 and y < b1:
            u = av[x]; v = bv[y]
            if u == v:
                wi += w[u]; x += 1; y += 1
            elif u < v:
                x += 1
            else:
                y += 1
        for x in range(a0, a1):
            wa += w[av[x]]
        for y in range(b0, b1):
            wb += w[bv[y]]
        if wa <= 0.0 or wb <= 0.0:
            out_cos[p] = 0.0; out_cmax[p] = 0.0
        else:
            out_cos[p] = wi / np.sqrt(wa * wb)
            ca = wi / wa; cb = wi / wb
            out_cmax[p] = ca if ca > cb else cb


def wlight_feats(qi, tj, A, B, idf):
    n = qi.shape[0]
    c = np.empty(n, np.float32); m = np.empty(n, np.float32)
    _wlight(qi, tj, A[0], A[1], B[0], B[1], idf, c, m)
    return c, m
