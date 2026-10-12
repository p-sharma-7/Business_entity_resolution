"""Exact-key blocking via numpy hash joins.

Keys are hashed to uint64 and joined with searchsorted rather than Python
dicts: at 10M targets x ~12 keys a dict costs both GB and minutes, while the
sorted-array join is a few seconds and a few hundred MB. 64-bit collisions
(~3e-6 expected over this volume) can only add spurious candidates, which the
downstream classifier removes, so they cost recall nothing.
"""
from __future__ import annotations
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def hash_keys(strings, salt: int = 0) -> np.ndarray:
    """Stable 64-bit hash of a list of strings (FNV-1a, vectorised per string)."""
    import hashlib
    out = np.empty(len(strings), dtype=np.uint64)
    for i, s in enumerate(strings):
        if not s:
            out[i] = 0
            continue
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8,
                            salt=str(salt).encode()[:16]).digest()
        out[i] = int.from_bytes(h, "little")
    return out


def hash_keys_fast(strings, salt: int = 0) -> np.ndarray:
    """Python hash() based; ~10x faster than blake2b and fine within a process."""
    out = np.empty(len(strings), dtype=np.int64)
    sp = salt * 0x9E3779B97F4A7C15
    for i, s in enumerate(strings):
        out[i] = 0 if not s else (hash(s) ^ sp)
    return out.view(np.uint64)


class KeyIndex:
    """Sorted index over target key hashes supporting bucket-guarded lookup."""

    def __init__(self, key_hashes: np.ndarray, valid: np.ndarray | None = None):
        self.order = np.argsort(key_hashes, kind="stable").astype(np.int64)
        self.sorted = key_hashes[self.order]
        if valid is not None:
            keep = valid[self.order]
            self.order = self.order[keep]
            self.sorted = self.sorted[keep]

    def lookup(self, query_hashes: np.ndarray, max_bucket: int = 300):
        """Return (query_row, target_row) arrays for all key-equal matches.

        Buckets larger than max_bucket are dropped: a key shared by hundreds of
        targets carries almost no identifying information (39% of S1 records
        share a normalized name, one bucket holds 253), and materialising it
        would dominate the candidate volume.
        """
        lo = np.searchsorted(self.sorted, query_hashes, "left")
        hi = np.searchsorted(self.sorted, query_hashes, "right")
        cnt = hi - lo
        cnt[query_hashes == 0] = 0          # empty key never matches
        cnt[cnt > max_bucket] = 0
        total = int(cnt.sum())
        if total == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64))
        qrow = np.repeat(np.arange(len(query_hashes), dtype=np.int64), cnt)
        # offsets within each bucket
        starts = np.repeat(lo, cnt)
        idx = np.arange(total, dtype=np.int64)
        base = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        offs = idx - np.repeat(base, cnt)
        trow = self.order[starts + offs]
        return qrow, trow


def pack(a: np.ndarray, b: np.ndarray, shift: int = 24) -> np.ndarray:
    return (a.astype(np.int64) << np.int64(shift)) | b.astype(np.int64)


def unpack(p: np.ndarray, shift: int = 24):
    return (p >> np.int64(shift)).astype(np.int64), (p & np.int64((1 << shift) - 1)).astype(np.int64)
