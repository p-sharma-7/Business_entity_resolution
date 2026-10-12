"""Fast, safe TSV reading for the challenge files.

The files are strictly 4-column tab-separated with no quoting; a handful of
records contain a literal double-quote, so csv quoting must stay OFF.
"""
from __future__ import annotations
import os
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("ER_ROOT", _HERE)
DATA = os.environ.get("ER_DATA", os.path.join(ROOT, "dataset"))
WORK = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))

SRC_HEADER = ["entity_id", "business_name", "business_address", "country"]


def read_source(split: str, src: int):
    """Return dict of parallel lists: ids, names, addrs, countries."""
    path = f"{DATA}/{split}/{split}_source{src}.tsv"
    ids, names, addrs, ctys = [], [], [], []
    with open(path, encoding="utf-8", newline="") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        if hdr != SRC_HEADER:
            raise ValueError(f"unexpected header in {path}: {hdr}")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                raise ValueError(f"bad column count in {path}: {p[:1]}")
            ids.append(p[0]); names.append(p[1]); addrs.append(p[2]); ctys.append(p[3])
    return {"id": ids, "name": names, "addr": addrs, "country": ctys}


def read_ground_truth(split: str = "train"):
    """Return {s1_id: [target ids]} preserving empty lists for singletons."""
    path = f"{DATA}/{split}/{split}_ground_truth.tsv"
    gt = {}
    with open(path, encoding="utf-8", newline="") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        if hdr != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"unexpected gt header: {hdr}")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            s1 = p[0]
            raw = p[1] if len(p) > 1 else ""
            gt[s1] = [x for x in raw.split(",") if x] if raw else []
    return gt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def fold_of(entity_id: str, folds: int = 5) -> int:
    """Deterministic fold for an S1 entity, stable across processes and runs.

    Used by BOTH the encoder fine-tuning and the LightGBM folds so that a
    held-out entity is held out everywhere. Python's hash() is salted per
    process and cannot be used for this.
    """
    import hashlib
    h = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % folds
