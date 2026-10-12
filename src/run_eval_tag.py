"""Honest fold-4 macro-F0.5 for a tagged stage-2 model family, per country.

Same protocol as run_decide.py rule B (out-of-fold isotonic calibration,
target-uniqueness over every pair, one threshold), so the language-agnostic
model is compared like-for-like with the production stage 2 (0.98428).
Saves models/calibrator<TAG>.pkl fit on all out-of-fold scores.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
import joblib
from sklearn.isotonic import IsotonicRegression
from er.io_utils import WORK, ROOT, fold_of, read_ground_truth
from er import decide as D

TAG = os.environ.get("ER_TAG", "")


def main():
    z = np.load(f"{WORK}/pred/oof{TAG}.npz")
    p2, y, q, t, src, fold = z["p2"], z["y"], z["q"], z["t"], z["src"], z["fold"]
    tb = pq.read_table(f"{WORK}/normalized/train_s1.parquet", columns=["entity_id", "country"])
    ids = tb.column("entity_id").to_pylist(); cty = np.asarray(tb.column("country").to_pylist())
    gt = read_ground_truth("train"); N = np.array([len(gt[e]) for e in ids], np.int32)
    ef = np.array([fold_of(e) for e in ids], np.int8)
    cal = np.zeros(len(p2), np.float32)
    for f in np.unique(fold):
        tr = fold != f
        cal[~tr] = IsotonicRegression(out_of_bounds="clip").fit(p2[tr], y[tr]).predict(p2[~tr])
    joblib.dump(IsotonicRegression(out_of_bounds="clip").fit(p2, y), f"{ROOT}/models/calibrator{TAG}.pkl")
    tk = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)
    uniq = D.unique_target_filter(tk, cal)
    res = {}
    for name, ents in (("fold4 all", np.flatnonzero(ef == 4)),
                       ("fold4 India", np.flatnonzero((ef == 4) & (cty == "India"))),
                       ("fold4 US", np.flatnonzero((ef == 4) & (cty == "US")))):
        rm = np.full(len(ids), -1, np.int64); rm[ents] = np.arange(len(ents))
        m = rm[q] >= 0
        line = []
        for th in (0.5, 0.6, 0.7, 0.8, 0.9):
            k = uniq & (cal >= th) & m
            line.append((th, D.macro_f05_fast(rm[q[k]], np.ones(int(k.sum()), bool), y[k], len(ents), N[ents])))
        best = max(line, key=lambda x: x[1])
        res[name] = best
        print(f"[{TAG or 'prod'}] {name:<12} " + "  ".join(f"th{a}:{b:.5f}" for a, b in line)
              + f"   best th={best[0]} -> {best[1]:.5f}", flush=True)
    json.dump({k: list(v) for k, v in res.items()}, open(f"{ROOT}/eda/eval{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
