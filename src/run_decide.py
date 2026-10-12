"""Tune the decision layer on out-of-fold predictions and report macro-F0.5.

Compares, on the same OOF scores:
  A  global probability threshold (the conventional baseline)
  B  threshold after target-uniqueness filtering
  C  per-entity expected-F0.5 maximisation
  D  C after target-uniqueness filtering
so the choice is made on measured grouped macro-F0.5, not on assumption.
"""
import sys, os, json, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from er.io_utils import read_ground_truth, fold_of, WORK, ROOT
from er import decide as D

t0 = time.time()
z = np.load(f"{WORK}/pred/oof.npz")
p1, p2, y, q, t, src, fold = (z["p1"], z["p2"], z["y"], z["q"], z["t"],
                              z["src"], z["fold"])
ids1 = pq.read_table(f"{WORK}/normalized/train_s1.parquet",
                     columns=["entity_id"]).column("entity_id").to_pylist()
n_ent = len(ids1)
gt = read_ground_truth("train")
true_counts = np.zeros(n_ent, np.int32)
for i, e in enumerate(ids1):
    true_counts[i] = len(gt[e])
print(f"pairs={len(y):,} entities={n_ent:,} pos={int(y.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

# candidate-stage ceiling: best achievable if the model were perfect
ceil_keep = y.astype(bool)
print(f"CEILING (perfect model on these candidates) = "
      f"{D.macro_f05_fast(q, ceil_keep, y, n_ent, true_counts):.5f}", flush=True)

tkey = t.astype(np.int64) + src.astype(np.int64) * np.int64(10_000_000)

# isotonic calibration, fit out-of-fold to avoid leaking into the thresholds
cal = np.zeros(len(p2), np.float32)
for f in np.unique(fold):
    tr = fold != f; va = ~tr
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p2[tr], y[tr])
    cal[va] = ir.predict(p2[va])
np.save(f"{WORK}/pred/oof_cal.npy", cal)
print(f"calibrated ({time.time()-t0:.0f}s)", flush=True)

results = {}


# Fold 4 is the honest slice: its entities were held out of the ENCODER
# fine-tuning as well as the classifier fold, so it is the only subset whose
# score transfers to a test set the encoder has never seen.
ent_fold = np.fromiter((fold_of(e) for e in ids1), np.int8, len(ids1))
HOLD = 4
hold_ent = np.flatnonzero(ent_fold == HOLD)
hold_mask_pair = ent_fold[q] == HOLD
tc_hold = true_counts[hold_ent]
remap = np.full(n_ent, -1, np.int64); remap[hold_ent] = np.arange(len(hold_ent))
print(f"honest slice: {len(hold_ent):,} entities, "
      f"{int(hold_mask_pair.sum()):,} candidate pairs", flush=True)


def rep(tag, keep):
    s = D.macro_f05_fast(q, keep, y, n_ent, true_counts)
    kh = keep & hold_mask_pair
    sh = D.macro_f05_fast(remap[q[kh]] if kh.any() else np.zeros(0, np.int64),
                          np.ones(int(kh.sum()), bool), y[kh],
                          len(hold_ent), tc_hold)
    npred = int(keep.sum()); tp = int(y[keep].sum())
    prec = tp / max(npred, 1); rec = tp / max(int(y.sum()), 1)
    results[tag] = dict(macroF05=float(s), honest=float(sh), pred=npred,
                        precision=prec, recall=rec)
    print(f"  {tag:<34} F0.5={s:.5f} | HONEST(fold4)={sh:.5f}  "
          f"pred={npred:,} P={prec:.4f} R={rec:.4f}", flush=True)
    return sh


print("\n--- A: global threshold on calibrated p ---", flush=True)
best = (None, -1)
for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
    s = rep(f"A th={th}", cal >= th)
    if s > best[1]: best = (("A", th), s)

print("\n--- B: target-uniqueness filter + threshold ---", flush=True)
uniq = D.unique_target_filter(tkey, cal)
for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    s = rep(f"B th={th}", uniq & (cal >= th))
    if s > best[1]: best = (("B", th), s)

print("\n--- C: expected-F0.5 set selection ---", flush=True)
for ms in (0.8, 1.0, 1.2):
    for pf in (0.0, 0.05, 0.10, 0.20):
        s = rep(f"C m={ms} floor={pf}", D.select_sets(q.astype(np.int64), cal, n_ent,
                                                      m_scale=ms, p_floor=pf))
        if s > best[1]: best = (("C", ms, pf), s)

print("\n--- D: uniqueness + expected-F0.5 ---", flush=True)
calu = np.where(uniq, cal, 0.0)
for ms in (0.8, 1.0, 1.2):
    for pf in (0.0, 0.05, 0.10, 0.20, 0.30):
        s = rep(f"D m={ms} floor={pf}", D.select_sets(q.astype(np.int64), calu, n_ent,
                                                      m_scale=ms, p_floor=pf) & uniq)
        if s > best[1]: best = (("D", ms, pf), s)

print(f"\nBEST (by honest fold-4 macro-F0.5): {best[0]}  {best[1]:.5f}")
json.dump({"best": list(map(str, best[0])), "best_score": best[1], "all": results},
          open(f"{ROOT}/eda/decide.json", "w"), indent=2)

# ---- artefacts consumed by run_predict.py --------------------------------
# The calibrator used at test time is refit on ALL out-of-fold scores; the
# per-fold version above existed only so threshold tuning never saw its own
# calibration data.
import joblib
final_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
final_cal.fit(p2, y)
joblib.dump(final_cal, f"{ROOT}/models/calibrator.pkl")
b = best[0]
cfg = {"rule": b[0], "honest_macro_f05": best[1]}
if b[0] in ("A", "B"):
    cfg["threshold"] = float(b[1])
else:
    cfg["m_scale"] = float(b[1]); cfg["p_floor"] = float(b[2])
json.dump(cfg, open(f"{ROOT}/models/decision.json", "w"), indent=2)
print("saved models/calibrator.pkl and models/decision.json:", cfg)
