#!/usr/bin/env python3
"""Validate submission files against every rule in the challenge spec.

Stdlib only. Prints PASS (exit 0) or a numbered list of issues (exit 1).

Rules checked:
  * exact header and tab separation, one row per test Source 1 entity
  * every Source 1 entity present, no duplicate source1_entity_id rows
  * ID lists reference only S2-/S3- ids that exist in the test set
  * no self-matches to Source 1, no duplicate ids within a list
  * matches are a subset of candidates (a matched id that was never a
    candidate signals a pipeline bug)
"""
import argparse, os, sys


def read_two_col(path, expect_col):
    issues, rows = [], {}
    order = []
    with open(path, encoding="utf-8", newline="") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        if hdr != ["source1_entity_id", expect_col]:
            issues.append(f"{os.path.basename(path)}: bad header {hdr!r}, "
                          f"expected ['source1_entity_id', {expect_col!r}]")
        for ln, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 1:
                parts = [parts[0], ""]
            if len(parts) != 2:
                issues.append(f"{os.path.basename(path)} line {ln}: {len(parts)} columns, expected 2")
                continue
            sid, raw = parts
            if sid in rows:
                issues.append(f"{os.path.basename(path)} line {ln}: duplicate source1_entity_id {sid}")
            ids = [x for x in raw.split(",") if x] if raw else []
            rows[sid] = ids
            order.append(sid)
    return rows, issues


def load_ids(path):
    out = set()
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            out.add(line.split("\t", 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()

    issues = []
    s1 = load_ids(f"{a.test_dir}/test_source1.tsv")
    s2 = load_ids(f"{a.test_dir}/test_source2.tsv")
    s3 = load_ids(f"{a.test_dir}/test_source3.tsv")
    valid = s2 | s3

    match, i1 = read_two_col(a.matching, "matched_entity_ids")
    cand, i2 = read_two_col(a.candidate, "candidate_entity_ids")
    issues += i1 + i2

    for label, rows in (("matching_results.tsv", match), ("candidate_pairs.tsv", cand)):
        missing = s1 - set(rows)
        extra = set(rows) - s1
        if missing:
            issues.append(f"{label}: {len(missing)} test Source 1 entities missing "
                          f"(e.g. {sorted(missing)[:3]})")
        if extra:
            issues.append(f"{label}: {len(extra)} rows not in test Source 1 "
                          f"(e.g. {sorted(extra)[:3]})")
        bad_self = bad_unknown = bad_dup = 0
        for sid, ids in rows.items():
            if len(ids) != len(set(ids)):
                bad_dup += 1
            for e in ids:
                if e.startswith("S1-"):
                    bad_self += 1
                elif e not in valid:
                    bad_unknown += 1
        if bad_dup:
            issues.append(f"{label}: {bad_dup} rows contain duplicate ids in one list")
        if bad_self:
            issues.append(f"{label}: {bad_self} Source 1 self-matches")
        if bad_unknown:
            issues.append(f"{label}: {bad_unknown} ids not present in the test set")

    not_cand = 0
    for sid, ids in match.items():
        cs = set(cand.get(sid, ()))
        for e in ids:
            if e not in cs:
                not_cand += 1
    if not_cand:
        issues.append(f"matched ids that never appeared as candidates: {not_cand}")

    n_nonempty = sum(1 for v in match.values() if v)
    n_ids = sum(len(v) for v in match.values())
    print(f"rows={len(match):,}  non-empty={n_nonempty:,} ({100*n_nonempty/max(len(match),1):.2f}%)  "
          f"matched ids={n_ids:,}  mean per non-empty={n_ids/max(n_nonempty,1):.2f}")
    print(f"candidate ids={sum(len(v) for v in cand.values()):,}")

    if issues:
        print("\nISSUES:")
        for i, m in enumerate(issues, 1):
            print(f"  {i}. {m}")
        sys.exit(1)
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
