#!/usr/bin/env python3
from __future__ import annotations

import math
import pandas as pd

ALFA_TABLE = {
    # sequence_name: (failure_type_str, pre_failure_s)
    # Fill from ALFA “Processed Sequences” table.
    # Example entries:
    "2018-07-30-17-36-35": ("Engine Full Power Loss", 133.4),
    "2018-10-05-15-55-10": ("Engine Full Power Loss", 100.1),
    "2018-10-18-11-08-24": ("No Failure", None),
    # ...continue for the sequences you include in per_run.csv
}

TOL_S = 0.25  # 0.2s is cited by ALFA GT topic rate; add a little cushion.

def main() -> None:
    per_run = pd.read_csv("results/uav_sweep/per_run.csv")

    # Expect these columns; adjust if you renamed them.
    # run_id should match sequence base name (no _1/_2 suffix unless that’s your convention).
    for required in ["run_id", "failure_type", "boundary_time_s"]:
        if required not in per_run.columns:
            raise SystemExit(f"Missing column: {required}")

    problems = []
    for _, r in per_run.iterrows():
        run_id = str(r["run_id"])
        boundary = r["boundary_time_s"]
        gt = ALFA_TABLE.get(run_id)

        if gt is None:
            problems.append((run_id, "missing_in_ALFA_TABLE", boundary, None))
            continue

        gt_type, gt_pre = gt

        if gt_pre is None:
            # no-failure case
            if boundary == boundary and not pd.isna(boundary):
                problems.append((run_id, "should_have_no_boundary", boundary, None))
            continue

        # failure case
        if boundary != boundary or pd.isna(boundary):
            problems.append((run_id, "missing_boundary_time_s", boundary, gt_pre))
            continue

        if abs(float(boundary) - float(gt_pre)) > TOL_S:
            problems.append((run_id, "boundary_mismatch", float(boundary), float(gt_pre)))

    if not problems:
        print("✅ per_run.csv boundaries match ALFA pre-failure times within tolerance.")
        return

    print("❌ Mismatches found:")
    for p in problems:
        print("  ", p)

if __name__ == "__main__":
    main()
