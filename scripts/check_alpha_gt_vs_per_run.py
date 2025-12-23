#check_alpha_gt_vs_per_run.py
from __future__ import annotations
import math
import argparse
import re
from pathlib import Path

import pandas as pd


def _extract_seq_id(run_id: str) -> str | None:
    """
    Extract ALFA processed sequence ID from a run_id string.
    Examples:
      carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj -> 2018-10-18-11-04-08_1
      2018-09-11-11-56-30__hard_spike -> 2018-09-11-11-56-30
    """
    m = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:_\d+)?", str(run_id))
    return m.group(0) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sanity-check ALFA ground-truth manifest vs per_run.csv")
    p.add_argument("--per-run", required=True, type=str, help="Path to per_run.csv")
    p.add_argument("--manifest", required=True, type=str, help="Path to alfa_gt_manifest.csv")
    p.add_argument(
        "--tolerance-s",
        type=float,
        default=0.25,
        help="Allowed absolute difference (seconds) between per_run boundary_time_s and manifest gt_failure_time_s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    per_run_path = Path(args.per_run)
    manifest_path = Path(args.manifest)

    per_run = pd.read_csv(per_run_path)
    manifest = pd.read_csv(manifest_path)

    if "run_id" not in per_run.columns:
        raise ValueError(f"per_run missing run_id column: {per_run_path}")
    if "run_id" not in manifest.columns:
        raise ValueError(f"manifest missing run_id column: {manifest_path}")
    if "gt_failure_time_s" not in manifest.columns:
        raise ValueError(f"manifest missing gt_failure_time_s column: {manifest_path}")
    if "boundary_time_s" not in per_run.columns:
        raise ValueError(f"per_run missing boundary_time_s column: {per_run_path}")

    merged = per_run.merge(
        manifest[["run_id", "gt_failure_time_s"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    )

    merged["seq_id"] = merged["run_id"].map(_extract_seq_id)
    merged["abs_diff_s"] = (merged["boundary_time_s"] - merged["gt_failure_time_s"]).abs()

    missing = merged["gt_failure_time_s"].isna()
    if missing.any():
        print("[check] WARNING: missing gt_failure_time_s for these run_id rows:")
        print(merged.loc[missing, ["run_id", "seq_id"]].to_string(index=False))

    bad = (~missing) & (merged["abs_diff_s"] > float(args.tolerance_s))
    if bad.any():
        print(f"[check] MISMATCHES (> {args.tolerance_s:.3f}s):")
        print(
            merged.loc[bad, ["run_id", "seq_id", "boundary_time_s", "gt_failure_time_s", "abs_diff_s"]]
            .sort_values("abs_diff_s", ascending=False)
            .to_string(index=False)
        )
        raise SystemExit(2)

    print(f"[check] OK: {len(merged)} rows; all within {args.tolerance_s:.3f}s tolerance.")


if __name__ == "__main__":
    main()

