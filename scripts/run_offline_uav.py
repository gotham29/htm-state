from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys
import pandas as pd

# Allow running as: python scripts/run_offline_uav_all.py
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from offline_demo_uav import evaluate_uav_csv, parse_args as offline_parse_args


def classify_failure(run_id: str) -> str:
    n = run_id.lower()

    if "no_ground_truth" in n:
        return "no_ground_truth"
    if "no_failure" in n:
        return "no_failure"

    has_engine = "engine_failure" in n
    has_elevator = "elevator_failure" in n
    has_rudder = ("rudder" in n and "failure" in n)
    has_aileron = ("aileron" in n and "failure" in n)

    families = sum([has_engine, has_elevator, has_rudder, has_aileron])
    if families >= 2 or "__" in n:
        return "multi_fault"

    if has_engine:
        return "engine_failure"
    if has_elevator:
        return "elevator_failure"
    if has_rudder:
        return "rudder_failure"
    if has_aileron:
        return "aileron_failure"

    return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Offline sweep over generated ALFA UAV streams (strict benchmark)")
    p.add_argument("--generated-dir", type=str, default="demos/uav_demo/generated")
    p.add_argument("--outdir", type=str, default="results/uav_sweep")
    p.add_argument("--glob", type=str, default="*.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(gen_dir.glob(args.glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {gen_dir} with glob {args.glob!r}")

    # IMPORTANT: do NOT consume sweep CLI args
    # Use defaults only for offline evaluation
    _argv0 = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]  # parse defaults only
        offline_args = offline_parse_args()
    finally:
        sys.argv = _argv0
    offline_args.strict_boundary = True  # enforce strict benchmark policy

    per_run_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        run_id = csv_path.stem
        failure_type = classify_failure(run_id)

        res = evaluate_uav_csv(csv_path, offline_args)
        # STRICT benchmark policy: require a ground-truth boundary
        if not res.get("has_boundary", False):
            res = {"included": False, "exclude_reason": "no_boundary_strict"}

        coverage_rows.append({
            "run_id": run_id,
            "csv_path": str(csv_path),
            "failure_type": failure_type,
            "included": bool(res.get("included", False)),
            "exclude_reason": res.get("exclude_reason", ""),
        })

        if not res.get("included", False):
            continue

        per_run_rows.append({
            "run_id": run_id,
            "csv_path": str(csv_path),
            "failure_type": failure_type,
            "boundary_time_s": res.get("boundary_time_s", None),
            "spike_detected": res.get("spike_detected", False),
            "spike_lag_s": res.get("spike_lag_s", None),
            "sustained_detected": res.get("sustained_detected", False),
            "sustained_lag_s": res.get("sustained_lag_s", None),
            "false_alarms_spm": res.get("false_alarms_spm", None),
            "post_elev_frac": res.get("post_elev_frac", None),
            "n_spikes_total": res.get("n_spikes_total", None),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    coverage_df.to_csv(outdir / "coverage.csv", index=False)
    per_run_df.to_csv(outdir / "per_run.csv", index=False)

    # If nothing was included, write what we have and exit gracefully with diagnostics.
    if per_run_df.empty:
        print(f"[run_offline_uav_all] discovered: {len(csv_paths)}")
        print(f"[run_offline_uav_all] included (strict): 0")
        print(f"[run_offline_uav_all] wrote: {outdir / 'coverage.csv'}")
        print(f"[run_offline_uav_all] wrote: {outdir / 'per_run.csv'}")
        if "exclude_reason" in coverage_df.columns:
            counts = coverage_df["exclude_reason"].fillna("").value_counts()
            print("[run_offline_uav_all] exclusion counts:")
            for k, v in counts.items():
                print(f"  {k or '(blank)'}: {v}")
        return

    summary_rows = []
    for ftype, g in per_run_df.groupby("failure_type", dropna=False):
        n = int(len(g))
        spike_rate = float(g["spike_detected"].astype(float).mean()) if n else float("nan")
        sust_rate = float(g["sustained_detected"].astype(float).mean()) if n else float("nan")

        spike_lag_med = float(g.loc[g["spike_detected"] == True, "spike_lag_s"].median()) if (g["spike_detected"] == True).any() else float("nan")
        sust_lag_med = float(g.loc[g["sustained_detected"] == True, "sustained_lag_s"].median()) if (g["sustained_detected"] == True).any() else float("nan")

        fa_med = float(g["false_alarms_spm"].median()) if n else float("nan")
        post_med = float(g["post_elev_frac"].median()) if n else float("nan")

        summary_rows.append({
            "failure_type": ftype,
            "n_runs": n,
            "spike_detect_rate": spike_rate,
            "median_spike_lag_s": spike_lag_med,
            "sust_detect_rate": sust_rate,
            "median_sust_lag_s": sust_lag_med,
            "median_false_alarms_spm": fa_med,
            "median_post_elev_frac": post_med,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["failure_type"])
    summary_df.to_csv(outdir / "summary_by_type.csv", index=False)

    print(f"[run_offline_uav_all] discovered: {len(csv_paths)}")
    print(f"[run_offline_uav_all] included (strict): {len(per_run_df)}")
    print(f"[run_offline_uav_all] wrote: {outdir / 'coverage.csv'}")
    print(f"[run_offline_uav_all] wrote: {outdir / 'per_run.csv'}")
    print(f"[run_offline_uav_all] wrote: {outdir / 'summary_by_type.csv'}")


if __name__ == "__main__":
    main()
