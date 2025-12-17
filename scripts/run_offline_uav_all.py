from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.offline_demo_uav import parse_args as offline_parse_args
from scripts.offline_demo_uav import evaluate_uav_csv


def classify_failure(name: str) -> str:
    n = name.lower()

    if "no_ground_truth" in n:
        return "no_ground_truth"
    if "no_failure" in n:
        return "no_failure"

    has_engine = "engine_failure" in n
    has_elevator = "elevator_failure" in n
    has_rudder = "rudder" in n and "failure" in n
    has_aileron = "aileron" in n and "failure" in n

    # multi-fault if >1 fault family appears
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
    p = argparse.ArgumentParser("Offline sweep over generated ALFA UAV CSV streams")
    p.add_argument("--generated-dir", type=str, default="demos/uav_demo/generated", help="Directory containing per-run CSVs.")
    p.add_argument("--outdir", type=str, default="results/uav_sweep", help="Output directory for result tables.")
    p.add_argument("--glob", type=str, default="**/*.csv", help="Glob pattern under generated-dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(gen_dir.glob(args.glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {gen_dir} with glob {args.glob!r}")

    # Reuse offline_demo_uav's default args as the evaluation config baseline
    offline_args = offline_parse_args()

    per_run_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        run_id = csv_path.stem
        failure_type = classify_failure(run_id)

        res = evaluate_uav_csv(csv_path, offline_args)

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
            "has_boundary": res["has_boundary"],
            "boundary_time_s": res["boundary_time_s"],
            "spike_detected": res["spike_detected"],
            "spike_lag_s": res["spike_lag_s"],
            "sustained_detected": res["sustained_detected"],
            "sustained_lag_s": res["sustained_lag_s"],
            "false_alarms_spm": res["false_alarms_spm"],
            "post_elev_frac": res["post_elev_frac"],
            "n_spikes_total": res["n_spikes_total"],
        })

    coverage_df = pd.DataFrame(coverage_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    coverage_df.to_csv(outdir / "coverage.csv", index=False)
    per_run_df.to_csv(outdir / "per_run.csv", index=False)

    # Summary by type
    def _rate(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else float("nan")

    # Only compute lag medians over detected runs
    summary_rows = []
    for ftype, g in per_run_df.groupby("failure_type", dropna=False):
        n = len(g)

        spike_rate = _rate(g["spike_detected"].astype(float)) if "spike_detected" in g else float("nan")
        sust_rate = _rate(g["sustained_detected"].astype(float)) if "sustained_detected" in g else float("nan")

        spike_lag_med = float(g.loc[g["spike_detected"] == True, "spike_lag_s"].median()) if (g["spike_detected"] == True).any() else float("nan")
        sust_lag_med = float(g.loc[g["sustained_detected"] == True, "sustained_lag_s"].median()) if (g["sustained_detected"] == True).any() else float("nan")

        fa_med = float(g["false_alarms_spm"].median()) if "false_alarms_spm" in g else float("nan")
        post_med = float(g["post_elev_frac"].median()) if "post_elev_frac" in g else float("nan")

        summary_rows.append({
            "failure_type": ftype,
            "n_runs": int(n),
            "spike_detect_rate": spike_rate,
            "median_spike_lag_s": spike_lag_med,
            "sust_detect_rate": sust_rate,
            "median_sust_lag_s": sust_lag_med,
            "median_false_alarms_spm": fa_med,
            "median_post_elev_frac": post_med,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["failure_type"])
    summary_df.to_csv(outdir / "summary_by_type.csv", index=False)

    print(f"[run_offline_uav_all] wrote: {outdir / 'coverage.csv'}")
    print(f"[run_offline_uav_all] wrote: {outdir / 'per_run.csv'}")
    print(f"[run_offline_uav_all] wrote: {outdir / 'summary_by_type.csv'}")
    print(f"[run_offline_uav_all] included runs: {len(per_run_df)} / discovered: {len(csv_paths)}")


if __name__ == "__main__":
    main()
