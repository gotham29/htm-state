from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys, copy, re
import pandas as pd


# Allow running as: python scripts/run_offline_uav_all.py
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_offline_uav import evaluate_uav_csv, parse_args as offline_parse_args


# Keep ALFA-22 definition in sync with scripts/alfa_apples_to_apples.py
ALFA_22_SEQ_IDS: set[str] = {
    "2018-07-18-15-53-31_1",
    "2018-07-18-15-53-31_2",
    "2018-07-18-16-37-39_1",
    "2018-07-30-16-39-00_1",
    "2018-07-30-16-39-00_2",
    "2018-07-30-16-39-00_3",
    "2018-09-11-11-56-30",
    "2018-09-11-14-16-55",
    "2018-09-11-14-22-07_1",
    "2018-09-11-14-22-07_2",
    "2018-09-11-14-41-38",
    "2018-09-11-14-41-51",
    "2018-09-11-14-52-54",
    "2018-09-11-15-05-11_1",
    "2018-09-11-15-05-11_2",
    "2018-09-11-15-06-34_1",
    "2018-09-11-15-06-34_2",
    "2018-09-11-15-06-34_3",
    "2018-09-11-17-27-13_1",
    "2018-09-11-17-27-13_2",
    "2018-09-11-17-55-30_1",
    "2018-09-11-17-55-30_2",
}


def _extract_seq_id(run_id: str) -> str | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:_\d+)?", str(run_id))
    return m.group(0) if m else None


def _strip_tag_suffix(run_id: str) -> str:
    """
    Our generated artifacts sometimes suffix run IDs with a visualization/tag label:
        2018-07-30-17-36-35__typical_spike
    Classification should be based on the underlying run identity, so strip __... .
    """
    return run_id.split("__", 1)[0]


def classify_failure(run_id: str) -> str:
    base = _strip_tag_suffix(run_id)
    n = base.lower()

    if "no_ground_truth" in n:
        return "no_ground_truth"
    if "no_failure" in n:
        return "no_failure"

    # ALFA sometimes describes engine faults as "full_power_loss"
    has_engine = ("engine_failure" in n) or ("full_power_loss" in n) or ("power_loss" in n)
    has_elevator = "elevator_failure" in n
    has_rudder = ("rudder" in n and "failure" in n)
    has_aileron = ("aileron" in n and "failure" in n)

    families = sum([has_engine, has_elevator, has_rudder, has_aileron])
    if families >= 2:
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


def failure_type_from_path(csv_path: Path, streams_root: Path) -> str:
    """
    Prefer directory label if streams are laid out as:
        generated/streams/<failure_type>/*.csv
    Fallback to name-based parsing if files are flat in streams_root.
    """
    p = csv_path.parent
    if p == streams_root:
        return classify_failure(csv_path.stem)
    # If immediate parent is a label folder, trust it.
    label = p.name.lower()
    allowed = {
        "engine_failure",
        "elevator_failure",
        "rudder_failure",
        "aileron_failure",
        "multi_fault",
        "no_failure",
        "no_ground_truth",
    }
    if label in allowed:
        return label
    # Sometimes you might have deeper nesting; try the first folder under streams_root.
    try:
        rel = csv_path.relative_to(streams_root)
        if len(rel.parts) >= 2:
            label2 = rel.parts[0].lower()
            if label2 in allowed:
                return label2
    except Exception:
        pass
    return classify_failure(csv_path.stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Offline sweep over generated ALFA UAV streams (strict benchmark)")
    p.add_argument("--generated-dir", type=str, default="demos/uav/generated/streams")
    p.add_argument("--outdir", type=str, default="demos/uav/generated/results/uav_sweep")
    p.add_argument("--glob", type=str, default="*.csv")
    p.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "alfa22"],
        help="If alfa22, only include dagger-marked ALFA-22 sequences in per_run + summary outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    streams_root = gen_dir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Support both:
    #   generated/*.csv
    #   generated/<failure_type>/*.csv
    if "**" in args.glob:
        csv_paths = sorted(gen_dir.rglob("*.csv"))
    else:
        csv_paths = sorted(gen_dir.rglob(args.glob))
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
        if args.subset == "alfa22":
            sid = _extract_seq_id(run_id)
            if sid is None or sid not in ALFA_22_SEQ_IDS:
                continue
        failure_type = failure_type_from_path(csv_path, streams_root)

        # For ALFA apples-to-apples metrics (accuracy/precision/recall),
        # we must include "no_failure" sequences as negative examples even though
        # they have no injection boundary. We therefore evaluate them with
        # strict_boundary disabled, but keep strict for all failure runs.
        if failure_type == "no_failure":
            args_nf = copy.copy(offline_args)
            args_nf.strict_boundary = False
            res = evaluate_uav_csv(csv_path, args_nf)
            # ensure boundary fields are empty for no-failure
            res["boundary_time_s"] = None
        else:
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
            "max_state_post_boundary": res.get("max_state_post_boundary", None),
            "n_spikes_total": res.get("n_spikes_total", None),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    coverage_df.to_csv(outdir / "coverage.csv", index=False)
    per_run_df.to_csv(outdir / "per_run.csv", index=False)

    # If nothing was included, write what we have and exit gracefully with diagnostics.
    if per_run_df.empty:
        print(f"[run_offline_uav_all] discovered: {len(csv_paths)}")
        print(f"[run_offline_uav_all] subset: {args.subset}")
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
        max_post_med = float(g["max_state_post_boundary"].median()) if ("max_state_post_boundary" in g.columns and n) else float("nan")

        summary_rows.append({
            "failure_type": ftype,
            "n_runs": n,
            "spike_detect_rate": spike_rate,
            "median_spike_lag_s": spike_lag_med,
            "sust_detect_rate": sust_rate,
            "median_sust_lag_s": sust_lag_med,
            "median_false_alarms_spm": fa_med,
            "median_post_elev_frac": post_med,
            "median_max_state_post_boundary": max_post_med,
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
