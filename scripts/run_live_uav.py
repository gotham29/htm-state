from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


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
    p = argparse.ArgumentParser("Generate representative live plots from ALFA UAV sweep results.")
    p.add_argument("--per-run", type=str, default="results/uav_sweep/per_run.csv")
    p.add_argument("--coverage", type=str, default="results/uav_sweep/coverage.csv")
    p.add_argument("--outdir", type=str, default="results/uav_sweep/figures")
    p.add_argument("--sleep", type=float, default=0.0, help="Passed to live_demo_uav.py --sleep (0 = as fast as possible).")
    p.add_argument("--dpi", type=int, default=200, help="Passed to live_demo_uav.py --save-fig-dpi.")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated failure types to plot (e.g. engine_failure,rudder_failure). Default: all.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess calls (default: current).",
    )
    return p.parse_args()


def pick_representatives(per_run: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """
    Deterministic picks (aligned with the story you want in demo_uav.md):
      - typical_spike: detected spike lag closest to median (within failure type)
      - hard_spike: detected spike lag near 90th percentile (or max if tiny n)
      - miss: neither spike nor sustained detected (lowest false_alarms_spm)
      - sustained_only: (elevator) sustained detected but spike not detected
    """
    out: Dict[str, List[Tuple[str, str]]] = {}

    df = per_run.copy()
    df["spike_detected"] = df["spike_detected"].apply(_as_bool)
    df["sustained_detected"] = df["sustained_detected"].apply(_as_bool)
    df["spike_lag_s"] = pd.to_numeric(df["spike_lag_s"], errors="coerce")
    df["sustained_lag_s"] = pd.to_numeric(df["sustained_lag_s"], errors="coerce")
    df["false_alarms_spm"] = pd.to_numeric(df["false_alarms_spm"], errors="coerce")

    for ftype in sorted(df["failure_type"].astype(str).unique().tolist()):
        g = df[df["failure_type"] == ftype].copy()
        picks: List[Tuple[str, str]] = []

        # typical/hard based on spike lag when available
        if g["spike_detected"].any():
            det = g[g["spike_detected"] & g["spike_lag_s"].notna()].copy()
            if len(det):
                med = float(det["spike_lag_s"].median())
                det["absdiff"] = (det["spike_lag_s"] - med).abs()
                typical = det.sort_values(["absdiff", "spike_lag_s"]).iloc[0]
                picks.append(("typical_spike", str(typical["run_id"])))

                if len(det) > 1:
                    q = float(det["spike_lag_s"].quantile(0.9))
                    det["absq"] = (det["spike_lag_s"] - q).abs()
                    hard = det.sort_values(["absq", "spike_lag_s"]).iloc[0]
                    if str(hard["run_id"]) == str(typical["run_id"]):
                        hard = det.sort_values("spike_lag_s", ascending=False).iloc[0]
                    if str(hard["run_id"]) != str(typical["run_id"]):
                        picks.append(("hard_spike", str(hard["run_id"])))

        # add elevator sustained-only contrast if present
        if ftype == "elevator_failure":
            alt = g[(~g["spike_detected"]) & (g["sustained_detected"])].copy()
            if len(alt):
                picks.append(("sustained_only", str(alt.iloc[0]["run_id"])))

        # add a clean miss if any
        miss = g[(~g["spike_detected"]) & (~g["sustained_detected"])].copy()
        if len(miss):
            miss = miss.sort_values(["false_alarms_spm", "run_id"])
            picks.append(("miss", str(miss.iloc[0]["run_id"])))

        # dedupe while preserving order
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for label, rid in picks:
            if rid in seen:
                continue
            uniq.append((label, rid))
            seen.add(rid)

        out[ftype] = uniq

    return out


def main() -> None:
    args = parse_args()
    per_run_path = Path(args.per_run)
    coverage_path = Path(args.coverage)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(per_run_path)
    coverage = pd.read_csv(coverage_path)

    # Build run_id -> csv_path map (coverage includes excluded runs too)
    run_to_csv: Dict[str, str] = dict(zip(coverage["run_id"].astype(str), coverage["csv_path"].astype(str)))

    picks = pick_representatives(per_run)

    # Add a single baseline no_failure run for visualization (even though strict excludes it)
    # Choose a stable-looking one by name (deterministic).
    baseline_id = "carbonZ_2018-10-18-11-08-24_no_failure"
    if baseline_id in run_to_csv:
        existing = picks.get("no_failure", [])
        # Put baseline first so the gallery reads nicely.
        picks["no_failure"] = [("baseline", baseline_id)] + [
            (lbl, rid) for (lbl, rid) in existing if rid != baseline_id
        ]

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    if only:
        picks = {k: v for k, v in picks.items() if k in set(only)}

    selected_rows: List[Dict[str, str]] = []

    live_script = (Path(__file__).resolve().parent / "live_demo_uav.py").resolve()
    if not live_script.exists():
        raise FileNotFoundError(f"Expected live demo script at: {live_script}")

    n_calls = 0
    for ftype, items in picks.items():
        for label, run_id in items:
            csv_path = run_to_csv.get(run_id)
            if not csv_path:
                print(f"[run_live_demo_uav_selected] SKIP (no csv_path): {run_id}")
                continue

            ft_dir = outdir / ftype
            ft_dir.mkdir(parents=True, exist_ok=True)
            out_png = ft_dir / f"{run_id}__{label}.png"

            cmd = [
                args.python,
                str(live_script),
                "--csv",
                str(csv_path),
                "--save-fig",
                str(out_png),
                "--save-fig-dpi",
                str(int(args.dpi)),
                "--sleep",
                str(float(args.sleep)),
            ]

            print(f"[run_live_demo_uav_selected] RUN: {run_id} ({ftype}/{label})")
            print("  " + " ".join(cmd))
            subprocess.check_call(cmd)
            n_calls += 1

            selected_rows.append(
                {
                    "failure_type": ftype,
                    "label": label,
                    "run_id": run_id,
                    "csv_path": str(csv_path),
                    "figure_path": str(out_png),
                }
            )

    sel_df = pd.DataFrame(selected_rows)
    sel_out = outdir.parent / "selected_runs.csv"
    sel_df.to_csv(sel_out, index=False)

    print(f"[run_live_demo_uav_selected] figures written under: {outdir}")
    print(f"[run_live_demo_uav_selected] wrote: {sel_out}")
    print(f"[run_live_demo_uav_selected] total runs plotted: {n_calls}")


if __name__ == "__main__":
    main()
