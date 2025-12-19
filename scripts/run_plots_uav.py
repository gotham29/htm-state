#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Render final (non-animated) UAV demo plots for *all* runs using demo_live_uav.py as a fast renderer."
    )
    p.add_argument(
        "--coverage",
        type=str,
        default="demos/uav/generated/results/uav_sweep/coverage.csv",
        help="coverage.csv produced by run_offline_uav_all.py",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="demos/uav/generated/figures/all",
        help="Root directory to write per-run PNGs.",
    )
    p.add_argument(
        "--only-included",
        action="store_true",
        help="If set, render only runs where coverage.included == True (recommended).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Passed to demo_live_uav.py --save-fig-dpi",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess calls (default: current).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coverage_path = Path(args.coverage)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cov = pd.read_csv(coverage_path)

    # Normalize included to bool (it may be stored as strings)
    if "included" in cov.columns and cov["included"].dtype == object:
        cov["included"] = cov["included"].astype(str).str.lower().isin(["true", "1", "yes"])

    if args.only_included and "included" in cov.columns:
        cov = cov[cov["included"] == True].copy()

    # Build run_id -> (csv_path, failure_type)
    run_to_csv: Dict[str, Path] = {}
    run_to_ft: Dict[str, str] = {}
    for rid, pth, ft in zip(
        cov["run_id"].astype(str),
        cov["csv_path"].astype(str),
        cov["failure_type"].astype(str),
    ):
        p = Path(pth)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        run_to_csv[rid] = p
        run_to_ft[rid] = ft

    demo_script = (Path(__file__).resolve().parent.parent / "demos" / "uav" / "demo_live_uav.py").resolve()
    if not demo_script.exists():
        # fallback if scripts live next to demo_live_uav.py
        demo_script = (Path(__file__).resolve().parent / "demo_live_uav.py").resolve()
    if not demo_script.exists():
        raise FileNotFoundError(f"Could not find demo_live_uav.py at expected path(s).")

    n = 0
    failed = 0
    for rid in sorted(run_to_csv.keys()):
        csv_path = run_to_csv[rid]
        ft = run_to_ft.get(rid, "unknown")

        ft_dir = outdir / ft
        ft_dir.mkdir(parents=True, exist_ok=True)
        out_png = ft_dir / f"{rid}.png"

        if out_png.exists():
            print(f"[render_offline_uav_plots] skip exists: {out_png}")
            continue

        cmd = [
            args.python,
            str(demo_script),
            "--render-final",
            "--no-show",
            "--csv",
            str(csv_path),
            "--save-fig",
            str(out_png),
            "--save-fig-dpi",
            str(int(args.dpi)),
        ]

        print(f"[render_offline_uav_plots] RUN: {rid} -> {out_png}")
        try:
            subprocess.check_call(cmd)
            n += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"[render_offline_uav_plots] FAILED: {rid} ({e})\n  cmd={' '.join(cmd)}")


    print(f"[render_offline_uav_plots] wrote {n} plots under: {outdir}")


if __name__ == "__main__":
    main()
