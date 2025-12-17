#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ORDER_DEFAULT = [
    "no_failure",
    "engine_failure",
    "elevator_failure",
    "aileron_failure",
    "rudder_failure",
    "multi_fault",
]


def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot HTM-State ALFA UAV summary figure (Figure 1).")
    p.add_argument("--summary", type=str, default="results/uav_sweep/summary_by_type.csv")
    p.add_argument("--per-run", type=str, default="results/uav_sweep/per_run.csv")
    p.add_argument("--out", type=str, default="results/uav_sweep/figure1_summary.png")
    p.add_argument("--title", type=str, default="HTM-State Benchmark on ALFA UAV Failures (Strict, Unsupervised)")
    p.add_argument(
        "--order",
        type=str,
        default=",".join(ORDER_DEFAULT),
        help="Comma-separated failure_type order for plotting.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    per_run_path = Path(args.per_run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    per_run = pd.read_csv(per_run_path)

    order: List[str] = [s.strip() for s in args.order.split(",") if s.strip()]

    # Ensure expected columns exist (fail fast with a readable message)
    needed_summary_cols = {
        "failure_type",
        "n_runs",
        "spike_detect_rate",
        "median_spike_lag_s",
        "sust_detect_rate",
        "median_sust_lag_s",
        "median_false_alarms_spm",
        "median_post_elev_frac",
    }
    missing = sorted(list(needed_summary_cols - set(summary.columns)))
    if missing:
        raise ValueError(f"summary_by_type.csv missing columns: {missing}")

    needed_per_run_cols = {"failure_type", "spike_detected", "spike_lag_s"}
    missing = sorted(list(needed_per_run_cols - set(per_run.columns)))
    if missing:
        raise ValueError(f"per_run.csv missing columns: {missing}")

    # Normalize failure_type categories
    summary["failure_type"] = summary["failure_type"].astype(str)
    per_run["failure_type"] = per_run["failure_type"].astype(str)

    # Reindex summary to desired order (keep any extras at end)
    extras = [ft for ft in summary["failure_type"].unique().tolist() if ft not in order]
    plot_order = order + sorted(extras)

    summary_idx = summary.set_index("failure_type").reindex(plot_order)
    n_runs = summary_idx["n_runs"].astype(float)

    # -------- Panel A: detection rates --------
    spike_rate = summary_idx["spike_detect_rate"].astype(float)
    sust_rate = summary_idx["sust_detect_rate"].astype(float)

    # For no_failure, rates are not meaningful; keep as NaN so they don't plot
    # (Your summary file might already have NaN; this is a safety net.)
    for ft in ["no_failure"]:
        if ft in summary_idx.index:
            spike_rate.loc[ft] = np.nan
            sust_rate.loc[ft] = np.nan

    # -------- Panel B: lag distribution (spike lag) --------
    # Use per-run rows where spike_detected True and spike_lag_s is finite
    per_run_lag = per_run.copy()
    # some CSVs may store booleans as strings; normalize
    if per_run_lag["spike_detected"].dtype == object:
        per_run_lag["spike_detected"] = per_run_lag["spike_detected"].astype(str).str.lower().isin(["true", "1", "yes"])
    per_run_lag["spike_lag_s"] = pd.to_numeric(per_run_lag["spike_lag_s"], errors="coerce")

    per_run_lag = per_run_lag[(per_run_lag["spike_detected"] == True) & (per_run_lag["spike_lag_s"].notna())]
    # exclude no_failure from lag plot
    per_run_lag = per_run_lag[per_run_lag["failure_type"] != "no_failure"]

    lag_groups = []
    lag_labels = []
    for ft in plot_order:
        if ft == "no_failure":
            continue
        g = per_run_lag.loc[per_run_lag["failure_type"] == ft, "spike_lag_s"].values
        if len(g) == 0:
            continue
        lag_groups.append(g)
        lag_labels.append(ft)

    # -------- Panel C: persistence vs false alarms --------
    fa = summary_idx["median_false_alarms_spm"].astype(float)
    post = summary_idx["median_post_elev_frac"].astype(float)

    # Exclude no_failure from persistence scatter (post_elev_frac not meaningful)
    scatter_mask = summary_idx.index != "no_failure"
    fa_sc = fa[scatter_mask]
    post_sc = post[scatter_mask]
    n_sc = n_runs[scatter_mask]

    # -------- Plot layout --------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    fig.suptitle(args.title, fontsize=16)

    # Panel A
    x = np.arange(len(plot_order))
    width = 0.38
    axA.bar(x - width / 2, spike_rate.values, width=width, label="Spike detect rate")
    axA.bar(x + width / 2, sust_rate.values, width=width, label="Sustained detect rate")
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("Detection rate")
    axA.set_title("(A) Detection performance by failure type")
    axA.set_xticks(x)
    axA.set_xticklabels(plot_order, rotation=25, ha="right")
    axA.legend(loc="upper right", frameon=True)

    # Slightly mute the no_failure category (grey label + faint band)
    if "no_failure" in plot_order:
        i0 = plot_order.index("no_failure")
        axA.axvspan(i0 - 0.5, i0 + 0.5, color="0.9", alpha=0.35, zorder=0)
        ticklabels = axA.get_xticklabels()
        if 0 <= i0 < len(ticklabels):
            ticklabels[i0].set_color("0.55")

    # Annotate n_runs above each category
    for i, ft in enumerate(plot_order):
        n = _safe_float(n_runs.loc[ft])
        if n is None:
            continue
        axA.text(i, 1.02, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    # Panel B
    if lag_groups:
        axB.boxplot(lag_groups, labels=lag_labels, showfliers=False)
        axB.set_ylabel("Lag (s)")
        axB.set_title("(B) Spike detection latency distribution")
        axB.tick_params(axis="x", rotation=25)
        # Optional reference line at 10s
        axB.axhline(10.0, linestyle="--", linewidth=1)
    else:
        axB.text(0.5, 0.5, "No detected spike lags available", ha="center", va="center")
        axB.set_axis_off()

    # Panel C
    # Point size ~ proportional to n_runs (clamped)
    sizes = []
    for ft in fa_sc.index:
        n = _safe_float(n_sc.loc[ft])
        if n is None:
            sizes.append(60.0)
        else:
            sizes.append(float(np.clip(30.0 + 25.0 * n, 50.0, 250.0)))

    axC.scatter(fa_sc.values, post_sc.values, s=np.array(sizes))
    axC.set_xlabel("Median false alarms (spikes/min, pre-boundary)")
    axC.set_ylabel("Median post-boundary elevation fraction")
    axC.set_title("(C) Persistence vs. false alarms (one point per failure type)")
    axC.set_xlim(left=0)

    # Label each point with failure type
    for ft, x0, y0 in zip(fa_sc.index.tolist(), fa_sc.values, post_sc.values):
        if np.isnan(x0) or np.isnan(y0):
            continue
        axC.text(x0, y0, f"  {ft}", va="center", fontsize=10)

    # Add simple interpretive arrows: "Compensable" -> engine, "Non-compensable" -> multi-fault
    coords = {}
    for ft, x0, y0 in zip(fa_sc.index.tolist(), fa_sc.values, post_sc.values):
        if np.isnan(x0) or np.isnan(y0):
            continue
        coords[ft] = (float(x0), float(y0))

    if "engine_failure" in coords:
        ex, ey = coords["engine_failure"]
        axC.annotate(
            "Compensable",
            xy=(ex, ey),
            xytext=(ex - 0.35, ey + 0.07),
            arrowprops=dict(arrowstyle="->", linewidth=1),
            fontsize=11,
            ha="right",
        )

    if "multi_fault" in coords:
        mx, my = coords["multi_fault"]
        axC.annotate(
            "Non-compensable",
            xy=(mx, my),
            xytext=(mx + 0.25, my + 0.06),
            arrowprops=dict(arrowstyle="->", linewidth=1),
            fontsize=11,
            ha="left",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    print(f"[plot_uav_summary] wrote: {out_path}")


if __name__ == "__main__":
    main()
