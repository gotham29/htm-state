#!/usr/bin/env python3
"""
eval_alfa_batch.py

Batch, apples-to-apples evaluation over the ALFA UAV runs specified by coverage.csv.

This script is *auditable by construction*:

1) Run list comes from coverage.csv (run_id, csv_path, failure_type, included).
2) Failure onset (toggle_step) is derived ONLY from the run CSV itself:
      toggle_step = first timestep where is_boundary == 1  (1-based)
   If no boundary exists, toggle_step=None (no-failure run).

3) HTM anomaly scores are generated from raw features using the same HTMSession
   pathway used in demo_offline_uav.py (via score_alfa_csv_with_htm()).
4) Apples-to-apples FP/TP/FN/TN accounting is computed using evaluate_run_end_to_end()
   with the chosen alarm_mode(s): persistence, spike, or, and.

Outputs
-------
Writes two CSVs:

A) Per-run ledger (one row per run per mode):
   <outdir>/per_run.csv

B) Summary-by-mode:
   <outdir>/summary_by_mode.csv

(Optionally) Summary-by-failure_type:
   <outdir>/summary_by_failure_type.csv

Notes
-----
- Baseline policy for this batch: HTM learns within each run (per-run training).
- No attempt is made here to train on held-out no-failure runs; that is Step 6/7.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from htm_state.alfa_htm_scorer import HTMScoringConfig, score_alfa_csv_with_htm
from htm_state.engine import StateEngineConfig
from htm_state.persistence_detector import PersistenceDetectorConfig
from htm_state.spike_detector import SpikeDetectorConfig
from htm_state.run_eval_pipeline import evaluate_run_end_to_end


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch apples-to-apples evaluation for ALFA UAV runs via coverage.csv.")
    p.add_argument(
        "--coverage",
        type=str,
        default="demos/uav/generated/results/uav_sweep/coverage.csv",
        help="Path to coverage.csv (run list + csv_path + included).",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repo root used to resolve relative csv_path entries from coverage.csv.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="demos/uav/generated/results/alfa_batch",
        help="Directory to write output CSVs.",
    )
    p.add_argument(
        "--only-included",
        action="store_true",
        help="If set, evaluate only rows where coverage.included == True (recommended).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="persistence,spike,or,and",
        help="Comma-separated alarm modes to evaluate: persistence,spike,or,and",
    )

    # Data / scoring
    p.add_argument("--rate-hz", type=float, default=25.0)
    p.add_argument("--warmup-seconds", type=float, default=8.0)
    p.add_argument(
        "--features",
        type=str,
        default="airspeed,climb,altitude,throttle,heading,pitch,roll,yaw",
        help="Comma-separated feature columns to feed to HTM.",
    )

    # HTM scoring knobs (match demo defaults)
    p.add_argument("--enc-n", type=int, default=64)
    p.add_argument("--enc-w", type=int, default=8)
    p.add_argument("--htm-ema-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--freeze-tm-after-boundary",
        action="store_true",
        help="If set, HTM learns pre-boundary then stops learning at/after boundary.",
    )

    # State engine smoothing (post-score, pre-detector)
    p.add_argument("--ema-alpha", type=float, default=0.1)

    # Spike detector (if used by mode)
    p.add_argument("--spike-recent", type=int, default=10)
    p.add_argument("--spike-prior", type=int, default=50)
    p.add_argument("--spike-threshold-pct", type=float, default=50.0)
    p.add_argument("--spike-min-sep", type=int, default=1)
    p.add_argument("--spike-min-delta", type=float, default=0.0)

    # Persistence detector (if used by mode)
    p.add_argument("--persist-baseline", type=int, default=200)
    p.add_argument("--persist-k-mad", type=float, default=3.0)
    p.add_argument("--persist-hold", type=int, default=10)
    p.add_argument("--persist-min-sep", type=int, default=1)

    p.add_argument(
        "--write-failure-type-summary",
        action="store_true",
        help="If set, write summary_by_failure_type.csv.",
    )
    return p.parse_args()


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_csv_path(repo_root: Path, csv_path: str) -> Path:
    p = Path(csv_path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _per_run_row(
    *,
    run_id: str,
    csv_path: str,
    failure_type: str,
    included: bool,
    mode: str,
    rate_hz: float,
    warmup_seconds: float,
    init_done_step: int,
    toggle_step: Optional[int],
    first_alarm_any: Optional[int],
    first_alarm_post: Optional[int],
    early_alarm: bool,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    n_alarm_steps: int,
    n_spike_events: Optional[int],
    n_persistence_events: Optional[int],
) -> Dict[str, object]:
    is_failure = toggle_step is not None

    lag_steps: Optional[int] = None
    lag_seconds: Optional[float] = None
    if is_failure and tp == 1 and first_alarm_post is not None and toggle_step is not None:
        lag_steps = int(first_alarm_post) - int(toggle_step)
        lag_seconds = float(lag_steps) / float(rate_hz)

    return {
        "run_id": run_id,
        "csv_path": csv_path,
        "failure_type": failure_type,
        "included": bool(included),
        "mode": mode,
        "is_failure": bool(is_failure),
        "rate_hz": float(rate_hz),
        "warmup_seconds": float(warmup_seconds),
        "init_done_step": int(init_done_step),
        "toggle_step": (int(toggle_step) if toggle_step is not None else None),
        "first_alarm_any": (int(first_alarm_any) if first_alarm_any is not None else None),
        "first_alarm_post": (int(first_alarm_post) if first_alarm_post is not None else None),
        "early_alarm": bool(early_alarm),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_alarm_steps": int(n_alarm_steps),
        "n_detector_events_spike": (int(n_spike_events) if n_spike_events is not None else None),
        "n_detector_events_persistence": (
            int(n_persistence_events) if n_persistence_events is not None else None
        ),
        "lag_steps": (int(lag_steps) if lag_steps is not None else None),
        "lag_seconds": (float(lag_seconds) if lag_seconds is not None else None),
    }


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    modes = _parse_list(args.modes)
    allowed = {"persistence", "spike", "or", "and"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Invalid mode(s) {bad}. Allowed: {sorted(allowed)}")

    feature_names = _parse_list(args.features)
    if not feature_names:
        raise ValueError("--features must include at least one feature column")

    coverage_path = _resolve_csv_path(repo_root, args.coverage)
    cov = pd.read_csv(coverage_path)

    required_cols = {"run_id", "csv_path", "failure_type", "included", "exclude_reason"}
    missing_cols = required_cols - set(cov.columns)
    if missing_cols:
        raise ValueError(f"coverage.csv missing columns: {sorted(missing_cols)}. Found: {list(cov.columns)}")

    if args.only_included:
        cov = cov[cov["included"].astype(bool) == True].copy()  # noqa: E712

    if len(cov) == 0:
        raise ValueError("No runs to evaluate after filtering (check --only-included and coverage.csv).")

    # Shared configs used by evaluate_run_end_to_end
    state_cfg = StateEngineConfig(ema_alpha=float(args.ema_alpha))

    spike_cfg = SpikeDetectorConfig(
        recent_window=int(args.spike_recent),
        prior_window=int(args.spike_prior),
        threshold_pct=float(args.spike_threshold_pct),
        edge_only=True,
        min_separation=int(args.spike_min_sep),
        min_delta=float(args.spike_min_delta),
        eps=1e-3,
    )
    persistence_cfg = PersistenceDetectorConfig(
        baseline_window=int(args.persist_baseline),
        k_mad=float(args.persist_k_mad),
        hold_steps=int(args.persist_hold),
        edge_only=True,
        min_separation=int(args.persist_min_sep),
    )

    per_run_rows: List[Dict[str, object]] = []

    # Iterate runs
    for _, row in cov.iterrows():
        run_id = str(row["run_id"])
        csv_path_raw = str(row["csv_path"])
        failure_type = str(row["failure_type"])
        included = bool(row["included"])

        run_csv_path = _resolve_csv_path(repo_root, csv_path_raw)
        if not run_csv_path.exists():
            raise FileNotFoundError(f"Run CSV not found for run_id={run_id}: {run_csv_path}")

        # 1) Score once per run (HTM)
        anomaly_scores, _mwl_states, boundary_step = score_alfa_csv_with_htm(
            str(run_csv_path),
            HTMScoringConfig(
                feature_names=list(feature_names),
                rate_hz=float(args.rate_hz),
                enc_n_per_feature=int(args.enc_n),
                enc_w_per_feature=int(args.enc_w),
                anomaly_ema_alpha=float(args.htm_ema_alpha),
                seed=int(args.seed),
                sp_learning=False,
            ),
            learn=True,
            freeze_tm_after_boundary=bool(args.freeze_tm_after_boundary),
        )

        toggle_step = boundary_step  # derived solely from is_boundary

        # 2) Evaluate each mode for this run (same anomaly_scores, different alarm definition)
        for mode in modes:
            use_spike = mode in ("spike", "or", "and")
            use_pers = mode in ("persistence", "or", "and")

            res, diag = evaluate_run_end_to_end(
                run_id=run_id,
                toggle_step=toggle_step,
                anomaly_scores=anomaly_scores,
                rate_hz=float(args.rate_hz),
                warmup_seconds=float(args.warmup_seconds),
                alarm_mode=mode,  # type: ignore[arg-type]
                state_cfg=state_cfg,
                spike_cfg=spike_cfg if use_spike else None,
                persistence_cfg=persistence_cfg if use_pers else None,
                include_detector_debug=False,  # batch: keep it light
                notes={
                    "csv_path": str(run_csv_path),
                    "failure_type": failure_type,
                    "included": included,
                },
            )

            # counts at/after init_done (diagnostic)
            init_done = diag.init_done_step
            # diag.alarm_steps are 1-based indices already
            n_alarm_steps = len([s for s in diag.alarm_steps if s >= init_done])

            n_spike_events: Optional[int] = None
            n_persistence_events: Optional[int] = None
            if use_spike and diag.detector_traces.spike_event:
                n_spike_events = sum(diag.detector_traces.spike_event[init_done - 1 :])
            if use_pers and diag.detector_traces.persistence_event:
                n_persistence_events = sum(diag.detector_traces.persistence_event[init_done - 1 :])

            per_run_rows.append(
                _per_run_row(
                    run_id=run_id,
                    csv_path=str(run_csv_path),
                    failure_type=failure_type,
                    included=included,
                    mode=mode,
                    rate_hz=float(args.rate_hz),
                    warmup_seconds=float(args.warmup_seconds),
                    init_done_step=int(init_done),
                    toggle_step=toggle_step,
                    first_alarm_any=res.first_alarm_any,
                    first_alarm_post=res.first_alarm_post,
                    early_alarm=res.has_early_alarm,
                    tp=res.confusion.tp,
                    fp=res.confusion.fp,
                    tn=res.confusion.tn,
                    fn=res.confusion.fn,
                    n_alarm_steps=int(n_alarm_steps),
                    n_spike_events=n_spike_events,
                    n_persistence_events=n_persistence_events,
                )
            )

    # ---- Write per-run ledger (Table A) ----
    per_run_df = pd.DataFrame(per_run_rows)
    per_run_csv = outdir / "per_run.csv"
    per_run_df.to_csv(per_run_csv, index=False)

    # ---- Summary by mode (Table B) ----
    def _summarize(group: pd.DataFrame) -> pd.Series:
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        tn = int(group["tn"].sum())
        fn = int(group["fn"].sum())
        n_runs = int(group["run_id"].nunique())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        accuracy = ((tp + tn) / n_runs) if n_runs > 0 else 0.0

        # Lag stats over TP runs
        lags = group.loc[group["lag_seconds"].notna(), "lag_seconds"].astype(float)
        if len(lags) > 0:
            mean_lag = float(lags.mean())
            med_lag = float(lags.median())
            p90_lag = float(lags.quantile(0.90))
            n_lag = int(lags.shape[0])
        else:
            mean_lag = med_lag = p90_lag = float("nan")
            n_lag = 0

        n_failure_runs = int(group["is_failure"].sum())  # counts rows; but one row per run per mode
        # Convert to unique run counts
        n_failure_runs = int(group[group["is_failure"] == True]["run_id"].nunique())  # noqa: E712
        n_no_failure_runs = int(group[group["is_failure"] == False]["run_id"].nunique())  # noqa: E712

        return pd.Series(
            {
                "n_runs": n_runs,
                "n_failure_runs": n_failure_runs,
                "n_no_failure_runs": n_no_failure_runs,
                "tp_total": tp,
                "fp_total": fp,
                "tn_total": tn,
                "fn_total": fn,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "tp_with_lag_n": n_lag,
                "mean_lag_seconds": mean_lag,
                "median_lag_seconds": med_lag,
                "p90_lag_seconds": p90_lag,
            }
        )

    summary_by_mode = per_run_df.groupby("mode", as_index=False).apply(_summarize).reset_index(drop=True)
    summary_csv = outdir / "summary_by_mode.csv"
    summary_by_mode.to_csv(summary_csv, index=False)

    # ---- Optional: Summary by failure_type (Table C) ----
    if args.write_failure_type_summary:
        summary_by_ft = (
            per_run_df.groupby(["failure_type", "mode"], as_index=False)
            .apply(_summarize)
            .reset_index(drop=True)
        )
        ft_csv = outdir / "summary_by_failure_type.csv"
        summary_by_ft.to_csv(ft_csv, index=False)

    print(f"Wrote: {per_run_csv}")
    print(f"Wrote: {summary_csv}")
    if args.write_failure_type_summary:
        print(f"Wrote: {outdir / 'summary_by_failure_type.csv'}")


if __name__ == "__main__":
    main()
