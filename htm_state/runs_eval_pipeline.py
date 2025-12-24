#!/usr/bin/env python3
"""
runs_eval_pipeline.py

Equally-auditable *multi-run* wrapper around the single-run end-to-end evaluation pipeline.

This module is intentionally "thin glue" that:
- calls evaluate_run_end_to_end() per run
- aggregates run-level confusion into precision/recall/accuracy (per the paper-style rules)
- returns:
    1) a per-run table of key fields (easy to print / save / plot)
    2) aggregate SummaryMetrics

It remains algorithm-agnostic: the only required per-run input is anomaly_scores[t].
If you can produce anomaly scores for each run from some model, you can evaluate it here.

What this does NOT do
---------------------
- It does not load ALFA data from disk.
- It does not generate plots.
- It does not decide detector parameters.
Those will be separate steps so we can keep this pipeline definition stable.

Indexing conventions
--------------------
- anomaly_scores is 0-based Python list/array length T (index 0 => step 1).
- toggle_step and init_done_step are 1-based indices.
- alarm_steps are 1-based indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .alarm_events import AlarmMode
from .engine import StateEngineConfig
from .persistence_detector import PersistenceDetectorConfig
from .spike_detector import SpikeDetectorConfig

from .eval_apples_to_apples import SummaryMetrics, sum_confusions, compute_metrics
from .run_eval_pipeline import RunEvalDiagnostics, evaluate_run_end_to_end, RunEvalResult


# -------------------------
# Inputs + outputs
# -------------------------

@dataclass(frozen=True)
class RunInput:
    """
    Minimal per-run inputs needed to evaluate a run.

    Attributes
    ----------
    run_id:
        Unique run identifier.
    toggle_step:
        1-based failure onset step, or None for no-failure runs.
    anomaly_scores:
        Per-step anomaly scores from any algorithm (length T).
    rate_hz:
        Sampling rate in Hz for this run. Used only to compute init_done_step.
    meta:
        Optional metadata (e.g., dataset split, chosen features, run duration, etc.).
    """
    run_id: str
    toggle_step: Optional[int]
    anomaly_scores: Sequence[float]
    rate_hz: float
    meta: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class PerRunRow:
    """
    Compact, human-readable per-run summary for tables/logging.

    This is meant to be the *first* artifact you look at when something seems off,
    before you open plots or deep diagnostics.

    Notes on fields
    ---------------
    - first_alarm_any: first alarm at/after init_done_step (paper-style "announcement")
    - first_alarm_post: first alarm at/after max(init_done_step, toggle_step) for failure runs
    - early_alarm: True if there was any alarm between init_done_step and toggle_step-1
                  (can produce an FP even on failure runs)
    """
    run_id: str
    toggle_step: Optional[int]
    rate_hz: float
    warmup_seconds: float
    init_done_step: int
    t_len: int
    alarm_mode: AlarmMode
    n_alarm_steps: int
    first_alarm_any: Optional[int]
    first_alarm_post: Optional[int]
    early_alarm: bool
    tp: int
    fp: int
    tn: int
    fn: int
    meta: Dict[str, object]


@dataclass(frozen=True)
class RunsEvaluation:
    """
    Full evaluation output for a collection of runs.

    Attributes
    ----------
    per_run:
        List of PerRunRow records (table-ready).
    run_results:
        Full per-run RunEvalResult objects (includes confusion object and first alarm fields).
    diagnostics:
        Optional full per-run diagnostics (state trace, event streams, alarm steps).
        This can be large, so include only if requested via include_diagnostics=True.
    summary:
        Aggregate SummaryMetrics (precision/recall/accuracy) computed over runs.
    totals:
        Aggregate TP/FP/TN/FN totals (note FP can overlap with FN on some failure runs).
    """
    per_run: List[PerRunRow]
    run_results: List[RunEvalResult]
    diagnostics: Optional[Dict[str, RunEvalDiagnostics]]
    summary: SummaryMetrics
    totals: Dict[str, int]


# -------------------------
# Multi-run evaluation
# -------------------------

def evaluate_runs_end_to_end(
    *,
    runs: Sequence[RunInput],
    warmup_seconds: float,
    alarm_mode: AlarmMode,
    # State construction
    state_cfg: Optional[StateEngineConfig] = None,
    # Detector configs (provide at least what's required by alarm_mode)
    spike_cfg: Optional[SpikeDetectorConfig] = None,
    persistence_cfg: Optional[PersistenceDetectorConfig] = None,
    # Debug/diagnostics controls
    include_detector_debug: bool = False,
    include_diagnostics: bool = False,
    shared_notes: Optional[Dict[str, object]] = None,
) -> RunsEvaluation:
    """
    Evaluate a set of runs end-to-end and compute aggregate metrics.

    Parameters
    ----------
    runs:
        Sequence of RunInput records (run_id, toggle_step, anomaly_scores, rate_hz, meta).
    warmup_seconds:
        Stabilization duration in seconds used to derive init_done_step per run.
        We use per-run rate_hz, but the same warmup_seconds across all runs is the
        apples-to-apples requirement.
    alarm_mode:
        How to combine detector event streams into alarm events:
        "spike" | "persistence" | "or" | "and"
    state_cfg:
        StateEngineConfig controlling EMA + clipping.
        If None, defaults from engine.py are used.
    spike_cfg:
        SpikeDetectorConfig. Required if alarm_mode includes spike ("spike", "or", "and").
    persistence_cfg:
        PersistenceDetectorConfig. Required if alarm_mode includes persistence
        ("persistence", "or", "and").
    include_detector_debug:
        If True, retain per-step detector debug dicts in diagnostics (large).
    include_diagnostics:
        If True, return full per-run RunEvalDiagnostics (state trace, event streams, alarm steps).
        If False, diagnostics is returned as None to keep outputs light.
    shared_notes:
        Optional metadata applied to all runs (e.g., algorithm name, commit hash).

    Returns
    -------
    RunsEvaluation:
        Includes a per-run table, aggregate metrics, and optional full diagnostics.

    Auditability guarantees
    -----------------------
    - Uses evaluate_run_end_to_end() as the single canonical run-level implementation.
    - Per-run rows preserve:
        - init_done_step (derived from warmup_seconds and each run's rate_hz)
        - first alarm times
        - early alarm flag
        - TP/FP/TN/FN contributions for that run
    - Aggregate metrics use eval_apples_to_apples.compute_metrics() with n_runs=len(runs).

    Failure mode note (FP overlap)
    ------------------------------
    On failure runs, a run can contribute both FP and FN (early alarm only). This is expected
    under the paper-matched rules. Therefore:
      - totals['tp'] + totals['fp'] + totals['tn'] + totals['fn'] is not necessarily == N_runs
      - Accuracy is computed as (TP_runs + TN_runs) / N_runs (handled in compute_metrics).

    Examples
    --------
    >>> runs = [
    ...   RunInput(run_id="a", toggle_step=None, anomaly_scores=[0.0]*10, rate_hz=25.0),
    ...   RunInput(run_id="b", toggle_step=7, anomaly_scores=[0.0]*6 + [5.0]*4, rate_hz=25.0),
    ... ]
    >>> out = evaluate_runs_end_to_end(
    ...   runs=runs,
    ...   warmup_seconds=0.0,
    ...   alarm_mode="persistence",
    ...   persistence_cfg=PersistenceDetectorConfig(baseline_window=3, threshold_sigma=0.0, sustain_count=2),
    ... )
    >>> out.summary.n_runs
    2
    """
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be >= 0")
    if len(runs) == 0:
        raise ValueError("runs must be non-empty")

    per_run_rows: List[PerRunRow] = []
    run_results: List[RunEvalResult] = []
    diagnostics_map: Optional[Dict[str, RunEvalDiagnostics]] = {} if include_diagnostics else None

    # Evaluate each run using the single canonical implementation.
    for r in runs:
        notes: Dict[str, object] = {}
        if shared_notes:
            notes.update(shared_notes)
        if r.meta:
            # Keep per-run metadata distinct, but include it in notes for provenance.
            notes.update({f"meta.{k}": v for k, v in r.meta.items()})

        run_eval, diag = evaluate_run_end_to_end(
            run_id=r.run_id,
            toggle_step=r.toggle_step,
            anomaly_scores=r.anomaly_scores,
            rate_hz=r.rate_hz,
            warmup_seconds=warmup_seconds,
            alarm_mode=alarm_mode,
            state_cfg=state_cfg,
            spike_cfg=spike_cfg,
            persistence_cfg=persistence_cfg,
            include_detector_debug=include_detector_debug,
            notes=notes,
        )

        run_results.append(run_eval)

        if include_diagnostics:
            assert diagnostics_map is not None
            diagnostics_map[r.run_id] = diag

        # Compact per-run row for tables
        c = run_eval.confusion
        per_run_rows.append(
            PerRunRow(
                run_id=r.run_id,
                toggle_step=r.toggle_step,
                rate_hz=float(r.rate_hz),
                warmup_seconds=float(warmup_seconds),
                init_done_step=int(diag.init_done_step),
                t_len=len(r.anomaly_scores),
                alarm_mode=alarm_mode,
                n_alarm_steps=len(diag.alarm_steps),
                first_alarm_any=run_eval.first_alarm_any,
                first_alarm_post=run_eval.first_alarm_post,
                early_alarm=run_eval.has_early_alarm,
                tp=c.tp,
                fp=c.fp,
                tn=c.tn,
                fn=c.fn,
                meta=r.meta or {},
            )
        )

    # Aggregate confusion + compute metrics (IMPORTANT: n_runs = len(runs))
    totals_obj = sum_confusions([rr.confusion for rr in run_results])
    summary = compute_metrics(totals_obj, n_runs=len(runs))

    totals = {
        "tp": totals_obj.tp,
        "fp": totals_obj.fp,
        "tn": totals_obj.tn,
        "fn": totals_obj.fn,
        "n_runs": len(runs),
    }

    return RunsEvaluation(
        per_run=per_run_rows,
        run_results=run_results,
        diagnostics=diagnostics_map,
        summary=summary,
        totals=totals,
    )


# -------------------------
# Helpers for printing/logging (optional)
# -------------------------

def per_run_rows_as_dicts(rows: Sequence[PerRunRow]) -> List[Dict[str, object]]:
    """
    Convert PerRunRow records into dicts (useful for JSON/CSV export).
    """
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "run_id": r.run_id,
                "toggle_step": r.toggle_step,
                "rate_hz": r.rate_hz,
                "warmup_seconds": r.warmup_seconds,
                "init_done_step": r.init_done_step,
                "t_len": r.t_len,
                "alarm_mode": r.alarm_mode,
                "n_alarm_steps": r.n_alarm_steps,
                "first_alarm_any": r.first_alarm_any,
                "first_alarm_post": r.first_alarm_post,
                "early_alarm": r.early_alarm,
                "tp": r.tp,
                "fp": r.fp,
                "tn": r.tn,
                "fn": r.fn,
                "meta": r.meta,
            }
        )
    return out


if __name__ == "__main__":
    # Minimal smoke demo (not a CLI).
    # This simply exercises the pipeline; it does not claim paper-level performance.
    demo_runs = [
        RunInput(run_id="nf_clean", toggle_step=None, anomaly_scores=[0.0] * 300, rate_hz=25.0),
        RunInput(run_id="nf_fp", toggle_step=None, anomaly_scores=[0.0] * 200 + [5.0] * 100, rate_hz=25.0),
        RunInput(run_id="fail_tp", toggle_step=250, anomaly_scores=[0.0] * 260 + [5.0] * 40, rate_hz=25.0),
    ]

    out = evaluate_runs_end_to_end(
        runs=demo_runs,
        warmup_seconds=8.0,
        alarm_mode="persistence",
        state_cfg=StateEngineConfig(ema_alpha=0.1),
        persistence_cfg=PersistenceDetectorConfig(baseline_window=80, threshold_sigma=3.0, sustain_count=10),
        include_diagnostics=False,
        shared_notes={"algorithm": "demo_scores"},
    )

    print("Totals:", out.totals)
    print("Summary:", out.summary)
    print("Per-run rows:")
    for row in out.per_run:
        print(row)
