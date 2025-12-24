#!/usr/bin/env python3
"""
eval_apples_to_apples.py

Apples-to-apples evaluation logic for *sequence-level* anomaly detection on UAV runs.

This module is deliberately **algorithm-agnostic**:
- It does NOT depend on HTM, HTM-State, or any specific detector.
- The only thing it assumes is that *someone* can produce "alarm events" over time,
  e.g., from an anomaly score stream passed through a detector.

Why this exists
---------------
The paper we are matching evaluates at the **flight/run level** (N runs), not
the timestep level. Each run contributes to TP/FP/TN/FN as follows:

Inputs per run:
- toggle_step (Optional[int]): 1-based index of the failure onset step.
    - None => no-failure run
- alarm_steps (Iterable[int]): 1-based step indices where the system "announced anomaly"
  (i.e., alarm events occurred) *after running any detector logic*
- init_done_step (int): 1-based first eligible step to count alarms, reflecting stabilization
  / warmup (derived from warmup_seconds * rate_hz).

Rules (run-level confusion):
- No-failure run:
    FP if any alarm at/after init_done_step
    TN otherwise
- Failure run:
    TP if any alarm at/after max(init_done_step, toggle_step)
    FN if no alarm at/after max(init_done_step, toggle_step)
    Additionally:
    FP if any alarm in [init_done_step, toggle_step-1]  (i.e., "early" alarm before failure)

Important: a single failure run can contribute BOTH FP and FN
(e.g., early alarm only, but never alarms again at/after failure).

This module implements those rules precisely, with clear docstrings and examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -------------------------
# Core utilities (generic)
# -------------------------

def warmup_steps(rate_hz: float, warmup_seconds: float) -> int:
    """
    Convert a warmup duration in seconds into a number of steps (samples).

    We mirror the paper's notion of an initialization / stabilization period by
    excluding alarms prior to init_done_step.

    Parameters
    ----------
    rate_hz:
        Sampling rate in samples/second (Hz). Must be > 0.
    warmup_seconds:
        Warmup duration in seconds. Must be >= 0.

    Returns
    -------
    steps:
        Number of steps to treat as warmup (integer). Computed as:
            ceil(rate_hz * warmup_seconds)

    Notes
    -----
    If warmup_seconds == 0, this returns 0 steps.

    Examples
    --------
    >>> warmup_steps(25.0, 8.0)
    200
    >>> warmup_steps(20.0, 8.0)
    160
    >>> warmup_steps(25.0, 0.0)
    0
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be >= 0")

    return int(ceil(rate_hz * warmup_seconds))


def init_done_step_from_warmup(rate_hz: float, warmup_seconds: float) -> int:
    """
    Return the 1-based step index at which evaluation becomes eligible.

    If we warm up for W steps, then:
      - steps 1..W are warmup (ineligible)
      - step W+1 is the first eligible step

    Examples
    --------
    >>> init_done_step_from_warmup(25.0, 8.0)  # 200 warmup steps
    201
    >>> init_done_step_from_warmup(25.0, 0.0)
    1
    """
    w = warmup_steps(rate_hz=rate_hz, warmup_seconds=warmup_seconds)
    return 1 + w


def _sorted_unique_steps(steps: Iterable[int]) -> List[int]:
    """
    Normalize alarm steps:
    - convert to sorted list
    - drop duplicates
    - validate steps are >= 1 (since we use 1-based indexing)

    This keeps downstream logic deterministic and easy to reason about.
    """
    uniq = set()
    for s in steps:
        if s < 1:
            raise ValueError("All step indices must be 1-based and >= 1")
        uniq.add(int(s))
    return sorted(uniq)


def first_alarm_at_or_after(alarm_steps: Iterable[int], step0: int) -> Optional[int]:
    """
    Return the first alarm step >= step0, using 1-based step indices.

    Parameters
    ----------
    alarm_steps:
        Iterable of alarm step indices (1-based).
    step0:
        1-based threshold step. Must be >= 1.

    Returns
    -------
    first_step:
        The earliest alarm step >= step0, or None if none exists.

    Examples
    --------
    >>> first_alarm_at_or_after([5, 2, 10], 6)
    10
    >>> first_alarm_at_or_after([2, 5], 1)
    2
    >>> first_alarm_at_or_after([], 1) is None
    True
    """
    if step0 < 1:
        raise ValueError("step0 must be 1-based and >= 1")

    steps = _sorted_unique_steps(alarm_steps)
    for s in steps:
        if s >= step0:
            return s
    return None


def any_alarm_in_window(
    alarm_steps: Iterable[int],
    start_step: int,
    end_step_inclusive: int,
) -> bool:
    """
    Check whether there exists any alarm step in [start_step, end_step_inclusive].

    Parameters
    ----------
    alarm_steps:
        Alarm steps, 1-based.
    start_step:
        Window start step, 1-based.
    end_step_inclusive:
        Window end step, 1-based inclusive.

    Returns
    -------
    has_alarm:
        True if any alarm is in the window, else False.

    Examples
    --------
    >>> any_alarm_in_window([3, 10], 1, 5)
    True
    >>> any_alarm_in_window([10], 1, 5)
    False
    >>> any_alarm_in_window([5], 6, 9)
    False
    """
    if start_step < 1 or end_step_inclusive < 1:
        raise ValueError("Window steps must be 1-based and >= 1")
    if end_step_inclusive < start_step:
        return False

    steps = _sorted_unique_steps(alarm_steps)
    for s in steps:
        if s < start_step:
            continue
        if s > end_step_inclusive:
            break
        return True
    return False


# -------------------------
# Run-level confusion logic
# -------------------------

@dataclass(frozen=True)
class RunConfusion:
    """
    Run-level confusion contributions.

    Note: On failure runs, FP and FN can both be 1 (early alarm only).
    """
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


def confusion_for_run(
    *,
    toggle_step: Optional[int],
    alarm_steps: Iterable[int],
    init_done_step: int,
) -> RunConfusion:
    """
    Compute run-level (sequence-level) confusion contributions using apples-to-apples logic.

    Parameters
    ----------
    toggle_step:
        Failure onset step index (1-based), or None for no-failure runs.
    alarm_steps:
        1-based indices of alarm events for this run (from ANY algorithm + detector).
        These are "announcement" moments, not raw anomaly scores.
    init_done_step:
        1-based first eligible step for alarms (after warmup/stabilization).

    Returns
    -------
    RunConfusion:
        tp/fp/tn/fn contributions for this run.

    Logic (exact)
    -------------
    1) No-failure run (toggle_step is None):
       - FP if any alarm at/after init_done_step
       - TN otherwise

    2) Failure run (toggle_step exists):
       - Let post_start = max(init_done_step, toggle_step)
       - TP if any alarm at/after post_start, else FN
       - Additionally, FP if any alarm in [init_done_step, toggle_step-1] (early alarm)

    This supports the paper's described possibility that one run contributes both
    FP and FN (early alarm only, but no alarm at/after failure).

    Examples
    --------
    No-failure:
    >>> confusion_for_run(toggle_step=None, alarm_steps=[], init_done_step=1).as_dict()
    {'tp': 0, 'fp': 0, 'tn': 1, 'fn': 0}
    >>> confusion_for_run(toggle_step=None, alarm_steps=[10], init_done_step=1).as_dict()
    {'tp': 0, 'fp': 1, 'tn': 0, 'fn': 0}

    Failure, clean detection at/after failure:
    >>> confusion_for_run(toggle_step=50, alarm_steps=[60], init_done_step=1).as_dict()
    {'tp': 1, 'fp': 0, 'tn': 0, 'fn': 0}

    Failure, early alarm only (counts as FP + FN):
    >>> confusion_for_run(toggle_step=50, alarm_steps=[20], init_done_step=1).as_dict()
    {'tp': 0, 'fp': 1, 'tn': 0, 'fn': 1}

    Failure, early and later alarm (FP + TP):
    >>> confusion_for_run(toggle_step=50, alarm_steps=[20, 55], init_done_step=1).as_dict()
    {'tp': 1, 'fp': 1, 'tn': 0, 'fn': 0}

    Warmup exclusion: early alarm before init_done_step ignored
    >>> confusion_for_run(toggle_step=None, alarm_steps=[5], init_done_step=10).as_dict()
    {'tp': 0, 'fp': 0, 'tn': 1, 'fn': 0}
    """
    if init_done_step < 1:
        raise ValueError("init_done_step must be 1-based and >= 1")
    if toggle_step is not None and toggle_step < 1:
        raise ValueError("toggle_step must be 1-based and >= 1 when provided")

    steps = _sorted_unique_steps(alarm_steps)

    # Helper: any alarm at/after threshold
    def has_alarm_at_or_after(th: int) -> bool:
        return first_alarm_at_or_after(steps, th) is not None

    # No-failure run
    if toggle_step is None:
        if has_alarm_at_or_after(init_done_step):
            return RunConfusion(fp=1)
        return RunConfusion(tn=1)

    # Failure run
    post_start = max(init_done_step, toggle_step)

    tp = 1 if has_alarm_at_or_after(post_start) else 0
    fn = 1 - tp

    # "Early alarm" window: [init_done_step, toggle_step-1]
    fp = 1 if any_alarm_in_window(steps, init_done_step, toggle_step - 1) else 0

    return RunConfusion(tp=tp, fp=fp, fn=fn, tn=0)


# -------------------------
# Aggregation + metrics
# -------------------------

@dataclass(frozen=True)
class ConfusionTotals:
    """
    Totals across runs.
    """
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n_runs(self) -> int:
        return self.tp + self.fp + self.tn + self.fn  # NOTE: FP can overlap on failure runs!

    def as_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


def sum_confusions(confusions: Sequence[RunConfusion]) -> ConfusionTotals:
    """
    Sum a list of RunConfusion records.

    Note: Because FP can overlap with FN on the same run (failure run with early-only alarm),
    tp+fp+tn+fn is not necessarily equal to the number of runs.
    Therefore, the correct run count is len(confusions).

    We keep `n_runs` as a property for convenience but do not rely on it
    for accuracy calculations.

    Examples
    --------
    >>> cs = [RunConfusion(tn=1), RunConfusion(fp=1), RunConfusion(fp=1, fn=1)]
    >>> sum_confusions(cs).as_dict()
    {'tp': 0, 'fp': 2, 'tn': 1, 'fn': 1}
    """
    tp = sum(c.tp for c in confusions)
    fp = sum(c.fp for c in confusions)
    tn = sum(c.tn for c in confusions)
    fn = sum(c.fn for c in confusions)
    return ConfusionTotals(tp=tp, fp=fp, tn=tn, fn=fn)


@dataclass(frozen=True)
class SummaryMetrics:
    """
    Standard metrics computed from run-level confusion totals.

    Precision and recall follow the usual definitions:
      precision = TP / (TP + FP)
      recall    = TP / (TP + FN)

    Accuracy is computed over runs as:
      accuracy = (TP_runs + TN_runs) / N_runs

    Here, TP_runs is the count of *failure runs* with a post-failure alarm.
    TN_runs is the count of *no-failure runs* with no alarms.

    IMPORTANT:
    FP totals may include "early FP" on failure runs and thus can overlap with FN.
    This does not affect precision/recall formulas, which use the totals as defined
    by the paper-style counting.
    """
    n_runs: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    accuracy: Optional[float]


def compute_metrics(
    totals: ConfusionTotals,
    *,
    n_runs: int,
) -> SummaryMetrics:
    """
    Compute precision, recall, and accuracy from totals.

    Parameters
    ----------
    totals:
        Summed TP/FP/TN/FN.
    n_runs:
        Number of runs evaluated (len(per_run_confusions)). Required explicitly
        because totals may "double count" runs in FP due to overlap.

    Returns
    -------
    SummaryMetrics

    Examples
    --------
    >>> totals = ConfusionTotals(tp=10, fp=2, tn=8, fn=2)
    >>> m = compute_metrics(totals, n_runs=22)
    >>> (round(m.precision, 4), round(m.recall, 4), round(m.accuracy, 4))
    (0.8333, 0.8333, 0.8182)
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    tp, fp, tn, fn = totals.tp, totals.fp, totals.tn, totals.fn

    precision = None if (tp + fp) == 0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0 else tp / (tp + fn)
    accuracy = (tp + tn) / n_runs if n_runs > 0 else None

    return SummaryMetrics(
        n_runs=n_runs,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
    )


# -------------------------
# Convenience: end-to-end
# -------------------------

@dataclass(frozen=True)
class RunEvalResult:
    """
    A single run's evaluation record.

    Fields are meant to be human-readable and easy to log/serialize.
    """
    run_id: str
    toggle_step: Optional[int]
    init_done_step: int
    first_alarm_any: Optional[int]
    first_alarm_post: Optional[int]
    has_early_alarm: bool
    confusion: RunConfusion


def evaluate_run(
    *,
    run_id: str,
    toggle_step: Optional[int],
    alarm_steps: Iterable[int],
    init_done_step: int,
) -> RunEvalResult:
    """
    Evaluate one run, returning both the confusion contribution and helpful diagnostics.

    Diagnostics included:
    - first_alarm_any: first alarm >= init_done_step
    - first_alarm_post: first alarm >= max(init_done_step, toggle_step) (failure runs)
    - has_early_alarm: any alarm in [init_done_step, toggle_step-1] (failure runs)

    This is useful for debugging and for generating an auditable per-run report.
    """
    steps = _sorted_unique_steps(alarm_steps)
    first_any = first_alarm_at_or_after(steps, init_done_step)

    if toggle_step is None:
        first_post = None
        has_early = False
    else:
        first_post = first_alarm_at_or_after(steps, max(init_done_step, toggle_step))
        has_early = any_alarm_in_window(steps, init_done_step, toggle_step - 1)

    conf = confusion_for_run(
        toggle_step=toggle_step,
        alarm_steps=steps,
        init_done_step=init_done_step,
    )

    return RunEvalResult(
        run_id=run_id,
        toggle_step=toggle_step,
        init_done_step=init_done_step,
        first_alarm_any=first_any,
        first_alarm_post=first_post,
        has_early_alarm=has_early,
        confusion=conf,
    )


def evaluate_runs(
    *,
    runs: Sequence[Tuple[str, Optional[int], Iterable[int]]],
    init_done_step: int,
) -> Tuple[List[RunEvalResult], SummaryMetrics]:
    """
    Evaluate a collection of runs.

    Parameters
    ----------
    runs:
        Sequence of tuples: (run_id, toggle_step, alarm_steps)
    init_done_step:
        Shared init_done_step used across runs (apples-to-apples requirement).

    Returns
    -------
    per_run_results, summary_metrics
    """
    per_run: List[RunEvalResult] = [
        evaluate_run(run_id=rid, toggle_step=t, alarm_steps=alarms, init_done_step=init_done_step)
        for rid, t, alarms in runs
    ]
    totals = sum_confusions([r.confusion for r in per_run])
    summary = compute_metrics(totals, n_runs=len(per_run))
    return per_run, summary


if __name__ == "__main__":
    # Minimal sanity demo (not a CLI yet; keeping this file "library-first").
    demo_runs = [
        ("no_fail_clean", None, []),
        ("no_fail_fp", None, [250]),
        ("fail_tp", 300, [320]),
        ("fail_early_only_fp_fn", 300, [200]),
        ("fail_early_and_tp_fp", 300, [200, 310]),
    ]

    init_done = 201  # e.g., rate_hz=25, warmup_seconds=8 => warmup_steps=200 => init_done_step=201
    per_run, summary = evaluate_runs(runs=demo_runs, init_done_step=init_done)

    print("Per-run:")
    for r in per_run:
        print(
            f"- {r.run_id}: toggle={r.toggle_step} init_done={r.init_done_step} "
            f"first_any={r.first_alarm_any} first_post={r.first_alarm_post} "
            f"early={r.has_early_alarm} confusion={r.confusion.as_dict()}"
        )

    print("\nSummary:")
    print(summary)
