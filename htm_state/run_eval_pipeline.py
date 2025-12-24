#!/usr/bin/env python3
"""
run_eval_pipeline.py

End-to-end, *algorithm-agnostic* evaluation pipeline for a single UAV run.

This module composes three layers, each intentionally kept simple and auditable:

Layer 1: "Apples-to-apples" run-level evaluation rules
  - eval_apples_to_apples.py
  - Consumes:
      toggle_step (Optional[int])
      init_done_step (int; from warmup_seconds * rate_hz)
      alarm_steps (list[int], 1-based)

Layer 2: Alarm event construction (still algorithm-agnostic)
  - alarm_events.py
  - Consumes:
      spike_event[t], persistence_event[t] boolean streams (optional)
      alarm_mode in {"spike","persistence","or","and"}
  - Produces:
      alarm_event[t] and alarm_steps (1-based)

Layer 3: HTM-State event adapter (detector adapter; algorithm-agnostic input)
  - htm_state_event_adapter.py
  - Consumes:
      anomaly_scores[t] (float stream from any algorithm)
      EMA/clipping config (StateEngineConfig)
      detector configs (SpikeDetectorConfig and/or PersistenceDetectorConfig)
  - Produces:
      spike_event[t], persistence_event[t] boolean streams
      plus diagnostic traces (state, debug, etc.)

Why this file exists
--------------------
This is the single canonical entry point we can point the paper authors to:
- It shows, top-to-bottom, how "announcement times" are derived and scored.
- It makes *no assumptions* about HTM specifically.
- It returns enough diagnostics to audit decisions and debug.

Important conventions
---------------------
- anomaly_scores and event streams are 0-based arrays where index 0 is step 1.
- alarm_steps are 1-based step indices, suitable for eval_apples_to_apples.

This module does NOT load ALFA data; it only evaluates a given run once you
provide the anomaly score stream and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .alarm_events import AlarmMode, combine_alarm_events, event_stream_to_alarm_steps
from .eval_apples_to_apples import RunEvalResult, evaluate_run, init_done_step_from_warmup
from .event_adapter import anomaly_scores_to_event_traces
from .engine import StateEngineConfig
from .spike_detector import SpikeDetectorConfig
from .persistence_detector import PersistenceDetectorConfig


@dataclass(frozen=True)
class RunEvalDiagnostics:
    """
    Diagnostics returned alongside the run-level evaluation result.

    These are intended for:
      - auditability (show exactly what the system did)
      - debugging (why did it alarm / not alarm?)
      - plotting (state + events vs time)

    Attributes
    ----------
    rate_hz:
        Sampling rate used to derive warmup steps.
    warmup_seconds:
        Warmup duration in seconds used to derive init_done_step.
    init_done_step:
        First eligible 1-based step after warmup.
    alarm_mode:
        How spike/persistence event streams were combined into the final alarm event stream.
    state_trace:
        anomaly (post-clip) and EMA state signals.
    detector_traces:
        spike_event and/or persistence_event streams (bool), plus optional debug dicts.
    combined_events:
        alarm_event stream (bool) produced by combining detector streams per alarm_mode.
    alarm_steps:
        1-based steps where alarm_event is True (computed from combined_events).
    notes:
        Free-form notes useful for provenance/logging.
    """
    rate_hz: float
    warmup_seconds: float
    init_done_step: int
    alarm_mode: AlarmMode
    state_trace: StateTrace
    detector_traces: DetectorEventTraces
    combined_events: CombinedEventStreams
    alarm_steps: List[int]
    notes: Dict[str, object]


def evaluate_run_end_to_end(
    *,
    run_id: str,
    toggle_step: Optional[int],
    anomaly_scores: Sequence[float],
    rate_hz: float,
    warmup_seconds: float,
    alarm_mode: AlarmMode,
    # State construction
    state_cfg: Optional[StateEngineConfig] = None,
    # Detector configs (provide at least the ones your alarm_mode needs)
    spike_cfg: Optional[SpikeDetectorConfig] = None,
    persistence_cfg: Optional[PersistenceDetectorConfig] = None,
    # Debug controls
    include_detector_debug: bool = False,
    notes: Optional[Dict[str, object]] = None,
) -> tuple[RunEvalResult, RunEvalDiagnostics]:
    """
    Evaluate a single run end-to-end, from anomaly scores to run-level confusion.

    Parameters
    ----------
    run_id:
        Identifier for the run/flight (e.g., "run_003").
    toggle_step:
        Failure onset step (1-based), or None for no-failure runs.
    anomaly_scores:
        Per-timestep anomaly score values from any algorithm (HTM, baseline, etc.).
        Length T. Index 0 corresponds to step 1.
    rate_hz:
        Sampling rate (Hz). Used only to compute init_done_step from warmup_seconds.
    warmup_seconds:
        Duration to exclude at the start of the run (stabilization). Converted via:
          warmup_steps = ceil(rate_hz * warmup_seconds)
          init_done_step = warmup_steps + 1
    alarm_mode:
        How to define "announcement times" from detector event streams.
        One of: "spike", "persistence", "or", "and".

    state_cfg:
        Controls EMA and optional clipping. If None, uses StateEngineConfig defaults.
    spike_cfg:
        Spike detector configuration. Required if alarm_mode is "spike" or includes spike ("or"/"and").
    persistence_cfg:
        Persistence detector configuration. Required if alarm_mode is "persistence" or includes persistence.
    include_detector_debug:
        If True, include per-step debug dicts from detector update() calls in diagnostics.
    notes:
        Optional free-form metadata you want carried along (e.g., algorithm name, hyperparams).

    Returns
    -------
    run_eval, diagnostics

    run_eval:
        RunEvalResult from eval_apples_to_apples.evaluate_run(), containing:
          - confusion contribution
          - first_alarm_any / first_alarm_post / early flag
          - init_done_step and toggle_step for auditability
    diagnostics:
        RunEvalDiagnostics containing intermediate signals and event streams.

    Raises
    ------
    ValueError:
        If alarm_mode requires event streams whose configs are not provided.

    Examples
    --------
    No-failure run with a late persistent alarm:
    >>> scores = [0.0]*100 + [5.0]*10
    >>> res, diag = evaluate_run_end_to_end(
    ...     run_id="nf",
    ...     toggle_step=None,
    ...     anomaly_scores=scores,
    ...     rate_hz=25.0,
    ...     warmup_seconds=0.0,
    ...     alarm_mode="persistence",
    ...     persistence_cfg=PersistenceDetectorConfig(baseline_window=20, threshold_sigma=0.0, sustain_count=3),
    ... )
    >>> res.confusion.fp in (0, 1)  # depends on detector parameters, but should be well-defined
    True

    Failure run where an early alarm occurs but no post-failure alarm:
    (This can legitimately yield FP=1 and FN=1 per our apples-to-apples rules.)
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be >= 0")

    # Ensure required detector configs are present for the chosen alarm_mode.
    if alarm_mode in ("spike", "or", "and") and spike_cfg is None:
        raise ValueError(f"alarm_mode='{alarm_mode}' requires spike_cfg")
    if alarm_mode in ("persistence", "or", "and") and persistence_cfg is None:
        raise ValueError(f"alarm_mode='{alarm_mode}' requires persistence_cfg")

    init_done_step = init_done_step_from_warmup(rate_hz=rate_hz, warmup_seconds=warmup_seconds)

    # 1) Convert anomaly_scores → EMA state → detector event streams
    state_trace, detector_traces = anomaly_scores_to_event_traces(
        anomaly_scores,
        state_cfg=state_cfg,
        spike_cfg=spike_cfg,
        persistence_cfg=persistence_cfg,
        include_debug=include_detector_debug,
    )

    # 2) Combine event streams per alarm_mode → alarm_event stream
    combined = combine_alarm_events(
        mode=alarm_mode,
        spike_event=detector_traces.spike_event,
        persistence_event=detector_traces.persistence_event,
    )

    # 3) Convert alarm_event stream → alarm_steps (1-based)
    alarm_steps = event_stream_to_alarm_steps(combined.alarm_event)

    # 4) Evaluate run-level confusion + diagnostics (pure, apples-to-apples logic)
    run_eval = evaluate_run(
        run_id=run_id,
        toggle_step=toggle_step,
        alarm_steps=alarm_steps,
        init_done_step=init_done_step,
    )

    diag = RunEvalDiagnostics(
        rate_hz=float(rate_hz),
        warmup_seconds=float(warmup_seconds),
        init_done_step=int(init_done_step),
        alarm_mode=alarm_mode,
        state_trace=state_trace,
        detector_traces=detector_traces,
        combined_events=combined,
        alarm_steps=alarm_steps,
        notes=notes or {},
    )

    return run_eval, diag


if __name__ == "__main__":
    # Minimal demo run (kept intentionally simple and readable).
    # This does not claim any particular performance; it simply demonstrates the pipeline.
    scores = [0.0] * 200 + [0.1] * 50 + [3.0] * 50 + [0.1] * 50

    res, diag = evaluate_run_end_to_end(
        run_id="demo_fail",
        toggle_step=280,  # 1-based
        anomaly_scores=scores,
        rate_hz=25.0,
        warmup_seconds=8.0,
        alarm_mode="persistence",
        state_cfg=StateEngineConfig(ema_alpha=0.1),
        persistence_cfg=PersistenceDetectorConfig(
            baseline_window=100,
            threshold_sigma=3.0,
            sustain_count=10,
        ),
        # spike_cfg omitted since we chose alarm_mode="persistence"
        include_detector_debug=False,
        notes={"algorithm": "demo_scores"},
    )

    print("Run eval:", res)
    print("Confusion:", res.confusion.as_dict())
    print("Alarm steps (first 10):", diag.alarm_steps[:10])
    print("init_done_step:", diag.init_done_step)
