#!/usr/bin/env python3
"""
htm_state_event_adapter.py

Thin adapter layer that bridges:

  anomaly_score[t]  (from *any* algorithm)
        ↓ (optional clipping + EMA smoothing)
     state[t]       (a continuous "state" signal)
        ↓ (HTM-State detector(s))
  spike_event[t], persistence_event[t]  (boolean event streams)
        ↓ (alarm_events.py can combine these into alarm_steps)
  alarm_steps (1-based indices)

Design goals
------------
1) **Algorithm-agnostic input**:
   - This module accepts a *scalar anomaly score stream* per timestep.
   - That stream could come from HTM, a baseline model, an LSTM, etc.

2) **Apples-to-apples clarity**:
   - The paper's evaluation is driven by "announcement times".
   - Here, "announcement times" are represented by detector event streams:
       - spike_event[t] or persistence_event[t]
   - Downstream, we convert events → alarm_steps → per-run confusion using
     eval_apples_to_apples.py.

3) **Thin + auditable**:
   - We do not invent new logic; we reuse the existing HTM-State components:
       - StateEngine EMA logic (matching engine.py semantics)
       - SpikeDetector (spike_detector.py)
       - PersistenceDetector (persistence_detector.py)
   - We keep interfaces explicit and return diagnostic traces to support auditing.

Indexing conventions
--------------------
- Input anomaly_scores is a Python sequence indexed 0..T-1, where index 0 is step 1.
- Output event streams are also 0-based sequences aligned with anomaly_scores.
- When you want alarm_steps for evaluation, use alarm_events.event_stream_to_alarm_steps().

Notes
-----
- This module intentionally does NOT know about toggle_step or warmup logic.
  Warmup exclusion is handled in eval_apples_to_apples.py by init_done_step.
- This module returns *event streams*, not alarm_steps, so you can combine events
  (spike/persistence OR/AND) outside this module in a single canonical place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .engine import StateEngineConfig  # existing HTM-State config dataclass
from .spike_detector import SpikeDetector, SpikeDetectorConfig
from .persistence_detector import PersistenceDetector, PersistenceDetectorConfig


# -------------------------
# State construction (EMA)
# -------------------------

@dataclass(frozen=True)
class StateTrace:
    """
    Diagnostic trace of anomaly→state conversion.

    Attributes
    ----------
    anomaly:
        The anomaly value after clipping (if enabled), aligned 1:1 with input length.
    state:
        EMA-smoothed state signal aligned 1:1 with anomaly.
    """
    anomaly: List[float]
    state: List[float]


def anomaly_to_state_trace(
    anomaly_scores: Sequence[float],
    *,
    state_cfg: Optional[StateEngineConfig] = None,
) -> StateTrace:
    """
    Convert a raw anomaly score stream into a continuous 'state' stream.

    This mirrors the logic in engine.py:StateEngine.step() for EMA smoothing:
      - optionally clip anomaly
      - state[0] = anomaly[0]
      - state[t] = a*anomaly[t] + (1-a)*state[t-1] for t>=1

    Parameters
    ----------
    anomaly_scores:
        Sequence of raw anomaly scores (length T).
    state_cfg:
        Optional StateEngineConfig controlling ema_alpha and clipping.
        If None, defaults are used (ema_alpha=0.05, no clipping), matching engine.py defaults.

    Returns
    -------
    StateTrace:
        anomaly (post-clip) and EMA state.

    Examples
    --------
    >>> tr = anomaly_to_state_trace([1.0, 0.0, 0.0], state_cfg=StateEngineConfig(ema_alpha=0.5))
    >>> tr.anomaly
    [1.0, 0.0, 0.0]
    >>> [round(x, 3) for x in tr.state]
    [1.0, 0.5, 0.25]
    """
    if state_cfg is None:
        state_cfg = StateEngineConfig()  # defaults from engine.py

    a = float(state_cfg.ema_alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("state_cfg.ema_alpha must be in (0, 1].")

    anomaly_out: List[float] = []
    state_out: List[float] = []

    def clip(x: float) -> float:
        if state_cfg.clip_min is not None and x < state_cfg.clip_min:
            x = float(state_cfg.clip_min)
        if state_cfg.clip_max is not None and x > state_cfg.clip_max:
            x = float(state_cfg.clip_max)
        return x

    initialized = False
    prev_state = 0.0

    for x in anomaly_scores:
        ax = clip(float(x))
        anomaly_out.append(ax)

        if not initialized:
            s = ax
            initialized = True
        else:
            s = a * ax + (1.0 - a) * prev_state

        state_out.append(s)
        prev_state = s

    return StateTrace(anomaly=anomaly_out, state=state_out)


# -------------------------
# Detector runners (events)
# -------------------------

@dataclass(frozen=True)
class DetectorEventTraces:
    """
    Diagnostic traces produced by running detectors over a state stream.

    Attributes
    ----------
    spike_event:
        Boolean stream aligned with state (0-based; index 0 is step 1). None if spike detector not run.
    persistence_event:
        Boolean stream aligned with state (0-based). None if persistence detector not run.
    spike_debug:
        Optional per-step debug dicts from SpikeDetector.update(). Present only if include_debug=True.
    persistence_debug:
        Optional per-step debug dicts from PersistenceDetector.update(). Present only if include_debug=True.
    """
    spike_event: Optional[List[bool]]
    persistence_event: Optional[List[bool]]
    spike_debug: Optional[List[Dict[str, object]]]
    persistence_debug: Optional[List[Dict[str, object]]]


def run_spike_detector(
    state: Sequence[float],
    *,
    cfg: SpikeDetectorConfig,
    include_debug: bool = False,
) -> Tuple[List[bool], Optional[List[Dict[str, object]]]]:
    """
    Run HTM-State SpikeDetector over a state stream.

    Parameters
    ----------
    state:
        Continuous state signal (length T).
    cfg:
        SpikeDetectorConfig from spike_detector.py.
    include_debug:
        If True, return the full dict returned by SpikeDetector.update() per step.

    Returns
    -------
    spike_event, debug
        spike_event is a list[bool] of length T where True means the detector fired a spike.
        debug is None unless include_debug=True.

    Notes
    -----
    SpikeDetector.update() returns a dict with key "spike" as 0/1.
    We convert that to bool for downstream consistency.

    Examples
    --------
    >>> ev, _ = run_spike_detector([0.0, 0.0, 10.0, 10.0], cfg=SpikeDetectorConfig(recent_window=1, prior_window=2, threshold_pct=50.0))
    >>> len(ev)
    4
    """
    det = SpikeDetector(cfg)
    events: List[bool] = []
    debug: Optional[List[Dict[str, object]]] = [] if include_debug else None

    for v in state:
        out = det.update(float(v))
        fired = bool(int(out.get("spike", 0)) == 1)
        events.append(fired)
        if include_debug:
            assert debug is not None
            debug.append(out)

    return events, debug


def run_persistence_detector(
    state: Sequence[float],
    *,
    cfg: PersistenceDetectorConfig,
    include_debug: bool = False,
) -> Tuple[List[bool], Optional[List[Dict[str, object]]]]:
    """
    Run HTM-State PersistenceDetector over a state stream.

    Parameters
    ----------
    state:
        Continuous state signal (length T).
    cfg:
        PersistenceDetectorConfig from persistence_detector.py.
    include_debug:
        If True, return the dict returned by PersistenceDetector.update() per step.

    Returns
    -------
    persistence_event, debug
        persistence_event is a list[bool] of length T where True means the detector fired
        a sustained/persistent event at that step.
        debug is None unless include_debug=True.

    Notes
    -----
    PersistenceDetector.update() returns a dict with keys:
      - "persistent" (bool)
      - "sustained"  (bool)
    and returns both spellings explicitly. We read "persistent".

    Examples
    --------
    >>> cfg = PersistenceDetectorConfig(
    ...     baseline_window=3,
    ...     k_mad=0.0,        # threshold = median + 0 * MAD => median
    ...     hold_steps=2,
    ... )
    >>> ev, _ = run_persistence_detector([0.0, 1.0, 2.0, 2.0, 2.0], cfg=cfg)
    >>> len(ev)
    5
    """
    det = PersistenceDetector(cfg)
    events: List[bool] = []
    debug: Optional[List[Dict[str, object]]] = [] if include_debug else None

    for v in state:
        out = det.update(float(v))
        fired = bool(out.get("persistent", False))
        events.append(fired)
        if include_debug:
            assert debug is not None
            debug.append(out)

    return events, debug


def anomaly_scores_to_event_traces(
    anomaly_scores: Sequence[float],
    *,
    state_cfg: Optional[StateEngineConfig] = None,
    spike_cfg: Optional[SpikeDetectorConfig] = None,
    persistence_cfg: Optional[PersistenceDetectorConfig] = None,
    include_debug: bool = False,
) -> Tuple[StateTrace, DetectorEventTraces]:
    """
    End-to-end adapter: anomaly_scores → (anomaly,state) → (spike_event,persistence_event).

    Parameters
    ----------
    anomaly_scores:
        Raw anomaly score stream (length T).
    state_cfg:
        Controls EMA + clipping. If None, uses StateEngineConfig defaults.
    spike_cfg:
        If provided, run SpikeDetector and return spike_event.
    persistence_cfg:
        If provided, run PersistenceDetector and return persistence_event.
    include_debug:
        If True, include per-step debug payloads from detector updates.

    Returns
    -------
    state_trace, detector_traces

    Typical use
    -----------
    1) Choose/compute anomaly_scores from your algorithm.
    2) Call this function to get event streams.
    3) Combine/convert events to alarm_steps using alarm_events.py.
    4) Evaluate with eval_apples_to_apples.py.

    Examples
    --------
    >>> scores = [0.0, 0.1, 0.2, 5.0, 5.0]
    >>> st, det = anomaly_scores_to_event_traces(scores, spike_cfg=SpikeDetectorConfig(), persistence_cfg=None)
    >>> len(st.state) == len(scores) == len(det.spike_event)
    True
    """
    if spike_cfg is None and persistence_cfg is None:
        raise ValueError("Provide at least one of spike_cfg or persistence_cfg to produce events.")

    st = anomaly_to_state_trace(anomaly_scores, state_cfg=state_cfg)

    spike_event: Optional[List[bool]] = None
    persistence_event: Optional[List[bool]] = None
    spike_debug: Optional[List[Dict[str, object]]] = None
    persistence_debug: Optional[List[Dict[str, object]]] = None

    if spike_cfg is not None:
        spike_event, spike_debug = run_spike_detector(st.state, cfg=spike_cfg, include_debug=include_debug)

    if persistence_cfg is not None:
        persistence_event, persistence_debug = run_persistence_detector(
            st.state, cfg=persistence_cfg, include_debug=include_debug
        )

    det_tr = DetectorEventTraces(
        spike_event=spike_event,
        persistence_event=persistence_event,
        spike_debug=spike_debug if include_debug else None,
        persistence_debug=persistence_debug if include_debug else None,
    )

    return st, det_tr


if __name__ == "__main__":
    # Minimal sanity demo: dummy anomaly scores; run both detectors.
    scores = [0.0] * 50 + [1.0] * 10 + [10.0] * 10 + [1.0] * 10

    st, det = anomaly_scores_to_event_traces(
        scores,
        state_cfg=StateEngineConfig(ema_alpha=0.2),
        spike_cfg=SpikeDetectorConfig(recent_window=5, prior_window=10, threshold_pct=50.0, edge_only=True),
        persistence_cfg=PersistenceDetectorConfig(
            baseline_window=30,
            k_mad=3.0,
            hold_steps=5,
        ),
        include_debug=False,
    )

    n_spikes = sum(1 for x in (det.spike_event or []) if x)
    n_persist = sum(1 for x in (det.persistence_event or []) if x)
    print(f"T={len(scores)}  spikes={n_spikes}  persistent={n_persist}")
