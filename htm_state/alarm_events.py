#!/usr/bin/env python3
"""
alarm_events.py

Algorithm-agnostic helpers for producing "alarm steps" from per-timestep event streams.

This module sits between:
  (A) any anomaly-scoring algorithm (HTM, Isolation Forest, LSTM, etc.) which produces
      a scalar "anomaly score" per timestep, and
  (B) apples-to-apples evaluation (eval_apples_to_apples.py), which consumes only:
      - init_done_step
      - toggle_step
      - alarm_steps  (1-based indices when "announcement" happens)

Key idea
--------
We standardize what an "alarm event" is:
- The detection layer produces boolean event streams per timestep, e.g.:
    spike_event[t]      in {False, True}
    persistence_event[t] in {False, True}
- We then combine those event streams with an explicit rule:
    mode = "spike" | "persistence" | "or" | "and"
- Finally we convert the resulting alarm_event[t] into alarm_steps: indices where alarm_event[t] is True.

Why keep this separate?
-----------------------
- Keeps evaluation logic clean and auditable.
- Lets us swap scoring algorithms without touching evaluation code.
- Lets us swap detectors or event definitions without touching evaluation code.
- Makes it trivial to reproduce the paper’s "announcement" logic consistently.

Indexing conventions
--------------------
Within Python arrays/lists, events are naturally 0-based (events[0] corresponds to step 1).
For evaluation, we use 1-based step indices. This module provides explicit conversion.

This module does NOT implement any particular detector (SpikeDetector/PersistenceDetector);
it only combines and converts event streams once they exist.

If we later want to expose detector choices (HTM-State SpikeDetector and PersistenceDetector),
we can add a separate adapter module (e.g., `htm_state_detectors.py`) that:
- takes anomaly_score[t],
- runs the configured HTM-State detector(s),
- returns spike_event[t] and persistence_event[t] for this module to combine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple


AlarmMode = Literal["spike", "persistence", "or", "and"]


@dataclass(frozen=True)
class CombinedEventStreams:
    """
    Convenience bundle for event streams.

    All sequences here are 0-based Python sequences where index 0 corresponds to step 1.
    """
    spike_event: Optional[Sequence[bool]]
    persistence_event: Optional[Sequence[bool]]
    alarm_event: Sequence[bool]


def _validate_bool_stream(name: str, stream: Sequence[bool]) -> None:
    """
    Validate that a stream is a sequence of bools.

    In practice, we accept ints 0/1 only if they are actual bool instances.
    If upstream produces numpy arrays, the caller should cast to Python bools,
    or we can broaden this later (keeping it strict for auditability now).
    """
    for i, v in enumerate(stream):
        if type(v) is not bool:
            raise TypeError(
                f"{name}[{i}] must be a Python bool, got {type(v).__name__}. "
                "Cast upstream to bool for clarity/auditability."
            )


def _validate_lengths(
    spike_event: Optional[Sequence[bool]],
    persistence_event: Optional[Sequence[bool]],
) -> int:
    """
    Validate stream lengths and return the common length.

    Rules:
    - If both provided, lengths must match.
    - If only one provided, we use its length.
    - If neither provided, error (no information).
    """
    if spike_event is None and persistence_event is None:
        raise ValueError("At least one of spike_event or persistence_event must be provided")

    if spike_event is not None:
        _validate_bool_stream("spike_event", spike_event)
    if persistence_event is not None:
        _validate_bool_stream("persistence_event", persistence_event)

    if spike_event is not None and persistence_event is not None:
        if len(spike_event) != len(persistence_event):
            raise ValueError(
                f"Event stream length mismatch: spike_event={len(spike_event)} "
                f"persistence_event={len(persistence_event)}"
            )
        return len(spike_event)

    return len(spike_event) if spike_event is not None else len(persistence_event)  # type: ignore[arg-type]


def combine_alarm_events(
    *,
    mode: AlarmMode,
    spike_event: Optional[Sequence[bool]] = None,
    persistence_event: Optional[Sequence[bool]] = None,
) -> CombinedEventStreams:
    """
    Combine boolean event streams into a single alarm_event stream.

    Parameters
    ----------
    mode:
        Combination rule:
        - "spike":        alarm_event[t] = spike_event[t]
        - "persistence":  alarm_event[t] = persistence_event[t]
        - "or":           alarm_event[t] = spike_event[t] OR persistence_event[t]
        - "and":          alarm_event[t] = spike_event[t] AND persistence_event[t]
    spike_event:
        Optional bool stream.
    persistence_event:
        Optional bool stream.

    Returns
    -------
    CombinedEventStreams:
        Contains original inputs plus the combined alarm_event stream.

    Examples
    --------
    >>> spike = [False, True, False]
    >>> pers  = [False, False, True]
    >>> combine_alarm_events(mode="spike", spike_event=spike, persistence_event=pers).alarm_event
    [False, True, False]
    >>> combine_alarm_events(mode="persistence", spike_event=spike, persistence_event=pers).alarm_event
    [False, False, True]
    >>> combine_alarm_events(mode="or", spike_event=spike, persistence_event=pers).alarm_event
    [False, True, True]
    >>> combine_alarm_events(mode="and", spike_event=spike, persistence_event=pers).alarm_event
    [False, False, False]
    """
    n = _validate_lengths(spike_event, persistence_event)

    # For modes requiring a stream, ensure it's present.
    if mode == "spike" and spike_event is None:
        raise ValueError('mode="spike" requires spike_event')
    if mode == "persistence" and persistence_event is None:
        raise ValueError('mode="persistence" requires persistence_event')
    if mode in ("or", "and"):
        if spike_event is None or persistence_event is None:
            raise ValueError(f'mode="{mode}" requires both spike_event and persistence_event')

    alarm: List[bool] = [False] * n

    if mode == "spike":
        assert spike_event is not None
        alarm = list(spike_event)
    elif mode == "persistence":
        assert persistence_event is not None
        alarm = list(persistence_event)
    elif mode == "or":
        assert spike_event is not None and persistence_event is not None
        for i in range(n):
            alarm[i] = bool(spike_event[i] or persistence_event[i])
    elif mode == "and":
        assert spike_event is not None and persistence_event is not None
        for i in range(n):
            alarm[i] = bool(spike_event[i] and persistence_event[i])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return CombinedEventStreams(
        spike_event=spike_event,
        persistence_event=persistence_event,
        alarm_event=alarm,
    )


def event_stream_to_alarm_steps(event_stream: Sequence[bool]) -> List[int]:
    """
    Convert a boolean event stream into a list of 1-based alarm step indices.

    Parameters
    ----------
    event_stream:
        0-based Python sequence of bools, where event_stream[0] corresponds to step 1.

    Returns
    -------
    alarm_steps:
        Sorted list of 1-based indices i where event_stream[i-1] is True.

    Examples
    --------
    >>> event_stream_to_alarm_steps([False, True, True, False])
    [2, 3]
    >>> event_stream_to_alarm_steps([])
    []
    """
    _validate_bool_stream("event_stream", event_stream)
    return [i + 1 for i, v in enumerate(event_stream) if v]


def first_alarm_step_from_event_stream(
    event_stream: Sequence[bool],
    *,
    init_done_step: int,
) -> Optional[int]:
    """
    Convenience: return the first alarm step >= init_done_step from a boolean event stream.

    This is often handy for per-run reporting, and matches the definition used by
    eval_apples_to_apples.first_alarm_at_or_after(alarm_steps, init_done_step).

    Parameters
    ----------
    event_stream:
        0-based bools where index 0 is step 1.
    init_done_step:
        1-based first eligible step to consider alarms.

    Returns
    -------
    first_alarm_step:
        The first 1-based step >= init_done_step with event=True, else None.

    Examples
    --------
    >>> first_alarm_step_from_event_stream([True, False, True], init_done_step=1)
    1
    >>> first_alarm_step_from_event_stream([True, False, True], init_done_step=2)
    3
    >>> first_alarm_step_from_event_stream([True, False, True], init_done_step=4) is None
    True
    """
    if init_done_step < 1:
        raise ValueError("init_done_step must be 1-based and >= 1")
    _validate_bool_stream("event_stream", event_stream)

    start_idx = init_done_step - 1  # convert 1-based step to 0-based index
    if start_idx >= len(event_stream):
        return None

    for idx in range(start_idx, len(event_stream)):
        if event_stream[idx]:
            return idx + 1  # back to 1-based
    return None


def combine_and_extract_alarm_steps(
    *,
    mode: AlarmMode,
    spike_event: Optional[Sequence[bool]] = None,
    persistence_event: Optional[Sequence[bool]] = None,
) -> List[int]:
    """
    Convenience wrapper: combine event streams (per mode) and return alarm_steps (1-based).

    This is the typical output that feeds directly into eval_apples_to_apples.confusion_for_run().

    Examples
    --------
    >>> spike = [False, True, False]
    >>> pers  = [False, False, True]
    >>> combine_and_extract_alarm_steps(mode="or", spike_event=spike, persistence_event=pers)
    [2, 3]
    """
    combined = combine_alarm_events(mode=mode, spike_event=spike_event, persistence_event=persistence_event)
    return event_stream_to_alarm_steps(combined.alarm_event)


if __name__ == "__main__":
    # Minimal sanity demo.
    spike = [False, True, False, False, True]
    pers = [False, False, False, True, True]

    for mode in ("spike", "persistence", "or", "and"):
        c = combine_alarm_events(mode=mode, spike_event=spike, persistence_event=pers)
        steps = event_stream_to_alarm_steps(c.alarm_event)
        print(f"mode={mode:11s} alarm_steps={steps}")
