from __future__ import annotations

from typing import Iterable, Optional, Tuple


def detection_lag_steps(
    spikes: Iterable[int],
    toggle_step: int,
) -> Optional[int]:
    """
    Compute detection lag in *steps* given a binary spike sequence and a ground-truth
    toggle step (1-based index).

    Returns
    -------
    lag_steps : Optional[int]
        first_spike_step - toggle_step, or None if no spike at/after toggle_step.
    """
    if toggle_step < 1:
        raise ValueError("toggle_step must be 1-based and >= 1")

    first_spike_step: Optional[int] = None
    for i, s in enumerate(spikes, start=1):
        if i < toggle_step:
            continue
        if s:
            first_spike_step = i
            break

    if first_spike_step is None:
        return None

    return first_spike_step - toggle_step


def detection_lag_seconds(
    spikes: Iterable[int],
    toggle_step: int,
    rate_hz: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Convenience wrapper: compute lag in steps *and* seconds.

    Parameters
    ----------
    spikes : Iterable[int]
        0/1 spike sequence.
    toggle_step : int
        1-based GT toggle index.
    rate_hz : float
        Sampling rate (steps per second).

    Returns
    -------
    (lag_steps, lag_seconds)
        lag_steps as in detection_lag_steps, and lag_seconds = lag_steps / rate_hz
        or (None, None) if no spike was found.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive.")

    lag_steps = detection_lag_steps(spikes, toggle_step)
    if lag_steps is None:
        return None, None

    lag_seconds = lag_steps / rate_hz
    return lag_steps, lag_seconds
