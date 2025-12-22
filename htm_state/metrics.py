from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class AlfaCounts:
    """
    Confusion-matrix counts for ALFA-style *sequence-level* detection evaluation.

    A sequence is TP if it contains a fault and has >=1 detection at/after boundary.
    A no-fault sequence is FP if it has >=1 detection anywhere.
    """

    tp: int
    fp: int
    tn: int
    fn: int


def accuracy_precision_recall(counts: AlfaCounts) -> Tuple[float, float, float]:
    """
    Returns (accuracy, precision, recall) with safe zero-division handling.
    """
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return acc, prec, rec


def alfa_counts_from_sequences(
    *,
    has_fault: Sequence[bool],
    detected: Sequence[bool],
) -> AlfaCounts:
    """
    Compute ALFA-style TP/FP/TN/FN from boolean vectors over sequences.
    """
    if len(has_fault) != len(detected):
        raise ValueError("has_fault and detected must have the same length")

    tp = fp = tn = fn = 0
    for hf, det in zip(has_fault, detected):
        if hf:
            if det:
                tp += 1
            else:
                fn += 1
        else:
            if det:
                fp += 1
            else:
                tn += 1
    return AlfaCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def alfa_detection_times_s(
    *,
    lags_s: Sequence[Optional[float]],
    has_fault: Sequence[bool],
    detected: Sequence[bool],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute (avg_detection_time_s, max_detection_time_s) over *faulted* sequences
    that were detected. Undetected faults are excluded from the timing average/max
    (consistent with many online-detection reporting conventions).
    """
    times: List[float] = []
    for lag, hf, det in zip(lags_s, has_fault, detected):
        if not hf:
            continue
        if not det:
            continue
        if lag is None:
            continue
        times.append(float(lag))

    if not times:
        return None, None

    avg = sum(times) / len(times)
    mx = max(times)
    return avg, mx


# --- Ground-truth sanity checking (ALFA official table) ---

def normalize_alfa_sequence_name(name: str) -> str:
    """
    Normalize a run/sequence identifier down to the ALFA 'Processed Sequences' key,
    e.g. '2018-10-18-11-04-08_1' or '2018-10-05-15-55-10'.
    """
    # already normalized?
    if name[:4].isdigit() and "-" in name:
        return name

    # common run_id format: carbonZ_<seq>_<rest>
    import re
    m = re.search(r"(20\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:_\d)?)", name)
    if m:
        return m.group(1)
    return name


def sanity_check_alfa_boundaries(
    *,
    per_run_rows: Sequence[dict],
    alfa_pre_failure_s_by_seq: Dict[str, Optional[float]],
    boundary_time_key: str = "boundary_time_s",
    run_id_key: str = "run_id",
    tol_s: float = 0.75,
) -> List[str]:
    """
    Compare per-run boundary time(s) vs ALFA official Flight Time Pre-Failure (s).

    Returns a list of human-readable mismatch strings. Empty list => OK.
    """
    mismatches: List[str] = []
    for row in per_run_rows:
        run_id = str(row.get(run_id_key, ""))
        seq = normalize_alfa_sequence_name(run_id)

        gt = alfa_pre_failure_s_by_seq.get(seq, None)
        bt = row.get(boundary_time_key, None)
        if bt is None:
            # If per_run excludes negatives or boundary missing, skip here.
            continue
        try:
            bt_f = float(bt)
        except Exception:
            continue

        # No-failure sequences have no GT pre-failure time
        if gt is None:
            continue

        if abs(bt_f - float(gt)) > tol_s:
            mismatches.append(
                f"{seq}: per_run {bt_f:.3f}s vs ALFA pre-failure {float(gt):.3f}s (Δ={bt_f-float(gt):+.3f}s)"
            )

    return mismatches

def boundary_step_from_pre_failure_seconds(
    pre_failure_s: float,
    sample_rate_hz: float,
    *,
    mode: str = "first_post",
) -> int:
    """
    Convert ALFA-style 'Flight Time Pre-Failure (s)' to a 1-based boundary step index.

    The ALFA table provides a *time* at which the failure is injected (relative to the
    start of the sequence). We must map that continuous time to a discrete sample index.

    Parameters
    ----------
    pre_failure_s : float
        Seconds before failure injection (ALFA table).
    sample_rate_hz : float
        Sampling frequency (e.g., ~25 Hz for many ALFA signals).
    mode : str
        - "last_pre": boundary is the last pre-failure sample (Option A / legacy).
        - "first_post": boundary is the first post-failure sample (Option B / recommended).

    Returns
    -------
    boundary_step : int
        1-based index into the sampled sequence.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if pre_failure_s < 0:
        raise ValueError("pre_failure_s must be >= 0")

    # Samples are assumed to occur at times:
    #   t_i = (i-1)/Fs   for i = 1,2,3,...
    # Failure injection time is t = pre_failure_s.
    if mode == "last_pre":
        # last sample with t_i <= pre_failure_s
        # i = floor(pre_failure_s*Fs) + 1
        return int(math.floor(pre_failure_s * sample_rate_hz)) + 1
    if mode == "first_post":
        # first sample with t_i >= pre_failure_s
        # (i-1)/Fs >= pre_failure_s  =>  i >= pre_failure_s*Fs + 1
        # i = ceil(pre_failure_s*Fs + 1)
        return int(math.ceil(pre_failure_s * sample_rate_hz + 1.0))

    raise ValueError(f"Unknown mode={mode!r}. Use 'last_pre' or 'first_post'.")

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


def detection_lag_steps_from_events(
    event_steps: Iterable[int],
    toggle_step: int,
) -> Optional[int]:
    """
    Compute detection lag in *steps* given a list of event step indices
    (e.g., [12, 44, 90]) and a ground-truth toggle step.

    Returns first_event_step - toggle_step, or None if no event at/after toggle_step.
    """
    if toggle_step < 0:
        raise ValueError("toggle_step must be >= 0")

    first_event: Optional[int] = None
    for s in event_steps:
        if s >= toggle_step:
            first_event = int(s)
            break

    if first_event is None:
        return None

    return first_event - toggle_step


def detection_lag_seconds_from_events(
    event_steps: Iterable[int],
    toggle_step: int,
    rate_hz: float,
) -> Tuple[Optional[int], Optional[float]]:
    """Wrapper returning (lag_steps, lag_seconds) for event step indices."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive.")
    lag_steps = detection_lag_steps_from_events(event_steps, toggle_step)
    if lag_steps is None:
        return None, None
    return lag_steps, lag_steps / rate_hz
