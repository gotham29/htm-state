from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metrics import (
    AlfaCounts,
    accuracy_precision_recall,
    alfa_counts_from_sequences,
    alfa_detection_times_s,
    sanity_check_alfa_boundaries,
)

@dataclass(frozen=True)
class RunGroundTruth:
    """
    Sequence-level ground truth for ALFA-style evaluation.

    - has_failure: whether this sequence contains an injected fault.
    - boundary_step: 1-based step index of the failure boundary (if has_failure).
      If has_failure=False, boundary_step must be None.
    """

    has_failure: bool
    boundary_step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.has_failure and (self.boundary_step is None or self.boundary_step < 1):
            raise ValueError("If has_failure=True, boundary_step must be a 1-based int.")
        if (not self.has_failure) and (self.boundary_step is not None):
            raise ValueError("If has_failure=False, boundary_step must be None.")


@dataclass(frozen=True)
class RunDetections:
    """
    Binary detection streams aligned to the run timeline.
    """

    spikes: Sequence[int]          # 0/1 per step
    sustained: Sequence[int]       # 0/1 per step

    def __post_init__(self) -> None:
        if len(self.spikes) != len(self.sustained):
            raise ValueError("spikes and sustained must have the same length.")


@dataclass(frozen=True)
class RunEvalResult:
    """
    Per-sequence outcome for ALFA-style evaluation.
    """

    # Sequence classification (ALFA-style)
    tp: bool
    fp: bool
    fn: bool
    tn: bool

    # Timing metrics (only meaningful for faulted sequences)
    first_detection_step: Optional[int]
    detection_time_s: Optional[float]

    # Helpful counts for reporting/debug
    num_spikes_total: int
    num_sustained_total: int


def _first_step_at_or_after(stream: Sequence[int], step0: int) -> Optional[int]:
    """
    Return the first 1-based step i >= step0 such that stream[i-1] == 1, else None.
    """
    if step0 < 1:
        raise ValueError("step0 must be 1-based and >= 1")
    for i in range(step0, len(stream) + 1):
        if stream[i - 1]:
            return i
    return None


def evaluate_run(
    det: RunDetections,
    gt: RunGroundTruth,
    *,
    step_seconds: float,
    decision_rule: str = "sustained_or_spike",
) -> RunEvalResult:
    """
    Evaluate one run using ALFA-style sequence-level metrics.

    Parameters
    ----------
    det : RunDetections
        spikes and sustained streams (0/1 per step), same length.
    gt : RunGroundTruth
        sequence-level fault presence and (optionally) boundary step.
    step_seconds : float
        seconds per step (e.g., 1/Fs). Used to convert detection delay to seconds.
    decision_rule : str
        - "sustained_only": DETECT iff sustained==1 at/after boundary
        - "sustained_or_spike": DETECT iff (sustained==1 OR spike==1) at/after boundary

    Returns
    -------
    RunEvalResult
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be > 0")

    n = len(det.spikes)
    num_spikes = int(sum(1 for x in det.spikes if x))
    num_sustained = int(sum(1 for x in det.sustained if x))

    if decision_rule == "sustained_only":
        stream = det.sustained
    elif decision_rule == "sustained_or_spike":
        stream = [1 if (det.sustained[i] or det.spikes[i]) else 0 for i in range(n)]
    else:
        raise ValueError("decision_rule must be 'sustained_only' or 'sustained_or_spike'")

    # Determine first detection step based on whether there is a boundary.
    if gt.has_failure:
        assert gt.boundary_step is not None
        first_det = _first_step_at_or_after(stream, gt.boundary_step)
        detected = first_det is not None

        tp = detected
        fn = not detected
        fp = False
        tn = False

        detection_time_s = None
        if first_det is not None:
            # Detection delay measured from boundary step (0 means at boundary step)
            detection_time_s = (first_det - gt.boundary_step) * step_seconds

        return RunEvalResult(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            first_detection_step=first_det,
            detection_time_s=detection_time_s,
            num_spikes_total=num_spikes,
            num_sustained_total=num_sustained,
        )

    # No-failure sequence: any detection at any time counts as FP (sequence-level).
    first_det = _first_step_at_or_after(stream, 1)
    detected = first_det is not None

    fp = detected
    tn = not detected
    tp = False
    fn = False

    return RunEvalResult(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        first_detection_step=first_det,
        detection_time_s=None,
        num_spikes_total=num_spikes,
        num_sustained_total=num_sustained,
    )


def load_alfa_ground_truth_csv(path: Path) -> Dict[str, Optional[float]]:
    """
    Load a CSV with at least:
      - sequence (e.g., 2018-10-18-11-04-08_1)
      - pre_failure_s (float or blank for no-failure)
      - has_fault (bool)

    Returns dict {sequence -> pre_failure_s or None}.
    """
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError("ground truth CSV must contain 'sequence'")
    if "pre_failure_s" not in df.columns:
        raise ValueError("ground truth CSV must contain 'pre_failure_s'")

    out: Dict[str, Optional[float]] = {}
    for _, r in df.iterrows():
        seq = str(r["sequence"])
        v = r["pre_failure_s"]
        if pd.isna(v):
            out[seq] = None
        else:
            out[seq] = float(v)
    return out


def compute_alfa_metrics_table(
    *,
    runs_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    # choose which detector to evaluate:
    detected_col: str,
    lag_s_col: str,
    # keys:
    run_id_col: str = "run_id",
    tol_boundary_s: float = 0.75,
) -> Dict[str, Any]:
    """
    Compute ALFA-style metrics for a chosen detector, given:
      - runs_df: per-run outputs (must include run_id, boundary_time_s, detector columns)
      - gt_df: ground-truth manifest with:
          sequence, has_fault, pre_failure_s, split (optional)

    IMPORTANT: this expects runs_df to include *both* fault and no-fault sequences
    if you want accuracy/precision/recall to be meaningful.
    """
    required = {run_id_col, "boundary_time_s", detected_col, lag_s_col}
    missing = required - set(runs_df.columns)
    if missing:
        raise ValueError(f"runs_df missing columns: {sorted(missing)}")

    # --- sanity-check boundary times vs ALFA pre-failure seconds ---
    alfa_pre_failure = {
        str(r["sequence"]): (None if pd.isna(r["pre_failure_s"]) else float(r["pre_failure_s"]))
        for _, r in gt_df.iterrows()
    }
    mismatches = sanity_check_alfa_boundaries(
        per_run_rows=runs_df.to_dict(orient="records"),
        alfa_pre_failure_s_by_seq=alfa_pre_failure,
        boundary_time_key="boundary_time_s",
        run_id_key=run_id_col,
        tol_s=tol_boundary_s,
    )
    if mismatches:
        msg = "Boundary sanity-check failed:\n" + "\n".join("  - " + m for m in mismatches[:50])
        if len(mismatches) > 50:
            msg += f"\n  ... and {len(mismatches)-50} more"
        raise RuntimeError(msg)

    # --- join to bring has_fault into same table ---
    # normalize to the same 'sequence' key
    from metrics import normalize_alfa_sequence_name
    tmp = runs_df.copy()
    tmp["sequence"] = tmp[run_id_col].map(normalize_alfa_sequence_name)

    merged = tmp.merge(gt_df[["sequence", "has_fault"]], on="sequence", how="left")
    if merged["has_fault"].isna().any():
        missing_seq = merged.loc[merged["has_fault"].isna(), "sequence"].unique().tolist()
        raise RuntimeError(f"Missing GT rows for sequences: {missing_seq[:20]}")

    has_fault = merged["has_fault"].astype(bool).tolist()
    detected = merged[detected_col].astype(bool).tolist()

    counts: AlfaCounts = alfa_counts_from_sequences(has_fault=has_fault, detected=detected)
    acc, prec, rec = accuracy_precision_recall(counts)

    lags_s: List[Optional[float]] = []
    for v in merged[lag_s_col].tolist():
        if pd.isna(v):
            lags_s.append(None)
        else:
            lags_s.append(float(v))

    avg_dt, max_dt = alfa_detection_times_s(lags_s=lags_s, has_fault=has_fault, detected=detected)

    return {
        "counts": counts,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "avg_detection_time_s": avg_dt,
        "max_detection_time_s": max_dt,
        "n_sequences": len(merged),
        "n_fault": int(sum(has_fault)),
        "n_no_fault": int(len(has_fault) - sum(has_fault)),
    }


def summarize_alfa_style(
    results: Sequence[RunEvalResult],
) -> Dict[str, Optional[float]]:
    """
    Aggregate ALFA-style metrics across sequences:
    - Accuracy: (TP + TN) / N
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)

    Also reports average/max detection time over detected faulted sequences.
    """
    tp = sum(1 for r in results if r.tp)
    fp = sum(1 for r in results if r.fp)
    fn = sum(1 for r in results if r.fn)
    tn = sum(1 for r in results if r.tn)
    n = len(results)

    acc = (tp + tn) / n if n else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None

    det_times = [r.detection_time_s for r in results if r.detection_time_s is not None]
    avg_dt = sum(det_times) / len(det_times) if det_times else None
    max_dt = max(det_times) if det_times else None

    return {
        "n": float(n),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "avg_detection_time_s": avg_dt,
        "max_detection_time_s": max_dt,
    }
