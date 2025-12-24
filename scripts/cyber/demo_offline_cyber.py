# scripts/offline_demo_cyber.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from htm_state.htm_session import HTMSession
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline HTM-State demo on UNSW-NB15 micro-drifts."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="demos/cyber_demo/unsw_cyber_stream.csv",
        help="Prepared cyber stream CSV (from prepare_unsw_cyber_stream.py).",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Sample rate used when generating the stream (default: 10 Hz).",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA alpha for HTM anomaly smoothing (MWL/state).",
    )
    p.add_argument(
        "--spike-recent-sec",
        type=float,
        default=2.0,
        help="Recent window size in seconds for spike detector.",
    )
    p.add_argument(
        "--spike-prior-sec",
        type=float,
        default=6.0,
        help="Prior window size in seconds for spike detector.",
    )
    p.add_argument(
        "--spike-threshold-pct",
        type=float,
        default=30.0,
        help="Growth %% threshold for spike detector.",
    )
    p.add_argument(
        "--spike-min-sep-sec",
        type=float,
        default=2.0,
        help="Minimum separation between spikes in seconds.",
    )
    p.add_argument(
        "--enc-n-per-feature",
        type=int,
        default=64,
        help="Encoder n (bits) per scalar feature.",
    )
    p.add_argument(
        "--enc-w-per-feature",
        type=int,
        default=8,
        help="Encoder w (active bits) per scalar feature.",
    )
    return p.parse_args()


def sec_to_steps(sec: float, rate_hz: float) -> int:
    return max(1, int(round(sec * rate_hz)))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "t" not in df.columns or "is_drift_boundary" not in df.columns:
        raise ValueError(
            "Expected 't' and 'is_drift_boundary' columns. "
            "Did you run scripts.prepare_unsw_cyber_stream?"
        )

    # Features: all numeric columns except metadata
    meta_cols = {"t", "label", "attack_cat", "phase_id", "is_drift_boundary"}
    feature_names = [c for c in df.columns if c not in meta_cols]

    if not feature_names:
        raise ValueError("No feature columns found for HTM input.")

    print(f"Using features: {feature_names}")

    # Build feature ranges from the data for HTMSession encoders
    feature_ranges: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        col = df[name].astype(float)
        vmin = float(col.min())
        vmax = float(col.max())
        if vmin == vmax:
            # avoid zero range
            vmax = vmin + 1.0
        feature_ranges[name] = {"min": vmin, "max": vmax}

    # Simple SP/TM parameter dictionaries; tweak later if needed
    sp_params = {
        "columnCount": 2048,
        "potentialPct": 0.8,
        "globalInhibition": True,
        "synPermActiveInc": 0.003,
        "synPermInactiveDec": 0.0005,
        "synPermConnected": 0.2,
        "boostStrength": 0.0,
    }

    tm_params = {
        "cellsPerColumn": 32,
        "activationThreshold": 20,
        "initialPerm": 0.21,
        "permanenceConnected": 0.5,
        "minThreshold": 13,
        "newSynapseCount": 31,
        "permanenceInc": 0.1,
        "permanenceDec": 0.0,
        "predictedSegmentDecrement": 0.001,
    }

    # Core HTM session (anomaly + EMA => "mwl"/state)
    session = HTMSession(
        feature_names=feature_names,
        enc_n_per_feature=args.enc_n_per_feature,
        enc_w_per_feature=args.enc_w_per_feature,
        sp_params=sp_params,
        tm_params=tm_params,
        seed=0,
        anomaly_ema_alpha=args.ema_alpha,
        feature_ranges=feature_ranges,
    )

    # Spike detector over the HTM "state"
    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec, args.rate_hz),
        min_delta=0.0,
        eps=1e-3,
    )
    spike_detector = SpikeDetector(spike_cfg)

    # Track drift boundaries
    drift_steps: List[int] = []
    drift_times: List[float] = []
    for i, row in df.iterrows():
        if int(row["is_drift_boundary"]) == 1:
            drift_steps.append(i)
            drift_times.append(float(row["t"]))

    print(f"Found {len(drift_steps)} drift boundaries at steps: {drift_steps}")

    # Map each drift to its first detected spike
    detection_info: List[Dict[str, Optional[float]]] = []
    for idx in range(len(drift_steps)):
        detection_info.append(
            {
                "drift_idx": idx,
                "boundary_step": drift_steps[idx],
                "boundary_time": drift_times[idx],
                "det_step": None,
                "det_time": None,
            }
        )

    next_drift_idx = 0

    for i, row in df.iterrows():
        t = float(row["t"])
        feats = {name: float(row[name]) for name in feature_names}

        # HTM step: anomaly + EMA ("mwl" = our scalar state)
        out = session.step(feats)
        state = float(out["mwl"])

        # Spike detector on the state
        spike_res = spike_detector.update(state)
        spike_flag = spike_res["spike"]

        if spike_flag and next_drift_idx < len(drift_steps):
            boundary_step = detection_info[next_drift_idx]["boundary_step"]
            if i >= boundary_step and detection_info[next_drift_idx]["det_step"] is None:
                detection_info[next_drift_idx]["det_step"] = i
                detection_info[next_drift_idx]["det_time"] = t
                next_drift_idx += 1

    # Report detection lags
    print("\n=== Drift Detection Results ===")
    lags_sec: List[float] = []
    for info in detection_info:
        idx = info["drift_idx"]
        b_step = info["boundary_step"]
        b_time = info["boundary_time"]
        d_step = info["det_step"]
        d_time = info["det_time"]

        if d_step is None:
            print(
                f"Drift {idx}: boundary at step {b_step} (t={b_time:.3f}s) "
                f"→ no spike detected after boundary."
            )
            continue

        lag_steps = d_step - b_step
        lag_sec = max(0.0, d_time - b_time)
        lags_sec.append(lag_sec)

        print(
            f"Drift {idx}: boundary at step {b_step} (t={b_time:.3f}s) "
            f"→ detected at step {d_step} (t={d_time:.3f}s), "
            f"lag = {lag_steps} steps ({lag_sec:.3f} s)"
        )

    if lags_sec:
        avg_lag = sum(lags_sec) / len(lags_sec)
        print(f"\nAverage detection lag over {len(lags_sec)} drifts: {avg_lag:.3f} s")
    else:
        print("\nNo drifts were detected by the spike detector.")


if __name__ == "__main__":
    main()
