# scripts/offline_demo_detection_lag.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from htm_state.engine import (
    StateEngine,
    StateEngineConfig,
    BaselineBackend,
    HTMBackend,
)
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig
from htm_state.metrics import detection_lag_seconds
from htm_state.htm_session import HTMSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline HTM State demo: run StateEngine + SpikeDetector over a CSV, "
            "then compute detection lag vs a ground-truth toggle."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="demos/workload_demo/synthetic_workload.csv",
        help="Path to CSV file with columns: t, control_x, control_y.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Sampling rate (steps per second). Used to convert lag to seconds.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="baseline",
        choices=["baseline", "htm"],
        help="Which anomaly backend to use: 'baseline' or 'htm'.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA alpha for state smoothing.",
    )
    parser.add_argument(
        "--spike-recent-sec",
        type=float,
        default=3.0,
        help="Recent window size in seconds for spike detector.",
    )
    parser.add_argument(
        "--spike-prior-sec",
        type=float,
        default=6.0,
        help="Prior window size in seconds for spike detector.",
    )
    parser.add_argument(
        "--spike-threshold-pct",
        type=float,
        default=50.0,
        help="Growth %% threshold for spike detector.",
    )
    parser.add_argument(
        "--spike-min-sep-sec",
        type=float,
        default=1.0,
        help="Minimum separation between spikes in seconds (debounce).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--toggle-step",
        type=int,
        default=None,
        help="1-based ground-truth toggle step. If not provided, defaults to mid-point of the file.",
    )
    group.add_argument(
        "--toggle-time-sec",
        type=float,
        default=None,
        help=(
            "Ground-truth toggle time in seconds (will be mapped to nearest row). "
            "If not provided, defaults to mid-point of the file."
        ),
    )
    # Optional: future hook for HTM config, if you want to read a YAML
    parser.add_argument(
        "--htm-config",
        type=str,
        default=None,
        help="Optional path to an HTM config file (used when --backend=htm).",
    )
    return parser.parse_args()


def sec_to_steps(sec: float, rate_hz: float) -> int:
    return max(1, int(round(sec * rate_hz)))


def build_backend(
    backend_kind: str,
    feature_names: List[str],
    htm_config_path: str | None = None,
):
    """
    Construct the requested backend.

    For now:
      - 'baseline' -> BaselineBackend over all feature_names
      - 'htm'      -> HTMBackend that wraps an HTMSession using some reasonable defaults

    Later, you can make this read a YAML/JSON config (htm_config_path) and forward
    detailed params into HTMSession.
    """
    if backend_kind == "baseline":
        return BaselineBackend(feature_names=feature_names)

    if backend_kind == "htm":
        # --- Simple, hard-coded HTM params for now ---
        # You can replace these with config-driven values later.
        enc_n_per_feature = 50
        enc_w_per_feature = 5

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

        seed = 42
        anomaly_ema_alpha = 0.2

        # For now we do not pass feature_ranges (defaults to [-1,1] per feature).
        # That’s okay for your synthetic sin/cos demo. Later we can compute
        # per-feature min/max from the data or take them from a config.
        htm_session = HTMSession(
            feature_names=feature_names,
            enc_n_per_feature=enc_n_per_feature,
            enc_w_per_feature=enc_w_per_feature,
            sp_params=sp_params,
            tm_params=tm_params,
            seed=seed,
            anomaly_ema_alpha=anomaly_ema_alpha,
            feature_ranges=None,
        )

        return HTMBackend(htm_session)

    raise ValueError(f"Unknown backend kind: {backend_kind}")


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError("Expected a 't' column for timestamps in the CSV.")

    # Features = all columns except time
    feature_names: List[str] = [c for c in df.columns if c != "t"]

    # Build backend: baseline or HTM
    backend = build_backend(
        backend_kind=args.backend,
        feature_names=feature_names,
        htm_config_path=args.htm_config,
    )

    # StateEngine config
    state_cfg = StateEngineConfig(ema_alpha=args.ema_alpha)

    # SpikeDetector config: convert windows in seconds -> steps
    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec, args.rate_hz),
        min_delta=0.0,
        eps=1e-3,
    )

    # Instantiate engine + detector
    engine = StateEngine(config=state_cfg, backend=backend)
    detector = SpikeDetector(spike_cfg)

    spikes: List[int] = []
    states: List[float] = []
    times: List[float] = []

    # Replay the time series
    for _, row in df.iterrows():
        t = float(row["t"])
        feats = {name: float(row[name]) for name in feature_names}

        state_out = engine.step(feats)
        state = float(state_out["state"])

        spike_res = detector.update(state)
        spike_flag = int(spike_res["spike"])

        times.append(t)
        states.append(state)
        spikes.append(spike_flag)

    n_steps = len(spikes)
    print(f"Processed {n_steps} steps from {csv_path}")

    # Determine ground-truth toggle
    if args.toggle_step is not None:
        toggle_step = args.toggle_step
    elif args.toggle_time_sec is not None:
        # Map toggle time to nearest index (1-based)
        toggle_time = args.toggle_time_sec
        abs_diffs = (df["t"] - toggle_time).abs()
        idx = int(abs_diffs.idxmin())  # 0-based index
        toggle_step = idx + 1  # 1-based
    else:
        # Default: step just after the mid-point
        toggle_step = n_steps // 2 + 1

    if toggle_step < 1 or toggle_step > n_steps:
        raise ValueError(f"toggle_step {toggle_step} is out of range (1..{n_steps}).")

    print(f"Using ground-truth toggle_step = {toggle_step} (t ≈ {times[toggle_step - 1]:.3f} s)")

    lag_steps, lag_seconds = detection_lag_seconds(
        spikes=spikes,
        toggle_step=toggle_step,
        rate_hz=args.rate_hz,
    )

    if lag_steps is None:
        print("No spike detected at or after the toggle step. Detection lag is undefined.")
    else:
        print(f"Detection lag: {lag_steps} steps")
        print(f"Detection lag: {lag_seconds:.3f} s at rate {args.rate_hz} Hz")


if __name__ == "__main__":
    main()
