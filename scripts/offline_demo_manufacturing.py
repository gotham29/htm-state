#!/usr/bin/env python3
"""
Offline evaluation for Demo 4 — Manufacturing Process Drift Detection.

Pipeline:
  manufacturing features -> anomaly proxy -> EMA state -> growth-based spike detector
Then:
  compare spike times to embedded regime boundaries [800, 1600]
  and report detection lags in steps and seconds.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Spike / state helpers (same structure as healthcare demo)
# ---------------------------------------------------------------------


@dataclass
class SpikeConfig:
    """
    Simple growth-based spike detector config.

    Threshold is expressed as % growth between recent and prior windows.
    """
    recent_window: int
    prior_window: int
    threshold_pct: float  # e.g., 40.0 means 40% growth

    @classmethod
    def from_seconds(
        cls,
        recent_sec: float,
        prior_sec: float,
        threshold_pct: float,
        rate_hz: float,
    ) -> "SpikeConfig":
        recent_window = max(1, int(recent_sec * rate_hz))
        prior_window = max(1, int(prior_sec * rate_hz))
        return cls(
            recent_window=recent_window,
            prior_window=prior_window,
            threshold_pct=threshold_pct,
        )


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average over a 1D array.
    """
    out = np.empty_like(x, dtype=float)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def detect_spikes_from_state(
    state: np.ndarray,
    cfg: SpikeConfig,
) -> np.ndarray:
    """
    Growth-based spike detector applied to the state signal.

    Compares mean of recent window vs prior window; if growth exceeds
    threshold_pct, and current point is a local maximum, mark a spike.
    """
    n = len(state)
    spikes = np.zeros(n, dtype=bool)

    r = cfg.recent_window
    p = cfg.prior_window
    total = r + p

    if n < total:
        return spikes

    for t in range(total, n):
        recent = state[t - r : t]
        prior = state[t - total : t - r]

        prior_mean = prior.mean() + 1e-8
        recent_mean = recent.mean()
        growth = (recent_mean - prior_mean) / prior_mean * 100.0

        # Local maximum check
        if (
            growth >= cfg.threshold_pct
            and state[t] >= state[t - 1]
            and state[t] >= state[t + 1 - 1 if t + 1 < n else t]
        ):
            spikes[t] = True

    return spikes


def run_htm_state_pipeline_for_demo(
    df: pd.DataFrame,
    features: List[str],
    ema_alpha: float,
    spike_cfg: SpikeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For this synthetic demo, approximate anomaly as the mean absolute
    z-score across manufacturing features, then:

      anomaly -> EMA -> state
      state   -> spike detector
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Manufacturing demo: missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    X = df[features].to_numpy(dtype=float)

    # crude anomaly proxy: mean absolute z-score across dimensions
    z = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    anomaly = np.abs(z).mean(axis=1)

    state = ema(anomaly, alpha=ema_alpha)
    spikes = detect_spikes_from_state(state, spike_cfg)

    return anomaly, state, spikes


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Offline HTM-State manufacturing drift demo")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to manufacturing line stream CSV",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Streaming rate (Hz); used only for time conversion",
    )
    parser.add_argument("--spike-recent-sec", type=float, default=3.0)
    parser.add_argument("--spike-prior-sec", type=float, default=6.0)
    parser.add_argument("--spike-threshold-pct", type=float, default=40.0)
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for anomaly → state",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading manufacturing stream from: {args.csv}")
    df = pd.read_csv(args.csv)
    n = len(df)
    print(f"Loaded {n} timesteps.")

    features = [
        "vibration_energy",
        "spindle_load",
        "feed_rate",
        "line_speed",
        "temperature",
        "part_time",
    ]

    spike_cfg = SpikeConfig.from_seconds(
        recent_sec=args.spike_recent_sec,
        prior_sec=args.spike_prior_sec,
        threshold_pct=args.spike_threshold_pct,
        rate_hz=args.rate_hz,
    )

    anomaly, state, spikes = run_htm_state_pipeline_for_demo(
        df,
        features=features,
        ema_alpha=args.ema_alpha,
        spike_cfg=spike_cfg,
    )

    is_boundary = df.get("is_boundary", pd.Series([0] * n)).to_numpy(dtype=int)
    boundary_indices = np.where(is_boundary == 1)[0]

    print("\nAnomaly/state summary:")
    print(f"  anomaly  min={anomaly.min():.3f}, max={anomaly.max():.3f}")
    print(f"  state    min={state.min():.3f}, max={state.max():.3f}")
    print(f"  spikes   count={int(spikes.sum())}")

    if len(boundary_indices) == 0:
        print("\nNo boundaries (is_boundary==1) found in CSV; nothing to evaluate.\n")
        return

    print("\n=== Manufacturing Drift Detection Results ===")
    lags_steps = []

    for i, b_idx in enumerate(boundary_indices):
        # first spike at or after boundary
        spike_candidates = np.where(spikes[b_idx:])[0]
        if len(spike_candidates) == 0:
            print(
                f"Transition {i}: boundary at step {b_idx} → "
                "no spike detected after this boundary in the stream."
            )
            continue

        first_spike_offset = spike_candidates[0]
        detected_idx = b_idx + first_spike_offset
        lag_steps = detected_idx - b_idx
        lag_sec = lag_steps / args.rate_hz
        lags_steps.append(lag_steps)

        print(
            f"Transition {i}: boundary at step {b_idx} "
            f"→ detected at step {detected_idx}, "
            f"lag = {lag_steps} steps ({lag_sec:.3f} s)"
        )

    if lags_steps:
        avg_lag_steps = float(np.mean(lags_steps))
        avg_lag_sec = avg_lag_steps / args.rate_hz
        print(
            f"\nAverage detection lag over {len(lags_steps)} transitions: "
            f"{avg_lag_steps:.1f} steps ({avg_lag_sec:.2f} s)\n"
        )
    else:
        print("\nNo transitions had a detected spike; average lag cannot be computed.\n")


if __name__ == "__main__":
    main()
