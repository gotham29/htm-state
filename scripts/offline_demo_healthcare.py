"""
Offline Demo 3 — Healthcare Operator Workload (Synthetic Behavioral Stream)

This script mirrors the structure of:
  - scripts/offline_demo_detection_lag.py   (Demo 1)
  - scripts/offline_demo_cyber.py           (Demo 2)

but targets a synthetic *clinical-style* operator stream:

  demos/healthcare_demo/operator_stream.csv

Expected columns:
  - timestep
  - motion_energy
  - cursor_velocity
  - interaction_density
  - task_variability
  - is_boundary  (1 at regime boundaries, else 0)

The logic is:
  1) Load the CSV stream.
  2) Run it through the HTM-State pipeline to get:
       - anomaly(t)
       - smoothed_state(t)   (EMA of anomaly)
       - spikes(t)           (growth-based detector)
  3) Compare spike times to true boundaries (is_boundary == 1).
  4) Report per-boundary detection lag and overall average lag.

To keep this file portable, the **HTM-specific call** is isolated in
`run_htm_state_pipeline_for_demo`, which you should wire to the same core
pipeline that Demo 1 and Demo 2 use.

For now, that function is a thin placeholder that simply:
  - constructs a crude anomaly proxy from the features
  - applies EMA
  - runs a growth-based spike detector

You can replace the body of `run_htm_state_pipeline_for_demo` with your
existing HTM-State backend call so all three demos share identical logic.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Spike detector / state engine config
# ---------------------------------------------------------------------

@dataclass
class SpikeConfig:
    recent_window: int = 30   # in steps
    prior_window: int = 90    # in steps
    growth_threshold_pct: float = 40.0  # percent growth required


# ---------------------------------------------------------------------
# Core helper: simple EMA + growth-based spike detector
# ---------------------------------------------------------------------

def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average."""
    out = np.zeros_like(series, dtype=float)
    if len(series) == 0:
        return out
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def detect_spikes_from_state(
    state: np.ndarray,
    cfg: SpikeConfig,
) -> np.ndarray:
    """
    Growth-based spike detector.

    For each time t >= prior_window + recent_window:
      - compute mean state over [t-recent_window, t)
      - compute mean state over [t-prior_window-recent_window, t-recent_window)
      - if recent_mean >= (1 + growth_threshold_pct/100) * prior_mean, mark spike.
    """
    n = len(state)
    spikes = np.zeros(n, dtype=int)

    rw = cfg.recent_window
    pw = cfg.prior_window
    min_t = rw + pw

    for t in range(min_t, n):
        recent = state[t - rw : t]
        prior = state[t - rw - pw : t - rw]

        prior_mean = prior.mean() if prior.size > 0 else 0.0
        recent_mean = recent.mean() if recent.size > 0 else 0.0

        # avoid division by ~0
        if prior_mean <= 1e-8:
            continue

        growth_pct = 100.0 * (recent_mean - prior_mean) / prior_mean
        if growth_pct >= cfg.growth_threshold_pct:
            spikes[t] = 1

    return spikes


# ---------------------------------------------------------------------
# HTM-State pipeline placeholder
# ---------------------------------------------------------------------

def run_htm_state_pipeline_for_demo(
    df: pd.DataFrame,
    ema_alpha: float,
    spike_cfg: SpikeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Placeholder for the real HTM-State pipeline.

    In your actual repo, you should:
      - Replace this body with the same HTM backend call used in Demo 1 & 2.
      - Return:
          anomaly[t], state[t], spikes[t]

    For now:
      - anomaly_proxy = normalized combination of behavioral features
      - state         = EMA(anomaly_proxy)
      - spikes        = growth-based spike detector on the state
    """
    features = ["motion_energy", "cursor_velocity",
                "interaction_density", "task_variability"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Healthcare demo: missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    X = df[features].to_numpy(dtype=float)

    # crude anomaly proxy: mean of z-scored features across dimensions
    z = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    anomaly = np.abs(z).mean(axis=1)

    state = ema(anomaly, alpha=ema_alpha)
    spikes = detect_spikes_from_state(state, spike_cfg)

    return anomaly, state, spikes


# ---------------------------------------------------------------------
# Detection lag evaluation
# ---------------------------------------------------------------------

@dataclass
class DetectionResult:
    boundary_step: int
    detect_step: int
    lag_steps: int
    lag_seconds: float


def compute_detection_lags(
    is_boundary: np.ndarray,
    spikes: np.ndarray,
    rate_hz: float,
    search_horizon_steps: int = 500,
) -> List[DetectionResult]:
    """
    For each step where is_boundary == 1, find the first spike after that
    boundary (up to search_horizon_steps) and compute detection lag.
    """
    n = len(is_boundary)
    boundaries = np.where(is_boundary == 1)[0]

    results: List[DetectionResult] = []

    for b in boundaries:
        # look forward for first spike
        search_end = min(n, b + search_horizon_steps)
        idx = np.where(spikes[b:search_end] == 1)[0]

        if len(idx) == 0:
            # no detection within horizon
            continue

        detect_step = b + int(idx[0])
        lag_steps = detect_step - b
        lag_seconds = lag_steps / float(rate_hz)

        results.append(
            DetectionResult(
                boundary_step=int(b),
                detect_step=int(detect_step),
                lag_steps=int(lag_steps),
                lag_seconds=float(lag_seconds),
            )
        )

    return results


def print_detection_summary(results: List[DetectionResult], rate_hz: float) -> None:
    if not results:
        print("No detections found for any boundaries within search horizon.")
        return

    print()
    print("=== Healthcare Workload Detection Results ===")
    for i, r in enumerate(results):
        print(
            f"Transition {i}: boundary at step {r.boundary_step} "
            f"→ detected at step {r.detect_step}, "
            f"lag = {r.lag_steps} steps ({r.lag_seconds:.3f} s)"
        )

    avg_lag = sum(r.lag_seconds for r in results) / len(results)
    print()
    print(f"Average detection lag over {len(results)} transitions: {avg_lag:.3f} s")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline Demo 3 — Healthcare Operator Workload (synthetic). "
            "Streams operator-like behavioral features through the HTM-State "
            "pipeline and measures detection lag to known regime boundaries."
        )
    )
    p.add_argument(
        "--csv",
        type=str,
        default="demos/healthcare_demo/operator_stream.csv",
        help="Path to healthcare operator stream CSV.",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Sample rate (Hz) used to convert steps → seconds.",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for state estimation.",
    )
    p.add_argument(
        "--spike-recent",
        type=int,
        default=30,
        help="Recent window size (in steps) for spike detector.",
    )
    p.add_argument(
        "--spike-prior",
        type=int,
        default=90,
        help="Prior window size (in steps) for spike detector.",
    )
    p.add_argument(
        "--spike-growth-threshold",
        type=float,
        default=40.0,
        help="Growth threshold (percent) for spike detection.",
    )
    p.add_argument(
        "--search-horizon-steps",
        type=int,
        default=500,
        help="Max steps to look ahead for a spike after each boundary.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading healthcare operator stream from: {args.csv}")
    df = pd.read_csv(args.csv)

    if "is_boundary" not in df.columns:
        raise ValueError(
            "Healthcare demo CSV must contain an 'is_boundary' column "
            "with 1 at regime boundaries and 0 elsewhere."
        )

    print(f"Loaded {len(df)} timesteps.")

    spike_cfg = SpikeConfig(
        recent_window=args.spike_recent,
        prior_window=args.spike_prior,
        growth_threshold_pct=args.spike_growth_threshold,
    )

    # ---- HTM-State pipeline call (replace with your real backend later) ----
    anomaly, state, spikes = run_htm_state_pipeline_for_demo(
        df=df,
        ema_alpha=args.ema_alpha,
        spike_cfg=spike_cfg,
    )
    # ------------------------------------------------------------------------

    # Basic sanity prints
    print()
    print("Anomaly/state summary:")
    print(f"  anomaly  min={anomaly.min():.3f}, max={anomaly.max():.3f}")
    print(f"  state    min={state.min():.3f}, max={state.max():.3f}")
    print(f"  spikes   count={int(spikes.sum())}")

    results = compute_detection_lags(
        is_boundary=df["is_boundary"].to_numpy(dtype=int),
        spikes=spikes,
        rate_hz=args.rate_hz,
        search_horizon_steps=args.search_horizon_steps,
    )

    print_detection_summary(results, rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
