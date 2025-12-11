#!/usr/bin/env python3
"""
Live visualization for Demo 4 — Manufacturing Process Drift Detection

This reuses the same simple anomaly → state → spike pipeline as
offline_demo_manufacturing.py, then streams it as a live plot.

For flexibility, it automatically uses all numeric columns except:
- 'timestep'
- 'is_boundary'
as the feature set driving anomaly/state.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Spike / state helpers (mirrored from other demos)
# ---------------------------------------------------------------------


@dataclass
class SpikeConfig:
    """
    Simple growth-based spike detector config.

    Threshold is expressed as a % growth between recent and prior windows.
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


def run_state_pipeline_for_demo(
    df: pd.DataFrame,
    features: List[str],
    ema_alpha: float,
    spike_cfg: SpikeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For this synthetic demo, we approximate anomaly as the mean absolute
    z-score across the selected manufacturing process features.
    Then:
      anomaly -> EMA -> state
      state   -> spike detector
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Manufacturing live demo: missing expected columns {missing}. "
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
# CLI + live plotting
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Live HTM manufacturing drift demo")
    parser.add_argument("--csv", required=True, help="Path to manufacturing line stream CSV")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Streaming rate (Hz); controls playback speed and spike windows",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Visualize every k-th timestep (1 = show all steps). "
             "Larger values speed up the demo and shrink GIFs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on number of timesteps to stream (for GIFs / faster runs)",
    )
    parser.add_argument("--spike-recent-sec", type=float, default=3.0)
    parser.add_argument("--spike-prior-sec", type=float, default=6.0)
    parser.add_argument("--spike-threshold-pct", type=float, default=40.0)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading manufacturing line stream: {args.csv}")
    df = pd.read_csv(args.csv)
    total_len = len(df)
    print(f"Loaded {total_len} timesteps.")
    print(f"Columns: {list(df.columns)}")

    # auto-select numeric feature columns (exclude timestep + is_boundary)
    candidate_cols = [
        c
        for c in df.columns
        if c not in ("timestep", "is_boundary")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidate_cols:
        raise ValueError(
            "No numeric feature columns found for manufacturing demo "
            "(expected numeric columns besides 'timestep' / 'is_boundary')."
        )

    features = candidate_cols
    print(f"Using features for HTM-State pipeline: {features}")

    # For plotting, restrict to a smaller, readable subset if available.
    preferred_order = [
        "vibration_energy",
        "spindle_load",
        "feed_rate",
        "line_speed",
        "part_time",
        "temperature",
    ]
    plot_features = [c for c in preferred_order if c in features][:4]
    if not plot_features:
        # Fallback: just take the first few numeric features
        plot_features = features[:4]
    print(f"Showing features in top plot: {plot_features}")

    spike_cfg = SpikeConfig.from_seconds(
        recent_sec=args.spike_recent_sec,
        prior_sec=args.spike_prior_sec,
        threshold_pct=args.spike_threshold_pct,
        rate_hz=args.rate_hz,
    )

    anomaly, state, spikes = run_state_pipeline_for_demo(
        df,
        features=features,
        ema_alpha=0.2,
        spike_cfg=spike_cfg,
    )

    is_boundary = df.get("is_boundary", pd.Series([0] * len(df))).to_numpy(dtype=int)

    rate_dt = 1.0 / args.rate_hz

    # -----------------------------------------------------------------
    # Matplotlib live figure setup
    # -----------------------------------------------------------------
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    top_ax, bot_ax = axs

    # number of points in rolling window (smaller for clearer plots)
    window = 120

    buffer_feat: Dict[str, List[float]] = {col: [] for col in plot_features}
    buffer_state: List[float] = []
    buffer_spikes: List[bool] = []
    buffer_boundary: List[int] = []

    n_steps = len(df) if args.max_steps is None else min(args.max_steps, len(df))
    step_stride = max(1, int(args.step_stride))
    print(f"Streaming up to {n_steps} timesteps (of {total_len}) "
          f"with step_stride={step_stride}.")

    # Precompute how many frames we'll actually draw (for nicer progress prints)
    total_frames = len(range(0, n_steps, step_stride))

    frame_idx = 0
    for t in range(0, n_steps, step_stride):
        # Lightweight progress print so you know how close you are to, e.g., step 1600
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"Frame {frame_idx+1}/{total_frames} (timestep {t}/{n_steps-1})")

        # Append current feature values for this frame (for plotting)
        for col in plot_features:
            buffer_feat[col].append(float(df.at[t, col]))

        # Append state / spike / boundary for this frame
        buffer_state.append(float(state[t]))
        buffer_spikes.append(bool(spikes[t]))
        buffer_boundary.append(int(is_boundary[t]))

        # Enforce rolling window
        if len(buffer_state) > window:
            buffer_state.pop(0)
            buffer_spikes.pop(0)
            buffer_boundary.pop(0)
            for col in plot_features:
                buffer_feat[col].pop(0)

        # Clear axes each frame
        top_ax.cla()
        bot_ax.cla()

        # ------------------ Top panel: signals -----------------------
        # Split plotted features into "temperature-like" and others,
        # so we can put temp on a second y-axis if present.
        temp_feature = next((c for c in plot_features if "temp" in c.lower()), None)
        primary_features = [c for c in plot_features if c != temp_feature]

        # Plot primary features on the left y-axis
        for col in primary_features:
            top_ax.plot(buffer_feat[col], label=col, alpha=0.8)

        top_ax.set_title("Manufacturing Signals")
        top_ax.set_xlabel("Time (relative steps)")
        top_ax.set_ylabel("Normalized units")

        # Optional second y-axis for temperature-like feature
        if temp_feature is not None:
            temp_ax = top_ax.twinx()
            temp_ax.plot(buffer_feat[temp_feature], alpha=0.85, color="black", linewidth=2.0)

            # Completely suppress any visible axis elements
            temp_ax.set_ylabel("")               # Remove axis label
            temp_ax.set_yticklabels([])          # Remove tick labels
            temp_ax.tick_params(axis="y", width=0)  # Remove tick marks

            # Build combined legend from both axes
            lines_left, labels_left = top_ax.get_legend_handles_labels()
            lines_right, labels_right = temp_ax.get_legend_handles_labels()
            top_ax.legend(
                lines_left + lines_right,
                labels_left + labels_right,
                loc="upper left",
            )
        else:
            top_ax.legend(loc="upper left")

        # ------------------ Bottom panel: state + spikes -------------
        x_vals = list(range(len(buffer_state)))
        bot_ax.plot(x_vals, buffer_state, label="HTM state (EMA of anomaly)", color="blue")

        # spikes
        for i, is_spike in enumerate(buffer_spikes):
            if is_spike:
                bot_ax.scatter(i, buffer_state[i], color="orange", s=40, zorder=3)

        # boundaries + lag bars
        boundary_indices = [i for i, b in enumerate(buffer_boundary) if b]
        for b_idx in boundary_indices:
            bot_ax.axvline(x=b_idx, linestyle="--", color="red", alpha=0.8)
            # lag: first spike after boundary in the visible window
            after = next((i for i in range(b_idx, len(buffer_spikes)) if buffer_spikes[i]), None)
            if after is not None:
                bot_ax.plot(
                    [b_idx, after],
                    [buffer_state[b_idx], buffer_state[after]],
                    color="magenta",
                    linewidth=2.5,
                    zorder=2,
                )

        bot_ax.set_title("HTM-State (synthetic) + Detected Process Drift")
        bot_ax.set_xlabel("Time (relative steps)")
        bot_ax.set_ylabel("State")
        bot_ax.legend(loc="upper left")

        plt.pause(rate_dt)
        frame_idx += 1

    print("\nFinished streaming.\n")
    # keep window open if user wants to inspect
    plt.tight_layout()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
