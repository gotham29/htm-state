# scripts/live_demo_cyber.py

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from htm_state.htm_session import HTMSession
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTM-State live demo on UNSW-NB15-based cyber micro-drifts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="demos/cyber_demo/unsw_cyber_stream.csv",
        help="Prepared cyber stream CSV (from scripts.prepare_unsw_cyber_stream).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Simulated sample rate in Hz (controls the real-time pacing).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Time window (seconds) to show in the scrolling plots.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA alpha for HTM anomaly smoothing into 'state'.",
    )
    # spike detector knobs (seconds → steps inside the demo)
    parser.add_argument(
        "--spike-recent-sec",
        type=float,
        default=6.0,
        help="Recent window size in seconds for spike detector.",
    )
    parser.add_argument(
        "--spike-prior-sec",
        type=float,
        default=15.0,
        help="Prior window size in seconds for spike detector.",
    )
    parser.add_argument(
        "--spike-threshold-pct",
        type=float,
        default=40.0,
        help="Growth %% threshold for spike detector.",
    )
    parser.add_argument(
        "--spike-min-sep-sec",
        type=float,
        default=3.0,
        help="Minimum separation between spikes in seconds (debounce).",
    )
    parser.add_argument(
        "--enc-n-per-feature",
        type=int,
        default=64,
        help="Encoder n (bits) per scalar feature.",
    )
    parser.add_argument(
        "--enc-w-per-feature",
        type=int,
        default=8,
        help="Encoder w (active bits) per scalar feature.",
    )
    parser.add_argument(
        "--max-plot-features",
        type=int,
        default=3,
        help="Max number of network features to plot on the top axis.",
    )

    return parser.parse_args()


def sec_to_steps(sec: float, rate_hz: float) -> int:
    return max(1, int(round(sec * rate_hz)))


class CyberLiveDemo:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        plot_features: List[str],
        rate_hz: float,
        window: float,
        ema_alpha: float,
        spike_cfg: SpikeDetectorConfig,
        enc_n_per_feature: int,
        enc_w_per_feature: int,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_names = feature_names
        self.plot_features = plot_features
        self.rate_hz = rate_hz
        self.window = window

        # Build feature ranges from data for encoders
        feature_ranges: Dict[str, Dict[str, float]] = {}
        for name in feature_names:
            col = self.df[name].astype(float)
            vmin = float(col.min())
            vmax = float(col.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            feature_ranges[name] = {"min": vmin, "max": vmax}

        # SP/TM params (same as offline demo)
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

        # Core HTM session (anomaly + EMA => "mwl" = cyber state)
        self.session = HTMSession(
            feature_names=feature_names,
            enc_n_per_feature=enc_n_per_feature,
            enc_w_per_feature=enc_w_per_feature,
            sp_params=sp_params,
            tm_params=tm_params,
            seed=0,
            anomaly_ema_alpha=ema_alpha,
            feature_ranges=feature_ranges,
        )

        # Spike detector on the HTM state
        self.spike_detector = SpikeDetector(spike_cfg)

        # Buffers for plotting
        self.ts: List[float] = []
        self.feature_buffers: Dict[str, List[float]] = {f: [] for f in plot_features}
        self.states: List[float] = []

        # Drift boundaries (ground truth) from the dataframe
        self.drift_times: List[float] = [
            float(row["t"])
            for _, row in self.df.iterrows()
            if "is_drift_boundary" in self.df.columns and int(row["is_drift_boundary"]) == 1
        ]

        # We'll redraw these as vertical lines each frame
        self.drift_artists = []

        # Spike buffers
        self.spike_ts: List[float] = []
        self.spike_states: List[float] = []

        # Matplotlib figure
        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6)
        )
        self.fig.suptitle("HTM-State Live Cyber Demo (UNSW-NB15 micro-drifts)")

        # Lines for plotted features
        self.feature_lines = []
        for feat in plot_features:
            (line,) = self.ax_top.plot([], [], label=feat)
            self.feature_lines.append(line)

        # State line and spike markers
        (self.state_line,) = self.ax_bottom.plot([], [], label="HTM cyber-state", linestyle="-")
        (self.spike_line,) = self.ax_bottom.plot([], [], "o", markersize=6, label="drift spike")

        self.ax_top.set_ylabel("Network features")
        self.ax_bottom.set_ylabel("HTM cyber-state")
        self.ax_bottom.set_xlabel("Time (s)")

        self.ax_top.legend(loc="upper left")
        self.ax_bottom.legend(loc="upper left")

        self._idx = 0

    def init_anim(self):
        for line in self.feature_lines:
            line.set_data([], [])
        self.state_line.set_data([], [])
        self.spike_line.set_data([], [])

        # clear any existing drift lines
        for ln in getattr(self, "drift_artists", []):
            ln.remove()
        self.drift_artists = []

        return (*self.feature_lines, self.state_line, self.spike_line)

    def update(self, frame):
        if self._idx >= len(self.df):
            # stop animation when we run out of data
            return (*self.feature_lines, self.state_line, self.spike_line)

        row = self.df.iloc[self._idx]
        t = float(row["t"])
        feats = {name: float(row[name]) for name in self.feature_names}

        # HTM step → anomaly + EMA ("mwl")
        out = self.session.step(feats)
        state = float(out["mwl"])

        # Spike detector on the state
        spike_res = self.spike_detector.update(state)
        spike_flag = spike_res["spike"]

        # Append to buffers
        self.ts.append(t)
        for feat in self.plot_features:
            self.feature_buffers[feat].append(float(row[feat]))
        self.states.append(state)

        if spike_flag:
            self.spike_ts.append(t)
            self.spike_states.append(state)

        # Sliding time window
        t_min = t - self.window
        start_idx = 0
        for j in range(len(self.ts)):
            if self.ts[j] >= t_min:
                start_idx = j
                break

        ts_window = self.ts[start_idx:]
        states_window = self.states[start_idx:]

        # Draw ground-truth drift boundaries as vertical dashed lines in the current window
        for ln in self.drift_artists:
            ln.remove()
        self.drift_artists = []

        if ts_window:
            t_start, t_end = ts_window[0], ts_window[-1]
            for dt in self.drift_times:
                if t_start <= dt <= t_end:
                    ln = self.ax_bottom.axvline(
                        dt, color="red", linestyle="--", alpha=0.4, label="_drift_gt"
                    )
                    self.drift_artists.append(ln)

        feat_windows = {
            f: vals[start_idx:] for f, vals in self.feature_buffers.items()
        }

        # Window spikes
        spike_ts_window: List[float] = []
        spike_states_window: List[float] = []
        for tt, ss in zip(self.spike_ts, self.spike_states):
            if tt >= t_min:
                spike_ts_window.append(tt)
                spike_states_window.append(ss)

        # Update line data
        for i, line in enumerate(self.feature_lines):
            feat = self.plot_features[i]
            line.set_data(ts_window, feat_windows[feat])

        self.state_line.set_data(ts_window, states_window)
        self.spike_line.set_data(spike_ts_window, spike_states_window)

        # Update axes limits
        if ts_window:
            self.ax_top.set_xlim(ts_window[0], ts_window[-1])
            self.ax_top.relim()
            self.ax_top.autoscale_view()

            self.ax_bottom.set_xlim(ts_window[0], ts_window[-1])
            self.ax_bottom.relim()
            self.ax_bottom.autoscale_view()

        self._idx += 1

        # Real-time pacing
        time.sleep(1.0 / self.rate_hz)

        return (*self.feature_lines, self.state_line, self.spike_line, *self.drift_artists)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Metadata columns
    meta_cols = {"t", "label", "attack_cat", "phase_id", "is_drift_boundary"}
    feature_names = [c for c in df.columns if c not in meta_cols]

    if not feature_names:
        raise ValueError("No numeric feature columns found for HTM input.")

    # Choose a small set of features for plotting (top panel)
    preferred_order = ["rate", "sload", "dload", "sbytes", "dbytes"]
    plot_features: List[str] = []
    for name in preferred_order:
        if name in feature_names and name not in plot_features:
            plot_features.append(name)
        if len(plot_features) >= args.max_plot_features:
            break

    if not plot_features:
        # fallback: just use the first few feature columns
        plot_features = feature_names[: args.max_plot_features]

    print(f"Using HTM features: {feature_names}")
    print(f"Plotting features: {plot_features}")

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec, args.rate_hz),
        min_delta=0.0,
        eps=1e-3,
    )

    demo = CyberLiveDemo(
        df=df,
        feature_names=feature_names,
        plot_features=plot_features,
        rate_hz=args.rate_hz,
        window=args.window,
        ema_alpha=args.ema_alpha,
        spike_cfg=spike_cfg,
        enc_n_per_feature=args.enc_n_per_feature,
        enc_w_per_feature=args.enc_w_per_feature,
    )

    anim = FuncAnimation(
        demo.fig,
        demo.update,
        init_func=demo.init_anim,
        interval=1,  # actual pacing via time.sleep inside update()
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
