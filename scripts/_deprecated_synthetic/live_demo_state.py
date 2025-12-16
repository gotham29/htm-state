from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from htm_state.engine import (
    StateEngine,
    StateEngineConfig,
    BaselineBackend,
    HTMBackend,
)
from htm_state.htm_session import HTMSession
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig
from htm_state.viz_helpers import TruthLagOverlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTM State live demo (synthetic workload)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="demos/workload_demo/synthetic_workload.csv",
        help=(
            "Path to CSV file with columns: t, control_x, control_y "
            "(default: synthetic workload)."
        ),
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=6.67,
        help="Simulated sample rate in Hz (controls the real-time feel).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Time window (seconds) to show in the scrolling plots.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA alpha for state smoothing.",
    )
    # optional tweak knobs for spike detector in steps (we’ll compute defaults from rate-hz)
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
    parser.add_argument(
        "--backend",
        type=str,
        default="htm",
        choices=["baseline", "htm"],
        help="Which backend to use for anomaly: 'baseline' or 'htm'.",
    )

    return parser.parse_args()


class LiveDemo:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        state_cfg: StateEngineConfig,
        rate_hz: float,
        window: float,
        spike_cfg: SpikeDetectorConfig,
        backend_kind: str = "baseline",
    ):
        self.df = df.reset_index(drop=True)
        self.feature_names = feature_names
        self.rate_hz = rate_hz
        self.window = window

        # Core state engine (baseline or HTM depending on backend_kind)
        if backend_kind == "baseline":
            backend = BaselineBackend(feature_names=feature_names)
        else:
            # Construct HTMSession with the same params you used in offline demo
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

            htm_session = HTMSession(
                feature_names=feature_names,
                enc_n_per_feature=enc_n_per_feature,
                enc_w_per_feature=enc_w_per_feature,
                sp_params=sp_params,
                tm_params=tm_params,
                seed=42,
                anomaly_ema_alpha=0.2,
                feature_ranges=None,
            )
            backend = HTMBackend(htm_session)

        self.engine = StateEngine(config=state_cfg, backend=backend)

        # Spike detector over the scalar state signal
        self.spike_detector = SpikeDetector(spike_cfg)

        # buffers for plotting
        self.ts: List[float] = []
        self.controls: List[List[float]] = [[] for _ in feature_names]
        self.states: List[float] = []
        self.growth_pct: List[float] = []

        # spike times / values (for plotting markers)
        self.spike_ts: List[float] = []
        self.spike_states: List[float] = []

        # set up matplotlib figure
        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6)
        )
        self.fig.suptitle("HTM State Live Demo (Synthetic Workload)")

        # lines for each control feature
        self.control_lines = []
        for name in feature_names:
            (line,) = self.ax_top.plot([], [], label=name)
            self.control_lines.append(line)

        # state line
        (self.state_line,) = self.ax_bottom.plot([], [], label="state", linestyle="-")
        # spike markers (scatter-like)
        (self.spike_line,) = self.ax_bottom.plot(
            [], [], "o", markersize=6, label="spike"
        )

        self.ax_top.set_ylabel("Control inputs")
        self.ax_bottom.set_ylabel("State")
        self.ax_bottom.set_xlabel("Time (s)")

        self.ax_top.legend(loc="upper left")
        self.ax_bottom.legend(loc="upper left")

        # Ground-truth workload transition (synthetic demo: midpoint toggle)
        mid_idx = len(self.df) // 2
        self.toggle_time: float = float(self.df["t"].iloc[mid_idx])
        self.det_time: Optional[float] = None
        self.det_lag_sec: Optional[float] = None

        # Overlay helper for truth + lag visualization
        self.overlay = TruthLagOverlay(self.ax_bottom)

        # animation index
        self._idx = 0

    def init_anim(self):
        for line in self.control_lines:
            line.set_data([], [])
        self.state_line.set_data([], [])
        self.spike_line.set_data([], [])

        # clear any truth/lag overlays
        self.overlay.clear()

        return (*self.control_lines, self.state_line, self.spike_line)

    def update(self, frame):
        if self._idx >= len(self.df):
            # stop animation when we run out of data
            return (*self.control_lines, self.state_line, self.spike_line)

        row = self.df.iloc[self._idx]
        t = float(row["t"])
        feat_vals = {name: float(row[name]) for name in self.feature_names}

        # Run through the state engine: anomaly + EMA state
        out = self.engine.step(feat_vals)
        state = float(out["state"])

        # Run spike detector on the state
        spike_res = self.spike_detector.update(state)
        spike_flag = spike_res["spike"]
        growth_pct = spike_res["growth_pct"]

        # If this is the first spike after the true transition, record detection time
        if spike_flag and self.det_time is None and t >= self.toggle_time:
            self.det_time = t
            self.det_lag_sec = self.det_time - self.toggle_time

        # append to buffers
        self.ts.append(t)
        for i, name in enumerate(self.feature_names):
            self.controls[i].append(feat_vals[name])
        self.states.append(state)
        self.growth_pct.append(growth_pct if growth_pct is not None else 0.0)

        if spike_flag:
            self.spike_ts.append(t)
            self.spike_states.append(state)

        # maintain a sliding time window
        t_min = t - self.window
        # find first index within window
        start_idx = 0
        for j in range(len(self.ts)):
            if self.ts[j] >= t_min:
                start_idx = j
                break

        ts_window = self.ts[start_idx:]
        controls_window = [c[start_idx:] for c in self.controls]
        states_window = self.states[start_idx:]

        # also window spikes
        spike_ts_window = []
        spike_states_window = []
        for tt, ss in zip(self.spike_ts, self.spike_states):
            if tt >= t_min:
                spike_ts_window.append(tt)
                spike_states_window.append(ss)

        # update line data
        for i, line in enumerate(self.control_lines):
            line.set_data(ts_window, controls_window[i])
        self.state_line.set_data(ts_window, states_window)
        self.spike_line.set_data(spike_ts_window, spike_states_window)

        # truth + lag overlay (single transition)
        overlay_artists: List = []
        if ts_window and states_window:
            overlay_artists = self.overlay.draw_single(
                toggle_time=self.toggle_time,
                det_time=self.det_time,
                lag_sec=self.det_lag_sec,
                ts_window=ts_window,
                states_window=states_window,
                truth_label="true workload transition",
                lag_label="detection lag",
            )

        # update axes limits
        if ts_window:
            self.ax_top.set_xlim(ts_window[0], ts_window[-1])
            self.ax_top.relim()
            self.ax_top.autoscale_view()

            self.ax_bottom.set_xlim(ts_window[0], ts_window[-1])
            self.ax_bottom.relim()
            self.ax_bottom.autoscale_view()

        self._idx += 1

        # simulate real-time pacing
        time.sleep(1.0 / self.rate_hz)

        return (
            *self.control_lines,
            self.state_line,
            self.spike_line,
            *overlay_artists,
        )


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Expecting at least columns: t, control_x, control_y
    feature_names = [c for c in df.columns if c != "t"]

    state_cfg = StateEngineConfig(ema_alpha=args.ema_alpha)

    # Convert spike detector windows in seconds -> steps
    def sec_to_steps(sec: float) -> int:
        return max(1, int(round(sec * args.rate_hz)))

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec),
        prior_window=sec_to_steps(args.spike_prior_sec),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec),
        min_delta=0.0,
        eps=1e-3,
    )

    demo = LiveDemo(
        df=df,
        feature_names=feature_names,
        state_cfg=state_cfg,
        rate_hz=args.rate_hz,
        window=args.window,
        spike_cfg=spike_cfg,
        backend_kind=args.backend,
    )

    anim = FuncAnimation(
        demo.fig,
        demo.update,
        init_func=demo.init_anim,
        interval=1,  # actual pacing is done via time.sleep in update()
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
