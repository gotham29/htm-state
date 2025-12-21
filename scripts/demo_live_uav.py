from __future__ import annotations
from dataclasses import dataclass
import argparse
import time
import os
from pathlib import Path

from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class _RunOutputs:
    ts: List[float]
    state_series: List[float]
    thr_series: List[Optional[float]]
    spike_steps: List[int]
    persist_steps: List[int]
    boundary_time: Optional[float]
    first_sustained_time: Optional[float]
    detection_lag_sec: Optional[float]
    persist_is_frozen: bool
    frozen_step: Optional[int]

def _compute_series_fast(
    df: pd.DataFrame,
    feature_names: List[str],
    rate_hz: float,
    boundary_step: Optional[int],
    boundary_time: Optional[float],
    session: "HTMSession",
    spike_detector: "SpikeDetector",
    persistence_detector: "PersistenceDetector",
    freeze_tm_after_boundary: bool,
    freeze_persist_baseline_at_boundary: bool,
    reset_detectors_at_boundary: bool,
) -> _RunOutputs:
    ts: List[float] = []
    state_series: List[float] = []
    thr_series: List[Optional[float]] = []
    spike_steps: List[int] = []
    persist_steps: List[int] = []
    persist_is_frozen: bool = False
    frozen_step: Optional[int] = None

    first_sustained_time: Optional[float] = None
    detection_lag_sec: Optional[float] = None

    for i, row in df.iterrows():
        t = float(row["t_sec"])
        feats = {name: float(row[name]) for name in feature_names}

        # Freeze persistence baseline at boundary (if requested).
        if freeze_persist_baseline_at_boundary and boundary_step is not None and i == boundary_step:
            persistence_detector.freeze_baseline()
            persist_is_frozen = True
            frozen_step = i

        learn = True
        if freeze_tm_after_boundary and boundary_step is not None and i >= boundary_step:
            learn = False

        # Optional: reset detectors exactly at boundary.
        if reset_detectors_at_boundary and boundary_step is not None and i == boundary_step:
            if hasattr(spike_detector, "reset"):
                spike_detector.reset()
            if hasattr(persistence_detector, "freeze_baseline"):
                persistence_detector.freeze_baseline()
                persist_is_frozen = True
                frozen_step = i

        out = session.step(feats, learn=learn)
        state = float(out["mwl"])

        spike_res = spike_detector.update(state)
        if int(spike_res.get("spike", 0)) == 1:
            spike_steps.append(i)

        pers_res = persistence_detector.update(state)
        thr = pers_res.get("thr", pers_res.get("threshold"))
        thr_series.append(float(thr) if thr is not None else None)

        is_sustained = bool(pers_res.get("sustained", pers_res.get("persistent", False)))
        if is_sustained:
            persist_steps.append(i)
            if boundary_time is not None and boundary_step is not None and i >= boundary_step and first_sustained_time is None:
                first_sustained_time = float(t)
                detection_lag_sec = first_sustained_time - float(boundary_time)

        ts.append(t)
        state_series.append(state)

    return _RunOutputs(
        ts=ts,
        state_series=state_series,
        thr_series=thr_series,
        spike_steps=spike_steps,
        persist_steps=persist_steps,
        boundary_time=boundary_time,
        first_sustained_time=first_sustained_time,
        detection_lag_sec=detection_lag_sec,
        persist_is_frozen=persist_is_frozen,
        frozen_step=frozen_step,
    )

def _render_final(
    csv_path: Path,
    out_png: Path | None,
    dpi: int,
    feature_names: List[str],
    rate_hz: float,
    ema_alpha: float,
    enc_n_per_feature: int,
    enc_w_per_feature: int,
    freeze_tm_after_boundary: bool,
    reset_detectors_at_boundary: bool,
    freeze_persist_baseline_at_boundary: bool,
    spike_recent_sec: float,
    spike_prior_sec: float,
    spike_threshold_pct: float,
    spike_min_sep_sec: float,
    spike_min_delta: float,
    elev_baseline_sec: float,
    elev_k_mad: float,
    elev_hold_sec: float,
    elev_min_sep_sec: float,
) -> None:
    df = pd.read_csv(csv_path)
    if "t_sec" not in df.columns:
        raise ValueError("CSV missing required column: t_sec")
    if "is_boundary" not in df.columns:
        raise ValueError("CSV missing required column: is_boundary")

    for f in feature_names:
        if f not in df.columns:
            raise ValueError(f"Requested feature '{f}' not in CSV columns: {list(df.columns)}")

    boundary_idxs = df.index[df["is_boundary"].astype(int) == 1].tolist()
    boundary_step: Optional[int] = int(boundary_idxs[0]) if boundary_idxs else None
    boundary_time: Optional[float] = float(df.loc[boundary_step, "t_sec"]) if boundary_step is not None else None

    session = build_session(
        df=df,
        feature_names=feature_names,
        ema_alpha=ema_alpha,
        enc_n_per_feature=enc_n_per_feature,
        enc_w_per_feature=enc_w_per_feature,
    )

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(spike_recent_sec, rate_hz),
        prior_window=sec_to_steps(spike_prior_sec, rate_hz),
        threshold_pct=spike_threshold_pct,
        min_delta=spike_min_delta,
    )
    spike_detector = SpikeDetector(spike_cfg)

    pers_cfg = PersistenceDetectorConfig(
        baseline_window=sec_to_steps(elev_baseline_sec, rate_hz),
        k_mad=elev_k_mad,
        hold_steps=sec_to_steps(elev_hold_sec, rate_hz),
    )
    persistence_detector = PersistenceDetector(pers_cfg)

    outs = _compute_series_fast(
        df=df,
        feature_names=feature_names,
        rate_hz=rate_hz,
        boundary_step=boundary_step,
        boundary_time=boundary_time,
        session=session,
        spike_detector=spike_detector,
        persistence_detector=persistence_detector,
        freeze_tm_after_boundary=freeze_tm_after_boundary,
        freeze_persist_baseline_at_boundary=freeze_persist_baseline_at_boundary,
        reset_detectors_at_boundary=reset_detectors_at_boundary,
    )

    # ---- final draw (single shot) ----
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"HTM-State Live Demo (UAV)\nFailure scenario: {csv_path.stem.replace('_', ' ')}")

    show_feats = feature_names[:3] if len(feature_names) >= 3 else feature_names

    # Boundary line on BOTH panels
    if outs.boundary_time is not None:
        ax0.axvline(outs.boundary_time, linestyle=":", linewidth=2)
        ax1.axvline(outs.boundary_time, linestyle=":", linewidth=2)

    # signals panel (display-only z-score normalization)
    for f in show_feats:
        ax0.plot(outs.ts, zscore(df[f]).values, label=f)
    ax0.set_ylabel("signals (z-score)")
    ax0.set_ylim(*fixed_ylim_zscore())
    ax0.legend(loc="upper left", ncol=min(3, len(show_feats)))

    # state panel
    ax1.set_ylim(*fixed_state_ylim())
    ax1.plot(outs.ts, outs.state_series, linewidth=2.75, color="C0", label="HTM-State (mwl)")

    if any(v is not None for v in outs.thr_series):
        thr_y = [v if v is not None else float("nan") for v in outs.thr_series]
        ax1.plot(
            outs.ts,
            thr_y,
            linestyle="--",
            linewidth=2,
            alpha=0.9 if outs.persist_is_frozen else 0.6,
            label="elev_thr (median + k*MAD)",
        )
        if outs.persist_is_frozen and outs.frozen_step is not None and outs.frozen_step < len(outs.ts):
            ax1.text(
                outs.ts[outs.frozen_step],
                fixed_state_ylim()[1] * 0.90,
                "persist baseline frozen",
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                alpha=0.8,
            )
    if outs.spike_steps:
        ax1.scatter(
            [outs.ts[j] for j in outs.spike_steps if j < len(outs.ts)],
            [outs.state_series[j] for j in outs.spike_steps if j < len(outs.state_series)],
            marker="^",
            label="spike",
            zorder=5,
        )

    if outs.persist_steps:
        ax1.scatter(
            [outs.ts[j] for j in outs.persist_steps if j < len(outs.ts)],
            [outs.state_series[j] + 0.03 for j in outs.persist_steps if j < len(outs.state_series)],
            marker="s",
            label="sustained",
            zorder=6,
        )

    if outs.boundary_time is not None:
        ax1.text(
            outs.boundary_time,
            fixed_state_ylim()[1] * 0.95,
            "Failure injected",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            alpha=0.8,
        )

    ax1.set_xlabel("t_sec")
    ax1.set_ylabel("state")
    ax1.legend(loc="upper left")

    # metrics summary box
    if outs.boundary_time is None:
        summary_lines = ["Metrics summary", "boundary: None", "first sustained: None", "lag: None"]
    else:
        summary_lines = [
            "Metrics summary",
            f"boundary: {outs.boundary_time:.1f}s",
            (f"first sustained: {outs.first_sustained_time:.1f}s" if outs.first_sustained_time is not None else "first sustained: None"),
            (f"lag: {outs.detection_lag_sec:.1f}s" if outs.detection_lag_sec is not None else "lag: None"),
        ]
    ax1.text(
        0.99,
        0.02,
        "\n".join(summary_lines),
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.5),
    )

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[demo_live_uav] wrote: {out_png}")
    plt.close(fig)

import sys
from pathlib import Path

# Allow running as: python scripts/offline_demo_uav.py
# by ensuring repo root is on sys.path for `import htm_state.*`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from htm_state.htm_session import HTMSession
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig
from htm_state.persistence_detector import PersistenceDetector, PersistenceDetectorConfig


def fixed_state_ylim():
    # Stable MWL display range across scenarios
    return (0.0, 1.6)

def fixed_ylim_zscore():
    # Stable display range for z-scored signals
    return (-4.0, 4.0)

def sec_to_steps(sec: float, rate_hz: float) -> int:
    return max(1, int(round(sec * rate_hz)))

def zscore(series: pd.Series) -> pd.Series:
    """Display-only normalization for plotting (does not affect modeling)."""
    s = series.astype(float)
    mu = float(s.mean())
    sd = float(s.std())
    return (s - mu) / (sd + 1e-6)

def default_save_path(csv_path: Path) -> Path:
    # Save alongside the CSV, same stem, .png extension
    return csv_path.with_suffix(".png")


def save_final_figure(fig: plt.Figure, save_path: Path, dpi: int) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[live_demo_uav] saved: {save_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Live HTM-State demo for ALFA UAV CSV streams")
    p.add_argument("--csv", type=str, required=True, help="Generated UAV CSV (from generate_uav_stream.py)")
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--sleep", type=float, default=0.10, help="Seconds to sleep per step (visual pacing)")

    # Offline renderer mode (single-shot, no animation)
    p.add_argument(
        "--render-final",
        action="store_true",
        help="Render a single end-state plot (no animation), then exit.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Never open a GUI window (useful for batch/offline rendering).",
    )

    p.add_argument(
        "--features",
        type=str,
        default="airspeed,climb,altitude,throttle,heading,pitch,roll,yaw",
        help="Comma-separated feature list to model (must exist in CSV).",
    )

    # HTM Session knobs (mirror offline defaults)
    p.add_argument("--ema-alpha", type=float, default=0.05)
    p.add_argument("--enc-n-per-feature", type=int, default=64)
    p.add_argument("--enc-w-per-feature", type=int, default=8)
    p.add_argument("--freeze-tm-after-boundary", action="store_true")

    # Detector boundary behavior
    p.add_argument(
        "--reset-detectors-at-boundary", action="store_true",
        help="Reset spike + persistence detectors right at boundary (improves post-boundary sensitivity).",
    )

    # Save fig
    p.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Path to save final figure PNG. Default: alongside CSV with same stem.",
    )
    p.add_argument("--save-fig-dpi", type=int, default=140)

    # Spike detector knobs
    p.add_argument("--spike-recent-sec", type=float, default=2.0)
    p.add_argument("--spike-prior-sec", type=float, default=6.0)
    p.add_argument("--spike-threshold-pct", type=float, default=60.0)
    p.add_argument("--spike-min-sep-sec", type=float, default=5.0)
    p.add_argument("--spike-min-delta", type=float, default=0.08)

    # Persistence detector knobs
    p.add_argument("--elev-baseline-sec", type=float, default=10.0, help="Baseline window length (sec)")
    p.add_argument("--elev-k-mad", type=float, default=3.0, help="Threshold = median + k*MAD")
    p.add_argument("--elev-hold-sec", type=float, default=1.0, help="Hold time above threshold (sec)")
    p.add_argument("--elev-min-sep-sec", type=float, default=2.0, help="Debounce between sustained triggers (sec)")
    p.add_argument(
        "--freeze-persist-baseline-at-boundary",
        action="store_true",
        help="Freeze persistence baseline at boundary (post-boundary compared to pre-boundary normal).",
    )
    return p.parse_args()


def build_session(df: pd.DataFrame, feature_names: List[str], ema_alpha: float,
                  enc_n_per_feature: int, enc_w_per_feature: int) -> HTMSession:
    # Robust per-file ranges (optionally avoid post-boundary "peeking")
    df_for_ranges = df
    if "is_boundary" in df.columns:
        b = df.index[df["is_boundary"].astype(int) == 1].tolist()
        if b:
            df_for_ranges = df.iloc[: int(b[0])]
    feature_ranges: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        col = df_for_ranges[name].astype(float)
        vmin = float(col.min())
        vmax = float(col.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        feature_ranges[name] = {"min": vmin, "max": vmax}

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

    session = HTMSession(
        feature_names=feature_names,
        enc_n_per_feature=enc_n_per_feature,
        enc_w_per_feature=enc_w_per_feature,
        sp_params=sp_params,
        tm_params=tm_params,
        seed=0,
        anomaly_ema_alpha=ema_alpha,
        feature_ranges=feature_ranges,
    )

    # Match your stated preference
    session.sp_learning = False
    return session


def main() -> None:
    args = parse_args()
    if args.no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    csv_path = Path(args.csv)

    # -------- FINAL (non-animated) fast path --------
    if args.render_final:
        out_png = Path(args.save_fig) if args.save_fig else default_save_path(csv_path)
        feature_names = [c.strip() for c in args.features.split(",") if c.strip()]
        _render_final(
            csv_path=csv_path,
            out_png=out_png,
            dpi=int(args.save_fig_dpi),
            feature_names=feature_names,
            rate_hz=float(args.rate_hz),
            ema_alpha=float(args.ema_alpha),
            enc_n_per_feature=int(args.enc_n_per_feature),
            enc_w_per_feature=int(args.enc_w_per_feature),
            freeze_tm_after_boundary=bool(args.freeze_tm_after_boundary),
            reset_detectors_at_boundary=bool(args.reset_detectors_at_boundary),
            freeze_persist_baseline_at_boundary=bool(args.freeze_persist_baseline_at_boundary),
            spike_recent_sec=float(args.spike_recent_sec),
            spike_prior_sec=float(args.spike_prior_sec),
            spike_threshold_pct=float(args.spike_threshold_pct),
            spike_min_sep_sec=float(args.spike_min_sep_sec),
            spike_min_delta=float(args.spike_min_delta),
            elev_baseline_sec=float(args.elev_baseline_sec),
            elev_k_mad=float(args.elev_k_mad),
            elev_hold_sec=float(args.elev_hold_sec),
            elev_min_sep_sec=float(args.elev_min_sep_sec),
        )
        return

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    save_path = Path(args.save_fig) if args.save_fig is not None else default_save_path(csv_path)

    df = pd.read_csv(csv_path)
    if "t_sec" not in df.columns:
        raise ValueError("CSV missing required column: t_sec")
    if "is_boundary" not in df.columns:
        raise ValueError("CSV missing required column: is_boundary")

    feature_names = [c.strip() for c in args.features.split(",") if c.strip()]
    for f in feature_names:
        if f not in df.columns:
            raise ValueError(f"Requested feature '{f}' not in CSV columns: {list(df.columns)}")

    boundary_idxs = df.index[df["is_boundary"].astype(int) == 1].tolist()
    boundary_step: Optional[int] = int(boundary_idxs[0]) if boundary_idxs else None
    boundary_time: Optional[float] = float(df.loc[boundary_step, "t_sec"]) if boundary_step is not None else None

    first_sustained_step: Optional[int] = None
    first_sustained_time: Optional[float] = None
    detection_lag_sec: Optional[float] = None

    print(f"CSV: {csv_path}")
    print(f"Using features: {feature_names}")
    if boundary_step is None:
        print("Boundary: None (no-failure scenario)")
    else:
        print(f"Boundary: step={boundary_step}, t_sec={boundary_time:.3f}")

    session = build_session(
        df=df,
        feature_names=feature_names,
        ema_alpha=args.ema_alpha,
        enc_n_per_feature=args.enc_n_per_feature,
        enc_w_per_feature=args.enc_w_per_feature,
    )

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        min_delta=args.spike_min_delta,
    )
    spike_detector = SpikeDetector(spike_cfg)

    pers_cfg = PersistenceDetectorConfig(
        baseline_window=sec_to_steps(args.elev_baseline_sec, args.rate_hz),
        k_mad=args.elev_k_mad,
        hold_steps=sec_to_steps(args.elev_hold_sec, args.rate_hz),
    )
    persistence_detector = PersistenceDetector(pers_cfg)

    # --- Live plot setup ---
    plt.ion()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # NOTE: signals panel is already display-normalized via z-score (does not affect modeling).
    fig.suptitle(
        f"HTM-State Live Demo (UAV)\nFailure scenario: {csv_path.stem.replace('_', ' ')}"
    )

    # Track whether the persistence threshold is frozen (for visualization)
    persist_is_frozen: bool = False

    # Choose a small subset to show as “signals”
    show_feats = feature_names[:3] if len(feature_names) >= 3 else feature_names

    ts: List[float] = []
    state_series: List[float] = []
    thr_series: List[Optional[float]] = []
    spike_steps: List[int] = []
    persist_steps: List[int] = []
    frozen_step: Optional[int] = None

    # Keep a handle so we can update/refresh the summary cleanly.
    metrics_text_artist = None

    for i, row in df.iterrows():
        t = float(row["t_sec"])
        feats = {name: float(row[name]) for name in feature_names}

        # Freeze persistence baseline right at the boundary (if requested).
        if (
            args.freeze_persist_baseline_at_boundary
            and boundary_step is not None
            and i == boundary_step
        ):
            persistence_detector.freeze_baseline()
            persist_is_frozen = True
            frozen_step = i

        learn = True
        if args.freeze_tm_after_boundary and boundary_step is not None and i >= boundary_step:
            learn = False

        # Optional: reset detectors exactly at boundary (helps avoid pre-boundary history masking post)
        if args.reset_detectors_at_boundary and boundary_step is not None and i == boundary_step:
            if hasattr(spike_detector, "reset"):
                spike_detector.reset()
            # For persistence, “reset detectors at boundary” should mean:
            # compare post-boundary against pre-boundary normal -> freeze baseline threshold.
            if hasattr(persistence_detector, "freeze_baseline"):
                persistence_detector.freeze_baseline()
                persist_is_frozen = True
                frozen_step = i

        out = session.step(feats, learn=learn)
        state = float(out["mwl"])

        spike_res = spike_detector.update(state)
        if int(spike_res["spike"]) == 1:
            spike_steps.append(i)

        pers_res = persistence_detector.update(state)

        thr = pers_res.get("thr", pers_res.get("threshold"))
        thr_series.append(float(thr) if thr is not None else None)

        is_sustained = bool(pers_res.get("sustained", pers_res.get("persistent", False)))
        if is_sustained:
            persist_steps.append(i)

            if (
                boundary_step is not None
                and i >= boundary_step
                and first_sustained_step is None
            ):
                first_sustained_step = i
                first_sustained_time = float(t)
                detection_lag_sec = first_sustained_time - float(boundary_time)

        ts.append(t)
        state_series.append(state)

        # --- redraw ---
        ax0.clear()
        ax1.clear()

        # Boundary line on BOTH panels (more intuitive for domain viewers)
        if boundary_time is not None:
            ax0.axvline(boundary_time, linestyle=":", linewidth=2)
            ax1.axvline(boundary_time, linestyle=":", linewidth=2)

        # signals panel (display-only z-score normalization)
        for f in show_feats:
            s = df.loc[:i, f]
            ax0.plot(ts, zscore(s).values, label=f)
        ax0.set_ylabel("signals (z-score)")
        ax0.set_ylim(*fixed_ylim_zscore())
        ax0.legend(loc="upper left", ncol=min(3, len(show_feats)))

        # state panel
        ax1.set_ylim(*fixed_state_ylim())
        # Always draw the HTM state line (explicit and first, so it can’t “disappear”)
        ax1.plot(ts, state_series, linewidth=2.75, color="C0", label="HTM-State (mwl)")

        # persistence threshold (when available)
        if any(v is not None for v in thr_series):
            thr_y = [v if v is not None else float("nan") for v in thr_series]
            # If we froze, make the post-freeze portion look “locked”
            # (simple but effective cue for specialists)
            ax1.plot(
                ts,
                thr_y,
                linestyle="--",
                linewidth=2,
                alpha=0.9 if persist_is_frozen else 0.6,
                label="elev_thr (median + k*MAD)"
            )

            if persist_is_frozen and frozen_step is not None and frozen_step < len(ts):
                ax1.text(
                    ts[frozen_step],
                    fixed_state_ylim()[1] * 0.90,
                    "persist baseline frozen",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontsize=8,
                    alpha=0.8,
                )

        # spike markers
        if spike_steps:
            ax1.scatter([ts[j] for j in spike_steps if j < len(ts)],
                        [state_series[j] for j in spike_steps if j < len(state_series)],
                        marker="^", label="spike", zorder=5)

        # sustained elevation trigger markers
        if persist_steps:
            ax1.scatter(
                [ts[j] for j in persist_steps if j < len(ts)],
                [state_series[j] + 0.03 for j in persist_steps if j < len(state_series)],
                marker="s",
                label="sustained",
                zorder=6,
            )

        # boundary label (keep only on state panel to reduce clutter)
        if boundary_time is not None:
            ax1.text(boundary_time, fixed_state_ylim()[1] * 0.95, "Failure injected",
                     rotation=90, verticalalignment="top", horizontalalignment="right",
                     fontsize=9, alpha=0.8)

        ax1.set_xlabel("t_sec")
        ax1.set_ylabel("state")
        ax1.legend(loc="upper left")

        # ---- metrics summary (rendered on-plot) ----
        # Build concise lines that domain viewers can read at a glance.
        if boundary_time is None:
            summary_lines = [
                "Metrics summary",
                "boundary: None",
                "first sustained: None",
                "lag: None",
            ]
        else:
            summary_lines = [
                "Metrics summary",
                f"boundary: {boundary_time:.1f}s",
                (
                    f"first sustained: {first_sustained_time:.1f}s"
                    if first_sustained_time is not None
                    else "first sustained: None"
                ),
                (
                    f"lag: {detection_lag_sec:.1f}s"
                    if detection_lag_sec is not None
                    else "lag: None"
                ),
            ]

        # Remove previous artist (no try/except; this should always be valid if present)
        if metrics_text_artist is not None:
            metrics_text_artist.remove()

        metrics_text_artist = ax1.text(
            0.99, 0.02,
            "\n".join(summary_lines),
            transform=ax1.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.5),
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(args.sleep)

    # Ensure summary exists on final frame even if loop never drew (edge cases).
    if metrics_text_artist is None:
        if boundary_time is None:
            summary_lines = [
                "Metrics summary",
                "boundary: None",
                "first sustained: None",
                "lag: None",
            ]
        else:
            summary_lines = [
                "Metrics summary",
                f"boundary: {boundary_time:.1f}s",
                (
                    f"first sustained: {first_sustained_time:.1f}s"
                    if first_sustained_time is not None
                    else "first sustained: None"
                ),
                (
                    f"lag: {detection_lag_sec:.1f}s"
                    if detection_lag_sec is not None
                    else "lag: None"
                ),
            ]
        metrics_text_artist = ax1.text(
            0.99, 0.02,
            "\n".join(summary_lines),
            transform=ax1.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.5),
        )
        fig.canvas.draw()

    # Save final frame on normal exit (no try/except).
    save_final_figure(fig, save_path, dpi=int(args.save_fig_dpi))

    print("\nDone.")
    print("\n=== Detection Summary ===")
    if boundary_time is not None:
        print(f"Boundary time (sec): {boundary_time:.3f}")
    else:
        print("Boundary time (sec): None")

    if first_sustained_time is not None:
        print(f"First sustained detection (sec): {first_sustained_time:.3f}")
        print(f"Detection lag (sec): {detection_lag_sec:.3f}")
    else:
        print("First sustained detection: None")
        print("Detection lag (sec): None")

    if not args.no_show:
        plt.show()
    else:
        # headless mode: if saving, you already saved; just close cleanly
        plt.close("all")

if __name__ == "__main__":
    main()

"""
# BASELINE
python scripts/demo_live_uav.py \
  --csv demos/uav/generated/streams/no_failure/carbonZ_2018-10-18-11-08-24_no_failure.csv \
  --sleep 0.1

# TYPICAL SPIKE DETECTION
python scripts/demo_live_uav.py \
  --csv demos/uav/generated/streams/engine_failure/carbonZ_2018-07-30-17-36-35_engine_failure_with_emr_traj.csv \
  --sleep 0.1

# HARD SPIKE
python scripts/demo_live_uav.py \
  --csv demos/uav/generated/streams/engine_failure/carbonZ_2018-10-05-15-55-10_engine_failure_with_emr_traj.csv \
  --sleep 0.1

# SUSTAINED 
python scripts/demo_live_uav.py \
  --csv demos/uav/generated/streams/elevator_failure/carbonZ_2018-09-11-14-41-51_elevator_failure.csv \
  --sleep 0.1

# MISS
python scripts/demo_live_uav.py \
  --csv demos/uav/generated/streams/engine_failure/carbonZ_2018-09-11-14-22-07_1_engine_failure.csv \
  --sleep 0.1

"""
