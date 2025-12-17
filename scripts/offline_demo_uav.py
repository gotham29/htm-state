# scripts/offline_demo_uav.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from htm_state.htm_session import HTMSession
from htm_state.spike_detector import SpikeDetector, SpikeDetectorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline HTM-State demo on ALFA UAV fault scenarios.")
    p.add_argument(
        "--csv",
        type=str,
        default="demos/uav_demo/generated/uav_engine_failure.csv",
        help="Prepared UAV stream CSV (from generate_uav_stream.py).",
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
        help="EMA alpha for HTM anomaly smoothing (state).",
    )

    # Spike detector settings
    p.add_argument("--spike-recent-sec", type=float, default=2.0)
    p.add_argument("--spike-prior-sec", type=float, default=6.0)
    p.add_argument("--spike-threshold-pct", type=float, default=30.0)
    p.add_argument("--spike-min-sep-sec", type=float, default=2.0)

    # Encoders
    p.add_argument("--enc-n-per-feature", type=int, default=64)
    p.add_argument("--enc-w-per-feature", type=int, default=8)

    # Learning control
    p.add_argument(
        "--freeze-tm-after-boundary",
        action="store_true",
        help="If set, TM learn=True before boundary and learn=False after boundary.",
    )
    p.add_argument(
        "--reset-spike-at-boundary",
        action="store_true",
        help="If set, reset spike detector state at the boundary step to focus on post-boundary edges.",
    )
    p.add_argument("--spike-min-delta", type=float, default=0.05,
                help="Minimum absolute increase (recent - prior) required to count as a spike.")
    p.add_argument("--persist-window-sec", type=float, default=None,
                   help="Seconds of pre-boundary state to use for persistence threshold. Default: boundary-window-sec.")
    p.add_argument("--persist-quantile", type=float, default=None,
                   help="If set (e.g. 0.99), use quantile(pre_window) as persistence threshold instead of median+k*MAD.")

    # Sustained elevation detection (level-crossing)
    p.add_argument("--elev-k-mad", type=float, default=6.0,
                   help="Threshold = pre_median + k*MAD for sustained elevation detection.")
    p.add_argument("--elev-hold-sec", type=float, default=2.0,
                   help="How long state must stay above threshold to count as detected (seconds).")
    # Elevator-case diagnosis helper: compare feature behavior right around boundary
    p.add_argument(
        "--boundary-window-sec",
        type=float,
        default=15.0,
        help="Window size (seconds) for pre/post boundary feature-delta report.",
    )
    # Debug/trace output
    p.add_argument("--trace-out", type=str, default=None,
                   help="Optional path to write a per-step trace CSV (for plotting/debug).")
    # Optional: choose feature subset
    p.add_argument(
        "--features",
        type=str,
        default="airspeed,climb,altitude,throttle,heading,pitch,roll,yaw",
        help="Comma-separated feature list to use (must exist in the CSV).",
    )

    return p.parse_args()


def sec_to_steps(sec: float, rate_hz: float) -> int:
    return max(1, int(round(sec * rate_hz)))

def _safe_mean_std(s: pd.Series) -> Dict[str, float]:
    s = s.astype(float)
    if len(s) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    # ddof=0 to avoid NaN std for len==1
    return {"mean": float(s.mean()), "std": float(s.std(ddof=0))}

 
def print_boundary_feature_report(
    df: pd.DataFrame,
    feature_names: List[str],
    boundary_step: int,
    window_steps: int,
) -> None:
    """
    Print a compact pre vs post feature delta report around the boundary.

    pre  = [boundary-window, boundary)
    post = [boundary, boundary+window)

    For each feature: mean_pre, mean_post, delta, pooled_std, z = delta / pooled_std
    """
    pre0 = max(0, boundary_step - window_steps)
    pre1 = boundary_step
    post0 = boundary_step
    post1 = min(len(df), boundary_step + window_steps)

    if pre1 - pre0 < 5 or post1 - post0 < 5:
        print("Boundary feature report: insufficient samples near boundary.")
        return
 
    print(f"\n=== Boundary feature deltas (±{window_steps} steps) ===")
    print(f"pre:  [{pre0}:{pre1})   post: [{post0}:{post1})")
    rows = []
    for f in feature_names:
        pre_stats = _safe_mean_std(df.loc[pre0:pre1 - 1, f])
        post_stats = _safe_mean_std(df.loc[post0:post1 - 1, f])
        delta = post_stats["mean"] - pre_stats["mean"]
        pooled = (pre_stats["std"] ** 2 + post_stats["std"] ** 2) ** 0.5
        z = delta / pooled if pooled > 1e-12 else float("inf") if delta != 0 else 0.0
        rows.append((f, pre_stats["mean"], post_stats["mean"], delta, pooled, z))

    # sort by |z| desc to surface which signals actually changed
    rows.sort(key=lambda r: abs(r[-1]), reverse=True)
    print("feature               mean_pre     mean_post    delta       pooled_std  z")
    for f, m0, m1, d, ps, z in rows:
        print(f"{f:20s} {m0:10.4f} {m1:11.4f} {d:10.4f} {ps:11.4f} {z:7.2f}")

def evaluate_uav_csv(csv_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    """
    Programmatic entrypoint for batch sweeps.
    Returns a dict of key metrics (same ones printed in main()).
    """
    df = pd.read_csv(csv_path)

    required = {"t_sec", "is_boundary"}
    missing = required - set(df.columns)
    if missing:
        return {
            "included": False,
            "exclude_reason": f"missing_required_columns:{sorted(missing)}",
        }

    feature_names = [c.strip() for c in args.features.split(",") if c.strip()]
    for f in feature_names:
        if f not in df.columns:
            return {
                "included": False,
                "exclude_reason": f"missing_feature:{f}",
            }

    # --- BEGIN: essentially your current main() logic, but without prints ---
    boundary_step: Optional[int] = None
    boundary_time: Optional[float] = None
    boundary_idxs = df.index[df["is_boundary"].astype(int) == 1].tolist()
    if boundary_idxs:
        boundary_step = int(boundary_idxs[0])
        boundary_time = float(df.loc[boundary_step, "t_sec"])

    # Build feature ranges from this CSV
    feature_ranges: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        col = df[name].astype(float)
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
        enc_n_per_feature=args.enc_n_per_feature,
        enc_w_per_feature=args.enc_w_per_feature,
        sp_params=sp_params,
        tm_params=tm_params,
        seed=0,
        anomaly_ema_alpha=args.ema_alpha,
        feature_ranges=feature_ranges,
    )
    session.sp_learning = False

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec, args.rate_hz),
        min_delta=args.spike_min_delta,
        eps=1e-3,
    )
    spike_detector = SpikeDetector(spike_cfg)

    spikes: List[int] = []
    states: List[float] = []
    det_spike_step: Optional[int] = None
    det_spike_time: Optional[float] = None

    for i, row in df.iterrows():
        t = float(row["t_sec"])
        feats = {name: float(row[name]) for name in feature_names}

        learn = True
        if args.freeze_tm_after_boundary and boundary_step is not None and i >= boundary_step:
            learn = False

        out = session.step(feats, learn=learn)
        state = float(out["mwl"])
        states.append(state)

        if args.reset_spike_at_boundary and boundary_step is not None and i == boundary_step:
            spike_detector.reset()

        spike_res = spike_detector.update(state)
        spike_flag = bool(spike_res["spike"])
        if spike_flag:
            spikes.append(i)
            if boundary_step is not None and det_spike_step is None and i >= boundary_step:
                det_spike_step = i
                det_spike_time = t

    # False alarms before boundary
    pre_end = len(df) if boundary_step is None else boundary_step
    pre_spikes = [s for s in spikes if s < pre_end]
    pre_minutes = (
        (df.loc[pre_end - 1, "t_sec"] - float(df.loc[0, "t_sec"])) / 60.0
        if pre_end > 1 else 0.0
    )
    false_alarms_spm = (len(pre_spikes) / pre_minutes) if pre_minutes > 0 else 0.0

    # Sustained elevation + post persistence (only if boundary exists)
    sustained_detected = False
    sustained_lag_s: Optional[float] = None
    post_elev_frac: Optional[float] = None

    if boundary_step is not None:
        persist_window_sec = args.persist_window_sec if args.persist_window_sec is not None else args.boundary_window_sec
        persist_steps = sec_to_steps(float(persist_window_sec), args.rate_hz)

        pre_start = max(0, boundary_step - persist_steps)
        pre_states = states[pre_start:boundary_step]
        post_states = states[boundary_step:]

        if len(pre_states) >= 10 and len(post_states) >= 10:
            pre_s = pd.Series(pre_states)
            if args.persist_quantile is not None:
                q = min(max(float(args.persist_quantile), 0.5), 0.999)
                thr = float(pre_s.quantile(q))
            else:
                pre_med = float(pre_s.median())
                pre_mad = float((pre_s - pre_med).abs().median())
                thr = pre_med + float(args.elev_k_mad) * (pre_mad if pre_mad > 1e-9 else 1e-3)

            post_elev_frac = float((pd.Series(post_states) > thr).mean())

            hold_steps = sec_to_steps(float(args.elev_hold_sec), args.rate_hz)
            det_elev_step: Optional[int] = None
            for j in range(boundary_step, len(states) - hold_steps + 1):
                if all(s > thr for s in states[j:j + hold_steps]):
                    det_elev_step = j
                    break

            if det_elev_step is not None:
                sustained_detected = True
                det_elev_time = float(df.loc[det_elev_step, "t_sec"])
                sustained_lag_s = float(det_elev_time - boundary_time) if boundary_time is not None else None

    spike_detected = (det_spike_step is not None) if boundary_step is not None else False
    spike_lag_s: Optional[float] = None
    if boundary_step is not None and det_spike_time is not None and boundary_time is not None:
        spike_lag_s = float(det_spike_time - boundary_time)

    return {
        "included": True,
        "exclude_reason": "",
        "has_boundary": boundary_step is not None,
        "boundary_time_s": boundary_time if boundary_time is not None else None,
        "spike_detected": bool(spike_detected),
        "spike_lag_s": spike_lag_s,
        "sustained_detected": bool(sustained_detected),
        "sustained_lag_s": sustained_lag_s,
        "false_alarms_spm": float(false_alarms_spm),
        "post_elev_frac": post_elev_frac,
        "n_spikes_total": int(len(spikes)),
    }

def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"t_sec", "is_boundary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {sorted(missing)}. "
            f"Did you run scripts.generate_uav_stream?"
        )

    # Feature names
    feature_names = [c.strip() for c in args.features.split(",") if c.strip()]
    for f in feature_names:
        if f not in df.columns:
            raise ValueError(f"Requested feature '{f}' not found in CSV. Columns: {list(df.columns)}")

    print(f"CSV: {csv_path}")
    print(f"Using features: {feature_names}")

    # Find boundary (if any)
    boundary_step: Optional[int] = None
    boundary_time: Optional[float] = None
    boundary_idxs = df.index[df["is_boundary"].astype(int) == 1].tolist()
    if len(boundary_idxs) > 0:
        boundary_step = int(boundary_idxs[0])
        boundary_time = float(df.loc[boundary_step, "t_sec"])
        print(f"Boundary: step={boundary_step}, t_sec={boundary_time:.3f}")
    else:
        print("Boundary: None (no-failure scenario)")

    # Quick boundary-local feature sanity report (helps diagnose elevator/no-detect cases)
    if boundary_step is not None:
        win_steps = sec_to_steps(float(args.boundary_window_sec), float(args.rate_hz))
        print_boundary_feature_report(df, feature_names, boundary_step, win_steps)  

    # Build feature ranges from this CSV (simple but robust)
    feature_ranges: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        col = df[name].astype(float)
        vmin = float(col.min())
        vmax = float(col.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        feature_ranges[name] = {"min": vmin, "max": vmax}

    # SP/TM params (same style as cyber demo; tune later)
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
        enc_n_per_feature=args.enc_n_per_feature,
        enc_w_per_feature=args.enc_w_per_feature,
        sp_params=sp_params,
        tm_params=tm_params,
        seed=0,
        anomaly_ema_alpha=args.ema_alpha,
        feature_ranges=feature_ranges,
    )

    # Default: SP learning OFF (as you wanted)
    session.sp_learning = False

    spike_cfg = SpikeDetectorConfig(
        recent_window=sec_to_steps(args.spike_recent_sec, args.rate_hz),
        prior_window=sec_to_steps(args.spike_prior_sec, args.rate_hz),
        threshold_pct=args.spike_threshold_pct,
        edge_only=True,
        min_separation=sec_to_steps(args.spike_min_sep_sec, args.rate_hz),
        min_delta=args.spike_min_delta,
        eps=1e-3,
    )
    spike_detector = SpikeDetector(spike_cfg)

    # --- Run + evaluate with Demo Template metrics ---
    spikes: List[int] = []
    # Optional trace rows (written at end)
    trace_rows: List[Dict[str, float]] = []
    # Spike-based detection time (first spike after boundary)
    det_spike_step: Optional[int] = None
    det_spike_time: Optional[float] = None

    # post-boundary persistence: fraction of time state is "elevated"
    # We'll compute this after the run using a simple threshold.
    states: List[float] = []

    for i, row in df.iterrows():
        t = float(row["t_sec"])
        feats = {name: float(row[name]) for name in feature_names}

        learn = True
        if args.freeze_tm_after_boundary and boundary_step is not None and i >= boundary_step:
            learn = False

        out = session.step(feats, learn=learn)
        anomaly = float(out["anomaly"])
        state = float(out["mwl"])
        states.append(state)
        # Optional: reset spike detector right at boundary to avoid pre-boundary history masking the post event
        if args.reset_spike_at_boundary and boundary_step is not None and i == boundary_step:
            spike_detector.reset()
        spike_res = spike_detector.update(state)
        spike_flag = bool(spike_res["spike"])

        # Save trace (we’ll fill thr/elev_flag later once thr is computed)
        if args.trace_out:
            tr ={
                "step": float(i),
                "t_sec": float(t),
                "is_boundary": float(row["is_boundary"]),
                "learn": float(1.0 if learn else 0.0),
                "anomaly": float(anomaly),
                "mwl": float(state),
                "spike": float(1.0 if spike_flag else 0.0),
                "growth_pct": float(spike_res["growth_pct"]) if spike_res["growth_pct"] is not None else float("nan"),
                "mr": float(spike_res["mr"]) if spike_res["mr"] is not None else float("nan"),
                "mp": float(spike_res["mp"]) if spike_res["mp"] is not None else float("nan"),
            }
            # Include raw features for plotting / elevator diagnosis
            for name in feature_names:
                tr[name] = float(row[name])
            trace_rows.append(tr)

        if spike_flag:
            spikes.append(i)
            if boundary_step is not None and det_spike_step is None and i >= boundary_step:
                det_spike_step = i
                det_spike_time = t

    det_elev_step: Optional[int] = None
    det_elev_time: Optional[float] = None

    # --- Report ---
    print("\n=== Results ===")

    # Detection lag (spike-based, optional)
    if boundary_step is None:
        print("Spike lag: N/A (no boundary in this scenario)")
    else:
        if det_spike_step is None:
            print(f"Spike lag: NOT DETECTED after boundary (boundary step={boundary_step})")
        else:
            lag_steps = int(det_spike_step - boundary_step)
            lag_sec = float(det_spike_time - boundary_time) if (det_spike_time is not None and boundary_time is not None) else None
            print(
                f"Spike lag: {lag_steps} steps"
                + (f" ({lag_sec:.3f} s)" if lag_sec is not None else "")
            )

    # False alarms before boundary (spikes/min)
    if boundary_step is None:
        pre_end = len(df)
    else:
        pre_end = boundary_step

    pre_spikes = [s for s in spikes if s < pre_end]
    pre_minutes = (df.loc[pre_end - 1, "t_sec"] - float(df.loc[0, "t_sec"])) / 60.0 if pre_end > 1 else 0.0
    fa_rate = (len(pre_spikes) / pre_minutes) if pre_minutes > 0 else 0.0
    print(f"False alarms before boundary: {len(pre_spikes)} spikes  ({fa_rate:.3f} spikes/min)")

    # Post-boundary persistence + sustained elevation detection
    if boundary_step is None:
        print("Post-boundary persistence: N/A (no boundary)")
        print("Sustained elevation lag: N/A (no boundary)")
    else:
        # Use a LOCAL pre-boundary window to avoid early-run transients / spikes polluting threshold
        persist_window_sec = args.persist_window_sec
        if persist_window_sec is None:
            persist_window_sec = args.boundary_window_sec
        persist_steps = sec_to_steps(float(persist_window_sec), args.rate_hz)

        pre_start = max(0, boundary_step - persist_steps)
        pre_states = states[pre_start:boundary_step]
        post_states = states[boundary_step:]

        if len(pre_states) < 10 or len(post_states) < 10:
            print("Post-boundary persistence: insufficient samples")
            print("Sustained elevation lag: insufficient samples")
        else:
            pre_med = float(pd.Series(pre_states).median())
            pre_mad = float((pd.Series(pre_states) - pre_med).abs().median())

            k = float(args.elev_k_mad)
            mad = pre_mad if pre_mad > 1e-9 else 1e-3
            thr = pre_med + k * mad
            pre_s = pd.Series(pre_states)
            if args.persist_quantile is not None:
                q = float(args.persist_quantile)
                # guardrails
                q = min(max(q, 0.5), 0.999)
                thr = float(pre_s.quantile(q))
            else:
                pre_med = float(pre_s.median())
                pre_mad = float((pre_s - pre_med).abs().median())
                thr = pre_med + float(args.elev_k_mad) * (pre_mad if pre_mad > 1e-9 else 1e-3)
            post_elev_frac = float((pd.Series(post_states) > thr).mean())
            print(f"Post-boundary persistence: elevated_fraction={post_elev_frac:.3f} (thr={thr:.4f})")

            # If tracing, backfill threshold + elevation flag
            if args.trace_out:
                for r in trace_rows:
                    r["thr"] = float(thr)
                    r["elev"] = float(1.0 if r["mwl"] > thr else 0.0)

            # Sustained elevation detection: first index >= boundary where state stays > thr for hold_steps
            hold_steps = sec_to_steps(float(args.elev_hold_sec), args.rate_hz)
            det_elev_step = None
            for j in range(boundary_step, len(states) - hold_steps + 1):
                if all(s > thr for s in states[j:j + hold_steps]):
                    det_elev_step = j
                    break

            if det_elev_step is None:
                print(f"Sustained elevation lag: NOT DETECTED (hold={hold_steps} steps)")
            else:
                det_elev_time = float(df.loc[det_elev_step, "t_sec"])
                lag_steps = int(det_elev_step - boundary_step)
                lag_sec = float(det_elev_time - boundary_time) if boundary_time is not None else None
                print(
                    f"Sustained elevation lag: {lag_steps} steps"
                    + (f" ({lag_sec:.3f} s)" if lag_sec is not None else "")
                    + f"  (hold={hold_steps} steps)"
                )

    # Optional: quick sanity on spikes
    print(f"Total spikes: {len(spikes)}")
    if spikes:
        print(f"First 10 spike steps: {spikes[:10]}")
    # Write trace CSV (if requested)
    if args.trace_out:
        outp = Path(args.trace_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(trace_rows[0].keys()) if trace_rows else []
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in trace_rows:
                w.writerow(r)
        print(f"Wrote trace: {outp} (rows={len(trace_rows)})")

if __name__ == "__main__":
    main()