#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from htm_state.alfa_htm_scorer import HTMScoringConfig, score_alfa_csv_with_htm
from htm_state.engine import StateEngineConfig
from htm_state.persistence_detector import PersistenceDetectorConfig
from htm_state.spike_detector import SpikeDetectorConfig
from htm_state.run_eval_pipeline import evaluate_run_end_to_end


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Score one ALFA run with HTM, then apples-to-apples evaluate + plot.")
    p.add_argument("--csv", required=True, help="ALFA per-run CSV (raw stream).")
    p.add_argument("--rate-hz", type=float, default=25.0)
    p.add_argument("--warmup-seconds", type=float, default=8.0)
    p.add_argument(
        "--toggle-step",
        type=int,
        default=None,
        help="Override boundary step (1-based). If omitted, uses first is_boundary==1 if present.",
    )
    p.add_argument("--features", type=str,
                   default="airspeed,climb,altitude,throttle,heading,pitch,roll,yaw")
    p.add_argument(
        "--mode",
        choices=["spike", "persistence", "or", "and"],
        default="persistence",
    )
    p.add_argument("--ema-alpha", type=float, default=0.1)

    # HTM session scoring knobs (match demo defaults)
    p.add_argument("--enc-n", type=int, default=64)
    p.add_argument("--enc-w", type=int, default=8)
    p.add_argument("--htm-ema-alpha", type=float, default=0.05)
    p.add_argument("--freeze-tm-after-boundary", action="store_true")

    # Detectors (start conservative; tune later)
    p.add_argument("--spike-recent", type=int, default=10)
    p.add_argument("--spike-prior", type=int, default=50)
    p.add_argument("--spike-threshold-pct", type=float, default=50.0)
    p.add_argument("--spike-min-sep", type=int, default=1)
    p.add_argument("--spike-min-delta", type=float, default=0.0)

    p.add_argument("--persist-baseline", type=int, default=200)
    p.add_argument("--persist-k-mad", type=float, default=3.0)
    p.add_argument("--persist-hold", type=int, default=10)
    p.add_argument("--persist-min-sep", type=int, default=1)

    p.add_argument("--out", default=None, help="Optional PNG output path instead of show().")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    feature_names = [c.strip() for c in args.features.split(",") if c.strip()]

    # 1) HTM scoring: raw CSV -> anomaly_scores
    anomaly_scores, mwl_states, boundary_step = score_alfa_csv_with_htm(
        str(csv_path),
        HTMScoringConfig(
            feature_names=feature_names,
            rate_hz=float(args.rate_hz),
            enc_n_per_feature=int(args.enc_n),
            enc_w_per_feature=int(args.enc_w),
            anomaly_ema_alpha=float(args.htm_ema_alpha),
            seed=0,
            sp_learning=False,
        ),
        learn=True,
        freeze_tm_after_boundary=bool(args.freeze_tm_after_boundary),
    )

    toggle_step = int(args.toggle_step) if args.toggle_step is not None else boundary_step

    # 2) Apples-to-apples evaluation (uses anomaly_scores)
    spike_cfg = SpikeDetectorConfig(
        recent_window=int(args.spike_recent),
        prior_window=int(args.spike_prior),
        threshold_pct=float(args.spike_threshold_pct),
        edge_only=True,
        min_separation=int(args.spike_min_sep),
        min_delta=float(args.spike_min_delta),
        eps=1e-3,
    )
    persistence_cfg = PersistenceDetectorConfig(
        baseline_window=int(args.persist_baseline),
        k_mad=float(args.persist_k_mad),
        hold_steps=int(args.persist_hold),
        edge_only=True,
        min_separation=int(args.persist_min_sep),
    )

    use_spike = args.mode in ("spike", "or", "and")
    use_pers = args.mode in ("persistence", "or", "and")

    res, diag = evaluate_run_end_to_end(
        run_id=csv_path.stem,
        toggle_step=toggle_step,
        anomaly_scores=anomaly_scores,
        rate_hz=float(args.rate_hz),
        warmup_seconds=float(args.warmup_seconds),
        alarm_mode=args.mode,  # type: ignore[arg-type]
        state_cfg=StateEngineConfig(ema_alpha=float(args.ema_alpha)),
        spike_cfg=spike_cfg if use_spike else None,
        persistence_cfg=persistence_cfg if use_pers else None,
        include_detector_debug=True,
        notes={"csv": str(csv_path)},
    )

    print("\n=== RUN RESULT ===")
    print(
        f"{res.run_id} | mode={args.mode} | TP={res.confusion.tp} FP={res.confusion.fp} "
        f"TN={res.confusion.tn} FN={res.confusion.fn} | first_any={res.first_alarm_any} "
        f"first_post={res.first_alarm_post} early={res.has_early_alarm} | "
        f"init_done={res.init_done_step} toggle={res.toggle_step}"
    )

    # 3) Plot diagnostics (same pattern as toy)
    T = len(anomaly_scores)
    x = list(range(1, T + 1))

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(x, diag.state_trace.anomaly, label="anomaly_score")
    ax.plot(x, diag.state_trace.state, label="ema_state")

    # Persistence threshold
    if diag.detector_traces.persistence_debug:
        thr_vals = [d.get("thr", None) for d in diag.detector_traces.persistence_debug]
        thr_x = [i + 1 for i, v in enumerate(thr_vals) if v is not None]
        thr_y = [float(v) for v in thr_vals if v is not None]
        ax.plot(thr_x, thr_y, label="persistence_thr")

    ymin, ymax = ax.get_ylim()
    y_event = ymin + 0.05 * (ymax - ymin)

    if diag.detector_traces.spike_event:
        sx = [i + 1 for i, v in enumerate(diag.detector_traces.spike_event) if v]
        ax.scatter(sx, [y_event] * len(sx), label="spike_fired", marker="|", s=200)

    if diag.detector_traces.persistence_event:
        px = [i + 1 for i, v in enumerate(diag.detector_traces.persistence_event) if v]
        ax.scatter(px, [y_event] * len(px), label="persistence_fired", marker="|", s=200)

    ax.scatter(
        [i + 1 for i, v in enumerate(diag.combined_events.alarm_event) if v],
        [y_event] * sum(diag.combined_events.alarm_event),
        label="alarm_event",
        marker="x",
    )

    ax.axvline(diag.init_done_step, linestyle="--", label="init_done_step")
    if toggle_step is not None:
        ax.axvline(int(toggle_step), linestyle="--", label="toggle_step")

    ax.set_title(
        f"{res.run_id} | mode={args.mode} | TP={res.confusion.tp} FP={res.confusion.fp} TN={res.confusion.tn} FN={res.confusion.fn}"
    )
    ax.set_xlabel("step (1-based)")
    ax.set_ylabel("value")
    ax.legend()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"Wrote: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
