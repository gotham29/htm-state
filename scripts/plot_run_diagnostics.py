#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from htm_state.engine import StateEngineConfig
from htm_state.persistence_detector import PersistenceDetectorConfig
from htm_state.run_eval_pipeline import evaluate_run_end_to_end


def _toy_scores(run_id: str) -> tuple[list[float], Optional[int]]:
    if run_id == "nf_clean":
        return [0.0] * 300, None
    if run_id == "nf_fp":
        return [0.0] * 200 + [5.0] * 100, None
    if run_id == "fail_early_only":
        return [0.0] * 200 + [5.0] * 40 + [0.0] * 60, 250
    raise ValueError(f"Unknown toy run_id: {run_id}")


def plot_run_diagnostics(run_id: str, outpath: Optional[str] = None) -> None:
    scores, toggle_step = _toy_scores(run_id)

    res, diag = evaluate_run_end_to_end(
        run_id=run_id,
        toggle_step=toggle_step,
        anomaly_scores=scores,
        rate_hz=25.0,
        warmup_seconds=8.0,
        alarm_mode="persistence",
        state_cfg=StateEngineConfig(ema_alpha=0.1),
        persistence_cfg=PersistenceDetectorConfig(
            baseline_window=80,
            k_mad=3.0,
            hold_steps=5,
            edge_only=True,
            min_separation=1,
        ),
        include_detector_debug=True,   # <-- so we can plot threshold too
        notes={"toy": True},
    )

    T = len(scores)
    x = list(range(1, T + 1))

    fig = plt.figure()
    ax = plt.gca()

    # Raw anomaly + EMA state
    ax.plot(x, diag.state_trace.anomaly, label="anomaly_score")
    ax.plot(x, diag.state_trace.state, label="ema_state")

    # Persistence threshold (if available)
    thr = None
    if diag.detector_traces.persistence_debug:
        thr = [d.get("thr", None) for d in diag.detector_traces.persistence_debug]
        # Plot thr where not None
        thr_x = [i + 1 for i, v in enumerate(thr) if v is not None]
        thr_y = [float(v) for v in thr if v is not None]
        ax.plot(thr_x, thr_y, label="persistence_thr")

    # Fired events as markers near the bottom
    ymin, ymax = ax.get_ylim()
    y_event = ymin + 0.05 * (ymax - ymin)

    if diag.detector_traces.persistence_event:
        pe = diag.detector_traces.persistence_event
        px = [i + 1 for i, v in enumerate(pe) if v]
        ax.scatter(px, [y_event] * len(px), label="persistence_fired", marker="|", s=200)

    # Combined alarm events (these are what gets converted to alarm_steps)
    ae = diag.combined_events.alarm_event
    ax.scatter([i + 1 for i, v in enumerate(ae) if v], [y_event] * sum(ae),
               label="alarm_event", marker="x")

    # init_done and toggle lines
    ax.axvline(diag.init_done_step, linestyle="--", label="init_done_step")
    if toggle_step is not None:
        ax.axvline(toggle_step, linestyle="--", label="toggle_step")

    ax.set_title(
        f"{run_id} | TP={res.confusion.tp} FP={res.confusion.fp} TN={res.confusion.tn} FN={res.confusion.fn} | "
        f"first_any={res.first_alarm_any} first_post={res.first_alarm_post} early={res.has_early_alarm}"
    )
    ax.set_xlabel("step (1-based)")
    ax.set_ylabel("value")
    ax.legend()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
        print(f"Wrote: {outpath}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser("Plot diagnostics for a single run using RunEvalDiagnostics (toy demo).")
    p.add_argument("--run", required=True, choices=["nf_clean", "nf_fp", "fail_early_only"])
    p.add_argument("--out", default=None, help="Optional path to save PNG instead of showing.")
    args = p.parse_args()
    plot_run_diagnostics(args.run, outpath=args.out)


if __name__ == "__main__":
    main()
