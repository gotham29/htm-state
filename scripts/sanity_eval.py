#sanity_eval.py

from htm_state.engine import StateEngineConfig
from htm_state.persistence_detector import PersistenceDetectorConfig
from htm_state.runs_eval_pipeline import RunInput, evaluate_runs_end_to_end


def main():
    # Three deliberately simple runs
    runs = [
        # No-failure, clean (TN)
        RunInput(
            run_id="nf_clean",
            toggle_step=None,
            anomaly_scores=[0.0] * 300,
            rate_hz=25.0,
        ),

        # No-failure, late alarm (FP)
        RunInput(
            run_id="nf_fp",
            toggle_step=None,
            anomaly_scores=[0.0] * 200 + [5.0] * 100,
            rate_hz=25.0,
        ),

        # Failure, early-only alarm (FP + FN)
        RunInput(
            run_id="fail_early_only",
            toggle_step=250,
            anomaly_scores=[0.0] * 200 + [5.0] * 40 + [0.0] * 60,
            rate_hz=25.0,
        ),
    ]

    out = evaluate_runs_end_to_end(
        runs=runs,
        warmup_seconds=8.0,  # => init_done_step = 201 at 25 Hz
        alarm_mode="persistence",
        state_cfg=StateEngineConfig(ema_alpha=0.1),
        persistence_cfg=PersistenceDetectorConfig(
            baseline_window=80,
            k_mad=3.0,
            hold_steps=5,
            edge_only=True,
            min_separation=1,
        ),
        include_diagnostics=False,
        shared_notes={"sanity": True},
    )

    print("\n=== PER-RUN TABLE ===")
    for r in out.per_run:
        print(
            f"{r.run_id:15s} "
            f"TP={r.tp} FP={r.fp} TN={r.tn} FN={r.fn} | "
            f"first_any={r.first_alarm_any} "
            f"first_post={r.first_alarm_post} "
            f"early={r.early_alarm}"
        )

    print("\n=== TOTALS ===")
    print(out.totals)

    print("\n=== SUMMARY ===")
    print(out.summary)

if __name__ == "__main__":
    main()
