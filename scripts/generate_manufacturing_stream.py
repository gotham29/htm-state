#!/usr/bin/env python3
"""
Generate synthetic manufacturing process stream for HTM-State Demo 4.

We simulate three regimes over time:

1) Baseline stable production
2) Drifting process (e.g., tool wear, misalignment, temperature rise)
3) High-drift / pre-fault regime

Two embedded regime boundaries:
- step 800  -> onset of drift from baseline
- step 1600 -> transition into high-drift / pre-fault regime
"""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic manufacturing process stream")
    parser.add_argument(
        "--out",
        type=str,
        default="demos/manufacturing_demo/line_stream.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2400,
        help="Number of timesteps to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    n = args.n_steps

    # Regime boundaries
    boundary_1 = 800
    boundary_2 = 1600

    if boundary_2 >= n:
        raise ValueError(f"boundary_2 ({boundary_2}) must be < n_steps ({n})")

    timesteps = np.arange(n)

    # -----------------------------
    # Feature definitions (per regime)
    # -----------------------------
    # Baseline means
    base_vib = 1.0
    base_load = 1.0
    base_feed = 1.0
    base_speed = 1.0
    base_temp = 30.0     # deg C
    base_part_time = 1.0 # normalized cycle time

    # Segment lengths
    n_base = boundary_1
    n_drift = boundary_2 - boundary_1
    n_prefault = n - boundary_2

    # Baseline: fairly tight noise
    vib_baseline = base_vib + 0.03 * np.random.randn(n_base)
    load_baseline = base_load + 0.03 * np.random.randn(n_base)
    feed_baseline = base_feed + 0.02 * np.random.randn(n_base)
    speed_baseline = base_speed + 0.02 * np.random.randn(n_base)
    temp_baseline = base_temp + 0.3 * np.random.randn(n_base)
    pt_baseline = base_part_time + 0.02 * np.random.randn(n_base)

    # Drift regime: means slowly ramp, variance slightly increases
    # Build linear ramps for means:
    vib_mean_drift = np.linspace(base_vib, base_vib + 0.3, n_drift)
    load_mean_drift = np.linspace(base_load, base_load + 0.4, n_drift)
    feed_mean_drift = np.linspace(base_feed, base_feed + 0.15, n_drift)
    speed_mean_drift = np.linspace(base_speed, base_speed - 0.1, n_drift)
    temp_mean_drift = np.linspace(base_temp, base_temp + 10.0, n_drift)
    pt_mean_drift = np.linspace(base_part_time, base_part_time + 0.15, n_drift)

    vib_drift = vib_mean_drift + 0.05 * np.random.randn(n_drift)
    load_drift = load_mean_drift + 0.05 * np.random.randn(n_drift)
    feed_drift = feed_mean_drift + 0.03 * np.random.randn(n_drift)
    speed_drift = speed_mean_drift + 0.03 * np.random.randn(n_drift)
    temp_drift = temp_mean_drift + 0.5 * np.random.randn(n_drift)
    pt_drift = pt_mean_drift + 0.03 * np.random.randn(n_drift)

    # High-drift / pre-fault: higher means + noticeably higher variance
    vib_prefault = (base_vib + 0.35) + 0.08 * np.random.randn(n_prefault)
    load_prefault = (base_load + 0.5) + 0.08 * np.random.randn(n_prefault)
    feed_prefault = (base_feed + 0.2) + 0.04 * np.random.randn(n_prefault)
    speed_prefault = (base_speed - 0.15) + 0.04 * np.random.randn(n_prefault)
    temp_prefault = (base_temp + 12.0) + 0.8 * np.random.randn(n_prefault)
    pt_prefault = (base_part_time + 0.2) + 0.04 * np.random.randn(n_prefault)

    # Concatenate segments
    vibration_energy = np.concatenate([vib_baseline, vib_drift, vib_prefault])
    spindle_load = np.concatenate([load_baseline, load_drift, load_prefault])
    feed_rate = np.concatenate([feed_baseline, feed_drift, feed_prefault])
    line_speed = np.concatenate([speed_baseline, speed_drift, speed_prefault])
    temperature = np.concatenate([temp_baseline, temp_drift, temp_prefault])
    part_time = np.concatenate([pt_baseline, pt_drift, pt_prefault])

    # Sanity check
    assert len(vibration_energy) == n

    # Boundary flag
    # We conceptually have two drift regimes (onset + pre-fault),
    # but for evaluation we only annotate the **major** pre-fault jump
    # as a ground-truth boundary. The earlier drift is treated as
    # gradual change that the system can respond to in its own time.
    is_boundary = np.zeros(n, dtype=int)
    is_boundary[boundary_2] = 1

    df = pd.DataFrame(
        {
            "timestep": timesteps,
            "vibration_energy": vibration_energy,
            "spindle_load": spindle_load,
            "feed_rate": feed_rate,
            "line_speed": line_speed,
            "temperature": temperature,
            "part_time": part_time,
            "is_boundary": is_boundary,
        }
    )

    # Ensure directory
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out, index=False)

    print(f"Wrote manufacturing line stream to: {args.out}")
    print(f"  rows: {len(df)}")
    print(f"  annotated boundary at step: {boundary_2}")


if __name__ == "__main__":
    main()
