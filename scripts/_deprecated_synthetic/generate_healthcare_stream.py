"""
Generate a synthetic healthcare-style operator stream for Demo 3.

Output CSV schema (per row / timestep):
  - timestep
  - motion_energy
  - cursor_velocity
  - interaction_density
  - task_variability
  - is_boundary  (1 at regime boundaries, else 0)

Regime structure (mirrors README examples):
  - 0   ..  599  : baseline routine operation
  - 600 .. 1449  : increasing workload / rising complexity
  - 1450.. N-1   : high-acuity / overload regime

You can change N or the boundary steps via CLI.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic healthcare operator stream for Demo 3."
    )
    p.add_argument(
        "--out",
        type=str,
        default="demos/healthcare_demo/operator_stream.csv",
        help="Output CSV path for the generated operator stream.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=2000,
        help="Total number of timesteps to generate.",
    )
    p.add_argument(
        "--boundary-steps",
        type=int,
        nargs="*",
        default=[600, 1450],
        help=(
            "Regime boundary steps where is_boundary=1 will be set. "
            "Typically [600, 1450] as in the README examples."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


def generate_stream(
    n_steps: int,
    boundary_steps: List[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate synthetic clinical-style operator time series with three regimes:
      baseline → rising workload → high-acuity / overload.
    """
    timesteps = np.arange(n_steps, dtype=int)

    # Define regime masks
    b0 = boundary_steps[0] if len(boundary_steps) > 0 else n_steps // 3
    b1 = boundary_steps[1] if len(boundary_steps) > 1 else 2 * n_steps // 3

    baseline_mask = timesteps < b0
    rising_mask = (timesteps >= b0) & (timesteps < b1)
    overload_mask = timesteps >= b1

    # Helper to build piecewise values with noise
    def piecewise(
        base_level: float,
        rising_level: float,
        overload_level: float,
        noise_scale: float,
    ) -> np.ndarray:
        x = np.empty(n_steps, dtype=float)

        # baseline: relatively low level, low noise
        x[baseline_mask] = base_level + noise_scale * rng.normal(
            loc=0.0, scale=0.2, size=baseline_mask.sum()
        )

        # rising: interpolate from base_level → rising_level with more noise
        rising_idx = np.where(rising_mask)[0]
        if rising_idx.size > 0:
            t_norm = np.linspace(0.0, 1.0, rising_idx.size)
            rising_trend = base_level + (rising_level - base_level) * t_norm
            x[rising_mask] = rising_trend + noise_scale * rng.normal(
                loc=0.0, scale=0.4, size=rising_idx.size
            )

        # overload: around overload_level with highest noise
        overload_idx = np.where(overload_mask)[0]
        if overload_idx.size > 0:
            x[overload_mask] = overload_level + noise_scale * rng.normal(
                loc=0.0, scale=0.6, size=overload_idx.size
            )

        return x

    # Feature construction:
    # - motion_energy: overall physical activity level
    # - cursor_velocity: control / pointing aggressiveness
    # - interaction_density: clicks, touches, interactions per unit time
    # - task_variability: variability / entropy of micro-actions
    motion_energy = piecewise(
        base_level=1.0,
        rising_level=1.6,
        overload_level=2.1,
        noise_scale=0.15,
    )
    cursor_velocity = piecewise(
        base_level=0.8,
        rising_level=1.4,
        overload_level=2.0,
        noise_scale=0.18,
    )
    interaction_density = piecewise(
        base_level=0.5,
        rising_level=1.0,
        overload_level=1.7,
        noise_scale=0.12,
    )
    task_variability = piecewise(
        base_level=0.4,
        rising_level=0.9,
        overload_level=1.5,
        noise_scale=0.20,
    )

    # Boundaries: mark requested steps that fall inside [0, n_steps)
    is_boundary = np.zeros(n_steps, dtype=int)
    for b in boundary_steps:
        if 0 <= b < n_steps:
            is_boundary[b] = 1

    df = pd.DataFrame(
        {
            "timestep": timesteps,
            "motion_energy": motion_energy,
            "cursor_velocity": cursor_velocity,
            "interaction_density": interaction_density,
            "task_variability": task_variability,
            "is_boundary": is_boundary,
        }
    )

    return df


def main() -> None:
    args = parse_args()

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    df = generate_stream(
        n_steps=args.n_steps,
        boundary_steps=args.boundary_steps,
        rng=rng,
    )

    df.to_csv(out_path, index=False)

    print(f"Wrote healthcare operator stream to: {out_path}")
    print(f"  rows: {len(df)}")
    print(f"  boundaries at steps: {sorted(args.boundary_steps)}")


if __name__ == "__main__":
    main()
