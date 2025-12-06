# scripts/make_synthetic_workload.py

import math
import csv
from pathlib import Path


def main():
    out_dir = Path("demos/workload_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_workload.csv"

    n_steps = 1000
    # first half: lower variance; second half: higher variance + offset
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "control_x", "control_y"])

        for t in range(n_steps):
            t_sec = t * 0.1
            if t < n_steps // 2:
                # baseline behavior
                x = math.sin(0.05 * t) + 0.1 * math.sin(0.5 * t)
                y = math.cos(0.05 * t) + 0.1 * math.sin(0.3 * t + 1.0)
            else:
                # "high workload" / changed behavior
                x = 0.5 * math.sin(0.05 * t) + 0.6 * math.sin(0.7 * t + 1.0)
                y = 0.5 * math.cos(0.05 * t) + 0.6 * math.sin(0.6 * t + 0.5)

            writer.writerow([t_sec, x, y])

    print(f"Wrote synthetic demo to {out_path}")


if __name__ == "__main__":
    main()
