#scripts/run_live_demo_uav_all.py
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run live_demo_uav over all generated UAV CSVs")
    p.add_argument(
        "--glob",
        type=str,
        default="demos/uav_demo/generated/uav_*.csv",
        help="Glob for UAV demo CSVs",
    )
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--freeze-tm-after-boundary", action="store_true")
    p.add_argument("--freeze-persist-baseline-at-boundary", action="store_true")
    p.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra args passed verbatim to scripts.live_demo_uav (e.g. \"--elev-k-mad 3 --elev-hold-sec 1\")",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    base_cmd: List[str] = [
        "python",
        "-m",
        "scripts.live_demo_uav",
        "--rate-hz",
        str(args.rate_hz),
        "--sleep",
        str(args.sleep),
    ]
    if args.freeze_tm_after_boundary:
        base_cmd.append("--freeze-tm-after-boundary")
    if args.freeze_persist_baseline_at_boundary:
        base_cmd.append("--freeze-persist-baseline-at-boundary")

    extra_tokens: List[str] = [t for t in args.extra.split(" ") if t.strip()]

    for csv_path in paths:
        cmd = base_cmd + ["--csv", str(csv_path)] + extra_tokens
        print("\n=== Running ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


"""
python -m scripts.run_live_demo_uav_all \
  --glob "demos/uav_demo/generated/uav_*.csv" \
  --rate-hz 10 --sleep 0.05 \
  --freeze-tm-after-boundary \
  --freeze-persist-baseline-at-boundary
"""