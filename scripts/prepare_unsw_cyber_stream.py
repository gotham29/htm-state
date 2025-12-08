# scripts/prepare_unsw_cyber_stream.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a small UNSW-NB15-based cyber stream with micro-drifts."
    )
    p.add_argument(
        "--raw-csv",
        type=str,
        default="demos/cyber_demo/raw/UNSW_NB15_training-set.csv",
        help="Path to UNSW_NB15_training-set.csv (downloaded from UNSW/Kaggle).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="demos/cyber_demo/unsw_cyber_stream.csv",
        help="Output path for the prepared stream.",
    )
    p.add_argument(
        "--rows-per-phase",
        type=int,
        default=500,
        help="Number of rows per phase (default: 500).",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Assumed sample rate for time column (default: 10 Hz).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw UNSW file not found: {raw_path}\n"
            "Download UNSW_NB15_training-set.csv from UNSW or Kaggle and "
            "place it here."
        )

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(raw_path)

    # Common UNSW columns:
    #  - 'label': 0 = normal, 1 = attack
    #  - 'attack_cat': attack category string (Reconnaissance, Fuzzers, etc.)
    if "label" not in df.columns or "attack_cat" not in df.columns:
        raise ValueError(
            "Expected columns 'label' and 'attack_cat' in UNSW_NB15_training-set.csv"
        )

    benign = df[df["label"] == 0]
    attacks = df[df["label"] == 1]

    recon = attacks[attacks["attack_cat"].str.lower() == "reconnaissance"]
    fuzzers = attacks[attacks["attack_cat"].str.lower() == "fuzzers"]
    exploits = attacks[attacks["attack_cat"].str.lower() == "exploits"]

    if len(benign) < args.rows_per_phase * 3:
        raise ValueError("Not enough benign samples to build the requested phases.")

    # Helper to sample without replacement (but safely)
    def sample(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
        n = min(n, len(df_in))
        idx = rng.choice(df_in.index.to_numpy(), size=n, replace=False)
        return df_in.loc[idx].copy()

    n = args.rows_per_phase

    # Phase 0: benign only
    phase0 = sample(benign, n)
    phase0["phase_id"] = 0

    # Phase 1: mostly benign + small Recon
    p1_benign = sample(benign, int(n * 0.9))
    p1_recon = sample(recon, int(n * 0.1))
    phase1 = pd.concat([p1_benign, p1_recon], ignore_index=True)
    phase1["phase_id"] = 1

    # Phase 2: mostly benign + small Fuzzers
    p2_benign = sample(benign, int(n * 0.9))
    p2_fuzz = sample(fuzzers, int(n * 0.1))
    phase2 = pd.concat([p2_benign, p2_fuzz], ignore_index=True)
    phase2["phase_id"] = 2

    # Phase 3: benign + Recon + Exploits
    p3_benign = sample(benign, int(n * 0.8))
    p3_recon = sample(recon, int(n * 0.1))
    p3_expl = sample(exploits, int(n * 0.1))
    phase3 = pd.concat([p3_benign, p3_recon, p3_expl], ignore_index=True)
    phase3["phase_id"] = 3

    stream = pd.concat([phase0, phase1, phase2, phase3], ignore_index=True)

    # Shuffle *within* each phase but preserve phase order
    stream = (
        stream.groupby("phase_id", group_keys=False)
        .apply(lambda g: g.sample(frac=1.0, random_state=args.seed))
        .reset_index(drop=True)
    )

    # Add time column
    stream.insert(0, "t", np.arange(len(stream), dtype=float) / float(args.rate_hz))

    # Mark drift boundaries (phase changes) with a flag
    stream["is_drift_boundary"] = 0
    prev_phase = stream.loc[0, "phase_id"]
    for i in range(1, len(stream)):
        cur_phase = stream.loc[i, "phase_id"]
        if cur_phase != prev_phase:
            stream.loc[i, "is_drift_boundary"] = 1
        prev_phase = cur_phase

    # Choose a set of numeric features for HTM input
    candidate_features: List[str] = [
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "spkts",
        "dpkts",
        "sload",
        "dload",
        "rate",
    ]
    numeric_features = [c for c in candidate_features if c in stream.columns]

    if not numeric_features:
        raise ValueError(
            "None of the expected UNSW numeric features were found. "
            "Check the column names in your CSV."
        )

    # Keep only what we need for the demo:
    # time + numeric features + labels for evaluation
    cols_to_keep = (
        ["t"] + numeric_features + ["label", "attack_cat", "phase_id", "is_drift_boundary"]
    )
    stream = stream[cols_to_keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.to_csv(out_path, index=False)
    print(f"Wrote cyber stream: {out_path}")
    print(f"Rows: {len(stream)}, numeric features: {numeric_features}")


if __name__ == "__main__":
    main()
