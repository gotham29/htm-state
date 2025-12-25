# htm_state/alfa_htm_scorer.py
from __future__ import annotations

"""
ALFA/ UAV scoring adapter: raw stream CSV -> per-step HTM anomaly score sequence.

Purpose
-------
This module is intentionally *not* an evaluator. It is a thin, auditable adapter
that turns a raw ALFA UAV CSV (features per timestep + is_boundary) into:

  - anomaly_scores[t] : float (HTM TM anomaly)
  - state_trace[t]    : float (HTM session EMA "mwl")

Those sequences are then consumed by the apples-to-apples evaluation pipeline:

  anomaly_scores[t] -> StateEngine EMA (optional) -> detector -> alarm_events
                    -> run_eval_pipeline -> confusion counts

Keeping this adapter separate ensures:
  - The paper-matching FP/TP/FN/TN logic stays isolated and reviewable.
  - Any anomaly backend can be swapped in later with the same evaluator.

Expected CSV columns
--------------------
- A time column: "t_sec" (optional for scoring; useful for plotting)
- Boundary indicator: "is_boundary" (optional for scoring; used by higher-level eval)
- Feature columns (float): e.g. airspeed, climb, altitude, throttle, heading, pitch, roll, yaw

Implementation notes
--------------------
We reuse the existing HTMSession used by demo_offline_uav.py. 
We compute per-run feature_ranges from the CSV (min/max per feature) exactly like the demo.

By default we:
  - keep SP learning OFF (stable mapping; matches your demo)
  - allow TM learning controlled via args (e.g., freeze after boundary later)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .htm_session import HTMSession


@dataclass
class HTMScoringConfig:
    """
    Configuration for scoring one ALFA run with HTMSession.

    Parameters mirror demo_offline_uav.py defaults to keep behavior aligned.
    """
    feature_names: List[str]
    rate_hz: float = 25.0

    # Encoders
    enc_n_per_feature: int = 64
    enc_w_per_feature: int = 8

    # Session EMA (HTMSession's internal EMA used to produce "mwl")
    anomaly_ema_alpha: float = 0.05

    # Determinism
    seed: int = 0

    # Learning
    sp_learning: bool = False


def _build_feature_ranges(df: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    feature_ranges: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        col = df[name].astype(float)
        vmin = float(col.min())
        vmax = float(col.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        feature_ranges[str(name)] = {"min": vmin, "max": vmax}
    return feature_ranges


def _default_sp_params() -> Dict[str, object]:
    # Keep aligned with demo_offline_uav.py :contentReference[oaicite:2]{index=2}
    return {
        "columnCount": 2048,
        "potentialPct": 0.8,
        "globalInhibition": True,
        "synPermActiveInc": 0.003,
        "synPermInactiveDec": 0.0005,
        "synPermConnected": 0.2,
        "boostStrength": 0.0,
    }


def _default_tm_params() -> Dict[str, object]:
    # Keep aligned with demo_offline_uav.py :contentReference[oaicite:3]{index=3}
    return {
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


def score_alfa_csv_with_htm(
    csv_path: str,
    cfg: HTMScoringConfig,
    *,
    learn: bool = True,
    freeze_tm_after_boundary: bool = False,
) -> Tuple[List[float], List[float], Optional[int]]:
    """
    Score a single ALFA UAV CSV with HTMSession and return per-step sequences.

    Parameters
    ----------
    csv_path : str
        Path to ALFA per-run CSV (raw feature stream).
    cfg : HTMScoringConfig
        HTM scoring configuration (features, encoder params, EMA alpha, etc.).
    learn : bool
        If True, TM learns throughout the run (unless frozen after boundary).
    freeze_tm_after_boundary : bool
        If True and a boundary exists, TM learn=True before boundary and learn=False after.

    Returns
    -------
    anomaly_scores : list[float]
        HTM TM anomaly per timestep (out["anomaly"] from HTMSession.step()).
    mwl_states : list[float]
        HTMSession EMA of anomaly (out["mwl"]).
    boundary_step : Optional[int]
        1-based index of first is_boundary==1, or None if not present.

    Notes
    -----
    We compute feature min/max ranges from the CSV for encoder scaling,
    mirroring demo_offline_uav.py. :contentReference[oaicite:4]{index=4}
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # Validate required feature columns
    missing = [f for f in cfg.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Available: {list(df.columns)}")

    # Boundary (1-based step for our evaluation system)
    boundary_step: Optional[int] = None
    if "is_boundary" in df.columns:
        idxs = df.index[df["is_boundary"].astype(int) == 1].tolist()
        if idxs:
            boundary_step = int(idxs[0]) + 1

    feature_ranges = _build_feature_ranges(df, cfg.feature_names)

    session = HTMSession(
        feature_names=list(cfg.feature_names),
        enc_n_per_feature=int(cfg.enc_n_per_feature),
        enc_w_per_feature=int(cfg.enc_w_per_feature),
        sp_params=_default_sp_params(),
        tm_params=_default_tm_params(),
        seed=int(cfg.seed),
        anomaly_ema_alpha=float(cfg.anomaly_ema_alpha),
        feature_ranges=feature_ranges,
    )
    session.sp_learning = bool(cfg.sp_learning)

    anomaly_scores: List[float] = []
    mwl_states: List[float] = []

    for i, row in df.iterrows():
        feats = {name: float(row[name]) for name in cfg.feature_names}

        step_1_based = int(i) + 1
        step_learn = bool(learn)
        if freeze_tm_after_boundary and boundary_step is not None and step_1_based >= boundary_step:
            step_learn = False

        out = session.step(feats, learn=step_learn)
        anomaly_scores.append(float(out["anomaly"]))
        mwl_states.append(float(out["mwl"]))

    return anomaly_scores, mwl_states, boundary_step
