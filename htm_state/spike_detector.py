from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from statistics import mean
from typing import Deque, Dict, Optional


@dataclass
class SpikeDetectorConfig:
    """
    Streaming spike detector over a scalar state signal (e.g., MWL or generic HTM state).

    Windows are in *steps* (you can convert from seconds using rate_hz at a higher layer).

    Attributes
    ----------
    recent_window : int
        Number of most recent steps for the "recent" mean (nr).
    prior_window : int
        Number of steps immediately before the recent window for the "prior" mean (np).
    threshold_pct : float
        Growth % threshold to declare a spike.
    edge_only : bool
        If True, only fire on upward *crossings* of the threshold (no latching).
    min_separation : int
        Minimum number of steps between spikes (debounce).
    min_delta : float
        Minimum absolute increase (mr - mp) required, in raw units (not %).
    eps : float
        Small value to stabilize division when |mp| is near zero.
    """

    recent_window: int = 30
    prior_window: int = 60
    threshold_pct: float = 50.0
    edge_only: bool = True
    min_separation: int = 0
    min_delta: float = 0.0
    eps: float = 1e-3


class SpikeDetector:
    """
    Stateful spike detector implementing the HTM-WL growth % logic:

        mr = mean(state[-nr:])
        mp = mean(state[-(nr+np):-nr])
        growth_pct = 100 * (mr - mp) / max(|mp|, eps)

    and then applying edge + separation controls.

    Call `update(state_value)` once per timestep.
    """

    def __init__(self, cfg: SpikeDetectorConfig) -> None:
        if cfg.recent_window <= 0 or cfg.prior_window <= 0:
            raise ValueError("recent_window and prior_window must be positive.")
        self.cfg = cfg

        self._window: Deque[float] = deque(
            maxlen=cfg.recent_window + cfg.prior_window
        )
        self._step: int = 0
        self._last_growth: Optional[float] = None
        self._last_spike_step: Optional[int] = None

    @property
    def step(self) -> int:
        """1-based step index (how many updates we've seen)."""
        return self._step

    def reset(self) -> None:
        """Clear all internal state."""
        self._window.clear()
        self._step = 0
        self._last_growth = None
        self._last_spike_step = None

    def update(self, state_value: float) -> Dict[str, Optional[float]]:
        """
        Ingest one new state value (e.g., MWL) and return spike decision + internals.

        Returns a dict with keys:
            - step          : 1-based step index
            - state         : the input state_value
            - mr            : recent mean (or None until warm)
            - mp            : prior mean (or None until warm)
            - growth_pct    : growth % (or None until warm)
            - spike         : 0 or 1
        """
        self._step += 1
        self._window.append(float(state_value))

        nr = self.cfg.recent_window
        np_ = self.cfg.prior_window
        needed = nr + np_

        # Not enough history yet -> can't compute windowed stats
        if len(self._window) < needed:
            return {
                "step": self._step,
                "state": state_value,
                "mr": None,
                "mp": None,
                "growth_pct": None,
                "spike": 0,
            }

        # Compute recent/prior means
        data = list(self._window)
        recent_slice = data[-nr:]
        prior_slice = data[-(nr + np_):-nr]

        mr = mean(recent_slice)
        mp = mean(prior_slice)

        denom = max(abs(mp), self.cfg.eps)
        growth_pct = 100.0 * (mr - mp) / denom

        # Thresholding
        above_thresh = growth_pct >= self.cfg.threshold_pct
        delta_ok = (mr - mp) >= self.cfg.min_delta

        # Edge-only logic: require crossing from below to above threshold
        if self.cfg.edge_only:
            was_above = (
                self._last_growth is not None
                and self._last_growth >= self.cfg.threshold_pct
            )
            crosses_up = above_thresh and not was_above
            trigger_candidate = crosses_up and delta_ok
        else:
            trigger_candidate = above_thresh and delta_ok

        # Min separation (debounce)
        if self._last_spike_step is not None:
            if self._step - self._last_spike_step <= self.cfg.min_separation:
                trigger_candidate = False

        spike = 1 if trigger_candidate else 0
        if spike:
            self._last_spike_step = self._step

        self._last_growth = growth_pct

        return {
            "step": self._step,
            "state": state_value,
            "mr": mr,
            "mp": mp,
            "growth_pct": growth_pct,
            "spike": spike,
        }
