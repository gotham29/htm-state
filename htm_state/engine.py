# htm_state/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Optional, List, Any


class Backend(Protocol):
    """
    Minimal interface for an anomaly backend.

    Any backend must implement:
      - reset()
      - compute_anomaly(features: dict[str, float]) -> float

    The 'features' dict is a single-timestep feature vector (e.g. control signals,
    cyber event counts, kinematic data, etc.) and the output is a scalar anomaly-like
    score that we will smooth into a 'state' value.
    """

    def reset(self) -> None:
        ...

    def compute_anomaly(self, features: Dict[str, float]) -> float:
        ...


@dataclass
class StateEngineConfig:
    """
    Configuration for the StateEngine.

    Attributes
    ----------
    ema_alpha : float
        Exponential moving average factor for smoothing the backend anomaly
        into a continuous 'state' estimate. Smaller = smoother, slower reaction.
    clip_min, clip_max : Optional[float]
        Optional clipping of the anomaly value before EMA. Set to None to disable.
    """
    ema_alpha: float = 0.05
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


class BaselineBackend:
    """
    Simple non-HTM baseline backend.

    Uses a running-mean model per feature and returns the mean absolute deviation
    from that running mean as an 'anomaly' score.

    This is just a placeholder so we can exercise the full HTM State pipeline
    (state smoothing, spike detection, detection lag) before plugging in a real HTM.
    """

    def __init__(self, feature_names: List[str]) -> None:
        if not feature_names:
            raise ValueError("BaselineBackend requires at least one feature name.")
        self.feature_names = list(feature_names)
        self._running_means: Dict[str, float] = {f: 0.0 for f in self.feature_names}
        self._count: int = 0

    def reset(self) -> None:
        self._running_means = {f: 0.0 for f in self.feature_names}
        self._count = 0

    def compute_anomaly(self, features: Dict[str, float]) -> float:
        """
        Online running-mean update, then average |x - mean| across features.
        """
        self._count += 1
        diffs = []

        for name in self.feature_names:
            x = float(features[name])
            old_mean = self._running_means[name]
            new_mean = old_mean + (x - old_mean) / self._count
            self._running_means[name] = new_mean
            diffs.append(abs(x - new_mean))

        if not diffs:
            return 0.0

        return float(sum(diffs) / len(diffs))


class HTMBackend:
    """
    Flexible wrapper for a real HTM model.

    You will plug in your own model from htm-wl / htm_py / htm_streamer here.

    The wrapped model can expose anomaly in a few ways:
      - model.compute_anomaly(features: dict) -> float
      - model.step(features: dict) -> float
      - model.step(features: dict) -> dict with key 'anomaly'

    Adapt this as needed to match your actual HTM API.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def reset(self) -> None:
        # Call reset() on the underlying model if it exists.
        if hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()

    def compute_anomaly(self, features: Dict[str, float]) -> float:
        # 1) Preferred: explicit compute_anomaly() method
        if hasattr(self.model, "compute_anomaly") and callable(self.model.compute_anomaly):
            return float(self.model.compute_anomaly(features))

        # 2) Next: step() that returns either a scalar or a dict with "anomaly"
        if hasattr(self.model, "step") and callable(self.model.step):
            result = self.model.step(features)
            if isinstance(result, dict) and "anomaly" in result:
                return float(result["anomaly"])
            return float(result)

        raise RuntimeError(
            "HTMBackend could not compute anomaly: expected model.compute_anomaly() "
            "or model.step() returning a scalar or dict with key 'anomaly'."
        )


class StateEngine:
    """
    HTM StateEngine.

    Responsibilities:
      - call a Backend to get an anomaly-like scalar per timestep
      - optionally clip that anomaly
      - smooth anomaly via EMA into a continuous 'state' value

    The intent is:
        raw features -> Backend (HTM or baseline) -> anomaly
        anomaly -> EMA -> 'state'
        state -> SpikeDetector (outside this class)

    Public API is intentionally simple and stable across backends.
    """

    def __init__(self, config: StateEngineConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend

        self._last_anomaly: float = 0.0
        self._state: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        """Reset internal state and backend."""
        self.backend.reset()
        self._last_anomaly = 0.0
        self._state = 0.0
        self._initialized = False

    @property
    def last_anomaly(self) -> float:
        return self._last_anomaly

    @property
    def state(self) -> float:
        return self._state

    def _clip(self, x: float) -> float:
        if self.config.clip_min is not None and x < self.config.clip_min:
            x = self.config.clip_min
        if self.config.clip_max is not None and x > self.config.clip_max:
            x = self.config.clip_max
        return x

    def step(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Process one timestep.

        Parameters
        ----------
        features : dict[str, float]
            Feature vector at the current time step.

        Returns
        -------
        dict with keys:
            - "anomaly": backend anomaly (after clipping)
            - "state"  : EMA-smoothed state value
        """
        anomaly = self.backend.compute_anomaly(features)
        anomaly = self._clip(float(anomaly))

        if not self._initialized:
            self._state = anomaly
            self._initialized = True
        else:
            a = self.config.ema_alpha
            self._state = a * anomaly + (1.0 - a) * self._state

        self._last_anomaly = anomaly

        return {
            "anomaly": self._last_anomaly,
            "state": self._state,
        }
