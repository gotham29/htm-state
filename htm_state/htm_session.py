# htm_state/htm_session.py

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# These imports assume htm.core is installed in your environment:
#   pip install htm.bindings   (or whatever you used in htm-wl)
from htm.bindings.sdr import SDR
from htm.algorithms import SpatialPooler as SP
from htm.algorithms import TemporalMemory as TM


class _ScalarBucket:
    """
    Minimal scalar encoder.

    If you want RDSE, see comment in HTMSession.__init__ to switch to
    htm.bindings.encoders RDSE. Uses (n, w) per feature.
    """

    def __init__(self, n: int, w: int, vmin: float = -1.0, vmax: float = 1.0):
        self.n, self.w = n, max(3, w)
        self.vmin, self.vmax = vmin, vmax
        self.range = max(1e-9, vmax - vmin)
        self.sp_learning = False  # default: OFF


    def encode(self, x: float) -> np.ndarray:
        x = min(self.vmax, max(self.vmin, x))
        frac = (x - self.vmin) / self.range
        center = int(frac * (self.n - 1))
        start = max(0, min(self.n - self.w, center - self.w // 2))
        sdr = np.zeros(self.n, dtype=np.int8)
        sdr[start:start + self.w] = 1
        return sdr


class HTMSession:
    """
    Builds encoders + SP + TM using htm.core. Produces:
      - anomaly (tm.anomaly)
      - mwl = EMA(anomaly) with alpha
    """

    def __init__(
        self,
        feature_names: List[str],
        enc_n_per_feature: int,
        enc_w_per_feature: int,
        sp_params: Dict,
        tm_params: Dict,
        seed: int,
        anomaly_ema_alpha: float = 0.2,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.feature_names = feature_names
        self.N = enc_n_per_feature * len(feature_names)
        self.C = int(sp_params.get("columnCount", 2048))
        self.alpha = anomaly_ema_alpha
        self._ema: Optional[float] = None

        # Encoders (you can swap this for RDSE if you want exact NuPIC behavior)
        self.encoders: Dict[str, _ScalarBucket] = {}
        for name in feature_names:
            vmin = feature_ranges.get(name, {}).get("min", -1.0) if feature_ranges else -1.0
            vmax = feature_ranges.get(name, {}).get("max", 1.0) if feature_ranges else 1.0
            self.encoders[name] = _ScalarBucket(enc_n_per_feature, enc_w_per_feature, vmin, vmax)

        # Input SDR
        self.input_sdr = SDR((self.N,))

        # Spatial Pooler
        self.sp = SP(
            inputDimensions=(self.N,),
            columnDimensions=(self.C,),
            potentialPct=float(sp_params.get("potentialPct", 0.8)),
            globalInhibition=bool(sp_params.get("globalInhibition", True)),
            synPermActiveInc=float(sp_params.get("synPermActiveInc", 0.003)),
            synPermInactiveDec=float(sp_params.get("synPermInactiveDec", 0.0005)),
            synPermConnected=float(sp_params.get("synPermConnected", 0.2)),
            boostStrength=float(sp_params.get("boostStrength", 0.0)),
            seed=int(seed),
        )

        # Temporal Memory
        self.tm = TM(
            columnDimensions=(self.C,),
            cellsPerColumn=int(tm_params.get("cellsPerColumn", 32)),
            activationThreshold=int(tm_params.get("activationThreshold", 20)),
            initialPermanence=float(tm_params.get("initialPerm", 0.21)),
            connectedPermanence=float(tm_params.get("permanenceConnected", 0.5)),
            minThreshold=int(tm_params.get("minThreshold", 13)),
            maxNewSynapseCount=int(tm_params.get("newSynapseCount", 31)),
            permanenceIncrement=float(tm_params.get("permanenceInc", 0.1)),
            permanenceDecrement=float(tm_params.get("permanenceDec", 0.0)),
            predictedSegmentDecrement=float(tm_params.get("predictedSegmentDecrement", 0.001)),
            seed=int(seed),
        )

    def reset(self) -> None:
        """
        Optional reset hook if you want to clear EMA and TM state between runs.
        """
        self._ema = None
        # The TM/SP classes do not expose a full reset API easily; for now we only
        # reset EMA and reuse learned structure across a sequence. If you want a full
        # reset, you can re-instantiate HTMSession.
        # (Or add explicit random reinit logic here.)

    def _encode(self, feats: Dict[str, float]) -> np.ndarray:
        chunks = [self.encoders[n].encode(float(feats[n])) for n in self.feature_names]
        return np.concatenate(chunks)

    def step(self, feats: Dict[str, float], learn: bool = True) -> Dict[str, float]:
        """
        One timestep of HTM processing.

        Parameters
        ----------
        feats : dict[str, float]
            Feature values for this timestep.
        learn : bool
            If False, Temporal Memory runs in inference-only mode (no learning updates).

        Returns
        -------
        dict with keys:
            - "anomaly": float
            - "mwl"    : EMA of anomaly (you may treat this as a domain-specific state)
        """
        x = self._encode(feats)
        self.input_sdr.sparse = np.nonzero(x)[0]

        active_columns = SDR(self.sp.getColumnDimensions())

        # Default recommendation: SP learning OFF (stable encoder->columns mapping, less compute).
        # Add self.sp_learning = False in __init__ (or wherever you configure the session).
        sp_learn = bool(getattr(self, "sp_learning", False))
        self.sp.compute(self.input_sdr, sp_learn, active_columns)

        # TM learning controlled by caller (so you can freeze after boundary)
        self.tm.compute(active_columns, learn=bool(learn))

        anomaly = float(self.tm.anomaly)
        if self._ema is None:
            self._ema = anomaly
        else:
            self._ema = (1.0 - self.alpha) * self._ema + self.alpha * anomaly

        return {"anomaly": anomaly, "mwl": float(self._ema)}
