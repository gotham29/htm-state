from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Dict

import numpy as np


@dataclass
class PersistenceDetectorConfig:
    baseline_window: int        # steps (rolling baseline length)
    k_mad: float                # threshold = median + k * MAD
    hold_steps: int             # consecutive steps required above threshold
    edge_only: bool = True      # emit only on rising edge of "sustained" condition
    min_separation: int = 1     # debounce between sustained triggers (steps)
    eps: float = 1e-9           # numeric stability for MAD


class PersistenceDetector:
    def __init__(self, cfg: PersistenceDetectorConfig):
        self.cfg = cfg
        self.hist: Deque[float] = deque(maxlen=cfg.baseline_window)
        self._count: int = 0
        self._armed: bool = True
        self._last_fire_step: Optional[int] = None
        self._step: int = -1

        # Optional: freeze baseline stats (useful at boundary)
        self._frozen_thr: Optional[float] = None
        self._freeze: bool = False

    def reset(self) -> None:
        self.hist.clear()
        self._count = 0
        self._armed = True
        self._last_fire_step = None
        self._step = -1
        self._frozen_thr = None
        self._freeze = False

    def freeze_baseline(self) -> None:
        """
        Freeze the threshold using the current baseline window.
        After freezing, we keep emitting based on this fixed thr (and we stop updating hist).
        """
        thr = self._compute_threshold()
        if thr is not None:
            self._frozen_thr = float(thr)
            self._freeze = True

    def _compute_threshold(self) -> Optional[float]:
        if len(self.hist) < self.cfg.baseline_window:
            return None
        x = np.asarray(self.hist, dtype=float)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) + float(self.cfg.eps)
        return float(med + float(self.cfg.k_mad) * mad)

    def update(self, value: float) -> dict:
        self._step += 1

        # Determine threshold
        if self._freeze and self._frozen_thr is not None:
            thr = float(self._frozen_thr)
        else:
            thr = self._compute_threshold()

        fired = False
        if thr is not None:
            if float(value) > float(thr):
                self._count += 1
            else:
                self._count = 0
                self._armed = True  # re-arm after dropping below thr

            sustained = self._count >= int(self.cfg.hold_steps)

            if sustained:
                # Debounce
                ok_sep = (
                    self._last_fire_step is None
                    or (self._step - self._last_fire_step) >= int(self.cfg.min_separation)
                )
                if ok_sep:
                    if self.cfg.edge_only:
                        if self._armed:
                            fired = True
                            self._armed = False
                    else:
                        fired = True
        else:
            sustained = False
            self._count = 0

        # Update baseline history unless frozen
        if not self._freeze:
            self.hist.append(float(value))

        # Return both key spellings (so callers don’t diverge)
        out: Dict[str, object] = {
            "sustained": bool(fired),
            "persistent": bool(fired),
            "thr": (float(thr) if thr is not None else None),
            "threshold": (float(thr) if thr is not None else None),
            "count": int(self._count),
            "is_frozen": bool(self._freeze),
        }
        return out
