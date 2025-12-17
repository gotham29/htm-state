from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class DriftInfo:
    """
    Simple container for drift boundary + detection metadata.

    Used in the cyber demo, but general enough for other domains.
    """

    boundary_time: float
    det_time: Optional[float] = None
    lag_sec: Optional[float] = None


class TruthLagOverlay:
    """
    Helper for drawing:
      - ground-truth transition / drift boundaries (vertical lines)
      - detection lag bars (horizontal lines + text)

    This encapsulates the Matplotlib artist bookkeeping so each demo can
    just call overlay.draw_*() each frame.
    """

    def __init__(self, ax_bottom, y_frac: float = 0.05):
        self.ax_bottom = ax_bottom
        self.y_frac = y_frac
        self._truth_artists: List = []
        self._lag_artists: List = []

    # ---------- internal helpers ----------

    def _clear(self) -> None:
        for ln in self._truth_artists + self._lag_artists:
            ln.remove()
        self._truth_artists = []
        self._lag_artists = []

    def _compute_y_level(self, states_window: Sequence[float]) -> float:
        y_min = min(states_window)
        y_max = max(states_window)
        span = max(1e-9, y_max - y_min)
        return y_min + self.y_frac * span

    # ---------- public API ----------

    def clear(self) -> None:
        """Clear any artists previously drawn by this overlay."""
        self._clear()

    def draw_single(
        self,
        toggle_time: float,
        det_time: Optional[float],
        lag_sec: Optional[float],
        ts_window: Sequence[float],
        states_window: Sequence[float],
        truth_label: str = "true transition",
        lag_label: str = "detection lag",
        color_truth: str = "red",
        color_lag: str = "magenta",
    ) -> List:
        """
        Draw a single ground-truth boundary + its detection lag bar.
        Used by the synthetic workload demo.
        """
        self._clear()
        artists: List = []

        if not ts_window or not states_window:
            return artists

        t_start, t_end = ts_window[0], ts_window[-1]
        y_level = self._compute_y_level(states_window)

        # Truth line
        if t_start <= toggle_time <= t_end:
            ln = self.ax_bottom.axvline(
                toggle_time,
                color=color_truth,
                linestyle="--",
                alpha=0.5,
                label=truth_label,
            )
            self._truth_artists.append(ln)
            artists.append(ln)

        # Detection lag bar + label
        if (
            det_time is not None
            and t_start <= toggle_time <= t_end
            and t_start <= det_time <= t_end
        ):
            lag = lag_sec if lag_sec is not None else max(0.0, det_time - toggle_time)
            lag_line = self.ax_bottom.plot(
                [toggle_time, det_time],
                [y_level, y_level],
                color=color_lag,
                linestyle="-",
                alpha=0.8,
                label=lag_label,
            )[0]
            self._lag_artists.append(lag_line)
            artists.append(lag_line)

            txt = self.ax_bottom.text(
                det_time,
                y_level,
                f"{lag:.1f}s",
                fontsize=8,
                ha="center",
                va="bottom",
                color=color_lag,
            )
            self._lag_artists.append(txt)
            artists.append(txt)

        return artists

    def draw_multi(
        self,
        drift_infos: Sequence[DriftInfo],
        ts_window: Sequence[float],
        states_window: Sequence[float],
        truth_label: str = "drift boundary",
        lag_label: str = "detection lag",
        color_truth: str = "red",
        color_lag: str = "magenta",
    ) -> List:
        """
        Draw multiple drift boundaries + their detection lag bars.
        Used by the cyber demo.
        """
        self._clear()
        artists: List = []

        if not drift_infos or not ts_window or not states_window:
            return artists

        t_start, t_end = ts_window[0], ts_window[-1]
        y_level = self._compute_y_level(states_window)

        truth_label_used = False
        lag_label_used = False

        for info in drift_infos:
            bt = info.boundary_time
            dt = info.det_time
            lag = info.lag_sec if info.lag_sec is not None else (
                None if dt is None else max(0.0, dt - bt)
            )

            # Truth line
            if t_start <= bt <= t_end:
                lbl = truth_label if not truth_label_used else "_nolegend_"
                ln = self.ax_bottom.axvline(
                    bt,
                    color=color_truth,
                    linestyle="--",
                    alpha=0.5,
                    label=lbl,
                )
                self._truth_artists.append(ln)
                artists.append(ln)
                truth_label_used = True

            # Detection lag bar + label
            if (
                dt is not None
                and t_start <= bt <= t_end
                and t_start <= dt <= t_end
            ):
                lbl = lag_label if not lag_label_used else "_nolegend_"
                lag_line = self.ax_bottom.plot(
                    [bt, dt],
                    [y_level, y_level],
                    color=color_lag,
                    linestyle="-",
                    alpha=0.8,
                    label=lbl,
                )[0]
                self._lag_artists.append(lag_line)
                artists.append(lag_line)
                lag_label_used = True

                if lag is not None:
                    txt = self.ax_bottom.text(
                        dt,
                        y_level,
                        f"{lag:.1f}s",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        color=color_lag,
                    )
                    self._lag_artists.append(txt)
                    artists.append(txt)

        return artists
