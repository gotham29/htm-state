# scripts/generate_uav_stream.py
import yaml
import argparse
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from dataclasses import dataclass


def _pick_best_numeric_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Fallback: pick a non-time numeric column that actually has signal.
    Preference: lots of non-NaNs, high nunique, high std.
    """
    # Exclude obvious time-like columns
    bad = set(["%time", "time", "timestamp", "t_sec"])
    candidates = [c for c in df.columns if c.lower() not in bad]
    best_col = None
    best_score = -1.0
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if not s.notna().any():
            continue
        score = float(s.notna().sum()) + float(s.nunique()) + float(s.std())
        if score > best_score:
            best_score = score
            best_col = c
    return best_col

def load_scenarios(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def infer_scenario_from_dir(d: Path) -> dict:
    """
    Build a scenario spec from a raw ALFA run directory.
    """
    spec = {"dir": d.name}
    # prefer ground-truth if present
    if any(p.name.startswith("failure_status") for p in d.glob("*.csv")):
        spec["boundary"] = "failure_status"
    return spec

def resolve_scenario_files(base_dir: Path, scenario: dict) -> dict:
    d = base_dir / scenario["dir"]
    if not d.exists():
        raise FileNotFoundError(d)

    def pick(pattern):
        matches = list(d.glob(pattern))
        if not matches:
            return None
        return matches[0]

    vfr = pick("*mavros-vfr_hud.csv")
    if vfr is None:
        raise FileNotFoundError("Missing mavros-vfr_hud.csv")

    # Optional attitude sources
    # Allow explicit overrides via YAML, e.g.:
    #   pitch: some_pitch.csv
    #   roll:  some_roll.csv
    #   yaw:   some_yaw.csv
    def _resolve_att_file(key: str, fallback_globs: Sequence[str]) -> Optional[Path]:
        if scenario.get(key):
            p = d / scenario[key]
            if not p.exists():
                raise FileNotFoundError(f"Scenario {key} file not found: {p}")
            return p
        for g in fallback_globs:
            p = pick(g)
            if p is not None:
                return p
        return None

    pitch = _resolve_att_file(
        "pitch",
        [
            "*mavros-nav_info-pitch.csv",
            "*mavros-nav_info-pitch*.csv",
            "*nav_info-pitch*.csv",
            "*mavctrl-rpy.csv",
        ],
    )
    roll = _resolve_att_file(
        "roll",
        [
            "*mavros-nav_info-roll.csv",
            "*mavros-nav_info-roll*.csv",
            "*nav_info-roll*.csv",
            "*mavctrl-rpy.csv",
        ],
    )
    yaw = _resolve_att_file(
        "yaw",
        [
            "*mavros-nav_info-yaw.csv",
            "*mavros-nav_info-yaw*.csv",
            "*nav_info-yaw*.csv",
            "*mavctrl-rpy.csv",
        ],
    )

    status = None
    if scenario.get("boundary") == "failure_status":
        status = pick("*failure_status*.csv")

    print(f"[generate_uav_stream] vfr   = {vfr}", file=sys.stderr)
    print(f"[generate_uav_stream] pitch = {pitch}", file=sys.stderr)
    print(f"[generate_uav_stream] roll  = {roll}", file=sys.stderr)
    print(f"[generate_uav_stream] yaw   = {yaw}", file=sys.stderr)
    print(f"[generate_uav_stream] status= {status}", file=sys.stderr)

    return {
        "vfr": vfr,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "status": status,
    }


def _pick_col(cols, keywords):
    cols_l = [c.lower() for c in cols]
    for kw in keywords:
        for c, cl in zip(cols, cols_l):
            if kw in cl:
                return c
    return None


@dataclass
class TimeInfo:
    # Start of the raw log in nanoseconds, if the input had ns-epoch timestamps.
    # If the input did NOT look like ns-epoch, this will be None.
    t0_ns: Optional[int]

def _pick_numeric_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_l = [c.lower() for c in cols]
    for kw in candidates:
        kw_l = kw.lower()
        for c, cl in zip(cols, cols_l):
            if kw_l in cl:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    return c
    return None

def _find_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("%time", "time", "timestamp"):
            return c
    raise ValueError("Could not find a time column (expected one of: %time, time, timestamp).")


def _to_time_index(df: pd.DataFrame) -> tuple[pd.DataFrame, TimeInfo]:
    # Common formats in these logs: "%time" as ns since epoch, or seconds-like.
    time_col = _find_time_col(df)
    t = df[time_col].astype(np.int64)

    # Heuristic: if values look like nanoseconds since epoch, convert to seconds.
    # (1e15 is safely above typical seconds timestamps)
    t0_ns: Optional[int] = None
    if t.median() > 1e15:
        t0_ns = int(t.iloc[0])
        t_sec = (t - t0_ns) / 1e9
    else:
        t_sec = t - t.iloc[0]

    df = df.copy()
    df["t_sec"] = t_sec.astype(float)
    df = df.drop(columns=[time_col])
    df = df.set_index("t_sec").sort_index()
    return df, TimeInfo(t0_ns=t0_ns)

def _make_time_grid(df_index: pd.Index, rate_hz: float) -> np.ndarray:
    dt = 1.0 / rate_hz
    t_min, t_max = float(df_index.min()), float(df_index.max())
    return np.arange(t_min, t_max, dt)

def _resample_to_grid(df: pd.DataFrame, t_grid: np.ndarray) -> pd.DataFrame:
    # df is indexed by t_sec
    df2 = df.reindex(df.index.union(t_grid)).sort_index()
    df2 = df2.interpolate(method="index").reindex(t_grid)
    df2.index.name = "t_sec"
    return df2.reset_index()

def _choose_speed_col(raw_df: pd.DataFrame, air_col: Optional[str], ground_col: Optional[str]) -> Optional[str]:
    # Prefer airspeed unless it is constant / near-constant (common in these logs: airspeed==0)
    def score(col: Optional[str]) -> float:
        if col is None:
            return -1.0
        s = pd.to_numeric(raw_df[col], errors="coerce").dropna()
        if len(s) == 0:
            return -1.0
        # higher variance / more unique values = better "signal"
        return float(s.nunique()) + float(s.std())
    if ground_col is None:
        return air_col
    if air_col is None:
        return ground_col
    return air_col if score(air_col) >= score(ground_col) else ground_col

def _boundary_heuristic(
    stream: pd.DataFrame,
    speed_col: Optional[str],
    climb_col: Optional[str],
) -> Optional[int]:
    # returns boundary index (row index in stream) or None
    if speed_col is None and climb_col is None:
        return None

    x = None
    if speed_col is not None:
        x = stream[speed_col].astype(float).to_numpy()
    elif climb_col is not None:
        x = stream[climb_col].astype(float).to_numpy()

    # Use a rolling baseline and look for sustained drop (engine failure tends to tank airspeed/climb).
    w = 200  # 20 seconds at 10Hz (adjustable)
    if len(x) < w * 2:
        return None

    baseline = pd.Series(x).rolling(w, min_periods=w).median()
    resid = pd.Series(x) - baseline
    # detect first point where resid is below a threshold for a sustained window
    thr = resid.rolling(w, min_periods=w).quantile(0.05)  # adaptive
    bad = resid < thr

    sustain = 30  # 3 seconds at 10Hz
    run = bad.rolling(sustain, min_periods=sustain).mean() >= 0.9
    idx = np.argmax(run.to_numpy()) if run.any() else None
    return int(idx) if idx is not None and run.any() else None

def _boundary_from_status(
    vfr_path: Path,
    status_path: Path,
    rate_hz: float,
    vfr_timeinfo: TimeInfo,
    stream_len: int,
) -> Optional[int]:
    """
    Preferred ground-truth boundary:
      boundary time = first %time in failure_status*.csv
      align into the resampled stream by converting to t_sec using vfr t0_ns
    Returns boundary index (row index in *resampled* stream) or None.
    """
    if status_path is None:
        return None
    if not status_path.exists():
        raise FileNotFoundError(f"Status file not found: {status_path}")

    # We need ns-epoch alignment for robust mapping.
    if vfr_timeinfo.t0_ns is None:
        # If vfr isn't ns-epoch, alignment is ambiguous; fall back to heuristic.
        return None

    status = pd.read_csv(status_path)
    if len(status) == 0:
        return None

    # Find status time column
    time_col = None
    for c in status.columns:
        if c.lower() in ("%time", "time", "timestamp"):
            time_col = c
            break
    if time_col is None:
        return None

    t0_fail_ns = int(pd.to_numeric(status[time_col], errors="coerce").dropna().iloc[0])

    # Convert failure onset to seconds relative to VFR start.
    t_fail_sec = (t0_fail_ns - int(vfr_timeinfo.t0_ns)) / 1e9

    # Map to resampled index (grid starts at ~0 sec, step = 1/rate_hz)
    if t_fail_sec < 0:
        return 0

    idx = int(round(t_fail_sec * rate_hz))
    if idx < 0:
        return 0
    if idx >= stream_len:
        return stream_len - 1
    return idx

def _merge_nav_info_signal(
    stream: pd.DataFrame,
    t_grid: np.ndarray,
    signal_name: str,
    signal_path: Optional[Path],
) -> pd.DataFrame:
    """
    Merge a nav_info-style signal (pitch/roll/yaw) onto the master VFR grid.
    Prefers field.measured, then field.commanded, and avoids header/time metadata.
    """
    if signal_path is None or not Path(signal_path).exists():
        return stream

    raw = pd.read_csv(signal_path)
    df, _ = _to_time_index(raw)

    def _is_bad_meta_col(c: str) -> bool:
        cl = c.lower()
        return (
            "header" in cl
            or "stamp" in cl
            or "seq" in cl
            or "frame_id" in cl
            or cl in ("%time", "time", "timestamp", "t_sec")
        )

    scol = None

    # 1) Exact known-good columns first
    if "field.measured" in raw.columns:
        scol = "field.measured"
    elif "field.commanded" in raw.columns:
        scol = "field.commanded"

    # 2) Next: any numeric col containing signal name / measured / commanded that isn't metadata
    if scol is None:
        candidates = []
        for c in raw.columns:
            if _is_bad_meta_col(c):
                continue
            cl = c.lower()
            if ("measured" in cl) or ("commanded" in cl) or (signal_name.lower() in cl):
                s = pd.to_numeric(raw[c], errors="coerce")
                if s.notna().any():
                    candidates.append(c)
        scol = candidates[0] if candidates else None

    # 3) LAST resort: best numeric column excluding metadata
    if scol is None:
        safe = raw.drop(
            columns=[c for c in raw.columns if _is_bad_meta_col(c)],
            errors="ignore",
        )
        scol = _pick_best_numeric_signal(safe)

    print(f"[generate_uav_stream] {signal_name} columns = {list(raw.columns)}", file=sys.stderr)
    print(f"[generate_uav_stream] selected {signal_name} column = {scol}", file=sys.stderr)

    if scol is not None and scol in df.columns:
        feat = df[[scol]]
        res = _resample_to_grid(feat, t_grid=t_grid).rename(columns={scol: signal_name})
        stream = stream.merge(res[["t_sec", signal_name]], on="t_sec", how="left")

        # sanity checks
        nun = int(stream[signal_name].nunique()) if signal_name in stream.columns else 0
        nn = int(stream[signal_name].notna().sum()) if signal_name in stream.columns else 0
        if nun < 5:
            print(
                f"[generate_uav_stream] WARNING: {signal_name} appears near-constant (nunique={nun})",
                file=sys.stderr,
            )
        if ("stamp" in str(scol).lower()) or (stream[signal_name].median() > 1e12):
            print(
                f"[generate_uav_stream] WARNING: {signal_name} looks timestamp-like (scol={scol}, median={stream[signal_name].median():.3e})",
                file=sys.stderr,
            )
        print(f"[generate_uav_stream] {signal_name} merged: non-null count = {nn}", file=sys.stderr)
    else:
        print(
            f"[generate_uav_stream] WARNING: could not extract {signal_name} from {signal_path} "
            f"(scol={scol}, df_cols={list(df.columns)})",
            file=sys.stderr,
        )

    return stream

def build_stream(vfr_path: Path, status_path: Optional[Path], pitch_path: Optional[Path], roll_path: Optional[Path], yaw_path: Optional[Path], rate_hz: float):
    vfr_raw = pd.read_csv(vfr_path)
    vfr_df, tinfo = _to_time_index(vfr_raw)

    # Pick candidate columns (robust to naming differences)
    cols = list(vfr_df.columns)
    air_col   = _pick_col(cols, ["airspeed"])
    ground_col= _pick_col(cols, ["groundspeed", "gs"])
    speed_col = _choose_speed_col(vfr_raw, air_col, ground_col)
    climb_col = _pick_col(cols, ["climb", "climb_rate", "vs", "vertical_speed"])
    alt_col   = _pick_col(cols, ["alt", "altitude", "amsl"])
    thr_col   = _pick_col(cols, ["throttle", "thr"])
    hdg_col   = _pick_col(cols, ["heading", "hdg"])

    # Keep only what we found
    keep = [c for c in [speed_col, climb_col, alt_col, thr_col, hdg_col] if c is not None]
    if not keep:
        raise ValueError(f"No usable feature columns detected in {vfr_path.name}. Columns were: {cols}")

    # Build a single master time grid from VFR and resample everything onto it
    t_grid = _make_time_grid(vfr_df.index, rate_hz=rate_hz)

    vfr_feat = vfr_df[keep]
    stream = _resample_to_grid(vfr_feat, t_grid=t_grid)

    # Rename to stable names (for demos/scripts)
    rename = {}
    if speed_col: rename[speed_col] = "airspeed"
    if climb_col: rename[climb_col] = "climb"
    if alt_col:   rename[alt_col]   = "altitude"
    if thr_col:   rename[thr_col]   = "throttle"
    if hdg_col:   rename[hdg_col]   = "heading"
    stream = stream.rename(columns=rename)

    # Optional: add attitude signals (pitch/roll/yaw)
    stream = _merge_nav_info_signal(stream, t_grid=t_grid, signal_name="pitch", signal_path=pitch_path)
    stream = _merge_nav_info_signal(stream, t_grid=t_grid, signal_name="roll",  signal_path=roll_path)
    stream = _merge_nav_info_signal(stream, t_grid=t_grid, signal_name="yaw",   signal_path=yaw_path)
    # NOTE: roll_path/yaw_path are now resolved via YAML; plumb them in from caller (see main() changes below)

    return stream, tinfo

def write_one(name: str, stream: pd.DataFrame, bidx: Optional[int], out_path: Path) -> None:
    stream = stream.copy()
    stream["is_boundary"] = 0
    if bidx is not None and 0 <= bidx < len(stream):
        stream.loc[bidx, "is_boundary"] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (boundary={bidx})")

def main():
    ap = argparse.ArgumentParser()

    # --- Directory sweep mode (NEW) ---
    ap.add_argument("--raw-dir", type=str, help="Directory containing raw UAV run folders")

    # --- YAML-driven mode (preferred, reproducible) ---
    ap.add_argument("--scenarios-yaml", type=str, help="YAML defining UAV scenarios")
    ap.add_argument("--scenario", type=str, help="Scenario key from YAML (e.g., engine_failure)")
    ap.add_argument("--all", action="store_true", help="Generate all scenarios in YAML")

    # --- Legacy explicit-path mode (back-compat) ---
    ap.add_argument("--engine-vfr", type=str)
    ap.add_argument("--no-failure-vfr", type=str)
    ap.add_argument(
        "--engine-failure-status",
        type=str,
        default=None,
        help="Optional failure status CSV (e.g., *failure_status-engines.csv) for ground-truth onset alignment",
    )

    ap.add_argument("--rate-hz", type=float, default=10.0)

    # outputs
    ap.add_argument("--out-dir", default="demos/uav_demo", help="Default output directory")
    ap.add_argument("--out", type=str, default=None, help="Optional explicit output CSV path (single scenario)")

    args = ap.parse_args()

    using_yaml = args.scenarios_yaml is not None
    using_raw_dir = args.raw_dir is not None

    if using_raw_dir and using_yaml:
        ap.error("Use either --raw-dir OR --scenarios-yaml, not both.")

    if using_raw_dir:
        raw_dir = Path(args.raw_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in sorted(raw_dir.iterdir()):
            if not d.is_dir():
                continue

            spec = infer_scenario_from_dir(d)
            try:
                files = resolve_scenario_files(raw_dir, spec)
                stream, tinfo = build_stream(
                    files["vfr"],
                    files["status"],
                    files["pitch"],
                    files["roll"],
                    files["yaw"],
                    rate_hz=args.rate_hz,
                )
            except Exception as e:
                print(f"[generate_uav_stream] SKIP {d.name}: {e}", file=sys.stderr)
                continue

            bidx = None
            if files["status"] is not None:
                bidx = _boundary_from_status(
                    vfr_path=files["vfr"],
                    status_path=files["status"],
                    rate_hz=args.rate_hz,
                    vfr_timeinfo=tinfo,
                    stream_len=len(stream),
                )

            out_path = out_dir / f"{d.name}.csv"
            write_one(d.name, stream, bidx, out_path)

        return

    if using_yaml:
        if not (args.scenario or args.all):
            ap.error("When using --scenarios-yaml, you must pass --scenario or --all.")
        if args.out is not None and args.all:
            ap.error("--out cannot be used with --all (use --out-dir instead).")
    else:
        if not (args.engine_vfr and args.no_failure_vfr):
            ap.error(
                "Either provide --engine-vfr and --no-failure-vfr, "
                "or use --scenarios-yaml with --scenario/--all."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if using_yaml:
        cfg = load_scenarios(Path(args.scenarios_yaml))
        base = Path(cfg["processed_dir"])
        scenarios = cfg["scenarios"]

        def run_one(name, spec):
            files = resolve_scenario_files(base, spec)
            stream, tinfo = build_stream(files["vfr"], files["status"], files["pitch"], files["roll"], files["yaw"], rate_hz=args.rate_hz)

            bidx = None
            if files["status"] is not None:
                bidx = _boundary_from_status(
                    vfr_path=files["vfr"],
                    status_path=files["status"],
                    rate_hz=args.rate_hz,
                    vfr_timeinfo=tinfo,
                    stream_len=len(stream),
                )
            if bidx is None and spec.get("boundary") == "heuristic":
                bidx = _boundary_heuristic(
                    stream,
                    speed_col="airspeed" if "airspeed" in stream.columns else None,
                    climb_col="climb" if "climb" in stream.columns else None,
                )

            out_path = Path(args.out) if args.out else out_dir / f"uav_{name}.csv"
            write_one(name, stream, bidx, out_path)

        if args.all:
            for name, spec in scenarios.items():
                run_one(name, spec)
        else:
            run_one(args.scenario, scenarios[args.scenario])

    else:
        # Legacy explicit-path mode
        eng, eng_tinfo = build_stream(Path(args.engine_vfr), rate_hz=args.rate_hz, pitch_path=None)
        nf, _ = build_stream(Path(args.no_failure_vfr), rate_hz=args.rate_hz, pitch_path=None)

        bidx = None
        if args.engine_failure_status:
            bidx = _boundary_from_status(
                vfr_path=Path(args.engine_vfr),
                status_path=Path(args.engine_failure_status),
                rate_hz=args.rate_hz,
                vfr_timeinfo=eng_tinfo,
                stream_len=len(eng),
            )
        if bidx is None:
            bidx = _boundary_heuristic(
                eng,
                speed_col="airspeed" if "airspeed" in eng.columns else None,
                climb_col="climb" if "climb" in eng.columns else None,
            )

        write_one("engine_failure", eng, bidx, out_dir / "uav_engine_failure.csv")
        write_one("no_failure", nf, None, out_dir / "uav_no_failure.csv")


if __name__ == "__main__":
    main()
