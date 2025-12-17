from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path

# --------- detection helpers ---------

def classify_failure(name: str) -> str:
    """Map folder name -> canonical failure type."""
    n = name.lower()

    if "no_failure" in n:
        return "no_failure"
    if "no_ground_truth" in n:
        return "no_ground_truth"

    has_engine = "engine_failure" in n
    has_elevator = "elevator_failure" in n
    has_rudder = ("rudder" in n and "failure" in n)
    has_aileron = ("aileron" in n and "failure" in n)

    # multi-fault if >1 fault family appears (or explicit double-underscore pattern)
    families = sum([has_engine, has_elevator, has_rudder, has_aileron])
    if families >= 2 or "__" in n:
        return "multi_fault"

    if has_engine:
        return "engine_failure"
    if has_elevator:
        return "elevator_failure"
    if has_rudder:
        return "rudder_failure"
    if has_aileron:
        return "aileron_failure"

    # fallback: everything after last timestamp chunk
    # e.g., carbonZ_2018-09-11-14-52-54_left_aileron__right_aileron__failure
    m = re.search(r"carbonz_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)$", n)
    if m:
        return m.group(1)

    return "unknown"

def is_eligible_failure_type(ftype: str) -> bool:
    return ftype in {
        "no_failure",
        "engine_failure",
        "elevator_failure",
        "rudder_failure",
        "aileron_failure",
        "multi_fault",
    }

def score_scenario(dirpath: Path) -> int:
    """
    Higher score = better representative scenario.
    Prefer having ground-truth + VFR.
    """
    files = [p.name.lower() for p in dirpath.glob("*.csv")]
    has_vfr = any("mavros-vfr_hud" in f for f in files)
    has_status = any("failure_status" in f for f in files)
    has_rcout = any("mavros-rc-out" in f for f in files)
    has_imu = any("mavros-imu-data" in f for f in files)
    has_pitch = any("mavros-nav_info-pitch" in f or "nav_info-pitch" in f for f in files)

    # weighted preference
    score = 0
    score += 50 if has_status else 0
    score += 30 if has_vfr else 0
    score += 10 if has_rcout else 0
    score += 5 if has_imu else 0
    score += 10 if has_pitch else 0

    # tie-breaker: prefer longer files (roughly)
    # (only if VFR exists)
    if has_vfr:
        vfr = next((dirpath / f for f in dirpath.iterdir()
                    if f.is_file() and "mavros-vfr_hud" in f.name.lower() and f.suffix.lower()==".csv"), None)
        if vfr and vfr.exists():
            score += min(20, int(vfr.stat().st_size / 50_000))  # +1 per ~50KB up to +20

    return score

def pick_all_eligible(processed_dir: Path):
    """
    Returns: dict[folder_name] = failure_type for all eligible runs.
    """
    picks = {}
    for d in processed_dir.iterdir():
        if not d.is_dir():
            continue
        if "ds_store" in d.name.lower():
            continue

        ftype = classify_failure(d.name)
        if ftype == "no_ground_truth":
            continue
        if not is_eligible_failure_type(ftype):
            continue
        picks[d.name] = ftype
    return picks

def pick_best_per_type(processed_dir: Path):
    """
    Returns: dict[failure_type] = selected_folder_name
    """
    candidates = {}
    for d in processed_dir.iterdir():
        if not d.is_dir():
            continue
        if "ds_store" in d.name.lower():
            continue

        ftype = classify_failure(d.name)
        if ftype in ("unknown",):
            continue

        s = score_scenario(d)
        if ftype not in candidates or s > candidates[ftype][0]:
            candidates[ftype] = (s, d.name)

    # keep only one baseline no_failure (best scored)
    return {ftype: name for ftype, (s, name) in candidates.items()}


def copy_needed_files(src_dir: Path, dst_dir: Path, include_optional: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)

    wanted_patterns = [
        "*mavros-vfr_hud*.csv",
        "*failure_status*.csv",
        "*failure-status*.csv",
        "*failureStatus*.csv",
        "*FailureStatus*.csv",
    ]
    if include_optional:
        wanted_patterns += [
            "*mavros-rc-out*.csv",
            "*mavros-imu-data*.csv",
            "*mavros-nav_info-*.csv",   # pitch/roll/yaw/airspeed/etc if present
            "*mavctrl-rpy*.csv",
        ]

    copied = []
    seen = set()
    for pat in wanted_patterns:
        for p in src_dir.glob(pat):
            if p.is_file():
                out = dst_dir / p.name
                if out.name in seen:
                    continue
                shutil.copy2(p, out)
                copied.append(out.name)
                seen.add(out.name)

    return copied


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Path to ALFA processed/ directory")
    ap.add_argument("--repo-root", required=True, help="Path to htm-state repo root")
    ap.add_argument("--include-optional", action="store_true", help="Also copy rc-out and imu-data if present")
    ap.add_argument("--mode", choices=["all", "best"], default="all",
                    help="all=copy all eligible runs (default); best=copy one best run per type")
    ap.add_argument("--print-only", action="store_true", help="Only print selections (no copying)")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()

    # New repo layout:
    # demos/uav/raw/<failure_type>/<run_id>/topic_files.csv
    out_raw = repo_root / "demos" / "uav" / "raw"
 
    out_raw.mkdir(parents=True, exist_ok=True)

    if args.mode == "best":
        picks_best = pick_best_per_type(processed_dir)  # dict[ftype] = folder
        folders = [(folder, ftype) for ftype, folder in picks_best.items()]
    else:
        picks_all = pick_all_eligible(processed_dir)     # dict[folder] = ftype
        folders = [(folder, ftype) for folder, ftype in picks_all.items()]

    # You can force “must have vfr” by filtering here if you want.
    print("\nSelected scenarios:")
    for folder, ftype in sorted(folders, key=lambda x: (x[1], x[0])):
        print(f"  {ftype:15s} -> {folder}")

    if args.print_only:
        return

    print("\nCopying files:")
    for folder, ftype in sorted(folders, key=lambda x: (x[1], x[0])):
        src = processed_dir / folder
        # Group by failure type to keep the repo navigable.
        # Example:
        # demos/uav/raw/engine_failure/carbonZ_2018-..._engine_failure/
        dst = out_raw / ftype / folder
        copied = copy_needed_files(src, dst, include_optional=args.include_optional)

        print(f"- {folder}")
        if copied:
            for c in copied:
                print(f"    copied: {c}")
        else:
            print("    (nothing matched wanted patterns)")

    print(f"\nDone. Raw scenarios are under: {out_raw}")


if __name__ == "__main__":
    main()
