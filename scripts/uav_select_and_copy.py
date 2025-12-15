import argparse
import os
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

    # common failures in your list
    if "engine_failure" in n:
        return "engine_failure"
    if "elevator_failure" in n:
        return "elevator_failure"
    if "rudder" in n:
        # rudder_right_failure / rudder_left_failure / rudder_zero__...
        return "rudder_failure"
    if "aileron" in n:
        # left_aileron_failure / both_ailerons_failure / left_aileron__right_aileron__failure
        return "aileron_failure"

    # fallback: everything after last timestamp chunk
    # e.g., carbonZ_2018-09-11-14-52-54_left_aileron__right_aileron__failure
    m = re.search(r"carbonz_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)$", n)
    if m:
        return m.group(1)

    return "unknown"


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
    ap.add_argument("--print-only", action="store_true", help="Only print selections (no copying)")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()

    out_raw = repo_root / "demos" / "uav_demo" / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    picks = pick_best_per_type(processed_dir)

    # You can force “must have vfr” by filtering here if you want.
    print("\nSelected scenarios:")
    for ftype in sorted(picks.keys()):
        print(f"  {ftype:15s} -> {picks[ftype]}")

    if args.print_only:
        return

    print("\nCopying files:")
    for ftype, folder in sorted(picks.items()):
        src = processed_dir / folder
        dst = out_raw / folder
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
