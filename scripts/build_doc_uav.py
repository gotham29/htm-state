#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional dependency: PyYAML.
# If you don't have it, the script can still work off CSV.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


AUTO_BEGIN = "<!-- AUTO-GALLERY:BEGIN -->"
AUTO_END = "<!-- AUTO-GALLERY:END -->"


DISPLAY_ORDER = [
    "no_failure",
    "engine_failure",
    "elevator_failure",
    "aileron_failure",
    "rudder_failure",
    "multi_fault",
]


LABEL_TITLES = {
    "baseline": "Baseline (no failure)",
    "typical_spike": "Typical spike",
    "hard_spike": "Hard spike",
    "sustained_only": "Sustained-only",
    "miss": "Miss",
}

DEFAULT_FIG_ROOT = "demos/uav/generated/figures"

@dataclass(frozen=True)
class Row:
    failure_type: str
    label: str
    run_id: str
    csv_path: str
    figure_path: str  # path to PNG (relative or absolute)
    # Optional metrics if present:
    boundary_s: Optional[float] = None
    first_spike_s: Optional[float] = None
    first_sustained_s: Optional[float] = None
    spike_lag_s: Optional[float] = None
    sustained_lag_s: Optional[float] = None
    false_alarms_spm: Optional[float] = None
    post_elev_frac: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build demos/uav/doc.md gallery from selected_runs manifest.")
    p.add_argument("--doc", type=str, default="demos/uav/doc.md")
    p.add_argument("--manifest", type=str, default="demos/uav/selected_runs.yaml")
    p.add_argument(
        "--manifest-csv",
        type=str,
        default="",
        help="Optional CSV manifest instead of YAML (e.g. results/uav_sweep/selected_runs.csv).",
    )
    p.add_argument(
        "--fig-root",
        type=str,
        default=DEFAULT_FIG_ROOT,
        help="Root directory for gallery images (expects subfolders per failure_type).",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Used to convert absolute paths into repo-relative links.",
    )
    return p.parse_args()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _to_repo_rel(path: str, repo_root: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(repo_root))
        except Exception:
            # fallback: just use basename if it's absolute outside repo
            return p.name
    return str(p)

def _to_doc_rel(repo_rel_path: str, repo_root: Path, doc_dir: Path) -> str:
    p = Path(repo_rel_path)
    abs_p = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
    return str(abs_p.relative_to(doc_dir.resolve()))

def load_manifest(manifest_path: Path, manifest_csv: Optional[Path], repo_root: Path, fig_root: Path) -> List[Row]:
    # CSV path still supported (old mode)
    if manifest_csv and manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
        rows: List[Row] = []
        for _, r in df.iterrows():
            rows.append(
                Row(
                    failure_type=str(r["failure_type"]),
                    label=str(r["label"]),
                    run_id=str(r["run_id"]),
                    csv_path=_to_repo_rel(str(r.get("csv_path", "")), repo_root),
                    figure_path=_to_repo_rel(str(r.get("figure_path", "")), repo_root),
                    boundary_s=_safe_float(r.get("boundary_s")),
                    first_spike_s=_safe_float(r.get("first_spike_s")),
                    first_sustained_s=_safe_float(r.get("first_sustained_s")),
                    spike_lag_s=_safe_float(r.get("spike_lag_s")),
                    sustained_lag_s=_safe_float(r.get("sustained_lag_s")),
                    false_alarms_spm=_safe_float(r.get("false_alarms_spm")),
                    post_elev_frac=_safe_float(r.get("post_elev_frac")),
                )
            )
        return rows

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed, and no --manifest-csv provided. "
            "Either: pip install pyyaml, or pass --manifest-csv."
        )

    data = yaml.safe_load(manifest_path.read_text())

    # --- Mode A: my original expected format: list OR {"runs": [...]}
    if isinstance(data, dict) and "runs" in data:
        data = data["runs"]
    if isinstance(data, list):
        rows: List[Row] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            rows.append(
                Row(
                    failure_type=str(obj["failure_type"]),
                    label=str(obj["label"]),
                    run_id=str(obj["run_id"]),
                    csv_path=_to_repo_rel(str(obj.get("csv_path", "")), repo_root),
                    figure_path=_to_repo_rel(str(obj.get("figure_path", "")), repo_root),
                    boundary_s=_safe_float(obj.get("boundary_s")),
                    first_spike_s=_safe_float(obj.get("first_spike_s")),
                    first_sustained_s=_safe_float(obj.get("first_sustained_s")),
                    spike_lag_s=_safe_float(obj.get("spike_lag_s")),
                    sustained_lag_s=_safe_float(obj.get("sustained_lag_s")),
                    false_alarms_spm=_safe_float(obj.get("false_alarms_spm")),
                    post_elev_frac=_safe_float(obj.get("post_elev_frac")),
                )
            )
        return rows

    # --- Mode B: YOUR schema: {"version":..., "fig_dir":..., "selected": {...}}
    if isinstance(data, dict) and "selected" in data:
        fig_dir = str(data.get("fig_dir", "")).strip()
        selected = data["selected"]

        if not isinstance(selected, dict):
            raise ValueError("selected_runs.yaml: 'selected' must be a mapping {failure_type: [items...]}")

        rows: List[Row] = []
        for failure_type, items in selected.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue

                run_id = str(it.get("run_id", "")).strip()
                tag = str(it.get("tag", "")).strip()  # <-- your key name
                fig_name = str(it.get("figure", "")).strip()

                if not run_id or not tag or not fig_name:
                    continue

                # Resolve figure path.
                # New repo layout: demos/uav/generated/figures/<failure_type>/<figure>.png
                # If the manifest figure already includes subdirs, respect it.
                fig_path: str
                fig_rel = Path(fig_name)
                if fig_rel.is_absolute():
                    fig_path = str(fig_rel)
                elif fig_rel.parts and fig_rel.parts[0] in ("demos", "results"):
                    # already repo-relative path
                    fig_path = str(fig_rel)
                else:
                    # Prefer --fig-root (new layout). Fall back to manifest fig_dir if provided.
                    base = fig_root if str(fig_root) else Path(fig_dir) if fig_dir else Path(".")
                    fig_path = str(base / str(failure_type) / fig_name)

                rows.append(
                    Row(
                        failure_type=str(failure_type),
                        label=tag,  # we store tag in label field
                        run_id=run_id,
                        csv_path="",  # optional: you can add later if you want
                        figure_path=_to_repo_rel(fig_path, repo_root),
                    )
                )

        return rows

    raise ValueError(
        "selected_runs.yaml must be either:\n"
        " - a list of run dicts\n"
        " - a dict with key 'runs'\n"
        " - a dict with key 'selected' (your schema)\n"
    )


def _format_metrics(r: Row) -> str:
    bits: List[str] = []
    if r.boundary_s is not None:
        bits.append(f"boundary={r.boundary_s:.1f}s")
    if r.first_spike_s is not None:
        bits.append(f"first_spike={r.first_spike_s:.1f}s")
    if r.first_sustained_s is not None:
        bits.append(f"first_sustained={r.first_sustained_s:.1f}s")
    if r.spike_lag_s is not None:
        bits.append(f"spike_lag={r.spike_lag_s:.1f}s")
    if r.sustained_lag_s is not None:
        bits.append(f"sust_lag={r.sustained_lag_s:.1f}s")
    if r.false_alarms_spm is not None:
        bits.append(f"FA={r.false_alarms_spm:.2f}/min")
    if r.post_elev_frac is not None:
        bits.append(f"post={r.post_elev_frac:.2f}")
    return ", ".join(bits) if bits else "—"


def render_gallery(rows: List[Row]) -> str:
    by_type: Dict[str, List[Row]] = {}
    for r in rows:
        by_type.setdefault(r.failure_type, []).append(r)

    # Stable ordering: prefer DISPLAY_ORDER, then alphabetical leftovers
    ordered_types = [t for t in DISPLAY_ORDER if t in by_type]
    leftovers = sorted([t for t in by_type.keys() if t not in ordered_types])
    ordered_types.extend(leftovers)

    out: List[str] = []
    out.append(AUTO_BEGIN)
    out.append("")
    out.append("> This section is generated from `demos/uav/selected_runs.yaml` (do not edit by hand).")
    out.append("")

    for ftype in ordered_types:
        out.append(f"### `{ftype}`")
        out.append("")
        out.append("| Example | Plot | Run ID | Quick metrics |")
        out.append("|---|---:|---|---|")

        # sort within type by label priority
        items = by_type[ftype]
        label_rank = {k: i for i, k in enumerate(["baseline", "typical_spike", "hard_spike", "sustained_only", "miss"])}
        items = sorted(items, key=lambda r: (label_rank.get(r.label, 999), r.run_id))

        for r in items:
            title = LABEL_TITLES.get(r.label, r.label)
            fig = r.figure_path
            run = r.run_id
            metrics = _format_metrics(r)
            # If you want clickable raw CSV links in GitHub:
            csv_link = r.csv_path if r.csv_path else ""
            run_cell = f"`{run}`"
            if csv_link:
                run_cell = f"[`{run}`]({csv_link})"
            out.append(f"| **{title}** | ![]({fig}) | {run_cell} | {metrics} |")

        out.append("")

    out.append(AUTO_END)
    out.append("")
    return "\n".join(out)


def splice_doc(doc_text: str, gallery_block: str) -> str:
    if AUTO_BEGIN not in doc_text or AUTO_END not in doc_text:
        raise ValueError(
            f"doc.md must contain both markers:\n{AUTO_BEGIN}\n{AUTO_END}"
        )
    pre = doc_text.split(AUTO_BEGIN)[0]
    post = doc_text.split(AUTO_END)[1]
    return pre + gallery_block + post


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    fig_root = Path(args.fig_root).resolve()

    doc_path = Path(args.doc).resolve()
    doc_dir = doc_path.parent.resolve()

    manifest_path = Path(args.manifest)
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else None

    rows = load_manifest(manifest_path, manifest_csv, repo_root, fig_root)
    # Convert repo-relative paths into doc-relative links so GitHub renders images correctly.
    fixed: List[Row] = []
    for r in rows:
        fig = r.figure_path
        if fig:
            fig = _to_doc_rel(fig, repo_root=repo_root, doc_dir=doc_dir)
        csvp = r.csv_path
        if csvp:
            csvp = _to_doc_rel(csvp, repo_root=repo_root, doc_dir=doc_dir)
        fixed.append(Row(**{**r.__dict__, "figure_path": fig, "csv_path": csvp}))
    rows = fixed
    gallery = render_gallery(rows)

    doc_text = doc_path.read_text()
    new_text = splice_doc(doc_text, gallery)

    doc_path.write_text(new_text)
    print(f"[build_doc_uav] wrote: {doc_path}")


if __name__ == "__main__":
    main()


"""
python scripts/build_demo_uav_md.py \
  --selected results/uav_sweep/selected_runs.csv \
  --demo-md demos/uav/doc.md
"""