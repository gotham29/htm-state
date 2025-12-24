#alfa_apples_to_apples.py

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ALFA-22 = dagger-marked sequences from the ALFA processed table,
# excluding the "2018-07-18-12-10-11" row with N/A pre/post fields.
ALFA_22_SEQ_IDS: set[str] = {
    "2018-07-18-15-53-31_1",
    "2018-07-18-15-53-31_2",
    "2018-07-18-16-37-39_1",  # no-failure
    "2018-07-30-16-39-00_1",
    "2018-07-30-16-39-00_2",
    "2018-07-30-16-39-00_3",  # no-failure
    "2018-09-11-11-56-30",
    "2018-09-11-14-16-55",    # no-failure
    "2018-09-11-14-22-07_1",
    "2018-09-11-14-22-07_2",
    "2018-09-11-14-41-38",    # no-failure
    "2018-09-11-14-41-51",
    "2018-09-11-14-52-54",
    "2018-09-11-15-05-11_1",
    "2018-09-11-15-05-11_2",  # no-failure
    "2018-09-11-15-06-34_1",
    "2018-09-11-15-06-34_2",
    "2018-09-11-15-06-34_3",
    "2018-09-11-17-27-13_1",
    "2018-09-11-17-27-13_2",
    "2018-09-11-17-55-30_1",
    "2018-09-11-17-55-30_2",
}


def _extract_seq_id(run_id: str) -> str | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:_\d+)?", str(run_id))
    return m.group(0) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate ALFA apples-to-apples table + optionally inject into doc.md")
    p.add_argument("--per-run", required=True, type=str, help="Path to per_run.csv")
    p.add_argument("--manifest", required=True, type=str, help="Path to alfa_gt_manifest.csv")
    p.add_argument("--out", required=True, type=str, help="Output CSV path (alfa_apples_to_apples.csv)")
    p.add_argument("--doc", default=None, type=str, help="Optional: doc.md to auto-populate")
    p.add_argument(
        "--mode",
        default="both",
        choices=["spike", "sustained", "both", "or", "all"],
        help="Which detector output defines a 'detection' for apples-to-apples metrics",
    )
    p.add_argument(
        "--subset",
        default="alfa22",
        choices=["alfa22", "all"],
        help="Evaluate only ALFA-22 subset (recommended) or all rows in per_run",
    )
    return p.parse_args()


def _compute_binary_metrics_from_series(
    df: pd.DataFrame,
    detected: pd.Series,
    lag_s: pd.Series,
) -> dict[str, Any]:
    detected = detected.fillna(False).astype(bool)
    # "has fault" = any failure type other than no-failure.
    # (Some pipelines use "none", yours uses "no_failure".)
    ft = df["failure_type"].astype(str).str.lower()
    has_fault = ~ft.isin({"no_failure", "none"})

    tp = int((detected & has_fault).sum())
    fp = int((detected & ~has_fault).sum())
    tn = int((~detected & ~has_fault).sum())
    fn = int((~detected & has_fault).sum())

    n = int(len(df))
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    # Detection time stats: over true positives only (avoids mixing misses into time averages)
    lag_s = pd.to_numeric(lag_s, errors="coerce")
    tp_lags = lag_s[(detected & has_fault) & np.isfinite(lag_s)]
    avg_dt = float(tp_lags.mean()) if len(tp_lags) else float("nan")
    max_dt = float(tp_lags.max()) if len(tp_lags) else float("nan")

    return {
        "n_sequences": n,
        "n_fault": int(has_fault.sum()),
        "n_no_fault": int((~has_fault).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "avg_detection_time_s": avg_dt,
        "max_detection_time_s": max_dt,
    }


def _compute_or_metrics(df: pd.DataFrame) -> dict[str, Any]:
    spike_det = df["spike_detected"].fillna(False).astype(bool)
    sust_det = df["sustained_detected"].fillna(False).astype(bool)
    detected = spike_det | sust_det

    # OR-lag = min of available lags for that row
    spike_lag = pd.to_numeric(df["spike_lag_s"], errors="coerce")
    sust_lag = pd.to_numeric(df["sustained_lag_s"], errors="coerce")
    lag = pd.concat([spike_lag, sust_lag], axis=1).min(axis=1, skipna=True)

    tmp = df.copy()
    tmp["or_detected"] = detected
    tmp["or_lag_s"] = lag
    return _compute_binary_metrics(tmp, detected_col="or_detected", lag_col="or_lag_s")


def _compute_binary_metrics(df: pd.DataFrame, detected_col: str, lag_col: str) -> dict[str, Any]:
    return _compute_binary_metrics_from_series(
        df=df,
        detected=df[detected_col],
        lag_s=df[lag_col],
    )


def _compute_or_lag(df: pd.DataFrame, lag_cols: list[str]) -> pd.Series:
    # row-wise min over lag columns, ignoring NaNs
    lags = [pd.to_numeric(df[c], errors="coerce") for c in lag_cols]
    if not lags:
        return pd.Series([np.nan] * len(df), index=df.index)
    m = lags[0].copy()
    for x in lags[1:]:
        m = np.fmin(m, x)  # fmin ignores NaNs if the other side is finite
    return m


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    # fixed column order
    cols = [
        "mode",
        "n_sequences",
        "n_fault",
        "n_no_fault",
        "accuracy",
        "precision",
        "recall",
        "avg_detection_time_s",
        "max_detection_time_s",
        "tp",
        "fp",
        "fn",
    ]

    def fmt(v: Any, col: str) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "–"
        if col in {"accuracy", "precision", "recall"}:
            return f"{float(v):.3f}"
        if col.endswith("_time_s"):
            return f"{float(v):.2f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(r.get(c), c) for c in cols) + " |"
        for r in rows
    )
    return "\n".join([header, sep, body])


def _inject_into_doc(doc_path: Path, md_table: str) -> None:
    text = doc_path.read_text()
    begin = "<!-- ALFA_APPLES_TO_APPLES:BEGIN -->"
    end = "<!-- ALFA_APPLES_TO_APPLES:END -->"

    if begin not in text or end not in text:
        raise ValueError(
            f"{doc_path} missing markers.\n"
            f"Add these around the section you want auto-populated:\n"
            f"{begin}\n...\n{end}"
        )

    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    new_block = (
        f"{begin}\n\n"
        f"{md_table}\n\n"
        f"{end}"
    )
    doc_path.write_text(pre + new_block + post)


def main() -> None:
    args = parse_args()
    per_run = pd.read_csv(Path(args.per_run))
    manifest = pd.read_csv(Path(args.manifest))

    # join for sanity: manifest provides gt_failure_time_s for traceability
    if "run_id" not in per_run.columns:
        raise ValueError("per_run.csv missing run_id")
    if "run_id" not in manifest.columns:
        raise ValueError("manifest missing run_id")

    df = per_run.merge(manifest[["run_id", "gt_failure_time_s"]], on="run_id", how="left")
    df["seq_id"] = df["run_id"].map(_extract_seq_id)

    if args.subset == "alfa22":
        df = df[df["seq_id"].isin(ALFA_22_SEQ_IDS)].copy()

    # Guard: if you still only have ~6 streams, fail loudly so you don’t publish bogus “ALFA-22” results.
    if args.subset == "alfa22" and df["seq_id"].nunique() < 20:
        raise SystemExit(
            f"[alfa_apples_to_apples] Only found {df['seq_id'].nunique()} ALFA-22 sequences in per_run.\n"
            f"Populate demos/uav/generated/streams with the 22 ALFA-22 CSVs, rerun the sweep,\n"
            f"then rerun this script."
        )

    rows: list[dict[str, Any]] = []
    if args.mode in ("spike", "both", "all"):
        r = _compute_binary_metrics(df, detected_col="spike_detected", lag_col="spike_lag_s")
        r["mode"] = "spike"
        rows.append(r)
    if args.mode in ("sustained", "both", "all"):
        r = _compute_binary_metrics(df, detected_col="sustained_detected", lag_col="sustained_lag_s")
        r["mode"] = "sustained"
        rows.append(r)
    if args.mode in ("or", "all"):
        r = _compute_or_metrics(df)
        r["mode"] = "or"
        rows.append(r)

    # OR row is only meaningful when we have both detector outputs available.
    if args.mode == "both":
        detected_or = (
            df["spike_detected"].fillna(False).astype(bool)
            | df["sustained_detected"].fillna(False).astype(bool)
        )
        lag_or = _compute_or_lag(df, ["spike_lag_s", "sustained_lag_s"])
        r = _compute_binary_metrics_from_series(df=df, detected=detected_or, lag_s=lag_or)
        r["mode"] = "or"
        rows.append(r)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[alfa_apples_to_apples] wrote: {out_csv}")

    md_table = _render_markdown_table(rows)
    if args.doc:
        doc_path = Path(args.doc)
        _inject_into_doc(doc_path, md_table)
        print(f"[alfa_apples_to_apples] updated doc: {doc_path}")


if __name__ == "__main__":
    main()
