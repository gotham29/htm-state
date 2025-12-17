# HTM-State UAV Demo (ALFA Dataset)

This demo evaluates **HTM-State** on real UAV flight data from the **ALFA (Autonomous Learning Flight Arena)** dataset.  
Rather than relying on a small number of hand-selected examples, we perform an **offline sweep across all eligible ALFA runs**, spanning:

- Engine failures  
- Control-surface failures (aileron, rudder, elevator)  
- Multi-fault scenarios  
- No-failure baselines  

The goal is to assess, in a broad and systematic way, whether HTM-State shows **early, stable, and interpretable detection behavior** across diverse failure modes — and to quantify that behavior using simple, transparent metrics.

After the offline sweep, a small number of representative runs are selected for **live streaming visualizations**, which provide intuition but do not drive the reported results.

## Dataset coverage

This demo uses flight logs from the **ALFA UAV dataset**, after preprocessing and stream generation.  
All evaluation is performed on generated per-run CSV streams located under:

```
demos/uav_demo/generated/
```

Each CSV corresponds to a single UAV flight and contains:
- Time index (`t_sec`)
- Modeled flight features (e.g., airspeed, climb, attitude, control inputs)
- A binary ground-truth boundary (`is_boundary`) when available

### Inclusion criteria

A run is included in the offline sweep if **all** of the following hold:

- The generated CSV contains required columns:
  - `t_sec`
  - `is_boundary`
- All requested feature columns are present
- The run represents one of the following scenario types:
  - `engine_failure`
  - `aileron_failure`
  - `rudder_failure`
  - `elevator_failure`
  - `multi_fault` (e.g., combined aileron + rudder failures)
  - `no_failure`

### Exclusion criteria

A run is excluded if **any** of the following hold:

- The scenario has **no ground-truth failure boundary** (e.g., `no_ground_truth`)
- Required columns are missing
- The generated stream is malformed or empty

Excluded runs are logged explicitly for transparency.

### Failure-type classification

Failure type is inferred directly from the source folder name used to generate each stream.  
Multi-fault scenarios are treated as a distinct category rather than being excluded.

Examples:
- `carbonZ_..._engine_failure` → `engine_failure`
- `carbonZ_..._rudder_left_failure` → `rudder_failure`
- `carbonZ_..._left_aileron__right_aileron__failure` → `multi_fault`
- `carbonZ_..._no_failure` → `no_failure`

This lightweight name-based classification allows the sweep to scale cleanly across the entire ALFA dataset without additional metadata.

## Metrics definition

Each UAV run is evaluated independently using a small set of transparent, time-aligned metrics derived from the HTM-State output. These metrics are designed to capture **timeliness**, **stability**, and **false-alarm behavior** relative to a known failure boundary.

### Primary metric: detection lag

Detection lag measures how quickly HTM-State responds **after** a ground-truth failure is injected.

Two complementary detection mechanisms are evaluated:

#### 1. Spike detection lag
- A **spike** is defined as a sharp increase in HTM-State relative to its recent history.
- Detection lag is computed as the elapsed time between the failure boundary and the **first spike occurring at or after the boundary**.

This captures fast, edge-like responses to abrupt changes.

#### 2. Sustained elevation lag
- A **sustained elevation** occurs when HTM-State remains above a threshold for a minimum hold duration.
- The threshold is computed from **pre-boundary state statistics** (median + *k*·MAD or equivalent).
- Detection lag is the elapsed time between the failure boundary and the first sustained elevation event.

This captures slower but more stable regime changes.

Both lags are reported in **seconds** when a boundary is present.

### Secondary metric: false alarms before boundary

False alarms quantify spurious detections **before** the failure occurs.

- Measured as the **rate of spike events per minute** in the pre-boundary interval.
- For `no_failure` scenarios, this metric is computed over the entire run.

Low false-alarm rates indicate stability under nominal conditions.

### Secondary metric: post-boundary persistence

Post-boundary persistence measures how strongly HTM-State remains elevated after a failure.

- Defined as the **fraction of post-boundary timesteps** where HTM-State exceeds the pre-boundary threshold.
- Values near 1.0 indicate sustained abnormal state; values near 0.0 indicate weak or transient response.

This metric complements detection lag by capturing **severity and consistency**, not just speed.

### Applicability notes

- For runs without a failure boundary (`no_failure`), detection lag metrics are not applicable.
- All metrics are computed without using future information beyond the current timestep.

## Results summary

The offline sweep produces two primary result tables:

1. A **per-run results table**, containing one row per UAV flight
2. An **aggregated summary table**, grouped by failure type

These tables are generated automatically and saved alongside the demo results.

### Per-run results table

The per-run table provides a complete, auditable record of HTM-State behavior on each individual flight.

Each row corresponds to a single generated UAV stream.

| Column | Description |
|------|------------|
| `run_id` | Unique identifier (derived from source folder / CSV name) |
| `failure_type` | One of: `engine_failure`, `aileron_failure`, `rudder_failure`, `elevator_failure`, `multi_fault`, `no_failure` |
| `has_boundary` | Boolean indicating whether a ground-truth boundary exists |
| `boundary_time_s` | Failure injection time (seconds), if applicable |
| `spike_detected` | Whether a spike was detected after the boundary |
| `spike_lag_s` | Detection lag to first spike (seconds) |
| `sustained_detected` | Whether a sustained elevation was detected |
| `sustained_lag_s` | Detection lag to sustained elevation (seconds) |
| `false_alarms_spm` | Spike rate before boundary (spikes per minute) |
| `post_elev_frac` | Fraction of post-boundary time state is elevated |
| `n_spikes_total` | Total number of spikes during the run |

This table is used both for quantitative reporting and for selecting representative runs for visualization.

**Output file:** `results/uav_sweep/per_run.csv`

### Aggregated results by failure type

To summarize performance across the dataset, per-run metrics are aggregated by inferred failure type.

| failure_type | n_runs | spike_detect_rate | median_spike_lag_s | sust_detect_rate | median_sust_lag_s | median_false_alarms_spm | median_post_elev_frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| engine_failure |  |  |  |  |  |  |  |
| aileron_failure |  |  |  |  |  |  |  |
| rudder_failure |  |  |  |  |  |  |  |
| elevator_failure |  |  |  |  |  |  |  |
| multi_fault |  |  |  |  |  |  |  |
| no_failure |  | N/A | N/A | N/A | N/A |  | N/A |

Notes:
- Detection rates are computed as the fraction of runs where detection occurred.
- Median lags are computed only over detected runs.
- For `no_failure`, detection lag metrics are not applicable; false alarms are the primary indicator.

**Output file:** `results/uav_sweep/summary_by_type.csv`

## Representative live visualizations

While the offline sweep provides comprehensive quantitative coverage, a small number of runs are selected for **live streaming visualizations** to build intuition and demonstrate how HTM-State evolves over time.

These figures are illustrative only; all reported metrics come from the offline sweep.

### Selection criteria

Representative runs are selected deterministically from the per-run results table:

- **Typical case**: run with detection lag closest to the median for a given failure type
- **Hard case**: run in the upper tail of detection lag (e.g., ~90th percentile) that is still successfully detected
- **Baseline case**: a `no_failure` run with low false-alarm rate
- **Miss case (if any)**: a run where detection did not occur, included for transparency

This selection ensures that live plots are **representative, not cherry-picked**.

### Visualization method

Each selected run is visualized using the live streaming demo:
- Flight signals (display-normalized)
- HTM-State output
- Spike detections
- Sustained elevation threshold and events
- Ground-truth failure boundary (when present)

### Figure organization

Generated figures are stored under:

```
results/uav_sweep/figures/{failure_type}/{run_id}.png
```

## Reproducibility

### 1. Prepare raw ALFA logs

Run from the repo root:

    python scripts/uav_select_and_copy.py \
      --processed-dir /path/to/ALFA/processed \
      --repo-root /path/to/htm-state \
      --include-optional

Raw scenarios are copied under:

```
demos/uav_demo/raw/
```

### 2. Generate per-run UAV stream CSVs

    python scripts/generate_uav_stream.py \
      --in demos/uav_demo/raw \
      --out demos/uav_demo/generated

### 3. Run the offline sweep

    python scripts/run_offline_uav_all.py \
      --generated-dir demos/uav_demo/generated \
      --outdir results/uav_sweep

This produces:
- `results/uav_sweep/per_run.csv`
- `results/uav_sweep/summary_by_type.csv`
- `results/uav_sweep/coverage.csv`

### 4. Generate live visualizations (optional)

    python scripts/run_live_demo_uav_selected.py \
      --per-run results/uav_sweep/per_run.csv \
      --outdir results/uav_sweep/figures
