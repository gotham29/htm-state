# UAV Demo (ALFA Dataset)

![ALFA UAV flight platform and telemetry context](generated/alfa.jpg)

This demo evaluates **HTM-State** on real UAV flight data from the **ALFA (Autonomous Learning Flight Arena)** dataset.

Rather than relying on a small number of hand-selected examples, we perform an **offline sweep across all eligible ALFA runs**, spanning:

- Engine failures  
- Control-surface failures (aileron, rudder, elevator)  
- Multi-fault scenarios  
- No-failure baselines  

The goal is to assess, in a broad and systematic way, whether HTM-State shows **early, stable, and interpretable detection behavior** across diverse failure modes — and to quantify that behavior using simple, transparent metrics.

After the offline sweep, a small number of representative runs are selected for **live streaming visualizations**, which provide intuition but do not drive the reported results.

## At a glance

### What a live run looks like

![Live HTM-State response during UAV engine failure](generated/figures/selected/engine_failure/carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj__typical_spike.png)

*Top: normalized flight signals. Bottom: HTM-State with spike and sustained detections relative to the injected failure boundary.*

**Figure 1** summarizes detection performance, latency, and persistence across all ALFA UAV failure types under a strict, unsupervised evaluation protocol.

![](generated/results/uav_sweep/figure1_summary.png)

> Spike-based and sustained-elevation detection capture complementary workload dynamics.  
> Control-surface and multi-fault scenarios exhibit higher persistence than compensable engine failures.

<details>
<summary><b>Dataset and evaluation scope</b></summary>

This demo uses flight logs from the **ALFA UAV dataset**, after preprocessing and stream generation.  
All evaluation is performed on per-run CSV streams located under:

```
demos/uav/generated/
```

Each CSV corresponds to a single UAV flight and is written under:

```
demos/uav/generated/streams/{failure_type}/{run_id}.csv
```
- Time index (`t_sec`)
- Flight and control features
- A binary ground-truth failure boundary (`is_boundary`) when available

Runs are **included** if required columns are present and a failure boundary exists; runs without ground truth or with malformed streams are **excluded** and logged.

Failure type is inferred directly from the source folder name (e.g., `engine_failure`, `rudder_failure`, `multi_fault`), allowing the sweep to scale without additional metadata.

Examples:
- `carbonZ_..._engine_failure` → `engine_failure`
- `carbonZ_..._rudder_left_failure` → `rudder_failure`
- `carbonZ_..._left_aileron__right_aileron__failure` → `multi_fault`
- `carbonZ_..._no_failure` → `no_failure`

This lightweight name-based classification allows the sweep to scale cleanly across the entire ALFA dataset without additional metadata.
</details>


<details>
<summary><b>Metrics definition</b></summary>

Each UAV run is evaluated independently using a small set of transparent, time-aligned metrics derived from the HTM-State output. These metrics are designed to capture **timeliness**, **stability**, and **false-alarm behavior** relative to a known failure boundary.

### Primary metric: detection lag

Detection lag measures how quickly HTM-State responds after a known failure boundary.

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
</details>

## Results summary

The offline sweep produces two primary result tables:

### Overview and key findings

The strict offline sweep across the ALFA UAV dataset demonstrates that **HTM-State exhibits consistent, fault-dependent behavior across diverse failure modes**, using a single unsupervised configuration and without per-scenario tuning.

Several high-level patterns emerge:

- **Detection behavior is strongly failure-type dependent.**  
  Control-surface and multi-fault scenarios are detected more reliably than engine failures, reflecting differences in how faults manifest in pilot control behavior.

- **Spike-based detection and sustained elevation capture complementary phenomena.**  
  Abrupt faults often produce transient novelty (spikes), while non-compensable control degradations produce prolonged state elevation. These mechanisms should not be conflated.

- **Detection latency is measured in seconds, not instantaneous steps.**  
  Median detection lags range from several seconds to tens of seconds depending on fault type, consistent with gradual workload emergence rather than threshold-triggered alarms.

- **False alarms remain bounded under nominal conditions.**  
  Pre-boundary spike rates remain low across all failure types, including no-failure baselines, indicating stability rather than hypersensitivity.

- **Post-boundary state persistence distinguishes compensable vs. sustained failures.**  
  Engine failures tend to show transient responses with low persistence, while elevator, aileron, and multi-fault scenarios exhibit sustained elevation consistent with prolonged workload impact.

These trends are summarized visually in **Figure 1**, which aggregates detection rate, latency, and persistence across all evaluated runs.

> **Figure 1 (Summary).** HTM-State benchmark performance on the ALFA UAV dataset under a strict, unsupervised evaluation protocol.  
> (A) Detection rates by failure type for spike-based and sustained-elevation mechanisms.  
> (B) Distribution of spike detection latency across failure types.  
> (C) Relationship between post-boundary state persistence and false-alarm rate, highlighting separation between compensable and non-compensable failures.

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

**Output file:** `demos/uav/generated/results/uav_sweep/summary_by_type.csv`

## Quantitative results

### Aggregated results by failure type

Notes:
- Detection rates are computed as the fraction of runs where detection occurred.
- Median lags are computed only over detected runs.
- For `no_failure`, detection lag metrics are not applicable; false alarms are the primary indicator.

Notably, HTM-State exhibits increasing post-boundary persistence from engine to control-surface to multi-fault scenarios, consistent with increasing workload severity rather than mere novelty.

**Output file:** `results/uav_sweep/summary_by_type.csv`

## Live animations (recommended for first-time readers)

These short clips show the **two-panel live view** used throughout this demo:

- **Top panel**: selected flight/control signals (what the airframe is doing / what the controller is commanding).
- **Bottom panel**: HTM-State outputs (state, spikes, sustained elevation), plus the **strict benchmark boundary** when available.

### 1) Baseline (no failure)

Use this as a “what normal looks like” reference. There is **no ground-truth boundary** in no-failure runs, so we show a
representative stable window (often mid-run or late-run, after warm-up).

<video controls playsinline width="900">
  <source src="generated/media/uav_no_failure_baseline.mp4" type="video/mp4">
</video>

**How to read it:** you want **low anomaly/spike activity** and no sustained elevation over long stretches.

### 2) Engine failure (transition window)

This clip focuses on the failure transition window for an engine-failure run.

<video controls playsinline width="900">
  <source src="generated/media/uav_engine_failure_transition.mp4" type="video/mp4">
</video>

**How to read it:** the vertical boundary line marks the strict “toggle” time. A clean detection shows (i) **spikes after**
the boundary and/or (ii) a sustained elevated state after the boundary with limited pre-boundary false alarms.
This particular clip is a **hard case** (useful as a miss/ambiguity example).

### 3) Control-surface multi-fault (transition window)

This clip shows a “control surfaces” multi-fault style run (e.g., rudder/aileron related).

<video controls playsinline width="900">
  <source src="generated/media/uav_control_surface_failure_transition.mp4" type="video/mp4">
</video>

**How to read it:** compare pre-boundary vs post-boundary behavior. In tough cases the signals may degrade gradually,
and strict scoring will penalize **pre-boundary spikes** as false alarms even if they look like plausible precursors.

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
demos/uav/generated/figures/selected/{failure_type}/{figure}.png
```

## Reproducibility

### 1. Prepare raw ALFA logs

Run from the repo root:

    python scripts/copy_data_uav.py \
      --processed-dir /path/to/ALFA/processed \
      --include-optional

Raw scenarios are copied under:

```
demos/uav/raw/
```

### 2. Generate per-run UAV stream CSVs

    python scripts/generate_stream_uav.py \
      --raw-dir demos/uav/raw \
      --out-dir demos/uav/generated/streams

Artifacts are written under:
- `demos/uav/generated/results/uav_sweep/` (coverage/per_run/summary CSVs + Figure 1)
- `demos/uav/generated/figures/selected/` (representative per-run PNGs)
- `demos/uav/generated/media/` (short MP4 clips used in this doc)

### 3. Run the offline sweep

    python scripts/run_offline_uav.py \
  --generated-dir demos/uav/generated \
  --outdir demos/uav/generated/results/uav_sweep

This produces:
- `demos/uav/generated/results/uav_sweep/per_run.csv`
- `demos/uav/generated/results/uav_sweep/summary_by_type.csv`
- `demos/uav/generated/results/uav_sweep/coverage.csv`

### 4. Generate live visualizations (optional)

    python scripts/run_live_uav.py \
      --per-run demos/uav/generated/results/uav_sweep/per_run.csv \
      --coverage demos/uav/generated/results/uav_sweep/coverage.csv \
      --outdir demos/uav/generated/figures/selected

<details>
<summary><b>Representative plot gallery (auto-generated)</b></summary>

<!-- AUTO-GALLERY:BEGIN -->

> This section is generated from `demos/uav/selected_runs.yaml` (do not edit by hand).

### `no_failure`

| Example | Plot | Run ID | Quick metrics |
|---|---:|---|---|
| **Baseline (no failure)** | ![](generated/figures/no_failure/carbonZ_2018-10-18-11-08-24_no_failure__baseline.png) | `carbonZ_2018-10-18-11-08-24_no_failure` | — |

### `engine_failure`

| Example | Plot | Run ID | Quick metrics |
|---|---:|---|---|
| **Typical spike** | ![](generated/figures/engine_failure/carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj__typical_spike.png) | `carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj` | — |
| **Hard spike** | ![](generated/figures/engine_failure/carbonZ_2018-09-11-11-56-30_engine_failure__hard_spike.png) | `carbonZ_2018-09-11-11-56-30_engine_failure` | — |
| **Miss** | ![](generated/figures/engine_failure/carbonZ_2018-09-11-14-22-07_2_engine_failure__miss.png) | `carbonZ_2018-09-11-14-22-07_2_engine_failure` | — |

### `elevator_failure`

| Example | Plot | Run ID | Quick metrics |
|---|---:|---|---|
| **Typical spike** | ![](generated/figures/elevator_failure/carbonZ_2018-09-11-15-05-11_1_elevator_failure__typical_spike.png) | `carbonZ_2018-09-11-15-05-11_1_elevator_failure` | — |
| **Sustained-only** | ![](generated/figures/elevator_failure/carbonZ_2018-09-11-14-41-51_elevator_failure__sustained_only.png) | `carbonZ_2018-09-11-14-41-51_elevator_failure` | — |

### `rudder_failure`

| Example | Plot | Run ID | Quick metrics |
|---|---:|---|---|
| **Typical spike** | ![](generated/figures/rudder_failure/carbonZ_2018-09-11-15-06-34_2_rudder_right_failure__typical_spike.png) | `carbonZ_2018-09-11-15-06-34_2_rudder_right_failure` | — |
| **Hard spike** | ![](generated/figures/rudder_failure/carbonZ_2018-09-11-15-06-34_1_rudder_right_failure__hard_spike.png) | `carbonZ_2018-09-11-15-06-34_1_rudder_right_failure` | — |

### `multi_fault`

| Example | Plot | Run ID | Quick metrics |
|---|---:|---|---|
| **Typical spike** | ![](generated/figures/multi_fault/carbonZ_2018-09-11-14-52-54_left_aileron__right_aileron__failure__typical_spike.png) | `carbonZ_2018-09-11-14-52-54_left_aileron__right_aileron__failure` | — |
| **Hard spike** | ![](generated/figures/multi_fault/carbonZ_2018-09-11-17-27-13_1_rudder_zero__left_aileron_failure__hard_spike.png) | `carbonZ_2018-09-11-17-27-13_1_rudder_zero__left_aileron_failure` | — |

<!-- AUTO-GALLERY:END -->

</details>

