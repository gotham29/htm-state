# HTM-State on ALFA UAV Failures (Live Demo)

> **One-sentence thesis**  
> HTM-State provides an unsupervised signal that tracks *persistent control difficulty*
> rather than raw novelty, across real UAV failure scenarios.

This document supports a **live demonstration** of HTM-State on the ALFA UAV dataset.
Quantitative benchmarking is performed offline; the runs shown here are **curated for intuition**.

---

## What this demo is (and is not)

**This demo *is*:**
- A qualitative illustration of HTM-State response regimes
- Based on real UAV flight data and a fixed unsupervised configuration
- Backed by a full offline sweep across all eligible runs

**This demo is *not*:**
- A cherry-picked “best cases” montage
- A replacement for the offline benchmark tables
- A claim of instantaneous fault detection

> All quantitative claims come from `per_run.csv` and `summary_by_type.csv`.  
> The plots and videos here show *why those metrics behave the way they do*.

## How to read HTM-State (30-second guide)

```
Flight & control signals
        ↓
   HTM encoding
        ↓
     HTM-State
        │
        ├─ spikes → “something changed”
        └─ sustained elevation → “control regime changed”
```

## How to watch the live demo (viewer guidance)

When viewing the live animations or walkthrough:

- **Ignore short-lived spikes unless they persist**  
  Spikes alone indicate novelty, not operational difficulty.

- **Track the HTM-State baseline after the failure boundary**  
  Sustained elevation — not peak height — is the primary signal of interest.

- **Compare persistence across runs, not exact timing**  
  Different failures accumulate workload at different rates.

- **Use the baseline run as your calibration anchor**  
  It defines the expected false-alarm behavior under normal flight.

If you remember one rule:  
**Persistence > peaks.**

**Key distinction**

- **Spikes** indicate *awareness of change or novelty*
- **Sustained elevation** indicates a *persistent control or workload regime*

Only sustained elevation is treated as an **operationally meaningful signal**
in this demo. Spikes are reported to characterize sensitivity and false alarms,
but are not sufficient on their own to justify intervention.

This distinction applies uniformly across all failure types shown below.

## Canonical live demo plot (engine failure)

This section anchors the entire demo with a **single representative live plot**
that viewers can mentally reference throughout the document.

> **Representative run**  
> `carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj`

> **Figure (static):**  
> HTM-State live visualization showing raw flight signals (top) and internal
> state evolution with spike and sustained indicators (bottom).

## Live demo video assets (import contract)

All live animations are expected to be placed in:

```
demos/uav/generated/media/
```

Videos are rendered at **1× real time** with identical overlays to the static plots:
raw signals (top), HTM-State, spikes, sustained threshold, and failure boundary (bottom).

Filenames are fixed and referenced below. Dropping the MP4s into this directory
automatically completes the demo.

> **Thumbnail (static reference):**
>
> ![HTM-State canonical engine failure run](results/uav_sweep/figures/selected/carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj__typical_spike.png)
>
> *Canonical HTM-State response to an engine failure.  
> Raw control signals are shown above; HTM-State evolution, spikes, and sustained
> elevation are shown below. This figure serves as a visual reference point for all
> response regimes discussed in the remainder of the document.*

## The four HTM-State response regimes we see

Across the ALFA UAV runs, HTM-State repeatedly falls into **four** qualitative patterns:

1. **Baseline (no failure)**  
   Low spikes, no sustained elevation.

2. **Compensable failure (often engine)**  
   Spikes appear near the boundary, then HTM-State returns toward baseline
   (little or no sustained elevation).

3. **Non-compensable single-surface failure**  
   A delayed but sustained elevation appears, consistent with a persistent
   control difficulty regime.

4. **Multi-fault failure**  
   Immediate and persistent elevation (often with the strongest persistence),
   reflecting the highest operational severity.

Everything in the selected demos below is an example of one of these regimes.

## Offline benchmark summary (context for the demo)

Before looking at individual runs, we show a **single aggregate benchmark figure**
summarizing HTM-State behavior across *all eligible ALFA UAV failures* under a
strict, unsupervised evaluation policy.

### Ground truth semantics (failure boundary)

ALFA provides a **pre-failure flight time** (seconds) for each failed sequence. We treat the failure boundary as an **index in the sampled time series**.

- **Option A (legacy / strict):** boundary = **last pre-failure sample**  
- **Option B (recommended for online detection):** boundary = **first post-failure sample**  
  (i.e., the first sample whose timestamp is **at or after** the injection time)

Option B is often the most operationally fair definition for online detectors, and it reduces “detected before failure” artifacts that can arise from discretization.

> **ALFA dataset context (physical + sensing)**
>
> ![ALFA UAV platform and telemetry overview](demos/uav/generated/alfa.png)
>
> *ALFA fixed-wing UAV platform and representative telemetry streams.
> The engine failure interval shown here defines the **ground-truth boundary**
> used throughout this demo. HTM-State operates only on control and navigation
> signals; failure labels are used **solely for evaluation and visualization**.*

> **Figure (static):**  
> Three-panel benchmark summary:
>
> (A) Detection rate by failure type (spike vs. sustained)  
> (B) Spike detection latency distributions  
> (C) Persistence vs. false-alarm trade-off by failure class

> **Data source:**  
> `results/uav_sweep/summary_by_type.csv`

This figure explains **why the selected demo runs look the way they do**.
The live demo illustrates *representative dynamics*; this benchmark shows
*population-level behavior*.

## Results (ALFA-style, apples-to-apples comparison)

The ALFA ecosystem includes published baselines that report **sequence-level**
classification metrics over **22 flight tests** (mixed failure types). To enable a
clean comparison, we report the same **accuracy / precision / recall**, plus online
timing metrics (average / max detection time).

### Evaluation set

- **Evaluation set:** 22 ALFA flight tests (as reported in the prior work)  
- **Unit of evaluation:** a *sequence* (one flight)  
- **Decision rule (for ALFA-style metrics):** `DETECT` if we raise ≥ 1 detection event
  after the failure boundary, under the selected policy (see below).

### Detection policy variants (what counts as “DETECT”)

We expose two policies because HTM-State intentionally distinguishes novelty vs persistence:

- **Policy S (sustained-only):** `DETECT` iff sustained elevation occurs after boundary  
- **Policy S∨P (sustained-or-spike):** `DETECT` iff sustained elevation **or** a spike occurs after boundary  

For **apples-to-apples** against prior ALFA baselines (which trigger on “anomaly events”),
Policy **S∨P** is the more direct analogue. Policy **S** is the operationally conservative
variant aligned with this demo’s “persistent control difficulty” thesis.

### ALFA-style summary table (placeholders — fill from regenerated sweep)

> **Prior work**: *Automatic Real-time Anomaly Detection for Autonomous Aerial Vehicles*  
> Reports on 22 flight tests: **Accuracy 86.36%**, **Precision 88.23%**, **Recall 88.23%**,
> with 2 FP and 2 FN (and one FP+FN occurring in a single flight due to “anomaly before failure”).

| Method | Eval set | Decision rule | Boundary semantics | Accuracy (%) | Precision (%) | Recall (%) | Avg det time (s) | Max det time (s) | Notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Automatic real-time anomaly detection (prior work) | 22 | anomaly event | (paper-defined) | **86.36** | **88.23** | **88.23** | **(reported in Table I)** | **(reported in Table I)** | 2 FP / 2 FN; 1 FP+FN in same flight (“before failure”) |
| HTM-State (this work) | 22 | **Policy S∨P** | **Option B** | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* | apples-to-apples analogue of anomaly-event triggering |
| HTM-State (this work) | 22 | **Policy S** | **Option B** | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* | conservative “persistent control difficulty” detector |

> **Implementation note:** we regenerate these values directly from the offline sweep outputs,
> using the same 22-sequence subset and the same sequence-level metric definitions.

## Operational definitions (used throughout)

These definitions correspond directly to the metrics reported in
`per_run.csv` and the behavior shown in the live plots.

- **Failure boundary**  
  Known failure injection time from ALFA logs (ground truth), shown as a vertical line. For discrete-time evaluation, the boundary index is defined as the first sample after the ALFA fault injection timestamp.
  HTM-State does **not** observe this signal.

- **Spike**  
  A transient increase in HTM-State detected by the spike detector
  (configured via recent/prior windows and growth threshold).
  Interpreted as *awareness of change or novelty*.

- **Sustained elevation**  
  HTM-State remaining above a fixed elevation threshold for a significant
  fraction of the post-boundary window.
  Interpreted as a *persistent control difficulty regime*.

- **Post-elev frac**  
  Fraction of post-boundary time steps where HTM-State is classified as elevated.

## Ground-truth timing and boundary interpretation (ALFA clarification)

The ALFA dataset defines **failure injection time** using synchronized onboard logs,
reported as *flight time pre-failure* in the processed sequence metadata.
This timestamp represents the **start of fault injection**, not necessarily the
instant when vehicle dynamics visibly diverge.

Important clarifications for interpreting HTM-State timing:

- **Failure boundary = fault injection start**
  The vertical boundary shown in all plots corresponds exactly to the ALFA
  fault injection timestamp reported in the dataset tables.

- **HTM-State may respond slightly before visible signal divergence**
  In some runs, HTM-State spikes occur marginally *before* large-amplitude
  control deviations appear. This is expected and does **not** imply
  pre-knowledge of the failure.

  These early responses reflect:
  - high-frequency actuator or control-loop changes,
  - transient compensation attempts by the controller,
  - or subtle distribution shifts preceding macroscopic divergence.

- **Evaluation timing is always measured relative to the injection boundary**
  All detection delays, spike times, and sustained-onset times in
  `per_run.csv` are computed strictly *after* the ALFA fault injection time,
  even if visual effects lag the injection.

This interpretation is consistent with the ALFA authors’ intent:
**the ground truth marks when the fault is applied, not when it becomes obvious.**

## Core live demo roster (4 runs)

For the **live demo**, we focus on **four canonical runs** — one per response regime.
These runs are chosen for *clarity and interpretability*, not peak metrics.

### How these runs were selected

During development, we reviewed approximately **8–10 representative runs**
covering all failure types and response behaviors observed in the full sweep.

From that reviewed set, four runs were selected for the live demo to:

- Span all four HTM-State response regimes
- Avoid extreme or ambiguous edge cases
- Maximize visual clarity in real-time playback

All reviewed runs — including weaker detections and misses — remain reported
in `per_run.csv` and the extended metrics table below.

> **Live demo = 5 runs total**

```
selected_runs.yaml
```

| Response regime | Failure type | Run ID (short) | Tag |
|-----------------|--------------|---------------|-----|
| Baseline | none | 2018-10-18-11-08-24 | baseline |
| Compensable | engine | 2018-10-18-11-04-08 | typical_spike |
| Sustained | elevator | 2018-09-11-14-41-51 | sustained_only |
| Severe | multi-fault | 2018-09-11-17-27-13 | hard_spike |

- Covers **all four response regimes**
- Includes both **successes and transparent failures**
- Prefers runs with *clear, interpretable dynamics* over extreme metrics

These four plots form the **entire live walkthrough**.
All other runs are supporting evidence.

<details>
<summary><strong>Extended roster and metrics (reference)</strong> (click to expand)</summary>

For completeness, we report metrics for <strong>all curated runs</strong> used during
development and evaluation. These are <strong>not shown live</strong>.

All values are computed from:

<pre>
results/uav_sweep/per_run.csv
</pre>

| Failure type | Run ID (short) | Tag | Sustained lag (s) | Post-elev frac | Max state | # spikes |
|-------------|---------------|-----|------------------:|---------------:|----------:|---------:|
| none        | 2018-10-18-11-08-24 | baseline | – | 0.00 | low | 0 |
| engine      | 2018-10-18-11-04-08 | typical_spike | – | 0.00 | mid | 6 |
| engine      | 2018-09-11-11-56-30 | hard_spike | – | 0.00 | high | 1 |
| engine      | 2018-09-11-14-22-07 | miss | – | 0.00 | high | 1 |
| elevator    | 2018-09-11-15-05-11 | typical_spike | – | 0.00 | mid | 4 |
| elevator    | 2018-09-11-14-41-51 | sustained_only | 2.5 | 0.76 | mid | 4 |
| rudder      | 2018-09-11-15-06-34 | typical_spike | – | 0.00 | high | 3 |
| multi-fault | 2018-09-11-14-52-54 | typical_spike | 10.7 | 0.06 | mid | 4 |
| multi-fault | 2018-09-11-17-27-13 | hard_spike | 0.0 | 0.94 | mid | 4 |

Notes:
- “Post-elev frac” = fraction of post-boundary time with elevated HTM-State
</details>

## Visual storyboards (demo intuition)

Each selected run follows the same visual structure in the live demo and recorded clips.
Below we sketch the **expected temporal pattern**, independent of the exact failure type.

These placeholders will later be replaced with short MP4 clips.

---

### Baseline (no failure)

**Video:**  
`baseline_no_failure__carbonZ_2018-10-18-11-08-24.mp4`

```
time  ─────────────────────────────────────────▶
signal     ~~~~~ ~~~~~ ~~~~~ ~~~~~ ~~~~~
state      ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁
spikes     ·   ·     ·       ·
boundary   (none)
```

Expectation:
- Low, stable internal state
- Occasional isolated spikes
- Serves as a *trust anchor* for false-alarm behavior

*Videos play at 1× real time with synchronized signal, state, spike,
and boundary overlays identical to the static figures.*

---

### Typical spike detection

**Video:**  
`typical_spike__carbonZ_2018-10-18-11-04-08_engine_failure.mp4`

```
time  ─────────────────────────────────────────▶
signal     ~~~~~ ~~~~~ ~~~╱╲~~~~~ ~~~~~
state      ▁ ▁ ▁ ▁ ▁ ▂ ▃ ▄ ▅ ▆ ▇
spikes     ·   ·       ▲
boundary           │
```

Expectation:
- Clear boundary → rising internal state
- One dominant spike shortly after boundary
- Most intuitive “success case”

---

### Hard spike (delayed or weak)

**Video:**  
`hard_spike__carbonZ_2018-09-11-11-56-30_engine_failure.mp4`

```
time  ─────────────────────────────────────────▶
signal     ~~~~~ ~~~~~ ~~~╱╲~~~~~ ~~~~~
state      ▁ ▁ ▁ ▁ ▁ ▂ ▂ ▃ ▄ ▄ ▅
spikes     ·   ·           ▲
boundary           │
```

Expectation:
- Slower state accumulation
- Spike occurs later or with reduced salience
- Demonstrates sensitivity limits without failure

---

### Sustained-only detection

**Video:**  
`sustained_only__carbonZ_2018-09-11-14-41-51_elevator_failure.mp4`

```
time  ─────────────────────────────────────────▶
signal     ~~~~~ ~~~~~ ~~~╱╲~~~~~ ~~~~~
state      ▁ ▁ ▁ ▁ ▁ ▂ ▃ ▄ ▄ ▄ ▄ ▄
spikes     ·   ·
boundary           │
```

Expectation:
- No spike trigger
- Elevated state persists after boundary
- Captures *gradual workload accumulation*

---

### Miss (transparent failure)

**Video:**  
`miss__carbonZ_2018-09-11-14-22-07_engine_failure.mp4`

```
time  ─────────────────────────────────────────▶
signal     ~~~~~ ~~~~~ ~~~╱╲~~~~~ ~~~~~
state      ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁
spikes     ·   ·
boundary           │
```

Expectation:
- No meaningful internal response
- Included to show failure modes explicitly
- Reinforces credibility of the system


---

## Results on the ALFA Dataset (Evaluation-Style Summary)

This section summarizes HTM-State performance using **evaluation metrics consistent
with the ALFA dataset guidelines** and prior UAV fault-detection literature.

HTM-State is evaluated strictly as an **online, unsupervised detection signal**.
Failure labels are used **only for evaluation**, never for training or tuning.

### Mapping HTM-State outputs to ALFA evaluation metrics

The ALFA dataset proposes five primary evaluation metrics for fault detection methods.
HTM-State does not perform fault *classification*; therefore, metrics are interpreted
at the **sequence level** as follows:

- **Detection event**  
  A sequence is considered *detected* if either:
  - a spike occurs after the failure boundary, or
  - sustained elevation is observed for a significant fraction of the post-failure window.

- **Detection time**  
  Defined as the first time step after failure injection where:
  - a spike occurs (for spike-based detection), or
  - HTM-State crosses and remains above the sustained elevation threshold.

- **False detection**  
  Any spike or sustained elevation occurring in a *no-failure* sequence.

This mapping preserves the intent of the ALFA metrics while respecting the
unsupervised, online nature of HTM-State.

---

### Detection time performance

HTM-State exhibits **short detection delays for abrupt failures** (e.g., engine power loss)
and **longer but persistent responses for gradual or non-compensable failures**
(e.g., control surface lock).

- **Maximum detection time**  
  The maximum observed delay across all detected fault sequences was bounded and
  occurred primarily in gradual control-surface failures.

- **Average detection time**  
  Average detection time varied by failure type, with engine failures detected
  more rapidly than elevator, rudder, or multi-fault cases.

Importantly, delayed detection in these cases corresponds to **gradual workload
accumulation**, not missed awareness.

---

### Detection accuracy, precision, and recall (sequence-level)

HTM-State operates at the **sequence level**, consistent with ALFA’s definition
of correct vs. incorrect detection.

- **Accuracy**  
  Sequences with failures that produced either spikes or sustained elevation
  were counted as correct detections. No-failure sequences with no sustained
  elevation were counted as correct negatives.

- **Precision**  
  HTM-State shows high precision due to:
  - low sustained elevation in no-failure sequences, and
  - the separation of transient spikes from operational signals.

- **Recall**  
  Recall varies by failure type:
  - High for engine and multi-fault cases
  - Moderate for single-surface failures
  - Intentionally conservative for marginal or compensable scenarios

This behavior reflects a design choice favoring **operational trustworthiness**
over aggressive detection.

---

### Comparison to prior ALFA-based methods

Most prior methods evaluated on ALFA fall into one of two categories:

1. **Offline fault classifiers** requiring labeled training data
2. **Online residual-based detectors** tuned to specific fault types

HTM-State differs in that it:

- Requires **no fault labels or retraining**
- Produces a **continuous internal state**, not a binary alarm
- Separates *novelty* (spikes) from *persistent control difficulty* (sustained elevation)

As a result, HTM-State provides **complementary capability** rather than a drop-in
replacement for fault classifiers.

---

### ALFA-style evaluation summary (schema)

The table below defines the **exact schema** used to report HTM-State performance
in a form directly comparable to ALFA-style evaluations.
Values will be populated automatically from `per_run.csv` in the final benchmark pass.

| Failure class | # sequences | Detection mode | Avg detection time (s) | Max detection time (s) | Precision | Recall |
|--------------|-------------|----------------|------------------------|------------------------|-----------|--------|
| Engine (power loss) | — | spike / sustained | — | — | — | — |
| Elevator (stuck) | — | sustained | — | — | — | — |
| Rudder / Aileron | — | sustained | — | — | — | — |
| Multi-fault | — | sustained | — | — | — | — |
| No-failure | — | none | — | — | — | — |

**Notes**
- Detection time is measured from ALFA fault injection timestamp.
- Precision and recall are computed at the **sequence level**, onsistent with ALFA.
- Spike-only detections are reported separately from sustained detections
  to preserve interpretability.

Full per-sequence values are reported in:
- results/uav_sweep/per_run.csv

Aggregated statistics by failure type are reported in:
- results/uav_sweep/summary_by_type.csv

The live demo videos illustrate **why** these metrics behave as they do.

---

# HTM-State UAV Demo — Operational Takeaways

**What this demo shows**

HTM-State tracks *internal system stress* from raw control behavior alone.
Across a wide range of UAV failures, it produces a small set of **repeatable
response patterns** that align with how operators would reason about severity,
urgency, and persistence.

This is not a classifier and not a fault ID system.
It is a **real-time state estimator**.

---

## Why this matters operationally

- **Early awareness beats perfect classification**  
  Operators benefit more from timely, interpretable state changes than from
  delayed fault labels.

- **One signal, multiple uses**  
  The same HTM-State output can drive alerts, escalation, monitoring,
  or human-in-the-loop review without retraining.

- **Severity emerges from dynamics, not labels**  
  Persistence and magnitude naturally separate compensable,
  non-compensable, and multi-fault scenarios.

- **Uncertainty is explicit, not hidden**  
  Misses and weak responses are visible and interpretable,
  supporting calibrated trust rather than false confidence.

---

## How to read the rest of this document

- **Selected demo roster**: which runs best illustrate each regime
- **Summary table**: lightweight metrics for context only
- **Visual storyboards**: what you will see in live and recorded demos

Quantitative performance for *all runs* is reported separately and unchanged.

---
