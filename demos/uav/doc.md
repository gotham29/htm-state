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

> **Video (placeholder):**  
> Short MP4 clip of the same run with real-time animation (to be added).

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

## Core live demo roster (4 runs)

For the **live demo**, we focus on **four canonical runs** — one per response regime.
These runs are chosen for *clarity and interpretability*, not peak metrics.

> **Live demo = 4 runs total**

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

---

### Typical spike detection

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
```

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
