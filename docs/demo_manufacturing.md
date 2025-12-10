# Demo 4 — Real-Time Manufacturing Process Drift Detection

This demo applies HTM-State to **manufacturing process behavior**  
(e.g., a production line, machining cell, or assembly station) to detect  
**slow drift, regime changes, and emerging instability** using the same  
online anomaly → state → spike pipeline as prior demos.


## 🎯 Domain / Use-Case

- **Primary context**: continuous manufacturing / assembly / machining lines

- **Typical signals**: vibration energy, spindle/load torque, feed rate, line speed, temperature, part/time metrics

- **Operational goal**: early detection of **process drift, instability, or impending faults**

This supports **predictive maintenance**, **quality protection**, and **line stability**  
without needing a supervised model per machine or per product.

## 🔍 Scenario

We stream a synthetic manufacturing line over time as a sequence of regimes:
- **Baseline stable production**

- **Drifting process** (e.g., tool wear, misalignment, temperature rise)

- **High-drift / pre-fault regime** where stability is threatened

In the synthetic dataset, drift is expressed through:

- slow changes in vibration and load

- increased variance in process metrics

- subtle but accumulating deviation in operating point

For this demo, we embed **two regime boundaries** in the stream:

- **step 600** — onset of drift from baseline
- **step 1450** — transition into high-drift / pre-fault regime

This mirrors Demo 1, Demo 2, and Demo 3:  
**regime shifts with detection latency measurement.**

## 📌 Core Question
> Can HTM-State detect **emerging manufacturing process drift**  
> fast enough to matter for **quality and maintenance decisions**,  
> *without* retraining, labels, or supervised classifiers?

---
## 💻 Offline Evaluation

### ▶️ Run

``` bash
python -m scripts.offline_demo_manufacturing \
    --csv demos/manufacturing_demo/line_stream.csv \
    --rate-hz 10
```

### 🧾 Example Output

```
Loaded 2000 timesteps.

=== Manufacturing Drift Detection Results ===
Transition 0: boundary at step 600  → detected at step 642, lag = 42 steps (4.200 s)
Transition 1: boundary at step 1450 → detected at step 1483, lag = 33 steps (3.300 s)

Average detection lag over 2 transitions: 3.8 s
```

🟢 In a typical configuration, HTM-State reacts within a few seconds @ 10 Hz  
to each shift in process regime.

🟢 This represents **model-free process drift detection** using HTM-State —  
🟢 no labels, no per-line classifier, and no retraining loop.