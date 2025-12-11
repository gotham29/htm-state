## 🔐 Demo 2 — Cyber Behavior Drift Detection (UNSW-NB15)

Cyber systems drift continuously — often without clear attack signatures.  
This demo applies HTM-State to **streaming network-flow behavior** from the UNSW-NB15 dataset,  
detecting **subtle and large-scale behavioral shifts** fully online and without retraining.

### 🔍 Scenario

We simulate a streaming sequence containing **three true drift boundaries**:

* stable period  
* small statistical change  
* larger behavioral shift  

Ground-truth boundaries are visualized using **vertical dashed red lines** in the live view.

### 📌 Question

> Can HTM-State detect emerging cyber behavior shifts  
> *without* retraining, classifiers, or labels?

✔ Yes — HTM-State learns online from raw features  
and adapts autonomously to shifting behavior distributions.

### 💻  Offline Evaluation

Run offline drift evaluation:

```bash
python -m scripts.offline_demo_cyber \
    --csv demos/cyber_demo/unsw_cyber_stream.csv \
    --rate-hz 10
```

Example output:

```text
Found 3 drift boundaries at steps: [500, 1000, 1500]

=== Drift Detection Results ===
Drift 0: boundary at step 500 (t=50.000s) → detected at step 535 (t=53.500s), lag = 35 steps (3.500 s)
Drift 1: boundary at step 1000 (t=100.000s) → detected at step 1073 (t=107.300s), lag = 73 steps (7.300 s)
Drift 2: boundary at step 1500 (t=150.000s) → detected at step 1534 (t=153.400s), lag = 34 steps (3.400 s)

Average detection lag over 3 drifts: **4.7 s**
```

This demonstrates **model-free cyber drift detection** using
the same unsupervised online pipeline validated in Demo 1.

### 🎥 Live Visualization

```bash
python -m scripts.live_demo_cyber \
    --csv demos/cyber_demo/unsw_cyber_stream.csv \
    --rate-hz 10
```

### 🔎 Interpretation
* HTM-State again detects the known shifts quickly & precisely, but not as much as Demo 1.

### 📌 Takeaway
* This second promising result supports the **domain generality** of HTM-State. 
