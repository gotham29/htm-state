## 🔐 Demo 2 — Cyber Behavior Drift Detection (UNSW-NB15)

Cyber systems drift continuously — often without clear attack signatures.  
This demo applies HTM-State to **streaming packet-flow behavior** from UNSW-NB15,  
detecting subtle and large-scale behavioral shifts in real time.

### 🔍 Scenario

We simulate a streaming sequence with **three true drift boundaries**:

* stable period  
* small statistical change  
* larger behavioral shift  

Ground-truth boundary times are marked visually with **vertical dashed red lines**.

### 📌 Question

> Can HTM-State detect emerging cyber behavior shifts  
> *without* retraining, classifiers, or labels?

✔ Yes — it learns online and responds autonomously.

### 💻  Offline Evaluation

Example command:

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

This represents **model-free cyber drift detection** using the same core pipeline that detected human workload changes.

### 🎥 Live Visualization

```bash
python -m scripts.live_demo_cyber \
    --csv demos/cyber_demo/unsw_cyber_stream.csv \
    --rate-hz 10
```

Live visualization shows:

- selected network features (e.g., rate, sload, dload)
- HTM cyber-state (anomaly-driven state estimate)
- true drift boundaries (red dashed lines)
- detected drift spikes (orange dots)
- magenta lag bars quantifying detection latency

This demonstrates **domain generality** —  
HTM-State adapts online whether its input is human control, network traffic,  
or machine sensor data.