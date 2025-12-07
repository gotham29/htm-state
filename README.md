# HTM-State  
**Continuous online anomaly learning and operator/system state awareness across domains.**

HTM-State is a real-time adaptive sensing system inspired by the neocortex.  
It continuously learns patterns in streaming data, estimates underlying state,  
detects transitions, and measures detection latency — all **without retraining or supervision**.

It is designed to operate across domains:

- Human workload / pilot cognition  
- Cyber intrusion / behavioral drift  
- Surgical performance change  
- Manufacturing process instability  
- UAV operator skill and safety  
- Edge intelligence / autonomous agents  

HTM-State provides a **unified operational pipeline** that can be deployed wherever  
behavioral pattern drift, safety monitoring, or cognitive state awareness is needed.

---

## 🌍 Why HTM-State exists

Conventional ML assumes:

- stationarity  
- batch retraining  
- labeled supervision  
- slow detection response  

**But real systems drift continuously.**

Human operators change mode.  
Networks degrade before they fail.  
Intrusions evolve stealthily.  
Machines deteriorate gradually.

HTM-State solves this by:
  
✔ **Online learning in nonstationary environments**  
✔ **No supervision or labels required**  
✔ **No retraining or fine-tuning needed**  
✔ **Sub-second response and change detection**
✔ **Works in domains where human + machine co-adapt**

---

## 🔁 Core Architecture

```text
Input features  ──► HTM Encoder + SP + TM  ─► Anomaly
                                            │
                                            ▼
                                     State Estimator (EMA)
                                            │
                                            ▼
                              Growth-based Spike Detector
                                            │
                                            ▼
                        Transition detection + latency metric
```

This structure is **domain-agnostic** — swapping input features yields new applications without rewriting logic.

---

## ✨ What the repo includes

- A pluggable HTM engine backend  
- Online state estimator (EMA/Fusion) 
- Growth-based spike detector  
- Detection lag metric  
- Live streaming visualizer  
- Offline evaluation tool  
- Synthetic workload dataset  
- A path to multiple domain demos  

---

# 🔬 Demo 1 — Real-Time Workload Transition Detection

This first demo illustrates HTM-State applied to **pilot-style psychomotor workload dynamics**  
(e.g., UAV control, piloting, teleoperation, manual tracking tasks).

### ✈️ Scenario

We simulate 1000 steps of control activity:

- First half: slower, smoother human control  
- Second half: higher tempo, more variability (increased workload)  

There is **no training data** and **no supervision**.

### 📌 Question

> Can the system autonomously detect this internal mode change just from streaming behavior?

✔ Yes — with detection lag ≈ **0.5 seconds** at 10 Hz.

This is significant because:
* conventional anomaly detectors require retraining
* supervised workload models need labeled sessions
* HTM-State learns on the fly and adapts autonomously

---

## 💻 Running Demo 1 (Offline Evaluation)

```bash
python -m scripts.offline_demo_detection_lag \
    --csv demos/workload_demo/synthetic_workload.csv \
    --backend htm \
    --rate-hz 10
```

Example output:

```text
Processed 1000 steps...
Using ground-truth toggle_step = 501
Detection lag: 5 steps
Detection lag: 0.500 s at 10 Hz
```

### ➤ Interpretation

HTM-State detects the workload shift  
**within half a second**  
of its onset.

That represents near-real-time awareness without supervision.

---

## 🎥 Live Visualization

```bash
python -m scripts.live_demo_state --backend htm --rate-hz 10
```

This shows two scrolling plots:

1. **Control signals**  
2. **HTM State + detected spikes**  

Spikes at transition points reflect **detected workload shifts**.

> Additional small spikes typically represent exploratory deviations or behavioral anomalies — useful for safety monitoring and drift awareness.

---
## 🎥 Demo 1 — Live Transition Animation

![HTM-State Demo 1 Live Transition](docs/gifs/demo1.gif)

*HTM-State continuously learns operator behavior in real time.  
This visualization shows the workload regime transition and HTM-State’s spike response:

* blue line → system’s real-time state estimate  
* orange marker → recognition of regime shift  

Detection occurs **within half a second**, without any offline learning.

---

## 🧠 Why Demo 1 matters

Demo 1 validates:

- ✔ online learning  
- ✔ unsupervised change detection  
- ✔ fast response  
- ✔ streaming embodiment  
- ✔ generality of approach  

This forms the baseline for domain expansion.
# 🚀 What Demo 1 proves

✔ HTM-State reacts in sub-second time  
✔ It requires **no labeled training data**  
✔ It adapts online like a human observer

---

# 🚧 Demo 2 — Cyber Behavioral Drift Detection *(coming soon)*

- Feature input: packet statistics, flow features, sequence patterns  
- Goal: detect stealthy gradual intrusions  
- Expected behavior: HTM-State reacts before classifiers flag attack states  

> This demo will illustrate how the same pipeline detects shifts in network behavior without oracles.

---

# 🏥 Demo 3 — Healthcare Operator Workload *(coming soon)*

- ICU nurse / surgical motion workload detection  
- Real-time fatigue shift / performance change monitoring  
- Valuable for patient safety, staffing load, augmented reality assistive systems  

---

# 🛠 Architecture Components

### ✔ HTM Backend

Encoders + Spatial Pooler + Temporal Memory using biologically-inspired learning.

### ✔ State Engine

Smooths anomaly into interpretable state estimates.

### ✔ Spike Detector

Detects transitions via growth differential logic.

### ✔ Detection Lag Metric

Measures adaptation time — critical in safety systems.

---

# 📦 Development Roadmap

| Phase  | Target                       |
|-------|------------------------------|
| Demo 1 | pilot workload transition    |
| Demo 2 | cyber drift detection        |
| Demo 3 | healthcare workload          |
| Demo 4 | industrial predictive change |
| Demo 5 | UAV safety horizon estimation |

---

# 🤝 Contributing

Future collaborators welcome —  
especially for new datasets in cyber, healthcare, robotics, or autonomy.

---

# 📌 Want to collaborate?

If you are interested in safety monitoring, autonomy, performance assessment,  
pilot modeling, cybersecurity drift detection, or cognitive systems — get in touch.

# 📧 Contact / Project Lead

Sam Heiserman  
Creator — HTM-State  
sheiser1@binghamton.edu