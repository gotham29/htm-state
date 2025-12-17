# HTM-State  
**Continuous online anomaly learning and operator/system state awareness across domains.**

HTM-State is a real-time adaptive sensing system inspired by the neocortex.  
It continuously learns patterns in streaming data, estimates underlying state,  
detects transitions, and measures detection latency — all **without retraining or supervision**.

It is designed to operate across domains:

- Human workload / pilot cognition  
- Cyber intrusion / behavioral drift  
- UAV operator skill and safety  
- Edge intelligence / autonomous agents  

HTM-State provides a **unified operational pipeline** that can be deployed wherever  
behavioral pattern drift, safety monitoring, or cognitive state awareness is needed.

---

## 📚 Contents

- [🌍 Why HTM-State exists](#-why-htm-state-exists)
- [🔁 Core Architecture](#-core-architecture)
- [✨ What the repo includes](#-what-the-repo-includes)
- [⚡ Quickstart](#-quickstart)
- [✈️ Demo 1 — UAV Failure Detection (ALFA UAV)](#️-demo-1--uav-failure-detection-alfa-uav)
- [🔐 Demo 2 — Cyber Behavior Drift Detection (UNSW-NB15)](#-demo-2--cyber-behavior-drift-detection-unsw-nb15)
- [🛠 Architecture Components](#-architecture-components)
- [📦 Development Roadmap](#-development-roadmap)
- [🤝 Contributing](#-contributing)
- [📧 Contact / Project Lead](#-contact--project-lead)

---

## 🌍 Why HTM-State exists

Conventional ML assumes:

- stationarity  
- batch retraining  
- labeled supervision  
- slow detection response  

**But real systems drift continuously — often subtly.**

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
- Persistence detector (median + k*MAD with hold-time)  
- Detection lag metric  
- Live streaming visualizer  
- Offline evaluation tool  
- Domain demo datasets + scripts (UAV + cyber today) 

---

## 🧩 API (Coming soon)

A minimal, importable API for using HTM-State in notebooks and production code:

- `HTMSession` for streaming anomaly + state (`step(feats, learn=...)`)
- A unified `SpikeDetector` interface
- Helpers to build feature ranges and encoder configs
- Example: “train on normal → freeze learning → monitor in deployment”

Tracked in: `docs/api.md` (placeholder)

---

## ⚡ Quickstart

```bash
# 1) Create & activate env (example)
conda create -n htm_env python=3.9 -y
conda activate htm_env

# 2) Install dependencies (from repo root)
pip install -r requirements.txt

# 3) Run Demo 1 (UAV failure scenarios)
python -m scripts.offline_demo_uav \
    --csv demos/uav_demo/generated/uav_engine_failure.csv \
    --rate-hz 10

python -m scripts.live_demo_uav \
    --csv demos/uav_demo/generated/uav_engine_failure.csv \
    --rate-hz 10

# 4) Run Demo 2 (cyber drift)
python -m scripts.offline_demo_cyber \
    --csv demos/cyber_demo/unsw_cyber_stream.csv \
    --rate-hz 10

python -m scripts.live_demo_cyber \
    --csv demos/cyber_demo/unsw_cyber_stream.csv \
    --rate-hz 10
```

Once those are working, you can tweak spike detector + HTM parameters via CLI flags  
to explore different sensitivities, detection latencies, and response profiles.

---

## ✈️ Demo 1 — UAV Failure Detection (ALFA UAV)

This demo evaluates **HTM-State on real UAV flight data** from the
**ALFA (Autonomous Learning Flight Arena)** dataset.

HTM-State continuously learns normal flight dynamics online from multivariate
telemetry (airspeed, climb rate, altitude, control behavior), then detects
**failure onset and regime change** without supervision or retraining.

### What this demo shows

- **Unsupervised online learning** on nonstationary UAV telemetry  
- **Fast spike-based detection** of abrupt failures  
- **Sustained state elevation** for persistent or non-compensable faults  
- **Quantified detection latency**, not just anomaly scores  
- **Systematic evaluation across all ALFA runs**, not cherry-picked cases

### Evaluation approach

- An **offline sweep** across all eligible ALFA failure scenarios
  (engine, control-surface, multi-fault, and no-failure baselines)
- Transparent metrics:
  - detection rate
  - detection lag (seconds)
  - false alarms before failure
  - post-failure persistence
- A small number of **representative live visualizations** for intuition only

👉 **Full UAV demo (offline results, figures, live plots):**  
[`demos/uav/doc.md`](demos/uav/doc.md)

---

## 🔐 Demo 2 — Cyber Behavior Drift Detection (UNSW-NB15)

HTM-State detects **behavioral drift** in streaming network flow statistics  
derived from the UNSW-NB15 dataset. The system adapts online to shifting  
traffic patterns and surfaces **emerging anomalies without signatures** —  
no supervised classifiers or retraining loops.

👉 **Full demo (offline + live + details):**  
[`docs/demos/cyber_demo.md`](docs/demos/cyber_demo.md)

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

## 📦 Development Roadmap

| Phase  | Target |
|-------|------------------------------|
| Demo 1 | UAV failure detection   |
| Demo 2 | Cyber drift detection   |
| Demo 3 | ?                       |

---

## 🤝 Contributing

Future collaborators welcome — especially for new datasets in  
cyber, healthcare, robotics, manufacturing, autonomy, or aviation.

---

## 📌 Want to collaborate?

If you're interested in safety monitoring, autonomy, behavior modeling,  
cybersecurity drift detection, or cognitive systems — get in touch.

## 📧 Contact / Project Lead

Sam Heiserman  
Creator — HTM-State  
sheiser1@binghamton.edu