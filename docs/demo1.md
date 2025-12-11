### 🔧 Scenario

We simulate a streaming human-control task with **1000 timesteps**:

- First half: slower, smoother human control  
- Second half: higher tempo, more variability (increased workload)  

There is **no training phase**, **no labels**, and **no supervision**.

### 📌 Question

> Can HTM-State autonomously detect this behavioral mode change  
> *directly from streaming behavior*?

✔ Yes — typically within **1–2 seconds** at 10 Hz.  
A representative run is shown below.

This matters because:
* conventional anomaly detectors require retraining  
* supervised workload models need labeled sessions  
* HTM-State learns adaptively and online  

---

### 💻 Offline Evaluation

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

### 🔎 Interpretation

HTM-State detects the behavioral state shift  
**within ~0.5–1.5 seconds** of onset.

This represents **near–real-time behavioral awareness** without supervision.

---

### 🎥 Live Visualization

```bash
python -m scripts.live_demo_state --backend htm --rate-hz 10
```

The live visualization shows two scrolling plots:

1. **Control signals**  
2. **HTM State + detected spikes (orange dots)**  

### ✅ What good detection looks like

✔ transition spike occurs shortly after the real change  
✔ few false alarms outside transition period  

### Failure modes

❌ spikes long after the boundary → slow reaction  
❌ spikes before the boundary → overly sensitive detector 

### 🔎 Interpretation
* HTM-State **detected behavioral state shifts** with strong speed & precision.

---

### 📌 Takeaway  

Demo 1 validates HTM-State for:

- ✔ online learning (no split training data)  
- ✔ unsupervised change detection  
- ✔ fast response  
- ✔ precision  

