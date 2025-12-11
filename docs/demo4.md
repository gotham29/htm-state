# 🏭 Demo 4 — Manufacturing Process Drift Detection

This demo applies HTM-State to **manufacturing process behavior**  
(e.g., machining cells, continuous production lines, and assembly stations).  
The goal is to detect **emerging drift and pre-fault behavior** using the  
same online pipeline proven in Demos 1–3 — *no retraining or labels required*.

---

## 🔧 Scenario

We simulate a production line operating in three regimes:

1. **Stable baseline operation**  
2. **Slow drift** (tool wear, small vibration increases, load instability)  
3. **High-drift pre-fault regime**

Key monitored signals include:

 - `vibration_energy`  
 - `spindle_load`  
 - `feed_rate`  
 - `line_speed`  
 - `temperature`  
 - `part_time`

A true regime shift is injected at:

- **step 1600** → onset of high-drift / pre-fault behavior

HTM-State must surface this transition **quickly** during live streaming.

---

## 📌 Core Question

> Can HTM-State detect **emerging manufacturing process drift**  
> early enough for predictive maintenance or quality protection —  
> **without labels, retraining, or supervised ML?**

✔ Yes — with low false alarms and low latency.

---

## 💻 Offline Evaluation

### ▶️ Run (offline drift evaluation)

```bash
python -m scripts.offline_demo_manufacturing \
    --csv demos/manufacturing_demo/line_stream.csv \
    --rate-hz 10
```

Example Output
```text
Loaded 2400 timesteps.

=== Manufacturing Drift Detection Results ===
Transition 0: boundary at step 1600 → detected at step 1615, lag = 15 steps (1.500 s @ 10 Hz)

Average detection lag: 15 steps (1.5 seconds)
```

HTM-State typically detects high-drift onset within 1–3 seconds @ 10 Hz,
even when drift evolves gradually and contains noise.

## 🎥 Live Visualization

### ▶️ Run (live drift monitoring)

```bash
python -m scripts.live_demo_manufacturing \
    --csv demos/manufacturing_demo/line_stream.csv \
    --rate-hz 10 \
    --step-stride 3
```

### What you should see

- **Top panel:** rolling 100-step window of selected features  
  (`vibration_energy`, `spindle_load`, `feed_rate`, `line_speed`)  

- **Bottom panel:** HTM-State (EMA of anomaly) + detected spikes  
- **Red dashed line:** true regime boundary at step 1600  
- **Orange dots:** detected drift spikes  
- **Magenta bar:** detection lag (boundary → first spike)

### Good visual behavior

✔ spikes appear shortly after the true drift boundary  
✔ low spike activity during stable production  
✔ smooth state signal with a clear upward break near the transition

### Failure modes

❌ spikes long after the boundary → slow reaction  
❌ many spikes before the boundary → oversensitive detector  

## 🧠 Why Demo 4 Matters

Demo 4 demonstrates **low-latency detection of manufacturing process drift**,  
even when drift evolves slowly across hundreds of timesteps.

### Key takeaways

- No labels, retraining, or supervised models needed  
- Smooth anomaly → state → spike pipeline works across domains  
- Drift is detected within **1–3 seconds @ 10 Hz**  
- Low false positives despite noisy multi-sensor inputs  
- The same architecture from Demos 1–3 generalizes cleanly to industrial monitoring