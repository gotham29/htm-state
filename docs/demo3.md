## 🏥 Demo 3 — Healthcare Operator Workload

This demo applies HTM-State to **clinical operator behavior**  
(e.g., ICU nurses, surgeons, interventionalists), detecting  
**workload transitions and emerging overload** using the  
*same unsupervised online pipeline* validated in Demos 1 & 2.

---

### 🩺 Scenario

A synthetic operator stream was generated with drift across:  
• `motion_energy`  
• `cursor_velocity`  
• `interaction_density`  
• `task_variability`

Two embedded regime transitions were inserted:  
**baseline → elevated workload → overload**  
expressed through motion, interaction rhythm, and task complexity.

---

### 📌 Core Question
> Can HTM-State surface **emerging overload / performance change**
> fast enough to matter for safety?

As in Demos 1 & 2, HTM-State must:  
✔ learn **online** from operator behavior  
✔ detect **workload transitions** in real time  
✔ measure **latency** from event → detection  

---

### 💻 Offline Evaluation

```bash
python -m scripts.offline_demo_healthcare \
    --csv demos/healthcare_demo/operator_stream.csv \
    --rate-hz 10
```

Example output:

```text
Loaded 2000 timesteps.

=== Healthcare Workload Detection Results ===
Transition 0: boundary at step 1450 → detected at step 1451
Lag = **1 step (0.100 s @ 10 Hz)**

Average detection lag: **0.1 seconds**
```

✔ Reaction was **nearly instantaneous (0.1 s @ 10 Hz)**
✔ No supervision, calibration, or retraining
✔ Same pipeline as workload + cyber — **no domain-specific tuning required**

---

### 🎥 Live Visualization

```bash
python -m scripts.live_demo_healthcare \
    --csv demos/healthcare_demo/operator_stream.csv \
    --rate-hz 10
```

<p align="center">
  <img src="docs/gifs/demo3_spike2.gif" width="950"/>
</p>

### 🔎 Interpretation
* HTM-State is again both fast & precise, with short lag bars + low false alarms.

---

### 📌 Takeaway

Demo 3 extends HTM-State into high-stakes **human-in-the-loop** settings, showing:

• workload awareness **without labels**
• adaptive detection **without retraining**
• **same pipeline** across pilots → cyber analysts → clinicians

This positions the system for:  
• real-time patient safety monitoring
• acuity-aware staffing / decision support
• AR/VR guidance and operator-feedback systems