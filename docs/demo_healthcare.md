# Demo 3 — Real-Time Clinical Operator Workload Detection

> 🔁 **Template Notes**
> All 🔴 red placeholders were replaced with healthcare-specific content.

🟢 This demo applies HTM-State to **clinical operator workload and performance shifts**  
🟢 using the same online anomaly → state → spike pipeline  
🟢 proven across other demos.

---

## 🎯 Domain / Use-Case

🟢 - Primary context: **surgeons, ICU nurses, proceduralists, operators**  
🟢 - Typical signals: **motion metrics, tool movement rate, interaction tempo, cursor path complexity, physiological control traces**  
🟢 - Operational goal: **early detection of rising workload, overload, or performance degradation**  

🟢 This supports safety awareness, staffing decisions, and adaptive assistive systems.

---

## 🔍 Scenario

🟢 We stream operator behavior over time — modeled as distinct regimes:

🟢 - **baseline routine operation**  
🟢 - **increasing workload / rising task complexity**  
🟢 - **high-acuity event / overload / safety-critical transition**  

🟢 If ground truth exists (e.g., annotated procedure timestamps),  
🟢 regime boundaries may be: **[600, 1450]** in this synthetic version.

🟢 This mirrors Demo 1 + Demo 2:  
🟢 regime shifts with latency measurement.

---

## 📌 Core Question

> 🟢 Can HTM-State detect **emerging overload and performance change**  
> 🟢 *without* retraining, labels, or supervised classifiers?

---

## 💻 Offline Evaluation

### ▶️ Run

```bash
python -m scripts.offline_demo_healthcare \
    --csv demos/healthcare_demo/operator_stream.csv \
    --rate-hz 10
```
---

### 🧾 Example Output

```text
Loaded 2000 timesteps.

=== Healthcare Workload Detection Results ===
Transition 0: boundary at step 1450 → detected at step 1451, lag = 1 steps (0.100 s)

Average detection lag over 1 transitions: 0.100 s
```

🟢 In this configuration, HTM-State reacted almost instantaneously (0.1 s @ 10 Hz)  
🟢 for the overload event at step 1450.

> 🟢 This represents **model-free operator workload change detection** using HTM-State.

---

## 📈 Live Visualization

### ▶️ Run

```bash
python -m scripts.live_demo_healthcare \
    --csv demos/healthcare_demo/operator_stream.csv \
    --rate-hz 10
```

---

### Optional tuning flags

    --spike-recent-sec 3 \
    --spike-prior-sec 6 \
    --spike-threshold-pct 40

### What you should see

🟢 **Top panel** — motion / interaction features (e.g., tool speed, cursor motion, interaction rate)  
🟢 **Bottom panel** — HTM state (EMA of anomaly) + spikes  
🟢 **Vertical dashed lines** — annotated workload / event transitions  
🟢 **Magenta bars** — detection lag visualization from event → spike  

### Good visual behavior

✔ spikes appear shortly after major workload transitions  
✔ magenta bars remain short (seconds, not tens of seconds)  
✔ relatively quiet behavior outside high-workload or event windows  

---

## 🎥 GIFs / Short Clips

If you capture short sequences (recommended), you can embed them like this:

<p align="center">
  <img src="docs/gifs/demo3_healthcare_1.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/demo3_healthcare_2.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/demo3_healthcare_3.gif" width="950"/>
</p>

### Interpretation

🟢 **Orange dots** — detected workload / performance spikes  
🟢 **Red dashed line** — annotated high-acuity / transition moment  
🟢 **Magenta bar** — time from event onset → HTM-State detection  

### What “good” looks like

✔ concise spike timing near the event markers  
✔ short lag bars (a few seconds at most)  
✔ minimal spurious spikes during routine operation  

### Failure modes

❌ spikes far after event markers → slow detection  
❌ repeated spikes with no annotated event → false alarms / oversensitivity  

---

## 🧠 Why This Demo Matters

🟢 Same structural goal as Demo 1 & Demo 2:

- 🟢 shows HTM-State generality in **high-stakes human–in-the-loop** settings  
- 🟢 validates **label-free, online** detection of overload and performance change  
- 🟢 supports autonomy, safety, and real-time decision support  

In the healthcare context, this points toward:

- early visibility into **operator overload, fatigue, or deteriorating performance**  
- continuous monitoring without dense manual labels or retraining  
- a single pipeline that can span **pilots → cyber analysts → clinicians**  
- future integration into **patient safety, staffing, and AR/VR assistive systems**  

---

# Healthcare Operator Workload Stream Dataset

Synthetic clinical-style operator behavior for HTM-State Demo 3.

Features:
- motion_energy
- cursor_velocity
- interaction_density
- task_variability

Boundaries:
[600, 1450] mark two workload regime shifts.
