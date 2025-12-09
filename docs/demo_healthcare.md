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
🟢 regime boundaries may be: **[example: 600, 1200, 1650]**

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
Found 2 workload transition boundaries at steps: [600, 1450]

=== Detection Lag Results ===
Transition 0: boundary at 600 → detected at 627, lag = 27 steps (2.7 s)
Transition 1: boundary at 1450 → detected at 1489, lag = 39 steps (3.9 s)

Average detection lag = 3.3 seconds
```

🟢 Detection lags are typically **2–4 seconds @ 10 Hz**,  
🟢 consistent with Demo 1 & Demo 2.

> 🟢 This represents **model-free operator workload change detection** using HTM-State.

---

## 📈 Live Visualization

### ▶️ Run

```bash
python -m scripts.live_demo_healthcare \
    --csv demos/healthcare_demo/operator_stream.csv \
    --rate-hz 10
```

