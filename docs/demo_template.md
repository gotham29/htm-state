# DEMO_NAME — DEMO_TAGLINE

> 🔁 **Template Notes**
> Replace everything in **🔴 red** while keeping everything in **🟢 green** unchanged.

🟢 This demo applies HTM-State to **🔴 DEMO_DOMAIN**  
🟢 using the same online anomaly → state → spike pipeline  
🟢 proven across other demos.

## 🎯 Domain / Use-Case

🟢 - Primary context: 🔴 DEMO_DOMAIN  
🟢 - Typical signals: 🔴 e.g., motion, rate, load, flow  
🟢 - Operational goal: 🔴 early drift/overload/change detection  

🟢 Keep this concise but concrete — enough to visualize the setting.

## 🔍 Scenario

🟢 We stream behavior over time — modeled as distinct regimes:

🟢 - 🔴 baseline state  
🟢 - 🔴 intermediate drift / rising complexity  
🟢 - 🔴 high-shift / overload / failure  

🟢 If ground truth exists, the regime boundaries are: 🔴 [STEP_LIST]

🟢 This mirrors Demo 1 + Demo 2:  
🟢 regime shifts with latency measurement.

## 📌 Core Question

> 🟢 Can HTM-State detect **🔴 DEMO_DOMAIN changes**  
> 🟢 *without* retraining, labels, or supervised classifiers?

## 💻 Offline Evaluation

### ▶️ Run

```bash
python -m 🔴 DEMO_OFFLINE_SCRIPT \
    --csv 🔴 DEMO_CSV_PATH \
    --rate-hz 🔴 DEMO_RATE_HZ

## 📈 Live Visualization

### ▶️ Run

python -m 🔴 DEMO_LIVE_SCRIPT \
    --csv 🔴 DEMO_CSV_PATH \
    --rate-hz 🔴 DEMO_RATE_HZ

### Optional tuning flags

    --spike-recent-sec 3 \
    --spike-prior-sec 6 \
    --spike-threshold-pct 40

### What you should see

🟢 - **Top panel** — domain features for 🔴 DEMO_DOMAIN  
🟢 - **Bottom panel** — HTM state + spikes  
🟢 - **Vertical dashed lines** — ground-truth transitions (if available)  
🟢 - **Magenta bars** — detection lag visualization  

### Good visual behavior

✔ Spikes appear shortly after regime boundaries  
✔ Magenta bars stay short  
✔ Quiet behavior outside regime transitions  

## 🎥 GIFs / Short Clips

<p align="center">
  <img src="docs/gifs/🔴 DEMO_GIF_1.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/🔴 DEMO_GIF_2.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/🔴 DEMO_GIF_3.gif" width="950"/>
</p>

### Interpretation

🟢 - **Orange dots** = detected spikes  
🟢 - **Red dashed line** = true drift boundary  
🟢 - **Magenta bar** = time-to-detection lag  

### What “good” looks like

✔ concise spike timing near true events  
✔ short lag bars  
✔ minimal spurious spikes  

### Failure modes

❌ spikes far after boundaries → slow detection  
❌ repeated spikes unrelated to boundaries → false alarms  

## 🧠 Why This Demo Matters

🟢 Same structure as Demo 1 & Demo 2:

- 🟢 shows HTM-State generality  
- 🟢 validates label-free detection  
- 🟢 supports autonomy / safety / monitoring  

🔴 Insert domain-specific value and implications here  
(e.g., staffing, intrusion detection, failure prevention, pilot safety).

