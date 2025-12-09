# DEMO_NAME — SHORT TAGLINE

> 🔁 **Template notes**
> - Search & replace these placeholders:
>   - `DEMO_NAME` — human-facing name (e.g., "Demo 4 — Industrial Predictive Change")
>   - `DEMO_DOMAIN` — short domain label (e.g., "Manufacturing line vibration")
>   - `DEMO_TAGLINE` — one-line value prop
>   - `DEMO_OFFLINE_SCRIPT` — offline script module path (e.g., `scripts.offline_demo_industrial`)
>   - `DEMO_LIVE_SCRIPT` — live script module path (e.g., `scripts.live_demo_industrial`)
>   - `DEMO_CSV_PATH` — default CSV path
>   - `DEMO_RATE_HZ` — sample rate (e.g., `10`)
>   - `DEMO_GIF_1`, `DEMO_GIF_2`, `DEMO_GIF_3` — GIF paths under `docs/gifs/`
> - Keep the **section structure** aligned with Demo 1 & 2 in README.

DEMO_TAGLINE

This demo applies HTM-State to **DEMO_DOMAIN** using the same  
online anomaly → state → spike pipeline as the other demos.

---

## 🎯 Domain / Use-Case

- Primary context: DEMO_DOMAIN  
- Typical signals: _e.g., sensor readings, control inputs, network metrics_  
- Operational goal: _e.g., early detection of drift, overload, faults_  

You can keep this short but concrete — enough for someone to visualize the setting.

---

## 🔍 Scenario

Describe the data stream and regime structure, following the same pattern as Demos 1 & 2:

- Sequence structure (e.g., "stable → mild drift → strong drift")  
- What changes, statistically or behaviorally (e.g., variance, rate, load)  
- Whether there are **known boundaries** or events (for detection lag)  

Example phrasing:

> We generate a streaming sequence with N distinct regimes:
> - baseline regime  
> - intermediate drift  
> - high-change regime (e.g., fault or overload)  

If applicable, note how ground truth is encoded (e.g., `is_drift_boundary` column).

---

## 📌 Question

> Can HTM-State detect DEMO_DOMAIN regime changes  
> *without* retraining, labels, or supervised classifiers?

You can tweak the wording, but keep the **yes/no diagnostic question** structure.

---

## 💻 Offline Evaluation

### ▶️ Command

```bash
python -m DEMO_OFFLINE_SCRIPT \
    --csv DEMO_CSV_PATH \
    --rate-hz DEMO_RATE_HZ
```

If you have extra flags (e.g., backend choice), include them here.

### 🧾 Example Output

```text
Found N drift boundaries at steps: [...]

=== Drift Detection Results ===
Drift 0: boundary at step ... → detected at step ..., lag = ... steps (... s)
Drift 1: ...
...

Average detection lag over N drifts: X.X s
```

Adapt this block to show:

- how many boundaries / events  
- per-event lag  
- average detection lag  

### 🔎 Interpretation

Summarize the result in plain language, mirroring Demo 1 & 2:

- ✔ Typical detection lag (in seconds @ DEMO_RATE_HZ Hz)  
- ✔ Whether detection is consistent across events  
- ✔ Whether this is competitive vs. retraining / supervised baselines  

> This represents **model-free drift / transition detection** using the same core  
> HTM-State pipeline as the other demos.

---

## 📈 Live Visualization

### ▶️ Command

```bash
python -m DEMO_LIVE_SCRIPT \
    --csv DEMO_CSV_PATH \
    --rate-hz DEMO_RATE_HZ
```

Add domain-specific knobs if relevant:

```bash
    --spike-recent-sec 3 \
    --spike-prior-sec 6 \
    --spike-threshold-pct 40
```

### What you should see

Describe the two panels in the same language as Demos 1 & 2:

- **Top panel** — selected DEMO_DOMAIN features  
- **Bottom panel** — HTM state (EMA of anomaly) + spikes  
- **Vertical dashed lines** — ground-truth transitions / drifts (if available)  
- **Magenta bars** — detection lag annotations  

Call out what "good" looks like:

- concise spikes near the transition  
- short magenta bars  
- relatively calm outside drift windows  

---

## 🎥 GIFs / Short Clips (optional)

If you create GIFs (recommended), use the same centered layout as Demo 2:

```html
<p align="center">
  <img src="docs/gifs/DEMO_GIF_1.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/DEMO_GIF_2.gif" width="950"/>
</p>

<p align="center">
  <img src="docs/gifs/DEMO_GIF_3.gif" width="950"/>
</p>
```

### 🔎 Interpretation

- ✔ What each color / marker means  
- ✔ What "good detection" looks like in this domain  
- ⚠️ What failure modes would look like (slow spikes, repeated false alarms)  

---

## 🧠 Why this demo matters

Use the same pattern as other demos:

- how this extends HTM-State into a new domain  
- why online, label-free detection is valuable here  
- how this connects to real deployments (e.g., safety, monitoring, autonomy)  

Keep it punchy and outcome-focused — this is the "so what?" section.