# Face Health Digital Twin
### Real-Time Vital Sign Estimation and Stress-Fatigue Prediction

> ⚠️ **IMPORTANT DISCLAIMER**  
> This system is for **educational and research purposes only** and is **NOT a medical diagnostic device**.  
> It must not be used for clinical diagnosis, medical monitoring, or any safety-critical application.  
> Always consult a qualified healthcare professional for medical advice.

---

## Overview

A real-time webcam-based physiological monitoring system that estimates heart rate (rPPG), fatigue level, stress level, and breathing rate from facial video — presented as an interactive **Digital Twin** dashboard.

The system runs entirely locally on a standard laptop with no cloud dependencies, no wearable sensors, and no proprietary hardware.

---

## Key Features

| Feature | Method | Accuracy Target |
|---|---|---|
| Heart Rate | rPPG / POS algorithm | ±5 BPM (good lighting) |
| Fatigue | EAR + PERCLOS + MAR | 3-level classification |
| Stress | HRV + blink + head motion | 3-level classification |
| Breathing | Nose-tip oscillation + RSA | ±2 br/min |
| Risk Level | Multi-signal trend analysis | Normal / Warning / High Risk |

---

## Architecture

```
digital_twins/
├── app/
│   ├── dashboard.py          # Streamlit UI (entry point)
│   └── components/
│       ├── charts.py         # Plotly chart builders
│       └── metrics_panel.py  # Metric card widgets
│
├── core/
│   ├── camera.py             # Thread-safe webcam manager
│   ├── face_tracker.py       # MediaPipe Face Mesh wrapper
│   ├── signal_buffer.py      # Circular time-series buffer
│   └── pipeline.py           # Central processing loop (background thread)
│
├── vitals/
│   ├── rppg.py               # POS rPPG heart rate estimator
│   ├── fatigue.py            # EAR / MAR / PERCLOS fatigue detector
│   ├── stress.py             # Multi-modal stress estimator
│   └── breathing.py          # Breathing rate estimator
│
├── prediction/
│   └── health_risk.py        # Rule-based health risk predictor
│
├── datasets/
│   └── loader.py             # Offline dataset loader (UBFC, PURE, WESAD)
│
├── config/
│   └── settings.py           # Central configuration (dataclasses)
│
├── utils/
│   ├── filters.py            # Bandpass, detrend, Welch PSD
│   └── visualization.py      # OpenCV HUD overlay helpers
│
├── tests/
│   └── test_pipeline.py      # Unit tests (pytest)
│
├── reports/
│   └── methodology_references.md  # Research paper citations
│
├── run.py                    # Project launcher
└── requirements.txt
```

### Design Decisions

**Layered + Modular Monolith**: Each layer has a single responsibility. The `core` layer handles hardware abstraction. The `vitals` layer contains all signal-processing science. The `prediction` layer synthesises vitals into risk levels. The `app` layer owns all UI concerns.

**Background pipeline thread**: The processing loop runs in a `daemon` thread. The Streamlit dashboard reads a thread-safe snapshot every refresh cycle. This prevents camera I/O and signal processing from blocking the UI render.

**No model training required**: All vital-sign estimators use signal-processing methods with physiologically-derived thresholds. The system is immediately usable out of the box.

---

## Methodology

### rPPG (Heart Rate)
Implements the **POS (Plane-Orthogonal-to-Skin)** algorithm (Wang et al., 2017). Extracts mean RGB from forehead and cheek ROIs defined by MediaPipe landmarks, applies temporal normalisation, projects onto the skin-orthogonal plane, bandpass filters at 0.7–3.0 Hz, and uses Welch's PSD to detect the dominant frequency.

### Fatigue
Computes **EAR** (Eye Aspect Ratio) for blink detection, **PERCLOS** (proportion of time eyes are closed) for drowsiness, **MAR** (Mouth Aspect Ratio) for yawn detection. A weighted composite score (PERCLOS 40% + blink deviation 25% + EAR 20% + yawn 15%) produces a 0–100 fatigue index.

### Stress
Rule-based fusion of: HRV-like metric (BPM standard deviation), head-movement jitter (frame-to-frame pose RMS), blink rate deviation, and brow-lowering proxy (AU4). Validated against WESAD-style stress indicators.

### Breathing
Bandpass filters vertical nose-tip oscillation at 0.13–0.42 Hz (8–25 br/min) to extract respiratory rhythm. Supplemented by RSA modulation of the rPPG amplitude envelope.

---

## Installation

### Prerequisites
- Python 3.10 or higher  
- A connected webcam  
- Windows / macOS / Linux

### Steps

```bash
# Clone or download the project
cd digital_twins

# Create a virtual environment (recommended)
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the system
python run.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## How to Run

```bash
# Option 1: Project launcher (recommended)
python run.py

# Option 2: Direct Streamlit launch
streamlit run app/dashboard.py

# Option 3: Run unit tests
python -m pytest tests/ -v
```

---

## Dashboard Guide

| Panel | Description |
|---|---|
| Live Feed | Webcam with facial landmarks overlay |
| Heart Rate | Current BPM + signal quality indicator |
| Fatigue | Score (0–100%) + state (Normal/Tired/Drowsy) |
| Stress | Score (0–100%) + state (Calm/Moderate/High) |
| Breathing | Estimated breaths per minute |
| Risk Banner | Green/Yellow/Red system status with specific alerts |
| rPPG Signal | Live pulse waveform |
| Score Bars | Horizontal bar chart for all scores |
| Vital History | 5-minute trend chart for BPM + fatigue + stress |
| Diagnostics | Raw EAR, MAR, blink rate, HRV, head movement |

---

## Datasets

The system operates fully in **live webcam mode** by default.

For offline validation, download these datasets and place them under `datasets/raw/`:

| Dataset | URL | License |
|---|---|---|
| UBFC-rPPG | https://sites.google.com/view/ybenezeth/ubfcrppg | Academic |
| PURE | https://www.tu-ilmenau.de (CV Group) | Academic |
| NTHU DDD | http://cv.cs.nthu.edu.tw/DDD | Academic |
| WESAD | UCI ML Repository | CC BY 4.0 |

---

## Performance

- **Camera FPS**: up to 30 FPS (hardware-limited)  
- **Processing latency**: < 50 ms per frame on a standard CPU  
- **BPM update rate**: every 0.5 s after 3 s warm-up  
- **Dashboard refresh**: configurable 100–1000 ms  
- **Memory footprint**: ~150–300 MB (MediaPipe model loaded)

---

## Limitations

1. **Lighting**: rPPG accuracy degrades significantly under poor or flickering light. Use in a well-lit, stable environment.
2. **Motion**: Large head movements corrupt the rPPG signal. The POS method provides partial motion robustness but is not immune.
3. **Skin tone**: Camera-based rPPG has documented accuracy disparities across skin tones. This is an active research area.
4. **Calibration**: No personalised baseline calibration — thresholds are population-level defaults.
5. **Breathing**: Nose oscillation method is sensitive to camera stability; low-confidence readings are common.
6. **Medical validity**: This system is NOT validated against medical-grade devices. BPM estimates carry ±5–15 BPM typical error.

---

## Future Work

- [ ] Personalised baseline calibration (per-user EAR/BPM norms)
- [ ] Lightweight CNN for skin-tone-robust rPPG (trained on UBFC + PURE)
- [ ] Multi-face tracking for group settings
- [ ] Session recording and export (CSV, PDF report)
- [ ] Mobile app (Flutter + TFLite)
- [ ] Integration with WESAD for ML-based stress classifier
- [ ] Electrodermal activity (EDA) sensor fusion via BLE
- [ ] Adaptive alert thresholds based on prior session history

---

## References

See `reports/methodology_references.md` for full citations with DOIs.

Key papers:
- Wang et al. (2017) — POS rPPG algorithm. IEEE TBME. DOI: 10.1109/TBME.2016.2609282
- Soukupová & Čech (2016) — EAR blink detection. CVWW.
- Dinges & Grace (1998) — PERCLOS drowsiness metric. FHWA.
- Giannakakis et al. (2019) — Stress detection biosignals review. IEEE RBME.
- Kartynnik et al. (2019) — MediaPipe Face Mesh. arXiv:1907.06724

---

## Privacy & Ethics

- **No data is stored**: All processing is in-memory and discarded when the application closes.
- **No network transmission**: The system is entirely local; no video or biometric data leaves the machine.
- **Consent**: In any deployment, users must be informed that their facial video is being processed.
- **Bias awareness**: rPPG accuracy varies with skin tone, lighting, and age. Results should be interpreted with awareness of these limitations.
- **Not for surveillance**: This system must not be used for covert monitoring of individuals.

---

*Developed as a university research project — Computer Science / Biomedical Engineering*
