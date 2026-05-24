# Face Health Digital Twin

Real-time webcam-based physiological monitoring: heart rate (rPPG), fatigue, stress, breathing, emotion detection, attention, and wellbeing — presented as an interactive AI dashboard.

> ⚠️ **Educational and research use only. Not a medical device.**

---

## Live Demo

| Service  | URL |
|----------|-----|
| Frontend | _your Vercel URL_ |
| Backend  | _your Render URL_ |

---

## Architecture

```
Browser (Vercel)          Backend (Render)
──────────────────        ─────────────────────────────────────
index.html / app.js  ←──WebSocket──→  FastAPI main.py
  getUserMedia()                         MediaPipe Face Landmarker
  send JPEG frame                        ONNX emotion models
  render results                         rPPG / fatigue / stress / breathing
```

- **GitHub** — source of truth for all code  
- **Render** — Python FastAPI backend, processes webcam frames via WebSocket  
- **Vercel** — static frontend (HTML/JS/CSS), no build step required

---

## Repository Structure

```
digital_twins/
├── main.py                  # FastAPI WebSocket backend (→ Render)
├── render.yaml              # Render deployment config
├── requirements.txt         # Python dependencies
│
├── ai/                      # AI models & emotion engine
│   ├── deep_emotion_model.py
│   ├── emotion_engine.py
│   ├── feature_extractor.py
│   └── state_fusion.py
│
├── core/                    # MediaPipe face tracking
│   ├── face_tracker.py
│   ├── model_manager.py
│   └── signal_buffer.py
│
├── vitals/                  # Signal processing
│   ├── rppg.py              # Heart rate (POS algorithm)
│   ├── fatigue.py           # EAR / PERCLOS
│   ├── stress.py            # HRV proxy + brow tension
│   ├── breathing.py         # Nose oscillation
│   └── attention.py         # Gaze + head stability
│
├── prediction/
│   └── health_risk.py       # Multi-signal risk classifier
│
├── config/
│   └── settings.py
│
├── utils/
│   └── filters.py
│
├── frontend/                # Static web app (→ Vercel)
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── config.js            # ← SET YOUR RENDER URL HERE
│   └── vercel.json
│
└── core/models/
    └── face_landmarker.task # MediaPipe model (~3.8 MB)
```

---

## Deployment Guide

### Prerequisites
- GitHub account
- [Render](https://render.com) account (free tier works; Starter plan recommended)
- [Vercel](https://vercel.com) account (free tier)

---

### Step 1 — Push to GitHub

```bash
cd digital_twins

git init
git add .
git commit -m "initial: Face Health Digital Twin"

# Create repo on GitHub then:
git remote add origin https://github.com/YOUR_USERNAME/digital-twin.git
git branch -M main
git push -u origin main
```

---

### Step 2 — Deploy Backend on Render

1. Go to [render.com/new/web-service](https://dashboard.render.com/new/web-service)
2. Connect your GitHub repo
3. Configure:
   - **Name**: `face-health-digital-twin`
   - **Branch**: `main`
   - **Root Directory**: _(leave blank — use repo root)_
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Starter ($7/mo) — Free tier works but has cold starts
4. Add environment variables:
   - `PYTHONUNBUFFERED` = `1`
5. Click **Create Web Service**
6. Wait ~5 minutes for the first build
7. Copy your Render URL: `https://face-health-digital-twin.onrender.com`

---

### Step 3 — Configure Frontend

Edit `frontend/config.js`:

```javascript
window.BACKEND_WS = "wss://face-health-digital-twin.onrender.com/ws";
```

Commit and push:
```bash
git add frontend/config.js
git commit -m "config: set Render backend URL"
git push
```

---

### Step 4 — Deploy Frontend on Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repo
3. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `frontend`
   - **Build Command**: _(leave blank)_
   - **Output Directory**: `.` (current directory)
4. Click **Deploy**
5. Your app is live at `https://your-project.vercel.app`

---

### Step 5 — Test

Open your Vercel URL and verify:

| Test | Expected |
|------|----------|
| Page loads | Dashboard renders with dark theme |
| Start Camera | Browser asks for camera permission |
| Camera active | Live feed shown, annotated feed appears |
| Face detected | `● FACE` badge turns green |
| After ~5s | Heart rate, fatigue, stress values appear |
| Emotion | Emoji and state update in real-time |
| Recording | Start/Stop/Download CSV works |
| Mobile | Open on phone (requires HTTPS — Vercel provides this) |

---

### Redeploying Updates

**Backend change** → push to GitHub → Render auto-deploys  
**Frontend change** → push to GitHub → Vercel auto-deploys  
**Both** → single `git push origin main`

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Edit frontend/config.js:
#   window.BACKEND_WS = "ws://localhost:8000/ws";

# Open frontend/index.html in browser
# (or serve with: python -m http.server 3000 -d frontend)
```

---

## Environment Variables

| Variable | Service | Value |
|----------|---------|-------|
| `PYTHONUNBUFFERED` | Render | `1` |
| `PORT` | Render | Set automatically |
| `BACKEND_WS` | Frontend (config.js) | Your Render WSS URL |

---

## Performance

| Metric | Value |
|--------|-------|
| Frame rate to backend | 10 fps |
| Backend processing | ~50–100 ms/frame (CPU) |
| Annotated frame size | ~15 KB (480×360 JPEG 75%) |
| Memory (Render) | ~400–500 MB |
| Heart rate warm-up | ~10 s |
| Emotion latency | ~1.5 s |

---

## Limitations

1. **Lighting** — rPPG needs stable, good light. Flickering or dim light degrades heart rate accuracy.
2. **Motion** — Large head movements corrupt the rPPG signal.
3. **Cold starts** — Free Render tier sleeps after 15 min inactivity (~30 s cold start). Upgrade to Starter to avoid this.
4. **Network latency** — Frame round-trip adds ~50–200 ms depending on distance from Render region.
5. **Medical validity** — Not validated against clinical devices. BPM error is typically ±5–15 BPM.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + uvicorn + WebSockets |
| Face tracking | MediaPipe Face Landmarker (Tasks API) |
| Emotion AI | EfficientNet-B0 ONNX (3-model ensemble) |
| Heart rate | rPPG POS algorithm (Wang et al. 2017) |
| Fatigue | EAR + PERCLOS (Soukupová & Čech 2016) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Hosting | Render (backend) + Vercel (frontend) |

---

*University research project — Computer Science / Biomedical Engineering*
