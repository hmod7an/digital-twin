---
title: Face Health Digital Twin
emoji: 🩺
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Real-time rPPG · Fatigue · Stress · Breathing from webcam
---

# Face Health Digital Twin

**Real-time vital sign estimation from a webcam.**

| Signal | Method | Reference |
|--------|--------|-----------|
| Heart Rate | rPPG / POS algorithm | Wang et al. 2017 |
| Fatigue | EAR + PERCLOS | Soukupová & Čech 2016 |
| Stress | Head pose + HR + blink | Dinges & Grace 1998 |
| Breathing | Vertical nose tip motion | — |

> ⚠️ For educational and research purposes only. Not a medical device.
