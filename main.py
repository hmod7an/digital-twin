"""
Face Health Digital Twin — FastAPI WebSocket backend.

Deployed on Render. Browser sends base64 JPEG frames over WebSocket;
server runs the full AI pipeline and returns JSON results + annotated frame.
"""
import asyncio
import base64
import csv
import io
import logging
import os
import sys
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Force software GL before any mediapipe/OpenCV import
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── ONNX model bootstrap ────────────────────────────────────────────────────────
_MODEL_DIR = os.path.expanduser("~/.hsemotion")
_ONNX_BASE = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition"
    "/raw/main/models/affectnet_emotions/onnx"
)
_ONNX_MODELS = [
    "enet_b0_8_best_vgaf.onnx",
    "enet_b0_8_best_afew.onnx",
    "enet_b2_7.onnx",
]

def _download_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for fname in _ONNX_MODELS:
        dst = os.path.join(_MODEL_DIR, fname)
        if not os.path.exists(dst):
            log.info(f"Downloading {fname} …")
            try:
                urllib.request.urlretrieve(f"{_ONNX_BASE}/{fname}", dst)
                log.info(f"{fname} ready ({os.path.getsize(dst) // 1024} KB)")
            except Exception as exc:
                log.warning(f"Could not download {fname}: {exc}")


# ── Lifespan: download all models and warm up MediaPipe before first request ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_models()
    # Download face landmarker model now so startup logs capture any failure
    try:
        from core.model_manager import ensure_face_landmarker
        model_path = ensure_face_landmarker()
        log.info(f"Face landmarker model ready: {model_path}")
    except Exception as exc:
        log.error(f"FATAL: Face landmarker model unavailable: {exc}")
    yield


app = FastAPI(title="Face Health Digital Twin API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool so synchronous CV/AI code doesn't block the event loop
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")


# ── Lazy imports (deferred until after model download) ───────────────────────
def _import_pipeline():
    from core.face_tracker import FaceTracker
    from ai.feature_extractor import FeatureExtractor
    from ai.deep_emotion_model import DeepEmotionModel
    from vitals.rppg import RPPGEstimator
    from vitals.fatigue import FatigueDetector
    from vitals.stress import StressEstimator
    from vitals.breathing import BreathingEstimator
    from vitals.attention import AttentionEstimator
    from ai.emotion_engine import EmotionEngine, EMOTION_COLOR
    from ai.state_fusion import StateFusion
    from prediction.health_risk import HealthRiskPredictor

    return {
        "FaceTracker": FaceTracker,
        "FeatureExtractor": FeatureExtractor,
        "DeepEmotionModel": DeepEmotionModel,
        "RPPGEstimator": RPPGEstimator,
        "FatigueDetector": FatigueDetector,
        "StressEstimator": StressEstimator,
        "BreathingEstimator": BreathingEstimator,
        "AttentionEstimator": AttentionEstimator,
        "EmotionEngine": EmotionEngine,
        "EMOTION_COLOR": EMOTION_COLOR,
        "StateFusion": StateFusion,
        "HealthRiskPredictor": HealthRiskPredictor,
    }


_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _import_pipeline()
    return _pipeline


# ── Session factory ────────────────────────────────────────────────────────────
def _new_session() -> dict:
    P = _get_pipeline()
    return {
        "tracker":    P["FaceTracker"](),
        "features":   P["FeatureExtractor"](),
        "deep_emo":   P["DeepEmotionModel"](),
        "rppg":       P["RPPGEstimator"](),
        "fatigue":    P["FatigueDetector"](),
        "stress":     P["StressEstimator"](),
        "breathing":  P["BreathingEstimator"](),
        "emotion":    P["EmotionEngine"](),
        "attention":  P["AttentionEstimator"](),
        "risk":       P["HealthRiskPredictor"](),
        "fusion":     P["StateFusion"](),
        "bpm_hist":   deque(maxlen=120),
        "fat_hist":   deque(maxlen=120),
        "str_hist":   deque(maxlen=120),
        "wb_hist":    deque(maxlen=120),
        "score_hist": deque(maxlen=120),
        "emo_hist":   deque(maxlen=150),
        "tick":       0,
        "recording":  False,
        "records":    [],
        "last_snap":  0,
    }


# ── Frame processor (runs in thread pool) ─────────────────────────────────────
def _process_frame(bgr: np.ndarray, session: dict) -> dict:
    P = _get_pipeline()
    EMOTION_COLOR = P["EMOTION_COLOR"]

    landmarks  = session["tracker"].process(bgr)
    face_feats = session["features"].extract(landmarks)

    rppg_r    = session["rppg"].update(landmarks)
    fatigue_r = session["fatigue"].update(landmarks)
    stress_r  = session["stress"].update(landmarks, rppg_r.bpm, fatigue_r.blink_rate)
    breath_r  = session["breathing"].update(
        landmarks, rppg_r.signal[-1] if len(rppg_r.signal) > 0 else None
    )

    deep_scores = None
    deep_emo    = session["deep_emo"]
    if landmarks is not None and deep_emo.available:
        x, y, wb, hb = landmarks.face_bbox
        if wb > 20 and hb > 20:
            fh, fw = bgr.shape[:2]
            crop = bgr[max(0, y):min(fh, y + hb), max(0, x):min(fw, x + wb)]
            deep_scores = deep_emo.predict(crop)

    emotion_r   = session["emotion"].update(
        face_feats, fatigue_r.score, stress_r.score,
        rppg_r.bpm, fatigue_r.blink_rate, fatigue_r.perclos, deep_scores,
    )
    attention_r = session["attention"].update(
        face_feats, fatigue_r.score, stress_r.score, fatigue_r.blink_rate,
    )
    risk_r   = session["risk"].update(rppg_r.bpm, fatigue_r.score, stress_r.score)
    mental_r = session["fusion"].update(
        emotion_r, attention_r, rppg_r, fatigue_r, stress_r, breath_r, risk_r,
    )

    # Annotated frame — resize to 480x360 to keep bandwidth reasonable
    ann_bgr = session["tracker"].draw_landmarks(bgr, landmarks) if landmarks else bgr
    h, w = ann_bgr.shape[:2]
    scale = min(480 / w, 360 / h)
    if scale < 1.0:
        ann_bgr = cv2.resize(ann_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    ann_b64 = base64.b64encode(buf.tobytes()).decode()

    # Histories
    session["tick"] += 1
    tick = session["tick"]
    session["bpm_hist"].append(rppg_r.bpm or 0.0)
    session["fat_hist"].append(fatigue_r.score)
    session["str_hist"].append(stress_r.score)
    session["wb_hist"].append(mental_r.wellbeing_score)
    session["emo_hist"].append(emotion_r.state)
    if emotion_r.scores:
        session["score_hist"].append(dict(emotion_r.scores))

    # Recording snapshot (every ~2 s at 10 fps)
    if session["recording"] and tick - session["last_snap"] >= 20:
        session["last_snap"] = tick
        session["records"].append({
            "time":        datetime.now().strftime("%H:%M:%S"),
            "emotion":     emotion_r.state,
            "confidence":  round(emotion_r.confidence * 100, 1),
            "bpm":         round(rppg_r.bpm, 1) if rppg_r.bpm else 0.0,
            "fatigue_%":   round(fatigue_r.score, 1),
            "stress_%":    round(stress_r.score, 1),
            "attention_%": round(attention_r.score, 1),
            "wellbeing_%": round(mental_r.wellbeing_score, 1),
        })

    ff = face_feats

    return {
        "type":         "frame",
        "annotated":    ann_b64,
        "face_ok":      landmarks is not None,
        "calibrating":  emotion_r.is_calibrating,
        "deep_ok":      deep_emo.available,
        "gpu":          getattr(deep_emo, "gpu_active", False),
        "tick":         tick,
        # Vitals
        "bpm":          round(rppg_r.bpm, 1) if rppg_r.bpm else None,
        "bpm_conf":     round(rppg_r.confidence, 2),
        "bpm_fill":     round(rppg_r.buffer_fill, 2),
        "rppg_signal":  rppg_r.signal[-120:].tolist() if len(rppg_r.signal) > 0 else [],
        "fatigue": {
            "score":    round(fatigue_r.score, 1),
            "state":    fatigue_r.state,
            "ear":      round(fatigue_r.ear, 3),
            "perclos":  round(fatigue_r.perclos, 1),
            "blinks":   round(fatigue_r.blink_rate, 1),
        },
        "stress": {
            "score": round(stress_r.score, 1),
            "state": stress_r.state,
            "hrv":   round(stress_r.hr_variability, 1),
            "move":  round(stress_r.head_movement, 2),
        },
        "breathing": {
            "rate": round(breath_r.rate_bpm, 1) if breath_r.rate_bpm else None,
            "conf": round(breath_r.confidence, 2),
        },
        "emotion": {
            "state":       emotion_r.state,
            "emoji":       emotion_r.emoji,
            "confidence":  round(emotion_r.confidence, 3),
            "color":       EMOTION_COLOR.get(emotion_r.state, "#9CA3AF"),
            "stable":      emotion_r.is_stable,
            "calibrating": emotion_r.is_calibrating,
            "scores":      {k: round(v, 3) for k, v in (emotion_r.scores or {}).items()},
            "dominant":    emotion_r.timeline_dominant,
            "stability":   round(emotion_r.stability_score, 3),
            "persistence": round(emotion_r.persistence_seconds, 1),
        },
        "attention": {
            "score": round(attention_r.score, 1),
            "level": attention_r.level,
            "cog":   round(attention_r.cognitive_load, 1),
            "gaze":  round(attention_r.gaze_score, 1),
            "stab":  round(attention_r.stability_score, 1),
        },
        "risk": {
            "code":     risk_r.level_code,
            "messages": risk_r.messages[:3],
        },
        "wellbeing": {
            "score":   round(mental_r.wellbeing_score, 1),
            "label":   mental_r.wellbeing_label,
            "trend":   mental_r.trend,
            "icon":    mental_r.trend_icon,
            "insights": mental_r.insights[:3],
            "recs":    mental_r.recommendations[:2],
        },
        "face": {
            "valid":    ff.valid,
            "smile":    round(ff.smile_score, 3) if ff.valid else 0,
            "frown":    round(ff.frown_score, 3) if ff.valid else 0,
            "furrow":   round(ff.brow_furrow_score, 3) if ff.valid else 0,
            "ibrow":    round(ff.inner_brow_raise_score, 3) if ff.valid else 0,
            "ear":      round(ff.ear, 3) if ff.valid else 0,
            "mouth":    round(ff.mouth_open_ratio, 3) if ff.valid else 0,
            "yaw":      round(ff.head_yaw, 1) if ff.valid else 0,
            "pitch":    round(ff.head_pitch, 1) if ff.valid else 0,
            "energy":   round(ff.facial_energy, 3) if ff.valid else 0,
        },
        # History slices for chart updates
        "bpm_hist": list(session["bpm_hist"])[-60:],
        "fat_hist": list(session["fat_hist"])[-60:],
        "str_hist": list(session["str_hist"])[-60:],
        "wb_hist":  list(session["wb_hist"])[-60:],
        "score_hist_last": session["score_hist"][-1] if session["score_hist"] else {},
        # Recording state
        "recording":  session["recording"],
        "rec_count":  len(session["records"]),
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    session = None  # lazy init on first frame

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")

            # ── Commands ──────────────────────────────────────────────
            if cmd == "start_rec":
                if session:
                    session["recording"] = True
                    session["last_snap"] = session["tick"]
                await ws.send_json({"type": "rec_state", "recording": True})
                continue
            if cmd == "stop_rec":
                if session:
                    session["recording"] = False
                await ws.send_json({"type": "rec_state", "recording": False})
                continue
            if cmd == "clear_rec":
                if session:
                    session["records"] = []
                    session["recording"] = False
                await ws.send_json({"type": "rec_cleared"})
                continue
            if cmd == "get_csv":
                records = session["records"] if session else []
                if records:
                    buf = io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
                    w.writeheader()
                    w.writerows(records)
                    await ws.send_json({"type": "csv", "data": buf.getvalue()})
                continue

            # ── Frame processing ──────────────────────────────────────
            frame_b64 = msg.get("frame")
            if not frame_b64:
                continue

            try:
                frame_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                # Init session on first valid frame
                if session is None:
                    session = await asyncio.get_event_loop().run_in_executor(
                        _EXECUTOR, _new_session
                    )

                result = await asyncio.get_event_loop().run_in_executor(
                    _EXECUTOR, _process_frame, bgr, session
                )
                await ws.send_json(result)

            except Exception as exc:
                log.error(f"Frame error: {exc}")
                await ws.send_json({"type": "error", "message": str(exc)[:300]})

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as exc:
        log.error(f"WS error: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "face-health-digital-twin"}


@app.get("/status")
async def status():
    """Diagnostic endpoint — shows what's loaded and what failed."""
    import importlib, sys
    checks = {}
    for mod in ["mediapipe", "cv2", "numpy", "scipy", "onnxruntime", "fastapi"]:
        try:
            m = importlib.import_module(mod)
            checks[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            checks[mod] = f"ERROR: {e}"
    return {"python": sys.version, "packages": checks}


@app.get("/debug")
async def debug():
    """Deep diagnostic — checks model files and pipeline init."""
    import os
    from pathlib import Path

    result = {}

    # Face landmarker model
    mp_model = Path(_ROOT) / "core" / "models" / "face_landmarker.task"
    result["face_landmarker"] = {
        "exists": mp_model.exists(),
        "size_kb": round(mp_model.stat().st_size / 1024) if mp_model.exists() else 0,
        "path": str(mp_model),
    }

    # ONNX models
    onnx_dir = os.path.expanduser("~/.hsemotion")
    result["onnx_models"] = {}
    for fname in _ONNX_MODELS:
        p = os.path.join(onnx_dir, fname)
        result["onnx_models"][fname] = {
            "exists": os.path.exists(p),
            "size_kb": round(os.path.getsize(p) / 1024) if os.path.exists(p) else 0,
        }

    # Try initialising FaceTracker
    try:
        from core.face_tracker import FaceTracker
        ft = FaceTracker()
        ft.close()
        result["face_tracker_init"] = "OK"
    except Exception as exc:
        result["face_tracker_init"] = f"ERROR: {exc}"

    # GL env vars
    result["env"] = {
        "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "NOT SET"),
        "EGL_PLATFORM": os.environ.get("EGL_PLATFORM", "NOT SET"),
        "MESA_GL_VERSION_OVERRIDE": os.environ.get("MESA_GL_VERSION_OVERRIDE", "NOT SET"),
    }

    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
