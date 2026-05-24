"""
Central processing pipeline — background thread.

Processing order per frame:
  1. CameraManager  → BGR frame
  2. FaceTracker    → FaceLandmarks
  3. FeatureExtractor → FaceFeatures   (geometry for emotion/attention)
  4. RPPGEstimator  → RPPGResult
  5. FatigueDetector → FatigueResult
  6. StressEstimator → StressResult
  7. BreathingEstimator → BreathingResult
  8. EmotionEngine  → EmotionResult    (geometry + physiology fusion)
  9. AttentionEstimator → AttentionResult
 10. HealthRiskPredictor → RiskResult
 11. StateFusion    → MentalStateResult (wellbeing + insights)
 12. Publish PipelineState snapshot (thread-safe)
"""
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.camera import CameraManager
from core.face_tracker import FaceTracker
from ai.feature_extractor import FeatureExtractor, FaceFeatures
from vitals.rppg import RPPGEstimator, RPPGResult
from vitals.fatigue import FatigueDetector, FatigueResult
from vitals.stress import StressEstimator, StressResult
from vitals.breathing import BreathingEstimator, BreathingResult
from vitals.attention import AttentionEstimator, AttentionResult
from ai.emotion_engine import EmotionEngine, EmotionResult
from ai.deep_emotion_model import DeepEmotionModel
from ai.state_fusion import StateFusion, MentalStateResult
from prediction.health_risk import HealthRiskPredictor, RiskResult


@dataclass
class PipelineState:
    """Thread-safe snapshot of the latest pipeline outputs."""
    timestamp: float = 0.0
    frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    face_detected: bool = False
    features: FaceFeatures = field(default_factory=FaceFeatures)
    rppg: RPPGResult = field(default_factory=RPPGResult)
    fatigue: FatigueResult = field(default_factory=FatigueResult)
    stress: StressResult = field(default_factory=StressResult)
    breathing: BreathingResult = field(default_factory=BreathingResult)
    emotion: EmotionResult = field(default_factory=EmotionResult)
    attention: AttentionResult = field(default_factory=AttentionResult)
    risk: RiskResult = field(default_factory=RiskResult)
    mental_state: MentalStateResult = field(default_factory=MentalStateResult)
    camera_fps: float = 0.0
    process_fps: float = 0.0


class ProcessingPipeline:
    """
    Manages the background processing loop.

    Usage:
        pipeline = ProcessingPipeline()
        pipeline.start()
        state = pipeline.get_state()   # call from any thread
        pipeline.stop()
    """

    def __init__(self):
        self._camera    = CameraManager()
        self._tracker   = FaceTracker()
        self._features  = FeatureExtractor()
        self._rppg      = RPPGEstimator()
        self._fatigue   = FatigueDetector()
        self._stress    = StressEstimator()
        self._breathing = BreathingEstimator()
        self._emotion      = EmotionEngine()
        self._deep_emotion = DeepEmotionModel()
        self._attention = AttentionEstimator()
        self._risk      = HealthRiskPredictor()
        self._fusion    = StateFusion()

        self._state      = PipelineState()
        self._state_lock = threading.Lock()
        self._running    = False
        self._thread: Optional[threading.Thread] = None

        self._proc_count = 0
        self._proc_fps   = 0.0
        self._fps_last   = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if not self._camera.start():
            return False
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._camera.stop()
        self._tracker.close()

    def get_state(self) -> PipelineState:
        """Return a shallow copy of the latest state (thread-safe)."""
        with self._state_lock:
            s = PipelineState(
                timestamp       = self._state.timestamp,
                frame           = self._state.frame,
                annotated_frame = self._state.annotated_frame,
                face_detected   = self._state.face_detected,
                features        = self._state.features,
                rppg            = self._state.rppg,
                fatigue         = self._state.fatigue,
                stress          = self._state.stress,
                breathing       = self._state.breathing,
                emotion         = self._state.emotion,
                attention       = self._state.attention,
                risk            = self._state.risk,
                mental_state    = self._state.mental_state,
                camera_fps      = self._state.camera_fps,
                process_fps     = self._state.process_fps,
            )
        return s

    @property
    def camera_available(self) -> bool:
        return self._camera.available

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            frame = self._camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # --- Face detection + landmark extraction ---
            landmarks = self._tracker.process(frame)

            # --- Geometric feature extraction ---
            face_features = self._features.extract(landmarks)

            # --- Physiological vitals ---
            rppg_r    = self._rppg.update(landmarks)
            fatigue_r = self._fatigue.update(landmarks)
            stress_r  = self._stress.update(
                landmarks, rppg_r.bpm, fatigue_r.blink_rate
            )
            breathing_r = self._breathing.update(
                landmarks,
                rppg_r.signal[-1] if len(rppg_r.signal) > 0 else None,
            )

            # --- Deep emotion model (runs on face crop every 3 frames) ---
            deep_scores = None
            if landmarks is not None and self._deep_emotion.available:
                x, y, w_box, h_box = landmarks.face_bbox
                if w_box > 20 and h_box > 20:
                    fh, fw = frame.shape[:2]
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(fw, x + w_box)
                    y1 = min(fh, y + h_box)
                    face_crop = frame[y0:y1, x0:x1]
                    deep_scores = self._deep_emotion.predict(face_crop)

            # --- AI inference (emotion + attention) ---
            emotion_r = self._emotion.update(
                face_features,
                fatigue_r.score,
                stress_r.score,
                rppg_r.bpm,
                fatigue_r.blink_rate,
                fatigue_r.perclos,
                deep_scores,
            )
            attention_r = self._attention.update(
                face_features,
                fatigue_r.score,
                stress_r.score,
                fatigue_r.blink_rate,
            )

            # --- Risk assessment ---
            risk_r = self._risk.update(
                rppg_r.bpm, fatigue_r.score, stress_r.score
            )

            # --- Mental state fusion + insights ---
            mental_r = self._fusion.update(
                emotion_r, attention_r, rppg_r,
                fatigue_r, stress_r, breathing_r, risk_r,
            )

            # --- Annotated frame ---
            if landmarks is not None:
                annotated = self._tracker.draw_landmarks(frame, landmarks)
            else:
                annotated = frame.copy()

            # --- FPS measurement ---
            self._proc_count += 1
            now = time.time()
            elapsed = now - self._fps_last
            if elapsed >= 1.0:
                self._proc_fps   = self._proc_count / elapsed
                self._proc_count = 0
                self._fps_last   = now

            with self._state_lock:
                self._state.timestamp       = now
                self._state.frame           = frame
                self._state.annotated_frame = annotated
                self._state.face_detected   = landmarks is not None
                self._state.features        = face_features
                self._state.rppg            = rppg_r
                self._state.fatigue         = fatigue_r
                self._state.stress          = stress_r
                self._state.breathing       = breathing_r
                self._state.emotion         = emotion_r
                self._state.attention       = attention_r
                self._state.risk            = risk_r
                self._state.mental_state    = mental_r
                self._state.camera_fps      = self._camera.fps
                self._state.process_fps     = self._proc_fps
