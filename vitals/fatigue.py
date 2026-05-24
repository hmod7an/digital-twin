"""
Fatigue detection via Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
PERCLOS, blink frequency, and yawn detection.

References:
  Soukupová & Čech (2016) — Real-Time Eye Blink Detection Using Facial Landmarks
  http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

  Dinges & Grace (1998) — PERCLOS: A Valid Psychophysiological Measure of Alertness
  (Federal Highway Administration, Report FHWA-MCRT-98-006)
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

from core.face_tracker import FaceLandmarks
from config.settings import settings


@dataclass
class FatigueResult:
    score: float = 0.0          # 0–100 composite fatigue score
    state: str = "Normal"       # Normal | Tired | Drowsy
    ear: float = 0.0            # Current Eye Aspect Ratio
    mar: float = 0.0            # Current Mouth Aspect Ratio
    blink_rate: float = 0.0     # Blinks per minute (60-s window)
    perclos: float = 0.0        # % time eyes closed in last 30 s
    is_yawning: bool = False
    blink_count_session: int = 0


class FatigueDetector:
    """
    Computes a composite fatigue score from real-time facial landmarks.

    Score composition:
      40% PERCLOS (proportion of time eyes are closed)
      25% Blink rate deviation from normal range
      20% EAR trend (sustained low EAR)
      15% Yawn frequency
    """

    # MediaPipe Face Mesh indices for EAR computation
    # Left eye:  P1=33, P2=160, P3=158, P4=133, P5=153, P6=144
    # Right eye: P1=362, P2=385, P3=387, P4=263, P5=373, P6=380
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks for MAR:  top-center=13, bottom-center=14
    # left-corner=61, right-corner=291, left-inner=78, right-inner=308
    MOUTH_OUTER = [61, 291, 13, 14]
    MOUTH_INNER = [78, 308, 95, 324]

    def __init__(self):
        cfg = settings.fatigue
        self._cfg = cfg

        # Rolling history for PERCLOS
        self._eye_closed_history: deque[tuple[float, bool]] = deque()
        # Timestamps of detected blinks
        self._blink_times: deque[float] = deque()

        # Blink state machine
        self._frames_below_ear = 0
        self._is_eye_open = True

        # Yawn state
        self._frames_mouth_open = 0
        self._yawn_times: deque[float] = deque()

        self._session_blinks = 0
        self._last_result = FatigueResult()

    def update(self, landmarks: Optional[FaceLandmarks]) -> FatigueResult:
        if landmarks is None:
            return FatigueResult(state="No Face")

        now = time.time()
        ear = self._compute_ear(landmarks)
        mar = self._compute_mar(landmarks)

        # --- Blink detection ---
        if ear < self._cfg.ear_threshold:
            self._frames_below_ear += 1
        else:
            if self._frames_below_ear >= self._cfg.ear_consec_frames:
                self._blink_times.append(now)
                self._session_blinks += 1
            self._frames_below_ear = 0

        eye_closed = ear < self._cfg.ear_drowsy_threshold
        self._eye_closed_history.append((now, eye_closed))
        self._evict_old(self._eye_closed_history, now, self._cfg.perclos_window)
        self._evict_old(self._blink_times, now, self._cfg.blink_window)

        # --- Yawn detection ---
        if mar > self._cfg.mar_threshold:
            self._frames_mouth_open += 1
        else:
            if self._frames_mouth_open >= 15:  # ~0.5 s at 30 fps
                self._yawn_times.append(now)
            self._frames_mouth_open = 0
        self._evict_old(self._yawn_times, now, 120.0)  # 2-min window

        # --- PERCLOS ---
        perclos = self._compute_perclos()

        # --- Blink rate ---
        blink_rate = len(self._blink_times) * (60.0 / self._cfg.blink_window)

        # --- Score fusion ---
        score = self._fuse_score(ear, perclos, blink_rate)
        state = self._classify_state(score, ear)

        result = FatigueResult(
            score=score,
            state=state,
            ear=round(ear, 3),
            mar=round(mar, 3),
            blink_rate=round(blink_rate, 1),
            perclos=round(perclos * 100, 1),
            is_yawning=mar > self._cfg.mar_threshold,
            blink_count_session=self._session_blinks,
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _compute_ear(self, lm: FaceLandmarks) -> float:
        """
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        Averaged over both eyes.
        """
        left  = self._eye_aspect_ratio(lm.pixel_coords, self.LEFT_EYE)
        right = self._eye_aspect_ratio(lm.pixel_coords, self.RIGHT_EYE)
        return (left + right) / 2.0

    @staticmethod
    def _eye_aspect_ratio(coords: np.ndarray, indices: list) -> float:
        try:
            p = [coords[i].astype(float) for i in indices]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h  = np.linalg.norm(p[0] - p[3])
            return float((v1 + v2) / (2.0 * h + 1e-9))
        except Exception:
            return 0.3  # Default open-eye value

    def _compute_mar(self, lm: FaceLandmarks) -> float:
        """
        MAR = vertical mouth opening / horizontal mouth width.
        """
        try:
            pts = lm.pixel_coords
            # vertical: distance between lip landmarks 13 (top) and 14 (bottom)
            top    = pts[13].astype(float)
            bottom = pts[14].astype(float)
            left   = pts[61].astype(float)
            right  = pts[291].astype(float)
            vertical   = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left - right)
            return float(vertical / (horizontal + 1e-9))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_perclos(self) -> float:
        """Fraction of time eyes were below drowsy-EAR threshold."""
        if not self._eye_closed_history:
            return 0.0
        closed = sum(1 for _, c in self._eye_closed_history if c)
        return closed / len(self._eye_closed_history)

    def _fuse_score(self, ear: float, perclos: float, blink_rate: float) -> float:
        cfg = self._cfg

        # PERCLOS component (0-100)
        perclos_score = min(100.0, perclos * 200.0)  # 50% PERCLOS → score 100

        # EAR component — how far below the normal open-eye level
        normal_ear = 0.30
        ear_score = max(0.0, (normal_ear - ear) / normal_ear * 100.0)

        # Blink rate deviation
        if cfg.normal_blink_low <= blink_rate <= cfg.normal_blink_high:
            blink_score = 0.0
        elif blink_rate < cfg.normal_blink_low:
            # Reduced blinking → concentration or early fatigue
            blink_score = (cfg.normal_blink_low - blink_rate) / cfg.normal_blink_low * 60.0
        else:
            # Excessive blinking → eye strain / fatigue
            blink_score = min(80.0, (blink_rate - cfg.normal_blink_high) * 3.0)

        # Yawn contribution
        yawn_score = min(100.0, len(self._yawn_times) * 20.0)

        score = (
            0.40 * perclos_score +
            0.25 * blink_score +
            0.20 * ear_score +
            0.15 * yawn_score
        )
        return round(min(100.0, max(0.0, score)), 1)

    @staticmethod
    def _classify_state(score: float, ear: float) -> str:
        if ear < settings.fatigue.ear_drowsy_threshold and score > 40:
            return "Drowsy"
        if score >= settings.risk.fatigue_warning:
            return "Tired"
        return "Normal"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _evict_old(buf: deque, now: float, window: float):
        while buf and (now - (buf[0] if isinstance(buf[0], float) else buf[0][0])) > window:
            buf.popleft()
