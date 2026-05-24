"""
Stress estimation using multi-modal fusion of physiological indicators.

Rule-based approach combining:
  - Heart rate variability (HRV-like metric from rPPG BPM variance)
  - Blink rate deviation
  - Head movement intensity
  - Facial action unit proxies (brow, jaw tension via landmark distances)

References:
  Giannakakis et al. (2019) — Review on Psychological Stress Detection Using Biosignals
  DOI: 10.1109/RBME.2019.2909969

  Aigrain et al. (2015) — Person-Independent and Person-Dependent Stress Detection
  Based on Instantaneous Variations in Physiological Signals
  DOI: 10.1145/2823465.2823491
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.face_tracker import FaceLandmarks
from core.signal_buffer import SignalBuffer
from config.settings import settings


@dataclass
class StressResult:
    score: float = 0.0          # 0–100
    state: str = "Calm"         # Calm | Moderate | High
    hr_variability: float = 0.0 # BPM standard deviation over window
    head_movement: float = 0.0  # RMS head displacement (degrees)
    brow_tension: float = 0.0   # Normalised brow-eye distance ratio
    components: dict = field(default_factory=dict)


class StressEstimator:
    """
    Estimates stress from rPPG BPM history, blink cadence, head motion,
    and facial geometry (brow furrowing as an AU4 proxy).

    Designed to be called every frame with the latest landmarks and BPM.
    """

    # MediaPipe landmarks for brow tension proxy
    # Inner brow: 107 (left), 336 (right)
    # Eye centre:  159 (left), 386 (right)
    LEFT_INNER_BROW  = 107
    RIGHT_INNER_BROW = 336
    LEFT_EYE_CENTER  = 159
    RIGHT_EYE_CENTER = 386

    def __init__(self):
        cfg = settings.stress
        self._cfg = cfg
        self._bpm_history = SignalBuffer(cfg.hr_trend_window)
        self._head_yaw_history: deque[float] = deque(maxlen=cfg.head_movement_window)
        self._head_pitch_history: deque[float] = deque(maxlen=cfg.head_movement_window)
        self._blink_rate_history: deque[float] = deque(maxlen=30)
        self._last_result = StressResult()

    def update(
        self,
        landmarks: Optional[FaceLandmarks],
        bpm: Optional[float],
        blink_rate: float,
    ) -> StressResult:

        if landmarks is None:
            return StressResult(state="No Face")

        now = time.time()

        # Track BPM
        if bpm is not None and 40 <= bpm <= 180:
            self._bpm_history.push(bpm, now)

        # Track head pose
        yaw, pitch, _ = landmarks.head_pose_angles
        self._head_yaw_history.append(yaw)
        self._head_pitch_history.append(pitch)

        # Track blink rate
        self._blink_rate_history.append(blink_rate)

        # --- Component scores ---
        hrv_score = self._hr_variability_score()
        head_score, head_rms = self._head_movement_score()
        blink_score = self._blink_stress_score()
        brow_score, brow_tension = self._brow_tension_score(landmarks)

        # Weighted fusion
        score = (
            self._cfg.weight_hr_variability * hrv_score +
            self._cfg.weight_blink_rate     * blink_score +
            self._cfg.weight_head_movement  * head_score +
            self._cfg.weight_facial_tension * brow_score
        )
        score = round(min(100.0, max(0.0, score)), 1)
        state = self._classify(score)

        self._last_result = StressResult(
            score=score,
            state=state,
            hr_variability=round(self._bpm_history.values.std()
                                 if len(self._bpm_history) > 2 else 0.0, 2),
            head_movement=round(head_rms, 2),
            brow_tension=round(brow_tension, 3),
            components={
                "hrv": round(hrv_score, 1),
                "blink": round(blink_score, 1),
                "head": round(head_score, 1),
                "brow": round(brow_score, 1),
            },
        )
        return self._last_result

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _hr_variability_score(self) -> float:
        """
        High short-term BPM variance → elevated sympathetic activity → stress.
        Normal HRV ~5–15 BPM std dev; >20 BPM std dev → stressed.
        """
        vals = self._bpm_history.values
        if len(vals) < 5:
            return 0.0
        std = float(vals.std())
        # Map: 0 std → 0, 20+ std → 100
        return min(100.0, std / 20.0 * 100.0)

    def _head_movement_score(self) -> tuple[float, float]:
        """
        High-frequency head jitter (tremor, agitation) signals stress.
        """
        if len(self._head_yaw_history) < 3:
            return 0.0, 0.0
        yaw = np.array(self._head_yaw_history)
        pitch = np.array(self._head_pitch_history)
        # RMS of frame-to-frame differences
        yaw_diff   = np.diff(yaw)
        pitch_diff = np.diff(pitch)
        rms = float(np.sqrt(np.mean(yaw_diff ** 2 + pitch_diff ** 2)))
        # Normal head movement ~0.5–2 deg/frame; >5 → high stress
        return min(100.0, rms / 5.0 * 100.0), rms

    def _blink_stress_score(self) -> float:
        """
        Both very high and very low blink rates associate with stress.
        Baseline ~15 blinks/min; under 8 or over 30 → stress.
        """
        if not self._blink_rate_history:
            return 0.0
        rate = float(np.mean(self._blink_rate_history))
        if 8.0 <= rate <= 25.0:
            return 0.0
        elif rate < 8.0:
            return min(100.0, (8.0 - rate) / 8.0 * 120.0)
        else:
            return min(100.0, (rate - 25.0) * 3.0)

    def _brow_tension_score(self, lm: FaceLandmarks) -> tuple[float, float]:
        """
        Brow-to-eye distance decreases with furrowing (AU4).
        Lower ratio → more tension → more stress.
        """
        try:
            coords = lm.pixel_coords.astype(float)
            left_brow_eye  = np.linalg.norm(
                coords[self.LEFT_INNER_BROW] - coords[self.LEFT_EYE_CENTER])
            right_brow_eye = np.linalg.norm(
                coords[self.RIGHT_INNER_BROW] - coords[self.RIGHT_EYE_CENTER])
            avg_dist = (left_brow_eye + right_brow_eye) / 2.0

            # Normalise by inter-eye distance for scale invariance
            inter_eye = np.linalg.norm(
                coords[self.LEFT_EYE_CENTER] - coords[self.RIGHT_EYE_CENTER])
            ratio = float(avg_dist / (inter_eye + 1e-9))

            # Typical relaxed ratio ~0.45; furrowed ~0.30
            # Below 0.30 → high tension
            tension_score = max(0.0, (0.45 - ratio) / 0.15 * 100.0)
            return min(100.0, tension_score), ratio
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _classify(score: float) -> str:
        if score >= settings.risk.stress_high_risk:
            return "High"
        elif score >= settings.risk.stress_warning:
            return "Moderate"
        return "Calm"
