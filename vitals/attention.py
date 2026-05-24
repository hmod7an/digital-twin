"""
Attention and cognitive-load estimation.

Attention is a composite of:
  - Head stability   (still head → engaged)
  - Forward gaze     (low head-yaw → looking at screen)
  - Eye alertness    (EAR in optimal alert range)
  - Blink cadence    (12-20 blinks/min optimal for sustained attention)
  - Absence of fatigue

Cognitive load is estimated as:
  - Physiological stress component
  - Brow-furrow tension (AU4)
  - Reduced blinking (suppressed during effortful processing)

All outputs are EMA-smoothed with α=0.10 (≈ 10-frame lag at 30 fps).

References:
  Rayner (1998) — Eye movements in reading and information processing.
  Psychological Bulletin, 124(3), 372-422. DOI: 10.1037/0033-2909.124.3.372

  Iqbal et al. (2004) — Task-evoked pupillary response to mental workload.
  CHI'04. DOI: 10.1145/985692.985802
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ai.feature_extractor import FaceFeatures


@dataclass
class AttentionResult:
    score: float = 50.0           # 0-100; higher = more attentive/focused
    level: str = "Normal"         # Distracted | Low | Normal | Focused | Deep Focus
    cognitive_load: float = 0.0   # 0-100; mental effort estimate
    gaze_score: float = 0.0       # 0-100; forward gaze quality
    stability_score: float = 0.0  # 0-100; head stillness


class AttentionEstimator:
    """Estimates real-time attention level and cognitive load."""

    EMA_ALPHA = 0.10  # lower = smoother, more lag

    def __init__(self):
        self._score_ema    = 50.0
        self._cog_ema      = 0.0
        self._gaze_ema     = 50.0
        self._stab_ema     = 50.0

    def update(
        self,
        features: FaceFeatures,
        fatigue_score: float,
        stress_score: float,
        blink_rate: float,
    ) -> AttentionResult:

        if not features.valid:
            return AttentionResult(level="No Face")

        fat = fatigue_score / 100.0
        str_ = stress_score / 100.0

        # --- Gaze quality (forward-facing proxy) ---
        gaze = float(np.clip(1.0 - features.head_yaw_norm * 1.4, 0.0, 1.0)) * 100.0

        # --- Head stability ---
        stab = float(np.clip(1.0 - features.head_movement_score * 1.3, 0.0, 1.0)) * 100.0

        # --- Eye alert state (optimal EAR range 0.26-0.34) ---
        ear_quality = float(np.clip(
            1.0 - abs(features.ear - 0.30) / 0.10, 0.0, 1.0
        )) * 100.0

        # --- Blink cadence (optimal 12-20 blinks/min) ---
        blink_quality = float(np.clip(
            1.0 - abs(blink_rate - 16.0) / 14.0, 0.0, 1.0
        )) * 100.0

        # --- Composite attention score ---
        raw_score = (
            0.30 * stab +
            0.28 * gaze +
            0.20 * ear_quality +
            0.12 * blink_quality +
            0.10 * (1.0 - fat) * 100.0
        )

        # --- Cognitive load (effort proxy) ---
        # Increases with stress, brow tension, and suppressed blinking
        raw_cog = (
            0.40 * str_ * 100.0 +
            0.35 * features.brow_furrow_score * 100.0 +
            0.25 * float(np.clip((15.0 - blink_rate) / 15.0, 0.0, 1.0)) * 100.0
        )

        a = self.EMA_ALPHA
        self._score_ema = a * raw_score + (1.0 - a) * self._score_ema
        self._cog_ema   = a * raw_cog   + (1.0 - a) * self._cog_ema
        self._gaze_ema  = a * gaze      + (1.0 - a) * self._gaze_ema
        self._stab_ema  = a * stab      + (1.0 - a) * self._stab_ema

        score = round(float(np.clip(self._score_ema, 0.0, 100.0)), 1)
        cog   = round(float(np.clip(self._cog_ema,   0.0, 100.0)), 1)

        return AttentionResult(
            score=score,
            level=_classify(score),
            cognitive_load=cog,
            gaze_score=round(self._gaze_ema, 1),
            stability_score=round(self._stab_ema, 1),
        )


def _classify(score: float) -> str:
    if score >= 82:
        return "Deep Focus"
    elif score >= 65:
        return "Focused"
    elif score >= 44:
        return "Normal"
    elif score >= 26:
        return "Low"
    return "Distracted"
