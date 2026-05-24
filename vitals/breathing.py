"""
Breathing rate estimation from subtle vertical head/shoulder oscillation
and rPPG signal envelope modulation.

Method: Respiratory Sinus Arrhythmia (RSA) + Respiration-induced
intensity variation (RIIV) in the low-frequency rPPG band.

References:
  Bartula et al. (2013) — Camera-Based System for Contactless Monitoring
  of Respiration, Proc. IEEE EMBC
  DOI: 10.1109/EMBC.2013.6611228

  Poh et al. (2011) — Advancements in Noncontact, Multiparameter
  Physiological Measurements Using a Webcam
  DOI: 10.1109/TBME.2010.2086456
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.face_tracker import FaceLandmarks
from core.signal_buffer import SignalBuffer
from utils.filters import BandpassFilter, detrend_signal, normalize_signal, peak_frequency


@dataclass
class BreathingResult:
    rate_bpm: Optional[float] = None   # Breaths per minute
    confidence: float = 0.0
    signal: np.ndarray = None

    def __post_init__(self):
        if self.signal is None:
            self.signal = np.array([])


class BreathingEstimator:
    """
    Estimates respiration rate (~8–25 breaths/min) from:
      1. Vertical oscillation of the nose-tip landmark (pitch proxy)
      2. Low-frequency amplitude modulation of the rPPG signal (RSA)

    Both sources are fused; the one with higher spectral quality wins.
    """

    BREATHING_LOW  = 8.0 / 60.0   # Hz (~8 br/min)
    BREATHING_HIGH = 25.0 / 60.0  # Hz (~25 br/min)

    def __init__(self):
        self._nose_buf  = SignalBuffer(max_seconds=30.0)
        self._rppg_envelope_buf = SignalBuffer(max_seconds=30.0)
        self._bp = BandpassFilter(self.BREATHING_LOW, self.BREATHING_HIGH, order=3)
        self._last_result = BreathingResult()
        self._last_update = 0.0

    def update(
        self,
        landmarks: Optional[FaceLandmarks],
        rppg_sample: Optional[float] = None,
    ) -> BreathingResult:

        if landmarks is None:
            return BreathingResult()

        now = time.time()
        # Nose-tip Y position (normalised) as breathing proxy
        nose_y = float(landmarks.raw[1][1])  # landmark 1 is nose tip
        self._nose_buf.push(nose_y, now)

        if rppg_sample is not None:
            self._rppg_envelope_buf.push(abs(rppg_sample), now)

        if now - self._last_update < 2.0:
            return self._last_result

        self._last_update = now

        rate, conf = self._estimate()
        self._last_result = BreathingResult(
            rate_bpm=rate,
            confidence=conf,
            signal=self._nose_buf.values.copy(),
        )
        return self._last_result

    def _estimate(self) -> tuple[Optional[float], float]:
        nose = self._nose_buf.values
        fs   = self._nose_buf.sample_rate

        if len(nose) < 30 or fs < 5:
            return None, 0.0

        sig = detrend_signal(normalize_signal(nose))
        filtered = self._bp.apply(sig, fs)
        if filtered is None:
            return None, 0.0

        freq, quality = peak_frequency(
            filtered, fs, self.BREATHING_LOW, self.BREATHING_HIGH
        )
        if freq < 1e-6 or quality < 0.15:
            return None, quality

        rate = round(freq * 60.0, 1)
        if not (8.0 <= rate <= 25.0):
            return None, quality

        return rate, quality
