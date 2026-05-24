"""
Remote Photoplethysmography (rPPG) heart-rate estimator.

Implements:
  1. POS (Plane-Orthogonal-to-Skin) algorithm
     Wang et al. (2017) — "Algorithmic Principles of Remote PPG"
     DOI: 10.1109/TBME.2016.2609282

  2. CHROM (CHROMatic difference) fallback
     De Haan & Jeanne (2013) — "Robust Pulse Rate from Chrominance-Based rPPG"
     DOI: 10.1109/TBME.2013.2266196

  3. Green-channel baseline (simplest, useful for quality check)

The pipeline:
  ROI mean RGB → temporal normalisation → POS projection
  → bandpass filter → Welch PSD → BPM with quality score
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from core.signal_buffer import SignalBuffer
from core.face_tracker import FaceLandmarks
from utils.filters import BandpassFilter, detrend_signal, normalize_signal, peak_frequency
from config.settings import settings


@dataclass
class RPPGResult:
    bpm: Optional[float] = None         # Estimated heart rate
    confidence: float = 0.0             # 0–1 signal quality
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    buffer_fill: float = 0.0            # Fraction of required buffer filled


class RPPGEstimator:
    """
    Estimates BPM from facial ROI colour changes captured by webcam.

    Usage:
        estimator = RPPGEstimator()
        for frame in ...:
            landmarks = tracker.process(frame)
            result = estimator.update(landmarks)
    """

    def __init__(self):
        cfg = settings.rppg
        self._bp_filter = BandpassFilter(cfg.filter_low, cfg.filter_high, cfg.filter_order)

        # Separate channel buffers (R, G, B) per ROI
        self._r_buf = SignalBuffer(cfg.buffer_seconds)
        self._g_buf = SignalBuffer(cfg.buffer_seconds)
        self._b_buf = SignalBuffer(cfg.buffer_seconds)

        self._last_result = RPPGResult()
        self._update_interval = 0.5        # Minimum seconds between BPM updates
        self._last_update_time: float = 0.0

    def update(self, landmarks: Optional[FaceLandmarks]) -> RPPGResult:
        """
        Ingest a new frame's ROI colours and return the latest BPM estimate.
        Call once per camera frame.
        """
        if landmarks is None:
            self._last_result = RPPGResult()
            return self._last_result

        rgb = self._extract_rgb(landmarks)
        if rgb is None:
            return self._last_result

        r, g, b = rgb
        t = time.time()
        self._r_buf.push(r, t)
        self._g_buf.push(g, t)
        self._b_buf.push(b, t)

        cfg = settings.rppg
        required_duration = 3.0  # Need at least 3 s of data
        fill = min(1.0, self._g_buf.duration / cfg.buffer_seconds)
        self._last_result.buffer_fill = fill
        self._last_result.signal = self._g_buf.values
        self._last_result.timestamps = self._g_buf.timestamps

        if (self._g_buf.duration < required_duration or
                time.time() - self._last_update_time < self._update_interval):
            return self._last_result

        self._last_update_time = time.time()
        bpm, quality = self._estimate_bpm()
        self._last_result.bpm = bpm
        self._last_result.confidence = quality
        return self._last_result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_rgb(self, lm: FaceLandmarks) -> Optional[tuple[float, float, float]]:
        """
        Average colour over available ROI patches.
        Returns mean (R, G, B) or None if no valid ROI.
        """
        patches = [lm.forehead_roi, lm.left_cheek_roi, lm.right_cheek_roi]
        valid = [p for p in patches if p is not None]
        if not valid:
            return None
        # ROI patches are BGR tuples from cv2.mean
        avg = np.mean(valid, axis=0)   # mean over ROIs
        b, g, r = avg[0], avg[1], avg[2]
        return float(r), float(g), float(b)

    def _estimate_bpm(self) -> tuple[Optional[float], float]:
        """Run POS algorithm then Welch PSD to get BPM + quality."""
        r = self._r_buf.values
        g = self._g_buf.values
        b = self._b_buf.values
        n = min(len(r), len(g), len(b))
        if n < 30:
            return None, 0.0

        r, g, b = r[-n:], g[-n:], b[-n:]
        fs = self._r_buf.sample_rate
        if fs < 5.0:
            return None, 0.0

        # --- POS algorithm (Wang 2017) ---
        # Temporal normalisation
        r_n = r / (r.mean() + 1e-9)
        g_n = g / (g.mean() + 1e-9)
        b_n = b / (b.mean() + 1e-9)

        # POS projection matrix
        s1 = g_n - b_n
        s2 = -2.0 * r_n + g_n + b_n

        # Tune alpha to equalize power
        std1 = s1.std() + 1e-9
        std2 = s2.std() + 1e-9
        alpha = std1 / std2
        pulse = s1 + alpha * s2

        # Detrend and normalise
        pulse = detrend_signal(pulse)
        pulse = normalize_signal(pulse)

        # Bandpass filter
        cfg = settings.rppg
        filtered = self._bp_filter.apply(pulse, fs)
        if filtered is None:
            # Fall back to green channel alone
            filtered = self._bp_filter.apply(detrend_signal(normalize_signal(g)), fs)
            if filtered is None:
                return None, 0.0

        # Peak frequency via Welch
        freq, quality = peak_frequency(filtered, fs, cfg.filter_low, cfg.filter_high)
        if freq < 1e-6 or quality < settings.rppg.min_signal_quality:
            return None, quality

        bpm = freq * 60.0
        if not (cfg.bpm_low <= bpm <= cfg.bpm_high):
            return None, quality

        return round(bpm, 1), quality
