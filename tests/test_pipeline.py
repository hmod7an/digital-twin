"""
Unit tests for the core signal processing pipeline.
Run with: python -m pytest tests/ -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.filters import (
    BandpassFilter, moving_average, detrend_signal,
    normalize_signal, peak_frequency,
)
from core.signal_buffer import SignalBuffer
from prediction.health_risk import HealthRiskPredictor, RiskResult


# ------------------------------------------------------------------
# Signal buffer tests
# ------------------------------------------------------------------

class TestSignalBuffer:
    def test_basic_push_and_read(self):
        buf = SignalBuffer(max_seconds=30.0)  # window > 10 s, no eviction
        for i in range(10):
            buf.push(float(i), timestamp=float(i))
        assert len(buf) == 10
        np.testing.assert_array_equal(buf.values, np.arange(10, dtype=float))

    def test_eviction(self):
        buf = SignalBuffer(max_seconds=3.0)
        for t in range(6):
            buf.push(float(t), timestamp=float(t))
        # Timestamps 0–2 should be evicted (now=5, window=3 → keep t>=2)
        assert len(buf) <= 4

    def test_sample_rate(self):
        buf = SignalBuffer(max_seconds=10.0)
        for i in range(31):
            buf.push(1.0, timestamp=i / 30.0)
        assert abs(buf.sample_rate - 30.0) < 1.0


# ------------------------------------------------------------------
# Filter tests
# ------------------------------------------------------------------

class TestBandpassFilter:
    def test_filters_sine_in_band(self):
        fs = 30.0
        t = np.linspace(0, 5, int(5 * fs))
        # 1.2 Hz sine (in rPPG band 0.7–3.0 Hz)
        sig = np.sin(2 * np.pi * 1.2 * t)
        filt = BandpassFilter(0.7, 3.0, order=4)
        out = filt.apply(sig, fs)
        assert out is not None
        assert out.std() > 0.1  # signal should pass through

    def test_attenuates_out_of_band(self):
        fs = 30.0
        t = np.linspace(0, 5, int(5 * fs))
        # 5 Hz signal (above rPPG band)
        sig = np.sin(2 * np.pi * 5.0 * t)
        filt = BandpassFilter(0.7, 3.0, order=4)
        out = filt.apply(sig, fs)
        assert out is not None
        assert out.std() < 0.5  # should be attenuated

    def test_returns_none_for_short_signal(self):
        filt = BandpassFilter(0.7, 3.0, order=4)
        out = filt.apply(np.array([1.0, 2.0]), fs=30.0)
        assert out is None


# ------------------------------------------------------------------
# Peak frequency tests
# ------------------------------------------------------------------

class TestPeakFrequency:
    def test_detects_known_frequency(self):
        fs = 30.0
        t = np.linspace(0, 10, int(10 * fs))
        freq_hz = 1.1  # ~66 BPM
        sig = np.sin(2 * np.pi * freq_hz * t)
        detected_freq, quality = peak_frequency(sig, fs, 0.7, 3.0)
        assert abs(detected_freq - freq_hz) < 0.15
        assert quality > 0.4

    def test_returns_zero_for_noise(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(300)
        freq, quality = peak_frequency(sig, 30.0, 0.7, 3.0)
        # Noise: quality should be low
        assert quality < 0.5


# ------------------------------------------------------------------
# Health risk predictor tests
# ------------------------------------------------------------------

class TestHealthRiskPredictor:
    def test_normal_state(self):
        pred = HealthRiskPredictor()
        result = pred.update(bpm=70.0, fatigue_score=20.0, stress_score=15.0)
        assert result.level == "Normal"
        assert result.level_code == 0

    def test_warning_high_bpm(self):
        pred = HealthRiskPredictor()
        result = pred.update(bpm=110.0, fatigue_score=10.0, stress_score=10.0)
        assert result.level_code >= 1

    def test_high_risk_combined(self):
        pred = HealthRiskPredictor()
        result = pred.update(bpm=75.0, fatigue_score=80.0, stress_score=80.0)
        assert result.level_code == 2
        assert result.level == "High Risk"

    def test_high_risk_bpm(self):
        pred = HealthRiskPredictor()
        result = pred.update(bpm=130.0, fatigue_score=10.0, stress_score=10.0)
        assert result.level_code == 2


# ------------------------------------------------------------------
# Detrend and normalize
# ------------------------------------------------------------------

class TestSignalProcessing:
    def test_detrend_removes_linear_trend(self):
        t = np.linspace(0, 1, 100)
        sig = t + np.sin(2 * np.pi * 2.0 * t)
        detrended = detrend_signal(sig)
        # After detrending, slope should be near zero
        coef = np.polyfit(t, detrended, 1)
        assert abs(coef[0]) < 0.5

    def test_normalize_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(200) * 5 + 10
        normed = normalize_signal(sig)
        assert abs(normed.mean()) < 0.01
        assert abs(normed.std() - 1.0) < 0.01
