"""
DSP utilities: bandpass filtering, detrending, moving averages.
"""
import numpy as np
from scipy import signal as sp_signal
from typing import Optional


class BandpassFilter:
    """
    Butterworth bandpass filter with cached coefficients.
    Re-computes only when sample rate changes by >5%.
    """

    def __init__(self, low_hz: float, high_hz: float, order: int = 4):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self._last_fs: float = 0.0
        self._b: Optional[np.ndarray] = None
        self._a: Optional[np.ndarray] = None

    def apply(self, data: np.ndarray, fs: float) -> Optional[np.ndarray]:
        """
        Filter 1-D signal `data` sampled at `fs` Hz.
        Returns filtered array or None if insufficient data.
        """
        if len(data) < self.order * 3 + 1:
            return None
        nyq = fs / 2.0
        if self.high_hz >= nyq:
            return None

        if abs(fs - self._last_fs) / max(self._last_fs, 1e-6) > 0.05:
            self._b, self._a = sp_signal.butter(
                self.order,
                [self.low_hz / nyq, self.high_hz / nyq],
                btype="band",
            )
            self._last_fs = fs

        try:
            return sp_signal.filtfilt(self._b, self._a, data)
        except Exception:
            return None


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average without look-ahead."""
    if window < 2 or len(data) < window:
        return data.copy()
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="full")[: len(data)]


def detrend_signal(data: np.ndarray) -> np.ndarray:
    """Remove linear trend from signal (baseline wander removal)."""
    if len(data) < 2:
        return data.copy()
    return sp_signal.detrend(data, type="linear")


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """Z-score normalisation."""
    std = data.std()
    if std < 1e-9:
        return np.zeros_like(data)
    return (data - data.mean()) / std


def peak_frequency(
    data: np.ndarray,
    fs: float,
    freq_low: float,
    freq_high: float,
) -> tuple[float, float]:
    """
    Return (dominant_frequency_hz, signal_quality) via Welch's PSD.
    signal_quality is the ratio of peak power to total band power.
    """
    if len(data) < 32:
        return 0.0, 0.0

    nperseg = min(len(data), 256)
    freqs, psd = sp_signal.welch(data, fs=fs, nperseg=nperseg)

    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not band_mask.any():
        return 0.0, 0.0

    band_psd = psd[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = np.argmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])

    total_power = psd.sum()
    band_power = band_psd.sum()
    quality = float(band_power / total_power) if total_power > 1e-12 else 0.0

    return peak_freq, quality
