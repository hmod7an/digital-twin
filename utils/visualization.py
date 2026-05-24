"""
OpenCV overlay helpers for the live camera feed.
"""
import cv2
import numpy as np
from typing import Optional


def draw_signal_overlay(
    frame: np.ndarray,
    signal: np.ndarray,
    bpm: Optional[float],
    fatigue_score: float,
    stress_score: float,
    fps: float,
) -> np.ndarray:
    """
    Draw a HUD overlay on the webcam frame for standalone preview.
    Used in the processing thread; NOT in Streamlit (Plotly handles that).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent black banner at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    bpm_text = f"HR: {bpm:.0f} BPM" if bpm else "HR: --"
    cv2.putText(out, bpm_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2)

    fat_color = _score_color(fatigue_score)
    cv2.putText(out, f"Fatigue: {fatigue_score:.0f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fat_color, 2)

    str_color = _score_color(stress_score)
    cv2.putText(out, f"Stress: {stress_score:.0f}%", (220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, str_color, 2)

    cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Mini rPPG waveform at bottom
    if len(signal) > 10:
        _draw_mini_waveform(out, signal, w, h)

    return out


def _score_color(score: float) -> tuple[int, int, int]:
    """Green → yellow → red gradient based on score 0-100."""
    if score < 40:
        return (50, 220, 50)
    elif score < 65:
        return (50, 220, 220)
    else:
        return (50, 50, 255)


def _draw_mini_waveform(
    frame: np.ndarray, signal: np.ndarray, w: int, h: int
):
    """Draw a 100-pixel tall waveform strip at the bottom of the frame."""
    strip_h = 60
    y_base = h - strip_h - 5
    x_start, x_end = 10, w - 10
    n_pts = x_end - x_start

    # Downsample or crop signal to n_pts
    if len(signal) > n_pts:
        sig = signal[-n_pts:]
    else:
        sig = signal

    sig_min, sig_max = sig.min(), sig.max()
    rng = sig_max - sig_min
    if rng < 1e-9:
        return

    normalized = (sig - sig_min) / rng
    pts = []
    for i, v in enumerate(normalized):
        x = x_start + int(i * n_pts / len(normalized))
        y = y_base + strip_h - int(v * (strip_h - 4))
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 212, 255), 1)
