"""
Short-term health risk prediction using trend analysis on vital sign history.

Approach:
  - Track sliding windows of BPM, fatigue, and stress
  - Detect monotone worsening trends using linear regression slope
  - Combine instantaneous thresholds with trend alarms
  - Produce 3-level risk: Normal | Warning | High Risk

This intentionally uses no external model — real-time rule-based systems
outperform under-trained ML models on small windows of noisy webcam data.
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from config.settings import settings


@dataclass
class RiskResult:
    level: str = "Normal"         # Normal | Warning | High Risk
    level_code: int = 0           # 0=Normal, 1=Warning, 2=High Risk
    messages: List[str] = field(default_factory=list)
    bpm_trend: float = 0.0        # BPM slope (BPM/min)
    fatigue_trend: float = 0.0    # Fatigue slope (%/min)
    stress_trend: float = 0.0


class HealthRiskPredictor:
    """
    Analyses rolling history of vitals to predict near-term health risk.

    Warning conditions (any one sufficient):
      - BPM > 100 or BPM < 50 (sustained >10 s)
      - Fatigue score > 50
      - Stress score > 50
      - Rising BPM trend (>5 BPM/min increase)
      - Rising fatigue trend (>3 %/min)

    High Risk conditions (any one sufficient):
      - BPM > 120 or no valid BPM for >20 s
      - Fatigue score > 75
      - Stress score > 75
      - Combined: fatigue > 50 AND stress > 50
    """

    def __init__(self):
        cfg = settings.risk
        self._cfg = cfg
        win = cfg.trend_window

        self._bpm_history:     deque[tuple[float, float]] = deque()
        self._fatigue_history: deque[tuple[float, float]] = deque()
        self._stress_history:  deque[tuple[float, float]] = deque()

        self._last_valid_bpm_time: float = time.time()
        self._last_result = RiskResult()

    def update(
        self,
        bpm: Optional[float],
        fatigue_score: float,
        stress_score: float,
    ) -> RiskResult:

        now = time.time()
        window = self._cfg.trend_window

        if bpm is not None:
            self._bpm_history.append((now, bpm))
            self._last_valid_bpm_time = now

        self._fatigue_history.append((now, fatigue_score))
        self._stress_history.append((now, stress_score))

        self._trim(self._bpm_history, now, window)
        self._trim(self._fatigue_history, now, window)
        self._trim(self._stress_history, now, window)

        messages: list[str] = []
        level_code = 0

        # --- BPM checks ---
        no_bpm_duration = now - self._last_valid_bpm_time
        if bpm is not None:
            if bpm > self._cfg.hr_high_risk:
                messages.append(f"Heart rate critically high: {bpm:.0f} BPM")
                level_code = max(level_code, 2)
            elif bpm > self._cfg.hr_high_warning:
                messages.append(f"Heart rate elevated: {bpm:.0f} BPM")
                level_code = max(level_code, 1)
            elif bpm < self._cfg.hr_low_warning:
                messages.append(f"Heart rate low: {bpm:.0f} BPM")
                level_code = max(level_code, 1)
        elif no_bpm_duration > 20.0:
            messages.append("Heart rate signal lost")
            level_code = max(level_code, 1)

        # --- Fatigue checks ---
        if fatigue_score >= self._cfg.fatigue_high_risk:
            messages.append(f"Severe fatigue detected ({fatigue_score:.0f}%)")
            level_code = max(level_code, 2)
        elif fatigue_score >= self._cfg.fatigue_warning:
            messages.append(f"Fatigue elevated ({fatigue_score:.0f}%)")
            level_code = max(level_code, 1)

        # --- Stress checks ---
        if stress_score >= self._cfg.stress_high_risk:
            messages.append(f"High stress detected ({stress_score:.0f}%)")
            level_code = max(level_code, 2)
        elif stress_score >= self._cfg.stress_warning:
            messages.append(f"Moderate stress ({stress_score:.0f}%)")
            level_code = max(level_code, 1)

        # --- Combined check ---
        if fatigue_score >= 50.0 and stress_score >= 50.0:
            messages.append("Combined fatigue and stress — consider a break")
            level_code = max(level_code, 2)

        # --- Trend analysis ---
        bpm_slope    = self._slope(self._bpm_history)
        fat_slope    = self._slope(self._fatigue_history)
        stress_slope = self._slope(self._stress_history)

        if bpm_slope > 5.0:    # BPM increasing >5/min
            messages.append(f"Heart rate rising ({bpm_slope:+.1f} BPM/min)")
            level_code = max(level_code, 1)
        if fat_slope > 3.0:    # Fatigue increasing >3%/min
            messages.append(f"Fatigue worsening ({fat_slope:+.1f} %/min)")
            level_code = max(level_code, 1)

        level = ["Normal", "Warning", "High Risk"][level_code]
        if not messages and level_code == 0:
            messages.append("All vitals within normal range")

        self._last_result = RiskResult(
            level=level,
            level_code=level_code,
            messages=messages,
            bpm_trend=round(bpm_slope, 2),
            fatigue_trend=round(fat_slope, 2),
            stress_trend=round(stress_slope, 2),
        )
        return self._last_result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _trim(buf: deque, now: float, window: float):
        while buf and (now - buf[0][0]) > window:
            buf.popleft()

    @staticmethod
    def _slope(buf: deque) -> float:
        """
        Linear regression slope in units/minute over the buffered history.
        Returns 0 if insufficient data.
        """
        if len(buf) < 5:
            return 0.0
        times  = np.array([t for t, _ in buf])
        values = np.array([v for _, v in buf])
        times_min = (times - times[0]) / 60.0
        if times_min[-1] < 0.05:
            return 0.0
        coef = np.polyfit(times_min, values, 1)
        return float(coef[0])
