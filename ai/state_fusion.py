"""
Multi-modal mental state fusion and AI insight generation.

Fuses all pipeline modalities into:
  - Overall wellbeing score (0-100)
  - Trend direction (Improving / Stable / Declining)
  - Natural-language AI insights (max 5 lines)
  - Actionable recommendations (max 3 lines)
  - Alert level (0=none, 1=notice, 2=warning)

No ML model required — fusion is rule-based with EMA smoothing.
The natural-language output is template-driven with signal interpolation.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from ai.emotion_engine import EmotionResult
from vitals.attention import AttentionResult
from vitals.rppg import RPPGResult
from vitals.fatigue import FatigueResult
from vitals.stress import StressResult
from vitals.breathing import BreathingResult
from prediction.health_risk import RiskResult


@dataclass
class MentalStateResult:
    wellbeing_score: float = 75.0     # 0-100; higher = better
    wellbeing_label: str = "Good"     # Poor | Fair | Good | Excellent
    trend: str = "Stable"             # Improving | Stable | Declining
    trend_icon: str = "→"             # ↑ | → | ↓
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alert_level: int = 0              # 0=none, 1=notice, 2=warning


class StateFusion:
    """
    Fuses all vitals + emotion + attention into a holistic mental state.
    Generates human-readable insights and personalised recommendations.
    """

    TREND_WINDOW = 30
    EMA_ALPHA    = 0.07   # low alpha → slow, stable wellbeing score

    def __init__(self):
        self._wb_ema = 75.0
        self._wb_history: deque[float] = deque(maxlen=self.TREND_WINDOW)

    def update(
        self,
        emotion:   EmotionResult,
        attention: AttentionResult,
        rppg:      RPPGResult,
        fatigue:   FatigueResult,
        stress:    StressResult,
        breathing: BreathingResult,
        risk:      RiskResult,
    ) -> MentalStateResult:

        raw_wb = self._wellbeing(emotion, attention, fatigue, stress, rppg)
        self._wb_ema = self.EMA_ALPHA * raw_wb + (1.0 - self.EMA_ALPHA) * self._wb_ema
        score = round(float(np.clip(self._wb_ema, 0.0, 100.0)), 1)
        self._wb_history.append(score)

        trend, icon = self._trend()
        label       = _wb_label(score)
        insights    = self._insights(emotion, attention, fatigue, stress, rppg, breathing)
        recs        = self._recommendations(fatigue, stress, attention, emotion)
        alert       = self._alert(fatigue, stress, rppg, risk)

        return MentalStateResult(
            wellbeing_score=score,
            wellbeing_label=label,
            trend=trend,
            trend_icon=icon,
            insights=insights,
            recommendations=recs,
            alert_level=alert,
        )

    # ------------------------------------------------------------------
    # Wellbeing score
    # ------------------------------------------------------------------

    @staticmethod
    def _wellbeing(em, at, fa, st, rp) -> float:
        base = 80.0
        base -= fa.score * 0.25        # fatigue penalty
        base -= st.score * 0.20        # stress penalty

        _emotion_delta = {
            "Happy":      +8.0,
            "Focused":    +5.0,
            "Surprised":  +2.0,
            "Neutral":     0.0,
            "Distracted": -4.0,
            "Tired":      -6.0,
            "Stressed":   -8.0,
            "Sad":        -10.0,
            "Angry":      -12.0,
        }
        base += _emotion_delta.get(em.state, 0.0)
        base += (at.score - 50.0) * 0.10   # attention bonus/penalty

        if rp.bpm and 55.0 <= rp.bpm <= 90.0:
            base += 3.0   # healthy resting HR bonus

        return float(np.clip(base, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def _trend(self):
        h = list(self._wb_history)
        if len(h) < 10:
            return "Stable", "→"
        n   = len(h)
        old = np.mean(h[:n // 3])
        new = np.mean(h[-n // 3:])
        delta = new - old
        if delta > 3.0:
            return "Improving", "↑"
        elif delta < -3.0:
            return "Declining", "↓"
        return "Stable", "→"

    # ------------------------------------------------------------------
    # Insight generation
    # ------------------------------------------------------------------

    def _insights(self, em, at, fa, st, rp, br) -> List[str]:
        out = []

        # Emotion
        if em.is_stable and em.state not in ("No Face",):
            conf_pct = int(em.confidence * 100)
            out.append(
                f"{em.emoji} Emotional state: {em.state} "
                f"(confidence {conf_pct}%)"
            )

        # Fatigue
        if fa.score >= 65:
            out.append(
                f"😴 High fatigue ({fa.score:.0f}%) — "
                f"PERCLOS {fa.perclos:.1f}%, blinks {fa.blink_rate:.0f}/min"
            )
        elif fa.score >= 38:
            out.append(f"😴 Moderate fatigue ({fa.score:.0f}%)")

        # Stress
        if st.score >= 60:
            out.append(
                f"🧠 Elevated stress ({st.score:.0f}%) — "
                f"HRV {st.hr_variability:.1f}, head motion {st.head_movement:.2f}"
            )
        elif st.score >= 35:
            out.append(f"🧠 Mild stress detected ({st.score:.0f}%)")

        # Attention
        if at.score < 30:
            out.append(
                f"🎯 Low attention ({at.score:.0f}%) — "
                f"gaze stability {at.gaze_score:.0f}%"
            )
        elif at.score >= 75:
            out.append(f"🎯 High focus state ({at.score:.0f}%)")

        # HR
        if rp.bpm:
            if rp.bpm > 100:
                out.append(f"❤️ Heart rate elevated: {rp.bpm:.0f} BPM")
            elif rp.bpm < 52:
                out.append(f"❤️ Heart rate low: {rp.bpm:.0f} BPM")
            else:
                out.append(f"❤️ Heart rate normal: {rp.bpm:.0f} BPM")
        else:
            out.append("❤️ Heart rate calibrating…")

        # Breathing
        if br.rate_bpm and br.confidence > 0.3:
            if br.rate_bpm > 20:
                out.append(f"🫁 Breathing elevated: {br.rate_bpm:.0f} br/min")
            elif br.rate_bpm < 8:
                out.append(f"🫁 Very slow breathing: {br.rate_bpm:.0f} br/min")

        return out[:5]

    def _recommendations(self, fa, st, at, em) -> List[str]:
        recs = []
        if fa.score >= 60:
            recs.append("💡 Take a 5-minute break — high fatigue detected")
        if st.score >= 55:
            recs.append("💡 Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s")
        if at.score < 30:
            recs.append("💡 Reduce distractions — silence notifications")
        if em.state == "Tired" and fa.score >= 50:
            recs.append("💡 A 10-20 min nap now can restore alertness significantly")
        if em.state == "Angry":
            recs.append("💡 Step away for 90 seconds — emotions peak then fade")
        if not recs:
            recs.append("✅ All vitals in healthy range — keep it up!")
        return recs[:3]

    # ------------------------------------------------------------------
    # Alert level
    # ------------------------------------------------------------------

    @staticmethod
    def _alert(fa, st, rp, risk) -> int:
        if risk.level_code == 2 or fa.score >= 72 or st.score >= 72:
            return 2
        if risk.level_code == 1 or fa.score >= 45 or st.score >= 45:
            return 1
        return 0


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _wb_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 63:
        return "Good"
    elif score >= 43:
        return "Fair"
    return "Poor"
