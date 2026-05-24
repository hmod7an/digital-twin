"""
Emotion classification engine — v2 (stable, human-like).

Architecture:
  Raw scorer
    → Transition-penalty dampening  (blocks unrealistic jumps)
    → Per-state EMA smoothing       (Sad slow, Surprised fast)
    → Inertia bias                  (current state resists displacement)
    → Per-state commit threshold    (Angry/Sad need longer evidence)
    → EmotionResult with stability + timeline

Key design principles:
  - Facial-energy gating: tired/passive face cannot fire Angry.
  - Sad > Angry protection: medium Sad evidence beats weak Angry.
  - Transition costs prevent mood whiplash (Happy → Angry instant block).
  - Inertia keeps state stable across brief neutral moments.
  - 60-second timeline → dominant-trend label.
"""
import numpy as np
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from ai.feature_extractor import FaceFeatures

EMOTION_STATES = [
    "Neutral", "Happy", "Sad", "Angry",
    "Stressed", "Tired", "Surprised",
    "Focused", "Distracted",
]

EMOTION_EMOJI: Dict[str, str] = {
    "Neutral":    "😐",
    "Happy":      "😊",
    "Sad":        "😢",
    "Angry":      "😠",
    "Stressed":   "😤",
    "Tired":      "😴",
    "Surprised":  "😲",
    "Focused":    "🎯",
    "Distracted": "😵",
    "No Face":    "❌",
}

EMOTION_COLOR: Dict[str, str] = {
    "Neutral":    "#9CA3AF",
    "Happy":      "#00FF88",
    "Sad":        "#60A5FA",
    "Angry":      "#FF4B4B",
    "Stressed":   "#FF8C00",
    "Tired":      "#A78BFA",
    "Surprised":  "#FBBF24",
    "Focused":    "#00D4FF",
    "Distracted": "#F87171",
    "No Face":    "#374151",
}


@dataclass
class EmotionResult:
    state: str = "Neutral"
    emoji: str = "😐"
    color: str = "#9CA3AF"
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    is_stable: bool = False

    # ── Raw signal debug ─────────────────────────────────────────────
    smile_score: float = 0.0
    furrow_score: float = 0.0
    frown_score: float = 0.0
    inner_brow_raise: float = 0.0
    facial_energy: float = 0.5

    # ── Temporal state tracking ──────────────────────────────────────
    stability_score: float = 0.0       # 0→1 over ~3 s of stable state
    persistence_seconds: float = 0.0   # seconds in current committed state
    is_calibrating: bool = True        # True during warm-up phase
    timeline_dominant: str = "Neutral" # dominant emotion in last 30 s


class EmotionEngine:
    """
    Classifies emotion with human-like temporal stability.

    Three mechanisms prevent frame-to-frame noise:
      1. Per-state EMA (different smoothing time-constants per emotion).
      2. Inertia bonus keeps current state competitive against brief dips.
      3. Per-state commit threshold — must dominate for N frames before switching.

    Unrealistic transitions are dampened by a penalty matrix applied to
    raw scores before EMA.  Facial energy gates Angry/Stressed so that a
    low-activation face (tired, passive, sad) never triggers those states.
    """

    # How fast each state reacts — lower α = more inertia, slower change.
    _EMA_ALPHA: Dict[str, float] = {
        "Neutral":    0.12,
        "Happy":      0.10,
        "Sad":        0.06,   # heavy emotion — slow to enter and leave
        "Angry":      0.08,
        "Stressed":   0.09,
        "Tired":      0.07,
        "Surprised":  0.18,   # fast-onset
        "Focused":    0.09,
        "Distracted": 0.11,
    }

    # Consecutive dominant frames required before committing to a state.
    _MIN_FRAMES: Dict[str, int] = {
        "Neutral":    15,
        "Happy":      15,
        "Sad":        18,   # was 30 — reduced so deep model can commit faster
        "Angry":      22,   # was 35
        "Stressed":   20,
        "Tired":      22,
        "Surprised":   6,
        "Focused":    18,
        "Distracted": 18,
    }

    # Transition penalty matrix.
    # {FROM: {TO: penalty ∈ [0,1]}} — multiplied into the raw TO score.
    # 0.0 = free;  1.0 = fully blocked.
    _TRANSITION_COST: Dict[str, Dict[str, float]] = {
        "Happy": {
            "Angry": 0.55,      # joy → rage requires extreme provocation
            "Sad":   0.40,      # mood whiplash
        },
        "Sad": {
            "Happy":     0.45,  # sadness does not instantly flip to joy
            "Angry":     0.35,
            "Surprised": 0.30,
        },
        "Angry": {
            "Happy":     0.45,
            "Surprised": 0.30,
        },
        "Tired": {
            "Angry":     0.55,  # exhausted face cannot suddenly be angry
            "Happy":     0.25,
            "Surprised": 0.40,
        },
        "Focused": {
            "Angry": 0.35,
        },
    }

    CONFIDENCE_THRESHOLD = 0.18   # minimum normalised share to commit
    WARM_UP_FRAMES       = 60     # ~2 s calibration before is_stable
    INERTIA_BONUS        = 0.05   # reduced so strong DL signal can overcome current state
    TIMELINE_LEN         = 1800   # 60 s at 30 fps
    TREND_WINDOW         = 900    # 30 s window for dominant trend

    def __init__(self):
        k = len(EMOTION_STATES)
        self._ema: Dict[str, float] = {s: 1.0 / k for s in EMOTION_STATES}
        self._ema["Neutral"] += 0.30  # start with Neutral bias

        self._current   = "Neutral"
        self._candidate = "Neutral"
        self._candidate_frames  = 0
        self._frames_in_current = 0   # frames since last committed state change
        self._prev_state = "Neutral"
        self._frame_count = 0

        self._timeline: deque = deque(maxlen=self.TIMELINE_LEN)

    # ── Public interface ─────────────────────────────────────────────

    # States where the deep model contributes; Tired/Focused/Distracted are rule-only.
    _DEEP_BLEND_STATES = frozenset(
        {"Neutral", "Happy", "Sad", "Angry", "Stressed", "Surprised"}
    )
    # Per-state deep-model blend weight. Sad/Angry are hardest to detect from
    # geometry alone so we trust the DL model more for them.
    _DEEP_ALPHA: Dict[str, float] = {
        "Neutral":   0.30,
        "Happy":     0.50,
        "Sad":       0.70,
        "Angry":     0.65,
        "Stressed":  0.45,
        "Surprised": 0.50,
    }
    # Minimum raw score floor when the deep model sees the emotion.
    # Sad/Angry use a lower threshold so subtle expressions are caught.
    _DEEP_FLOOR_THRESHOLD: Dict[str, float] = {
        "Sad":       0.12,
        "Angry":     0.15,
        "Happy":     0.20,
        "Stressed":  0.22,
        "Neutral":   0.25,
        "Surprised": 0.20,
    }
    _DEEP_FLOOR_VALUE = 0.20

    def update(
        self,
        features: FaceFeatures,
        fatigue_score: float,
        stress_score: float,
        bpm: Optional[float],
        blink_rate: float,
        perclos: float,
        deep_scores: Optional[Dict[str, float]] = None,
    ) -> EmotionResult:

        self._frame_count += 1

        if not features.valid:
            return EmotionResult(
                state="No Face", emoji="❌", color="#374151",
                is_calibrating=self._frame_count < self.WARM_UP_FRAMES,
            )

        # 1. Raw evidence scores
        raw = self._score(features, fatigue_score, stress_score,
                          bpm, blink_rate, perclos)

        # 1b. Blend with deep-model prior for covered states
        if deep_scores is not None:
            for s in self._DEEP_BLEND_STATES:
                a = self._DEEP_ALPHA[s]
                d = deep_scores.get(s, 0.0)
                raw[s] = a * d + (1.0 - a) * raw[s]
                # Floor: if DL model meets per-state threshold, guarantee a minimum
                thresh = self._DEEP_FLOOR_THRESHOLD.get(s, 0.25)
                if d >= thresh:
                    raw[s] = max(raw[s], self._DEEP_FLOOR_VALUE)

        # 1c. Strong facial-expression override.
        # If clear sadness landmarks fire (frown + ibrow both high), boost Sad
        # and cap competing non-emotional states (Distracted, Focused, Neutral).
        sad_signal = features.frown_score * 0.5 + features.inner_brow_raise_score * 0.5
        if sad_signal > 0.20:
            boost = min(0.40, sad_signal * 0.70)
            raw["Sad"] = max(raw["Sad"], boost)
            # Attentive gaze (Focused) should not override genuine sadness
            raw["Focused"]    = min(raw["Focused"],    raw["Sad"] * 0.75)
            raw["Distracted"] = min(raw["Distracted"], raw["Sad"] * 0.60)
            raw["Neutral"]    = min(raw["Neutral"],    raw["Sad"] * 0.80)

        # 2. Transition penalty (dampens unrealistic jumps)
        penalized = self._penalize(raw, self._current)

        # 3. Per-state EMA smoothing
        for s in EMOTION_STATES:
            a = self._EMA_ALPHA[s]
            self._ema[s] = a * penalized[s] + (1.0 - a) * self._ema[s]

        # 4. Normalise → probability-like distribution
        total = sum(self._ema.values()) + 1e-9
        norm  = {s: self._ema[s] / total for s in EMOTION_STATES}

        # 5. Inertia: current state gets a bonus so brief dips don't trigger switch
        biased = dict(norm)
        biased[self._current] = min(1.0, biased[self._current] + self.INERTIA_BONUS)
        total2  = sum(biased.values()) + 1e-9
        biased  = {s: v / total2 for s, v in biased.items()}

        # 6. State persistence with per-state thresholds
        best = max(biased, key=biased.get)
        if best == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate   = best
            self._candidate_frames = 1

        min_f = self._MIN_FRAMES.get(best, 25)
        if (self._candidate_frames >= min_f
                and norm[best] >= self.CONFIDENCE_THRESHOLD):
            if self._current != best:
                self._prev_state        = self._current
                self._current           = best
                self._frames_in_current = 0
            else:
                self._frames_in_current += 1
        else:
            if self._current == best:
                self._frames_in_current += 1

        # 7. Timeline memory
        self._timeline.append(self._current)

        # 8. Derived metrics
        stability = float(np.clip(self._frames_in_current / 90.0, 0.0, 1.0))
        persist_s = round(self._frames_in_current / 30.0, 1)

        recent = list(self._timeline)[-self.TREND_WINDOW:]
        dominant = Counter(recent).most_common(1)[0][0] if len(recent) >= 30 else self._current

        is_cal   = self._frame_count < self.WARM_UP_FRAMES
        is_stable = not is_cal

        return EmotionResult(
            state=self._current,
            emoji=EMOTION_EMOJI.get(self._current, "😐"),
            color=EMOTION_COLOR.get(self._current, "#9CA3AF"),
            confidence=round(norm[self._current], 3),
            scores={s: round(v, 3) for s, v in norm.items()},
            is_stable=is_stable,
            smile_score=round(features.smile_score, 3),
            furrow_score=round(features.brow_furrow_score, 3),
            frown_score=round(features.frown_score, 3),
            inner_brow_raise=round(features.inner_brow_raise_score, 3),
            facial_energy=round(features.facial_energy, 3),
            stability_score=round(stability, 3),
            persistence_seconds=persist_s,
            is_calibrating=is_cal,
            timeline_dominant=dominant,
        )

    # ── Transition penalty ───────────────────────────────────────────

    def _penalize(self, raw: Dict[str, float], from_state: str) -> Dict[str, float]:
        costs = self._TRANSITION_COST.get(from_state, {})
        if not costs:
            return raw
        return {s: v * (1.0 - costs.get(s, 0.0)) for s, v in raw.items()}

    # ── Evidence scoring ─────────────────────────────────────────────

    def _score(
        self,
        f: FaceFeatures,
        fatigue: float,
        stress: float,
        bpm: Optional[float],
        blink_rate: float,
        perclos: float,
    ) -> Dict[str, float]:

        fat = fatigue / 100.0
        str_ = stress / 100.0
        per  = perclos / 100.0

        # HR features
        if bpm and 40.0 <= bpm <= 180.0:
            hr_elev  = _clip((bpm - 85.0) / 35.0)
            hr_relax = _clip(1.0 - abs(bpm - 70.0) / 25.0)
        else:
            hr_elev  = 0.0
            hr_relax = 0.5

        # Blink signals
        blink_low  = _clip((15.0 - blink_rate) / 12.0)
        blink_high = _clip((blink_rate - 22.0) / 15.0)

        # Eye openness
        ear_low   = _clip((0.28 - f.ear) / 0.10)
        ear_alert = _clip((f.ear - 0.24) / 0.08)

        # ── Facial energy gates ───────────────────────────────────────
        # A low-energy (passive/tired/sad) face cannot fire Angry or Stressed.
        # sqrt gives a gentler ramp: energy=0.25 → gate≈0.84 (not hard cut-off).
        energy_gate = float(np.sqrt(_clip(f.facial_energy / 0.35)))
        low_energy  = _clip(1.0 - f.facial_energy)

        scores: Dict[str, float] = {}

        # ── Happy ─────────────────────────────────────────────────────
        scores["Happy"] = _w(
            (0.70, f.smile_score),
            (0.15, 1.0 - f.brow_furrow_score),
            (0.10, f.mouth_width_norm * 0.6),
            (0.05, hr_relax * (1.0 - fat * 0.3)),
        )

        # ── Sad ───────────────────────────────────────────────────────
        # Low facial energy + AU1 inner brow raise are the primary cues.
        # Downward head pitch (looking down) is a natural sadness posture cue.
        head_down = _clip(-f.head_pitch / 20.0)   # positive when looking down
        scores["Sad"] = _w(
            (0.40, f.frown_score),
            (0.28, f.inner_brow_raise_score),
            (0.15, head_down * 0.7),
            (0.10, f.brow_raise_score * (1.0 - f.brow_furrow_score * 0.5)),
            (0.07, low_energy * 0.55),
        )
        # Only suppress Sad once smile is clearly genuine (> 0.25 threshold).
        scores["Sad"] = scores["Sad"] * _clip(1.0 - max(0.0, f.smile_score - 0.25) * 3.5)

        # ── Angry ─────────────────────────────────────────────────────
        # Primary driver is brow furrow.  Gated by facial energy —
        # a tired or passive face cannot be angry.
        angry_base = _w(
            (0.80, f.brow_furrow_score),
            (0.15, f.frown_score * 0.7),
            (0.05, blink_high * 0.4),
        )
        scores["Angry"] = angry_base * _clip(1.0 - f.smile_score * 4.0) * energy_gate

        # Sad > Angry protection: medium Sad blocks weak Angry
        if scores["Sad"] > 0.18 and scores["Sad"] >= scores["Angry"]:
            scores["Angry"] = min(scores["Angry"], scores["Sad"] * 0.40)

        # ── Stressed ──────────────────────────────────────────────────
        # Physiological stress + brow tension.  Gated by energy.
        scores["Stressed"] = _w(
            (0.35, str_),
            (0.30, f.brow_furrow_score),
            (0.20, f.head_movement_score),
            (0.15, blink_high),
        )
        scores["Stressed"] = (scores["Stressed"]
                               * _clip(1.0 - f.smile_score * 2.0)
                               * energy_gate)

        # ── Tired ─────────────────────────────────────────────────────
        scores["Tired"] = _w(
            (0.35, fat),
            (0.30, ear_low),
            (0.20, f.mouth_open_ratio * 0.65),
            (0.15, per),
        )

        # ── Surprised ─────────────────────────────────────────────────
        scores["Surprised"] = _w(
            (0.45, f.eye_wide_score),
            (0.35, f.mouth_open_ratio),
            (0.20, f.brow_raise_score),
        )

        # ── Focused ───────────────────────────────────────────────────
        gaze_fwd = 1.0 - f.head_yaw_norm
        scores["Focused"] = _w(
            (0.28, 1.0 - f.head_movement_score),
            (0.25, gaze_fwd),
            (0.22, ear_alert),
            (0.15, 1.0 - fat * 0.7),
            (0.10, blink_low * 0.5),
        )

        # ── Distracted ────────────────────────────────────────────────
        # Movement-driven, not pose-driven. A still angled head (looking
        # down when sad, or just resting) should NOT score as Distracted.
        scores["Distracted"] = _w(
            (0.20, f.head_yaw_norm),        # reduced: tilt alone ≠ distracted
            (0.55, f.head_movement_score),  # primary signal: active fidgeting
            (0.25, fat * 0.55),
        )

        # ── Neutral ───────────────────────────────────────────────────
        # Raised floor so a resting face settles here, not Angry/Stressed.
        max_other = max(v for v in scores.values())
        scores["Neutral"] = float(np.clip(0.50 - max_other * 0.65, 0.08, 0.50))

        return scores


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _w(*pairs) -> float:
    return float(np.clip(sum(w * v for w, v in pairs), 0.0, 1.0))
