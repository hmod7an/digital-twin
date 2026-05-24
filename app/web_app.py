"""
Face Health Digital Twin — Professional AI Dashboard v3

Redesigned to closely match the desktop (PyQt5) application:
  - Same color palette (#0B0F1A / #141925 / #1E2A3A)
  - Same per-vital accent colors (HR=#FF5E7A, Fatigue=#FF8C00, etc.)
  - Same color thresholds (score_color, bpm_color, wellbeing_color)
  - Larger value typography (36–40 px / 800 weight)
  - PERCLOS added to fatigue sub-label
  - HRV + head-motion added to stress sub-label
  - Emotion card debug rows (smile/frown/furrow/ibrow scores)
  - AI row layout: Attention + Wellbeing + Insights (matches desktop row 2)
  - Facial signal analysis panel (separate, like desktop sidebar)
  - Risk banner attached to vital-cards section

Launch:
    python run_web.py                  # local  → http://localhost:7860
    python run_web.py --share          # public HTTPS tunnel (mobile camera)
    python run_web.py --share --auth   # with login
"""
import sys
import os
import cv2
import numpy as np
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr

from core.face_tracker import FaceTracker
from ai.feature_extractor import FeatureExtractor, FaceFeatures
from ai.deep_emotion_model import DeepEmotionModel
from vitals.rppg import RPPGEstimator
from vitals.fatigue import FatigueDetector
from vitals.stress import StressEstimator
from vitals.breathing import BreathingEstimator
from vitals.attention import AttentionEstimator
from ai.emotion_engine import EmotionEngine, EMOTION_COLOR, EMOTION_STATES
from ai.state_fusion import StateFusion, MentalStateResult
from prediction.health_risk import HealthRiskPredictor

# ── Colour palette — exact match to desktop gui/styles.py ────────────────────
_DARK    = "#0B0F1A"   # QMainWindow / QWidget background
_PANEL   = "#141925"   # QFrame#card background-color
_HEADER  = "#0F1520"   # QFrame#header_card / insights_card
_BORDER  = "#1E2A3A"   # card border color
_CYAN    = "#00D4FF"
_GREEN   = "#00FF88"
_YELLOW  = "#FFD700"
_RED     = "#FF4B4B"
_ORANGE  = "#FF8C00"
_GRAY    = "#6B7280"
_GRAY2   = "#64748B"   # QLabel#section_header / metric_label color
_PURPLE  = "#A78BFA"
_WHITE   = "#E2E8F0"   # global text color

# Fixed per-vital accent colors — matches desktop _BaseCard(accent_color=…)
_HR_ACCENT      = "#FF5E7A"
_FATIGUE_ACCENT = "#FF8C00"
_STRESS_ACCENT  = "#FF4B4B"
_BREATH_ACCENT  = "#00FF88"
_ATTN_ACCENT    = "#00D4FF"
_WB_ACCENT      = "#00FF88"

_EMOTION_EMOJI = {
    "Neutral": "😐", "Happy": "😊", "Sad": "😢", "Angry": "😠",
    "Stressed": "😤", "Tired": "😴", "Surprised": "😲",
    "Focused": "🎯", "Distracted": "😵",
}


# ── Color helpers — exact match to desktop gui/styles.py ─────────────────────

def _score_color(score: float) -> str:
    """Fatigue & stress coloring — matches desktop score_color()."""
    if score < 35:   return _GREEN
    elif score < 60: return _YELLOW
    return _RED


def _bpm_color(bpm: float) -> str:
    """Matches desktop bpm_color()."""
    if 50 <= bpm <= 100:   return _GREEN
    elif 40 <= bpm <= 120: return _YELLOW
    return _RED


def _attention_color(score: float) -> str:
    """Matches desktop attention_color()."""
    if score >= 65:   return _CYAN
    elif score >= 44: return _GREEN
    elif score >= 26: return _YELLOW
    return _RED


def _wellbeing_color(score: float) -> str:
    """Matches desktop wellbeing_color()."""
    if score >= 72:   return _GREEN
    elif score >= 52: return _YELLOW
    return _RED


# ─────────────────────────────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────────────────────────────

def _new_session() -> dict:
    return {
        "tracker":        FaceTracker(),
        "features":       FeatureExtractor(),
        "deep_emo":       DeepEmotionModel(),
        "rppg":           RPPGEstimator(),
        "fatigue":        FatigueDetector(),
        "stress":         StressEstimator(),
        "breathing":      BreathingEstimator(),
        "emotion":        EmotionEngine(),
        "attention":      AttentionEstimator(),
        "risk":           HealthRiskPredictor(),
        "fusion":         StateFusion(),
        "bpm_hist":       deque([0.0] * 60, maxlen=60),
        "fat_hist":       deque([0.0] * 60, maxlen=60),
        "str_hist":       deque([0.0] * 60, maxlen=60),
        "wb_hist":        deque([75.0] * 60, maxlen=60),
        "emo_label_hist": deque(maxlen=60),
        "score_hist":     deque(maxlen=50),
        "tick":           0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main frame processor
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray, session: dict):
    if session is None:
        session = _new_session()
    if frame is None:
        return _empty_outputs(session)

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    landmarks  = session["tracker"].process(bgr)
    face_feats = session["features"].extract(landmarks)

    rppg_r    = session["rppg"].update(landmarks)
    fatigue_r = session["fatigue"].update(landmarks)
    stress_r  = session["stress"].update(landmarks, rppg_r.bpm, fatigue_r.blink_rate)
    breath_r  = session["breathing"].update(
        landmarks, rppg_r.signal[-1] if len(rppg_r.signal) > 0 else None
    )

    deep_scores = None
    deep_emo = session["deep_emo"]
    if landmarks is not None and deep_emo.available:
        x, y, wb, hb = landmarks.face_bbox
        if wb > 20 and hb > 20:
            fh, fw = bgr.shape[:2]
            crop = bgr[max(0, y):min(fh, y + hb), max(0, x):min(fw, x + wb)]
            deep_scores = deep_emo.predict(crop)

    emotion_r   = session["emotion"].update(
        face_feats, fatigue_r.score, stress_r.score,
        rppg_r.bpm, fatigue_r.blink_rate, fatigue_r.perclos, deep_scores,
    )
    attention_r = session["attention"].update(
        face_feats, fatigue_r.score, stress_r.score, fatigue_r.blink_rate,
    )
    risk_r   = session["risk"].update(rppg_r.bpm, fatigue_r.score, stress_r.score)
    mental_r = session["fusion"].update(
        emotion_r, attention_r, rppg_r, fatigue_r, stress_r, breath_r, risk_r,
    )

    if landmarks is not None:
        ann_bgr = session["tracker"].draw_landmarks(bgr, landmarks)
    else:
        ann_bgr = bgr
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

    session["tick"] += 1
    tick = session["tick"]
    if tick % 30 == 0:
        session["bpm_hist"].append(rppg_r.bpm or 0.0)
        session["fat_hist"].append(fatigue_r.score)
        session["str_hist"].append(stress_r.score)
        session["wb_hist"].append(mental_r.wellbeing_score)
        session["emo_label_hist"].append(
            (emotion_r.state, EMOTION_COLOR.get(emotion_r.state, _GRAY))
        )
    if tick % 3 == 0 and emotion_r.scores:
        session["score_hist"].append(dict(emotion_r.scores))

    face_ok = landmarks is not None

    system_html   = _system_html(face_ok, deep_emo.available, emotion_r.is_calibrating, tick)
    vital_html    = _vital_html(rppg_r, fatigue_r, stress_r, breath_r, risk_r)
    emotion_html  = _emotion_main_html(emotion_r)
    emo_bars_html = _emotion_bars_html(emotion_r)
    # secondary_out = AI row (Attention + Wellbeing + Insights)
    secondary_html = _ai_row_html(attention_r, mental_r)
    # wellbeing_out = Facial signal analysis panel
    wellbeing_html = _facial_signals_html(face_feats)

    rppg_fig  = _rppg_figure(rppg_r.signal, rppg_r.bpm)
    trend_fig = _trend_figure(session)
    emo_fig   = _emotion_timeline_figure(session)

    return (
        ann_rgb,
        system_html, vital_html,
        emotion_html, emo_bars_html,
        secondary_html, wellbeing_html,
        rppg_fig, trend_fig, emo_fig,
        session,
    )


def _empty_outputs(session):
    ph = (
        f'<div style="color:{_GRAY2};padding:16px;background:{_PANEL};'
        f'border:1px solid {_BORDER};border-radius:12px;font-family:monospace;">'
        f'Waiting for camera feed…</div>'
    )
    empty_fig = go.Figure().update_layout(
        paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
        font=dict(color=_WHITE), height=200,
        margin=dict(l=30, r=10, t=30, b=10),
    )
    return (
        None,
        _system_html(False, False, True, 0),
        ph, ph, ph, ph, ph,
        empty_fig, empty_fig, empty_fig,
        session,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(content: str, border_color: str = None, accent: str = None) -> str:
    """Base card — matches desktop QFrame#card style."""
    bc  = border_color or _BORDER
    top = f"border-top:4px solid {accent};" if accent else ""
    return (
        f'<div style="background:{_PANEL};border:1px solid {bc};{top}'
        f'border-radius:12px;padding:14px 16px;margin-bottom:8px;">'
        f'{content}</div>'
    )


def _bar(value: float, color: str, height: int = 8) -> str:
    """Progress bar matching desktop QProgressBar style."""
    pct = int(max(0, min(100, value * 100)))
    return (
        f'<div style="background:#1A2030;border-radius:5px;height:{height}px;'
        f'overflow:hidden;margin-top:3px;">'
        f'<div style="background:{color};width:{pct}%;height:{height}px;'
        f'border-radius:5px;transition:width 0.3s ease;"></div></div>'
    )


def _section_hdr(icon: str, title: str) -> str:
    """10px / 700 / #64748B / letter-spacing 2px — matches QLabel#section_header."""
    return (
        f'<div style="font-size:10px;font-weight:700;color:{_GRAY2};'
        f'letter-spacing:2px;margin-bottom:6px;">{icon} {title}</div>'
    )


def _big_value(val: str, unit: str, color: str) -> str:
    """40px / 800 weight value + 14px unit — matches QLabel#metric_value."""
    return (
        f'<div style="display:flex;align-items:baseline;gap:4px;margin:4px 0 2px;">'
        f'<span style="font-size:38px;font-weight:800;color:{color};'
        f'letter-spacing:-1px;line-height:1;">{val}</span>'
        f'<span style="font-size:14px;color:#94A3B8;font-weight:600;">{unit}</span>'
        f'</div>'
    )


def _sub(text: str) -> str:
    return f'<div style="font-size:11px;color:{_GRAY2};margin-bottom:4px;">{text}</div>'


# ── System / header bar ───────────────────────────────────────────────────────

def _system_html(face_ok: bool, deep_ok: bool, calibrating: bool, tick: int) -> str:
    face_dot = f'<span style="color:{_GREEN};">●</span>' if face_ok else f'<span style="color:{_RED};">●</span>'
    deep_dot = f'<span style="color:{_CYAN};">●</span>' if deep_ok  else f'<span style="color:{_GRAY};">●</span>'
    cal_dot  = f'<span style="color:{_YELLOW};">●</span>' if calibrating else f'<span style="color:{_GREEN};">●</span>'
    live_col = _GREEN if face_ok else _RED
    live_lbl = "🟢  LIVE" if face_ok else "🔴  NO FACE"

    return (
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:12px;'
        f'background:{_HEADER};border:1px solid {_BORDER};border-radius:12px;'
        f'padding:10px 18px;font-family:\'Segoe UI\',Consolas,monospace;">'
        # Title block
        f'<div style="display:flex;flex-direction:column;min-width:280px;">'
        f'<span style="color:{_CYAN};font-weight:800;font-size:16px;letter-spacing:1px;">'
        f'🩺  Face Health Digital Twin</span>'
        f'<span style="color:{_GRAY2};font-size:10px;letter-spacing:0.5px;margin-top:1px;">'
        f'Real-time AI health monitoring  |  rPPG · Fatigue · Stress · Emotion · Attention · Wellbeing'
        f'</span></div>'
        # Status indicators
        f'<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-left:auto;">'
        f'<span style="color:#9CA3AF;font-size:12px;">{face_dot} {"FACE DETECTED" if face_ok else "NO FACE"}</span>'
        f'<span style="color:#9CA3AF;font-size:12px;">{deep_dot} {"DEEP AI" if deep_ok else "DEEP AI OFFLINE"}</span>'
        f'<span style="color:#9CA3AF;font-size:12px;">{cal_dot} {"CALIBRATING" if calibrating else "SYSTEM READY"}</span>'
        f'<span style="color:#4B5563;font-size:11px;">#{tick:,}</span>'
        f'<span style="font-size:13px;font-weight:bold;color:{live_col};">{live_lbl}</span>'
        f'</div>'
        f'</div>'
    )


# ── Vital cards + Risk banner ─────────────────────────────────────────────────

def _vital_html(rppg_r, fatigue_r, stress_r, breath_r, risk_r) -> str:

    def _vcard(icon, title, val, unit, val_color, accent, sub, fill):
        return (
            f'<div style="flex:1;min-width:155px;background:{_PANEL};'
            f'border:1px solid {_BORDER};border-top:4px solid {accent};'
            f'border-radius:12px;padding:12px 14px 14px;">'
            + _section_hdr(icon, title)
            + _big_value(val, unit, val_color)
            + _sub(sub)
            + _bar(fill, val_color, 8)
            + f'</div>'
        )

    # Heart rate
    if rppg_r.bpm:
        bpm_val  = f"{rppg_r.bpm:.0f}"
        bpm_col  = _bpm_color(rppg_r.bpm)
        bpm_sub  = f"Quality: {rppg_r.confidence*100:.0f}%"
        bpm_fill = max(0.0, min(1.0, (rppg_r.bpm - 40) / 100.0))
    else:
        bpm_val  = "--"
        bpm_col  = _GRAY
        bpm_sub  = f"Calibrating… {rppg_r.buffer_fill*100:.0f}%"
        bpm_fill = rppg_r.buffer_fill

    # Fatigue — matches desktop FatigueCard.refresh sub-label
    fat_col = _score_color(fatigue_r.score)
    fat_sub = (
        f"{fatigue_r.state}  ·  EAR {fatigue_r.ear:.3f}"
        f"  ·  PERCLOS {fatigue_r.perclos:.1f}%"
    )

    # Stress — matches desktop StressCard.refresh sub-label
    str_col = _score_color(stress_r.score)
    str_sub = (
        f"{stress_r.state}  ·  HRV {stress_r.hr_variability:.1f}"
        f"  ·  Motion {stress_r.head_movement:.2f}"
    )

    # Breathing
    if breath_r.rate_bpm and breath_r.confidence > 0.2:
        br_val  = f"{breath_r.rate_bpm:.0f}"
        br_col  = _GREEN if 12 <= breath_r.rate_bpm <= 20 else _YELLOW
        br_sub  = f"Confidence: {breath_r.confidence*100:.0f}%"
        br_fill = min(1.0, breath_r.rate_bpm / 25.0)
    else:
        br_val  = "--"
        br_col  = _GRAY
        br_sub  = "Estimating breathing pattern…"
        br_fill = 0.0

    cards = "".join([
        _vcard("❤️", "HEART RATE", bpm_val, "BPM",  bpm_col, _HR_ACCENT,      bpm_sub, bpm_fill),
        _vcard("😴", "FATIGUE",    f"{fatigue_r.score:.0f}", "%", fat_col, _FATIGUE_ACCENT, fat_sub, fatigue_r.score / 100),
        _vcard("🧠", "STRESS",     f"{stress_r.score:.0f}",  "%", str_col, _STRESS_ACCENT,  str_sub, stress_r.score / 100),
        _vcard("🫁", "BREATHING",  br_val, "/min",  br_col, _BREATH_ACCENT,   br_sub, br_fill),
    ])

    # Risk banner — matches desktop RiskBanner
    _risk_colors  = {0: _GREEN,  1: _YELLOW,  2: _RED}
    _risk_headings = {
        0: "✅  SYSTEM STATUS: NORMAL",
        1: "⚠️  STATUS: WARNING",
        2: "🚨  STATUS: HIGH RISK",
    }
    rc  = _risk_colors.get(risk_r.level_code, _GRAY)
    hdg = _risk_headings.get(risk_r.level_code, "✅  SYSTEM STATUS: NORMAL")
    msg = "  ·  ".join(risk_r.messages[:3]) if risk_r.messages else "All vitals in healthy range"

    risk_banner = (
        f'<div style="background:{rc}12;border:2px solid {rc}44;'
        f'border-radius:12px;padding:10px 16px;margin-top:6px;">'
        f'<div style="font-size:14px;font-weight:bold;color:{rc};">{hdg}</div>'
        f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">{msg}</div>'
        f'</div>'
    )

    return (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{cards}</div>'
        + risk_banner
    )


# ── Emotion main card ─────────────────────────────────────────────────────────

def _emotion_main_html(er) -> str:
    color = EMOTION_COLOR.get(er.state, _GRAY)
    stab  = int(er.stability_score * 100)

    # Persist time — matches desktop EmotionCard format
    pers_s = er.persistence_seconds
    if er.is_calibrating:
        persist_str = "⏳ Calibrating…"
    else:
        mins = int(pers_s // 60)
        secs = int(pers_s % 60)
        t    = f"{mins}m {secs}s" if mins else f"{secs}s"
        persist_str = f"⏱ {t} stable  ·  {stab}% certainty"

    cal_tag = (
        f'<span style="background:#1E293B;color:{_YELLOW};font-size:10px;'
        f'padding:2px 7px;border-radius:4px;font-weight:bold;">CALIBRATING</span>'
        if er.is_calibrating else
        f'<span style="background:{color}22;color:{color};font-size:10px;'
        f'padding:2px 7px;border-radius:4px;font-weight:bold;">STABLE</span>'
        if er.is_stable else ""
    )

    dom_color = EMOTION_COLOR.get(er.timeline_dominant, _GRAY)
    dom_emoji = _EMOTION_EMOJI.get(er.timeline_dominant, "😐")

    # Debug rows — matches desktop EmotionCard dev mode
    debug_rows = ""
    if er.scores:
        h  = er.scores.get("Happy",   0.0) * 100
        s  = er.scores.get("Sad",     0.0) * 100
        an = er.scores.get("Angry",   0.0) * 100
        n  = er.scores.get("Neutral", 0.0) * 100
        debug_rows = (
            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid {_BORDER};">'
            f'<div style="font-size:10px;color:#475569;line-height:1.6;">'
            f'smile:{er.smile_score:.2f}  frown:{er.frown_score:.2f}  '
            f'furrow:{er.furrow_score:.2f}  ibrow:{er.inner_brow_raise:.2f}  '
            f'nrg:{er.facial_energy:.2f}'
            f'</div>'
            f'<div style="font-size:10px;color:#475569;">'
            f'H:{h:.0f}%  S:{s:.0f}%  An:{an:.0f}%  N:{n:.0f}%'
            f'</div>'
            f'<div style="font-size:10px;color:#334155;margin-top:2px;">30s trend → {er.timeline_dominant}</div>'
            f'</div>'
        )

    content = (
        _section_hdr("🧬", "EMOTIONAL STATE") +
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
        f'<span style="font-size:44px;line-height:1;">{er.emoji}</span>'
        f'<div style="flex:1;">'
        f'<div style="font-size:22px;font-weight:700;color:{color};">{er.state}</div>'
        f'<div style="font-size:11px;color:{_GRAY2};margin:2px 0;">{persist_str}</div>'
        f'{cal_tag}'
        f'</div></div>'
        # Confidence bar — purple like desktop QProgressBar#emotion_conf
        f'<div style="font-size:10px;color:{_GRAY2};margin-bottom:2px;">Confidence</div>'
        f'<div style="background:#1A2030;border-radius:5px;height:8px;overflow:hidden;">'
        f'<div style="background:{_PURPLE};width:{int(er.confidence*100)}%;height:8px;'
        f'border-radius:5px;transition:width 0.3s;"></div></div>'
        f'<div style="font-size:11px;color:{_GRAY2};margin:3px 0 6px;">'
        f'Confidence: {er.confidence*100:.0f}%</div>'
        # Stability bar — cyan like desktop
        f'<div style="font-size:10px;color:{_GRAY2};margin-bottom:2px;">Stability</div>'
        f'<div style="background:#1A2030;border-radius:5px;height:6px;overflow:hidden;margin-bottom:10px;">'
        f'<div style="background:{_CYAN}88;width:{stab}%;height:6px;'
        f'border-radius:5px;transition:width 0.3s;"></div></div>'
        # 30s dominant
        f'<div style="padding:8px 0;border-top:1px solid {_BORDER};">'
        f'<div style="font-size:10px;font-weight:700;color:{_GRAY2};letter-spacing:1px;margin-bottom:4px;">30s DOMINANT</div>'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="font-size:18px;">{dom_emoji}</span>'
        f'<span style="color:{dom_color};font-weight:bold;">{er.timeline_dominant}</span>'
        f'</div></div>'
        + debug_rows
    )
    return _card(content, border_color=color + "66", accent=color)


# ── Emotion confidence bars ───────────────────────────────────────────────────

def _emotion_bars_html(er) -> str:
    scores  = er.scores or {}
    ordered = sorted(EMOTION_STATES, key=lambda s: scores.get(s, 0.0), reverse=True)
    rows = []
    for state in ordered:
        pct    = scores.get(state, 0.0) * 100
        color  = EMOTION_COLOR.get(state, _GRAY)
        emoji  = _EMOTION_EMOJI.get(state, "")
        active = state == er.state
        bold   = "bold"   if active else "normal"
        glow   = f"text-shadow:0 0 8px {color};" if active else ""
        rows.append(
            f'<div style="margin-bottom:7px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
            f'<span style="font-size:12px;font-weight:{bold};'
            f'color:{"#E2E8F0" if active else _GRAY2};{glow}">{emoji} {state}</span>'
            f'<span style="font-size:11px;color:{color if active else "#4B5563"};font-weight:{bold};">'
            f'{pct:.0f}%</span></div>'
            f'<div style="background:#1A2030;border-radius:3px;height:6px;overflow:hidden;">'
            f'<div style="background:{color};width:{pct:.1f}%;height:6px;border-radius:3px;'
            f'transition:width 0.4s ease;opacity:{"1" if active else "0.5"};"></div>'
            f'</div></div>'
        )
    header = (
        f'<div style="font-size:10px;font-weight:700;color:{_CYAN};letter-spacing:2px;'
        f'margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid {_BORDER};">'
        f'🎭 EMOTION CONFIDENCE SCORES</div>'
    )
    return _card(header + "".join(rows))


# ── AI row: Attention + Wellbeing + Insights ──────────────────────────────────

def _ai_row_html(attention_r, mental_r: MentalStateResult) -> str:
    """
    Matches desktop layout row 2:
      EmotionCard | AttentionCard | WellbeingCard | InsightsPanel
    Here we render AttentionCard + WellbeingCard + InsightsPanel.
    """

    # Attention card — matches desktop AttentionCard
    att_col = _attention_color(attention_r.score)
    att_card = (
        f'<div style="flex:1;min-width:160px;background:{_PANEL};'
        f'border:1px solid {_BORDER};border-top:4px solid {_ATTN_ACCENT};'
        f'border-radius:12px;padding:12px 14px 14px;">'
        + _section_hdr("🎯", "ATTENTION")
        + _big_value(f"{attention_r.score:.0f}", "%", att_col)
        + _sub(f"{attention_r.level}  ·  Cog. Load {attention_r.cognitive_load:.0f}%")
        + _bar(attention_r.score / 100, att_col, 8)
        # Sub-bars matching desktop AttentionCard cog/gaze/stability rows
        + f'<div style="margin-top:8px;">'
        + f'<div style="font-size:10px;color:{_GRAY2};margin-bottom:2px;">Cog. Load</div>'
        + _bar(attention_r.cognitive_load / 100, _ORANGE, 5)
        + f'<div style="font-size:10px;color:{_GRAY2};margin-top:5px;margin-bottom:2px;">Gaze Quality</div>'
        + _bar(attention_r.gaze_score / 100, _CYAN, 5)
        + f'<div style="font-size:10px;color:{_GRAY2};margin-top:5px;margin-bottom:2px;">Head Stability</div>'
        + _bar(attention_r.stability_score / 100, _GREEN, 5)
        + f'</div>'
        + f'</div>'
    )

    # Wellbeing card — matches desktop WellbeingCard
    wb_col = _wellbeing_color(mental_r.wellbeing_score)
    wb_card = (
        f'<div style="flex:1;min-width:160px;background:{_PANEL};'
        f'border:1px solid {_BORDER};border-top:4px solid {_WB_ACCENT};'
        f'border-radius:12px;padding:12px 14px 14px;">'
        + _section_hdr("🌡️", "WELLBEING")
        + _big_value(f"{mental_r.wellbeing_score:.0f}", "%", wb_col)
        + _sub(f"{mental_r.wellbeing_label}  {mental_r.trend_icon} {mental_r.trend}")
        + _bar(mental_r.wellbeing_score / 100, wb_col, 8)
        + f'</div>'
    )

    # Insights panel — matches desktop InsightsPanel
    ins_lines = "".join(
        f'<div style="padding:3px 0;font-size:12px;color:#CBD5E1;'
        f'border-bottom:1px solid #1E293B;">▸  {ln}</div>'
        for ln in mental_r.insights
    ) or f'<div style="font-size:12px;color:{_GRAY2};padding:4px 0;">Collecting data…</div>'

    rec_lines = "".join(
        f'<div style="padding:2px 0;font-size:11px;color:{_GRAY2};">→  {ln}</div>'
        for ln in mental_r.recommendations
    )

    insights_card = (
        f'<div style="flex:2;min-width:200px;background:{_HEADER};'
        f'border:1px solid #1E3050;border-radius:12px;padding:12px 14px 14px;">'
        + f'<div style="font-size:10px;font-weight:700;color:{_CYAN};'
        + f'letter-spacing:2px;margin-bottom:8px;">💡 AI INSIGHTS</div>'
        + ins_lines
        + (f'<div style="margin-top:8px;">'
           f'<div style="font-size:10px;font-weight:700;color:{_GRAY2};margin-bottom:4px;">'
           f'RECOMMENDATIONS</div>'
           f'{rec_lines}</div>' if mental_r.recommendations else '')
        + f'</div>'
    )

    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{att_card}{wb_card}{insights_card}</div>'


# ── Facial signal analysis panel ──────────────────────────────────────────────

def _facial_signals_html(ff: FaceFeatures) -> str:
    header = (
        f'<div style="font-size:10px;font-weight:700;color:#A78BFA;'
        f'letter-spacing:2px;margin-bottom:10px;">📡 FACIAL SIGNAL ANALYSIS</div>'
    )

    if not ff.valid:
        return _card(
            header +
            f'<div style="color:#4B5563;font-size:12px;">'
            f'No face detected — position your face in frame</div>'
        )

    def _sig(label, value, scale=1.0):
        raw   = float(value) * scale
        clamp = min(1.0, max(0.0, raw))
        c = _GREEN if clamp < 0.35 else (_YELLOW if clamp < 0.65 else _RED)
        return (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
            f'<span style="font-size:11px;color:{_GRAY2};">{label}</span>'
            f'<span style="font-size:11px;color:{c};font-weight:bold;">{raw:.3f}</span></div>'
            + _bar(clamp, c, 5) + f'</div>'
        )

    grid = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px;">'
        + _sig("Smile (AU12)",           ff.smile_score)
        + _sig("Frown (corner droop)",   ff.frown_score)
        + _sig("Brow Furrow (AU4)",      ff.brow_furrow_score)
        + _sig("Inner Brow Raise (AU1)", ff.inner_brow_raise_score)
        + _sig("Eye Openness (EAR)",     ff.ear, scale=3.0)
        + _sig("Eye Wide (AU5)",         ff.eye_wide_score)
        + _sig("Mouth Open (MAR)",       ff.mouth_open_ratio)
        + _sig("Head Movement",          ff.head_movement_score)
        + f'</div>'
    )
    pose_row = (
        f'<div style="display:flex;gap:20px;margin-top:8px;font-size:11px;color:{_GRAY2};'
        f'border-top:1px solid {_BORDER};padding-top:6px;">'
        f'<span>Yaw {ff.head_yaw:.1f}°</span>'
        f'<span>Pitch {ff.head_pitch:.1f}°</span>'
        f'<span>Brow Raise {ff.brow_raise_score:.2f}</span>'
        f'<span>Energy {ff.facial_energy:.2f}</span>'
        f'</div>'
    )
    return _card(header + grid + pose_row)


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=_DARK, plot_bgcolor=_PANEL,
        font=dict(color=_WHITE, family="'Segoe UI', Consolas, monospace"),
        margin=dict(l=40, r=16, t=40, b=24),
        **kw,
    )


def _rppg_figure(signal: np.ndarray, bpm) -> go.Figure:
    fig = go.Figure()
    if len(signal) > 5:
        s   = signal[-300:]
        rng = s.max() - s.min()
        if rng > 1e-9:
            s = (s - s.min()) / rng * 2 - 1
        fig.add_trace(go.Scatter(
            y=s, mode="lines",
            line=dict(color=_CYAN, width=1.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ))
    title = f"❤️ rPPG — {bpm:.0f} BPM" if bpm else "❤️ rPPG Signal — calibrating…"
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(color=_CYAN, size=13)),
        height=190, showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1E2A3A", zeroline=True,
                   zerolinecolor=_GRAY, showticklabels=False),
    ))
    return fig


def _trend_figure(session: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.04)
    x   = list(range(len(session["bpm_hist"])))
    bpm = list(session["bpm_hist"])
    fat = list(session["fat_hist"])
    st  = list(session["str_hist"])
    wb  = list(session["wb_hist"])

    fig.add_trace(go.Scatter(x=x, y=bpm, mode="lines",
                             line=dict(color=_CYAN, width=2), name="BPM"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=wb,  mode="lines",
                             line=dict(color=_GREEN, width=1.5, dash="dot"),
                             name="Wellbeing"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=fat, mode="lines",
                             line=dict(color=_ORANGE, width=2), name="Fatigue %",
                             fill="tozeroy", fillcolor="rgba(255,140,0,0.10)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=st,  mode="lines",
                             line=dict(color=_RED, width=2), name="Stress %"), row=2, col=1)

    for row in [1, 2]:
        fig.update_xaxes(showgrid=False, showticklabels=False, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1E2A3A", row=row, col=1)

    fig.update_layout(**_base_layout(
        title=dict(text="📊 Vital Trends (60s)", font=dict(color=_WHITE, size=13)),
        height=270, margin=dict(l=40, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def _emotion_timeline_figure(session: dict) -> go.Figure:
    hist = list(session["score_hist"])
    fig  = go.Figure()

    if hist:
        x   = list(range(len(hist)))
        top = sorted(
            EMOTION_STATES,
            key=lambda s: max((h.get(s, 0.0) for h in hist), default=0.0),
            reverse=True,
        )[:5]
        for state in top:
            ys    = [h.get(state, 0.0) * 100 for h in hist]
            color = EMOTION_COLOR.get(state, _GRAY)
            fig.add_trace(go.Scatter(
                x=x, y=ys, mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy" if state == top[-1] else "none",
                fillcolor=color + "18",
                name=f"{_EMOTION_EMOJI.get(state,'')} {state}",
                hovertemplate=f"{state}: %{{y:.1f}}%<extra></extra>",
            ))

    label_hist = list(session["emo_label_hist"])
    if label_hist and hist:
        xs = [i * (len(hist) / max(1, len(label_hist))) for i in range(len(label_hist))]
        for i, (state, color) in enumerate(label_hist):
            fig.add_trace(go.Scatter(
                x=[xs[i]], y=[-5], mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                showlegend=False,
                hovertemplate=f"{state}<extra></extra>",
            ))

    fig.update_layout(**_base_layout(
        title=dict(text="🎭 Emotion Timeline", font=dict(color=_WHITE, size=13)),
        height=240, margin=dict(l=40, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#1E2A3A",
                   range=[-10, 100], title="Score %",
                   titlefont=dict(size=10, color=_GRAY)),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSS — matches desktop dark theme
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Global ── */
body, .gradio-container {
    background: #0B0F1A !important;
    color: #E2E8F0 !important;
    font-family: "Segoe UI", "Consolas", monospace !important;
    font-size: 13px !important;
}
footer { display: none !important; }

/* ── Markdown headings ── */
.prose h1 { color: #00D4FF !important; font-family: monospace; font-size: 18px !important; }
.prose h2 { color: #9CA3AF !important; font-size: 14px !important; font-weight: 500; }
.prose h3 { color: #64748B !important; font-size: 12px !important; }
.prose p, .prose li { color: #9CA3AF !important; font-size: 12px; }
.prose table { border-collapse: collapse; width: 100%; }
.prose th { background: #141925; color: #00D4FF; font-size: 11px;
            padding: 6px 10px; border: 1px solid #1E2A3A; }
.prose td { font-size: 11px; color: #D1D5DB; padding: 5px 10px; border: 1px solid #1E2A3A; }
.prose tr:nth-child(even) td { background: #111827; }
.prose blockquote { border-left: 3px solid #1E2A3A; padding-left: 10px; color: #64748B !important; }

/* ── Buttons ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #00D4FF, #0090CC) !important;
    color: #000 !important; font-weight: bold !important;
    border-radius: 8px !important; border: none !important;
}
button { border-radius: 8px !important; }

/* ── Input labels ── */
.label-wrap label, .block label span {
    color: #64748B !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── Camera / image panels ── */
.gr-image, .image-container img {
    border-radius: 10px !important;
    border: 1px solid #1E2A3A !important;
}

/* ── Panels and blocks ── */
.block, .panel, .gradio-block {
    background: #0B0F1A !important;
    border-color: #1E2A3A !important;
}

/* ── Accordion ── */
details summary {
    background: #141925 !important;
    border: 1px solid #1E2A3A !important;
    border-radius: 8px !important;
    color: #64748B !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
}
details[open] summary { border-radius: 8px 8px 0 0 !important; }
details > div {
    background: #0F1520 !important;
    border: 1px solid #1E2A3A !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #141925; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #2A3040; border-radius: 3px; min-height: 20px; }
::-webkit-scrollbar-thumb:hover { background: #3A4050; }

/* ── Plots ── */
.plot-container, .js-plotly-plot { background: #0B0F1A !important; }

/* ── Row spacing ── */
.gap-4 { gap: 10px !important; }

/* ── Responsive ── */
@media (max-width: 900px) {
    .image-container { max-height: 260px !important; }
}
@media (max-width: 600px) {
    .image-container { max-height: 200px !important; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

_ABOUT_MD = """
## 🔬 How It Works

| Signal | Technology | Latency |
|--------|-----------|---------|
| **Heart Rate** | rPPG — subtle skin-colour variations from blood flow | ~10 s warm-up |
| **Fatigue** | EAR (Eye Aspect Ratio) + PERCLOS + blink rate | Instant |
| **Stress** | HRV proxy + blink suppression + brow tension (AU4) | ~5 s |
| **Breathing** | Nose-tip oscillation / RSA motion | ~8 s |
| **Emotion** | EfficientNet-B0 deep ensemble + FACS geometry fusion | ~1.5 s |
| **Attention** | Head stability + gaze + blink cadence | Instant |
| **Wellbeing** | Multi-signal fusion (all of the above) | ~5 s |

## 🤖 AI Architecture

The **Emotion Engine** uses two parallel systems fused per-state:
1. **FACS Geometry** — 478 MediaPipe landmarks → 8 action unit proxies (AU1/4/5/12 etc.) → weighted evidence scoring with adaptive per-person calibration
2. **Deep Model** (EfficientNet-B0) — frame classifier ensemble from VGGFace2 + AffectNet data, blended by per-state alpha weights (Sad α=0.70, Angry α=0.65, Happy α=0.50)

Per-person adaptive calibration (rolling-median baseline) adapts within ~30 frames so your natural resting expression reads as Neutral.

> ⚠️ For educational and research purposes only. Not a medical diagnostic device.
"""

_HOWTO_MD = """
### 📖 Quick Start
1. Click **▶ Start** on the camera panel and allow browser camera access
2. Sit **30–60 cm** from the camera in **good, stable lighting**
3. Keep your face **visible** for 5–10 seconds during calibration — watch **SYSTEM READY**
4. Try different expressions — the emotion timeline and confidence bars update in real-time

> **Mobile:** Camera requires HTTPS. Run `python run_web.py --share` for a secure public tunnel URL.
"""


def build_ui():
    with gr.Blocks(title="Face Health Digital Twin — AI") as demo:

        session_state = gr.State(None)

        # ── Header / system status ────────────────────────────────────
        system_out = gr.HTML(value=_system_html(False, False, True, 0))

        # ── Camera row ───────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                webcam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="📷 LIVE CAMERA  —  allow access when prompted",
                )
            with gr.Column(scale=3):
                video_out = gr.Image(
                    type="numpy",
                    label="🔍 ANNOTATED FEED  —  face mesh & landmarks",
                )

        # ── Vital cards + Risk banner ─────────────────────────────────
        vital_out = gr.HTML()

        # ── Emotion section ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                emotion_out = gr.HTML()
            with gr.Column(scale=3):
                emo_bars_out = gr.HTML()

        # ── AI row: Attention + Wellbeing + Insights ──────────────────
        secondary_out = gr.HTML()

        # ── Facial signal analysis ─────────────────────────────────────
        wellbeing_out = gr.HTML()

        # ── Charts ───────────────────────────────────────────────────
        with gr.Row():
            rppg_plot    = gr.Plot(label="rPPG Pulse Signal")
            trend_plot   = gr.Plot(label="Vital Trends")
            emotion_plot = gr.Plot(label="Emotion Timeline")

        # ── Info accordion ─────────────────────────────────────────────
        with gr.Accordion("📖 Quick Start Guide", open=False):
            gr.Markdown(_HOWTO_MD)
        with gr.Accordion("🔬 About This System", open=False):
            gr.Markdown(_ABOUT_MD)

        # ── Wire streaming ────────────────────────────────────────────
        webcam_in.stream(
            fn=process_frame,
            inputs=[webcam_in, session_state],
            outputs=[
                video_out,
                system_out, vital_out,
                emotion_out, emo_bars_out,
                secondary_out, wellbeing_out,
                rppg_plot, trend_plot, emotion_plot,
                session_state,
            ],
            time_limit=None,
            stream_every=0.067,
        )

    return demo


def launch(share: bool = False, auth=None, port: int = 7860):
    demo = build_ui()
    demo.launch(
        share=share,
        auth=auth,
        server_port=port,
        server_name="0.0.0.0",
        css=_CSS,
    )
