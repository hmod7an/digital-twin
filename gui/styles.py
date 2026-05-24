"""
Dark professional QSS stylesheet — Face Health Digital Twin.
"""

DARK_THEME = """
/* ── Global ─────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #0B0F1A;
    color: #E2E8F0;
    font-family: "Segoe UI", "Consolas", monospace;
    font-size: 13px;
}

/* ── Labels ─────────────────────────────────────── */
QLabel { color: #E2E8F0; }

QLabel#title {
    font-size: 24px;
    font-weight: 800;
    color: #00D4FF;
    letter-spacing: 1px;
}
QLabel#subtitle {
    font-size: 11px;
    color: #64748B;
    letter-spacing: 0.5px;
}
QLabel#section_header {
    font-size: 10px;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 2px;
}
QLabel#metric_value {
    font-size: 40px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}
QLabel#metric_unit {
    font-size: 14px;
    color: #94A3B8;
    font-weight: 600;
}
QLabel#metric_label {
    font-size: 11px;
    color: #64748B;
}
QLabel#emotion_emoji {
    font-size: 36px;
}
QLabel#emotion_state {
    font-size: 20px;
    font-weight: 700;
}
QLabel#insight_line {
    font-size: 12px;
    color: #CBD5E1;
    padding: 1px 0;
}
QLabel#rec_line {
    font-size: 11px;
    color: #64748B;
    padding: 1px 0;
}
QLabel#status_normal  { color: #00FF88; font-size: 13px; font-weight: bold; }
QLabel#status_warning { color: #FFD700; font-size: 13px; font-weight: bold; }
QLabel#status_danger  { color: #FF4B4B; font-size: 13px; font-weight: bold; }

/* ── Cards ───────────────────────────────────────── */
QFrame#card {
    background-color: #141925;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
}
QFrame#header_card {
    background-color: #0F1520;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
}
QFrame#insights_card {
    background-color: #0F1520;
    border: 1px solid #1E3050;
    border-radius: 12px;
}

/* Risk banners */
QFrame#risk_card_normal  {
    background-color: #071A10;
    border: 2px solid #00FF8855;
    border-radius: 12px;
}
QFrame#risk_card_warning {
    background-color: #1A1100;
    border: 2px solid #FFD70055;
    border-radius: 12px;
}
QFrame#risk_card_danger  {
    background-color: #1A0505;
    border: 2px solid #FF4B4B55;
    border-radius: 12px;
}

/* ── Progress bars ───────────────────────────────── */
QProgressBar {
    background-color: #1A2030;
    border: none;
    border-radius: 5px;
    height: 10px;
}
QProgressBar#hr::chunk       { background-color: #FF5E7A; border-radius: 5px; }
QProgressBar#fatigue::chunk  { background-color: #FF8C00; border-radius: 5px; }
QProgressBar#stress::chunk   { background-color: #FF4B4B; border-radius: 5px; }
QProgressBar#buffer::chunk   { background-color: #00D4FF; border-radius: 5px; }
QProgressBar#breathing::chunk { background-color: #00FF88; border-radius: 5px; }
QProgressBar#attention::chunk { background-color: #00D4FF; border-radius: 5px; }
QProgressBar#emotion_conf::chunk { background-color: #A78BFA; border-radius: 5px; }
QProgressBar#wellbeing::chunk { background-color: #00FF88; border-radius: 5px; }
QProgressBar#cogload::chunk  { background-color: #FBBF24; border-radius: 5px; }

/* ── Scroll bars ─────────────────────────────────── */
QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical {
    background: #141925; width: 6px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #2A3040; border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Status bar ─────────────────────────────────── */
QStatusBar {
    background-color: #0B0F1A;
    color: #475569;
    font-size: 11px;
    border-top: 1px solid #1E2A3A;
}
"""


def score_color(score: float) -> str:
    if score < 35:   return "#00FF88"
    elif score < 60: return "#FFD700"
    return "#FF4B4B"


def bpm_color(bpm: float) -> str:
    if 50 <= bpm <= 100:  return "#00FF88"
    elif 40 <= bpm <= 120: return "#FFD700"
    return "#FF4B4B"


def attention_color(score: float) -> str:
    if score >= 65:   return "#00D4FF"
    elif score >= 44: return "#00FF88"
    elif score >= 26: return "#FFD700"
    return "#FF4B4B"


def wellbeing_color(score: float) -> str:
    if score >= 72:   return "#00FF88"
    elif score >= 52: return "#FFD700"
    return "#FF4B4B"
