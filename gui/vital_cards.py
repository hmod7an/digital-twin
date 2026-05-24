"""
Vital-sign metric card widgets — Face Health Digital Twin.
Each card has a coloured 4px accent strip on top, large numerics, and a progress bar.
"""
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt
from gui.styles import score_color, bpm_color, attention_color, wellbeing_color


# ── Base card ────────────────────────────────────────────────────────────────

class _BaseCard(QFrame):
    """
    Shared card skeleton:
      [4 px accent strip via border-top]
      [icon  TITLE HEADER]
      [BIG VALUE   unit]
      [sub-label text]
    Subclasses add extra rows via self._card_layout.
    """

    def __init__(self, icon: str, title: str,
                 accent_color: str = "#00D4FF", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._accent_color = accent_color
        self._apply_accent(accent_color)

        self._card_layout = QVBoxLayout(self)
        self._card_layout.setContentsMargins(14, 10, 14, 12)
        self._card_layout.setSpacing(5)

        # Header row
        header = QHBoxLayout()
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet("font-size:16px;")
        lbl_title = QLabel(title)
        lbl_title.setObjectName("section_header")
        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()
        self._card_layout.addLayout(header)

        # Value row
        val_row = QHBoxLayout()
        self._lbl_value = QLabel("--")
        self._lbl_value.setObjectName("metric_value")
        self._lbl_unit = QLabel("")
        self._lbl_unit.setObjectName("metric_unit")
        self._lbl_unit.setAlignment(Qt.AlignBottom)
        val_row.addWidget(self._lbl_value)
        val_row.addWidget(self._lbl_unit)
        val_row.addStretch()
        self._card_layout.addLayout(val_row)

        # Sub-label
        self._lbl_sub = QLabel("")
        self._lbl_sub.setObjectName("metric_label")
        self._card_layout.addWidget(self._lbl_sub)

    def _apply_accent(self, color: str, border_side: str = "#1E2A3A"):
        self.setStyleSheet(
            f"QFrame#card {{"
            f" background-color:#141925;"
            f" border:1px solid {border_side};"
            f" border-top:4px solid {color};"
            f" border-radius:12px; }}"
        )

    def _set_value_color(self, color: str):
        self._lbl_value.setStyleSheet(
            f"font-size:40px; font-weight:800; color:{color}; letter-spacing:-1px;"
        )


# ── Vital cards ───────────────────────────────────────────────────────────────

class HeartRateCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("❤️", "HEART RATE", accent_color="#FF5E7A", parent=parent)
        self._lbl_unit.setText("BPM")

        self._buf_bar = QProgressBar()
        self._buf_bar.setObjectName("buffer")
        self._buf_bar.setRange(0, 100)
        self._buf_bar.setTextVisible(True)
        self._buf_bar.setFormat("Buffer %p%")
        self._buf_bar.setFixedHeight(10)
        self._card_layout.addWidget(self._buf_bar)

        self._hr_bar = QProgressBar()
        self._hr_bar.setObjectName("hr")
        self._hr_bar.setRange(40, 180)
        self._hr_bar.setTextVisible(False)
        self._hr_bar.setFixedHeight(10)
        self._card_layout.addWidget(self._hr_bar)

    def refresh(self, bpm, confidence: float, buffer_fill: float):
        if bpm is not None:
            self._lbl_value.setText(f"{bpm:.0f}")
            self._set_value_color(bpm_color(bpm))
            self._lbl_sub.setText(f"Quality: {confidence*100:.0f}%")
            self._buf_bar.setVisible(False)
            self._hr_bar.setVisible(True)
            self._hr_bar.setValue(int(bpm))
        else:
            self._lbl_value.setText("--")
            self._set_value_color("#6B7280")
            pct = int(buffer_fill * 100)
            self._lbl_sub.setText(f"Calibrating… {pct}%")
            self._buf_bar.setVisible(True)
            self._buf_bar.setValue(pct)
            self._hr_bar.setVisible(False)


class FatigueCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("😴", "FATIGUE", accent_color="#FF8C00", parent=parent)
        self._lbl_unit.setText("%")
        self._bar = QProgressBar()
        self._bar.setObjectName("fatigue")
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._card_layout.addWidget(self._bar)

    def refresh(self, score: float, state: str, ear: float, perclos: float):
        self._lbl_value.setText(f"{score:.0f}")
        self._set_value_color(score_color(score))
        self._lbl_sub.setText(f"{state}  ·  EAR {ear:.3f}  ·  PERCLOS {perclos:.1f}%")
        self._bar.setValue(int(score))


class StressCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("🧠", "STRESS", accent_color="#FF4B4B", parent=parent)
        self._lbl_unit.setText("%")
        self._bar = QProgressBar()
        self._bar.setObjectName("stress")
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._card_layout.addWidget(self._bar)

    def refresh(self, score: float, state: str, hrv: float, head_mv: float):
        self._lbl_value.setText(f"{score:.0f}")
        self._set_value_color(score_color(score))
        self._lbl_sub.setText(f"{state}  ·  HRV {hrv:.1f}  ·  Motion {head_mv:.2f}")
        self._bar.setValue(int(score))


class BreathingCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("🫁", "BREATHING", accent_color="#00FF88", parent=parent)
        self._lbl_unit.setText("/min")
        self._bar = QProgressBar()
        self._bar.setObjectName("breathing")
        self._bar.setRange(0, 30)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._card_layout.addWidget(self._bar)

    def refresh(self, rate, confidence: float):
        if rate is not None and confidence > 0.2:
            self._lbl_value.setText(f"{rate:.0f}")
            self._set_value_color("#00FF88")
            self._lbl_sub.setText(f"Confidence: {confidence*100:.0f}%")
            self._bar.setValue(int(rate))
        else:
            self._lbl_value.setText("--")
            self._set_value_color("#6B7280")
            self._lbl_sub.setText("Estimating breathing pattern…")
            self._bar.setValue(0)


# ── AI cards ─────────────────────────────────────────────────────────────────

class EmotionCard(QFrame):
    """
    Big emoji + state name + confidence + stability bar.
    Press 'D' in the main window to toggle the developer debug rows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._cur_color  = "#A78BFA"
        self._debug_mode = True     # toggled by MainWindow 'D' key
        self._refresh_frame_style(self._cur_color)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(4)

        # ── Header ────────────────────────────────────────────────────
        header = QHBoxLayout()
        lbl_icon = QLabel("🧬")
        lbl_icon.setStyleSheet("font-size:16px;")
        lbl_title = QLabel("EMOTIONAL STATE")
        lbl_title.setObjectName("section_header")
        self._lbl_dev_tag = QLabel("DEV")
        self._lbl_dev_tag.setStyleSheet(
            "font-size:9px; color:#00D4FF; font-weight:bold; "
            "background:#0B0F1A; border:1px solid #00D4FF44; "
            "border-radius:3px; padding:1px 4px;"
        )
        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()
        header.addWidget(self._lbl_dev_tag)
        lay.addLayout(header)

        # ── Emoji + State ─────────────────────────────────────────────
        emotion_row = QHBoxLayout()
        emotion_row.setSpacing(10)
        self._lbl_emoji = QLabel("😐")
        self._lbl_emoji.setObjectName("emotion_emoji")
        self._lbl_emoji.setAlignment(Qt.AlignVCenter)
        vcol = QVBoxLayout()
        vcol.setSpacing(2)
        self._lbl_state = QLabel("Neutral")
        self._lbl_state.setObjectName("emotion_state")
        self._lbl_persist = QLabel("Settling…")
        self._lbl_persist.setStyleSheet("font-size:10px; color:#64748B;")
        vcol.addWidget(self._lbl_state)
        vcol.addWidget(self._lbl_persist)
        emotion_row.addWidget(self._lbl_emoji)
        emotion_row.addLayout(vcol)
        emotion_row.addStretch()
        lay.addLayout(emotion_row)

        # ── Confidence bar ────────────────────────────────────────────
        self._conf_bar = QProgressBar()
        self._conf_bar.setObjectName("emotion_conf")
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setFixedHeight(8)
        lay.addWidget(self._conf_bar)

        self._lbl_conf = QLabel("Confidence: --")
        self._lbl_conf.setObjectName("metric_label")
        lay.addWidget(self._lbl_conf)

        # ── Stability bar (teal) ──────────────────────────────────────
        self._stab_bar = QProgressBar()
        self._stab_bar.setRange(0, 100)
        self._stab_bar.setTextVisible(False)
        self._stab_bar.setFixedHeight(6)
        self._stab_bar.setStyleSheet(
            "QProgressBar{background:#1A2030;border:none;border-radius:3px;}"
            "QProgressBar::chunk{background:#00D4FF88;border-radius:3px;}"
        )
        lay.addWidget(self._stab_bar)

        # ── Developer debug rows (hidden when debug_mode off) ─────────
        self._lbl_debug1 = QLabel("")
        self._lbl_debug1.setStyleSheet("font-size:10px; color:#475569;")
        lay.addWidget(self._lbl_debug1)

        self._lbl_debug2 = QLabel("")
        self._lbl_debug2.setStyleSheet("font-size:10px; color:#475569;")
        self._lbl_debug2.setWordWrap(True)
        lay.addWidget(self._lbl_debug2)

        # Trend line
        self._lbl_trend = QLabel("")
        self._lbl_trend.setStyleSheet("font-size:10px; color:#334155;")
        lay.addWidget(self._lbl_trend)

    # ── Developer mode toggle ─────────────────────────────────────────
    def toggle_debug(self):
        self._debug_mode = not self._debug_mode
        for w in (self._lbl_debug1, self._lbl_debug2, self._lbl_trend):
            w.setVisible(self._debug_mode)
        self._lbl_dev_tag.setVisible(self._debug_mode)

    def _refresh_frame_style(self, color: str):
        self.setStyleSheet(
            f"QFrame#card {{"
            f" background-color:#141925;"
            f" border:1px solid {color}44;"
            f" border-top:4px solid {color};"
            f" border-radius:12px; }}"
        )

    def refresh(self, state: str, emoji: str, color: str, confidence: float,
                scores: dict = None, smile_score: float = 0.0,
                furrow_score: float = 0.0, frown_score: float = 0.0,
                stability: float = 0.0, persist_s: float = 0.0,
                is_calibrating: bool = False, trend: str = "Neutral",
                facial_energy: float = 0.5, inner_brow_raise: float = 0.0):
        self._lbl_emoji.setText(emoji)
        self._lbl_state.setText(state)
        self._lbl_state.setStyleSheet(f"font-size:20px; font-weight:700; color:{color};")
        self._conf_bar.setValue(int(confidence * 100))
        self._stab_bar.setValue(int(stability * 100))
        self._lbl_conf.setText(f"Confidence: {confidence*100:.0f}%")

        if is_calibrating:
            self._lbl_persist.setText("⏳ Calibrating…")
        else:
            mins = int(persist_s // 60)
            secs = int(persist_s % 60)
            t = f"{mins}m {secs}s" if mins else f"{secs}s"
            stab_pct = int(stability * 100)
            self._lbl_persist.setText(f"⏱ {t} stable  ·  {stab_pct}% certainty")

        if color != self._cur_color:
            self._cur_color = color
            self._refresh_frame_style(color)

        # Debug rows (only updated when visible)
        if self._debug_mode:
            self._lbl_debug1.setText(
                f"smile:{smile_score:.2f}  frown:{frown_score:.2f}"
                f"  furrow:{furrow_score:.2f}  ibrow:{inner_brow_raise:.2f}"
                f"  nrg:{facial_energy:.2f}"
            )
            if scores:
                h  = scores.get("Happy",   0.0)
                s  = scores.get("Sad",     0.0)
                an = scores.get("Angry",   0.0)
                n  = scores.get("Neutral", 0.0)
                self._lbl_debug2.setText(
                    f"H:{h*100:.0f}%  S:{s*100:.0f}%"
                    f"  An:{an*100:.0f}%  N:{n*100:.0f}%"
                )
            self._lbl_trend.setText(f"30s trend → {trend}")


class AttentionCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("🎯", "ATTENTION", accent_color="#00D4FF", parent=parent)
        self._lbl_unit.setText("%")

        self._att_bar = QProgressBar()
        self._att_bar.setObjectName("attention")
        self._att_bar.setRange(0, 100)
        self._att_bar.setTextVisible(False)
        self._att_bar.setFixedHeight(10)
        self._card_layout.addWidget(self._att_bar)

        cog_row = QHBoxLayout()
        cog_lbl = QLabel("Cog. Load")
        cog_lbl.setObjectName("metric_label")
        self._cog_bar = QProgressBar()
        self._cog_bar.setObjectName("cogload")
        self._cog_bar.setRange(0, 100)
        self._cog_bar.setTextVisible(False)
        self._cog_bar.setFixedHeight(7)
        cog_row.addWidget(cog_lbl)
        cog_row.addWidget(self._cog_bar)
        self._card_layout.addLayout(cog_row)

    def refresh(self, score: float, level: str, cognitive_load: float):
        self._lbl_value.setText(f"{score:.0f}")
        self._set_value_color(attention_color(score))
        self._lbl_sub.setText(f"{level}  ·  Cog. Load {cognitive_load:.0f}%")
        self._att_bar.setValue(int(score))
        self._cog_bar.setValue(int(cognitive_load))


class WellbeingCard(_BaseCard):
    def __init__(self, parent=None):
        super().__init__("🌡️", "WELLBEING", accent_color="#00FF88", parent=parent)
        self._lbl_unit.setText("%")
        self._wb_bar = QProgressBar()
        self._wb_bar.setObjectName("wellbeing")
        self._wb_bar.setRange(0, 100)
        self._wb_bar.setTextVisible(False)
        self._wb_bar.setFixedHeight(10)
        self._card_layout.addWidget(self._wb_bar)

    def refresh(self, score: float, label: str, trend: str, icon: str):
        self._lbl_value.setText(f"{score:.0f}")
        self._set_value_color(wellbeing_color(score))
        self._lbl_sub.setText(f"{label}  {icon} {trend}")
        self._wb_bar.setValue(int(score))


# ── Insights panel ───────────────────────────────────────────────────────────

class InsightsPanel(QFrame):
    """AI insights + recommendations with coloured bullets."""

    MAX_INSIGHTS = 5
    MAX_RECS     = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("insights_card")
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(3)

        header = QHBoxLayout()
        lbl_icon  = QLabel("💡")
        lbl_icon.setStyleSheet("font-size:16px;")
        lbl_title = QLabel("AI INSIGHTS")
        lbl_title.setObjectName("section_header")
        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()
        lay.addLayout(header)

        self._insight_labels: list[QLabel] = []
        for _ in range(self.MAX_INSIGHTS):
            lbl = QLabel("")
            lbl.setObjectName("insight_line")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            self._insight_labels.append(lbl)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1E2A3A;")
        lay.addWidget(sep)

        self._rec_labels: list[QLabel] = []
        for _ in range(self.MAX_RECS):
            lbl = QLabel("")
            lbl.setObjectName("rec_line")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            self._rec_labels.append(lbl)

    def refresh(self, insights: list, recommendations: list):
        for i, lbl in enumerate(self._insight_labels):
            text = insights[i] if i < len(insights) else ""
            lbl.setText(f"▸  {text}" if text else "")
            lbl.setVisible(bool(text))

        for i, lbl in enumerate(self._rec_labels):
            text = recommendations[i] if i < len(recommendations) else ""
            lbl.setText(f"→  {text}" if text else "")
            lbl.setVisible(bool(text))


# ── Risk banner ───────────────────────────────────────────────────────────────

class RiskBanner(QFrame):
    """Full-width risk status banner."""

    _STYLES = {
        0: ("risk_card_normal",  "✅  SYSTEM STATUS: NORMAL"),
        1: ("risk_card_warning", "⚠️  STATUS: WARNING"),
        2: ("risk_card_danger",  "🚨  STATUS: HIGH RISK"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(4)
        self._lbl_title = QLabel()
        self._lbl_title.setStyleSheet("font-size:15px; font-weight:bold;")
        self._lbl_msgs  = QLabel()
        self._lbl_msgs.setStyleSheet("font-size:11px; color:#9CA3AF;")
        self._lbl_msgs.setWordWrap(True)
        lay.addWidget(self._lbl_title)
        lay.addWidget(self._lbl_msgs)

    def refresh(self, level_code: int, messages: list):
        obj_name, heading = self._STYLES.get(level_code, self._STYLES[0])
        self.setObjectName(obj_name)
        self.style().unpolish(self)
        self.style().polish(self)
        colors = {0: "#00FF88", 1: "#FFD700", 2: "#FF4B4B"}
        self._lbl_title.setStyleSheet(
            f"font-size:15px; font-weight:bold; color:{colors[level_code]};"
        )
        self._lbl_title.setText(heading)
        self._lbl_msgs.setText("  ·  ".join(messages))
