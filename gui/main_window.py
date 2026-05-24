"""
Main PyQt5 window — Face Health Digital Twin.

Layout (maximised):
  ┌─ Header ──────────────────────────────────────────────────────────┐
  ├─ Video (left) ───────────┬─ Vitals 2×2 grid (right) ─────────────┤
  │  Live annotated feed     │  ❤️ HR     😴 Fatigue                  │
  │                          │  🧠 Stress  🫁 Breathing               │
  │                          │  ─── Risk Banner ───────────────────   │
  ├──────────────────────────┴───────────────────────────────────────┤
  │  🧬 Emotion  │  🎯 Attention  │  🌡️ Wellbeing  │  💡 Insights    │
  ├──────────────────────────────────────────────────────────────────┤
  │  rPPG waveform  │  Vital history chart                           │
  └──────────────────────────────────────────────────────────────────┘

Driven by a 33 ms QTimer (~30 fps).
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFrame, QLabel, QStatusBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeyEvent

from core.pipeline import ProcessingPipeline
from gui.video_widget import VideoWidget
from gui.vital_cards import (
    HeartRateCard, FatigueCard, StressCard, BreathingCard, RiskBanner,
    EmotionCard, AttentionCard, WellbeingCard, InsightsPanel,
)
from gui.live_charts import RPPGChart, HistoryChart
from gui.styles import DARK_THEME


class MainWindow(QMainWindow):
    REFRESH_MS = 33  # ~30 fps

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Health Digital Twin")
        self.setMinimumSize(1500, 900)
        self.setStyleSheet(DARK_THEME)

        self._pipeline    = ProcessingPipeline()
        self._camera_ok   = self._pipeline.start()
        self._history_tick = 0

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(self.REFRESH_MS)

        if not self._camera_ok:
            self._status_bar.showMessage(
                "⚠️  Camera not available — check connection and permissions"
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        root.addWidget(self._make_header())

        # ── Row 1: video + 2×2 vital cards ───────────────────────────
        body = QHBoxLayout()
        body.setSpacing(10)

        self._video = VideoWidget()
        self._video.setMinimumWidth(580)
        body.addWidget(self._video, stretch=3)
        body.addWidget(self._make_right_panel(), stretch=2)
        root.addLayout(body, stretch=5)

        # ── Row 2: emotion / attention / wellbeing / insights ─────────
        root.addWidget(self._make_ai_row(), stretch=2)

        # ── Row 3: rPPG + history charts ─────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self._rppg_chart = RPPGChart()
        self._rppg_chart.setMinimumHeight(160)
        bottom.addWidget(self._rppg_chart, stretch=3)
        self._history_chart = HistoryChart()
        self._history_chart.setMinimumHeight(160)
        bottom.addWidget(self._history_chart, stretch=4)
        root.addLayout(bottom, stretch=2)

        # ── Status bar ───────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(
            "background-color:#0B0F1A; color:#475569; font-size:11px;"
        )
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Initialising AI pipeline…")

    def _make_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("header_card")
        frame.setMaximumHeight(64)
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 8, 16, 8)

        title    = QLabel("🩺  Face Health Digital Twin")
        title.setObjectName("title")
        subtitle = QLabel(
            "Real-time AI health monitoring  |  "
            "rPPG · Fatigue · Stress · Emotion · Attention · Wellbeing"
        )
        subtitle.setObjectName("subtitle")

        vcol = QVBoxLayout()
        vcol.setSpacing(2)
        vcol.addWidget(title)
        vcol.addWidget(subtitle)
        lay.addLayout(vcol)
        lay.addStretch()

        self._lbl_live = QLabel("⚫  OFFLINE")
        self._lbl_live.setStyleSheet("font-size:13px; color:#6B7280; font-weight:bold;")
        lay.addWidget(self._lbl_live)
        return frame

    def _make_right_panel(self) -> QWidget:
        """2×2 grid of vital cards + full-width risk banner."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._hr_card        = HeartRateCard()
        self._fatigue_card   = FatigueCard()
        self._stress_card    = StressCard()
        self._breathing_card = BreathingCard()
        self._risk_banner    = RiskBanner()

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self._hr_card,        0, 0)
        grid.addWidget(self._fatigue_card,   0, 1)
        grid.addWidget(self._stress_card,    1, 0)
        grid.addWidget(self._breathing_card, 1, 1)
        lay.addLayout(grid)
        lay.addWidget(self._risk_banner)
        return widget

    def _make_ai_row(self) -> QWidget:
        """Emotion | Attention | Wellbeing | AI Insights — full-width row."""
        frame = QFrame()
        frame.setObjectName("card")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(10)

        self._emotion_card   = EmotionCard()
        self._attention_card = AttentionCard()
        self._wellbeing_card = WellbeingCard()
        self._insights_panel = InsightsPanel()

        for w in [self._emotion_card, self._attention_card, self._wellbeing_card]:
            lay.addWidget(w, stretch=1)

        lay.addWidget(self._insights_panel, stretch=2)
        return frame

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _on_tick(self):
        state = self._pipeline.get_state()

        # Video
        if state.annotated_frame is not None:
            self._video.update_frame(state.annotated_frame)

        # Vital cards
        self._hr_card.refresh(
            state.rppg.bpm, state.rppg.confidence, state.rppg.buffer_fill
        )
        self._fatigue_card.refresh(
            state.fatigue.score, state.fatigue.state,
            state.fatigue.ear, state.fatigue.perclos,
        )
        self._stress_card.refresh(
            state.stress.score, state.stress.state,
            state.stress.hr_variability, state.stress.head_movement,
        )
        self._breathing_card.refresh(
            state.breathing.rate_bpm, state.breathing.confidence
        )
        self._risk_banner.refresh(state.risk.level_code, state.risk.messages)

        # AI cards
        self._emotion_card.refresh(
            state.emotion.state, state.emotion.emoji,
            state.emotion.color, state.emotion.confidence,
            state.emotion.scores,
            state.emotion.smile_score,
            state.emotion.furrow_score,
            state.emotion.frown_score,
            state.emotion.stability_score,
            state.emotion.persistence_seconds,
            state.emotion.is_calibrating,
            state.emotion.timeline_dominant,
            state.emotion.facial_energy,
            state.emotion.inner_brow_raise,
        )
        self._attention_card.refresh(
            state.attention.score, state.attention.level,
            state.attention.cognitive_load,
        )
        self._wellbeing_card.refresh(
            state.mental_state.wellbeing_score,
            state.mental_state.wellbeing_label,
            state.mental_state.trend,
            state.mental_state.trend_icon,
        )
        self._insights_panel.refresh(
            state.mental_state.insights,
            state.mental_state.recommendations,
        )

        # Charts
        self._rppg_chart.update(
            state.rppg.signal, state.rppg.bpm, state.rppg.confidence
        )

        # History chart ~1 sample/s
        self._history_tick += 1
        if self._history_tick >= 30:
            self._history_tick = 0
            self._history_chart.push(
                state.rppg.bpm or 0.0, state.fatigue.score, state.stress.score
            )

        # Live indicator
        if state.face_detected:
            self._lbl_live.setText("🟢  LIVE")
            self._lbl_live.setStyleSheet(
                "font-size:13px; color:#00FF88; font-weight:bold;"
            )
        else:
            self._lbl_live.setText("🔴  NO FACE")
            self._lbl_live.setStyleSheet(
                "font-size:13px; color:#FF4B4B; font-weight:bold;"
            )

        bpm_txt = f"{state.rppg.bpm:.0f} BPM" if state.rppg.bpm else "BPM: calibrating"
        cal_txt = "⏳ Calibrating…  |  " if state.emotion.is_calibrating else ""
        self._status_bar.showMessage(
            f"  {cal_txt}"
            f"Cam: {state.camera_fps:.1f} fps  |  "
            f"Proc: {state.process_fps:.1f} fps  |  "
            f"{bpm_txt}  |  Fatigue: {state.fatigue.score:.0f}%  |  "
            f"Stress: {state.stress.score:.0f}%  |  "
            f"Emotion: {state.emotion.state} ({state.emotion.confidence*100:.0f}%)  |  "
            f"Attn {state.attention.score:.0f}%  |  "
            f"WB {state.mental_state.wellbeing_score:.0f}%  |  "
            f"[D] debug"
        )

    # ------------------------------------------------------------------
    # Developer mode
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_D:
            self._emotion_card.toggle_debug()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        self._pipeline.stop()
        event.accept()
