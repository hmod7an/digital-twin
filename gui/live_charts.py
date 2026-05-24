"""
pyqtgraph-based real-time charts: rPPG waveform + vital history.
"""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from collections import deque

# Global pyqtgraph dark config
pg.setConfigOption("background", "#0E1117")
pg.setConfigOption("foreground", "#9CA3AF")
pg.setConfigOptions(antialias=True)


class RPPGChart(QWidget):
    """Live rPPG pulse waveform (last 10 s of signal)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._lbl = QLabel("rPPG Signal — calibrating…")
        self._lbl.setStyleSheet("font-size:12px; color:#00D4FF; padding-left:8px;")
        lay.addWidget(self._lbl)

        self._plot = pg.PlotWidget()
        self._plot.setMinimumHeight(140)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=False, y=True, alpha=0.15)

        pen = pg.mkPen(color="#00D4FF", width=1.5)
        self._curve = self._plot.plot(pen=pen)
        lay.addWidget(self._plot)

        # Confidence badge (top-right)
        self._conf_label = QLabel("Quality: 0%")
        self._conf_label.setStyleSheet("font-size:10px; color:#6B7280; padding-right:8px;")
        self._conf_label.setAlignment(Qt.AlignRight)
        lay.addWidget(self._conf_label)

    def update(self, signal: np.ndarray, bpm, confidence: float):
        if len(signal) > 5:
            s = signal[-300:]
            rng = s.max() - s.min()
            if rng > 1e-9:
                s = (s - s.min()) / rng * 2 - 1
            self._curve.setData(s)

        title = f"rPPG Signal — {bpm:.0f} BPM" if bpm else "rPPG Signal — calibrating…"
        self._lbl.setText(title)

        c_color = "#00FF88" if confidence > 0.6 else ("#FFD700" if confidence > 0.3 else "#FF4B4B")
        self._conf_label.setStyleSheet(f"font-size:10px; color:{c_color}; padding-right:8px;")
        self._conf_label.setText(f"Quality: {confidence*100:.0f}%")


class HistoryChart(QWidget):
    """
    Two-panel history chart:
    Top:    BPM over time (cyan line)
    Bottom: Fatigue (orange fill) + Stress (red line)
    """

    HISTORY_LEN = 300  # ~5 min at 1 sample/s

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        lbl = QLabel("Vital Trends")
        lbl.setStyleSheet("font-size:12px; color:#9CA3AF; padding-left:8px;")
        lay.addWidget(lbl)

        self._bpm_buf     = deque([0.0] * self.HISTORY_LEN, maxlen=self.HISTORY_LEN)
        self._fatigue_buf = deque([0.0] * self.HISTORY_LEN, maxlen=self.HISTORY_LEN)
        self._stress_buf  = deque([0.0] * self.HISTORY_LEN, maxlen=self.HISTORY_LEN)

        # Top: BPM
        self._bpm_plot = pg.PlotWidget()
        self._bpm_plot.setMinimumHeight(100)
        self._bpm_plot.hideAxis("bottom")
        self._bpm_plot.setMouseEnabled(False, False)
        self._bpm_plot.setMenuEnabled(False)
        self._bpm_plot.showGrid(x=False, y=True, alpha=0.15)
        self._bpm_plot.setLabel("left", "BPM", color="#9CA3AF", size="9pt")
        self._bpm_plot.setYRange(40, 150, padding=0.05)
        self._bpm_curve = self._bpm_plot.plot(
            pen=pg.mkPen(color="#00D4FF", width=2)
        )
        lay.addWidget(self._bpm_plot)

        # Bottom: Fatigue + Stress
        self._fs_plot = pg.PlotWidget()
        self._fs_plot.setMinimumHeight(80)
        self._fs_plot.hideAxis("bottom")
        self._fs_plot.setMouseEnabled(False, False)
        self._fs_plot.setMenuEnabled(False)
        self._fs_plot.showGrid(x=False, y=True, alpha=0.15)
        self._fs_plot.setLabel("left", "%", color="#9CA3AF", size="9pt")
        self._fs_plot.setYRange(0, 100, padding=0)

        self._fat_curve = self._fs_plot.plot(
            pen=pg.mkPen(color="#FF8C00", width=2), name="Fatigue"
        )
        self._fat_fill = pg.FillBetweenItem(
            self._fat_curve,
            self._fs_plot.plot(np.zeros(self.HISTORY_LEN)),
            brush=pg.mkBrush(255, 140, 0, 30),
        )
        self._fs_plot.addItem(self._fat_fill)

        self._str_curve = self._fs_plot.plot(
            pen=pg.mkPen(color="#FF4B4B", width=2), name="Stress"
        )
        lay.addWidget(self._fs_plot)

        # Legend
        legend_row = QHBoxLayout()
        for color, text in [("#00D4FF", "BPM"), ("#FF8C00", "Fatigue %"), ("#FF4B4B", "Stress %")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:12px;")
            lbl2 = QLabel(text)
            lbl2.setStyleSheet("font-size:10px; color:#9CA3AF;")
            legend_row.addWidget(dot)
            legend_row.addWidget(lbl2)
        legend_row.addStretch()
        lay.addLayout(legend_row)

    def push(self, bpm_val: float, fatigue: float, stress: float):
        self._bpm_buf.append(bpm_val)
        self._fatigue_buf.append(fatigue)
        self._stress_buf.append(stress)

        x = np.arange(self.HISTORY_LEN)
        self._bpm_curve.setData(x, np.array(self._bpm_buf))
        self._fat_curve.setData(x, np.array(self._fatigue_buf))
        self._str_curve.setData(x, np.array(self._stress_buf))
