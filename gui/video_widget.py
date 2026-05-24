"""
OpenCV frame → QLabel video display widget.
"""
import numpy as np
import cv2
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class VideoWidget(QLabel):
    """Displays a live OpenCV BGR frame, scaled to the widget size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #0A0D14;"
            "border: 1px solid #2A3040;"
            "border-radius: 10px;"
            "color: #6B7280;"
            "font-size: 14px;"
        )
        self.setText("Waiting for camera…")

    def update_frame(self, bgr: np.ndarray):
        """Convert BGR OpenCV frame to QPixmap and display it."""
        # Convert colour space
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # IMPORTANT: use tobytes() to create a stable copy — using rgb.data
        # directly can produce black frames if the array is GC'd before Qt
        # finishes rendering the pixmap.
        qi = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qi)

        scaled = pix.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)
