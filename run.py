"""
Face Health Digital Twin — desktop application entry point.

Usage:
    python run.py

Opens the PyQt5 real-time dashboard with live webcam input.
No browser required.
"""
import sys
import os

# Project root on path so all relative imports work
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow


def main():
    # Enable high-DPI scaling on Windows
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Face Health Digital Twin")
    app.setOrganizationName("University Research")

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
