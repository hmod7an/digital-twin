"""
Hugging Face Spaces entry point.
HF Spaces expects a file named app.py at the repo root that exposes `demo`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.web_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
