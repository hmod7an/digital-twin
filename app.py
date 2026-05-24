"""
Hugging Face Spaces entry point — Face Health Digital Twin.

HF Spaces runs this file on every cold start. It downloads the ONNX
emotion models from the HSEmotion GitHub release (once, ~60 MB total)
then launches the Gradio UI on the standard port.
"""
import os, sys, urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Download ONNX models if missing ──────────────────────────────────────────
_MODEL_DIR = os.path.expanduser("~/.hsemotion")
_ONNX_BASE = ("https://github.com/HSE-asavchenko/face-emotion-recognition"
              "/raw/main/models/affectnet_emotions/onnx")
_ONNX_MODELS = [
    "enet_b0_8_best_vgaf.onnx",
    "enet_b0_8_best_afew.onnx",
    "enet_b2_7.onnx",
]

os.makedirs(_MODEL_DIR, exist_ok=True)
for _fname in _ONNX_MODELS:
    _dst = os.path.join(_MODEL_DIR, _fname)
    if not os.path.exists(_dst):
        print(f"[startup] Downloading {_fname} …", flush=True)
        try:
            urllib.request.urlretrieve(f"{_ONNX_BASE}/{_fname}", _dst)
            print(f"[startup] {_fname} OK ({os.path.getsize(_dst)//1024} KB)",
                  flush=True)
        except Exception as exc:
            print(f"[startup] Warning: {_fname} failed: {exc}", flush=True)

# ── Launch ────────────────────────────────────────────────────────────────────
from app.web_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
