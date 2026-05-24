"""
Hugging Face Spaces entry point — Face Health Digital Twin.

HF Spaces runs this file directly. It downloads any missing models
then launches the Gradio UI on the standard port.

Deploy:
    1. Create a new Space at https://huggingface.co/new-space
       - SDK: Gradio
       - Hardware: CPU Basic (free) or T4 GPU (paid, faster inference)
    2. Push this repo: git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
                       git push hf master
    3. Done — your permanent URL is: https://YOUR_USER-YOUR_SPACE.hf.space
"""
import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Download ONNX emotion models if missing ───────────────────────────────────
_MODEL_DIR  = os.path.expanduser("~/.hsemotion")
_ONNX_BASE  = ("https://github.com/HSE-asavchenko/face-emotion-recognition"
               "/raw/main/models/affectnet_emotions/onnx")
_ONNX_MODELS = [
    "enet_b0_8_best_vgaf.onnx",
    "enet_b0_8_best_afew.onnx",
    "enet_b2_7.onnx",
]

def _download_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for fname in _ONNX_MODELS:
        dst = os.path.join(_MODEL_DIR, fname)
        if not os.path.exists(dst):
            url = f"{_ONNX_BASE}/{fname}"
            print(f"[startup] Downloading {fname} …", flush=True)
            try:
                urllib.request.urlretrieve(url, dst)
                print(f"[startup] {fname} — OK ({os.path.getsize(dst)//1024} KB)",
                      flush=True)
            except Exception as e:
                print(f"[startup] Warning: could not download {fname}: {e}",
                      flush=True)

_download_models()

# ── Launch ────────────────────────────────────────────────────────────────────
from app.web_app import launch

# HF Spaces expects port 7860 and server_name 0.0.0.0
launch(share=False, auth=None, port=7860)
