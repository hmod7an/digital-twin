"""
Downloads required MediaPipe task model files on first run.
Models are cached in core/models/ and never re-downloaded.
"""
import os
import urllib.request
import urllib.error
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"


def ensure_face_landmarker() -> Path:
    """
    Return path to the face landmarker model, downloading it if necessary.
    Raises RuntimeError if download fails and model is absent.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 1_000_000:
        return FACE_LANDMARKER_PATH

    print(f"[ModelManager] Downloading face landmarker model (~5 MB)…")
    print(f"  From: {FACE_LANDMARKER_URL}")
    print(f"  To:   {FACE_LANDMARKER_PATH}")

    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        size_mb = FACE_LANDMARKER_PATH.stat().st_size / 1_048_576
        print(f"[ModelManager] Downloaded {size_mb:.1f} MB — OK")
        return FACE_LANDMARKER_PATH
    except urllib.error.URLError as e:
        if FACE_LANDMARKER_PATH.exists():
            FACE_LANDMARKER_PATH.unlink()
        raise RuntimeError(
            f"Failed to download face landmarker model: {e}\n"
            "Please check your internet connection, then restart the application.\n"
            f"Alternatively, manually download from:\n  {FACE_LANDMARKER_URL}\n"
            f"and save to:\n  {FACE_LANDMARKER_PATH}"
        ) from e
