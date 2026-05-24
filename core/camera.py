"""
Thread-safe webcam capture manager with graceful error handling.
"""
import threading
import time
import cv2
import numpy as np
from config.settings import settings


class CameraManager:
    """
    Captures frames from a webcam in a background thread so the main
    pipeline always has the latest frame without blocking on I/O.
    """

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps_actual: float = 0.0
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self.available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the webcam and start the background capture thread."""
        cfg = settings.camera

        # Try backends in order: MSMF → DirectShow → auto
        # Only require that the device opens and returns any frame — dark startup
        # frames are normal and cleared by the warm-up read loop below.
        self._cap = None
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(cfg.device_index, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()

        if self._cap is None:
            self.available = False
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Discard first 20 frames — cameras commonly output dark/green frames on startup
        for _ in range(20):
            self._cap.read()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.available = True
        return True

    def stop(self):
        """Stop capture and release the device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self.available = False

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent captured frame (thread-safe copy)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def fps(self) -> float:
        return self._fps_actual

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            with self._lock:
                self._frame = frame

            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps_actual = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now
