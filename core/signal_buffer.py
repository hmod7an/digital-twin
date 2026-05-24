"""
Fixed-length circular buffer for real-time physiological signals.
"""
import time
import numpy as np
from collections import deque
from typing import Optional


class SignalBuffer:
    """
    Stores (timestamp, value) pairs up to a maximum duration window.
    Provides numpy arrays of values and timestamps for DSP operations.
    """

    def __init__(self, max_seconds: float = 10.0):
        self._max_seconds = max_seconds
        self._timestamps: deque[float] = deque()
        self._values: deque[float] = deque()

    def push(self, value: float, timestamp: Optional[float] = None):
        t = timestamp if timestamp is not None else time.time()
        self._timestamps.append(t)
        self._values.append(value)
        self._evict_old(t)

    def _evict_old(self, now: float):
        cutoff = now - self._max_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._values.popleft()

    @property
    def values(self) -> np.ndarray:
        return np.array(self._values, dtype=np.float64)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps, dtype=np.float64)

    @property
    def duration(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def sample_rate(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / self.duration

    def __len__(self) -> int:
        return len(self._values)

    def clear(self):
        self._timestamps.clear()
        self._values.clear()
