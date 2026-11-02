"""Camera capture with resolution request and automatic reconnect."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def parse_source(text: str) -> int | str:
    """'0' -> 0 (device index). Anything else is a URL or file path."""
    return int(text) if text.isdigit() else text


class Camera:
    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        max_failures: int = 5,
        reconnect_delay: float = 1.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.max_failures = max_failures
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0
        self.reconnects = 0

    def _backend(self) -> int:
        if isinstance(self.source, int) and sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    def open(self) -> bool:
        self.release()
        cap = cv2.VideoCapture(self.source, self._backend())
        if not cap.isOpened():
            return False
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Warm-up read: some drivers apply the new size only after the first grab.
        cap.read()
        self._cap = cap
        self._failures = 0
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def size(self) -> tuple[int, int]:
        if not self._cap:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def reported_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    def read(self) -> np.ndarray | None:
        """Next frame, or None when the camera is down. Reconnects on repeated failure."""
        if not self.is_open and not self.open():
            time.sleep(self.reconnect_delay)
            return None
        ok, frame = self._cap.read()  # type: ignore[union-attr]
        if ok and frame is not None:
            self._failures = 0
            return frame
        self._failures += 1
        if self._failures >= self.max_failures:
            self.reconnects += 1
            self.release()
            time.sleep(self.reconnect_delay)
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
