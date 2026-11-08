"""Camera capture with resolution request and automatic reconnect.

Two kinds of source:
- a device index (Camo, Continuity Camera, built-in camera) through AVFoundation on macOS,
- a network stream URL (http MJPEG, rtsp) through FFmpeg, read on a background thread that
  keeps only the newest frame, so slow processing does not build up lag.

`CameraSettings` is everything about the camera in data/camera.toml: the source that
`tools/list_cameras.py` saved (the live commands use it as their defaults), the nominal
focal length, and the lens calibration from `battlesight calibrate`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .calib import Lens

DEFAULT_CAMERA_FILE = Path(__file__).resolve().parents[2] / "data" / "camera.toml"
STREAM_OPEN_TIMEOUT_MS = 5000
STREAM_READ_TIMEOUT_S = 1.0


def parse_source(text: str) -> int | str:
    """'0' -> 0 (device index). Anything else is a URL or file path."""
    return int(text) if text.isdigit() else text


def is_stream(source: int | str) -> bool:
    """True for a network URL (http://, rtsp://, ...). File paths and indices are not streams."""
    return isinstance(source, str) and "://" in source


@dataclass(frozen=True)
class CameraSettings:
    """Everything about the camera, saved in data/camera.toml.

    [camera]: source and requested frame (`tools/list_cameras.py`), and the 35 mm
    equivalent focal length of the nominal lens. For a stream, width and height have no
    effect. [lens]: the calibration from `battlesight calibrate`.
    """

    source: int | str = 0
    width: int = 1920
    height: int = 1080
    fps: float | None = None
    focal_35mm: float = 24.0  # Xiaomi 13T main lens
    lens: Lens | None = field(default=None, compare=False)
    lens_views: int | None = None
    lens_date: str | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CAMERA_FILE) -> CameraSettings:
        """Saved settings, or the built-in defaults when the file does not exist."""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        cam, lens = raw.get("camera", {}), raw.get("lens")
        source = cam.get("source", 0)
        if isinstance(source, str):
            source = parse_source(source)
        fps = cam.get("fps")
        return cls(
            source=source,
            width=int(cam.get("width", cls.width)),
            height=int(cam.get("height", cls.height)),
            fps=None if fps is None else float(fps),
            focal_35mm=float(cam.get("focal_35mm", cls.focal_35mm)),
            lens=None if lens is None else Lens.from_toml(lens),
            lens_views=None if lens is None or "views" not in lens else int(lens["views"]),
            lens_date=None if lens is None else lens.get("date"),
        )

    def save(self, path: str | Path = DEFAULT_CAMERA_FILE) -> None:
        lines = [
            "# Camera settings. tools/list_cameras.py writes [camera], battlesight calibrate",
            "# writes [lens]. The --source option overrides the source for one run.",
            "",
            "[camera]",
            f"source = {json.dumps(self.source)}",
        ]
        if not is_stream(self.source):
            lines += [f"width = {self.width}", f"height = {self.height}"]
        if self.fps is not None:
            lines.append(f"fps = {self.fps}")
        lines += [
            "# Nominal lens when [lens] is missing or does not fit the frame: pinhole, no distortion.",
            f"focal_35mm = {self.focal_35mm}",
        ]
        if self.lens is not None:
            lines += [
                "",
                "# Lens calibration. Used when image_size has the aspect ratio of the frame (any scale).",
                "[lens]",
            ]
            raw = self.lens.to_toml()
            if self.lens_views is not None:
                raw["views"] = self.lens_views
            if self.lens_date is not None:
                raw["date"] = self.lens_date
            lines += [f"{k} = {json.dumps(v)}" for k, v in raw.items()]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

    def override(self, **values) -> CameraSettings:
        """Copy with the given values; None means keep the saved value."""
        return replace(self, **{k: v for k, v in values.items() if v is not None})

    def with_lens(self, lens: Lens, views: int) -> CameraSettings:
        return replace(self, lens=lens, lens_views=views, lens_date=time.strftime("%Y-%m-%d %H:%M:%S"))


class LatestFrameReader:
    """Reads a capture on a thread and keeps only the newest frame.

    `next()` waits for a frame newer than the last one it returned. The thread stops on the
    first failed read, so the owner can reconnect.
    """

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._taken = 0
        self._stop = False
        self.alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok or frame is None:
                    self.alive = False
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def next(self, timeout: float) -> np.ndarray | None:
        """Newest unseen frame, or None after `timeout` s or when the stream ended."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._taken or not self.alive, timeout)
            if self._seq <= self._taken:
                return None
            self._taken = self._seq
            return self._frame

    def stop(self) -> None:
        self._stop = True
        # A blocked read returns within the FFmpeg read timeout. Wait for it before release.
        self._thread.join(timeout=STREAM_OPEN_TIMEOUT_MS / 1000 + 1.0)


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
        self._reader: LatestFrameReader | None = None
        self._failures = 0
        self.reconnects = 0

    @property
    def stream(self) -> bool:
        return is_stream(self.source)

    def _backend(self) -> int:
        if isinstance(self.source, int) and sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        if self.stream:
            return cv2.CAP_FFMPEG
        return cv2.CAP_ANY

    def _open_capture(self) -> cv2.VideoCapture:
        if not self.stream:
            return cv2.VideoCapture(self.source, self._backend())
        # RTSP over TCP: UDP drops packets on Wi-Fi and gives torn frames. A user value wins.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS,
        ]
        return cv2.VideoCapture(self.source, self._backend(), params)

    def open(self) -> bool:
        self.release()
        cap = self._open_capture()
        if not cap.isOpened():
            return False
        if self.stream:
            # The phone app sets the stream size and rate. Requests have no effect here.
            self._cap = cap
            self._reader = LatestFrameReader(cap)
            self._failures = 0
            return True
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

    def _grab(self) -> np.ndarray | None:
        if self._reader is not None:
            frame = self._reader.next(STREAM_READ_TIMEOUT_S)
            if frame is None and not self._reader.alive:
                self._failures = self.max_failures  # stream ended: reconnect now
            return frame
        ok, frame = self._cap.read()  # type: ignore[union-attr]
        return frame if ok else None

    def read(self) -> np.ndarray | None:
        """Next frame, or None when the camera is down. Reconnects on repeated failure."""
        if not self.is_open and not self.open():
            time.sleep(self.reconnect_delay)
            return None
        frame = self._grab()
        if frame is not None:
            self._failures = 0
            return frame
        self._failures += 1
        if self._failures >= self.max_failures:
            self.reconnects += 1
            self.release()
            time.sleep(self.reconnect_delay)
        return None

    def release(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
