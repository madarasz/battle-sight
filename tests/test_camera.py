import threading

import numpy as np

from battlesight.camera import Camera, CameraSettings, LatestFrameReader, is_stream, parse_source


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("12") == 12
    assert parse_source("http://192.168.1.20:8080/video") == "http://192.168.1.20:8080/video"
    assert parse_source("clip.mp4") == "clip.mp4"


def test_is_stream():
    assert is_stream("http://192.168.1.20:8080/video")
    assert is_stream("rtsp://192.168.1.20:8554/live")
    assert not is_stream(1)
    assert not is_stream("clip.mp4")
    assert Camera("rtsp://x/live").stream and not Camera(1).stream


class FakeCap:
    """Gives numbered frames on demand. `read` blocks until the test releases a frame."""

    def __init__(self, n: int):
        self.n = n
        self.i = 0
        self.go = threading.Semaphore(0)

    def read(self):
        self.go.acquire()
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((2, 2), self.i, np.uint8)


def test_reader_keeps_only_newest_frame():
    cap = FakeCap(3)
    reader = LatestFrameReader(cap)
    for _ in range(3):
        cap.go.release()
    # All three arrived before we asked: only the newest comes back, once.
    deadline = 50
    while cap.i < 3 and deadline:
        threading.Event().wait(0.01)
        deadline -= 1
    frame = reader.next(timeout=1.0)
    assert frame is not None and frame[0, 0] == 3
    assert reader.next(timeout=0.05) is None
    cap.go.release()  # end of stream
    assert reader.next(timeout=1.0) is None
    assert not reader.alive
    reader.stop()


def test_reader_waits_for_new_frame():
    cap = FakeCap(5)
    reader = LatestFrameReader(cap)
    assert reader.next(timeout=0.05) is None
    cap.go.release()
    frame = reader.next(timeout=1.0)
    assert frame is not None and frame[0, 0] == 1
    cap.n = 0
    cap.go.release()
    reader.stop()


def test_settings_defaults_without_file(tmp_path):
    s = CameraSettings.load(tmp_path / "none.toml")
    assert s == CameraSettings(source=0, width=1920, height=1080, fps=None, focal_35mm=24.0)
    assert s.lens is None


def test_settings_round_trip_device(tmp_path):
    path = tmp_path / "camera.toml"
    CameraSettings(source=1, width=3840, height=2160, fps=30.0, focal_35mm=26.0).save(path)
    assert CameraSettings.load(path) == CameraSettings(source=1, width=3840, height=2160, fps=30.0, focal_35mm=26.0)


def test_settings_round_trip_stream(tmp_path):
    path = tmp_path / "camera.toml"
    CameraSettings(source="http://192.168.0.217:8080/video").save(path)
    assert "width =" not in path.read_text()  # the phone app sets the stream size
    assert CameraSettings.load(path).source == "http://192.168.0.217:8080/video"


def test_settings_override_keeps_saved_values():
    s = CameraSettings(source=1, width=3840, height=2160).override(source=None, width=1280, height=None, fps=None)
    assert s == CameraSettings(source=1, width=1280, height=2160)


def _capture_open(monkeypatch):
    """Stop the live command at the camera open and record its arguments."""
    from battlesight import app

    seen = {}

    def fake_open(args):
        seen.update(source=args.source, width=args.width, height=args.height, fps=args.fps)
        return None

    monkeypatch.setattr(app, "_open_camera", fake_open)
    return seen


def test_cli_uses_saved_settings(tmp_path, monkeypatch):
    from battlesight.app import main

    path = tmp_path / "camera.toml"
    CameraSettings(source="http://192.168.0.217:8080/video").save(path)
    seen = _capture_open(monkeypatch)
    assert main(["--camera-file", str(path)]) == 2
    assert seen["source"] == "http://192.168.0.217:8080/video"


def test_cli_options_override_saved_settings(tmp_path, monkeypatch):
    from battlesight.app import main

    path = tmp_path / "camera.toml"
    CameraSettings(source=1, width=3840, height=2160).save(path)
    seen = _capture_open(monkeypatch)
    main(["--camera-file", str(path), "--source", "0"])
    assert seen == {"source": 0, "width": 3840, "height": 2160, "fps": None}


def test_cli_without_file_uses_built_in_defaults(tmp_path, monkeypatch):
    from battlesight.app import main

    seen = _capture_open(monkeypatch)
    main(["--camera-file", str(tmp_path / "none.toml")])
    assert seen == {"source": 0, "width": 1920, "height": 1080, "fps": None}


def test_list_cameras_save_keeps_lens(tmp_path):
    """Saving a new source in tools/list_cameras.py keeps the focal length and the lens."""
    import importlib.util
    from pathlib import Path

    from battlesight.calib import Lens

    spec = importlib.util.spec_from_file_location("list_cameras", Path(__file__).parents[1] / "tools" / "list_cameras.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    path = tmp_path / "camera.toml"
    K = np.array([[3000.0, 0, 1920], [0, 3000.0, 1080], [0, 0, 1]])
    CameraSettings(source=1, focal_35mm=26.0).with_lens(Lens(K, np.zeros(5), (3840, 2160), "calibrated", 0.2), 14).save(path)

    class Args:
        no_save = False
        camera_file = str(path)

    assert tool._save(Args, source="http://192.168.0.217:8080/video") == 0
    s = CameraSettings.load(path)
    assert s.source == "http://192.168.0.217:8080/video" and s.focal_35mm == 26.0
    assert s.lens is not None and s.lens.size == (3840, 2160) and s.lens_views == 14
