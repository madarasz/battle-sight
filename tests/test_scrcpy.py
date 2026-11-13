"""scrcpy source: source text, server file, arguments, adb errors. No phone needed."""

import subprocess
import zipfile

import pytest

from battlesight import scrcpy
from battlesight.camera import Camera, CameraSettings, is_stream, parse_source
from battlesight.scrcpy import ScrcpyError, ScrcpyServer, is_scrcpy, serial_of, server_args, server_version


def _fake_server(path, version="5.0.1"):
    with zipfile.ZipFile(path, "w") as jar:
        jar.writestr("classes.dex", b"\x00junk\x00The server version (" + version.encode() + b") does not match the client (\x00")
    return path


def test_source_text():
    assert is_scrcpy("scrcpy") and is_scrcpy("scrcpy:VKX123")
    assert not is_scrcpy(1) and not is_scrcpy("http://x/video") and not is_scrcpy("scrcpy.mp4")
    assert serial_of("scrcpy") is None and serial_of("scrcpy:VKX123") == "VKX123"
    assert parse_source("scrcpy") == "scrcpy"
    assert not is_stream("scrcpy")
    cam = Camera("scrcpy:VKX123", 1920, 1080)
    assert cam.scrcpy and not cam.stream
    assert not Camera(0).scrcpy and not Camera("http://x/video").scrcpy


def test_settings_keep_camera_size(tmp_path):
    """The size of a scrcpy source is the camera size asked from the phone: it is saved."""
    path = tmp_path / "camera.toml"
    CameraSettings(source="scrcpy", width=3840, height=2160, fps=30.0).save(path)
    assert CameraSettings.load(path) == CameraSettings(source="scrcpy", width=3840, height=2160, fps=30.0)


def test_server_version(tmp_path):
    assert server_version(_fake_server(tmp_path / "scrcpy-server", "5.0.1")) == "5.0.1"
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ScrcpyError):
        server_version(bad)


def test_find_server_env(tmp_path):
    jar = _fake_server(tmp_path / "srv")
    assert scrcpy.find_server({"SCRCPY_SERVER_PATH": str(jar)}) == jar
    with pytest.raises(ScrcpyError, match="no such file"):
        scrcpy.find_server({"SCRCPY_SERVER_PATH": str(tmp_path / "none")})


def test_server_args():
    args = server_args("5.0.1", "0000abcd", 1920, 1080, 30.0)
    assert args[0] == "5.0.1"
    for a in ("scid=0000abcd", "tunnel_forward=true", "audio=false", "control=false", "video_source=camera",
              "raw_stream=true", "camera_size=1920x1080", "camera_fps=30"):
        assert a in args
    rest = server_args("5.0.1", "1", None, None, None)
    assert not any(a.startswith(("camera_size", "camera_fps")) for a in rest)


class FakeAdb:
    """Records adb calls. `fail` maps an adb command to its stderr."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def run(self, cmd, **kw):
        self.calls.append(cmd)
        sub = next(c for c in cmd[1:] if not c.startswith("-") and c not in ("VKX",))
        if sub in self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", self.fail[sub])
        return subprocess.CompletedProcess(cmd, 0, "51234\n" if sub == "forward" else "", "")


def _patch(monkeypatch, tmp_path, adb):
    monkeypatch.setattr(scrcpy, "find_adb", lambda: "adb")
    monkeypatch.setattr(scrcpy, "find_server", lambda: _fake_server(tmp_path / "scrcpy-server"))
    monkeypatch.setattr(scrcpy.subprocess, "run", adb.run)


def test_start_without_device_gives_hint(monkeypatch, tmp_path):
    adb = FakeAdb(fail={"push": "adb: no devices/emulators found"})
    _patch(monkeypatch, tmp_path, adb)
    with pytest.raises(ScrcpyError, match="USB debugging"):
        ScrcpyServer("scrcpy").start()
    cam = Camera("scrcpy", 1920, 1080)
    assert not cam.open()
    assert "USB debugging" in cam.error


def test_start_forwards_port_with_serial(monkeypatch, tmp_path):
    adb = FakeAdb()
    _patch(monkeypatch, tmp_path, adb)
    started = []

    class FakeProc:
        stdout = iter(["[server] INFO: Device: phone"])

        def __init__(self, cmd, **kw):
            started.append(cmd)

        def poll(self):
            return 0

    monkeypatch.setattr(scrcpy.subprocess, "Popen", FakeProc)
    server = ScrcpyServer("scrcpy:VKX", 1920, 1080)
    assert server.start() == "tcp://127.0.0.1:51234"
    push, forward = adb.calls[:2]
    assert push[:3] == ["adb", "-s", "VKX"] and push[3] == "push"
    assert forward[3:5] == ["forward", "tcp:0"] and forward[5].startswith("localabstract:scrcpy_")
    scid = forward[5].removeprefix("localabstract:scrcpy_")
    assert f"scid={scid}" in started[0] and "camera_size=1920x1080" in started[0]
    server.stop()
    assert adb.calls[-1][3:] == ["forward", "--remove", "tcp:51234"]
    assert server.port is None
