"""Android phone camera over USB through the scrcpy server, no scrcpy window.

The scrcpy desktop app only shows the video in its own window, so other apps cannot read
it on macOS. Its device part, `scrcpy-server`, can send plain H.264 (`raw_stream=true`)
over an adb socket. `ScrcpyServer` pushes the server to the phone, forwards a local TCP
port to its socket, and starts it with the camera as video source. OpenCV (FFmpeg) then
reads `tcp://127.0.0.1:<port>` like any other stream.

Source text: `scrcpy` (the one adb device) or `scrcpy:SERIAL` (one of more devices,
serial from `adb devices`). Needs `adb` and the `scrcpy-server` file of an installed
scrcpy (Homebrew: `brew install scrcpy`). `ADB` and `SCRCPY_SERVER_PATH` override where
they are found, as for scrcpy itself.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import threading
import zipfile
from collections import deque
from pathlib import Path

SCHEME = "scrcpy"
DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
VIDEO_BIT_RATE = 20_000_000  # scrcpy default is 8 Mbps; USB carries more, less blur on bases
ADB_TIMEOUT_S = 15.0
SERVER_SEARCH = (
    "/opt/homebrew/share/scrcpy/scrcpy-server",
    "/usr/local/share/scrcpy/scrcpy-server",
    "/usr/share/scrcpy/scrcpy-server",
)


class ScrcpyError(RuntimeError):
    """The scrcpy source cannot start. The message tells the user what to do."""


def is_scrcpy(source: int | str) -> bool:
    return isinstance(source, str) and (source == SCHEME or source.startswith(SCHEME + ":"))


def serial_of(source: str) -> str | None:
    """'scrcpy' -> None (the one device), 'scrcpy:ABC' -> 'ABC'."""
    _, _, serial = source.partition(":")
    return serial or None


def find_adb() -> str:
    adb = os.environ.get("ADB") or shutil.which("adb")
    if not adb:
        raise ScrcpyError("adb not found. Install scrcpy (brew install scrcpy), it brings adb.")
    return adb


def find_server(env: dict[str, str] | None = None) -> Path:
    """The scrcpy-server file: SCRCPY_SERVER_PATH, next to the scrcpy program, or a known place."""
    env = os.environ if env is None else env
    if env.get("SCRCPY_SERVER_PATH"):
        path = Path(env["SCRCPY_SERVER_PATH"])
        if not path.is_file():
            raise ScrcpyError(f"SCRCPY_SERVER_PATH={path}: no such file")
        return path
    places = []
    exe = shutil.which("scrcpy")
    if exe:
        places.append(Path(exe).resolve().parents[1] / "share" / "scrcpy" / "scrcpy-server")
    places += [Path(p) for p in SERVER_SEARCH]
    for path in places:
        if path.is_file():
            return path
    raise ScrcpyError("scrcpy-server not found. Install scrcpy (brew install scrcpy) or set SCRCPY_SERVER_PATH.")


def server_version(path: Path) -> str:
    """Version of a scrcpy-server file. The server starts only when it gets its own version."""
    try:
        with zipfile.ZipFile(path) as jar:
            dex = jar.read("classes.dex")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ScrcpyError(f"{path}: not a scrcpy-server file ({exc})") from exc
    m = re.search(rb"The server version \(([0-9][^)]*)\)", dex)
    if not m:
        raise ScrcpyError(f"{path}: cannot read the server version")
    return m.group(1).decode()


def server_args(version: str, scid: str, width: int | None, height: int | None, fps: float | None) -> list[str]:
    """Arguments of com.genymobile.scrcpy.Server: camera video only, plain H.264, no control."""
    args = [
        version,
        f"scid={scid}",
        "log_level=info",
        "tunnel_forward=true",
        "audio=false",
        "control=false",
        "video_source=camera",
        "camera_facing=back",
        "video_codec=h264",
        f"video_bit_rate={VIDEO_BIT_RATE}",
        "raw_stream=true",
    ]
    if width and height:
        args.append(f"camera_size={width}x{height}")
    if fps:
        args.append(f"camera_fps={int(round(fps))}")
    return args


class ScrcpyServer:
    """One run of scrcpy-server on the phone with a forwarded local port.

    `start()` returns the TCP URL of the H.264 stream. `stop()` ends the server and removes
    the forward. The server accepts one connection: open the URL once per start.
    """

    def __init__(self, source: str, width: int | None = None, height: int | None = None, fps: float | None = None):
        self.serial = serial_of(source)
        self.width, self.height, self.fps = width, height, fps
        self.port: int | None = None
        self._proc: subprocess.Popen | None = None
        self._log: deque[str] = deque(maxlen=30)
        self._adb: str | None = None
        self._scid: str | None = None

    def _adb_cmd(self, *args: str) -> list[str]:
        cmd = [self._adb or find_adb()]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + list(args)

    def _run(self, *args: str) -> str:
        try:
            done = subprocess.run(self._adb_cmd(*args), capture_output=True, text=True, timeout=ADB_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise ScrcpyError(f"adb {args[0]} did not answer in {ADB_TIMEOUT_S:.0f} s. Reconnect the USB cable.") from exc
        if done.returncode != 0:
            raise ScrcpyError(_adb_hint(done.stderr.strip() or done.stdout.strip()))
        return done.stdout

    def start(self) -> str:
        self.stop()
        self._adb = find_adb()
        server = find_server()
        version = server_version(server)
        self._run("push", str(server), DEVICE_SERVER_PATH)
        scid = self._scid = f"{secrets.randbits(31):08x}"
        out = self._run("forward", "tcp:0", f"localabstract:scrcpy_{scid}").strip()
        self.port = int(out.splitlines()[-1]) if out else None
        if not self.port:
            raise ScrcpyError(f"adb forward gave no port: {out!r}")
        cmd = self._adb_cmd(
            "shell", f"CLASSPATH={DEVICE_SERVER_PATH}", "app_process", "/", "com.genymobile.scrcpy.Server",
            *server_args(version, scid, self.width, self.height, self.fps),
        )
        self._log.clear()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._read_log, args=(self._proc,), daemon=True).start()
        return f"tcp://127.0.0.1:{self.port}"

    def _read_log(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            self._log.append(line.rstrip())

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def log(self) -> list[str]:
        return list(self._log)

    def problem(self) -> str:
        """The server error lines, for a failed start."""
        lines = [ln for ln in self._log if "ERROR" in ln or "Exception" in ln or "WARN" in ln]
        text = "\n".join(lines or self.log()[-5:])
        if "camera size" in text.lower() or "camera_size" in text.lower() or "No camera" in text:
            text += "\nList the sizes the phone gives: scrcpy --video-source=camera --list-camera-sizes"
        return text

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._adb is not None and self._scid is not None:
            # The server on the phone ends some seconds after the socket closes. End it now,
            # so a quick restart does not find the camera in use.
            _quiet(self._adb_cmd("shell", "pkill", "-f", f"scid={self._scid}"))
        if self.port is not None and self._adb is not None:
            _quiet(self._adb_cmd("forward", "--remove", f"tcp:{self.port}"))
        self.port = self._scid = None


def _quiet(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, timeout=ADB_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _adb_hint(text: str) -> str:
    text = text.removeprefix("adb: ")
    if "no devices" in text or "not found" in text:
        return f"adb: {text}. Connect the phone with USB and allow USB debugging (docs/setup.md)."
    if "more than one" in text:
        return f"adb: {text}. Use the source scrcpy:SERIAL with a serial from: adb devices"
    if "unauthorized" in text:
        return f"adb: {text}. Unlock the phone and allow USB debugging for this computer."
    return f"adb: {text}"
