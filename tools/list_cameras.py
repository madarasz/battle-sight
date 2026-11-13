"""Find the camera, check it, and save it as the default for battlesight.

    uv run tools/list_cameras.py [--max 6] [--width 3840 --height 2160]   list device indices
    uv run tools/list_cameras.py --use 1 [--width 3840 --height 2160]     check index 1 and save it
    uv run tools/list_cameras.py --url http://PHONE-IP:8080/video         check a stream and save it
    uv run tools/list_cameras.py --scrcpy [SERIAL] --width 1920 --height 1080   Android phone over USB

--use, --url and --scrcpy write the [camera] table of data/camera.toml (--camera-file to change).
The other values in the file (focal_35mm, the [lens] calibration) stay. The live commands
(battlesight, enroll, calibrate) then use the saved values. --source overrides the source
for one run. --no-save only checks.

macOS asks for camera permission on the first run. Camo shows up as an extra index.
"out device of bound (0-0): 1" warnings mean OpenCV has only index 0: no extra camera.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from battlesight.camera import DEFAULT_CAMERA_FILE, Camera, CameraSettings


def probe_index(idx: int, width: int, height: int):
    """(default size, size after the request, fps), or None when the index does not open."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    default = tuple(frame.shape[1::-1]) if ok and frame is not None else None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    got = tuple(frame.shape[1::-1]) if ok and frame is not None else None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return default, got, fps


def _size(s) -> str:
    return f"{s[0]}x{s[1]}" if s else "?x?"


def list_indices(args) -> int:
    found = 0
    for idx in range(args.max):
        res = probe_index(idx, args.width, args.height)
        if res is None:
            continue
        found += 1
        default, got, fps = res
        print(f"index {idx}: default {_size(default)}, after request {args.width}x{args.height}: {_size(got)}, fps {fps:.0f}")
    if not found:
        print("no camera opened. Check System Settings > Privacy & Security > Camera for your terminal.")
        return 1
    print("save one as the default: uv run tools/list_cameras.py --use INDEX [--width W --height H]")
    return 0


def use_index(args) -> int:
    res = probe_index(args.use, args.width, args.height)
    if res is None or res[1] is None:
        print(f"index {args.use}: cannot open or no frame. Run the script without --use to list the cameras.")
        return 1
    _, got, fps = res
    print(f"index {args.use}: {_size(got)}, fps {fps:.0f} (requested {args.width}x{args.height})")
    if got != (args.width, args.height):
        print(f"the camera gives {_size(got)}, not {args.width}x{args.height}. Saving the size it gives.")
    return _save(args, source=args.use, width=got[0], height=got[1], fps=args.fps)


def use_url(args) -> int:
    cam = Camera(args.url)
    if not cam.open():
        print(f"cannot open {args.url}. Check the URL in a browser, the Wi-Fi, the VPN, and Local Network access.")
        return 1
    if _measure(cam, args.url) is None:
        return 1
    return _save(args, source=args.url)


def use_scrcpy(args) -> int:
    source = f"scrcpy:{args.scrcpy}" if args.scrcpy else "scrcpy"
    cam = Camera(source, args.width, args.height, args.fps)
    if not cam.open():
        print(f"cannot open {source}: {cam.error}")
        return 1
    if _measure(cam, source) is None:
        return 1
    # Save the camera size that the phone accepted, not the frame size: the frame can be turned.
    return _save(args, source=source, width=args.width, height=args.height, fps=args.fps)


def _measure(cam: Camera, name: str, warmup: float = 1.0, span: float = 2.0):
    """Read an open camera for some seconds and print the size and rate. The frame size, or None.
    The first second is not counted: a stream can give its start-up frames in a burst."""
    t0 = time.perf_counter()
    got = 0
    shape = None
    start = None
    while time.perf_counter() - t0 < warmup + span + 3.0:
        frame = cam.read()
        now = time.perf_counter()
        if frame is None:
            continue
        shape = frame.shape[1::-1]
        if now - t0 < warmup:
            continue
        if start is None:
            start = now
        else:
            got += 1
        if now - start >= span:
            break
    cam.release()
    if shape is None:
        print(f"{name}: opened, but no frames arrived")
        return None
    rate = got / (time.perf_counter() - start) if start is not None and got else 0.0
    print(f"{name}: {shape[0]}x{shape[1]}, {rate:.1f} fps measured")
    return shape


def _save(args, **camera) -> int:
    """Write [camera] to the camera file. The focal length and the [lens] table stay."""
    if args.no_save:
        return 0
    old = CameraSettings.load(args.camera_file)
    settings = old.override(**camera)
    settings.save(args.camera_file)
    print(f"saved to {args.camera_file}. Now run: uv run battlesight")
    if old.lens is not None and old.source != settings.source:
        print(
            f"note: the file keeps the lens calibration from {old.lens_date or 'before'} "
            f"({old.lens.size[0]}x{old.lens.size[1]}, source {old.source}). "
            "If it is for another camera, run: uv run battlesight calibrate"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=6, help="device indices to try when listing")
    ap.add_argument("--width", type=int, default=3840, help="frame width to request from a device")
    ap.add_argument("--height", type=int, default=2160, help="frame height to request from a device")
    ap.add_argument("--fps", type=float, default=None, help="frame rate to save with --use or --scrcpy")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--use", type=int, metavar="INDEX", help="check this device index and save it")
    pick.add_argument("--url", help="check this network stream and save it")
    pick.add_argument("--scrcpy", nargs="?", const="", metavar="SERIAL", help="check an Android phone camera over USB (scrcpy server) and save it; SERIAL from 'adb devices' when more than one is connected")
    ap.add_argument("--no-save", action="store_true", help="only check, do not write the camera file")
    ap.add_argument("--camera-file", default=str(DEFAULT_CAMERA_FILE), help="camera settings file to write")
    args = ap.parse_args(argv)
    if args.url:
        return use_url(args)
    if args.scrcpy is not None:
        return use_scrcpy(args)
    if args.use is not None:
        return use_index(args)
    return list_indices(args)


if __name__ == "__main__":
    sys.exit(main())
