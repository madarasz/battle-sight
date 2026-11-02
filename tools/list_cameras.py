"""Probe camera indices and report the resolution each one delivers.

    uv run tools/list_cameras.py [--max 6] [--width 3840 --height 2160]

macOS asks for camera permission on the first run. Camo shows up as an extra index.
"""

from __future__ import annotations

import argparse
import sys

import cv2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    args = ap.parse_args(argv)
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    found = 0
    for idx in range(args.max):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        found += 1
        ok, frame = cap.read()
        default = frame.shape[1::-1] if ok and frame is not None else ("?", "?")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = cap.read()
        got = frame.shape[1::-1] if ok and frame is not None else ("?", "?")
        print(f"index {idx}: default {default[0]}x{default[1]}, after request {args.width}x{args.height}: {got[0]}x{got[1]}, fps {cap.get(cv2.CAP_PROP_FPS):.0f}")
        cap.release()
    if not found:
        print("no camera opened. Check System Settings > Privacy & Security > Camera for your terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
