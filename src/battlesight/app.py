"""Command line entry point.

    battlesight corners [--source 0] [--width 3840 --height 2160]
    battlesight image PATH... [--until STAGE] [--out DIR] [--json FILE|-] [--show]

corners: opens the camera, detects the four corner tags on every frame, draws
them, and warns when a tag is lost or the board moved.
Keys: q / Esc quit, r re-lock reference, s save snapshot.

image: runs the analysis pipeline on saved photos (files or folders). Writes a
JSON report and overlay images. Exit code 0 when every stage passed on every
image, 1 when a stage failed, 2 when an input could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from . import __version__
from .calib import DriftMonitor, TagDetector, board_corners_px
from .camera import Camera, parse_source
from .config import DEFAULT_CONFIG, BoardConfig
from .overlay import draw_banner, draw_board_corners, draw_tags, fit_to_width
from .pipeline import SCHEMA_VERSION, Pipeline, stage_names

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="0", help="camera index or stream URL (default 0)")
    p.add_argument("--width", type=int, default=1920, help="requested frame width")
    p.add_argument("--height", type=int, default=1080, help="requested frame height")
    p.add_argument("--fps", type=float, default=None, help="requested frame rate")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.add_argument("--display-width", type=int, default=1600, help="window width on screen")


def cmd_corners(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    detector = TagDetector(cfg)
    monitor = DriftMonitor(cfg)
    cam = Camera(parse_source(args.source), args.width, args.height, args.fps)
    if not cam.open():
        print(f"cannot open camera source {args.source!r}", file=sys.stderr)
        return 2
    w, h = cam.size
    print(f"camera {args.source}: {w}x{h} @ {cam.reported_fps:.0f} fps (requested {args.width}x{args.height})")
    print(f"board {cfg.name} {cfg.width_mm:.0f}x{cfg.height_mm:.0f} mm, tags {cfg.tag_ids}, {cfg.tag_size_mm:.0f} mm")
    print("keys: q quit, r re-lock reference, s snapshot")

    window = "Battle Sight - corners"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    t_last = time.perf_counter()
    fps = 0.0
    detect_ms = 0.0
    last_status = None
    frames = 0
    while True:
        frame = cam.read()
        if frame is None:
            print("camera: no frame, retrying...", file=sys.stderr)
            if cv2.waitKey(100) in (27, ord("q")):
                break
            continue
        frames += 1
        t0 = time.perf_counter()
        obs = detector.detect(frame)
        detect_ms = 0.9 * detect_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
        report = monitor.update(obs)
        corners = board_corners_px(cfg, obs)
        ref_corners = board_corners_px(cfg, monitor.reference) if monitor.reference else None

        if report.status is not last_status:
            print(f"[{frames:6d}] {report.summary()}")
            last_status = report.status

        draw_tags(frame, obs, cfg, report)
        draw_board_corners(frame, corners, ref_corners)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now
        draw_banner(frame, report, f"{w}x{h}  {fps:4.1f} fps  detect {detect_ms:4.1f} ms")
        cv2.imshow(window, fit_to_width(frame, args.display_width))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            monitor.reset()
            print("reference reset, locking again")
        if key == ord("s"):
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            path = SNAPSHOT_DIR / time.strftime("snap_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
    cam.release()
    cv2.destroyAllWindows()
    return 0


def collect_images(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """Expand files and folders (not recursive) into a sorted image list.
    Returns (images, errors)."""
    images: list[Path] = []
    errors: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES)
            if not found:
                errors.append(f"{p}: no images in folder")
            images.extend(found)
        elif p.is_file():
            images.append(p)
        else:
            errors.append(f"{p}: not found")
    return images, errors


def cmd_image(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    pipe = Pipeline(cfg, args.until)
    images, errors = collect_images(args.inputs)
    # Keep stdout clean for the JSON report when it goes there.
    log = sys.stderr if args.json == "-" else sys.stdout
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    window = "Battle Sight - image"
    reports = []
    any_failed = False
    for path in images:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)  # applies EXIF orientation
        if frame is None:
            errors.append(f"{path}: cannot read image")
            continue
        result = pipe.analyze(frame, source=str(path))
        reports.append(result.to_dict())
        any_failed |= not result.ok
        if not args.quiet:
            print(f"{'ok  ' if result.ok else 'FAIL'} {path}  {result.summary()}", file=log)
        if out_dir or args.show:
            overlay = pipe.annotate(frame, result)
            if out_dir:
                cv2.imwrite(str(out_dir / f"{path.stem}.overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out_dir / f"{path.stem}.json").write_text(json.dumps(result.to_dict(), indent=2))
            if args.show:
                cv2.imshow(window, fit_to_width(overlay, args.display_width))
                if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                    break
    if args.show:
        cv2.destroyAllWindows()
    for e in errors:
        print(f"error: {e}", file=sys.stderr)

    report = {
        "schema": SCHEMA_VERSION,
        "config": str(args.config),
        "stages": pipe.names,
        "ok": not any_failed and not errors,
        "errors": errors,
        "images": reports,
    }
    if args.json == "-":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    if errors or not images:
        return 2
    return 1 if any_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="battlesight", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"battlesight {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("corners", help="track the four AprilTag board corners")
    _add_camera_args(p)
    p.set_defaults(func=cmd_corners)

    p = sub.add_parser("image", help="run the analysis pipeline on saved photos")
    p.add_argument("inputs", nargs="+", help="image files or folders of images")
    p.add_argument("--until", choices=stage_names(), default=None, help="last stage to run (default: all stages)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.add_argument("--out", default=None, help="folder for <name>.overlay.jpg and <name>.json per image")
    p.add_argument("--json", default=None, metavar="FILE", help="write the combined JSON report to FILE, or - for stdout")
    p.add_argument("--show", action="store_true", help="show each overlay in a window (any key next, q quit)")
    p.add_argument("--display-width", type=int, default=1600, help="window width on screen")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-image status lines")
    p.set_defaults(func=cmd_image)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
