"""Command line entry point.

    battlesight corners [--source 0] [--width 3840 --height 2160]
    battlesight rectify [--source 0] [--width 3840 --height 2160]
    battlesight calibrate [--source 0 | --images PATH...] [--camera-file data/camera.toml]

Everything about the camera is in data/camera.toml (--camera-file): the live
commands take --source, --width, --height and --fps from it when the options are
not given (tools/list_cameras.py saves them), and the lens from its [lens] table
(battlesight calibrate writes it).
    battlesight image PATH... [--until STAGE] [--out DIR] [--json FILE|-] [--show]

corners: opens the camera, detects the four corner tags on every frame, draws
them, and warns when a tag is lost or the board moved.
Keys: q / Esc quit, r re-lock reference, s save snapshot.

rectify: live top-down board view with the grid. Keys: q quit, v camera or
top-down view, g grid on/off, r re-lock reference, s save snapshot.

calibrate: lens calibration with the ChArUco sheet (tools/make_charuco.py),
from the live camera (space grab view, c calibrate and save, q quit) or from
saved images.

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
from .calib import CharucoFinder, DriftMonitor, DriftStatus, TagDetector, board_corners_px, calibrate_lens, lens_for_frame
from .camera import DEFAULT_CAMERA_FILE, Camera, CameraSettings, parse_source
from .config import DEFAULT_CONFIG, BoardConfig
from .overlay import GREEN, RED, YELLOW, draw_banner, draw_board_corners, draw_grid, draw_tags, draw_text_banner, fit_to_width
from .pipeline import SCHEMA_VERSION, Pipeline, stage_names
from .rectify import fit_view, mm_to_rect, warp_board

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _add_camera_args(p: argparse.ArgumentParser) -> None:
    saved = "saved in --camera-file"
    p.add_argument("--source", default=None, help=f"camera index, or stream URL such as http://PHONE-IP:8080/video (default: {saved}, else 0)")
    p.add_argument("--width", type=int, default=None, help=f"requested frame width (default: {saved}, else 1920)")
    p.add_argument("--height", type=int, default=None, help=f"requested frame height (default: {saved}, else 1080)")
    p.add_argument("--fps", type=float, default=None, help=f"requested frame rate (default: {saved}, else camera default)")
    p.add_argument("--camera-file", default=str(DEFAULT_CAMERA_FILE), help="camera settings and lens calibration (tools/list_cameras.py, calibrate)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.add_argument("--display-width", type=int, default=1600, help="window width on screen")


def cmd_corners(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    detector = TagDetector(cfg)
    monitor = DriftMonitor(cfg)
    cam = _open_camera(args)
    if cam is None:
        return 2
    w, h = cam.size
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
            _save_snapshot(frame)
    cam.release()
    cv2.destroyAllWindows()
    return 0


def _apply_camera_settings(args: argparse.Namespace) -> None:
    """Fill the camera options that the command line left out from the saved settings."""
    path = Path(args.camera_file)
    settings = CameraSettings.load(path).override(
        source=None if args.source is None else parse_source(args.source),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    if path.exists():
        print(f"camera settings: {path} (command options override them)")
    args.camera = settings
    args.source, args.width, args.height, args.fps = settings.source, settings.width, settings.height, settings.fps


def _open_camera(args: argparse.Namespace) -> Camera | None:
    cam = Camera(args.source, args.width, args.height, args.fps)
    if not cam.open():
        print(f"cannot open camera source {args.source!r}. See docs/setup.md, Troubleshooting.", file=sys.stderr)
        return None
    w, h = cam.size
    asked = "size set by the stream" if cam.stream else f"requested {args.width}x{args.height}"
    print(f"camera {args.source}: {w}x{h} @ {cam.reported_fps:.0f} fps ({asked})")
    return cam


def _save_snapshot(frame) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / time.strftime("snap_%Y%m%d_%H%M%S.png")
    cv2.imwrite(str(path), frame)
    print(f"saved {path}")


def cmd_rectify(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    detector = TagDetector(cfg)
    monitor = DriftMonitor(cfg)
    cam = _open_camera(args)
    if cam is None:
        return 2
    lens = lens_for_frame(args.camera, cam.size)
    print(f"lens: {lens.source}, fx {lens.K[0, 0]:.0f} px")
    print("keys: q quit, v camera/top-down view, g grid, r re-lock reference, s snapshot")
    window = "Battle Sight - rectify"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    top_down, grid = True, True
    view = None
    last_status = None
    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(100) in (27, ord("q")):
                break
            continue
        obs = detector.detect(frame)
        report = monitor.update(obs)
        if report.status is not last_status:
            print(report.summary())
            last_status = report.status
        # The locked reference gives a still homography. A moved board falls
        # back to the current frame until the user re-locks with r.
        tags = monitor.reference if report.status is DriftStatus.OK else obs
        if tags and all(i in tags for i in cfg.tag_ids):
            try:
                view = fit_view(cfg, tags, lens)
            except ValueError as exc:
                print(f"rectify: {exc}", file=sys.stderr)
        text = report.summary()
        color = {DriftStatus.OK: GREEN, DriftStatus.LOST: RED}.get(report.status, YELLOW)
        if view is not None:
            text += f"  fit max {view.reproj_max_mm:.2f} mm"
        if top_down and view is not None:
            out = warp_board(frame, view, cfg)
            if grid:
                draw_grid(out, cfg, lambda p: mm_to_rect(cfg, p))
        else:
            out = frame.copy()
            if grid and view is not None:
                draw_grid(out, cfg, view.mm_to_px)
        draw_text_banner(out, text, color, f"{lens.source} lens")
        cv2.imshow(window, fit_to_width(out, args.display_width))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("v"):
            top_down = not top_down
        if key == ord("g"):
            grid = not grid
        if key == ord("r"):
            monitor.reset()
            print("reference reset, locking again")
        if key == ord("s"):
            _save_snapshot(frame)
    cam.release()
    cv2.destroyAllWindows()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    finder = CharucoFinder()
    views = []
    size = None
    if args.images:
        images, errors = collect_images(args.images)
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        if errors or not images:
            return 2
        for path in images:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"error: {path}: cannot read image", file=sys.stderr)
                return 2
            h, w = frame.shape[:2]
            if size is not None and size != (w, h):
                print(f"error: {path}: size {w}x{h}, other images {size[0]}x{size[1]}", file=sys.stderr)
                return 2
            size = (w, h)
            found = finder.detect(frame)
            print(f"{path}: {0 if found is None else len(found[0])} corners")
            if found is not None:
                views.append(found)
    else:
        cam = _open_camera(args)
        if cam is None:
            return 2
        size = cam.size
        print("keys: space grab view, c calibrate and save, q quit")
        window = "Battle Sight - calibrate"
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(100) in (27, ord("q")):
                    break
                continue
            found = finder.detect(frame)
            out = frame.copy()
            if found is not None:
                for p in found[1]:
                    cv2.circle(out, (int(p[0]), int(p[1])), max(3, out.shape[1] // 500), GREEN, -1)
            n = 0 if found is None else len(found[0])
            draw_text_banner(out, f"{len(views)} views, {n} corners in frame", GREEN if found is not None else YELLOW, "space grab  c calibrate  q quit")
            cv2.imshow(window, fit_to_width(out, args.display_width))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found is not None:
                views.append(found)
                print(f"view {len(views)}: {n} corners")
            if key == ord("c") or key in (27, ord("q")):
                break
        cam.release()
        cv2.destroyAllWindows()
    if len(views) < max(3, args.min_views):
        print(f"need {max(3, args.min_views)} views with the target, got {len(views)}", file=sys.stderr)
        return 1
    lens = calibrate_lens(views, size)
    # Load the file again: only [lens] changes, not the camera options of this run.
    CameraSettings.load(args.camera_file).with_lens(lens, len(views)).save(args.camera_file)
    print(f"rms {lens.rms_px:.3f} px, fx {lens.K[0, 0]:.1f} fy {lens.K[1, 1]:.1f}, dist {lens.dist.round(4).tolist()}")
    print(f"wrote [lens] to {args.camera_file}")
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
    pipe = Pipeline(cfg, args.until, CameraSettings.load(args.camera_file))
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
            views = pipe.views(frame, result)
            if out_dir:
                cv2.imwrite(str(out_dir / f"{path.stem}.overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                for name, img in views.items():
                    cv2.imwrite(str(out_dir / f"{path.stem}.{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out_dir / f"{path.stem}.json").write_text(json.dumps(result.to_dict(), indent=2))
            if args.show:
                cv2.imshow(window, fit_to_width(overlay, args.display_width))
                for name, img in views.items():
                    cv2.imshow(f"{window} - {name}", fit_to_width(img, args.display_width))
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

    p = sub.add_parser("rectify", help="live top-down board view with the grid")
    _add_camera_args(p)
    p.set_defaults(func=cmd_rectify)

    p = sub.add_parser("calibrate", help="lens calibration with the ChArUco sheet")
    _add_camera_args(p)
    p.add_argument("--images", nargs="+", default=None, help="calibrate from image files or folders instead of the camera")
    p.add_argument("--min-views", type=int, default=10, help="fewest views with the target")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("image", help="run the analysis pipeline on saved photos")
    p.add_argument("inputs", nargs="+", help="image files or folders of images")
    p.add_argument("--until", choices=stage_names(), default=None, help="last stage to run (default: all stages)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.add_argument("--camera-file", default=str(DEFAULT_CAMERA_FILE), help="camera settings with the lens calibration ([lens])")
    p.add_argument("--out", default=None, help="folder for <name>.overlay.jpg and <name>.json per image")
    p.add_argument("--json", default=None, metavar="FILE", help="write the combined JSON report to FILE, or - for stdout")
    p.add_argument("--show", action="store_true", help="show each overlay in a window (any key next, q quit)")
    p.add_argument("--display-width", type=int, default=1600, help="window width on screen")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-image status lines")
    p.set_defaults(func=cmd_image)
    args = parser.parse_args(argv)
    if hasattr(args, "source"):
        _apply_camera_settings(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
