"""Command line entry point.

    battlesight corners [--source 0] [--width 3840 --height 2160]
    battlesight rectify [--source 0] [--width 3840 --height 2160]
    battlesight bases [--source 0] [--reference data/snapshots/empty_board.png]
    battlesight ident --team NAME [--team NAME2] [--reference FILE]
    battlesight enroll --team NAME --operative NAME [--base-mm 32] [--images PATH...]
    battlesight gallery --team NAME [--team NAME2]
    battlesight calibrate [--source 0 | --images PATH...] [--camera-file data/camera.toml]

Everything about the camera is in data/camera.toml (--camera-file): the live
commands take --source, --width, --height and --fps from it when the options are
not given (tools/list_cameras.py saves them), and the lens from its [lens] table
(battlesight calibrate writes it).
    battlesight image PATH... [--until STAGE] [--reference FILE] [--team NAME] [--out DIR] [--json FILE|-] [--show]

corners: opens the camera, detects the four corner tags on every frame, draws
them, and warns when a tag is lost or the board moved.
Keys: q / Esc quit, r re-lock reference, s save snapshot.

rectify: live top-down board view with the grid. Keys: q quit, v camera or
top-down view, g grid on/off, r re-lock reference, s save snapshot.

bases: snapshot base detection on the live camera. Keys: space detect, e take
the empty-board reference, c clear it, v camera or top-down view, s save
snapshot, q quit.

ident: snapshot base detection and operative names from the gallery
(data/gallery/<team>/). Keys: space detect and name, click a base then a name
to fix it (the crop goes into the gallery), a accept all names, u undo the last
gallery photo, e empty-board reference, c clear it, v camera or top-down view,
s save snapshot, q quit.

enroll: adds gallery photos of one operative. Put only this operative on the
board (or click its base), push space, turn and move it, push space again.
10 to 20 photos. With --images it takes the crops from saved photos.

gallery: lists the operatives of the teams with their base size and photo
count, and checks each photo against the rest (leave-one-out).

calibrate: lens calibration with the ChArUco sheet (tools/make_charuco.py),
from the live camera (space grab view, c calibrate and save, q quit) or from
saved images.

image: runs the analysis pipeline on saved photos (files or folders). Writes a
JSON report and overlay images. With --reference (an empty-board photo) the
bases stage looks only where the board changed. Exit code 0 when every stage
passed on every image, 1 when a stage failed, 2 when an input could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from . import ident as idt
from .calib import CharucoFinder, DriftMonitor, DriftStatus, TagDetector, board_corners_px, calibrate_lens, lens_for_frame
from .camera import DEFAULT_CAMERA_FILE, Camera, CameraSettings, parse_source
from .config import CORNER_NAMES, DEFAULT_CONFIG, BoardConfig
from .overlay import GREEN, RED, YELLOW, Menu, draw_banner, draw_board_corners, draw_grid, draw_idents, draw_menu, draw_tags, draw_text_banner, fit_to_width
from .pipeline import SCHEMA_VERSION, Pipeline, select_stages, stage_names
from .rectify import fit_view, mm_to_rect, warp_board

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
DEFAULT_REFERENCE = SNAPSHOT_DIR / "empty_board.png"
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


def _add_gallery_args(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("--team", action="append", required=required, default=None, help="team folder under --gallery-root, for the ident stage (repeat for two teams)")
    p.add_argument("--gallery-root", default=str(idt.DEFAULT_GALLERY_ROOT), help="folder with one folder per team")


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


def _save_snapshot(frame) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / time.strftime("snap_%Y%m%d_%H%M%S.png")
    cv2.imwrite(str(path), frame)
    print(f"saved {path}")
    return path


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


def cmd_bases(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    pipe = Pipeline(cfg, None, args.camera)
    ref_path = Path(args.reference)
    if ref_path.is_file():
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        try:
            pipe.set_reference(ref)
            print(f"empty-board reference: {ref_path}")
        except ValueError as exc:
            print(f"{ref_path}: {exc}. Push e on the empty board to take a new one.", file=sys.stderr)
    cam = _open_camera(args)
    if cam is None:
        return 2
    print("keys: space detect, e empty-board reference, c clear reference, v camera/top-down view, s snapshot, q quit")
    window = "Battle Sight - bases"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    top_down = True
    shown = None  # overlay of the last detection, kept until the next one
    result = snap = None
    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(100) in (27, ord("q")):
                break
            continue
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("e"):
            try:
                pipe.set_reference(frame)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ref_path), frame)
                print(f"empty-board reference saved to {ref_path}")
            except ValueError as exc:
                print(f"no reference: {exc}", file=sys.stderr)
        if key == ord("c"):
            pipe.clear_reference()
            print("reference cleared: single-frame detection")
        if key == ord("s"):
            _save_snapshot(frame)
        if key == ord(" "):
            snap = frame.copy()
            result = pipe.analyze(snap, source="camera")
            print(result.summary())
            for b in result.stages["bases"].data.get("bases", []):
                print(f"  {b['diameter_mm']:g} mm at {b['center_mm']} mm, rim {b['coverage']:.0%} visible")
        if key in (ord(" "), ord("v")):
            if key == ord("v"):
                top_down = not top_down
            if result is not None:
                views = pipe.views(snap, result) if top_down else {}
                shown = views["bases"] if "bases" in views else pipe.annotate(snap, result)
                draw_text_banner(shown, result.summary(), GREEN if result.ok else RED, "reference" if pipe.has_reference else "no reference")
        if shown is None:
            out = frame.copy()
            hint = "" if pipe.has_reference else "  (no empty-board reference: push e on the empty board)"
            draw_text_banner(out, "push space to find the bases" + hint, YELLOW)
        else:
            out = shown
        cv2.imshow(window, fit_to_width(out, args.display_width))
    cam.release()
    cv2.destroyAllWindows()
    return 0


def _load_gallery(args: argparse.Namespace, cfg: BoardConfig, log=sys.stdout, quiet: bool = False) -> idt.Gallery | None:
    """Gallery of the --team options, or None (with a message) on an error."""
    try:
        gallery = idt.Gallery.for_teams(args.team, cfg, args.gallery_root)
    except (ValueError, OSError) as exc:
        print(f"error: gallery: {exc}", file=sys.stderr)
        return None
    if not quiet:
        print(f"gallery {', '.join(sorted(gallery.team_dirs))}: {len(gallery.operatives)} operatives, {len(gallery)} photos, model {gallery.embedder.name}", file=log)
        for label, n in gallery.counts().items():
            if n == 0:
                print(f"  {label}: no photos yet (battlesight enroll)", file=log)
    return gallery


def _load_reference(pipe: Pipeline, path: Path) -> None:
    if path.is_file():
        ref = cv2.imread(str(path), cv2.IMREAD_COLOR)
        try:
            pipe.set_reference(ref)
            print(f"empty-board reference: {path}")
        except ValueError as exc:
            print(f"{path}: {exc}. Push e on the empty board to take a new one.", file=sys.stderr)


class _BoardWindow:
    """Shows a snapshot top-down (rectified) or as the camera frame, and maps
    window clicks to board mm. `fit_to_width` scales the image down."""

    def __init__(self, cfg: BoardConfig, display_width: int):
        self.cfg = cfg
        self.display_width = display_width
        self.top_down = True
        self.scale = 1.0
        self.view = None  # rectify.BoardView of the snapshot

    def base_image(self, frame, result) -> np.ndarray:
        self.view = result.scratch.get("view")
        img = result.scratch.get("rectified") if self.top_down else None
        return (img if img is not None else frame).copy()

    def to_px(self, pts_mm):
        if self.top_down:
            return mm_to_rect(self.cfg, pts_mm)
        return self.view.mm_to_px(np.asarray(pts_mm, float).reshape(-1, 2))

    def to_mm(self, x: float, y: float) -> np.ndarray | None:
        px = np.array([[x / self.scale, y / self.scale]])
        if self.top_down:
            return (px / self.cfg.rectify_px_per_mm - self.cfg.rectify_margin_mm)[0]
        return None if self.view is None else self.view.px_to_mm(px)[0]

    def fit(self, img: np.ndarray) -> np.ndarray:
        out = fit_to_width(img, self.display_width)
        self.scale = out.shape[1] / img.shape[1]
        return out


def corner_warning(cfg: BoardConfig, obs: dict) -> str | None:
    """Warning text when the camera does not see all four corner cards, else None."""
    missing = [i for i in cfg.tag_ids if i not in obs]
    if not missing:
        return None
    names = ", ".join(f"{i} {CORNER_NAMES[cfg.corner_index(i)]}" for i in missing)
    return f"CORNERS NOT SEEN: {4 - len(missing)}/4 cards, missing {names}"


def _failed_stage(result) -> str:
    """Short text of the first failed stage of a snapshot."""
    for name, r in result.stages.items():
        if not r.ok and not r.skipped:
            return f"{name} failed: {r.message}"
    return result.summary()


class _CornerCheck:
    """Finds the corner cards on live frames (every `every` frames, to keep the
    window fast) and draws a red warning strip when one is not seen."""

    def __init__(self, cfg: BoardConfig, every: int = 3):
        self.cfg = cfg
        self.detector = TagDetector(cfg)
        self.every = max(1, every)
        self.count = 0
        self.warning: str | None = None

    def update(self, frame) -> str | None:
        if self.count % self.every == 0:
            self.warning = corner_warning(self.cfg, self.detector.detect(frame))
        self.count += 1
        return self.warning

    def draw(self, img, notice: _Notice | None = None):
        """`img` with the strips under the status strip: the corner warning,
        then the last message of `notice`. A copy when something is drawn."""
        strips = []
        if self.warning is not None:
            strips.append((self.warning, RED, "space will fail"))
        if notice is not None and notice.current() is not None:
            strips.append((*notice.current(), ""))
        if not strips:
            return img
        img = img.copy()
        for row, (text, color, extra) in enumerate(strips, start=1):
            draw_text_banner(img, text, color, extra, row=row)
        return img


class _Notice:
    """Last message of a live window. `say` prints it (errors to stderr) and
    keeps it on the screen for `seconds`, so the user sees it without the
    Terminal. `clock` is for tests."""

    def __init__(self, seconds: float = 6.0, clock=time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self.text: str | None = None
        self.color = YELLOW
        self.until = 0.0

    def say(self, text: str, color=YELLOW, error: bool = False) -> None:
        print(text, file=sys.stderr if error else sys.stdout)
        self.text, self.color, self.until = text, color, self.clock() + self.seconds

    def error(self, text: str) -> None:
        self.say(text, RED, error=True)

    def ok(self, text: str) -> None:
        self.say(text, GREEN)

    def current(self) -> tuple[str, tuple[int, int, int]] | None:
        if self.text is None or self.clock() > self.until:
            return None
        return self.text, self.color


def _session(result, gallery: idt.Gallery) -> idt.LabelSession | None:
    found = result.scratch.get("bases")
    if found is None:
        return None
    idents = result.scratch.get("idents") or [idt.Ident(None, float("-inf"), float("-inf")) for _ in found]
    centers = np.array([c.center_mm for c in found]).reshape(-1, 2)
    return idt.LabelSession(centers, [c.diameter_mm for c in found], list(result.scratch.get("base_crops", [])), list(idents), gallery)


def cmd_ident(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    gallery = _load_gallery(args, cfg)
    if gallery is None:
        return 2
    pipe = Pipeline(cfg, None, args.camera, gallery=gallery)
    ref_path = Path(args.reference)
    _load_reference(pipe, ref_path)
    cam = _open_camera(args)
    if cam is None:
        return 2
    print("keys: space detect and name, click a base then a name to fix it, a accept all, u undo, e empty-board reference, c clear reference, v view, s snapshot, q quit")
    window = "Battle Sight - ident"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    win = _BoardWindow(cfg, args.display_width)
    state = {"menu": None, "hover": None, "session": None, "dirty": True}
    notice = _Notice()

    def on_mouse(event, x, y, flags, _):
        sess: idt.LabelSession | None = state["session"]
        menu: Menu | None = state["menu"]
        if event == cv2.EVENT_MOUSEMOVE and menu is not None:
            hover = menu.hit(x, y)
            state["dirty"] |= hover != state["hover"]
            state["hover"] = hover
        if event == cv2.EVENT_RBUTTONDOWN:
            state["menu"] = None
            if sess:
                sess.selected = None
            state["dirty"] = True
        if event != cv2.EVENT_LBUTTONDOWN or sess is None:
            return
        state["dirty"] = True
        if menu is not None:
            row = menu.hit(x, y)
            state["menu"] = None
            if row is not None and sess.selected is not None:
                label = menu.items[row]
                path = sess.choose(label)
                print(f"named {label}: added {path}")
                notice.ok(f"named {idt.split_label(label)[1]}, photo added to the gallery")
                return
        p_mm = win.to_mm(x, y)
        if p_mm is None or sess.pick(p_mm) is None:
            sess.selected = None
            return
        items = sess.menu()
        if not items:
            notice.error(f"no operative with a {sess.diameters_mm[sess.selected]:g} mm base in the gallery: battlesight enroll")
            return
        state["menu"] = Menu(items, (x + 12, y), row_h=max(22, int(28 * win.scale * 1.5)), width=280)

    cv2.setMouseCallback(window, on_mouse)
    check = _CornerCheck(cfg)
    result = snap = None
    shown = None
    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(100) in (27, ord("q")):
                break
            continue
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        sess: idt.LabelSession | None = state["session"]
        if key == ord("e"):
            try:
                pipe.set_reference(frame)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ref_path), frame)
                notice.ok(f"empty-board reference saved to {ref_path}")
            except ValueError as exc:
                notice.error(f"no reference: {exc}")
        if key == ord("c"):
            pipe.clear_reference()
            notice.say("reference cleared: single-frame detection")
        if key == ord("s"):
            notice.ok(f"saved {_save_snapshot(frame).name}")
        if key == ord("v"):
            win.top_down = not win.top_down
            state["menu"] = None
            state["dirty"] = True
        if key in (ord("a"), ord("u")) and sess is None:
            notice.error("push space first")
        if key == ord("a") and sess is not None:
            added = sess.accept_all()
            notice.ok(f"accepted {len(added)} names, added to the gallery")
            state["dirty"] = True
        if key == ord("u") and sess is not None:
            path = sess.undo()
            if path is None:
                notice.error("nothing to undo")
            else:
                notice.say(f"removed {path.name}")
            state["dirty"] = True
        if key == ord(" "):
            snap = frame.copy()
            result = pipe.analyze(snap, source="camera")
            print(result.summary())
            state["session"] = sess = _session(result, gallery)
            state["menu"] = None
            for b in result.stages["ident"].data.get("bases", []):
                name = b["label"] or "unknown"
                print(f"  {name:30s} {b['diameter_mm']:g} mm at {b['center_mm']} mm, score {b['score']}")
            if not result.ok:
                notice.error(_failed_stage(result))
            elif not result.scratch.get("bases"):
                notice.error("no base found")
            elif result.stages["ident"].message in ("no gallery",) or result.stages["ident"].message.startswith("gallery has no photos"):
                notice.error(result.stages["ident"].message)
            state["dirty"] = True
        if result is not None and state["dirty"]:
            img = win.base_image(snap, result)
            if result.ok and sess is not None:
                draw_idents(img, result.scratch["bases"], sess.idents, win.to_px, sess.selected)
            named = 0 if sess is None else sum(x.label is not None for x in sess.idents)
            total = 0 if sess is None else len(sess.idents)
            text = f"{named}/{total} named" if result.ok else result.summary()
            draw_text_banner(img, text, GREEN if result.ok else RED, "click a base to fix its name")
            shown = win.fit(img)
            if state["menu"] is not None:
                state["menu"] = state["menu"].fit((shown.shape[1], shown.shape[0]))
                draw_menu(shown, state["menu"], state["hover"])
            state["dirty"] = False
        if shown is None:
            out = frame.copy()
            hint = "" if pipe.has_reference else "  (no empty-board reference: push e on the empty board)"
            draw_text_banner(out, "push space to find and name the operatives" + hint, YELLOW)
            out = win.fit(out)
        else:
            out = shown
        check.update(frame)
        cv2.imshow(window, check.draw(out, notice))
    cam.release()
    cv2.destroyAllWindows()
    return 0


def _enroll_candidates(result, base_mm: float | None) -> list[int]:
    found = result.scratch.get("bases") or []
    return [i for i, c in enumerate(found) if base_mm is None or c.diameter_mm == base_mm]


def cmd_enroll(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    args.team = [args.team]
    gallery = _load_gallery(args, cfg)
    if gallery is None:
        return 2
    team = args.team[0]
    try:
        name = idt.check_name(args.operative, "operative")
        if args.base_mm is not None and args.base_mm not in cfg.bases_diameters_mm:
            raise ValueError(f"--base-mm {args.base_mm:g} is not in [bases] diameters_mm {list(cfg.bases_diameters_mm)}")
        op = gallery.ensure_operative(team, name, args.base_mm)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    label = op.label
    pipe = Pipeline(cfg, "ident", args.camera)
    if args.images:
        if args.reference:
            _load_reference(pipe, Path(args.reference))
        images, errors = collect_images(args.images)
        added = 0
        for path in images:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                errors.append(f"{path}: cannot read image")
                continue
            result = pipe.analyze(frame, source=str(path))
            if not result.ok:
                print(f"skip {path}: {result.summary()}")
                continue
            base_mm = op.base_mm
            cand = _enroll_candidates(result, base_mm)
            if len(cand) != 1:
                print(f"skip {path}: {len(cand)} bases{'' if base_mm is None else f' of {base_mm:g} mm'}, need exactly one")
                continue
            c = result.scratch["bases"][cand[0]]
            out = gallery.add(label, result.scratch["base_crops"][cand[0]], c.diameter_mm)
            op = gallery.operatives[gallery.index(label)]
            added += 1
            print(f"added {out} ({c.diameter_mm:g} mm at {np.round(c.center_mm, 1).tolist()} mm)")
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"{label}: {gallery.counts()[label]} photos")
        return 2 if errors else (0 if added else 1)

    _load_reference(pipe, Path(args.reference or DEFAULT_REFERENCE))
    cam = _open_camera(args)
    if cam is None:
        return 2
    print(f"enroll {label}: space grab, click the base when there are more, u undo, v view, q quit. Turn and move the model between grabs.")
    window = "Battle Sight - enroll"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    win = _BoardWindow(cfg, args.display_width)
    state = {"result": None, "snap": None, "added": [], "click": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state["result"] is not None:
            state["click"] = win.to_mm(x, y)

    cv2.setMouseCallback(window, on_mouse)
    check = _CornerCheck(cfg)
    notice = _Notice()

    def add(i: int) -> None:
        nonlocal op
        r = state["result"]
        c = r.scratch["bases"][i]
        path = gallery.add(label, r.scratch["base_crops"][i], c.diameter_mm)
        op = gallery.operatives[gallery.index(label)]
        state["added"].append(path)
        state["result"] = None
        print(f"added {path}")
        notice.ok(f"photo {gallery.counts()[label]} added. Turn and move {op.name}, then push space.")

    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(100) in (27, ord("q")):
                break
            continue
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("v"):
            win.top_down = not win.top_down
        if key == ord("u"):
            if state["added"]:
                path = state["added"].pop()
                gallery.remove(path)
                notice.say(f"removed {path.name}: {gallery.counts()[label]} photos")
            else:
                notice.error("nothing to undo")
        if key == ord(" "):
            snap = frame.copy()
            result = pipe.analyze(snap, source="camera")
            state["result"], state["snap"] = (result, snap) if result.ok else (None, None)
            if not result.ok:
                print(result.summary())
                notice.error(_failed_stage(result))
            else:
                cand = _enroll_candidates(result, op.base_mm)
                if len(cand) == 1:
                    add(cand[0])
                elif not cand:
                    notice.error("no base found" + ("" if op.base_mm is None else f" of {op.base_mm:g} mm") + ": the base edge must be visible")
                    state["result"] = None
                else:
                    notice.say(f"{len(cand)} bases: click the base of {op.name}")
        r = state["result"]
        if r is not None and state["click"] is not None:
            p_mm, state["click"] = state["click"], None
            found = r.scratch["bases"]
            cand = _enroll_candidates(r, op.base_mm)
            if p_mm is not None and cand:
                d = [np.linalg.norm(found[i].center_mm - p_mm) for i in cand]
                k = int(np.argmin(d))
                if d[k] <= 0.5 * found[cand[k]].diameter_mm + 5.0:
                    add(cand[k])
                else:
                    notice.error("no base here: click inside a yellow circle")
        if r is not None and state["result"] is not None:
            img = win.base_image(state["snap"], r)
            from .overlay import draw_bases

            draw_bases(img, [r.scratch["bases"][i] for i in _enroll_candidates(r, op.base_mm)], win.to_px, YELLOW)
            draw_text_banner(img, f"click the base of {op.name}", YELLOW, f"{gallery.counts()[label]} photos")
        else:
            img = frame.copy()
            draw_text_banner(img, f"enroll {op.name}: push space", GREEN, f"{gallery.counts()[label]} photos, aim for 10 to 20")
        check.update(frame)
        cv2.imshow(window, check.draw(win.fit(img), notice))
    cam.release()
    cv2.destroyAllWindows()
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    gallery = _load_gallery(args, cfg)
    if gallery is None:
        return 2
    counts = gallery.counts()
    for op in gallery.operatives:
        base = "?" if op.base_mm is None else f"{op.base_mm:g} mm"
        print(f"  {op.label:32s} base {base:7s} {counts[op.label]:3d} photos")
    right, total, misses = gallery.leave_one_out()
    if total:
        print(f"leave-one-out: {right}/{total} photos named right ({right / total:.0%})")
        for true, guess in misses:
            print(f"  {true} looked like {guess}")
    else:
        print("leave-one-out: needs two photos or more of operatives with the same base size")
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
    # Keep stdout clean for the JSON report when it goes there.
    log = sys.stderr if args.json == "-" else sys.stdout
    gallery = None
    runs_ident = "ident" in select_stages(args.until)
    if runs_ident and not args.no_ident and not args.team:
        # No --team: every team folder of the gallery root.
        args.team = idt.list_teams(args.gallery_root) or None
        if args.team is None and not args.quiet:
            print(f"no gallery teams in {args.gallery_root}: the ident stage names nothing (battlesight enroll)", file=log)
    if runs_ident and not args.no_ident and args.team:
        gallery = _load_gallery(args, cfg, log, args.quiet)
        if gallery is None:
            return 2
    pipe = Pipeline(cfg, args.until, CameraSettings.load(args.camera_file), gallery=gallery)
    if args.reference:
        ref = cv2.imread(args.reference, cv2.IMREAD_COLOR)
        if ref is None:
            print(f"error: {args.reference}: cannot read the empty-board reference", file=sys.stderr)
            return 2
        try:
            pipe.set_reference(ref)
        except ValueError as exc:
            print(f"error: {args.reference}: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"empty-board reference: {args.reference}", file=log)
    images, errors = collect_images(args.inputs)
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
        "reference": args.reference,
        "teams": None if gallery is None else sorted(gallery.team_dirs),
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

    p = sub.add_parser("bases", help="find the operative bases on the live camera (snapshot on a key)")
    _add_camera_args(p)
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="empty-board photo: loaded at start, written by the e key")
    p.set_defaults(func=cmd_bases)

    p = sub.add_parser("ident", help="find the bases and name the operatives from the gallery (snapshot on a key)")
    _add_camera_args(p)
    _add_gallery_args(p, required=True)
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="empty-board photo: loaded at start, written by the e key")
    p.set_defaults(func=cmd_ident)

    p = sub.add_parser("enroll", help="add gallery photos of one operative")
    _add_camera_args(p)
    p.add_argument("--team", required=True, help="team folder under --gallery-root")
    p.add_argument("--gallery-root", default=str(idt.DEFAULT_GALLERY_ROOT), help="folder with one folder per team")
    p.add_argument("--operative", required=True, help="operative name, for example \"Boss Nob\"")
    p.add_argument("--base-mm", type=float, default=None, help="base diameter; only bases of this size are taken (default: from team.toml, else the found base)")
    p.add_argument("--images", nargs="+", default=None, help="take the crops from image files or folders (one base each) instead of the camera")
    p.add_argument("--reference", default=None, metavar="FILE", help="empty-board photo for the bases stage")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("gallery", help="list the gallery and check each photo against the rest")
    _add_gallery_args(p, required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.set_defaults(func=cmd_gallery)

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
    p.add_argument("--reference", default=None, metavar="FILE", help="empty-board photo (terrain, no operatives) from the same camera and light, for the bases stage")
    _add_gallery_args(p, required=False)
    p.add_argument("--no-ident", action="store_true", help="do not load the gallery: the ident stage names nothing (default: the --team teams, else all teams in --gallery-root)")
    p.add_argument("--out", default=None, help="folder for <name>.overlay.jpg, <name>.<view>.jpg and <name>.json per image")
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
