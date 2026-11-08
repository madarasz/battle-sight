"""Battle Sight: augmented reality overlay for Kill Team.

    battlesight [--team A [B]] [--source S] [--reference FILE]   live app
    battlesight enroll --team NAME --operative NAME [--base-mm 32] [--images PATH...]
    battlesight gallery --team A [B]
    battlesight calibrate [--source S | --images PATH...]
    battlesight image-pipeline PATH... [--reference FILE] [--team A [B]] [--out DIR] [--json FILE|-] [--show]

Live app: tracks the four corner cards on every frame (the board position
updates by itself when the board or the camera moves) and shows the camera or
the board from above. Keys: space find the bases and name them, b find the
bases only, v camera / top-down view, g grid, e take the empty-board
reference, c clear it, click a base then a name to fix it, a accept all
names, u undo the last gallery photo, s save the frame, h help, Esc close
the menu or the snapshot, q quit.

Camera values (source, frame size, frame rate, lens) are in data/camera.toml
(--camera-file): tools/list_cameras.py and battlesight calibrate write it.
--source overrides the saved source for one run.

enroll: adds gallery photos of one operative (live, or --images).
gallery: lists the gallery and checks each photo against the rest.
calibrate: lens calibration with the ChArUco sheet (tools/make_charuco.py).
image-pipeline: runs all stages on saved photos, writes a JSON report and
overlay images. Exit code 0 all stages passed, 1 a stage failed, 2 bad input.
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
from .overlay import GREEN, RED, WHITE, YELLOW, Menu, draw_banner, draw_bases, draw_board_corners, draw_grid, draw_idents, draw_menu, draw_tags, draw_text_banner, fit_to_width
from .pipeline import SCHEMA_VERSION, Pipeline
from .rectify import fit_view, mm_to_rect, warp_board

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
DEFAULT_REFERENCE = SNAPSHOT_DIR / "empty_board.png"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DISPLAY_WIDTH = 1600  # image width in the windows; the windows can be resized
MAX_TEAMS = 2  # two teams play


def _add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default=None, help="camera index, or stream URL such as http://PHONE-IP:8080/video, for this run (default: saved in --camera-file, else 0)")
    p.add_argument("--camera-file", default=str(DEFAULT_CAMERA_FILE), help="camera settings and lens calibration (tools/list_cameras.py, calibrate)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")


def _add_gallery_args(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("--team", nargs="+", required=required, default=None, metavar="NAME", help=f"team folders under --gallery-root, 1 or {MAX_TEAMS}")
    p.add_argument("--gallery-root", default=str(idt.DEFAULT_GALLERY_ROOT), help="folder with one folder per team")


def _check_teams(args: argparse.Namespace) -> bool:
    if args.team and len(args.team) > MAX_TEAMS:
        print(f"error: --team takes 1 or {MAX_TEAMS} teams, got {len(args.team)}", file=sys.stderr)
        return False
    return True


def _apply_camera_settings(args: argparse.Namespace) -> None:
    """Fill the camera options that the command line left out from the saved settings."""
    path = Path(args.camera_file)
    settings = CameraSettings.load(path).override(source=None if args.source is None else parse_source(args.source), width=None, height=None, fps=None)
    if path.exists():
        print(f"camera settings: {path} (--source overrides the saved source)")
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


HELP_KEYS = [
    ("space", "find the bases and name them"),
    ("b", "find the bases only"),
    ("v", "camera view / top-down view"),
    ("g", "grid on / off"),
    ("e", "empty-board reference: take it (no operatives on the board)"),
    ("c", "empty-board reference: clear it"),
    ("click", "a base: name menu. A name: set it"),
    ("a", "accept all names"),
    ("u", "undo the last gallery photo"),
    ("s", "save the camera frame"),
    ("h", "show / hide this help"),
    ("Esc", "close the menu, else close the snapshot"),
    ("q", "quit"),
]


def _draw_help(img: np.ndarray, first_row: int) -> None:
    for row, (key, text) in enumerate(HELP_KEYS, start=first_row):
        draw_text_banner(img, f"{key:6s} {text}", WHITE, row=row)


def _gallery_teams(args: argparse.Namespace) -> list[str]:
    """--team, else every team folder of the gallery root."""
    return list(args.team) if args.team else idt.list_teams(args.gallery_root)


class _Live:
    """State of the live app: live view or a snapshot, the name menu, the
    keys that are on. No camera or window calls, so the key logic stays
    simple to read."""

    def __init__(self):
        self.top_down = False
        self.grid = True
        self.help = False
        self.result = None  # FrameResult of the shown snapshot, or None = live view
        self.snap = None  # camera frame of the snapshot
        self.session: idt.LabelSession | None = None
        self.menu: Menu | None = None
        self.hover: int | None = None
        self.dirty = True
        self.shown: np.ndarray | None = None  # rendered snapshot

    def close_snapshot(self) -> None:
        self.result = self.snap = self.session = self.menu = self.shown = None

    def escape(self) -> None:
        """Esc: close the menu when it is open, else go back to the live view."""
        if self.menu is not None:
            self.menu = None
            if self.session is not None:
                self.session.selected = None
            self.dirty = True
        else:
            self.close_snapshot()


def cmd_live(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    if not _check_teams(args):
        return 2
    pipe = Pipeline(cfg, None, args.camera)
    ref_path = Path(args.reference)
    _load_reference(pipe, ref_path)
    detector = TagDetector(cfg)
    monitor = DriftMonitor(cfg)
    cam = _open_camera(args)
    if cam is None:
        return 2
    lens = lens_for_frame(args.camera, cam.size)
    print(f"board {cfg.name} {cfg.width_mm:.0f}x{cfg.height_mm:.0f} mm, lens {lens.source}, fx {lens.K[0, 0]:.0f} px")
    print("keys: space find and name, b bases only, v view, g grid, e/c empty-board reference, click a base to fix its name, a accept all, u undo, s save, h help, Esc back, q quit")
    window = "Battle Sight"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    win = _BoardWindow(cfg, DISPLAY_WIDTH)
    st = _Live()
    notice = _Notice()
    gallery: idt.Gallery | None = None
    gallery_tried = False

    def on_mouse(event, x, y, flags, _):
        sess, menu = st.session, st.menu
        if event == cv2.EVENT_MOUSEMOVE and menu is not None:
            hover = menu.hit(x, y)
            st.dirty |= hover != st.hover
            st.hover = hover
        if event != cv2.EVENT_LBUTTONDOWN or st.result is None:
            return
        if sess is None:
            notice.error("push space to name the operatives")
            return
        st.dirty = True
        if menu is not None:
            row = menu.hit(x, y)
            st.menu = None
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
        st.menu = Menu(items, (x + 12, y), row_h=max(22, int(28 * win.scale * 1.5)), width=280)

    cv2.setMouseCallback(window, on_mouse)

    def snapshot(frame, name: bool) -> None:
        nonlocal gallery, gallery_tried
        if name and not gallery_tried:
            gallery_tried = True
            teams = _gallery_teams(args)
            if not teams:
                notice.error(f"no gallery teams in {args.gallery_root}: battlesight enroll")
            else:
                args.team = teams
                gallery = _load_gallery(args, cfg)
                if gallery is None:
                    notice.error("gallery: cannot load, see the Terminal")
                pipe.set_gallery(gallery)
        st.close_snapshot()
        st.snap = frame.copy()
        st.result = result = pipe.analyze(st.snap, source="camera", until=None if name else "bases")
        print(result.summary())
        if name and gallery is not None and result.ok:
            st.session = _session(result, gallery)
            for b in result.stages["ident"].data.get("bases", []):
                print(f"  {b['label'] or 'unknown':30s} {b['diameter_mm']:g} mm at {b['center_mm']} mm, score {b['score']}")
        else:
            for b in result.stages["bases"].data.get("bases", []) if "bases" in result.stages else []:
                print(f"  {b['diameter_mm']:g} mm at {b['center_mm']} mm, rim {b['coverage']:.0%} visible")
        if not result.ok:
            notice.error(_failed_stage(result))
        elif not result.scratch.get("bases"):
            notice.error("no base found")
        elif name and gallery is not None and result.stages["ident"].message.startswith("gallery has no photos"):
            notice.error(result.stages["ident"].message)
        st.dirty = True

    def render_snapshot() -> np.ndarray:
        r = st.result
        if not r.ok:
            img = pipe.annotate(st.snap, r)
            return win.fit(img)
        img = win.base_image(st.snap, r)
        if st.grid:
            draw_grid(img, cfg, (lambda p: mm_to_rect(cfg, p)) if win.top_down else win.view.mm_to_px)
        found = r.scratch.get("bases") or []
        if st.session is not None:
            draw_idents(img, found, st.session.idents, win.to_px, st.session.selected)
            named = sum(x.label is not None for x in st.session.idents)
            draw_text_banner(img, f"{named}/{len(found)} named", GREEN, "click a base to fix its name, Esc back to live")
        else:
            draw_bases(img, found, win.to_px)
            draw_text_banner(img, f"{len(found)} bases", GREEN, "space to name them, Esc back to live")
        out = win.fit(img)
        if st.menu is not None:
            st.menu = st.menu.fit((out.shape[1], out.shape[0]))
            draw_menu(out, st.menu, st.hover)
        return out

    def render_live(frame, obs, report, view) -> np.ndarray:
        if st.top_down and view is not None:
            out = warp_board(frame, view, cfg)
            if st.grid:
                draw_grid(out, cfg, lambda p: mm_to_rect(cfg, p))
        else:
            out = frame.copy()
            draw_tags(out, obs, cfg, report)
            draw_board_corners(out, board_corners_px(cfg, obs), board_corners_px(cfg, monitor.reference) if monitor.reference else None)
            if st.grid and view is not None:
                draw_grid(out, cfg, view.mm_to_px)
        draw_banner(out, report, f"{lens.source} lens  h help")
        return win.fit(out)

    sized = False
    view = None
    last_status = None
    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            continue
        obs = detector.detect(frame)
        report = monitor.update(obs)
        if report.relocked:
            notice.ok("board moved: new position saved")
        if report.status is not last_status:
            print(report.summary())
            last_status = report.status
        # A still board uses the saved position, so the view does not jitter.
        tags = monitor.reference if report.status is DriftStatus.OK else obs
        view = None
        if tags and all(i in tags for i in cfg.tag_ids):
            try:
                view = fit_view(cfg, tags, lens)
            except ValueError as exc:
                print(f"rectify: {exc}", file=sys.stderr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 27:
            st.escape()
        if key == ord("v"):
            st.top_down = not st.top_down
            st.menu, st.dirty = None, True
        if key == ord("g"):
            st.grid, st.dirty = not st.grid, True
        if key == ord("h"):
            st.help = not st.help
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
            notice.say("empty-board reference cleared: single-frame detection")
        if key == ord("s"):
            notice.ok(f"saved {_save_snapshot(frame).name}")
        if key in (ord("a"), ord("u")) and st.session is None:
            notice.error("push space first")
        if key == ord("a") and st.session is not None:
            added = st.session.accept_all()
            notice.ok(f"accepted {len(added)} names, added to the gallery")
            st.dirty = True
        if key == ord("u") and st.session is not None:
            path = st.session.undo()
            if path is None:
                notice.error("nothing to undo")
            else:
                notice.say(f"removed {path.name}")
            st.dirty = True
        if key in (ord(" "), ord("b")):
            snapshot(frame, name=key == ord(" "))

        if st.result is not None:
            if st.dirty:
                st.shown = render_snapshot()
                st.dirty = False
            out = st.shown.copy()
            strips = [] if report.ok else [(report.summary(), RED if report.status is not DriftStatus.MOVED else YELLOW, "live")]
        else:
            out = render_live(frame, obs, report, view)
            hint = "push space to find and name the operatives, b for the bases only"
            if not pipe.has_reference:
                hint += "  (no empty-board reference: e)"
            strips = [(hint, YELLOW, "")]
        if notice.current() is not None:
            strips.append((*notice.current(), ""))
        for row, (text, color, extra) in enumerate(strips, start=1):
            draw_text_banner(out, text, color, extra, row=row)
        if st.help:
            _draw_help(out, len(strips) + 1)
        cv2.imshow(window, out)
        if not sized:
            cv2.resizeWindow(window, out.shape[1], out.shape[0])
            sized = True
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
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    win = _BoardWindow(cfg, DISPLAY_WIDTH)
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
    if not _check_teams(args):
        return 2
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
            cv2.imshow(window, fit_to_width(out, DISPLAY_WIDTH))
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


def cmd_image_pipeline(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    if not _check_teams(args):
        return 2
    # Keep stdout clean for the JSON report when it goes there.
    log = sys.stderr if args.json == "-" else sys.stdout
    gallery = None
    # No --team: every team folder of the gallery root.
    args.team = _gallery_teams(args) or None
    if args.team is None and not args.quiet:
        print(f"no gallery teams in {args.gallery_root}: the ident stage names nothing (battlesight enroll)", file=log)
    if args.team:
        gallery = _load_gallery(args, cfg, log, args.quiet)
        if gallery is None:
            return 2
    pipe = Pipeline(cfg, None, CameraSettings.load(args.camera_file), gallery=gallery)
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
    window = "Battle Sight - image-pipeline"
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
                cv2.imshow(window, fit_to_width(overlay, DISPLAY_WIDTH))
                for name, img in views.items():
                    cv2.imshow(f"{window} - {name}", fit_to_width(img, DISPLAY_WIDTH))
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
    # No command: the live app. Its options are on the main parser.
    _add_camera_args(parser)
    _add_gallery_args(parser, required=False)
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="empty-board photo: loaded at start, written by the e key")
    parser.set_defaults(func=cmd_live)
    sub = parser.add_subparsers(dest="command", metavar="command")

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

    p = sub.add_parser("image-pipeline", help="run all stages on saved photos")
    p.add_argument("inputs", nargs="+", help="image files or folders of images")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board TOML config")
    p.add_argument("--camera-file", default=str(DEFAULT_CAMERA_FILE), help="camera settings with the lens calibration ([lens])")
    p.add_argument("--reference", default=None, metavar="FILE", help="empty-board photo (terrain, no operatives) from the same camera and light, for the bases stage")
    _add_gallery_args(p, required=False)
    p.add_argument("--out", default=None, help="folder for <name>.overlay.jpg, <name>.<view>.jpg and <name>.json per image")
    p.add_argument("--json", default=None, metavar="FILE", help="write the combined JSON report to FILE, or - for stdout")
    p.add_argument("--show", action="store_true", help="show each overlay in a window (any key next, q quit)")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-image status lines")
    p.set_defaults(func=cmd_image_pipeline)
    args = parser.parse_args(argv)
    if args.func in (cmd_live, cmd_enroll, cmd_calibrate):
        _apply_camera_settings(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
