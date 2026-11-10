"""Battle Sight: augmented reality overlay for Kill Team.

    battlesight [--team A [B]] [--source S | --image-source FILE] [--reference FILE]   live app
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from . import ident as idt
from .bench import is_reference
from .calib import CharucoFinder, TagDetector, calibrate_lens
from .camera import DEFAULT_CAMERA_FILE, Camera, CameraSettings, ImageSource, parse_source
from .config import CORNER_NAMES, DEFAULT_CONFIG, BoardConfig
from .live import DEFAULT_REFERENCE, DISPLAY_WIDTH, BoardWindow, FpsMeter, LiveSession, Notice, ZoomView, failed_stage, load_gallery, load_reference
from .overlay import GREEN, RED, YELLOW, draw_bases, draw_text_banner, fit_to_width
from .pipeline import SCHEMA_VERSION, BasesSearch, Pipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
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


def _open_image_source(path: str) -> ImageSource | None:
    source = ImageSource(path)
    if not source.open():
        print(f"error: cannot read image {path}", file=sys.stderr)
        return None
    w, h = source.size
    print(f"image source {path}: {w}x{h}, the same frame on each read")
    return source


def _load_gallery(args: argparse.Namespace, cfg: BoardConfig, log=sys.stdout, quiet: bool = False) -> idt.Gallery | None:
    """Gallery of the --team options, or None (with a message) on an error."""
    return load_gallery(args.team, args.gallery_root, cfg, log, quiet)


def corner_warning(cfg: BoardConfig, obs: dict) -> str | None:
    """Warning text when the camera does not see all four corner cards, else None."""
    missing = [i for i in cfg.tag_ids if i not in obs]
    if not missing:
        return None
    names = ", ".join(f"{i} {CORNER_NAMES[cfg.corner_index(i)]}" for i in missing)
    return f"CORNERS NOT SEEN: {4 - len(missing)}/4 cards, missing {names}"


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

    def draw(self, img, notice: Notice | None = None, info: tuple[str, tuple[int, int, int]] | None = None):
        """`img` with the strips under the status strip: the corner warning,
        `info` (text, color), then the last message of `notice`. A copy when
        something is drawn."""
        strips = []
        if self.warning is not None:
            strips.append((self.warning, RED, "space will fail"))
        if info is not None:
            strips.append((*info, ""))
        if notice is not None and notice.current() is not None:
            strips.append((*notice.current(), ""))
        if not strips:
            return img
        img = img.copy()
        for row, (text, color, extra) in enumerate(strips, start=1):
            draw_text_banner(img, text, color, extra, row=row)
        return img


def _gallery_teams(args: argparse.Namespace) -> list[str]:
    """--team, else every team folder of the gallery root."""
    return list(args.team) if args.team else idt.list_teams(args.gallery_root)


def cmd_live(args: argparse.Namespace) -> int:
    cfg = BoardConfig.load(args.config)
    if not _check_teams(args):
        return 2
    cam = _open_image_source(args.image_source) if args.image_source else _open_camera(args)
    if cam is None:
        return 2
    # The saved reference is from the live camera: it does not fit another image.
    reference = args.reference or (None if args.image_source else DEFAULT_REFERENCE)
    session = LiveSession(cfg, args.camera, cam.size, args.team, args.gallery_root, reference)
    lens = session.lens
    print(f"board {cfg.name} {cfg.width_mm:.0f}x{cfg.height_mm:.0f} mm, lens {lens.source}, fx {lens.K[0, 0]:.0f} px")
    from .ui.shell import run_window  # Qt loads only for the live app

    return run_window(session, cam)


def _enroll_candidates(result, base_mm: float | None) -> list[int]:
    found = result.scratch.get("bases") or []
    return [i for i, c in enumerate(found) if base_mm is None or c.diameter_mm == base_mm]


def leave_one_out_text(right: int, total: int, misses: list[tuple[str, str]], shown: int = 2) -> str:
    """One line for the info bar: the score and the most frequent misses
    (operative names without the team, so that the line fits)."""
    if not total:
        return "leave-one-out: needs two photos or more of operatives with the same base size"
    text = f"leave-one-out: {right}/{total} photos named right ({right / total:.0%})"
    if not misses:
        return text
    pairs = Counter((idt.split_label(true)[1], idt.split_label(guess)[1]) for true, guess in misses).most_common()
    parts = [f"{true} as {guess}" + (f" x{n}" if n > 1 else "") for (true, guess), n in pairs[:shown]]
    if len(pairs) > shown:
        parts.append(f"+{len(pairs) - shown} more")
    return f"{text}: {', '.join(parts)}"


WEAK_MIN_WRONG = 2  # an operative is weak with this many wrong photos or more,
WEAK_MIN_RATE = 0.1  # and this part of its photos or more


def weak_operatives(misses: list[tuple[str, str]], counts: dict[str, int]) -> list[tuple[str, int, int]]:
    """(label, wrong photos, photos) of the operatives that the leave-one-out
    test names wrong too often, worst first. These need more photos."""
    wrong = Counter(true for true, _ in misses)
    out = [(label, n, counts.get(label, 0)) for label, n in wrong.items() if n >= WEAK_MIN_WRONG and n >= WEAK_MIN_RATE * counts.get(label, 0)]
    return sorted(out, key=lambda w: (-w[1] / max(w[2], 1), w[0]))


def weak_text(weak: list[tuple[str, int, int]], shown: int = 3) -> str:
    """Short advice for the info bar, "" when no operative is weak."""
    if not weak:
        return ""
    names = [idt.split_label(label)[1] for label, _, _ in weak[:shown]]
    return "add photos of " + ", ".join(names) + (f", +{len(weak) - shown} more" if len(weak) > shown else "")


def _print_leave_one_out(right: int, total: int, misses: list[tuple[str, str]], counts: dict[str, int] | None = None) -> None:
    print(leave_one_out_text(right, total, []))
    for true, guess in misses:
        print(f"  {true} looked like {guess}")
    for label, n, photos in weak_operatives(misses, counts or {}):
        looks = ", ".join(idt.split_label(g)[1] for g, _ in Counter(guess for true, guess in misses if true == label).most_common(2))
        print(f"  weak: {label}, {n} of {photos} photos wrong (looks like {looks}). Add photos of it: other places on the board, other directions.")


def _enroll_background(result, i: int) -> np.ndarray | None:
    """Empty-board crop of found base i, None when there is no reference."""
    backgrounds = result.scratch["base_backgrounds"]
    return backgrounds[i] if backgrounds else None


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
            load_reference(pipe, Path(args.reference))
        if not pipe.has_reference:
            print("no empty-board reference: the photos get no background crop (--reference)")
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
            try:
                out = gallery.add(label, result.scratch["base_crops"][cand[0]], c.diameter_mm, _enroll_background(result, cand[0]))
            except idt.DuplicatePhoto as exc:
                print(f"skip {path}: {exc}")
                continue
            op = gallery.operatives[gallery.index(label)]
            added += 1
            print(f"added {out} ({c.diameter_mm:g} mm at {np.round(c.center_mm, 1).tolist()} mm)")
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"{label}: {gallery.counts()[label]} photos")
        return 2 if errors else (0 if added else 1)

    reference_path = Path(args.reference or DEFAULT_REFERENCE)
    load_reference(pipe, reference_path)
    cam = _open_camera(args)
    if cam is None:
        return 2
    print(f"enroll {label}: space grab, click the base when there are more (Esc back when the right base has no circle), u undo, e take the empty-board reference, t test the gallery, + - zoom at the pointer (space searches the zoomed area only), 0 no zoom, v view, q quit. Turn and move the model between grabs.")
    window = "Battle Sight - enroll"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    win = BoardWindow(cfg, DISPLAY_WIDTH)
    zoom = ZoomView(DISPLAY_WIDTH)
    win.zoom = zoom
    state = {"result": None, "snap": None, "added": [], "click": None, "test": None, "relaxed": False, "pointer": None, "top_down": win.top_down}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            state["pointer"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and state["result"] is not None:
            state["click"] = win.to_mm(x, y)

    cv2.setMouseCallback(window, on_mouse)
    check = _CornerCheck(cfg)
    notice = Notice()
    meter = FpsMeter()

    def find(snap, relaxed: bool):
        size = None if op.base_mm is None else (op.base_mm,)
        return pipe.analyze(snap, source="camera", search=BasesSearch(zoom.area(), size, relaxed))

    def anchor():
        """Frame point under the pointer, None when the top-down view is on the screen."""
        if state["pointer"] is None or (state["result"] is not None and win.top_down):
            return None
        return zoom.to_frame(state["pointer"][0] / win.scale, state["pointer"][1] / win.scale)

    def add(i: int) -> None:
        nonlocal op
        r = state["result"]
        c = r.scratch["bases"][i]
        try:
            path = gallery.add(label, r.scratch["base_crops"][i], c.diameter_mm, _enroll_background(r, i))
        except idt.DuplicatePhoto:
            state["result"] = None
            notice.error(f"the gallery has this photo already: turn and move {op.name}, then push space")
            return
        op = gallery.operatives[gallery.index(label)]
        state["added"].append(path)
        state["result"] = None
        state["test"] = None  # the gallery changed
        print(f"added {path}")
        notice.ok(f"photo {gallery.counts()[label]} added. Turn and move {op.name}, then push space.")

    def leave_pick() -> bool:
        """Back to the live view from the base choice. False when no choice is open."""
        if state["result"] is None:
            return False
        state["result"], state["snap"], state["click"] = None, None, None
        notice.say("no photo added: move the model or zoom, then push space")
        return True

    while True:
        frame = cam.read()
        if frame is None:
            meter.reset()
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") or (key == 27 and not leave_pick()):
                break
            continue
        meter.tick(time.monotonic())
        zoom.set_frame_size(frame.shape[1], frame.shape[0])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or (key == 27 and not leave_pick()):
            break
        if key == ord("v"):
            state["top_down"] = not state["top_down"]
        if key in (ord("+"), ord("=")):
            zoom.step(1, anchor())
        if key in (ord("-"), ord("_")):
            zoom.step(-1, anchor())
        if key == ord("0"):
            zoom.reset()
        win.top_down = state["top_down"] and not zoom.active  # zoom shows the camera view
        if key == ord("u"):
            if state["added"]:
                path = state["added"].pop()
                gallery.remove(path)
                state["test"] = None  # the gallery changed
                notice.say(f"removed {path.name}: {gallery.counts()[label]} photos")
            else:
                notice.error("nothing to undo")
        if key == ord("e"):
            try:
                pipe.set_reference(frame)
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(reference_path), frame)
                notice.ok(f"empty-board reference saved to {reference_path}")
            except ValueError as exc:
                notice.error(f"no reference: {exc}")
        if key == ord("t"):
            right, total, misses = gallery.leave_one_out()
            _print_leave_one_out(right, total, misses, gallery.counts())
            advice = weak_text(weak_operatives(misses, gallery.counts()))
            state["test"] = (leave_one_out_text(right, total, misses) + (f". {advice[0].upper()}{advice[1:]}" if advice else ""), GREEN if total and not misses else YELLOW)
        if key == ord(" "):
            snap = frame.copy()
            result = find(snap, False)
            state["relaxed"] = result.ok and not _enroll_candidates(result, op.base_mm)
            if state["relaxed"]:
                result = find(snap, True)
            state["result"], state["snap"] = (result, snap) if result.ok else (None, None)
            if not result.ok:
                print(result.summary())
                notice.error(failed_stage(result))
            else:
                cand = _enroll_candidates(result, op.base_mm)
                if not cand:
                    notice.error("no base found" + ("" if op.base_mm is None else f" of {op.base_mm:g} mm") + ": the base edge must be visible")
                    state["result"] = None
                elif state["relaxed"]:
                    notice.say(f"not sure: click the base of {op.name} only if a yellow circle fits it, else push space")
                elif len(cand) == 1:
                    add(cand[0])
                else:
                    notice.say(f"{len(cand)} bases: click the base of {op.name}")
        r = state["result"]
        video = f"{frame.shape[1]}x{frame.shape[0]} {'--' if meter.fps is None else f'{meter.fps:.0f}'} fps"
        if zoom.active:
            video += f" zoom {zoom.factor:.1f}x"
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
            if not win.top_down:
                img = zoom.image(img)
            ask = f"not sure: click the base of {op.name}" if state["relaxed"] else f"click the base of {op.name}"
            draw_text_banner(img, ask, YELLOW, f"{video}, {gallery.counts()[label]} photos, Esc back")
        else:
            img = zoom.image(frame).copy()
            if pipe.has_reference:
                draw_text_banner(img, f"enroll {op.name}: push space", GREEN, f"{video}, {gallery.counts()[label]} photos, aim for 10 to 20")
            else:
                draw_text_banner(img, f"enroll {op.name}: no background saved, push e on the empty board", YELLOW, f"{video}, {gallery.counts()[label]} photos, aim for 10 to 20")
        check.update(frame)
        cv2.imshow(window, check.draw(win.fit(img), notice, state["test"]))
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
    _print_leave_one_out(*gallery.leave_one_out(), counts)
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
    A folder gives no `<name>.empty.*` references. A file named here always counts.
    Returns (images, errors)."""
    images: list[Path] = []
    errors: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES and not is_reference(q))
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
    parser.add_argument("--reference", default=None, metavar="FILE", help=f"empty-board photo: loaded at start, written by the e key (default: {DEFAULT_REFERENCE})")
    parser.add_argument("--image-source", default=None, metavar="FILE", help="use one image file instead of the camera, for debugging and tests")
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
    if args.image_source and args.func is not cmd_live:
        parser.error("--image-source works only with the live app (no command)")
    if args.image_source and args.source is not None:
        parser.error("use --source or --image-source, not both")
    if args.func in (cmd_live, cmd_enroll, cmd_calibrate):
        _apply_camera_settings(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
