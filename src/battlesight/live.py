"""Game core of the live app. No window, no camera, no GUI toolkit.

A front end reads frames from the camera and gives each one to
`LiveSession.step`. It shows the returned image and state, and it sends the
user's commands with `LiveSession.dispatch`.

The contract between the core and a front end (specs/ui.md, section 8):

- `ACTIONS` lists all things that the user can do.
- `LiveSession.state()` is a JSON-safe dict with all that the user can see.

The image holds the board overlays only. Status text, messages and the name
menu belong to the front end.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from . import ident as idt
from . import measure
from . import terrain as ter
from .calib import DriftMonitor, DriftReport, DriftStatus, TagDetector, board_corners_px, lens_for_frame
from .camera import CameraSettings
from .config import BoardConfig
from .overlay import GREEN, MAGENTA, RED, WHITE, YELLOW, draw_bases, draw_board_corners, draw_circle_mm, draw_grid, draw_idents, draw_rim_line, draw_tags, draw_terrain, fit_to_width
from .pipeline import Pipeline
from .rectify import fit_view, mm_to_rect, warp_board

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
DEFAULT_REFERENCE = SNAPSHOT_DIR / "empty_board.png"
DISPLAY_WIDTH = 1600  # image width in the windows; the windows can be resized


# --- helpers shared with the other live commands --------------------------------


class Notice:
    """Last message of a live window. `say` prints it (errors to stderr) and
    keeps it on the screen for `seconds`, so the user sees it without the
    Terminal. `clock` is for tests."""

    def __init__(self, seconds: float = 6.0, clock: Callable[[], float] = time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self.text: str | None = None
        self.color = YELLOW
        self.level = "hint"
        self.until = 0.0

    def say(self, text: str, color=YELLOW, error: bool = False) -> None:
        print(text, file=sys.stderr if error else sys.stdout)
        self.text, self.color, self.until = text, color, self.clock() + self.seconds
        self.level = "error" if error else "done" if color == GREEN else "hint"

    def error(self, text: str) -> None:
        self.say(text, RED, error=True)

    def ok(self, text: str) -> None:
        self.say(text, GREEN)

    def current(self) -> tuple[str, tuple[int, int, int]] | None:
        if self.text is None or self.clock() > self.until:
            return None
        return self.text, self.color


class FpsMeter:
    """Smoothed frame rate of a live loop. `fps` is None before the second tick."""

    def __init__(self) -> None:
        self.fps: float | None = None
        self.last: float | None = None

    def reset(self) -> None:
        """The camera is down: the next tick starts a new measurement."""
        self.fps = self.last = None

    def tick(self, now: float) -> None:
        if self.last is not None and now > self.last:
            rate = 1.0 / (now - self.last)
            self.fps = rate if self.fps is None else 0.9 * self.fps + 0.1 * rate
        self.last = now


class ZoomView:
    """Digital zoom of the camera image: a rectangle of the frame, shown at
    the display size. No window calls."""

    STEP = 2.0**0.5
    MAX = 8.0

    def __init__(self, display_width: int):
        self.display_width = display_width
        self.factor = 1.0
        self.size: tuple[int, int] | None = None  # frame (w, h)
        self.center = np.zeros(2)
        self.out: tuple[int, int] | None = None  # shown image (w, h)

    @property
    def active(self) -> bool:
        return self.factor > 1.0

    def _rect(self) -> tuple[int, int, int, int]:
        w, h = self.size
        half = np.array([w, h], float) / (2.0 * self.factor)
        c = np.clip(self.center, half, np.array([w, h], float) - half)
        x0, y0 = np.rint(c - half).astype(int)
        x1, y1 = np.rint(c + half).astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    def area(self) -> tuple[int, int, int, int] | None:
        """The shown rectangle of the frame (x0, y0, x1, y1), None without zoom."""
        return self._rect() if self.active and self.size is not None else None

    def step(self, direction: int, anchor: Sequence[float] | None = None) -> None:
        """One step in (+1) or out (-1). The frame point `anchor` stays where it is."""
        if self.size is None:
            return
        new = float(np.clip(self.factor * self.STEP**direction, 1.0, self.MAX))
        new = 1.0 if new < 1.01 else new
        if anchor is not None and new != self.factor:
            a = np.asarray(anchor, float)
            self.center = a + (self.center - a) * self.factor / new
        self.factor = new
        w, h = self.size
        half = np.array([w, h], float) / (2.0 * new)
        self.center = np.clip(self.center, half, np.array([w, h], float) - half)

    def reset(self) -> None:
        self.factor = 1.0
        self.out = None
        if self.size is not None:
            self.center = np.array(self.size, float) / 2.0

    def set_frame_size(self, w: int, h: int) -> None:
        if self.size != (w, h):
            self.size = (w, h)
            self.center = np.array([w, h], float) / 2.0

    def image(self, frame: np.ndarray) -> np.ndarray:
        """The shown part of `frame`. The frame itself without zoom."""
        h, w = frame.shape[:2]
        self.set_frame_size(w, h)
        if not self.active:
            self.out = None
            return frame
        x0, y0, x1, y1 = self._rect()
        out_w = min(w, self.display_width)
        self.out = (out_w, max(1, round(h * out_w / w)))
        crop = frame[y0:y1, x0:x1]
        return cv2.resize(crop, self.out, interpolation=cv2.INTER_AREA if x1 - x0 > out_w else cv2.INTER_LINEAR)

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Pixel of the shown image -> pixel of the frame."""
        if not self.active or self.out is None:
            return x, y
        x0, y0, x1, y1 = self._rect()
        return x0 + x * (x1 - x0) / self.out[0], y0 + y * (y1 - y0) / self.out[1]


class BoardWindow:
    """Shows a snapshot top-down (rectified) or as the camera frame, and maps
    window clicks to board mm. `fit_to_width` scales the image down."""

    def __init__(self, cfg: BoardConfig, display_width: int):
        self.cfg = cfg
        self.display_width = display_width
        self.top_down = True
        self.scale = 1.0
        self.view = None  # rectify.BoardView of the snapshot
        self.zoom: ZoomView | None = None  # zoom of the camera view

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
        if self.view is None:
            return None
        if self.zoom is not None:
            px = np.array([self.zoom.to_frame(px[0, 0], px[0, 1])])
        return self.view.px_to_mm(px)[0]

    def fit(self, img: np.ndarray) -> np.ndarray:
        out = fit_to_width(img, self.display_width)
        self.scale = out.shape[1] / img.shape[1]
        return out


def failed_stage(result) -> str:
    """Short text of the first failed stage of a snapshot."""
    for name, r in result.stages.items():
        if not r.ok and not r.skipped:
            return f"{name} failed: {r.message}"
    return result.summary()


def load_gallery(teams: Sequence[str], root, cfg: BoardConfig, log=sys.stdout, quiet: bool = False) -> idt.Gallery | None:
    """Gallery of the teams, or None (with a message) on an error."""
    try:
        gallery = idt.Gallery.for_teams(teams, cfg, root)
    except (ValueError, OSError) as exc:
        print(f"error: gallery: {exc}", file=sys.stderr)
        return None
    if not quiet:
        print(f"gallery {', '.join(sorted(gallery.team_dirs))}: {len(gallery.operatives)} operatives, {len(gallery)} photos, model {gallery.embedder.name}", file=log)
        for label, n in gallery.counts().items():
            if n == 0:
                print(f"  {label}: no photos yet (battlesight enroll)", file=log)
    return gallery


def load_reference(pipe: Pipeline, path: Path) -> None:
    if path.is_file():
        ref = cv2.imread(str(path), cv2.IMREAD_COLOR)
        try:
            pipe.set_reference(ref)
            print(f"empty-board reference: {path}")
        except ValueError as exc:
            print(f"{path}: {exc}. Push e on the empty board to take a new one.", file=sys.stderr)


# --- the contract: actions ------------------------------------------------------


def _always(_state: dict) -> bool:
    return True


def _has_selection(state: dict) -> bool:
    return state["selected"] is not None


def _in_snapshot(state: dict) -> bool:
    return state["mode"] == "snapshot"


@dataclass(frozen=True)
class Action:
    """One thing that the user can do. `enabled` and `checked` only tell a
    front end how to show the control: `dispatch` does not read them, so a key
    on a control that is off still gives its message. `control = False` keeps
    the action off the panels (key and menu only). `args` names the values
    that `dispatch` needs. An action with `shell = True` belongs to the front
    end (help, quit), and the core has no handler for it."""

    id: str
    label: str
    key: str | None = None
    enabled: Callable[[dict], bool] = _always
    checked: Callable[[dict], bool] | None = None
    args: tuple[str, ...] = ()
    shell: bool = False
    control: bool = True  # False: key and menu only, no button in a panel


ACTIONS: tuple[Action, ...] = (
    Action("snapshot.name", "Identify operatives", "space", enabled=lambda s: not s["busy"]),
    Action("snapshot.bases", "Find bases only", "b", enabled=lambda s: not s["busy"], control=False),
    Action("snapshot.back", "Back", "Esc", enabled=_in_snapshot),
    Action("view.top_down", "Adjust perspective", "v", checked=lambda s: s["top_down"]),
    Action("view.grid", "Grid", "g", checked=lambda s: s["grid"]),
    Action("view.terrain", "Layout", checked=lambda s: s["terrain"], enabled=lambda s: s["map_layout"] is not None),
    Action("view.zones", "Drop zones", checked=lambda s: s["zones"], enabled=lambda s: s["map_layout"] is not None),
    Action("view.keys", "Terrain keys", checked=lambda s: s["keys"], enabled=lambda s: s["map_layout"] is not None),
    Action("map.terrain_type", "Terrain type", args=("name",)),
    Action("map.layout", "Layout", args=("name",)),
        Action("reference.take", "Take empty-board reference", "e"),
    Action("reference.clear", "Clear reference", "c", enabled=lambda s: s["reference_set"]),
    Action("board.click", "Click the board", args=("x", "y"), enabled=_in_snapshot),
    Action("board.select", "Select for names", args=("x", "y"), enabled=_in_snapshot),
    Action("base.choose", "Set name", args=("label", "index"), enabled=_has_selection),
    Action("base.deselect", "Clear selection", enabled=_has_selection),
    *(Action(f"ring.{n}", f"Ring {n} in", str(n), enabled=_has_selection, checked=partial(lambda n, s: s["ring_in"] == n, n)) for n in range(1, 10)),
    Action("ring.clear", "Clear ring", "0", enabled=lambda s: s["ring_in"] is not None),
    Action("names.accept", "Accept all names", "a", enabled=lambda s: s["unconfirmed"] > 0),
    Action("names.undo", "Undo last gallery photo", "u", enabled=lambda s: s["can_undo"]),
    Action("frame.save", "Save camera frame", "s"),
    Action("app.layout", "Next layout", "l", shell=True),
    Action("app.help", "Keys", "h", shell=True),
    Action("app.quit", "Quit", "q", shell=True),
)


# --- the core -------------------------------------------------------------------


@dataclass
class LiveFrame:
    """Result of one `LiveSession.step`. `image` is None until the first camera
    frame. `scale` is image pixels per source pixel. `crop` is the board around
    the selected base, or None."""

    image: np.ndarray | None
    state: dict[str, Any]
    scale: float
    crop: np.ndarray | None = None


def _num(value) -> float | None:
    """JSON-safe number: None for a missing or infinite value."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 4)


class LiveSession:
    """State and logic of the live app. All calls come from one thread."""

    def __init__(
        self,
        cfg: BoardConfig,
        camera: CameraSettings,
        frame_size: tuple[int, int],
        teams: Sequence[str] | None = None,
        gallery_root=idt.DEFAULT_GALLERY_ROOT,
        reference_path: Path | str | None = DEFAULT_REFERENCE,
        snapshot_dir: Path = SNAPSHOT_DIR,
        display_width: int = DISPLAY_WIDTH,
        clock: Callable[[], float] = time.monotonic,
        maps_root: Path | str = ter.DEFAULT_MAPS_ROOT,
    ):
        self.cfg = cfg
        self.lens = lens_for_frame(camera, frame_size)
        self.pipe = Pipeline(cfg, None, camera)
        self.reference_path = None if reference_path is None else Path(reference_path)  # None: no load, no save
        self.snapshot_dir = Path(snapshot_dir)
        if self.reference_path is not None:
            load_reference(self.pipe, self.reference_path)
        self.detector = TagDetector(cfg)
        self.monitor = DriftMonitor(cfg)
        self.win = BoardWindow(cfg, display_width)
        self.clock = clock
        self.notice = Notice(clock=clock)
        self.teams = list(teams) if teams else None
        self.gallery_root = gallery_root
        # Base sizes of the teams, from the team files: also for the b key, before the gallery loads.
        self.pipe.set_roster(idt.roster_sizes(gallery_root, self.teams or idt.list_teams(gallery_root)))
        self.gallery: idt.Gallery | None = None
        self.gallery_tried = False

        self.top_down = True
        self.grid = True
        self.terrain = True  # terrain overlay on, when a map layout is chosen
        self.zones = False  # drop zones of the layout off
        self.keys = False  # wall keys (A1, B2, ...) off
        self.maps_root = Path(maps_root)
        self.maps: list[ter.MapEntry] = []
        self.terrain_type: str | None = None
        self.map_layout: str | None = None
        self.terrain_shapes: list[ter.Shape] = []
        self._read_maps()
        self.busy = False
        self.camera_up = True
        self.on_busy: Callable[[LiveFrame], None] | None = None  # called before a long action

        self.frame: np.ndarray | None = None
        self.obs: dict = {}
        self.report: DriftReport | None = None
        self.view = None  # rectify.BoardView of the last frame
        self.last_status = None
        self.image: np.ndarray | None = None
        self.detect_ms: float | None = None  # tag detection of the last frame
        self.meter = FpsMeter()

        self.result = None  # FrameResult of the shown snapshot, or None = live view
        self.snap: np.ndarray | None = None  # camera frame of the snapshot
        self.session: idt.LabelSession | None = None
        self.selected: int | None = None  # index of the selected base
        self.target: int | None = None  # index of the second base of a distance
        self.ring_in: int | None = None  # ring on the selected base, in inches
        self.shown: np.ndarray | None = None  # rendered snapshot
        self.dirty = True

        self._handlers: dict[str, Callable[..., None]] = {
            "snapshot.name": lambda: self._snapshot(name=True),
            "snapshot.bases": lambda: self._snapshot(name=False),
            "snapshot.back": self._back,
            "view.top_down": self._toggle_view,
            "view.grid": self._toggle_grid,
            "view.terrain": self._toggle_terrain,
            "view.zones": self._toggle_zones,
            "view.keys": self._toggle_keys,
            "map.terrain_type": self._set_terrain_type,
            "map.layout": self._set_map_layout,
            "reference.take": self._take_reference,
            "reference.clear": self._clear_reference,
            "board.click": self._click,
            "board.select": self._select_for_names,
            "base.choose": self._choose,
            "base.deselect": self._deselect,
            **{f"ring.{n}": partial(self._set_ring, n) for n in range(1, 10)},
            "ring.clear": partial(self._set_ring, None),
            "names.accept": self._accept_all,
            "names.undo": self._undo,
            "frame.save": self._save_frame,
        }

    # --- frames -----------------------------------------------------------------

    def step(self, frame: np.ndarray | None) -> LiveFrame:
        """Process one camera frame. None = the camera is down."""
        self.camera_up = frame is not None
        if frame is None:
            self.meter.reset()
            return self.current()
        self.frame = frame
        started = self.clock()
        self.meter.tick(started)
        self.obs = self.detector.detect(frame)
        self.detect_ms = (self.clock() - started) * 1000.0
        self.report = report = self.monitor.update(self.obs)
        if report.relocked:
            self.notice.ok("board moved: new position saved")
        if report.status is not self.last_status:
            print(report.summary())
            self.last_status = report.status
        # A still board uses the saved position, so the view does not jitter.
        tags = self.monitor.reference if report.status is DriftStatus.OK else self.obs
        self.view = None
        if tags and all(i in tags for i in self.cfg.tag_ids):
            try:
                self.view = fit_view(self.cfg, tags, self.lens)
            except ValueError as exc:
                print(f"rectify: {exc}", file=sys.stderr)
        self.image = self._snapshot_image() if self.result is not None else self._render_live()
        return self.current()

    def current(self) -> LiveFrame:
        """The last image with the state of now."""
        state = self.state()
        return LiveFrame(self.image, state, self.win.scale, self._selected_crop())

    def _selected_crop(self) -> np.ndarray | None:
        s, i = self.session, self.selected
        return None if s is None or i is None or i >= len(s.crops) else s.crops[i]

    def set_display_width(self, width: int) -> None:
        """Largest width of the returned image. A front end sets it to the width on the screen."""
        self.win.display_width = width
        self.dirty = True

    def _render_live(self) -> np.ndarray:
        cfg, frame, view = self.cfg, self.frame, self.view
        if self.top_down and view is not None:
            out = warp_board(frame, view, cfg)
            self._draw_terrain(out, lambda p: mm_to_rect(cfg, p))
            if self.grid:
                draw_grid(out, cfg, lambda p: mm_to_rect(cfg, p))
        else:
            out = frame.copy()
            draw_tags(out, self.obs, cfg, self.report)
            reference = self.monitor.reference
            draw_board_corners(out, board_corners_px(cfg, self.obs), board_corners_px(cfg, reference) if reference else None)
            if view is not None:
                self._draw_terrain(out, view.mm_to_px)
            if self.grid and view is not None:
                draw_grid(out, cfg, view.mm_to_px)
        return self.win.fit(out)

    def _snapshot_image(self) -> np.ndarray:
        if self.dirty:
            self.shown = self._render_snapshot()
            self.dirty = False
        return self.shown

    def _render_snapshot(self) -> np.ndarray:
        r, win, cfg = self.result, self.win, self.cfg
        if not r.ok:
            return win.fit(self.pipe.annotate(self.snap, r, banner=False))
        img = win.base_image(self.snap, r)
        self._draw_terrain(img, win.to_px)
        if self.grid:
            draw_grid(img, cfg, (lambda p: mm_to_rect(cfg, p)) if win.top_down else win.view.mm_to_px)
        found = r.scratch.get("bases") or []
        if self.session is not None:
            draw_idents(img, found, self.session.idents, win.to_px)
        else:
            draw_bases(img, found, win.to_px)
        self._draw_measure(img, found)
        return win.fit(img)

    def _draw_measure(self, img: np.ndarray, found: list) -> None:
        """The selected base, its ring, the second base and the distance."""
        if self.selected is None:
            return
        to_px, a = self.win.to_px, found[self.selected]
        draw_circle_mm(img, a.center_mm, 0.5 * a.diameter_mm, to_px, WHITE)
        if self.ring_in is not None:
            draw_circle_mm(img, a.center_mm, measure.ring_radius_mm(a.diameter_mm, self.ring_in), to_px, MAGENTA)
        if self.target is not None:
            b = found[self.target]
            gap, start, end = measure.rim_gap(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)
            draw_circle_mm(img, b.center_mm, 0.5 * b.diameter_mm, to_px, MAGENTA)
            draw_rim_line(img, start, end, f'{gap / measure.MM_PER_INCH:.2f}"', to_px)

    # --- state ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """All that a front end can show. JSON-safe."""
        if self.result is not None:
            self._snapshot_image()  # brings the scale and the view up to date
        s = self.session
        current = self.notice.current()
        return {
            "mode": "live" if self.result is None else "snapshot",
            "busy": self.busy,
            "top_down": self.win.top_down if self.result is not None else self.top_down,
            "grid": self.grid,
            "terrain": self.terrain,
            "zones": self.zones,
            "keys": self.keys,
            "terrain_types": sorted({e.terrain for e in self.maps}),
            "terrain_type": self.terrain_type,
            "map_layouts": [e.name for e in self.maps if e.terrain == self.terrain_type],
            "map_layout": self.map_layout,
            "drift": self._drift_state(),
            "camera": self.camera_up,
            "lens": self.lens.source,
            "teams": sorted(self.gallery.team_dirs) if self.gallery is not None else self.teams,
            "reference_set": self.pipe.has_reference,
            "snapshot_ok": None if self.result is None else bool(self.result.ok),
            "naming": s is not None,
            "bases": self._bases_state(),
            "selected": self.selected,
            "target": self.target,
            "ring_in": self.ring_in,
            "distance": self._distance_state(),
            "candidates": self._candidates_state(),
            "can_undo": s is not None and bool(s.added),
            "unconfirmed": 0 if s is None else sum(x.label is not None and not x.confirmed for x in s.idents),
            "notice": None if current is None else {"text": current[0], "level": self.notice.level},
            "debug": {
                "tags": None if self.report is None else int(self.report.visible),
                "fit_max_mm": None if self.view is None else _num(self.view.reproj_max_mm),
                "fps": _num(self.meter.fps),
                "detect_ms": _num(self.detect_ms),
            },
        }

    def _drift_state(self) -> dict[str, Any] | None:
        r = self.report
        if r is None:
            return None
        return {
            "status": r.status.name,
            "ok": bool(r.ok),
            "text": r.summary(),
            "lost": [int(i) for i in r.lost],
            "card": None if r.card is None else int(r.card),
        }

    def _found(self) -> list:
        """The bases of a snapshot that passed, else an empty list."""
        if self.result is None or not self.result.ok:
            return []
        return self.result.scratch.get("bases") or []

    def _distance_state(self) -> dict[str, float] | None:
        if self.selected is None or self.target is None:
            return None
        found = self._found()
        a, b = found[self.selected], found[self.target]
        gap = measure.rim_gap(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)[0]
        return {"mm": _num(gap), "in": _num(gap / measure.MM_PER_INCH)}

    def _bases_state(self) -> list[dict[str, Any]]:
        found = self._found()
        if not found:
            return []
        centers = np.array([c.center_mm for c in found]).reshape(-1, 2)
        positions = np.asarray(self.win.to_px(centers), float).reshape(-1, 2) * self.win.scale
        idents = self.session.idents if self.session is not None else [None] * len(found)
        return [
            {
                "index": i,
                "diameter_mm": float(c.diameter_mm),
                "label": None if x is None else x.label,
                "score": None if x is None else _num(x.score),
                "sure": x is not None and bool(x.sure),
                "confirmed": x is not None and bool(x.confirmed),
                "position_px": [round(float(p[0]), 1), round(float(p[1]), 1)],
            }
            for i, (c, x, p) in enumerate(zip(found, idents, positions))
        ]

    def _candidates_state(self) -> list[dict[str, Any]]:
        s, i = self.session, self.selected
        if s is None or i is None:
            return []
        scores = dict(s.idents[i].candidates)
        return [{"label": label, "score": _num(scores.get(label))} for label in s.menu(i)]

    # --- actions ----------------------------------------------------------------

    def dispatch(self, action_id: str, **args) -> None:
        """Do one action of `ACTIONS`. An unknown or a shell action is a KeyError."""
        self._handlers[action_id](**args)

    def _need_frame(self) -> bool:
        if self.frame is None:
            self.notice.error("no camera frame yet")
            return False
        return True

    def _close_snapshot(self) -> None:
        self.result = self.snap = self.session = self.shown = None
        self.selected = self.target = self.ring_in = None

    def _load_gallery(self) -> None:
        self.gallery_tried = True
        teams = self.teams or idt.list_teams(self.gallery_root)
        if not teams:
            self.notice.error(f"no gallery teams in {self.gallery_root}: battlesight enroll")
            return
        self.gallery = load_gallery(teams, self.gallery_root, self.cfg)
        if self.gallery is None:
            self.notice.error("gallery: cannot load, see the Terminal")
        self.pipe.set_gallery(self.gallery)

    def _snapshot(self, name: bool) -> None:
        if not self._need_frame():
            return
        self.busy = True
        try:
            if self.on_busy is not None:
                self.on_busy(self.current())
            self._run_snapshot(self.frame, name)
        finally:
            self.busy = False

    def _run_snapshot(self, frame: np.ndarray, name: bool) -> None:
        if name and not self.gallery_tried:
            self._load_gallery()
        self._close_snapshot()
        self.win.top_down = True  # a snapshot opens top-down, the v key changes it
        self.snap = frame.copy()
        self.result = result = self.pipe.analyze(self.snap, source="camera", until=None if name else "bases")
        print(result.summary())
        gallery = self.gallery
        if name and gallery is not None and result.ok:
            self.session = self._label_session(result, gallery)
            for b in result.stages["ident"].data.get("bases", []):
                print(f"  {b['label'] or 'unknown':30s} {b['diameter_mm']:g} mm at {b['center_mm']} mm, score {b['score']}")
        else:
            for b in result.stages["bases"].data.get("bases", []) if "bases" in result.stages else []:
                print(f"  {b['diameter_mm']:g} mm at {b['center_mm']} mm, rim {b['coverage']:.0%} visible")
        if not result.ok:
            self.notice.error(failed_stage(result))
        elif not result.scratch.get("bases"):
            self.notice.error("no base found")
        elif name and gallery is not None and result.stages["ident"].message.startswith("gallery has no photos"):
            self.notice.error(result.stages["ident"].message)
        self.dirty = True

    @staticmethod
    def _label_session(result, gallery: idt.Gallery) -> idt.LabelSession | None:
        found = result.scratch.get("bases")
        if found is None:
            return None
        idents = result.scratch.get("idents") or [idt.Ident(None, float("-inf"), float("-inf")) for _ in found]
        centers = np.array([c.center_mm for c in found]).reshape(-1, 2)
        return idt.LabelSession(centers, [c.diameter_mm for c in found], list(result.scratch.get("base_crops", [])), list(idents), gallery, backgrounds=list(result.scratch.get("base_backgrounds", [])))

    def _set_selected(self, index: int | None) -> None:
        """A new selected base starts with no ring and no second base."""
        self.selected = index
        self.target = self.ring_in = None
        self.dirty = True

    def _deselect(self) -> None:
        self._set_selected(None)

    def _back(self) -> None:
        """First clear the selected base, then close the snapshot."""
        if self.selected is not None:
            self._deselect()
        else:
            self._close_snapshot()

    def _toggle_view(self) -> None:
        """The live view and the snapshot each have their own view."""
        if self.result is not None:
            self.win.top_down = not self.win.top_down
        else:
            self.top_down = not self.top_down
        self.dirty = True

    def _toggle_grid(self) -> None:
        self.grid = not self.grid
        self.dirty = True

    # --- terrain map ----------------------------------------------------------------

    def _read_maps(self) -> None:
        self.maps, problems = ter.list_maps(self.maps_root)
        for text in problems:
            self.notice.error(f"map file not used: {text}")
        types = sorted({e.terrain for e in self.maps})
        self.terrain_type = types[0] if types else None

    def _draw_terrain(self, img: np.ndarray, to_px) -> None:
        switch = {"zone": self.zones, "wall_key": self.keys, "teleporter_key": self.keys}
        shown = [s for s in self.terrain_shapes if switch.get(s.kind, self.terrain)]
        if shown:
            draw_terrain(img, shown, to_px)

    def _toggle_terrain(self) -> None:
        self.terrain = not self.terrain
        self.dirty = True

    def _toggle_zones(self) -> None:
        self.zones = not self.zones
        self.dirty = True

    def _toggle_keys(self) -> None:
        self.keys = not self.keys
        self.dirty = True

    def _set_terrain_type(self, name: str) -> None:
        """A new terrain type starts with no layout."""
        if name not in {e.terrain for e in self.maps}:
            self.notice.error(f"no maps of terrain type {name!r} in {self.maps_root}")
            return
        if name != self.terrain_type:
            self.terrain_type = name
            self._set_map_layout(None)

    def _set_map_layout(self, name: str | None) -> None:
        """Load the layout `name` of the terrain type, or no layout for None.
        A bad file gives a notice and leaves no layout."""
        self.dirty = True
        self.map_layout, self.terrain_shapes = None, []
        if name is None:
            return
        entry = next((e for e in self.maps if e.terrain == self.terrain_type and e.name == name), None)
        if entry is None:
            self.notice.error(f"no layout {name!r} of terrain type {self.terrain_type!r}")
            return
        try:
            tmap = ter.load_map(entry.path, self.cfg)
        except ter.MapError as exc:
            self.notice.error(f"map not loaded: {exc}")
            return
        self.map_layout, self.terrain_shapes, self.terrain = name, ter.shapes(tmap, self.cfg), True
        self.notice.ok(f"map {tmap.terrain} / {tmap.name}")

    def _take_reference(self) -> None:
        if not self._need_frame():
            return
        try:
            self.pipe.set_reference(self.frame)
            if self.reference_path is None:
                self.notice.ok("empty-board reference set for this run, not saved")
                return
            self.reference_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.reference_path), self.frame)
            self.notice.ok(f"empty-board reference saved to {self.reference_path}")
        except ValueError as exc:
            self.notice.error(f"no reference: {exc}")

    def _clear_reference(self) -> None:
        self.pipe.clear_reference()
        self.notice.say("empty-board reference cleared: single-frame detection")

    def _base_under(self, x: float, y: float) -> int | None:
        """The base at pixel (x, y) of the returned image."""
        found = self._found()
        p_mm = self.win.to_mm(x, y) if found else None
        if p_mm is None:
            return None
        return measure.base_at([c.center_mm for c in found], [c.diameter_mm for c in found], p_mm)

    def _click(self, x: float, y: float) -> None:
        """A left click. The first base is the selected base, and another
        base is the target of the distance. A click on one of them or on the
        empty board takes it away."""
        if self.result is None:
            return
        index = self._base_under(x, y)
        if index is None or index == self.selected:
            self._set_selected(None)
        elif self.selected is None:
            self._set_selected(index)
        else:
            self.target = None if index == self.target else index
            self.dirty = True

    def _select_for_names(self, x: float, y: float) -> None:
        """A right click: the base there becomes the selected base, to fix its name."""
        if self.result is None:
            return
        index = self._base_under(x, y)
        self._set_selected(index)
        s = self.session
        if index is None:
            return
        if s is None:
            self.notice.error("push space to name the operatives")
        elif not s.menu(index):
            self.notice.error(f"no operative with a {s.diameters_mm[index]:g} mm base in the gallery: battlesight enroll")

    def _set_ring(self, inches: int | None) -> None:
        if self.selected is None:
            self.notice.say("click a base first")
            return
        self.ring_in = inches
        self.dirty = True

    def _choose(self, label: str, index: int | None = None) -> None:
        """Set the name of base `index` (default: the selected base). A front
        end can give the index, so the order of a close and a choice of its
        popup does not matter."""
        s = self.session
        index = self.selected if index is None else index
        if s is None or index is None:
            return
        path = s.choose(label, index)
        self._set_selected(None)
        name = idt.split_label(label)[1]
        if path is None:
            print(f"named {label}: the gallery has this photo already, no photo added")
            self.notice.ok(f"named {name}, the gallery has this photo already")
        else:
            print(f"named {label}: added {path}")
            self.notice.ok(f"named {name}, photo added to the gallery")
        self.dirty = True

    def _accept_all(self) -> None:
        if self.session is None:
            self.notice.error("push space first")
            return
        added = self.session.accept_all()
        new = sum(p is not None for p in added)
        same = len(added) - new
        self.notice.ok(f"accepted {len(added)} names, {new} photos added to the gallery" + (f", {same} already in the gallery" if same else ""))
        self.dirty = True

    def _undo(self) -> None:
        if self.session is None:
            self.notice.error("push space first")
            return
        done, path = self.session.undo()
        if not done:
            self.notice.error("nothing to undo")
        else:
            self.notice.say("name removed" if path is None else f"removed {path.name}")
        self.dirty = True

    def _save_frame(self) -> None:
        if not self._need_frame():
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / time.strftime("snap_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(path), self.frame)
        print(f"saved {path}")
        self.notice.ok(f"saved {path.name}")
